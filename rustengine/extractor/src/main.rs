//! trackp-extract -- fast, memory-frugal Rust port of the BC corpus builder.
//!
//! Streams `D:\gm_dataset\replays_*.parquet`, keeps 1.32.7 episodes (winner seat
//! + loser seat rated >= 2100), encodes each per-turn decision with the exact
//! Python token layout (see encode.rs), and writes sharded Parquet identical in
//! schema to `data/trackp_corpus`. One process, tiny RAM, no multiprocessing
//! orphan hazard -- reap-proof on a memory-starved box.
//!
//!   trackp-extract --gm D:\gm_dataset --out data\trackp_corpus_rust \
//!                  --rows-per-shard 50000 --jobs 1 [--limit N] [--only id,id]
//!
//! Companions (index.parquet / stats.json / norm.json) are finalised by
//! `python -m kaggriculture.data.trackp_corpus --finalize <out>` (light post-pass).
mod encode;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, BinaryBuilder, Float32Array, Int16Array, Int64Array, Int8Array,
                   LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const ENGINE_KEEP: &str = "1.32.7";

type Banks = HashMap<String, [Option<(f64, f64)>; 2]>; // eid -> [seat0,(bank,rating), seat1]

struct Args {
    gm: PathBuf,
    out: PathBuf,
    rows_per_shard: usize,
    jobs: u64,
    limit: usize,
    only: HashSet<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        gm: PathBuf::from(r"D:\gm_dataset"),
        out: PathBuf::from("data/trackp_corpus_rust"),
        rows_per_shard: 50_000,
        jobs: 1,
        limit: 0,
        only: HashSet::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--gm" => a.gm = PathBuf::from(it.next().unwrap()),
            "--out" => a.out = PathBuf::from(it.next().unwrap()),
            "--rows-per-shard" => a.rows_per_shard = it.next().unwrap().parse().unwrap(),
            "--jobs" => a.jobs = it.next().unwrap().parse().unwrap(),
            "--limit" => a.limit = it.next().unwrap().parse().unwrap(),
            "--only" => {
                a.only = it.next().unwrap().split(',').map(|s| s.trim().to_string()).collect()
            }
            other => eprintln!("[warn] ignoring unknown arg {other}"),
        }
    }
    a
}

fn shard_schema() -> Schema {
    Schema::new(vec![
        Field::new("episode_id", DataType::Int64, false),
        Field::new("seat", DataType::Int8, false),
        Field::new("day", DataType::Int16, false),
        Field::new("step", DataType::Int16, false),
        Field::new("split", DataType::Int8, false),
        Field::new("rtg", DataType::Float32, false),
        Field::new("rating", DataType::Float32, false),
        Field::new("world_family", DataType::Int16, false),
        Field::new("n_tokens", DataType::Int16, false),
        Field::new("tokens", DataType::Binary, false),
        Field::new("action", DataType::Binary, false),
    ])
}

// --- lookups (engine keep + banks/ratings) -----------------------------------
fn load_keep(gm: &Path) -> Result<HashSet<String>> {
    let path = gm.join("episode_features.csv");
    let mut rdr = csv::Reader::from_path(&path)
        .with_context(|| format!("open {}", path.display()))?;
    let hdr = rdr.headers()?.clone();
    let ei = hdr.iter().position(|h| h == "episode_id").ok_or_else(|| anyhow!("no episode_id"))?;
    let vi = hdr.iter().position(|h| h == "engine_version")
        .ok_or_else(|| anyhow!("no engine_version"))?;
    let mut keep = HashSet::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.get(vi) == Some(ENGINE_KEEP) {
            if let Some(e) = rec.get(ei) {
                keep.insert(e.to_string());
            }
        }
    }
    Ok(keep)
}

fn load_banks(gm: &Path) -> Result<Banks> {
    let path = gm.join("agents.csv");
    let mut rdr = csv::Reader::from_path(&path)
        .with_context(|| format!("open {}", path.display()))?;
    let hdr = rdr.headers()?.clone();
    let col = |name: &str| hdr.iter().position(|h| h == name);
    let ei = col("episode_id").ok_or_else(|| anyhow!("no episode_id"))?;
    let ai = col("agent_index").ok_or_else(|| anyhow!("no agent_index"))?;
    let bi = col("final_bank").ok_or_else(|| anyhow!("no final_bank"))?;
    let ri = col("rating_after").ok_or_else(|| anyhow!("no rating_after"))?;
    let mut banks: Banks = HashMap::new();
    for rec in rdr.records() {
        let rec = rec?;
        let (eid, seat) = match (rec.get(ei), rec.get(ai)) {
            (Some(e), Some(s)) => (e.to_string(), s.parse::<usize>().ok()),
            _ => continue,
        };
        let seat = match seat {
            Some(s) if s < 2 => s,
            _ => continue,
        };
        // Python: float(x or 0), skip row (continue) on parse failure.
        let bank = parse_or_zero(rec.get(bi));
        let rat = parse_or_zero(rec.get(ri));
        let (bank, rat) = match (bank, rat) {
            (Some(b), Some(r)) => (b, r),
            _ => continue,
        };
        banks.entry(eid).or_insert([None, None])[seat] = Some((bank, rat));
    }
    Ok(banks)
}

/// Python `float(s or 0)`: empty -> 0.0; unparseable -> None (skip).
fn parse_or_zero(s: Option<&str>) -> Option<f64> {
    match s {
        None => Some(0.0),
        Some(x) if x.is_empty() => Some(0.0),
        Some(x) => x.parse::<f64>().ok(),
    }
}

// --- replay parquet iteration ------------------------------------------------
fn replay_files(gm: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = fs::read_dir(gm)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("replays_") && n.ends_with(".parquet"))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v.reverse(); // newest shard first (match iter_replays reverse=True)
    Ok(v)
}

fn projection(builder: &ParquetRecordBatchReaderBuilder<File>, names: &[&str]) -> ProjectionMask {
    let sd = builder.parquet_schema();
    let arrow_schema = builder.schema();
    let roots: Vec<usize> = names
        .iter()
        .filter_map(|n| arrow_schema.index_of(n).ok())
        .collect();
    ProjectionMask::roots(sd, roots)
}

/// episode_ids present in a file (cheap: int64 column only).
fn file_episode_ids(path: &Path) -> Result<Vec<i64>> {
    let f = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(f)?;
    let mask = projection(&builder, &["episode_id"]);
    let reader = builder.with_projection(mask).with_batch_size(16384).build()?;
    let mut ids = Vec::new();
    for batch in reader {
        let b = batch?;
        if let Some(c) = b.column(0).as_any().downcast_ref::<Int64Array>() {
            ids.extend((0..c.len()).map(|i| c.value(i)));
        }
    }
    Ok(ids)
}

fn get_str<'a>(arr: &'a dyn Array, i: usize) -> Option<&'a str> {
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return if a.is_null(i) { None } else { Some(a.value(i)) };
    }
    if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
        return if a.is_null(i) { None } else { Some(a.value(i)) };
    }
    None
}

// --- shard writer ------------------------------------------------------------
struct ShardWriter {
    out: PathBuf,
    prefix: String,
    schema: Arc<Schema>,
    next_shard: usize,
    // buffers
    episode_id: Vec<i64>,
    seat: Vec<i8>,
    day: Vec<i16>,
    step: Vec<i16>,
    split: Vec<i8>,
    rtg: Vec<f32>,
    rating: Vec<f32>,
    world_family: Vec<i16>,
    n_tokens: Vec<i16>,
    tokens: Vec<Vec<u8>>,
    action: Vec<Vec<u8>>,
    rows_per_shard: usize,
    rows_written: usize,
}

impl ShardWriter {
    fn new(out: &Path, prefix: &str, rows_per_shard: usize) -> Self {
        let next_shard = existing_max_shard(out, prefix).map(|m| m + 1).unwrap_or(0);
        ShardWriter {
            out: out.to_path_buf(),
            prefix: prefix.to_string(),
            schema: Arc::new(shard_schema()),
            next_shard,
            episode_id: vec![],
            seat: vec![],
            day: vec![],
            step: vec![],
            split: vec![],
            rtg: vec![],
            rating: vec![],
            world_family: vec![],
            n_tokens: vec![],
            tokens: vec![],
            action: vec![],
            rows_per_shard,
            rows_written: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, eid: i64, seat: i8, day: i16, step: i16, split: i8, rtg: f32,
            rating: f32, wf: i16, ntok: i16, toks: Vec<u8>, act: Vec<u8>) -> Result<()> {
        self.episode_id.push(eid);
        self.seat.push(seat);
        self.day.push(day);
        self.step.push(step);
        self.split.push(split);
        self.rtg.push(rtg);
        self.rating.push(rating);
        self.world_family.push(wf);
        self.n_tokens.push(ntok);
        self.tokens.push(toks);
        self.action.push(act);
        if self.episode_id.len() >= self.rows_per_shard {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.episode_id.is_empty() {
            return Ok(());
        }
        let mut tb = BinaryBuilder::new();
        for v in &self.tokens {
            tb.append_value(v);
        }
        let mut ab = BinaryBuilder::new();
        for v in &self.action {
            ab.append_value(v);
        }
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(Int64Array::from(std::mem::take(&mut self.episode_id))),
                Arc::new(Int8Array::from(std::mem::take(&mut self.seat))),
                Arc::new(Int16Array::from(std::mem::take(&mut self.day))),
                Arc::new(Int16Array::from(std::mem::take(&mut self.step))),
                Arc::new(Int8Array::from(std::mem::take(&mut self.split))),
                Arc::new(Float32Array::from(std::mem::take(&mut self.rtg))),
                Arc::new(Float32Array::from(std::mem::take(&mut self.rating))),
                Arc::new(Int16Array::from(std::mem::take(&mut self.world_family))),
                Arc::new(Int16Array::from(std::mem::take(&mut self.n_tokens))),
                Arc::new(tb.finish()),
                Arc::new(ab.finish()),
            ],
        )?;
        let name = format!("{}{:04}.parquet", self.prefix, self.next_shard);
        let path = self.out.join(&name);
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(10)?))
            .build();
        let mut w = ArrowWriter::try_new(file, self.schema.clone(), Some(props))?;
        w.write(&batch)?;
        w.close()?;
        self.rows_written += batch.num_rows();
        self.next_shard += 1;
        self.tokens.clear();
        self.action.clear();
        Ok(())
    }
}

fn existing_max_shard(out: &Path, prefix: &str) -> Option<usize> {
    let mut mx: Option<usize> = None;
    if let Ok(rd) = fs::read_dir(out) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str() {
                if let Some(rest) = n.strip_prefix(prefix) {
                    if let Some(num) = rest.strip_suffix(".parquet") {
                        if let Ok(k) = num.parse::<usize>() {
                            mx = Some(mx.map_or(k, |m| m.max(k)));
                        }
                    }
                }
            }
        }
    }
    mx
}

/// episode_ids already present in ANY shard_*.parquet (resume support).
/// A shard we wrote (`shard_r*`) that no longer reads back is a partial write
/// from a killed run: delete it so those episodes are regenerated cleanly and
/// the finalize post-pass never trips on a truncated footer (mirrors the Python
/// `_done_episodes` corrupt-shard removal).
fn done_episodes(out: &Path) -> Result<HashSet<String>> {
    let mut done = HashSet::new();
    if let Ok(rd) = fs::read_dir(out) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("shard_") && name.ends_with(".parquet") {
                match file_episode_ids(&e.path()) {
                    Ok(ids) => done.extend(ids.into_iter().map(|i| i.to_string())),
                    Err(_) => {
                        if name.starts_with("shard_r") {
                            let _ = fs::remove_file(e.path());
                            eprintln!("[trackp-extract] removed corrupt shard {name}");
                        }
                    }
                }
            }
        }
    }
    Ok(done)
}

// --- per-episode row extraction (mirror episode_rows) ------------------------
fn world_family_of(steps: &[Value]) -> i32 {
    for cell in steps {
        let obs = cell.get(0).and_then(|c| c.get("observation"));
        if let Some(obs) = obs {
            if encode::as_int(obs.get("day").unwrap_or(&Value::Null)) == 6 {
                if let Some(shops) =
                    obs.get("town").and_then(|t| t.get("unlocked_shops")).and_then(|v| v.as_array())
                {
                    let mut top: Vec<String> = shops
                        .iter()
                        .take(2)
                        .filter_map(|s| s.as_str().map(|x| x.to_string()))
                        .collect();
                    top.sort();
                    return encode::world_family(&top.join("|"));
                }
                return encode::world_family("");
            }
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn extract_episode(eid: &str, rep: &Value, banks: &Banks, w: &mut ShardWriter) -> Result<usize> {
    let steps = match rep.get("steps").and_then(|v| v.as_array()) {
        Some(s) if s.len() >= 24 => s,
        _ => return Ok(0),
    };
    let bk = match banks.get(eid) {
        Some(b) if b[0].is_some() && b[1].is_some() => b,
        _ => return Ok(0), // len(bk) < 2
    };
    let b0 = bk[0].unwrap().0;
    let b1 = bk[1].unwrap().0;
    let win0: f32 = if b0 > b1 { 1.0 } else if b0 == b1 { 0.5 } else { 0.0 };
    let win = [win0, if win0 == 0.5 { 0.5 } else { 1.0 - win0 }];
    let sp = encode::split_of(eid) as i8;
    let winner = if b0 >= b1 { 0usize } else { 1 };
    let loser = 1 - winner;
    let mut seats = vec![winner];
    if bk[loser].unwrap().1 >= encode::LOSER_RATING_MIN {
        seats.push(loser);
    }
    let wf = world_family_of(steps) as i16;
    let eid_i64: i64 = eid.parse().unwrap_or(0);
    let mut n = 0usize;
    for &seat in &seats {
        let rat = bk[seat].unwrap().1 as f32;
        for (t, stepcell) in steps.iter().enumerate() {
            let cell = match stepcell.get(seat) {
                Some(c) if !c.is_null() => c,
                _ => continue,
            };
            let obs = match cell.get("observation") {
                Some(o) if !o.is_null() => o,
                _ => continue,
            };
            // obs is seat-specific: keep iff player == seat (Python `!= seat`
            // skips absent/mismatched; value-equality so 0.0 == 0).
            let player_ok = match obs.get("player") {
                Some(v) if !v.is_null() => encode::as_int(v) == seat as i64,
                _ => false,
            };
            if !player_ok {
                continue;
            }
            let toks = encode::encode_tokens(obs, seat);
            let ntok = (toks.len() / encode::TOK_W) as i16;
            let act = encode::encode_action(cell.get("action").unwrap_or(&Value::Null));
            let day = encode::as_int(obs.get("day").unwrap_or(&Value::Null)) as i16;
            let step = match obs.get("step") {
                Some(v) if !v.is_null() => encode::as_int(v),
                _ => t as i64,
            } as i16;
            let tok_bytes = i32s_to_le(&toks);
            let act_bytes = i32s_to_le(&act);
            w.push(eid_i64, seat as i8, day, step, sp, win[seat], rat, wf, ntok, tok_bytes,
                   act_bytes)?;
            n += 1;
        }
    }
    Ok(n)
}

#[inline]
fn i32s_to_le(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// --- worker ------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn run_worker(
    wid: u64,
    njobs: u64,
    files: &[PathBuf],
    keep: &HashSet<String>,
    done: &HashSet<String>,
    only: &HashSet<String>,
    banks: &Banks,
    out: &Path,
    rows_per_shard: usize,
    limit: usize,
    processed: &AtomicUsize,
) -> Result<(usize, usize)> {
    let prefix = if njobs > 1 { format!("shard_r{wid}_") } else { "shard_r".to_string() };
    let mut w = ShardWriter::new(out, &prefix, rows_per_shard);
    let mut n_rows = 0usize;
    let mut n_eps = 0usize;
    for path in files {
        let f = match File::open(path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let builder = match ParquetRecordBatchReaderBuilder::try_new(f) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // ROW-GROUP split (not episode-hash): each worker decompresses a disjoint
        // 1/njobs of the row-groups -> no redundant reads, balanced -> CPU-bound,
        // scales with cores. (rg r -> worker r % njobs)
        let nrg = builder.metadata().num_row_groups();
        let my_rgs: Vec<usize> = (0..nrg)
            .filter(|r| njobs <= 1 || (*r as u64) % njobs == wid)
            .collect();
        if my_rgs.is_empty() {
            continue;
        }
        let mask = projection(&builder, &["episode_id", "replay_json"]);
        let reader = builder.with_projection(mask).with_row_groups(my_rgs)
            .with_batch_size(16).build()?;
        for batch in reader {
            let b = batch?;
            let eids = b.column(0).as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("episode_id not int64"))?;
            let rjs = b.column(1);
            for i in 0..b.num_rows() {
                let eid = eids.value(i).to_string();
                if !keep.contains(&eid) || done.contains(&eid) {
                    continue;
                }
                if !only.is_empty() && !only.contains(&eid) {
                    continue;
                }
                // (no episode-hash filter: row-group split already makes the
                // workers' row sets disjoint)
                let rj = match get_str(rjs.as_ref(), i) {
                    Some(s) => s,
                    None => continue,
                };
                let rep: Value = match serde_json::from_str(rj) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let added = extract_episode(&eid, &rep, banks, &mut w)?;
                if added > 0 {
                    n_rows += added;
                    n_eps += 1;
                }
                let p = processed.fetch_add(1, Ordering::Relaxed) + 1;
                if limit > 0 && p >= limit {
                    w.flush()?;
                    return Ok((n_rows, n_eps));
                }
            }
        }
    }
    w.flush()?;
    Ok((n_rows, n_eps))
}

fn main() -> Result<()> {
    let a = parse_args();
    fs::create_dir_all(&a.out)?;
    let t0 = std::time::Instant::now();

    eprintln!("[trackp-extract] loading lookups from {}", a.gm.display());
    let keep = load_keep(&a.gm)?;
    let banks = load_banks(&a.gm)?;
    eprintln!("[trackp-extract] {} @ {} ; {} episodes with banks", keep.len(), ENGINE_KEEP,
              banks.len());
    let done = done_episodes(&a.out)?;
    eprintln!("[trackp-extract] {} episodes already done (resume)", done.len());

    let all_files = replay_files(&a.gm)?;
    // Skip files with no kept, not-done episode -- avoids decompressing the huge
    // 1.32.6 shard entirely.
    let mut files: Vec<PathBuf> = Vec::new();
    for f in &all_files {
        let ids = file_episode_ids(f).unwrap_or_default();
        let hit = ids.iter().any(|id| {
            let s = id.to_string();
            keep.contains(&s) && !done.contains(&s) && (a.only.is_empty() || a.only.contains(&s))
        });
        eprintln!("[trackp-extract] {}  ({} rows){}", f.file_name().unwrap().to_string_lossy(),
                  ids.len(), if hit { "" } else { "  [skip: no kept]" });
        if hit {
            files.push(f.clone());
        }
    }

    let keep = Arc::new(keep);
    let done = Arc::new(done);
    let only = Arc::new(a.only);
    let banks = Arc::new(banks);
    let processed = Arc::new(AtomicUsize::new(0));

    let (n_rows, n_eps) = if a.jobs > 1 {
        let results: Vec<Result<(usize, usize)>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..a.jobs)
                .map(|wid| {
                    let (keep, done, only, banks, processed) =
                        (keep.clone(), done.clone(), only.clone(), banks.clone(), processed.clone());
                    let files = files.clone();
                    let out = a.out.clone();
                    sc.spawn(move || {
                        run_worker(wid, a.jobs, &files, &keep, &done, &only, &banks, &out,
                                   a.rows_per_shard, a.limit, &processed)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut r = 0;
        let mut e = 0;
        for res in results {
            let (rr, ee) = res?;
            r += rr;
            e += ee;
        }
        (r, e)
    } else {
        run_worker(0, 1, &files, &keep, &done, &only, &banks, &a.out, a.rows_per_shard,
                   a.limit, &processed)?
    };

    write_companions(&a.out, n_rows, n_eps)?;
    let dt = t0.elapsed().as_secs_f64();
    let total_shards = count_shards(&a.out);
    eprintln!(
        "[trackp-extract] +{n_rows} rows / +{n_eps} eps in {dt:.1}s ({:.1} eps/s); \
         {total_shards} shards -> {}",
        n_eps as f64 / dt.max(1e-6),
        a.out.display()
    );
    eprintln!("[trackp-extract] NEXT: python -m kaggriculture.data.trackp_corpus --finalize {}",
              a.out.display());
    Ok(())
}

fn count_shards(out: &Path) -> usize {
    fs::read_dir(out)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    let n = e.file_name();
                    let n = n.to_string_lossy();
                    n.starts_with("shard_") && n.ends_with(".parquet")
                })
                .count()
        })
        .unwrap_or(0)
}

fn write_companions(out: &Path, n_rows: usize, n_eps: usize) -> Result<()> {
    let vocab = serde_json::json!({
        "crops": encode::CROPS, "animals": encode::ANIMALS, "products": encode::PRODUCTS,
        "shops": encode::SHOPS, "mover_verbs": encode::MOVER_VERBS,
        "market_verbs": encode::MARKET_VERBS,
        "kinds": ["EMPTY","PLANT","PASTURE","COOP","WEED"],
        "max_hands": encode::MAX_HANDS, "tok_w": encode::TOK_W, "max_tokens": encode::MAX_TOKENS,
        "act_w": encode::ACT_W, "max_market": encode::MAX_MARKET,
    });
    fs::write(out.join("vocab.json"), serde_json::to_vec_pretty(&vocab)?)?;
    let manifest = serde_json::json!({
        "token_layout_version": 2, "engine": ENGINE_KEEP, "val_fraction": encode::VAL_FRACTION,
        "loser_rating_min": encode::LOSER_RATING_MIN, "source": "rust",
        "n_shards": count_shards(out), "rows_added_this_run": n_rows,
        "eps_added_this_run": n_eps, "complete": false,
        "built": chrono_now(),
    });
    fs::write(out.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

fn chrono_now() -> String {
    // no chrono dep: seconds since epoch is enough as a build marker
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}
