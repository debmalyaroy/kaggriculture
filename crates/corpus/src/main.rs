//! corpus-extract: Kaggriculture replays -> macro (per game day) Parquet corpus.
//!
//!   corpus-extract --out DIR [--gm DIR] [--daily DIR]... [--replays DIR]...
//!                  [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N] [--threads N] [--plan]
//!                  [--only ID,ID] [--checkpoint-eps N] [--rows-per-group N] [--engine V]
//!
//! See docs/rl/corpus.md for the schema and the ledger/delta rules.
mod extract;
mod files;
mod gm;
mod slim;
mod slimsrc;
mod store;
mod util;

use anyhow::{bail, Result};
use extract::{extract, Meta, Out};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

pub const EXTRACTOR_VERSION: &str = "corpus-1";
/// `--mode slim` writes only the slim table under its own ledger version.
pub const SLIM_VERSION: &str = "slim-1";
/// `--mode obs`: the macro policy observation (crates/dayobs) + labels, table `day_obs`.
pub const OBS_VERSION: &str = "obs-1";
pub const SCHEMA_VERSION: u32 = 1;

struct Args {
    out: PathBuf,
    gm: Option<PathBuf>,
    daily: Vec<PathBuf>,
    replays: Vec<PathBuf>,
    slim: Vec<PathBuf>,
    since: Option<String>,
    until: Option<String>,
    limit: usize,
    threads: usize,
    plan: bool,
    only: HashSet<i64>,
    checkpoint_eps: usize,
    rows_per_group: usize,
    engine: String,
    mode: String,
}

const USAGE: &str = "usage: corpus-extract --out DIR [--gm DIR] [--daily DIR]... [--replays DIR]... \
[--since YYYY-MM-DD] [--until YYYY-MM-DD] [--limit N] [--threads N=2] [--plan] [--only ID,ID] \
[--checkpoint-eps N=500|50(slim)] [--rows-per-group N=50000] [--engine 1.32.7] [--mode features|slim]";

fn parse_args() -> Result<Args> {
    let mut a = Args {
        out: PathBuf::new(),
        gm: None,
        daily: vec![],
        replays: vec![],
        slim: vec![],
        since: None,
        until: None,
        limit: 0,
        threads: 2,
        plan: false,
        only: HashSet::new(),
        checkpoint_eps: 0,
        rows_per_group: 50_000,
        engine: "1.32.7".into(),
        mode: "features".into(),
    };
    let mut it = std::env::args().skip(1);
    let mut need = |it: &mut dyn Iterator<Item = String>, k: &str| -> Result<String> {
        it.next().ok_or_else(|| anyhow::anyhow!("{k} needs a value\n{USAGE}"))
    };
    while let Some(k) = it.next() {
        match k.as_str() {
            "--out" => a.out = PathBuf::from(need(&mut it, &k)?),
            "--gm" => a.gm = Some(PathBuf::from(need(&mut it, &k)?)),
            "--daily" => a.daily.push(PathBuf::from(need(&mut it, &k)?)),
            "--replays" => a.replays.push(PathBuf::from(need(&mut it, &k)?)),
            "--slim" => a.slim.push(PathBuf::from(need(&mut it, &k)?)),
            "--since" => a.since = Some(need(&mut it, &k)?),
            "--until" => a.until = Some(need(&mut it, &k)?),
            "--limit" => a.limit = need(&mut it, &k)?.parse()?,
            "--threads" => a.threads = need(&mut it, &k)?.parse::<usize>()?.max(1),
            "--plan" => a.plan = true,
            "--only" => {
                a.only = need(&mut it, &k)?
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            "--checkpoint-eps" => a.checkpoint_eps = need(&mut it, &k)?.parse::<usize>()?.max(1),
            "--rows-per-group" => a.rows_per_group = need(&mut it, &k)?.parse()?,
            "--engine" => a.engine = need(&mut it, &k)?,
            "--mode" => a.mode = need(&mut it, &k)?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    anyhow::ensure!(a.mode == "features" || a.mode == "slim" || a.mode == "obs", "--mode must be features, slim or obs");
    if a.checkpoint_eps == 0 {
        a.checkpoint_eps = if a.mode == "slim" { 50 } else { 500 };
    }
    if a.out.as_os_str().is_empty() {
        bail!("--out is required\n{USAGE}");
    }
    anyhow::ensure!(a.slim.is_empty() || a.mode == "features" || a.mode == "obs", "--slim input is for --mode features / obs");
    anyhow::ensure!(a.mode != "obs" || (a.gm.is_none() && a.daily.is_empty() && a.replays.is_empty()), "--mode obs reads --slim input only");
    if a.gm.is_none() && a.daily.is_empty() && a.replays.is_empty() && a.slim.is_empty() {
        bail!("give at least one of --gm / --daily / --replays\n{USAGE}");
    }
    Ok(a)
}

/// Where a work item's replay(s) live.
enum Loc {
    Gm { shard: usize, rg: usize },
    Slim(usize),
    File(PathBuf),
}

struct Work {
    loc: Loc,
    metas: Vec<Meta>,
}

/// Per-source plan counters.
#[derive(Default, Clone)]
struct PlanLine {
    files: usize,
    ids: usize,
    done: usize,
    engine_skip: usize,
    dup: usize,
    before_since: usize,
    todo: usize,
    note: String,
}

fn date_of(end_time: &str) -> String {
    util::find_date(end_time).unwrap_or_else(|| "unknown".into())
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let slim_mode = a.mode == "slim";
    let obs_mode = a.mode == "obs";
    let ver: &str = if slim_mode { SLIM_VERSION } else if obs_mode { OBS_VERSION } else { EXTRACTOR_VERSION };
    let t0 = Instant::now();
    std::fs::create_dir_all(&a.out)?;
    let ledger = store::read_ledger(&a.out, ver)?;
    if !a.plan {
        let n = store::remove_orphans(&a.out, &ledger.tags);
        if n > 0 {
            eprintln!("[corpus] removed {n} orphaned data files from interrupted checkpoints");
        }
    }
    eprintln!(
        "[corpus] ledger: {} rows, {} ids done under {ver}",
        ledger.n_rows,
        ledger.done.len()
    );
    let done = &ledger.done;
    let mut sources = store::read_sources(&a.out);
    let mut plan: BTreeMap<String, PlanLine> = BTreeMap::new();
    let mut seen: HashSet<i64> = HashSet::new();
    let mut cands: Vec<(String, Meta, Loc)> = Vec::new(); // (end_time, meta, loc)
    let since = a.since.clone().unwrap_or_default();
    let until = a.until.clone().unwrap_or_default();
    let only = &a.only;
    let keep_id = |e: i64| only.is_empty() || only.contains(&e);

    // ---------------------------------------------------------------- GM ----------------
    let mut shards: Vec<gm::Shard> = Vec::new();
    let mut gm_engine: HashMap<i64, bool> = HashMap::new();
    if let Some(gmdir) = &a.gm {
        gm_engine = gm::load_engine(gmdir, &a.engine)?;
        let mut gm_pending: Vec<(i64, usize, usize)> = Vec::new(); // eid, shard, rg
        for path in gm::shard_files(gmdir)? {
            let key = path.to_string_lossy().to_string();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let fp = util::fingerprint(&path)?;
            let line = plan.entry(format!("gm {name}")).or_default();
            line.files = 1;
            if let Some(s) = sources.get(&key) {
                if s["fingerprint"] == fp.as_str()
                    && s["complete"] == true
                    && s["extractor_version"] == ver
                    && only.is_empty()
                {
                    line.note = "frozen+complete: skipped wholesale".into();
                    continue;
                }
            }
            let sh = gm::Shard::open(&path, fp)?;
            let si = shards.len();
            for &(e, rg) in &sh.ids {
                line.ids += 1;
                if !keep_id(e) {
                    continue;
                }
                if done.contains(&e) {
                    line.done += 1;
                } else if gm_engine.get(&e) == Some(&false) {
                    line.engine_skip += 1;
                } else if !seen.insert(e) {
                    line.dup += 1;
                } else {
                    gm_pending.push((e, si, rg));
                }
            }
            shards.push(sh);
        }
        let want: HashSet<i64> = gm_pending.iter().map(|x| x.0).collect();
        let eps = gm::load_episodes(gmdir, &want)?;
        for (e, si, rg) in gm_pending {
            let g = eps.get(&e).cloned();
            let end_time = g.as_ref().map(|g| g.end_time.clone()).unwrap_or_default();
            let meta = Meta {
                eid: e,
                source: "gm",
                source_file: shards[si].path.file_name().unwrap().to_string_lossy().to_string(),
                file_sha: shards[si].fingerprint.clone(),
                end_date: date_of(&end_time),
                end_time: end_time.clone(),
                gm: g,
                ..Default::default()
            };
            cands.push((end_time, meta, Loc::Gm { shard: si, rg }));
        }
    }

    // ----------------------------------------------------------- daily / replays ---------
    let mut file_lists: Vec<(&'static str, PathBuf, Vec<files::FileEp>)> = Vec::new();
    for d in &a.daily {
        for dd in files::daily_dirs(d) {
            let eps = files::scan_daily(&dd)?;
            file_lists.push(("daily", dd, eps));
        }
    }
    for d in &a.replays {
        file_lists.push(("replays", d.clone(), files::scan_replays(d)?));
    }
    for (src, dir, eps) in file_lists {
        let line = plan.entry(format!("{src} {}", dir.display())).or_default();
        line.files = eps.len();
        for fe in eps {
            line.ids += 1;
            if !keep_id(fe.eid) {
                continue;
            }
            if done.contains(&fe.eid) {
                line.done += 1;
                continue;
            }
            if !seen.insert(fe.eid) {
                line.dup += 1;
                continue;
            }
            let meta = Meta {
                eid: fe.eid,
                source: src,
                source_file: fe.path.to_string_lossy().to_string(),
                end_date: date_of(&fe.end_time),
                end_time: fe.end_time.clone(),
                rating_min: fe.rating_min,
                rating_max: fe.rating_max,
                ..Default::default()
            };
            cands.push((fe.end_time, meta, Loc::File(fe.path)));
        }
    }

    // ------------------------------------------------------------- slim corpus ----------
    let mut slim_files: Vec<PathBuf> = Vec::new();
    for root in &a.slim {
        for f in slimsrc::files(root) {
            let rows = slimsrc::index(&f)?;
            let key = format!("slim {}", f.display());
            let line = plan.entry(key).or_default();
            line.files = 1;
            let fi = slim_files.len();
            slim_files.push(f.clone());
            for r in rows {
                line.ids += 1;
                if !keep_id(r.eid) {
                    continue;
                }
                if done.contains(&r.eid) {
                    line.done += 1;
                    continue;
                }
                if !seen.insert(r.eid) {
                    line.dup += 1;
                    continue;
                }
                let src: &'static str = match r.source.as_str() {
                    "gm" => "gm",
                    "daily" => "daily",
                    "replays" => "replays",
                    _ => "slim",
                };
                let meta = Meta {
                    eid: r.eid,
                    source: src,
                    source_file: f.to_string_lossy().to_string(),
                    end_date: if r.end_date.is_empty() { date_of(&r.end_time) } else { r.end_date.clone() },
                    end_time: r.end_time.clone(),
                    gm: Some(gm::GmEp {
                        end_time: r.end_time.clone(),
                        episode_type: r.episode_type.clone(),
                        sub: r.sub,
                        team: [None, None],
                        rating: r.rating,
                    }),
                    rating_min: r.rating_min,
                    rating_max: r.rating_max,
                    coverage: r.coverage,
                    ..Default::default()
                };
                cands.push((r.end_time, meta, Loc::Slim(fi)));
            }
        }
    }
    let slim_files = Arc::new(slim_files);

    // ------------------------------------------------- since / order / limit -------------
    let plan_keys: Vec<String> = plan.keys().cloned().collect();
    let source_key = |m: &Meta| -> String {
        if m.source == "gm" {
            format!("gm {}", m.source_file)
        } else {
            plan_keys
                .iter()
                .find(|k| k.starts_with(m.source) && m.source_file.starts_with(&k[m.source.len() + 1..]))
                .cloned()
                .unwrap_or_default()
        }
    };
    let mut kept = Vec::with_capacity(cands.len());
    let mut since_drop: HashMap<String, usize> = HashMap::new();
    for c in cands {
        let after_until = !until.is_empty() && (c.1.end_date == "unknown" || c.1.end_date.as_str() > until.as_str());
        if after_until || (!since.is_empty() && (c.1.end_date == "unknown" || c.1.end_date.as_str() < since.as_str())) {
            *since_drop.entry(source_key(&c.1)).or_default() += 1;
            continue;
        }
        kept.push(c);
    }
    for (k, n) in since_drop {
        plan.entry(k).or_default().before_since += n;
    }
    kept.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.eid.cmp(&x.1.eid)));
    if a.limit > 0 && kept.len() > a.limit {
        kept.truncate(a.limit);
    }
    for c in &kept {
        plan.entry(source_key(&c.1)).or_default().todo += 1;
    }

    // ------------------------------------------------------------------- plan ----------
    let total = kept.len();
    eprintln!("[corpus] plan ({} sources):", plan.len());
    eprintln!("  {:<58} {:>7} {:>7} {:>7} {:>7} {:>5} {:>7} {:>7}",
              "source", "files", "ids", "done", "eng!=", "dup", "<since", "todo");
    for (k, l) in &plan {
        eprintln!("  {:<58} {:>7} {:>7} {:>7} {:>7} {:>5} {:>7} {:>7} {}",
                  k, l.files, l.ids, l.done, l.engine_skip, l.dup, l.before_since, l.todo, l.note);
    }
    if let (Some(f), Some(l)) = (kept.first(), kept.last()) {
        eprintln!("[corpus] {total} episodes to extract, newest {} .. oldest {}", f.1.end_date, l.1.end_date);
    } else {
        eprintln!("[corpus] nothing to extract");
    }
    if a.plan {
        return Ok(());
    }

    // GM side facts for the work set only
    if let Some(gmdir) = &a.gm {
        let want: HashSet<i64> = kept.iter().filter(|c| c.1.source == "gm").map(|c| c.1.eid).collect();
        if !want.is_empty() {
            let hashes = gm::load_hashes(gmdir, &want)?;
            let subs: HashSet<i64> = kept
                .iter()
                .filter_map(|c| c.1.gm.as_ref())
                .flat_map(|g| g.sub.iter().flatten().copied())
                .collect();
            let cov = gm::load_coverage(gmdir, &subs)?;
            for c in kept.iter_mut() {
                if c.1.source != "gm" {
                    continue;
                }
                c.1.gm_hash = hashes.get(&c.1.eid).copied();
                if let Some(g) = &c.1.gm {
                    c.1.coverage = [g.sub[0].and_then(|s| cov.get(&s).copied()),
                                    g.sub[1].and_then(|s| cov.get(&s).copied())];
                }
            }
        }
    }

    // group GM rows of one row group into one work item (decoded once)
    let mut works: Vec<Work> = Vec::new();
    let mut rg_index: HashMap<(usize, usize), usize> = HashMap::new();
    for (_, meta, loc) in kept {
        match loc {
            Loc::Gm { shard, rg } => {
                if let Some(&w) = rg_index.get(&(shard, rg)) {
                    works[w].metas.push(meta);
                } else {
                    rg_index.insert((shard, rg), works.len());
                    works.push(Work { loc: Loc::Gm { shard, rg }, metas: vec![meta] });
                }
            }
            Loc::File(p) => works.push(Work { loc: Loc::File(p), metas: vec![meta] }),
            Loc::Slim(fi) => {
                // one work item per slim file: its records are streamed once
                if let Some(&w) = rg_index.get(&(usize::MAX, fi)) {
                    works[w].metas.push(meta);
                } else {
                    rg_index.insert((usize::MAX, fi), works.len());
                    works.push(Work { loc: Loc::Slim(fi), metas: vec![meta] });
                }
            }
        }
    }

    // ------------------------------------------------------------------ run -------------
    let run = format!(
        "{}p{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        std::process::id()
    );
    let mut st = store::Store::new(&a.out, run, a.rows_per_group, ver);
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::sync_channel::<Out>(a.threads * 2);
    let shards = Arc::new(shards);
    let engine = a.engine.clone();
    let mut finished: HashSet<i64> = HashSet::new();
    let (mut n_ok, mut n_skip, mut n_err, mut hash_ok, mut hash_bad) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut last_print = Instant::now();
    let t_run = Instant::now();
    let mut since_ckpt = 0usize;
    let mut first_errors: Vec<String> = Vec::new();

    std::thread::scope(|sc| -> Result<()> {
        for _ in 0..a.threads {
            let tx = tx.clone();
            let (works, next, shards, engine) = (&works, &next, shards.clone(), engine.clone());
            let slim_files = slim_files.clone();
            let ver = ver;
            sc.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(w) = works.get(i) else { break };
                    run_work(w, &shards, &slim_files, &engine, &tx, slim_mode, ver);
                }
            });
        }
        drop(tx);
        for out in rx {
            let eid = out.meta.eid;
            let date = out.meta.end_date.clone();
            if out.status == "ok" {
                n_ok += 1;
            } else if out.status.starts_with("skipped") {
                n_skip += 1;
            } else {
                n_err += 1;
                if first_errors.len() < 5 {
                    first_errors.push(format!("{eid}: {}", out.status));
                }
            }
            match out.hash_match {
                Some(true) => hash_ok += 1,
                Some(false) => hash_bad += 1,
                None => {}
            }
            if let Some(r) = out.episode {
                st.table("episodes").push(&date, r)?;
            }
            for r in out.states {
                st.table("daily_states").push(&date, r)?;
            }
            for r in out.behaviour {
                st.table("daily_behaviour").push(&date, r)?;
            }
            for r in out.obs {
                st.table("day_obs").push(&date, r)?;
            }
            if let Some(r) = out.slim {
                st.table("slim").push(&date, r)?;
            }
            if !out.status.starts_with("error") {
                finished.insert(eid);
            }
            st.pending.push(store::LedgerRow {
                episode_id: eid,
                source: out.meta.source.to_string(),
                source_file: out.meta.source_file.clone(),
                source_file_sha: out.meta.file_sha.clone(),
                engine_version: out.engine.clone(),
                end_date: date,
                status: out.status.clone(),
            });
            since_ckpt += 1;
            if since_ckpt >= a.checkpoint_eps {
                st.checkpoint()?;
                store::write_schema(&a.out, SCHEMA_VERSION, ver, &st.tables)?;
                since_ckpt = 0;
            }
            let n = n_ok + n_skip + n_err;
            if last_print.elapsed().as_secs_f64() >= 10.0 {
                last_print = Instant::now();
                let dt = t_run.elapsed().as_secs_f64();
                let rate = n as f64 / dt.max(1e-9);
                let eta = (total.saturating_sub(n)) as f64 / rate.max(1e-9);
                eprintln!(
                    "[corpus] {n}/{total}  ok {n_ok} skip {n_skip} err {n_err}  {rate:.2} eps/s  ETA {:.0}s  peak RSS {} MB",
                    eta,
                    util::peak_rss() >> 20
                );
            }
        }
        Ok(())
    })?;
    st.checkpoint()?;
    store::write_schema(&a.out, SCHEMA_VERSION, ver, &st.tables)?;

    // GM shard completeness for the wholesale skip next time
    for sh in shards.iter() {
        let complete = sh.ids.iter().all(|(e, _)| {
            done.contains(e) || finished.contains(e) || gm_engine.get(e) == Some(&false)
        });
        sources.insert(
            sh.path.to_string_lossy().to_string(),
            serde_json::json!({
                "fingerprint": sh.fingerprint,
                "complete": complete && only.is_empty(),
                "extractor_version": ver,
                "rows": sh.ids.len(),
            }),
        );
    }
    store::write_sources(&a.out, &sources)?;

    let dt = t_run.elapsed().as_secs_f64();
    let n = n_ok + n_skip + n_err;
    eprintln!(
        "[corpus] done: {n} episodes (ok {n_ok}, skipped {n_skip}, error {n_err}) in {dt:.1}s = {:.2} eps/s ({:.3} s/episode/thread); rows: episodes {} daily_states {} daily_behaviour {} slim {}",
        n as f64 / dt.max(1e-9),
        dt * a.threads as f64 / (n.max(1)) as f64,
        st.tables[0].rows_total,
        st.tables[1].rows_total,
        st.tables[2].rows_total,
        st.tables[3].rows_total
    );
    if hash_ok + hash_bad > 0 {
        eprintln!("[corpus] stream-hash check vs GM csv: {hash_ok} match, {hash_bad} mismatch");
    }
    for e in &first_errors {
        eprintln!("[corpus] error sample: {e}");
    }
    if obs_mode {
        use std::sync::atomic::Ordering::Relaxed;
        eprintln!("[obs-check] rows checked {}; shared-column diffs {}", extract::OBS_CHECK_ROWS.load(Relaxed), extract::OBS_CHECK_DIFFS.load(Relaxed));
    }
    eprintln!(
        "[corpus] total wall {:.1}s; peak RSS {} MB; out {}",
        t0.elapsed().as_secs_f64(),
        util::peak_rss() >> 20,
        a.out.display()
    );
    Ok(())
}

fn run_work(w: &Work, shards: &[gm::Shard], slim_files: &[PathBuf], engine: &str, tx: &mpsc::SyncSender<Out>, slim_mode: bool, ver: &str) {
    let conv = |json: &str, m: Meta| -> Out {
        if slim_mode { slim::slim(json, m, engine, ver) } else { extract(json, m, engine, ver) }
    };
    match &w.loc {
        Loc::Gm { shard, rg } => {
            let sh = &shards[*shard];
            let by_id: HashMap<i64, &Meta> = w.metas.iter().map(|m| (m.eid, m)).collect();
            let wanted: HashSet<i64> = by_id.keys().copied().collect();
            let mut seen = HashSet::new();
            let res = sh.for_each_replay(*rg, &wanted, |eid, json| {
                seen.insert(eid);
                let m = (*by_id[&eid]).clone();
                let _ = tx.send(conv(json, m));
            });
            for m in &w.metas {
                if !seen.contains(&m.eid) {
                    let msg = match &res {
                        Err(e) => format!("error:read {e}"),
                        Ok(()) => "error:replay missing in row group".to_string(),
                    };
                    let _ = tx.send(Out::status_only(m.clone(), String::new(), msg));
                }
            }
        }
        Loc::Slim(fi) => {
            let by_id: HashMap<i64, &Meta> = w.metas.iter().map(|m| (m.eid, m)).collect();
            let wanted: HashSet<i64> = by_id.keys().copied().collect();
            let mut seen = HashSet::new();
            let res = slimsrc::for_each(&slim_files[*fi], &wanted, |eid, json| {
                seen.insert(eid);
                let m = (*by_id[&eid]).clone();
                let out = if ver == OBS_VERSION { extract::extract_obs(json, m, engine, ver) } else { extract::extract_slim(json, m, engine, ver) };
                let _ = tx.send(out);
            });
            for m in &w.metas {
                if !seen.contains(&m.eid) {
                    let msg = match &res { Err(e) => format!("error:read {e}"), Ok(()) => "error:slim record missing".to_string() };
                    let _ = tx.send(Out::status_only(m.clone(), String::new(), msg));
                }
            }
        }
        Loc::File(p) => {
            let mut m = w.metas[0].clone();
            m.file_sha = util::fingerprint(p).unwrap_or_default();
            if let Some(v) = files::peek_version(p) {
                if v != engine {
                    let st = format!("skipped:engine={v}");
                    let _ = tx.send(Out::status_only(m, v, st));
                    return;
                }
            }
            let out = match std::fs::read_to_string(p) {
                Ok(s) => conv(&s, m),
                Err(e) => Out::status_only(m, String::new(), format!("error:read {e}")),
            };
            let _ = tx.send(out);
        }
    }
}
