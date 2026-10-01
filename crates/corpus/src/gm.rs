//! GM dataset adapter (`D:\gm_dataset` layout): replay shards + CSV side tables.
//!
//! The CSVs are streamed record by record and only the columns / ids that are needed are
//! kept, as compact id -> value maps.
use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Int64Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::ProjectionMask;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

/// Side-table facts about one GM episode.
#[derive(Clone, Debug, Default)]
pub struct GmEp {
    pub end_time: String,
    pub episode_type: String,
    pub sub: [Option<i64>; 2],
    pub team: [Option<i64>; 2],
    pub rating: [Option<f64>; 2],
}

pub fn shard_files(gm: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = fs::read_dir(gm)
        .with_context(|| format!("read_dir {}", gm.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("replays_") && n.ends_with(".parquet"))
                .unwrap_or(false)
        })
        .collect();
    v.sort();
    v.reverse();
    Ok(v)
}

fn csv_reader(path: &Path) -> Result<(csv::Reader<File>, csv::StringRecord)> {
    let mut r = csv::Reader::from_path(path).with_context(|| format!("open {}", path.display()))?;
    let h = r.headers()?.clone();
    Ok((r, h))
}

fn col(h: &csv::StringRecord, name: &str) -> Result<usize> {
    h.iter().position(|x| x == name).ok_or_else(|| anyhow!("column {name} missing"))
}

fn opt_i64(s: Option<&str>) -> Option<i64> {
    s.and_then(|x| x.trim().parse::<f64>().ok()).map(|f| f as i64)
}

fn opt_f64(s: Option<&str>) -> Option<f64> {
    s.and_then(|x| x.trim().parse::<f64>().ok())
}

/// episode_id -> engine_version is 1.32.7 (true) / something else (false). Ids absent
/// from the CSV are unknown (decided later from the replay's own module_version).
pub fn load_engine(gm: &Path, keep: &str) -> Result<HashMap<i64, bool>> {
    let path = gm.join("episode_features.csv");
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let (mut r, h) = csv_reader(&path)?;
    let (ei, vi) = (col(&h, "episode_id")?, col(&h, "engine_version")?);
    let mut m = HashMap::new();
    let mut rec = csv::StringRecord::new();
    while r.read_record(&mut rec)? {
        if let Some(e) = opt_i64(rec.get(ei)) {
            let v = rec.get(vi).unwrap_or("");
            if !v.is_empty() {
                m.insert(e, v == keep);
            }
        }
    }
    Ok(m)
}

/// Side facts for the wanted ids only.
pub fn load_episodes(gm: &Path, want: &HashSet<i64>) -> Result<HashMap<i64, GmEp>> {
    let path = gm.join("episodes.csv");
    let mut out = HashMap::with_capacity(want.len());
    if !path.exists() {
        return Ok(out);
    }
    let (mut r, h) = csv_reader(&path)?;
    let ei = col(&h, "episode_id")?;
    let et = col(&h, "end_time")?;
    let ty = col(&h, "type").ok();
    let c = |n: &str| h.iter().position(|x| x == n);
    let (s0, s1, t0, t1, r0, r1) =
        (c("sub_0"), c("sub_1"), c("team_0"), c("team_1"), c("rating_0"), c("rating_1"));
    fn g(rec: &csv::StringRecord, i: Option<usize>) -> Option<&str> {
        i.and_then(|i| rec.get(i))
    }
    let mut rec = csv::StringRecord::new();
    while r.read_record(&mut rec)? {
        let e = match opt_i64(rec.get(ei)) {
            Some(e) if want.contains(&e) => e,
            _ => continue,
        };
        out.insert(
            e,
            GmEp {
                end_time: rec.get(et).unwrap_or("").to_string(),
                episode_type: g(&rec, ty).unwrap_or("").to_string(),
                sub: [opt_i64(g(&rec, s0)), opt_i64(g(&rec, s1))],
                team: [opt_i64(g(&rec, t0)), opt_i64(g(&rec, t1))],
                rating: [opt_f64(g(&rec, r0)), opt_f64(g(&rec, r1))],
            },
        );
    }
    Ok(out)
}

/// GM stream-hash cuts (24/48/100/136) for the wanted ids: eid -> [seat][cut] as u64.
pub fn load_hashes(gm: &Path, want: &HashSet<i64>) -> Result<HashMap<i64, [[Option<u64>; 4]; 2]>> {
    let path = gm.join("stream_hashes.csv");
    let mut out = HashMap::new();
    if !path.exists() {
        return Ok(out);
    }
    let (mut r, h) = csv_reader(&path)?;
    let ei = col(&h, "episode_id")?;
    let si = col(&h, "seat")?;
    let cuts: Vec<Option<usize>> = features::consts::HASH_CUTS
        .iter()
        .map(|c| h.iter().position(|x| x == format!("stream_h{c}")))
        .collect();
    let mut rec = csv::StringRecord::new();
    while r.read_record(&mut rec)? {
        let e = match opt_i64(rec.get(ei)) {
            Some(e) if want.contains(&e) => e,
            _ => continue,
        };
        let seat = match opt_i64(rec.get(si)) {
            Some(s) if (0..2).contains(&s) => s as usize,
            _ => continue,
        };
        let ent = out.entry(e).or_insert([[None; 4]; 2]);
        for (k, ci) in cuts.iter().enumerate() {
            ent[seat][k] = ci
                .and_then(|i| rec.get(i))
                .and_then(|x| u64::from_str_radix(x.trim(), 16).ok());
        }
    }
    Ok(out)
}

/// submission_id -> coverage for the wanted submissions.
pub fn load_coverage(gm: &Path, want: &HashSet<i64>) -> Result<HashMap<i64, f64>> {
    let path = gm.join("per_submission_coverage.csv");
    let mut out = HashMap::new();
    if !path.exists() {
        return Ok(out);
    }
    let (mut r, h) = csv_reader(&path)?;
    let (si, ci) = (col(&h, "submission_id")?, col(&h, "coverage")?);
    let mut rec = csv::StringRecord::new();
    while r.read_record(&mut rec)? {
        if let (Some(s), Some(c)) = (opt_i64(rec.get(si)), opt_f64(rec.get(ci))) {
            if want.contains(&s) {
                out.insert(s, c);
            }
        }
    }
    Ok(out)
}

/// One opened shard: footer metadata (shared by all readers) and the id -> row-group map.
pub struct Shard {
    pub path: PathBuf,
    pub fingerprint: String,
    pub meta: ArrowReaderMetadata,
    /// (episode_id, row_group) for every row, file order.
    pub ids: Vec<(i64, usize)>,
}

fn projection(meta: &ArrowReaderMetadata, names: &[&str]) -> ProjectionMask {
    let roots: Vec<usize> = names.iter().filter_map(|n| meta.schema().index_of(n).ok()).collect();
    ProjectionMask::roots(meta.metadata().file_metadata().schema_descr(), roots)
}

impl Shard {
    pub fn open(path: &Path, fingerprint: String) -> Result<Shard> {
        let f = File::open(path)?;
        let meta = ArrowReaderMetadata::load(&f, ArrowReaderOptions::default())?;
        let mut ids = Vec::new();
        let nrg = meta.metadata().num_row_groups();
        let mask = projection(&meta, &["episode_id"]);
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(f, meta.clone())
            .with_projection(mask)
            .with_batch_size(65_536)
            .build()?;
        // row groups are read in order; map each row back to its group by row counts
        let mut rg_ends = Vec::with_capacity(nrg);
        let mut acc = 0usize;
        for i in 0..nrg {
            acc += meta.metadata().row_group(i).num_rows() as usize;
            rg_ends.push(acc);
        }
        let mut row = 0usize;
        let mut rg = 0usize;
        for b in reader {
            let b = b?;
            let c = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("episode_id not int64"))?;
            for i in 0..c.len() {
                while rg < nrg && row >= rg_ends[rg] {
                    rg += 1;
                }
                ids.push((c.value(i), rg));
                row += 1;
            }
        }
        Ok(Shard { path: path.to_path_buf(), fingerprint, meta, ids })
    }

    /// Stream the replays of one row group; calls `f(episode_id, replay_json)` for the
    /// wanted rows only, one decoded replay at a time.
    pub fn for_each_replay(
        &self,
        rg: usize,
        wanted: &HashSet<i64>,
        mut f: impl FnMut(i64, &str),
    ) -> Result<()> {
        let file = File::open(&self.path)?;
        let mask = projection(&self.meta, &["episode_id", "replay_json"]);
        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.meta.clone())
            .with_projection(mask)
            .with_row_groups(vec![rg])
            .with_batch_size(1)
            .build()?;
        for b in reader {
            let b = b?;
            let ids = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("episode_id not int64"))?;
            let js = b.column(1);
            for i in 0..b.num_rows() {
                let e = ids.value(i);
                if !wanted.contains(&e) {
                    continue;
                }
                if let Some(s) = get_str(js.as_ref(), i) {
                    f(e, s);
                }
            }
        }
        Ok(())
    }
}

fn get_str(arr: &dyn Array, i: usize) -> Option<&str> {
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        return (!a.is_null(i)).then(|| a.value(i));
    }
    if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
        return (!a.is_null(i)).then(|| a.value(i));
    }
    None
}
