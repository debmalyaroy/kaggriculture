//! Directory adapters: official daily datasets (`<id>.json` + manifest.csv) and plain
//! replay directories (our self-play, same Kaggle replay format).
use crate::util::{find_date, hashed_id, mtime_iso};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct FileEp {
    pub eid: i64,
    pub path: PathBuf,
    pub end_time: String,
    /// Daily manifest ratings: the pair is {min, 2*avg - min}; seat order unknown.
    pub rating_min: Option<f64>,
    pub rating_max: Option<f64>,
}

fn json_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        v.sort();
        for p in v {
            if p.is_dir() {
                if depth > 0 {
                    json_files(&p, depth - 1, out);
                }
            } else if p.extension().and_then(|e| e.to_str()) == Some("json") {
                out.push(p);
            }
        }
    }
}

fn stem_id(p: &Path) -> i64 {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    stem.parse::<i64>().unwrap_or_else(|_| hashed_id(stem))
}

struct ManRow {
    time: Option<String>,
    avg: Option<f64>,
    min: Option<f64>,
}

fn read_manifest(path: &Path) -> HashMap<i64, ManRow> {
    let mut out = HashMap::new();
    let mut r = match csv::Reader::from_path(path) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let h = match r.headers() {
        Ok(h) => h.clone(),
        Err(_) => return out,
    };
    let c = |n: &str| h.iter().position(|x| x == n);
    let (ei, ti, ai, mi) = (
        c("episode_id"),
        c("end_time").or(c("create_time")).or(c("date")),
        c("avg_score"),
        c("min_score"),
    );
    let ei = match ei {
        Some(e) => e,
        None => return out,
    };
    let mut rec = csv::StringRecord::new();
    while let Ok(true) = r.read_record(&mut rec) {
        if let Some(e) = rec.get(ei).and_then(|x| x.trim().parse::<f64>().ok()) {
            let f = |i: Option<usize>| i.and_then(|i| rec.get(i)).and_then(|x| x.trim().parse::<f64>().ok());
            out.insert(
                e as i64,
                ManRow {
                    time: ti.and_then(|i| rec.get(i)).map(|s| s.to_string()).filter(|s| !s.is_empty()),
                    avg: f(ai),
                    min: f(mi),
                },
            );
        }
    }
    out
}

/// A directory is a daily dataset if it holds a manifest.csv and/or `<id>.json` files; a
/// parent of several daily datasets (e.g. `/kaggle/input`) is expanded one level.
pub fn daily_dirs(root: &Path) -> Vec<PathBuf> {
    let has_own = root.join("manifest.csv").exists()
        || fs::read_dir(root)
            .map(|rd| {
                rd.flatten().any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
            })
            .unwrap_or(false);
    if has_own {
        return vec![root.to_path_buf()];
    }
    let mut v: Vec<PathBuf> = fs::read_dir(root)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    v.sort();
    v.into_iter().flat_map(|p| daily_dirs_shallow(&p)).collect()
}

fn daily_dirs_shallow(p: &Path) -> Vec<PathBuf> {
    let direct = p.join("manifest.csv").exists();
    if direct {
        return vec![p.to_path_buf()];
    }
    // Kaggle mounts sometimes nest the files one level down
    fs::read_dir(p)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|q| q.is_dir() && q.join("manifest.csv").exists())
                .collect()
        })
        .unwrap_or_default()
}

pub fn scan_daily(dir: &Path) -> Result<Vec<FileEp>> {
    let man = read_manifest(&dir.join("manifest.csv"));
    let dir_date = find_date(&dir.to_string_lossy());
    let mut files = Vec::new();
    json_files(dir, 1, &mut files);
    Ok(files
        .into_iter()
        .map(|p| {
            let eid = stem_id(&p);
            let m = man.get(&eid);
            let end_time = m
                .and_then(|m| m.time.clone())
                .or_else(|| dir_date.clone())
                .unwrap_or_else(|| mtime_iso(&p));
            let (rmin, rmax) = match m {
                Some(ManRow { avg: Some(a), min: Some(mn), .. }) => (Some(*mn), Some(2.0 * a - mn)),
                _ => (None, None),
            };
            FileEp { eid, path: p, end_time, rating_min: rmin, rating_max: rmax }
        })
        .collect())
}

pub fn scan_replays(dir: &Path) -> Result<Vec<FileEp>> {
    let mut files = Vec::new();
    json_files(dir, 3, &mut files);
    Ok(files
        .into_iter()
        .map(|p| {
            let eid = stem_id(&p);
            let end_time = mtime_iso(&p);
            FileEp { eid, path: p, end_time, rating_min: None, rating_max: None }
        })
        .collect())
}

/// `module_version` from the first 64 KiB of a replay file (top-level keys come sorted,
/// so it precedes `steps`). None if not found there.
pub fn peek_version(path: &Path) -> Option<String> {
    let mut f = fs::File::open(path).ok()?;
    let mut buf = vec![0u8; 64 * 1024];
    let n = f.read(&mut buf).ok()?;
    let s = String::from_utf8_lossy(&buf[..n]);
    let k = s.find("\"module_version\"")?;
    let rest = &s[k + 16..];
    let q1 = rest.find('"')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(rest[..q2].to_string())
}
