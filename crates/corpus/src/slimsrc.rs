//! `--slim DIR` input adapter: slim-corpus Parquet files -> features.
use anyhow::{Context, Result};
use arrow::array::{Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

pub struct SlimRow {
    pub eid: i64,
    pub end_date: String,
    pub end_time: String,
    pub source: String,
    pub episode_type: String,
    pub sub: [Option<i64>; 2],
    pub rating: [Option<f64>; 2],
    pub rating_min: Option<f64>,
    pub rating_max: Option<f64>,
    pub coverage: [Option<f64>; 2],
}

/// Every slim data file under `root` (skips ledger/ and .tmp).
pub fn files(root: &Path) -> Vec<PathBuf> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if p.is_dir() {
            if p.file_name().and_then(|n| n.to_str()) == Some("ledger") {
                return;
            }
            if let Ok(rd) = std::fs::read_dir(p) {
                let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                v.sort();
                for q in v {
                    walk(&q, out);
                }
            }
        } else if p.extension().and_then(|e| e.to_str()) == Some("parquet") {
            out.push(p.to_path_buf());
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

fn open(path: &Path, cols: &[&str], batch: usize) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = b.parquet_schema();
    let idx: Vec<usize> = cols
        .iter()
        .filter_map(|c| (0..schema.num_columns()).find(|&i| schema.column(i).name() == *c))
        .collect();
    let mask = ProjectionMask::leaves(schema, idx);
    Ok(b.with_projection(mask).with_batch_size(batch).build()?)
}

fn f(b: &arrow::record_batch::RecordBatch, name: &str, i: usize) -> Option<f64> {
    b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<arrow::array::Float64Array>()).filter(|a| !a.is_null(i)).map(|a| a.value(i))
}

fn n(b: &arrow::record_batch::RecordBatch, name: &str, i: usize) -> Option<i64> {
    b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<Int64Array>()).filter(|a| !a.is_null(i)).map(|a| a.value(i))
}

fn s(b: &arrow::record_batch::RecordBatch, name: &str, i: usize) -> String {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .filter(|a| !a.is_null(i))
        .map(|a| a.value(i).to_string())
        .unwrap_or_default()
}

/// Ids and dates of one file (light: no slim column).
pub fn index(path: &Path) -> Result<Vec<SlimRow>> {
    let mut out = Vec::new();
    let cols = ["episode_id", "end_date", "end_time", "source", "episode_type", "submission_id_0", "submission_id_1", "rating_after_0", "rating_after_1", "rating_min", "rating_max", "coverage_0", "coverage_1"];
    for b in open(path, &cols, 8192)? {
        let b = b?;
        let ids = b.column_by_name("episode_id").context("episode_id")?.as_any().downcast_ref::<Int64Array>().context("i64")?;
        for i in 0..b.num_rows() {
            out.push(SlimRow {
                eid: ids.value(i), end_date: s(&b, "end_date", i), end_time: s(&b, "end_time", i), source: s(&b, "source", i),
                episode_type: s(&b, "episode_type", i),
                sub: [n(&b, "submission_id_0", i), n(&b, "submission_id_1", i)],
                rating: [f(&b, "rating_after_0", i), f(&b, "rating_after_1", i)],
                rating_min: f(&b, "rating_min", i), rating_max: f(&b, "rating_max", i),
                coverage: [f(&b, "coverage_0", i), f(&b, "coverage_1", i)],
            });
        }
    }
    Ok(out)
}

/// Stream the slim JSON of the wanted ids in one file (4 records per batch: bounded memory).
pub fn for_each(path: &Path, wanted: &HashSet<i64>, mut f: impl FnMut(i64, &str)) -> Result<()> {
    for b in open(path, &["episode_id", "slim"], 4)? {
        let b = b?;
        let ids = b.column_by_name("episode_id").context("episode_id")?.as_any().downcast_ref::<Int64Array>().context("i64")?;
        let js = b.column_by_name("slim").context("slim")?.as_any().downcast_ref::<StringArray>().context("utf8")?;
        for i in 0..b.num_rows() {
            let e = ids.value(i);
            if wanted.contains(&e) && !js.is_null(i) {
                f(e, js.value(i));
            }
        }
    }
    Ok(())
}
