//! Output store: partitioned Parquet tables, the append-only ledger, sources.json and
//! schema.json.
//!
//! Layout under `<out>`:
//!   episodes/date=YYYY-MM-DD/part-<tag>.parquet        (same for daily_states, daily_behaviour)
//!   ledger/part-<tag>.parquet
//!   sources.json, schema.json
//! `<tag>` = `<run>-<seq>` names one checkpoint. A checkpoint's data files are renamed into
//! place first and its ledger part last, so on start-up any data file whose tag has no
//! ledger part is an interrupted checkpoint and is deleted (its episodes get redone).
use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use features::{Row, Val};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const TABLES: [&str; 5] = ["episodes", "daily_states", "daily_behaviour", "slim", "day_obs"];

#[derive(Clone, Debug, PartialEq)]
pub struct ColSpec {
    pub name: String,
    pub ty: &'static str,
    pub doc: &'static str,
}

fn ty_of(v: &Val) -> &'static str {
    match v {
        Val::I(_) => "int64",
        Val::F(_) => "float64",
        Val::S(_) => "string",
    }
}

pub fn spec_of(row: &Row) -> Vec<ColSpec> {
    row.cols
        .iter()
        .map(|c| ColSpec { name: c.name.clone(), ty: ty_of(&c.val), doc: c.doc })
        .collect()
}

enum Buf {
    I(Vec<Option<i64>>),
    F(Vec<Option<f64>>),
    S(Vec<Option<String>>),
}

struct Part {
    bufs: Vec<Buf>,
    rows: usize,
}

/// Column buffers for one table, bucketed by partition date.
pub struct Table {
    pub name: &'static str,
    pub spec: Option<Vec<ColSpec>>,
    parts: BTreeMap<String, Part>,
    pub rows_total: usize,
}

impl Table {
    fn new(name: &'static str) -> Self {
        Table { name, spec: None, parts: BTreeMap::new(), rows_total: 0 }
    }

    pub fn push(&mut self, date: &str, row: Row) -> Result<()> {
        if self.spec.is_none() {
            self.spec = Some(spec_of(&row));
        }
        let spec = self.spec.as_ref().unwrap();
        anyhow::ensure!(
            spec.len() == row.cols.len()
                && spec.iter().zip(&row.cols).all(|(s, c)| s.name == c.name && s.ty == ty_of(&c.val)),
            "table {}: row layout differs from the schema",
            self.name
        );
        let part = self.parts.entry(date.to_string()).or_insert_with(|| Part {
            bufs: spec
                .iter()
                .map(|s| match s.ty {
                    "int64" => Buf::I(Vec::new()),
                    "float64" => Buf::F(Vec::new()),
                    _ => Buf::S(Vec::new()),
                })
                .collect(),
            rows: 0,
        });
        for (b, c) in part.bufs.iter_mut().zip(row.cols) {
            match (b, c.val) {
                (Buf::I(v), Val::I(x)) => v.push(x),
                (Buf::F(v), Val::F(x)) => v.push(x),
                (Buf::S(v), Val::S(x)) => v.push(x),
                _ => unreachable!("type checked above"),
            }
        }
        part.rows += 1;
        self.rows_total += 1;
        Ok(())
    }

    fn arrow_schema(&self) -> Arc<Schema> {
        let spec = self.spec.as_ref().unwrap();
        Arc::new(Schema::new(
            spec.iter()
                .map(|s| {
                    let dt = match s.ty {
                        "int64" => DataType::Int64,
                        "float64" => DataType::Float64,
                        _ => DataType::Utf8,
                    };
                    Field::new(&s.name, dt, true)
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// Write every buffered partition as `<out>/<table>/date=<d>/part-<tag>.parquet`.
    fn flush(&mut self, out: &Path, tag: &str, rows_per_group: usize) -> Result<usize> {
        if self.parts.is_empty() {
            return Ok(0);
        }
        let schema = self.arrow_schema();
        let mut files = 0;
        for (date, part) in std::mem::take(&mut self.parts) {
            let cols: Vec<ArrayRef> = part
                .bufs
                .into_iter()
                .map(|b| -> ArrayRef {
                    match b {
                        Buf::I(v) => Arc::new(Int64Array::from(v)),
                        Buf::F(v) => Arc::new(Float64Array::from(v)),
                        Buf::S(v) => Arc::new(StringArray::from(v)),
                    }
                })
                .collect();
            let batch = RecordBatch::try_new(schema.clone(), cols)?;
            let dir = out.join(self.name).join(format!("date={date}"));
            fs::create_dir_all(&dir)?;
            let fin = dir.join(format!("part-{tag}.parquet"));
            let tmp = dir.join(format!("part-{tag}.parquet.tmp"));
            write_parquet(&tmp, &batch, rows_per_group)?;
            fs::rename(&tmp, &fin)?;
            files += 1;
        }
        Ok(files)
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch, rows_per_group: usize) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(6)?))
        .set_max_row_group_row_count(Some(rows_per_group.max(1)))
        .build();
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    w.write(batch)?;
    w.close()?;
    Ok(())
}

/// One ledger line.
#[derive(Clone, Debug)]
pub struct LedgerRow {
    pub episode_id: i64,
    pub source: String,
    pub source_file: String,
    pub source_file_sha: String,
    pub engine_version: String,
    pub end_date: String,
    pub status: String,
}

pub struct Store {
    pub out: PathBuf,
    pub run: String,
    pub seq: usize,
    pub rows_per_group: usize,
    pub extractor_version: String,
    pub tables: Vec<Table>,
    pub pending: Vec<LedgerRow>,
}

impl Store {
    pub fn new(out: &Path, run: String, rows_per_group: usize, extractor_version: &str) -> Self {
        Store {
            out: out.to_path_buf(),
            run,
            seq: 0,
            rows_per_group,
            extractor_version: extractor_version.to_string(),
            tables: TABLES.iter().map(|n| Table::new(n)).collect(),
            pending: Vec::new(),
        }
    }

    pub fn table(&mut self, name: &str) -> &mut Table {
        self.tables.iter_mut().find(|t| t.name == name).expect("known table")
    }

    /// Close the current checkpoint: data files first, ledger part last.
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let tag = format!("{}-{:05}", self.run, self.seq);
        for t in self.tables.iter_mut() {
            t.flush(&self.out, &tag, self.rows_per_group)?;
        }
        self.write_ledger(&tag)?;
        self.pending.clear();
        self.seq += 1;
        Ok(())
    }

    fn write_ledger(&self, tag: &str) -> Result<()> {
        let dir = self.out.join("ledger");
        fs::create_dir_all(&dir)?;
        let p = &self.pending;
        let s = |f: fn(&LedgerRow) -> String| -> ArrayRef {
            Arc::new(StringArray::from(p.iter().map(|r| Some(f(r))).collect::<Vec<_>>()))
        };
        let konst = |v: &str| -> ArrayRef {
            Arc::new(StringArray::from(p.iter().map(|_| Some(v.to_string())).collect::<Vec<_>>()))
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("episode_id", DataType::Int64, false),
            Field::new("source", DataType::Utf8, true),
            Field::new("source_file", DataType::Utf8, true),
            Field::new("source_file_sha", DataType::Utf8, true),
            Field::new("engine_version", DataType::Utf8, true),
            Field::new("end_date", DataType::Utf8, true),
            Field::new("extractor_version", DataType::Utf8, true),
            Field::new("status", DataType::Utf8, true),
            Field::new("checkpoint", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(p.iter().map(|r| r.episode_id).collect::<Vec<_>>())),
                s(|r| r.source.clone()),
                s(|r| r.source_file.clone()),
                s(|r| r.source_file_sha.clone()),
                s(|r| r.engine_version.clone()),
                s(|r| r.end_date.clone()),
                konst(&self.extractor_version),
                s(|r| r.status.clone()),
                konst(tag),
            ],
        )?;
        let fin = dir.join(format!("part-{tag}.parquet"));
        let tmp = dir.join(format!("part-{tag}.parquet.tmp"));
        write_parquet(&tmp, &batch, 1 << 20)?;
        fs::rename(&tmp, &fin)?;
        Ok(())
    }
}

/// Ledger summary read at start-up.
#[derive(Default)]
pub struct LedgerState {
    /// ids with an ok / skipped status under the current extractor version
    pub done: HashSet<i64>,
    /// checkpoint tags that have a ledger part
    pub tags: HashSet<String>,
    pub n_rows: usize,
    pub n_ok: usize,
}

fn str_col<'a>(b: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    b.column_by_name(name).and_then(|c| c.as_any().downcast_ref::<StringArray>())
}

pub fn read_ledger(out: &Path, extractor_version: &str) -> Result<LedgerState> {
    let mut st = LedgerState::default();
    let dir = out.join("ledger");
    let rd = match fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Ok(st),
    };
    let mut ok_ids: HashMap<i64, bool> = HashMap::new();
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        if name.ends_with(".tmp") {
            let _ = fs::remove_file(&p);
            continue;
        }
        let tag = match name.strip_prefix("part-").and_then(|n| n.strip_suffix(".parquet")) {
            Some(t) => t.to_string(),
            None => continue,
        };
        let reader = match File::open(&p)
            .map_err(anyhow::Error::from)
            .and_then(|f| Ok(ParquetRecordBatchReaderBuilder::try_new(f)?.build()?))
        {
            Ok(r) => r,
            Err(_) => {
                eprintln!("[corpus] removing unreadable ledger part {name}");
                let _ = fs::remove_file(&p);
                continue;
            }
        };
        st.tags.insert(tag);
        for b in reader {
            let b = b?;
            let ids = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let (ver, status) = (str_col(&b, "extractor_version"), str_col(&b, "status"));
            for i in 0..b.num_rows() {
                st.n_rows += 1;
                let v = ver.map(|a| a.value(i)).unwrap_or("");
                let s = status.map(|a| a.value(i)).unwrap_or("");
                let good = v == extractor_version && (s == "ok" || s.starts_with("skipped"));
                let e = ok_ids.entry(ids.value(i)).or_insert(false);
                *e |= good;
                if good && s == "ok" {
                    st.n_ok += 1;
                }
            }
        }
    }
    st.done = ok_ids.into_iter().filter(|(_, g)| *g).map(|(k, _)| k).collect();
    Ok(st)
}

/// Delete data files from checkpoints that never got their ledger part (and stray .tmp).
pub fn remove_orphans(out: &Path, tags: &HashSet<String>) -> usize {
    let mut n = 0;
    for t in TABLES {
        let tdir = out.join(t);
        let Ok(rd) = fs::read_dir(&tdir) else { continue };
        for d in rd.flatten() {
            let Ok(files) = fs::read_dir(d.path()) else { continue };
            for f in files.flatten() {
                let name = f.file_name().to_string_lossy().to_string();
                let tag = name.strip_prefix("part-").and_then(|n| n.strip_suffix(".parquet"));
                let orphan = name.ends_with(".tmp") || tag.map(|t| !tags.contains(t)).unwrap_or(false);
                if orphan && fs::remove_file(f.path()).is_ok() {
                    n += 1;
                }
            }
        }
    }
    n
}

/// `sources.json`: per GM shard fingerprint + whether every eligible id in it is done.
pub type Sources = BTreeMap<String, serde_json::Value>;

pub fn read_sources(out: &Path) -> Sources {
    fs::read(out.join("sources.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

pub fn write_sources(out: &Path, s: &Sources) -> Result<()> {
    let tmp = out.join("sources.json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(s)?)?;
    fs::rename(&tmp, out.join("sources.json"))?;
    Ok(())
}

pub fn write_schema(out: &Path, schema_version: u32, extractor_version: &str, tables: &[Table]) -> Result<()> {
    let path = out.join("schema.json");
    let mut existing: serde_json::Value = fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::json!({}));
    existing["SCHEMA_VERSION"] = serde_json::json!(schema_version);
    existing["extractor_version"] = serde_json::json!(extractor_version);
    existing["partitioning"] = serde_json::json!(
        "<table>/date=YYYY-MM-DD/part-<run>-<seq>.parquet (date = episode end date, UTC); zstd"
    );
    if existing.get("tables").is_none() {
        existing["tables"] = serde_json::json!({});
    }
    for t in tables {
        if let Some(spec) = &t.spec {
            existing["tables"][t.name] = serde_json::Value::Array(
                spec.iter()
                    .map(|c| serde_json::json!({"name": c.name, "type": c.ty, "doc": c.doc}))
                    .collect(),
            );
        }
    }
    fs::write(path, serde_json::to_vec_pretty(&existing)?)?;
    Ok(())
}
