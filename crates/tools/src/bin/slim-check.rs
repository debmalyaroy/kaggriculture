//! slim-check: engine parity against REAL games.
//!
//!   slim-check <slim parquet file or dir> [--max N]
//!
//! For every slim record: State::new(seed), then for t in 0..n-1 apply the recorded actions
//! (steps[t+1].a[seat] is the action applied at engine step t) with kagg-engine's step
//! function, and require both farms' money after every step to equal the replay's
//! steps[t+1].f[seat].money exactly. The replays were produced by Kaggle's official engine,
//! so a clean pass = our engine reproduces real ladder games bit for bit.
use anyhow::{Context, Result};
use arrow::array::{Array, StringArray};
use kagg_engine::engine::{self, PlayerAction, UnitAction};
use kagg_engine::state::State;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Kaggle action element -> engine token (strings verbatim, numbers through int()).
fn tok(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.as_i64().map(|i| i.to_string()).unwrap_or_else(|| (n.as_f64().unwrap_or(0.0).trunc() as i64).to_string()),
        Value::Bool(b) => (*b as i64).to_string(),
        _ => "?".into(),
    }
}

fn unit(v: &Value) -> UnitAction {
    match v.as_array() {
        Some(a) if !a.is_empty() => {
            let t: Vec<String> = a.iter().map(tok).collect();
            let r: Vec<&str> = t.iter().map(String::as_str).collect();
            UnitAction::parse(&r)
        }
        _ => UnitAction::parse(&[]),
    }
}

/// Mirrors kaggsim.tape.action_to_line: non-dict = empty action; positional hands/market.
fn action(v: &Value) -> PlayerAction {
    let Some(o) = v.as_object() else { return PlayerAction::default() };
    let farmer = match o.get("farmer") {
        Some(f) if f.is_array() && !f.as_array().unwrap().is_empty() => unit(f),
        _ => UnitAction::parse(&["PASS"]),
    };
    let hands = o.get("hands").and_then(|h| h.as_array()).map(|h| h.iter().map(unit).collect()).unwrap_or_default();
    let market = o
        .get("market")
        .and_then(|m| m.as_array())
        .map(|m| m.iter().map(|x| x.as_array().map(|a| a.iter().map(tok).collect()).unwrap_or_default()).collect())
        .unwrap_or_default();
    PlayerAction { farmer, hands, market }
}

fn check(doc: &Value) -> Result<Option<(usize, String)>> {
    let seed = doc["info"]["seed"].as_i64().context("no seed")?;
    let steps = doc["steps"].as_array().context("no steps")?;
    let n = steps.len();
    let mut st = State::new(seed);
    for t in 0..n - 1 {
        let a = &steps[t + 1]["a"];
        engine::step(&mut st, &[action(&a[0]), action(&a[1])]);
        for s in 0..2 {
            let want = steps[t + 1]["f"][s]["money"].as_f64().unwrap_or(f64::NAN);
            if st.farms[s].money != want {
                return Ok(Some((t, format!("seat {s}: engine {} replay {}", st.farms[s].money, want))));
            }
        }
    }
    Ok(None)
}

fn files(p: &Path, out: &mut Vec<PathBuf>) {
    if p.is_dir() {
        if let Ok(rd) = std::fs::read_dir(p) {
            let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            v.sort();
            for q in v {
                files(&q, out);
            }
        }
    } else if p.extension().and_then(|e| e.to_str()) == Some("parquet") && p.to_string_lossy().contains("slim") && !p.to_string_lossy().contains("ledger") {
        out.push(p.to_path_buf());
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(args.get(1).context("usage: slim-check <file|dir> [--max N]")?);
    let max: usize = args.iter().position(|a| a == "--max").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let mut fs_ = Vec::new();
    files(&root, &mut fs_);
    let (mut ok, mut bad, mut n) = (0usize, 0usize, 0usize);
    'outer: for f in fs_ {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&f)?)?.with_batch_size(8).build()?;
        for b in reader {
            let b = b?;
            let ids = b.column_by_name("episode_id").context("episode_id")?;
            let slim = b.column_by_name("slim").context("slim")?.as_any().downcast_ref::<StringArray>().context("slim utf8")?;
            for i in 0..b.num_rows() {
                if n >= max {
                    break 'outer;
                }
                n += 1;
                let doc: Value = serde_json::from_str(slim.value(i))?;
                let eid = arrow::array::cast::as_primitive_array::<arrow::datatypes::Int64Type>(ids.as_ref()).value(i);
                match check(&doc)? {
                    None => ok += 1,
                    Some((t, msg)) => {
                        bad += 1;
                        println!("MISMATCH episode {eid} first at step {t}: {msg}");
                    }
                }
            }
        }
    }
    println!("slim-check: {n} episodes, {ok} bit-exact every step, {bad} mismatched");
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}
