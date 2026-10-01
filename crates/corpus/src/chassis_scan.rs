//! chassis-scan: the economy of EVERY game in the slim corpus, both seats, per realized world (v63.12 chassis C1).
//!
//!     chassis-scan --slim data/slim/s1 --out data/chassis/c1_scan.tsv [--threads 8]
//!
//! No replay is needed: each slim record keeps per step both seats' actions (`a`), both farms' money (`f[s].money`) and
//! the town's unlocked shops when they change (`tw`). Files are processed in parallel; each thread streams its file
//! 4 records at a time (slimsrc::for_each), so memory stays bounded. One row per (episode, seat):
//!   eid seat rating source date n world bank_d6 bank_d12 bank_d18 bank_d24 bank_d30 opp_d30 farm24 farm72 farm144
//! world = the first two unlocked shops at the end; bank_dD = money after day D (step min(24 D, n-1));
//! farmN = hash of this seat's farmer + hand actions over steps 1..N (what a chassis route fixes; games with the same
//! farm144 play the same opening and can switch continuations at step 144). Duplicated episode ids (re-extracts) are
//! written once per file; dedupe on eid downstream.
#[path = "slimsrc.rs"]
mod slimsrc;

use serde_json::Value;
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn fnv(h: u64, s: &str) -> u64 {
    s.bytes().fold(h, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

#[derive(serde::Deserialize)]
struct Money {
    #[serde(default)]
    money: Option<f64>,
    #[serde(default)]
    hands: Option<Vec<Value>>,
    #[serde(default)]
    unlocked_quadrants: Option<Vec<String>>,
}

/// Day-6 farm state keys of seat `s` from the tile snapshot at step 145 (the state the route switch at 144 lands in):
/// layout6 = every tile's LOCKED / empty / PLANT:crop / PASTURE:animal in place (weeds count as empty) + hands +
/// unlocked quadrants; econ6 = the same contents as counts (positions ignored).
fn day6_keys(tl: &Value, s: usize, hands: usize, quads: &[String]) -> (u64, u64) {
    let mut lay = 0xcbf2_9ce4_8422_2325u64;
    let mut cnt: std::collections::BTreeMap<String, u32> = Default::default();
    if let Some(rows) = tl.get(s).and_then(|x| x.as_array()) {
        for row in rows {
            for t in row.as_array().into_iter().flatten() {
                let k = match t {
                    Value::Null => "_".to_string(),
                    Value::String(x) => x.clone(),
                    Value::Object(o) => {
                        let kind = o.get("kind").and_then(|x| x.as_str()).unwrap_or("?");
                        let sub = o.get("crop").or_else(|| o.get("animal")).and_then(|x| x.as_str()).unwrap_or("");
                        if kind == "WEED" || sub == "WEED" { "_".to_string() } else { format!("{kind}:{sub}") }
                    }
                    _ => "?".to_string(),
                };
                lay = fnv(lay, &k);
                lay = fnv(lay, ";");
                if k != "_" && k != "LOCKED" {
                    *cnt.entry(k).or_default() += 1;
                }
            }
        }
    }
    let tail = format!("|h{hands}|q{}", quads.join(","));
    lay = fnv(lay, &tail);
    let econ = fnv(0xcbf2_9ce4_8422_2325u64, &format!("{cnt:?}{tail}"));
    (lay, econ)
}

#[derive(serde::Deserialize)]
struct Step<'a> {
    #[serde(borrow, default)]
    a: Option<Vec<&'a serde_json::value::RawValue>>,
    #[serde(default)]
    tw: Option<Vec<String>>,
    #[serde(default)]
    f: Option<Vec<Option<Money>>>,
    #[serde(borrow, default)]
    tl: Option<&'a serde_json::value::RawValue>,
}

#[derive(serde::Deserialize)]
struct Rec<'a> {
    #[serde(borrow)]
    steps: Vec<Step<'a>>,
}

fn row(eid: i64, meta: &slimsrc::SlimRow, js: &str) -> Option<String> {
    // borrowed parse: market / private / tiles are skipped without allocation
    let v: Rec = serde_json::from_str(js).ok()?;
    let steps = &v.steps;
    let n = steps.len();
    if n < 700 {
        return None;
    }
    // farm24 / farm72 / farm144 = the opening; farm_all = the whole game's farm stream (the same full tape seen against
    // many opponents measures its economy and its robustness directly)
    let mut hs = [[0xcbf2_9ce4_8422_2325u64; 4]; 2];
    let mut shops: Vec<String> = vec![];
    for (t, st) in steps.iter().enumerate() {
        if let Some(tw) = st.tw.as_ref() {
            shops = tw.clone();
        }
        if t >= 1 {
            for s in 0..2 {
                let a: Option<Value> = st.a.as_ref().and_then(|a| a.get(s)).and_then(|r| serde_json::from_str(r.get()).ok());
                let farm = format!(
                    "{}|{}",
                    a.as_ref().and_then(|a| a.get("farmer")).map(|x| x.to_string()).unwrap_or_default(),
                    a.as_ref().and_then(|a| a.get("hands")).map(|x| x.to_string()).unwrap_or_default()
                );
                for (k, lim) in [24usize, 72, 144].iter().enumerate() {
                    if t <= *lim {
                        hs[s][k] = fnv(hs[s][k], &farm);
                    }
                }
                hs[s][3] = fnv(hs[s][3], &farm);
            }
        }
    }
    let money = |t: usize, s: usize| -> f64 { steps.get(t.min(n - 1)).and_then(|st| st.f.as_ref()).and_then(|f| f.get(s)).and_then(|m| m.as_ref()).and_then(|m| m.money).unwrap_or(f64::NAN) };
    let mut d6 = [(0u64, 0u64); 2];
    if let Some(tl) = steps.get(145).and_then(|st| st.tl).and_then(|r| serde_json::from_str::<Value>(r.get()).ok()) {
        for s in 0..2 {
            let fs = steps.get(144).and_then(|st| st.f.as_ref()).and_then(|f| f.get(s)).and_then(|m| m.as_ref());
            let hands = fs.and_then(|m| m.hands.as_ref()).map(|h| h.len()).unwrap_or(0);
            let quads = fs.and_then(|m| m.unlocked_quadrants.clone()).unwrap_or_default();
            d6[s] = day6_keys(&tl, s, hands, &quads);
        }
    }
    let world = format!("{}|{}", shops.first().map(|s| s.as_str()).unwrap_or("-"), shops.get(1).map(|s| s.as_str()).unwrap_or("-"));
    let mut out = String::new();
    for s in 0..2 {
        let r = meta.rating[s].map(|r| format!("{r:.0}")).unwrap_or_default();
        let b: Vec<String> = [6usize, 12, 18, 24].iter().map(|d| format!("{:.0}", money(d * 24, s))).collect();
        out.push_str(&format!(
            "{eid}	{s}	{r}	{}	{}	{n}	{world}	{}	{:.0}	{:.0}	{:x}	{:x}	{:x}	{:x}	{:x}	{:x}
",
            meta.source, meta.end_date, b.join("	"), money(n - 1, s), money(n - 1, 1 - s), hs[s][0], hs[s][1], hs[s][2], hs[s][3], d6[s].0, d6[s].1
        ));
    }
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let root = get("--slim").unwrap_or_else(|| "data/slim/s1".into());
    let out = get("--out").expect("--out FILE");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let files: Vec<PathBuf> = slimsrc::files(std::path::Path::new(&root)).into_iter().filter(|p| !p.to_string_lossy().contains("index")).collect();
    let nf = files.len();
    eprintln!("[chassis-scan] {nf} slim files, {threads} threads");
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    w.lock()
        .unwrap()
        .write_all(b"eid\tseat\trating\tsource\tdate\tn\tworld\tbank_d6\tbank_d12\tbank_d18\tbank_d24\tbank_d30\topp_d30\tfarm24\tfarm72\tfarm144\n")
        .unwrap();
    let q = Arc::new(Mutex::new(files));
    let done = Arc::new(Mutex::new((0usize, 0usize)));
    let t0 = std::time::Instant::now();
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, w, done) = (q.clone(), w.clone(), done.clone());
            std::thread::spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                let Ok(meta) = slimsrc::index(&p) else { continue };
                let by: std::collections::HashMap<i64, &slimsrc::SlimRow> = meta.iter().map(|m| (m.eid, m)).collect();
                let wanted: HashSet<i64> = by.keys().copied().collect();
                let mut buf = String::new();
                let mut k = 0;
                let _ = slimsrc::for_each(&p, &wanted, |eid, js| {
                    if let Some(r) = by.get(&eid).and_then(|m| row(eid, m, js)) {
                        buf.push_str(&r);
                        k += 1;
                    }
                });
                w.lock().unwrap().write_all(buf.as_bytes()).unwrap();
                let mut d = done.lock().unwrap();
                d.0 += 1;
                d.1 += k;
                if d.0 % 50 == 0 {
                    eprintln!("[chassis-scan] {} / {nf} files, {} games ({:.0}s)", d.0, d.1, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    w.lock().unwrap().flush().unwrap();
    let d = done.lock().unwrap();
    eprintln!("[chassis-scan] done: {} games in {:.0}s -> {out}", d.1, t0.elapsed().as_secs_f64());
}
