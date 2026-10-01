//! slim-tape: write replay tapes (python/replay_to_tape.py format) for chosen slim games, straight from the corpus.
//!
//!     slim-tape --slim data/slim/s1 --games EID_SEAT[,EID_SEAT..] | --games-file F (one EID_SEAT per line, first
//!               column) --out-dir DIR [--threads 8]
//!
//! Tape: {id, seed, seat, rewards, actions[t] = [a0, a1] (the action taken AT step t = slim steps[t+1].a), band,
//! opp_rating}. `seat` = the seat named in EID_SEAT (the seat a chassis replaces); the other seat is the opponent.
#[path = "slimsrc.rs"]
mod slimsrc;

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let root = get("--slim").unwrap_or_else(|| "data/slim/s1".into());
    let out = get("--out-dir").expect("--out-dir DIR");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut list: Vec<String> = get("--games").map(|g| g.split(',').map(|s| s.to_string()).collect()).unwrap_or_default();
    if let Some(f) = get("--games-file") {
        for l in std::io::BufReader::new(std::fs::File::open(&f).expect("games-file")).lines().map_while(Result::ok) {
            if let Some(x) = l.split('\t').next().filter(|x| x.contains('_')) {
                list.push(x.to_string());
            }
        }
    }
    let mut seats: HashMap<i64, Vec<usize>> = HashMap::new();
    for g in &list {
        let (e, s) = g.split_once('_').expect("EID_SEAT");
        if let (Ok(e), Ok(s)) = (e.parse::<i64>(), s.parse::<usize>()) {
            seats.entry(e).or_default().push(s);
        }
    }
    std::fs::create_dir_all(&out).unwrap();
    let wanted: Arc<HashSet<i64>> = Arc::new(seats.keys().copied().collect());
    let seats = Arc::new(seats);
    let done: Arc<Mutex<HashSet<(i64, usize)>>> = Default::default();
    let files: Vec<_> = slimsrc::files(std::path::Path::new(&root)).into_iter().filter(|p| !p.to_string_lossy().contains("index")).collect();
    let q = Arc::new(Mutex::new(files));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (q, wanted, seats, done, out) = (q.clone(), wanted.clone(), seats.clone(), done.clone(), out.clone());
            std::thread::spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                let _ = slimsrc::for_each(&p, &wanted, |eid, js| {
                    let Ok(v) = serde_json::from_str::<Value>(js) else { return };
                    let Some(steps) = v.get("steps").and_then(|s| s.as_array()) else { return };
                    // seed / rewards: a dict / list, or their Python repr as a string ("{'TeamNames': [...], 'seed': N}")
                    let seed: i64 = match v.get("info") {
                        Some(Value::Object(o)) => o.get("seed").and_then(|x| x.as_i64()).unwrap_or(-1),
                        Some(Value::String(i)) => i.split("'seed':").nth(1).and_then(|s| s.trim().trim_end_matches('}').trim().parse().ok()).unwrap_or(-1),
                        _ => -1,
                    };
                    let rewards: Value = match v.get("rewards") {
                        Some(Value::String(r)) => serde_json::from_str(&r.replace("None", "null")).unwrap_or(Value::Null),
                        Some(x) => x.clone(),
                        None => Value::Null,
                    };
                    if seed < 0 {
                        eprintln!("[slim-tape] {eid}: no seed (info {:?})", v.get("info").map(|x| x.to_string().chars().take(120).collect::<String>()));
                    }
                    let actions: Vec<Value> = (0..719).map(|t| steps.get(t + 1).and_then(|st| st.get("a")).cloned().unwrap_or_else(|| json!([null, null]))).collect();
                    for &s in seats.get(&eid).into_iter().flatten() {
                        if !done.lock().unwrap().insert((eid, s)) {
                            continue;
                        }
                        let t = json!({"id": eid, "seed": seed, "seat": s, "rewards": rewards, "actions": actions, "band": "slim", "opp_rating": null});
                        std::fs::write(format!("{out}/{eid}_{s}.json"), serde_json::to_string(&t).unwrap()).unwrap();
                    }
                });
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[slim-tape] {} of {} tapes -> {out}", done.lock().unwrap().len(), list.len());
}
