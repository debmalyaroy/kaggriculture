//! chassis-build: a synced world chassis from one opening family (v63.12 chassis C3).
//!
//!     chassis-build --slim data/slim/s1 --family-worlds data/chassis/family_worlds.tsv --family HASH \
//!         --base-router data/builds/v63.12_rl_c4/stage/base/router.json --out-dir weights/chassis/NAME
//!
//! Every route shares the family's farm opening (steps 1..144 identical farmer + hand actions), so the router can switch
//! to the world's route at step 144 without desyncing the farm. Route 100 = the family's best game overall (default and
//! the opening played before the switch); route 101.. = per world the best WINNING game of that family in that world.
//! The action a route plays at step t is the recorded action that led to state t+1 (slim `steps[t+1].a[seat]`).
//! The router is the base router with `shop_routes_new` replaced, `default_new` = 100 and the shared endgame route
//! switched off (endgame_step 720): each world keeps the endgame of the game it came from.
#[path = "slimsrc.rs"]
mod slimsrc;

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let root = get("--slim").unwrap_or_else(|| "data/slim/s1".into());
    let fw = get("--family-worlds").unwrap_or_else(|| "data/chassis/family_worlds.tsv".into());
    let fams: Vec<String> = get("--families").expect("--families H1,H2,..").split(',').map(|s| s.to_string()).collect();
    let base_router = get("--base-router").expect("--base-router FILE");
    let out_root = get("--out-root").unwrap_or_else(|| "weights/chassis".into());
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    // per family: (world, eid, seat, bank) of its best winning game per world
    let mut picks: HashMap<String, Vec<(String, i64, usize, f64)>> = HashMap::new();
    for (i, l) in std::io::BufReader::new(std::fs::File::open(&fw).expect("family_worlds")).lines().enumerate() {
        let l = l.unwrap();
        let x: Vec<&str> = l.split('\t').collect();
        if i == 0 || x.len() < 7 || !fams.iter().any(|f| f == x[0]) || x[6].is_empty() {
            continue;
        }
        let (e, s) = x[6].split_once('_').unwrap();
        picks.entry(x[0].to_string()).or_default().push((x[1].to_string(), e.parse().unwrap(), s.parse().unwrap(), x[5].parse().unwrap_or(0.0)));
    }
    let wanted: std::sync::Arc<HashSet<i64>> = std::sync::Arc::new(picks.values().flatten().map(|p| p.1).collect());
    // one parallel pass over the corpus collects every wanted game's two action streams
    let streams: std::sync::Arc<std::sync::Mutex<HashMap<(i64, usize), Vec<Value>>>> = Default::default();
    let q = std::sync::Arc::new(std::sync::Mutex::new(slimsrc::files(std::path::Path::new(&root)).into_iter().filter(|p| !p.to_string_lossy().contains("index")).collect::<Vec<_>>()));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (q, wanted, streams) = (q.clone(), wanted.clone(), streams.clone());
            std::thread::spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                let _ = slimsrc::for_each(&p, &wanted, |eid, js| {
                    let Ok(v) = serde_json::from_str::<Value>(js) else { return };
                    let Some(steps) = v.get("steps").and_then(|s| s.as_array()) else { return };
                    let mut m = streams.lock().unwrap();
                    for s in 0..2 {
                        m.entry((eid, s)).or_insert_with(|| (0..719).map(|t| steps.get(t + 1).and_then(|st| st.get("a")).and_then(|a| a.get(s)).cloned().unwrap_or(Value::Null)).collect());
                    }
                });
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let streams = streams.lock().unwrap();
    for fam in &fams {
        let Some(ps) = picks.get(fam) else { eprintln!("[chassis-build] no games for {fam}"); continue };
        let best = ps.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap().clone();
        let mut routes = serde_json::Map::new();
        let mut table = serde_json::Map::new();
        let Some(d) = streams.get(&(best.1, best.2)) else { eprintln!("[chassis-build] {fam}: best stream missing"); continue };
        routes.insert("100".into(), Value::Array(d.clone()));
        let (mut id, mut missing) = (101, 0);
        for (world, e, s, _) in ps {
            match streams.get(&(*e, *s)) {
                Some(a) => {
                    routes.insert(id.to_string(), Value::Array(a.clone()));
                    table.insert(world.clone(), Value::from(id));
                    id += 1;
                }
                None => missing += 1,
            }
        }
        let mut router: Value = serde_json::from_str(&std::fs::read_to_string(&base_router).expect("router")).expect("router json");
        let r = router.as_object_mut().unwrap();
        r.insert("shop_routes_new".into(), Value::Object(table));
        r.insert("default_new".into(), Value::from(100));
        r.insert("endgame_step".into(), Value::from(720));
        r.insert("chassis_family".into(), Value::from(fam.clone()));
        let out = format!("{out_root}/{fam}");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(format!("{out}/routes.json"), serde_json::to_string(&Value::Object(routes)).unwrap()).unwrap();
        std::fs::write(format!("{out}/router.json"), serde_json::to_string_pretty(&router).unwrap()).unwrap();
        eprintln!("[chassis-build] {fam}: {} world routes + default (best {}_{} bank {:.0}); {missing} missing -> {out}", id - 101, best.1, best.2, best.3);
    }
}
