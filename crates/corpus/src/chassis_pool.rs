//! chassis-pool: the candidate route pool of opening families (chassis route search, step 1).
//!
//!     chassis-pool --scan data/chassis/c1_scan.tsv --families H1,H2 --out-root weights/chassis_pool
//!                  [--k-fam 24] [--k-cross 16] [--k-any 8] [--threads 24]
//!
//! Per family F (farm144 hash) and per world W, the candidates are
//!   fam   : games playing F's exact day-6 farm opening, top k-fam by final bank
//!   cross : games sharing only F's day-3 opening (farm72), top k-cross
//!   any   : the world's best games overall, top k-any (a switch at step 144 may desync; the screen measures it)
//! OUT/F/routes.json: "0" = F's best game (the opening, played before the switch and as default), 1000.. = candidates;
//! OUT/F/pool.tsv: route world game kind bank opp; OUT/F/router.json: a plain router (no step-0 override, switch at
//! 144 from the fam candidate with the best bank; the screen replaces the table).
#[path = "slimsrc.rs"]
mod slimsrc;

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct G {
    eid: i64,
    seat: usize,
    world: String,
    bank: f64,
    opp: f64,
    f72: String,
    f144: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let scan = get("--scan").unwrap_or_else(|| "data/chassis/c1_scan.tsv".into());
    let root = get("--slim").unwrap_or_else(|| "data/slim/s1".into());
    let fams: Vec<String> = get("--families").expect("--families").split(',').map(|s| s.to_string()).collect();
    let out_root = get("--out-root").unwrap_or_else(|| "weights/chassis_pool".into());
    let (kf, kc, ka, threads) = (num("--k-fam", 24), num("--k-cross", 16), num("--k-any", 8), num("--threads", 24));
    // grouping columns of the scan: --fam-col (default 15 = farm144; 17 = layout6) and --cross-col (14 = farm72;
    // 18 = econ6)
    let (fc, cc) = (num("--fam-col", 15), num("--cross-col", 14));
    let mut games: Vec<G> = vec![];
    let mut seen = HashSet::new();
    for (i, l) in std::io::BufReader::new(std::fs::File::open(&scan).expect("scan")).lines().enumerate() {
        let Ok(l) = l else { continue };
        let x: Vec<&str> = l.split('\t').collect();
        if i == 0 || x.len() <= fc.max(cc) || x[6].starts_with('-') || x[6].ends_with('-') {
            continue;
        }
        let (Ok(b), Ok(o)) = (x[11].parse::<f64>(), x[12].parse::<f64>()) else { continue };
        if !b.is_finite() || !o.is_finite() || !seen.insert((x[0].to_string(), x[1].to_string())) {
            continue;
        }
        games.push(G { eid: x[0].parse().unwrap(), seat: x[1].parse().unwrap(), world: x[6].into(), bank: b, opp: o, f72: x[cc].into(), f144: x[fc].into() });
    }
    games.sort_by(|a, b| b.bank.partial_cmp(&a.bank).unwrap());
    let worlds: Vec<String> = {
        let mut w: Vec<String> = games.iter().map(|g| g.world.clone()).collect::<HashSet<_>>().into_iter().collect();
        w.sort();
        w
    };
    eprintln!("[chassis-pool] {} games, {} worlds", games.len(), worlds.len());
    // per family: (route id, world, game, kind)
    let mut plan: HashMap<String, Vec<(i64, String, G, &str)>> = HashMap::new();
    let mut want: HashSet<(i64, usize)> = HashSet::new();
    for f in &fams {
        let Some(best) = games.iter().find(|g| &g.f144 == f) else { eprintln!("[chassis-pool] no games for {f}"); continue };
        let f72 = best.f72.clone();
        let mut v = vec![(0i64, best.world.clone(), best.clone(), "open")];
        let mut id = 1000;
        for w in &worlds {
            let mut used = HashSet::new();
            let mut take = |pred: &dyn Fn(&G) -> bool, k: usize, kind: &'static str, v: &mut Vec<(i64, String, G, &str)>, id: &mut i64| {
                for g in games.iter().filter(|g| &g.world == w && pred(g)).take(k * 3) {
                    if !used.insert((g.eid, g.seat)) {
                        continue;
                    }
                    v.push((*id, w.clone(), g.clone(), kind));
                    *id += 1;
                    if v.iter().filter(|x| &x.1 == w && x.3 == kind).count() >= k {
                        break;
                    }
                }
            };
            take(&|g: &G| &g.f144 == f, kf, "fam", &mut v, &mut id);
            take(&|g: &G| g.f72 == f72 && &g.f144 != f, kc, "cross", &mut v, &mut id);
            take(&|g: &G| g.f72 != f72 && g.bank > g.opp, ka, "any", &mut v, &mut id);
        }
        want.extend(v.iter().map(|x| (x.2.eid, x.2.seat)));
        eprintln!("[chassis-pool] {f}: {} candidates", v.len() - 1);
        plan.insert(f.clone(), v);
    }
    // one parallel corpus pass: the wanted seats' action streams
    let eids: Arc<HashSet<i64>> = Arc::new(want.iter().map(|x| x.0).collect());
    let want = Arc::new(want);
    let streams: Arc<Mutex<HashMap<(i64, usize), String>>> = Default::default();
    let files: Vec<_> = slimsrc::files(std::path::Path::new(&root)).into_iter().filter(|p| !p.to_string_lossy().contains("index")).collect();
    let q = Arc::new(Mutex::new(files));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let (q, eids, want, streams) = (q.clone(), eids.clone(), want.clone(), streams.clone());
            std::thread::spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                let _ = slimsrc::for_each(&p, &eids, |eid, js| {
                    let Ok(v) = serde_json::from_str::<Value>(js) else { return };
                    let Some(steps) = v.get("steps").and_then(|s| s.as_array()) else { return };
                    for s in 0..2 {
                        if !want.contains(&(eid, s)) {
                            continue;
                        }
                        let a: Vec<Value> = (0..719).map(|t| steps.get(t + 1).and_then(|st| st.get("a")).and_then(|a| a.get(s)).cloned().unwrap_or(Value::Null)).collect();
                        streams.lock().unwrap().entry((eid, s)).or_insert_with(|| serde_json::to_string(&a).unwrap());
                    }
                });
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let streams = streams.lock().unwrap();
    for (f, v) in &plan {
        let dir = format!("{out_root}/{f}");
        std::fs::create_dir_all(&dir).unwrap();
        let mut routes: Vec<String> = vec![];
        let mut pool = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/pool.tsv")).unwrap());
        writeln!(pool, "route\tworld\tgame\tkind\tbank\topp").unwrap();
        let mut table = serde_json::Map::new();
        let mut miss = 0;
        for (id, w, g, kind) in v {
            let Some(s) = streams.get(&(g.eid, g.seat)) else { miss += 1; continue };
            routes.push(format!("\"{id}\":{s}"));
            writeln!(pool, "{id}\t{w}\t{}_{}\t{kind}\t{:.0}\t{:.0}", g.eid, g.seat, g.bank, g.opp).unwrap();
            if *kind == "fam" && !table.contains_key(w) {
                table.insert(w.clone(), json!(id));
            }
        }
        let router = json!({
            "select_step": 144, "endgame_step": 720, "endgame_route": 0, "yarn_uses_old": false,
            "default_new": 0, "default_old": 0, "shop_routes_new": table.clone(), "shop_routes_old": table, "shop_routes_v92": {},
            "chassis_family": f,
            "settings": {"hand_align": true, "weed_repair": true, "sell_lead": false, "front_run": false, "budget_guard": false,
                         "room_guard": false, "clamp_sells": false, "dead_stock": false, "terminal_liquidation": false}
        });
        std::fs::write(format!("{dir}/routes.json"), format!("{{{}}}", routes.join(","))).unwrap();
        std::fs::write(format!("{dir}/router.json"), serde_json::to_string_pretty(&router).unwrap()).unwrap();
        eprintln!("[chassis-pool] {f}: {} routes written ({miss} missing) -> {dir}", v.len() - miss);
    }
}
