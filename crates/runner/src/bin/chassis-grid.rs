//! chassis-grid: guard settings grid for one chassis (every combination x a world-stratified tape sample, parallel).
//!
//!     chassis-grid --chassis weights/chassis/NAME --tapes DIR[,DIR] --out-dir data/chassis/grid/NAME
//!                  [--bools hand_align,weed_repair,...] [--grid F.json] [--base JSON] [--per-world 12] [--threads 24]
//!                  [--worlds worlds.tsv]
//!
//! Cells: `--bools a,b,c` = all 2^k ON/OFF combinations of those guards (others as the router says), and/or
//! `--grid F.json` = [{"name": .., "settings": {...}}, ...] (knob sweeps); `--base JSON` is applied under every cell.
//! Cell 0 is the router's own settings (+ --base): every other cell is compared with it PAIRED (same tapes):
//! better / worse = games whose result (win 1 / draw .5 / loss 0) went up / down.
//! Worlds: `--worlds` (chassis-screen worlds.tsv: tape -> realized world) or a realize pass (play to step 200).
//! Each game plays with only the routes it needs (the opening + the route of its world), so jobs stay cheap.
//! Writes OUT/cells.tsv (per cell: n, score, wins, losses, mean gap, better, worse, guard hits) and OUT/games.tsv.
use agent::act::Action;
use agent::base::Base;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::Json;
use kagg_engine::state::State;
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

fn walk(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let q = e.path();
            if q.is_dir() {
                walk(&q, out);
            } else if q.extension().is_some_and(|x| x == "json") {
                out.push(q);
            }
        }
    }
}

fn load_tape(f: &std::path::Path) -> Option<runner::Tape> {
    let txt = std::fs::read_to_string(f).ok()?;
    let j = kagg_engine::json::parse(&txt).ok()?;
    let seat = j.get("seat").i64() as usize;
    let stream: Vec<Action> = j.get("actions").arr().iter().map(|p| p.arr().get(1 - seat).map(Action::from_json).unwrap_or_else(Action::pass)).collect();
    Some(runner::Tape { id: f.file_stem()?.to_string_lossy().to_string(), seed: j.get("seed").i64(), seat, band: j.get("band").str().to_string(), stream: Arc::new(stream) })
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

struct Ctx {
    routes: HashMap<i64, Vec<Action>>,
    router: Json,
    used: Vec<i64>,
    openings: HashMap<i64, Vec<Action>>,
    guard: Json,
}

impl Ctx {
    /// Our base: the opening (route 0, or the cell's `__opening` pool route) + every world route of the table (a
    /// different opening can land in a different world, so the router picks at 144 as it will live).
    fn ours(&self, _world: &str, cell: &Json) -> Base {
        let open = if cell.get("__opening").is_null() { &self.routes[&0] } else { &self.openings[&cell.get("__opening").i64()] };
        let mut rs = vec![("0".to_string(), open.clone())];
        for r in &self.used {
            if *r != 0 {
                rs.push((r.to_string(), self.routes[r].clone()));
            }
        }
        let mut b = Base::from_parts("grid", rs, &self.router).expect("base");
        b.cut = agent::base::cut_index("chassis").unwrap();
        b.chassis.cfg.apply(cell);
        b.chassis.diagnostics.log = true;
        b
    }
    fn play(&self, tape: &runner::Tape, world: &str, cell: &Json, stop: i64) -> (String, f64, f64, [u32; 9]) {
        let opp = runner::guarded_tape_base(tape, &self.guard);
        let seat = tape.seat;
        let mut ag: [Base; 2] = if seat == 0 { [self.ours(world, cell), opp] } else { [opp, self.ours(world, cell)] };
        let mut st = State::new(tape.seed);
        while st.step < stop {
            let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
            for (s, a) in ag.iter_mut().enumerate() {
                let (act, _) = a.act(agent::obs::Obs::from_state(&st, s));
                acts.push(runner::to_engine(&act));
            }
            let pair = [acts.remove(0), acts.remove(0)];
            if !engine::step(&mut st, &pair) {
                break;
            }
        }
        let sh = &st.town.unlocked_shops;
        let w = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
        (w, st.farms[seat].money, st.farms[1 - seat].money, ag[seat].chassis.diagnostics.guard_hits)
    }
}

fn par<J: Send + Clone + 'static, R: Send + 'static>(jobs: Vec<J>, threads: usize, tag: &'static str, f: Arc<dyn Fn(&J) -> R + Send + Sync>) -> Vec<(J, R)> {
    let n = jobs.len();
    let q = Arc::new(Mutex::new(jobs));
    let out: Arc<Mutex<Vec<(J, R)>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let t0 = std::time::Instant::now();
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, out, f) = (q.clone(), out.clone(), f.clone());
            std::thread::spawn(move || loop {
                let Some(j) = q.lock().unwrap().pop() else { break };
                let r = f(&j);
                let mut o = out.lock().unwrap();
                o.push((j, r));
                if o.len() % 5000 == 0 {
                    eprintln!("[chassis-grid] {tag} {} / {n} ({:.0}s)", o.len(), t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[chassis-grid] {tag} done {n} in {:.0}s", t0.elapsed().as_secs_f64());
    Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap()
}

fn pts(us: f64, them: f64) -> f64 {
    if us > them { 1.0 } else if us == them { 0.5 } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let dir = get("--chassis").expect("--chassis DIR");
    let out = get("--out-dir").expect("--out-dir DIR");
    let (per_world, threads) = (num("--per-world", 12), num("--threads", 24));
    std::fs::create_dir_all(&out).unwrap();
    let parsed = agent::obs::parse_routes(&std::fs::read_to_string(format!("{dir}/routes.json")).expect("routes")).expect("parse");
    let routes: HashMap<i64, Vec<Action>> = parsed.into_iter().map(|(k, t)| (k.parse().unwrap(), t)).collect();
    let router = kagg_engine::json::parse(&std::fs::read_to_string(format!("{dir}/router.json")).expect("router")).expect("router json");
    let table: HashMap<String, i64> = router.get("shop_routes_new").obj().iter().map(|(k, v)| (k.clone(), v.i64())).collect();
    let default = if router.get("default_new").is_null() { 0 } else { router.get("default_new").i64() };
    let guard = runner::guard_router(&get("--guard-base").unwrap_or_else(|| "configs/bases/v61.1".into()));
    // cells
    let base_cell: Vec<(String, Json)> = match get("--base").map(|s| kagg_engine::json::parse(&s).expect("--base json")) {
        Some(Json::Obj(o)) => o,
        _ => vec![],
    };
    let mk = |extra: Vec<(String, Json)>| {
        let mut o = base_cell.clone();
        for (k, v) in extra {
            o.retain(|(a, _)| *a != k);
            o.push((k, v));
        }
        Json::Obj(o)
    };
    // the chassis carries only the REPAIR guards; the market guards live in the reactive shell / layers and are
    // pinned off in every cell (sell_lead/r36/racepx -> rshell/shell, front_run -> preempt, dead_stock -> shell,
    // terminal_liquidation -> terminal/endgame)
    const MARKET: [&str; 6] = ["sell_lead", "front_run", "dead_stock", "terminal_liquidation", "r36", "racepx"];
    let mk = |extra: Vec<(String, Json)>| {
        let Json::Obj(mut o) = mk(extra) else { unreachable!() };
        for m in MARKET {
            if o.iter().any(|(k, v)| k == m && v.bool()) {
                panic!("{m} is a reactive-layer guard; it cannot be switched on in the chassis");
            }
            o.retain(|(k, _)| k != m);
            o.push((m.to_string(), Json::Bool(false)));
        }
        Json::Obj(o)
    };
    let mut cells: Vec<(String, Json)> = vec![("router".into(), mk(vec![]))];
    if let Some(b) = get("--bools") {
        let names: Vec<&str> = b.split(',').filter(|s| !s.is_empty()).collect();
        for m in 0..(1u32 << names.len()) {
            let kv: Vec<(String, Json)> = names.iter().enumerate().map(|(i, n)| (n.to_string(), Json::Bool(m >> i & 1 == 1))).collect();
            let name = names.iter().enumerate().map(|(i, n)| format!("{}{}", if m >> i & 1 == 1 { "+" } else { "-" }, n)).collect::<Vec<_>>().join("");
            cells.push((name, mk(kv)));
        }
    }
    if let Some(g) = get("--grid") {
        let j = kagg_engine::json::parse(&std::fs::read_to_string(&g).expect("grid")).expect("grid json");
        for c in j.arr() {
            let kv = match c.get("settings") { Json::Obj(o) => o.clone(), _ => vec![] };
            cells.push((c.get("name").str().to_string(), mk(kv)));
        }
    }
    // tapes and worlds
    let mut files = vec![];
    for d in get("--tapes").expect("--tapes").split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    files.dedup_by(|a, b| a.file_name() == b.file_name());
    // every route the router can dispatch: the world table, the default and the (world, rival key) overrides
    let cluster_vals: Vec<i64> = router.get("cluster_routes").obj().iter().map(|(_, v)| v.i64()).collect();
    let mut used: Vec<i64> = table.values().copied().chain([default]).chain(cluster_vals).filter(|r| routes.contains_key(r)).collect();
    used.sort();
    used.dedup();
    // --openings POOL:N : one cell per opening = the N best-bank family games of the pool (their steps 0..143)
    let mut openings = HashMap::new();
    if let Some(o) = get("--openings") {
        let (pd, n) = o.split_once(':').expect("--openings POOL:N");
        let n: usize = n.parse().unwrap();
        let mut fam: Vec<(f64, i64)> = std::fs::read_to_string(format!("{pd}/pool.tsv")).expect("pool.tsv").lines().skip(1).filter_map(|l| {
            let x: Vec<&str> = l.split('\t').collect();
            (x.len() >= 5 && x[3] == "fam").then(|| (x[4].parse().unwrap_or(0.0), x[0].parse().unwrap()))
        }).collect();
        fam.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let keep: std::collections::HashSet<i64> = fam.iter().take(n).map(|x| x.1).collect();
        for (k, t) in agent::obs::parse_routes(&std::fs::read_to_string(format!("{pd}/routes.json")).expect("pool routes")).expect("parse") {
            let id: i64 = k.parse().unwrap();
            if keep.contains(&id) {
                openings.insert(id, t);
            }
        }
        let mut ids: Vec<i64> = openings.keys().copied().collect();
        ids.sort();
        for id in ids {
            cells.push((format!("open{id}"), mk(vec![("__opening".into(), Json::Num(id as f64))])));
        }
    }
    let ctx = Arc::new(Ctx { routes, router, used, openings, guard });
    let worlds: HashMap<String, String> = match get("--worlds") {
        Some(w) => std::fs::read_to_string(&w).expect("worlds").lines().filter_map(|l| l.split_once('\t')).map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        None => {
            let c = ctx.clone();
            let cell0 = cells[0].1.clone();
            par(files.clone(), threads, "realize", Arc::new(move |f: &std::path::PathBuf| load_tape(f).map(|t| c.play(&t, "", &cell0, 200).0)))
                .into_iter()
                .filter_map(|(f, w)| w.map(|w| (f.file_stem().unwrap().to_string_lossy().to_string(), w)))
                .collect()
        }
    };
    let mut by_world: HashMap<String, Vec<&std::path::PathBuf>> = HashMap::new();
    for f in &files {
        if let Some(w) = worlds.get(&*f.file_stem().unwrap().to_string_lossy()) {
            by_world.entry(w.clone()).or_default().push(f);
        }
    }
    let mut tapes: Vec<(Arc<runner::Tape>, String)> = vec![];
    for (w, v) in by_world.iter_mut() {
        v.sort_by_key(|p| fnv(&format!("grid{}", p.to_string_lossy())));
        for p in v.iter().take(per_world) {
            if let Some(t) = load_tape(p) {
                tapes.push((Arc::new(t), w.clone()));
            }
        }
    }
    eprintln!("[chassis-grid] {} cells x {} tapes ({} worlds) = {} games", cells.len(), tapes.len(), by_world.len(), cells.len() * tapes.len());
    let cells = Arc::new(cells);
    let tapes = Arc::new(tapes);
    let jobs: Vec<(usize, usize)> = (0..cells.len()).flat_map(|c| (0..tapes.len()).map(move |t| (c, t))).collect();
    let (c2, cl, tp) = (ctx.clone(), cells.clone(), tapes.clone());
    let res = par(jobs, threads, "grid", Arc::new(move |j: &(usize, usize)| {
        let (t, w) = &tp[j.1];
        c2.play(t, w, &cl[j.0].1, 720)
    }));
    let mut r: HashMap<(usize, usize), (f64, f64, [u32; 9])> = HashMap::new();
    let mut gw = std::io::BufWriter::new(std::fs::File::create(format!("{out}/games.tsv")).unwrap());
    writeln!(gw, "cell\ttape\tworld\tour_bank\ttheir_bank").unwrap();
    for ((c, t), (_, us, them, h)) in res {
        writeln!(gw, "{}\t{}\t{}\t{us:.0}\t{them:.0}", cells[c].0, tapes[t].0.id, tapes[t].1).unwrap();
        r.insert((c, t), (us, them, h));
    }
    gw.flush().unwrap();
    let mut rows = vec![];
    for (c, (name, js)) in cells.iter().enumerate() {
        let (mut n, mut p, mut w, mut l, mut gap, mut better, mut worse) = (0.0, 0.0, 0, 0, 0.0, 0, 0);
        let mut hits = [0u64; 9];
        for t in 0..tapes.len() {
            let Some(&(us, them, h)) = r.get(&(c, t)) else { continue };
            let &(u0, t0, _) = &r[&(0, t)];
            n += 1.0;
            p += pts(us, them);
            w += (us > them) as i32;
            l += (us < them) as i32;
            gap += us - them;
            let d = pts(us, them) - pts(u0, t0);
            better += (d > 0.0) as i32;
            worse += (d < 0.0) as i32;
            for (i, x) in h.iter().enumerate() {
                hits[i] += *x as u64;
            }
        }
        let hs: Vec<String> = agent::chassis::GUARDS.iter().zip(hits.iter()).filter(|(_, x)| **x > 0).map(|(g, x)| format!("{g}:{:.0}", *x as f64 / n)).collect();
        rows.push((p / n, name.clone(), format!("{name}\t{n}\t{:.4}\t{w}\t{l}\t{:.0}\t{better}\t{worse}\t{}\t{}", p / n, gap / n, hs.join(","), js.dump())));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut cw = std::io::BufWriter::new(std::fs::File::create(format!("{out}/cells.tsv")).unwrap());
    writeln!(cw, "cell\tn\tscore\twins\tlosses\tmean_gap\tbetter_vs_router\tworse_vs_router\tguard_hits_per_game\tsettings").unwrap();
    for (_, _, l) in &rows {
        writeln!(cw, "{l}").unwrap();
    }
    cw.flush().unwrap();
    for (_, _, l) in rows.iter().take(10) {
        eprintln!("[chassis-grid] {}", l.split('\t').take(8).collect::<Vec<_>>().join("  "));
    }
}
