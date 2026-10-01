//! chassis-screen: closed-loop per-world route search for one chassis pool (crates/corpus chassis-pool output).
//!
//!     chassis-screen --pool weights/chassis_pool/FAM --tapes DIR[,DIR] --out-dir data/chassis/screen/FAM
//!                    [--per-world 16] [--val-per-world 40] [--top 4] [--kinds fam,cross,any] [--settings JSON]
//!                    [--threads 24] [--guard-base configs/bases/v61.1]
//!
//! Phase 1 (realize): every tape is played to step 200 with the pool's opening (route 0) in our seat against the
//!   recorded opponent (guarded tape); the world = the first two shops then (both are fixed by step 146).
//! Phase 2 (screen): per world, a SELECT sample (`--per-world` tapes, id-hash split) plays every candidate route of
//!   that world (route 0 until the switch at 144, then the candidate).
//! Phase 3 (validate): the `--top` candidates per world by select score replay on a disjoint VALIDATE sample.
//! Pick per world = best on select+validate (wins + draws/2, then mean gap). Writes OUT/screen.tsv (every game),
//! OUT/summary.tsv (per world x route), OUT/table.json {world: route}, OUT/worlds.tsv (tape -> realized world).
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
    open: Vec<Action>,
    cands: HashMap<i64, Vec<Action>>,
    router: Json,
    settings: Option<Json>,
    guard: Json,
}

impl Ctx {
    /// Our base: route 0 = the opening, route 1 = the candidate (forced after the switch); `None` = opening only.
    fn ours(&self, cand: Option<i64>) -> Base {
        let mut routes = vec![("0".to_string(), self.open.clone())];
        if let Some(c) = cand {
            routes.push(("1".to_string(), self.cands[&c].clone()));
        }
        let mut b = Base::from_parts("screen", routes, &self.router).expect("base");
        b.cut = agent::base::cut_index("chassis").unwrap();
        if let Some(s) = &self.settings {
            b.chassis.cfg.apply(s);
        }
        b.chassis.router.force(if cand.is_some() { 1 } else { 0 });
        b
    }

    /// Play one game to `stop` (720 = the end); returns (world, our money, their money).
    fn play(&self, tape: &runner::Tape, cand: Option<i64>, stop: i64) -> (String, f64, f64, u64) {
        let opp = runner::guarded_tape_base(tape, &self.guard);
        let ours = self.ours(cand);
        let seat = tape.seat;
        let mut ag: [Base; 2] = if seat == 0 { [ours, opp] } else { [opp, ours] };
        let mut st = State::new(tape.seed);
        let mut cluster = 0u64;
        while st.step < stop {
            if st.step == 145 {
                cluster = agent::cluster::econ_key(&agent::obs::Obs::from_state(&st, seat).farms[1 - seat]);
            }
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
        let world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
        (world, st.farms[seat].money, st.farms[1 - seat].money, cluster)
    }
}

/// Exact two-sided sign test on discordant pairs.
fn sign_p(better: u64, worse: u64) -> f64 {
    let m = better + worse;
    let k = better.min(worse);
    let (mut p, mut lc) = (0.0f64, 0.0f64);
    for i in 0..=k {
        if i > 0 {
            lc += ((m - i + 1) as f64).ln() - (i as f64).ln();
        }
        p += (lc - m as f64 * 2f64.ln()).exp();
    }
    (2.0 * p).min(1.0)
}

/// Run `jobs` on `threads` workers; `f` maps a job to an output value.
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
                    eprintln!("[chassis-screen] {tag} {} / {n} ({:.0}s)", o.len(), t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[chassis-screen] {tag} done {n} in {:.0}s", t0.elapsed().as_secs_f64());
    Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap()
}

#[derive(Default, Clone)]
struct Score {
    n: f64,
    pts: f64,
    gap: f64,
}
impl Score {
    fn add(&mut self, us: f64, them: f64) {
        self.n += 1.0;
        self.pts += if us > them { 1.0 } else if us == them { 0.5 } else { 0.0 };
        self.gap += us - them;
    }
    fn key(&self) -> (f64, f64) {
        if self.n == 0.0 {
            (-1.0, 0.0)
        } else {
            (self.pts / self.n, self.gap / self.n)
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let pool = get("--pool").expect("--pool DIR");
    let out = get("--out-dir").expect("--out-dir DIR");
    let (per_world, val_per_world, top, threads) = (num("--per-world", 16), num("--val-per-world", 40), num("--top", 4), num("--threads", 24));
    let kinds: Vec<String> = get("--kinds").unwrap_or_else(|| "fam,cross,any".into()).split(',').map(|s| s.to_string()).collect();
    std::fs::create_dir_all(&out).unwrap();
    // pool
    let routes = agent::obs::parse_routes(&std::fs::read_to_string(format!("{pool}/routes.json")).expect("routes")).expect("parse routes");
    let mut world_of: HashMap<i64, (String, String)> = HashMap::new();
    for l in std::fs::read_to_string(format!("{pool}/pool.tsv")).expect("pool.tsv").lines().skip(1) {
        let x: Vec<&str> = l.split('\t').collect();
        if x.len() >= 4 && kinds.iter().any(|k| k == x[3]) {
            world_of.insert(x[0].parse().unwrap(), (x[1].to_string(), x[3].to_string()));
        }
    }
    let mut open = vec![];
    let mut cands = HashMap::new();
    for (k, t) in routes {
        let id: i64 = k.parse().unwrap();
        if id == 0 {
            open = t;
        } else if world_of.contains_key(&id) {
            cands.insert(id, t);
        }
    }
    let router = kagg_engine::json::parse(&std::fs::read_to_string(format!("{pool}/router.json")).expect("router")).expect("router json");
    let settings = get("--settings").map(|s| kagg_engine::json::parse(&s).expect("--settings json"));
    let guard = runner::guard_router(&get("--guard-base").unwrap_or_else(|| "configs/bases/v61.1".into()));
    let ctx = Arc::new(Ctx { open, cands, router, settings, guard });
    // phase 1: realized worlds
    let mut files = vec![];
    for d in get("--tapes").expect("--tapes").split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    files.dedup_by(|a, b| a.file_name() == b.file_name());
    // --eval N: a FIXED hold-out (the same N tapes for every class and for the baseline, by file-name hash) that
    // never takes part in picking routes; the assembled chassis is judged on it, paired with --baseline
    let n_eval = num("--eval", 0);
    let mut eval_files: Vec<std::path::PathBuf> = vec![];
    // --fold k/K: K-fold cross-fit -- the tapes of fold k (file-name hash) are the hold-out, the others pick
    if let Some(fk) = get("--fold") {
        let (k, kk) = fk.split_once('/').expect("--fold k/K");
        let (k, kk): (u64, u64) = (k.parse().unwrap(), kk.parse().unwrap());
        let (ev, pk): (Vec<_>, Vec<_>) = files.drain(..).partition(|p| fnv(&format!("fold{}", p.file_name().unwrap().to_string_lossy())) % kk == k);
        eval_files = ev;
        files = pk;
    }
    if n_eval > 0 {
        files.sort_by_key(|p| fnv(&format!("eval{}", p.file_name().unwrap().to_string_lossy())));
        eval_files = files.drain(..n_eval.min(files.len())).collect();
        files.sort();
    }
    // --max-tapes N: a fixed hash sample of the tapes (a cheap prescreen)
    if let Some(m) = get("--max-tapes").and_then(|s| s.parse::<usize>().ok()) {
        files.sort_by_key(|p| fnv(&format!("max{}", p.file_name().unwrap().to_string_lossy())));
        files.truncate(m);
    }
    eprintln!("[chassis-screen] {} candidates, {} tapes", ctx.cands.len(), files.len());
    let c1 = ctx.clone();
    let realized = par(files, threads, "realize", Arc::new(move |f: &std::path::PathBuf| load_tape(f).map(|t| {
        let r = c1.play(&t, None, 200);
        (r.0, r.3)
    })));
    let mut by_world: HashMap<String, Vec<std::path::PathBuf>> = HashMap::new();
    let mut cluster_of: HashMap<String, u64> = HashMap::new();
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{out}/worlds.tsv")).unwrap());
        for (f, r) in &realized {
            if let Some((wd, cl)) = r {
                let stem = f.file_stem().unwrap().to_string_lossy().to_string();
                writeln!(w, "{stem}\t{wd}\t{cl:x}").unwrap();
                by_world.entry(wd.clone()).or_default().push(f.clone());
                cluster_of.insert(stem, *cl);
            }
        }
    }
    for v in by_world.values_mut() {
        v.sort_by_key(|p| fnv(&p.to_string_lossy()));
    }
    // successive halving: round r plays the survivors on a fresh, disjoint slice of each world's tapes (loaded per
    // round, dropped after); keep = a fraction (< 1) or a count (>= 1); scores accumulate over rounds
    let spec = get("--rounds").unwrap_or_else(|| format!("{per_world}:{top},{val_per_world}"));
    let rounds: Vec<(usize, f64)> = spec.split(',').map(|r| {
        let mut it = r.split(':');
        (it.next().unwrap().parse().unwrap(), it.next().map(|k| k.parse().unwrap()).unwrap_or(1.0))
    }).collect();
    let mut alive: HashMap<String, Vec<i64>> = HashMap::new();
    for (id, (wd, _)) in &world_of {
        if ctx.cands.contains_key(id) {
            alive.entry(wd.clone()).or_default().push(*id);
        }
    }
    let mut games = std::io::BufWriter::new(std::fs::File::create(format!("{out}/screen.tsv")).unwrap());
    writeln!(games, "round\troute\tkind\tcand_world\ttape\trealized_world\tour_bank\ttheir_bank").unwrap();
    let mut score: HashMap<(String, i64), Score> = HashMap::new();
    let mut pts_of: HashMap<(i64, String), f64> = HashMap::new();
    let mut offset = 0usize;
    let key = |score: &HashMap<(String, i64), Score>, w: &str, c: i64| score.get(&(w.to_string(), c)).map(|s| s.key()).unwrap_or((-1.0, 0.0));
    for (ri, (size, keep)) in rounds.iter().enumerate() {
        let mut tapes: HashMap<String, Arc<runner::Tape>> = HashMap::new();
        let mut jobs = vec![];
        for (wd, cs) in &alive {
            let Some(v) = by_world.get(wd) else { continue };
            for p in v.iter().skip(offset).take(*size) {
                let Some(t) = load_tape(p) else { continue };
                let id = t.id.clone();
                tapes.insert(id.clone(), Arc::new(t));
                for c in cs {
                    jobs.push((*c, id.clone()));
                }
            }
        }
        offset += size;
        let tapes = Arc::new(tapes);
        let tag: &'static str = Box::leak(format!("round{}", ri + 1).into_boxed_str());
        eprintln!("[chassis-screen] {tag}: {} candidates x {size} tapes/world = {} games", alive.values().map(|v| v.len()).sum::<usize>(), jobs.len());
        let (c, t) = (ctx.clone(), tapes.clone());
        for ((cid, id), (rw, us, them, _)) in par(jobs, threads, tag, Arc::new(move |j: &(i64, String)| c.play(&t[&j.1], Some(j.0), 720))) {
            let (wd, kind) = &world_of[&cid];
            writeln!(games, "{}\t{cid}\t{kind}\t{wd}\t{id}\t{rw}\t{us:.0}\t{them:.0}", ri + 1).unwrap();
            score.entry((wd.clone(), cid)).or_default().add(us, them);
            pts_of.insert((cid, id.clone()), if us > them { 1.0 } else if us == them { 0.5 } else { 0.0 });
        }
        if ri + 1 < rounds.len() {
            for (wd, cs) in alive.iter_mut() {
                cs.sort_by(|a, b| key(&score, wd, *b).partial_cmp(&key(&score, wd, *a)).unwrap());
                let k = if *keep < 1.0 { ((cs.len() as f64 * keep).ceil() as usize).max(1) } else { *keep as usize };
                cs.truncate(k.max(1));
            }
        }
    }
    games.flush().unwrap();
    // summary + pick (best accumulated score among the last round's survivors)
    let mut sw = std::io::BufWriter::new(std::fs::File::create(format!("{out}/summary.tsv")).unwrap());
    writeln!(sw, "world\troute\tkind\tn\tscore\tmean_gap\tfinalist").unwrap();
    let mut keys: Vec<&(String, i64)> = score.keys().collect();
    keys.sort();
    for k in keys {
        let s = &score[k];
        let fin = alive.get(&k.0).is_some_and(|v| v.contains(&k.1));
        writeln!(sw, "{}\t{}\t{}\t{}\t{:.3}\t{:.0}\t{}", k.0, k.1, world_of[&k.1].1, s.n, s.key().0, s.key().1, fin as i32).unwrap();
    }
    sw.flush().unwrap();
    // ROBUST final pick among the last round's survivors: the base route must hold against every kind of rival, so
    // each rival cluster of the world counts equally (clusters with < 5 tapes pooled as one) and the worst cluster
    // counts extra: key = 0.75 x cluster-balanced score + 0.25 x worst-cluster score (ties: raw score)
    let world_ids: HashMap<String, Vec<String>> = by_world.iter().map(|(w, v)| (w.clone(), v.iter().map(|p| p.file_stem().unwrap().to_string_lossy().to_string()).collect())).collect();
    let robust = |c: i64, ids: &[String], skip: Option<u64>| -> (f64, f64) {
        let mut per: HashMap<u64, (f64, f64)> = HashMap::new();
        for id in ids {
            let cl = *cluster_of.get(id).unwrap_or(&0);
            if Some(cl) == skip {
                continue;
            }
            if let Some(p) = pts_of.get(&(c, id.clone())) {
                let e = per.entry(cl).or_default();
                e.0 += p;
                e.1 += 1.0;
            }
        }
        let (mut big, mut rest) = (vec![], (0.0, 0.0));
        for (_, (p, n)) in per {
            if n >= 5.0 {
                big.push(p / n)
            } else {
                rest.0 += p;
                rest.1 += n
            }
        }
        if rest.1 > 0.0 {
            big.push(rest.0 / rest.1);
        }
        if big.is_empty() {
            return (-1.0, 0.0);
        }
        let bal = big.iter().sum::<f64>() / big.len() as f64;
        let min = big.iter().cloned().fold(f64::MAX, f64::min);
        (0.75 * bal + 0.25 * min, bal)
    };
    let mut table: Vec<(String, i64, (f64, f64), f64)> = vec![];
    for (wd, cs) in &alive {
        let ids = world_ids.get(wd).cloned().unwrap_or_default();
        if let Some(b) = cs.iter().max_by(|a, b| (robust(**a, &ids, None).0, key(&score, wd, **a)).partial_cmp(&(robust(**b, &ids, None).0, key(&score, wd, **b))).unwrap()) {
            table.push((wd.clone(), *b, key(&score, wd, *b), score[&(wd.clone(), *b)].n));
        }
    }
    // leave-cluster-out report (no extra games): per world and big rival cluster, re-pick the base route WITHOUT that
    // cluster's tapes and score it on them -- how the base holds against an opponent type it never saw
    {
        let (mut lco, mut ins, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for (wd, pick, _, _) in &table {
            let ids = world_ids.get(wd).cloned().unwrap_or_default();
            let mut cls: HashMap<u64, Vec<String>> = HashMap::new();
            for id in &ids {
                if pts_of.contains_key(&(*pick, id.clone())) {
                    cls.entry(*cluster_of.get(id).unwrap_or(&0)).or_default().push(id.clone());
                }
            }
            for (cl, cids) in cls.iter().filter(|(_, v)| v.len() >= 10) {
                let alt = alive[wd].iter().max_by(|a, b| robust(**a, &ids, Some(*cl)).0.partial_cmp(&robust(**b, &ids, Some(*cl)).0).unwrap()).copied().unwrap_or(*pick);
                for id in cids {
                    lco += pts_of.get(&(alt, id.clone())).copied().unwrap_or(0.0);
                    ins += pts_of.get(&(*pick, id.clone())).copied().unwrap_or(0.0);
                    n += 1.0;
                }
            }
        }
        eprintln!("[chassis-screen] LCO leave-cluster-out base score {:.4} vs in-sample pick {:.4} on {n} tapes", lco / n.max(1.0), ins / n.max(1.0));
    }
    table.sort_by(|a, b| a.0.cmp(&b.0));
    // per (world, rival cluster) overrides: among the world's last-round finalists (they played every pick tape of the
    // world), the best one on the cell's tapes replaces the world route when it is better, paired, with sign-test
    // p < --cluster-p on >= --cluster-min tapes
    let (cmin, cp) = (num("--cluster-min", 30), get("--cluster-p").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.1));
    let mut overrides: Vec<(String, i64, usize, u64, u64, f64)> = vec![];
    for (wd, pick, _, _) in &table {
        let mut cells: HashMap<u64, Vec<String>> = HashMap::new();
        for p in by_world.get(wd).into_iter().flatten() {
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            if pts_of.contains_key(&(*pick, stem.clone())) {
                if let Some(c) = cluster_of.get(&stem) {
                    cells.entry(*c).or_default().push(stem);
                }
            }
        }
        for (cl, ids) in cells {
            if ids.len() < cmin {
                continue;
            }
            let mut best: Option<(f64, i64, u64, u64)> = None;
            for f in alive.get(wd).into_iter().flatten().filter(|f| *f != pick) {
                let (mut d, mut b, mut w, mut n) = (0.0, 0u64, 0u64, 0);
                for id in &ids {
                    let (Some(a), Some(z)) = (pts_of.get(&(*f, id.clone())), pts_of.get(&(*pick, id.clone()))) else { continue };
                    n += 1;
                    d += a - z;
                    b += (a > z) as u64;
                    w += (a < z) as u64;
                }
                if n >= cmin && d > 0.0 && best.map_or(true, |x| d > x.0) {
                    best = Some((d, *f, b, w));
                }
            }
            if let Some((_, f, b, w)) = best {
                let p = sign_p(b, w);
                if p < cp {
                    overrides.push((format!("{wd}#{cl:x}"), f, ids.len(), b, w, p));
                }
            }
        }
    }
    overrides.sort_by(|a, b| a.0.cmp(&b.0));
    {
        let mut ow = std::io::BufWriter::new(std::fs::File::create(format!("{out}/cluster_overrides.tsv")).unwrap());
        writeln!(ow, "world_cluster\troute\ttapes\tbetter\tworse\tp").unwrap();
        for (k, r, n, b, w, p) in &overrides {
            writeln!(ow, "{k}\t{r}\t{n}\t{b}\t{w}\t{p:.3e}").unwrap();
        }
    }
    eprintln!("[chassis-screen] {} (world, cluster) overrides (>= {cmin} tapes, p < {cp})", overrides.len());
    let js: Vec<String> = table.iter().map(|(w, r, _, _)| format!("\"{w}\": {r}")).collect();
    std::fs::write(format!("{out}/table.json"), format!("{{{}}}", js.join(", "))).unwrap();
    // the assembled chassis: route 0 = the opening, one route per picked world, the pool router with this table
    {
        let cd = format!("{out}/chassis");
        std::fs::create_dir_all(&cd).unwrap();
        let tape = |t: &Vec<Action>| format!("[{}]", t.iter().map(|a| a.dump()).collect::<Vec<_>>().join(","));
        let mut parts = vec![format!("\"0\":{}", tape(&ctx.open))];
        let mut ids: Vec<i64> = table.iter().map(|x| x.1).chain(overrides.iter().map(|o| o.1)).collect();
        ids.sort();
        ids.dedup();
        for r in ids {
            parts.push(format!("\"{r}\":{}", tape(&ctx.cands[&r])));
        }
        std::fs::write(format!("{cd}/routes.json"), format!("{{{}}}", parts.join(","))).unwrap();
        let Json::Obj(mut rv) = kagg_engine::json::parse(&std::fs::read_to_string(format!("{pool}/router.json")).unwrap()).unwrap() else { panic!("router") };
        let tj = kagg_engine::json::parse(&format!("{{{}}}", js.join(", "))).unwrap();
        rv.retain(|(k, _)| k != "shop_routes_new" && k != "shop_routes_old");
        rv.push(("shop_routes_new".into(), tj.clone()));
        rv.push(("shop_routes_old".into(), tj));
        rv.retain(|(k, _)| k != "cluster_routes" && k != "cluster_step");
        let cj = kagg_engine::json::parse(&format!("{{{}}}", overrides.iter().map(|o| format!("\"{}\": {}", o.0, o.1)).collect::<Vec<_>>().join(", "))).unwrap();
        rv.push(("cluster_routes".into(), cj));
        rv.push(("cluster_step".into(), Json::Num(145.0)));
        std::fs::write(format!("{cd}/router.json"), Json::Obj(rv).dump()).unwrap();
    }
    // family score: per-world pick score weighted by how often our opening lands in that world
    let total: usize = by_world.values().map(|v| v.len()).sum();
    let (mut fs, mut wsum) = (0.0f64, 0.0f64);
    let mut pw = std::io::BufWriter::new(std::fs::File::create(format!("{out}/picks.tsv")).unwrap());
    writeln!(pw, "world\troute\tkind\tn\tscore\tmean_gap\tworld_share").unwrap();
    for (w, r, (s, g), n) in &table {
        let share = by_world.get(w).map(|v| v.len()).unwrap_or(0) as f64 / total.max(1) as f64;
        writeln!(pw, "{w}\t{r}\t{}\t{n}\t{s:.3}\t{g:.0}\t{share:.4}", world_of[r].1).unwrap();
        fs += share * s;
        wsum += share;
    }
    pw.flush().unwrap();
    eprintln!("[chassis-screen] {} worlds picked; family score (world-weighted pick score) {:.4} -> {out}/table.json", table.len(), fs / wsum.max(1e-9));
    // hold-out evaluation of the assembled chassis, paired with the baseline on the same tapes
    if !eval_files.is_empty() {
        let cd = format!("{out}/chassis");
        let routes = agent::obs::parse_routes(&std::fs::read_to_string(format!("{cd}/routes.json")).unwrap()).unwrap();
        let router = kagg_engine::json::parse(&std::fs::read_to_string(format!("{cd}/router.json")).unwrap()).unwrap();
        let settings = ctx.settings.clone();
        let full = Arc::new((routes, router, settings, ctx.guard.clone()));
        let res = par(eval_files, threads, "eval", Arc::new(move |f: &std::path::PathBuf| {
            let t = load_tape(f)?;
            // two variants per tape: the chassis as built, and its base (every cluster override stripped)
            let mut res = [(0.0, 0.0); 2];
            for (vi, strip) in [false, true].iter().enumerate() {
                let mut b = Base::from_parts("chassis", full.0.clone(), &full.1).ok()?;
                b.cut = agent::base::cut_index("chassis").unwrap();
                if let Some(s) = &full.2 {
                    b.chassis.cfg.apply(s);
                }
                if *strip {
                    b.chassis.router.cluster_routes.clear();
                }
                let opp = runner::guarded_tape_base(&t, &full.3);
                let seat = t.seat;
                let mut ag: [Base; 2] = if seat == 0 { [b, opp] } else { [opp, b] };
                let mut st = State::new(t.seed);
                loop {
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
                res[vi] = (st.farms[seat].money, st.farms[1 - seat].money);
            }
            Some((t.id.clone(), res[0].0, res[0].1, res[1].0, res[1].1))
        }));
        let base: HashMap<String, (f64, f64)> = get("--baseline").map(|b| std::fs::read_to_string(&b).expect("baseline").lines().skip(1).filter_map(|l| {
            let x: Vec<&str> = l.split('\t').collect();
            Some((x.first()?.to_string(), (x.get(4)?.parse().ok()?, x.get(5)?.parse().ok()?)))
        }).collect()).unwrap_or_default();
        let pts = |u: f64, t: f64| if u > t { 1.0 } else if u == t { 0.5 } else { 0.0 };
        let (mut n, mut sc, mut bs, mut better, mut worse) = (0.0f64, 0.0, 0.0, 0u64, 0u64);
        let (mut so, mut ob, mut ow) = (0.0f64, 0u64, 0u64);
        let mut ew = std::io::BufWriter::new(std::fs::File::create(format!("{out}/eval.tsv")).unwrap());
        writeln!(ew, "tape\tour_bank\ttheir_bank\tbase_our\tbase_their\tbaseonly_our\tbaseonly_their").unwrap();
        for (_, r) in res {
            let Some((id, u, t, u0, t0)) = r else { continue };
            let Some(&(bu, bt)) = base.get(&id) else { continue };
            writeln!(ew, "{id}\t{u:.0}\t{t:.0}\t{bu:.0}\t{bt:.0}\t{u0:.0}\t{t0:.0}").unwrap();
            so += pts(u0, t0);
            ob += (pts(u0, t0) > pts(bu, bt)) as u64;
            ow += (pts(u0, t0) < pts(bu, bt)) as u64;
            n += 1.0;
            sc += pts(u, t);
            bs += pts(bu, bt);
            let d = pts(u, t) - pts(bu, bt);
            better += (d > 0.0) as u64;
            worse += (d < 0.0) as u64;
        }
        ew.flush().unwrap();
        let p = sign_p(better, worse);
        std::fs::write(format!("{out}/eval_summary.tsv"), format!("n\tscore\tbase_score\tbetter\tworse\tp\n{n}\t{:.4}\t{:.4}\t{better}\t{worse}\t{p:.3e}\n", sc / n.max(1.0), bs / n.max(1.0))).unwrap();
        eprintln!("[chassis-screen] EVALBASE (world routes only, no cluster overrides) n {n} score {:.4} vs baseline better {ob} worse {ow} p {:.2e}", so / n.max(1.0), sign_p(ob, ow));
        eprintln!("[chassis-screen] EVAL n {n} score {:.4} baseline {:.4} better {better} worse {worse} p {p:.2e}", sc / n.max(1.0), bs / n.max(1.0));
    }
}
