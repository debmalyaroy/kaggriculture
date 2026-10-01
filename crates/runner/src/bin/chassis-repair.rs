//! chassis-repair: drive a chassis to beat EVERY tape (operator rule: 100%), by (world, rival key) route overrides.
//!
//!     chassis-repair --chassis DIR --pools P1[,P2..] --tapes DIR[,DIR] --out-dir OUT [--top 6] [--threads 24]
//!
//! 1. accept : the chassis plays every tape; per tape its realized world, the rival's three keys at step 145
//!             (cluster::keys: exact layout "l..", exact counts "x..", bucketed counts) and the result.
//! 2. scout  : per world with losses, every candidate route of that world (all pools: pool.tsv world column) plays
//!             every LOST tape of the world (forced after the switch).
//! 3. cells  : level 1 (bucketed key) -> 2 (exact counts) -> 3 (exact layout). For each cell that still has losses, the
//!             `--top` candidates that beat most of its lost tapes also play the cell's other tapes; a candidate that
//!             wins the whole cell becomes the cell's override; else the best one (more cell wins than the parent
//!             route) is kept and its remaining losses go down a level (a finer key overrides a coarser one).
//! 4. write  : OUT/chassis (routes: the chassis' + every override route re-numbered from max(existing id)+1 (>= 5000); router: cluster_routes
//!             with the new keys), OUT/accept.tsv, OUT/overrides.tsv, OUT/unbeaten.tsv (tapes no candidate beats).
//! Re-run chassis-vs-tapes (or this tool) on OUT/chassis to confirm.
use agent::act::Action;
use agent::base::Base;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::Json;
use kagg_engine::state::State;
use std::collections::{HashMap, HashSet};
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

fn par<J: Send + Clone + 'static, R: Send + 'static>(jobs: Vec<J>, threads: usize, tag: &str, f: Arc<dyn Fn(&J) -> R + Send + Sync>) -> Vec<(J, R)> {
    let n = jobs.len();
    let q = Arc::new(Mutex::new(jobs));
    let out: Arc<Mutex<Vec<(J, R)>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let t0 = std::time::Instant::now();
    let tag = tag.to_string();
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, out, f, tag) = (q.clone(), out.clone(), f.clone(), tag.clone());
            std::thread::spawn(move || loop {
                let Some(j) = q.lock().unwrap().pop() else { break };
                let r = f(&j);
                let mut o = out.lock().unwrap();
                o.push((j, r));
                if o.len() % 5000 == 0 {
                    eprintln!("[chassis-repair] {tag} {} / {n} ({:.0}s)", o.len(), t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[chassis-repair] {tag} done {n} in {:.0}s", t0.elapsed().as_secs_f64());
    Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap()
}

struct Ctx {
    /// the chassis' own routes (id -> tape) and router
    routes: Vec<(String, Vec<Action>)>,
    router: Json,
    /// candidate routes: global id -> (world, tape)
    cands: HashMap<i64, (String, Vec<Action>)>,
    guard: Json,
}

struct Res {
    world: String,
    keys: [String; 3],
    us: f64,
    them: f64,
}

impl Ctx {
    /// `cand` = None: the chassis as it is; Some(c): the chassis' opening, then candidate c forced after the switch.
    /// `cand` = None: the chassis as it is; Some((c, dispatch key)): the chassis with candidate c dispatched exactly
    /// as it will live -- the world route until the rival-key lookup at step 145, then c (key = "WORLD#l<layout>").
    fn play(&self, t: &runner::Tape, cand: Option<(i64, &str)>) -> Res {
        let mut b = match cand {
            None => Base::from_parts("chassis", self.routes.clone(), &self.router).expect("base"),
            Some((c, key)) => {
                let mut rs = self.routes.clone();
                rs.push(("99999".into(), self.cands[&c].1.clone()));
                let mut b = Base::from_parts("cand", rs, &self.router).expect("base");
                b.chassis.router.cluster_routes.insert(0, (key.to_string(), 99999));
                b
            }
        };
        b.cut = agent::base::cut_index("chassis").unwrap();
        let opp = runner::guarded_tape_base(t, &self.guard);
        let seat = t.seat;
        let mut ag: [Base; 2] = if seat == 0 { [b, opp] } else { [opp, b] };
        let mut st = State::new(t.seed);
        let mut keys = [String::new(), String::new(), String::new()];
        loop {
            if st.step == 145 {
                keys = agent::cluster::keys(&agent::obs::Obs::from_state(&st, seat).farms[1 - seat]);
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
        Res { world, keys, us: st.farms[seat].money, them: st.farms[1 - seat].money }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
    let dir = get("--chassis").expect("--chassis DIR");
    let out = get("--out-dir").expect("--out-dir DIR");
    let (top, threads) = (num("--top", 6), num("--threads", 24));
    std::fs::create_dir_all(format!("{out}/chassis")).unwrap();
    let routes = agent::obs::parse_routes(&std::fs::read_to_string(format!("{dir}/routes.json")).expect("routes")).expect("parse");
    let router = kagg_engine::json::parse(&std::fs::read_to_string(format!("{dir}/router.json")).expect("router")).expect("router json");
    // candidates from every pool, re-numbered (global id -> world, tape) with their origin
    let mut cands: HashMap<i64, (String, Vec<Action>)> = HashMap::new();
    let mut origin: HashMap<i64, String> = HashMap::new();
    let mut gid = 100_000i64;
    for p in get("--pools").expect("--pools").split(',').filter(|s| !s.is_empty()) {
        // --scout-max N: per world only the N highest-bank candidates of this pool
        let smax = num("--scout-max", 80);
        let mut rows: Vec<(String, String, f64)> = vec![];
        for l in std::fs::read_to_string(format!("{p}/pool.tsv")).expect("pool.tsv").lines().skip(1) {
            let x: Vec<&str> = l.split('\t').collect();
            if x.len() >= 5 {
                rows.push((x[0].to_string(), x[1].to_string(), x[4].parse().unwrap_or(0.0)));
            }
        }
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        let mut per: HashMap<String, usize> = HashMap::new();
        let mut world_of: HashMap<String, String> = HashMap::new();
        for (id, w, _) in rows {
            let c = per.entry(w.clone()).or_default();
            if *c < smax {
                *c += 1;
                world_of.insert(id, w);
            }
        }
        for (k, t) in agent::obs::parse_routes(&std::fs::read_to_string(format!("{p}/routes.json")).expect("pool routes")).expect("parse") {
            if let Some(w) = world_of.get(&k) {
                cands.insert(gid, (w.clone(), t));
                origin.insert(gid, format!("{p}:{k}"));
                gid += 1;
            }
        }
    }
    let guard = runner::guard_router(&get("--guard-base").unwrap_or_else(|| "configs/bases/v61.1".into()));
    let ctx = Arc::new(Ctx { routes, router, cands, guard });
    let mut files = vec![];
    for d in get("--tapes").expect("--tapes").split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    files.dedup_by(|a, b| a.file_name() == b.file_name());
    // tapes are STREAMED: only id -> file is kept, every game loads its tape (keeps two repairs within RAM)
    let tapes: Arc<HashMap<String, std::path::PathBuf>> = Arc::new(files.iter().map(|f| (f.file_stem().unwrap().to_string_lossy().to_string(), f.clone())).collect());
    eprintln!("[chassis-repair] {} tapes, {} candidates", tapes.len(), ctx.cands.len());
    let pts = |u: f64, t: f64| if u > t { 1.0 } else if u == t { 0.5 } else { 0.0 };
    // 1. accept
    let (c, tp) = (ctx.clone(), tapes.clone());
    let acc: HashMap<String, Res> = par(tapes.keys().cloned().collect(), threads, "accept", Arc::new(move |id: &String| load_tape(&tp[id]).map(|t| c.play(&t, None))))
        .into_iter()
        .filter_map(|(k, r)| r.map(|r| (k, r)))
        .collect();
    // each tape's exact dispatch key (its realized world + the rival's exact layout at step 145)
    let keyof: Arc<HashMap<String, String>> = Arc::new(acc.iter().map(|(k, r)| (k.clone(), format!("{}#{}", r.world, r.keys[0]))).collect());
    let n = acc.len() as f64;
    let score: f64 = acc.values().map(|r| pts(r.us, r.them)).sum::<f64>() / n;
    // --lost-margin M: repair only the losses within $M (the reachable ones); deeper losses are noted, not scouted
    let lm: f64 = get("--lost-margin").and_then(|x| x.parse().ok()).unwrap_or(f64::MAX);
    let lost: Vec<&String> = acc.iter().filter(|(_, r)| r.us <= r.them && r.them - r.us <= lm).map(|(k, _)| k).collect();
    eprintln!("[chassis-repair] accept: score {score:.4}, {} of {} tapes not won", lost.len(), acc.len());
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{out}/accept.tsv")).unwrap());
        writeln!(w, "tape\tworld\tkey_l\tkey_x\tkey_b\tour_bank\ttheir_bank").unwrap();
        for (k, r) in &acc {
            writeln!(w, "{k}\t{}\t{}\t{}\t{}\t{:.0}\t{:.0}", r.world, r.keys[0], r.keys[1], r.keys[2], r.us, r.them).unwrap();
        }
    }
    // memo of candidate results: (cand, tape) -> points
    let memo: Arc<Mutex<HashMap<(i64, String), f64>>> = Default::default();
    let run = |jobs: Vec<(i64, String)>, tag: &str| {
        let todo: Vec<(i64, String)> = {
            let m = memo.lock().unwrap();
            jobs.into_iter().filter(|j| !m.contains_key(j)).collect()
        };
        let (c, tp) = (ctx.clone(), tapes.clone());
        let ks = keyof.clone();
        for ((cid, id), r) in par(todo, threads, tag, Arc::new(move |j: &(i64, String)| c.play(&load_tape(&tp[&j.1]).expect("tape"), Some((j.0, ks[&j.1].as_str()))))) {
            memo.lock().unwrap().insert((cid, id), if r.us > r.them { 1.0 } else if r.us == r.them { 0.5 } else { 0.0 });
        }
    };
    // candidates by world
    let mut by_world: HashMap<String, Vec<i64>> = HashMap::new();
    for (id, (w, _)) in ctx.cands.iter() {
        by_world.entry(w.clone()).or_default().push(*id);
    }
    // 2. scout: every candidate of the world on every lost tape of the world
    let mut jobs = vec![];
    for id in &lost {
        for c in by_world.get(&acc[*id].world).into_iter().flatten() {
            jobs.push((*c, (*id).clone()));
        }
    }
    run(jobs, "scout");
    // 3. cells, level 1 -> 3. `cur[tape]` = the result the tape currently gets (chassis, then overrides)
    let mut cur: HashMap<String, f64> = acc.iter().map(|(k, r)| (k.clone(), pts(r.us, r.them))).collect();
    let mut overrides: Vec<(String, i64, usize, f64, f64)> = vec![]; // key, cand, cell size, cell score before, after
    for level in [2usize, 1, 0] {
        // cells at this level that still hold a lost tape
        let mut cells: HashMap<String, Vec<String>> = HashMap::new();
        for (k, r) in &acc {
            cells.entry(format!("{}#{}", r.world, r.keys[level])).or_default().push(k.clone());
        }
        let bad: Vec<(String, Vec<String>)> = cells.into_iter().filter(|(_, ids)| ids.iter().any(|i| cur[i] < 1.0)).collect();
        // shortlist per cell by lost tapes beaten (from the memo), then play the shortlist on the whole cell
        let mut short: Vec<(String, Vec<String>, Vec<i64>)> = vec![];
        let mut jobs = vec![];
        {
            let m = memo.lock().unwrap();
            for (key, ids) in bad {
                let w = key.split('#').next().unwrap().to_string();
                let lost_here: Vec<&String> = ids.iter().filter(|i| cur[*i] < 1.0).collect();
                let mut sc: Vec<(f64, i64)> = by_world.get(&w).into_iter().flatten().map(|c| (lost_here.iter().map(|i| *m.get(&(*c, (*i).clone())).unwrap_or(&0.0)).sum::<f64>(), *c)).filter(|x| x.0 > 0.0).collect();
                sc.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let sl: Vec<i64> = sc.iter().take(top).map(|x| x.1).collect();
                for c in &sl {
                    for i in &ids {
                        jobs.push((*c, i.clone()));
                    }
                }
                short.push((key, ids, sl));
            }
        }
        run(jobs, &format!("cells-level{}", 3 - level));
        let m = memo.lock().unwrap();
        let (mut fixed, mut improved) = (0, 0);
        for (key, ids, sl) in short {
            let before: f64 = ids.iter().map(|i| cur[i]).sum();
            let best = sl.iter().map(|c| (ids.iter().map(|i| *m.get(&(*c, i.clone())).unwrap_or(&0.0)).sum::<f64>(), *c)).max_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some((after, c)) = best {
                if after > before {
                    for i in &ids {
                        cur.insert(i.clone(), m[&(c, i.clone())]);
                    }
                    if after >= ids.len() as f64 { fixed += 1 } else { improved += 1 }
                    overrides.push((key, c, ids.len(), before, after));
                }
            }
        }
        let s: f64 = cur.values().sum::<f64>() / n;
        eprintln!("[chassis-repair] level {}: {fixed} cells fully fixed, {improved} improved; score now {s:.4}", 3 - level);
    }
    // a finer key must win over a coarser one: overrides were applied coarse -> fine, so keep them all (router looks up
    // the finest first). 4. write
    let mut routes_out: Vec<String> = ctx.routes.iter().map(|(k, t)| format!("\"{k}\":[{}]", t.iter().map(|a| a.dump()).collect::<Vec<_>>().join(","))).collect();
    let mut new_id: HashMap<i64, i64> = HashMap::new();
    // new routes are numbered ABOVE every existing route id (30 Sep: numbering from 5000 on a second pass overwrote the
    // first pass's routes 5000.. that its cluster overrides still used)
    let mut next = ctx.routes.iter().filter_map(|(k, _)| k.to_string().parse::<i64>().ok()).max().map(|m| (m + 1).max(5000)).unwrap_or(5000);
    let mut cr: Vec<String> = match ctx.router.get("cluster_routes") {
        Json::Obj(o) => o.iter().map(|(k, v)| format!("\"{k}\": {}", v.i64())).collect(),
        _ => vec![],
    };
    let mut ow = std::io::BufWriter::new(std::fs::File::create(format!("{out}/overrides.tsv")).unwrap());
    writeln!(ow, "key\troute\torigin\tcell\tbefore\tafter").unwrap();
    let mut seen_keys: HashSet<String> = HashSet::new();
    for (key, c, sz, b, a) in overrides.iter().rev() {
        if !seen_keys.insert(key.clone()) {
            continue;
        }
        let id = *new_id.entry(*c).or_insert_with(|| {
            let i = next;
            next += 1;
            routes_out.push(format!("\"{i}\":[{}]", ctx.cands[c].1.iter().map(|a| a.dump()).collect::<Vec<_>>().join(",")));
            i
        });
        cr.retain(|x| !x.starts_with(&format!("\"{key}\"")));
        cr.push(format!("\"{key}\": {id}"));
        writeln!(ow, "{key}\t{id}\t{}\t{sz}\t{b}\t{a}", origin[c]).unwrap();
    }
    ow.flush().unwrap();
    std::fs::write(format!("{out}/chassis/routes.json"), format!("{{{}}}", routes_out.join(","))).unwrap();
    let Json::Obj(mut rv) = ctx.router.clone() else { panic!("router") };
    rv.retain(|(k, _)| k != "cluster_routes" && k != "cluster_step");
    rv.push(("cluster_routes".into(), kagg_engine::json::parse(&format!("{{{}}}", cr.join(", "))).unwrap()));
    rv.push(("cluster_step".into(), Json::Num(145.0)));
    std::fs::write(format!("{out}/chassis/router.json"), Json::Obj(rv).dump()).unwrap();
    let mut uw = std::io::BufWriter::new(std::fs::File::create(format!("{out}/unbeaten.tsv")).unwrap());
    writeln!(uw, "tape\tworld\tkey_b\tkey_x\tkey_l\tbest_points").unwrap();
    let mut un = 0;
    for (k, v) in &cur {
        if *v < 1.0 {
            un += 1;
            let r = &acc[k];
            writeln!(uw, "{k}\t{}\t{}\t{}\t{}\t{v}", r.world, r.keys[2], r.keys[1], r.keys[0]).unwrap();
        }
    }
    uw.flush().unwrap();
    let s: f64 = cur.values().sum::<f64>() / n;
    eprintln!("[chassis-repair] done: projected score {s:.4} ({un} tapes still not won) -> {out}/chassis (re-run to confirm)");
}
