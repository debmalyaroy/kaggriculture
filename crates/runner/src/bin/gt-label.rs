//! Closed-loop game-theory payoffs (v63.12 G1b): at collision states against REAL adaptive opponents, branch our next
//! 24 steps of one item's sales four ways and play every branch to the end (the opponent keeps reacting).
//!
//!     gt-label --base DIR --a-args "<our agent>" --opps "name:<args>|..." --bank data/worlds/w64_bank.json
//!              --split train --per-world 1 [--seed-skip 0] --checkpoints 360,552,648,684 --min 15 --out F [--threads 8]
//! A collision: at a checkpoint both players' sheds (engine truth) hold >= min units of STRAWBERRY/MELON/MILK/WOOL.
//! Branches (crate::gt::GtLayer::force): 0 HOLD (release <= 25% of the shed in 24 steps), 1 TRANCHE (<= 75%, <= 12 per
//! step), 2 DUMP (the whole shed now, front of the queue), 3 TAPE (unchanged). The checkpoint prefix is shared, so
//! each label is an exact counterfactual. Row:
//!   seed seat opp world step item price our_shed rival_shed rival_est gap_hold gap_tranche gap_dump gap_tape
use agent::base::Base;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::state::State;
use std::io::Write;
use std::sync::{Arc, Mutex};

const ITEMS: [&str; 4] = ["STRAWBERRY", "MELON", "MILK", "WOOL"];

fn step1(st: &mut State, agents: &mut [Base; 2]) -> bool {
    let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
    for (s, ag) in agents.iter_mut().enumerate() {
        let o = agent::obs::Obs::from_state(st, s);
        let (a, _) = ag.act(o);
        acts.push(runner::to_engine(&a));
    }
    let pair = [acts.remove(0), acts.remove(0)];
    engine::step(st, &pair)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let split = |s: &str| -> Vec<String> { s.split_whitespace().map(|x| x.to_string()).collect() };
    let out = get("--out").expect("--out FILE");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let base_dir = get("--base").expect("--base DIR");
    let min: i64 = get("--min").and_then(|s| s.parse().ok()).unwrap_or(15);
    let cps: Vec<i64> = get("--checkpoints").unwrap_or_else(|| "360,552,648,684".into()).split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let mut a = Base::load(&base_dir).expect("base");
    runner::configure(&mut a, &split(&get("--a-args").unwrap_or_default()));
    let off = agent::gt::GtCfg { groups: vec![], ..Default::default() };
    a.gtl = Some(agent::gt::GtLayer::new(Arc::new(off)));
    let opps: Vec<(String, Base)> = get("--opps")
        .expect("--opps")
        .split('|')
        .map(|spec| {
            let (name, rest) = spec.split_once(':').unwrap_or((spec, ""));
            let mut b = Base::load(&base_dir).expect("base");
            runner::configure(&mut b, &split(rest));
            (name.to_string(), b)
        })
        .collect();
    let bank = json::parse(&std::fs::read_to_string(get("--bank").expect("--bank")).expect("bank")).expect("bank json");
    let per: usize = get("--per-world").and_then(|s| s.parse().ok()).unwrap_or(1);
    let skip: usize = get("--seed-skip").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut jobs: Vec<(i64, usize, usize)> = vec![];
    for (_, seeds) in bank.get(&get("--split").unwrap_or_else(|| "train".into())).obj() {
        for s in seeds.arr().iter().skip(skip).take(per) {
            for oi in 0..opps.len() {
                jobs.push((s.i64(), 0, oi));
                jobs.push((s.i64(), 1, oi));
            }
        }
    }
    if let Some(n) = get("--limit").and_then(|s| s.parse::<usize>().ok()) {
        jobs.truncate(n);
    }
    eprintln!("[gt-label] {} games, checkpoints {:?}, min {min}", jobs.len(), cps);
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    let t0 = std::time::Instant::now();
    let q = Arc::new(Mutex::new(jobs));
    let (opps, a) = (Arc::new(opps), Arc::new(a));
    let done = Arc::new(Mutex::new(0usize));
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, opps, a, w, done, cps) = (q.clone(), opps.clone(), a.clone(), w.clone(), done.clone(), cps.clone());
            std::thread::spawn(move || loop {
                let Some((seed, me, oi)) = q.lock().unwrap().pop() else { break };
                let (name, ob) = &opps[oi];
                let mut ag: [Base; 2] = if me == 0 { [a.fresh(), ob.fresh()] } else { [ob.fresh(), a.fresh()] };
                let mut st = State::new(seed);
                let mut rows: Vec<String> = vec![];
                for &cp in &cps {
                    let mut alive = true;
                    while st.step < cp {
                        if !step1(&mut st, &mut ag) {
                            alive = false;
                            break;
                        }
                    }
                    if !alive {
                        break;
                    }
                    for item in ITEMS {
                        let (ours, theirs) = (st.private[me].shed.get(item), st.private[1 - me].shed.get(item));
                        if ours < min || theirs < min {
                            continue;
                        }
                        let px = kagg_engine::market::price(kagg_engine::market::param(item).unwrap(), st.market.inventory.get(item) as f64);
                        let est = ag[me].gt.stock(item);
                        let mut gaps = [0f64; 4];
                        for (k, g) in gaps.iter_mut().enumerate() {
                            let mut s2 = st.clone();
                            let mut a2 = ag.clone();
                            if let Some(l) = a2[me].gtl.as_mut() {
                                l.force = Some((item, cp, k));
                            }
                            while step1(&mut s2, &mut a2) {}
                            *g = s2.farms[me].money - s2.farms[1 - me].money;
                        }
                        let sh = &st.town.unlocked_shops;
                        let world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
                        rows.push(format!("{seed}\t{me}\t{name}\t{world}\t{cp}\t{item}\t{px}\t{ours}\t{theirs}\t{est}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\n", gaps[0], gaps[1], gaps[2], gaps[3]));
                    }
                }
                let mut wl = w.lock().unwrap();
                for r in rows {
                    wl.write_all(r.as_bytes()).unwrap();
                }
                wl.flush().unwrap();
                drop(wl);
                let mut n = done.lock().unwrap();
                *n += 1;
                if *n % 20 == 0 {
                    eprintln!("[gt-label] {} games ({:.0}s)", *n, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[gt-label] done in {:.0}s -> {out}", t0.elapsed().as_secs_f64());
}
