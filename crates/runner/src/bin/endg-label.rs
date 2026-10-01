//! Exact counterfactual labels for the learned endgame controller (crates/agent/src/endg.rs).
//!
//!     endg-label --a-args "<our agent>" --endg configs/endg/proposals.json --opps "name:<args>|..|rand:RAND" \
//!                --bank data/worlds/w64_bank.json --split train --per-world 8 [--seed-skip 0] [--all-opps] \
//!                --out F [--threads 16]
//! Every game (seed from the bank, seat, opponent) is played once per endgame proposal k = 0..K-1 with the
//! proposal FORCED from the controller's decision step (proposal 0 = the agent as-is). Games are deterministic
//! and the prefix up to the decision step is identical, so each replay is an exact closed-loop counterfactual
//! (the opponent keeps reacting). Row: id seat seed opp world x[0..NF] gap_0 .. gap_{K-1}
//! (gap = our final bank minus theirs).
use agent::base::Base;
use agent::endg::{EndgCfg, EndgCtl, Mode, NF};
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::state::State;
use std::io::Write;
use std::sync::{Arc, Mutex};

fn splitmix(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Plays both agents one step; false when the episode is over.
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

/// Branching labels (v63.12 T2): play once to the decision step, then copy the game and both agents and finish it
/// once per proposal. The prefix is identical by construction, so this equals K full replays at ~1/10 of the cost.
fn play_branch(seed: i64, mut agents: [Base; 2], me: usize, from: i64, cfgs: &[Arc<EndgCfg>]) -> (Vec<f64>, String, Option<Vec<f32>>) {
    let mut st = State::new(seed);
    while st.step < from {
        if !step1(&mut st, &mut agents) {
            break;
        }
    }
    let mut gaps = vec![0f64; cfgs.len()];
    let (mut world, mut x0) = (String::new(), None);
    for (kk, c) in cfgs.iter().enumerate() {
        let mut s2 = st.clone();
        let mut a2 = agents.clone();
        a2[me].endg = Some(EndgCtl::new(c.clone()));
        if let (Some(e0), Some(e2)) = (agents[me].endg.as_ref(), a2[me].endg.as_mut()) {
            e2.seats = e0.seats.clone();
        }
        while step1(&mut s2, &mut a2) {}
        gaps[kk] = s2.farms[me].money - s2.farms[1 - me].money;
        if kk == 0 {
            let sh = &s2.town.unlocked_shops;
            world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
            x0 = a2[me].endg.as_ref().and_then(|e| e.seats.first().map(|s| s.2.clone()));
        }
    }
    (gaps, world, x0)
}

/// One closed-loop game; returns (our margin, the world, our endgame inputs if the decision happened).
#[allow(dead_code)]
fn play(seed: i64, agents: &mut [Base; 2], me: usize) -> (f64, String, Option<Vec<f32>>) {
    let mut st = State::new(seed);
    loop {
        let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
        for (s, ag) in agents.iter_mut().enumerate() {
            let o = agent::obs::Obs::from_state(&st, s);
            let (a, _) = ag.act(o);
            acts.push(runner::to_engine(&a));
        }
        let pair = [acts.remove(0), acts.remove(0)];
        if !engine::step(&mut st, &pair) {
            break;
        }
    }
    let sh = &st.town.unlocked_shops;
    let world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
    let x = agents[me].endg.as_ref().and_then(|e| e.seats.first().map(|s| s.2.clone()));
    (st.farms[me].money - st.farms[1 - me].money, world, x)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = get("--out").expect("--out FILE");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let endg = get("--endg").expect("--endg proposals.json");
    let split = |s: &str| s.split_whitespace().map(|x| x.to_string()).collect::<Vec<String>>();
    let mut a = Base::load(&base_dir).expect("base");
    let mut a_args = split(&get("--a-args").unwrap_or_default());
    a_args.extend(["--endg".to_string(), endg.clone()]);
    runner::configure(&mut a, &a_args);
    let cfg0 = EndgCfg::load(&endg).expect("endg");
    let k = cfg0.patches.len();
    let cfgs: Vec<Arc<EndgCfg>> = (0..k)
        .map(|i| {
            let mut c = cfg0.clone();
            c.mode = Mode::Force(i);
            Arc::new(c)
        })
        .collect();
    let opps: Vec<(String, Option<Base>)> = get("--opps")
        .expect("--opps")
        .split('|')
        .map(|spec| {
            let (name, rest) = spec.split_once(':').unwrap_or((spec, ""));
            if rest.trim() == "RAND" {
                (name.to_string(), None)
            } else {
                let mut b = Base::load(&base_dir).expect("base");
                runner::configure(&mut b, &split(rest));
                (name.to_string(), Some(b))
            }
        })
        .collect();
    let bank = json::parse(&std::fs::read_to_string(get("--bank").expect("--bank")).expect("bank")).expect("bank json");
    let per: usize = get("--per-world").and_then(|s| s.parse().ok()).unwrap_or(8);
    let skip: usize = get("--seed-skip").and_then(|s| s.parse().ok()).unwrap_or(0);
    let all_opps = args.iter().any(|x| x == "--all-opps");
    let both_seats = args.iter().any(|x| x == "--both-seats");
    // --branch: play the prefix once and branch at the decision step (exact, ~10x faster)
    let branch = args.iter().any(|x| x == "--branch");
    let mut jobs: Vec<(i64, usize, usize)> = vec![];
    for (_, seeds) in bank.get(&get("--split").unwrap_or_else(|| "train".into())).obj() {
        for (i, s) in seeds.arr().iter().skip(skip).take(per).enumerate() {
            let seed = s.i64();
            if all_opps {
                for oi in 0..opps.len() {
                    jobs.push((seed, 0, oi));
                    jobs.push((seed, 1, oi));
                }
            } else if both_seats {
                let oi = (seed as usize / 7 + i) % opps.len();
                jobs.push((seed, 0, oi));
                jobs.push((seed, 1, oi));
            } else {
                jobs.push((seed, i % 2, (seed as usize / 7 + i) % opps.len()));
            }
        }
    }
    if let Some(n) = get("--limit").and_then(|s| s.parse::<usize>().ok()) {
        jobs.truncate(n);
    }
    eprintln!("[endg-label] {} games x {k} proposals, {} opponents, {NF} inputs", jobs.len(), opps.len());
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    let t0 = std::time::Instant::now();
    let q = Arc::new(Mutex::new(jobs));
    let (opps, a, cfgs) = (Arc::new(opps), Arc::new(a), Arc::new(cfgs));
    let done = Arc::new(Mutex::new(0usize));
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, opps, a, w, done, cfgs) = (q.clone(), opps.clone(), a.clone(), w.clone(), done.clone(), cfgs.clone());
            std::thread::spawn(move || loop {
                let Some((seed, seat, oi)) = q.lock().unwrap().pop() else { break };
                let (name, ob) = &opps[oi];
                let mk = |me: usize, kk: usize| -> [Base; 2] {
                    let mut o = match ob {
                        Some(b) => b.fresh(),
                        None => {
                            let mut b = a.fresh();
                            b.rshell = None;
                            b.policy = None;
                            b.obs_track = None;
                            b.endg = None;
                            b.shell = None;
                            let mut r = splitmix(seed as u64 ^ 0xB2A2);
                            let kn = agent::layers::knobs::Knobs::random(&mut r);
                            b.chain.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), kn)];
                            b.chain.schedule = Some(vec![1; 30]);
                            b
                        }
                    };
                    o.endg = None;
                    let mut mine = a.fresh();
                    mine.endg = Some(EndgCtl::new(cfgs[kk].clone()));
                    if me == 0 { [mine, o] } else { [o, mine] }
                };
                let (gaps, world, x0) = if branch {
                    play_branch(seed, mk(seat, 0), seat, cfgs[0].from, &cfgs)
                } else {
                    let mut gaps = vec![0f64; cfgs.len()];
                    let mut x0: Option<Vec<f32>> = None;
                    let mut world = String::new();
                    for kk in 0..cfgs.len() {
                        let (g, wd, x) = play(seed, &mut mk(seat, kk), seat);
                        gaps[kk] = g;
                        if kk == 0 {
                            x0 = x;
                            world = wd;
                        }
                    }
                    (gaps, world, x0)
                };
                let Some(x) = x0 else { continue };
                let xs: Vec<String> = x.iter().map(|v| format!("{v}")).collect();
                let gs: Vec<String> = gaps.iter().map(|v| format!("{v:.0}")).collect();
                let line = format!("s{seed}\t{seat}\t{seed}\t{name}\t{world}\t{}\t{}\n", xs.join("\t"), gs.join("\t"));
                w.lock().unwrap().write_all(line.as_bytes()).unwrap();
                let mut n = done.lock().unwrap();
                *n += 1;
                if *n % 50 == 0 {
                    eprintln!("[endg-label] {} games ({:.0}s)", *n, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    w.lock().unwrap().flush().unwrap();
    eprintln!("[endg-label] done in {:.0}s -> {out}", t0.elapsed().as_secs_f64());
}
