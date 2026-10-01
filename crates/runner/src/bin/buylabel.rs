//! Exact-engine labels for the shell v3 buy head (BUY_PRODUCT WHEAT, crates/agent/src/rshell.rs).
//!
//!     buylabel --a-args "<our agent incl. --rshell>" --opps "name:<args>|...|rand:RAND" \
//!              --bank data/worlds/w64_bank.json --split train --per-world 6 [--seed-skip 0] --k 12 --out F [--threads 24]
//!
//! Seat A = our agent as shipped (shell on). Per game, up to --k decision steps are sampled among steps 144..700 at
//! hours 1,5,9,13,17,21 where A does not sell wheat and has a free market slot; each is replayed FROM THE START with
//! `BUY_PRODUCT WHEAT q` appended to A's action at that step, q in {5, 10, 20}. Both agents keep reacting after
//! the fork (closed loop); games are deterministic, so each replay is an exact counterfactual (the shell sells the
//! bought wheat whenever it judges best). Label = A's final margin.
//! Row: id seat step opp x[0..NX] m0 m5 m10 m20   (m0 = the unforced game)
use agent::base::Base;
use agent::rshell::NX;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::state::State;
use std::io::Write;
use std::sync::{Arc, Mutex};

const QS: [i64; 3] = [5, 10, 20];

fn splitmix(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

type Dec = (i64, [f32; NX]);

fn play(seed: i64, agents: &mut [Base; 2], me: usize, force: Option<(i64, i64)>, collect: bool) -> (f64, Vec<Dec>) {
    let mut st = State::new(seed);
    let mut decs = vec![];
    loop {
        let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
        for (s, ag) in agents.iter_mut().enumerate() {
            let o = agent::obs::Obs::from_state(&st, s);
            let step = o.step();
            let (mut a, _) = ag.act(o);
            if s == me {
                let sells_wheat = a.market.iter().any(|c| c.is_sell3() && c.s(1) == "WHEAT" && c.n(2) > 0);
                if collect && (144..=700).contains(&step) && step % 4 == 1 && !sells_wheat && a.market.len() < 10 {
                    if let Some(x) = ag.rshell.as_ref().and_then(|r| r.buy_x) {
                        decs.push((step, x));
                    }
                }
                if let Some((t, q)) = force {
                    if t == step && a.market.len() < 10 {
                        a.market.push(agent::act::Cmd::order("BUY_PRODUCT", "WHEAT", q));
                    }
                }
            }
            acts.push(runner::to_engine(&a));
        }
        let pair = [acts.remove(0), acts.remove(0)];
        if !engine::step(&mut st, &pair) {
            break;
        }
    }
    (st.farms[me].money - st.farms[1 - me].money, decs)
}

/// Open loop: `a` plays `me` live vs the other seat's recorded actions (a real ladder game's opponent).
fn play_tape(seed: i64, opp: &[PlayerAction], a: &mut Base, me: usize, force: Option<(i64, i64)>, collect: bool) -> (f64, Vec<Dec>) {
    let mut st = State::new(seed);
    let mut decs = vec![];
    let mut t = 0usize;
    loop {
        let o = agent::obs::Obs::from_state(&st, me);
        let step = o.step();
        let (mut act, _) = a.act(o);
        let sells_wheat = act.market.iter().any(|c| c.is_sell3() && c.s(1) == "WHEAT" && c.n(2) > 0);
        if collect && (144..=700).contains(&step) && step % 4 == 1 && !sells_wheat && act.market.len() < 10 {
            if let Some(x) = a.rshell.as_ref().and_then(|r| r.buy_x) {
                decs.push((step, x));
            }
        }
        if let Some((ft, q)) = force {
            if ft == step && act.market.len() < 10 {
                act.market.push(agent::act::Cmd::order("BUY_PRODUCT", "WHEAT", q));
            }
        }
        let mine = runner::to_engine(&act);
        let other = opp.get(t).cloned().unwrap_or_default();
        let pair = if me == 0 { [mine, other] } else { [other, mine] };
        t += 1;
        if !engine::step(&mut st, &pair) {
            break;
        }
    }
    (st.farms[me].money - st.farms[1 - me].money, decs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = get("--out").expect("--out FILE");
    let k_max: usize = get("--k").and_then(|s| s.parse().ok()).unwrap_or(12);
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let split = |s: &str| s.split_whitespace().map(|x| x.to_string()).collect::<Vec<String>>();
    let mut a = Base::load(&base_dir).expect("base");
    runner::configure(&mut a, &split(&get("--a-args").unwrap_or_default()));
    assert!(a.rshell.is_some(), "--a-args must install the shipped shell (--rshell)");
    a.rshell.as_mut().unwrap().want_buy_x = true;
    if let Some(dir) = get("--tapes") {
        // OPEN LOOP on real ladder games (the players who beat us, and the games we won): tape seat = OUR seat
        let mut files: Vec<std::path::PathBuf> = vec![];
        for d in dir.split(',') {
            files.extend(std::fs::read_dir(d).expect("tapes dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")));
        }
        files.sort();
        eprintln!("[buylabel] open loop: {} tapes x {} decisions x {} quantities", files.len(), k_max, QS.len());
        let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
        let q = Arc::new(Mutex::new(files));
        let a = Arc::new(a);
        let t0 = std::time::Instant::now();
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                let (q, a, w) = (q.clone(), a.clone(), w.clone());
                std::thread::spawn(move || loop {
                    let Some(f) = q.lock().unwrap().pop() else { break };
                    let Ok(j) = json::parse(&std::fs::read_to_string(&f).unwrap()) else { continue };
                    let id = f.file_stem().unwrap().to_string_lossy().to_string();
                    let seed = j.get("seed").i64();
                    let seat = j.get("seat").i64() as usize;
                    let opp: Vec<PlayerAction> = j.get("actions").arr().iter().map(|p| p.arr().get(1 - seat).map(agent::act::Action::from_json).unwrap_or_else(agent::act::Action::pass)).map(|x| runner::to_engine(&x)).collect();
                    let (real, mut decs) = play_tape(seed, &opp, &mut a.fresh(), seat, None, true);
                    let mut rng = splitmix(seed as u64 ^ 0x7A9E);
                    for i in (1..decs.len()).rev() {
                        let jx = (rng() % (i as u64 + 1)) as usize;
                        decs.swap(i, jx);
                    }
                    decs.truncate(k_max);
                    let mut buf = String::new();
                    for d in &decs {
                        let ms: Vec<String> = QS.iter().map(|qn| format!("{:.0}", play_tape(seed, &opp, &mut a.fresh(), seat, Some((d.0, *qn)), false).0)).collect();
                        let xs: Vec<String> = d.1.iter().map(|v| format!("{v}")).collect();
                        buf.push_str(&format!("{id}	{seat}	{}	tape	{}	{real:.0}	{}
", d.0, xs.join("	"), ms.join("	")));
                    }
                    w.lock().unwrap().write_all(buf.as_bytes()).unwrap();
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        w.lock().unwrap().flush().unwrap();
        eprintln!("[buylabel] done in {:.0}s -> {out}", t0.elapsed().as_secs_f64());
        return;
    }
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
    let per: usize = get("--per-world").and_then(|s| s.parse().ok()).unwrap_or(6);
    let skip: usize = get("--seed-skip").and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut jobs: Vec<(i64, usize, usize)> = vec![];
    for (_, seeds) in bank.get(&get("--split").unwrap_or_else(|| "train".into())).obj() {
        for (i, s) in seeds.arr().iter().skip(skip).take(per).enumerate() {
            let seed = s.i64();
            jobs.push((seed, i % 2, (seed as usize / 7 + i) % opps.len()));
        }
    }
    eprintln!("[buylabel] {} games x {} decisions x {} quantities, {} opponents", jobs.len(), k_max, QS.len(), opps.len());
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    let t0 = std::time::Instant::now();
    let q = Arc::new(Mutex::new(jobs));
    let opps = Arc::new(opps);
    let a = Arc::new(a);
    let done = Arc::new(Mutex::new(0usize));
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, opps, a, w, done) = (q.clone(), opps.clone(), a.clone(), w.clone(), done.clone());
            std::thread::spawn(move || loop {
                let Some((seed, seat, oi)) = q.lock().unwrap().pop() else { break };
                let (name, ob) = &opps[oi];
                let mk = |me: usize| -> [Base; 2] {
                    let mut o = match ob {
                        Some(b) => b.fresh(),
                        None => {
                            let mut b = a.fresh();
                            b.rshell = None;
                            b.policy = None;
                            b.obs_track = None;
                            let mut r = splitmix(seed as u64 ^ 0xB2A2);
                            let kn = agent::layers::knobs::Knobs::random(&mut r);
                            b.chain.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), kn)];
                            b.chain.schedule = Some(vec![1; 30]);
                            b
                        }
                    };
                    if ob.is_none() {
                        o.rshell = None;
                    }
                    let mine = a.fresh();
                    if me == 0 { [mine, o] } else { [o, mine] }
                };
                let (real, mut decs) = play(seed, &mut mk(seat), seat, None, true);
                let mut rng = splitmix(seed as u64 ^ (seat as u64) << 40 ^ 0xB0B);
                for i in (1..decs.len()).rev() {
                    let j = (rng() % (i as u64 + 1)) as usize;
                    decs.swap(i, j);
                }
                decs.truncate(k_max);
                let mut buf = String::new();
                for d in &decs {
                    let ms: Vec<String> = QS.iter().map(|qn| format!("{:.0}", play(seed, &mut mk(seat), seat, Some((d.0, *qn)), false).0)).collect();
                    let xs: Vec<String> = d.1.iter().map(|v| format!("{v}")).collect();
                    buf.push_str(&format!("s{seed}\t{seat}\t{}\t{name}\t{}\t{real:.0}\t{}\n", d.0, xs.join("\t"), ms.join("\t")));
                }
                w.lock().unwrap().write_all(buf.as_bytes()).unwrap();
                let mut n = done.lock().unwrap();
                *n += 1;
                if *n % 25 == 0 {
                    eprintln!("[buylabel] {} games ({:.0}s)", *n, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    eprintln!("[buylabel] done in {:.0}s -> {out}", t0.elapsed().as_secs_f64());
}
