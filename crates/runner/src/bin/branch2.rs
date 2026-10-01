//! Exact-engine labels for reactive shell v2 (crates/agent/src/rshell.rs).
//!
//! CLOSED LOOP (the discriminating substrate: copies, same lineage, clones react):
//!     branch2 --selfplay --a-args "<our agent>" --opps "copy:<args>|v611:<args>|rand:RAND" \
//!             --bank data/worlds/w64_bank.json --split train --per-world 4 [--seed-skip 0] --k 24 --out F [--threads 24]
//!   seat A = our agent with the shell in observe mode (inputs recorded, action unchanged). For each game
//!   (seed from the 64-world bank, both seats, opponent chosen per game) up to --k (step, item) decisions are
//!   sampled; each is replayed FROM THE START five times with A's sale of that item at that step forced to
//!   each fraction (0, 1/4, 1/2, 3/4, all of the projected shed). Both agents keep reacting after the fork;
//!   games are deterministic, so each replay is an exact counterfactual. Label = A's final margin.
//! OPEN LOOP (real players' tapes; the opponent keeps its recorded actions):
//!     branch2 --tapes DIR --a-args "<our agent>" --k 24 --out F
//!   the shadow agent (our agent, observe mode) supplies the inputs on the seat's history; the seat's recorded
//!   action gets each fraction for that item, then both seats follow their recorded actions to the end.
//! Row: id seat step item cur_cls stock kind x[0..NX] m0 m1 m2 m3 m4 m_real
use agent::act::Action;
use agent::base::Base;
use agent::rshell::{qty_of, NC, NI, NX};
use agent::shell::set_sell;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::state::State;
use std::io::Write;
use std::sync::{Arc, Mutex};

const FRACS: [f32; NC] = [0.0, 0.25, 0.5, 0.75, 1.0];

fn splitmix(mut x: u64) -> impl FnMut() -> u64 {
    move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A decision: (step, item index, chain class, stock, inputs).
type Dec = (i64, usize, usize, i64, [f32; NX]);

/// Play one closed-loop game. `force` = (step, item, qty) applied to seat `me`'s action after the agent acts.
/// Returns the final margin of `me` and, when `collect`, every decision's inputs.
fn play(seed: i64, agents: &mut [Base; 2], me: usize, force: Option<(i64, usize, i64)>, collect: bool) -> (f64, Vec<Dec>) {
    let mut st = State::new(seed);
    let mut decs = vec![];
    loop {
        let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
        for (s, ag) in agents.iter_mut().enumerate() {
            let o = agent::obs::Obs::from_state(&st, s);
            let step = o.step();
            let (mut a, _) = ag.act(o);
            if s == me {
                if collect {
                    if let Some(rs) = ag.rshell.as_ref() {
                        for (k, it) in rs.last.iter().enumerate() {
                            if let Some(it) = it {
                                decs.push((step, k, it.cur, it.stock, it.x));
                            }
                        }
                    }
                }
                if let Some((t, k, q)) = force {
                    if t == step {
                        set_sell(&mut a, agent::rshell::ITEMS[k], q);
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

fn pick(decs: Vec<Dec>, k: usize, seed: u64) -> Vec<Dec> {
    let mut rng = splitmix(seed);
    let (mut act, mut hold): (Vec<Dec>, Vec<Dec>) = decs.into_iter().filter(|d| d.0 < 718).partition(|d| {
        // "active": the chain sells it, or a sale is planned within 8 steps, or the rival sold it last step
        let x = &d.4;
        let f = |n: &str| x[agent::rshell::ITEMF.iter().position(|y| *y == n).unwrap()];
        d.2 > 0 || f("plan_8") > 0.0 || f("rival_sold_1") > 0.0 || f("fc_lin_4") > 0.0
    });
    for v in [&mut act, &mut hold] {
        for i in (1..v.len()).rev() {
            let j = (rng() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
    let na = act.len().min(k * 3 / 4);
    let mut out: Vec<Dec> = act.into_iter().take(na).collect();
    let nh = k.saturating_sub(out.len());
    out.extend(hold.into_iter().take(nh));
    out
}

fn row(id: &str, seat: usize, d: &Dec, kind: &str, m: &[f64; NC], real: f64) -> String {
    let xs: Vec<String> = d.4.iter().map(|v| format!("{v}")).collect();
    let ms: Vec<String> = m.iter().map(|v| format!("{v:.0}")).collect();
    format!("{id}\t{seat}\t{}\t{}\t{}\t{}\t{kind}\t{}\t{}\t{real:.0}\n", d.0, agent::rshell::ITEMS[d.1], d.2, d.3, xs.join("\t"), ms.join("\t"))
}

fn observe_shell(b: &mut Base, cfg: &str) {
    let mut c = agent::rshell::RConfig::load(cfg).expect("--shell-cfg");
    c.on = false;
    b.rshell = Some(agent::rshell::RShellCtl::new(Arc::new(c)));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = get("--out").expect("--out FILE");
    let k_max: usize = get("--k").and_then(|s| s.parse().ok()).unwrap_or(24);
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let shell_cfg = get("--shell-cfg").unwrap_or_else(|| "configs/rshell/observe.json".into());
    let split = |s: &str| s.split_whitespace().map(|x| x.to_string()).collect::<Vec<String>>();
    let mut a = Base::load(&base_dir).expect("base");
    runner::configure(&mut a, &split(&get("--a-args").unwrap_or_default()));
    // on-policy (DAgger): when --a-args already installs a shell (--rshell), it stays ON and the labels are
    // taken on the states the candidate itself visits; otherwise the shell runs in observe mode
    if a.rshell.is_none() {
        observe_shell(&mut a, &shell_cfg);
    }
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    let t0 = std::time::Instant::now();
    if args.iter().any(|x| x == "--selfplay") {
        // opponents: name:args|name:args (args "RAND" = a random clone-lineage knob vector per game)
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
        let per: usize = get("--per-world").and_then(|s| s.parse().ok()).unwrap_or(4);
        let skip: usize = get("--seed-skip").and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut jobs: Vec<(i64, usize, usize)> = vec![]; // seed, seat, opponent
        for (_, seeds) in bank.get(&get("--split").unwrap_or_else(|| "train".into())).obj() {
            for (i, s) in seeds.arr().iter().skip(skip).take(per).enumerate() {
                let seed = s.i64();
                jobs.push((seed, i % 2, (seed as usize / 7 + i) % opps.len()));
            }
        }
        eprintln!("[branch2] closed loop: {} games x {} decisions x {NC} fractions, {} opponents", jobs.len(), k_max, opps.len());
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
                        o.rshell = None;
                        let mine = a.fresh();
                        if me == 0 { [mine, o] } else { [o, mine] }
                    };
                    let (real, decs) = play(seed, &mut mk(seat), seat, None, true);
                    let picked = pick(decs, k_max, seed as u64 ^ (seat as u64) << 40);
                    let id = format!("s{seed}");
                    let mut buf = String::new();
                    for d in &picked {
                        let mut m = [0f64; NC];
                        for c in 0..NC {
                            let qn = qty_of(&FRACS, c, d.3);
                            m[c] = play(seed, &mut mk(seat), seat, Some((d.0, d.1, qn)), false).0;
                        }
                        buf.push_str(&row(&id, seat, d, name, &m, real));
                    }
                    w.lock().unwrap().write_all(buf.as_bytes()).unwrap();
                    let mut n = done.lock().unwrap();
                    *n += 1;
                    if *n % 50 == 0 {
                        eprintln!("[branch2] {} games ({:.0}s)", *n, t0.elapsed().as_secs_f64());
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    } else {
        // --tapes DIR[,DIR...] [--max-per-dir N] (seeded sample per folder)
        let dirs = get("--tapes").expect("--tapes DIR or --selfplay");
        let other_seat = args.iter().any(|x| x == "--other-seat");
        let max_per: usize = get("--max-per-dir").and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
        let mut files = vec![];
        for dir in dirs.split(',') {
            let mut fs: Vec<_> = std::fs::read_dir(dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            fs.sort();
            let mut rng = splitmix(0x7A9E_0001 ^ dir.len() as u64);
            for i in (1..fs.len()).rev() {
                let j = (rng() % (i as u64 + 1)) as usize;
                fs.swap(i, j);
            }
            files.extend(fs.into_iter().take(max_per));
        }
        eprintln!("[branch2] open loop: {} tapes", files.len());
        let q = Arc::new(Mutex::new(files));
        let a = Arc::new(a);
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                let (q, a, w) = (q.clone(), a.clone(), w.clone());
                std::thread::spawn(move || loop {
                    let Some(f) = q.lock().unwrap().pop() else { break };
                    let Ok(j) = json::parse(&std::fs::read_to_string(&f).unwrap()) else { continue };
                    let id = f.file_stem().unwrap().to_string_lossy().to_string();
                    let seed = j.get("seed").i64();
                    let seat = if other_seat { 1 - j.get("seat").i64() as usize } else { j.get("seat").i64() as usize };
                    let acts = j.get("actions").arr();
                    let rec = |s: usize| -> Vec<Action> { acts.iter().map(|p| p.arr().get(s).map(Action::from_json).unwrap_or_else(Action::pass)).collect() };
                    let (r0, r1) = (rec(0), rec(1));
                    let eng = |v: &[Action]| -> Vec<PlayerAction> { v.iter().map(runner::to_engine).collect() };
                    let (e0, e1) = (eng(&r0), eng(&r1));
                    let mine = if seat == 0 { &r0 } else { &r1 };
                    let mut st = State::new(seed);
                    let mut shadow = a.fresh();
                    let mut states: Vec<State> = Vec::with_capacity(720);
                    let mut decs: Vec<Dec> = vec![];
                    let mut projs: Vec<(i64, usize, i64)> = vec![]; // step, item, projected stock under the SEAT's action
                    let mut t = 0usize;
                    loop {
                        states.push(st.clone());
                        let o = agent::obs::Obs::from_state(&st, seat);
                        if let Ok(v) = agent::view::View::new(o.clone()) {
                            let _ = shadow.act(o);
                            if let (Some(rs), Some(theirs)) = (shadow.rshell.as_ref(), mine.get(t)) {
                                let proj = shadow.chassis.projected_shed(theirs, &v);
                                for (k, it) in rs.last.iter().enumerate() {
                                    if let Some(it) = it {
                                        let stock = agent::obs::qget(&proj, agent::rshell::ITEMS[k]);
                                        if stock > 0 {
                                            let cls = agent::rshell::class_of(&FRACS, agent::shell::sell_qty(theirs, agent::rshell::ITEMS[k]), stock);
                                            decs.push((v.step, k, cls, stock, it.x));
                                            projs.push((v.step, k, stock));
                                        }
                                    }
                                }
                            }
                        }
                        let pair = [e0.get(t).cloned().unwrap_or_default(), e1.get(t).cloned().unwrap_or_default()];
                        let alive = engine::step(&mut st, &pair);
                        t += 1;
                        if !alive {
                            break;
                        }
                    }
                    let real = st.farms[seat].money - st.farms[1 - seat].money;
                    let picked = pick(decs, k_max, seed as u64 ^ 0x7A9E);
                    let mut buf = String::new();
                    for d in &picked {
                        let ts = d.0 as usize;
                        let mut m = [0f64; NC];
                        for c in 0..NC {
                            let mut act = mine[ts].clone();
                            set_sell(&mut act, agent::rshell::ITEMS[d.1], qty_of(&FRACS, c, d.3));
                            let pa = runner::to_engine(&act);
                            let other = if seat == 0 { e1.get(ts) } else { e0.get(ts) }.cloned().unwrap_or_default();
                            let first = if seat == 0 { [pa, other] } else { [other, pa] };
                            let mut s2 = states[ts].clone();
                            let mut alive = engine::step(&mut s2, &first);
                            let mut u = ts + 1;
                            while alive {
                                let pair = [e0.get(u).cloned().unwrap_or_default(), e1.get(u).cloned().unwrap_or_default()];
                                alive = engine::step(&mut s2, &pair);
                                u += 1;
                            }
                            m[c] = s2.farms[seat].money - s2.farms[1 - seat].money;
                        }
                        buf.push_str(&row(&id, seat, d, "tape", &m, real));
                    }
                    let _ = NI;
                    w.lock().unwrap().write_all(buf.as_bytes()).unwrap();
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }
    w.lock().unwrap().flush().unwrap();
    eprintln!("[branch2] done in {:.0}s -> {out}", t0.elapsed().as_secs_f64());
}
