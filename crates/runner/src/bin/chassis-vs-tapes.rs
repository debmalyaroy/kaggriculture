//! chassis-vs-tapes: the chassis ALONE (no layers) replaces one seat of every recorded game and plays the recorded
//! opponent on the same seed (v63.12 chassis acceptance test). Parallel, one tape in memory per thread.
//!
//!     chassis-vs-tapes --chassis weights/chassis/HASH --tapes DIR[,DIR...] --out F.tsv [--threads 12]
//!     chassis-vs-tapes --agent CANDIDATE/agent.json --tapes ... --out F.tsv   (the full agent, every manager)
//!                      [--guard-base configs/bases/v61.1]
//!
//! Tape = python/replay_to_tape.py format (actions[t] = [seat0, seat1], `seat` = the seat we take; the other seat's
//! recorded stream is the opponent, played as a guarded route: runner::guarded_tape_base -- its recorded stream kept
//! legal against a different us). Our side = Base::load(chassis dir) with the chain cut at `chassis`.
//! Row: id seat band world our_bank their_bank gap route (the chassis route we ended on) recorded_our_bank.
use agent::base::Base;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::state::State;
use std::io::Write;
use std::sync::{Arc, Mutex};

/// KRL_TRACE_DIR=D: one D/<id>.trace.tsv per tape. Rows:
///   D step who money quads hands empty locked weed plants{crop:n} dry animals{kind:n} empty_structs shed{item:n}
///      at the first step of every day, for us and them
///   M step who market-orders   whenever a seat sends a non-empty market order list
fn trace_step(w: &mut impl Write, st: &State, seat: usize, pair: &[PlayerAction; 2]) {
    use kagg_engine::state::Cell;
    let who = |s: usize| if s == seat { "us" } else { "them" };
    if st.step % 24 == 0 {
        for s in [seat, 1 - seat] {
            let fm = &st.farms[s];
            let (mut empty, mut locked, mut weed, mut dry, mut estr) = (0, 0, 0, 0, 0);
            let mut plants: std::collections::BTreeMap<&str, i64> = Default::default();
            let mut animals: std::collections::BTreeMap<&str, i64> = Default::default();
            for row in &fm.tiles {
                for c in row {
                    match c {
                        Cell::Empty => empty += 1,
                        Cell::Locked => locked += 1,
                        Cell::Weed => weed += 1,
                        Cell::Plant { crop, consecutive_unwatered, .. } => {
                            *plants.entry(crop).or_default() += 1;
                            if *consecutive_unwatered > 0 {
                                dry += 1;
                            }
                        }
                        Cell::Structure { kind, animal } => match animal {
                            Some(a) => *animals.entry(a.animal).or_default() += 1,
                            None => {
                                let _ = kind;
                                estr += 1
                            }
                        },
                    }
                }
            }
            let shed: Vec<String> = st.private[s].shed.0.iter().filter(|(_, n)| *n > 0).map(|(k, n)| format!("{k}:{n}")).collect();
            let _ = writeln!(
                w,
                "D\t{}\t{}\t{:.0}\t{}\t{}\t{empty}\t{locked}\t{weed}\t{:?}\t{dry}\t{:?}\t{estr}\t{}",
                st.step,
                who(s),
                fm.money,
                fm.unlocked_quadrants.join(","),
                fm.hands.len(),
                plants,
                animals,
                shed.join(",")
            );
        }
    }
    for s in [seat, 1 - seat] {
        let m: Vec<String> = pair[s].market.iter().filter(|o| !o.is_empty() && !(o.len() > 2 && o[0] == "SELL" && o[2] == "0")).map(|o| o.join(" ")).collect();
        if !m.is_empty() {
            let _ = writeln!(w, "M\t{}\t{}\t{}", st.step, who(s), m.join(" | "));
        }
    }
}

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let chassis = get("--chassis").unwrap_or_else(|| "-".into());
    let dirs = get("--tapes").expect("--tapes DIR[,DIR]");
    let out = get("--out").expect("--out FILE");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(12);
    let guard_base = get("--guard-base").unwrap_or_else(|| "configs/bases/v61.1".into());
    // --agent agent.json : the FULL agent (every manager of the config, cut = full) instead of the chassis alone; the
    // agent is built by agent::cli exactly as agent-stdio builds the submission (fast-track tournament, 30 Sep)
    let mut ours = match get("--agent") {
        Some(cfg) => agent::cli::build(vec!["agent-stdio".into(), "--config".into(), cfg]).agent.base,
        None => {
            let mut b = Base::load(&chassis).expect("chassis");
            b.cut = agent::base::cut_index("chassis").unwrap();
            b
        }
    };
    // --settings '{"budget_guard":true,...}' : a guard-grid cell over the router's settings
    if let Some(js) = get("--settings") {
        ours.chassis.cfg.apply(&kagg_engine::json::parse(&js).expect("--settings json"));
    }
    // --route R : play route R in every world after the switch (route screening)
    if let Some(r) = get("--route").and_then(|s| s.parse::<i64>().ok()) {
        ours.chassis.router.force(r);
    }
    ours.chassis.diagnostics.log = true;
    let ours = Arc::new(ours);
    let guard = Arc::new(runner::guard_router(&guard_base));
    let mut files = vec![];
    for d in dirs.split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    let n = files.len();
    eprintln!("[chassis-vs-tapes] {n} tapes, {threads} threads, chassis {chassis}");
    let w = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out).expect("out"))));
    w.lock().unwrap().write_all(b"id\tseat\tband\tworld\tour_bank\ttheir_bank\tgap\troute\trecorded_our_bank\n").unwrap();
    let q = Arc::new(Mutex::new(files));
    let tally = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
    let t0 = std::time::Instant::now();
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, w, tally, ours, guard) = (q.clone(), w.clone(), tally.clone(), ours.clone(), guard.clone());
            std::thread::spawn(move || loop {
                let Some(f) = q.lock().unwrap().pop() else { break };
                let Ok(txt) = std::fs::read_to_string(&f) else { continue };
                let Ok(j) = kagg_engine::json::parse(&txt) else { continue };
                let seat = j.get("seat").i64() as usize;
                let seed = j.get("seed").i64();
                let stream: Vec<agent::act::Action> = j.get("actions").arr().iter().map(|p| p.arr().get(1 - seat).map(agent::act::Action::from_json).unwrap_or_else(agent::act::Action::pass)).collect();
                let rec: f64 = j.get("rewards").arr().get(seat).map(|x| x.f64()).unwrap_or(f64::NAN);
                let rec_them: f64 = j.get("rewards").arr().get(1 - seat).map(|x| x.f64()).unwrap_or(f64::NAN);
                let mut days = [0f64; 6];
                let tape = runner::Tape { id: j.get("id").str().to_string(), seed, seat, band: j.get("band").str().to_string(), stream: Arc::new(stream) };
                let opp = runner::guarded_tape_base(&tape, &guard);
                let mut ag: [Base; 2] = if seat == 0 { [ours.fresh(), opp] } else { [opp, ours.fresh()] };
                let mut st = State::new(seed);
                let tid = f.file_stem().unwrap().to_string_lossy().to_string();
                let mut trace = std::env::var("KRL_TRACE_DIR").ok().map(|d| {
                    std::io::BufWriter::new(std::fs::File::create(std::path::Path::new(&d).join(format!("{tid}.trace.tsv"))).expect("trace file"))
                });
                loop {
                    let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
                    for (s, a) in ag.iter_mut().enumerate() {
                        let (act, _) = a.act(agent::obs::Obs::from_state(&st, s));
                        acts.push(runner::to_engine(&act));
                    }
                    let pair = [acts.remove(0), acts.remove(0)];
                    if let Some(tr) = trace.as_mut() {
                        trace_step(tr, &st, seat, &pair);
                    }
                    for (k, d) in [144i64, 288, 576].iter().enumerate() {
                        if st.step == *d {
                            days[2 * k] = st.farms[seat].money;
                            days[2 * k + 1] = st.farms[1 - seat].money;
                        }
                    }
                    if !engine::step(&mut st, &pair) {
                        break;
                    }
                }
                if let Some(tr) = trace.as_mut() {
                    // E: the final state -- what each side still holds unsold (shed + carried) when the game ends
                    for s in [seat, 1 - seat] {
                        let mut left: std::collections::BTreeMap<&str, i64> = Default::default();
                        for (k, n) in st.private[s].shed.0.iter().chain(st.private[s].inventories.iter().flat_map(|i| i.0.iter())) {
                            if *n > 0 {
                                *left.entry(k).or_default() += n;
                            }
                        }
                        let _ = writeln!(tr, "E	{}	{}	{:.0}	{:?}", st.step, if s == seat { "us" } else { "them" }, st.farms[s].money, left);
                    }
                }
                let sh = &st.town.unlocked_shops;
                let world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
                let (us, them) = (st.farms[seat].money, st.farms[1 - seat].money);
                let route = ag[seat].chassis.players.iter().find(|(p, _)| *p == seat as i64).and_then(|(_, s)| s.route).unwrap_or(-1);
                let id = f.file_stem().unwrap().to_string_lossy().to_string();
                let g = ag[seat].chassis.diagnostics.summary();
                let dd: Vec<String> = days.iter().map(|x| format!("{x:.0}")).collect();
                let line = format!("{id}\t{seat}\t{}\t{world}\t{us:.0}\t{them:.0}\t{:.0}\t{route}\t{rec:.0}\t{rec_them:.0}\t{}\t{g}\n", tape.band, us - them, dd.join("\t"));
                w.lock().unwrap().write_all(line.as_bytes()).unwrap();
                let mut t = tally.lock().unwrap();
                t.0 += 1;
                if us > them {
                    t.1 += 1;
                } else if us < them {
                    t.2 += 1;
                }
                if t.0 % 1000 == 0 {
                    eprintln!("[chassis-vs-tapes] {} / {n}: W {} L {} ({:.0}s)", t.0, t.1, t.2, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    w.lock().unwrap().flush().unwrap();
    let t = tally.lock().unwrap();
    eprintln!("[chassis-vs-tapes] done: {} tapes W {} L {} D {} in {:.0}s -> {out}", t.0, t.1, t.2, t.0 - t.1 - t.2, t0.elapsed().as_secs_f64());
}
