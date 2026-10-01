//! PPO rollouts: the learner seat plays the macro policy (sampling), the opponent comes from the
//! league mix or a frozen snapshot of earlier policy weights. Called by python/learn/ppo.py once
//! per iteration.
//!
//!     ppo-rollout --weights W.bin --out PREFIX --games 1024 --seed0 S [--threads 12]
//!                 [--snaps DIR] [--allow 0,1,...] [--profiles configs/profiles/v2.json]
//!                 [--shield configs/shield/v1.json]   (opponent-group mask on the policy's choice)
//!                 [--tapes data/tapes/train --tape-frac 0.3]   (real players' streams on their own seeds)
//!                 [--greedy --oracle-days 0,1,2,3,12,20,23,26 --oracle-cands 0,2,13,19,...]   (branch oracle:
//!                  PREFIX.oracle = per key day the learner's margin under each forced candidate profile)
//!                 [--learner-fixed K]   (reference: learner plays profile K, no policy; --weights optional)
//!                 [--greedy] [--opp v611|fixed:K]   (validation: argmax policy vs a fixed opponent;
//!                                                    seeds below 1,000,000 allowed with --greedy)
//!
//!                 [--mix mirror:25,snap:15,v611:10,fixed:20,v63:10,rand:20]   (ppo2: opponent mix in percent;
//!                  mirror = the learner's own policy, v63 = fixed profile 35; absent = the original mix)
//!                 [--seed-bank data/worlds/w64_bank.json]   (non-tape games play a bank seed of a world chosen
//!                  uniformly over the 64 realized worlds: every world is trained)
//!                 [--rshell F] [--chain-off S] [--knob-over F]   (learner only: runner::apply_extras)
//!                 [--shell F]   (learner only: the v1 learned sales shell)
//! Mid-day decisions: when the learner's weights carry a mid-day trailer (policy::MID_MAGIC), each game has
//! dayobs::n_slots(mid) decision slots in chronological order (days 25-29 get a second one at hour 13) and
//! --oracle-days takes "d" (hour 1 of day d) and "d.13" (hour 13 of mid day d). Without it, slot = day.
//! Writes PREFIX.tsv (seed seat opp_kind opp_desc bank_learner bank_opp score) and PREFIX.traj
//! (games x 30 days x [dayobs::N obs, action, logp, value_logit, mask] little-endian f32, `mask`
//! being the allowed-profile bits (u64 as two u32 slots: low, high) the choice was sampled under; days the game never reached
//! are all zero with action -1). Both written atomically.
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

// rl3 ids: v2 escalated/AFR + 33 p19_v92, 34 p19_v92ext, 35 = v63 (the live agent to beat)
const FIXED_OPP: [usize; 14] = [2, 12, 13, 19, 21, 22, 23, 24, 29, 30, 31, 33, 34, 35];
/// Per day: obs[N], action, logp, value logit, allowed-profile mask (low / high u32 bits, each stored in an f32 slot).
pub const W: usize = dayobs::N + 5;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let prof = get("--profiles").unwrap_or_else(|| "configs/profiles/rl3.json".into());
    let out = get("--out").expect("--out PREFIX");
    let games: u64 = get("--games").and_then(|s| s.parse().ok()).unwrap_or(256);
    let seed0: u64 = get("--seed0").and_then(|s| s.parse().ok()).expect("--seed0");
    let greedy = args.iter().any(|a| a == "--greedy");
    let fixed_opp = get("--opp");
    assert!(greedy || seed0 >= 1_000_000, "seeds below 1,000,000 are reserved for validation");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let learner_fixed: Option<usize> = get("--learner-fixed").and_then(|s| s.parse().ok());
    let net: Option<Arc<policy::Net>> = match get("--weights") {
        Some(w) => Some(Arc::new(policy::Net::load(&w).expect("weights"))),
        None => {
            assert!(learner_fixed.is_some(), "--weights required unless --learner-fixed");
            None
        }
    };
    let snaps: Vec<Arc<policy::Net>> = get("--snaps")
        .map(|d| {
            let mut v: Vec<_> = std::fs::read_dir(d).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).collect()).unwrap_or_default();
            v.sort();
            v.into_iter().filter(|p| p.extension().is_some_and(|x| x == "bin")).filter_map(|p| policy::Net::load(p.to_str().unwrap()).ok()).map(Arc::new).collect()
        })
        .unwrap_or_default();
    let mut base = agent::base::Base::load(&base_dir).expect("base");
    base.set_profiles(&prof, None).expect("profiles");
    let n_prof = base.chain.profiles.len();
    if let Some(n) = net.as_ref() {
        assert_eq!(n_prof, n.n_act, "policy action count != profile table");
    }
    let mut allow = vec![true; n_prof];
    if let Some(s) = get("--allow") {
        allow = vec![false; n_prof];
        for x in s.split(',').filter_map(|x| x.trim().parse::<usize>().ok()) {
            if x < n_prof {
                allow[x] = true;
            }
        }
    }
    // real ladder players (python/band_tapes.py, a training window disjoint from the gate's): with
    // probability --tape-frac a game is that player's recorded stream on ITS seed, in its seat
    let tapes: Arc<Vec<runner::Tape>> = Arc::new(get("--tapes").map(|d| runner::load_tapes(&d)).unwrap_or_default());
    let tape_frac: f64 = get("--tape-frac").and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let router = Arc::new(runner::guard_router(&base_dir));
    // --shell F: the learner plays with the v1 learned sales shell, as the shipped agent does (27 Sep: PPO3 trained
    // without it scored +0.132 in validation but lost lineage +41/-74 when played with it)
    let learn_shell = get("--shell").map(|f| Arc::new(agent::shell::ShellModel::load(&f).expect("shell model")));
    if !tapes.is_empty() {
        eprintln!("[ppo-rollout] {} tape opponents, {:.0}% of games", tapes.len(), tape_frac * 100.0);
    }
    if let Some(f) = get("--shield") {
        base.chain.group_ctl = Some(load_shield(&f).expect("shield"));
    }
    base.obs_track = Some(dayobs::Builder::new());
    let t0 = std::time::Instant::now();
    let queue = Arc::new(Mutex::new((seed0..seed0 + games).rev().collect::<Vec<_>>()));
    let results = Arc::new(Mutex::new(Vec::<(u64, String, Vec<f32>, Vec<String>)>::new()));
    // branch oracle: key days and candidate profiles (default: no oracle)
    let oracle_days: Arc<Vec<usize>> = Arc::new(get("--oracle-days").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default());
    let oracle_max_margin: f64 = get("--oracle-max-margin").and_then(|s| s.parse().ok()).unwrap_or(f64::INFINITY);
    let oracle_cands: Arc<Vec<usize>> = Arc::new(get("--oracle-cands").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| (0..32).collect()));
    // decision slots per game (30 without mid-day decisions)
    let mid: Vec<usize> = net.as_ref().map(|n| n.mid.clone()).unwrap_or_default();
    let n_slots = dayobs::n_slots(&mid);
    // oracle days as slots: "d" = hour 1 of day d, "d.13" = the hour-13 decision of mid day d
    let oracle_slots: Arc<Vec<usize>> = Arc::new(
        get("--oracle-days")
            .map(|s| {
                s.split(',')
                    .filter_map(|x| {
                        let x = x.trim();
                        let (d, h) = match x.split_once('.') {
                            Some((d, h)) => (d.parse::<usize>().ok()?, h.parse::<usize>().ok()?),
                            None => (x.parse::<usize>().ok()?, 1),
                        };
                        if h != 1 && !(h == dayobs::MID_HOUR && mid.contains(&d)) {
                            return None;
                        }
                        Some(dayobs::slot_of(d * dayobs::TURNS_PER_DAY + h, &mid))
                    })
                    .collect()
            })
            .unwrap_or_default(),
    );
    let _ = &oracle_days;
    // --mix: opponent kinds with percent weights (cumulative thresholds over 0..1000)
    let mix: Option<Arc<Vec<(String, u64)>>> = get("--mix").map(|s| {
        let parts: Vec<(String, f64)> = s
            .split(',')
            .filter_map(|x| x.split_once(':').and_then(|(k, w)| w.trim().parse::<f64>().ok().map(|w| (k.trim().to_string(), w))))
            .collect();
        let tot: f64 = parts.iter().map(|p| p.1).sum::<f64>().max(1e-9);
        let mut acc = 0.0;
        let mut v = vec![];
        for (k, w) in parts {
            assert!(["mirror", "snap", "v611", "fixed", "v63", "rand"].contains(&k.as_str()), "--mix kind {k}");
            acc += w / tot;
            v.push((k, (acc * 1000.0).round() as u64));
        }
        Arc::new(v)
    });
    // --seed-bank: {"train": {"SHOP|SHOP": [seed, ...], ...}} (python: the 64-world bank)
    let bank: Arc<Vec<Vec<i64>>> = Arc::new(
        get("--seed-bank")
            .map(|f| {
                let j = kagg_engine::json::parse(&std::fs::read_to_string(&f).expect("seed bank")).expect("seed bank json");
                let mut ws: Vec<(String, Vec<i64>)> = j.get("train").obj().iter().map(|(k, v)| (k.clone(), v.arr().iter().map(|x| x.i64()).collect())).collect();
                ws.sort();
                ws.into_iter().map(|w| w.1).filter(|v: &Vec<i64>| !v.is_empty()).collect()
            })
            .unwrap_or_default(),
    );
    if !bank.is_empty() {
        eprintln!("[ppo-rollout] seed bank: {} worlds", bank.len());
    }
    let args = Arc::new(args.clone());
    let mut hs = vec![];
    for _ in 0..threads.max(1) {
        let (q, res, b, net, snaps, allow, fixed_opp) = (queue.clone(), results.clone(), base.fresh(), net.clone(), snaps.clone(), allow.clone(), fixed_opp.clone());
        let (tapes, router, learn_shell) = (tapes.clone(), router.clone(), learn_shell.clone());
        let (oracle_days, oracle_cands) = (oracle_slots.clone(), oracle_cands.clone());
        let (mix, bank, args) = (mix.clone(), bank.clone(), args.clone());
        hs.push(std::thread::spawn(move || loop {
            let Some(seed) = q.lock().unwrap().pop() else { break };
            // the whole game setup is a pure function of the seed, so an oracle branch can rebuild the
            // identical game (same opponent, world, seat and rng stream) and change one day's choice
            let setup = |seed: u64| {
                let mut rng = splitmix(seed ^ 0x5EED_0F_99AA_BB11);
                let tape = (fixed_opp.is_none() && !tapes.is_empty() && (rng() % 10_000) as f64 / 10_000.0 < tape_frac)
                    .then(|| tapes[(rng() % tapes.len() as u64) as usize].clone());
                let learner = tape.as_ref().map(|t| t.seat).unwrap_or((seed & 1) as usize);
                let mut agents = [b.fresh(), b.fresh()];
                match (learner_fixed, net.as_ref()) {
                    (Some(k), _) => agents[learner].chain.schedule = Some(vec![k.min(n_prof - 1); 30]),
                    (None, Some(n)) => agents[learner].policy = Some(agent::base::PolicyCtl::new(n.clone(), allow.clone(), if greedy { None } else { Some(rng()) })),
                    (None, None) => unreachable!(),
                }
                // --rshell / --chain-off / --knob-over: the learner only
                runner::apply_extras(&mut agents[learner], &args);
                if let Some(m) = &learn_shell {
                    agents[learner].shell = Some(agent::shell::ShellCtl::new(m.clone()));
                }
                let opp = &mut agents[1 - learner];
                // --mix: pick the opponent kind by weight (tapes and a fixed --opp take precedence)
                let mixed: Option<String> = match (mix.as_ref(), tape.is_none() && fixed_opp.is_none()) {
                    (Some(m), true) => {
                        let u = rng() % 1000;
                        m.iter().find(|(_, c)| u < *c).or(m.last()).map(|(k, _)| k.clone())
                    }
                    _ => None,
                };
                let r = if fixed_opp.is_some() || mixed.is_some() { 1000 } else { rng() % 100 };
                let (kind, desc) = if let Some(t) = tape.as_ref() {
                    *opp = runner::guarded_tape_base(t, &router);
                    ("tape", format!("{}:{}", t.band, t.id))
                } else if let Some(f) = fixed_opp.as_deref() {
                    opp.obs_track = None;
                    let p = f.strip_prefix("fixed:").and_then(|x| x.parse::<usize>().ok()).unwrap_or(0).min(n_prof - 1);
                    opp.chain.schedule = Some(vec![p; 30]);
                    ("fixedval", p.to_string())
                } else if let Some(k) = mixed.as_deref() {
                    match k {
                        "mirror" if net.is_some() => {
                            // a closed-loop copy: the learner's own current policy in the other seat
                            opp.policy = Some(agent::base::PolicyCtl::new(net.clone().unwrap(), allow.clone(), if greedy { None } else { Some(rng()) }));
                            ("mirror", "cur".to_string())
                        }
                        "snap" if !snaps.is_empty() => {
                            let i = (rng() % snaps.len() as u64) as usize;
                            opp.policy = Some(agent::base::PolicyCtl::new(snaps[i].clone(), allow.clone(), Some(rng())));
                            ("snap", i.to_string())
                        }
                        "fixed" => {
                            opp.obs_track = None;
                            let p = FIXED_OPP[(rng() % FIXED_OPP.len() as u64) as usize].min(n_prof - 1);
                            opp.chain.schedule = Some(vec![p; 30]);
                            ("fixed", p.to_string())
                        }
                        "v63" => {
                            opp.obs_track = None;
                            opp.chain.schedule = Some(vec![35usize.min(n_prof - 1); 30]);
                            ("v63", "35".to_string())
                        }
                        "rand" => {
                            opp.obs_track = None;
                            let kn = agent::layers::knobs::Knobs::random(&mut rng);
                            opp.chain.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), kn)];
                            opp.chain.schedule = Some(vec![1; 30]);
                            ("randknobs", format!("{seed:x}"))
                        }
                        _ => {
                            opp.obs_track = None;
                            opp.chain.schedule = Some(vec![0; 30]);
                            ("v611", "0".to_string())
                        }
                    }
                } else if r < 20 && !snaps.is_empty() {
                    let i = (rng() % snaps.len() as u64) as usize;
                    opp.policy = Some(agent::base::PolicyCtl::new(snaps[i].clone(), allow.clone(), Some(rng())));
                    ("snap", i.to_string())
                } else if r < 40 {
                    opp.obs_track = None;
                    opp.chain.schedule = Some(vec![0; 30]);
                    ("v611", "0".to_string())
                } else if r < 75 {
                    opp.obs_track = None;
                    let p = FIXED_OPP[(rng() % FIXED_OPP.len() as u64) as usize].min(n_prof - 1);
                    opp.chain.schedule = Some(vec![p; 30]);
                    ("fixed", p.to_string())
                } else {
                    opp.obs_track = None;
                    let kn = agent::layers::knobs::Knobs::random(&mut rng);
                    opp.chain.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), kn)];
                    opp.chain.schedule = Some(vec![1; 30]);
                    ("randknobs", format!("{seed:x}"))
                };
                // --seed-bank: a uniformly chosen realized world, then one of its seeds (own rng stream, so the
                // setup draws above are unchanged)
                let bank_seed = (!bank.is_empty()).then(|| {
                    let mut r2 = splitmix(seed ^ 0xB4E4_64B0_0C0F_FEE1);
                    let w = &bank[(r2() % bank.len() as u64) as usize];
                    w[(r2() % w.len() as u64) as usize]
                });
                let world = tape.as_ref().map(|t| t.seed).or(bank_seed).unwrap_or(seed as i64);
                (agents, learner, kind, desc, world)
            };
            let (mut agents, learner, kind, desc, world) = setup(seed);
            let g = runner::play(world, &mut agents);
            let (bl, bo) = (g.banks[learner], g.banks[1 - learner]);
            let score = if bl > bo { 1.0 } else if bl < bo { 0.0 } else { 0.5 };
            // --oracle-days: exact counterfactual branches of this (greedy) game. For each key day d and
            // each allowed candidate k, replay the identical game with the learner forced to k on day d
            // (every other day stays the policy's own play) and record the learner's bank margin.
            let mut orows: Vec<String> = vec![];
            // --oracle-max-margin M: branch only games lost or won by less than M (where one day can flip
            // the result; 73% of v62.1's band-gate losses are within $3,000)
            if !oracle_days.is_empty() && (bl - bo) < oracle_max_margin {
                let trace = agents[learner].policy.as_ref().map(|p| p.trace.clone()).unwrap_or_default();
                for &d in oracle_days.iter() {
                    let Some(&(_, base_k, _, _, bits)) = trace.iter().find(|t| t.0 == d) else { continue };
                    let mut cells = vec![format!("{base_k}:{}", bl - bo)];
                    if std::env::var("KRL_ORACLE_SELFCHECK").is_ok() {
                        // exactness: forcing the policy's own choice must reproduce the base game to the dollar
                        let (mut ag, l2, _, _, w2) = setup(seed);
                        if let Some(p) = ag[l2].policy.as_mut() {
                            p.force = Some((d, base_k));
                        }
                        let gb = runner::play(w2, &mut ag);
                        let m = gb.banks[l2] - gb.banks[1 - l2];
                        eprintln!("[oracle-selfcheck] seed {seed} day {d} base {} replay {} {}", bl - bo, m, if m == bl - bo { "EXACT" } else { "MISMATCH" });
                    }
                    for &k in oracle_cands.iter() {
                        if k == base_k || k >= 64 || (bits >> k) & 1 == 0 {
                            continue; // the base game already scored base_k; masked profiles are not options
                        }
                        let (mut ag, l2, _, _, w2) = setup(seed);
                        if let Some(p) = ag[l2].policy.as_mut() {
                            p.force = Some((d, k));
                        }
                        let gb = runner::play(w2, &mut ag);
                        cells.push(format!("{k}:{}", gb.banks[l2] - gb.banks[1 - l2]));
                    }
                    orows.push(format!("{seed}\t{d}\t{}", cells.join(",")));
                }
            }
            let mut traj = vec![0f32; n_slots * W];
            for d in 0..n_slots {
                traj[d * W + dayobs::N] = -1.0;
            }
            for (d, v) in &agents[learner].day_obs {
                if *d < n_slots {
                    traj[d * W..d * W + dayobs::N].copy_from_slice(v);
                }
            }
            for &(d, a, lp, v, bits) in agents[learner].policy.as_ref().map(|p| p.trace.as_slice()).unwrap_or(&[]) {
                if d < n_slots {
                    traj[d * W + dayobs::N] = a as f32;
                    traj[d * W + dayobs::N + 1] = lp;
                    traj[d * W + dayobs::N + 2] = v;
                    traj[d * W + dayobs::N + 3] = f32::from_bits(bits as u32);
                    traj[d * W + dayobs::N + 4] = f32::from_bits((bits >> 32) as u32);
                }
            }
            res.lock().unwrap().push((seed, format!("{seed}\t{learner}\t{kind}\t{desc}\t{bl}\t{bo}\t{score}"), traj, orows));
        }));
    }
    for h in hs {
        h.join().expect("worker");
    }
    let mut rs = std::mem::take(&mut *results.lock().unwrap());
    rs.sort_by_key(|x| x.0);
    let tr = format!("{out}.traj");
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{tr}.part")).unwrap());
    for (_, _, t, _) in &rs {
        for x in t {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    w.flush().unwrap();
    drop(w);
    std::fs::rename(format!("{tr}.part"), &tr).unwrap();
    let tsv = format!("{out}.tsv");
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{tsv}.part")).unwrap());
    for (_, l, _, _) in &rs {
        writeln!(w, "{l}").unwrap();
    }
    w.flush().unwrap();
    drop(w);
    std::fs::rename(format!("{tsv}.part"), &tsv).unwrap();
    if !oracle_days.is_empty() {
        // PREFIX.oracle: seed<TAB>day<TAB>k:margin,k:margin,... (first cell = the policy's own choice)
        let of = format!("{out}.oracle");
        let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{of}.part")).unwrap());
        for (_, _, _, o) in &rs {
            for line in o {
                writeln!(w, "{line}").unwrap();
            }
        }
        w.flush().unwrap();
        drop(w);
        std::fs::rename(format!("{of}.part"), &of).unwrap();
    }
    let wins = rs.iter().filter(|x| x.1.ends_with("\t1")).count();
    eprintln!("[ppo-rollout] {} games in {:.1}s ({:.2} g/s); learner wins {wins}", rs.len(), t0.elapsed().as_secs_f64(), rs.len() as f64 / t0.elapsed().as_secs_f64());
}

/// `--shield FILE`: the opponent-group controller with only the RL shield set (no group profiles, no
/// jitter), so the policy's choice is masked per group and pinned after an exact day-23 mirror.
fn load_shield(path: &str) -> Result<agent::layers::group::GroupCtl, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let j = kagg_engine::json::parse(&txt).map_err(|e| format!("{path}: {e}"))?;
    let (sh, tol) = agent::layers::group::Shield::from_json(&j)?;
    Ok(agent::layers::group::GroupCtl::shield_only(sh, tol))
}
