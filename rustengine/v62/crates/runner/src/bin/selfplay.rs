//! Self-play on the Rust engine: base A (seat 0) vs base B (seat 1) over a seed list, in
//! parallel threads. One line per game: `seed<TAB>bank0<TAB>bank1<TAB>max_us0<TAB>max_us1`.
//!
//!     selfplay --a configs/bases/v61.1 --b configs/bases/v61.1 --seeds 268028965,1863162774 --threads 4
//!     selfplay --a ... --b ... --n 100 --seed0 1   (seeds seed0..seed0+n)
//!     selfplay --profiles configs/profiles/v1.json --pa 3 --pb 0 --n 200   (fixed profiles)
//!     selfplay --profiles ... --ca 13 --rand-b 7 --n 400   (B = random clone-lineage knobs per game)
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let a_dir = get("--a").unwrap_or_else(|| "configs/bases/v61.1".into());
    let b_dir = get("--b").unwrap_or_else(|| a_dir.clone());
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(4);
    let seeds: Vec<i64> = match get("--seeds") {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None => {
            let n: i64 = get("--n").and_then(|s| s.parse().ok()).unwrap_or(10);
            let s0: i64 = get("--seed0").and_then(|s| s.parse().ok()).unwrap_or(1);
            (s0..s0 + n).collect()
        }
    };
    let mut a = agent::base::Base::load(&a_dir).expect("load a");
    let mut b = agent::base::Base::load(&b_dir).expect("load b");
    if let Some(p) = get("--profiles") {
        let pa = get("--pa").and_then(|s| s.parse().ok());
        let pb = get("--pb").and_then(|s| s.parse().ok());
        a.set_profiles(&p, pa).expect("profiles a");
        b.set_profiles(&p, pb).expect("profiles b");
        a.chain.clone_profile = get("--ca").and_then(|s| s.parse().ok());
        b.chain.clone_profile = get("--cb").and_then(|s| s.parse().ok());
        a.chain.clone_strict = args.iter().any(|x| x == "--strict-a");
        a.chain.afr_profile = get("--fa").and_then(|s| s.parse().ok());
        b.chain.afr_profile = get("--fb").and_then(|s| s.parse().ok());
        let trig: usize = get("--ftrig").and_then(|s| s.parse().ok()).unwrap_or(2);
        a.chain.afr_trigger = trig;
        b.chain.afr_trigger = trig;
        b.chain.clone_strict = args.iter().any(|x| x == "--strict-b");
        // --group-a/-b P_DIFF,P_PARTIAL,P_COPY and --jitter-a/-b J_DIFF,J_PARTIAL,J_COPY: opponent-group controller
        let tri = |k: &str| -> Option<[i64; 3]> {
            let v: Vec<i64> = get(k)?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (v.len() == 3).then(|| [v[0], v[1], v[2]])
        };
        for (who, g, j, e, je) in [(0usize, "--group-a", "--jitter-a", "--endgame-a", "--jitter-end-a"), (1, "--group-b", "--jitter-b", "--endgame-b", "--jitter-end-b")] {
            if let Some(p) = tri(g) {
                let jit = tri(j).unwrap_or([0, 0, 0]);
                let eg = tri(e).map(|v| v.map(|x| (x >= 0).then_some(x as usize))).unwrap_or([None; 3]);
                let jend = tri(je).unwrap_or(jit);
                let mut ctl = agent::layers::group::GroupCtl::new([p[0] as usize, p[1] as usize, p[2] as usize], jit).with_endgame(eg, jend);
                ctl.mirror_tol = get(if who == 0 { "--mirror-tol-a" } else { "--mirror-tol-b" }).and_then(|s| s.parse().ok());
                if who == 0 { a.chain.group_ctl = Some(ctl) } else { b.chain.group_ctl = Some(ctl) }
            }
        }
    }
    let slow_us: Option<f32> = get("--slow-us").and_then(|s| s.parse().ok());
    let lat_dump: Option<String> = get("--lat-dump");
    let rand_b: Option<u64> = get("--rand-b").and_then(|s| s.parse().ok());
    let queue = Arc::new(Mutex::new(seeds.clone()));
    let t0 = std::time::Instant::now();
    let mut hs = vec![];
    for _ in 0..threads.max(1) {
        let (a, b, q, lat_dump) = (a.fresh(), b.fresh(), queue.clone(), lat_dump.clone());
        hs.push(std::thread::spawn(move || {
            let mut out = vec![];
            loop {
                let Some(seed) = q.lock().unwrap().pop() else { break };
                let mut agents = [a.fresh(), b.fresh()];
                let mut tag = String::new();
                if let Some(salt) = rand_b {
                    // seat B = a random clone-lineage member (knobs derived from seed^salt), fixed all game
                    let mut x = (seed as u64) ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let mut rng = move || {
                        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        let mut z = x;
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        z ^ (z >> 31)
                    };
                    let k = agent::layers::knobs::Knobs::random(&mut rng);
                    tag = format!("\t{k:?}");
                    let ch = &mut agents[1].chain;
                    ch.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), k)];
                    ch.schedule = Some(vec![1; 30]);
                    ch.clone_profile = None;
                }
                let r = runner::play_prof(seed, &mut agents, slow_us);
                if let Ok(path) = std::env::var("KAGG_FIRED_DUMP") {
                    // per game, per seat: steps on which each chain stage changed the action
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("fired dump");
                    for (s, ag) in agents.iter().enumerate() {
                        let v: Vec<String> = ag.chain.fired.iter().map(|x| x.to_string()).collect();
                        let _ = writeln!(f, "{}	{s}	{}	{}	{}", r.seed, r.banks[s], r.banks[1 - s], v.join(","));
                    }
                }
                for (st, seat, us, ph, top) in &r.slow {
                    let names: Vec<String> = top.iter().map(|(k, u)| format!("{}={:.0}", agent::base::CUTS[*k], u)).collect();
                    eprintln!("SLOW seed {} step {st} seat {seat} {us:.0}us pre {:.0} core {:.0} post {:.0} | {}", r.seed, ph[0], ph[1], ph[2], names.join(" "));
                }
                println!("{}\t{}\t{}\t{:.0}\t{:.0}\t{}{tag}", r.seed, r.banks[0], r.banks[1], r.max_us[0], r.max_us[1], r.world);
                if let Some(path) = lat_dump.as_ref() {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("lat dump");
                    for (st, u) in r.step_us.iter().enumerate() {
                        let _ = writeln!(f, "{}	{st}	{:.0}	{:.0}", r.seed, u[0], u[1]);
                    }
                }
                out.push(r);
            }
            out
        }));
    }
    let mut n = 0;
    let mut wins = [0, 0, 0];
    let mut prof = runner::Prof::default();
    for h in hs {
        for r in h.join().unwrap() {
            n += 1;
            prof.json_us += r.prof.json_us;
            prof.parse_us += r.prof.parse_us;
            prof.act_us += r.prof.act_us;
            prof.engine_us += r.prof.engine_us;
            for s in 0..2 {
                if prof.stage_us[s].len() < r.prof.stage_us[s].len() {
                    prof.stage_us[s].resize(r.prof.stage_us[s].len(), 0.0);
                }
                for (i, x) in r.prof.stage_us[s].iter().enumerate() {
                    prof.stage_us[s][i] += x;
                }
                for i in 0..3 {
                    prof.phase_us[s][i] += r.prof.phase_us[s][i];
                }
            }
            let k = if r.banks[0] > r.banks[1] { 0 } else if r.banks[1] > r.banks[0] { 1 } else { 2 };
            wins[k] += 1;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let tp: Vec<u64> = agent::terminal::TPROF.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
    if tp[0] > 0 {
        eprintln!("[tprof] plans {} sims {} sim_us {:.0} liq_miss {} liq_us {:.0} proposals_us {:.0}", tp[5], tp[0], tp[1] as f64 / 1e3, tp[2], tp[3] as f64 / 1e3, tp[4] as f64 / 1e3);
    }
    if args.iter().any(|x| x == "--stage-prof") && n > 0 {
        let g = n as f64 * 1e3;
        eprintln!("[prof] per game ms: obs json {:.1} | obs parse {:.1} | agents act {:.1} | engine {:.1}", prof.json_us / g, prof.parse_us / g, prof.act_us / g, prof.engine_us / g);
        for s in 0..2 {
            let p = prof.phase_us[s];
            let mut st: Vec<(usize, f64)> = prof.stage_us[s].iter().copied().enumerate().filter(|x| x.1 > 0.0).collect();
            st.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top: Vec<String> = st.iter().take(12).map(|(k, u)| format!("{}={:.1}", agent::base::CUTS[*k], u / g)).collect();
            eprintln!("[prof] seat {s} per game ms: pre {:.1} core {:.1} post {:.1} | stages {}", p[0] / g, p[1] / g, p[2] / g, top.join(" "));
        }
    }
    eprintln!("[selfplay] {n} games in {secs:.1}s ({:.2} games/s) A wins {} B wins {} draws {}", n as f64 / secs, wins[0], wins[1], wins[2]);
}
