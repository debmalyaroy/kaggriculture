//! Random-profile league: training data for the macro policy (task P5.1, queue Q06/Q07).
//!
//! Each game: the LEARNER seat (seat = seed & 1) plays a random per-day profile schedule
//! ("sticky": keep yesterday's profile with probability --stay, else uniform over --allow);
//! the opponent is drawn from a mix of v61.1, fixed escalated/front-running profiles, random
//! clone-lineage knob sets, and another random-schedule learner. The learner's dayobs vector at
//! every decision step is recorded with the profile it then played and the game's outcome.
//!
//!     league-rand --profiles configs/profiles/v2.json --out data/leagues/rand --batches 120 [--batch 500]
//!                 [--seed0 10000000] [--first K] [--threads 12] [--stay 0.5] [--allow 0,1,2,...]
//! `--first K` runs batches K .. K+batches-1 (so parallel workers on disjoint ranges write
//! globally unique batch files).
//!
//! Output per batch k (written atomically, a finished batch is never replayed -> resumable):
//!   batch-<k>.tsv  one line per game: seed seat opp_kind opp_desc sched(30 ids, comma) bank_learner bank_opp score
//!   batch-<k>.obs  games x 30 days x dayobs::N little-endian f32 (row order = tsv order;
//!                  days the game never reached are zero)
//! Seeds: batch k covers seed0 + k*batch .. + batch. Seeds below 1,000,000 are reserved for
//! validation panels and never used here.
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

fn sched(rng: &mut impl FnMut() -> u64, allow: &[usize], stay: f64) -> Vec<usize> {
    let mut s = Vec::with_capacity(30);
    let mut cur = allow[(rng() % allow.len() as u64) as usize];
    for _ in 0..30 {
        if (rng() % 10_000) as f64 / 10_000.0 >= stay {
            cur = allow[(rng() % allow.len() as u64) as usize];
        }
        s.push(cur);
    }
    s
}

/// Fixed opponent profiles: v61.1-escalated and front-running variants of our own lineage.
// rl3 ids: v2 escalated/AFR + 33 p19_v92, 34 p19_v92ext, 35 = v63 (the live agent to beat)
const FIXED_OPP: [usize; 14] = [2, 12, 13, 19, 21, 22, 23, 24, 29, 30, 31, 33, 34, 35];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let prof = get("--profiles").unwrap_or_else(|| "configs/profiles/rl3.json".into());
    let out = std::path::PathBuf::from(get("--out").expect("--out DIR"));
    let batches: u64 = get("--batches").and_then(|s| s.parse().ok()).unwrap_or(10);
    let first: u64 = get("--first").and_then(|s| s.parse().ok()).unwrap_or(0);
    let batch: u64 = get("--batch").and_then(|s| s.parse().ok()).unwrap_or(500);
    let seed0: u64 = get("--seed0").and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    assert!(seed0 >= 1_000_000, "seeds below 1,000,000 are reserved for validation");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let stay: f64 = get("--stay").and_then(|s| s.parse().ok()).unwrap_or(0.5);
    std::fs::create_dir_all(&out).expect("out dir");

    let mut base = agent::base::Base::load(&base_dir).expect("base");
    base.set_profiles(&prof, None).expect("profiles");
    let n_prof = base.chain.profiles.len();
    let allow: Vec<usize> = match get("--allow") {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).filter(|&x| x < n_prof).collect(),
        None => (0..n_prof).collect(),
    };
    assert!(!allow.is_empty(), "empty --allow");
    base.obs_track = Some(dayobs::Builder::new());
    let t_all = std::time::Instant::now();
    let mut done_games = 0u64;
    for k in first..first + batches {
        let tsv = out.join(format!("batch-{k:05}.tsv"));
        if tsv.exists() {
            continue;
        }
        let t0 = std::time::Instant::now();
        let seeds: Vec<u64> = (seed0 + k * batch..seed0 + (k + 1) * batch).collect();
        let queue = Arc::new(Mutex::new(seeds.iter().rev().copied().collect::<Vec<_>>()));
        let results = Arc::new(Mutex::new(Vec::<(u64, String, Vec<f32>)>::new()));
        let mut hs = vec![];
        for _ in 0..threads.max(1) {
            let (q, res, b, allow) = (queue.clone(), results.clone(), base.fresh(), allow.clone());
            hs.push(std::thread::spawn(move || loop {
                let Some(seed) = q.lock().unwrap().pop() else { break };
                let mut rng = splitmix(seed ^ 0xA5A5_5A5A_1234_5678);
                let learner = (seed & 1) as usize;
                let mut agents = [b.fresh(), b.fresh()];
                let s_l = sched(&mut rng, &allow, stay);
                agents[learner].chain.schedule = Some(s_l.clone());
                let opp = &mut agents[1 - learner];
                opp.obs_track = None;
                let r = rng() % 100;
                let (kind, desc) = if r < 25 {
                    opp.chain.schedule = Some(vec![0; 30]);
                    ("v611", "0".to_string())
                } else if r < 60 {
                    let p = FIXED_OPP[(rng() % FIXED_OPP.len() as u64) as usize].min(n_prof - 1);
                    opp.chain.schedule = Some(vec![p; 30]);
                    ("fixed", p.to_string())
                } else if r < 85 {
                    let kn = agent::layers::knobs::Knobs::random(&mut rng);
                    let d = format!("{:x}", seed);
                    opp.chain.profiles = vec![("v61.1".into(), Default::default()), ("rand".into(), kn)];
                    opp.chain.schedule = Some(vec![1; 30]);
                    ("randknobs", d)
                } else {
                    let s = sched(&mut rng, &allow, stay);
                    let d = s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                    opp.chain.schedule = Some(s);
                    ("randsched", d)
                };
                let g = runner::play(seed as i64, &mut agents);
                let (bl, bo) = (g.banks[learner], g.banks[1 - learner]);
                let score = if bl > bo { 1.0 } else if bl < bo { 0.0 } else { 0.5 };
                let mut obs = vec![0f32; 30 * dayobs::N];
                for (d, v) in &agents[learner].day_obs {
                    if *d < 30 {
                        obs[d * dayobs::N..(d + 1) * dayobs::N].copy_from_slice(v);
                    }
                }
                let line = format!("{seed}\t{learner}\t{kind}\t{desc}\t{}\t{bl}\t{bo}\t{score}",
                                   s_l.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
                res.lock().unwrap().push((seed, line, obs));
            }));
        }
        for h in hs {
            h.join().expect("worker");
        }
        let mut rs = std::mem::take(&mut *results.lock().unwrap());
        rs.sort_by_key(|x| x.0);
        // obs first, tsv last (atomically): the tsv's existence marks the batch complete
        let obs_path = out.join(format!("batch-{k:05}.obs"));
        let mut w = std::io::BufWriter::new(std::fs::File::create(obs_path.with_extension("obs.part")).unwrap());
        for (_, _, o) in &rs {
            for x in o {
                w.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        w.flush().unwrap();
        drop(w);
        std::fs::rename(obs_path.with_extension("obs.part"), &obs_path).unwrap();
        let tmp = tsv.with_extension("tsv.part");
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap());
        for (_, l, _) in &rs {
            writeln!(w, "{l}").unwrap();
        }
        w.flush().unwrap();
        drop(w);
        std::fs::rename(&tmp, &tsv).unwrap();
        done_games += rs.len() as u64;
        let wins = rs.iter().filter(|x| x.1.ends_with("\t1")).count();
        eprintln!("[league-rand] batch {k} {} games in {:.1}s ({:.2} g/s); learner wins {wins}; total this run {done_games} in {:.0}s",
                  rs.len(), t0.elapsed().as_secs_f64(), rs.len() as f64 / t0.elapsed().as_secs_f64(), t_all.elapsed().as_secs_f64());
    }
    eprintln!("[league-rand] finished: {done_games} new games; {batches} batches present under {}", out.display());
}
