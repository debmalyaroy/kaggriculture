//! Offline measurement of play-time search (queue Q24): would picking each key day's profile by
//! simulating the rest of the game beat v63 against real players, when the rival's future is a
//! PREDICTION (as it must be at play time)?
//!
//!     search-eval --tapes data/tapes/band --pool data/tapes/train --out PREFIX [--profiles configs/profiles/rl3.json]
//!                 [--ref 35] [--days 6,12,14,16,18,20,22,24,26] [--cands 35,19,31,34,33,0,13,2] [--nn 2]
//!                 [--switch-min 0.05] [--threads 16] [--limit N]
//!
//! Per eval tape (a real player's recorded stream on its own seed, played through the chassis guards
//! as in the band gate), three learners in our seat, each scored on the real game:
//!   ref    profile --ref every day (v63);
//!   model  sequential search: on each key day d, for each candidate k, simulate the whole game with the
//!          learner's choices so far, k on day d and --ref afterwards, against a rival that plays its REAL
//!          stream up to step 24d and then the stream of its --nn nearest pool games (nearest by requested
//!          sales per product per day up to day d; the same game is never its own neighbour). Switch from
//!          --ref only if k's mean value beats --ref's by --switch-min (value = mean W/D/L score +
//!          mean margin / 100,000);
//!   truth  the same search against the rival's real future (the oracle's view; the upper bound).
//! Every game is deterministic given the world seed and both streams, so a simulation from step 0 with
//! the real prefix reaches exactly the state the learner sees at day d.
//! PREFIX.tsv: id band ref_margin model_margin truth_margin model_sched truth_sched (sched = day:profile of
//! the switches). stderr: W/L totals and the paired counts vs ref.
use agent::act::Action;
use std::io::Write;
use std::sync::{Arc, Mutex};

const PRODUCTS: [&str; 9] = dayobs::PRODUCTS;

/// Requested SELL units per product per day (cumulative), [30][9].
fn sells(stream: &[Action]) -> Vec<[f32; 9]> {
    let mut out = vec![[0f32; 9]; 30];
    let mut cum = [0f32; 9];
    for (t, a) in stream.iter().enumerate() {
        for o in &a.market {
            if o.len() > 2 && o.op() == "SELL" {
                if let Some(i) = PRODUCTS.iter().position(|p| *p == o.s(1)) {
                    cum[i] += o.n(2).max(0) as f32;
                }
            }
        }
        let d = t / 24;
        if d < 30 && (t % 24 == 23 || t + 1 == stream.len()) {
            out[d] = cum;
        }
    }
    for d in 1..30 {
        for i in 0..9 {
            if out[d][i] < out[d - 1][i] {
                out[d][i] = out[d - 1][i];
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let list = |k: &str, d: &str| -> Vec<usize> { get(k).unwrap_or_else(|| d.into()).split(',').filter_map(|x| x.trim().parse().ok()).collect() };
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let prof = get("--profiles").unwrap_or_else(|| "configs/profiles/rl3.json".into());
    let out = get("--out").expect("--out PREFIX");
    let r#ref: usize = get("--ref").and_then(|s| s.parse().ok()).unwrap_or(35);
    let days = Arc::new(list("--days", "6,12,14,16,18,20,22,24,26"));
    let cands = Arc::new(list("--cands", "35,19,31,34,33,0,13,2"));
    let nn: usize = get("--nn").and_then(|s| s.parse().ok()).unwrap_or(2);
    let switch_min: f64 = get("--switch-min").and_then(|s| s.parse().ok()).unwrap_or(0.05);
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(16);
    let mut evals = runner::load_tapes(&get("--tapes").expect("--tapes DIR"));
    if let Some(n) = get("--limit").and_then(|s| s.parse::<usize>().ok()) {
        evals.truncate(n);
    }
    let pool = runner::load_tapes(&get("--pool").expect("--pool DIR"));
    let pool_sig: Arc<Vec<Vec<[f32; 9]>>> = Arc::new(pool.iter().map(|t| sells(&t.stream)).collect());
    let pool = Arc::new(pool);
    let mut base = agent::base::Base::load(&base_dir).expect("base");
    base.set_profiles(&prof, None).expect("profiles");
    let n_prof = base.chain.profiles.len();
    assert!(r#ref < n_prof && cands.iter().all(|&k| k < n_prof), "profile id out of range");
    let router = Arc::new(runner::guard_router(&base_dir));
    eprintln!("[search-eval] {} eval tapes, pool {}, days {:?}, cands {:?}, nn {nn}", evals.len(), pool.len(), days, cands);
    let t0 = std::time::Instant::now();
    let queue = Arc::new(Mutex::new((0..evals.len()).rev().collect::<Vec<_>>()));
    let evals = Arc::new(evals);
    let results = Arc::new(Mutex::new(Vec::<(usize, String, [f64; 3])>::new()));
    let mut hs = vec![];
    for _ in 0..threads.max(1) {
        let (q, res, b, evals, pool, pool_sig, router, days, cands) =
            (queue.clone(), results.clone(), base.fresh(), evals.clone(), pool.clone(), pool_sig.clone(), router.clone(), days.clone(), cands.clone());
        hs.push(std::thread::spawn(move || loop {
            let Some(i) = q.lock().unwrap().pop() else { break };
            let t = &evals[i];
            let me = t.seat;
            // one game: our schedule vs a rival stream; returns our margin
            let game = |sched: &[usize], stream: Arc<Vec<Action>>| -> f64 {
                let mut ag = [b.fresh(), b.fresh()];
                ag[me].chain.schedule = Some(sched.to_vec());
                let rt = runner::Tape { id: String::new(), seed: t.seed, seat: me, band: String::new(), stream };
                ag[1 - me] = runner::guarded_tape_base(&rt, &router);
                let g = runner::play(t.seed, &mut ag);
                g.banks[me] - g.banks[1 - me]
            };
            let value = |ms: &[f64]| -> f64 {
                let n = ms.len().max(1) as f64;
                ms.iter().map(|&m| if m > 0.0 { 1.0 } else if m < 0.0 { 0.0 } else { 0.5 }).sum::<f64>() / n + ms.iter().sum::<f64>() / n / 100_000.0
            };
            let sig = sells(&t.stream);
            let real = t.stream.clone();
            // the rival's predicted streams at day d: real prefix (steps <= 24d) + nearest pool games' suffix
            let predicted = |d: usize| -> Vec<Arc<Vec<Action>>> {
                let mut dist: Vec<(f32, usize)> = pool
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.id != t.id && p.seed != t.seed)
                    .map(|(j, _)| {
                        let s = &pool_sig[j];
                        let mut x = 0f32;
                        for dd in 0..d.min(30) {
                            for k in 0..9 {
                                x += (s[dd][k] - sig[dd][k]).abs();
                            }
                        }
                        (x, j)
                    })
                    .collect();
                dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                dist.iter()
                    .take(nn)
                    .map(|&(_, j)| {
                        let cut = (24 * d + 1).min(real.len());
                        let mut s: Vec<Action> = real[..cut].to_vec();
                        s.extend(pool[j].stream.iter().skip(cut).cloned());
                        s.resize(real.len().max(s.len()), Action::pass());
                        Arc::new(s)
                    })
                    .collect()
            };
            let search = |use_truth: bool| -> (Vec<usize>, Vec<(usize, usize)>) {
                let mut sched = vec![r#ref; 30];
                let mut switches = vec![];
                for &d in days.iter() {
                    let rivals: Vec<Arc<Vec<Action>>> = if use_truth { vec![real.clone()] } else { predicted(d) };
                    let score_of = |k: usize| -> f64 {
                        let mut s = sched.clone();
                        s[d] = k;
                        let ms: Vec<f64> = rivals.iter().map(|r| game(&s, r.clone())).collect();
                        value(&ms)
                    };
                    let v_ref = score_of(r#ref);
                    let (mut best_k, mut best_v) = (r#ref, v_ref);
                    for &k in cands.iter().filter(|&&k| k != r#ref) {
                        let v = score_of(k);
                        if v > best_v {
                            best_v = v;
                            best_k = k;
                        }
                    }
                    if best_k != r#ref && best_v >= v_ref + switch_min {
                        sched[d] = best_k;
                        switches.push((d, best_k));
                    }
                }
                (sched, switches)
            };
            let m_ref = game(&vec![r#ref; 30], real.clone());
            let (s_model, sw_model) = search(false);
            let m_model = game(&s_model, real.clone());
            let (s_truth, sw_truth) = search(true);
            let m_truth = game(&s_truth, real.clone());
            let fmt = |sw: &[(usize, usize)]| if sw.is_empty() { "-".to_string() } else { sw.iter().map(|(d, k)| format!("{d}:{k}")).collect::<Vec<_>>().join(",") };
            let line = format!("{}\t{}\t{m_ref}\t{m_model}\t{m_truth}\t{}\t{}", t.id, t.band, fmt(&sw_model), fmt(&sw_truth));
            res.lock().unwrap().push((i, line, [m_ref, m_model, m_truth]));
        }));
    }
    for h in hs {
        h.join().expect("worker");
    }
    let mut rs = std::mem::take(&mut *results.lock().unwrap());
    rs.sort_by_key(|x| x.0);
    let tsv = format!("{out}.tsv");
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{tsv}.part")).unwrap());
    writeln!(w, "id\tband\tref\tmodel\ttruth\tmodel_sched\ttruth_sched").unwrap();
    for (_, l, _) in &rs {
        writeln!(w, "{l}").unwrap();
    }
    w.flush().unwrap();
    drop(w);
    std::fs::rename(format!("{tsv}.part"), &tsv).unwrap();
    let wl = |c: usize| {
        let w = rs.iter().filter(|x| x.2[c] > 0.0).count();
        let l = rs.iter().filter(|x| x.2[c] < 0.0).count();
        (w, l)
    };
    let sc = |m: f64| (m > 0.0) as i32 - (m < 0.0) as i32;
    let pair = |c: usize| {
        let b = rs.iter().filter(|x| sc(x.2[c]) > sc(x.2[0])).count();
        let wo = rs.iter().filter(|x| sc(x.2[c]) < sc(x.2[0])).count();
        (b, wo)
    };
    let (r, m, tr) = (wl(0), wl(1), wl(2));
    let (pm, pt) = (pair(1), pair(2));
    eprintln!("[search-eval] {} tapes in {:.0}s: ref {}W/{}L, model {}W/{}L (vs ref +{}/-{}), truth {}W/{}L (vs ref +{}/-{})",
              rs.len(), t0.elapsed().as_secs_f64(), r.0, r.1, m.0, m.1, pm.0, pm.1, tr.0, tr.1, pt.0, pt.1);
}
