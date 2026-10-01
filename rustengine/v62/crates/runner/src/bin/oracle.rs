//! Headroom test for a SELL search: could one different selling decision have flipped the game?
//!
//! For each seed the baseline game (A vs B) is played; then, for every decision step t in --steps and every
//! candidate, the game is replayed from the start with A's action at step t rewritten (both agents are
//! deterministic, so history up to t is identical) and the result compared:
//!   hold  = drop A's SELL orders at t (their slots kept empty)
//!   dump  = sell A's whole shed at t (every product A holds, except WHEAT and FERTILIZER), first in the queue
//! Perfect information and one decision only: an UPPER bound on what a one-step sell search could flip.
//!
//!     oracle --a BASE --b BASE --profiles P --pa N --pb M --seeds S1,S2 [--from 288 --to 696 --every 24] [--threads 8]
//!
//! Output per seed: seed, base_a, base_b, n_flips, best_gain, and the (step, cand, gain) of the best override.
use agent::act::{Action, Cmd};
use kagg_engine::state::{State, PRODUCTS};
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let a_dir = get("--a").expect("--a");
    let b_dir = get("--b").unwrap_or_else(|| a_dir.clone());
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let from: i64 = get("--from").and_then(|s| s.parse().ok()).unwrap_or(288);
    let to: i64 = get("--to").and_then(|s| s.parse().ok()).unwrap_or(696);
    let every: i64 = get("--every").and_then(|s| s.parse().ok()).unwrap_or(24);
    // --rule R: instead of the one-decision search, apply a fixed rule to A all game and print both results
    //   hold0 = drop A's SELL orders at every hour-0 step from --from on
    let rule = get("--rule");
    let seeds: Vec<i64> = get("--seeds").expect("--seeds").split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let mut a = agent::base::Base::load(&a_dir).expect("a");
    let mut b = agent::base::Base::load(&b_dir).expect("b");
    if let Some(p) = get("--profiles") {
        a.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("pa");
        b.set_profiles(&p, get("--pb").and_then(|s| s.parse().ok())).expect("pb");
    }
    let q = Arc::new(Mutex::new(seeds));
    let out = Arc::new(Mutex::new(vec![]));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (q, out, a, b, rule) = (q.clone(), out.clone(), &a, &b, &rule);
            s.spawn(move || loop {
                let Some(seed) = q.lock().unwrap().pop() else { break };
                let base = runner::play(seed, &mut [a.fresh(), b.fresh()]);
                let margin = base.banks[0] - base.banks[1];
                if let Some(rule) = rule.as_deref() {
                    let hook = |step: i64, seat: usize, _st: &State, act: &mut Action| {
                        if seat == 0 && rule == "hold0" && step >= from && step % 24 == 0 {
                            for m in act.market.iter_mut() {
                                if !m.is_empty() && m.op() == "SELL" {
                                    *m = Cmd(vec![]);
                                }
                            }
                        }
                    };
                    let r = runner::play_with(seed, &mut [a.fresh(), b.fresh()], None, &hook);
                    out.lock().unwrap().push(format!("{seed}	{}	{}	{}	{}", base.banks[0], base.banks[1], r.banks[0], r.banks[1]));
                    continue;
                }
                let (mut flips, mut best) = (0usize, (f64::NEG_INFINITY, -1i64, ""));
                let mut t = from;
                while t <= to {
                    for cand in ["hold", "dump"] {
                        let hook = |step: i64, seat: usize, st: &State, act: &mut Action| {
                            if seat != 0 || step != t {
                                return;
                            }
                            if cand == "hold" {
                                for m in act.market.iter_mut() {
                                    if !m.is_empty() && m.op() == "SELL" {
                                        *m = Cmd(vec![]);
                                    }
                                }
                            } else {
                                let mut front: Vec<Cmd> = PRODUCTS
                                    .iter()
                                    .filter(|p| **p != "WHEAT" && **p != "FERTILIZER" && st.private[0].shed.get(p) > 0)
                                    .map(|p| Cmd::order("SELL", p, st.private[0].shed.get(p)))
                                    .collect();
                                front.extend(act.market.drain(..).filter(|m| !(m.len() > 1 && m.op() == "SELL" && front_has(m))));
                                front.truncate(10);
                                act.market = front;
                            }
                        };
                        let r = runner::play_with(seed, &mut [a.fresh(), b.fresh()], None, &hook);
                        let m = r.banks[0] - r.banks[1];
                        if (margin <= 0.0) != (m <= 0.0) && m > 0.0 {
                            flips += 1;
                        }
                        if m - margin > best.0 {
                            best = (m - margin, t, cand);
                        }
                    }
                    t += every;
                }
                out.lock().unwrap().push(format!("{seed}	{}	{}	{flips}	{:.0}	{}	{}", base.banks[0], base.banks[1], best.0, best.1, best.2));
            });
        }
    });
    let mut o = Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap();
    o.sort();
    for l in o {
        println!("{l}");
    }
}

/// every product sell is replaced by the dump (the dump already sells the whole shed of it)
fn front_has(m: &Cmd) -> bool {
    let p = m.s(1);
    p != "WHEAT" && p != "FERTILIZER"
}
