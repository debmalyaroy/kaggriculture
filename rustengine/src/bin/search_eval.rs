//! Offline evaluation harness for the Track-P searcher.
//!
//! Headless: no Kaggle, no transport, no Python agent in the loop. It plays
//! the searcher against a Rust opponent on the bit-exact engine, over a fixed
//! seed set, both seats, and reports own bank / opp bank / win rate split by
//! the LOW/HIGH price regime.
//!
//! The regime spec is `src/band_panel.py`'s, frozen in
//! `.local/band_panel/regime.json` (calibrated on 8,000 ladder traces):
//!   statistic = mean over {MILK, STRAWBERRY, EGG, WOOL} of
//!               median(price / base) over the window's turns
//!   d3-5   threshold 1.1065  (EARLY, actionable)
//!   d9-12  threshold 1.0630  (PRIMARY, AUC 0.832 / Cohen d 1.35)
//!
//! Usage:
//!   search_eval --seeds 4000:4032 --budget-ms 150
//!   search_eval --seeds 4000:4016 --budget-ms 10,50,150,300 --profile
//!   search_eval --seeds 4000:4032 --opp searcher       (self-play)
//!   search_eval --seeds 4000:4032 --control            (skeleton vs skeleton)

use kaggriculture_engine::engine;
use kaggriculture_engine::plan::{DayKnobs, SkeletonPolicy, Style, TPD};
use kaggriculture_engine::search::{SearchCfg, Searcher, Stats};
use kaggriculture_engine::state::State;

const STAT_PRODUCTS: [(&str, f64); 4] = [
    ("MILK", 160.0), ("STRAWBERRY", 120.0), ("EGG", 50.0), ("WOOL", 200.0),
];
const THR_EARLY: f64 = 1.1065; // d3-5
const THR_PRIMARY: f64 = 1.0630; // d9-12

#[derive(Clone, Debug)]
struct Cell {
    seed: i64,
    seat: usize,
    own: f64,
    opp: f64,
    stat_early: f64,
    stat_primary: f64,
    stats: Stats,
}

impl Cell {
    fn win(&self) -> f64 {
        if self.own > self.opp {
            1.0
        } else if self.own < self.opp {
            0.0
        } else {
            0.5
        }
    }
    fn regime(&self) -> &'static str {
        if self.stat_primary < THR_PRIMARY { "LOW" } else { "HIGH" }
    }
    fn regime_early(&self) -> &'static str {
        if self.stat_early < THR_EARLY { "LOW" } else { "HIGH" }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Side {
    Search,
    Skeleton(Style, f64),
}

fn median(v: &mut Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// One full episode. `a` plays seat 0, `b` plays seat 1.
fn play(seed: i64, a: Side, b: Side, budget_ms: u64, cfg: &SearchCfg,
        report_seat: usize) -> Cell
{
    play_traced(seed, a, b, budget_ms, budget_ms, cfg, report_seat, false)
}

/// `budget_a` / `budget_b` are per-seat so the search-quality curve can be run
/// as searcher(B) vs searcher(reference) -- against the skeleton the score
/// saturates at 30-0-2 and a saturated score cannot show a curve.
#[allow(clippy::too_many_arguments)]
fn play_traced(seed: i64, a: Side, b: Side, budget_a: u64, budget_b: u64,
               cfg: &SearchCfg, report_seat: usize, trace: bool) -> Cell
{
    let mut st = State::new(seed);
    let mut sa_search = Searcher::with_cfg(0, cfg.clone());
    let mut sb_search = Searcher::with_cfg(1, cfg.clone());
    let mut sa_skel = match a {
        Side::Skeleton(sty, g) => Some(SkeletonPolicy::styled(0, sty, g)),
        _ => None,
    };
    let mut sb_skel = match b {
        Side::Skeleton(sty, g) => Some(SkeletonPolicy::styled(1, sty, g)),
        _ => None,
    };

    // price rows for the regime statistic
    let mut rows: Vec<(i64, [f64; 4])> = Vec::with_capacity(24 * 10);

    loop {
        let day = st.step / TPD;
        if (3..=12).contains(&day) {
            let mut r = [0f64; 4];
            for (i, (name, base)) in STAT_PRODUCTS.iter().enumerate() {
                r[i] = st.market.prices.get(name) as f64 / base;
            }
            rows.push((day, r));
        }
        let a0 = match a {
            Side::Search => sa_search.decide(&st, 0, budget_a),
            Side::Skeleton(..) => sa_skel.as_mut().unwrap().act(&st),
        };
        let a1 = match b {
            Side::Search => sb_search.decide(&st, 1, budget_b),
            Side::Skeleton(..) => sb_skel.as_mut().unwrap().act(&st),
        };
        let running = engine::step(&mut st, &[a0, a1]);
        if !running {
            break;
        }
        if trace && st.step % TPD == 0 {
            let s = if report_seat == 0 { &sa_search } else { &sb_search };
            if let Some(k) = s.committed_knobs() {
                println!("trace d{:<2} bank {:>8.0}/{:<8.0} hire {} land {} \
                          buy {:?} plant {:?} caps {:?} floors {:?} v {:?}",
                         st.step / TPD - 1, st.farms[report_seat].money,
                         st.farms[1 - report_seat].money, k.hire, k.buy_land,
                         k.buy, k.plant, k.sell_cap, k.sell_floor,
                         s.pending_value().map(|v| (v * 1000.0).round() / 1000.0));
            }
        }
    }

    let stat = |lo: i64, hi: i64| -> f64 {
        let mut acc = 0.0;
        for i in 0..4 {
            let mut v: Vec<f64> = rows.iter()
                .filter(|(d, _)| *d >= lo && *d <= hi)
                .map(|(_, r)| r[i])
                .collect();
            acc += median(&mut v);
        }
        acc / 4.0
    };

    let (own, opp) = if report_seat == 0 {
        (st.farms[0].money, st.farms[1].money)
    } else {
        (st.farms[1].money, st.farms[0].money)
    };
    let stats = if report_seat == 0 {
        sa_search.stats.clone()
    } else {
        sb_search.stats.clone()
    };
    Cell {
        seed,
        seat: report_seat,
        own,
        opp,
        stat_early: stat(3, 5),
        stat_primary: stat(9, 12),
        stats,
    }
}

// ------------------------------------------------------------------ report --

/// Per-cell dump: `seed \t seat \t own \t opp`, in cell order.
///
/// The in-process `paired` below can only compare two BUDGET rows of one run.
/// Anything else worth a sign test -- a value-function change, an opponent
/// model, a flag -- lives in two separate runs, and pairing those needs the
/// cells, not the summary. `src/win_metric.py::paired_test` consumes this.
fn dump_cells(path: &str, cells: &[Cell]) {
    use std::io::Write;
    let mut s = String::from("seed\tseat\town\topp\n");
    for c in cells {
        s.push_str(&format!("{}\t{}\t{}\t{}\n", c.seed, c.seat, c.own, c.opp));
    }
    match std::fs::File::create(path) {
        Ok(mut f) => {
            let _ = f.write_all(s.as_bytes());
            println!("cells -> {path}");
        }
        Err(e) => eprintln!("cannot write {path}: {e}"),
    }
}

fn summarise(label: &str, cells: &[Cell], profile: bool) {
    if cells.is_empty() {
        println!("{label}: no cells");
        return;
    }
    let n = cells.len() as f64;
    let score: f64 = cells.iter().map(|c| c.win()).sum::<f64>() / n;
    let own: f64 = cells.iter().map(|c| c.own).sum::<f64>() / n;
    let opp: f64 = cells.iter().map(|c| c.opp).sum::<f64>() / n;
    let mut owns: Vec<f64> = cells.iter().map(|c| c.own).collect();
    let mut opps: Vec<f64> = cells.iter().map(|c| c.opp).collect();
    let mown = median(&mut owns);
    let mopp = median(&mut opps);
    let wins = cells.iter().filter(|c| c.own > c.opp).count();
    let losses = cells.iter().filter(|c| c.own < c.opp).count();
    let draws = cells.len() - wins - losses;

    println!("\n=== {label} ===");
    println!("cells {:<4} score {:.3}  W-D-L {}-{}-{}", cells.len(), score,
             wins, draws, losses);
    println!("bank  own mean {:>9.0} median {:>9.0} | opp mean {:>9.0} \
              median {:>9.0} | edge {:>+9.0}",
             own, mown, opp, mopp, own - opp);

    for (tag, f) in [("d9-12", 0usize), ("d3-5", 1usize)] {
        let mut lo: Vec<&Cell> = Vec::new();
        let mut hi: Vec<&Cell> = Vec::new();
        for c in cells {
            let r = if f == 0 { c.regime() } else { c.regime_early() };
            if r == "LOW" { lo.push(c) } else { hi.push(c) }
        }
        let s = |v: &Vec<&Cell>| -> String {
            if v.len() < 6 {
                format!("{:>2} cells (too few to report)", v.len())
            } else {
                let sc: f64 = v.iter().map(|c| c.win()).sum::<f64>()
                    / v.len() as f64;
                let mut b: Vec<f64> = v.iter().map(|c| c.own).collect();
                format!("{:>2} cells  score {:.3}  median bank {:>8.0}",
                        v.len(), sc, median(&mut b))
            }
        };
        println!("regime {tag:<6} LOW  {}", s(&lo));
        println!("regime {tag:<6} HIGH {}", s(&hi));
    }

    // WITHIN-RUN split, exactly as band_panel.py's "regime REL" row.
    //
    // Needed because the absolute d9-12 threshold (1.0630) is calibrated on
    // LADDER traces and this substrate does not reach it: a skeleton mirror
    // banks ~60k and never gluts the four tracked products the way a real
    // 88k-median ladder game does -- and the skeleton buys no geese at all,
    // so EGG sits permanently on its (1.32.7 hinge) SCARCITY side and drags
    // the four-product mean above the cut in every cell. An absolute regime
    // readout with 0 LOW cells is not a LOW result, it is no result; the
    // relative split at least ranks this run's own worlds.
    {
        let mut xs: Vec<f64> = cells.iter().map(|c| c.stat_primary).collect();
        let thr = median(&mut xs);
        let lo: Vec<&Cell> =
            cells.iter().filter(|c| c.stat_primary < thr).collect();
        let hi: Vec<&Cell> =
            cells.iter().filter(|c| c.stat_primary >= thr).collect();
        let s = |v: &Vec<&Cell>| -> String {
            if v.len() < 6 {
                format!("{:>2} cells (too few)", v.len())
            } else {
                let sc: f64 =
                    v.iter().map(|c| c.win()).sum::<f64>() / v.len() as f64;
                let mut b: Vec<f64> = v.iter().map(|c| c.own).collect();
                format!("{:>2} cells  score {:.3}  median bank {:>8.0}",
                        v.len(), sc, median(&mut b))
            }
        };
        println!("regime REL    thr {thr:.4} (within-run median of the d9-12 \
                  statistic)");
        println!("regime REL    LOW  {}", s(&lo));
        println!("regime REL    HIGH {}", s(&hi));
    }

    if profile {
        let mut r = 0u64;
        let mut ss = 0u64;
        let mut acc = 0u64;
        let mut dec = 0u64;
        let (mut nr, mut nro, mut ne, mut nt) = (0u64, 0u64, 0u64, 0u64);
        for c in cells {
            r += c.stats.rollouts;
            ss += c.stats.sim_steps;
            acc += c.stats.accepted;
            dec += c.stats.decides;
            nr += c.stats.ns_root;
            nro += c.stats.ns_rollout;
            ne += c.stats.ns_exec;
            nt += c.stats.ns_total;
        }
        if dec > 0 {
            let ms = |x: u64| x as f64 / 1e6;
            println!("profile: {:.1} rollouts/turn, {:.0} sim-steps/turn, \
                      {} accepted moves total", r as f64 / dec as f64,
                     ss as f64 / dec as f64, acc);
            println!("profile: per turn ms -- total {:.1}  rollout {:.1}  \
                      root {:.1}  execute {:.3}",
                     ms(nt) / dec as f64, ms(nro) / dec as f64,
                     ms(nr) / dec as f64, ms(ne) / dec as f64);
            if ss > 0 {
                println!("profile: {:.2} us per simulated step \
                          (engine + both policies)",
                         (nro + nr) as f64 / 1000.0 / ss as f64);
            }
        }
    }
}

/// Paired sign test over seeds: how many seeds did A win that B lost.
fn paired(label: &str, a: &[Cell], b: &[Cell]) {
    use std::collections::HashMap;
    let key = |c: &Cell| (c.seed, c.seat);
    let mb: HashMap<(i64, usize), &Cell> =
        b.iter().map(|c| (key(c), c)).collect();
    let (mut up, mut down, mut same) = (0, 0, 0);
    let mut d_sum = 0.0;
    let mut n = 0;
    for ca in a {
        let Some(cb) = mb.get(&key(ca)) else { continue };
        n += 1;
        d_sum += ca.own - cb.own;
        let (wa, wb) = (ca.win(), cb.win());
        if wa > wb {
            up += 1
        } else if wa < wb {
            down += 1
        } else {
            same += 1
        }
    }
    if n == 0 {
        return;
    }
    // Two-sided exact binomial on the discordant pairs (McNemar exact).
    let k = up.min(down);
    let m = up + down;
    let mut p = 0.0f64;
    if m > 0 {
        let mut c = 1.0f64;
        let mut tail = 0.0f64;
        for i in 0..=k {
            if i > 0 {
                c = c * (m - i + 1) as f64 / i as f64;
            }
            tail += c;
        }
        p = (2.0 * tail / (2f64).powi(m as i32)).min(1.0);
    }
    println!("\npaired {label}: n {n}  discordant {up}+/{down}-  ties {same}  \
              mean bank delta {:+.0}  McNemar exact p {:.4}",
             d_sum / n as f64, p);
}

// --------------------------------------------------------------------- cli --

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| -> Option<String> {
        args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
    };
    let has = |k: &str| args.iter().any(|a| a == k);

    let seeds_arg = get("--seeds").unwrap_or_else(|| "4000:4016".into());
    let (lo, hi) = {
        let mut it = seeds_arg.split(':');
        let a: i64 = it.next().unwrap().parse().expect("--seeds LO:HI");
        let b: i64 = it.next().expect("--seeds LO:HI").parse().unwrap();
        (a, b)
    };
    let budgets: Vec<u64> = get("--budget-ms")
        .unwrap_or_else(|| "150".into())
        .split(',')
        .map(|s| s.trim().parse().expect("--budget-ms N[,N...]"))
        .collect();
    let mut cfg = SearchCfg::default();
    if let Some(v) = get("--k") { cfg.k_days = v.parse().unwrap(); }
    if let Some(v) = get("--lambda") { cfg.lambda = v.parse().unwrap(); }
    if let Some(v) = get("--ensemble") { cfg.ensemble = v.parse().unwrap(); }
    if let Some(v) = get("--opp-aggression") {
        cfg.opp_aggression = v.parse().unwrap();
    }
    // `--rollouts N` = a reproducible budget (N candidate evaluations per
    // turn). `--budget-ms` = the realistic one, which is wall-clock and so is
    // NOT reproducible. Paired tests use the former, the headline curve the
    // latter, and the report says which is which.
    let fixed_rollouts = get("--rollouts").map(|v| {
        let n: usize = v.parse().unwrap();
        cfg.max_rollouts = n;
        n
    });
    let threads: usize = get("--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|n| n.get() / 3).unwrap_or(4).max(1)
        });
    // The REAL opponent's aggression, independent of the search's internal
    // model. Setting them apart is the transfer test: a searcher that only
    // beats the exact policy it models has learned nothing transferable.
    let real_agg: f64 = get("--real-agg")
        .map(|v| v.parse().unwrap())
        .unwrap_or(1.0);
    // A structurally DIFFERENT real opponent (field|geese|wheat|melon). The
    // search only ever models `field`, so anything else is a transfer test.
    let real_style = Style::parse(
        &get("--real-style").unwrap_or_else(|| "field".into()));
    // Reference budget for the opposing searcher in self-play, so the curve
    // is measured against a FIXED-strength opponent instead of itself.
    let ref_budget: u64 = get("--ref-budget-ms")
        .map(|v| v.parse().unwrap()).unwrap_or(10);
    let trace = has("--trace-knobs");
    let profile = has("--profile");
    let control = has("--control");
    let self_play = get("--opp").map(|o| o == "searcher").unwrap_or(false);
    let seats: Vec<usize> = match get("--seat").as_deref() {
        Some("0") => vec![0],
        Some("1") => vec![1],
        _ => vec![0, 1],
    };

    let seeds: Vec<i64> = (lo..hi).collect();
    println!("search_eval: seeds {lo}..{hi} ({} seeds) x seats {:?}  \
              threads {threads}  k {}  lambda {}  ensemble {}",
             seeds.len(), seats, cfg.k_days, cfg.lambda, cfg.ensemble);

    // ---- control: the opponent against itself, so the seed set's own bias
    // ---- is visible before any searcher number is believed.
    if control {
        let jobs: Vec<(i64, usize)> =
            seeds.iter().flat_map(|s| seats.iter().map(move |t| (*s, *t)))
                 .collect();
        let cells = run(jobs, threads, |seed, seat| {
            play(seed, Side::Skeleton(Style::Field, 1.0),
                 Side::Skeleton(Style::Field, 1.0), 0,
                 &SearchCfg::default(), seat)
        });
        summarise("CONTROL skeleton vs skeleton", &cells, false);
    }

    let mut prev: Option<(u64, Vec<Cell>)> = None;
    for &b in budgets.iter() {
        let jobs: Vec<(i64, usize)> =
            seeds.iter().flat_map(|s| seats.iter().map(move |t| (*s, *t)))
                 .collect();
        let c2 = cfg.clone();
        let cells = run(jobs, threads, move |seed, seat| {
            let (a, bb) = if self_play {
                (Side::Search, Side::Search)
            } else if seat == 0 {
                (Side::Search, Side::Skeleton(real_style, real_agg))
            } else {
                (Side::Skeleton(real_style, real_agg), Side::Search)
            };
            let (ba, bbud) = if self_play {
                // the REFERENCE seat always runs at `ref_budget`
                if seat == 0 { (b, ref_budget) } else { (ref_budget, b) }
            } else {
                (b, b)
            };
            play_traced(seed, a, bb, ba, bbud, &c2, seat, trace)
        });
        let label = format!("searcher @ {}{} vs {}",
                            match fixed_rollouts {
                                Some(n) => format!("{n} rollouts/turn"),
                                None => format!("{b} ms/turn"),
                            },
                            "",
                            if self_play {
                                format!("searcher@{ref_budget}ms")
                            } else {
                                format!("skeleton({:?} agg {real_agg})",
                                        real_style)
                            });
        summarise(&label, &cells, profile);
        if let Some(p) = get("--cells-tsv") {
            let path = if budgets.len() > 1 {
                format!("{p}.{b}")
            } else {
                p
            };
            dump_cells(&path, &cells);
        }
        if let Some((pb, pc)) = &prev {
            paired(&format!("{b} ms vs {pb} ms"), &cells, pc);
        }
        prev = Some((b, cells));
    }
}

fn run<F>(jobs: Vec<(i64, usize)>, threads: usize, f: F) -> Vec<Cell>
where
    F: Fn(i64, usize) -> Cell + Send + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Cell>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (seed, seat) = jobs[i];
                let c = f(seed, seat);
                out.lock().unwrap().push(c);
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|c| (c.seed, c.seat));
    v
}

#[allow(dead_code)]
fn _knobs_touch(st: &State) -> DayKnobs {
    DayKnobs::skeleton(0, st, 0)
}
