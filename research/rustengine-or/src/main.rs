//! Slot-1 OR tape-generator solver — MVP driver.
//! Tasks: B0.2 (solver links + toy solve), B1.1 (staircase PWL), B1.2/B1.3
//! (single-day sell allocation), B2-stub (emit .tape SELL rows).
//! Standalone crate — see docs/history/or-economic-model-spec-2026-09-18.md.

mod market;
mod model;
mod pwl;
mod season;

use market::*;
use model::{emit_tape_sells, solve_day, Item, BACKEND_NAME};

fn main() {
    println!("=== Kaggriculture OR tape-generator MVP ===");
    println!("solver backend : {BACKEND_NAME}\n");

    if std::env::args().any(|a| a == "--season") {
        // B1.4 + B3 multi-day model over the engine-derived absorption table and
        // the realized market scenario.
        season::run_season();
        return;
    }

    // --- B0.2 sanity: a trivial LP through the same backend, to confirm the
    // crate links and the solver actually runs end-to-end on this machine. ---
    toy_lp_smoke();

    // --- B1.1 context: each product's $1-floor "knee" (staircase width). The
    // optimiser will refuse to push a fragile product much past this. ---
    println!("--- B1.1: staircase knees (units above I0 until the $1 floor) ---");
    for p in [STRAWBERRY, WOOL, MILK, MELON, CARROT, TOMATO, WHEAT] {
        let pw = pwl::Pwl::build(&p, MARKET_I0);
        println!("  {:11} knee @ +{:6}  (rev of full 100u = ${})", p.name, pw.width, pwl::exact_rev(&p, 100, MARKET_I0));
    }
    println!();

    // --- B1.2/B1.3: single-day sell allocation. Supplies deliberately > shed
    // budget so the optimiser must choose which units are worth a scarce slot.
    // WHEAT is an abundant $1-decay-resistant staple filler (spec §3.5). ---
    let items = vec![
        Item { p: STRAWBERRY, v0: MARKET_I0, supply: 100 },
        Item { p: WOOL, v0: MARKET_I0, supply: 100 },
        Item { p: MILK, v0: MARKET_I0, supply: 100 },
        Item { p: MELON, v0: MARKET_I0, supply: 100 },
        Item { p: WHEAT, v0: MARKET_I0, supply: 1000 },
    ];

    for shed_cap in [100i64, 300] {
        println!("--- B1.2/B1.3: single-day sell allocation (shed/sell budget = {shed_cap}) ---");
        println!("supply: 100 ea STRAWBERRY/WOOL/MILK/MELON + 1000 WHEAT filler, all seeded at I0");
        match solve_day(&items, shed_cap) {
            Ok(res) => {
                for ((name, q), (_, r)) in res.sells.iter().zip(res.revenues.iter()) {
                    let per = if *q > 0 { *r as f64 / *q as f64 } else { 0.0 };
                    println!("  {name:11} sell {q:4}  rev ${r:8}  (${per:6.1}/u)");
                }
                println!("  TOTAL ${}  |  shed slots used {}/{}", res.total_revenue, res.shed_used, shed_cap);

                println!("  .tape SELL rows (market col, ints only, priority-ordered, <=10/turn):");
                for row in emit_tape_sells(&res) {
                    println!("    {row}");
                }
                println!();
            }
            Err(e) => {
                eprintln!("SOLVE FAILED: {e}");
                std::process::exit(1);
            }
        }
    }
    println!("Scarcity read: premium products are held at/near their knees; extra budget");
    println!("spills into WHEAT (~$22/u forever) rather than dumping fragile units at $1 —");
    println!("higher $/u, lower volume = the F3.1 winner edge (spec §0.1).");
    println!("\nB2 routing would pack these SELL rows across the day's 24 turns and prepend");
    println!("the farmer/hands op columns; an oversized SELL n is harmless (per-unit partial fill, §9).");
}

/// Minimal LP: maximise 3a + 2b s.t. a+b<=4, a<=2, b<=3. Optimum a=2,b=2 -> 10.
fn toy_lp_smoke() {
    use good_lp::{constraint, variable, variables, Solution, SolverModel};
    #[cfg(feature = "highs")]
    use good_lp::highs as backend;
    #[cfg(all(feature = "microlp", not(feature = "highs")))]
    use good_lp::microlp as backend;

    let mut vars = variables!();
    let a = vars.add(variable().min(0.0).max(2.0));
    let b = vars.add(variable().min(0.0).max(3.0));
    let sol = vars
        .maximise(3.0 * a + 2.0 * b)
        .using(backend)
        .with(constraint!(a + b <= 4.0))
        .solve()
        .expect("toy LP solve");
    let (av, bv, obj) = (sol.value(a), sol.value(b), 3.0 * sol.value(a) + 2.0 * sol.value(b));
    println!("--- B0.2 toy LP smoke: max 3a+2b s.t. a+b<=4, a<=2, b<=3 ---");
    println!("  a={av:.3} b={bv:.3} objective={obj:.3}  (expected a=2 b=2 obj=10)\n");
    assert!((obj - 10.0).abs() < 1e-6, "toy LP wrong: {obj}");
}

// ===================== unit tests (B1.1) =====================
#[cfg(test)]
mod tests {
    use super::market::*;
    use super::pwl::{exact_rev, Pwl};

    // Ground truth computed against the real engine market_price (Python), spec §3.5.
    #[test]
    fn staircase_widths_match_spec() {
        let cases = [
            (STRAWBERRY, 62i64),
            (WOOL, 59),
            (MELON, 158),
            (MILK, 76),
            (CARROT, 842),
            (TOMATO, 529),
        ];
        for (p, w) in cases {
            let pw = Pwl::build(&p, MARKET_I0);
            assert_eq!(pw.width, w, "{} width", p.name);
        }
    }

    #[test]
    fn marginal_prices_match_spec() {
        // price(I0), price(I0+10), +20, +30, +50  (spec §3.5 table)
        let expect: &[(Params, [i64; 5])] = &[
            (STRAWBERRY, [120, 101, 82, 62, 24]),
            (WOOL, [200, 194, 177, 148, 55]),
            (MELON, [250, 249, 246, 241, 225]),
            (MILK, [160, 139, 118, 97, 55]),
            (CARROT, [35, 31, 30, 29, 27]),
            (TOMATO, [60, 52, 49, 46, 42]),
        ];
        for (p, exp) in expect {
            for (j, off) in [0, 10, 20, 30, 50].iter().enumerate() {
                assert_eq!(market_price(p, MARKET_I0 + off), exp[j], "{} +{off}", p.name);
            }
        }
    }

    #[test]
    fn pwl_equals_exact_staircase() {
        // The PWL upper envelope must equal the exact Σ price(v0+m) at integers.
        for p in [STRAWBERRY, WOOL, MILK, MELON, CARROT, TOMATO] {
            let pw = Pwl::build(&p, MARKET_I0);
            for q in [0, 1, 5, 10, 30, 50, 62, 80, 100, 200] {
                assert_eq!(
                    pw.eval(q),
                    exact_rev(&p, q, MARKET_I0),
                    "{} PWL vs exact at q={q}",
                    p.name
                );
            }
        }
    }

    #[test]
    fn strawberry_revenue_ground_truth() {
        // Exact values from the Python engine (spec-verified).
        assert_eq!(exact_rev(&STRAWBERRY, 1, MARKET_I0), 120);
        assert_eq!(exact_rev(&STRAWBERRY, 10, MARKET_I0), 1113);
        assert_eq!(exact_rev(&STRAWBERRY, 30, MARKET_I0), 2764);
        assert_eq!(exact_rev(&STRAWBERRY, 62, MARKET_I0), 3809);
        assert_eq!(exact_rev(&STRAWBERRY, 80, MARKET_I0), 3827); // +18 floor units @ $1
    }

    #[test]
    fn floor_units_are_pure_filler() {
        // Past the knee every unit adds exactly $1 and never advances inventory.
        let base = exact_rev(&STRAWBERRY, 62, MARKET_I0);
        assert_eq!(exact_rev(&STRAWBERRY, 62 + 25, MARKET_I0), base + 25);
    }
}
