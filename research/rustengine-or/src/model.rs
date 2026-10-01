//! B1.2/B1.3 MVP: single-day sell allocation. Given supplies already in hand and
//! a shed/sell budget of `shedCap = 100` units (spec §5.2), choose per-product
//! sell quantities to maximise total staircase-PWL revenue. This is a pure LP
//! (concave PWL ⇒ secant cuts ⇒ no binaries, spec §3.3) and demonstrates the
//! scarcity edge: the optimiser CAPS volume on fragile products at their knee
//! instead of dumping into the $1 floor.

use crate::market::Params;
use crate::pwl::Pwl;
use good_lp::{constraint, variable, variables, Expression, Solution, SolverModel, Variable};

// Backend selection: HiGHS if built with --features highs, else pure-Rust microlp.
#[cfg(feature = "highs")]
use good_lp::highs as backend;
#[cfg(feature = "highs")]
pub const BACKEND_NAME: &str = "HiGHS (good_lp/highs, C++ build)";

#[cfg(all(feature = "microlp", not(feature = "highs")))]
use good_lp::microlp as backend;
#[cfg(all(feature = "microlp", not(feature = "highs")))]
pub const BACKEND_NAME: &str = "microlp (good_lp pure-Rust simplex)";

pub struct Item {
    pub p: Params,
    pub v0: i64,     // expected market inventory at the window open (spec §7.2)
    pub supply: i64, // units available to sell this day
}

pub struct SolveResult {
    pub sells: Vec<(&'static str, i64)>,
    pub revenues: Vec<(&'static str, i64)>,
    pub total_revenue: i64,
    pub shed_used: i64,
}

/// Solve the single-day allocation with a total sell/shed budget.
pub fn solve_day(items: &[Item], shed_cap: i64) -> Result<SolveResult, String> {
    let mut vars = variables!();
    let sell: Vec<Variable> = items
        .iter()
        .map(|it| vars.add(variable().min(0.0).max(it.supply as f64)))
        .collect();
    let rev: Vec<Variable> = items.iter().map(|_| vars.add(variable().min(0.0))).collect();

    let objective: Expression = rev.iter().map(|&v| Expression::from(v)).sum();
    let mut model = vars.maximise(objective).using(backend);

    // Concave-PWL secant cuts: rev_i <= R(k) + delta_k*(sell_i - k)
    let pwls: Vec<Pwl> = items.iter().map(|it| Pwl::build(&it.p, it.v0)).collect();
    for (i, pw) in pwls.iter().enumerate() {
        for c in &pw.cuts {
            let rhs: Expression = (c.r_k as f64) + (c.delta_k as f64) * (sell[i] - c.k as f64);
            model = model.with(constraint!(rev[i] <= rhs));
        }
    }

    // Shed / sell budget couples the products (spec §5.2): scarce slots force a
    // choice, and the optimiser spends them on units that still pay > $1.
    let total_sell: Expression = sell.iter().map(|&v| Expression::from(v)).sum();
    model = model.with(constraint!(total_sell <= shed_cap as f64));

    let sol = model.solve().map_err(|e| format!("solver error: {e:?}"))?;

    let mut sells = Vec::new();
    let mut revenues = Vec::new();
    let mut total_revenue = 0i64;
    let mut shed_used = 0i64;
    for (i, it) in items.iter().enumerate() {
        let q = sol.value(sell[i]).round() as i64;
        // Re-price revenue with the EXACT engine staircase at the chosen integer q
        // (removes any LP linearisation slack; equals the PWL value at integers).
        let r = crate::pwl::exact_rev(&it.p, q, it.v0);
        sells.push((it.p.name, q));
        revenues.push((it.p.name, r));
        total_revenue += r;
        shed_used += q;
    }
    Ok(SolveResult { sells, revenues, total_revenue, shed_used })
}

/// Emit the SELL rows of a `.tape` for the chosen day (spec §9 output contract).
/// Market column is an ordered list of `[op, item, n]`, integer args only,
/// <= 10 orders/turn. B2 (routing) consumes these by packing them across the
/// day's 24 turns and prepending the farmer/hands op columns.
pub fn emit_tape_sells(res: &SolveResult) -> Vec<String> {
    // One SELL line per product with q>0, priority-ordered by revenue (highest
    // first) so that if a turn ever exceeds 10 orders the low-value lines drop.
    let mut rows: Vec<(&str, i64, i64)> = res
        .sells
        .iter()
        .zip(res.revenues.iter())
        .filter(|((_, q), _)| *q > 0)
        .map(|((name, q), (_, r))| (*name, *q, *r))
        .collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    rows.into_iter()
        .map(|(name, q, _)| format!("[\"SELL\", \"{name}\", {q}]"))
        .collect()
}
