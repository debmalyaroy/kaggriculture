//! The horizon value function and the search objective.
//!
//! Everything here is arithmetic in the engine's OWN price curve
//! (`market::price`), never a fitted model. That is deliberate: a learned value
//! head would make the searcher unauditable, and `train_gates.policy_allowed()`
//! has no positive evidence to license one. Every number this file produces can
//! be checked by hand against the interpreter.
//!
//! See docs/history/trackp-search-design-2026-09-03.md §1 and §4 for the argument.

use crate::market;
use crate::plan::{base_price, N_CROP, N_PROD, LAST_DAY, TPD};
use crate::rules;
use crate::state::{Cell, State, ANIMAL_NAMES, BOARD, CROP_NAMES, PRODUCTS};

/// Residual uncertainty in the FINAL margin at game start, in dollars.
///
/// Calibrated, not invented. Measured over real games: P(margin < $3,000) is
/// 0.352 in HIGH-price worlds and 0.435 in LOW. For a zero-mean margin,
/// P(|m| < 3000) = 2*Phi(3000/sigma) - 1, giving sigma ~= $6,570 (HIGH) and
/// ~= $5,210 (LOW). We take the round number between them.
pub const MARGIN_SIGMA0: f64 = 6000.0;
/// Floor on the scale, so the last day's objective is effectively the win
/// indicator rather than a bank maximiser.
pub const MARGIN_SIGMA_MIN: f64 = 400.0;

/// Marginal revenue of selling `n` units of product `p` into `inv`, priced by
/// the engine's per-unit lockstep rule. The walk-down is EXACT, which is why
/// the search never needs a hand-written "don't dump wool" rule.
pub fn liquidate(p: usize, n: i64, inv: i64) -> f64 {
    if n <= 0 {
        return 0.0;
    }
    let par = &market::PARAMS[p];
    let mut total = 0.0;
    // Prices only refresh once per order index in the engine, but over a
    // multi-day liquidation the walk-down is what actually happens; summing
    // per unit is the conservative (and correct-in-the-limit) reading.
    for k in 0..n {
        total += market::price(par, (inv + k) as f64) as f64;
    }
    total
}

/// Price of one more unit sold on top of `extra` already-planned units.
fn marginal(p: usize, inv: i64, extra: i64) -> f64 {
    market::price(&market::PARAMS[p], (inv + extra) as f64) as f64
}

fn market_inv(st: &State, p: usize) -> i64 {
    st.market.inventory.get(PRODUCTS[p])
}

/// Horizon asset value of one seat, in dollars.
///
/// money + liquidation(shed) + standing crops + standing animals + seeds that
/// can still be planted AND matured before day 29.
pub fn asset_value(st: &State, seat: usize) -> f64 {
    asset_value_off(st, seat, &[0; N_PROD]).0
}

/// `asset_value`, with each product's walk-down STARTING `pre[p]` units deep
/// into the curve, and also returning how many units of each product this seat
/// can realise.
///
/// `pre` is how the joint-liquidation model (§ `objective`) tells one seat that
/// it is not the only seller. `pre = 0` reproduces the sole-seller valuation
/// exactly, unit for unit, which is what makes the two modes a clean A/B.
fn asset_value_off(st: &State, seat: usize, pre: &[i64; N_PROD])
    -> (f64, [i64; N_PROD])
{
    let day = st.step / TPD;
    let days_left = (LAST_DAY - day).max(0);
    let farm = &st.farms[seat];
    let mut v = farm.money;

    // ---- shed ------------------------------------------------------------
    // `pre` is the depth into the curve that the OTHER seat's liquidation has
    // already used up by the time ours lands (0 in sole-seller mode).
    let mut sold_so_far = *pre;
    for (p, name) in PRODUCTS.iter().enumerate() {
        let n = st.private[seat].shed.get(name);
        if n > 0 {
            v += liquidate(p, n, market_inv(st, p) + sold_so_far[p]);
            sold_so_far[p] += n;
        }
    }
    for (i, name) in ANIMAL_NAMES.iter().enumerate() {
        // An unplaced animal is worth its future production only if there are
        // days to place and run it; otherwise it is a sunk cost.
        let a = &rules::ANIMALS[i];
        let n = st.private[seat].shed.get(name);
        if n > 0 && days_left > a.first_yield_day {
            let prods = 1 + (days_left - a.first_yield_day) / a.interval.max(1);
            let p = product_index(a.product);
            v += n as f64 * prods as f64
                * marginal(p, market_inv(st, p), sold_so_far[p])
                * 0.6;
        }
    }
    for inv in st.private[seat].inventories.iter() {
        for (p, name) in PRODUCTS.iter().enumerate() {
            let n = inv.get(name);
            if n > 0 {
                v += n as f64 * marginal(p, market_inv(st, p), sold_so_far[p]);
                sold_so_far[p] += n;
            }
        }
    }

    if days_left == 0 {
        // Nothing standing can still be realised; only banked money and stock
        // that could be sold in the remaining turns count.
        return (v, sub(sold_so_far, pre));
    }

    // ---- standing tiles ---------------------------------------------------
    let mut feed_animals = 0i64;
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            match &farm.tiles[y][x] {
                Cell::Plant { crop, planted_day, yield_units,
                              consecutive_unwatered, .. } => {
                    let ci = CROP_NAMES.iter().position(|n| n == crop)
                        .unwrap_or(0);
                    let cd = &rules::CROPS[ci];
                    let p = ci; // CROP_NAMES and PRODUCTS agree on 0..5
                    let age = day - planted_day;
                    // A plant that missed a watering is one day from a weed.
                    let alive = if *consecutive_unwatered >= 1 { 0.5 } else { 1.0 };
                    let mut units = 0i64;
                    if cd.ongoing {
                        let last = cd.first_yield_day
                            + cd.interval * (cd.max_yield - 1);
                        let mut a = age.max(cd.first_yield_day);
                        // count remaining production days inside the horizon
                        while a <= last && day + (a - age) <= LAST_DAY {
                            if (a - cd.first_yield_day) % cd.interval.max(1) == 0
                                && a > age
                            {
                                units += 1;
                            }
                            a += 1;
                        }
                        units += *yield_units;
                    } else if age >= cd.first_yield_day {
                        units = (*yield_units).max(0);
                    } else if day + (cd.first_yield_day - age) <= LAST_DAY {
                        // still growing and it CAN mature: the whole crop
                        units = cd.max_yield.min(*yield_units + cd.max_yield);
                    }
                    // A crop that cannot mature before the last day is worth
                    // nothing -- the day-28-planting leak, closed structurally.
                    if units > 0 {
                        v += alive * units as f64
                            * marginal(p, market_inv(st, p), sold_so_far[p]);
                        sold_so_far[p] += units;
                    }
                }
                Cell::Structure { animal: Some(a), .. } => {
                    let ai = ANIMAL_NAMES.iter().position(|n| *n == a.animal)
                        .unwrap_or(0);
                    let ad = &rules::ANIMALS[ai];
                    let p = product_index(ad.product);
                    let age = day - a.placed_day;
                    let mut units = a.yield_units;
                    let mut d = age + 1;
                    while day + (d - age) <= LAST_DAY {
                        let since = d - ad.first_yield_day;
                        if since >= 0 && since % ad.interval.max(1) == 0 {
                            units += 1;
                        }
                        d += 1;
                    }
                    units += a.pending_care_bonus;
                    // One missed feed from escaping: halve it. This is the
                    // "starving animals day 10" leak, priced.
                    let alive = if a.consecutive_unfed >= 1 { 0.5 } else { 1.0 };
                    v += alive * units as f64
                        * marginal(p, market_inv(st, p), sold_so_far[p]);
                    sold_so_far[p] += units;
                    feed_animals += 1;
                }
                _ => {}
            }
        }
    }
    // Feed is a real, unavoidable cost of holding the herd.
    v -= feed_animals as f64 * days_left as f64
        * base_price(0) * 0.9;

    // ---- seeds ------------------------------------------------------------
    let free_tiles = count_free(farm);
    let mut room = free_tiles;
    for ci in 0..N_CROP {
        let n = st.private[seat].seeds.get(CROP_NAMES[ci]);
        if n <= 0 {
            continue;
        }
        let cd = &rules::CROPS[ci];
        if day + cd.first_yield_day > LAST_DAY {
            continue; // dead cash: the "unplanted seed d15-d24" leak
        }
        let plantable = n.min(room);
        room -= plantable;
        let p = ci;
        let units = plantable as f64 * cd.max_yield as f64 * 0.5;
        v += units * marginal(p, market_inv(st, p), sold_so_far[p]);
        // Counted AFTER pricing, so this seat's own numbers are unchanged; it
        // matters only to the other seat's `pre`, which is what the joint
        // model needs it for.
        sold_so_far[p] += units as i64;
    }

    (v, sub(sold_so_far, pre))
}

fn sub(a: [i64; N_PROD], b: &[i64; N_PROD]) -> [i64; N_PROD] {
    let mut o = [0i64; N_PROD];
    for i in 0..N_PROD {
        o[i] = (a[i] - b[i]).max(0);
    }
    o
}

fn count_free(farm: &crate::state::Farm) -> i64 {
    let mut n = 0;
    for y in 0..BOARD as usize {
        for x in 0..BOARD as usize {
            if farm.tiles[y][x] == Cell::Empty {
                n += 1;
            }
        }
    }
    n
}

fn product_index(name: &str) -> usize {
    PRODUCTS.iter().position(|p| *p == name).unwrap_or(0)
}

/// Residual final-margin scale at `day`.
pub fn margin_scale(day: i64) -> f64 {
    let left = ((LAST_DAY + 1 - day).max(0) as f64) / 30.0;
    (MARGIN_SIGMA0 * left.sqrt()).max(MARGIN_SIGMA_MIN)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// THE OBJECTIVE: a smooth surrogate for P(own final > opp final).
///
/// Not own bank (rank 1 and rank 400 bank the same 88k median), not linear
/// own-minus-opp (which prices the 200,000th dollar like the one that flips a
/// $500 loss). The sigmoid saturates when we are far ahead, is steepest when
/// the game is close -- which is exactly the LOW-price, thin-margin regime
/// where the entire top-10 edge lives (61.5% vs 48.0%) -- and shrinks its
/// scale to $400 by the last day so the endgame plays to win, not to bank.
/// The tie-breaker below is not a second objective; it is a NUMERICAL guard.
/// Once the edge is ~40 sigmas the logistic is 1.0 in f64 and every candidate
/// ties, so the hill climb stops accepting anything and the searcher silently
/// degrades to the skeleton for the rest of the game (observed on seed 4000:
/// V pinned at 1.000 from day 7). The guard is 1e-9 per dollar, which is
/// ~1e-4 of the logistic's own gradient near an even game, so it can never
/// outvote the win-probability term where that term has any resolution.
const TIEBREAK_PER_DOLLAR: f64 = 1e-9;

/// JOINT LIQUIDATION: price both seats' remaining stock against ONE market.
///
/// The sole-seller reading (`pre = 0` for both seats) is the design doc's
/// suspect for the searcher's weakest measured cell -- 0.875 and a collapsed
/// +13.8k edge against a wheat monoculture that dumps everything (§11.4). If we
/// value a shed of 40 WOOL as though only our 40 units walk the curve down,
/// then in a world where the opponent is also holding 40, both seats' stock is
/// systematically overvalued and the search happily accumulates inventory that
/// the market will not actually absorb.
///
/// Under a FAIR INTERLEAVE (neither seat gets to sell first), seat s's `u_s`
/// units sit at merged positions `k * U / u_s`, and for a locally linear price
/// curve that is exactly equal to walking s's own units down starting
/// `u_other / 2` deep. So the model is one number per product per seat: half
/// the OTHER seat's realisable units, handed in as `pre`.
///
/// Two properties make this the honest version rather than a fudge:
///   * it conserves revenue -- the two seats' liquidations sum to what one
///     seller would get for the combined stack, instead of double-counting the
///     top of the curve;
///   * it is symmetric, so it cannot flatter our own seat.
///
/// It is deliberately NOT a "don't dump" rule. The search still discovers what
/// each product tolerates from the engine's own price curve; it just stops
/// being told that the curve is all ours.
fn joint_values(st: &State, me: usize) -> (f64, f64) {
    let zero = [0i64; N_PROD];
    let opp = 1 - me;
    let (_, u_me) = asset_value_off(st, me, &zero);
    let (_, u_op) = asset_value_off(st, opp, &zero);
    let mut pre_me = [0i64; N_PROD];
    let mut pre_op = [0i64; N_PROD];
    for p in 0..N_PROD {
        pre_me[p] = u_op[p] / 2;
        pre_op[p] = u_me[p] / 2;
    }
    (
        asset_value_off(st, me, &pre_me).0,
        asset_value_off(st, opp, &pre_op).0,
    )
}

/// Whether the joint model is on. Env-gated so the two readings are a paired
/// A/B on ONE binary rather than two builds that might differ elsewhere.
///
/// **Default OFF.** It is a hypothesis until a paired test says otherwise, and
/// an unmeasured change must not be able to ride into a gauntlet number.
/// `TRACKP_JOINT_LIQ=1` turns it on.
fn joint_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("TRACKP_JOINT_LIQ").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    })
}

pub fn objective(st: &State, me: usize) -> f64 {
    let day = st.step / TPD;
    let (a, b) = if joint_enabled() {
        joint_values(st, me)
    } else {
        (asset_value(st, me), asset_value(st, 1 - me))
    };
    let d = a - b;
    let x = (d / margin_scale(day)).clamp(-40.0, 40.0);
    sigmoid(x) + TIEBREAK_PER_DOLLAR * d.clamp(-2.0e6, 2.0e6)
}

/// Raw dollar difference, for diagnostics and for the harness's reports.
pub fn edge(st: &State, me: usize) -> f64 {
    asset_value(st, me) - asset_value(st, 1 - me)
}
