//! Exact port of the engine market price function
//! (`vendor/kaggle_environments/envs/kaggriculture/kaggriculture.py`,
//! `market_price` / `_shape`, lines 61-206). Ground truth for the PWL.
//!
//! CRITICAL: the engine uses Python `round()` = round-half-to-EVEN (banker's
//! rounding), then `max(1, int(round(p)))`. round-half-up drifts unit values by
//! $1 exactly at half-dollar quotes (spec §3.1). We replicate round-half-even.

pub const MARKET_I0: i64 = 10_000;
pub const PRICE_FLOOR: i64 = 1;
const HINGE_GAIN: f64 = 8.0;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub name: &'static str,
    pub base: f64,
    pub t: f64,
    pub below_func: Func,
    pub below_target: f64,
    pub above_func: Func,
    pub above_target: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum Func {
    Linear,
    Sq,
    Sqrt,
    Log,
    Log10,
    Hinge,
}

fn shape(f: Func, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match f {
        Func::Linear => x,
        Func::Sq => x * x,
        Func::Sqrt => x.sqrt(),
        Func::Log => (1.0 + x).ln(),
        Func::Log10 => (1.0 + x).log10(),
        Func::Hinge => {
            if t <= 0.0 {
                x
            } else {
                let u = x / t;
                u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
            }
        }
    }
}

/// Python3 round() semantics: round half to even.
fn round_half_even(v: f64) -> f64 {
    let r = v.round(); // Rust rounds half AWAY from zero
    if (v - v.trunc()).abs() == 0.5 {
        // exactly a half: pick the even neighbour
        let lower = v.floor();
        if (lower as i64) % 2 == 0 {
            lower
        } else {
            v.ceil()
        }
    } else {
        r
    }
}

/// Marginal price to sell ONE unit when the market currently holds `inventory`
/// of `item` (priced at pre-sell inventory for a SELL). Exact engine port.
pub fn market_price(p: &Params, inventory: i64) -> i64 {
    let inv = inventory as f64;
    let i0 = MARKET_I0 as f64;
    let price = if inventory < MARKET_I0 {
        let amp = p.below_target * p.base / shape(p.below_func, p.t, p.t);
        p.base + amp * shape(p.below_func, i0 - inv, p.t)
    } else {
        let amp = p.above_target * p.base / shape(p.above_func, p.t, p.t);
        p.base - amp * shape(p.above_func, inv - i0, p.t)
    };
    (round_half_even(price) as i64).max(PRICE_FLOOR)
}

/// Verified MARKET_PARAMS subset (spec §2.5 / kaggriculture.py lines 42-51).
/// Only the six scarcity products needed for the toy model + PWL test, plus
/// the two staples used to show the "dumping is free but worthless" contrast.
pub const STRAWBERRY: Params = Params { name: "STRAWBERRY", base: 120.0, t: 100.0, below_func: Func::Sqrt, below_target: 0.70, above_func: Func::Linear, above_target: 1.60 };
pub const WOOL: Params = Params { name: "WOOL", base: 200.0, t: 105.0, below_func: Func::Log, below_target: 0.20, above_func: Func::Sq, above_target: 3.20 };
pub const MELON: Params = Params { name: "MELON", base: 250.0, t: 300.0, below_func: Func::Log, below_target: 0.20, above_func: Func::Sq, above_target: 3.60 };
pub const MILK: Params = Params { name: "MILK", base: 160.0, t: 122.0, below_func: Func::Sqrt, below_target: 0.60, above_func: Func::Linear, above_target: 1.60 };
pub const CARROT: Params = Params { name: "CARROT", base: 35.0, t: 450.0, below_func: Func::Hinge, below_target: 1.00, above_func: Func::Sqrt, above_target: 0.70 };
pub const TOMATO: Params = Params { name: "TOMATO", base: 60.0, t: 200.0, below_func: Func::Hinge, below_target: 0.40, above_func: Func::Sqrt, above_target: 0.60 };
pub const WHEAT: Params = Params { name: "WHEAT", base: 25.0, t: 400.0, below_func: Func::Sqrt, below_target: 0.80, above_func: Func::Log, above_target: 0.20 };
pub const EGG: Params = Params { name: "EGG", base: 50.0, t: 332.0, below_func: Func::Hinge, below_target: 0.40, above_func: Func::Log, above_target: 0.20 };
