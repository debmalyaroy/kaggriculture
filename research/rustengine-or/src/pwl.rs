//! Staircase-revenue piecewise-linear model (spec §3.2-3.3).
//!
//! Selling `q` units of product `p` starting from market inventory `v0` earns
//! the concave integer staircase `R(q) = Σ_{m=0}^{q-1} price(v0+m)`, where a unit
//! sold at the $1 floor does NOT advance inventory (pure filler, slope 1 forever).
//!
//! For a MAXIMISATION we encode `R` with SECANT CUTS (no binaries, no SOS2):
//!     rev ≤ R(k) + Δ(k)·(sell − k)     for each breakpoint k
//! with Δ(k) = price(v0+k) = R(k+1)−R(k), non-increasing ⇒ concave ⇒ these
//! upper-bound lines are all valid and tight at integer `sell` (spec §3.3).

use crate::market::{market_price, Params, MARKET_I0};

/// Exact engine revenue for selling `q` units seeded at inventory `v0`.
/// Replicates the $1-floor-adds-no-supply rule (`_commit_unit`, spec §3.2).
pub fn exact_rev(p: &Params, q: i64, v0: i64) -> i64 {
    let mut inv = v0;
    let mut total = 0;
    for _ in 0..q {
        let price = market_price(p, inv);
        total += price;
        if price > 1 {
            inv += 1; // floor units never advance inventory
        }
    }
    total
}

/// One secant cut: rev ≤ r_k + delta_k * (sell − k).
#[derive(Clone, Copy, Debug)]
pub struct Cut {
    pub k: i64,
    pub r_k: i64,     // R(k)
    pub delta_k: i64, // marginal price of unit k
}

#[derive(Clone, Debug)]
pub struct Pwl {
    pub name: &'static str,
    pub v0: i64,
    pub width: i64,     // units above v0 until the $1 floor (staircase width)
    pub cuts: Vec<Cut>, // k = 0 .. width inclusive (tail capped by the slope-1 cut at width)
}

impl Pwl {
    /// Build the breakpoint table seeded at `v0`. Cuts run k=0..=width where
    /// `width` is the first k with price(v0+k)==1; that final Δ=1 cut caps the
    /// whole flat tail in one segment (all tail lines coincide), so the table
    /// stays small (spec §3.3: "represent the flat $1 tail as one segment").
    pub fn build(p: &Params, v0: i64) -> Pwl {
        let mut cuts = Vec::new();
        let mut inv = v0;
        let mut r_k: i64 = 0;
        let mut k: i64 = 0;
        loop {
            let delta = market_price(p, inv);
            cuts.push(Cut { k, r_k, delta_k: delta });
            if delta <= 1 {
                // reached the $1 floor: this slope-1 cut caps the tail. stop.
                break;
            }
            r_k += delta;
            inv += 1;
            k += 1;
            if k > 200_000 {
                break; // safety for staples that never truly floor
            }
        }
        let width = k;
        Pwl { name: p.name, v0, width, cuts }
    }

    /// Evaluate the PWL upper envelope at integer `sell` = min over cuts of the
    /// secant RHS. At integer `sell` this equals `exact_rev` (used to unit-test
    /// the PWL against the engine staircase without a solver).
    pub fn eval(&self, sell: i64) -> i64 {
        self.cuts
            .iter()
            .map(|c| c.r_k + c.delta_k * (sell - c.k))
            .min()
            .unwrap_or(0)
    }
}
