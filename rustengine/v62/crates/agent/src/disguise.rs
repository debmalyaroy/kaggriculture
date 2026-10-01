//! Disguise (knob-driven, off by default), applied after every layer.
//!
//! * STREAM (`disguise_stream`): replays of our games are public and our action stream can be matched byte for
//!   byte (GM's stream_hashes.csv). Zero-quantity SELL orders of products we hold none of are appended at the
//!   END of the market list (never moving a real order's slot, never past the 10-order cap) on a per-game
//!   pseudo-random pattern. The engine drops a 0-unit order, so state and banks are unchanged; the stream (any
//!   hash, prefix or exact-match lineage) differs per game. Port of kaggriculture-rl `agent/src/disguise.rs`.
//! * MONEY (`disguise_money` = N > 0): opponents never see our orders, only the visible farm state; public
//!   mirror detectors key on it (e.g. `_v58_exact_public_mirror`: both public farms exactly equal). Selling N
//!   wheat at step 1 makes our money and shed differ from an exact copy from step 2 on (cost ~N x $30), so
//!   exact-equality detectors never fire and money fingerprints taken at step 2 change. Tolerance-based
//!   detectors (herd / hands / quadrants / money within a margin) still see a near mirror.
use crate::act::{Action, Cmd};
use crate::view::{View, PRODUCTS};

const SALT: u64 = 0x5eed_d15c_2026_0926;

#[derive(Clone, Debug, Default)]
pub struct Disguise {
    h: Option<u64>,
    pub added: usize,
}

impl Disguise {
    fn next(&mut self) -> u64 {
        let mut x = self.h.unwrap_or(SALT);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.h = Some(x);
        x
    }

    pub fn stream(&mut self, a: &mut Action, v: &View) {
        if v.step == 0 {
            self.h = None;
        }
        if self.h.is_none() {
            let mut s = SALT ^ (v.rival().money.to_bits()).rotate_left(17) ^ (v.step as u64).wrapping_mul(0x9e37_79b9);
            for (k, n) in v.obs.mkt_inventory.iter() {
                for b in k.bytes() {
                    s = (s ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                }
                s ^= *n as u64;
            }
            self.h = Some(s | 1);
        }
        let r = self.next();
        if r % 3 != 0 || a.market.len() >= 10 {
            return;
        }
        let empty: Vec<&str> = PRODUCTS.iter().copied().filter(|p| v.shed(p) <= 0).collect();
        if empty.is_empty() {
            return;
        }
        a.market.push(Cmd::order("SELL", empty[(r >> 8) as usize % empty.len()], 0));
        self.added += 1;
    }

    pub fn money(a: &mut Action, v: &View, n: i64) {
        if n > 0 && v.step == 1 && v.shed("WHEAT") >= n && a.market.len() < 10 {
            a.market.push(Cmd::order("SELL", "WHEAT", n));
        }
    }
}
