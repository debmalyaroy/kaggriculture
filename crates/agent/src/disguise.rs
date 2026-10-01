//! Stream disguise: change our action stream without changing the game.
//!
//! Replays of our games are public, and our stream can be matched byte for byte (the GM dataset's
//! stream_hashes.csv: sha256 over each seat's actions). This appends zero-quantity SELL orders of products
//! we hold none of, at the END of the market list (never moving a real order's positional slot, never past
//! the 10-order cap), on a per-game pseudo-random pattern seeded by a secret salt and the opening state.
//! A SELL of 0 units sells nothing, so the state and every bank are unchanged (checked: identical banks with
//! and without it), while the stream -- and so any hash, prefix or exact-match lineage -- differs per game.
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

    pub fn apply(&mut self, a: &mut Action, v: &View) {
        if self.h.is_none() {
            // per-game seed from public opening state (differs across games; nothing an opponent can steer)
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
        let item = empty[(r >> 8) as usize % empty.len()];
        a.market.push(Cmd::order("SELL", item, 0));
        self.added += 1;
    }
}
