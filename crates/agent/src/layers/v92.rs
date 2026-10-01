//! v9/2 PREDICT (herd_safe ca25 lines 3392-3556; shepherds_ledger's confidence-gated window
//! extension, lines 7090-7160): forecast the rival's premium sales from a library of recorded
//! top-player sale streams and sell our planned lots just before theirs.
//!
//! Each turn the rival's executed sales are recovered exactly like RACE does (inventory delta +
//! town draw - own sales, from RACE's previous-turn snapshot). Library streams with the same
//! first two shops are scored against the rival's recovered sale ticks of the last 240 turns;
//! when the best stream sells >= K units of MILK / WOOL / STRAWBERRY in the next two turns
//! (extension: next W turns, when the stream has matched the rival's recent sales), our tape's
//! planned sales of that product within H turns are sold now.
//!
//! Ablation on the faithful harness (24 worlds x 2 seats, 2026-09-25): removing v92 from ca25
//! moved v62.1's score vs ca25 from 0.521 to 0.729 (paired +10/-0, p 0.002) -- the one piece of
//! that family's clone game we lack. Knob-driven (`v92_*`), off by default (= v61.1).
use super::knobs::Knobs;
use super::v9::V9;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::obs::{qget, Qty};
use crate::view::View;
use std::collections::HashMap;

static LIB: &[u8] = include_bytes!("../../data/v92_lib.bin");
const ITEMS: [&str; 5] = ["MILK", "WOOL", "STRAWBERRY", "EGG", "MELON"];
const USE: [&str; 3] = ["MILK", "WOOL", "STRAWBERRY"];
const NAMES: [&str; 8] = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"];
const MAX_ORDERS: usize = 10;

type Ev = HashMap<(i64, usize), i64>;

/// Bitset over steps 0..768 for one product.
type Bits = [u64; 12];

fn bit(b: &Bits, t: i64) -> bool {
    (0..768).contains(&t) && b[(t >> 6) as usize] >> (t & 63) & 1 == 1
}

/// A stream's keys sorted by step, and per product the steps within 1 of a recorded sale (the
/// `near` test of the forecast as a bit lookup instead of three hash probes).
#[derive(Clone)]
struct EvIdx {
    keys: Vec<(i64, usize)>,
    near: [Bits; 5],
}

fn index(ev: &Ev) -> EvIdx {
    let mut keys: Vec<(i64, usize)> = ev.keys().copied().collect();
    keys.sort();
    let mut near = [[0u64; 12]; 5];
    for &(t, i) in &keys {
        for u in [t - 1, t, t + 1] {
            if (0..768).contains(&u) && i < 5 {
                near[i][(u >> 6) as usize] |= 1 << (u & 63);
            }
        }
    }
    EvIdx { keys, near }
}

/// One shop pair's recorded streams, in library order.
fn pair_streams(a: &str, b: &str) -> Vec<Ev> {
    let (Some(i), Some(j)) = (NAMES.iter().position(|n| *n == a), NAMES.iter().position(|n| *n == b)) else { return vec![] };
    if LIB.len() < 8 + 64 * 8 || &LIB[..4] != b"V92L" {
        return vec![];
    }
    let k = i * 8 + j;
    let at = 8 + k * 8;
    let start = u32::from_le_bytes(LIB[at..at + 4].try_into().unwrap()) as usize;
    let raw = &LIB[8 + 64 * 8..];
    let mut pos = start;
    let rd = |p: usize| raw.get(p).copied().unwrap_or(0) as i64;
    let n = rd(pos) | rd(pos + 1) << 8;
    pos += 2;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let m = rd(pos) | rd(pos + 1) << 8;
        pos += 2;
        let mut ev = Ev::new();
        let mut last = 0i64;
        for _ in 0..m {
            let d = rd(pos);
            pos += 1;
            let t = if d == 255 {
                let t = rd(pos) | rd(pos + 1) << 8;
                pos += 2;
                t
            } else {
                last + d
            };
            ev.insert((t, rd(pos) as usize), rd(pos + 1));
            pos += 2;
            last = t;
        }
        out.push(ev);
    }
    out
}

#[derive(Default, Clone)]
struct St {
    step: i64,
    seen: HashMap<(i64, usize), i64>,
    /// `seen` as per-product bitsets (for the fast forecast)
    seen_bits: [Bits; 5],
    best: Option<Vec<usize>>,
    /// library pair index of the current game (set on the first forecast)
    pair: Option<usize>,
    /// v29 market pressure per ITEMS entry (decaying recovered rival flow) and units already advanced
    pressure: [f64; 5],
    press_added: [i64; 5],
}

#[derive(Default, Clone)]
pub struct V92 {
    players: Vec<(i64, St)>,
    lib: Vec<((&'static str, &'static str), Vec<Ev>)>,
    idx: Vec<Vec<EvIdx>>,
}

impl V92 {
    fn streams(&mut self, a: &'static str, b: &'static str) -> usize {
        if let Some(i) = self.lib.iter().position(|(k, _)| *k == (a, b)) {
            return i;
        }
        let evs = pair_streams(a, b);
        self.idx.push(evs.iter().map(index).collect());
        self.lib.push(((a, b), evs));
        self.lib.len() - 1
    }

    /// `_v92_p_update`: record the rival's recovered sales of the previous turn.
    fn update(st: &mut St, v: &View, v9: &V9) {
        let Some(prev) = v9.race_prev(v.obs.player).filter(|p| p.step == v.step - 1) else { return };
        let draw = super::v9::town_draw(&prev.shops, prev.step);
        for (i, item) in ITEMS.iter().enumerate() {
            if qget(&prev.prices, item) <= 3 {
                continue;
            }
            let sold = qget(&v.obs.mkt_inventory, item) - qget(&prev.inventory, item) + qget(&draw, item) - qget(&prev.own, item);
            // v29 adaptive-market pressure: a decaying public-market signal of rival selling
            st.pressure[i] = (st.pressure[i] * 0.72 + (sold.max(0) as f64).min(24.0)).max(0.0);
            if sold >= 2 {
                st.seen.insert((v.step - 1, i), sold);
                let t = v.step - 1;
                if (0..768).contains(&t) {
                    st.seen_bits[i][(t >> 6) as usize] |= 1 << (t & 63);
                }
            }
        }
    }

    /// `_v92_p_forecast`: indices of the top-`top` streams of this shop pair.
    fn forecast(evs: &[Ev], seen: &HashMap<(i64, usize), i64>, step: i64, top: usize) -> Vec<usize> {
        let lo = step - 240;
        let near = |m: &dyn Fn(&(i64, usize)) -> bool, t: i64, i: usize| m(&(t, i)) || m(&(t - 1, i)) || m(&(t + 1, i));
        let in_seen = |k: &(i64, usize)| seen.contains_key(k);
        let recent: Vec<(i64, usize)> = seen.keys().filter(|(t, _)| *t >= lo).copied().collect();
        let mut scored: Vec<(f64, usize)> = evs
            .iter()
            .enumerate()
            .map(|(c, ev)| {
                let (mut m, mut f) = (0i64, 0i64);
                for &(tt, i) in ev.keys() {
                    if lo <= tt && tt < step - 1 {
                        if near(&in_seen, tt, i) {
                            m += 1;
                        } else {
                            f += 1;
                        }
                    }
                }
                let in_ev = |k: &(i64, usize)| ev.contains_key(k);
                let miss = recent.iter().filter(|&&(tt, i)| !near(&in_ev, tt, i)).count() as f64;
                (m as f64 - 0.5 * f as f64 - 0.5 * miss, c)
            })
            .collect();
        // Python: stable sort by -score, keep the first `top`.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top).map(|(_, c)| c).collect()
    }

    /// `forecast` on the bitset index: the same integer counts (m, f, miss), the same score and the
    /// same stable sort, so the same streams in the same order.
    fn forecast_fast(idx: &[EvIdx], seen: &[Bits; 5], step: i64, top: usize) -> Vec<usize> {
        let lo = step - 240;
        let near_seen = |t: i64, i: usize| bit(&seen[i], t) || bit(&seen[i], t - 1) || bit(&seen[i], t + 1);
        // recent = every seen tick at t >= lo (seen holds ticks up to step - 1 only)
        let mut recent: Vec<(i64, usize)> = vec![];
        for (i, b) in seen.iter().enumerate() {
            for t in lo.max(0)..step.min(768) {
                if bit(b, t) {
                    recent.push((t, i));
                }
            }
        }
        let mut scored: Vec<(f64, usize)> = idx
            .iter()
            .enumerate()
            .map(|(c, ev)| {
                let (mut m, mut f) = (0i64, 0i64);
                let start = ev.keys.partition_point(|k| k.0 < lo);
                for &(tt, i) in &ev.keys[start..] {
                    if tt >= step - 1 {
                        break;
                    }
                    if near_seen(tt, i) {
                        m += 1;
                    } else {
                        f += 1;
                    }
                }
                let miss = recent.iter().filter(|&&(tt, i)| !(i < 5 && bit(&ev.near[i], tt))).count() as f64;
                (m as f64 - 0.5 * f as f64 - 0.5 * miss, c)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top).map(|(_, c)| c).collect()
    }

    /// Units the stream predicts for item `i` (shepherds `_hp_quantity` when `ext_window` > 0).
    fn quantity(ev: &Ev, i: usize, step: i64, seen: &HashMap<(i64, usize), i64>, k: i64, ext_window: i64) -> i64 {
        let g = |t: i64| ev.get(&(t, i)).copied().unwrap_or(0);
        let immediate = g(step + 1) + g(step + 2);
        if ext_window <= 0 || immediate >= k {
            return immediate;
        }
        let extended: i64 = (1..=ext_window).map(|d| g(step + d)).sum();
        if extended < k {
            return immediate;
        }
        let prior: Vec<i64> = ev.iter().filter(|(&(t, it), &q)| step - 240 <= t && t < step - 1 && it == i && q >= 2).map(|(&(t, _), _)| t).collect();
        let hits = prior.iter().filter(|&&t| (-1..=1).any(|d| seen.contains_key(&(t + d, i)))).count();
        if hits >= 3 && hits as f64 >= 0.7 * prior.len() as f64 {
            extended
        } else {
            immediate
        }
    }

    /// The layer (runs right after the V9 opening, before RACE). State updates every turn;
    /// orders change only when `k.v92_on`.
    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis, v9: &V9, k: &Knobs) -> Action {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.players.iter().position(|(p, _)| *p == player) {
            Some(i) => i,
            None => {
                self.players.push((player, St { step: -1, ..Default::default() }));
                self.players.len() - 1
            }
        };
        if step <= self.players[i].1.step {
            self.players[i].1 = St { step: -1, ..Default::default() };
        }
        self.players[i].1.step = step;
        Self::update(&mut self.players[i].1, v, v9);
        // the forecast also feeds counter D's rival model and the end-game sale search
        let need = k.v92_on || k.cxd_model > 0 || (k.tsell_on && k.tsell_model > 0);
        if need && step >= 150 && v.obs.shops.len() >= 2 {
            let pair = self.streams(v.obs.shops[0], v.obs.shops[1]);
            self.players[i].1.pair = Some(pair);
            let every = k.v92_every.max(1);
            if step % every == 0 || self.players[i].1.best.is_none() {
                let best = if std::env::var_os("KAGG_V92_SLOW").is_some() {
                    Self::forecast(&self.lib[pair].1, &self.players[i].1.seen, step, k.v92_top.max(1) as usize)
                } else {
                    Self::forecast_fast(&self.idx[pair], &self.players[i].1.seen_bits, step, k.v92_top.max(1) as usize)
                };
                self.players[i].1.best = Some(best);
            }
        }
        let action = if k.press_on { self.press(action, v, ch, k, i) } else { action };
        if !k.v92_on || step < 150 || step >= 700 || v.obs.shops.len() < 2 {
            return action;
        }
        let pair = self.streams(v.obs.shops[0], v.obs.shops[1]);
        let best = self.players[i].1.best.clone().unwrap_or_default();
        if best.is_empty() {
            return action;
        }
        let Some(route) = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route) else { return action };
        if ch.route_idx(route).is_none() {
            return action;
        }
        let tape = &ch.route(route).tape;
        let mut market = action.market.clone();
        let already: Vec<&str> = market.iter().filter(|o| o.len() > 1 && matches!(o.op(), "SELL" | "BUY_PRODUCT")).map(|o| o.s(1)).collect();
        let stock: Qty = ch.projected_shed(&action, v);
        let mut changed = false;
        let st = &self.players[i].1;
        for (idx, item) in ITEMS.iter().enumerate() {
            if !USE.contains(item) || already.contains(item) || market.len() >= MAX_ORDERS {
                continue;
            }
            let votes = best.iter().filter(|&&c| Self::quantity(&self.lib[pair].1[c], idx, step, &st.seen, k.v92_k, k.v92_ext_window) >= k.v92_k).count();
            if votes < 1 {
                continue;
            }
            let mut ours = 0i64;
            let end = (tape.len() as i64).min(step + k.v92_h + 1);
            for t in step + 1..end {
                ours += tape[t as usize].market.iter().filter(|o| o.is_sell3() && o.s(1) == *item).map(|o| o.n(2).min(100)).sum::<i64>();
            }
            let qty = qget(&stock, item).min(ours);
            if qty > 0 {
                market.insert(0, Cmd::order("SELL", item, qty));
                changed = true;
            }
        }
        if !changed {
            return action;
        }
        market.truncate(MAX_ORDERS);
        Action { market, ..action }
    }

    /// The tape action of `player` at step `t` (route 2 from step 648, like RSA).
    fn tape_at(ch: &Chassis, player: i64, t: i64) -> Option<&Action> {
        let route = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route)?;
        let r = if t >= 648 { 2 } else { route };
        ch.route(r).tape.get(t as usize)
    }

    /// Planned SELL units of `item` per step in from..=to (capped at step 718).
    fn planned(ch: &Chassis, player: i64, item: &str, from: i64, to: i64) -> Vec<i64> {
        (from..=to.min(718))
            .map(|t| {
                Self::tape_at(ch, player, t)
                    .map(|a| a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).clamp(0, 100)).sum::<i64>())
                    .unwrap_or(0)
            })
            .collect()
    }

    /// v29 market-pressure layer: when the recovered rival flow of an item is high, advance up to
    /// `press_tranche` of our planned sales of it (within `press_h` turns), at most `press_max`
    /// per game. Knobs `press_*`, off by default.
    fn press(&mut self, action: Action, v: &View, ch: &Chassis, k: &Knobs, i: usize) -> Action {
        let step = v.step;
        if step < k.press_from || step >= 718 {
            return action;
        }
        let mut market = action.market.clone();
        let already: Vec<&str> = market.iter().filter(|o| o.len() > 1 && matches!(o.op(), "SELL" | "BUY_PRODUCT")).map(|o| o.s(1)).collect();
        let stock: Qty = ch.projected_shed(&action, v);
        let mut changed = false;
        for (idx, item) in ITEMS.iter().enumerate() {
            let (pressure, added) = (self.players[i].1.pressure[idx], self.players[i].1.press_added[idx]);
            if pressure < k.press_trigger || already.contains(item) || market.len() >= MAX_ORDERS {
                continue;
            }
            let ours: i64 = Self::planned(ch, v.obs.player, item, step + 1, step + k.press_h).iter().sum();
            let qty = qget(&stock, item).min(ours).min(k.press_tranche).min(k.press_max - added);
            if qty > 0 {
                market.insert(0, Cmd::order("SELL", item, qty));
                self.players[i].1.press_added[idx] += qty;
                changed = true;
            }
        }
        if !changed {
            return action;
        }
        market.truncate(MAX_ORDERS);
        Action { market, ..action }
    }

    /// The best stream's predicted rival sale of ITEMS[idx] at tick `t` (None: no forecast).
    fn rival_units(&self, player: i64, idx: usize, t: i64) -> Option<i64> {
        let st = &self.players.iter().find(|(p, _)| *p == player)?.1;
        let (pair, c) = (st.pair?, *st.best.as_ref()?.first()?);
        Some(self.lib[pair].1[c].get(&(t, idx)).copied().unwrap_or(0))
    }

    /// Counter D's rival model from the forecast: this turn's predicted rival SELLs.
    pub fn predicted_rival(&self, player: i64, step: i64) -> Option<Vec<Cmd>> {
        let mut out = vec![];
        for (idx, item) in ITEMS.iter().enumerate() {
            let q = self.rival_units(player, idx, step)?;
            if q > 0 {
                out.push(Cmd::order("SELL", item, q));
            }
        }
        Some(out)
    }

    /// End-game sale-timing search (knobs `tsell_*`): from `tsell_from`, for each premium item pick
    /// how many turns of planned sales to pull into this turn by simulating the item's market over
    /// `tsell_window` turns (the rival's sales first, then ours, then the town draw) with the engine
    /// price curve. Rival model: 0 = copy (our own tape), 1 = v92 forecast (copy when none), 2 =
    /// worst case of both. Acts only when the best advance beats no advance by `tsell_min` dollars.
    pub fn tsell(&self, action: Action, v: &View, ch: &Chassis, k: &Knobs) -> Action {
        let (step, player) = (v.step, v.obs.player);
        if step < k.tsell_from || step >= 718 {
            return action;
        }
        const ADV: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
        let w = k.tsell_window.clamp(1, 48);
        let mut market = action.market.clone();
        let stock: Qty = ch.projected_shed(&action, v);
        let draws: Vec<Qty> = (0..=w).map(|d| super::v9::town_draw(&v.obs.shops, step + d)).collect();
        let mut changed = false;
        for (idx, item) in ITEMS.iter().enumerate() {
            if market.len() >= MAX_ORDERS || market.iter().any(|o| o.len() > 1 && o.op() == "BUY_PRODUCT" && o.s(1) == *item) {
                continue;
            }
            let cur: i64 = market.iter().filter(|o| o.is_sell3() && o.s(1) == *item).map(|o| o.n(2).clamp(0, 100)).sum();
            let avail = (qget(&stock, item) - cur).max(0);
            if avail <= 0 {
                continue;
            }
            let fut = Self::planned(ch, player, item, step + 1, step + w);
            if fut.iter().sum::<i64>() <= 0 {
                continue;
            }
            let pidx = crate::market::item_index(item);
            let inv0 = qget(&v.obs.mkt_inventory, item);
            let ours_for = |a: i64| -> Vec<i64> {
                let pool: i64 = fut.iter().take(a as usize).sum();
                let mut extra = pool.min(avail);
                let mut out = vec![cur + extra];
                for &f in &fut {
                    let take = f.min(extra);
                    extra -= take;
                    out.push(f - take);
                }
                out
            };
            let copy: Vec<i64> = std::iter::once(cur).chain(fut.iter().copied()).collect();
            let pred: Option<Vec<i64>> = (0..=w).map(|d| self.rival_units(player, idx, step + d)).collect();
            let rivals: Vec<Vec<i64>> = match (k.tsell_model, pred) {
                (1, Some(p)) => vec![p],
                (2, Some(p)) => vec![copy.clone(), p],
                _ => vec![copy.clone()],
            };
            let revenue = |ours: &[i64], rival: &[i64]| -> f64 {
                let mut inv = inv0;
                let mut rev = 0.0;
                for d in 0..ours.len() {
                    inv += rival.get(d).copied().unwrap_or(0).max(0);
                    for _ in 0..ours[d] {
                        rev += crate::market::price_i(pidx, item, inv) as f64;
                        inv += 1;
                    }
                    inv = (inv - qget(&draws[d.min(draws.len() - 1)], item)).max(0);
                }
                rev
            };
            let score = |a: i64| -> f64 { rivals.iter().map(|r| revenue(&ours_for(a), r)).fold(f64::INFINITY, f64::min) };
            let base = score(0);
            let (mut best_a, mut best) = (0, base);
            for &a in &ADV[1..] {
                if a > w {
                    break;
                }
                let sc = score(a);
                if sc > best {
                    best = sc;
                    best_a = a;
                }
            }
            if best_a > 0 && best - base >= k.tsell_min {
                let extra = ours_for(best_a)[0] - cur;
                if extra > 0 {
                    market.insert(0, Cmd::order("SELL", item, extra));
                    changed = true;
                }
            }
        }
        if !changed {
            return action;
        }
        market.truncate(MAX_ORDERS);
        Action { market, ..action }
    }
}
