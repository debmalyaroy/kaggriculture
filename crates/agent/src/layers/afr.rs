//! AFR — anti-front-run (ours, not part of v61.1; off unless `knobs.afr_on`).
//!
//! Detect: the rival sold item X last step (public market-inventory rise net of town drain, on an
//! item we did not sell) while we were HOLDING X in the shed with a route sale of X still ahead of
//! us. That is a pre-emption: they reach the market before our schedule.
//!
//! Respond, per flagged item only: pull our planned route sales of X forward over a horizon of
//! (observed lead + `afr_extra` + per-game jitter) steps, front of the queue, above a price floor.
//! Unflagged items keep the base timing, so a rival that never pre-empts sees the base agent.
//!
//! Jitter (0..=`afr_jitter` extra steps per item) is drawn once per game from a hash of the
//! game's own opening state and a build salt: replays show the distribution, not the draw.
//! Market-side only: WHEN and HOW MUCH we sell, never what the farm does.
use super::knobs::Knobs;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::obs::{qadd, qget, Qty};
use crate::view::View;

const ITEMS: [&str; 7] = ["STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"];
const FROM: i64 = 144;
const TO: i64 = 718;
const SALT: u64 = 0x5eed_af12_2026_0925;

fn base(item: &str) -> i64 {
    match item {
        "STRAWBERRY" => 120,
        "WOOL" => 200,
        "EGG" => 50,
        "MILK" => 160,
        "MELON" => 250,
        "CARROT" => 35,
        "TOMATO" => 60,
        _ => 0,
    }
}

fn shop_items(s: &str) -> &'static [&'static str] {
    match s {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => &[],
    }
}

#[derive(Clone)]
struct Snap {
    step: i64,
    inventory: Qty,
    shed: Qty,
    shops: Vec<&'static str>,
    sold: Vec<&'static str>,
}

#[derive(Clone, Default)]
struct St {
    prev: Option<Snap>,
    /// (item, flagged until step, lead in steps)
    flags: Vec<(&'static str, i64, i64)>,
    jitter: Vec<(&'static str, i64)>,
    events: u32,
    /// steps at which pre-emptions were detected
    event_steps: Vec<i64>,
}

#[derive(Clone, Default)]
pub struct Afr {
    players: Vec<(i64, St)>,
}

fn fnv(h: u64, s: &str) -> u64 {
    let mut h = h;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl Afr {
    fn st(&mut self, p: i64) -> &mut St {
        let i = match self.players.iter().position(|(q, _)| *q == p) {
            Some(i) => i,
            None => {
                self.players.push((p, St::default()));
                self.players.len() - 1
            }
        };
        &mut self.players[i].1
    }

    /// Pre-emption events so far for `player` (diagnostics).
    pub fn events(&self, player: i64) -> u32 {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s.events).unwrap_or(0)
    }

    /// Pre-emptions detected in the `window` steps before `step`.
    pub fn recent(&self, player: i64, step: i64, window: i64) -> usize {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s.event_steps.iter().filter(|t| **t >= step - window && **t < step).count()).unwrap_or(0)
    }

    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis, k: &Knobs) -> Action {
        let step = v.step;
        let player = v.obs.player;
        let route = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route);
        let planned_sell = |t: i64, item: &str| -> bool {
            let Some(route) = route else { return false };
            let r = if t >= 648 { 2 } else { route };
            ch.route(r).tape.get(t as usize).is_some_and(|a| a.market.iter().any(|o| o.is_sell3() && o.s(1) == item && o.n(2) > 0))
        };
        let st = self.st(player);
        if st.jitter.is_empty() {
            // once per game: a draw only this game's own opening state (and the salt) determines
            let mut h = fnv(0xcbf2_9ce4_8422_2325 ^ SALT, &format!("{player}"));
            for s in &v.obs.shops {
                h = fnv(h, s);
            }
            for (it, q) in &v.obs.mkt_inventory {
                h = fnv(h, &format!("{it}{q}"));
            }
            h = fnv(h, &format!("{:.0}", v.rival().money));
            st.jitter = ITEMS
                .iter()
                .enumerate()
                .map(|(i, it)| (*it, ((h >> (i * 8)) % (k.afr_jitter.max(0) as u64 + 1)) as i64))
                .collect();
        }

        // ---- detect ----
        if let Some(prev) = st.prev.as_ref().filter(|p| p.step + 1 == step && step % 24 != 0) {
            let t0 = step - 1;
            let mut town: Qty = vec![];
            if t0 % 4 == 0 {
                for s in &prev.shops {
                    let items = shop_items(s);
                    for it in items {
                        qadd(&mut town, it, if items.len() == 1 { 2 } else { 1 });
                    }
                }
            }
            if t0 % 24 == 0 {
                for it in ITEMS {
                    qadd(&mut town, it, 1);
                }
            }
            let mut new_flags = vec![];
            for item in ITEMS {
                if qget(&prev.shed, item) <= 0 || prev.sold.contains(&item) {
                    continue;
                }
                let rival = qget(&v.obs.mkt_inventory, item) - qget(&prev.inventory, item) + qget(&town, item);
                if rival <= 0 {
                    continue;
                }
                if let Some(next) = (step..TO.min(step + 24)).find(|t| planned_sell(*t, item)) {
                    new_flags.push((item, next - t0));
                }
            }
            for (item, lead) in new_flags {
                st.events += 1;
                st.event_steps.push(step);
                let until = step + k.afr_hold;
                match st.flags.iter_mut().find(|f| f.0 == item) {
                    Some(f) => {
                        f.1 = until;
                        f.2 = f.2.max(lead);
                    }
                    None => st.flags.push((item, until, lead)),
                }
            }
        }

        // ---- respond ----
        let mut out = action;
        if (FROM..TO).contains(&step) && step % 24 != 23 && !out.market.iter().any(|o| o.len() > 1 && o.op() == "BUY_PRODUCT") {
            let active: Vec<(&'static str, i64)> = st
                .flags
                .iter()
                .filter(|f| f.1 >= step)
                .map(|f| (f.0, (f.2 + k.afr_extra + st.jitter.iter().find(|j| j.0 == f.0).map(|j| j.1).unwrap_or(0)).clamp(1, k.afr_look_max)))
                .collect();
            if !active.is_empty() {
                let stock = ch.projected_shed(&out, v);
                let mut selling: Qty = vec![];
                for o in &out.market {
                    if o.is_sell3() {
                        qadd(&mut selling, o.s(1), o.n(2).max(0));
                    }
                }
                let mut market = out.market.clone();
                let mut extra: Vec<Cmd> = vec![];
                for (item, look) in active {
                    let floor = 2f64.max(k.afr_min_frac * base(item) as f64);
                    if (v.price(item) as f64) < floor {
                        continue;
                    }
                    let mut planned = 0;
                    for t in (step + 1)..=(step + look).min(TO) {
                        let Some(route) = route else { break };
                        let r = if t >= 648 { 2 } else { route };
                        if let Some(a) = ch.route(r).tape.get(t as usize) {
                            planned += a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum::<i64>();
                        }
                    }
                    let want = (qget(&stock, item) - qget(&selling, item)).min(planned);
                    if want < 1 {
                        continue;
                    }
                    if let Some(o) = market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                        let nv = o.n(2) + want;
                        o.set_n(2, nv);
                    } else if market.len() + extra.len() < 10 {
                        extra.push(Cmd::order("SELL", item, want));
                    }
                    qadd(&mut selling, item, want);
                }
                if !extra.is_empty() || market != out.market {
                    out.market = extra.into_iter().chain(market).collect();
                }
            }
        }

        // ---- remember ----
        let sold: Vec<&'static str> = ITEMS.iter().copied().filter(|it| out.market.iter().any(|o| o.is_sell3() && o.s(1) == *it)).collect();
        st.prev = Some(Snap { step, inventory: v.obs.mkt_inventory.clone(), shed: v.shed.clone(), shops: v.obs.shops.clone(), sold });
        st.flags.retain(|f| f.1 >= step);
        out
    }
}
