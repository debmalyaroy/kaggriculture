//! SALE manager: demand-aware batching + small-batch front-running (operator 30 Sep).
//!
//! Runs after the chain's sale decisions (chain -> reactive shell -> sales shell -> HERE -> preempt / standoff), on
//! market orders only, so it can never desync the tape. Per item and turn, with the engine's own price curve:
//!
//!   glut      an item is GLUTTED when its quote is <= `glut_ratio` x base (the market holds more than it drains).
//!   big       NOT glutted: we may sell big -- every unit whose own fill price stays >= `glut_ratio` x base
//!             (selling down to the glut line, never into it) -- plus the demand batch below.
//!   demand    glutted (or past the glut line): only the DEMAND batch = ceil(`demand_mult` x town drain over the next
//!             `window` steps) + `min_batch`: what the town absorbs before our next chance, so we never flood it.
//!   carry     units the cap held back are carried (`carry`) and offered again on later turns under the same rule.
//!   flush     no cap from `flush_from` (the endgame liquidates), nor when the projected shed is within `room_margin`
//!             of full, nor for items outside `items`.
//!   front     FRONT-RUN (operator: "sell first, in a small batch, to force the rival"): when the rival is forecast to
//!             sell item X within `front_look` steps -- its own schedule this game (hour-of-day table, p >= `front_p`
//!             after `front_days` days) AND it holds >= `front_min_stock` units, OR it holds >= `front_state_stock`
//!             while X quotes above base (a state-driven dump) -- we sell a batch of X NOW, ahead of it: the demand
//!             batch when X is glutted, down to the glut line when it is not. Only stock the tape sells later anyway
//!             (our planned sells over `front_horizon` steps), never inputs (WHEAT / FERTILIZER by default).
//!
//! Everything is a knob (`SaleCfg`, JSON keys = field names); `on: false` (default) = inert.
use crate::act::{Action, Cmd};
use crate::chassis::{base_price, Chassis};
use crate::obs::{qadd, qget, Qty};
use crate::view::*;
use kagg_engine::json::Json;

#[derive(Clone, Debug)]
pub struct SaleCfg {
    pub on: bool,
    pub items: Vec<&'static str>,
    pub glut_ratio: f64,
    pub window: i64,
    pub demand_mult: f64,
    pub min_batch: i64,
    pub carry: bool,
    pub flush_from: i64,
    pub room_margin: i64,
    pub from: i64,
    pub front: bool,
    pub front_items: Vec<&'static str>,
    pub front_look: i64,
    pub front_p: f64,
    pub front_days: i64,
    pub front_min_stock: i64,
    pub front_state_stock: i64,
    pub front_horizon: i64,
    pub front_from: i64,
    pub front_to: i64,
    pub max_orders: usize,
    /// PRICE MODE (30 Sep, operator: "rule-based selling must use the price calculator; sell to demand only what
    /// dumps"): an item's sale is cut to the demand batch only when the price calculator says holding the rest earns
    /// more than `hold_margin` over selling it all now -- the market simulated `horizon` steps ahead with the town's
    /// drain AND the rival's tracked stock arriving over `rival_h` steps -- and the rival is not about to sell it.
    pub price_mode: bool,
    pub horizon: i64,
    pub rival_h: i64,
    pub hold_margin: f64,
}

const SELLABLE: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];

impl Default for SaleCfg {
    fn default() -> Self {
        SaleCfg {
            on: false,
            items: SELLABLE.to_vec(),
            glut_ratio: 1.0,
            window: 4,
            demand_mult: 1.0,
            min_batch: 1,
            carry: true,
            flush_from: 700,
            room_margin: 10,
            from: 0,
            front: false,
            front_items: SELLABLE.to_vec(),
            front_look: 1,
            front_p: 0.5,
            front_days: 2,
            front_min_stock: 3,
            front_state_stock: 15,
            front_horizon: 48,
            front_from: 144,
            front_to: 711,
            max_orders: 10,
            price_mode: false,
            horizon: 24,
            rival_h: 12,
            hold_margin: 0.03,
        }
    }
}

fn item_list(j: &Json) -> Result<Vec<&'static str>, String> {
    j.arr()
        .iter()
        .map(|x| PRODUCTS.iter().copied().chain(["FERTILIZER"]).find(|p| *p == x.str()).ok_or(format!("unknown item {:?}", x.str())))
        .collect()
}

impl SaleCfg {
    /// `self` with the keys present in `j` overridden; unknown keys are an error.
    pub fn with(&self, j: &Json) -> Result<SaleCfg, String> {
        let mut c = self.clone();
        for (k, v) in j.obj() {
            match k.as_str() {
                "on" => c.on = v.bool(),
                "items" => c.items = item_list(v)?,
                "glut_ratio" => c.glut_ratio = v.f64(),
                "window" => c.window = v.i64().max(1),
                "demand_mult" => c.demand_mult = v.f64(),
                "min_batch" => c.min_batch = v.i64().max(0),
                "carry" => c.carry = v.bool(),
                "flush_from" => c.flush_from = v.i64(),
                "room_margin" => c.room_margin = v.i64(),
                "from" => c.from = v.i64(),
                "front" => c.front = v.bool(),
                "front_items" => c.front_items = item_list(v)?,
                "front_look" => c.front_look = v.i64().max(1),
                "front_p" => c.front_p = v.f64(),
                "front_days" => c.front_days = v.i64(),
                "front_min_stock" => c.front_min_stock = v.i64(),
                "front_state_stock" => c.front_state_stock = v.i64(),
                "front_horizon" => c.front_horizon = v.i64().max(1),
                "front_from" => c.front_from = v.i64(),
                "front_to" => c.front_to = v.i64(),
                "max_orders" => c.max_orders = v.i64().clamp(1, 10) as usize,
                "price_mode" => c.price_mode = v.bool(),
                "horizon" => c.horizon = v.i64().max(1),
                "rival_h" => c.rival_h = v.i64().max(1),
                "hold_margin" => c.hold_margin = v.f64(),
                "note" | "name" => {}
                o => return Err(format!("sale: unknown knob {o:?}")),
            }
        }
        Ok(c)
    }

    /// Every knob with its current value (for `--dump-knobs`).
    pub fn dump(&self) -> String {
        let l = |v: &[&str]| format!("[{}]", v.iter().map(|x| format!("\"{x}\"")).collect::<Vec<_>>().join(","));
        format!(
            "{{\"on\":{},\"items\":{},\"glut_ratio\":{},\"window\":{},\"demand_mult\":{},\"min_batch\":{},\"carry\":{},\"flush_from\":{},\"room_margin\":{},\"from\":{},\"front\":{},\"front_items\":{},\"front_look\":{},\"front_p\":{},\"front_days\":{},\"front_min_stock\":{},\"front_state_stock\":{},\"front_horizon\":{},\"front_from\":{},\"front_to\":{},\"max_orders\":{},\"price_mode\":{},\"horizon\":{},\"rival_h\":{},\"hold_margin\":{}}}",
            self.on, l(&self.items), self.glut_ratio, self.window, self.demand_mult, self.min_batch, self.carry, self.flush_from, self.room_margin,
            self.from, self.front, l(&self.front_items), self.front_look, self.front_p, self.front_days, self.front_min_stock,
            self.front_state_stock, self.front_horizon, self.front_from, self.front_to, self.max_orders,
            self.price_mode, self.horizon, self.rival_h, self.hold_margin
        )
    }
}

fn shop_items(shop: &str) -> &'static [&'static str] {
    match shop {
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

/// Town drain of `item` over steps [s, s + n): shop ticks every 4 steps (x2 for single-product shops) + the town
/// centre once a day (engine semantics, as in rshell / preempt).
pub fn drain(item: &str, shops: &[&'static str], s: i64, n: i64) -> i64 {
    let per: i64 = shops.iter().map(|sh| {
        let its = shop_items(sh);
        if its.contains(&item) { if its.len() == 1 { 2 } else { 1 } } else { 0 }
    }).sum();
    let mut d = 0;
    for u in s..s + n {
        if u % 4 == 0 {
            d += per;
        }
        if u % 24 == 0 && item != "FERTILIZER" {
            d += 1;
        }
    }
    d
}

/// Units we can sell starting at market inventory `inv` while each unit's own fill price stays >= `floor`.
fn units_above(item: &str, inv: i64, floor: f64, cap: i64) -> i64 {
    let idx = crate::market::item_index(item);
    let mut n = 0;
    while n < cap && crate::market::price_i(idx, item, inv + n) as f64 >= floor {
        n += 1;
    }
    n
}

#[derive(Clone, Debug, Default)]
pub struct Sale {
    pub cfg: std::sync::Arc<SaleCfg>,
    /// per player: units the cap held back, offered again later
    carry: Vec<(i64, Qty)>,
    pub capped: u32,
    pub released: u32,
    pub fronts: u32,
}

impl Sale {
    pub fn new(cfg: std::sync::Arc<SaleCfg>) -> Sale {
        Sale { cfg, ..Default::default() }
    }

    fn carry_mut(&mut self, p: i64) -> &mut Qty {
        let i = match self.carry.iter().position(|(q, _)| *q == p) {
            Some(i) => i,
            None => {
                self.carry.push((p, vec![]));
                self.carry.len() - 1
            }
        };
        &mut self.carry[i].1
    }

    /// How many units of `item` may go to market this turn (the rest is carried).
    fn allowance(&self, item: &str, v: &View) -> i64 {
        let c = &self.cfg;
        let inv = qget(&v.obs.mkt_inventory, item);
        let floor = c.glut_ratio * base_price(item) as f64;
        let big = units_above(item, inv, floor, 10_000);
        let demand = (c.demand_mult * drain(item, &v.obs.shops, v.step, c.window) as f64).ceil() as i64 + c.min_batch;
        big + demand
    }

    /// PRICE MODE: units of `want` to sell now. Everything, unless the price calculator says the demand batch now +
    /// the rest at the best later step (market moved by the town's drain and the rival's stock) earns > hold_margin
    /// more -- and never when the rival is about to sell the item (the race: sell first).
    fn priced(&self, item: &str, v: &View, want: i64, gt: &crate::gt::RivalTracker) -> i64 {
        let c = &self.cfg;
        let keep = self.allowance(item, v).min(want);
        if keep >= want || self.rival_will_sell(item, v, gt) {
            return want;
        }
        let idx = crate::market::item_index(item);
        let px = |x: i64| crate::market::price_i(idx, item, x.max(0)) as f64;
        let sum = |x0: i64, n: i64| (0..n).map(|j| px(x0 + j)).sum::<f64>();
        let inv0 = qget(&v.obs.mkt_inventory, item);
        let rival = gt.stock(item) as f64;
        let rate = rival / c.rival_h as f64;
        let all_now = sum(inv0, want);
        let rest = want - keep;
        let mut x = (inv0 + keep) as f64;
        let mut best_later = 0.0f64;
        let last = (c.flush_from - 1).min(LAST_ACT_STEP - 1);
        for t in 1..=c.horizon {
            let u = v.step + t;
            if u > last {
                break;
            }
            if t <= c.rival_h {
                x += rate;
            }
            x = (x - drain(item, &v.obs.shops, u - 1, 1) as f64).max(0.0);
            best_later = best_later.max(sum(x.round() as i64, rest));
        }
        let split = sum(inv0, keep) + best_later;
        if split > all_now * (1.0 + c.hold_margin) {
            keep
        } else {
            want
        }
    }

    /// Rival forecast: will it sell `item` within the next `front_look` steps?
    fn rival_will_sell(&self, item: &str, v: &View, gt: &crate::gt::RivalTracker) -> bool {
        let c = &self.cfg;
        let stock = gt.stock(item);
        let days = v.step / 24;
        if days >= c.front_days && stock >= c.front_min_stock {
            for k in 1..=c.front_look {
                if gt.sale_forecast(item, v.step + k, days).0 >= c.front_p {
                    return true;
                }
            }
        }
        stock >= c.front_state_stock && v.price(item) > base_price(item)
    }

    pub fn apply(&mut self, a: &mut Action, v: &View, ch: &Chassis, gt: &crate::gt::RivalTracker) {
        let c = self.cfg.clone();
        if !c.on || v.step < c.from || v.step > LAST_ACT_STEP {
            return;
        }
        let p = v.obs.player;
        let proj = ch.projected_shed(a, v);
        let flush = v.step >= c.flush_from || proj.iter().map(|(_, q)| (*q).max(0)).sum::<i64>() >= 100 - c.room_margin;
        // 1. cap this turn's SELLs of glut-prone items to the allowance; carry the rest
        let mut selling: Qty = vec![];
        let mut held: Qty = vec![];
        let mut allow_left: Qty = vec![];
        if c.price_mode {
            let mut want_tot: Qty = vec![];
            for o in a.market.iter().filter(|o| o.is_sell3()) {
                qadd(&mut want_tot, o.s(1), o.n(2).max(0));
            }
            for (it, w) in want_tot.iter() {
                if !flush && c.items.contains(it) {
                    allow_left.push((it, self.priced(it, v, *w, gt)));
                }
            }
        }
        for o in a.market.iter_mut() {
            if !o.is_sell3() {
                continue;
            }
            let item = o.s(1);
            let want = o.n(2).max(0);
            if flush || !c.items.contains(&item) {
                qadd(&mut selling, item, want);
                continue;
            }
            if !allow_left.iter().any(|(k, _)| *k == item) {
                let al = self.allowance(item, v);
                allow_left.push((item, al));
            }
            let left = qget(&allow_left, item);
            let q = want.min(left.max(0));
            if q < want {
                o.set_n(2, q); // keep the slot (a zero order holds the positional race)
                qadd(&mut held, item, want - q);
                self.capped += 1;
            }
            qadd(&mut allow_left, item, -q);
            qadd(&mut selling, item, q);
        }
        // 2. carried units: release what the allowance still permits (or everything on flush); never beyond the shed
        let mut carry = std::mem::take(self.carry_mut(p));
        if c.carry {
            for (item, q) in held.iter() {
                qadd(&mut carry, item, *q);
            }
            let mut keep: Qty = vec![];
            for (item, q) in carry.iter() {
                let avail = qget(&proj, item) - qget(&selling, item);
                let q = (*q).min(avail.max(0));
                if q <= 0 {
                    continue;
                }
                let room = if flush || !c.items.contains(item) {
                    q
                } else {
                    let al = if allow_left.iter().any(|(k, _)| k == item) {
                        qget(&allow_left, item)
                    } else if c.price_mode {
                        self.priced(item, v, q, gt)
                    } else {
                        self.allowance(item, v)
                    };
                    q.min(al.max(0))
                };
                if room > 0 && sell_more(a, item, room, c.max_orders) {
                    qadd(&mut selling, item, room);
                    self.released += 1;
                    if q > room {
                        qadd(&mut keep, item, q - room);
                    }
                } else {
                    qadd(&mut keep, item, q);
                }
            }
            *self.carry_mut(p) = keep;
        }
        // 3. front-run: the rival is about to sell X -> sell a batch of X first
        if c.front && v.step >= c.front_from && v.step <= c.front_to {
            let r = ch.players.iter().find(|(q, _)| *q == p).and_then(|(_, s)| s.route);
            for item in c.front_items.iter().copied() {
                if !self.rival_will_sell(item, v, gt) {
                    continue;
                }
                let planned = r.map(|rid| horizon_sells(ch, rid, item, v.step + 1, c.front_horizon)).unwrap_or(0);
                let avail = (qget(&proj, item) - qget(&selling, item)).min(planned);
                if avail <= 0 {
                    continue;
                }
                let q = avail.min(self.allowance(item, v).max(0));
                if q > 0 && sell_more(a, item, q, c.max_orders) {
                    qadd(&mut selling, item, q);
                    self.fronts += 1;
                }
            }
        }
    }
}

/// Our tape's SELLs of `item` over steps [from, from + h).
fn horizon_sells(ch: &Chassis, route: i64, item: &str, from: i64, h: i64) -> i64 {
    let r = ch.route(route);
    (from..from + h).filter_map(|t| r.at(t)).flat_map(|a| a.market.iter()).filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum()
}

/// Add `q` to this turn's SELL of `item` (the first non-empty SELL of it), else append an order if a slot is free.
fn sell_more(a: &mut Action, item: &'static str, q: i64, max_orders: usize) -> bool {
    if let Some(o) = a.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
        let n = o.n(2).max(0) + q;
        o.set_n(2, n);
        return true;
    }
    if a.market.len() >= max_orders {
        return false;
    }
    a.market.push(Cmd::order("SELL", item, q));
    true
}
