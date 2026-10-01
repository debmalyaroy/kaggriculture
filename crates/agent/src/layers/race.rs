//! RACE clone-gated reservation horizon (agent 5432-5549; the clone level is 9: the module
//! global is reassigned at 5692 before any state is created) and R127 (5552-5647): drop
//! last-hour plants that would end the turn unwatered, and fund urgent grain in slot 0.
use super::r37::similarity;
use super::r85;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qget, Qty};
use crate::sim;
use crate::view::View;

pub const HORIZON_CLONE: i64 = 9;
pub const HORIZON_ESCALATED: i64 = 24;
pub const HORIZON_MIRROR: i64 = 24;
const ITEMS: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];
const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];

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
    prices: Qty,
    shops: Vec<&'static str>,
    shed: Qty,
}
#[derive(Clone)]
struct St {
    step: i64,
    hist: Vec<bool>,
    horizon: i64,
    level: i64,
    prev: Option<Snap>,
    prev_sold: Option<Vec<&'static str>>,
    snapshot: Option<Snap>,
}
impl St {
    fn new(clone_level: i64) -> St {
        St { step: -1, hist: vec![], horizon: 0, level: clone_level, prev: None, prev_sold: None, snapshot: None }
    }
}

#[derive(Default, Clone)]
pub struct Race {
    players: Vec<(i64, St)>,
    r127: Vec<(i64, i64)>,
}

impl Race {
    /// Pre phase; returns the horizon R36 sees this step.
    pub fn pre(&mut self, v: &View, ch: &Chassis, k: &super::knobs::Knobs) -> i64 {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.players.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.players[i].1.step => i,
            Some(i) => {
                self.players[i].1 = St::new(k.race_clone);
                i
            }
            None => {
                self.players.push((player, St::new(k.race_clone)));
                self.players.len() - 1
            }
        };
        let st = &mut self.players[i].1;
        st.step = step;
        st.horizon = 0;
        if step == 1 {
            let (own, rival) = (v.farm().money, v.rival().money);
            if (rival - own).abs() < 0.5 && k.race_mirror > st.level {
                st.level = k.race_mirror;
            }
        }
        if (k.race_from..696).contains(&step) && clone(v, st) {
            if st.level < k.race_escalated && lost(v, st, ch) {
                st.level = k.race_escalated;
            }
            st.horizon = st.level;
        }
        st.snapshot = if (215..696).contains(&step) {
            Some(Snap {
                step,
                inventory: v.obs.mkt_inventory.clone(),
                prices: v.obs.prices.clone(),
                shops: v.obs.shops.clone(),
                shed: v.shed.clone(),
            })
        } else {
            None
        };
        st.horizon
    }

    pub fn post(&mut self, action: &Action, v: &View) {
        if let Some((_, st)) = self.players.iter_mut().find(|(p, _)| *p == v.obs.player) {
            st.prev = st.snapshot.take();
            st.prev_sold = st.prev.as_ref().map(|_| {
                action.market.iter().filter(|o| o.is_sell3() && ITEMS.contains(&o.s(1))).map(|o| o.s(1)).collect()
            });
        }
    }

    /// ADV (6290): `prev_action` becomes ADV's output when RACE captured one this step.
    pub fn set_prev_action(&mut self, action: &Action, v: &View) {
        if let Some((_, st)) = self.players.iter_mut().find(|(p, _)| *p == v.obs.player) {
            if st.step == v.step && st.prev_sold.is_some() {
                st.prev_sold = Some(action.market.iter().filter(|o| o.is_sell3() && ITEMS.contains(&o.s(1))).map(|o| o.s(1)).collect());
            }
        }
    }

    /// Clone signal for a profile controller: the RACE clone gate is (or would be) on — 4 of the
    /// last 6 hand-position checks equal to the rival's and layout similarity >= .95 — or the
    /// step-1 cash mirror fired.
    pub fn clone_signal(&self, v: &View) -> bool {
        let Some((_, st)) = self.players.iter().find(|(p, _)| *p == v.obs.player) else { return false };
        let mirror = st.level >= HORIZON_MIRROR && st.step >= 1;
        let pos = st.hist.len() >= 4 && st.hist.iter().filter(|b| **b).count() >= 4;
        mirror || (pos && similarity(v) >= 0.95)
    }

    /// The player's race level (clone / escalated / mirror horizon; 0 before the first step).
    pub fn level(&self, player: i64) -> i64 {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s.level).unwrap_or(0)
    }

    /// Is the clone gate on for this player (`_RACE_STATE[p]['horizon'] > 0`)? (V44Y gate)
    pub fn horizon(&self, player: i64) -> i64 {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s.horizon).unwrap_or(0)
    }
}

/// `_race_clone`.
fn clone(v: &View, st: &mut St) -> bool {
    let (own, rival) = (v.farm(), v.rival());
    if !own.hands.is_empty() {
        st.hist.push(!own.hands.is_empty() && own.hands == rival.hands && own.farmer == rival.farmer);
        if st.hist.len() > 6 {
            st.hist.remove(0);
        }
    }
    st.hist.len() >= 4 && st.hist.iter().filter(|b| **b).count() >= 4 && similarity(v) >= 0.95
}

/// `_race_lost`: the rival sold a race product last turn well ahead of our own tape sale.
fn lost(v: &View, st: &St, ch: &Chassis) -> bool {
    let (Some(prev), Some(sold)) = (st.prev.as_ref(), st.prev_sold.as_ref()) else { return false };
    let step = v.step;
    if step != prev.step + 1 || step % 24 == 0 {
        return false;
    }
    let mut town: Qty = vec![];
    let t0 = step - 1;
    if t0 % 4 == 0 {
        for s in &prev.shops {
            let items = shop_items(s);
            for it in items {
                crate::obs::qadd(&mut town, it, if items.len() == 1 { 2 } else { 1 });
            }
        }
    }
    if t0 % 24 == 0 {
        for it in ITEMS {
            crate::obs::qadd(&mut town, it, 1);
        }
    }
    let Some(route) = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route) else { return false };
    let planned = |t: i64, item: &str| -> bool {
        let r = if t >= 648 { 2 } else { route };
        ch.route(r).tape.get(t as usize).is_some_and(|a| a.market.iter().any(|o| o.is_sell3() && o.s(1) == item))
    };
    for item in ITEMS {
        if qget(&prev.shed, item) <= 0 || sold.contains(&item) || qget(&prev.prices, item) <= 1 {
            continue;
        }
        let rival = qget(&v.obs.mkt_inventory, item) - qget(&prev.inventory, item) + qget(&town, item);
        if rival <= 0 {
            continue;
        }
        if ((step - 1)..719.min(step + 5)).any(|t| planned(t, item)) {
            continue;
        }
        if ((step + 5)..719.min(step + 24)).any(|t| planned(t, item)) {
            return true;
        }
    }
    false
}

// ---- R127 ------------------------------------------------------------------------------------
/// `_r127_fields`: our farm + private after this step's unit actions (over-subscribed PLANTs skipped).
pub fn fields(v: &View, action: &Action) -> (kagg_engine::state::Farm, kagg_engine::state::Private) {
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let commands = action.units();
    let mut demand: Qty = vec![];
    for c in &commands {
        if c.len() > 1 && c.op() == "PLANT" {
            crate::obs::qadd(&mut demand, c.s(1), 1);
        }
    }
    let day = v.step.div_euclid(24);
    for (actor, c) in commands.iter().enumerate().take(v.obs.invs.len()) {
        if c.len() > 1 && c.op() == "PLANT" && demand.iter().any(|(k, n)| *k == c.s(1) && *n > qget(&v.obs.seeds, k)) {
            continue;
        }
        sim::apply(&mut farm, &mut private, actor, c, day);
    }
    (farm, private)
}

pub fn to_qty(m: &kagg_engine::state::OMap) -> Qty {
    m.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect()
}

fn last_hour(action: Action, v: &View) -> Action {
    if v.step.rem_euclid(24) != 23 {
        return action;
    }
    let mut commands = action.units();
    if !commands.iter().any(|c| !c.is_empty() && c.op() == "PLANT") {
        return action;
    }
    let day = v.step.div_euclid(24);
    let mut result = action.clone();
    let mut changed = false;
    for _ in 0..=commands.len() {
        let (farm, _) = fields(v, &result);
        let mut drop = vec![];
        for (actor, c) in commands.iter().enumerate() {
            if c.is_empty() || c.op() != "PLANT" {
                continue;
            }
            let ok = v.positions.get(actor).is_some_and(|&(x, y)| {
                matches!(&farm.tiles[y as usize][x as usize],
                    kagg_engine::state::Cell::Plant { crop, planted_day, watered_today, .. }
                    if *crop == c.s(1) && *planted_day == day && *watered_today)
            });
            if !ok {
                drop.push(actor);
            }
        }
        if drop.is_empty() {
            break;
        }
        for a in drop {
            commands[a] = Cmd::pass();
        }
        result.set_units(commands.clone());
        changed = true;
    }
    if changed {
        result
    } else {
        action
    }
}

fn priority(action: Action, v: &View, ch: &Chassis) -> Action {
    let step = v.step;
    if !(144..695).contains(&step) {
        return action;
    }
    let orders = &action.market;
    let is = |o: &Cmd, op: &str, it: &str| o.len() >= 2 && o.op() == op && o.s(1) == it;
    if orders.len() > 9 || orders.iter().any(|o| is(o, "BUY_PRODUCT", "WHEAT") || is(o, "SELL", "WHEAT")) {
        return action;
    }
    let Some(route) = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route) else { return action };
    let r = if step + 1 >= 648 { 2 } else { route };
    let Some(future) = ch.route(r).tape.get((step + 1) as usize) else { return action };
    let commands = future.units();
    if !commands.iter().any(|c| is(c, "PICKUP", "WHEAT")) {
        return action;
    }
    if r85::budget(v, orders) != Some(true) {
        return action;
    }
    let (farm, private) = fields(v, &action);
    let night = step.rem_euclid(24) == 23;
    let mut positions: Vec<(i64, i64)> = vec![farm.farmer];
    positions.extend(farm.hands.iter().copied());
    if night {
        positions = vec![(4, 4)];
    } else {
        for o in orders {
            if !o.is_empty() && o.op() == "HIRE" {
                let mut best = ACCESS[0];
                let mut key = (usize::MAX, usize::MAX);
                for (ai, p) in ACCESS.iter().enumerate() {
                    let k = (positions.iter().filter(|q| *q == p).count(), ai);
                    if k < key {
                        key = k;
                        best = *p;
                    }
                }
                positions.push(best);
            }
        }
    }
    let need: i64 = positions
        .iter()
        .zip(commands.iter())
        .filter(|(p, c)| ACCESS.contains(p) && is(c, "PICKUP", "WHEAT"))
        .map(|(_, c)| if c.len() > 2 { c.n(2) } else { 1 }.max(0))
        .sum();
    let shed = to_qty(&private.shed);
    let invs: Vec<Qty> = private.inventories.iter().map(to_qty).collect();
    let (stock, buys, _) = r85::market_stock(&shed, orders);
    let (before, loss) = r85::delivery(&stock, &invs, night);
    let shortage = (need - qget(&before, "WHEAT")).max(0);
    if shortage == 0 || shortage > 100 - shed.iter().map(|(_, n)| n).sum::<i64>() {
        return action;
    }
    let inv = qget(&v.obs.mkt_inventory, "WHEAT");
    let cost: i64 = (1..=shortage).map(|j| market::price("WHEAT", (inv - (2 * j - 1)) as f64)).sum();
    if r85::budget_with(v, orders, v.farm().money - cost as f64) != Some(true) {
        return action;
    }
    let mut proposed = vec![Cmd::order("BUY_PRODUCT", "WHEAT", shortage)];
    proposed.extend(orders.iter().cloned());
    let (stock2, after_buys, _) = r85::market_stock(&shed, &proposed);
    let (after, after_loss) = r85::delivery(&stock2, &invs, night);
    let buy_at = |i: usize| after_buys.iter().find(|(j, _)| *j == i).map(|(_, q)| *q).unwrap_or(0);
    if qget(&after, "WHEAT") < need
        || buys.iter().any(|(i, q)| buy_at(i + 1) < *q)
        || after_loss.iter().any(|(p, q)| *q > qget(&loss, p))
    {
        return action;
    }
    let mut r = action;
    r.market = proposed;
    r
}

impl Race {
    pub fn r127(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let _ = &self.r127;
        let a = last_hour(action, v);
        priority(a, v, ch)
    }
}
