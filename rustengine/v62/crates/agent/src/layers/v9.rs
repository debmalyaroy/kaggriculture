//! The v9 group (agent 2894-3870): COURIER, CARROT, HERD (wrapper is a no-op), FERT, OPENING,
//! RACE (+ its `_V9_ITEM_HZ` horizon), RACEPX / RACEGATE wrappers (their logic lives in the
//! chassis patch and in `r36`), CTRTABLE and OVERFLOW.
use super::r85;
use crate::act::{Action, Cmd, Tok};
use crate::chassis::Chassis;
use crate::obs::{qadd, qget, Qty};
use crate::router::round3;
use crate::sim;
use crate::view::View;

const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
type P = (i64, i64);

fn native_route(ch: &Chassis, player: i64) -> Option<(i64, &crate::chassis::PlayerState)> {
    let s = ch.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s)?;
    let r = s.route?;
    ch.route_idx(r)?;
    Some((r, s))
}

fn step_state(v: &mut Vec<(i64, i64)>, player: i64, step: i64) -> bool {
    // returns true when the state was (re)created
    match v.iter_mut().find(|(p, _)| *p == player) {
        Some((_, s)) if step > *s => {
            *s = step;
            false
        }
        Some((_, s)) => {
            *s = step;
            true
        }
        None => {
            v.push((player, step));
            true
        }
    }
}

// ---- COURIER ---------------------------------------------------------------------------------
const COURIER_ITEMS: [&str; 4] = ["STRAWBERRY", "MILK", "WOOL", "MELON"];
const IDLE: [&str; 6] = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP"];

#[derive(Default, Clone)]
struct CourierSt {
    day: Option<i64>,
    /// unit -> (route, start)
    plans: Vec<(usize, Vec<Cmd>, i64)>,
}

fn courier_plan(tape: &[Action], unit: usize, pos: P, commands: &[Cmd], step: i64, end: i64) -> Option<Vec<Cmd>> {
    let c = &commands[unit];
    if !c.is_empty() && !IDLE.contains(&c.op()) {
        return None;
    }
    for t in (step + 1)..=end {
        let units = tape.get(t as usize).map(|a| a.units()).unwrap_or_else(|| vec![Cmd::pass()]);
        let command = units.get(unit).cloned().unwrap_or_else(Cmd::pass);
        if !command.is_empty() && !IDLE.contains(&command.op()) {
            return None;
        }
    }
    let mut target = ACCESS[0];
    for a in ACCESS {
        if (a.0 - pos.0).abs() + (a.1 - pos.1).abs() < (target.0 - pos.0).abs() + (target.1 - pos.1).abs() {
            target = a;
        }
    }
    let mut walk: Vec<Cmd> = vec![];
    walk.extend((0..(target.0 - pos.0).max(0)).map(|_| Cmd::new("EAST")));
    walk.extend((0..(pos.0 - target.0).max(0)).map(|_| Cmd::new("WEST")));
    walk.extend((0..(target.1 - pos.1).max(0)).map(|_| Cmd::new("SOUTH")));
    walk.extend((0..(pos.1 - target.1).max(0)).map(|_| Cmd::new("NORTH")));
    if walk.len() as i64 <= end - step {
        walk.push(Cmd::new("DROP"));
        Some(walk)
    } else {
        None
    }
}

// ---- CARROT ----------------------------------------------------------------------------------
#[derive(Default, Clone)]
struct CarrotSt {
    tiles: Vec<(P, i64)>,
    swapped: bool,
}

// ---- RACE ------------------------------------------------------------------------------------
pub const RACE_DEFAULT: i64 = 44;
pub const RACE_MAX: i64 = 48;
pub const RACE_MARGIN: i64 = 12;
pub const RACE_GAP: i64 = 3;
pub const RACE_WINDOW: i64 = 30;
pub const RACE_ITEMS: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];

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

pub(crate) fn town_draw(shops: &[&str], step: i64) -> Qty {
    let mut draw: Qty = RACE_ITEMS.iter().map(|i| (*i, 0)).collect();
    if step % 4 == 0 {
        for s in shops {
            let items = shop_items(s);
            for item in items {
                if draw.iter().any(|(k, _)| k == item) {
                    qadd(&mut draw, item, if items.len() == 1 { 2 } else { 1 });
                }
            }
        }
    }
    if step % 24 == 0 {
        for e in draw.iter_mut() {
            e.1 += 1;
        }
    }
    draw
}

fn planned_sells(tape: &[Action], item: &str, step: i64) -> Vec<i64> {
    let lo = (step - RACE_WINDOW).max(0);
    let hi = (tape.len() as i64).min(step + RACE_WINDOW + 1);
    (lo..hi)
        .filter(|&t| tape[t as usize].market.iter().any(|o| o.is_sell3() && o.s(1) == item && o.n(2) > 0))
        .collect()
}

#[derive(Clone)]
pub(crate) struct Prev {
    pub(crate) step: i64,
    pub(crate) inventory: Qty,
    pub(crate) prices: Qty,
    pub(crate) own: Qty,
    left: Qty,
    pub(crate) shops: Vec<&'static str>,
}
#[derive(Clone)]
struct RaceSt {
    step: i64,
    lead: i64,
    prev: Option<Prev>,
}

#[derive(Default)]
pub struct V9 {
    courier_step: Vec<(i64, i64)>,
    courier: Vec<(i64, CourierSt)>,
    carrot_step: Vec<(i64, i64)>,
    carrot: Vec<(i64, CarrotSt)>,
    race: Vec<(i64, RaceSt)>,
    ct_step: Vec<(i64, i64)>,
    ct: Vec<(i64, Option<&'static [(i64, usize, (&'static str, &'static str, i64))]>)>,
    /// PIPE's install empties `CT_TABLE`.
    pub ct_disabled: bool,
}

impl V9 {
    // ---- COURIER -----------------------------------------------------------------------------
    pub fn courier_pre(&mut self, v: &View) {
        let p = v.obs.player;
        if step_state(&mut self.courier_step, p, v.step) {
            match self.courier.iter_mut().find(|(k, _)| *k == p) {
                Some((_, s)) => *s = CourierSt::default(),
                None => self.courier.push((p, CourierSt::default())),
            }
        }
    }

    pub fn courier(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let player = v.obs.player;
        if step >= 718 || step.rem_euclid(24) < 12 {
            return action;
        }
        let Some(i) = self.courier.iter().position(|(k, _)| *k == player) else { return action };
        let day = step.div_euclid(24);
        let st = &mut self.courier[i].1;
        if st.day != Some(day) {
            st.day = Some(day);
            st.plans.clear();
        }
        let Some((route, native)) = native_route(ch, player) else { return action };
        let tape = &ch.route(route).tape;
        let positions = &v.positions;
        let invs = &v.obs.invs;
        let mut commands = action.units();
        while commands.len() < positions.len() {
            commands.push(Cmd::pass());
        }
        let end = day * 24 + 23;
        let lo = (day * 24) as usize;
        let hi = ((end + 1) as usize).min(tape.len());
        let crew = 1 + tape[lo.min(hi)..hi].iter().map(|a| a.hands.len()).max().unwrap_or(0);
        let mut delivered: Qty = vec![];
        let mut changed = false;
        for (unit, &pos) in positions.iter().enumerate().take(crew) {
            static EMPTY: Qty = Vec::new();
            let inventory = invs.get(unit).unwrap_or(&EMPTY);
            let cargo: Qty = inventory.iter().filter(|(k, v)| COURIER_ITEMS.contains(k) && *v > 0).cloned().collect();
            let pi = st.plans.iter().position(|(u, _, _)| *u == unit);
            let pi = match pi {
                Some(pi) => pi,
                None => {
                    let pend = native.pending.iter().any(|(k, q)| *k == unit && !q.is_empty());
                    if cargo.is_empty() || pend {
                        continue;
                    }
                    let Some(r) = courier_plan(tape, unit, pos, &commands, step, end) else { continue };
                    st.plans.push((unit, r, step));
                    st.plans.len() - 1
                }
            };
            let (_, route_cmds, start) = &st.plans[pi];
            let index = step - start;
            if index < 0 || index as usize >= route_cmds.len() {
                continue;
            }
            let command = route_cmds[index as usize].clone();
            if command.len() == 1 && command.op() == "DROP" {
                if !ACCESS.contains(&pos) {
                    st.plans[pi] = (unit, vec![], step);
                    continue;
                }
                for (item, n) in &cargo {
                    qadd(&mut delivered, item, *n);
                }
            }
            commands[unit] = command;
            changed = true;
        }
        if !changed {
            return action;
        }
        let mut result = action;
        let market = result.market.clone();
        result.set_units(commands);
        if !delivered.is_empty() {
            let mut market = market;
            let mut order = delivered.clone();
            order.sort_by_key(|(k, n)| -v.price(k) * n);
            for (item, n) in order {
                if v.price(item) < 2 {
                    continue;
                }
                if let Some(ei) = market.iter().position(|o| o.is_sell3() && o.s(1) == item) {
                    let mut e = market.remove(ei);
                    let nv = e.n(2) + n;
                    e.set_n(2, nv);
                    market.insert(0, e);
                } else if market.len() < 10 {
                    market.insert(0, Cmd::order("SELL", item, n));
                }
            }
            result.market = market;
        }
        result
    }

    // ---- CARROT ------------------------------------------------------------------------------
    pub fn carrot_pre(&mut self, v: &View) {
        let p = v.obs.player;
        if step_state(&mut self.carrot_step, p, v.step) {
            match self.carrot.iter_mut().find(|(k, _)| *k == p) {
                Some((_, s)) => *s = CarrotSt::default(),
                None => self.carrot.push((p, CarrotSt::default())),
            }
        }
    }

    pub fn carrot(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let Some(i) = self.carrot.iter().position(|(k, _)| *k == v.obs.player) else { return action };
        let st = &mut self.carrot[i].1;
        let step = v.step;
        let day = step.div_euclid(24);
        let mut commands = action.units();
        let mut market = action.market.clone();
        let mut changed = false;
        let farm = v.farm();
        let pos = &v.positions;
        if !st.tiles.is_empty() {
            for (i, c) in commands.iter_mut().enumerate().take(pos.len()) {
                if !(c.len() == 1 && c.op() == "WATER") {
                    continue;
                }
                let p = pos[i];
                if st.tiles.iter().find(|(k, _)| *k == p).map(|(_, d)| *d) != Some(day - 3) {
                    continue;
                }
                let t = farm.tile(p.0, p.1);
                if t.is_dict() && t.crop == "CARROT" && t.planted_day == day - 3 && t.yield_units > 0 {
                    *c = Cmd::new("HARVEST");
                    changed = true;
                }
            }
        }
        let wheat_held = qget(&v.obs.shed, "WHEAT") + v.obs.invs.iter().map(|m| qget(m, "WHEAT")).sum::<i64>();
        let pw = if v.obs.prices.iter().any(|(k, _)| *k == "WHEAT") { v.price("WHEAT") } else { 99 };
        let ratio = v.price("CARROT") as f64 / pw.max(1) as f64;
        let boom = ratio >= 4.5;
        let reserve = if boom { 10 } else { 40 };
        if boom && (10..=23).contains(&day) && wheat_held < 40 {
            let topup = 40 - wheat_held;
            let budget = farm.money - 1500.0;
            let qty = topup.min((budget / (pw + 5).max(1) as f64).floor() as i64);
            if qty > 0 && market.len() < 10 && !market.iter().any(|o| o.len() >= 2 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT") {
                market.push(Cmd::order("BUY_PRODUCT", "WHEAT", qty));
                changed = true;
            }
        }
        if (10..=23).contains(&day) && wheat_held >= reserve && ratio >= 2.0 {
            let mut seeds = qget(&v.obs.seeds, "CARROT")
                - commands.iter().filter(|c| c.len() >= 2 && c.op() == "PLANT" && c.s(1) == "CARROT").count() as i64;
            for (i, c) in commands.iter_mut().enumerate() {
                if c.len() >= 2 && c.op() == "PLANT" && c.s(1) == "WHEAT" && seeds > 0 {
                    c.0[1] = Tok::S("CARROT");
                    seeds -= 1;
                    changed = true;
                    st.swapped = true;
                    if i < pos.len() {
                        let p = pos[i];
                        match st.tiles.iter_mut().find(|(k, _)| *k == p) {
                            Some(e) => e.1 = day,
                            None => st.tiles.push((p, day)),
                        }
                    }
                }
            }
            for o in market.iter_mut() {
                if o.len() >= 3 && o.op() == "BUY_SEED" && o.s(1) == "WHEAT" {
                    o.0[1] = Tok::S("CARROT");
                    changed = true;
                    st.swapped = true;
                }
            }
        }
        let mut result = action.clone();
        result.set_units(commands);
        result.market = market;
        if st.swapped && day < 24 {
            let stock = qget(&ch.projected_shed(&result, v), "CARROT");
            let selling: i64 = result.market.iter().filter(|o| o.is_sell3() && o.s(1) == "CARROT").map(|o| o.n(2)).sum();
            if stock > selling && result.market.len() < 10 && v.price("CARROT") >= 2 {
                result.market.insert(0, Cmd::order("SELL", "CARROT", stock - selling));
                changed = true;
            }
        }
        if changed {
            result
        } else {
            action
        }
    }

    // ---- FERT --------------------------------------------------------------------------------
    pub fn fert(action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let day = step.div_euclid(24);
        if day < 14 || step >= 700 {
            return action;
        }
        let farm = v.farm();
        let mut commands = action.units();
        let mut planned: Vec<(usize, i64)> = vec![];
        if let Some((route, _)) = native_route(ch, v.obs.player) {
            let tape = &ch.route(route).tape;
            for t in step..(tape.len() as i64).min(day * 24 + 24) {
                for (u, c) in tape[t as usize].units().iter().enumerate() {
                    if !c.is_empty() && c.op() == "FERTILIZE" {
                        match planned.iter_mut().find(|(k, _)| *k == u) {
                            Some(e) => e.1 += 1,
                            None => planned.push((u, 1)),
                        }
                    }
                }
            }
        }
        let mut carried: Vec<(usize, i64)> = vec![];
        let mut targeted: Vec<P> = vec![];
        let mut changed = false;
        let n = commands.len().min(v.positions.len());
        for unit in 0..n {
            let c = &commands[unit];
            if c.is_empty() || c.op() != "WATER" {
                continue;
            }
            let (x, y) = v.positions[unit];
            let t = farm.tile(x, y);
            if !(t.is_dict() && t.kind == "PLANT" && matches!(t.crop, "WHEAT" | "CARROT")) {
                continue;
            }
            if day - t.planted_day != 1 || t.watered_today {
                continue;
            }
            if t.consecutive_unwatered != 0 || t.fertilized_until_day >= day {
                continue;
            }
            let have = match carried.iter().find(|(k, _)| *k == unit) {
                Some((_, h)) => *h,
                None => {
                    let h = qget(v.inv(unit), "FERTILIZER") - planned.iter().find(|(k, _)| *k == unit).map(|(_, n)| *n).unwrap_or(0);
                    carried.push((unit, h));
                    h
                }
            };
            if have <= 0 || targeted.contains(&(x, y)) {
                continue;
            }
            commands[unit] = Cmd::new("FERTILIZE");
            carried.iter_mut().find(|(k, _)| *k == unit).unwrap().1 = have - 1;
            targeted.push((x, y));
            changed = true;
        }
        if !changed {
            return action;
        }
        let mut r = action;
        r.set_units(commands);
        r
    }

    // ---- OPENING -----------------------------------------------------------------------------
    pub fn opening(action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        if step > 1 || native_route(ch, v.obs.player).is_none() {
            return action;
        }
        let wheat: Vec<Cmd> = action
            .market
            .iter()
            .filter(|o| o.len() >= 3 && matches!(o.op(), "BUY_PRODUCT" | "SELL") && o.s(1) == "WHEAT")
            .cloned()
            .collect();
        let sig: Vec<(&str, &str, i64)> = wheat.iter().map(|o| (o.op(), o.s(1), o.n(2))).collect();
        let tape_sig: &[(&str, &str, i64)] = if step == 0 {
            &[("BUY_PRODUCT", "WHEAT", 13), ("BUY_PRODUCT", "WHEAT", 30), ("SELL", "WHEAT", 30)]
        } else {
            &[("SELL", "WHEAT", 13), ("BUY_PRODUCT", "WHEAT", 5)]
        };
        if sig != tape_sig {
            return action;
        }
        let rest: Vec<Cmd> = action.market.iter().filter(|o| !wheat.contains(o)).cloned().collect();
        let mut r = action;
        r.market = if step == 0 { vec![Cmd::order("BUY_PRODUCT", "WHEAT", 20), Cmd::order("SELL", "WHEAT", 15)] } else { vec![] };
        r.market.extend(rest);
        r
    }

    // ---- RACE --------------------------------------------------------------------------------
    /// Pre phase: returns `_V9_ITEM_HZ[player]`.
    pub fn race_pre(&mut self, v: &View, ch: &Chassis, k: &super::knobs::Knobs) -> Vec<(&'static str, i64)> {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.race.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.race[i].1.step => i,
            Some(i) => {
                self.race[i].1 = RaceSt { step: -1, lead: -k.v9_race_margin, prev: None };
                i
            }
            None => {
                self.race.push((player, RaceSt { step: -1, lead: -k.v9_race_margin, prev: None }));
                self.race.len() - 1
            }
        };
        let st = &mut self.race[i].1;
        st.step = step;
        // _v9_race_update
        if let Some(prev) = st.prev.as_ref().filter(|p| p.step == step - 1) {
            if let Some((route, _)) = native_route(ch, player) {
                let tape = &ch.route(route).tape;
                let draw = town_draw(&prev.shops, prev.step);
                let t = prev.step;
                for item in RACE_ITEMS {
                    if qget(&prev.prices, item) <= 3 {
                        continue;
                    }
                    let sold = qget(&v.obs.mkt_inventory, item) - qget(&prev.inventory, item) + qget(&draw, item) - qget(&prev.own, item);
                    if sold < 2 {
                        continue;
                    }
                    if qget(&prev.left, item) <= 0 {
                        continue;
                    }
                    let planned = planned_sells(tape, item, t);
                    let after: Vec<i64> = planned.iter().copied().filter(|s| *s >= t).collect();
                    let before: Vec<i64> = planned.iter().copied().filter(|s| *s < t).collect();
                    if after.is_empty() || before.last().is_some_and(|b| t - b < RACE_GAP) {
                        continue;
                    }
                    st.lead = st.lead.max(after[0] - t);
                }
            }
        }
        let horizon = k.v9_race_max.min(k.v9_race_default.max(st.lead + k.v9_race_margin));
        RACE_ITEMS.iter().map(|i| (*i, horizon)).collect()
    }

    /// RACE's previous-turn snapshot for `player` (read by v92).
    pub(crate) fn race_prev(&self, player: i64) -> Option<&Prev> {
        self.race.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.prev.as_ref())
    }

    pub fn race_post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let Some(i) = self.race.iter().position(|(p, _)| *p == v.obs.player) else { return action };
        let stock = ch.projected_shed(&action, v);
        let mut own: Qty = vec![];
        for o in &action.market {
            if o.is_sell3() && RACE_ITEMS.contains(&o.s(1)) {
                let item = o.s(1);
                let n = o.n(2).max(0).min((qget(&stock, item) - qget(&own, item)).max(0));
                qadd(&mut own, item, n);
            }
        }
        let left: Qty = RACE_ITEMS.iter().map(|it| (*it, qget(&stock, it) - qget(&own, it))).collect();
        self.race[i].1.prev = Some(Prev {
            step: v.step,
            inventory: v.obs.mkt_inventory.clone(),
            prices: v.obs.prices.clone(),
            own,
            left,
            shops: v.obs.shops.clone(),
        });
        action
    }

    // ---- CTRTABLE ----------------------------------------------------------------------------
    pub fn ctrtable(&mut self, action: Action, v: &View) -> Action {
        type Row = (i64, usize, (&'static str, &'static str, i64));
        static AGI: [Row; 4] = [
            (7, 0, ("BUY_PRODUCT", "WHEAT", 5)),
            (7, 1, ("SELL", "WHEAT", 5)),
            (8, 0, ("BUY_PRODUCT", "WHEAT", 5)),
            (8, 1, ("SELL", "WHEAT", 5)),
        ];
        static GOOSE: [Row; 2] = [(3, 0, ("BUY_PRODUCT", "WHEAT", 20)), (3, 1, ("SELL", "WHEAT", 20))];
        let (player, step) = (v.obs.player, v.step);
        if step_state(&mut self.ct_step, player, step) {
            match self.ct.iter_mut().find(|(k, _)| *k == player) {
                Some(e) => e.1 = None,
                None => self.ct.push((player, None)),
            }
        }
        let i = self.ct.iter().position(|(k, _)| *k == player).unwrap();
        if step == 2 && !self.ct_disabled {
            let key = (round3(v.rival().money), qget(&v.obs.mkt_inventory, "WHEAT"));
            self.ct[i].1 = if key == (979.0, 9989) {
                Some(&AGI[..])
            } else if key == (33.0, 9990) {
                Some(&GOOSE[..])
            } else {
                None
            };
        }
        let Some(plan) = self.ct[i].1 else { return action };
        let mut items: Vec<(usize, Cmd)> =
            plan.iter().filter(|(t, _, _)| *t == step).map(|(_, s, o)| (*s, Cmd::order(o.0, o.1, o.2))).collect();
        if items.is_empty() {
            return action;
        }
        items.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| format!("{:?}", a.1).cmp(&format!("{:?}", b.1))));
        let mut market = action.market.clone();
        for (slot, o) in items {
            while market.len() < slot {
                market.push(Cmd::order("SELL", "WHEAT", 0));
            }
            market.insert(slot.min(market.len()), o);
        }
        market.truncate(10);
        let mut r = action;
        r.market = market;
        r
    }
}

// ---- OVERFLOW --------------------------------------------------------------------------------
/// `_ov_apply` (hour 23): sell exactly the shed stock that cargo destroyed at midnight would replace.
pub fn overflow(action: Action, v: &View) -> Action {
    if v.step.rem_euclid(24) != 23 {
        return action;
    }
    let orders = &action.market;
    if orders.len() >= 10 || r85::budget(v, orders) != Some(true) {
        return action;
    }
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let commands = action.units();
    let mut demand: Qty = vec![];
    for c in &commands {
        if c.len() > 1 && c.op() == "PLANT" {
            qadd(&mut demand, c.s(1), 1);
        }
    }
    let day = v.step.div_euclid(24);
    for (actor, c) in commands.iter().enumerate().take(v.obs.invs.len()) {
        if c.len() > 1 && c.op() == "PLANT" && demand.iter().any(|(k, n)| *k == c.s(1) && *n > qget(&v.obs.seeds, k)) {
            continue;
        }
        sim::apply(&mut farm, &mut private, actor, c, day);
    }
    let shed: Qty = private.shed.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect();
    let invs: Vec<Qty> = private.inventories.iter().map(|m| m.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect()).collect();
    let (stock, _, _) = r85::market_stock(&shed, orders);
    let (original, loss) = r85::delivery(&stock, &invs, true);
    if loss.is_empty() {
        return action;
    }
    let mut remaining = (100 - stock.iter().map(|(_, n)| n).sum::<i64>()).max(0);
    let mut tail: Vec<&'static str> = vec![];
    for bag in &invs {
        for (item, n) in bag {
            let n = (*n).max(0);
            let take = n.min(remaining);
            remaining -= take;
            for _ in 0..(n - take) {
                tail.push(item);
            }
        }
    }
    let mut released: Qty = vec![];
    let mut best: Option<Vec<Cmd>> = None;
    for item in tail {
        qadd(&mut released, item, 1);
        if !v.obs.prices.iter().any(|(k, _)| *k == item) || qget(&released, item) > qget(&stock, item) {
            break;
        }
        if orders.len() + released.len() > 10 {
            break;
        }
        let mut proposed = orders.clone();
        proposed.extend(released.iter().map(|(p, n)| Cmd::order("SELL", p, *n)));
        let (after, _, _) = r85::market_stock(&shed, &proposed);
        let (fin, _) = r85::delivery(&after, &invs, true);
        let same = original.iter().chain(fin.iter()).all(|(p, _)| qget(&original, p) == qget(&fin, p));
        if same {
            best = Some(proposed);
        }
    }
    match best {
        Some(m) => {
            let mut r = action;
            r.market = m;
            r
        }
        None => action,
    }
}
