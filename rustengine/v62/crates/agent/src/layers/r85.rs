//! R85 economic overlay (agent 2537-2630: discretionary feed skip, surplus fertilizer sale,
//! second hour-23 warehouse pass) with `_r86_next_feed` (2635) and `_r88_feed_bonus_cost`
//! (2673); R95 wheat-replenishment trim (2695-2761); R97 wheat-supply guard (2763-2891).
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qadd, qget, qset, Qty, Tile};
use crate::sim;
use crate::view::{animal_cost, fib, move_delta, seed_price, View};

const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
type P = (i64, i64);

fn route_of(ch: &Chassis, player: i64) -> Option<i64> {
    ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route)
}
fn tape_at(ch: &Chassis, route: i64, t: i64) -> Option<&Action> {
    ch.route(if t >= 648 { 2 } else { route }).tape.get(t as usize)
}
fn qty_or1(c: &Cmd) -> i64 {
    if c.len() > 2 { c.n(2) } else { 1 }.max(0)
}
fn is(c: &Cmd, op: &str, item: &str) -> bool {
    c.len() >= 2 && c.op() == op && c.s(1) == item
}

// ---- R88 / R86 ---------------------------------------------------------------------------
/// `_r88_feed_bonus_cost`.
fn feed_bonus_cost(t: &Tile, day: i64) -> i64 {
    let (first, interval) = match t.animal {
        "GOOSE" => (4, 1),
        "COW" => (8, 2),
        _ => (6, 3),
    };
    let first = first + t.placed_day;
    let tomorrow = day + 1;
    let produces = tomorrow >= first && (tomorrow - first).rem_euclid(interval) == 0;
    let mut pending = t.pending_care_bonus.max(0);
    if !produces {
        pending = 0;
    }
    let mut care = 1;
    let mut next_use = first;
    if next_use <= tomorrow {
        next_use += ((tomorrow - next_use).div_euclid(interval) + 1) * interval;
    }
    if next_use > 29 {
        care = 0;
    }
    pending + care
}

#[derive(Default)]
pub struct R85 {
    /// `_R86_FEED_CACHE`: (route, day) -> tiles fed by hour 21 tomorrow.
    feed_cache: Vec<((i64, i64), Vec<P>)>,
    /// `native_reserves`: route -> backward fertilizer reserve (per game).
    reserves: Vec<(i64, i64, Vec<i64>)>,
    last_step: Vec<(i64, i64)>,
    /// `_R95_RESERVES`: (route, step) -> tape wheat demand.
    r95: Vec<((i64, i64), i64)>,
}

impl R85 {
    /// `_r86_next_feed`: will tomorrow's tape feed `target` by hour 21?
    fn next_feed(&mut self, ch: &Chassis, route: i64, day: i64, target: P) -> bool {
        if day == 28 {
            return true;
        }
        let tomorrow = day + 1;
        let r = if tomorrow >= 27 { 2 } else { route };
        let key = (r, tomorrow);
        if !self.feed_cache.iter().any(|(k, _)| *k == key) {
            let mut positions: Vec<P> = vec![(4, 4)];
            let mut wheat: Vec<i64> = vec![0];
            let mut feeds: Vec<P> = vec![];
            for hour in 0..24 {
                let a = &ch.route(r).tape[(tomorrow * 24 + hour) as usize];
                let cmds = a.units();
                let n = cmds.len().min(positions.len());
                for actor in 0..n {
                    let c = &cmds[actor];
                    if c.is_empty() {
                        continue;
                    }
                    let pos = positions[actor];
                    if let Some((dx, dy)) = move_delta(c.op()) {
                        positions[actor] = ((pos.0 + dx).clamp(0, 9), (pos.1 + dy).clamp(0, 9));
                    } else if is(c, "PICKUP", "WHEAT") && ACCESS.contains(&pos) {
                        wheat[actor] += qty_or1(c);
                    } else if c.op() == "FEED" && wheat[actor] > 0 {
                        wheat[actor] -= 1;
                        if hour <= 21 && !feeds.contains(&pos) {
                            feeds.push(pos);
                        }
                    } else if c.op() == "DROP" && ACCESS.contains(&pos) {
                        wheat[actor] = 0;
                    } else if is(c, "PLACE", "WHEAT") && ACCESS.contains(&pos) {
                        wheat[actor] = (wheat[actor] - qty_or1(c)).max(0);
                    }
                }
                for o in &a.market {
                    if !o.is_empty() && o.op() == "HIRE" {
                        let mut chosen = ACCESS[0];
                        let mut key = (usize::MAX, usize::MAX);
                        for (ai, p) in ACCESS.iter().enumerate() {
                            let k = (positions.iter().filter(|q| *q == p).count(), ai);
                            if k < key {
                                key = k;
                                chosen = *p;
                            }
                        }
                        positions.push(chosen);
                        wheat.push(0);
                    }
                }
            }
            self.feed_cache.push((key, feeds));
        }
        self.feed_cache.iter().find(|(k, _)| *k == key).unwrap().1.contains(&target)
    }

    /// `_r85_feed`: skip a FEED whose care bonus is worth less than the wheat.
    fn feed(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let day = step.div_euclid(24);
        if !(10..=28).contains(&day) || step.rem_euclid(24) > 21 {
            return action;
        }
        let Some(route) = route_of(ch, v.obs.player) else { return action };
        let tape = &ch.route(route).tape;
        let lo = ((day * 24) as usize).min(tape.len());
        let hi = (((day + 1) * 24).min(719) as usize).min(tape.len());
        let expected = tape[lo..hi].iter().map(|a| a.hands.len()).max().unwrap_or(0);
        let mut commands = action.units();
        let mut changed = false;
        let n = commands.len().min(expected + 1);
        for actor in 0..n {
            if !(commands[actor].len() == 1 && commands[actor].op() == "FEED") || actor >= v.positions.len() {
                continue;
            }
            let t = v.tile(v.positions[actor]);
            if !t.is_dict() || !matches!(t.animal, "GOOSE" | "COW" | "SHEEP") {
                continue;
            }
            if t.fed_today || t.consecutive_unfed != 0 {
                continue;
            }
            if qget(v.inv(actor), "WHEAT") <= 0 {
                continue;
            }
            let item = match t.animal {
                "GOOSE" => "EGG",
                "COW" => "MILK",
                _ => "WOOL",
            };
            let bonus = feed_bonus_cost(t, day);
            if bonus as f64 * (v.price(item) as f64 + 5.0) * 1.25 >= v.price("WHEAT") as f64 {
                continue;
            }
            if !self.next_feed(ch, route, day, v.positions[actor]) {
                continue;
            }
            commands[actor] = Cmd::pass();
            changed = true;
        }
        if !changed {
            return action;
        }
        let mut r = action;
        r.set_units(commands);
        r
    }

    /// `_r85_reserve`.
    fn reserve(&mut self, v: &View, ch: &Chassis, player: i64, route: i64, dedicated: i64) -> i64 {
        let step = v.step;
        if !self.reserves.iter().any(|(p, r, _)| *p == player && *r == route) {
            let mut reserve = vec![0i64; 720];
            for t in (0..=718).rev() {
                let a = tape_at(ch, route, t).unwrap();
                let pickup: i64 = a.units().iter().filter(|c| is(c, "PICKUP", "FERTILIZER")).map(qty_or1).sum();
                let purchase: i64 = a.market.iter().filter(|o| o.len() > 2 && is(o, "BUY_PRODUCT", "FERTILIZER")).map(|o| o.n(2).max(0)).sum();
                reserve[t as usize] = pickup + (reserve[t as usize + 1] - purchase).max(0);
            }
            self.reserves.push((player, route, reserve));
        }
        let r = &self.reserves.iter().find(|(p, r, _)| *p == player && *r == route).unwrap().2;
        14.max(r[(step + 1).min(719) as usize] + dedicated)
    }

    /// `_r85_fertilizer`: sell fertilizer beyond every scheduled and dedicated need.
    fn fertilizer(&mut self, action: Action, v: &View, ch: &Chassis, dedicated: i64) -> Action {
        let day = v.step.div_euclid(24);
        if !(6..=28).contains(&day) {
            return action;
        }
        if action.market.len() >= 10 || action.market.iter().any(|o| !o.is_empty() && o.op() != "SELL") {
            return action;
        }
        let stock = ch.projected_shed(&action, v);
        let held = qget(&stock, "FERTILIZER").max(0);
        let sold: i64 = action.market.iter().filter(|o| o.len() > 2 && is(o, "SELL", "FERTILIZER")).map(|o| o.n(2).max(0)).sum();
        let player = v.obs.player;
        let Some(route) = route_of(ch, player) else { return action };
        let extra = held - sold - self.reserve(v, ch, player, route, dedicated);
        if extra <= 0 {
            return action;
        }
        let mut r = action;
        r.market.push(Cmd::order("SELL", "FERTILIZER", extra));
        r
    }

    /// The R85 wrapper body (after its parent).
    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis, dedicated: i64, feed_on: bool) -> Action {
        let player = v.obs.player;
        match self.last_step.iter_mut().find(|(p, _)| *p == player) {
            Some((_, s)) if v.step > *s => *s = v.step,
            Some((_, s)) => {
                *s = v.step;
                self.reserves.retain(|(p, _, _)| *p != player);
            }
            None => self.last_step.push((player, v.step)),
        }
        let r = if feed_on { self.feed(action, v, ch) } else { action };
        let r = self.fertilizer(r, v, ch, dedicated);
        if v.step.rem_euclid(24) == 23 {
            return super::r51::warehouse(r, v, ch);
        }
        r
    }

    // ---- R95 ---------------------------------------------------------------------------
    fn r95_reserve(&mut self, v: &View, ch: &Chassis, route: i64) -> i64 {
        let step = v.step;
        let key = (route, step);
        if !self.r95.iter().any(|(k, _)| *k == key) {
            let mut demand = 6;
            for t in (step + 1)..719.min(step + 49) {
                let a = tape_at(ch, route, t).unwrap();
                demand += a.units().iter().filter(|c| is(c, "PICKUP", "WHEAT")).map(qty_or1).sum::<i64>();
                demand += a.market.iter().filter(|o| o.len() > 2 && is(o, "SELL", "WHEAT")).map(|o| o.n(2).max(0)).sum::<i64>();
            }
            self.r95.push((key, demand));
        }
        let mut demand = self.r95.iter().find(|(k, _)| *k == key).unwrap().1;
        if v.shops().iter().filter(|s| **s == "YARN_STORE").count() >= 2 {
            let mut days: Vec<i64> = ((step + 1)..(step + 49)).map(|t| t / 24).filter(|d| *d >= 12).collect();
            days.dedup();
            demand += 6 * days.len() as i64;
        }
        demand
    }

    /// `_r95_replenish`: on days 10-11 trim wheat buys to the two-day physical reserve.
    pub fn r95(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        if !(10..=11).contains(&step.div_euclid(24)) {
            return action;
        }
        if !action.market.iter().any(|o| o.len() > 2 && is(o, "BUY_PRODUCT", "WHEAT") && o.n(2) > 0) {
            return action;
        }
        if action.market.iter().any(|o| is(o, "SELL", "WHEAT")) {
            return action;
        }
        let Some(route) = route_of(ch, v.obs.player) else { return action };
        let mut farm = sim::farm(v.farm());
        let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
        for (actor, c) in action.units().iter().enumerate().take(v.obs.invs.len()) {
            sim::apply(&mut farm, &mut private, actor, c, step.div_euclid(24));
        }
        let mut held = private.shed.get("WHEAT").max(0);
        let reserve = self.r95_reserve(v, ch, route);
        let mut result = action;
        for o in result.market.iter_mut() {
            if o.len() < 3 || !is(o, "BUY_PRODUCT", "WHEAT") {
                continue;
            }
            let quantity = o.n(2).max(0);
            let retained = quantity.min((reserve - held).max(0));
            held += retained;
            if retained < quantity {
                o.set_n(2, retained); // zero keeps every later order slot
            }
        }
        result
    }
}

// ---- R97 -------------------------------------------------------------------------------------
pub fn market_stock(shed: &Qty, orders: &[Cmd]) -> (Qty, Vec<(usize, i64)>, Vec<(usize, i64)>) {
    let mut stock = shed.clone();
    let (mut buys, mut sales) = (vec![], vec![]);
    for (i, o) in orders.iter().enumerate() {
        if o.len() < 3 {
            continue;
        }
        let (op, item, n) = (o.op(), o.s(1), o.n(2).max(0));
        if op == "SELL" {
            let q = n.min(qget(&stock, item).max(0));
            let cur = qget(&stock, item);
            qset(&mut stock, item, cur - q);
            sales.push((i, q));
        } else if op == "BUY_PRODUCT" || op == "BUY_ANIMAL" {
            let total: i64 = stock.iter().map(|(_, n)| n).sum();
            let q = n.min((100 - total).max(0));
            let cur = qget(&stock, item);
            qset(&mut stock, item, cur + q);
            buys.push((i, q));
        }
    }
    (stock, buys, sales)
}

pub fn delivery(stock: &Qty, invs: &[Qty], night: bool) -> (Qty, Qty) {
    let mut stock = stock.clone();
    let mut lost: Qty = vec![];
    if night {
        for inv in invs {
            for (item, q) in inv {
                let q = (*q).max(0);
                let total: i64 = stock.iter().map(|(_, n)| n).sum();
                let take = q.min((100 - total).max(0));
                qadd(&mut stock, item, take);
                if q > take {
                    qadd(&mut lost, item, q - take);
                }
            }
        }
    }
    (stock, lost)
}

/// `_r97_budget`; None = Python would raise (unknown BUY_PRODUCT item).
pub fn budget(v: &View, orders: &[Cmd]) -> Option<bool> {
    budget_with(v, orders, v.farm().money)
}

/// `_r97_budget` against an explicit cash amount (R127 subtracts its own grain cost first).
pub fn budget_with(v: &View, orders: &[Cmd], money: f64) -> Option<bool> {
    let farm = v.farm();
    let mut cost = 0i64;
    let mut hires = farm.hires_today;
    let pw = market::price("WHEAT", (qget(&v.obs.mkt_inventory, "WHEAT") - 2000) as f64);
    let pf = market::price("FERTILIZER", (qget(&v.obs.mkt_inventory, "FERTILIZER") - 2000) as f64);
    for o in orders {
        if o.is_empty() {
            continue;
        }
        match o.op() {
            "HIRE" => {
                cost += fib(hires);
                hires += 1;
            }
            "BUY_LAND" => cost += 4000,
            op if o.len() > 2 => {
                let (item, q) = (o.s(1), o.n(2).max(0));
                match op {
                    "BUY_PRODUCT" => {
                        cost += q * match item {
                            "WHEAT" => pw,
                            "FERTILIZER" => pf,
                            _ => return None,
                        }
                    }
                    "BUY_ANIMAL" => cost += q * animal_cost(item)?,
                    "BUY_SEED" => cost += q * seed_price(item)?,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Some(cost as f64 <= money)
}

/// `_r97_supply`: make sure the wheat the next tape step picks up is in the shed.
pub fn r97(action: Action, v: &View, ch: &Chassis) -> Action {
    let step = v.step;
    if !(144..695).contains(&step) {
        return action;
    }
    let day = step.div_euclid(24);
    let Some(route) = route_of(ch, v.obs.player) else { return action };
    let future = tape_at(ch, route, step + 1).unwrap();
    let commands = future.units();
    let following = tape_at(ch, route, step + 2).unwrap();
    let demand = |cs: &[Cmd]| cs.iter().filter(|c| is(c, "PICKUP", "WHEAT")).map(qty_or1).sum::<i64>();
    let mut prefund = 0;
    if future.market.len() == 10 && !future.market.iter().any(|o| is(o, "BUY_PRODUCT", "WHEAT") || is(o, "SELL", "WHEAT")) {
        let later = following.units();
        if demand(&later) != 0 {
            prefund = demand(&commands) + demand(&later);
        }
    }
    if prefund == 0 && !commands.iter().any(|c| is(c, "PICKUP", "WHEAT")) {
        return action;
    }
    let orders = &action.market;
    if orders.len() > 10 || budget(v, orders) != Some(true) {
        return action;
    }
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    for (actor, c) in action.units().iter().enumerate().take(v.obs.invs.len()) {
        sim::apply(&mut farm, &mut private, actor, c, day);
    }
    let night = step.rem_euclid(24) == 23;
    let mut positions: Vec<P> = vec![farm.farmer];
    positions.extend(farm.hands.iter().copied());
    if night {
        positions = vec![(4, 4)];
    } else {
        for o in orders {
            if !o.is_empty() && o.op() == "HIRE" {
                let mut chosen = ACCESS[0];
                let mut key = (usize::MAX, usize::MAX);
                for (ai, p) in ACCESS.iter().enumerate() {
                    let k = (positions.iter().filter(|q| *q == p).count(), ai);
                    if k < key {
                        key = k;
                        chosen = *p;
                    }
                }
                positions.push(chosen);
            }
        }
    }
    let mut need: i64 =
        positions.iter().zip(commands.iter()).filter(|(p, c)| ACCESS.contains(p) && is(c, "PICKUP", "WHEAT")).map(|(_, c)| qty_or1(c)).sum();
    need = need.max(prefund);
    if need == 0 {
        return action;
    }
    let shed: Qty = private.shed.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect();
    let invs: Vec<Qty> = private.inventories.iter().map(|m| m.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect()).collect();
    let (ostock, obuys, _) = market_stock(&shed, orders);
    let (ofinal, oloss) = delivery(&ostock, &invs, night);
    if qget(&ofinal, "WHEAT") >= need {
        return action;
    }
    let project = |cand: &[Cmd]| {
        let (stock, buys, sales) = market_stock(&shed, cand);
        let (fin, loss) = delivery(&stock, &invs, night);
        let safe = obuys.iter().all(|(i, q)| buys.iter().find(|(j, _)| j == i).map(|(_, b)| *b).unwrap_or(0) >= *q)
            && loss.iter().all(|(item, q)| *q <= qget(&oloss, item));
        (fin, sales, safe)
    };
    let mut proposed: Vec<Cmd> = orders.clone();
    for index in (0..proposed.len()).rev() {
        if !is(&proposed[index], "SELL", "WHEAT") {
            continue;
        }
        let (fin, sales, _) = project(&proposed);
        let shortage = (need - qget(&fin, "WHEAT")).max(0);
        if shortage == 0 {
            break;
        }
        let sold = sales.iter().find(|(j, _)| *j == index).map(|(_, q)| *q).unwrap_or(0);
        if sold == 0 {
            continue;
        }
        let old = proposed[index].0[2].clone();
        proposed[index].set_n(2, (sold - shortage).max(0));
        let (after, _, safe) = project(&proposed);
        if !safe || qget(&after, "WHEAT") <= qget(&fin, "WHEAT") {
            proposed[index].0[2] = old;
        }
    }
    let (fin, _, _) = project(&proposed);
    let shortage = (need - qget(&fin, "WHEAT")).max(0);
    if shortage != 0 {
        let last_sale = proposed.iter().rposition(|o| is(o, "SELL", "WHEAT")).map(|i| i as i64).unwrap_or(-1);
        let idx = (0..proposed.len()).rev().take_while(|&i| i as i64 > last_sale).find(|&i| is(&proposed[i], "BUY_PRODUCT", "WHEAT"));
        if let Some(i) = idx {
            let nv = proposed[i].n(2).max(0) + shortage;
            proposed[i].set_n(2, nv);
        } else if proposed.len() < 10 {
            proposed.push(Cmd::order("BUY_PRODUCT", "WHEAT", shortage));
        } else {
            return action;
        }
    }
    let (fin, _, safe) = project(&proposed);
    if !safe || qget(&fin, "WHEAT") < need {
        return action;
    }
    if budget(v, &proposed) != Some(true) {
        return action;
    }
    let mut result = action;
    result.market = proposed;
    result
}
