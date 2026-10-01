//! OR2 "sales ordered by the rival's estimated sellable stock" (agent 4391-4636) and
//! CH "harvest animals that would overflow tonight" (4639-4786).
use super::r37::quote_priority;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qadd, qget, qset, FarmObs, Qty};
use crate::view::{move_delta, View};

pub const CAP: i64 = 24;
pub const SLOT_H: i64 = 6;
pub const SLOT_MARGIN: f64 = 12.0;
const SN_ITEMS: [&str; 5] = ["MILK", "STRAWBERRY", "WOOL", "MELON", "EGG"];
const ITEMS: [&str; 8] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];
const ONGOING: [&str; 2] = ["TOMATO", "STRAWBERRY"];
type P = (i64, i64);

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

fn draw(shops: &[&str], step: i64) -> Qty {
    let mut d: Qty = vec![];
    if step % 4 == 0 {
        for s in shops {
            let p = shop_items(s);
            for item in p {
                qadd(&mut d, item, if p.len() == 1 { 2 } else { 1 });
            }
        }
    }
    if step % 24 == 0 {
        for item in ITEMS {
            qadd(&mut d, item, 1);
        }
    }
    d
}

/// `(kind 'P'|'A', item, born, yield)` per occupied tile.
type TileSig = (char, &'static str, i64, i64);
fn tiles(f: &FarmObs) -> Vec<(P, TileSig)> {
    let mut out = vec![];
    for y in 0..f.rows {
        for x in 0..f.cols {
            let t = &f.tiles[y * f.cols + x];
            if !t.is_dict() {
                continue;
            }
            if t.kind == "PLANT" && !t.crop.is_empty() {
                out.push(((x as i64, y as i64), ('P', t.crop, t.planted_day, t.yield_units)));
            } else if let Some(p) = match t.animal {
                "GOOSE" => Some("EGG"),
                "COW" => Some("MILK"),
                "SHEEP" => Some("WOOL"),
                _ => None,
            } {
                out.push(((x as i64, y as i64), ('A', p, t.placed_day, t.yield_units)));
            }
        }
    }
    out
}

/// `_or2_exposure`.
fn exposure(v: &View, item: &str, qty: i64, batch: i64) -> f64 {
    if qty <= 0 || batch <= 0 || market::params(item).is_none() {
        return 0.0;
    }
    let inv = qget(&v.obs.mkt_inventory, item) as f64;
    let mut s = 0i64;
    for j in 0..qty {
        s += market::price(item, inv + j as f64) - market::price(item, inv + (batch + j) as f64);
    }
    s as f64
}

#[derive(Clone)]
struct Prev {
    step: i64,
    tiles: Vec<(P, TileSig)>,
    inv: Qty,
    own: Qty,
    prices: Qty,
    shops: Vec<&'static str>,
}
#[derive(Clone)]
struct St {
    step: i64,
    stock: Qty,
    prev: Option<Prev>,
}

#[derive(Default, Clone)]
pub struct Or2 {
    players: Vec<(i64, St)>,
    ch: Vec<(i64, (i64, Qty))>,
    /// profile knob `or2_slot_margin` (None = SLOT_MARGIN)
    pub margin: Option<f64>,
}

fn debts_of(ch: &mut Chassis, player: i64) -> Option<&mut Vec<(i64, Qty)>> {
    ch.players.iter_mut().find(|(p, _)| *p == player).map(|(_, s)| &mut s.sell.r36_debts)
}
fn owed(d: &[(i64, Qty)], t: i64, item: &str) -> i64 {
    d.iter().find(|(s, _)| *s == t).map(|(_, m)| qget(m, item)).unwrap_or(0)
}
fn book(d: &mut Vec<(i64, Qty)>, t: i64, item: &'static str, q: i64) {
    match d.iter_mut().find(|(s, _)| *s == t) {
        Some((_, m)) => qadd(m, item, q),
        None => d.push((t, vec![(item, q)])),
    }
}

impl Or2 {
    pub fn post(&mut self, action: Action, v: &View, ch: &mut Chassis) -> Action {
        let (step, seat) = (v.step, v.obs.player);
        let i = match self.players.iter().position(|(p, _)| *p == seat) {
            Some(i) if step != 0 && step > self.players[i].1.step => i,
            Some(i) => {
                self.players[i].1 = St { step: -1, stock: ITEMS.iter().map(|x| (*x, 0)).collect(), prev: None };
                i
            }
            None => {
                self.players.push((seat, St { step: -1, stock: ITEMS.iter().map(|x| (*x, 0)).collect(), prev: None }));
                self.players.len() - 1
            }
        };
        let mut st = self.players[i].1.clone();
        let rival = v.rival();
        let rtiles = tiles(rival);
        let inv_now: Qty = ITEMS.iter().map(|x| (*x, qget(&v.obs.mkt_inventory, x))).collect();
        if let Some(prev) = st.prev.as_ref().filter(|p| p.step == step - 1) {
            for (pos, old) in &prev.tiles {
                let (kind, item, born, y) = *old;
                if y <= 0 {
                    continue;
                }
                let new = rtiles.iter().find(|(p, _)| p == pos).map(|(_, s)| *s);
                let mut got = 0;
                if kind == 'P' && !ONGOING.contains(&item) {
                    let weed = {
                        let t = rival.tile(pos.0, pos.1);
                        t.is_dict() && t.kind == "WEED"
                    };
                    if (new.is_none() && !weed) || new.is_some_and(|n| n.2 != born) {
                        got = y;
                    }
                } else if let Some(n) = new.filter(|n| n.0 == kind && n.1 == item && n.2 == born && n.3 < y) {
                    if step % 24 != 0 {
                        got = y - n.3;
                    } else if n.3 == 0 {
                        got = y;
                    }
                }
                if got > 0 {
                    qadd(&mut st.stock, item, got);
                }
            }
            let d = draw(&prev.shops, prev.step);
            for item in ITEMS {
                if qget(&prev.prices, item) <= 1 {
                    continue;
                }
                let moved = qget(&inv_now, item) - qget(&prev.inv, item) + qget(&d, item) - qget(&prev.own, item);
                if moved > 0 {
                    let cur = qget(&st.stock, item);
                    qset(&mut st.stock, item, (cur - moved).max(0));
                }
            }
        }
        let mut own: Qty = vec![];
        let mut action = action;
        let mut orders: Vec<Cmd> = action.market.clone();
        let proj = ch.projected_shed(&action, v);
        // slot swap
        if SLOT_H > 0 && (288..694).contains(&step) && orders.len() >= 10 {
            let route = ch.players.iter().find(|(p, _)| *p == seat).and_then(|(_, s)| s.route);
            if let (Some(route), true) = (route, debts_of(ch, seat).is_some()) {
                let sells: Vec<usize> = (0..orders.len()).filter(|&k| orders[k].is_sell3()).collect();
                let selling: Vec<&str> = sells.iter().map(|&k| orders[k].s(1)).collect();
                let bought: Vec<&str> =
                    orders.iter().filter(|o| o.len() >= 2 && matches!(o.op(), "BUY_PRODUCT" | "BUY_ANIMAL")).map(|o| o.s(1)).collect();
                let mut best: Option<(f64, &'static str, i64, Vec<(i64, i64)>)> = None;
                {
                    let debts = debts_of(ch, seat).unwrap().clone();
                    for item in SN_ITEMS {
                        if selling.contains(&item) || bought.contains(&item) {
                            continue;
                        }
                        let avail = qget(&proj, item);
                        if avail <= 0 || v.price(item) < 2 {
                            continue;
                        }
                        let (mut plan, mut take) = (vec![], 0i64);
                        for t in (step + 1)..=(694.min(step + SLOT_H)) {
                            if take >= avail {
                                break;
                            }
                            let r = if t >= 648 { 2 } else { route };
                            let planned: i64 = ch
                                .route(r)
                                .tape
                                .get(t as usize)
                                .map(|a| a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum())
                                .unwrap_or(0);
                            let q = (planned - owed(&debts, t, item)).min(avail - take);
                            if q > 0 {
                                plan.push((t, q));
                                take += q;
                            }
                        }
                        if take <= 0 {
                            continue;
                        }
                        let b = CAP.min(qget(&st.stock, item));
                        let value = exposure(v, item, take, b.max(1));
                        if best.as_ref().is_none_or(|x| value > x.0) {
                            best = Some((value, item, take, plan));
                        }
                    }
                }
                if let (Some(best), false) = (best, sells.is_empty()) {
                    let sval = |o: &Cmd| {
                        let q = o.n(2).max(0).min(qget(&proj, o.s(1)).max(0));
                        exposure(v, o.s(1), q, CAP.min(qget(&st.stock, o.s(1))).max(1))
                    };
                    let mut wk = sells[0];
                    for &k in &sells[1..] {
                        if sval(&orders[k]) < sval(&orders[wk]) {
                            wk = k;
                        }
                    }
                    let weakest = orders[wk].clone();
                    let sw = sval(&weakest);
                    if (best.0 > sw + self.margin.unwrap_or(SLOT_MARGIN) && !matches!(weakest.s(1), "WHEAT" | "FERTILIZER"))
                        || (best.0 > sw + self.margin.unwrap_or(SLOT_MARGIN) && weakest.n(2) <= 2)
                    {
                        let debts = debts_of(ch, seat).unwrap();
                        let mut refund = weakest.n(2).max(0);
                        for t in (step + 1)..(step + 49) {
                            if refund <= 0 {
                                break;
                            }
                            let o = owed(debts, t, weakest.s(1));
                            let back = o.min(refund);
                            if back > 0 {
                                let m = &mut debts.iter_mut().find(|(s, _)| *s == t).unwrap().1;
                                qset(m, weakest.s(1), o - back);
                                refund -= back;
                            }
                        }
                        let first = orders.iter().position(|o| *o == weakest).unwrap();
                        orders.remove(first);
                        orders.push(Cmd::order("SELL", best.1, best.2));
                        for (t, q) in &best.3 {
                            book(debts, *t, best.1, *q);
                        }
                        action.market = orders.clone();
                    }
                }
            }
        }
        // reorder by rival-exposure
        let mut left = proj.clone();
        let mut movable: Vec<(usize, Cmd)> = vec![];
        let mut fixed: Vec<(usize, Cmd)> = vec![];
        let mut bought: Vec<&str> = vec![];
        for (idx, o) in orders.iter().enumerate() {
            if o.len() >= 2 && matches!(o.op(), "BUY_PRODUCT" | "BUY_ANIMAL") {
                bought.push(o.s(1));
            }
            if o.is_sell3() && o.n(2) > 0 && !bought.contains(&o.s(1)) {
                movable.push((idx, o.clone()));
            } else {
                fixed.push((idx, o.clone()));
            }
        }
        if !movable.is_empty() && step >= 1 {
            let key = |io: &(usize, Cmd)| {
                let item = io.1.s(1);
                let qty = io.1.n(2).min(qget(&proj, item).max(0));
                let b = CAP.min(qget(&st.stock, item));
                (-exposure(v, item, qty, b), -(quote_priority(v, item, io.1.n(2), &proj) as f64), io.0)
            };
            let mut scored: Vec<((f64, f64, usize), Cmd)> = movable.iter().map(|io| (key(io), io.1.clone())).collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let new: Vec<Cmd> = scored.into_iter().map(|(_, o)| o).chain(fixed.into_iter().map(|(_, o)| o)).collect();
            if new != orders {
                action.market = new.clone();
                orders = new;
            }
        }
        for o in orders.iter().take(10) {
            if o.is_sell3() && ITEMS.contains(&o.s(1)) {
                let item = o.s(1);
                let got = o.n(2).max(0).min(qget(&left, item).max(0));
                let cur = qget(&left, item);
                qset(&mut left, item, cur - got);
                qadd(&mut own, item, got);
            }
        }
        st.prev = Some(Prev { step, tiles: rtiles, inv: inv_now, own, prices: v.obs.prices.clone(), shops: v.obs.shops.clone() });
        st.step = step;
        self.players[i].1 = st;
        action
    }

    // ---- CH ----------------------------------------------------------------------------------
    pub fn ch(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let (step, seat) = (v.step, v.obs.player);
        let i = match self.ch.iter().position(|(p, _)| *p == seat) {
            Some(i) if step != 0 && step > self.ch[i].1 .0 => i,
            Some(i) => {
                self.ch[i].1 = (-1, vec![]);
                i
            }
            None => {
                self.ch.push((seat, (-1, vec![])));
                self.ch.len() - 1
            }
        };
        self.ch[i].1 .0 = step;
        if step > 717 {
            return action;
        }
        let day = step.div_euclid(24);
        let farm = v.farm();
        let mut units = action.units();
        let mut market = action.market.clone();
        let mut changed = false;
        let mut visits: Option<Vec<(P, Vec<(i64, &'static str)>)>> = None;
        let n = v.positions.len().min(units.len());
        for k in 0..n {
            let pos = v.positions[k];
            let cmd = &units[k];
            if cmd.is_empty() || !matches!(cmd.op(), "CARE" | "COLLECT_FERTILIZER") {
                continue;
            }
            let t = farm.tile(pos.0, pos.1);
            let (product, cap, first, interval) = match (t.is_dict(), t.animal) {
                (true, "GOOSE") => ("EGG", 4, 4, 1),
                (true, "COW") => ("MILK", 6, 8, 2),
                (true, "SHEEP") => ("WOOL", 6, 6, 3),
                _ => continue,
            };
            let since = day + 1 - t.placed_day - first;
            if since < 0 || since.rem_euclid(interval) != 0 {
                continue;
            }
            let y = t.yield_units;
            let vis = visits.get_or_insert_with(|| visits_today(v, ch, &action));
            static NONE: Vec<(i64, &str)> = Vec::new();
            let later = vis.iter().find(|(p, _)| *p == pos).map(|(_, l)| l).unwrap_or(&NONE);
            if later.iter().any(|(_, op)| *op == "HARVEST") {
                continue;
            }
            let fed = t.fed_today || later.iter().any(|(_, op)| *op == "FEED");
            let prod = 1 + if fed { t.pending_care_bonus } else { 0 };
            let overflow = y + prod - cap;
            if overflow <= 0 || y <= 0 {
                continue;
            }
            let quote = v.price(product);
            if cmd.op() == "COLLECT_FERTILIZER" {
                if overflow * quote <= v.price("FERTILIZER") {
                    continue;
                }
            } else if later.iter().any(|(_, op)| *op == "COLLECT_FERTILIZER") || overflow <= 1 {
                continue;
            }
            let carried: i64 = v.obs.invs.iter().flat_map(|m| m.iter().map(|(_, n)| *n)).sum();
            if v.obs.shed.iter().map(|(_, n)| n).sum::<i64>() + carried + y >= 100 {
                continue;
            }
            let saved = if cmd.op() == "COLLECT_FERTILIZER" { overflow } else { overflow - 1 };
            units[k] = Cmd::new("HARVEST");
            qadd(&mut self.ch[i].1 .1, product, saved);
            changed = true;
        }
        if self.ch[i].1 .1.iter().any(|(_, c)| *c > 0) {
            let va = Action { farmer: units[0].clone(), hands: units[1..].to_vec(), market: market.clone() };
            let stock = ch.projected_shed(&va, v);
            let credits = self.ch[i].1 .1.clone();
            for (product, credit) in credits {
                if credit <= 0 || market.len() >= 10 || v.price(product) < 2 {
                    continue;
                }
                let selling: i64 = market.iter().filter(|o| o.is_sell3() && o.s(1) == product).map(|o| o.n(2)).sum();
                let q = credit.min(qget(&stock, product) - selling);
                if q > 0 {
                    if let Some(o) = market.iter_mut().find(|o| o.is_sell3() && o.s(1) == product) {
                        let nv = o.n(2) + q;
                        o.set_n(2, nv);
                    } else {
                        market.insert(0, Cmd::order("SELL", product, q));
                    }
                    qset(&mut self.ch[i].1 .1, product, credit - q);
                    changed = true;
                }
            }
        }
        if !changed {
            return action;
        }
        let mut r = action;
        r.set_units(units);
        market.truncate(10);
        r.market = market;
        r
    }
}

/// `_ch_visits_today`: {pos: [(t, op)]} for non-move commands after this step, today.
fn visits_today(v: &View, ch: &Chassis, action: &Action) -> Vec<(P, Vec<(i64, &'static str)>)> {
    let step = v.step;
    let board = v.farm().rows as i64;
    let half = board / 2;
    let access = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)];
    let mut positions = v.positions.clone();
    let mut out: Vec<(P, Vec<(i64, &'static str)>)> = vec![];
    let native = ch.players.iter().find(|(p, _)| *p == v.obs.player).map(|(_, s)| s);
    for t in step..((step / 24 + 1) * 24) {
        let act: Option<&Action> = if t == step {
            Some(action)
        } else {
            native.and_then(|n| {
                if t > 719 {
                    return None;
                }
                let r = if t >= 648 { Some(2) } else { n.route }?;
                ch.route(r).tape.get(t as usize)
            })
        };
        let units = act.map(|a| a.units()).unwrap_or_else(|| vec![Cmd::pass()]);
        for i in 0..positions.len() {
            let op = units.get(i).filter(|c| !c.is_empty()).map(|c| c.op()).unwrap_or("PASS");
            if let Some((dx, dy)) = move_delta(op) {
                let (nx, ny) = (positions[i].0 + dx, positions[i].1 + dy);
                if (0..board).contains(&nx) && (0..board).contains(&ny) {
                    positions[i] = (nx, ny);
                }
            } else if t > step {
                let p = positions[i];
                match out.iter_mut().find(|(q, _)| *q == p) {
                    Some((_, l)) => l.push((t, op)),
                    None => out.push((p, vec![(t, op)])),
                }
            }
        }
        if let Some(a) = act {
            for o in &a.market {
                if !o.is_empty() && o.op() == "HIRE" {
                    let mut best = access[0];
                    let mut key = (usize::MAX, usize::MAX);
                    for (ai, a) in access.iter().enumerate() {
                        let k = (positions.iter().filter(|p| *p == a).count(), ai);
                        if k < key {
                            key = k;
                            best = *a;
                        }
                    }
                    positions.push(best);
                }
            }
        }
    }
    out
}
