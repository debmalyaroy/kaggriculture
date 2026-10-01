//! SR "shedroom" (agent 4789-4901), HD2 herd choice at the tape's goose purchase (4904-5187)
//! and CS cow->goose swap at the first cow purchase (5190-5427).
use super::r37::similarity;
use crate::act::{Action, Cmd, Tok};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qadd, qget, Qty};
use crate::view::{shed_adjacent, View, PRODUCTS};

type P = (i64, i64);

fn tape_at<'a>(ch: &'a Chassis, player: i64, t: i64) -> Option<&'a Action> {
    let n = ch.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s)?;
    if t > 719 {
        return None;
    }
    let r = if t >= 648 { 2 } else { n.route? };
    ch.route(r).tape.get(t as usize)
}

// ---- SR --------------------------------------------------------------------------------------
pub fn sr(action: Action, v: &View, ch: &Chassis) -> Action {
    let step = v.step;
    if !(21..=23).contains(&step.rem_euclid(24)) || step >= 717 {
        return action;
    }
    let proj = ch.projected_shed(&action, v);
    let mut market = action.market.clone();
    let mut left = proj.clone();
    let mut night: i64 = proj.iter().map(|(_, n)| (*n).max(0)).sum();
    for o in &market {
        if o.is_sell3() {
            let got = o.n(2).max(0).min(qget(&left, o.s(1)).max(0));
            qadd(&mut left, o.s(1), -got);
            night -= got;
        } else if o.len() >= 3 && matches!(o.op(), "BUY_PRODUCT" | "BUY_ANIMAL") {
            night += o.n(2).max(0);
        }
    }
    let units = action.units();
    let farm = v.farm();
    let mut carried = 0;
    for (i, &pos) in v.positions.iter().enumerate() {
        let mut held: i64 = v.inv(i).iter().map(|(_, n)| (*n).max(0)).sum();
        let cmd = units.get(i).filter(|c| !c.is_empty()).cloned().unwrap_or_else(Cmd::pass);
        let tile = farm.tile(pos.0, pos.1);
        match cmd.op() {
            "DROP" if shed_adjacent(pos, v.board) => held = 0,
            "HARVEST" if tile.is_dict() => held += tile.yield_units.max(0),
            "COLLECT_FERTILIZER" if tile.is_dict() && tile.fertilizer_available => held += 1,
            "FEED" | "FERTILIZE" if held > 0 => held -= 1,
            "PICKUP" if cmd.len() >= 2 && shed_adjacent(pos, v.board) => held += (if cmd.len() >= 3 { cmd.n(2) } else { 1 }).max(1),
            _ => {}
        }
        carried += held;
    }
    let mut overflow = night + carried - 100 + 8;
    if overflow <= 0 {
        return action;
    }
    let (mut nw, mut nf) = (0i64, 0i64);
    for t in (step + 1)..719.min(step + 25) {
        if let Some(a) = tape_at(ch, v.obs.player, t) {
            for c in a.units() {
                if c.is_empty() {
                    continue;
                }
                match c.op() {
                    "FEED" => nw += 1,
                    "FERTILIZE" => nf += 1,
                    _ => {}
                }
            }
        }
    }
    let mut cands: Vec<(i64, &'static str, i64)> = vec![];
    for item in PRODUCTS {
        let need = match item {
            "WHEAT" => nw,
            "FERTILIZER" => nf,
            _ => 0,
        };
        let spare = qget(&left, item) - need;
        if spare > 0 && v.price(item) >= 2 {
            cands.push((v.price(item), item, spare));
        }
    }
    cands.sort();
    let mut sold = 0;
    for (_, item, spare) in cands {
        if overflow <= 0 {
            break;
        }
        let q = spare.min(overflow);
        if let Some(o) = market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
            let nv = o.n(2) + q;
            o.set_n(2, nv);
        } else {
            if market.len() >= 10 {
                continue;
            }
            market.push(Cmd::order("SELL", item, q));
        }
        overflow -= q;
        sold += q;
    }
    if sold == 0 {
        return action;
    }
    let mut r = action;
    r.market = market;
    r
}

// ---- HD2 -------------------------------------------------------------------------------------
pub const HD2_FROM: i64 = 192;
pub const HD2_TO: i64 = 360;
pub const HD2_RATIO: f64 = 1.3;
pub const HD2_MIN_GAIN: f64 = 600.0;
pub const HD2_CARE: f64 = 0.8;
pub const HD2_FUTURE: f64 = 0.0;

/// (cost, first, interval, per, product)
fn spec(animal: &str) -> (i64, i64, i64, i64, &'static str) {
    match animal {
        "GOOSE" => (300, 4, 1, 2, "EGG"),
        "COW" => (400, 8, 2, 3, "MILK"),
        _ => (500, 6, 3, 4, "WOOL"),
    }
}
fn shop_types(s: &str) -> &'static [&'static str] {
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
const ALL_SHOPS: [&str; 8] =
    ["BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"];

fn schedule(animal: &str, placed: i64, from: i64, out: &mut [f64; 30], scale: f64) {
    let (_, first, interval, per, _) = spec(animal);
    let lo = from.max(placed + first);
    for d in lo.max(0)..30 {
        if (d - placed - first).rem_euclid(interval) == 0 {
            out[d as usize] += (1.0 + (per - 1) as f64 * HD2_CARE) * scale;
        }
    }
}
fn pyround(x: f64) -> i64 {
    market::round_half_even(x) as i64
}

/// `_hd2_ev(option, k)` -> margin value.
pub fn ev(option: &str, k: i64, v: &View, ch: &Chassis) -> f64 {
    let (cost, _, _, _, item) = spec(option);
    let animal_of = match item {
        "EGG" => "GOOSE",
        "MILK" => "COW",
        _ => "SHEEP",
    };
    let step = v.step;
    let day = step.div_euclid(24);
    let seat = v.me;
    let mut existing = [[0f64; 30]; 2];
    for (fi, f) in v.obs.farms.iter().enumerate().take(2) {
        for t in &f.tiles {
            if t.is_dict() && t.animal == animal_of {
                schedule(animal_of, t.placed_day, day + 1, &mut existing[fi], 1.0);
            }
        }
    }
    let carried: i64 = v.obs.invs.iter().map(|m| qget(m, animal_of)).sum();
    for _ in 0..(qget(&v.obs.shed, animal_of) + carried).max(0) {
        schedule(animal_of, day + 1, day + 1, &mut existing[seat], 1.0);
    }
    if let Some(native) = ch.players.iter().find(|(p, _)| *p == v.obs.player).map(|(_, s)| s) {
        let similar = similarity(v) >= 0.9;
        for t in (step + 1)..696 {
            let r = if t >= 648 { Some(2) } else { native.route };
            let Some(a) = r.and_then(|r| ch.route(r).tape.get(t as usize)) else { continue };
            for o in &a.market {
                if o.len() >= 3 && o.op() == "BUY_ANIMAL" && o.s(1) == animal_of {
                    for _ in 0..o.n(2).max(0) {
                        schedule(animal_of, t / 24 + 1, day + 1, &mut existing[seat], 1.0);
                        if similar {
                            schedule(animal_of, t / 24 + 1, day + 1, &mut existing[1 - seat], 1.0);
                        }
                    }
                }
            }
        }
    }
    let (ours, rival) = (existing[seat], existing[1 - seat]);
    let mut new = [0f64; 30];
    schedule(option, day + 1, day + 1, &mut new, k as f64);
    let shops = v.shops();
    let unlocks_left = (8 - shops.len() as i64).max(0);
    let base_demand: f64 = shops
        .iter()
        .map(|s| {
            let p = shop_types(s);
            if p.contains(&item) {
                6.0 * if p.len() == 1 { 2.0 } else { 1.0 }
            } else {
                0.0
            }
        })
        .sum::<f64>()
        + 1.0;
    let fut: f64 = ALL_SHOPS
        .iter()
        .map(|s| {
            let p = shop_types(s);
            if p.contains(&item) {
                6.0 * if p.len() == 1 { 2.0 } else { 1.0 }
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / ALL_SHOPS.len() as f64;
    let extra_demand = fut * HD2_FUTURE;
    let inv0 = qget(&v.obs.mkt_inventory, item) as f64;
    let path = |with_new: bool| -> ([i64; 30], f64) {
        let mut inv = inv0;
        let mut prices = [0i64; 30];
        let mut revenue = 0.0;
        for d in (day + 1)..30 {
            let opened = unlocks_left.min((d / 3 - day / 3).max(0));
            inv -= base_demand + extra_demand * opened as f64;
            inv += ours[d as usize] + rival[d as usize];
            prices[d as usize] = market::price(item, pyround(inv) as f64);
            if with_new {
                let units = pyround(new[d as usize]);
                for _ in 0..units.max(0) {
                    let price = market::price(item, pyround(inv) as f64);
                    revenue += price as f64;
                    if price > 1 {
                        inv += 1.0;
                    }
                }
            }
        }
        (prices, revenue)
    };
    let (bp, _) = path(false);
    let (np, revenue) = path(true);
    let mut swing = 0.0;
    for d in (day + 1)..30 {
        let d = d as usize;
        swing += (np[d] - bp[d]) as f64 * (ours[d] - rival[d]);
    }
    revenue + swing - (cost * k) as f64
}

#[derive(Clone, Default)]
struct Hd2St {
    step: i64,
    decided: bool,
    mode: Option<&'static str>,
    pending: Vec<(i64, i64, i64)>,
    sites: Vec<(P, i64)>,
    credit: i64,
}

#[derive(Clone, Default)]
struct CsSt {
    step: i64,
    decided: bool,
    mode: Option<&'static str>,
    plan: Option<Plan>,
    broken: bool,
    sites: Vec<(P, i64)>,
    pending: Vec<(i64, i64, i64)>,
    credit: i64,
}

#[derive(Clone, Default)]
struct Plan {
    builds: Vec<((i64, usize), P)>,
    pickups: Vec<((i64, usize), i64)>,
    places: Vec<((i64, usize), P)>,
}

impl Herd {
    /// HD2 / CS started for this player (a herd decision taken or a rewrite in flight): option gates only block
    /// NEW decisions.
    pub fn hd2_active(&self, player: i64) -> bool {
        self.hd2.iter().find(|(p, _)| *p == player).is_some_and(|(_, s)| (s.decided && s.mode.is_some()) || !s.pending.is_empty() || !s.sites.is_empty())
    }
    pub fn cs_active(&self, player: i64) -> bool {
        self.cs.iter().find(|(p, _)| *p == player).is_some_and(|(_, s)| (s.decided && s.mode.is_some()) || s.plan.is_some() || !s.pending.is_empty() || !s.sites.is_empty())
    }
}

#[derive(Default, Clone)]
pub struct Herd {
    hd2: Vec<(i64, Hd2St)>,
    cs: Vec<(i64, CsSt)>,
}

fn slot<T: Clone + Default>(v: &mut Vec<(i64, T)>, player: i64, step: i64, last: impl Fn(&T) -> i64, fresh: impl Fn() -> T) -> usize {
    match v.iter().position(|(p, _)| *p == player) {
        Some(i) if step > last(&v[i].1) => i,
        Some(i) => {
            v[i].1 = fresh();
            i
        }
        None => {
            v.push((player, fresh()));
            v.len() - 1
        }
    }
}

impl Herd {
    pub fn hd2(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let i = slot(&mut self.hd2, v.obs.player, step, |s| s.step, || Hd2St { step: -1, ..Default::default() });
        let mut st = std::mem::take(&mut self.hd2[i].1);
        st.step = step;
        let farm = v.farm();
        if let Some(mode) = st.mode {
            let day = step.div_euclid(24);
            let mut keep = vec![];
            for &(x, y, d) in &st.pending {
                let t = farm.tile(x, y);
                if t.is_dict() && t.animal == mode && t.placed_day == d {
                    match st.sites.iter_mut().find(|(p, _)| *p == (x, y)) {
                        Some(e) => e.1 = d,
                        None => st.sites.push(((x, y), d)),
                    }
                } else if day <= d + 1 {
                    keep.push((x, y, d));
                }
            }
            st.pending = keep;
        } else if !st.decided {
            // _hd2_decide
            let buys: Vec<&Cmd> = action.market.iter().filter(|o| o.len() >= 3 && o.op() == "BUY_ANIMAL" && o.s(1) == "GOOSE").collect();
            if !buys.is_empty() {
                st.decided = true;
                let coop = farm.tiles.iter().any(|t| t.is_dict() && t.kind == "COOP");
                if (HD2_FROM..HD2_TO).contains(&step) && !coop {
                    let k: i64 = buys.iter().map(|o| o.n(2).max(0)).sum();
                    let kp = k.max(3);
                    let g = ev("GOOSE", kp, v, ch);
                    let c = ev("COW", kp, v, ch);
                    let s = ev("SHEEP", kp, v, ch);
                    let (best, eb) = if s > c { ("SHEEP", s) } else { ("COW", c) };
                    let gain = eb - g;
                    let ok = eb >= HD2_RATIO * g.max(1.0);
                    let extra = (spec(best).0 - 300) * k;
                    if gain >= HD2_MIN_GAIN && ok && farm.money >= (300 * k + extra + 50) as f64 {
                        st.mode = Some(best);
                    }
                }
            }
        }
        let out = match st.mode {
            Some(mode) => hd2_rewrite(action, v, ch, &mut st, mode),
            None => action,
        };
        self.hd2[i].1 = st;
        out
    }

    pub fn cs(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let i = slot(&mut self.cs, v.obs.player, step, |s| s.step, || CsSt { step: -1, ..Default::default() });
        let mut st = std::mem::take(&mut self.cs[i].1);
        st.step = step;
        let farm = v.farm();
        let positions = &v.positions;
        let mut market = action.market.clone();
        let mut action = action;
        if !st.pending.is_empty() {
            let mut keep = vec![];
            for &(x, y, d) in &st.pending {
                let t = farm.tile(x, y);
                if t.is_dict() && Some(t.animal) == st.mode && t.placed_day == d {
                    match st.sites.iter_mut().find(|(p, _)| *p == (x, y)) {
                        Some(e) => e.1 = d,
                        None => st.sites.push(((x, y), d)),
                    }
                } else if step.div_euclid(24) <= d {
                    keep.push((x, y, d));
                }
            }
            st.pending = keep;
        }
        if !st.decided && (144..192).contains(&step) {
            let buys: Vec<&Cmd> = market.iter().filter(|o| o.len() >= 3 && o.op() == "BUY_ANIMAL" && o.s(1) == "COW" && o.n(2) > 0).collect();
            let shops = v.shops();
            let no_milk = !shops.iter().any(|s| matches!(*s, "PIZZA_SHOP" | "ICE_CREAM_SHOP" | "SMOOTHIE_SHOP"));
            let shop_ok = no_milk && !shops.contains(&"YARN_STORE");
            if !buys.is_empty() && !shop_ok {
                st.decided = true;
            } else if !buys.is_empty() {
                st.decided = true;
                let k: i64 = buys.iter().map(|o| o.n(2)).sum();
                let cow = ev("COW", k, v, ch);
                let goose = ev("GOOSE", k, v, ch);
                if goose - cow >= HD2_MIN_GAIN && goose >= HD2_RATIO * cow.max(1.0) {
                    if let Some(plan) = cs_plan(v, ch, &action, k, "GOOSE") {
                        st.mode = Some("GOOSE");
                        st.plan = Some(plan);
                        for o in market.iter_mut() {
                            if o.len() >= 3 && o.op() == "BUY_ANIMAL" && o.s(1) == "COW" {
                                o.0[1] = Tok::S("GOOSE");
                            }
                        }
                    }
                }
            }
        }
        if let (Some(mode), Some(plan), false) = (st.mode, st.plan.clone(), st.broken) {
            let mut units = action.units();
            for i in 0..units.len().min(positions.len()) {
                let cmd = if units[i].is_empty() { Cmd::pass() } else { units[i].clone() };
                let pos = positions[i];
                let key = (step, i);
                if let Some((_, bp)) = plan.builds.iter().find(|(k, _)| *k == key) {
                    if cmd.len() == 1 && cmd.op() == "BUILD_PASTURE" && pos == *bp {
                        units[i] = Cmd::new("BUILD_COOP");
                    } else {
                        st.broken = true;
                    }
                }
                if plan.pickups.iter().any(|(k, _)| *k == key) {
                    if cmd.len() >= 2 && cmd.op() == "PICKUP" && cmd.s(1) == "COW" {
                        let mut c = cmd.clone();
                        c.0[1] = Tok::S(mode);
                        units[i] = c;
                    } else {
                        st.broken = true;
                    }
                }
                if let Some((_, pp)) = plan.places.iter().find(|(k, _)| *k == key) {
                    if cmd.len() >= 2 && cmd.op() == "PLACE" && cmd.s(1) == "COW" && pos == *pp {
                        let mut c = cmd.clone();
                        c.0[1] = Tok::S(mode);
                        units[i] = c;
                        st.pending.push((pos.0, pos.1, step.div_euclid(24)));
                    } else {
                        st.broken = true;
                    }
                }
            }
            action.set_units(units);
        }
        if !st.sites.is_empty() {
            let product = spec(st.mode.unwrap_or("GOOSE")).4;
            let units = action.units();
            for i in 0..units.len().min(positions.len()) {
                let (x, y) = positions[i];
                let t = farm.tile(x, y);
                if units[i].len() == 1
                    && units[i].op() == "HARVEST"
                    && st.sites.iter().any(|(p, _)| *p == (x, y))
                    && t.is_dict()
                    && Some(t.animal) == st.mode
                {
                    st.credit += t.yield_units.max(0);
                }
            }
            if st.credit > 0 {
                let stock = ch.projected_shed(&action, v);
                let planned: i64 = market.iter().filter(|o| o.is_sell3() && o.s(1) == product).map(|o| o.n(2).max(0)).sum();
                let mut extra = st.credit.min((qget(&stock, product) - planned).max(0));
                if extra > 0 {
                    if let Some(o) = market.iter_mut().find(|o| o.is_sell3() && o.s(1) == product) {
                        let nv = o.n(2) + extra;
                        o.set_n(2, nv);
                    } else if market.len() < 10 {
                        market.insert(0, Cmd::order("SELL", product, extra));
                    } else {
                        extra = 0;
                    }
                    st.credit -= extra;
                }
            }
        }
        action.market = market;
        self.cs[i].1 = st;
        action
    }
}

fn hd2_rewrite(mut action: Action, v: &View, ch: &Chassis, st: &mut Hd2St, mode: &'static str) -> Action {
    let (cost, _, _, _, item) = spec(mode);
    let farm = v.farm();
    let mut cash = farm.money;
    for o in action.market.iter_mut() {
        if o.len() >= 3 && o.op() == "BUY_ANIMAL" && o.s(1) == "GOOSE" {
            let n = o.n(2).max(0);
            let affordable = (cash.max(0.0) / cost as f64).floor() as i64;
            let q = n.min(affordable);
            *o = Cmd::order("BUY_ANIMAL", mode, q);
            cash -= (q * cost) as f64;
        } else if o.len() >= 3 && matches!(o.op(), "BUY_ANIMAL" | "BUY_SEED" | "BUY_PRODUCT") {
            let q = o.n(2).max(0);
            cash -= match o.op() {
                "BUY_ANIMAL" => (q * crate::view::animal_cost(o.s(1)).unwrap_or(0)) as f64,
                "BUY_SEED" => (q * crate::view::seed_price(o.s(1)).unwrap_or(0)) as f64,
                _ => (q * v.price(o.s(1))) as f64,
            };
        } else if !o.is_empty() && o.op() == "BUY_LAND" {
            cash -= 4000.0;
        }
    }
    let mut workers = action.units();
    let day = v.step.div_euclid(24);
    for actor in 0..workers.len().min(v.positions.len()) {
        let w = workers[actor].clone();
        if w.is_empty() {
            continue;
        }
        let (x, y) = v.positions[actor];
        let t = farm.tile(x, y);
        if w.op() == "BUILD_COOP" {
            workers[actor] = Cmd::new("BUILD_PASTURE");
        } else if w.len() >= 2 && matches!(w.op(), "PICKUP" | "PLACE") && w.s(1) == "GOOSE" {
            let mut c = w.clone();
            c.0[1] = Tok::S(mode);
            workers[actor] = c;
            if w.op() == "PLACE" {
                st.pending.push((x, y, day));
            }
        } else if w.len() >= 2 && w.op() == "PLACE" && w.s(1) == "EGG" {
            let mut c = w.clone();
            c.0[1] = Tok::S(item);
            workers[actor] = c;
        } else if w.len() == 1 && w.op() == "HARVEST" && st.sites.iter().any(|(p, _)| *p == (x, y)) && t.is_dict() && t.animal == mode {
            st.credit += t.yield_units.max(0);
        }
    }
    action.set_units(workers);
    if st.credit > 0 {
        let stock = ch.projected_shed(&action, v);
        let planned: i64 = action.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum();
        let mut extra = st.credit.min((qget(&stock, item) - planned).max(0));
        if extra > 0 {
            if let Some(o) = action.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                let nv = o.n(2) + extra;
                o.set_n(2, nv);
            } else if action.market.len() < 10 {
                action.market.insert(0, Cmd::order("SELL", item, extra));
            } else {
                extra = 0;
            }
            st.credit -= extra;
        }
    }
    action
}

/// `_cs_plan`.
fn cs_plan(v: &View, ch: &Chassis, action: &Action, k: i64, mode: &str) -> Option<Plan> {
    let step = v.step;
    let farm = v.farm();
    let board = farm.rows as i64;
    let half = board / 2;
    let access = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)];
    let mut positions = v.positions.clone();
    let end = (step / 24 + 1) * 24 - 1;
    let (mut builds, mut pickups, mut places): (Vec<(i64, usize, P)>, Vec<(i64, usize, i64)>, Vec<(i64, usize, P)>) = (vec![], vec![], vec![]);
    for t in step..=end {
        let act = if t == step { Some(action) } else { tape_at(ch, v.obs.player, t) };
        let units = act.map(|a| a.units()).unwrap_or_else(|| vec![Cmd::pass()]);
        for i in 0..positions.len() {
            let cmd = units.get(i).filter(|c| !c.is_empty()).cloned().unwrap_or_else(Cmd::pass);
            let pos = positions[i];
            if let Some((dx, dy)) = crate::view::move_delta(cmd.op()) {
                let (nx, ny) = (pos.0 + dx, pos.1 + dy);
                if (0..board).contains(&nx) && (0..board).contains(&ny) {
                    positions[i] = (nx, ny);
                }
            } else if cmd.op() == "BUILD_PASTURE" {
                builds.push((t, i, pos));
            } else if cmd.len() >= 2 && cmd.op() == "PICKUP" && cmd.s(1) == "COW" {
                pickups.push((t, i, (if cmd.len() > 2 { cmd.n(2) } else { 1 }).max(1)));
            } else if cmd.len() >= 2 && cmd.op() == "PLACE" && cmd.s(1) == "COW" {
                places.push((t, i, pos));
            }
        }
        if let Some(a) = act {
            for o in &a.market {
                if !o.is_empty() && o.op() == "HIRE" {
                    let mut best = access[0];
                    let mut key = (usize::MAX, usize::MAX);
                    for (ai, a) in access.iter().enumerate() {
                        let kk = (positions.iter().filter(|p| *p == a).count(), ai);
                        if kk < key {
                            key = kk;
                            best = *a;
                        }
                    }
                    positions.push(best);
                }
            }
        }
    }
    let mut chosen: Vec<(i64, usize, P)> = vec![];
    let mut tiles: Vec<P> = vec![];
    for &(t, i, pos) in &places {
        if !tiles.contains(&pos) {
            chosen.push((t, i, pos));
            tiles.push(pos);
        }
        if chosen.len() as i64 == k {
            break;
        }
    }
    if (chosen.len() as i64) < k {
        return None;
    }
    let mut plan = Plan::default();
    let buying_land = action.market.iter().any(|o| !o.is_empty() && o.op() == "BUY_LAND");
    let unlocked: Vec<&str> = if farm.quadrants.is_empty() { vec!["NW"] } else { farm.quadrants.clone() };
    let next_q: Option<&str> = ["NE", "SW", "SE"].into_iter().find(|q| !unlocked.contains(q));
    for (t, i, pos) in chosen {
        let (x, y) = pos;
        let tile = farm.tile(x, y);
        if mode == "GOOSE" {
            let quadrant = format!("{}{}", if y < half { "N" } else { "S" }, if x < half { "W" } else { "E" });
            let lbb = tile.is_locked() && buying_land && next_q == Some(quadrant.as_str());
            if !tile.is_none() && !lbb {
                return None;
            }
            let prior: Vec<(i64, usize)> = builds.iter().filter(|(tb, _, pb)| *pb == pos && *tb < t).map(|(tb, ib, _)| (*tb, *ib)).collect();
            let &(tb, ib) = prior.last()?;
            match plan.builds.iter_mut().find(|(k2, _)| *k2 == (tb, ib)) {
                Some(e) => e.1 = pos,
                None => plan.builds.push(((tb, ib), pos)),
            }
        }
        let carrier: Vec<&(i64, usize, i64)> = pickups.iter().filter(|(tp, ip, _)| *ip == i && *tp < t).collect();
        let &&(tp, ip, _) = carrier.last()?;
        match plan.pickups.iter_mut().find(|(k2, _)| *k2 == (tp, ip)) {
            Some(e) => e.1 += 1,
            None => plan.pickups.push(((tp, ip), 1)),
        }
        match plan.places.iter_mut().find(|(k2, _)| *k2 == (t, i)) {
            Some(e) => e.1 = pos,
            None => plan.places.push(((t, i), pos)),
        }
    }
    for (key, n) in &plan.pickups {
        let q = pickups.iter().find(|(tp, ip, _)| (*tp, *ip) == *key).map(|(_, _, q)| *q).unwrap();
        if q != *n {
            return None;
        }
    }
    Some(plan)
}

#[allow(dead_code)]
fn _unused(_: Qty) {}
