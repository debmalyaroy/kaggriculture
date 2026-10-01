//! R51 finite-harvest fertilizer tour (agent 2143-2297, with `_r68_joint_plans` 2424 and
//! `_r62_input_start` 2408) and R51 warehouse close (2299-2362).
use super::v219::ParentInfo;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qget, qset, Qty};
use crate::sim;
use crate::view::{fib, move_delta, shed_adjacent, View, PRODUCTS};

const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
pub const MAX_WORKERS: i64 = 2;
type P = (i64, i64);

fn dist(a: P, b: P) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
fn walk(pos: P, t: P) -> Option<Cmd> {
    if pos.0 != t.0 {
        return Some(Cmd::new(if pos.0 < t.0 { "EAST" } else { "WEST" }));
    }
    if pos.1 != t.1 {
        return Some(Cmd::new(if pos.1 < t.1 { "SOUTH" } else { "NORTH" }));
    }
    None
}
fn spawn(positions: &[P]) -> P {
    let mut chosen = ACCESS[0];
    let mut key = (usize::MAX, usize::MAX);
    for (ai, p) in ACCESS.iter().enumerate() {
        let k = (positions.iter().filter(|q| *q == p).count(), ai);
        if k < key {
            key = k;
            chosen = *p;
        }
    }
    chosen
}
fn crops(item: &str) -> Option<(i64, i64, i64)> {
    match item {
        "WHEAT" => Some((2, 4, 6)),
        "CARROT" => Some((2, 3, 4)),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct Target {
    crop: &'static str,
    birth: i64,
    yield_: i64,
    until: i64,
    watered: bool,
    water: Vec<i64>,
    harvest: Option<i64>,
    first: i64,
    last: i64,
    cap: i64,
}

/// `(x, y, crop, birth)`.
type Stop = (i64, i64, &'static str, i64);

#[derive(Clone, Debug)]
struct Plan {
    path: Vec<Stop>,
    quantity: i64,
    loaded: bool,
}

#[derive(Clone, Debug, Default)]
struct State {
    step: i64,
    day: Option<i64>,
    /// actor -> plan (insertion order).
    workers: Vec<(usize, Plan)>,
    pending: Option<Vec<(usize, Plan)>>,
}

#[derive(Default)]
pub struct R51 {
    players: Vec<(i64, State)>,
}

impl R51 {
    pub fn pre(&mut self, v: &View) {
        let p = v.obs.player;
        match self.players.iter_mut().find(|(k, _)| *k == p) {
            Some((_, s)) if v.step > s.step => s.step = v.step,
            Some((_, s)) => *s = State { step: v.step, ..Default::default() },
            None => self.players.push((p, State { step: v.step, ..Default::default() })),
        }
    }

    /// E410: actors of the live fertilizer tour workers.
    pub fn worker_actors(&self, player: i64) -> Vec<usize> {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s.workers.iter().map(|(a, _)| *a).collect()).unwrap_or_default()
    }

    /// R85: fertilizer committed to unloaded tour plans (`{**workers, **pending}`).
    pub fn undelivered(&self, player: i64) -> i64 {
        let Some((_, s)) = self.players.iter().find(|(p, _)| *p == player) else { return 0 };
        let mut merged: Vec<(usize, &Plan)> = s.workers.iter().map(|(a, p)| (*a, p)).collect();
        for (a, p) in s.pending.iter().flatten() {
            match merged.iter_mut().find(|(k, _)| k == a) {
                Some(e) => e.1 = p,
                None => merged.push((*a, p)),
            }
        }
        merged.iter().filter(|(_, p)| !p.loaded).map(|(_, p)| p.quantity.max(0)).sum()
    }

    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis, parents: [ParentInfo; 2]) -> Action {
        let Some(i) = self.players.iter().position(|(k, _)| *k == v.obs.player) else { return action };
        let mut st = std::mem::take(&mut self.players[i].1);
        let out = control(v, action, &mut st, ch, &parents);
        self.players[i].1 = st;
        out
    }
}

fn units_of(a: &Action) -> Vec<Cmd> {
    a.units()
}

/// `_r51_input_forecast`.
fn forecast(v: &View, ch: &Chassis, route: i64, expected: usize) -> Vec<(P, Target)> {
    let step = v.step;
    let day = step.div_euclid(24);
    let farm = v.farm();
    let mut pos: Vec<P> = vec![farm.farmer.unwrap()];
    pos.extend(farm.hands.iter().take(expected).copied());
    let mut targets: Vec<(P, Target)> = vec![];
    for y in 0..farm.rows {
        for x in 0..farm.cols {
            let t = &farm.tiles[y * farm.cols + x];
            if !t.is_dict() {
                continue;
            }
            let Some((first, last, cap)) = crops(t.crop) else { continue };
            let age = day - t.planted_day;
            if 1 <= age && age < last {
                targets.push((
                    (x as i64, y as i64),
                    Target {
                        crop: t.crop,
                        birth: t.planted_day,
                        yield_: t.yield_units,
                        until: t.fertilized_until_day,
                        watered: t.watered_today,
                        water: vec![],
                        harvest: None,
                        first,
                        last,
                        cap,
                    },
                ));
            }
        }
    }
    let mut seen: Vec<(i64, P)> = vec![];
    let end = 712.min((day + 4) * 24);
    for t in step..end {
        let r = if t >= 648 { 2 } else { route };
        let a = &ch.route(r).tape[t as usize];
        let units = units_of(a);
        let n = units.len().min(pos.len());
        for actor in 0..n {
            let c = &units[actor];
            if c.is_empty() {
                continue;
            }
            let xy = pos[actor];
            if let Some((_, tg)) = targets.iter_mut().find(|(p, _)| *p == xy) {
                if tg.harvest.is_none() {
                    if c.op() == "WATER" && !seen.contains(&(t / 24, xy)) {
                        seen.push((t / 24, xy));
                        let age = t / 24 - tg.birth;
                        if !(t / 24 == day && tg.watered) && tg.first <= age && age <= tg.last {
                            tg.water.push(t);
                        }
                    }
                    if c.op() == "HARVEST" {
                        tg.harvest = Some(t);
                    }
                }
            }
            if let Some((dx, dy)) = move_delta(c.op()) {
                pos[actor] = ((pos[actor].0 + dx).clamp(0, 9), (pos[actor].1 + dy).clamp(0, 9));
            }
        }
        for o in &a.market {
            if !o.is_empty() && o.op() == "HIRE" {
                let s = spawn(&pos);
                pos.push(s);
            }
        }
        if (t + 1) % 24 == 0 {
            pos = vec![(4, 4)];
        }
    }
    targets
}

/// `_r51_input_gain`.
fn gain(t: &Target, arrival: i64, day: i64) -> i64 {
    let Some(h) = t.harvest else { return 0 };
    if h <= arrival {
        return 0;
    }
    let extra = t.water.iter().filter(|&&w| arrival < w && w <= h && day <= w / 24 && w / 24 <= day + 2 && w / 24 > t.until).count() as i64;
    let baseline = t.yield_ + t.water.iter().map(|&w| if w / 24 <= t.until { 2 } else { 1 }).sum::<i64>();
    extra.min(t.cap - baseline).max(0)
}

/// `_r62_input_start`.
fn input_start(v: &View, action: &Action, index: i64) -> (i64, P) {
    let mut positions = v.positions.clone();
    for (actor, c) in action.units().iter().enumerate().take(positions.len()) {
        if !c.is_empty() {
            if let Some((dx, dy)) = move_delta(c.op()) {
                positions[actor] = ((positions[actor].0 + dx).clamp(0, 9), (positions[actor].1 + dy).clamp(0, 9));
            }
        }
    }
    let hires = action.market.iter().filter(|o| !o.is_empty() && o.op() == "HIRE").count() as i64;
    let mut chosen = ACCESS[0];
    for _ in 0..hires + index + 1 {
        chosen = spawn(&positions);
        positions.push(chosen);
    }
    (v.step + 2, chosen)
}

#[derive(Clone)]
struct Beam {
    score: f64,
    gross: i64,
    now: i64,
    pos: P,
    path: Vec<Stop>,
    used: Vec<P>,
    wheat: i64,
    carrot: i64,
}
fn beam_cmp(a: &Beam, b: &Beam) -> std::cmp::Ordering {
    // key (-score, -gross, now, path) ascending
    b.score
        .partial_cmp(&a.score)
        .unwrap()
        .then(b.gross.cmp(&a.gross))
        .then(a.now.cmp(&b.now))
        .then(a.path.cmp(&b.path))
}

/// `_r51_input_path` -> (path, wheat units, carrot units).
fn input_path(v: &View, targets: &[(P, Target)], action: &Action, index: i64) -> (Vec<Stop>, i64, i64) {
    let day = v.step.div_euclid(24);
    let (ready, start) = input_start(v, action, index);
    let pw = (v.price("WHEAT") - 2).max(1);
    let pc = (v.price("CARROT") - 2).max(1);
    let fert = (market::price("FERTILIZER", (qget(&v.obs.mkt_inventory, "FERTILIZER") - 16) as f64) + 2).max(1) as f64;
    let mut beam = vec![Beam { score: 0.0, gross: 0, now: ready, pos: start, path: vec![], used: vec![], wheat: 0, carrot: 0 }];
    let mut best: Option<Beam> = None;
    // Candidates are ranked without materialising their path/used vectors; only the 8 survivors
    // are built. Every beam at a depth has a path of the same length, so the `path` tiebreak of
    // `beam_cmp` is (parent path, new stop).
    struct Cand {
        score: f64,
        gross: i64,
        now: i64,
        parent: usize,
        stop: Stop,
        xy: P,
        wheat: i64,
        carrot: i64,
    }
    for depth in 0..8 {
        let mut expanded: Vec<Cand> = vec![];
        for (bi, b) in beam.iter().enumerate() {
            for (xy, t) in targets {
                if b.used.contains(xy) {
                    continue;
                }
                let arrival = b.now + dist(b.pos, *xy);
                if arrival >= day * 24 + 23 {
                    continue;
                }
                let g = gain(t, arrival, day);
                if g == 0 {
                    continue;
                }
                let price = if t.crop == "WHEAT" { pw } else { pc };
                let gross = b.gross + g * price;
                expanded.push(Cand {
                    score: gross as f64 - 1.5 * fert * (b.path.len() + 1) as f64,
                    gross,
                    now: arrival + 1,
                    parent: bi,
                    stop: (xy.0, xy.1, t.crop, t.birth),
                    xy: *xy,
                    wheat: b.wheat + if t.crop == "WHEAT" { g } else { 0 },
                    carrot: b.carrot + if t.crop == "CARROT" { g } else { 0 },
                });
            }
        }
        if expanded.is_empty() {
            break;
        }
        let cmp = |a: &Cand, b: &Cand| {
            b.score
                .partial_cmp(&a.score)
                .unwrap()
                .then(b.gross.cmp(&a.gross))
                .then(a.now.cmp(&b.now))
                .then_with(|| beam[a.parent].path.cmp(&beam[b.parent].path))
                .then_with(|| a.stop.cmp(&b.stop))
        };
        if expanded.len() > 8 {
            expanded.select_nth_unstable_by(8, cmp);
            expanded.truncate(8);
        }
        expanded.sort_by(cmp);
        let next: Vec<Beam> = expanded
            .iter()
            .map(|c| {
                let p = &beam[c.parent];
                let mut path = Vec::with_capacity(p.path.len() + 1);
                path.extend_from_slice(&p.path);
                path.push(c.stop);
                let mut used = Vec::with_capacity(p.used.len() + 1);
                used.extend_from_slice(&p.used);
                used.push(c.xy);
                Beam { score: c.score, gross: c.gross, now: c.now, pos: c.xy, path, used, wheat: c.wheat, carrot: c.carrot }
            })
            .collect();
        beam = next;
        if depth >= 2 {
            let c = &beam[0];
            if best.as_ref().is_none_or(|b| beam_cmp(c, b) == std::cmp::Ordering::Less) {
                best = Some(c.clone());
            }
        }
    }
    match best {
        None => (vec![], 0, 0),
        Some(b) => (b.path, b.wheat, b.carrot),
    }
}

/// `_r68_joint_plans` -> (plans, total_q).
fn joint_plans(v: &View, action: &Action, targets: &[(P, Target)], stock: &Qty, purchases: i64, topup: i64) -> (Vec<Plan>, i64) {
    let farm = v.farm();
    let inv = |i: &str| qget(&v.obs.mkt_inventory, i);
    let stock_sum: i64 = stock.iter().map(|(_, n)| n).sum();
    let mut best: Option<((f64, f64, f64, i64, i64), Vec<Plan>, i64)> = None;
    for (mode, first_crop) in [None, Some("WHEAT"), Some("CARROT")].into_iter().enumerate() {
        let mut remaining: Vec<(P, Target)> = targets.to_vec();
        let mut plans: Vec<Plan> = vec![];
        let (mut total_q, mut total_cost, mut total_value) = (0i64, 0f64, 0f64);
        let (mut all_w, mut all_c) = (0i64, 0i64);
        for i in 0..MAX_WORKERS {
            let subset: Vec<(P, Target)> = match first_crop {
                Some(fc) if i == 0 => remaining.iter().filter(|(_, t)| t.crop == fc).cloned().collect(),
                _ => remaining.clone(),
            };
            let (path, uw, uc) = input_path(v, &subset, action, i);
            let q = path.len() as i64;
            if q < 3 || action.market.len() as i64 + 2 + i > 10 || stock_sum + purchases + total_q + q + topup > 95 {
                break;
            }
            let quote = market::price("FERTILIZER", (inv("FERTILIZER") - total_q - q - topup) as f64);
            let cost = ((q + if i == 0 { topup } else { 0 }) * (quote + 2) + fib(farm.hires_today + i)) as f64;
            let mut value = 0i64;
            for (item, n, all) in [("WHEAT", uw, all_w), ("CARROT", uc, all_c)] {
                value += n * (market::price(item, (inv(item) + all + n) as f64) - 2).max(1);
            }
            let value = value as f64;
            if value < 1.5 * cost + 50.0 || farm.money < total_cost + cost + 3000.0 {
                break;
            }
            plans.push(Plan { path: path.clone(), quantity: q, loaded: false });
            total_q += q;
            total_cost += cost;
            total_value += value;
            all_w += uw;
            all_c += uc;
            for (x, y, _, _) in &path {
                remaining.retain(|(p, _)| *p != (*x, *y));
            }
        }
        let score = (total_value - total_cost, total_value, -total_cost, -(plans.len() as i64), -(mode as i64));
        let better = match &best {
            None => true,
            Some((b, _, _)) => score.partial_cmp(b) == Some(std::cmp::Ordering::Greater),
        };
        if better {
            best = Some((score, plans, total_q));
        }
    }
    let (_, plans, q) = best.unwrap();
    (plans, q)
}

/// `_r51_input_control`.
fn control(v: &View, action: Action, st: &mut State, ch: &Chassis, parents: &[ParentInfo; 2]) -> Action {
    let step = v.step;
    let day = step.div_euclid(24);
    let hour = step.rem_euclid(24);
    let farm = v.farm();
    let Some(native) = ch.players.iter().find(|(p, _)| *p == v.obs.player).map(|(_, s)| s) else { return action };
    if st.day != Some(day) {
        st.day = Some(day);
        st.workers.clear();
        st.pending = None;
    }
    if let Some(pending) = st.pending.take() {
        for (actor, plan) in pending {
            if farm.hands.len() >= actor {
                match st.workers.iter_mut().find(|(a, _)| *a == actor) {
                    Some(e) => e.1 = plan,
                    None => st.workers.push((actor, plan)),
                }
            }
        }
    }
    if !st.workers.is_empty() {
        let mut changed = action;
        for k in 0..st.workers.len() {
            let actor = st.workers[k].0;
            if actor == 0 || actor > farm.hands.len() || actor > changed.hands.len() {
                continue; // Python: IndexError -> the wrapper returns PASS (never observed)
            }
            let pos = farm.hands[actor - 1];
            let inv_f = qget(v.inv(actor), "FERTILIZER");
            let mut cmd = Cmd::pass();
            if !st.workers[k].1.loaded {
                let stock = ch.projected_shed(&changed, v);
                let q = st.workers[k].1.quantity.min(qget(&stock, "FERTILIZER").max(0));
                if q != 0 && shed_adjacent(pos, 10) {
                    cmd = Cmd::order("PICKUP", "FERTILIZER", q);
                    st.workers[k].1.loaded = true;
                }
            } else if inv_f != 0 {
                let plan = &mut st.workers[k].1;
                while let Some(&(x, y, crop, birth)) = plan.path.first() {
                    let t = farm.tile(x, y);
                    if !t.is_dict() || t.crop != crop || t.planted_day != birth || t.fertilized_until_day >= day + 2 {
                        plan.path.remove(0);
                        continue;
                    }
                    cmd = walk(pos, (x, y)).unwrap_or_else(|| Cmd::new("FERTILIZE"));
                    if cmd.len() == 1 && cmd.op() == "FERTILIZE" {
                        plan.path.remove(0);
                    }
                    break;
                }
            }
            changed.hands[actor - 1] = cmd;
        }
        return changed;
    }
    if !(1..=3).contains(&hour) || !(12..=28).contains(&day) {
        return action;
    }
    let Some(route) = native.route else { return action };
    let tape = &ch.route(route).tape;
    let lo = ((day * 24) as usize).min(tape.len());
    let hi = (((day + 1) * 24).min(719) as usize).min(tape.len());
    let planned = &tape[lo..hi];
    let expected = planned.iter().map(|a| a.hands.len()).max().unwrap_or(0);
    let h = (hour as usize).min(planned.len());
    if planned[h..].iter().any(|a| a.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE")) || !native.pending.is_empty() {
        return action;
    }
    if day == 12 || day == 18 || parents.iter().any(|p| p.committed && p.requested_day != Some(day)) {
        return action;
    }
    if parents.iter().any(|p| p.pending) || action.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE") {
        return action;
    }
    let mut owned: Vec<usize> = (1..=expected).collect();
    for p in parents {
        if p.actors.iter().any(|a| owned.contains(a)) {
            return action;
        }
        owned.extend(p.actors.iter().copied());
    }
    owned.sort();
    owned.dedup();
    if owned != (1..=farm.hands.len()).collect::<Vec<_>>() {
        return action;
    }
    let targets = forecast(v, ch, route, expected);
    let stock = ch.projected_shed(&action, v);
    let purchases: i64 =
        action.market.iter().filter(|o| o.len() > 2 && matches!(o.op(), "BUY_PRODUCT" | "BUY_ANIMAL")).map(|o| o.n(2).max(0)).sum();
    let mut available = qget(&stock, "FERTILIZER").max(0);
    for o in &action.market {
        if o.len() >= 3 && o.op() == "SELL" && o.s(1) == "FERTILIZER" {
            available = (available - o.n(2).max(0)).max(0);
        } else if o.len() >= 3 && o.op() == "BUY_PRODUCT" && o.s(1) == "FERTILIZER" {
            available += o.n(2).max(0);
        }
    }
    let Some(next_native) = planned.get(h + 1) else { return action };
    let native_pickups: i64 = next_native
        .units()
        .iter()
        .filter(|c| c.len() > 1 && c.op() == "PICKUP" && c.s(1) == "FERTILIZER")
        .map(|c| if c.len() > 2 { c.n(2) } else { 1 }.max(0))
        .sum();
    let topup = (native_pickups - available).max(0);
    let (plans, total_q) = joint_plans(v, &action, &targets, &stock, purchases, topup);
    if plans.is_empty() {
        return action;
    }
    let n = plans.len();
    st.pending = Some(plans.into_iter().enumerate().map(|(i, p)| (farm.hands.len() + 1 + i, p)).collect());
    let mut changed = action;
    changed.market.push(Cmd::order("BUY_PRODUCT", "FERTILIZER", total_q + topup));
    for _ in 0..n {
        changed.market.push(Cmd::new("HIRE"));
    }
    changed
}

/// `_r51_close_warehouse` (hour 23, days 12-28): project the final hour's unit actions on the
/// engine and sell what would not fit the shed at midnight.
pub fn warehouse(action: Action, v: &View, ch: &Chassis) -> Action {
    let step = v.step;
    let day = step.div_euclid(24);
    if step.rem_euclid(24) != 23 || !(12..=28).contains(&day) {
        return action;
    }
    if action.market.iter().any(|o| !o.is_empty() && o.op() != "SELL") {
        return action;
    }
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let commands = action.units();
    let mut demand: Qty = vec![];
    for c in &commands {
        if c.len() > 1 && c.op() == "PLANT" {
            crate::obs::qadd(&mut demand, c.s(1), 1);
        }
    }
    let blocked: Vec<&str> = demand.iter().filter(|(k, q)| *q > qget(&v.obs.seeds, k)).map(|(k, _)| *k).collect();
    let n_inv = v.obs.invs.len();
    for (actor, c) in commands.iter().enumerate().take(n_inv) {
        let c = if c.len() > 1 && c.op() == "PLANT" && blocked.contains(&c.s(1)) { Cmd::pass() } else { c.clone() };
        sim::apply(&mut farm, &mut private, actor, &c, day);
    }
    let mut post: Qty = private.shed.0.iter().map(|(k, n)| (crate::act::intern(k), *n)).collect();
    for o in &action.market {
        if o.len() >= 3 && o.op() == "SELL" {
            let cur = qget(&post, o.s(1));
            qset(&mut post, o.s(1), (cur - o.n(2).max(0)).max(0));
        }
    }
    let carried: i64 = private.inventories.iter().flat_map(|m| m.0.iter().map(|(_, q)| (*q).max(0))).sum();
    let mut needed = post.iter().map(|(_, n)| n).sum::<i64>() + carried - 100;
    if needed <= 0 {
        return action;
    }
    let mut result = action;
    let mut items: Vec<&'static str> = PRODUCTS.iter().copied().filter(|p| *p != "WHEAT" && *p != "FERTILIZER").collect();
    items.sort_by_key(|p| -v.price(p));
    for item in items {
        let qty = needed.min(qget(&post, item));
        if qty == 0 {
            continue;
        }
        if let Some(o) = result.market.iter_mut().find(|o| o.len() >= 3 && o.op() == "SELL" && o.s(1) == item) {
            let nv = o.n(2).max(0) + qty;
            o.set_n(2, nv);
        } else if result.market.len() < 10 {
            result.market.push(Cmd::order("SELL", item, qty));
        } else {
            continue;
        }
        needed -= qty;
        let cur = qget(&post, item);
        qset(&mut post, item, cur - qty);
        if needed <= 0 {
            break;
        }
    }
    if needed > 0 {
        let Some(route) = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route) else { return result };
        let mut reserve = 0i64;
        for t in (step + 1)..719 {
            let r = if t >= 648 { 2 } else { route };
            let fut = &ch.route(r).tape[t as usize];
            for c in fut.units() {
                if c.len() > 1 && c.op() == "PICKUP" && c.s(1) == "WHEAT" {
                    reserve += if c.len() > 2 { c.n(2) } else { 1 }.max(0);
                }
            }
            if fut.market.iter().any(|o| o.len() > 1 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT") {
                break;
            }
        }
        let incoming: i64 = private.inventories.iter().map(|m| m.get("WHEAT").max(0)).sum();
        let others: i64 = post.iter().filter(|(p, _)| *p != "WHEAT").map(|(_, q)| q).sum::<i64>()
            + private.inventories.iter().flat_map(|m| m.0.iter().filter(|(p, _)| *p != "WHEAT").map(|(_, q)| (*q).max(0))).sum::<i64>();
        let pw = qget(&post, "WHEAT");
        let qty = if 100 - others >= reserve { needed.min(pw).min((pw + incoming - reserve).max(0)) } else { 0 };
        let has = result.market.iter().any(|o| o.len() >= 3 && o.op() == "SELL" && o.s(1) == "WHEAT");
        if qty != 0 && (has || result.market.len() < 10) {
            if let Some(o) = result.market.iter_mut().find(|o| o.len() >= 3 && o.op() == "SELL" && o.s(1) == "WHEAT") {
                let nv = o.n(2).max(0) + qty;
                o.set_n(2, nv);
            } else {
                result.market.push(Cmd::order("SELL", "WHEAT", qty));
            }
        }
    }
    result
}
