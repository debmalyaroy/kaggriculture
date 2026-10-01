//! V219 finite late tomato investment (agent lines 1237-1462) with its final-bound helpers:
//! `_v219_request` as wrapped by V13V (line 6098, skip days 19/21/23), `_r53_labor_assignment`
//! (2367), `_r70_parent_fert_qty` (2450), `_r79_tomato_fertilizer_worthwhile` (2518).
//! Report-only bookkeeping (seen/lost plants, telemetry) is omitted: it never feeds a decision.
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qget, Qty};
use crate::view::{fib, is_move, move_delta, View, PRODUCTS};

pub const CROP_MIN_PRICE: i64 = 70;
const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];

#[derive(Clone, Debug)]
pub struct Cfg {
    pub fertilize: bool,
    pub skip_days: Vec<i64>,
}
impl Default for Cfg {
    fn default() -> Self {
        Cfg { fertilize: true, skip_days: vec![19, 21, 23] }
    }
}

#[derive(Clone, Debug)]
pub struct Labor {
    pub paths: Vec<Vec<(i64, i64)>>,
    pub spawns: Vec<(i64, i64)>,
    pub workers: i64,
    pub fertilizer: bool,
}

#[derive(Clone, Debug)]
struct Pending {
    first_actor: usize,
    count: usize,
    crop_workers: usize,
    fertilizer: bool,
    labor: Option<Labor>,
}

#[derive(Clone, Debug)]
pub struct Role {
    pub fert_kind: bool,
    pub targets: Vec<(i64, i64)>,
    pub needs_fertilizer: bool,
    pub fertilizer_quantity: Option<i64>,
    pub loaded: bool,
    pub pickup_requested: bool,
}

#[derive(Clone, Debug)]
pub struct State {
    last_step: i64,
    day: i64,
    /// actor -> role (insertion order).
    pub workers: Vec<(usize, Role)>,
    targets: Vec<(i64, i64)>,
    pub eligible: bool,
    pub committed: bool,
    requested_day: Option<i64>,
    pending: Option<Pending>,
}

impl State {
    fn new(step: i64) -> State {
        State {
            last_step: step,
            day: -1,
            workers: vec![],
            targets: [5, 6].iter().flat_map(|&y| (5..10).map(move |x| (x, y))).collect(),
            eligible: false,
            committed: false,
            requested_day: None,
            pending: None,
        }
    }
}

/// What R51 reads of a parent project's state (`_V219_STATES` / `_V233_STATES`).
#[derive(Clone, Debug, Default)]
pub struct ParentInfo {
    pub committed: bool,
    pub requested_day: Option<i64>,
    pub pending: bool,
    pub actors: Vec<usize>,
}
#[derive(Default, Clone)]
pub struct V219 {
    pub cfg: Cfg,
    pub players: Vec<(i64, State)>,
    /// Precomputed: some route buys land or plants tomato in steps 432..719 (disqualifies).
    pub routes_block: Option<bool>,
}

fn walk(pos: (i64, i64), t: (i64, i64)) -> Option<Cmd> {
    if pos.0 != t.0 {
        return Some(Cmd::new(if pos.0 < t.0 { "EAST" } else { "WEST" }));
    }
    if pos.1 != t.1 {
        return Some(Cmd::new(if pos.1 < t.1 { "SOUTH" } else { "NORTH" }));
    }
    None
}
fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
fn home(pos: (i64, i64)) -> (i64, i64) {
    let mut best = ACCESS[0];
    for p in ACCESS {
        if dist(pos, p) < dist(pos, best) {
            best = p;
        }
    }
    best
}
fn n_hires(a: &Action) -> i64 {
    a.market.iter().filter(|o| !o.is_empty() && o.op() == "HIRE").count() as i64
}

impl V219 {
    pub fn info(&self, player: i64) -> ParentInfo {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| ParentInfo {
            committed: s.committed,
            requested_day: s.requested_day,
            pending: s.pending.is_some(),
            actors: s.workers.iter().map(|(a, _)| *a).collect(),
        }).unwrap_or_default()
    }

    /// R85 `_r85_reserve` share: fertilizer still owed to unloaded fertilizer roles, +10 for a
    /// pending fertilizer request.
    pub fn fert_dedicated(&self, player: i64, invs: &[Qty]) -> i64 {
        let Some((_, s)) = self.players.iter().find(|(p, _)| *p == player) else { return 0 };
        let mut d = 0;
        for (actor, role) in &s.workers {
            if !role.needs_fertilizer || role.loaded {
                continue;
            }
            let desired = role.fertilizer_quantity.unwrap_or(if role.fert_kind { 10 } else { 5 });
            let carried = invs.get(*actor).map(|m| qget(m, "FERTILIZER")).unwrap_or(0);
            d += (desired - carried).max(0);
        }
        if s.pending.as_ref().is_some_and(|p| p.fertilizer) {
            d += 10;
        }
        d
    }

    fn routes_block(&mut self, ch: &Chassis) -> bool {
        *self.routes_block.get_or_insert_with(|| {
            ch.routes.iter().any(|r| {
                r.tape.iter().skip(432).take(719 - 432).any(|a| {
                    a.market.iter().any(|o| !o.is_empty() && o.op() == "BUY_LAND")
                        || std::iter::once(&a.farmer).chain(a.hands.iter()).any(|c| c.len() == 2 && c.op() == "PLANT" && c.s(1) == "TOMATO")
                })
            })
        })
    }

    fn qualifies(&mut self, v: &View, ch: &Chassis, min_shops: usize) -> bool {
        let f = v.farm();
        let mut q: Vec<&str> = f.quadrants.clone();
        q.sort();
        q.dedup();
        if f.rows != 10 || q != ["NE", "NW", "SW"] {
            return false;
        }
        if f.money < 12000.0 || v.price("TOMATO") < CROP_MIN_PRICE {
            return false;
        }
        let tomato_shop = |s: &&str| matches!(*s, "PIZZA_SHOP" | "FARMERS_MARKET");
        let world_pair = min_shops >= 100 && v.shops().len() >= 2 && v.shops()[..2].iter().all(tomato_shop);
        if !world_pair && v.shops().iter().filter(|s| tomato_shop(s)).count() < min_shops % 100 {
            return false;
        }
        if [5i64, 6].iter().any(|&y| (5..10).any(|x| !f.tile(x, y).is_locked())) {
            return false;
        }
        if qget(&v.obs.seeds, "TOMATO") != 0 || qget(&v.obs.shed, "TOMATO") != 0 {
            return false;
        }
        if f.tiles.iter().any(|t| t.is_dict() && t.crop == "TOMATO") {
            return false;
        }
        !self.routes_block(ch)
    }

    /// `min_shops`: PIZZA / FARMERS_MARKET shops that must be unlocked on day 18 (knobs.v219_min_shops; v61.1: 3).
    pub fn apply(&mut self, action: Action, v: &View, ch: &Chassis, min_shops: usize) -> Action {
        let step = v.step;
        let player = v.obs.player;
        let day = step.div_euclid(24);
        let idx = match self.players.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.players[i].1.last_step => i,
            Some(i) => {
                self.players[i].1 = State::new(step);
                i
            }
            None => {
                self.players.push((player, State::new(step)));
                self.players.len() - 1
            }
        };
        let mut st = std::mem::replace(&mut self.players[idx].1, State::new(step));
        st.last_step = step;
        if step == 432 {
            st.eligible = self.qualifies(v, ch, min_shops);
            if std::env::var_os("KRL_DEBUG_V219").is_some() {
                // diagnostics only (RCA 2026-09-27): which qualification test decides
                let f = v.farm();
                let mut q: Vec<&str> = f.quadrants.clone();
                q.sort();
                q.dedup();
                let shops = v.shops().iter().filter(|s| matches!(**s, "PIZZA_SHOP" | "FARMERS_MARKET")).count();
                let row56 = [5i64, 6].iter().any(|&y| (5..10).any(|x| !f.tile(x, y).is_locked()));
                eprintln!(
                    "V219 p{} eligible={} rows={} quads={:?} money={:.0} tomato_px={} min_px={} pizza_farmers_shops={} rows56_unlocked={} tomato_seeds={} tomato_shed={} tomato_tiles={} routes_block={:?}",
                    player, st.eligible, f.rows, q, f.money, v.price("TOMATO"), CROP_MIN_PRICE, shops, row56,
                    qget(&v.obs.seeds, "TOMATO"), qget(&v.obs.shed, "TOMATO"), f.tiles.iter().filter(|t| t.is_dict() && t.crop == "TOMATO").count(), self.routes_block
                );
            }
        }
        let out = if !st.eligible || day < 18 { action } else { self.run(&mut st, action, v, ch, step, day) };
        self.players[idx].1 = st;
        out
    }

    fn run(&self, st: &mut State, action: Action, v: &View, ch: &Chassis, step: i64, day: i64) -> Action {
        if st.day != day {
            st.day = day;
            st.workers.clear();
        }
        let farm = v.farm();
        if let Some(p) = st.pending.take() {
            if farm.hands.len() + 1 >= p.first_actor + p.count && farm.quadrants.contains(&"SE") {
                for index in 0..p.count {
                    let fw = index == p.crop_workers;
                    let mut targets: Vec<(i64, i64)> = if fw || p.crop_workers == 1 {
                        st.targets.clone()
                    } else if p.crop_workers == 2 {
                        st.targets[index * 5..(index * 5 + 5).min(st.targets.len())].to_vec()
                    } else {
                        let g: [&[(i64, i64)]; 3] = [&[(5, 5), (6, 5), (7, 5)], &[(8, 5), (9, 5), (9, 6), (8, 6)], &[(5, 6), (6, 6), (7, 6)]];
                        g[index].to_vec()
                    };
                    let mut role = Role {
                        fert_kind: fw,
                        targets: vec![],
                        needs_fertilizer: p.fertilizer && (day == 24 || fw),
                        fertilizer_quantity: None,
                        loaded: false,
                        pickup_requested: false,
                    };
                    if let Some(l) = &p.labor {
                        targets = l.paths[index].clone();
                        role.needs_fertilizer = l.fertilizer;
                        role.fertilizer_quantity = Some(targets.len() as i64);
                    }
                    role.targets = targets;
                    let key = p.first_actor + index;
                    match st.workers.iter_mut().find(|(k, _)| *k == key) {
                        Some(e) => e.1 = role,
                        None => st.workers.push((key, role)),
                    }
                }
            }
        }
        let mut action = self.request(v, action, st, ch, step, day);
        if !st.workers.is_empty() {
            let mut commands = action.units();
            while commands.len() < farm.hands.len() + 1 {
                commands.push(Cmd::pass());
            }
            let mut workers = std::mem::take(&mut st.workers);
            for (actor, role) in workers.iter_mut() {
                if *actor >= commands.len() {
                    continue;
                }
                commands[*actor] = worker(v, *actor, role, step, day);
            }
            st.workers = workers;
            action.set_units(commands);
        }
        if st.committed
            && action.market.len() < 10
            && !action.market.iter().any(|o| o.op() == "SELL" && o.s(1) == "TOMATO")
        {
            let q = qget(&ch.projected_shed(&action, v), "TOMATO");
            if q > 0 {
                action.market.push(Cmd::order("SELL", "TOMATO", q));
            }
        }
        action
    }

    /// `_v219_request` (V13V-wrapped).
    fn request(&self, v: &View, action: Action, st: &mut State, ch: &Chassis, step: i64, day: i64) -> Action {
        let farm = v.farm();
        // V13V skip days.
        if self.cfg.skip_days.contains(&day) && st.committed && st.requested_day != Some(day) {
            let mut tomatoes = 0;
            let mut safe = true;
            for &(x, y) in &st.targets {
                let t = farm.tile(x, y);
                if t.is_dict() && t.crop == "TOMATO" {
                    tomatoes += 1;
                    if t.consecutive_unwatered != 0 {
                        safe = false;
                    }
                }
            }
            if tomatoes > 0 && safe {
                st.requested_day = Some(day);
                return action;
            }
        }
        let offset = step.rem_euclid(24);
        if !st.committed && day != 18 {
            return action;
        }
        if st.requested_day == Some(day) {
            return action;
        }
        let Some(route) = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route) else {
            return action;
        };
        let tape = &ch.route(route).tape;
        let lo = ((day * 24) as usize).min(tape.len());
        let hi = (((day + 1) * 24).min(719) as usize).min(tape.len());
        let planned = &tape[lo..hi];
        let latest_hire = planned
            .iter()
            .enumerate()
            .filter(|(_, a)| a.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE"))
            .map(|(i, _)| i as i64)
            .max()
            .unwrap_or(-1);
        let deadline = if st.committed && 3 < latest_hire && latest_hire <= 6 { 6 } else { 3 };
        if offset > deadline {
            return action;
        }
        let from = ((offset + 1) as usize).min(planned.len());
        if planned[from..].iter().any(|a| a.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE")) {
            return action;
        }
        let parent_hires = n_hires(&action);
        let expected = planned.iter().map(|a| a.hands.len() as i64).max().unwrap_or(0);
        if farm.hands.len() as i64 + parent_hires != expected {
            return action;
        }
        let fertilizer = self.cfg.fertilize && (day == 24 || day == 27) && r79_worthwhile(v, &action, day);
        let mut crop_workers: i64 = if [19, 20, 21, 22, 23, 25].contains(&day) && offset <= 2 {
            1
        } else if (26..=28).contains(&day) {
            3
        } else {
            2
        };
        let labor = r53_labor(v, &action, fertilizer, step, day);
        if let Some(l) = &labor {
            crop_workers = l.workers;
        }
        let count = crop_workers + (fertilizer && day == 27 && labor.is_none()) as i64;
        let mut extra: Vec<Cmd> = vec![];
        if !st.committed {
            extra.push(Cmd::new("BUY_LAND"));
            extra.push(Cmd::order("BUY_SEED", "TOMATO", 10));
        }
        let fq = if fertilizer { r70_fert_qty(v, &action, ch, planned, offset) } else { 0 };
        if fertilizer {
            extra.push(Cmd::order("BUY_PRODUCT", "FERTILIZER", fq));
        }
        for _ in 0..count {
            extra.push(Cmd::new("HIRE"));
        }
        if action.market.len() + extra.len() > 10 {
            return action;
        }
        let h0 = farm.hires_today;
        let mut budget: i64 = (h0..h0 + parent_hires + count).map(fib).sum();
        if !st.committed {
            budget += 4500;
        }
        if fertilizer {
            budget += fq * (v.price("FERTILIZER") + 5);
        }
        for o in &action.market {
            if o.is_empty() {
                continue;
            }
            let q = o.n(2);
            match o.op() {
                "BUY_PRODUCT" => budget += q * (v.price(o.s(1)) + 10),
                "BUY_ANIMAL" => budget += q * crate::view::animal_cost(o.s(1)).unwrap_or(0),
                "BUY_SEED" => budget += q * crate::view::seed_price(o.s(1)).unwrap_or(0),
                _ => {}
            }
        }
        if farm.money < (budget + 3000) as f64 {
            return action;
        }
        st.pending = Some(Pending {
            first_actor: (expected + 1) as usize,
            count: count as usize,
            crop_workers: crop_workers as usize,
            fertilizer,
            labor,
        });
        st.requested_day = Some(day);
        st.committed = true;
        let mut changed = action;
        changed.market.extend(extra);
        changed
    }
}

/// `_v219_worker`.
fn worker(v: &View, actor: usize, role: &mut Role, step: i64, day: i64) -> Cmd {
    let pos = v.positions[actor];
    let inv = v.inv(actor);
    let fert = qget(inv, "FERTILIZER");
    if role.needs_fertilizer && !role.loaded {
        let h = home(pos);
        if let Some(w) = walk(pos, h) {
            return w;
        }
        let desired = role.fertilizer_quantity.unwrap_or(if role.fert_kind { 10 } else { 5 });
        if fert >= desired {
            role.loaded = true;
        } else if role.pickup_requested {
            role.loaded = true;
        } else if v.shed("FERTILIZER") >= desired {
            role.pickup_requested = true;
            return Cmd::order("PICKUP", "FERTILIZER", desired);
        } else {
            role.loaded = true;
        }
    }
    let mut todo: Vec<((i64, i64), Cmd, usize)> = vec![];
    for (ti, &(x, y)) in role.targets.iter().enumerate() {
        let tile = v.farm().tile(x, y);
        let tomato = tile.is_dict() && tile.crop == "TOMATO";
        let mut command = None;
        if role.fert_kind {
            if tomato && tile.fertilized_until_day < day + 2 && fert > 0 {
                command = Some(Cmd::new("FERTILIZE"));
            }
        } else if day == 18 && !tomato {
            if tile.is_none() && qget(&v.obs.seeds, "TOMATO") > 0 {
                command = Some(Cmd(vec![crate::act::Tok::S("PLANT"), crate::act::Tok::S("TOMATO")]));
            } else if tile.is_dict() && tile.kind == "WEED" {
                command = Some(Cmd::new("DIG"));
            }
        } else if tomato {
            if day < 29 && !tile.watered_today {
                command = Some(Cmd::new("WATER"));
            } else if role.needs_fertilizer && tile.fertilized_until_day < day + 2 && fert > 0 {
                command = Some(Cmd::new("FERTILIZE"));
            } else if tile.yield_units > 0 {
                command = Some(Cmd::new("HARVEST"));
            }
        }
        if let Some(c) = command {
            // targets.index(v[0]) = the FIRST occurrence of that target.
            let first = role.targets.iter().position(|t| *t == (x, y)).unwrap_or(ti);
            todo.push(((x, y), c, first));
        }
    }
    let h = home(pos);
    let distance = dist(pos, h);
    let tom = qget(inv, "TOMATO");
    let place = || Cmd::order("PLACE", "TOMATO", tom);
    if step >= 718 - distance && tom != 0 {
        return walk(pos, h).unwrap_or_else(place);
    }
    if !todo.is_empty() {
        let mut best = 0;
        for i in 1..todo.len() {
            let k = (dist(pos, todo[i].0), todo[i].2);
            if k < (dist(pos, todo[best].0), todo[best].2) {
                best = i;
            }
        }
        let (t, c, _) = todo.swap_remove(best);
        return walk(pos, t).unwrap_or(c);
    }
    if tom != 0 {
        return walk(pos, h).unwrap_or_else(place);
    }
    if inv.iter().any(|(_, n)| *n != 0) {
        return walk(pos, h).unwrap_or_else(|| Cmd::new("DROP"));
    }
    Cmd::pass()
}

/// `_r53_labor_assignment`.
pub fn r53_labor(v: &View, action: &Action, fertilizer: bool, step: i64, day: i64) -> Option<Labor> {
    if !(26..=28).contains(&day) || step.rem_euclid(24) > 2 {
        return None;
    }
    if day == 27 && !fertilizer {
        return None;
    }
    let count = if fertilizer { 3 } else { 2 };
    let mut positions: Vec<(i64, i64)> = v.positions.clone();
    let units = action.units();
    for (i, c) in units.iter().enumerate().take(positions.len()) {
        if !c.is_empty() {
            if let Some((dx, dy)) = move_delta(c.op()) {
                positions[i] = ((positions[i].0 + dx).clamp(0, 9), (positions[i].1 + dy).clamp(0, 9));
            }
        }
    }
    let native_hires = n_hires(action);
    let mut spawns = vec![];
    for i in 0..native_hires + count {
        let mut chosen = ACCESS[0];
        let mut key = (i64::MAX, usize::MAX);
        for (ai, p) in ACCESS.iter().enumerate() {
            let k = (positions.iter().filter(|q| *q == p).count() as i64, ai);
            if k < key {
                key = k;
                chosen = *p;
            }
        }
        positions.push(chosen);
        if i >= native_hires {
            spawns.push(chosen);
        }
    }
    let groups: Vec<Vec<(i64, i64)>> = if fertilizer {
        vec![vec![(5, 5), (6, 5), (7, 5), (8, 5)], vec![(9, 5), (9, 6), (8, 6)], vec![(5, 6), (6, 6), (7, 6)]]
    } else {
        vec![(5..10).map(|x| (x, 5)).collect(), (5..10).map(|x| (x, 6)).collect()]
    };
    let remaining = 23 - step.rem_euclid(24);
    let mut best: Option<(i64, i64, Vec<Vec<(i64, i64)>>)> = None;
    for perm in permutations(groups.len()) {
        let assignment: Vec<Vec<(i64, i64)>> = perm.iter().map(|&i| groups[i].clone()).collect();
        let costs: Vec<i64> = spawns
            .iter()
            .zip(assignment.iter())
            .map(|(s, path)| {
                let mut d = dist(*s, path[0]);
                d += path.windows(2).map(|w| dist(w[0], w[1])).sum::<i64>();
                d += ACCESS.iter().map(|a| dist(*path.last().unwrap(), *a)).min().unwrap();
                d + (if fertilizer { 3 } else { 2 }) * path.len() as i64 + 1 + fertilizer as i64
            })
            .collect();
        let mx = costs.iter().copied().max().unwrap_or(0);
        if mx <= remaining {
            let cand = (mx, costs.iter().sum::<i64>(), assignment);
            if best.as_ref().is_none_or(|b| cand < *b) {
                best = Some(cand);
            }
        }
    }
    let (_, _, paths) = best?;
    let _ = is_move;
    Some(Labor { paths, spawns, workers: count, fertilizer })
}

/// `itertools.permutations(range(n))` in its lexicographic order.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(cur: &mut Vec<usize>, used: &mut Vec<bool>, n: usize, out: &mut Vec<Vec<usize>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !used[i] {
                used[i] = true;
                cur.push(i);
                rec(cur, used, n, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = vec![];
    rec(&mut vec![], &mut vec![false; n], n, &mut out);
    out
}

/// `_r70_parent_fert_qty`.
fn r70_fert_qty(v: &View, action: &Action, ch: &Chassis, planned: &[Action], offset: i64) -> i64 {
    let mut stock: Qty = ch.projected_shed(action, v);
    for o in &action.market {
        if o.len() < 3 {
            continue;
        }
        let (op, item, q) = (o.op(), o.s(1), o.n(2).max(0));
        let total: i64 = stock.iter().map(|(_, n)| n).sum();
        match op {
            "SELL" => {
                let cur = qget(&stock, item);
                crate::obs::qset(&mut stock, item, (cur - q).max(0));
            }
            "BUY_PRODUCT" | "BUY_ANIMAL" => {
                let cur = qget(&stock, item);
                crate::obs::qset(&mut stock, item, cur + q.min((100 - total).max(0)));
            }
            _ => {}
        }
    }
    let nxt = planned.get((offset + 1) as usize);
    let native_need: i64 = nxt
        .map(|a| {
            a.units()
                .iter()
                .filter(|c| c.len() > 1 && c.op() == "PICKUP" && c.s(1) == "FERTILIZER")
                .map(|c| if c.len() > 2 { c.n(2) } else { 1 }.max(0))
                .sum()
        })
        .unwrap_or(0);
    let quantity = 10.max(10 + native_need - qget(&stock, "FERTILIZER").max(0));
    let total: i64 = stock.iter().map(|(_, n)| n).sum();
    if quantity > (100 - total).max(0) {
        return 10;
    }
    quantity
}

/// `_r79_tomato_fertilizer_worthwhile`.
fn r79_worthwhile(v: &View, action: &Action, day: i64) -> bool {
    if v.price("FERTILIZER") <= 30 {
        return true;
    }
    let farm = v.farm();
    let mut bonus = 0i64;
    for y in [5, 6] {
        for x in 5..10 {
            let t = farm.tile(x, y);
            if !t.is_dict() || t.crop != "TOMATO" {
                continue;
            }
            let birth = t.planted_day;
            let until = t.fertilized_until_day;
            bonus += (day..day + 3).filter(|d| until < *d && (8..=11).contains(&(d + 1 - birth))).count() as i64;
        }
    }
    if bonus == 0 {
        return false;
    }
    let inv = |i: &str| qget(&v.obs.mkt_inventory, i) as f64;
    let price = (market::price("TOMATO", inv("TOMATO") + bonus as f64 + 10.0) - 2).max(1);
    let fert = (market::price("FERTILIZER", inv("FERTILIZER") - 10.0) + 2).max(1);
    let extra_labor = fib(farm.hires_today + n_hires(action) + 3);
    bonus * price >= 2 * (10 * fert + extra_labor) + 100
}

#[allow(dead_code)]
fn _products() -> [&'static str; 9] {
    PRODUCTS
}
