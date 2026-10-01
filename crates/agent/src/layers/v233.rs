//! V233 six-sheep SE expansion (agent lines 1950-2141) in its FINAL composition:
//! eligibility = VE (3990) over base (1959); request = VT (4058) ∘ SL (3890) ∘ base (1971);
//! worker = VT (4082) ∘ SL (3924) ∘ base (2014); rescue = VT (4124) ∘ base (2047).
//! Uses `_r62_input_start` (2408) and `_sl_path` (3886).
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::obs::{qget, Qty};
use crate::view::{animal_cost, fib, move_delta, seed_price, View};

const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const SL_TILES: [(i64, i64); 6] = [(5, 5), (6, 5), (7, 5), (7, 6), (6, 6), (5, 6)];
const VT_TILES: [(i64, i64); 6] = [(5, 5), (6, 5), (7, 5), (5, 6), (6, 6), (7, 6)];

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
/// `_v219_home`: nearest access tile, first in ACCESS order on ties.
fn home219(pos: P) -> P {
    let mut best = ACCESS[0];
    for p in ACCESS {
        if dist(pos, p) < dist(pos, best) {
            best = p;
        }
    }
    best
}
/// base worker's home: min by (distance, tile) — tie broken by the tile tuple.
fn home_by_tuple(pos: P) -> P {
    *ACCESS.iter().min_by_key(|p| (dist(pos, **p), **p)).unwrap()
}
fn path_cost(pos: P, path: &[P]) -> i64 {
    if path.is_empty() {
        return 0;
    }
    dist(pos, path[0]) + path.windows(2).map(|w| dist(w[0], w[1])).sum::<i64>()
}
/// `_sl_path`: the permutation of `targets` minimising (walk length, path).
pub fn sl_path(pos: P, targets: &[P]) -> Vec<P> {
    if targets.is_empty() {
        return vec![];
    }
    let mut best: Option<(i64, Vec<P>)> = None;
    let mut idx: Vec<usize> = (0..targets.len()).collect();
    permute(&mut idx, 0, &mut |perm| {
        let path: Vec<P> = perm.iter().map(|&i| targets[i]).collect();
        let key = (path_cost(pos, &path), path);
        if best.as_ref().is_none_or(|b| key < *b) {
            best = Some(key);
        }
    });
    best.unwrap().1
}
fn permute(a: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
    // Order does not matter: the minimum over (cost, path) is unique.
    if k == a.len() {
        f(a);
        return;
    }
    for i in k..a.len() {
        a.swap(k, i);
        permute(a, k + 1, f);
        a.swap(k, i);
    }
}
/// `_r62_input_start(obs, action, index)` -> (ready step, spawn tile).
fn r62_input_start(v: &View, action: &Action, index: i64) -> (i64, P) {
    let mut positions = v.positions.clone();
    for (actor, c) in action.units().iter().enumerate().take(positions.len()) {
        if !c.is_empty() {
            if let Some((dx, dy)) = move_delta(c.op()) {
                positions[actor] = ((positions[actor].0 + dx).clamp(0, 9), (positions[actor].1 + dy).clamp(0, 9));
            }
        }
    }
    let native_hires = n_hires(action);
    let mut chosen = ACCESS[0];
    for _ in 0..native_hires + index + 1 {
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
    (v.step + 2, chosen)
}
fn n_hires(a: &Action) -> i64 {
    a.market.iter().filter(|o| !o.is_empty() && o.op() == "HIRE").count() as i64
}

#[derive(Clone, Debug)]
struct Pending {
    first: usize,
    initial: bool,
    count: usize,
    targets: Vec<P>,
}
#[derive(Clone, Debug)]
struct Work {
    step: i64,
    op: &'static str,
    inventory: Qty,
}
#[derive(Clone, Debug)]
struct State {
    last_step: i64,
    day: i64,
    workers: Vec<(usize, Vec<P>)>,
    work: Vec<(usize, Work)>,
    credit_wool: i64,
    credit_fert: i64,
    rescue_today: i64,
    committed: bool,
    requested_day: Option<i64>,
    pending: Option<Pending>,
}
impl State {
    fn new(step: i64) -> State {
        State {
            last_step: step,
            day: -1,
            workers: vec![],
            work: vec![],
            credit_wool: 0,
            credit_fert: 0,
            rescue_today: 0,
            committed: false,
            requested_day: None,
            pending: None,
        }
    }
}

#[derive(Default, Clone)]
pub struct V233 {
    players: Vec<(i64, State)>,
}

fn native_day<'a>(ch: &'a Chassis, route: i64, day: i64) -> &'a [Action] {
    let tape = &ch.route(route).tape;
    let lo = ((day * 24) as usize).min(tape.len());
    let hi = (((day + 1) * 24).min(719) as usize).min(tape.len());
    &tape[lo..hi]
}

impl V233 {
    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let step = v.step;
        let player = v.obs.player;
        let day = step.div_euclid(24);
        let i = match self.players.iter().position(|(p, _)| *p == player) {
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
        let mut st = std::mem::replace(&mut self.players[i].1, State::new(step));
        st.last_step = step;
        let out = if day < 11 { action } else { run(&mut st, action, v, ch, step, day) };
        self.players[i].1 = st;
        out
    }
    pub fn info(&self, player: i64) -> super::v219::ParentInfo {
        self.players.iter().find(|(p, _)| *p == player).map(|(_, s)| super::v219::ParentInfo {
            committed: s.committed,
            requested_day: s.requested_day,
            pending: s.pending.is_some(),
            actors: s.workers.iter().map(|(a, _)| *a).collect(),
        }).unwrap_or_default()
    }
    pub fn committed(&self, player: i64) -> bool {
        self.players.iter().any(|(p, s)| *p == player && s.committed)
    }
}

fn run(st: &mut State, action: Action, v: &View, ch: &Chassis, step: i64, day: i64) -> Action {
    let farm = v.farm();
    let invs = &v.obs.invs;
    if st.day != day {
        st.day = day;
        st.workers.clear();
        st.work.clear();
        st.rescue_today = 0;
    }
    for (actor, prev) in &st.work {
        if prev.step != step - 1 || *actor >= invs.len() {
            continue;
        }
        let item = match prev.op {
            "HARVEST" => "WOOL",
            "COLLECT_FERTILIZER" => "FERTILIZER",
            _ => continue,
        };
        let gained = (qget(&invs[*actor], item) - qget(&prev.inventory, item)).max(0);
        if item == "WOOL" {
            st.credit_wool += gained;
        } else {
            st.credit_fert += gained;
        }
    }
    if let Some(p) = st.pending.take() {
        let funded = farm.quadrants.contains(&"SE") && (!p.initial || qget(&v.obs.shed, "SHEEP") >= 6);
        if funded && farm.hands.len() + 1 >= p.first + p.count {
            if p.count == 1 {
                set_worker(&mut st.workers, p.first, p.targets.clone());
            } else {
                for k in 0..2 {
                    set_worker(&mut st.workers, p.first + k, (5..8).map(|x| (x, 5 + k as i64)).collect());
                }
            }
            if p.initial {
                st.committed = true;
            }
        }
    }
    let route = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route);
    let action = match route {
        Some(r) => request_vt(v, action, st, ch, r, step, day).0,
        None => action,
    };
    if !st.committed {
        return action;
    }
    let mut result = action;
    let mut commands = result.units();
    while commands.len() < farm.hands.len() + 1 {
        commands.push(Cmd::pass());
    }
    st.work.clear();
    for (actor, targets) in &st.workers {
        if *actor == 0 || *actor > farm.hands.len() {
            continue; // Python would index hands[actor-1] (wrong actor / IndexError); never happens
        }
        let c = worker_vt(v, *actor, targets, step, day);
        let op = c.op();
        commands[*actor] = c;
        let inventory = invs.get(*actor).cloned().unwrap_or_default();
        st.work.push((*actor, Work { step, op, inventory }));
    }
    result.set_units(commands);
    result = rescue_vt(v, result, st, ch, step, day);
    let stock = ch.projected_shed(&result, v);
    for item in ["WOOL", "FERTILIZER"] {
        let scheduled: i64 = result.market.iter().filter(|o| o.op() == "SELL" && o.s(1) == item && o.len() >= 2).map(|o| o.n(2)).sum();
        let credit = if item == "WOOL" { st.credit_wool } else { st.credit_fert };
        let count = credit.min((qget(&stock, item) - scheduled).max(0));
        if count != 0 && result.market.len() < 10 {
            result.market.push(Cmd::order("SELL", item, count));
            if item == "WOOL" {
                st.credit_wool -= count;
            } else {
                st.credit_fert -= count;
            }
        }
    }
    result
}

fn set_worker(w: &mut Vec<(usize, Vec<P>)>, k: usize, t: Vec<P>) {
    match w.iter_mut().find(|(a, _)| *a == k) {
        Some(e) => e.1 = t,
        None => w.push((k, t)),
    }
}

// ---- eligibility --------------------------------------------------------------------------
/// base `_v233_eligible`; `clean` = the VE view with every SHEEP removed from shed/inventories.
fn eligible_base(v: &View, ch: &Chassis, route: i64, clean: bool) -> bool {
    let f = v.farm();
    let mut q: Vec<&str> = f.quadrants.clone();
    q.sort();
    q.dedup();
    if f.rows != 10 || q != ["NE", "NW", "SW"] {
        return false;
    }
    if v.shops().iter().filter(|s| **s == "YARN_STORE").count() < 2 || v.price("WOOL") < 220 || v.price("WHEAT") > 45 {
        return false;
    }
    if [5i64, 6].iter().any(|&y| (5..8).any(|x| !f.tile(x, y).is_locked())) {
        return false;
    }
    if !clean && (qget(&v.obs.shed, "SHEEP") != 0 || v.obs.invs.iter().any(|m| qget(m, "SHEEP") != 0)) {
        return false;
    }
    for day in 12..30 {
        for a in native_day(ch, route, day) {
            if a.market.iter().any(|o| !o.is_empty() && (o.op() == "BUY_LAND" || (o.op() == "BUY_ANIMAL" && o.s(1) == "SHEEP"))) {
                return false;
            }
            if std::iter::once(&a.farmer)
                .chain(a.hands.iter())
                .any(|c| !c.is_empty() && matches!(c.op(), "PICKUP" | "PLACE") && c.len() > 1 && c.s(1) == "SHEEP")
            {
                return false;
            }
        }
    }
    true
}

/// VE `_v233_eligible` (day 11 variant).
fn eligible(v: &View, ch: &Chassis, route: i64) -> bool {
    let step = v.step;
    if step.div_euclid(24) != 11 {
        return eligible_base(v, ch, route, false);
    }
    let tape = &ch.route(route).tape;
    let held = qget(&v.obs.shed, "SHEEP") + v.obs.invs.iter().map(|m| qget(m, "SHEEP")).sum::<i64>();
    let mut pickups = 0;
    for t in step..(12 * 24) {
        let Some(a) = tape.get(t as usize) else { break };
        for c in std::iter::once(&a.farmer).chain(a.hands.iter()) {
            if !c.is_empty() && c.op() == "PICKUP" && c.len() > 1 && c.s(1) == "SHEEP" {
                pickups += if c.len() > 2 { c.n(2) } else { 1 };
            }
        }
    }
    if held > pickups {
        return false;
    }
    if !eligible_base(v, ch, route, true) {
        return false;
    }
    let mut spend = 0i64;
    let mut hires: Vec<(i64, i64)> = vec![];
    for t in (step + 1)..(tape.len() as i64).min(13 * 24) {
        for o in &tape[t as usize].market {
            if o.is_empty() {
                continue;
            }
            if o.op() == "BUY_LAND" || (o.op() == "BUY_ANIMAL" && o.s(1) == "SHEEP") {
                return false;
            }
            let q = o.n(2);
            match o.op() {
                "BUY_SEED" => spend += q * seed_price(o.s(1)).unwrap_or(100),
                "BUY_PRODUCT" => {
                    let p = if v.obs.prices.iter().any(|(k, _)| *k == o.s(1)) { v.price(o.s(1)) } else { 50 };
                    spend += q * (p + 10)
                }
                "BUY_ANIMAL" => spend += q * animal_cost(o.s(1)).unwrap_or(500),
                "HIRE" => {
                    let d = t / 24;
                    let h = hires.iter().find(|(k, _)| *k == d).map(|(_, n)| *n).unwrap_or(0);
                    spend += fib(h);
                    match hires.iter_mut().find(|(k, _)| *k == d) {
                        Some(e) => e.1 += 1,
                        None => hires.push((d, 1)),
                    }
                }
                _ => {}
            }
        }
    }
    v.money() >= (10000 + spend) as f64
}

// ---- request ------------------------------------------------------------------------------
/// base `_v233_request`; returns (action, changed).
fn request_base(v: &View, action: Action, st: &mut State, ch: &Chassis, route: i64, step: i64, day: i64) -> (Action, bool) {
    let hour = step.rem_euclid(24);
    if st.requested_day == Some(day) {
        return (action, false);
    }
    let committed = st.committed;
    if !committed && (hour > (if day == 11 { 3 } else { 1 }) || !(day == 11 || day == 12) || !eligible(v, ch, route)) {
        return (action, false);
    }
    let planned = native_day(ch, route, day);
    let mut deadline = if committed { 2 } else if day == 11 { 3 } else { 1 };
    if committed {
        let last = planned
            .iter()
            .enumerate()
            .filter(|(_, a)| a.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE"))
            .map(|(h, _)| h as i64)
            .max()
            .unwrap_or(0);
        if 2 < last && last <= 6 {
            deadline = 6;
        }
    }
    if hour > deadline {
        return (action, false);
    }
    let from = ((hour + 1) as usize).min(planned.len());
    if planned[from..].iter().any(|a| a.market.iter().any(|o| !o.is_empty() && o.op() == "HIRE")) {
        return (action, false);
    }
    let farm = v.farm();
    let parent_hires = n_hires(&action);
    let expected = planned.iter().map(|a| a.hands.len() as i64).max().unwrap_or(0);
    if farm.hands.len() as i64 + parent_hires != expected {
        return (action, false);
    }
    let initial = !committed;
    let mut extra: Vec<Cmd> = vec![];
    if initial {
        extra.push(Cmd::new("BUY_LAND"));
        extra.push(Cmd::order("BUY_ANIMAL", "SHEEP", 6));
    }
    extra.push(Cmd::order("BUY_PRODUCT", "WHEAT", 6));
    extra.push(Cmd::new("HIRE"));
    extra.push(Cmd::new("HIRE"));
    if action.market.len() + extra.len() > 10 {
        return (action, false);
    }
    let stock = ch.projected_shed(&action, v);
    let mut incoming = 6 + 6 * initial as i64;
    let mut budget = 7000 * initial as i64 + 6 * (v.price("WHEAT") + 10);
    let h0 = farm.hires_today;
    budget += (h0..h0 + parent_hires + 2).map(fib).sum::<i64>();
    for o in &action.market {
        if o.is_empty() {
            continue;
        }
        let q = o.n(2);
        match o.op() {
            "BUY_LAND" => return (action, false),
            "BUY_PRODUCT" => {
                incoming += q;
                budget += q * (v.price(o.s(1)) + 10);
            }
            "BUY_ANIMAL" => {
                incoming += q;
                budget += q * animal_cost(o.s(1)).unwrap_or(0);
            }
            "BUY_SEED" => budget += q * seed_price(o.s(1)).unwrap_or(0),
            _ => {}
        }
    }
    if stock.iter().map(|(_, n)| n).sum::<i64>() + incoming > 100 {
        return (action, false);
    }
    if farm.money < (budget + if initial { 3000 } else { 1000 }) as f64 {
        return (action, false);
    }
    st.requested_day = Some(day);
    st.pending = Some(Pending { first: (expected + 1) as usize, initial, count: 2, targets: vec![] });
    let mut result = action;
    result.market.extend(extra);
    (result, true)
}

/// SL `_v233_request`: a compact one-hand service day when all six sheep are empty.
fn request_sl(v: &View, action: Action, st: &mut State, ch: &Chassis, route: i64, step: i64, day: i64) -> (Action, bool) {
    let parent = action.clone();
    let (result, changed) = request_base(v, action, st, ch, route, step, day);
    let Some(p) = st.pending.as_ref() else { return (result, changed) };
    if !changed || p.initial {
        return (result, changed);
    }
    let farm = v.farm();
    let tiles: Vec<_> = SL_TILES.iter().map(|&(x, y)| farm.tile(x, y)).collect();
    if !tiles.iter().all(|t| t.is_dict() && t.animal == "SHEEP" && t.yield_units == 0) {
        return (result, changed);
    }
    let (ready, spawn) = r62_input_start(v, &parent, 0);
    let path = sl_path(spawn, &SL_TILES);
    let travel = path_cost(spawn, &path);
    let mandatory = tiles.iter().filter(|t| !t.fed_today).count() as i64 + tiles.iter().filter(|t| !t.cared_today).count() as i64;
    let available = ((day + 1) * 24).min(719) - ready;
    if travel + mandatory > available {
        return (result, changed);
    }
    let saved = fib(farm.hires_today + n_hires(&parent) + 1);
    let last = *path.last().unwrap();
    let delivery = if day == 29 { dist(last, home219(last)) + 1 } else { 0 };
    let possible = (available - travel - mandatory - delivery).min(6).max(0);
    if saved <= (6 - possible) * v.price("FERTILIZER") {
        return (result, changed);
    }
    let mut out = result;
    out.market.pop();
    let p = st.pending.as_mut().unwrap();
    p.count = 1;
    p.targets = path;
    (out, true)
}

/// VT `_v233_request`: day 29 hires nothing without wool, else one harvest-only hand.
fn request_vt(v: &View, action: Action, st: &mut State, ch: &Chassis, route: i64, step: i64, day: i64) -> (Action, bool) {
    let parent = action.clone();
    let (result, changed) = request_sl(v, action, st, ch, route, step, day);
    if !changed || day != 29 || !st.committed {
        return (result, changed);
    }
    match st.pending.as_ref() {
        Some(p) if !p.initial => {}
        _ => return (result, changed),
    }
    let farm = v.farm();
    let wool: Vec<P> = VT_TILES
        .iter()
        .copied()
        .filter(|&(x, y)| {
            let t = farm.tile(x, y);
            t.is_dict() && t.animal == "SHEEP" && t.yield_units > 0
        })
        .collect();
    if wool.is_empty() {
        st.pending = None;
        return (parent, false);
    }
    let mut out = parent.clone();
    out.market.push(Cmd::new("HIRE"));
    let spawn = r62_input_start(v, &parent, 0).1;
    let p = st.pending.as_mut().unwrap();
    p.count = 1;
    p.targets = sl_path(spawn, &wool);
    (out, true)
}

// ---- workers ------------------------------------------------------------------------------
fn place(item: &str, n: i64) -> Cmd {
    Cmd::order("PLACE", item, n)
}

/// base `_v233_worker`.
fn worker_base(v: &View, actor: usize, targets: &[P], step: i64) -> Cmd {
    let farm = v.farm();
    let pos = farm.hands[actor - 1];
    let inv = v.inv(actor);
    let shed = &v.obs.shed;
    let home = home_by_tuple(pos);
    let distance = dist(pos, home);
    let cargo: Vec<&str> = ["WOOL", "FERTILIZER"].into_iter().filter(|i| qget(inv, i) != 0).collect();
    let day = step.div_euclid(24);
    if !cargo.is_empty() && step.rem_euclid(24) >= (if day == 29 { 22 } else { 23 }) - distance {
        return walk(pos, home).unwrap_or_else(|| place(cargo[0], qget(inv, cargo[0])));
    }
    let missing = targets.iter().filter(|&&(x, y)| !(farm.tile(x, y).is_dict() && farm.tile(x, y).animal == "SHEEP")).count() as i64;
    if missing != 0 && qget(inv, "SHEEP") == 0 && qget(shed, "SHEEP") != 0 {
        return walk(pos, home).unwrap_or_else(|| Cmd::order("PICKUP", "SHEEP", missing.min(qget(shed, "SHEEP"))));
    }
    let hungry = targets.iter().filter(|&&(x, y)| !(farm.tile(x, y).is_dict() && farm.tile(x, y).fed_today)).count() as i64;
    if hungry != 0 && qget(inv, "WHEAT") == 0 && qget(shed, "WHEAT") != 0 {
        return walk(pos, home).unwrap_or_else(|| Cmd::order("PICKUP", "WHEAT", hungry.min(qget(shed, "WHEAT"))));
    }
    let mut tasks: Vec<(i64, usize, P, Cmd)> = vec![];
    for (ti, &(x, y)) in targets.iter().enumerate() {
        let t = farm.tile(x, y);
        let command = if t.is_none() {
            Some(Cmd::new("BUILD_PASTURE"))
        } else if t.is_dict() && t.kind == "WEED" {
            Some(Cmd::new("DIG"))
        } else if t.is_dict() && t.kind == "PASTURE" && !t.has_animal() {
            if qget(inv, "SHEEP") != 0 {
                Some(Cmd(vec![crate::act::Tok::S("PLACE"), crate::act::Tok::S("SHEEP")]))
            } else {
                None
            }
        } else if t.is_dict() && t.animal == "SHEEP" {
            if !t.fed_today && qget(inv, "WHEAT") != 0 {
                Some(Cmd::new("FEED"))
            } else if !t.cared_today {
                Some(Cmd::new("CARE"))
            } else if t.yield_units != 0 {
                Some(Cmd::new("HARVEST"))
            } else {
                None
            }
        } else {
            None
        };
        if let Some(c) = command {
            let first = targets.iter().position(|q| *q == (x, y)).unwrap_or(ti);
            tasks.push((dist(pos, (x, y)), first, (x, y), c));
        }
    }
    if !tasks.is_empty() {
        let best = (0..tasks.len()).min_by_key(|&i| (tasks[i].0, tasks[i].1, tasks[i].2)).unwrap();
        let (_, _, t, c) = tasks.swap_remove(best);
        return walk(pos, t).unwrap_or(c);
    }
    if !cargo.is_empty() {
        return walk(pos, home).unwrap_or_else(|| place(cargo[0], qget(inv, cargo[0])));
    }
    Cmd::pass()
}

/// SL `_v233_worker` (six-tile compact service).
fn worker_sl(v: &View, actor: usize, targets: &[P], step: i64) -> Cmd {
    if targets.len() != 6 {
        return worker_base(v, actor, targets, step);
    }
    let farm = v.farm();
    let pos = farm.hands[actor - 1];
    let inv = v.inv(actor);
    let day = step.div_euclid(24);
    let mut needed: Vec<P> = vec![];
    let mut hungry = 0;
    for &(x, y) in targets {
        let t = farm.tile(x, y);
        if !t.is_dict() || t.animal != "SHEEP" {
            return worker_base(v, actor, targets, step);
        }
        if !t.fed_today {
            hungry += 1;
        }
        if !t.fed_today || !t.cared_today {
            needed.push((x, y));
        }
    }
    let home = home219(pos);
    if hungry > qget(inv, "WHEAT") {
        return walk(pos, home).unwrap_or_else(|| Cmd::order("PICKUP", "WHEAT", hungry.min(qget(&v.obs.shed, "WHEAT"))));
    }
    if !needed.is_empty() {
        let path = sl_path(pos, &needed);
        let cur = farm.tile(pos.0, pos.1);
        if targets.contains(&pos) && cur.is_dict() && cur.fed_today && cur.cared_today && cur.fertilizer_available {
            let travel = path_cost(pos, &path);
            let work = path.iter().filter(|&&(x, y)| !farm.tile(x, y).fed_today).count() as i64
                + path.iter().filter(|&&(x, y)| !farm.tile(x, y).cared_today).count() as i64;
            let last = *path.last().unwrap();
            let delivery = if day == 29 { dist(last, home219(last)) + 1 } else { 0 };
            let remaining = ((day + 1) * 24).min(719) - step;
            if 1 + travel + work + delivery <= remaining {
                return Cmd::new("COLLECT_FERTILIZER");
            }
        }
        let target = path[0];
        let t = farm.tile(target.0, target.1);
        return walk(pos, target).unwrap_or_else(|| Cmd::new(if !t.fed_today { "FEED" } else { "CARE" }));
    }
    let remaining = if day == 29 { 719 - step } else { 24 - step.rem_euclid(24) };
    let mut tasks: Vec<(i64, P)> = vec![];
    for &target in targets {
        let t = farm.tile(target.0, target.1);
        if !t.fertilizer_available {
            continue;
        }
        let d = dist(pos, target);
        let ret = if day == 29 { dist(target, home219(target)) + 1 } else { 0 };
        if d + 1 + ret <= remaining {
            tasks.push((d, target));
        }
    }
    if let Some(&(_, target)) = tasks.iter().min() {
        return walk(pos, target).unwrap_or_else(|| Cmd::new("COLLECT_FERTILIZER"));
    }
    let f = qget(inv, "FERTILIZER");
    if f != 0 {
        return walk(pos, home).unwrap_or_else(|| place("FERTILIZER", f));
    }
    Cmd::pass()
}

/// VT `_v233_worker` (days 28-29).
fn worker_vt(v: &View, actor: usize, targets: &[P], step: i64, day: i64) -> Cmd {
    if day != 28 && day != 29 {
        return worker_sl(v, actor, targets, step);
    }
    let farm = v.farm();
    let pos = farm.hands[actor - 1];
    let inv = v.inv(actor);
    let home = home219(pos);
    let distance = dist(pos, home);
    let cargo: Vec<&str> = ["WOOL", "FERTILIZER"].into_iter().filter(|i| qget(inv, i) != 0).collect();
    let last = if day == 29 { 717 } else { day * 24 + 23 };
    if !cargo.is_empty() && step >= last - distance {
        return walk(pos, home).unwrap_or_else(|| place(cargo[0], qget(inv, cargo[0])));
    }
    let sheep: Vec<P> = targets.iter().copied().filter(|&(x, y)| farm.tile(x, y).is_dict() && farm.tile(x, y).animal == "SHEEP").collect();
    if day == 28 {
        let hungry = sheep.iter().filter(|&&(x, y)| !farm.tile(x, y).fed_today).count() as i64;
        let shed_w = qget(&v.obs.shed, "WHEAT");
        if hungry != 0 && qget(inv, "WHEAT") == 0 && shed_w != 0 {
            return walk(pos, home).unwrap_or_else(|| Cmd::order("PICKUP", "WHEAT", hungry.min(shed_w)));
        }
    }
    let mut tasks: Vec<(i64, P, &'static str)> = vec![];
    for &(x, y) in &sheep {
        let t = farm.tile(x, y);
        let command = if day == 28 && !t.fed_today && qget(inv, "WHEAT") != 0 {
            Some("FEED")
        } else if t.yield_units != 0 {
            Some("HARVEST")
        } else if day == 28 && t.fertilizer_available {
            Some("COLLECT_FERTILIZER")
        } else {
            None
        };
        if let Some(c) = command {
            tasks.push((dist(pos, (x, y)), (x, y), c));
        }
    }
    if let Some(&(_, target, c)) = tasks.iter().min() {
        return walk(pos, target).unwrap_or_else(|| Cmd::new(c));
    }
    if !cargo.is_empty() {
        return walk(pos, home).unwrap_or_else(|| place(cargo[0], qget(inv, cargo[0])));
    }
    Cmd::pass()
}

// ---- rescue -------------------------------------------------------------------------------
/// VT `_v234_rescue` over base: buy the wheat the sheep hands are short of (not on day 29).
fn rescue_vt(v: &View, action: Action, st: &mut State, ch: &Chassis, step: i64, day: i64) -> Action {
    if day == 29 || st.workers.is_empty() || step.rem_euclid(24) > 14 {
        return action;
    }
    if action.market.len() >= 10 {
        return action;
    }
    if action.market.iter().any(|o| {
        !o.is_empty() && (matches!(o.op(), "HIRE" | "BUY_LAND" | "BUY_ANIMAL" | "BUY_PRODUCT" | "BUY_SEED") || (o.len() > 1 && o.s(1) == "WHEAT"))
    }) {
        return action;
    }
    let farm = v.farm();
    let commands = action.units();
    let (mut hungry, mut carried) = (0i64, 0i64);
    for (actor, targets) in &st.workers {
        let Some(c) = commands.get(*actor) else { return action };
        if (c.len() == 1 && c.op() == "FEED") || (c.len() >= 2 && c.op() == "PICKUP" && c.s(1) == "WHEAT") {
            return action;
        }
        carried += qget(v.inv(*actor), "WHEAT");
        hungry += targets
            .iter()
            .filter(|&&(x, y)| {
                let t = farm.tile(x, y);
                t.is_dict() && t.animal == "SHEEP" && !t.fed_today
            })
            .count() as i64;
    }
    let stock = ch.projected_shed(&action, v);
    let shortage = hungry - carried - qget(&stock, "WHEAT");
    if !(0 < shortage && shortage <= 6) || st.rescue_today + shortage > 6 {
        return action;
    }
    let quote = v.price("WHEAT");
    let total: i64 = stock.iter().map(|(_, n)| n).sum();
    if quote < 1 || farm.money < (1000 + shortage * (quote + 10)) as f64 || total + shortage > 100 {
        return action;
    }
    let mut result = action;
    result.market.push(Cmd::order("BUY_PRODUCT", "WHEAT", shortage));
    st.rescue_today += shortage;
    result
}
