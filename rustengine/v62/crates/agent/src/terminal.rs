//! The 7-turn terminal closure planner (agent lines 1018-1117 + the `_PLANNER_NS` payload
//! `python/ref/payloads/shop0909_closure_planner.py`), with the final `_shadow_terminal`
//! (R46 over V219 over base: abstain when a V233 / V219 project is committed).
//!
//! At step 712 the chassis's own remaining actions are replayed on a shadow (players state saved
//! and restored), simulated with the engine's unit semantics, and a bounded search over worker
//! harvest/collect/deliver routes proposes a plan that must physically dominate the baseline.
//! Steps 713-718 follow the plan while the observed farm matches the expected one.
use crate::act::{intern, Action, Cmd, Tok};
use crate::chassis::{Chassis, PlayerState};
use crate::obs::{FarmObs, Obs, Qty, Tile};
use crate::sim;
use crate::view::{shed_adjacent, View, PRODUCTS};
use kagg_engine::engine::decay_plants;
use kagg_engine::state::{Cell, Farm, OMap, Private};

pub const START: i64 = 712;
pub const FINAL: i64 = 718;

/// Planner knobs (profile knobs `term_*`). Defaults = v61.1: on, start 712, 64 sims, 1 pass, 4 proposals/actor.
#[derive(Clone, Copy, Debug)]
pub struct TermCfg {
    pub on: bool,
    pub start: i64,
    pub sims: usize,
    pub passes: usize,
    pub props: usize,
}
impl Default for TermCfg {
    fn default() -> Self {
        TermCfg { on: true, start: START, sims: 64, passes: 1, props: 4 }
    }
}
const ACCESS: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const OPS: [&str; 18] = [
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "PICKUP", "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER",
];
const ITEMS: [&str; 12] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"];

type P = (i64, i64);

// ---- engine <-> observation ------------------------------------------------------------------
fn tile_of(c: &Cell) -> Tile {
    let mut t = Tile::default();
    match c {
        Cell::Empty => {}
        Cell::Locked => t = Tile::LOCKED,
        Cell::Weed => t.kind = "WEED",
        Cell::Plant { crop, planted_day, watered_today, consecutive_unwatered, yield_units, max_lifespan_step, fertilized_until_day } => {
            t.kind = "PLANT";
            t.crop = intern(crop);
            t.planted_day = *planted_day;
            t.watered_today = *watered_today;
            t.consecutive_unwatered = *consecutive_unwatered;
            t.yield_units = *yield_units;
            t.max_lifespan_step = *max_lifespan_step;
            t.fertilized_until_day = *fertilized_until_day;
        }
        Cell::Structure { kind, animal } => {
            t.kind = intern(kind);
            if let Some(a) = animal {
                t.animal = intern(&a.animal);
                t.placed_day = a.placed_day;
                t.yield_units = a.yield_units;
                t.consecutive_unfed = a.consecutive_unfed;
                t.fed_today = a.fed_today;
                t.cared_today = a.cared_today;
                t.fertilizer_available = a.fertilizer_available;
                t.pending_care_bonus = a.pending_care_bonus;
            }
        }
    }
    t
}
fn farm_obs(f: &Farm) -> FarmObs {
    let rows = f.tiles.len();
    let cols = f.tiles.first().map(|r| r.len()).unwrap_or(0);
    FarmObs {
        money: f.money,
        farmer: Some(f.farmer),
        hands: f.hands.clone(),
        hires_today: f.hires_today,
        quadrants: f.unlocked_quadrants.iter().map(|s| intern(s)).collect(),
        tiles: f.tiles.iter().flat_map(|r| r.iter().map(tile_of)).collect(),
        rows,
        cols,
    }
}
fn qty(m: &OMap) -> Qty {
    m.0.iter().map(|(k, n)| (intern(k), *n)).collect()
}
/// Python dict equality (order-insensitive).
fn omap_eq(a: &OMap, b: &OMap) -> bool {
    a.0.len() == b.0.len() && a.0.iter().all(|(k, v)| b.0.iter().any(|(k2, v2)| k2 == k && v2 == v))
}
fn ge(a: &OMap, b: &OMap) -> bool {
    b.0.iter().all(|(k, v)| a.get(k) >= *v)
}
/// `physical_state(obs)` equality: the farm without money, and private.
fn physical_eq(f: &Farm, p: &Private, g: &Farm, q: &Private) -> bool {
    f.farmer == g.farmer
        && f.hands == g.hands
        && f.hires_today == g.hires_today
        && f.unlocked_quadrants == g.unlocked_quadrants
        && f.tiles == g.tiles
        && omap_eq(&p.shed, &q.shed)
        && omap_eq(&p.seeds, &q.seeds)
        && p.inventories.len() == q.inventories.len()
        && p.inventories.iter().zip(q.inventories.iter()).all(|(a, b)| omap_eq(a, b))
}

/// The observation with our farm and private replaced by (`farm`, `private`) at `step`.
fn project(base: &Obs, me: usize, farm: &Farm, private: &Private, step: i64) -> Obs {
    let mut o = base.clone();
    o.step = Some(step);
    o.day = step / 24;
    o.hour = step % 24;
    let money = o.farms[me].money;
    o.farms[me] = farm_obs(farm);
    o.farms[me].money = money;
    o.shed = qty(&private.shed);
    o.seeds = qty(&private.seeds);
    o.invs = private.inventories.iter().map(qty).collect();
    o
}

/// `_parent_liquidate(farm, private, prices)` (= the payload's `shop_liquidation`).
fn liquidate(ch: &Chassis, base: &Obs, me: usize, farm: &Farm, private: &Private) -> Action {
    let o = project(base, me, farm, private, base.step());
    let Ok(v) = View::new(o) else { return Action::pass() };
    let units: Vec<Cmd> = v
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| if shed_adjacent(*p, v.board) && !v.inv(i).is_empty() { Cmd::new("DROP") } else { Cmd::pass() })
        .collect();
    let mut a = Action { farmer: units[0].clone(), hands: units[1..].to_vec(), market: vec![] };
    let stock = ch.projected_shed(&a, &v);
    let mut m: Vec<Cmd> = PRODUCTS
        .iter()
        .filter(|p| crate::obs::qget(&stock, p) > 0)
        .map(|p| Cmd::order("SELL", p, crate::obs::qget(&stock, p)))
        .collect();
    m.sort_by_key(|o| -v.price(o.s(1)) * o.n(2));
    a.market = m;
    a
}

// ---- simulate --------------------------------------------------------------------------------
#[derive(Clone)]
struct Row {
    pre_market: OMap,
    deposited: Vec<OMap>,
    sold: OMap,
}
#[derive(Clone)]
struct Event {
    actor: usize,
    op: &'static str,
    xy: P,
    acquired: Option<OMap>,
}
#[derive(Clone)]
struct Run {
    rows: Vec<Row>,
    states: Vec<(Farm, Private)>,
    events: Vec<Event>,
    actions: Vec<Action>,
    overflow: i64,
    farm: Farm,
    private: Private,
    sold: OMap,
}

/// `_commands(action, n)`.
fn commands(a: &Action, n: usize) -> Vec<Cmd> {
    let mut c: Vec<Cmd> = std::iter::once(a.farmer.clone()).chain(a.hands.iter().cloned()).take(n).collect();
    let pad = (n as i64 - 1 - a.hands.len() as i64).max(0);
    for _ in 0..pad {
        c.push(Cmd::pass());
    }
    c
}

fn is_int(t: &Tok) -> bool {
    matches!(t, Tok::I(_))
}

/// `_validate` (Err = Unsupported).
fn validate(schedule: &[Action], orders: usize) -> Result<(), ()> {
    for a in schedule {
        for c in std::iter::once(&a.farmer).chain(a.hands.iter()) {
            if c.is_empty() || !OPS.contains(&c.op()) {
                return Err(());
            }
            if matches!(c.op(), "PICKUP" | "PLACE" | "PLANT") {
                if c.len() < 2 || !ITEMS.contains(&c.s(1)) {
                    return Err(());
                }
                if c.len() > 2 && !is_int(&c.0[2]) {
                    return Err(());
                }
            }
        }
        if a.market.len() > orders {
            return Err(());
        }
        for o in &a.market {
            if o.len() != 3 || o.op() != "SELL" || !PRODUCTS.contains(&o.s(1)) || !is_int(&o.0[2]) || o.n(2) <= 0 {
                return Err(());
            }
        }
    }
    Ok(())
}

struct Sim<'a> {
    ch: &'a Chassis,
    base: &'a Obs,
    me: usize,
    step: i64,
    start: i64,
    farm0: Farm,
    private0: Private,
    /// `liquidate` memo: its output is a function of the physical state alone (the rest of the
    /// projected observation is `base`). Keyed by exact, order-sensitive state equality.
    liq: std::cell::RefCell<Vec<(Farm, Private, Action)>>,
}

impl Sim<'_> {
    fn liquidate(&self, farm: &Farm, private: &Private) -> Action {
        let hit = self.liq.borrow().iter().find(|(f, p, _)| {
            f.farmer == farm.farmer
                && f.hands == farm.hands
                && p.inventories == private.inventories
                && p.shed == private.shed
                && physical_eq(f, p, farm, private)
        }).map(|(_, _, a)| a.clone());
        if let Some(a) = hit {
            return a;
        }
        let t0 = std::time::Instant::now();
        let a = liquidate(self.ch, self.base, self.me, farm, private);
        tprof(2, 1);
        tprof(3, t0.elapsed().as_nanos() as u64);
        self.liq.borrow_mut().push((farm.clone(), private.clone(), a.clone()));
        a
    }

    fn simulate(&self, schedule: &[Action], detailed: bool) -> Result<Run, ()> {
        let t0 = std::time::Instant::now();
        let r = self.simulate_inner(schedule, detailed);
        tprof(0, 1);
        tprof(1, t0.elapsed().as_nanos() as u64);
        r
    }

    fn simulate_inner(&self, schedule: &[Action], detailed: bool) -> Result<Run, ()> {
        let step = self.step;
        if step < self.start || step + schedule.len() as i64 - 1 > FINAL || schedule.is_empty() {
            return Err(());
        }
        let mut farm = self.farm0.clone();
        let mut private = self.private0.clone();
        let n = 1 + farm.hands.len();
        if private.inventories.len() != n || n > 32 {
            return Err(());
        }
        validate(schedule, 10)?;
        let mut deposited: Vec<OMap> = vec![OMap::default(); n];
        let mut sold = OMap::default();
        let mut states = vec![];
        let mut rows = vec![];
        let mut events = vec![];
        let mut executed: Vec<Action> = schedule.to_vec();
        let mut overflow = 0;
        for offset in 0..executed.len() {
            let t = step + offset as i64;
            if detailed {
                states.push((farm.clone(), private.clone()));
            }
            if t == FINAL {
                executed[offset] = self.liquidate(&farm, &private);
            }
            let action = executed[offset].clone();
            let all: Vec<&Cmd> = std::iter::once(&action.farmer).chain(action.hands.iter()).collect();
            let mut demand: Vec<(&str, i64)> = vec![];
            for c in &all {
                if !c.is_empty() && c.op() == "PLANT" {
                    match demand.iter_mut().find(|(k, _)| *k == c.s(1)) {
                        Some(e) => e.1 += 1,
                        None => demand.push((c.s(1), 1)),
                    }
                }
            }
            let blocked: Vec<&str> = demand.iter().filter(|(k, n)| *n > private.seeds.get(k)).map(|(k, _)| *k).collect();
            for (actor, command) in commands(&action, n).into_iter().enumerate() {
                if command.is_empty() {
                    return Err(());
                }
                let command = if command.op() == "PLANT" && blocked.contains(&command.s(1)) { Cmd::pass() } else { command };
                let op = command.op();
                let xy = if actor == 0 { farm.farmer } else { farm.hands[actor - 1] };
                let before_inv = if matches!(op, "DROP" | "HARVEST" | "COLLECT_FERTILIZER") { Some(private.inventories[actor].clone()) } else { None };
                let before_shed = if matches!(op, "DROP" | "PLACE") { Some(private.shed.clone()) } else { None };
                sim::apply(&mut farm, &mut private, actor, &command, t / 24);
                if let Some(bs) = before_shed {
                    let mut delta = OMap::default();
                    for (item, amount) in &private.shed.0 {
                        if *amount > bs.get(item) {
                            delta.0.push((*item, amount - bs.get(item)));
                        }
                    }
                    for (item, amount) in &delta.0 {
                        deposited[actor].add(item, *amount);
                    }
                    if !delta.0.is_empty() {
                        events.push(Event { actor, op, xy, acquired: None });
                    }
                    if op == "DROP" {
                        let inv = &private.inventories[actor];
                        overflow += before_inv
                            .as_ref()
                            .unwrap()
                            .0
                            .iter()
                            .map(|(item, amount)| (amount - inv.get(item) - delta.get(item)).max(0))
                            .sum::<i64>();
                    }
                }
                if matches!(op, "HARVEST" | "COLLECT_FERTILIZER") {
                    let bi = before_inv.as_ref().unwrap();
                    let mut delta = OMap::default();
                    for (item, amount) in &private.inventories[actor].0 {
                        if *amount > bi.get(item) {
                            delta.0.push((*item, amount - bi.get(item)));
                        }
                    }
                    if !delta.0.is_empty() {
                        events.push(Event { actor, op, xy, acquired: Some(delta) });
                    }
                }
            }
            let pre_market = private.shed.clone();
            for o in &action.market {
                let (item, requested) = (o.s(1), o.n(2));
                let q = requested.min(private.shed.get(item)).min(99999);
                if q > 0 {
                    private.shed.add(item, -q);
                    sold.add(item, q);
                }
            }
            decay_plants(&mut farm, t);
            rows.push(Row { pre_market, deposited: deposited.clone(), sold: sold.clone() });
        }
        if detailed {
            states.push((farm.clone(), private.clone()));
        }
        Ok(Run { rows, states, events, actions: executed, overflow, farm, private, sold })
    }
}

fn dominates(c: &Run, b: &Run) -> bool {
    if c.overflow != 0 {
        return false;
    }
    for (new, old) in c.rows.iter().zip(b.rows.iter()) {
        if !ge(&new.pre_market, &old.pre_market) || !ge(&new.sold, &old.sold) {
            return false;
        }
        if new.deposited.iter().zip(old.deposited.iter()).any(|(a, b)| !ge(a, b)) {
            return false;
        }
    }
    true
}

fn value(run: &Run, prices: &[(&str, f64)]) -> f64 {
    let mut s = 0.0;
    for (item, p) in prices {
        s += (run.sold.get(item) + run.private.shed.get(item)) as f64 * p;
    }
    s
}

fn walk(s: P, e: P) -> Vec<&'static str> {
    let mut r = vec![];
    r.extend(std::iter::repeat_n("EAST", (e.0 - s.0).max(0) as usize));
    r.extend(std::iter::repeat_n("WEST", (s.0 - e.0).max(0) as usize));
    r.extend(std::iter::repeat_n("SOUTH", (e.1 - s.1).max(0) as usize));
    r.extend(std::iter::repeat_n("NORTH", (s.1 - e.1).max(0) as usize));
    r
}
fn ret(pos: P) -> Vec<&'static str> {
    let d = |a: P| (pos.0 - a.0).abs() + (pos.1 - a.1).abs();
    let mut best = ACCESS[0];
    for a in ACCESS {
        if d(a) < d(best) {
            best = a;
        }
    }
    let mut r = walk(pos, best);
    r.push("DROP");
    r
}

fn animal_product(a: &str) -> Option<&'static str> {
    Some(match a {
        "GOOSE" => "EGG",
        "COW" => "MILK",
        "SHEEP" => "WOOL",
        _ => return None,
    })
}
fn first_yield_day(crop: &str) -> i64 {
    match crop {
        "WHEAT" | "CARROT" => 2,
        "TOMATO" => 8,
        _ => 10,
    }
}

/// `_proposals`: (score, offset, route, bundle count), best first.
fn proposals(run: &Run, actor: usize, prices: &[(&str, f64)], max_per_actor: usize, start: i64) -> Vec<(f64, usize, Vec<&'static str>, usize)> {
    let price = |i: &str| prices.iter().find(|(k, _)| *k == i).map(|(_, p)| *p).unwrap_or(0.0);
    let mut owners: Vec<((P, &str), Vec<usize>)> = vec![];
    for e in &run.events {
        if e.acquired.is_some() {
            match owners.iter_mut().find(|(k, _)| *k == (e.xy, e.op)) {
                Some((_, s)) => {
                    if !s.contains(&e.actor) {
                        s.push(e.actor)
                    }
                }
                None => owners.push(((e.xy, e.op), vec![e.actor])),
            }
        }
    }
    let foreign = |xy: P, op: &str| owners.iter().find(|(k, _)| *k == (xy, op)).is_some_and(|(_, s)| s.iter().any(|a| *a != actor));
    let mut out: Vec<(f64, usize, Vec<&'static str>, usize)> = vec![];
    let mut seen: Vec<(usize, Vec<&'static str>)> = vec![];
    let horizon = run.rows.len();
    for offset in 0..horizon {
        let (farm, private) = &run.states[offset];
        let pos = if actor == 0 { farm.farmer } else { farm.hands[actor - 1] };
        let inventory = &private.inventories[actor];
        let carried: f64 = inventory.0.iter().map(|(i, c)| price(i) * *c as f64).sum();
        let empty = OMap::default();
        let prefix = if offset > 0 { &run.rows[offset - 1].deposited[actor] } else { &empty };
        let future = &run.rows[horizon - 1].deposited[actor];
        let obligation: f64 = future.0.iter().map(|(i, c)| price(i) * (c - prefix.get(i)) as f64).sum();
        let mut bundles: Vec<(P, Vec<&'static str>, f64, i64)> = vec![];
        for (y, row) in farm.tiles.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                let xy = (x as i64, y as i64);
                let mut ops = vec![];
                let mut val = 0.0;
                let (yu, item, mature, fert_avail, has_animal) = match cell {
                    Cell::Plant { crop, planted_day, yield_units, .. } => {
                        let mature = (start + offset as i64) / 24 - planted_day >= first_yield_day(crop);
                        (*yield_units, Some(intern(crop)), mature, false, false)
                    }
                    Cell::Structure { animal: Some(a), .. } => (a.yield_units, animal_product(&a.animal), true, a.fertilizer_available, true),
                    Cell::Structure { animal: None, .. } | Cell::Weed => (0, None, false, false, false),
                    _ => continue,
                };
                if yu > 0 && item.is_some() && mature && !foreign(xy, "HARVEST") {
                    ops.push("HARVEST");
                    val += price(item.unwrap()) * yu as f64;
                }
                if fert_avail && has_animal && !foreign(xy, "COLLECT_FERTILIZER") {
                    ops.push("COLLECT_FERTILIZER");
                    val += price("FERTILIZER");
                }
                if !ops.is_empty() {
                    let distance = (walk(pos, xy).len() + ops.len() + ret(xy).len()) as i64;
                    if distance <= (horizon - offset) as i64 {
                        bundles.push((xy, ops, val, distance));
                    }
                }
            }
        }
        bundles.sort_by(|a, b| {
            (-a.2 / a.3 as f64).partial_cmp(&(-b.2 / b.3 as f64)).unwrap().then((-a.2).partial_cmp(&-b.2).unwrap()).then(a.0.cmp(&b.0))
        });
        let mut variants: Vec<(Vec<(P, Vec<&'static str>)>, f64)> = vec![];
        if carried != 0.0 {
            variants.push((vec![], carried));
        }
        for b in bundles.iter().take(6) {
            variants.push((vec![(b.0, b.1.clone())], carried + b.2));
        }
        for f in bundles.iter().take(3) {
            for s in bundles.iter().take(3) {
                if f.0 != s.0 {
                    variants.push((vec![(f.0, f.1.clone()), (s.0, s.1.clone())], carried + f.2 + s.2));
                }
            }
        }
        for (stops, val) in variants {
            let mut route: Vec<&'static str> = vec![];
            let mut cursor = pos;
            for (xy, ops) in &stops {
                route.extend(walk(cursor, *xy));
                route.extend(ops.iter().copied());
                cursor = *xy;
            }
            route.extend(ret(cursor));
            if route.len() > horizon - offset {
                continue;
            }
            while route.len() < horizon - offset {
                route.push("PASS");
            }
            if !seen.iter().any(|(o, r)| *o == offset && *r == route) {
                seen.push((offset, route.clone()));
                out.push((val - obligation, offset, route, stops.len()));
            }
        }
    }
    out.sort_by(|a, b| (-a.0).partial_cmp(&-b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let direct: Vec<(f64, usize, Vec<&'static str>, usize)> = out.iter().filter(|p| p.3 == 0 && p.0 > 0.0).take(2).cloned().collect();
    let mut chosen = direct.clone();
    chosen.extend(out.into_iter().filter(|p| !direct.contains(p)));
    chosen.truncate(max_per_actor);
    chosen
}

#[derive(Clone)]
pub struct Plan {
    baseline: Vec<Action>,
    actions: Vec<Action>,
    expected: Vec<(Farm, Private)>,
    abandoned: bool,
    deviated: bool,
    parent_states_before: Vec<PlayerState>,
    start: i64,
}

/// `plan_terminal(obs, config, baseline, max_simulations=64, passes=1, proposals_per_actor=4)`.
/// Profiling: [simulate calls, simulate ns, liquidate misses, liquidate ns, proposals ns, plan calls].
pub static TPROF: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
fn tprof(i: usize, n: u64) {
    TPROF[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

fn plan_terminal(s: &Sim, baseline_remaining: &[Action], cfg: TermCfg) -> Option<Plan> {
    tprof(5, 1);
    if s.step != s.start || baseline_remaining.len() as i64 != FINAL - s.start + 1 {
        return None;
    }
    let max_sims = cfg.sims;
    let baseline = s.simulate(baseline_remaining, true).ok()?;
    let prices: Vec<(&str, f64)> = PRODUCTS
        .iter()
        .map(|p| (*p, (s.base.prices.iter().find(|(k, _)| k == p).map(|(_, v)| *v as f64).unwrap_or(1.0)).max(1.0)))
        .collect();
    let mut current: Vec<Action> = baseline_remaining.to_vec();
    let mut best = baseline.clone();
    let baseline_value = value(&baseline, &prices);
    let mut best_value = baseline_value;
    let mut changes = 0;
    let mut sims = 0;
    let n = baseline.private.inventories.len();
    for _sweep in 0..cfg.passes.max(1) {
        let mut improved = false;
        for actor in 0..n {
            let mut winner: Option<Vec<Action>> = None;
            let tp = std::time::Instant::now();
            let props = proposals(&best, actor, &prices, cfg.props.max(1), s.start);
            tprof(4, tp.elapsed().as_nanos() as u64);
            for (_, offset, route, _) in props {
                if sims >= max_sims {
                    break;
                }
                let mut trial = current.clone();
                for (k, cmd) in route.iter().enumerate() {
                    let i = offset + k;
                    let c = Cmd::new(cmd);
                    if actor == 0 {
                        trial[i].farmer = c;
                    } else {
                        while trial[i].hands.len() < n - 1 {
                            trial[i].hands.push(Cmd::pass());
                        }
                        trial[i].hands[actor - 1] = c;
                    }
                }
                let Ok(evaluated) = s.simulate(&trial, false) else { return None };
                sims += 1;
                let score = value(&evaluated, &prices);
                if score > best_value && dominates(&evaluated, &baseline) {
                    let mut required: Vec<((P, &str, usize), OMap)> = vec![];
                    for e in &best.events {
                        if let Some(a) = &e.acquired {
                            if e.actor != actor {
                                let k = (e.xy, e.op, e.actor);
                                match required.iter_mut().find(|(kk, _)| *kk == k) {
                                    Some(x) => x.1 = a.clone(),
                                    None => required.push((k, a.clone())),
                                }
                            }
                        }
                    }
                    let mut acquired: Vec<((P, &str, usize), OMap)> = vec![];
                    for e in &evaluated.events {
                        if let Some(a) = &e.acquired {
                            let k = (e.xy, e.op, e.actor);
                            let i = match acquired.iter().position(|(kk, _)| *kk == k) {
                                Some(i) => i,
                                None => {
                                    acquired.push((k, OMap::default()));
                                    acquired.len() - 1
                                }
                            };
                            for (item, amount) in &a.0 {
                                acquired[i].1.add(item, *amount);
                            }
                        }
                    }
                    let empty = OMap::default();
                    if required.iter().all(|(k, v)| ge(acquired.iter().find(|(kk, _)| kk == k).map(|(_, m)| m).unwrap_or(&empty), v)) {
                        winner = Some(trial);
                        best_value = score;
                    }
                }
            }
            if let Some(w) = winner {
                current = w;
                best = s.simulate(&current, true).ok()?;
                changes += 1;
                improved = true;
            }
            if sims >= max_sims {
                break;
            }
        }
        if !improved || sims >= max_sims {
            break;
        }
    }
    if changes == 0 || best_value <= baseline_value {
        return None;
    }
    let fin = s.simulate(&current, true).ok()?;
    let physical = s.simulate(&current, false).ok()?;
    if !dominates(&physical, &baseline) {
        return None;
    }
    let delta: Vec<i64> = PRODUCTS.iter().map(|p| fin.sold.get(p) - baseline.sold.get(p)).collect();
    let last = fin.rows.len() - 1;
    let deposited_gain = (0..n).any(|a| PRODUCTS.iter().any(|p| fin.rows[last].deposited[a].get(p) > baseline.rows[last].deposited[a].get(p)));
    let worker_change = fin.actions.iter().zip(baseline.actions.iter()).any(|(a, b)| commands(a, n) != commands(b, n));
    let accepted = worker_change && deposited_gain && delta.iter().any(|d| *d > 0) && delta.iter().all(|d| *d >= 0);
    if !accepted {
        return None;
    }
    let mut expected = fin.states.clone();
    expected.pop();
    Some(Plan {
        baseline: baseline_remaining.to_vec(),
        actions: fin.actions,
        expected,
        abandoned: false,
        deviated: false,
        parent_states_before: vec![],
        start: s.start,
    })
}

// ---- following the plan ----------------------------------------------------------------------
/// `_recover_observed`.
fn recover(ch: &Chassis, v: &View, parent: &Action) -> Action {
    let farm = sim::farm(v.farm());
    let private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let step = v.step;
    let remaining = FINAL - step + 1;
    let mut room = (100 - private.shed.0.iter().map(|(_, n)| n).sum::<i64>()).max(0);
    let mut cmds = vec![];
    let mut positions = vec![farm.farmer];
    positions.extend(farm.hands.iter().copied());
    for (pos, inv) in positions.iter().zip(private.inventories.iter()) {
        let mut c = Cmd::pass();
        if inv.0.iter().any(|(_, q)| *q > 0) {
            let route = ret(*pos);
            let held: i64 = inv.0.iter().map(|(_, q)| (*q).max(0)).sum();
            if route.len() as i64 > remaining {
            } else if route.len() > 1 {
                c = Cmd::new(route[0]);
            } else if held <= room {
                c = Cmd::new("DROP");
                room -= held;
            } else {
                let items: Vec<&str> = PRODUCTS.iter().copied().filter(|i| inv.get(i) > 0).collect();
                if room > 0 && !items.is_empty() {
                    let price = |i: &str| v.obs.prices.iter().find(|(k, _)| *k == i).map(|(_, p)| *p).unwrap_or(1);
                    let key = |i: &str| (price(i) * inv.get(i).min(room), -(PRODUCTS.iter().position(|p| *p == i).unwrap() as i64));
                    let item = *items.iter().max_by_key(|i| key(i)).unwrap();
                    let q = inv.get(item).min(room);
                    c = Cmd::order("PLACE", item, q);
                    room -= q;
                }
            }
        }
        cmds.push(c);
    }
    let mut a = Action { farmer: cmds[0].clone(), hands: cmds[1..].to_vec(), market: parent.market.clone() };
    if step == FINAL {
        // simulate(..., [action], final_liquidate=True, preserve_final_commands=True): keep the
        // unit commands, market = the post-unit shed liquidation sorted by value.
        let mut f = farm.clone();
        let mut p = private.clone();
        for (actor, c) in commands(&a, positions.len()).iter().enumerate() {
            sim::apply(&mut f, &mut p, actor, c, step / 24);
        }
        let mut m: Vec<Cmd> = PRODUCTS.iter().filter(|i| p.shed.get(i) > 0).map(|i| Cmd::order("SELL", i, p.shed.get(i))).collect();
        m.sort_by_key(|o| -v.price(o.s(1)) * o.n(2));
        a.market = m;
    }
    let _ = ch;
    a
}

#[derive(Default)]
pub struct Terminal {
    seats: Vec<(i64, (Option<i64>, Option<Plan>))>,
    /// Planner knobs for this turn (set by the base from the active profile; None = defaults).
    pub cfg: Option<TermCfg>,
}

pub struct Ctx<'a> {
    pub committed_project: bool,
    pub obs: &'a Obs,
}

impl Terminal {
    /// The terminal wrapper around `pre` (= chassis + `_SHOP` rescue).
    pub fn act(&mut self, ch: &mut Chassis, v: &View, cx: Ctx, pre: &dyn Fn(&mut Chassis, &View) -> Action) -> Action {
        let (step, seat) = (v.step, v.obs.player);
        let cfg = self.cfg.unwrap_or_default();
        let i = match self.seats.iter().position(|(p, _)| *p == seat) {
            Some(i) => i,
            None => {
                self.seats.push((seat, (None, None)));
                self.seats.len() - 1
            }
        };
        let previous = self.seats[i].1 .0;
        if step == 0 || previous.is_some_and(|p| step <= p) {
            self.seats[i].1 .1 = None;
        }
        self.seats[i].1 .0 = Some(step);
        if let Some(plan) = self.seats[i].1 .1.as_mut() {
            if (plan.start..=FINAL).contains(&step) {
                if previous != Some(step - 1) {
                    plan.abandoned = true;
                }
                let idx = (step - plan.start) as usize;
                let parent = plan.baseline[idx].clone();
                let result = follow(ch, v, &parent, plan);
                if plan.abandoned {
                    if !plan.deviated {
                        let st = plan.parent_states_before[idx].clone();
                        if let Some((_, s)) = ch.players.iter_mut().find(|(p, _)| *p == seat) {
                            *s = st;
                        }
                        self.seats[i].1 .1 = None;
                        return pre(ch, v);
                    }
                }
                return result;
            }
        }
        if !cfg.on || step != cfg.start {
            return pre(ch, v);
        }
        let Some((baseline, states)) = shadow(ch, v, cx.obs, cx.committed_project, cfg.start) else { return pre(ch, v) };
        let actual = pre(ch, v);
        if actual != baseline[0] {
            return actual;
        }
        let me = v.me;
        let s = Sim {
            ch,
            base: cx.obs,
            me,
            step,
            start: cfg.start,
            farm0: sim::farm(v.farm()),
            private0: sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs),
            liq: Default::default(),
        };
        let Some(mut plan) = plan_terminal(&s, &baseline, cfg) else { return actual };
        plan.parent_states_before = states;
        let result = follow(ch, v, &actual, &mut plan);
        self.seats[i].1 .1 = Some(plan);
        result
    }
}

/// `terminal_action`.
fn follow(ch: &Chassis, v: &View, parent: &Action, plan: &mut Plan) -> Action {
    let step = v.step;
    if plan.abandoned {
        return if plan.deviated { recover(ch, v, parent) } else { parent.clone() };
    }
    let idx = (step - plan.start) as usize;
    let n = v.positions.len();
    let farm = sim::farm(v.farm());
    let private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let (ef, ep) = &plan.expected[idx];
    let mismatch = !physical_eq(&farm, &private, ef, ep)
        || commands(parent, n) != commands(&plan.baseline[idx], n)
        || parent.market != plan.baseline[idx].market;
    if mismatch {
        plan.abandoned = true;
        return if plan.deviated { recover(ch, v, parent) } else { parent.clone() };
    }
    let mut result = plan.actions[idx].clone();
    if step == FINAL {
        result = liquidate(ch, &v.obs, v.me, &farm, &private);
    }
    if commands(&result, n) != commands(parent, n) {
        plan.deviated = true;
    }
    result
}

/// `_shadow_terminal`: the chassis's own 712..718 actions on a simulated shadow.
fn shadow(ch: &mut Chassis, v: &View, obs: &Obs, committed_project: bool, start: i64) -> Option<(Vec<Action>, Vec<PlayerState>)> {
    if committed_project {
        return None;
    }
    let seat = v.obs.player;
    let st = ch.players.iter().find(|(p, _)| *p == seat).map(|(_, s)| s)?;
    if st.last_step != start - 1 || st.route != Some(2) || !st.pending.is_empty() {
        return None;
    }
    let saved = ch.players.clone();
    let saved_diag = ch.diagnostics.clone();
    ch.diagnostics = Default::default();
    let me = v.me;
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let mut baseline = vec![];
    let mut states = vec![];
    let mut ok = true;
    for step in start..=FINAL {
        let projected = project(obs, me, &farm, &private, step);
        states.push(ch.players.iter().find(|(p, _)| *p == seat).map(|(_, s)| s.clone()).unwrap());
        let Ok(pv) = View::new(projected.clone()) else {
            ok = false;
            break;
        };
        let mut action = ch.act(&pv);
        if step == FINAL {
            action = liquidate(ch, &projected, me, &farm, &private);
        } else {
            let m = &action.market;
            let items: Vec<&str> = m.iter().map(|o| o.s(1)).collect();
            let all = PRODUCTS.iter().all(|p| items.contains(p)) && items.iter().all(|i| PRODUCTS.contains(i));
            if m.len() != 9 || !all || m.iter().any(|o| o.op() != "SELL" || o.len() != 3 || !is_int(&o.0[2]) || o.n(2) < 100) {
                ok = false;
                break;
            }
        }
        if ch.diagnostics.layer_fallbacks != 0 || ch.diagnostics.entry_fallbacks != 0 || ch.diagnostics.terminal_rescue_errors != 0 {
            ok = false;
            break;
        }
        let s = Sim { ch, base: &projected, me, step, start, farm0: farm.clone(), private0: private.clone(), liq: Default::default() };
        let Ok(run) = s.simulate(std::slice::from_ref(&action), false) else {
            ok = false;
            break;
        };
        if run.actions[0] != action {
            ok = false;
            break;
        }
        baseline.push(action);
        let money = farm.money;
        farm = run.farm;
        farm.money = money;
        private = run.private;
    }
    ch.players = saved;
    ch.diagnostics = saved_diag;
    if ok {
        Some((baseline, states))
    } else {
        None
    }
}
