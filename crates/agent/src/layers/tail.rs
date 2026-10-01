//! cha22 tail, part 2 (agent 7173-7583): CXD exact best-response ordering, E410 fertilizer
//! guard, E402 late seed cap, MG same-item merge, IG queue hole-closure, then our RSA route
//! sale advance (glut guard 0.5 x base) on top of the last callable (`ig_agent`).
use super::ca;
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qadd, qget, Qty};
use crate::sim;
use crate::view::View;
use kagg_engine::state::Cell;


/// Allocation-free per-item schedule key for the margin caches (CXD / V44Y): up to 16 queue
/// entries, each packed as (qty << 5 | sell << 4 | position). Only the first `n` entries are
/// compared/hashed. Queues that do not pack (long, or qty outside 0..2^58) take the `Vec` path.
pub const SCHED_CAP: usize = 16;
#[derive(Clone, Copy)]
pub struct Sched {
    pub n: u8,
    pub e: [u64; SCHED_CAP],
}
impl PartialEq for Sched {
    fn eq(&self, o: &Sched) -> bool {
        self.entries() == o.entries()
    }
}
impl Eq for Sched {}
impl std::hash::Hash for Sched {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        for x in self.entries() {
            h.write_u64(*x);
        }
        h.write_u8(self.n);
    }
}
impl Sched {
    pub const EMPTY: Sched = Sched { n: 0, e: [0; SCHED_CAP] };
    pub fn push(&mut self, i: usize, sell: bool, q: i64) {
        self.e[self.n as usize] = ((q as u64) << 5) | ((sell as u64) << 4) | i as u64;
        self.n += 1;
    }
    pub fn entries(&self) -> &[u64] {
        &self.e[..self.n as usize]
    }
    pub fn unpack(x: u64) -> (usize, bool, i64) {
        ((x & 15) as usize, x & 16 != 0, (x >> 5) as i64)
    }
}

/// FNV-1a — the cache keys are tiny and trusted; SipHash dominated the lookup.
#[derive(Default, Clone, Copy)]
pub struct Fnv(u64);
impl std::hash::Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write_u64(&mut self, x: u64) {
        let h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        self.0 = (h ^ x).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        self.0 = h;
    }
}
pub type FnvBuild = std::hash::BuildHasherDefault<Fnv>;
pub type SchedCache = std::collections::HashMap<(&'static str, Sched), f64, FnvBuild>;

// ---- CXD -------------------------------------------------------------------------------------
pub const CXD_BUDGET: usize = 800;
const CXD_FIXED: [&str; 4] = ["HIRE", "BUY_SEED", "BUY_ANIMAL", "BUY_LAND"];

/// Single-item lockstep race (money-unbounded), as `_v44y_lockstep`.
fn lockstep(item: &'static str, me: &[Option<(bool, i64)>], opp: &[Option<(bool, i64)>], inv0: i64, stock0: i64) -> (f64, f64) {
    let mut inv = inv0;
    let mut stock = [stock0, stock0];
    let mut rev = [0f64, 0f64];
    let buyable = item == "WHEAT" || item == "FERTILIZER";
    let idx = market::item_index(item);
    for i in 0..me.len().max(opp.len()) {
        let mut rem = [me.get(i).copied().flatten().filter(|x| x.1 > 0), opp.get(i).copied().flatten().filter(|x| x.1 > 0)];
        let mut guard = 0;
        loop {
            guard += 1;
            if guard > 5000 {
                break;
            }
            let mut quoted: [Option<(bool, i64)>; 2] = [None, None];
            for p in 0..2 {
                let Some((sell, q)) = rem[p] else { continue };
                if q <= 0 {
                    continue;
                }
                if sell {
                    quoted[p] = Some((true, market::price_i(idx, item, inv)));
                } else if buyable {
                    quoted[p] = Some((false, market::price_i(idx, item, inv - 1)));
                } else {
                    rem[p] = None;
                }
            }
            if quoted.iter().all(|q| q.is_none()) {
                break;
            }
            let mut committed = false;
            for p in 0..2 {
                let Some((sell, price)) = quoted[p] else { continue };
                if sell {
                    if stock[p] <= 0 {
                        rem[p] = None;
                        continue;
                    }
                    stock[p] -= 1;
                    rev[p] += price as f64;
                    if price > 1 {
                        inv += 1;
                    }
                } else {
                    stock[p] += 1;
                    rev[p] -= price as f64;
                    inv -= 1;
                }
                if let Some(r) = rem[p].as_mut() {
                    r.1 -= 1;
                }
                committed = true;
            }
            if !committed {
                break;
            }
        }
    }
    (rev[0], rev[1])
}

pub fn valid(o: &Cmd) -> Option<(&'static str, bool, i64)> {
    if o.is_empty() || o.len() < 3 || !matches!(o.op(), "SELL" | "BUY_PRODUCT") || market::params(o.s(1)).is_none() {
        return None;
    }
    Some((o.s(1), o.op() == "SELL", o.n(2)))
}

/// `_v44y_factor_margin(opp, ...)` with its per-(item, schedule) cache.
pub struct Margin<'a> {
    opp: Vec<(&'static str, Vec<Option<(bool, i64)>>)>,
    inv: &'a Qty,
    stock: &'a Qty,
    cache: std::collections::HashMap<(&'static str, Vec<(usize, bool, i64)>), f64>,
    fast: SchedCache,
}

impl<'a> Margin<'a> {
    pub fn new(opp_orders: &[Cmd], inv: &'a Qty, stock: &'a Qty) -> Margin<'a> {
        let mut opp: Vec<(&'static str, Vec<Option<(bool, i64)>>)> = vec![];
        for (i, o) in opp_orders.iter().enumerate() {
            if let Some((item, sell, n)) = valid(o) {
                let idx = match opp.iter().position(|(k, _)| *k == item) {
                    Some(p) => p,
                    None => {
                        opp.push((item, vec![None; opp_orders.len()]));
                        opp.len() - 1
                    }
                };
                opp[idx].1[i] = Some((sell, n));
            }
        }
        Margin { opp, inv, stock, cache: Default::default(), fast: Default::default() }
    }
    pub fn eval(&mut self, cand: &[Cmd]) -> f64 {
        let parsed: Vec<Option<(&'static str, bool, i64)>> = cand.iter().map(valid).collect();
        self.eval_parsed(&parsed)
    }

    /// `eval` over pre-parsed orders (`valid(o)` per queue position).
    pub fn eval_parsed(&mut self, cand: &[Option<(&'static str, bool, i64)>]) -> f64 {
        if cand.len() <= SCHED_CAP
            && self.opp.len() + cand.len() <= SCHED_CAP
            && cand.iter().all(|o| o.is_none_or(|(_, _, n)| (0..1 << 58).contains(&n)))
        {
            return self.eval_fast(cand);
        }
        self.eval_slow(cand)
    }

    /// `eval_slow` without allocation: same schedules, same item order, same values.
    fn eval_fast(&mut self, cand: &[Option<(&'static str, bool, i64)>]) -> f64 {
        let mut items: [&'static str; SCHED_CAP] = [""; SCHED_CAP];
        let mut scheds = [Sched::EMPTY; SCHED_CAP];
        let mut k = 0;
        for (item, _) in &self.opp {
            items[k] = item;
            k += 1;
        }
        for (i, o) in cand.iter().enumerate() {
            if let Some((item, sell, n)) = *o {
                match items[..k].iter().position(|x| *x == item) {
                    Some(p) => scheds[p].push(i, sell, n),
                    None => {
                        items[k] = item;
                        scheds[k].push(i, sell, n);
                        k += 1;
                    }
                }
            }
        }
        let mut total = 0.0;
        for x in 0..k {
            let key = (items[x], scheds[x]);
            let value = match self.fast.get(&key) {
                Some(v) => *v,
                None => {
                    let mut mine: Vec<Option<(bool, i64)>> = vec![None; cand.len()];
                    for &x in key.1.entries() {
                        let (i, sell, n) = Sched::unpack(x);
                        mine[i] = Some((sell, n));
                    }
                    let item = key.0;
                    let theirs = self.opp.iter().find(|(k, _)| *k == item).map(|(_, s)| s.clone()).unwrap_or_else(|| vec![None; cand.len()]);
                    let (a, b) = lockstep(item, &mine, &theirs, qget(self.inv, item), qget(self.stock, item));
                    self.fast.insert(key, a - b);
                    a - b
                }
            };
            total += value;
        }
        total
    }

    fn eval_slow(&mut self, cand: &[Option<(&'static str, bool, i64)>]) -> f64 {
        let mut schedules: Vec<(&'static str, Vec<(usize, bool, i64)>)> = self.opp.iter().map(|(k, _)| (*k, vec![])).collect();
        for (i, o) in cand.iter().enumerate() {
            if let Some((item, sell, n)) = *o {
                match schedules.iter_mut().find(|(k, _)| *k == item) {
                    Some((_, s)) => s.push((i, sell, n)),
                    None => schedules.push((item, vec![(i, sell, n)])),
                }
            }
        }
        let mut total = 0.0;
        for (item, schedule) in schedules {
            let key = (item, schedule);
            let value = match self.cache.get(&key) {
                Some(v) => *v,
                None => {
                    let mut mine: Vec<Option<(bool, i64)>> = vec![None; cand.len()];
                    for &(i, sell, n) in &key.1 {
                        mine[i] = Some((sell, n));
                    }
                    let theirs = self.opp.iter().find(|(k, _)| *k == item).map(|(_, s)| s.clone()).unwrap_or_else(|| vec![None; cand.len()]);
                    let (a, b) = lockstep(item, &mine, &theirs, qget(self.inv, item), qget(self.stock, item));
                    self.cache.insert(key, a - b);
                    a - b
                }
            };
            total += value;
        }
        total
    }
}

/// `itertools.permutations(pool, r)` in its order, calling `f` until it returns false.
fn r_permutations(pool: &[usize], r: usize, f: &mut impl FnMut(&[usize]) -> bool) {
    let n = pool.len();
    if r > n {
        return;
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let mut cycles: Vec<usize> = (0..r).map(|i| n - i).collect();
    let emit = |idx: &[usize]| -> Vec<usize> { idx[..r].iter().map(|&i| pool[i]).collect() };
    if !f(&emit(&indices)) {
        return;
    }
    loop {
        let mut done = true;
        for i in (0..r).rev() {
            cycles[i] -= 1;
            if cycles[i] == 0 {
                let x = indices.remove(i);
                indices.push(x);
                cycles[i] = n - i;
            } else {
                let j = cycles[i];
                indices.swap(i, n - j);
                if !f(&emit(&indices)) {
                    return;
                }
                done = false;
                break;
            }
        }
        if done {
            return;
        }
    }
}

/// `cxd` against explicit rival order lists (worst case over them); empty = `cxd` (copy model).
pub fn cxd_models(action: Action, v: &View, ch: &Chassis, models: &[Vec<Cmd>]) -> Action {
    if models.is_empty() {
        return cxd(action, v, ch);
    }
    cxd_impl(action, v, ch, models)
}

pub fn cxd(action: Action, v: &View, ch: &Chassis) -> Action {
    let m = vec![action.market.clone()];
    cxd_impl(action, v, ch, &m)
}

fn cxd_impl(action: Action, v: &View, ch: &Chassis, models: &[Vec<Cmd>]) -> Action {
    let market = &action.market;
    if market.len() < 2 {
        return action;
    }
    let orders = market.clone();
    let bought: Vec<&str> = orders.iter().filter(|o| !o.is_empty() && o.len() > 1 && o.op() == "BUY_PRODUCT").map(|o| o.s(1)).collect();
    let (mut slots, mut sells, mut fixed) = (vec![], vec![], vec![]);
    for (i, o) in orders.iter().enumerate() {
        if o.is_empty() {
            continue;
        }
        if CXD_FIXED.contains(&o.op()) {
            slots.push(i);
            fixed.push(o.clone());
        } else if o.op() == "SELL" && o.len() > 1 && !bought.contains(&o.s(1)) {
            slots.push(i);
            sells.push(o.clone());
        }
    }
    if sells.is_empty() || slots.len() < 2 {
        return action;
    }
    let stock: Qty = ch.projected_shed(&action, v).into_iter().map(|(k, n)| (k, n.max(0))).collect();
    let inv = v.obs.mkt_inventory.clone();
    let mut ms: Vec<Margin> = models.iter().map(|o| Margin::new(o, &inv, &stock)).collect();
    // Candidates are index maps (queue position -> index into `orders`); nothing is cloned per
    // candidate. `class[i]` = first index whose order equals orders[i] (the `out == orders` test).
    let parsed: Vec<Option<(&'static str, bool, i64)>> = orders.iter().map(valid).collect();
    let class: Vec<usize> = (0..orders.len()).map(|i| (0..=i).find(|&j| orders[j] == orders[i]).unwrap()).collect();
    let sell_idx: Vec<usize> = slots.iter().copied().filter(|&i| {
        let o = &orders[i];
        !CXD_FIXED.contains(&o.op())
    }).collect();
    let fixed_idx: Vec<usize> = slots.iter().copied().filter(|&i| CXD_FIXED.contains(&orders[i].op())).collect();
    debug_assert_eq!(sell_idx.len(), sells.len());
    debug_assert_eq!(fixed_idx.len(), fixed.len());
    let base = ms.iter_mut().map(|m| m.eval_parsed(&parsed)).fold(f64::INFINITY, f64::min);
    let mut best = base;
    let mut best_map: Option<Vec<usize>> = None;
    let mut evals = 0usize;
    let mut map: Vec<usize> = (0..orders.len()).collect();
    let mut cand = parsed.clone();
    r_permutations(&slots, sells.len(), &mut |positions| {
        for (&i, &src) in positions.iter().zip(sell_idx.iter()) {
            map[i] = src;
        }
        let mut f = fixed_idx.iter();
        for &i in slots.iter() {
            if !positions.contains(&i) {
                match f.next() {
                    Some(&src) => map[i] = src,
                    None => map[i] = i,
                }
            }
        }
        if (0..orders.len()).all(|i| class[map[i]] == class[i]) {
            return true;
        }
        evals += 1;
        if evals > CXD_BUDGET {
            return false;
        }
        for i in 0..orders.len() {
            cand[i] = parsed[map[i]];
        }
        let val = ms.iter_mut().map(|m| m.eval_parsed(&cand)).fold(f64::INFINITY, f64::min);
        if val > best + 0.5 {
            best = val;
            best_map = Some(map.clone());
        }
        true
    });
    match best_map {
        Some(b) => {
            let mut r = action;
            r.market = b.iter().map(|&k| orders[k].clone()).collect();
            r
        }
        None => action,
    }
}

// ---- E410 ------------------------------------------------------------------------------------
pub fn e410(action: Action, v: &View, ch: &Chassis, reactive: &[usize]) -> Action {
    let (step, seat) = (v.step, v.obs.player);
    let day = step.div_euclid(24);
    let mut units = action.units();
    if !units.iter().any(|c| c.len() == 1 && c.op() == "FERTILIZE") {
        return action;
    }
    let mut farm = sim::farm(v.farm());
    let mut private = sim::private(&v.obs.shed, &v.obs.seeds, &v.obs.invs);
    let mut positions: Vec<(i64, i64)> = vec![farm.farmer];
    positions.extend(farm.hands.iter().copied());
    let Some(route) = ch.players.iter().find(|(p, _)| *p == seat).and_then(|(_, s)| s.route) else { return action };
    let tape = &ch.route(route).tape;
    let lo = ((day * 24) as usize).min(tape.len());
    let hi = (((day + 1) * 24).min(719) as usize).min(tape.len());
    let expected = tape[lo..hi].iter().map(|a| a.hands.len()).max().unwrap_or(0);
    let mut changed = false;
    let n = units.len().min(positions.len());
    for i in 0..n {
        let pos = positions[i];
        let mut cmd = units[i].clone();
        if cmd.len() == 1 && cmd.op() == "FERTILIZE" {
            let fert = private.inventories.get(i).map(|m| m.get("FERTILIZER")).unwrap_or(0);
            if let Cell::Plant { crop, planted_day, watered_today, yield_units, fertilized_until_day, .. } =
                farm.tiles[pos.1 as usize][pos.0 as usize].clone()
            {
                if (crop == "WHEAT" || crop == "CARROT") && fert > 0 {
                    let until = fertilized_until_day;
                    let covered = until >= day + 2;
                    let mut skip = covered;
                    if !skip && i <= expected && !reactive.contains(&i) {
                        let vis = ca::visits(v, ch, &action, pos, 718.min((planted_day + 6) * 24), Some(step + 1));
                        let wd = if watered_today { day } else { -1 };
                        let old = ca::yield_path(&crop, planted_day, &vis, yield_units, until, wd, step).0;
                        let new = ca::yield_path(&crop, planted_day, &vis, yield_units, until.max(day + 2), wd, step).0;
                        skip = old > 0 && old == new;
                    }
                    if skip {
                        cmd = Cmd::pass();
                        units[i] = cmd.clone();
                        changed = true;
                    }
                }
            }
        }
        sim::apply(&mut farm, &mut private, i, &cmd, day);
    }
    if !changed {
        return action;
    }
    let mut r = action;
    r.set_units(units);
    r
}

// ---- E402 ------------------------------------------------------------------------------------
/// Returns the new action and the carrot seed it cut (for CA's `spare_carrot`).
pub fn e402(action: Action, v: &View, ch: &Chassis) -> (Action, i64) {
    let (step, seat) = (v.step, v.obs.player);
    if step < 624 {
        return (action, 0);
    }
    let crop = |c: &str| c == "WHEAT" || c == "CARROT";
    if !action.market.iter().any(|o| o.len() >= 3 && o.op() == "BUY_SEED" && crop(o.s(1))) {
        return (action, 0);
    }
    let Some(native) = ch.players.iter().find(|(p, _)| *p == seat).map(|(_, s)| s) else { return (action, 0) };
    let Some(route) = native.route else { return (action, 0) };
    let mut remaining: i64 = 0;
    for t in (step + 1)..719 {
        let r = if t >= 648 { 2 } else { route };
        if let Some(a) = ch.route(r).tape.get(t as usize) {
            remaining += a.units().iter().filter(|c| c.len() > 1 && c.op() == "PLANT" && crop(c.s(1))).count() as i64;
        }
    }
    remaining += native.pending.iter().flat_map(|(_, q)| q.iter()).filter(|(_, c)| c.len() > 1 && c.op() == "PLANT" && crop(c.s(1))).count() as i64;
    let units = action.units();
    let mut available: Qty = ["WHEAT", "CARROT"]
        .iter()
        .map(|p| (*p, (qget(&v.obs.seeds, p) - units.iter().filter(|c| c.len() >= 2 && c.op() == "PLANT" && c.s(1) == *p).count() as i64).max(0)))
        .collect();
    let mut out: Vec<Cmd> = vec![];
    let mut changed = false;
    let mut carrot_cut = 0;
    for o in &action.market {
        if o.len() >= 3 && o.op() == "BUY_SEED" && crop(o.s(1)) {
            let p = o.s(1);
            let qty = o.n(2).max(0);
            let keep = qty.min((remaining - qget(&available, p)).max(0));
            qadd(&mut available, p, keep);
            if keep < qty {
                changed = true;
                if p == "CARROT" {
                    carrot_cut += qty - keep;
                }
                out.push(if keep != 0 { Cmd::order("BUY_SEED", p, keep) } else { Cmd::default() });
                continue;
            }
        }
        out.push(o.clone());
    }
    if !changed {
        return (action, 0);
    }
    let mut r = action;
    r.market = out;
    (r, carrot_cut)
}

// ---- MG / IG ---------------------------------------------------------------------------------
pub fn mg(action: Action) -> Action {
    if action.market.len() < 2 {
        return action;
    }
    let mut seen: Vec<((&str, &str), usize)> = vec![];
    let mut out: Vec<Cmd> = vec![];
    let mut changed = false;
    for o in &action.market {
        if !o.is_empty() && o.len() >= 3 && matches!(o.op(), "SELL" | "BUY_PRODUCT" | "BUY_SEED") {
            let key = (o.op(), o.s(1));
            if let Some((_, at)) = seen.iter().find(|(k, _)| *k == key) {
                let nv = out[*at].n(2) + o.n(2);
                out[*at].set_n(2, nv);
                changed = true;
                continue;
            }
            seen.push((key, out.len()));
        }
        out.push(o.clone());
    }
    if !changed {
        return action;
    }
    let mut r = action;
    r.market = out;
    r
}

const IG_CASH: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];

pub fn ig(action: Action, v: &View, ch: &Chassis) -> Action {
    let market = &action.market;
    if market.len() < 2 {
        return action;
    }
    let projected = ch.projected_shed(&action, v);
    let mut remaining: Qty = IG_CASH.iter().map(|i| (*i, qget(&projected, i).max(0))).collect();
    let mut revised: Vec<Cmd> = vec![];
    for o in market {
        if o.is_sell3() && IG_CASH.contains(&o.s(1)) {
            let requested = o.n(2).max(0);
            let executed = requested.min(qget(&remaining, o.s(1)));
            qadd(&mut remaining, o.s(1), -executed);
            revised.push(if executed <= 0 { Cmd::default() } else { o.clone() });
        } else {
            revised.push(o.clone());
        }
    }
    let mut holes: Vec<usize> = vec![];
    for index in 0..revised.len() {
        if revised[index].is_empty() {
            holes.push(index);
            continue;
        }
        let o = &revised[index];
        let movable = o.is_sell3() && IG_CASH.contains(&o.s(1)) && o.n(2) > 0;
        if !movable || holes.is_empty() {
            continue;
        }
        let target = holes.remove(0);
        revised[target] = revised[index].clone();
        revised[index] = Cmd::default();
        holes.push(index);
    }
    if revised == *market {
        return action;
    }
    let mut r = action;
    r.market = revised;
    r
}

// ---- RSA -------------------------------------------------------------------------------------
pub const RSA_LOOK: i64 = 5;
pub const RSA_FROM: i64 = 144;
pub const RSA_TO: i64 = 718;
pub const RSA_MIN_FRAC: f64 = 0.5;
const RSA_ITEMS: [&str; 7] = ["STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"];
fn rsa_base(item: &str) -> i64 {
    match item {
        "STRAWBERRY" => 120,
        "WOOL" => 200,
        "EGG" => 50,
        "MILK" => 160,
        "MELON" => 250,
        "CARROT" => 35,
        "TOMATO" => 60,
        _ => 0,
    }
}

pub fn rsa(action: Action, v: &View, ch: &Chassis, look: i64, min_frac: f64) -> Action {
    let step = v.step;
    if step % 24 == 23 || !(RSA_FROM..RSA_TO).contains(&step) {
        return action;
    }
    let Some(route) = ch.players.iter().find(|(p, _)| *p == v.obs.player).and_then(|(_, s)| s.route) else { return action };
    let mut plan: Vec<(&'static str, i64)> = vec![];
    let mut first: Option<Cmd> = None;
    for off in 1..=look {
        let fs = step + off;
        if fs > 718 {
            break;
        }
        let r = if fs >= 648 { 2 } else { route };
        let Some(a) = ch.route(r).tape.get(fs as usize) else { continue };
        for o in &a.market {
            if o.is_empty() || o.len() < 3 {
                continue;
            }
            if first.is_none() {
                first = Some(o.clone());
            }
            if o.op() == "SELL" && RSA_ITEMS.contains(&o.s(1)) && o.n(2) > 0 {
                plan.push((o.s(1), o.n(2)));
            }
        }
    }
    let protected = first.as_ref().filter(|f| f.op() == "SELL").map(|f| f.s(1));
    plan.retain(|(i, _)| Some(*i) != protected);
    if plan.is_empty() {
        return action;
    }
    let mut market = action.market.clone();
    if market.iter().any(|o| o.len() > 1 && o.op() == "BUY_PRODUCT") {
        return action;
    }
    let stock = ch.projected_shed(&action, v);
    let mut selling: Qty = vec![];
    for o in &market {
        if o.is_sell3() {
            qadd(&mut selling, o.s(1), o.n(2).max(0));
        }
    }
    let picked: Vec<&str> = action.units().iter().filter(|c| c.len() > 1 && c.op() == "PICKUP").map(|c| c.s(1)).collect();
    // sorted(set(items), key=-price): equal prices fall back to RSA_ITEMS order.
    let mut items: Vec<&'static str> = RSA_ITEMS.iter().copied().filter(|it| plan.iter().any(|(p, _)| p == it)).collect();
    items.sort_by_key(|it| -v.price(it));
    let mut extra: Vec<Cmd> = vec![];
    for item in items {
        let floor = 2f64.max(min_frac * rsa_base(item) as f64);
        if picked.contains(&item) || (v.price(item) as f64) < floor {
            continue;
        }
        let avail = qget(&stock, item) - qget(&selling, item);
        let want = avail.min(plan.iter().filter(|(i, _)| *i == item).map(|(_, q)| q).sum());
        if want < 1 {
            continue;
        }
        if let Some(o) = market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
            let nv = o.n(2) + want;
            o.set_n(2, nv);
        } else if market.len() + extra.len() < 10 {
            extra.push(Cmd::order("SELL", item, want));
        }
    }
    if extra.is_empty() && market == action.market {
        return action;
    }
    let mut r = action;
    r.market = extra.into_iter().chain(market).collect();
    r
}
