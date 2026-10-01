//! v44y group (agent 5650-6081): PG pre-guard (hours 21-22), V44Y lockstep best-response SELL
//! ordering (always on from step 216: `_V44Y_REORDER_GATE = False`), Y shop-aware herd
//! (`_Y_CFG` yarnsheep/yarngeese, days 8-11) and E334 / E335 sale-slot compaction.
use super::r85;
use super::race::{fields, to_qty};
use crate::act::{Action, Cmd, Tok};
use crate::chassis::Chassis;
use crate::obs::{qadd, qget, qset, Qty};
use crate::view::{fib, View, PRODUCTS};

// ---- PG ----------------------------------------------------------------------------------------
const Y_ITEMS: [&str; 5] = ["MILK", "STRAWBERRY", "MELON", "WOOL", "TOMATO"];
const Y_MARGIN: i64 = -6;

pub fn preguard(action: Action, v: &View) -> Action {
    let step = v.step;
    let h = step.rem_euclid(24);
    if !(h == 21 || h == 22) || step.div_euclid(24) < 1 || step >= 696 {
        return action;
    }
    let orders = &action.market;
    if orders.len() >= 10 {
        return action;
    }
    let (_, private) = fields(v, &action);
    let (stock, _, _) = r85::market_stock(&to_qty(&private.shed), orders);
    let carried: i64 = private.inventories.iter().flat_map(|m| m.0.iter().map(|(_, n)| (*n).max(0))).sum();
    let mut needed = stock.iter().map(|(_, n)| (*n).max(0)).sum::<i64>() + carried - 99 - Y_MARGIN;
    if needed <= 0 {
        return action;
    }
    let mut items: Vec<&'static str> = PRODUCTS.to_vec();
    items.sort_by_key(|it| -v.price(it));
    let mut extra: Vec<Cmd> = vec![];
    for item in items {
        let avail = qget(&stock, item).max(0);
        let qty = needed.min(avail);
        if qty <= 0 {
            continue;
        }
        if Y_ITEMS.contains(&item) && v.price(item) >= 2 {
            extra.push(Cmd::order("SELL", item, qty));
        }
        needed -= qty;
        if needed <= 0 {
            break;
        }
    }
    if extra.is_empty() || orders.len() + extra.len() > 10 {
        return action;
    }
    let mut r = action;
    r.market.extend(extra);
    r
}

// ---- V44Y ------------------------------------------------------------------------------------
// The margin model (lockstep race + per-(item, schedule) cache) is shared with CXD: `tail::Margin`.

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn rec(cur: &mut Vec<usize>, used: &mut [bool], n: usize, out: &mut Vec<Vec<usize>>) {
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

/// `_v44y_reorder`.
pub fn v44y(action: Action, v: &View, ch: &Chassis) -> Action {
    if v.step < 216 || action.market.len() < 2 {
        return action;
    }
    let mut orders = action.market.clone();
    let mut blocks = vec![];
    let mut i = 0;
    while i < orders.len() {
        if !orders[i].is_empty() && orders[i].op() == "SELL" {
            let mut j = i;
            while j < orders.len() && !orders[j].is_empty() && orders[j].op() == "SELL" {
                j += 1;
            }
            if (2..=6).contains(&(j - i)) {
                blocks.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if blocks.is_empty() {
        return action;
    }
    let stock: Qty = ch.projected_shed(&action, v).into_iter().map(|(k, n)| (k, n.max(0))).collect();
    let inv = v.obs.mkt_inventory.clone();
    let mut m = super::tail::Margin::new(&orders, &inv, &stock);
    let mut parsed: Vec<Option<(&'static str, bool, i64)>> = orders.iter().map(super::tail::valid).collect();
    let base = m.eval_parsed(&parsed);
    let mut best = base;
    for (i, j) in blocks {
        let blk: Vec<Cmd> = orders[i..j].to_vec();
        let blk_parsed: Vec<Option<(&'static str, bool, i64)>> = parsed[i..j].to_vec();
        let mut seen: std::collections::HashSet<Vec<(&str, i64)>> = Default::default();
        let mut best_perm: Option<Vec<usize>> = None;
        let mut cand = parsed.clone();
        for perm in permutations(j - i) {
            let key: Vec<(&str, i64)> = perm.iter().map(|&p| (blk[p].s(1), blk[p].n(2))).collect();
            if !seen.insert(key) {
                continue;
            }
            for (k, &p) in perm.iter().enumerate() {
                cand[i + k] = blk_parsed[p];
            }
            let val = m.eval_parsed(&cand);
            if val > best + 0.5 {
                best = val;
                best_perm = Some(perm);
            }
        }
        if let Some(perm) = best_perm {
            for (k, &p) in perm.iter().enumerate() {
                orders[i + k] = blk[p].clone();
                parsed[i + k] = blk_parsed[p];
            }
        }
    }
    if best <= base + 0.5 {
        return action;
    }
    let mut r = action;
    r.market = orders;
    r
}

// ---- Y (shop-aware herd) -----------------------------------------------------------------------
fn y_cost(a: &str) -> i64 {
    match a {
        "COW" => 400,
        "SHEEP" => 500,
        "GOOSE" => 300,
        _ => 500,
    }
}
fn y_struct(a: &str) -> &'static str {
    if a == "GOOSE" {
        "COOP"
    } else {
        "PASTURE"
    }
}
fn y_product(a: &str) -> &'static str {
    match a {
        "COW" => "MILK",
        "SHEEP" => "WOOL",
        _ => "EGG",
    }
}
const Y_ANIMALS: [&str; 3] = ["COW", "SHEEP", "GOOSE"];

#[derive(Clone, Default)]
struct YSt {
    last: i64,
    pending: Vec<(&'static str, &'static str, i64, i64)>,
    credit: Vec<((&'static str, &'static str), i64)>,
    sites: Vec<((i64, i64), &'static str)>,
    sale: Qty,
    coop_swap: i64,
}

impl Y {
    /// A Y herd swap in flight for this player (option gates only block NEW swaps).
    pub fn active(&self, player: i64) -> bool {
        self.players.iter().find(|(p, _)| *p == player).is_some_and(|(_, s)| !s.pending.is_empty() || !s.credit.is_empty() || !s.sites.is_empty() || s.coop_swap != 0)
    }
}

#[derive(Default, Clone)]
pub struct Y {
    players: Vec<(i64, YSt)>,
}

fn y_cash(v: &View, ch: &Chassis, units_action: &Action, market: &[Cmd]) -> f64 {
    let farm = v.farm();
    let probe = Action { farmer: units_action.farmer.clone(), hands: units_action.hands.clone(), market: vec![] };
    let shed = ch.projected_shed(&probe, v);
    let mut cash = farm.money;
    let mut cost = 0.0f64;
    let mut hires = farm.hires_today;
    let quads = farm.quadrants.len() as i64;
    for o in market {
        if o.is_empty() {
            continue;
        }
        match o.op() {
            "SELL" if o.len() >= 3 => cash += 0.8 * (o.n(2).max(0).min(qget(&shed, o.s(1))) as f64) * v.price(o.s(1)) as f64,
            "BUY_ANIMAL" if o.len() >= 3 => cost += (o.n(2) * y_cost(o.s(1))) as f64,
            "BUY_PRODUCT" if o.len() >= 3 => cost += o.n(2) as f64 * (v.price(o.s(1)) as f64 + 10.0),
            "BUY_SEED" if o.len() >= 3 => cost += (o.n(2) * crate::view::seed_price(o.s(1)).unwrap_or(100)) as f64,
            "BUY_LAND" => cost += [1000.0, 2000.0, 4000.0][(quads - 1).clamp(0, 2) as usize],
            "HIRE" => {
                cost += fib(hires) as f64;
                hires += 1;
            }
            _ => {}
        }
    }
    cash - cost
}

impl Y {
    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let (seat, step) = (v.obs.player, v.step);
        let i = match self.players.iter().position(|(p, _)| *p == seat) {
            Some(i) if step > self.players[i].1.last => i,
            Some(i) => {
                self.players[i].1 = YSt { last: -1, ..Default::default() };
                i
            }
            None => {
                self.players.push((seat, YSt { last: -1, ..Default::default() }));
                self.players.len() - 1
            }
        };
        let mut st = std::mem::take(&mut self.players[i].1);
        st.last = step;
        let out = controller(action, v, ch, &mut st);
        self.players[i].1 = st;
        out
    }
}

fn controller(action: Action, v: &View, ch: &Chassis, st: &mut YSt) -> Action {
    let step = v.step;
    let day = step.div_euclid(24);
    let farm = v.farm();
    let shed = &v.obs.shed;
    let invs = &v.obs.invs;
    let shops = v.shops();
    let center = farm.rows as i64 / 2;
    // 1. confirm swapped purchases
    let mut gained: Vec<(&str, i64)> = vec![];
    for &(from, to, qty, before) in &st.pending.clone() {
        if !gained.iter().any(|(k, _)| *k == to) {
            gained.push((to, (qget(shed, to) - before).max(0)));
        }
        let g = gained.iter_mut().find(|(k, _)| *k == to).unwrap();
        let got = qty.min(g.1);
        g.1 -= got;
        if got > 0 {
            match st.credit.iter_mut().find(|(k, _)| *k == (from, to)) {
                Some(e) => e.1 += got,
                None => st.credit.push(((from, to), got)),
            }
            if y_struct(from) != y_struct(to) {
                st.coop_swap += got;
            }
        }
    }
    st.pending.clear();
    let mut result = action;
    // 2. purchase-point substitution
    if (8..=11).contains(&day) {
        for k in 0..result.market.len() {
            let o = result.market[k].clone();
            if !(o.len() >= 3 && o.op() == "BUY_ANIMAL" && Y_ANIMALS.contains(&o.s(1)) && (1..=2).contains(&o.n(2))) {
                continue;
            }
            let yarn = shops.contains(&"YARN_STORE");
            let to = match o.s(1) {
                "COW" if yarn => "SHEEP",
                "GOOSE" if yarn => "SHEEP",
                _ => continue,
            };
            if to == o.s(1) {
                continue;
            }
            let trial: Vec<Cmd> = result
                .market
                .iter()
                .map(|t| {
                    if *t == o {
                        let mut t = t.clone();
                        t.0[1] = Tok::S(to);
                        t
                    } else {
                        t.clone()
                    }
                })
                .collect();
            if y_cash(v, ch, &result, &trial) < 100.0 {
                continue;
            }
            st.pending.push((o.s(1), to, o.n(2), qget(shed, to)));
            result.market[k].0[1] = Tok::S(to);
        }
    }
    // 3. worker rewrites + harvest credit
    let mut workers = result.units();
    let mut avail: Vec<(&str, i64)> = Y_ANIMALS.iter().map(|a| (*a, qget(shed, a))).collect();
    let mut seen: Vec<(i64, i64)> = vec![];
    let mut occupied: Vec<(i64, i64)> = vec![];
    let n = workers.len().min(v.positions.len());
    for actor in 0..n {
        if workers[actor].is_empty() {
            continue;
        }
        static EMPTY: Qty = Vec::new();
        let inv = invs.get(actor).unwrap_or(&EMPTY);
        let (x, y) = v.positions[actor];
        let tile = farm.tile(x, y);
        let site = (x, y);
        let work = &mut workers[actor];
        let op = work.op();
        if op == "PICKUP" && work.len() >= 2 && Y_ANIMALS.contains(&work.s(1)) {
            let kind = work.s(1);
            let qty = if work.len() > 2 { work.n(2).max(1) } else { 1 };
            let a = avail.iter_mut().find(|(k, _)| *k == kind).unwrap();
            if a.1 >= qty {
                a.1 -= qty;
                continue;
            }
            if !((x == center - 1 || x == center) && (y == center - 1 || y == center)) {
                continue;
            }
            if Y_ANIMALS.iter().any(|a| qget(inv, a) != 0) {
                continue;
            }
            for k in 0..st.credit.len() {
                let ((frm, to), c) = st.credit[k];
                let at = avail.iter().find(|(kk, _)| *kk == to).map(|(_, n)| *n).unwrap_or(0);
                if frm == kind && c >= qty && at >= qty {
                    work.0[1] = Tok::S(to);
                    st.credit[k].1 = c - qty;
                    avail.iter_mut().find(|(kk, _)| *kk == to).unwrap().1 -= qty;
                    break;
                }
            }
        } else if op == "PLACE" && work.len() >= 2 && Y_ANIMALS.contains(&work.s(1)) {
            let kind = work.s(1);
            if qget(inv, kind) > 0 {
                continue;
            }
            for to in ["COW", "SHEEP", "GOOSE"] {
                if to != kind && qget(inv, to) > 0 && tile.is_dict() && tile.kind == y_struct(to) && !tile.has_animal() && !occupied.contains(&site) {
                    work.0[1] = Tok::S(to);
                    match st.sites.iter_mut().find(|(s, _)| *s == site) {
                        Some(e) => e.1 = to,
                        None => st.sites.push((site, to)),
                    }
                    occupied.push(site);
                    break;
                }
            }
        } else if op == "BUILD_COOP" && st.coop_swap > 0 {
            work.0[0] = Tok::S("BUILD_PASTURE");
        } else if op == "HARVEST" && !seen.contains(&site) {
            if let Some((_, animal)) = st.sites.iter().find(|(s, _)| *s == site) {
                if tile.is_dict() && tile.animal == *animal {
                    let units = tile.yield_units.max(0);
                    if units != 0 {
                        qadd(&mut st.sale, y_product(tile.animal), units);
                    }
                }
                seen.push(site);
            }
        }
    }
    result.set_units(workers);
    // 4. sell extra production at existing sale slots
    for k in 0..st.sale.len() {
        let (prod, credit) = st.sale[k];
        let credit = credit.min(qget(shed, prod));
        st.sale[k].1 = credit;
        if credit <= 0 {
            continue;
        }
        let planned: i64 = result.market.iter().filter(|o| o.is_sell3() && o.s(1) == prod).map(|o| o.n(2).max(0)).sum();
        if planned <= 0 {
            continue;
        }
        let extra = credit.min(qget(shed, prod) - planned);
        if extra <= 0 {
            continue;
        }
        if let Some(o) = result.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == prod && o.n(2) > 0) {
            let nv = o.n(2) + extra;
            o.set_n(2, nv);
            st.sale[k].1 = credit - extra;
        }
    }
    result
}

// ---- E334 / E335 -------------------------------------------------------------------------------
const E334_ITEMS: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];

fn e334_original(action: Action, v: &View) -> Action {
    let market = &action.market;
    let mut segments = vec![];
    let mut i = 0;
    let is_item_sell = |o: &Cmd| o.is_sell3() && E334_ITEMS.contains(&o.s(1));
    while i < market.len() {
        if is_item_sell(&market[i]) {
            let mut j = i + 1;
            while j < market.len() && is_item_sell(&market[j]) {
                j += 1;
            }
            if j - i >= 2 {
                segments.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if segments.is_empty() {
        return action;
    }
    let (_, private) = fields(v, &action);
    let mut remaining = to_qty(&private.shed);
    let mut new = market.clone();
    for (s, e) in segments {
        let mut q: Qty = vec![];
        for o in &market[s..e] {
            qadd(&mut q, o.s(1), o.n(2).max(0));
        }
        let mut kept: Vec<Cmd> = vec![];
        for (p, n) in &q {
            let k = (*n).min(qget(&remaining, p).max(0));
            if k != 0 {
                kept.push(Cmd::order("SELL", p, k));
                let cur = qget(&remaining, p);
                qset(&mut remaining, p, cur - k);
            }
        }
        let mut replacement = kept.clone();
        while replacement.len() < e - s {
            replacement.push(Cmd::default());
        }
        if replacement[..] != market[s..e] {
            new.splice(s..e, replacement);
        }
    }
    if new == *market {
        return action;
    }
    let mut r = action;
    r.market = new;
    r
}

/// `_e334_compact` as replaced by E335.
pub fn e335(action: Action, v: &View) -> Action {
    let market = &action.market;
    if v.step < 144 || market.len() < 2 {
        return action;
    }
    if !market.iter().all(|o| o.is_sell3()) {
        return e334_original(action, v);
    }
    let (_, private) = fields(v, &action);
    let mut remaining = to_qty(&private.shed);
    let mut effective: Vec<Cmd> = vec![];
    for o in market {
        let item = o.s(1);
        let q = o.n(2).max(0).min(qget(&remaining, item).max(0));
        let cur = qget(&remaining, item);
        qset(&mut remaining, item, (cur - q).max(0));
        effective.push(if q != 0 { Cmd::order("SELL", item, q) } else { Cmd::default() });
    }
    let mut new = effective.clone();
    let mut i = 0;
    while i < effective.len() {
        if !effective[i].is_empty() && !E334_ITEMS.contains(&effective[i].s(1)) {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < effective.len() && (effective[j].is_empty() || E334_ITEMS.contains(&effective[j].s(1))) {
            j += 1;
        }
        let mut q: Qty = vec![];
        for o in &effective[i..j] {
            if !o.is_empty() {
                qadd(&mut q, o.s(1), o.n(2));
            }
        }
        let mut kept: Vec<Cmd> = q.iter().map(|(p, n)| Cmd::order("SELL", p, *n)).collect();
        while kept.len() < j - i {
            kept.push(Cmd::default());
        }
        new.splice(i..j, kept);
        i = j;
    }
    if new == *market {
        return action;
    }
    let mut r = action;
    r.market = new;
    r
}
