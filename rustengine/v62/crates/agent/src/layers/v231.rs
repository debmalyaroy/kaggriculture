//! V231 bounded livestock substitution (agent lines 1538-1647): swap a scheduled 1-2 SHEEP
//! purchase for COW in milk worlds, carry/place the cows in the sheep's slots, and add the
//! extra milk they produce to an existing MILK sale.
use crate::act::{Action, Cmd, Tok};
use crate::chassis::Chassis;
use crate::obs::qget;
use crate::view::View;

pub const CAP: i64 = 4;
const ANIMALS: [&str; 3] = ["COW", "SHEEP", "GOOSE"];

#[derive(Clone, Debug, Default)]
struct State {
    last: i64,
    confirmed: i64,
    reserved: i64,
    pending_buy: Option<(i64, i64)>,
    /// actor -> cows carried.
    carrying: Vec<(usize, i64)>,
    /// (actor, site, day).
    pending_places: Vec<(usize, (i64, i64), i64)>,
    /// site -> placed day.
    sites: Vec<((i64, i64), i64)>,
    milk_credit: i64,
}

fn get<K: PartialEq, V: Copy + Default>(m: &[(K, V)], k: &K) -> V {
    m.iter().find(|(a, _)| a == k).map(|(_, v)| *v).unwrap_or_default()
}
fn set<K: PartialEq, V>(m: &mut Vec<(K, V)>, k: K, v: V) {
    if let Some(e) = m.iter_mut().find(|(a, _)| *a == k) {
        e.1 = v;
    } else {
        m.push((k, v));
    }
}

#[derive(Default)]
pub struct V231 {
    players: Vec<(i64, State)>,
}

impl V231 {
    /// The state is (re)created before the parent runs.
    pub fn pre(&mut self, v: &View) {
        let p = v.obs.player;
        match self.players.iter_mut().find(|(k, _)| *k == p) {
            Some((_, s)) if v.step > s.last => {}
            Some((_, s)) => *s = State { last: -1, ..Default::default() },
            None => self.players.push((p, State { last: -1, ..Default::default() })),
        }
    }

    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let p = v.obs.player;
        let Some(i) = self.players.iter().position(|(k, _)| *k == p) else { return action };
        let mut st = std::mem::take(&mut self.players[i].1);
        let out = controller(v, action, &mut st, CAP, ch);
        self.players[i].1 = st;
        out
    }
}

fn controller(v: &View, action: Action, st: &mut State, cap: i64, ch: &Chassis) -> Action {
    let step = v.step;
    let farm = v.farm();
    let shed = &v.obs.shed;
    let invs = &v.obs.invs;
    let positions = &v.positions;
    if let Some((before, q)) = st.pending_buy.take() {
        let gained = (qget(shed, "COW") - before).max(0);
        let confirmed = q.min(gained);
        st.confirmed += confirmed;
        st.reserved += confirmed;
    }
    for &(actor, site, day) in &st.pending_places.clone() {
        let t = farm.tile(site.0, site.1);
        if t.is_dict() && t.animal == "COW" && t.placed_day == day {
            set(&mut st.sites, site, day);
            let c = get(&st.carrying, &actor);
            set(&mut st.carrying, actor, (c - 1).max(0));
        }
    }
    st.pending_places.clear();
    st.last = step;
    let mut result = action;
    let mut workers = result.units();
    let mut seen_harvest: Vec<(i64, i64)> = vec![];
    let mut cow_available = qget(shed, "COW");
    let mut occupied: Vec<(i64, i64)> = vec![];
    let n = workers.len().min(positions.len());
    for actor in 0..n {
        static EMPTY: Vec<(&str, i64)> = Vec::new();
        let inventory = invs.get(actor).unwrap_or(&EMPTY);
        let (x, y) = positions[actor];
        let tile = farm.tile(x, y);
        let site = (x, y);
        let work = &mut workers[actor];
        let site_day = st.sites.iter().find(|(s, _)| *s == site).map(|(_, d)| *d);
        if work.len() == 1
            && work.op() == "HARVEST"
            && site_day.is_some()
            && !seen_harvest.contains(&site)
            && tile.is_dict()
            && tile.animal == "COW"
            && Some(tile.placed_day) == site_day
        {
            st.milk_credit += tile.yield_units.max(0);
            seen_harvest.push(site);
        }
        if work.len() >= 2 && work.op() == "PICKUP" && work.s(1) == "SHEEP" {
            let quantity = if work.len() > 2 { work.n(2) } else { 1 }.max(0);
            let center = farm.rows as i64 / 2;
            if quantity != 0
                && st.reserved >= quantity
                && cow_available >= quantity
                && (x == center - 1 || x == center)
                && (y == center - 1 || y == center)
                && !ANIMALS.iter().any(|a| qget(inventory, a) != 0)
            {
                work.0[1] = Tok::S("COW");
                st.reserved -= quantity;
                cow_available -= quantity;
                let c = get(&st.carrying, &actor);
                set(&mut st.carrying, actor, c + quantity);
            }
        }
        if work.len() >= 2
            && work.op() == "PLACE"
            && work.s(1) == "SHEEP"
            && get(&st.carrying, &actor) > 0
            && qget(inventory, "COW") > 0
            && tile.is_dict()
            && tile.kind == "PASTURE"
            && !tile.has_animal()
            && !occupied.contains(&site)
        {
            work.0[1] = Tok::S("COW");
            st.pending_places.push((actor, site, step.div_euclid(24)));
        }
        if work.len() >= 2 && work.op() == "PLACE" && ANIMALS.contains(&work.s(1)) && qget(inventory, work.s(1)) > 0 {
            occupied.push(site);
        }
    }
    result.set_units(workers);
    let shops = v.shops();
    let mut cows = 0;
    let mut sheep = 0;
    for t in &farm.tiles {
        if t.is_dict() {
            if t.animal == "COW" {
                cows += 1;
            } else if t.animal == "SHEEP" {
                sheep += 1;
            }
        }
    }
    let cargo: i64 = invs.iter().map(|m| ANIMALS.iter().map(|a| qget(m, a)).sum::<i64>()).sum();
    let stock_animals: i64 = ANIMALS.iter().map(|a| qget(shed, a)).sum();
    let milk_shops = shops.iter().filter(|s| matches!(**s, "PIZZA_SHOP" | "ICE_CREAM_SHOP" | "SMOOTHIE_SHOP")).count();
    let animal_orders: Vec<usize> =
        (0..result.market.len()).filter(|&k| result.market[k].len() >= 3 && result.market[k].op() == "BUY_ANIMAL").collect();
    if (216..=227).contains(&step)
        && shops.len() >= 3
        && st.confirmed < cap
        && st.reserved == 0
        && !st.carrying.iter().any(|(_, c)| *c != 0)
        && st.pending_places.is_empty()
        && cargo == 0
        && stock_animals == 0
        && animal_orders.len() == 1
        && result.market[animal_orders[0]].s(1) == "SHEEP"
        && milk_shops >= 2
        && !shops.contains(&"YARN_STORE")
        && v.price("MILK") >= v.price("WOOL")
        && cows >= 4
        && sheep >= 2
    {
        let o = &mut result.market[animal_orders[0]];
        let quantity = o.n(2);
        if (1..=2).contains(&quantity) && quantity <= cap - st.confirmed {
            o.0[1] = Tok::S("COW");
            st.pending_buy = Some((qget(shed, "COW"), quantity));
        }
    }
    if st.milk_credit > 0 {
        let stock = ch.projected_shed(&result, v);
        let total_planned: i64 =
            result.market.iter().filter(|o| o.len() >= 3 && o.op() == "SELL" && o.s(1) == "MILK").map(|o| o.n(2).max(0)).sum();
        let extra = st.milk_credit.min((qget(&stock, "MILK") - total_planned).max(0));
        if extra != 0 {
            for o in result.market.iter_mut() {
                if o.len() >= 3 && o.op() == "SELL" && o.s(1) == "MILK" && o.n(2) > 0 {
                    let nv = o.n(2) + extra;
                    o.set_n(2, nv);
                    st.milk_credit -= extra;
                    break;
                }
            }
        }
    }
    let _ = Cmd::pass;
    result
}
