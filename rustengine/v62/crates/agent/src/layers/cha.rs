//! cha20/cha22 tail, part 1 (agent 6136-7170): E343_WL weed-lag replay, ADV ready-stock sale
//! advance, T62A terminal clear, PIPE EarlyCycle opening (route-0 install), MA mirror classifier,
//! WB3 wheat buy-first, FX/EV/DP/MP quote-history lead-sells, BD buy-the-dip, MPX model lead,
//! SM shield-milk.
use crate::act::{Action, Cmd, Tok};
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qadd, qget, Qty};
use crate::view::{seed_price, View};

fn route_of(ch: &Chassis, player: i64) -> Option<i64> {
    ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route)
}
fn tape_units(ch: &Chassis, player: i64, t: i64) -> Vec<Cmd> {
    let Some(r) = route_of(ch, player) else { return vec![] };
    if !(0..=718).contains(&t) {
        return vec![];
    }
    let r = if t >= 648 { 2 } else { r };
    ch.route(r).tape.get(t as usize).map(|a| a.units()).unwrap_or_default()
}
fn tape_market(ch: &Chassis, player: i64, t: i64) -> Vec<Cmd> {
    let Some(r) = route_of(ch, player) else { return vec![] };
    if !(0..=718).contains(&t) {
        return vec![];
    }
    let r = if t >= 648 { 2 } else { r };
    ch.route(r).tape.get(t as usize).map(|a| a.market.clone()).unwrap_or_default()
}

// ---- E343_WL ---------------------------------------------------------------------------------
pub const WL_STEPS: i64 = 8;

#[derive(Clone, Default)]
struct WlSt {
    step: i64,
    active: Vec<(usize, i64, Cmd)>,
}

// ---- FX quote history (shared with EV / DP / MP) ---------------------------------------------
const FX_ITEMS: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];
const FX_QUOTE_WIN: i64 = 12;
const FX_FLOW_WIN: i64 = 8;
const FX_FLOW_MIN: i64 = 999;
const FX_LEAD_H: i64 = 12;
const FX_MIN_STEP: i64 = 96;

#[derive(Clone, Default)]
struct FxPrev {
    step: i64,
    inventory: Qty,
    prices: Qty,
    own: Qty,
    shops: Vec<&'static str>,
}
#[derive(Clone, Default)]
struct FxSt {
    step: i64,
    flow: Vec<((i64, &'static str), i64)>,
    quotes: Vec<(i64, Qty)>,
    prev: Option<FxPrev>,
}

#[derive(Clone, Default)]
struct BdSt {
    hist: Vec<(i64, i64)>,
    pending: i64,
    since: Option<i64>,
}

#[derive(Clone, Default)]
struct MpxSt {
    prev: Option<(i64, Qty, Qty)>,
    hist: Vec<(&'static str, Vec<i64>)>,
}

#[derive(Default)]
pub struct Cha {
    wl: Vec<(i64, WlSt)>,
    pipe: Vec<(i64, i64)>,
    ma_prior: Option<i64>,
    pub ma_mirror: bool,
    fx: Vec<(i64, FxSt)>,
    bd: Vec<(i64, BdSt)>,
    mpx: Vec<(i64, MpxSt)>,
    /// Pristine route-0 prefix (`_PIPE_RAW`).
    pipe_raw: Option<Vec<Action>>,
}

impl Cha {
    // ---- E343_WL -----------------------------------------------------------------------------
    pub fn wl_pre(&mut self, v: &View) {
        if v.step == 0 {
            self.wl.clear();
        }
    }

    pub fn wl(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let (step, player) = (v.step, v.obs.player);
        let i = match self.wl.iter().position(|(p, _)| *p == player) {
            Some(i) => i,
            None => {
                self.wl.push((player, WlSt { step: -1, active: vec![] }));
                self.wl.len() - 1
            }
        };
        let st = &mut self.wl[i].1;
        if step <= st.step {
            st.active.clear();
        }
        st.step = step;
        if step % 24 == 0 {
            st.active.clear();
        }
        let farm = v.farm();
        let positions = &v.positions;
        let mut units: Vec<Cmd> = vec![if action.farmer.is_empty() { Cmd::pass() } else { action.farmer.clone() }];
        units.extend(action.hands.iter().cloned());
        let tape_now = tape_units(ch, player, step);
        let tape_prev = tape_units(ch, player, step - 1);
        let mut changed = false;
        let mut keep: Vec<(usize, i64, Cmd)> = vec![];
        for (k, start, intended) in st.active.drain(..) {
            if k >= units.len() || k >= positions.len() {
                continue;
            }
            let age = step - start;
            if age == 1 {
                units[k] = intended.clone();
                changed = true;
                keep.push((k, start, intended));
            } else if (2..=1 + WL_STEPS).contains(&age) {
                let prev = tape_prev.get(k).filter(|c| !c.is_empty()).cloned().unwrap_or_else(Cmd::pass);
                units[k] = prev;
                changed = true;
                keep.push((k, start, intended));
            }
        }
        st.active = keep;
        let n = units.len().min(positions.len()).min(tape_now.len());
        for k in 0..n {
            if st.active.iter().any(|(a, _, _)| *a == k) {
                continue;
            }
            let intent = &tape_now[k];
            if intent.is_empty() || !matches!(intent.op(), "BUILD_PASTURE" | "BUILD_COOP") {
                continue;
            }
            let (x, y) = positions[k];
            let t = farm.tile(x, y);
            if !(t.is_dict() && t.kind == "WEED") {
                continue;
            }
            if !(units[k].len() == 1 && units[k].op() == "DIG") {
                units[k] = Cmd::new("DIG");
                changed = true;
            }
            st.active.push((k, step, intent.clone()));
        }
        if !changed {
            return action;
        }
        let mut r = action;
        r.set_units(units);
        r
    }

    // ---- PIPE --------------------------------------------------------------------------------
    /// PIPE pre phase: a player's first call of a game (re)installs the EarlyCycle opening into
    /// route 0 (and `CT_TABLE = {}`, i.e. CTRTABLE off).
    pub fn pipe_pre(&mut self, v: &View, ch: &mut Chassis) -> bool {
        let (player, step) = (v.obs.player, v.step);
        let fresh = match self.pipe.iter_mut().find(|(p, _)| *p == player) {
            Some((_, s)) if step > *s => false,
            Some(_) => true,
            None => {
                self.pipe.push((player, -1));
                true
            }
        };
        if fresh {
            pipe_install(ch, &mut self.pipe_raw);
        }
        fresh
    }

    pub fn pipe_post(&mut self, action: Action, v: &View) -> Action {
        let (player, step) = (v.obs.player, v.step);
        if let Some((_, s)) = self.pipe.iter_mut().find(|(p, _)| *p == player) {
            *s = step;
        }
        let farm = v.farm();
        if (step == 57 || step == 91) && !farm.hands.is_empty() {
            let drop0 = action.hands.first().is_some_and(|h| h.len() == 1 && h.op() == "DROP");
            if farm.hands[0] == (4, 4) && drop0 {
                let n = qget(v.inv(1), "WHEAT");
                if n > 0 {
                    let mut r = action.clone();
                    if let Some(o) = r.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == "WHEAT") {
                        let nv = o.n(2) + n;
                        o.set_n(2, nv);
                    } else if r.market.len() < 10 {
                        r.market.push(Cmd::order("SELL", "WHEAT", n));
                    } else {
                        return action;
                    }
                    return r;
                }
            }
        }
        action
    }

    // ---- MA / WB3 ----------------------------------------------------------------------------
    pub fn ma_pre(&mut self, v: &View) {
        let step = v.step;
        if step == 0 {
            self.ma_prior = None;
            self.ma_mirror = false;
        }
        if step == 1 {
            self.ma_prior = v.obs.mkt_inventory.iter().find(|(k, _)| *k == "WHEAT").map(|(_, n)| *n);
        }
        if step == 2 {
            if let (Some(prior), Some(now)) = (self.ma_prior, v.obs.mkt_inventory.iter().find(|(k, _)| *k == "WHEAT").map(|(_, n)| *n)) {
                self.ma_mirror = prior - now >= 15;
            }
        }
    }

    pub fn wb3(&self, action: Action, v: &View) -> Action {
        let s: Vec<&str> = v.shops().iter().take(2).copied().collect();
        let pair = |a: &str, b: &str| s.len() == 2 && s[0] == a && s[1] == b;
        let in_pairs = pair("BRUNCH_SPOT", "BRUNCH_SPOT") || pair("BAKERY", "BRUNCH_SPOT") || pair("PIZZA_SHOP", "SMOOTHIE_SHOP");
        let target = (self.ma_mirror && in_pairs) || pair("BRUNCH_SPOT", "BRUNCH_SPOT") || pair("BAKERY", "BRUNCH_SPOT");
        if !target || action.market.len() <= 1 {
            return action;
        }
        let buys: Vec<Cmd> = action.market.iter().filter(|o| o.len() >= 3 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT").cloned().collect();
        if buys.is_empty() || buys.contains(&action.market[0]) {
            return action;
        }
        let rest: Vec<Cmd> = action.market.iter().filter(|o| !buys.contains(o)).cloned().collect();
        let mut r = action;
        r.market = buys.into_iter().chain(rest).collect();
        r
    }

    // ---- FX / EV / DP / MP -------------------------------------------------------------------
    pub fn fx_pre(&mut self, v: &View) {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.fx.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.fx[i].1.step => i,
            Some(i) => {
                self.fx[i].1 = FxSt { step: -1, ..Default::default() };
                i
            }
            None => {
                self.fx.push((player, FxSt { step: -1, ..Default::default() }));
                self.fx.len() - 1
            }
        };
        let st = &mut self.fx[i].1;
        st.step = step;
        // _fx_update
        if let Some(prev) = st.prev.as_ref().filter(|p| p.step == step - 1) {
            let draw = town_draw(&prev.shops, prev.step);
            let prev = prev.clone();
            for item in FX_ITEMS {
                if qget(&prev.prices, item) <= 3 {
                    continue;
                }
                let sold = qget(&v.obs.mkt_inventory, item) - qget(&prev.inventory, item) + qget(&draw, item) - qget(&prev.own, item);
                if sold > 0 {
                    let key = (prev.step, item);
                    match st.flow.iter_mut().find(|(k, _)| *k == key) {
                        Some(e) => e.1 = sold,
                        None => st.flow.push((key, sold)),
                    }
                }
            }
            if st.flow.len() > 512 {
                let mut keys: Vec<(i64, &str)> = st.flow.iter().map(|(k, _)| *k).collect();
                keys.sort();
                let drop: Vec<(i64, &str)> = keys.into_iter().take(256).collect();
                st.flow.retain(|(k, _)| !drop.contains(k));
            }
        }
    }

    fn quotes(&self, player: i64) -> &[(i64, Qty)] {
        self.fx.iter().find(|(p, _)| *p == player).map(|(_, s)| s.quotes.as_slice()).unwrap_or(&[])
    }

    pub fn fx_post(&mut self, action: Action, v: &View, ch: &Chassis, ev: Option<i64>) -> Action {
        let (player, step) = (v.obs.player, v.step);
        let Some(i) = self.fx.iter().position(|(p, _)| *p == player) else { return action };
        // _fx_apply
        {
            let st = &mut self.fx[i].1;
            let q: Qty = FX_ITEMS.iter().map(|it| (*it, v.price(it))).collect();
            match st.quotes.iter_mut().find(|(t, _)| *t == step) {
                Some(e) => e.1 = q,
                None => st.quotes.push((step, q)),
            }
            if st.quotes.len() > 96 {
                let mut ks: Vec<i64> = st.quotes.iter().map(|(t, _)| *t).collect();
                ks.sort();
                let drop: Vec<i64> = ks.into_iter().take(48).collect();
                st.quotes.retain(|(t, _)| !drop.contains(t));
            }
        }
        let mut out = action.clone();
        if (FX_MIN_STEP..700).contains(&step) && !self.ma_mirror && action.market.len() < 10 {
            if let Some(route) = route_of(ch, player) {
                let st = &self.fx[i].1;
                let mut market = action.market.clone();
                let already: Vec<&str> = market.iter().filter(|o| o.len() > 1 && matches!(o.op(), "SELL" | "BUY_PRODUCT")).map(|o| o.s(1)).collect();
                let stock = ch.projected_shed(&action, v);
                let tape = &ch.route(route).tape;
                let mut added = false;
                for item in FX_ITEMS {
                    if already.contains(&item) || market.len() >= 10 {
                        continue;
                    }
                    let flow: i64 = st.flow.iter().filter(|((t, it), _)| *it == item && step - FX_FLOW_WIN <= *t && *t < step).map(|(_, q)| q).sum();
                    if flow < FX_FLOW_MIN {
                        continue;
                    }
                    let q = v.price(item);
                    if q <= 3 || below_avg(&st.quotes, step, item, q) {
                        continue;
                    }
                    let planned = planned_sells(tape, item, step, FX_LEAD_H);
                    let qty = qget(&stock, item).min(planned);
                    if qty <= 0 {
                        continue;
                    }
                    market.insert(0, Cmd::order("SELL", item, qty));
                    added = true;
                }
                if added {
                    market.truncate(10);
                    out.market = market;
                }
            }
        }
        let out = match ev {
            Some(h) => window_lead(out, v, ch, self.quotes(player), &[15, 16, 17, 18, 19, 20], h),
            None => out,
        };
        let mut own: Qty = vec![];
        for o in &out.market {
            if !o.is_empty() && o.op() == "SELL" && o.len() >= 3 {
                qadd(&mut own, o.s(1), o.n(2).max(0));
            }
        }
        self.fx[i].1.prev = Some(FxPrev {
            step,
            inventory: v.obs.mkt_inventory.clone(),
            prices: v.obs.prices.clone(),
            own,
            shops: v.obs.shops.clone(),
        });
        out
    }

    pub fn dp(&self, action: Action, v: &View, ch: &Chassis, h: i64) -> Action {
        window_lead(action, v, ch, self.quotes(v.obs.player), &[0, 1, 2], h)
    }
    pub fn mp(&self, action: Action, v: &View, ch: &Chassis, h: i64) -> Action {
        window_lead(action, v, ch, self.quotes(v.obs.player), &[10, 11, 12, 13], h)
    }

    // ---- BD ----------------------------------------------------------------------------------
    pub fn bd_pre(&mut self, v: &View) {
        if v.step == 0 {
            self.bd.clear();
        }
    }
    pub fn bd(&mut self, action: Action, v: &View) -> Action {
        let step = v.step;
        if !(24..690).contains(&step) {
            return action;
        }
        let player = v.obs.player;
        let i = match self.bd.iter().position(|(p, _)| *p == player) {
            Some(i) => i,
            None => {
                self.bd.push((player, BdSt::default()));
                self.bd.len() - 1
            }
        };
        let st = &mut self.bd[i].1;
        let price = v.price("WHEAT");
        if price > 0 {
            st.hist.push((step, price));
            if st.hist.len() > 48 {
                st.hist.drain(..24);
            }
        }
        let mut market = action.market.clone();
        let avg = |h: &[(i64, i64)]| -> f64 {
            let vals: Vec<i64> = h.iter().filter(|(s, _)| step - 12 <= *s && *s < step).map(|(_, p)| *p).collect();
            if vals.is_empty() {
                price as f64
            } else {
                vals.iter().sum::<i64>() as f64 / vals.len() as f64
            }
        };
        if st.pending > 0 {
            let a = avg(&st.hist);
            let due = st.since.is_some_and(|s| step - s >= 6);
            if price > 0 && ((price as f64) <= a || due) {
                let has = market.iter().any(|o| o.len() >= 3 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT");
                if market.len() < 10 && !has {
                    let qty = st.pending.min(8);
                    market.push(Cmd::order("BUY_PRODUCT", "WHEAT", qty));
                    st.pending -= qty;
                }
                if st.pending <= 0 {
                    st.since = None;
                }
            }
        }
        if st.pending < 64 {
            let a = avg(&st.hist);
            for o in market.iter_mut() {
                if o.len() >= 3 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT" && o.n(2) >= 8 {
                    let q = o.n(2);
                    if price > 0 && (price as f64) > a {
                        let hold = (q - 1).min(64 - st.pending).min(q.div_euclid(2));
                        if hold > 0 {
                            o.set_n(2, q - hold);
                            st.pending += hold;
                            if st.since.is_none() {
                                st.since = Some(step);
                            }
                        }
                    }
                }
            }
        }
        let mut r = action;
        r.market = market;
        r
    }

    // ---- MPX ---------------------------------------------------------------------------------
    pub fn mpx_pre(&mut self, v: &View) {
        if v.step == 0 {
            self.mpx.clear();
        }
    }
    pub fn mpx(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        const ITEMS: [&str; 3] = ["MILK", "STRAWBERRY", "WOOL"];
        let step = v.step;
        if !(144..696).contains(&step) || !(12..23).contains(&step.rem_euclid(24)) {
            return action;
        }
        let player = v.obs.player;
        let i = match self.mpx.iter().position(|(p, _)| *p == player) {
            Some(i) => i,
            None => {
                self.mpx.push((player, MpxSt::default()));
                self.mpx.len() - 1
            }
        };
        let draw = |s: i64| (s % 4 == 0) as i64 + (s % 24 == 0) as i64;
        let st = &mut self.mpx[i].1;
        let inv_now: Qty = ITEMS.iter().map(|it| (*it, qget(&v.obs.mkt_inventory, it))).collect();
        if let Some((pstep, pinv, pown)) = st.prev.clone() {
            if pstep == step - 1 {
                for it in ITEMS {
                    let d = qget(&inv_now, it) - qget(&pinv, it) + draw(step - 1) - qget(&pown, it);
                    let h = match st.hist.iter_mut().find(|(k, _)| *k == it) {
                        Some(e) => &mut e.1,
                        None => {
                            st.hist.push((it, vec![]));
                            &mut st.hist.last_mut().unwrap().1
                        }
                    };
                    h.push(d.max(0));
                    if h.len() > 12 {
                        h.drain(..6);
                    }
                }
            }
        }
        let mut own: Qty = vec![];
        let mut orders = action.market.clone();
        for o in &orders {
            if o.is_sell3() && ITEMS.contains(&o.s(1)) {
                qadd(&mut own, o.s(1), o.n(2));
            }
        }
        st.prev = Some((step, inv_now.clone(), own.clone()));
        let already: Vec<&str> = orders.iter().filter(|o| o.len() > 1 && o.op() == "SELL").map(|o| o.s(1)).collect();
        if route_of(ch, player).is_none() {
            return action;
        }
        let stock = ch.projected_shed(&action, v);
        let mut added = false;
        for item in ITEMS {
            if already.contains(&item) || orders.len() >= 10 {
                continue;
            }
            let avail = qget(&stock, item);
            if avail <= 0 {
                continue;
            }
            if v.price(item) <= 1 {
                continue;
            }
            let rival = st.hist.iter().find(|(k, _)| *k == item).map(|(_, h)| h.clone()).unwrap_or_default();
            let tail: Vec<i64> = rival.iter().rev().take(4).rev().copied().collect();
            let rival_avg = if rival.is_empty() { 0.0 } else { tail.iter().sum::<i64>() as f64 / tail.len() as f64 };
            let inv = qget(&v.obs.mkt_inventory, item);
            let p_cur = market::price(item, inv as f64) as f64;
            let inv_next = inv as f64 + rival_avg + 6.0 - draw(step) as f64;
            let p_next = market::price(item, (inv_next.trunc() as i64).max(0) as f64) as f64;
            if p_next < p_cur - 0.5 {
                let take = avail.min(3);
                if take <= 0 {
                    continue;
                }
                orders.insert(0, Cmd::order("SELL", item, take));
                if let Some((_, _, own)) = st.prev.as_mut() {
                    qadd(own, item, take);
                }
                added = true;
            }
        }
        if !added {
            return action;
        }
        orders.truncate(10);
        let mut r = action;
        r.market = orders;
        r
    }
}

fn town_draw(shops: &[&str], step: i64) -> Qty {
    let items_of = |s: &str| -> &'static [&'static str] {
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
    };
    let mut d: Qty = FX_ITEMS.iter().map(|i| (*i, 0)).collect();
    if step % 4 == 0 {
        for s in shops {
            let items = items_of(s);
            for it in items {
                if d.iter().any(|(k, _)| k == it) {
                    qadd(&mut d, it, if items.len() == 1 { 2 } else { 1 });
                }
            }
        }
    }
    if step % 24 == 0 {
        for e in d.iter_mut() {
            e.1 += 1;
        }
    }
    d
}

fn below_avg(quotes: &[(i64, Qty)], step: i64, item: &str, q: i64) -> bool {
    let vals: Vec<i64> =
        quotes.iter().filter(|(t, _)| step - FX_QUOTE_WIN <= *t && *t < step).map(|(_, m)| qget(m, item)).filter(|p| *p > 0).collect();
    !vals.is_empty() && (q as f64) < vals.iter().sum::<i64>() as f64 / vals.len() as f64
}

fn planned_sells(tape: &[Action], item: &str, step: i64, h: i64) -> i64 {
    let hi = (tape.len() as i64).min(step + h + 1);
    ((step + 1)..hi).map(|t| tape[t as usize].market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum::<i64>()).sum()
}

/// `_ev_apply` / `_dp_apply` / `_mp_apply`: in the given hours, lead-sell up to 3/4 of the next
/// `h` turns' planned sells of an item quoted at or above its trailing average.
fn window_lead(action: Action, v: &View, ch: &Chassis, quotes: &[(i64, Qty)], hours: &[i64], h: i64) -> Action {
    let step = v.step;
    if !(96..700).contains(&step) || !hours.contains(&step.rem_euclid(24)) {
        return action;
    }
    let mut market = action.market.clone();
    if market.len() >= 10 {
        return action;
    }
    let already: Vec<&str> = market.iter().filter(|o| o.len() > 1 && matches!(o.op(), "SELL" | "BUY_PRODUCT")).map(|o| o.s(1)).collect();
    let stock = ch.projected_shed(&action, v);
    let Some(route) = route_of(ch, v.obs.player) else { return action };
    let tape = &ch.route(route).tape;
    let mut added = false;
    for item in FX_ITEMS {
        if already.contains(&item) || market.len() >= 10 {
            continue;
        }
        let q = v.price(item);
        if q <= 3 || below_avg(quotes, step, item, q) {
            continue;
        }
        let planned = planned_sells(tape, item, step, h);
        if planned <= 0 {
            continue;
        }
        let qty = qget(&stock, item).min(((3 * planned + 3).div_euclid(4)).max(1));
        if qty <= 0 {
            continue;
        }
        market.insert(0, Cmd::order("SELL", item, qty));
        added = true;
    }
    if !added {
        return action;
    }
    market.truncate(10);
    let mut r = action;
    r.market = market;
    r
}

// ---- ADV -------------------------------------------------------------------------------------
const ADV_ITEMS: [&str; 7] = ["STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"];

pub fn adv(action: Action, v: &View, ch: &Chassis, look: i64) -> Action {
    let (step, player) = (v.step, v.obs.player);
    if step % 24 == 23 || !(216..718).contains(&step) {
        return action;
    }
    let Some(native) = ch.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s) else { return action };
    let debts = &native.sell.r36_debts;
    let mut plan: Vec<(i64, &'static str, i64)> = vec![];
    let mut first: Option<Cmd> = None;
    for off in 1..=look {
        let t = step + off;
        if t > 718 {
            break;
        }
        for o in tape_market(ch, player, t) {
            if o.is_empty() || o.len() < 3 {
                continue;
            }
            if first.is_none() {
                first = Some(o.clone());
            }
            if o.op() == "SELL" && ADV_ITEMS.contains(&o.s(1)) {
                let owed = debts.iter().find(|(s, _)| *s == t).map(|(_, m)| qget(m, o.s(1))).unwrap_or(0);
                let q = o.n(2).max(0) - owed;
                if q > 0 {
                    plan.push((t, o.s(1), q));
                }
            }
        }
    }
    let protected = first.as_ref().filter(|f| f.op() == "SELL").map(|f| f.s(1));
    plan.retain(|(_, it, _)| Some(*it) != protected);
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
    // sorted(set(items), key=-price): equal prices fall back to ADV_ITEMS order (Python's set
    // order is hash-randomised per process there).
    let mut items: Vec<&'static str> = ADV_ITEMS.iter().copied().filter(|it| plan.iter().any(|(_, p, _)| p == it)).collect();
    items.sort_by_key(|it| -v.price(it));
    let mut extra: Vec<Cmd> = vec![];
    let mut added = 0;
    for item in items {
        if picked.contains(&item) || v.price(item) < 2 {
            continue;
        }
        let mut avail = qget(&stock, item) - qget(&selling, item);
        if avail < 1 {
            continue;
        }
        let hit = market.iter().position(|o| o.is_sell3() && o.s(1) == item);
        if hit.is_none() && market.len() + extra.len() >= 10 {
            continue;
        }
        let mut n = 0;
        for &(_, it, q) in &plan {
            if it != item || avail <= 0 {
                continue;
            }
            let take = q.min(avail);
            n += take;
            avail -= take;
        }
        if n < 1 {
            continue;
        }
        match hit {
            Some(h) => {
                let nv = market[h].n(2) + n;
                market[h].set_n(2, nv);
            }
            None => extra.push(Cmd::order("SELL", item, n)),
        }
        added += n;
    }
    if added == 0 {
        return action;
    }
    let mut r = action;
    r.market = extra.into_iter().chain(market).collect();
    r
}

// ---- T62A ------------------------------------------------------------------------------------
pub fn t62a(action: Action, v: &View) -> Action {
    if v.step < 712 {
        return action;
    }
    let mut items: Vec<(&'static str, i64)> = v.obs.shed.iter().filter(|(k, q)| *q > 0 && v.price(k) >= 1).cloned().collect();
    if items.is_empty() {
        return action;
    }
    items.sort_by_key(|(k, _)| -v.price(k));
    let mut r = action;
    r.market = items.iter().take(10).map(|(k, q)| Cmd::order("SELL", k, *q)).collect();
    r
}

// ---- PIPE install ------------------------------------------------------------------------------
fn set_hand(a: &mut Action, h: usize, c: Cmd) {
    if h < a.hands.len() {
        a.hands[h] = c;
    }
}
fn c1(op: &str) -> Cmd {
    Cmd::new(op)
}

/// `_pipe_install('EarlyCycle')` on route 0 (`routes[0]` = the first route in dict order).
fn pipe_install(ch: &mut Chassis, raw: &mut Option<Vec<Action>>) {
    let Some(i0) = ch.route_idx(0) else { return };
    let r0 = &mut ch.routes[i0];
    let pristine = raw.get_or_insert_with(|| r0.tape[..96.min(r0.tape.len())].to_vec()).clone();
    let tape = &mut r0.tape;
    for (i, a) in pristine.into_iter().enumerate() {
        tape[i] = a;
    }
    tape[0].market = vec![Cmd::order("BUY_PRODUCT", "WHEAT", 5), Cmd::order("BUY_SEED", "WHEAT", 1)];
    tape[1].market.retain(|o| !(o.len() >= 3 && matches!(o.op(), "BUY_PRODUCT" | "SELL") && o.s(1) == "WHEAT"));
    let plant = Cmd(vec![Tok::S("PLANT"), Tok::S("WHEAT")]);
    for (s, c) in [(2, c1("WEST")), (3, c1("WEST")), (4, c1("WEST")), (5, plant), (6, c1("WATER"))] {
        set_hand(&mut tape[s], 1, c);
    }
    set_hand(&mut tape[29], 2, c1("WATER"));
    let cmds = ["WEST", "WEST", "WEST", "WATER", "HARVEST", "BUILD_PASTURE", "EAST", "EAST", "DROP"];
    for (s, c) in (49..58).zip(cmds) {
        set_hand(&mut tape[s], 0, c1(c));
    }
    for s in 53..58 {
        set_hand(&mut tape[s], 0, Cmd::pass());
    }
    let omw = ["EAST", "EAST", "WATER", "HARVEST", "BUILD_PASTURE", "EAST", "EAST", "DROP"];
    for (s, c) in (84..92).zip(omw) {
        set_hand(&mut tape[s], 0, c1(c));
    }
    r0.refresh();
}

// ---- SM --------------------------------------------------------------------------------------
pub fn sm(action: Action, v: &View) -> Action {
    const PRODUCTS: [&str; 9] = ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "TOMATO", "CARROT", "WHEAT", "FERTILIZER"];
    if !v.shops().iter().any(|s| matches!(*s, "PIZZA_SHOP" | "ICE_CREAM_SHOP" | "SMOOTHIE_SHOP")) {
        return action;
    }
    let market = &action.market;
    if market.is_empty() {
        return action;
    }
    let farm = v.farm();
    let shed = &v.obs.shed;
    let quads = farm.quadrants.len() as i64;
    let mut cash = farm.money;
    let mut sold: Qty = vec![];
    for o in market {
        if o.is_empty() {
            continue;
        }
        match o.op() {
            "SELL" if o.len() >= 3 => {
                let item = o.s(1);
                let take = o.n(2).max(0).min(qget(shed, item) - qget(&sold, item));
                if take > 0 {
                    cash += take as f64 * v.price(item) as f64;
                    qadd(&mut sold, item, take);
                }
            }
            "BUY_ANIMAL" if o.len() >= 3 => {
                cash -= (crate::view::animal_cost(o.s(1)).unwrap_or(400) * o.n(2).max(0)) as f64;
            }
            "BUY_SEED" if o.len() >= 3 => cash -= (seed_price(o.s(1)).unwrap_or(100) * o.n(2).max(0)) as f64,
            "BUY_PRODUCT" if o.len() >= 3 => cash -= v.price(o.s(1)) as f64 * o.n(2).max(0) as f64,
            "BUY_LAND" => {
                let extra = quads - 1;
                cash -= if (0..3).contains(&extra) { [1000.0, 2000.0, 4000.0][extra as usize] } else { 4000.0 };
            }
            _ => {}
        }
    }
    let mut need = -cash;
    if need <= 0.0 {
        return action;
    }
    let mut cands: Vec<(f64, &'static str, i64)> = vec![];
    for item in PRODUCTS {
        let price = v.price(item) as f64;
        let avail = qget(shed, item) - qget(&sold, item);
        if price > 1.0 && avail > 0 {
            cands.push((price, item, avail));
        }
    }
    cands.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut added: Vec<Cmd> = vec![];
    for (price, item, avail) in cands {
        if need <= 0.0 || market.len() + added.len() >= 10 {
            break;
        }
        let q = avail.min((need / price).trunc() as i64 + 1);
        if q <= 0 {
            continue;
        }
        added.push(Cmd::order("SELL", item, q));
        need -= q as f64 * price;
    }
    if added.is_empty() {
        return action;
    }
    let idx = market.iter().position(|o| !o.is_empty() && o.op().starts_with("BUY")).unwrap_or(market.len());
    let mut m: Vec<Cmd> = market[..idx].to_vec();
    m.extend(added);
    m.extend(market[idx..].iter().cloned());
    m.truncate(10);
    let mut r = action;
    r.market = m;
    r
}
