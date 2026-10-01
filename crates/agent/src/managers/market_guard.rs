//! MARKET GUARD manager (operator 30 Sep): the lead-sell / race guards that used to be chassis settings. The chassis
//! keeps only the repair guards (hand_align, weed_repair, budget_guard, room_guard, clamp_sells); these run at two
//! HOOK points inside the chassis step, exactly where they always ran (so the c4 behaviour is bit-identical):
//!   hook A (after weed_repair): suppression + R36 debt settlement, sell_lead (RACEPX glut gate -> R36 window ->
//!                               native lead), front_run
//!   hook B (after clamp_sells):  dead_stock, terminal_liquidation
//! and they are switched on ONLY by this manager (managers::Config "market_guard"), never by a chassis router.json
//! (chassis::Settings defaults them OFF). `final_check` is the manager's last pass over every order (legality, caps).
use crate::act::{Action, Cmd};
use crate::chassis::{add_sell, base_price, Chassis, SellState};
use crate::obs::{qadd, qget, Qty};
use crate::view::*;

impl Chassis {
    // ---- layer: sell_lead --------------------------------------------------------------
    /// `Chassis._sell_lead` as patched: RACEPX (`_v9_racepx_lead`) -> R36 window (`_r36_native_lead`)
    /// -> native.
    pub(crate) fn sell_lead(&self, action: &mut Action, v: &View, projected: &mut Qty, route: i64, step: i64, next_sup: &mut SellState) {
        if self.cfg.racepx {
            let blocked: Vec<&'static str> =
                PRODUCTS.iter().copied().filter(|i| v.price(i) <= base_price(i) + self.cfg.racepx_margin).collect();
            if !blocked.is_empty() {
                return self.lead_core(action, v, projected, route, step, next_sup, &blocked, false);
            }
        }
        if self.cfg.r36 && !(step < self.cfg.r36_lead_from || step >= self.cfg.r36_lead_to) {
            return;
        }
        self.lead_core(action, v, projected, route, step, next_sup, &[], true)
    }

    /// The lead-sale body. `skip_inputs` = the native WHEAT/FERTILIZER exclusion.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lead_core(&self, action: &mut Action, v: &View, projected: &mut Qty, route: i64, step: i64, next_sup: &mut SellState,
                 blocked: &[&str], skip_inputs: bool) {
        let cfg = &self.cfg;
        let nxt = step + 1;
        let unlock_period = 3 * cfg.turns_per_day;
        if nxt > LAST_ACT_STEP || nxt % unlock_period == 0 || step % cfg.lead_every == 0 {
            return;
        }
        let mut planned: Qty = vec![];
        if let Some(fut) = self.route(route).at(nxt) {
            for o in &fut.market {
                if o.is_sell3() && PRODUCTS.contains(&o.s(1)) {
                    qadd(&mut planned, o.s(1), o.n(2).max(0));
                }
            }
        }
        let already: Vec<&str> = action.market.iter().filter(|o| o.op() == "SELL" && o.len() > 1).map(|o| o.s(1)).collect();
        let race = self.race_signal.as_ref();
        for item in PRODUCTS {
            if race.is_some_and(|r| !r.contains(&item)) {
                continue; // lead_signal: no race on this item -> the tape's own sale stands
            }
            if (skip_inputs && (item == "WHEAT" || item == "FERTILIZER"))
                || blocked.contains(&item)
                || qget(&planned, item) <= 0
                || already.contains(&item)
            {
                continue;
            }
            let qty = qget(projected, item).min(qget(&planned, item));
            if qty <= 0 || v.price(item) < cfg.min_sell_price {
                continue;
            }
            if !add_sell(action, item, qty, cfg.max_orders, false) {
                break;
            }
            qadd(projected, item, -qty);
            qadd(&mut next_sup.suppress, item, qty);
        }
        if !next_sup.suppress.is_empty() {
            next_sup.due_step = nxt;
        }
    }

    // ---- layer: front_run (hook) ---------------------------------------------------------
    pub(crate) fn front_run(&self, action: &mut Action, v: &View, projected: &mut Qty, route: i64, step: i64, next_sup: &mut SellState) {
        const FRONT_RUN_ITEMS: [&str; 4] = ["MILK", "WOOL", "STRAWBERRY", "MELON"];
        let cfg = &self.cfg;
        let nxt = step + 1;
        let Some(plan) = self.opponent_plan.as_ref() else { return };
        if nxt > LAST_ACT_STEP || nxt as usize >= plan.len() {
            return;
        }
        let mut already: Vec<&str> = action.market.iter().filter(|o| o.op() == "SELL" && o.len() > 1).map(|o| o.s(1)).collect();
        for o in &plan[nxt as usize].market {
            if !(o.is_sell3() && FRONT_RUN_ITEMS.contains(&o.s(1))) {
                continue;
            }
            let item = o.s(1);
            if already.contains(&item) || v.price(item) < cfg.min_sell_price {
                continue;
            }
            let own_next: i64 = self
                .route(route)
                .at(nxt)
                .map(|a| a.market.iter().filter(|x| x.is_sell3() && x.s(1) == item).map(|x| x.n(2).max(0)).sum())
                .unwrap_or(0);
            let qty = qget(projected, item).min(o.n(2).max(0)).min(own_next);
            if qty <= 0 {
                continue;
            }
            if !add_sell(action, item, qty, cfg.max_orders, false) {
                break;
            }
            qadd(projected, item, -qty);
            already.push(item);
            qadd(&mut next_sup.suppress, item, qty);
        }
        if !next_sup.suppress.is_empty() {
            next_sup.due_step = nxt;
        }
    }

    // ---- layer: dead_stock ---------------------------------------------------------------
    pub(crate) fn dead_stock(&self, action: &mut Action, v: &View, projected: &Qty, route: i64, step: i64) {
        let mut planned: Qty = vec![];
        for o in &action.market {
            if o.is_sell3() {
                qadd(&mut planned, o.s(1), o.n(2));
            }
        }
        let day = step.div_euclid(self.cfg.turns_per_day);
        let r = self.route(route);
        let mut extra: Vec<Cmd> = vec![];
        for item in PRODUCTS {
            let have = qget(projected, item) - qget(&planned, item);
            if have <= 0 {
                continue;
            }
            let surplus = if day >= self.cfg.dead_stock_day { have } else { have - r.future_sells(item, step + 1) };
            if surplus > 0 && v.price(item) > self.cfg.dead_stock_min_price {
                extra.push(Cmd::order("SELL", item, surplus));
            }
        }
        extra.sort_by_key(|o| -v.price(o.s(1)) * o.n(2));
        action.market.extend(extra);
    }

    // ---- layer: terminal_liquidation -------------------------------------------------------
    pub(crate) fn terminal_liquidation(&self, action: &mut Action, projected: &Qty, step: i64) {
        if step < self.cfg.terminal_from {
            return;
        }
        action.market = projected
            .iter()
            .filter(|(it, q)| *q > 0 && PRODUCTS.contains(it))
            .map(|(it, q)| Cmd::order("SELL", it, *q))
            .take(self.cfg.max_orders)
            .collect();
    }

}

/// `_apply_suppression`: remove from this step's SELLs what was already sold a step early.
pub(crate) fn apply_suppression(action: &mut Action, s: &SellState, step: i64) {
    if s.due_step != step {
        return;
    }
    let mut remaining = s.suppress.clone();
    for o in action.market.iter_mut() {
        if o.is_sell3() && qget(&remaining, o.s(1)) > 0 {
            let item = o.s(1);
            let removed = o.n(2).max(0).min(qget(&remaining, item));
            let nv = o.n(2) - removed;
            o.set_n(2, nv);
            qadd(&mut remaining, item, -removed);
            // A zero-quantity order keeps later market race slots intact.
        }
    }
}

/// `_r36_suppress` tail: pop this step's R36 debts and take them off this step's SELLs.
pub(crate) fn settle_r36_debts(action: &mut Action, s: &mut SellState, step: i64) {
    let Some(i) = s.r36_debts.iter().position(|(k, _)| *k == step) else { return };
    let (_, mut due) = s.r36_debts.remove(i);
    for o in action.market.iter_mut() {
        if o.is_sell3() {
            let item = o.s(1);
            let removed = o.n(2).max(0).min(qget(&due, item));
            let nv = o.n(2) - removed;
            o.set_n(2, nv);
            qadd(&mut due, item, -removed);
        }
    }
}


/// The MARKET GUARD's final check over every order after all managers (knobs: `clamp` = SELLs sequentially clamped to
/// the projected shed keeping every slot, `min_price` = a SELL of an item quoting below it becomes a zero order (slot
/// kept), `max_orders` = the order list cut to the engine's cap, highest priority first as queued).
#[derive(Clone, Debug)]
pub struct FinalCheck {
    pub clamp: bool,
    pub min_price: i64,
    pub max_orders: usize,
}

impl Default for FinalCheck {
    fn default() -> Self {
        FinalCheck { clamp: true, min_price: 0, max_orders: 10 }
    }
}

impl FinalCheck {
    pub fn with(&self, j: &kagg_engine::json::Json) -> Result<FinalCheck, String> {
        let mut c = self.clone();
        for (k, v) in j.obj() {
            match k.as_str() {
                "on" | "note" => {}
                "clamp" => c.clamp = v.bool(),
                "min_price" => c.min_price = v.i64(),
                "max_orders" => c.max_orders = v.i64().clamp(1, 10) as usize,
                o => return Err(format!("final_check: unknown knob {o:?}")),
            }
        }
        Ok(c)
    }

    pub fn dump(&self) -> String {
        format!("{{\"clamp\":{},\"min_price\":{},\"max_orders\":{}}}", self.clamp, self.min_price, self.max_orders)
    }

    pub fn apply(&self, a: &mut Action, v: &View, ch: &Chassis) {
        if self.clamp {
            let mut avail = ch.projected_shed(a, v);
            for o in a.market.iter_mut() {
                if o.is_sell3() {
                    let item = o.s(1);
                    let have = qget(&avail, item).max(0);
                    let n = o.n(2).max(0).min(have);
                    qadd(&mut avail, item, -n);
                    o.set_n(2, n);
                }
            }
        }
        if self.min_price > 0 {
            for o in a.market.iter_mut() {
                if o.is_sell3() && v.price(o.s(1)) < self.min_price && v.step < LAST_ACT_STEP {
                    o.set_n(2, 0);
                }
            }
        }
        a.market.truncate(self.max_orders);
    }
}

/// `verify_fills`: at the start of step `step`, check the early sales made on `step - 1` against the shed we now
/// observe. Sold = projected shed then - shed now (the early sale was the only SELL of that item that step: sell_lead
/// skips items already sold, R36 blocks them). A shortfall is taken back from the R36 debts booked then (latest due
/// first), then from this step's suppression, so the tape keeps the sales that were never actually made early.
pub(crate) fn verify_fills(s: &mut SellState, v: &View, step: i64) {
    if s.early_step != step - 1 || s.early.is_empty() {
        s.early.clear();
        s.early_debts.clear();
        return;
    }
    let early = std::mem::take(&mut s.early);
    let mut booked = std::mem::take(&mut s.early_debts);
    booked.sort_by_key(|(due, _, _)| -due);
    for (item, q) in early.iter() {
        let sold = (qget(&s.early_proj, item) - v.shed(item)).max(0);
        let mut short = q - (*q).min(sold);
        if short <= 0 {
            continue;
        }
        for (due, it, bq) in booked.iter_mut() {
            if short <= 0 {
                break;
            }
            if it != item || *bq <= 0 {
                continue;
            }
            let take = short.min(*bq);
            if let Some((_, m)) = s.r36_debts.iter_mut().find(|(d, _)| d == due) {
                let have = qget(m, item);
                let t = take.min(have);
                qadd(m, item, -t);
                short -= t;
                *bq -= t;
            }
        }
        if short > 0 && s.due_step == step {
            let have = qget(&s.suppress, item);
            qadd(&mut s.suppress, item, -short.min(have));
        }
    }
}
