//! `Chassis` (agent lines 404-936): replays the routed tape through the reactive layers.
//! Statement order, dict iteration order and tie-breaks follow the Python line by line; the
//! Python method a block ports is named in its comment.
use crate::act::{Action, Cmd};
use crate::router::{RouterState, ShopRouter};
use crate::view::*;
use crate::obs::{qadd, qget, qset, Obs, Qty};
use crate::managers::market_guard::{apply_suppression, settle_r36_debts};
use kagg_engine::json::Json;

#[derive(Clone, Debug)]
pub struct Settings {
    pub hand_align: bool,
    pub weed_repair: bool,
    pub sell_lead: bool,
    pub front_run: bool,
    pub budget_guard: bool,
    pub room_guard: bool,
    pub clamp_sells: bool,
    pub dead_stock: bool,
    pub terminal_liquidation: bool,
    pub block_turns: i64,
    pub shed_capacity: i64,
    pub board_size: i64,
    pub max_orders: usize,
    pub turns_per_day: i64,
    pub min_sell_price: i64,
    /// R36 class patch (agent line 1670): native lead only at step < r36_lead_from or >= r36_lead_to;
    /// suppression also settles the `r36_debts` due this step.
    pub r36: bool,
    pub r36_lead_from: i64,
    pub r36_lead_to: i64,
    /// RACEPX class patch (line 3617): while any product quotes <= base + margin, lead-sell only
    /// the non-glutted products (every step, WHEAT/FERTILIZER included).
    pub racepx: bool,
    pub racepx_margin: i64,
    /// sell_lead: lead-sell at most every `lead_every` steps (steps with step % lead_every == 0 are skipped).
    pub lead_every: i64,
    /// dead_stock: from this day every held unit is surplus; sell only while price > dead_stock_min_price.
    pub dead_stock_day: i64,
    pub dead_stock_min_price: i64,
    /// room_guard: check every `room_every` steps (at step % room_every == room_every - 1), keep `room_margin`
    /// free shed slots.
    pub room_every: i64,
    pub room_margin: i64,
    /// terminal_liquidation: dump the projected shed from this step.
    pub terminal_from: i64,
    /// market guard (30 Sep): early sales (sell_lead suppressions, R36 debts) are booked only for the units that
    /// actually FILLED -- measured next step as projected shed minus observed shed -- so an early sale that was
    /// clamped, truncated or cut never cancels the tape's own sale (it blocked 138 tape sales in one f898 game).
    pub verify_fills: bool,
    /// lead_signal: gate the lead sells / R36 reservations on a rival race signal (see Chassis::race_signal): the rival
    /// sold the item at the coming hours on >= lead_signal_p of the days so far (look lead_signal_look steps ahead) and
    /// holds >= lead_signal_stock units of it.
    pub lead_signal: bool,
    pub lead_signal_look: i64,
    pub lead_signal_p: f64,
    pub lead_signal_stock: i64,
    /// lead_price (30 Sep, operator: "sale timing based on price"): a lead sell / R36 reservation of an item is allowed
    /// only when the price calculator says selling now pays >= (1 - lead_price_tol) x the price expected at the
    /// planned step (lead_price_h ahead; market moved by the town's drain and the rival's stock arriving over
    /// lead_price_rival_h steps) -- OR the rival is racing it (lead_signal forecast). Uses Chassis::race_signal.
    pub lead_price: bool,
    pub lead_price_h: i64,
    pub lead_price_rival_h: i64,
    pub lead_price_tol: f64,
    /// animal_guard (1 Oct, loss 115943662: 4 cows stranded in the shed for 17 days after a cash crunch): drop a
    /// BUY_ANIMAL of a kind while >= animal_guard_max animals of that kind already sit unplaced in our shed.
    pub animal_guard: bool,
    pub animal_guard_max: i64,
    /// steps an animal kind must have sat in the shed continuously before it counts as stranded (a bought animal
    /// normally waits one step for a hand to place it)
    pub animal_guard_steps: i64,
    /// land_repair (1 Oct, losses 115943662 / 115949853: the tape's BUY_LAND missed after a cash dip, every later build
    /// on the locked quadrant failed, the animals bought for it were stranded): a PLANT / BUILD on a LOCKED tile is
    /// queued for replay (weed_repair's queue) and the next quadrant is bought this step when affordable.
    pub land_repair: bool,
}

impl Default for Settings {
    /// `DEFAULT_SETTINGS`, except the MARKET guards (sell_lead, front_run, dead_stock, terminal_liquidation; r36 and
    /// racepx were already off) default OFF (operator order 2026-09-30): they are not part of the chassis. A chassis'
    /// router.json carries only the repair guards; a full agent that wants a market guard must switch it on
    /// explicitly (layer config), so a key missing from a chassis can never silently turn one on.
    fn default() -> Self {
        Settings {
            hand_align: true, weed_repair: true, sell_lead: false, front_run: false, budget_guard: true,
            room_guard: true, clamp_sells: true, dead_stock: false, terminal_liquidation: false,
            block_turns: 72, shed_capacity: 100, board_size: 10, max_orders: 10, turns_per_day: 24,
            min_sell_price: 2,
            r36: false, r36_lead_from: 288, r36_lead_to: 696, racepx: false, racepx_margin: 0,
            lead_every: 4, dead_stock_day: 29, dead_stock_min_price: 1, room_every: 24, room_margin: 1, terminal_from: LAST_ACT_STEP,
            verify_fills: false,
            lead_signal: false,
            lead_signal_look: 3,
            lead_signal_p: 0.3,
            lead_signal_stock: 2,
            lead_price: false,
            lead_price_h: 2,
            lead_price_rival_h: 12,
            lead_price_tol: 0.0,
            animal_guard: false,
            animal_guard_max: 1,
            animal_guard_steps: 24,
            land_repair: false,
        }
    }
}

impl Settings {
    /// DEFAULT_SETTINGS updated with the keys present in `j`.
    pub fn from_json(j: &Json) -> Settings {
        let mut s = Settings::default();
        s.apply(j);
        s
    }

    /// Overwrite the keys present in `j` (a guard-grid cell on top of a router's settings).
    pub fn apply(&mut self, j: &Json) {
        let s = self;
        for (k, v) in j.obj() {
            match k.as_str() {
                "hand_align" => s.hand_align = v.bool(),
                "weed_repair" => s.weed_repair = v.bool(),
                "sell_lead" => s.sell_lead = v.bool(),
                "front_run" => s.front_run = v.bool(),
                "budget_guard" => s.budget_guard = v.bool(),
                "room_guard" => s.room_guard = v.bool(),
                "clamp_sells" => s.clamp_sells = v.bool(),
                "dead_stock" => s.dead_stock = v.bool(),
                "terminal_liquidation" => s.terminal_liquidation = v.bool(),
                "block_turns" => s.block_turns = v.i64(),
                "shed_capacity" => s.shed_capacity = v.i64(),
                "board_size" => s.board_size = v.i64(),
                "max_orders" => s.max_orders = v.i64().max(0) as usize,
                "turns_per_day" => s.turns_per_day = v.i64(),
                "min_sell_price" => s.min_sell_price = v.i64(),
                "r36" => s.r36 = v.bool(),
                "r36_lead_from" => s.r36_lead_from = v.i64(),
                "r36_lead_to" => s.r36_lead_to = v.i64(),
                "racepx" => s.racepx = v.bool(),
                "racepx_margin" => s.racepx_margin = v.i64(),
                "lead_every" => s.lead_every = v.i64().max(1),
                "dead_stock_day" => s.dead_stock_day = v.i64(),
                "dead_stock_min_price" => s.dead_stock_min_price = v.i64(),
                "room_every" => s.room_every = v.i64().max(1),
                "room_margin" => s.room_margin = v.i64(),
                "terminal_from" => s.terminal_from = v.i64(),
                "verify_fills" => s.verify_fills = v.bool(),
                "lead_signal" => s.lead_signal = v.bool(),
                "lead_signal_look" => s.lead_signal_look = v.i64().max(1),
                "lead_signal_p" => s.lead_signal_p = v.f64(),
                "lead_signal_stock" => s.lead_signal_stock = v.i64(),
                "lead_price" => s.lead_price = v.bool(),
                "lead_price_h" => s.lead_price_h = v.i64().max(1),
                "lead_price_rival_h" => s.lead_price_rival_h = v.i64().max(1),
                "lead_price_tol" => s.lead_price_tol = v.f64(),
                "animal_guard" => s.animal_guard = v.bool(),
                "animal_guard_max" => s.animal_guard_max = v.i64(),
                "animal_guard_steps" => s.animal_guard_steps = v.i64(),
                "land_repair" => s.land_repair = v.bool(),
                _ => {}
            }
        }
    }
}

/// `sell_state`: what sell_lead / front_run sold a step early, and the R36 debt ledger.
#[derive(Clone, Debug)]
pub struct SellState {
    pub due_step: i64,
    pub suppress: Qty,
    /// `{step: {item: qty}}` (written by the R36 layers).
    pub r36_debts: Vec<(i64, Qty)>,
    /// verify_fills: this step's early sales (item -> units), the R36 debts they booked (due, item, units) and the
    /// projected shed they were sized on; checked against the observed shed on the next step.
    pub early_step: i64,
    pub early: Qty,
    pub early_debts: Vec<(i64, &'static str, i64)>,
    pub early_proj: Qty,
}
impl Default for SellState {
    fn default() -> Self {
        SellState { due_step: -1, suppress: vec![], r36_debts: vec![], early_step: -9, early: vec![], early_debts: vec![], early_proj: vec![] }
    }
}

/// Weed-repair replay queue of one unit: `[(pos, act), ...]`.
pub type Queue = Vec<((i64, i64), Cmd)>;

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub last_step: i64,
    pub route: Option<i64>,
    pub router: RouterState,
    /// `pending`: unit index -> queue (dict insertion order).
    pub pending: Vec<(usize, Queue)>,
    pub sell: SellState,
    /// animal_guard: per animal kind, the step since which >= 1 of it has sat in our shed continuously
    pub animal_since: Vec<(&'static str, i64)>,
}
impl PlayerState {
    fn new() -> Self {
        PlayerState { last_step: -1, route: None, router: RouterState::default(), pending: vec![], sell: SellState::default(), animal_since: vec![] }
    }
}

#[derive(Clone, Debug)]
pub struct Diagnostics {
    pub layer_fallbacks: u64,
    pub entry_fallbacks: u64,
    pub terminal_rescue_errors: u64,
    /// Chassis debug log (set `log` to record): per guard (GUARDS order) the steps it changed the action,
    /// and the first such step (-1 = never).
    pub log: bool,
    pub guard_hits: [u32; GUARDS.len()],
    pub guard_first: [i64; GUARDS.len()],
}

impl Default for Diagnostics {
    fn default() -> Self {
        Diagnostics { layer_fallbacks: 0, entry_fallbacks: 0, terminal_rescue_errors: 0, log: false, guard_hits: [0; GUARDS.len()], guard_first: [-1; GUARDS.len()] }
    }
}

/// Guard names in pipeline order (index into Diagnostics::guard_hits).
pub const GUARDS: [&str; 9] = ["hand_align", "weed_repair", "suppress", "sell_lead", "front_run", "budget_guard", "room_guard", "clamp_sells", "dead_stock"];

impl Diagnostics {
    fn mark(&mut self, g: usize, before: &Action, after: &Action, step: i64) {
        if before != after {
            self.guard_hits[g] += 1;
            if self.guard_first[g] < 0 {
                self.guard_first[g] = step;
            }
        }
    }
    /// "name:hits@first" for every guard that fired.
    pub fn summary(&self) -> String {
        let v: Vec<String> = GUARDS.iter().enumerate().filter(|(i, _)| self.guard_hits[*i] > 0).map(|(i, g)| format!("{g}:{}@{}", self.guard_hits[i], self.guard_first[i])).collect();
        if v.is_empty() { "-".into() } else { v.join(",") }
    }
}

/// One route: its tape and the per-product suffix sums of planned SELL quantities.
#[derive(Clone)]
pub struct Route {
    pub id: i64,
    pub tape: Vec<Action>,
    /// `future[p][t]` = planned SELL of PRODUCTS[p] in steps >= t.
    pub future: Vec<Vec<i64>>,
}

impl Route {
    pub fn new(id: i64, tape: Vec<Action>) -> Route {
        let mut r = Route { id, tape, future: vec![] };
        r.refresh();
        r
    }
    /// Recompute the suffix sums (call after mutating the tape, e.g. PIPE's EarlyCycle).
    pub fn refresh(&mut self) {
        let n = self.tape.len();
        let mut f = vec![vec![0i64; n + 1]; PRODUCTS.len()];
        for t in (0..n).rev() {
            for p in 0..PRODUCTS.len() {
                f[p][t] = f[p][t + 1];
            }
            for o in &self.tape[t].market {
                if o.is_sell3() {
                    if let Some(p) = PRODUCTS.iter().position(|x| *x == o.s(1)) {
                        f[p][t] += o.n(2).max(0);
                    }
                }
            }
        }
        self.future = f;
    }
    pub fn future_sells(&self, item: &str, step: i64) -> i64 {
        let Some(p) = PRODUCTS.iter().position(|x| *x == item) else { return 0 };
        let col = &self.future[p];
        if step >= 0 && (step as usize) < col.len() {
            col[step as usize]
        } else {
            0
        }
    }
    pub fn at(&self, step: i64) -> Option<&Action> {
        if step >= 0 {
            self.tape.get(step as usize)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Chassis {
    /// Routes in the Python dict's insertion order (the first is `next(iter(routes))`).
    pub routes: Vec<Route>,
    pub router: ShopRouter,
    pub cfg: Settings,
    pub opponent_plan: Option<Vec<Action>>,
    /// Per-player state keyed by `observation["player"]`.
    pub players: Vec<(i64, PlayerState)>,
    pub diagnostics: Diagnostics,
    /// lead_signal (30 Sep): the items the rival is forecast to sell soon, set by Base every turn when the market
    /// guard's `lead_signal` is on. Some(list) = sell_lead / RACEPX / R36 may pull a sale forward ONLY for these items
    /// (no race signal -> the tape's own sale stands); None = ungated (v61.1).
    pub race_signal: Option<Vec<&'static str>>,
}

impl Chassis {
    pub fn route_idx(&self, id: i64) -> Option<usize> {
        self.routes.iter().position(|r| r.id == id)
    }
    pub fn route(&self, id: i64) -> &Route {
        &self.routes[self.route_idx(id).unwrap_or(0)]
    }
    pub fn player(&mut self, p: i64) -> Option<&mut PlayerState> {
        self.players.iter_mut().find(|(k, _)| *k == p).map(|(_, s)| s)
    }

    /// `_state`: a new game resets the player's state.
    fn state_idx(&mut self, player: i64, step: i64) -> usize {
        let i = match self.players.iter().position(|(k, _)| *k == player) {
            Some(i) => i,
            None => {
                self.players.push((player, PlayerState::new()));
                self.players.len() - 1
            }
        };
        let st = &mut self.players[i].1;
        if step == 0 || step <= st.last_step {
            *st = PlayerState::new();
        }
        st.last_step = step;
        i
    }

    /// `_route_action`.
    fn route_action(&self, route: i64, step: i64) -> Action {
        self.route(route).at(step).cloned().unwrap_or_else(Action::pass)
    }

    /// `Chassis.act`. `Err(())` = Python raised before the layer try-block (the factory
    /// then falls back to the tape).
    pub fn act(&mut self, v: &View) -> Action {
        let step = v.step;
        let player = v.obs.player;
        let pi = self.state_idx(player, step);
        let mut rs = std::mem::take(&mut self.players[pi].1.router);
        let mut route = self.router.route(v, step, &mut rs);
        let st_route = self.players[pi].1.route;
        self.players[pi].1.router = rs;
        if self.route_idx(route).is_none() {
            route = match st_route {
                Some(r) if self.route_idx(r).is_some() => r,
                _ => self.routes[0].id,
            };
        }
        self.players[pi].1.route = Some(route);
        let mut action = self.route_action(route, step);
        let raw = action.clone();
        match self.layers(&mut action, v, pi, route, step) {
            Ok(()) => {
                action.market.truncate(self.cfg.max_orders);
                action
            }
            Err(()) => {
                self.diagnostics.layer_fallbacks += 1;
                raw
            }
        }
    }

    fn layers(&mut self, action: &mut Action, v: &View, pi: usize, route: i64, step: i64) -> Result<(), ()> {
        let cfg = self.cfg.clone();
        let log = self.diagnostics.log;
        let mut prev = if log { action.clone() } else { Action::default() };
        macro_rules! mark {
            ($g:expr) => {
                if log {
                    self.diagnostics.mark($g, &prev, action, step);
                    prev = action.clone();
                }
            };
        }
        if cfg.hand_align {
            hand_align(action, v);
            mark!(0);
        }
        if cfg.weed_repair {
            let mut pending = std::mem::take(&mut self.players[pi].1.pending);
            self.weed_repair(action, v, &mut pending, route, step);
            self.players[pi].1.pending = pending;
            mark!(1);
        }
        if cfg.land_repair {
            let mut pending = std::mem::take(&mut self.players[pi].1.pending);
            let mut units = action.units();
            let n = units.len().min(v.positions.len());
            let mut want_land = false;
            for i in 0..n {
                let pos = v.positions[i];
                if matches!(units[i].op(), "PLANT" | "BUILD_COOP" | "BUILD_PASTURE") && v.tile(pos).is_locked() {
                    match pending.iter().position(|(k, _)| *k == i) {
                        Some(q) => pending[q].1.push((pos, units[i].clone())),
                        None => pending.push((i, vec![(pos, units[i].clone())])),
                    }
                    units[i] = Cmd::pass();
                    want_land = true;
                }
            }
            action.set_units(units);
            let q = v.quadrants();
            let buying = action.market.iter().any(|o| o.op() == "BUY_LAND");
            if want_land && !buying && q >= 1 && ((q - 1) as usize) < LAND_PRICES.len() && v.money() >= LAND_PRICES[(q - 1) as usize] as f64 && action.market.len() < cfg.max_orders {
                action.market.push(Cmd::new("BUY_LAND"));
            }
            self.players[pi].1.pending = pending;
        }
        if cfg.verify_fills {
            crate::managers::market_guard::verify_fills(&mut self.players[pi].1.sell, v, step);
        }
        if cfg.sell_lead || cfg.front_run {
            apply_suppression(action, &self.players[pi].1.sell, step);
            if cfg.r36 {
                settle_r36_debts(action, &mut self.players[pi].1.sell, step);
            }
            mark!(2);
        }
        let projected = self.projected_shed(action, v);
        let mut lead_available = projected.clone();
        let mut next_sup = SellState {
            due_step: -1,
            suppress: vec![],
            r36_debts: std::mem::take(&mut self.players[pi].1.sell.r36_debts),
            ..Default::default()
        };
        if cfg.sell_lead {
            self.sell_lead(action, v, &mut lead_available, route, step, &mut next_sup);
            mark!(3);
        }
        if cfg.front_run && self.opponent_plan.as_ref().is_some_and(|p| !p.is_empty()) {
            self.front_run(action, v, &mut lead_available, route, step, &mut next_sup);
            mark!(4);
        }
        if cfg.verify_fills {
            next_sup.early_step = step;
            next_sup.early = next_sup.suppress.clone();
            next_sup.early_proj = projected.clone();
        }
        self.players[pi].1.sell = next_sup;
        if cfg.budget_guard {
            self.budget_guard(action, v, route, step);
            mark!(5);
        }
        if cfg.animal_guard {
            // stranded animals: no new purchase of a kind that has sat in the shed for animal_guard_steps (no pasture)
            let since = &mut self.players[pi].1.animal_since;
            for kind in ["COW", "SHEEP", "GOOSE"] {
                let n = v.shed(kind);
                let i = since.iter().position(|(k, _)| *k == kind);
                match (n >= cfg.animal_guard_max, i) {
                    (true, None) => since.push((kind, step)),
                    (false, Some(j)) => {
                        since.remove(j);
                    }
                    _ => {}
                }
            }
            let stranded: Vec<&'static str> = since.iter().filter(|(_, s0)| step - s0 >= cfg.animal_guard_steps).map(|(k, _)| *k).collect();
            for o in action.market.iter_mut() {
                if o.op() == "BUY_ANIMAL" && o.len() >= 3 && stranded.contains(&o.s(1)) {
                    o.set_n(2, 0);
                }
            }
        }
        if cfg.room_guard {
            self.room_guard(action, v, route, step);
            mark!(6);
        }
        if cfg.clamp_sells {
            clamp_sells(action, &projected);
            mark!(7);
        }
        if cfg.dead_stock {
            self.dead_stock(action, v, &projected, route, step);
            mark!(8);
        }
        if cfg.terminal_liquidation {
            self.terminal_liquidation(action, &projected, step);
        }
        let _ = &prev;
        Ok(())
    }

    // ---- layer: weed_repair ----------------------------------------------------------
    fn weed_repair(&self, action: &mut Action, v: &View, pending: &mut Vec<(usize, Queue)>, route: i64, step: i64) {
        let mut units = action.units();
        let r = self.route(route);
        let nxt = r.at(step + 1);
        let next_units: Vec<Cmd> = nxt.map(|a| a.units()).unwrap_or_else(|| vec![Cmd::pass()]);
        let n = units.len().min(v.positions.len());
        for i in 0..n {
            let pos = v.positions[i];
            let tile = v.tile(pos);
            let mut act = units[i].clone();
            let qi = pending.iter().position(|(k, _)| *k == i);
            let mut has_queue = qi.is_some_and(|q| !pending[q].1.is_empty());
            if has_queue && pending[qi.unwrap()].1[0].0 != pos {
                pending.remove(qi.unwrap());
                has_queue = false;
            }
            let is_weed = tile.is_dict() && tile.kind == "WEED";
            let noop = is_noop(&act, tile, v.inv(i), v, pos);
            let next_op = next_units.get(i).filter(|c| !c.is_empty()).map(|c| c.op()).unwrap_or("PASS");
            let op = act.op();
            if matches!(op, "PLANT" | "BUILD_COOP" | "BUILD_PASTURE") && is_weed {
                match pending.iter().position(|(k, _)| *k == i) {
                    Some(q) => pending[q].1.push((pos, act.clone())),
                    None => pending.push((i, vec![(pos, act.clone())])),
                }
                act = Cmd::new("DIG");
            } else if has_queue && noop {
                let q = pending.iter().position(|(k, _)| *k == i).unwrap();
                let replay = pending[q].1[0].1.clone();
                if replay.op() == "PLANT" && is_move(next_op) {
                    pending.remove(q); // WATER could never follow: keep the seed
                } else {
                    pending[q].1.remove(0);
                    if !act.is_empty() && act.op() != "PASS" && !is_move(act.op()) {
                        pending[q].1.push((pos, act.clone()));
                    }
                    act = replay;
                    if pending[q].1.is_empty() {
                        pending.remove(q);
                    }
                }
            } else if is_weed && noop {
                act = Cmd::new("DIG");
            }
            units[i] = act;
        }
        action.set_units(units);
    }

    // ---- projected shed --------------------------------------------------------------
    /// `_projected_shed`: the shed after this step's unit actions, before the market.
    pub fn projected_shed(&self, action: &Action, v: &View) -> Qty {
        let cap = self.cfg.shed_capacity;
        let mut proj: Qty = PRODUCTS.iter().map(|p| (*p, v.shed(p))).collect();
        for (k, val) in &v.shed {
            if !proj.iter().any(|(n, _)| n == k) {
                proj.push((k, *val));
            }
        }
        let mut total: i64 = proj.iter().map(|(_, n)| n).sum();
        let n = action.n_units().min(v.positions.len());
        for i in 0..n {
            if !shed_adjacent(v.positions[i], v.board) {
                continue;
            }
            let act = action.unit(i).unwrap();
            let op = if act.is_empty() { "PASS" } else { act.op() };
            let inv = v.inv(i);
            if op == "PICKUP" && act.len() >= 2 && proj.iter().any(|(k, _)| *k == act.s(1)) {
                let item = act.s(1);
                let want = if act.len() >= 3 { act.n(2).max(0) } else { 1 };
                let qty = qget(&proj, item).min(want);
                qadd(&mut proj, item, -qty);
                total -= qty;
            } else if op == "DROP" {
                for (item, held) in inv {
                    let take = (*held).max(0).min((cap - total).max(0));
                    if take > 0 {
                        qadd(&mut proj, item, take);
                        total += take;
                    }
                }
            } else if op == "PLACE" && act.len() >= 2 && animal_structure(act.s(1)).is_none() {
                let item = act.s(1);
                let want = if act.len() >= 3 { act.n(2).max(0) } else { 1 };
                let take = want.min(qget(inv, item).max(0)).min((cap - total).max(0));
                if take > 0 {
                    qadd(&mut proj, item, take);
                    total += take;
                }
            }
        }
        proj
    }

    // ---- layer: budget_guard -----------------------------------------------------------
    /// `_block_requirements`: planned purchase cost and item reserves for tape steps [start, end).
    fn block_requirements(&self, v: &View, route: i64, start: i64, end: i64) -> (f64, Qty) {
        let tape = &self.route(route).tape;
        let mut budget = 0.0f64;
        let (mut seed_bal, mut item_bal, mut seed_need, mut item_need): (Qty, Qty, Qty, Qty) = (vec![], vec![], vec![], vec![]);
        let mut hires_by_day: Vec<(i64, i64)> = vec![];
        let mut quadrants = v.quadrants();
        let need = |bal: &mut Qty, need: &mut Qty, k: &'static str, d: i64| {
            qadd(bal, k, d);
            let b = qget(bal, k);
            let cur = qget(need, k);
            qset(need, k, cur.max(-b));
        };
        let hi = end.min(tape.len() as i64);
        let mut t = start;
        while t < hi {
            let a = &tape[t as usize];
            for u in a.units() {
                if u.is_empty() {
                    continue;
                }
                let op = u.op();
                let arg = u.s(1);
                let qty = (if u.len() > 2 { u.n(2) } else { 1 }).max(1);
                if op == "PLANT" && seed_price(arg).is_some() {
                    need(&mut seed_bal, &mut seed_need, arg, -1);
                } else if op == "FEED" {
                    need(&mut item_bal, &mut item_need, "WHEAT", -1);
                } else if op == "FERTILIZE" {
                    need(&mut item_bal, &mut item_need, "FERTILIZER", -1);
                } else if op == "PLACE" && u.len() > 1 {
                    need(&mut item_bal, &mut item_need, arg, -qty);
                }
            }
            for o in &a.market {
                if o.is_empty() {
                    continue;
                }
                let op = o.op();
                let item = o.s(1);
                let qty = (if o.len() > 2 { o.n(2) } else { 1 }).max(1);
                if op == "HIRE" {
                    let day = (t - start).div_euclid(self.cfg.turns_per_day);
                    match hires_by_day.iter_mut().find(|(d, _)| *d == day) {
                        Some(e) => e.1 += 1,
                        None => hires_by_day.push((day, 1)),
                    }
                } else if op == "BUY_LAND" {
                    let extra = quadrants - 1;
                    if extra >= 0 && (extra as usize) < LAND_PRICES.len() {
                        budget += LAND_PRICES[extra as usize] as f64;
                        quadrants += 1;
                    }
                } else if op == "BUY_SEED" && seed_price(item).is_some() {
                    budget += (seed_price(item).unwrap() * qty) as f64;
                    qadd(&mut seed_bal, item, qty);
                } else if op == "BUY_PRODUCT" && (item == "WHEAT" || item == "FERTILIZER") {
                    budget += (v.price(item) * qty) as f64;
                    qadd(&mut item_bal, item, qty);
                } else if op == "BUY_ANIMAL" && animal_cost(item).is_some() {
                    budget += (animal_cost(item).unwrap() * qty) as f64;
                    qadd(&mut item_bal, item, qty);
                }
            }
            t += 1;
        }
        for (day, n) in hires_by_day {
            let first = if day == 0 { v.hires_today() } else { 0 };
            for k in 0..n {
                budget += fib(first + k) as f64;
            }
        }
        (budget, item_need)
    }

    fn budget_guard(&self, action: &mut Action, v: &View, route: i64, step: i64) {
        let cfg = &self.cfg;
        let block = cfg.block_turns;
        if block <= 0 || step % block != 0 {
            return;
        }
        let (budget, item_need) = self.block_requirements(v, route, step, step + block);
        let mut existing: Qty = vec![];
        for o in &action.market {
            if o.is_sell3() {
                qadd(&mut existing, o.s(1), o.n(2).max(0));
            }
        }
        let r = self.route(route);
        let mut cash = v.money();
        for item in PRODUCTS {
            let planned = qget(&existing, item).max(r.future_sells(item, step) - r.future_sells(item, step + block));
            cash += (v.shed(item).min(planned) * v.price(item)) as f64;
        }
        let mut shortfall = budget - cash;
        if shortfall <= 0.0 {
            return;
        }
        let mut cands: Vec<(i64, &'static str, i64, i64)> = vec![];
        for item in PRODUCTS {
            let price = v.price(item);
            if price < cfg.min_sell_price {
                continue;
            }
            let protected = (qget(&item_need, item) - v.in_hands(item)).max(0);
            let avail = v.shed(item) - protected - qget(&existing, item);
            if avail > 0 {
                cands.push((-price, item, avail, price));
            }
        }
        cands.sort();
        let mut added = false;
        for (_, item, avail, price) in cands {
            if shortfall <= 0.0 {
                break;
            }
            let s = shortfall.trunc() as i64;
            let qty = avail.min(-((-s).div_euclid(price)));
            if add_sell(action, item, qty, cfg.max_orders, true) {
                shortfall -= (qty * price) as f64;
                added = true;
            }
        }
        if added {
            let (sells, others): (Vec<Cmd>, Vec<Cmd>) = action.market.drain(..).partition(|o| o.op() == "SELL");
            action.market = sells.into_iter().chain(others).collect();
        }
    }

    // ---- layer: room_guard ---------------------------------------------------------------
    fn room_guard(&self, action: &mut Action, v: &View, route: i64, step: i64) {
        let cfg = &self.cfg;
        if step % cfg.room_every != cfg.room_every - 1 {
            return;
        }
        let cap = cfg.shed_capacity;
        let carried: i64 = v.invs().iter().flat_map(|m| m.iter().map(|(_, n)| (*n).max(0))).sum();
        let (mut produced, mut consumed) = (0i64, 0i64);
        let n = action.n_units().min(v.positions.len());
        for i in 0..n {
            let tile = v.tile(v.positions[i]);
            let a = action.unit(i).unwrap();
            if a.is_empty() {
                continue;
            }
            match a.op() {
                "HARVEST" if tile.is_dict() => produced += tile.yield_units.max(0),
                "COLLECT_FERTILIZER" if tile.is_dict() && tile.fertilizer_available => produced += 1,
                "FEED" | "FERTILIZE" => consumed += 1,
                "PLACE" if a.len() > 1 && animal_structure(a.s(1)).is_some() => consumed += 1,
                _ => {}
            }
        }
        let mut planned_sells: Qty = vec![];
        let mut planned_buys = 0i64;
        for o in &action.market {
            if o.is_empty() {
                continue;
            }
            if o.is_sell3() {
                qadd(&mut planned_sells, o.s(1), o.n(2).max(0));
            } else if (o.op() == "BUY_PRODUCT" || o.op() == "BUY_ANIMAL") && o.len() >= 3 {
                planned_buys += o.n(2).max(0);
            }
        }
        let fillable: i64 = planned_sells.iter().map(|(it, n)| v.shed(it).min(*n)).sum();
        let mut needed = v.shed_total() + carried + produced - consumed + planned_buys - fillable - (cap - cfg.room_margin);
        if needed <= 0 {
            return;
        }
        let r = self.route(route);
        let mut priority: Vec<&'static str> = PRODUCTS.to_vec();
        priority.sort_by_key(|it| (r.future_sells(it, step + 1) > 0, -v.price(it), *it));
        for item in priority {
            let avail = (v.shed(item) - qget(&planned_sells, item)).max(0);
            let qty = needed.min(avail);
            if qty <= 0 || v.price(item) < 1 {
                continue;
            }
            if !add_sell(action, item, qty, cfg.max_orders, true) {
                continue;
            }
            qadd(&mut planned_sells, item, qty);
            needed -= qty;
            if needed <= 0 {
                break;
            }
        }
    }

    /// `make_agent`'s closure: never fails. A missing/incomplete observation falls back to
    /// the player's tape action, else PASS with the hands padded.
    pub fn entry(&mut self, obs: Obs) -> (Action, Option<View>) {
        let obs = match View::new(obs) {
            Ok(v) => {
                let a = self.act(&v);
                return (a, Some(v));
            }
            Err(o) => o,
        };
        self.diagnostics.entry_fallbacks += 1;
        let step = obs.step();
        let player = obs.player;
        let rid = self
            .players
            .iter()
            .find(|(k, _)| *k == player)
            .and_then(|(_, s)| s.route)
            .filter(|r| self.route_idx(*r).is_some())
            .unwrap_or(self.routes[0].id);
        if let Some(a) = self.route(rid).at(step) {
            return (a.clone(), None);
        }
        let p = player.max(0) as usize;
        if p < obs.farms.len() {
            let n = obs.farms[p].hands.len();
            return (Action { farmer: Cmd::pass(), hands: vec![Cmd::pass(); n], market: vec![] }, None);
        }
        (Action::pass(), None)
    }
}

// ---- layer: hand_align ---------------------------------------------------------------------
fn hand_align(action: &mut Action, v: &View) {
    let expected = v.positions.len().saturating_sub(1);
    while action.hands.len() < expected {
        action.hands.push(Cmd::pass());
    }
    action.hands.truncate(expected);
}

/// The engine's base quote per product (market PARAMS `base`).
pub fn base_price(item: &str) -> i64 {
    match item {
        "WHEAT" => 25,
        "CARROT" => 35,
        "TOMATO" => 60,
        "STRAWBERRY" => 120,
        "MELON" => 250,
        "EGG" => 50,
        "MILK" => 160,
        "WOOL" => 200,
        "FERTILIZER" => 100,
        _ => 0,
    }
}

/// `_add_sell`.
pub fn add_sell(action: &mut Action, item: &'static str, qty: i64, max_orders: usize, merge: bool) -> bool {
    if merge {
        for o in action.market.iter_mut() {
            if o.op() == "SELL" && o.s(1) == item {
                let nv = o.n(2) + qty;
                if o.len() >= 3 {
                    o.set_n(2, nv);
                } else {
                    return false; // Python: IndexError on order[2] -> layer fallback (never in route data)
                }
                return true;
            }
        }
    }
    if action.market.len() >= max_orders {
        return false;
    }
    action.market.push(Cmd::order("SELL", item, qty));
    true
}

/// `_clamp_sells`: clamp SELLs to a sequential stock bound, keeping every market slot.
fn clamp_sells(action: &mut Action, projected: &Qty) {
    let mut avail = projected.clone();
    for o in action.market.iter_mut() {
        if o.is_sell3() {
            let item = o.s(1);
            let have = qget(&avail, item);
            let n = o.n(2).min(have).max(0);
            qset(&mut avail, item, have - n);
            *o = Cmd::order("SELL", item, n);
        } else if (o.op() == "BUY_PRODUCT" || o.op() == "BUY_ANIMAL") && o.len() >= 3 {
            let it = o.s(1);
            qadd(&mut avail, it, o.n(2).max(0));
        }
    }
}
