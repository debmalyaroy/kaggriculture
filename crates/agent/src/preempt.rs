//! Pre-emption (operator 28 Sep): the rival fingerprints our lineage and sells a small batch just before our
//! known sale, so our bulk sale lands on a depressed price. We fingerprint THEM the same way and sell first.
//!
//! Track: every step, the rival's sales per product are read off the public market inventory (rise net of the
//! town's drain, on products we did not sell that step) and binned by hour of day. Route bots repeat their daily
//! schedule, so after a few days `seen(item, hour) / days` is a good forecast of "they sell X at hour h".
//! Act: when the rival is forecast to sell X within the next `look` steps (+ a per-game jitter of 0..=`jitter`
//! extra steps) and we hold X with our own sale of X planned within `horizon` steps (route tape), we sell that
//! planned amount NOW, ahead of them, above a price floor. The final liquidation (from `to`) is left alone.
//! Runs last in Base::act (after the chain, reactive shell and sales shell), so no later layer undoes it.
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::obs::{qadd, qget, Qty};
use crate::view::View;

pub const ITEMS: [&str; 8] = ["WHEAT", "STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"];

#[derive(Clone, Debug)]
pub struct PreCfg {
    pub from: i64,
    pub to: i64,
    /// forecast window: rival sale expected within this many steps
    pub look: i64,
    /// extra 0..=jitter steps per product, drawn once per game from the opening state
    pub jitter: i64,
    /// our planned sales within this many steps are pulled forward
    pub horizon: i64,
    /// forecast probability to act on (share of observed days the rival sold X at that hour)
    pub p_min: f64,
    /// days of observation required before acting
    pub min_days: i64,
    /// price floor as a fraction of base
    pub floor: f64,
    pub items: Vec<&'static str>,
    /// forecast-driven pre-emption on/off (the rival-sale fingerprint)
    pub threat: bool,
    /// random lead: each planned route sale is released 0..=lead_max steps early (a fresh draw per sale); 0 = off
    pub lead_max: i64,
    /// price keeping: sell early only the units whose price is >= (1 - tol) x the price projected at the scheduled step
    pub tol: f64,
    /// offline opponent clusters (python/opp_cluster2.py): per cluster, P(rival sells item at step t)
    pub clusters: Option<std::sync::Arc<Vec<Vec<Vec<f32>>>>>,
    /// per cluster, per item: median lot (units per sale order)
    pub lots: Option<std::sync::Arc<Vec<Vec<f32>>>>,
    /// per cluster: true = the BACKGROUND entry (unpredictable / RL): when it is the most likely, no pre-emption
    pub background: Vec<bool>,
    /// "sell the demand" (operator 28 Sep): when pre-empting, sell only the units that bring the price down to
    /// demand_floor x base -- the premium the town's drain created -- so the rival's sale lands at base/glut.
    /// 0 = off (pre-empt the whole planned amount).
    pub demand_floor: f64,
    /// wheat trade (the one good both sides can buy back): buy when the price is `arb_margin` below the price
    /// projected `arb_look` steps ahead (town drain + the rival's expected sales), sell the bought units when the
    /// price is `arb_margin` above their cost; at most `arb_max` units held, cash kept above `arb_cash`. 0 = off.
    pub arb_max: i64,
    pub arb_margin: f64,
    pub arb_look: i64,
    pub arb_cash: f64,
    /// wheat trade: minimum $ edge per unit (buy vs projected resale; resale vs cost) and the first step
    pub arb_abs: f64,
    pub arb_from: i64,
    /// lineage identification: the rival's sales are scored only from this step (D5; nothing decisive before the
    /// day-6 shop unlock)
    pub fp_from: i64,
    /// wheat surplus resale: at phase-1 steps (right after a town drain) sell shed wheat beyond the tape's pickups
    /// of the next `wsell_need_h` steps + `wsell_buffer`, while each unit's price >= the trailing 24-step mean.
    /// false = off.
    pub wsell: bool,
    pub wsell_need_h: i64,
    pub wsell_buffer: i64,
    /// glut guard (operator 28 Sep): before `guard_to`, our own SELL orders of the `guard_items` keep only the units
    /// whose price now is >= guard_ratio x the price projected `guard_look` steps ahead (town drain + the rival's
    /// expected sales); the rest wait for the tape's later sales. 0 = off. The end-stage dump is never guarded.
    pub guard_ratio: f64,
    pub guard_look: i64,
    pub guard_to: i64,
    pub guard_items: Vec<&'static str>,
    /// SELL TO DEMAND (operator 28 Sep; README "Market Mechanics"): the town consumes each product every 4 turns
    /// (after the market step), so the premium exists right after each tick. For `demand_items`, before `demand_to`:
    /// the chain's own SELLs are capped to the units priced >= demand_ratio x base (no dumping below demand), and at
    /// every tick step (step % 4 == 1) we sell the units the town just consumed (down to demand_ratio x base) --
    /// unless the shed is above `shed_press` items (held stock would be discarded at the cap). 0 = off.
    pub demand_ratio: f64,
    pub demand_to: i64,
    pub shed_press: i64,
    pub demand_items: Vec<&'static str>,
    /// per-product selling level (x base) overriding demand_ratio: hinge goods (carrot/tomato/egg) sell only into a
    /// deep drain, premium goods to base at each tick, melon (no shop demand) just under base
    pub demand_by: Vec<(&'static str, f64)>,
    /// SALE OPTIMISER (operator 28 Sep): per product an exact DP over the next `opt_h` steps, re-solved every turn
    /// (receding horizon). Price = the engine curve; our k-th unit sold meets the projected market stock (town drain
    /// from the unlocked shops + the rival's expected sales from its lineage table) + k. Units only once they are in
    /// our shed / arriving (ripe yield + carried goods, at the end of the day). Units left at the horizon are valued
    /// at the projected price there x `opt_salvage`. Replaces the chain's SELLs of `opt_items` before `opt_to`.
    pub opt: bool,
    pub opt_h: i64,
    pub opt_to: i64,
    pub opt_salvage: f64,
    pub opt_items: Vec<&'static str>,
    /// per-game noise on the projected stock path (units, uniform +-) -- near-ties resolve differently each game
    pub opt_noise: f64,
    pub opt_max_units: i64,
    /// per-step discount on future revenue (competition / uncertainty): waiting must beat the risk of waiting
    pub opt_gamma: f64,
    /// front-loading (random lead, optimiser, pre-emption) only against a MATCHED tape lineage: the non-background
    /// lineages must hold >= `gate_w` of the posterior. Unmatched (RL / adaptive / unknown): tape + sale layer only,
    /// still capped to demand. 0 = no gate.
    pub gate_w: f64,
    /// Rival-stock pre-emption (endgame): from `stock_from` to `stock_to`, when the rival's estimated stock of an item
    /// (visible harvests - recovered sales) is >= `stock_min`, sell our planned sales of it within `stock_h` steps now,
    /// ahead of the dump it must make before the end. 0 = off. `stock_any` = any rival (else lineage-matched only).
    pub stock_from: i64,
    pub stock_to: i64,
    pub stock_min: i64,
    pub stock_h: i64,
    pub stock_any: bool,
    /// Stock estimate window in steps (collected - sold over the last `stock_w` steps; 0 = whole game). The rival's
    /// deliveries to town shops are invisible, so a whole-game count overestimates; 24 tracked best (29 Sep check).
    pub stock_w: i64,
    /// Items the rival-stock rule may pre-empt (default: glut-prone only; milk / egg never glut, and the rival's milk
    /// stock is not observable -- 29 Sep RCA: selling 9 milk early on a phantom 105-unit estimate lost $700).
    pub stock_items: Vec<&'static str>,
    /// Final SELL slot order (29 Sep ladder RCA: 16 of 20 live mirror losses were same-step races lost on slot
    /// order -- the rival's premium SELL in slot 0, ours in slot 1). "value": our SELLs permuted among their own
    /// slots by quote x quantity, highest first. "" = unchanged.
    pub sell_order: String,
    /// Rival groups the stock rule acts against (0 DIFFERENT, 1 PARTIAL, 2 COPY; empty = all).
    pub stock_groups: Vec<usize>,
    /// Rival stock source: "" = this layer's own tracker (v63.11), "gt" = the shared crate::gt tracker (v63.12).
    pub stock_src: String,
    /// Mixed strategy (v63.12, operator's game-theory notes): per game, one parameter set is drawn with probability
    /// p from a hash of a secret salt and the opening state (replays show the distribution, never the next draw).
    /// Empty = the single set from the stock_* keys.
    pub stock_variants: Vec<(f64, StockSel)>,
}

impl Default for PreCfg {
    fn default() -> Self {
        PreCfg { from: 96, to: 647, look: 2, jitter: 1, horizon: 12, p_min: 0.5, min_days: 2, floor: 0.5, items: ITEMS.to_vec(), threat: true, lead_max: 0, tol: 0.05, clusters: None, lots: None, background: vec![], demand_floor: 0.0, arb_max: 0, arb_margin: 0.08, arb_look: 8, arb_cash: 1500.0, arb_abs: 1.5, arb_from: 192, fp_from: 120, wsell: false, wsell_need_h: 48, wsell_buffer: 5, guard_ratio: 0.0, guard_look: 24, guard_to: 647, guard_items: vec!["STRAWBERRY", "MILK", "WOOL", "MELON", "CARROT", "TOMATO"], demand_ratio: 0.0, demand_to: 647, shed_press: 85, demand_items: ITEMS.to_vec(), demand_by: vec![], opt: false, opt_h: 72, opt_to: 647, opt_salvage: 0.85, opt_items: vec!["STRAWBERRY", "MILK", "WOOL", "MELON", "CARROT", "TOMATO", "EGG"], opt_noise: 2.0, opt_max_units: 150, opt_gamma: 0.99, gate_w: 0.0, stock_from: 0, stock_to: 711, stock_min: 5, stock_h: 48, stock_any: true, stock_w: 0, stock_items: vec!["STRAWBERRY", "MELON", "TOMATO", "CARROT", "WOOL"], sell_order: String::new(), stock_groups: vec![], stock_variants: vec![], stock_src: String::new() }
    }
}

impl PreCfg {
    pub fn load(path: &str) -> Result<PreCfg, String> {
        Self::load_over(path, &kagg_engine::json::Json::Null)
    }

    /// `load` with the keys of `over` (an agent.json manager override) replacing the file's.
    pub fn load_over(path: &str, over: &kagg_engine::json::Json) -> Result<PreCfg, String> {
        let mut j = kagg_engine::json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        crate::managers::merge(&mut j, over);
        let mut c = PreCfg::default();
        let i = |k: &str, d: i64| if j.get(k).is_null() { d } else { j.get(k).i64() };
        let f = |k: &str, d: f64| if j.get(k).is_null() { d } else { j.get(k).f64() };
        c.from = i("from", c.from);
        c.to = i("to", c.to);
        c.look = i("look", c.look);
        c.jitter = i("jitter", c.jitter);
        c.horizon = i("horizon", c.horizon);
        c.p_min = f("p_min", c.p_min);
        c.min_days = i("min_days", c.min_days);
        c.floor = f("floor", c.floor);
        c.threat = if j.get("threat").is_null() { c.threat } else { j.get("threat").bool() };
        c.lead_max = i("lead_max", c.lead_max);
        c.tol = f("tol", c.tol);
        c.demand_floor = f("demand_floor", c.demand_floor);
        c.arb_max = i("arb_max", c.arb_max);
        c.arb_margin = f("arb_margin", c.arb_margin);
        c.arb_look = i("arb_look", c.arb_look);
        c.arb_cash = f("arb_cash", c.arb_cash);
        c.arb_abs = f("arb_abs", c.arb_abs);
        c.arb_from = i("arb_from", c.arb_from);
        c.fp_from = i("fp_from", c.fp_from);
        c.wsell = if j.get("wsell").is_null() { c.wsell } else { j.get("wsell").bool() };
        c.wsell_need_h = i("wsell_need_h", c.wsell_need_h);
        c.wsell_buffer = i("wsell_buffer", c.wsell_buffer);
        c.guard_ratio = f("guard_ratio", c.guard_ratio);
        c.demand_ratio = f("demand_ratio", c.demand_ratio);
        c.demand_to = i("demand_to", c.demand_to);
        c.shed_press = i("shed_press", c.shed_press);
        c.opt = if j.get("opt").is_null() { c.opt } else { j.get("opt").bool() };
        c.opt_h = i("opt_h", c.opt_h);
        c.opt_to = i("opt_to", c.opt_to);
        c.opt_salvage = f("opt_salvage", c.opt_salvage);
        c.opt_noise = f("opt_noise", c.opt_noise);
        c.opt_max_units = i("opt_max_units", c.opt_max_units);
        c.opt_gamma = f("opt_gamma", c.opt_gamma);
        c.gate_w = f("gate_w", c.gate_w);
        c.stock_from = f("stock_from", c.stock_from as f64) as i64;
        c.stock_to = f("stock_to", c.stock_to as f64) as i64;
        c.stock_min = f("stock_min", c.stock_min as f64) as i64;
        c.stock_h = f("stock_h", c.stock_h as f64) as i64;
        c.stock_w = f("stock_w", c.stock_w as f64) as i64;
        if !j.get("sell_order").is_null() {
            c.sell_order = j.get("sell_order").str().to_string();
        }
        if !j.get("stock_items").is_null() {
            c.stock_items = j.get("stock_items").arr().iter().filter_map(|x| ITEMS.iter().copied().find(|i| *i == x.str())).collect();
        }
        c.stock_any = if j.get("stock_any").is_null() { c.stock_any } else { j.get("stock_any").bool() };
        if !j.get("stock_src").is_null() {
            c.stock_src = j.get("stock_src").str().to_string();
        }
        if !j.get("stock_groups").is_null() {
            c.stock_groups = j.get("stock_groups").arr().iter().map(|x| x.i64() as usize).collect();
        }
        if !j.get("stock_variants").is_null() {
            for v in j.get("stock_variants").arr() {
                let g = |k: &str, d: i64| if v.get(k).is_null() { d } else { v.get(k).f64() as i64 };
                let sel = StockSel { from: g("stock_from", c.stock_from), to: g("stock_to", c.stock_to), min: g("stock_min", c.stock_min), h: g("stock_h", c.stock_h), w: g("stock_w", c.stock_w) };
                c.stock_variants.push((v.get("p").f64(), sel));
            }
        }
        if !j.get("opt_items").is_null() {
            c.opt_items = j.get("opt_items").arr().iter().map(|x| crate::act::intern(x.str())).collect();
        }
        c.demand_by = j.get("demand_by").obj().iter().map(|(n, v)| (crate::act::intern(n), v.f64())).collect();
        if !j.get("demand_items").is_null() {
            c.demand_items = j.get("demand_items").arr().iter().map(|x| crate::act::intern(x.str())).collect();
        }
        c.guard_look = i("guard_look", c.guard_look);
        c.guard_to = i("guard_to", c.guard_to);
        if !j.get("guard_items").is_null() {
            c.guard_items = j.get("guard_items").arr().iter().map(|x| crate::act::intern(x.str())).collect();
        }
        if !j.get("clusters").is_null() {
            let mut cp = j.get("clusters").str().to_string();
            // relative: the current directory first (main.py runs in the tarball root), else beside the config file
            if !std::path::Path::new(&cp).exists() {
                if let Some(d) = std::path::Path::new(path).parent() {
                    let alt = d.join(&cp);
                    if alt.exists() {
                        cp = alt.to_string_lossy().into_owned();
                    }
                }
            }
            let cj = kagg_engine::json::parse(&std::fs::read_to_string(&cp).map_err(|e| format!("{cp}: {e}"))?)?;
            let tab: Vec<Vec<Vec<f32>>> = cj
                .get("clusters")
                .arr()
                .iter()
                .map(|cl| ITEMS.iter().map(|it| { let v = cl.get("p_step").get(it); if v.is_null() { vec![0.0; 720] } else { v.arr().iter().map(|x| x.f64() as f32).collect() } }).collect())
                .collect();
            c.clusters = Some(std::sync::Arc::new(tab));
            let lots: Vec<Vec<f32>> = cj.get("clusters").arr().iter().map(|cl| ITEMS.iter().map(|it| { let v = cl.get("lot").get(it); if v.is_null() { 0.0 } else { (v.f64() as f32).min(200.0) } }).collect()).collect();
            c.lots = Some(std::sync::Arc::new(lots));
            c.background = cj.get("clusters").arr().iter().map(|cl| !cl.get("background").is_null() && cl.get("background").bool()).collect();
        }
        if !j.get("items").is_null() {
            c.items = j.get("items").arr().iter().map(|x| crate::act::intern(x.str())).collect();
        }
        Ok(c)
    }
}

fn shop_items(s: &str) -> &'static [&'static str] {
    kagg_engine::engine::shop_products(s)
}

fn fnv_item(item: &str) -> u64 {
    item.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

fn base_price(item: &str) -> f64 {
    kagg_engine::market::param(item).map(|p| p.base).unwrap_or(0.0)
}

#[derive(Clone, Debug, Default)]
pub struct Preempt {
    pub cfg: std::sync::Arc<PreCfg>,
    prev: Option<(i64, Qty, Vec<&'static str>, Vec<&'static str>)>, // step, market inventory, shops, items we sold
    /// (item, hour) -> number of days the rival sold it at that hour
    seen: Vec<(&'static str, i64, i64)>,
    jit: Vec<(&'static str, i64)>,
    /// units already sold early: (item, scheduled step, units) -- deducted from the order at that step
    presold: Vec<(&'static str, i64, i64)>,
    salt: u64,
    /// log-likelihood of the rival's observed sales under each offline cluster
    ll: Vec<f64>,
    /// wheat trade: units bought and not yet resold, their total cost
    arb_held: i64,
    arb_cost: f64,
    /// trailing wheat quotes (last 24 steps)
    wpx: Vec<i64>,
    pub fired: u32,
    /// rival stock estimate: per rival tile the (product, yield) last step; units collected and sold per item
    rtiles: Vec<(Option<&'static str>, i64)>,
    rcollected: Qty,
    rsold: Qty,
    /// (step, item, +collected / -sold) for the windowed estimate
    revents: Vec<(i64, &'static str, i64)>,
    /// The rival's group (set by Base before apply) and this game's stock-rule parameters (drawn once the group is known).
    pub group: Option<usize>,
    sel: Option<StockSel>,
    /// The shared Opponent State Tracker's estimate (crate::gt::ITEMS order), set by Base every turn; used when
    /// `stock_src` is "gt" (v63.12 G3).
    pub gt_stock: [i64; 7],
}

/// One stock-rule parameter set (see PreCfg::stock_*).
#[derive(Clone, Copy, Debug, Default)]
pub struct StockSel {
    pub from: i64,
    pub to: i64,
    pub min: i64,
    pub h: i64,
    pub w: i64,
}

/// Town drain of `item` over steps [s, t): shop ticks every 4 steps (x2 for single-product shops) + the town
/// centre once a day (not FERTILIZER), with the shops unlocked now.
fn drain(item: &str, shops: &[&'static str], s: i64, t: i64) -> i64 {
    let mut d = 0;
    let per: i64 = shops.iter().map(|sh| { let its = shop_items(sh); if its.contains(&item) { if its.len() == 1 { 2 } else { 1 } } else { 0 } }).sum();
    for u in s..t {
        if u % 4 == 0 {
            d += per;
        }
        if u % 24 == 0 && item != "FERTILIZER" {
            d += 1;
        }
    }
    d
}

fn mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

impl Preempt {
    pub fn new(cfg: std::sync::Arc<PreCfg>) -> Preempt {
        Preempt { cfg, ..Default::default() }
    }

    /// The rival's estimated stock of `item`: visible harvests / collections minus recovered sales.
    pub fn rival_stock(&self, item: &str) -> i64 {
        if self.cfg.stock_src == "gt" {
            return crate::gt::ITEMS.iter().position(|x| *x == item).map(|i| self.gt_stock[i]).unwrap_or(0);
        }
        let w = self.sel.map(|s| s.w).unwrap_or(self.cfg.stock_w);
        if w > 0 {
            return self.revents.iter().filter(|e| e.1 == item).map(|e| e.2).sum::<i64>().max(0);
        }
        (qget(&self.rcollected, item) - qget(&self.rsold, item)).max(0)
    }

    fn forecast(&self, item: &str, hour: i64, days: i64) -> f64 {
        let n = self.seen.iter().find(|s| s.0 == item && s.1 == hour).map(|s| s.2).unwrap_or(0);
        n as f64 / days.max(1) as f64
    }

    /// The rival is matched to a known tape lineage (step >= 144 and lineage posterior >= gate_w, or >= 0.8 if unset).
    pub fn matched(&self, step: i64) -> bool {
        step >= 144 && self.lineage_weight() >= if self.cfg.gate_w > 0.0 { self.cfg.gate_w } else { 0.8 }
    }

    /// Posterior weight of the tape lineages (everything but the background entry).
    pub fn lineage_weight(&self) -> f64 {
        if self.ll.is_empty() {
            return 0.0;
        }
        let m = self.ll.iter().cloned().fold(f64::MIN, f64::max);
        let (mut lin, mut all) = (0.0, 0.0);
        for (c, l) in self.ll.iter().enumerate() {
            let w = (l - m).exp();
            all += w;
            if !self.cfg.background.get(c).copied().unwrap_or(false) {
                lin += w;
            }
        }
        lin / all.max(1e-12)
    }

    fn rival_step(&self, i: usize, u: i64) -> f64 {
        let (Some(tab), Some(lots)) = (self.cfg.clusters.as_ref(), self.cfg.lots.as_ref()) else { return 0.0 };
        if self.ll.is_empty() || !(0..720).contains(&u) {
            return 0.0;
        }
        let m = self.ll.iter().cloned().fold(f64::MIN, f64::max);
        let (mut num, mut den) = (0.0, 0.0);
        for (c, l) in self.ll.iter().enumerate() {
            let w = (l - m).exp();
            num += w * tab[c][i][u as usize] as f64 * lots[c][i] as f64;
            den += w;
        }
        num / den.max(1e-12)
    }

    /// The optimiser's sale quantity for `item` now (None = not handled).
    fn optimise(&self, item: &'static str, v: &View, incoming: i64, cap: i64, noise: &mut dyn FnMut() -> f64) -> Option<i64> {
        let c = &self.cfg;
        let p = kagg_engine::market::param(item)?;
        let i = ITEMS.iter().position(|x| *x == item)?;
        let s0 = v.step;
        let end = (s0 + c.opt_h).min(c.opt_to + 1);
        let h = (end - s0).max(1) as usize;
        let have = v.shed(item).max(0);
        let n = (have + incoming).min(c.opt_max_units).max(0) as usize;
        if n == 0 {
            return Some(0);
        }
        // projected pre-sale market stock at each step: drain after each step's market, rival sells first
        let per: i64 = v.obs.shops.iter().map(|sh| { let its = shop_items(sh); if its.contains(&item) { if its.len() == 1 { 2 } else { 1 } } else { 0 } }).sum();
        let mut b = vec![0f64; h + 1];
        let mut inv = qget(&v.obs.mkt_inventory, item) as f64;
        for t in 0..=h {
            let u = s0 + t as i64;
            inv += self.rival_step(i, u);
            b[t] = inv + noise() * c.opt_noise;
            if u % 4 == 0 {
                inv -= per as f64;
            }
            if u % 24 == 0 {
                inv -= 1.0;
            }
        }
        // units available by step t: shed now, the incoming at the next day boundary
        let day_end = (24 - s0.rem_euclid(24)) as usize;
        let avail = |t: usize| -> usize { if t >= day_end { n } else { (have as usize).min(n) } };
        // shed capacity: units of this product we may still hold at step t (held beyond it are discarded)
        let cap = cap.max(0) as usize;
        let pr = |x: f64| kagg_engine::market::price(p, x.round()) as f64;
        // prefix revenue: P_t(m) = sum_{j<m} price(b_t + j)
        let mut pre = vec![vec![0f64; n + 1]; h + 1];
        for t in 0..=h {
            for m in 0..n {
                pre[t][m + 1] = pre[t][m] + pr(b[t] + m as f64);
            }
        }
        // V[t][k]: best revenue from step t with k of our units already sold (they sit in market stock)
        let mut nxt = vec![0f64; n + 1];
        for k in 0..=n {
            nxt[k] = c.opt_salvage * c.opt_gamma.powi(h as i32) * (pre[h][n] - pre[h][k]); // left over: sold at the horizon's price level
        }
        let mut first_q = 0usize;
        for t in (0..h).rev() {
            let mut cur = vec![f64::MIN; n + 1];
            let a = avail(t);
            for k in 0..=n {
                // holding a - k units here must fit the shed: otherwise selling at least the excess is forced
                let q_min = a.saturating_sub(k).saturating_sub(cap);
                let mut best = if q_min == 0 { nxt[k] } else { f64::MIN };
                let mut bq = 0usize;
                if k < a {
                    for q in q_min.max(1)..=(a - k) {
                        let val = c.opt_gamma.powi(t as i32) * (pre[t][k + q] - pre[t][k]) + nxt[k + q];
                        if val > best + 1e-9 {
                            best = val;
                            bq = q;
                        }
                    }
                }
                cur[k] = best;
                if t == 0 && k == 0 {
                    first_q = bq;
                }
            }
            nxt = cur;
        }
        Some(first_q.min(have as usize) as i64)
    }

    /// Expected units the rival sells of `item` in steps (s, t): per-step sale probability x lot, cluster-weighted.
    fn rival_units(&self, item: &str, s: i64, t: i64) -> f64 {
        let (Some(tab), Some(lots)) = (self.cfg.clusters.as_ref(), self.cfg.lots.as_ref()) else { return 0.0 };
        let Some(i) = ITEMS.iter().position(|x| *x == item) else { return 0.0 };
        if self.ll.is_empty() {
            return 0.0;
        }
        let m = self.ll.iter().cloned().fold(f64::MIN, f64::max);
        let (mut num, mut den) = (0.0, 0.0);
        for (c, l) in self.ll.iter().enumerate() {
            let w = (l - m).exp();
            let mut e = 0.0;
            for u in (s + 1).max(0)..t.min(720) {
                e += tab[c][i][u as usize] as f64;
            }
            num += w * e * lots[c][i] as f64;
            den += w;
        }
        num / den.max(1e-12)
    }

    /// P(rival sells `item` at step `t`): the clusters' per-step tables weighted by how well each explains the
    /// rival's sales so far (softmax of the log-likelihood); None without clusters.
    fn cluster_forecast(&self, item: &str, t: i64) -> Option<f64> {
        let tab = self.cfg.clusters.as_ref()?;
        let i = ITEMS.iter().position(|x| *x == item)?;
        if t < 0 || t >= 720 || self.ll.is_empty() {
            return None;
        }
        let m = self.ll.iter().cloned().fold(f64::MIN, f64::max);
        // the rival looks unpredictable (the background explains it best): no forecast, no pre-emption
        if let Some(best) = self.ll.iter().position(|l| *l == m) {
            if self.cfg.background.get(best).copied().unwrap_or(false) {
                return Some(0.0);
            }
        }
        let (mut num, mut den) = (0.0, 0.0);
        for (c, l) in self.ll.iter().enumerate() {
            if self.cfg.background.get(c).copied().unwrap_or(false) {
                continue;
            }
            let w = (l - m).exp();
            num += w * tab[c][i][t as usize] as f64;
            den += w;
        }
        Some(num / den.max(1e-12))
    }

    pub fn apply(&mut self, out: &mut Action, v: &View, ch: &Chassis) {
        let step = v.step;
        let player = v.obs.player;
        let c = self.cfg.clone();
        if self.jit.is_empty() {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ 0x9e37_79b9_2026_0928 ^ (player as u64);
            for s in &v.obs.shops {
                for b in s.bytes() {
                    h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                }
            }
            h = (h ^ (v.rival().money as u64)).wrapping_mul(0x0100_0000_01b3);
            self.salt = mix(h ^ 0x51ab_0e42);
            self.jit = ITEMS.iter().enumerate().map(|(i, it)| (*it, ((h >> (i * 7)) % (c.jitter.max(0) as u64 + 1)) as i64)).collect();
        }
        // ---- track the rival's sales ----
        if let Some((ps, inv, shops, sold)) = self.prev.as_ref() {
            if *ps + 1 == step {
                let t0 = *ps;
                let mut town: Qty = vec![];
                if t0 % 4 == 0 {
                    for s in shops {
                        let its = shop_items(s);
                        for it in its {
                            qadd(&mut town, it, if its.len() == 1 { 2 } else { 1 });
                        }
                    }
                }
                if t0 % 24 == 0 {
                    for it in ITEMS {
                        qadd(&mut town, it, 1);
                    }
                }
                let mut obs_sold: Vec<(usize, bool)> = vec![];
                for (ii, it) in ITEMS.iter().enumerate() {
                    if sold.contains(it) {
                        continue;
                    }
                    let r = qget(&v.obs.mkt_inventory, it) - qget(inv, it) + qget(&town, it);
                    obs_sold.push((ii, r > 0));
                }
                if let (Some(tab), true) = (self.cfg.clusters.clone(), t0 >= self.cfg.fp_from) {
                    if self.ll.is_empty() {
                        self.ll = vec![0.0; tab.len()];
                    }
                    let tt = t0.clamp(0, 719) as usize;
                    for (c, l) in self.ll.iter_mut().enumerate() {
                        for &(ii, yes) in &obs_sold {
                            let p = (tab[c][ii][tt] as f64).clamp(0.01, 0.99);
                            *l += if yes { p.ln() } else { (1.0 - p).ln() };
                        }
                    }
                }
                for it in ITEMS {
                    if sold.contains(&it) {
                        continue;
                    }
                    let rival = qget(&v.obs.mkt_inventory, it) - qget(inv, it) + qget(&town, it);
                    if rival > 0 {
                        qadd(&mut self.rsold, it, rival);
                        self.revents.push((step, it, -rival));
                        let hour = t0.rem_euclid(24);
                        match self.seen.iter_mut().find(|s| s.0 == it && s.1 == hour) {
                            Some(s) => s.2 += 1,
                            None => self.seen.push((it, hour, 1)),
                        }
                    }
                }
            }
        }
        // ---- rival stock: a tile's yield that dropped (collected) or a crop tile that was cleared (harvested) ----
        {
            let now: Vec<(Option<&'static str>, i64)> = v.rival().tiles.iter().map(|t| (product_of_tile(t), t.yield_units.max(0))).collect();
            if self.rtiles.len() == now.len() {
                for (a, b) in self.rtiles.iter().zip(&now) {
                    if let (Some(it), y0) = *a {
                        let got = if b.0 == Some(it) { (y0 - b.1).max(0) } else { y0 };
                        if got > 0 {
                            qadd(&mut self.rcollected, it, got);
                            self.revents.push((step, it, got));
                        }
                    }
                }
            }
            self.rtiles = now;
            let sw = self.sel.map(|s| s.w).unwrap_or(c.stock_w);
            if sw > 0 {
                let w = sw;
                self.revents.retain(|e| e.0 > step - w);
            } else {
                self.revents.clear();
            }
        }
        let matched = c.gate_w <= 0.0 || (step >= 144 && self.lineage_weight() >= c.gate_w);
        // ---- rival-stock pre-emption (endgame) ----
        if self.sel.is_none() && step >= 25 {
            let base = StockSel { from: c.stock_from, to: c.stock_to, min: c.stock_min, h: c.stock_h, w: c.stock_w };
            let ok = c.stock_groups.is_empty() || self.group.is_some_and(|g| c.stock_groups.contains(&g));
            let mut sel = if ok { base } else { StockSel { from: 0, ..base } };
            if ok && !c.stock_variants.is_empty() {
                let tot: f64 = c.stock_variants.iter().map(|x| x.0).sum::<f64>().max(1e-9);
                let r = (mix(self.salt ^ 0x6a09_e667_f3bc_c909) >> 11) as f64 / (1u64 << 53) as f64 * tot;
                let mut acc = 0.0;
                for (p, v) in &c.stock_variants {
                    acc += p;
                    if r < acc {
                        sel = *v;
                        break;
                    }
                }
            }
            self.sel = Some(sel);
            pdbg(step, "-", sel.from as f64, "stock_variant", sel.min, sel.h);
        }
        let sp = self.sel.unwrap_or_default();
        if sp.from > 0 && step >= sp.from && step <= sp.to && (c.stock_any || matched) {
            let route = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route);
            for &item in &c.stock_items {
                let rst = self.rival_stock(item);
                if rst < sp.min {
                    continue;
                }
                if (v.price(item) as f64) < c.floor * base_price(item) {
                    pdbg(step, item, rst as f64, "stock_price_floor", v.shed(item), rst);
                    continue;
                }
                let Some(route) = route else { continue };
                let selling: i64 = out.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum();
                let mut planned = 0;
                for t in (step + 1)..=(step + sp.h).min(718) {
                    let r = if t >= 648 { 2 } else { route };
                    if let Some(a) = ch.route(r).tape.get(t as usize) {
                        planned += a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum::<i64>();
                    }
                }
                let want = (v.shed(item) - selling).min(planned);
                if want < 1 {
                    pdbg(step, item, rst as f64, "stock_nothing", v.shed(item), planned);
                    continue;
                }
                pdbg(step, item, rst as f64, "STOCK_FIRE", v.shed(item), want);
                if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                    let nv = o.n(2) + want;
                    o.set_n(2, nv);
                } else if out.market.len() < 10 {
                    out.market.insert(0, Cmd::order("SELL", item, want));
                } else {
                    continue;
                }
                self.fired += 1;
            }
        }
        // ---- sale optimiser ----
        if matched && c.opt && step >= c.from && step <= c.opt_to {
            let f = v.farm();
            let mut inc: Qty = vec![];
            for t in &f.tiles {
                if t.kind == "PLANT" && t.yield_units > 0 {
                    qadd(&mut inc, t.crop, t.yield_units);
                }
            }
            for inv in v.invs() {
                for (it, n) in inv.iter() {
                    qadd(&mut inc, it, (*n).max(0));
                }
            }
            let mut rng = self.salt ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut noise = move || {
                rng = mix(rng.wrapping_add(0x632b_e59b_d9b4_e019));
                (rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            };
            let items = c.opt_items.clone();
            let inc_total: i64 = inc.iter().map(|x| x.1).sum();
            for item in items {
                let have = v.shed(item);
                // only the real overflow past the shed cap forces sales, shared in proportion to stock; free room
                // means nothing is forced
                let mine = have + qget(&inc, item);
                let over = v.shed_total() + inc_total - 100;
                let cap = if over <= 0 || mine <= 0 {
                    mine
                } else {
                    let share = (over as f64 * mine as f64 / (v.shed_total() + inc_total).max(1) as f64).ceil() as i64;
                    (mine - share).max(0)
                };
                let Some(q) = self.optimise(item, v, qget(&inc, item), cap, &mut noise) else { continue };
                if std::env::var_os("KRL_OPT_TRACE").is_some() && have > 0 {
                    let chain: i64 = out.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2)).sum();
                    eprintln!("[opt] step {step} {item} have {have} inc {} cap {cap} price {} chain {chain} -> {q}", qget(&inc, item), v.price(item));
                }
                // replace the chain's own SELL of this product in place (never drop another order: the day-start
                // hires and seed buys sit at the end of the list)
                match out.market.iter().position(|o| o.is_sell3() && o.s(1) == item) {
                    Some(ix) => {
                        if q > 0 {
                            out.market[ix].set_n(2, q);
                        } else {
                            out.market.remove(ix);
                        }
                        out.market.retain({
                            let mut seen = false;
                            move |o| {
                                if o.is_sell3() && o.s(1) == item {
                                    if seen {
                                        return false;
                                    }
                                    seen = true;
                                }
                                true
                            }
                        });
                    }
                    None => {
                        if q > 0 && out.market.len() < 10 {
                            out.market.insert(0, Cmd::order("SELL", item, q));
                        }
                    }
                }
            }
        }
        // ---- sell to demand ----
        if c.demand_ratio > 0.0 && step >= c.from && step < c.demand_to {
            // incoming supply: ripe yield standing on our tiles + what our units carry (crops take days to grow;
            // holding for demand must leave room in the 100-item shed for the harvests already on the way)
            let f = v.farm();
            let mut incoming = 0i64;
            for t in &f.tiles {
                if t.kind == "PLANT" || !t.animal.is_empty() {
                    incoming += t.yield_units.max(0);
                }
            }
            for inv in v.invs() {
                incoming += inv.iter().map(|(_, n)| (*n).max(0)).sum::<i64>();
            }
            let press = v.shed_total() + incoming > c.shed_press;
            if !press {
                // cap the chain's own sales to the demand-priced units
                for o in out.market.iter_mut() {
                    if !o.is_sell3() || !c.demand_items.contains(&o.s(1)) {
                        continue;
                    }
                    let item = o.s(1);
                    let Some(p) = kagg_engine::market::param(item) else { continue };
                    let inv = qget(&v.obs.mkt_inventory, item);
                    let mut n = 0i64;
                    let lvl = c.demand_by.iter().find(|d| d.0 == item).map(|d| d.1).unwrap_or(c.demand_ratio);
                    while n < o.n(2) && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= lvl * p.base {
                        n += 1;
                    }
                    o.set_n(2, n);
                }
                out.market.retain(|o| !(o.is_sell3() && o.n(2) <= 0));
            }
            // at each tick: sell what the town just consumed (the premium), ahead of the rival
            if step % 4 == 1 {
                let mut selling: Qty = vec![];
                for o in &out.market {
                    if o.is_sell3() {
                        qadd(&mut selling, o.s(1), o.n(2).max(0));
                    }
                }
                for &item in &c.demand_items {
                    let Some(p) = kagg_engine::market::param(item) else { continue };
                    let have = v.shed(item) - qget(&selling, item);
                    if have <= 0 {
                        continue;
                    }
                    let inv = qget(&v.obs.mkt_inventory, item) + qget(&selling, item);
                    let mut n = 0i64;
                    let lvl = c.demand_by.iter().find(|d| d.0 == item).map(|d| d.1).unwrap_or(c.demand_ratio);
                    while n < have && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= lvl * p.base {
                        n += 1;
                    }
                    if n <= 0 {
                        continue;
                    }
                    if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                        let nv = o.n(2) + n;
                        o.set_n(2, nv);
                    } else if out.market.len() < 10 {
                        out.market.insert(0, Cmd::order("SELL", item, n));
                    }
                }
            }
        }
        // ---- glut guard: never dump into a glut the drain will clear before the end stage ----
        if c.guard_ratio > 0.0 && step < c.guard_to {
            for o in out.market.iter_mut() {
                if !o.is_sell3() || !c.guard_items.contains(&o.s(1)) {
                    continue;
                }
                let item = o.s(1);
                let Some(p) = kagg_engine::market::param(item) else { continue };
                let inv = qget(&v.obs.mkt_inventory, item);
                let t = (step + c.guard_look).min(c.guard_to);
                let later = kagg_engine::market::price(p, inv as f64 - drain(item, &v.obs.shops, step, t) as f64 + self.rival_units(item, step, t)) as f64;
                let mut n = 0i64;
                while n < o.n(2) && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= c.guard_ratio * later {
                    n += 1;
                }
                o.set_n(2, n);
            }
            out.market.retain(|o| !(o.is_sell3() && o.n(2) <= 0));
        }
        // ---- deduct units already sold early from the chain's orders due now ----
        if !self.presold.is_empty() {
            for o in out.market.iter_mut() {
                if !o.is_sell3() {
                    continue;
                }
                let item = o.s(1);
                let due: i64 = self.presold.iter().filter(|p| p.0 == item && p.1 <= step).map(|p| p.2).sum();
                if due > 0 {
                    let nv = (o.n(2) - due).max(0);
                    o.set_n(2, nv);
                }
            }
            out.market.retain(|o| !(o.is_sell3() && o.n(2) <= 0));
            self.presold.retain(|p| p.1 > step);
        }
        // ---- random lead + price keeping: release planned sales early ----
        if matched && c.lead_max > 0 && step >= c.from && step <= c.to {
            let player_route = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route);
            if let Some(route) = player_route {
                let mut selling: Qty = vec![];
                for o in &out.market {
                    if o.is_sell3() {
                        qadd(&mut selling, o.s(1), o.n(2).max(0));
                    }
                }
                for &item in &c.items {
                    let Some(p) = kagg_engine::market::param(item) else { continue };
                    for t in (step + 1)..=(step + c.lead_max).min(c.to) {
                        let r = if t >= 648 { 2 } else { route };
                        let q: i64 = ch.route(r).tape.get(t as usize).map(|a| a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum()).unwrap_or(0);
                        if q <= 0 || self.presold.iter().any(|x| x.0 == item && x.1 == t) {
                            continue;
                        }
                        let lead = (mix(self.salt ^ (t as u64).wrapping_mul(0x9e37_79b9) ^ fnv_item(item)) % (c.lead_max as u64 + 1)) as i64;
                        if t - lead != step {
                            continue;
                        }
                        let avail = v.shed(item) - qget(&selling, item);
                        let inv = qget(&v.obs.mkt_inventory, item);
                        // what our first unit would fetch at the scheduled step (town drain until then, no rival)
                        // price calculator: stock at the scheduled step = now - town drain + the rival's expected sales
                        // (its cluster's decision table) -- the price our first unit would actually meet there
                        let inv_t = inv as f64 - drain(item, &v.obs.shops, step, t) as f64 + self.rival_units(item, step, t);
                        let p_sched = kagg_engine::market::price(p, inv_t) as f64;
                        let mut n = 0i64;
                        while n < q.min(avail) && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= (1.0 - c.tol) * p_sched {
                            n += 1;
                        }
                        if n <= 0 {
                            continue;
                        }
                        if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                            let nv = o.n(2) + n;
                            o.set_n(2, nv);
                        } else if out.market.len() < 10 {
                            out.market.insert(0, Cmd::order("SELL", item, n));
                        } else {
                            continue;
                        }
                        qadd(&mut selling, item, n);
                        self.presold.push((item, t, n));
                        self.fired += 1;
                    }
                }
            }
        }
        // ---- wheat surplus resale after the drain (corpus: top players resell in ~9 steps at +$2.06, we hold ~20) ----
        self.wpx.push(v.price("WHEAT"));
        if self.wpx.len() > 24 {
            self.wpx.remove(0);
        }
        if c.wsell && step >= c.arb_from && step < c.to && step % 4 == 1 {
            if let (Some(p), Some(route)) = (kagg_engine::market::param("WHEAT"), ch.players.iter().find(|(pl, _)| *pl == player).and_then(|(_, s)| s.route)) {
                // what the tape itself will pick up from the shed (feed runs) and sell over the next hours
                let mut need = c.wsell_buffer;
                for t in (step + 1)..=(step + c.wsell_need_h).min(719) {
                    let r = if t >= 648 { 2 } else { route };
                    if let Some(a) = ch.route(r).tape.get(t as usize) {
                        for u in std::iter::once(&a.farmer).chain(a.hands.iter()) {
                            if u.op() == "PICKUP" && u.s(1) == "WHEAT" {
                                need += if u.len() >= 3 { u.n(2).max(1) } else { 1 };
                            }
                        }
                    }
                }
                let selling: i64 = out.market.iter().filter(|o| o.is_sell3() && o.s(1) == "WHEAT").map(|o| o.n(2).max(0)).sum();
                let surplus = v.shed("WHEAT") - need - selling;
                if surplus > 0 {
                    let mean = self.wpx.iter().sum::<i64>() as f64 / self.wpx.len().max(1) as f64;
                    let inv = qget(&v.obs.mkt_inventory, "WHEAT") + selling;
                    let mut n = 0i64;
                    while n < surplus && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= mean {
                        n += 1;
                    }
                    if n > 0 {
                        if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == "WHEAT") {
                            let nv = o.n(2) + n;
                            o.set_n(2, nv);
                        } else if out.market.len() < 10 {
                            out.market.insert(0, Cmd::order("SELL", "WHEAT", n));
                        }
                    }
                }
            }
        }
        // ---- wheat tick trade (corpus study 29 Sep: top players buy at drain phase 0 and resell after the drain,
        // 218 units / game, +$453 vs the field's $307). Only units WE bought are ever resold: farm wheat untouched.
        if c.arb_max > 0 && step >= c.arb_from && step < 700 {
            if let Some(p) = kagg_engine::market::param("WHEAT") {
                let inv = qget(&v.obs.mkt_inventory, "WHEAT");
                let chain_sells = out.market.iter().any(|o| o.is_sell3() && o.s(1) == "WHEAT" && o.n(2) > 0);
                let chain_buys = out.market.iter().any(|o| o.len() >= 2 && o.op() == "BUY_PRODUCT" && o.s(1) == "WHEAT");
                if self.arb_held > 0 && step % 4 != 0 {
                    // resell our units while each still beats their average cost by arb_abs (all of them from 690)
                    let avg = self.arb_cost / self.arb_held as f64;
                    let base_n: i64 = out.market.iter().filter(|o| o.is_sell3() && o.s(1) == "WHEAT").map(|o| o.n(2).max(0)).sum();
                    let mut n = 0i64;
                    let last = step >= 690;
                    while n < self.arb_held.min((v.shed("WHEAT") - base_n).max(0)) && (last || kagg_engine::market::price(p, (inv + base_n + n) as f64) as f64 >= avg + c.arb_abs) {
                        n += 1;
                    }
                    if n > 0 {
                        if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == "WHEAT") {
                            let nv = o.n(2) + n;
                            o.set_n(2, nv);
                        } else if out.market.len() < 10 {
                            out.market.insert(0, Cmd::order("SELL", "WHEAT", n));
                        } else {
                            n = 0;
                        }
                        if n > 0 {
                            self.arb_cost -= avg * n as f64;
                            self.arb_held -= n;
                        }
                    }
                } else if step % 4 == 0 && !chain_sells && !chain_buys && step < 660 && self.arb_held < c.arb_max {
                    // buy at drain phase 0: the town consumes after the market step, so next step's price is higher;
                    // each unit only if the projected next-step price (drain - the rival's expected wheat sales) beats it
                    let inv_next = inv as f64 - drain("WHEAT", &v.obs.shops, step, step + 1) as f64 + self.rival_units("WHEAT", step, step + 2);
                    let mut n = 0i64;
                    let mut cost = 0.0;
                    let cash = v.money();
                    while self.arb_held + n < c.arb_max {
                        let q = kagg_engine::market::price(p, (inv - n - 1) as f64) as f64; // BUY quotes at post-buy stock
                        let resell = kagg_engine::market::price(p, inv_next - (n + 1) as f64) as f64;
                        if resell < q + c.arb_abs || cash - cost - q < c.arb_cash {
                            break;
                        }
                        cost += q;
                        n += 1;
                    }
                    if n > 0 && out.market.len() < 10 {
                        out.market.push(Cmd::order("BUY_PRODUCT", "WHEAT", n));
                        self.arb_held += n;
                        self.arb_cost += cost;
                    }
                }
            }
        }
        // ---- pre-empt the rival's forecast sales ----
        let days = step / 24;
        if matched && c.threat && step >= c.from && step <= c.to && days >= c.min_days {
            let route = ch.players.iter().find(|(p, _)| *p == player).and_then(|(_, s)| s.route);
            let mut selling: Qty = vec![];
            for o in &out.market {
                if o.is_sell3() {
                    qadd(&mut selling, o.s(1), o.n(2).max(0));
                }
            }
            for &item in &c.items {
                let look = c.look + self.jit.iter().find(|j| j.0 == item).map(|j| j.1).unwrap_or(0);
                let pk: f64 = (1..=look).map(|k| match self.cluster_forecast(item, step + k) {
                    Some(p) => p,
                    None => self.forecast(item, (step + k).rem_euclid(24), days),
                }).fold(0.0, f64::max);
                let threat = pk >= c.p_min;
                if !threat {
                    if pk >= 0.05 && v.shed(item) > 0 {
                        pdbg(step, item, pk, "below_p_min", v.shed(item), 0);
                    }
                    continue;
                }
                if (v.price(item) as f64) < c.floor * base_price(item) {
                    pdbg(step, item, pk, "price_floor", v.shed(item), 0);
                    continue;
                }
                let Some(route) = route else {
                    pdbg(step, item, pk, "no_route", v.shed(item), 0);
                    continue;
                };
                let mut planned = 0;
                for t in (step + 1)..=(step + c.horizon).min(c.to) {
                    let r = if t >= 648 { 2 } else { route };
                    if let Some(a) = ch.route(r).tape.get(t as usize) {
                        planned += a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum::<i64>();
                    }
                }
                let mut want = (v.shed(item) - qget(&selling, item)).min(planned);
                if c.demand_floor > 0.0 {
                    if let Some(p) = kagg_engine::market::param(item) {
                        let inv = qget(&v.obs.mkt_inventory, item);
                        let floor = c.demand_floor * p.base;
                        let mut n = 0i64;
                        while n < want && kagg_engine::market::price(p, (inv + n) as f64) as f64 >= floor {
                            n += 1;
                        }
                        want = n;
                    }
                }
                if want < 1 {
                    pdbg(step, item, pk, if v.shed(item) - qget(&selling, item) < 1 { "nothing_left" } else { "tape_plans_none" }, v.shed(item), planned);
                    continue;
                }
                pdbg(step, item, pk, "FIRE", v.shed(item), want);
                if let Some(o) = out.market.iter_mut().find(|o| o.is_sell3() && o.s(1) == item) {
                    let nv = o.n(2) + want;
                    o.set_n(2, nv);
                } else if out.market.len() < 10 {
                    // front of the queue: the per-unit lockstep race is won by being early in the list too
                    out.market.insert(0, Cmd::order("SELL", item, want));
                } else {
                    continue;
                }
                qadd(&mut selling, item, want);
                self.fired += 1;
            }
        }
        if c.sell_order == "value" {
            let slots: Vec<usize> = (0..out.market.len()).filter(|&i| out.market[i].is_sell3() && out.market[i].n(2) > 0).collect();
            let mut sells: Vec<Cmd> = slots.iter().map(|&i| out.market[i].clone()).collect();
            let val = |o: &Cmd| v.price(o.s(1)) as f64 * o.n(2).min(v.shed(o.s(1)).max(0)) as f64;
            sells.sort_by(|a, b| val(b).partial_cmp(&val(a)).unwrap_or(std::cmp::Ordering::Equal));
            for (k, &i) in slots.iter().enumerate() {
                out.market[i] = sells[k].clone();
            }
        }
        let sold: Vec<&'static str> = ITEMS.iter().copied().filter(|it| out.market.iter().any(|o| o.is_sell3() && o.s(1) == *it && o.n(2) > 0)).collect();
        self.prev = Some((step, v.obs.mkt_inventory.clone(), v.obs.shops.clone(), sold));
    }
}

/// KRL_PRE_DBG=<file>: one line per (step, item) the pre-empt considered: forecast, the gate that stopped it (or FIRE).
fn pdbg(step: i64, item: &str, p: f64, why: &str, shed: i64, n: i64) {
    if let Ok(path) = std::env::var("KRL_PRE_DBG") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{step}	{item}	{p:.3}	{why}	shed={shed}	n={n}");
        }
    }
}

/// The product a tile yields (crop, or the animal's product).
fn product_of_tile(t: &crate::obs::Tile) -> Option<&'static str> {
    if !t.is_dict() {
        return None;
    }
    if t.kind == "PLANT" && !t.crop.is_empty() {
        return ITEMS.iter().copied().find(|i| *i == t.crop);
    }
    match t.animal {
        "GOOSE" => Some("EGG"),
        "COW" => Some("MILK"),
        "SHEEP" => Some("WOOL"),
        _ => None,
    }
}
