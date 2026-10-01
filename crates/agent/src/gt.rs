//! Opponent State Tracker (v63.12 G3, operator's game-theory notes Module 3): the rival's unsold stock per item,
//! rebuilt from public state every turn and shared by the market layers (rshell tranching, preempt standoff).
//!
//!   rival stock = rival harvests - rival market sales
//!   harvests: a rival tile whose yield went to 0 (HARVEST takes every unit; an ongoing crop / an animal stays) or a
//!             crop tile that became EMPTY. A yield that fell by less (rot past the lifespan, 1 unit per 2 steps) or a
//!             tile that became a WEED (died / rotted out) is not a harvest (engine.rs HARVEST / decay / death).
//!   sales:    market inventory change + the town's drain over the step - our own filled sales, where our fills are
//!             exact from our own books: (shed + carried) before - after + what we harvested this step.
//! Deliveries to town shops are invisible, so a whole-game count can only overestimate; `window` > 0 keeps the
//! events of the last `window` steps only.
use crate::obs::{qget, Qty, Tile};
use crate::view::View;

pub const ITEMS: [&str; 7] = ["CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];

fn product(t: &Tile) -> Option<&'static str> {
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

/// Units harvested per item between two snapshots of one farm's tiles.
fn harvested(prev: &[(Option<&'static str>, i64, &'static str)], now: &[Tile]) -> [i64; 7] {
    let mut out = [0i64; 7];
    if prev.len() != now.len() {
        return out;
    }
    for (a, b) in prev.iter().zip(now) {
        let (Some(it), y0, _) = *a else { continue };
        if y0 <= 0 {
            continue;
        }
        let same = product(b) == Some(it);
        // harvested: the yield went to 0 on the same tile (ongoing crop / animal) or the crop tile became empty (null)
        let took = (same && b.yield_units.max(0) == 0) || (!same && !b.is_dict());
        if took {
            out[ITEMS.iter().position(|x| *x == it).unwrap()] += y0;
        }
    }
    out
}

fn snap(tiles: &[Tile]) -> Vec<(Option<&'static str>, i64, &'static str)> {
    tiles.iter().map(|t| (product(t), t.yield_units.max(0), t.kind)).collect()
}

fn shop_items(s: &str) -> &'static [&'static str] {
    kagg_engine::engine::shop_products(s)
}

#[derive(Clone, Debug, Default)]
pub struct RivalTracker {
    prev_step: i64,
    prev_inv: Qty,
    prev_shops: Vec<&'static str>,
    prev_ours: [i64; 7],
    prev_px: Qty,
    prev_rmoney: f64,
    rtiles: Vec<(Option<&'static str>, i64, &'static str)>,
    otiles: Vec<(Option<&'static str>, i64, &'static str)>,
    pub collected: [i64; 7],
    pub sold: [i64; 7],
    /// (step, item index, +harvested / -sold)
    events: Vec<(i64, usize, i64)>,
    pub window: i64,
    started: bool,
    /// KRL_GT_LOG=FILE: per step (our estimate of the rival's stock, our own true stock); one JSON line per game at
    /// the last step, so a self-play run can score each seat's estimate against the other seat's truth.
    log: Vec<([i64; 7], [i64; 7])>,
    /// diagnostics: our own cumulative harvests / filled sales (exact), to score the rival-side estimates
    own_h: [i64; 7],
    own_s: [i64; 7],
    /// A2 rival fingerprint through `fp_to` (D6 = step 144): steps seen, steps with the rival's units on exactly our
    /// squares, steps with exactly our money, the largest |money gap|, the first step the money differed.
    pub fp_n: i64,
    pub fp_pos: i64,
    pub fp_cash: i64,
    pub fp_gap: f64,
    pub fp_div: i64,
    /// Sale manager's rival-sale model (30 Sep): per item and hour of day, the days on which the rival sold it at that
    /// hour, and the units it sold then (the rival's own daily schedule as observed in THIS game).
    pub sale_days: [[i64; 24]; 7],
    pub sale_units: [[i64; 24]; 7],
    last_sale_day: [[i64; 24]; 7],
}

static GT_LOG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ours_stock(v: &View) -> [i64; 7] {
    let mut s = [0i64; 7];
    for (i, it) in ITEMS.iter().enumerate() {
        s[i] = qget(&v.obs.shed, it).max(0) + v.obs.invs.iter().map(|q| qget(q, it).max(0)).sum::<i64>();
    }
    s
}

impl RivalTracker {
    pub fn new(window: i64) -> RivalTracker {
        RivalTracker { window, ..Default::default() }
    }

    /// Call once per turn, before any layer reads it.
    pub fn update(&mut self, v: &View) {
        let step = v.step;
        let ours = ours_stock(v);
        if step >= 1 && step <= 144 {
            let mut a: Vec<(i64, i64)> = v.farm().hands.clone();
            a.extend(v.farm().farmer);
            let mut b: Vec<(i64, i64)> = v.rival().hands.clone();
            b.extend(v.rival().farmer);
            a.sort();
            b.sort();
            let gap = (v.money() - v.rival().money).abs();
            self.fp_n += 1;
            self.fp_pos += (a == b) as i64;
            self.fp_cash += (gap < 0.5) as i64;
            self.fp_gap = self.fp_gap.max(gap);
            if gap >= 0.5 && self.fp_div == 0 {
                self.fp_div = step;
            }
        }
        if self.started && step == self.prev_step + 1 {
            let rh = harvested(&self.rtiles, &v.rival().tiles);
            let oh = harvested(&self.otiles, &v.farm().tiles);
            let t0 = self.prev_step;
            // visible rival revenue this step (units recovered from the market inventory x last quote); a unit sold
            // at the $1 floor does not enter the market inventory (engine commit_unit), so the rest of the rival's
            // money gain is attributed to its floor items below
            let mut vis_rev = 0.0;
            let mut floor: Vec<usize> = vec![];
            for (i, it) in ITEMS.iter().enumerate() {
                if rh[i] > 0 {
                    self.collected[i] += rh[i];
                    self.events.push((step, i, rh[i]));
                }
                let mut town = 0;
                if t0 % 4 == 0 {
                    for s in &self.prev_shops {
                        let its = shop_items(s);
                        if its.contains(it) {
                            town += if its.len() == 1 { 2 } else { 1 };
                        }
                    }
                }
                if t0 % 24 == 0 {
                    town += 1;
                }
                let our_sold = (self.prev_ours[i] - ours[i] + oh[i]).max(0);
                self.own_h[i] += oh[i];
                self.own_s[i] += our_sold;
                let rs = qget(&v.obs.mkt_inventory, it) - qget(&self.prev_inv, it) + town - our_sold;
                if rs > 0 {
                    self.sold[i] += rs;
                    self.events.push((step, i, -rs));
                    self.note_sale(t0, i, rs);
                    vis_rev += rs as f64 * qget(&self.prev_px, it).max(1) as f64;
                }
                if qget(&self.prev_px, it) <= 1 {
                    floor.push(i);
                }
            }
            let resid = (v.rival().money - self.prev_rmoney - vis_rev).round() as i64;
            let mut left = resid;
            for &i in &floor {
                if left <= 0 {
                    break;
                }
                let have = (self.collected[i] - self.sold[i]).max(0);
                let q = have.min(left);
                if q > 0 {
                    self.sold[i] += q;
                    self.events.push((step, i, -q));
                    self.note_sale(t0, i, q);
                    left -= q;
                }
            }
        }
        if self.window > 0 {
            let w = self.window;
            self.events.retain(|e| e.0 > step - w);
        }
        if std::env::var_os("KRL_GT_LOG").is_some() {
            self.log.push((ITEMS.map(|it| self.stock(it)), ours));
            if step >= 718 {
                self.dump(v);
            }
        }
        self.started = true;
        self.prev_step = step;
        self.prev_inv = v.obs.mkt_inventory.clone();
        self.prev_shops = v.obs.shops.clone();
        self.prev_ours = ours;
        self.prev_px = v.obs.prices.clone();
        self.prev_rmoney = v.rival().money;
        self.rtiles = snap(&v.rival().tiles);
        self.otiles = snap(&v.farm().tiles);
    }

    fn dump(&mut self, v: &View) {
        let Ok(path) = std::env::var("KRL_GT_LOG") else { return };
        let arr = |x: &[i64; 7]| format!("[{}]", x.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","));
        let est: Vec<String> = self.log.iter().map(|e| arr(&e.0)).collect();
        let own: Vec<String> = self.log.iter().map(|e| arr(&e.1)).collect();
        let line = format!("{{\"player\":{},\"money\":{},\"rmoney\":{},\"shops\":\"{}\",\"window\":{},\"est\":[{}],\"own\":[{}],\"coll\":{},\"sold\":{},\"own_h\":{},\"own_s\":{},\"fp\":[{},{},{},{:.0},{}]}}
",
            v.obs.player, v.money(), v.rival().money, v.obs.shops.join("|"), self.window, est.join(","), own.join(","), arr(&self.collected), arr(&self.sold), arr(&self.own_h), arr(&self.own_s), self.fp_n, self.fp_pos, self.fp_cash, self.fp_gap, self.fp_div);
        let _g = GT_LOG_LOCK.lock();
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
        }
        self.log.clear();
    }

    /// The rival sold `q` units of item `i` on step `t`: count the (item, hour) once per day.
    fn note_sale(&mut self, t: i64, i: usize, q: i64) {
        let (d, h) = (t.div_euclid(24) + 1, t.rem_euclid(24) as usize);
        if self.last_sale_day[i][h] != d {
            self.last_sale_day[i][h] = d;
            self.sale_days[i][h] += 1;
        }
        self.sale_units[i][h] += q;
    }

    /// P(the rival sells `item` at hour-of-day `h`) from its schedule so far (days observed = `days`) and its mean lot then.
    pub fn sale_forecast(&self, item: &str, h: i64, days: i64) -> (f64, f64) {
        let Some(i) = ITEMS.iter().position(|x| *x == item) else { return (0.0, 0.0) };
        let h = h.rem_euclid(24) as usize;
        let n = self.sale_days[i][h];
        let p = n as f64 / days.max(1) as f64;
        let lot = if n > 0 { self.sale_units[i][h] as f64 / n as f64 } else { 0.0 };
        (p.min(1.0), lot)
    }

    /// The rival's estimated unsold stock of `item` (whole game, or the last `window` steps when window > 0).
    pub fn stock(&self, item: &str) -> i64 {
        let Some(i) = ITEMS.iter().position(|x| *x == item) else { return 0 };
        if self.window > 0 {
            return self.events.iter().filter(|e| e.1 == i).map(|e| e.2).sum::<i64>().max(0);
        }
        (self.collected[i] - self.sold[i]).max(0)
    }
}

// ------------------------------------------------------------------------------------------------------------------
// Game-theoretic liquidation layer (v63.12 G4 tranching, G5 endgame standoff, G6 queue order, G7 guards).
// Market-side only: it edits SELL quantities and the order list; farm actions and the tape stay untouched.
// ------------------------------------------------------------------------------------------------------------------

use crate::act::{Action, Cmd};

pub const GT_ITEMS: [&str; 4] = ["STRAWBERRY", "MELON", "MILK", "WOOL"];

/// `gt.json` (python/v6312/g2_msne.py writes `mix`; the rest are knobs):
/// {"groups":[1,2], "mid_from":144, "mid_to":647, "mid_min":15, "tranche":12, "end_from":648, "end_to":711,
///  "end_min":12, "end_step":680, "end_jitter":4, "queue":["WOOL","STRAWBERRY","MILK","MELON"],
///  "mix": {"STRAWBERRY|mid|low": {"p":[h,t,d]}, ...}, "items":[...], "salt":1, "only":"mid"|"end"|""}
#[derive(Clone, Debug, Default)]
pub struct GtCfg {
    pub groups: Vec<usize>,
    pub mid_from: i64,
    pub mid_to: i64,
    pub mid_min: i64,
    pub tranche: i64,
    pub end_from: i64,
    pub end_to: i64,
    pub end_min: i64,
    pub end_step: i64,
    pub end_jitter: i64,
    pub queue: Vec<&'static str>,
    pub items: Vec<&'static str>,
    /// (item|regime|bin) -> [P(hold), P(tranche), P(dump)]
    pub mix: Vec<(String, [f64; 3])>,
    pub salt: u64,
    /// "mid" / "end" / "" (both): which regimes act (ablation)
    pub only: String,
    /// G5 (G1b-calibrated): our own shed must hold >= this before an endgame dump; one dump per item per `end_every` steps
    pub end_self_min: i64,
    pub end_every: i64,
}

impl GtCfg {
    pub fn load(path: &str) -> Result<GtCfg, String> {
        Self::load_over(path, &kagg_engine::json::Json::Null)
    }

    /// `load` with the keys of `over` (an agent.json manager override) replacing the file's.
    pub fn load_over(path: &str, over: &kagg_engine::json::Json) -> Result<GtCfg, String> {
        let mut j = kagg_engine::json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        crate::managers::merge(&mut j, over);
        let n = |k: &str, d: f64| if j.get(k).is_null() { d } else { j.get(k).f64() };
        let strs = |k: &str, d: &[&'static str]| -> Vec<&'static str> {
            if j.get(k).is_null() { d.to_vec() } else { j.get(k).arr().iter().map(|x| crate::act::intern(x.str())).collect() }
        };
        let mut mix = vec![];
        if !j.get("mix").is_null() {
            for (k, v) in j.get("mix").obj() {
                let p = v.get("p").arr();
                if p.len() == 3 {
                    mix.push((k.clone(), [p[0].f64(), p[1].f64(), p[2].f64()]));
                }
            }
        }
        Ok(GtCfg {
            groups: if j.get("groups").is_null() { vec![1, 2] } else { j.get("groups").arr().iter().map(|x| x.f64() as usize).collect() },
            mid_from: n("mid_from", 144.0) as i64,
            mid_to: n("mid_to", 647.0) as i64,
            mid_min: n("mid_min", 15.0) as i64,
            tranche: n("tranche", 12.0) as i64,
            end_from: n("end_from", 648.0) as i64,
            end_to: n("end_to", 711.0) as i64,
            end_min: n("end_min", 12.0) as i64,
            end_step: n("end_step", 680.0) as i64,
            end_jitter: n("end_jitter", 4.0) as i64,
            queue: strs("queue", &["WOOL", "STRAWBERRY", "MILK", "MELON"]),
            items: strs("items", &GT_ITEMS),
            mix,
            salt: n("salt", 1.0) as u64,
            only: if j.get("only").is_null() { String::new() } else { j.get("only").str().to_string() },
            end_self_min: n("end_self_min", 1.0) as i64,
            end_every: n("end_every", 0.0) as i64,
        })
    }
}

fn base_price(item: &str) -> f64 {
    kagg_engine::market::param(item).map(|p| p.base).unwrap_or(1.0)
}

fn mix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

#[derive(Clone, Debug, Default)]
pub struct GtLayer {
    pub cfg: std::sync::Arc<GtCfg>,
    pub group: Option<usize>,
    h: Option<u64>,
    /// per item: this game's endgame activation step (sampled once)
    end_at: Vec<(&'static str, i64)>,
    /// hold, tranche, dump (mid game), endgame fire
    pub hits: [u32; 4],
    /// open mid-game windows: (item, start step, strategy, units still allowed)
    win: Vec<(&'static str, i64, usize, i64)>,
    /// G1b labels: force one strategy for one item over [step, step+24): 0 hold, 1 tranche, 2 dump now, 3 tape (off)
    pub force: Option<(&'static str, i64, usize)>,
    /// last endgame dump step per item
    last_dump: Vec<(&'static str, i64)>,
}

impl GtLayer {
    pub fn new(cfg: std::sync::Arc<GtCfg>) -> GtLayer {
        GtLayer { cfg, ..Default::default() }
    }

    fn unit(&self, a: u64, b: u64) -> f64 {
        (mix64(self.h.unwrap_or(0) ^ a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// This game's draw from the MSNE mix of (item, regime, price bin) for `day` (0 hold, 1 tranche, 2 dump).
    fn strategy(&self, item: &str, regime: &str, px: i64, day: i64) -> usize {
        let r = px as f64 / base_price(item).max(1.0);
        let bin = if r < 0.6 { "low" } else if r < 1.2 { "mid" } else { "high" };
        let key = format!("{item}|{regime}|{bin}");
        let Some((_, p)) = self.cfg.mix.iter().find(|(k, _)| *k == key) else { return 2 };
        let u = self.unit(item.bytes().fold(7u64, |h, b| h.wrapping_mul(31) ^ b as u64), day as u64) * (p[0] + p[1] + p[2]).max(1e-9);
        if u < p[0] {
            0
        } else if u < p[0] + p[1] {
            1
        } else {
            2
        }
    }

    fn forced(&mut self, a: &mut Action, v: &View, item: &'static str, from: i64, st: usize) {
        let step = v.step;
        if st == 3 || step < from || step >= from + 24 {
            return;
        }
        if step == from {
            let s0 = v.shed(item);
            self.win = vec![(item, from, st, match st { 0 => s0 / 4, 1 => (s0 * 3) / 4, _ => i64::MAX })];
        }
        if st == 2 {
            if step == from && v.shed(item) > 0 {
                a.market.retain(|o| !(o.is_sell3() && o.s(1) == item));
                a.market.insert(0, Cmd::order("SELL", item, v.shed(item)));
                a.market.truncate(crate::tpp::MAXM);
            }
            return;
        }
        let Some(w) = self.win.first().copied() else { return };
        let cap = if st == 1 { w.3.min(self.cfg.tranche.max(0)) } else { w.3 };
        let mut room = cap.max(0);
        let mut used = 0;
        for o in a.market.iter_mut().filter(|o| o.is_sell3() && o.s(1) == item) {
            let q = o.n(2).max(0).min(room).min((v.shed(item) - used).max(0));
            o.set_n(2, q);
            room -= q;
            used += q;
        }
        self.win[0].3 = w.3 - used;
    }

    pub fn apply(&mut self, a: &mut Action, v: &View, tr: &RivalTracker) {
        let c = self.cfg.clone();
        let step = v.step;
        if self.h.is_none() {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ c.salt.wrapping_mul(0x51ab_0e42) ^ (v.obs.player as u64);
            for s in &v.obs.shops {
                for b in s.bytes() {
                    h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
                }
            }
            h = (h ^ (v.rival().money as u64)).wrapping_mul(0x0100_0000_01b3);
            self.h = Some(h);
        }
        if let Some((item, from, st)) = self.force {
            self.forced(a, v, item, from, st);
            return;
        }
        let Some(g) = self.group else { return };
        if !c.groups.contains(&g) {
            return;
        }
        let day = step / 24;
        // ---- G4: mid-game tranching over 24-step windows (G1's classes: of the stock held at the window start,
        // HOLD releases < 25%, TRANCHE 25-75% at most `tranche` per step, DUMP anything) ----
        if c.only != "end" && step >= c.mid_from && step <= c.mid_to {
            for &item in &c.items {
                let k = self.win.iter().position(|w| w.0 == item);
                let expired = k.map_or(true, |k| step >= self.win[k].1 + 24);
                if expired {
                    if let Some(k) = k {
                        self.win.remove(k);
                    }
                    if tr.stock(item) < c.mid_min || v.shed(item) < 1 {
                        continue;
                    }
                    let st = self.strategy(item, "mid", v.price(item), day);
                    let s0 = v.shed(item);
                    let budget = match st {
                        0 => s0 / 4,
                        1 => (s0 * 3) / 4,
                        _ => i64::MAX,
                    };
                    self.hits[st] += 1;
                    self.win.push((item, step, st, budget));
                }
                let Some(k) = self.win.iter().position(|w| w.0 == item) else { continue };
                let (_, _, st, left) = self.win[k];
                if st == 2 {
                    continue;
                }
                let cap = if st == 1 { left.min(c.tranche.max(0)) } else { left };
                let mut room = cap.max(0);
                let mut used = 0;
                for o in a.market.iter_mut().filter(|o| o.is_sell3() && o.s(1) == item) {
                    let q = o.n(2).max(0).min(room).min(v.shed(item) - used);
                    let q = q.max(0);
                    o.set_n(2, q);
                    room -= q;
                    used += q;
                }
                self.win[k].3 = left - used;
            }
        }
        // ---- G5: endgame standoff (the rival holds stock -> front-run its dump at a per-game sampled step) ----
        if c.only != "mid" && step >= c.end_from && step <= c.end_to {
            if self.end_at.is_empty() {
                for (i, &item) in c.items.iter().enumerate() {
                    let j = (self.unit(0xE17D ^ i as u64, 0) * (2 * c.end_jitter + 1) as f64) as i64 - c.end_jitter;
                    self.end_at.push((item, c.end_step + j));
                }
            }
            let mut fire: Vec<&'static str> = vec![];
            for &item in &c.queue {
                if !c.items.contains(&item) || v.shed(item) < c.end_self_min.max(1) {
                    continue;
                }
                if c.end_every > 0 && self.last_dump.iter().any(|x| x.0 == item && step < x.1 + c.end_every) {
                    continue;
                }
                let at = self.end_at.iter().find(|x| x.0 == item).map(|x| x.1).unwrap_or(c.end_step);
                let dump = self.strategy(item, "end", v.price(item), day) == 2;
                if tr.stock(item) >= c.end_min && (step >= at || dump) {
                    fire.push(item);
                }
            }
            if !fire.is_empty() {
                // G6/G7: the whole shed of each fired item (never more), at the front in queue order
                let mut front: Vec<Cmd> = vec![];
                for &item in &fire {
                    a.market.retain(|o| !(o.is_sell3() && o.s(1) == item));
                    let want = v.shed(item);
                    if want > 0 {
                        front.push(Cmd::order("SELL", item, want));
                        self.hits[3] += 1;
                        self.last_dump.retain(|x| x.0 != item);
                        self.last_dump.push((item, step));
                    }
                }
                let keep = crate::tpp::MAXM.saturating_sub(front.len());
                if a.market.len() > keep {
                    // drop zero-quantity no-ops first, then the tail
                    let mut rest: Vec<Cmd> = a.market.iter().filter(|o| !(o.len() >= 3 && o.n(2) == 0)).cloned().collect();
                    rest.truncate(keep);
                    a.market = rest;
                }
                front.append(&mut a.market);
                a.market = front;
            }
        }
    }
}
