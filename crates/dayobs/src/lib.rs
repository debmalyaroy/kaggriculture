//! The macro policy's per-day observation, built ONLINE from what one seat sees each step.
//!
//! One code path for training and play: the corpus adapter (`features::dayobs_adapter`) feeds
//! a slim episode through [`Builder::push`] step by step exactly as the agent feeds its live
//! observations, so the vector the policy is trained on is the vector it acts on.
//!
//! Per step the caller supplies a [`StepView`] (public state + the seat's own shed, from the
//! seat's point of view: index 0 = own, 1 = rival) and the seat's own market orders applied at
//! that step ([`OwnAct`]). At each decision step (hour 1 of day d, engine step 24d+1)
//! [`Builder::vector`] returns the [`N`]-float vector named by [`NAMES`].
//!
//! Only information the seat has at play time is used. The rival's sales are inferred from
//! public market-inventory changes (+ the town's draw - own fills + own buys), as in the corpus
//! `rival_sold_*` columns.
//!
//! Versioning: any change to the vector bumps [`FEAT_VERSION`]; weights record the version they
//! were trained on and the builder refuses a mismatch.

pub const FEAT_VERSION: u32 = 2;
pub const TURNS_PER_DAY: usize = 24;
const SHOP_SELL_INTERVAL: usize = 4;
const CENTER_SELL_INTERVAL: usize = 24;

/// Market products, alphabetical (same order as the corpus columns).
pub const PRODUCTS: [&str; 9] = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK", "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"];
pub const CROPS: [&str; 5] = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"];
pub const CROP_FIRST_YIELD_DAY: [i64; 5] = [2, 10, 10, 8, 2];
pub const ANIMALS: [&str; 3] = ["COW", "GOOSE", "SHEEP"];
pub const SHOPS: [&str; 8] = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"];
const P_FERTILIZER: usize = 2;
const P_WHEAT: usize = 7;

pub fn index_of(list: &[&str], s: &str) -> Option<usize> {
    list.iter().position(|&x| x == s)
}

/// Products each shop instance pulls per shop tick (engine `SHOPS`).
pub fn shop_items(shop: usize) -> &'static [usize] {
    match shop {
        0 => &[1, 7],
        1 => &[1, 7, 5],
        2 => &[7, 0, 6, 5],
        3 => &[5, 4, 7],
        4 => &[0],
        5 => &[4, 6, 7],
        6 => &[5, 4],
        7 => &[8],
        _ => &[],
    }
}

/// Engine `_town_consume` at step t for the given shop instances (SHOPS indices).
pub fn town_draw(shops: &[u8], t: usize) -> [f64; 9] {
    let mut d = [0.0; 9];
    if t % SHOP_SELL_INTERVAL == 0 {
        for &s in shops {
            let items = shop_items(s as usize);
            let mult = if items.len() == 1 { 2.0 } else { 1.0 };
            for &i in items {
                d[i] += mult;
            }
        }
    }
    if t % CENTER_SELL_INTERVAL == 0 {
        for (i, x) in d.iter_mut().enumerate() {
            if i != P_FERTILIZER {
                *x += 1.0;
            }
        }
    }
    d
}

/// Tile counts for one farm at a snapshot (same definitions as the corpus `*_tiles_*` columns).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TileCount {
    pub planted: u16,
    pub growing: u16,
    pub ripe: u16,
    pub weed: u16,
    pub empty: u16,
    pub pasture: u16,
    pub coop: u16,
    pub by_crop: [u16; 5],
    pub animals: [u16; 3],
}

/// The tile categories the counter needs (callers map their own tile types onto this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileKind {
    Locked,
    Empty,
    Plant,
    Weed,
    Pasture,
    Coop,
    Other,
}

impl TileCount {
    /// Count one tile. `crop`/`animal` are CROPS/ANIMALS indices (None if absent/unknown).
    pub fn add(&mut self, kind: TileKind, crop: Option<usize>, animal: Option<usize>, yield_units: f64, planted_day: i64, day: i64) {
        match kind {
            TileKind::Empty => self.empty += 1,
            TileKind::Weed => self.weed += 1,
            TileKind::Pasture => self.pasture += 1,
            TileKind::Coop => self.coop += 1,
            TileKind::Plant => {
                self.planted += 1;
                if let Some(c) = crop {
                    self.by_crop[c] += 1;
                    if yield_units > 0.0 && day - planted_day >= CROP_FIRST_YIELD_DAY[c] {
                        self.ripe += 1;
                    } else {
                        self.growing += 1;
                    }
                } else {
                    self.growing += 1;
                }
            }
            TileKind::Locked | TileKind::Other => {}
        }
        // any tile carrying an animal counts (corpus tile_stats convention)
        if let Some(a) = animal {
            self.animals[a] += 1;
        }
    }
}

/// What one seat sees at one engine step. Index 0 = own farm, 1 = rival.
#[derive(Clone, Debug, Default)]
pub struct StepView {
    pub step: usize,
    pub money: [f64; 2],
    pub hires: [f64; 2],
    pub hands: [u16; 2],
    pub quads: [u8; 2],
    /// Farmer position and hands identical across the two farms.
    pub pos_eq: bool,
    pub prices: [f64; 9],
    pub inv: [f64; 9],
    /// Unlocked shop instances (SHOPS indices; unknown names dropped).
    pub shops: Vec<u8>,
    pub own_shed: [f64; 9],
    /// Tile counts [own, rival]; needed at decision steps only.
    pub tiles: Option<[TileCount; 2]>,
    /// Layout similarity own vs rival (r37 detector); decision steps only.
    pub layout_sim: Option<f64>,
}

/// The seat's own market orders applied at a step, summed per product over the first 10 orders.
#[derive(Clone, Copy, Debug, Default)]
pub struct OwnAct {
    pub sell: [f64; 9],
    pub buy: [f64; 9],
}

/// Online builder. Feed every step in order with [`push`](Builder::push).
#[derive(Clone, Debug, Default)]
pub struct Builder {
    prev: Option<StepView>,
    prev_act: OwnAct,
    /// rival flow per step, last 24 steps (flow(t) for t = s-24 .. s-1 when at step s)
    flows: std::collections::VecDeque<[f64; 9]>,
    poseq: std::collections::VecDeque<bool>,
    /// money gap at each decision step, by day
    gaps: Vec<(usize, f64)>,
    cash_eq1: Option<bool>,
    cur: Option<StepView>,
    /// v2: day-0 position equality (steps 0..23), the opponent-group signal the shield uses
    pos0: (u32, u32),
    /// v2: step of each entry in `flows` (for the rival's sell hours)
    flow_steps: std::collections::VecDeque<usize>,
}

pub const N: usize = 93;

/// Hour of the second decision on `mid` days.
pub const MID_HOUR: usize = 13;

/// Is engine step `step` a decision step (hour 1 of every day, hour 13 of the `mid` days)?
pub fn is_decision(step: usize, mid: &[usize]) -> bool {
    let h = step % TURNS_PER_DAY;
    h == 1 || (h == MID_HOUR && mid.contains(&(step / TURNS_PER_DAY)))
}

/// Decision slots in a game: 30 days + one per mid day.
pub fn n_slots(mid: &[usize]) -> usize {
    30 + mid.iter().filter(|d| **d < 30).count()
}

/// Chronological slot index of the decision at `step` (hour 1 of day d = d + mid days before d; hour 13
/// of a mid day = the slot right after that day's hour-1 slot). With no mid days, slot = day.
pub fn slot_of(step: usize, mid: &[usize]) -> usize {
    let d = step / TURNS_PER_DAY;
    let before = mid.iter().filter(|x| **x < d).count();
    d + before + usize::from(step % TURNS_PER_DAY == MID_HOUR && mid.contains(&d))
}

/// The day of each slot (for per-day masks in the learner).
pub fn slot_days(mid: &[usize]) -> Vec<usize> {
    let mut v = vec![];
    for d in 0..30 {
        v.push(d);
        if mid.contains(&d) {
            v.push(d);
        }
    }
    v
}

fn slog(x: f64) -> f32 {
    (x.signum() * x.abs().ln_1p()) as f32
}

/// Column names of [`Builder::vector`], in order.
pub fn names() -> Vec<String> {
    let mut v: Vec<String> = vec!["day".into(), "own_money".into(), "rival_money".into(), "gap".into(), "gap_d1".into(), "gap_d3".into(),
                                  "own_hands".into(), "rival_hands".into(), "own_quads".into(), "rival_quads".into(),
                                  "own_hires".into(), "rival_hires".into()];
    for who in ["own", "rival"] {
        for k in ["planted", "growing", "ripe", "weed", "empty", "pasture", "coop"] {
            v.push(format!("{who}_tiles_{k}"));
        }
        for c in CROPS {
            v.push(format!("{who}_tiles_{}", c.to_ascii_lowercase()));
        }
        for a in ANIMALS {
            v.push(format!("{who}_animals_{}", a.to_ascii_lowercase()));
        }
    }
    for p in PRODUCTS {
        v.push(format!("own_shed_{}", p.to_ascii_lowercase()));
    }
    for p in PRODUCTS {
        v.push(format!("price_{}", p.to_ascii_lowercase()));
    }
    for p in PRODUCTS {
        v.push(format!("inv_{}", p.to_ascii_lowercase()));
    }
    for s in SHOPS {
        v.push(format!("shop_{}", s.to_ascii_lowercase()));
    }
    v.push("shops_n".into());
    for p in PRODUCTS {
        v.push(format!("rival_sold_{}", p.to_ascii_lowercase()));
    }
    v.extend(["layout_sim".into(), "pos_equal_frac".into(), "cash_equal_step1".into()]);
    // v2: the shield's group signal (with cash_equal_step1) and when in the day the rival sells
    v.extend(["pos_equal_day0".into(), "rival_sell_hour".into(), "rival_sell_early".into()]);
    v
}

impl Builder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed step `v.step` (the state BEFORE the engine runs it) and the own orders applied at it.
    /// Steps must arrive in order starting at 0 (or 1); a gap resets the rolling windows.
    pub fn push(&mut self, v: StepView, act: OwnAct) {
        self.observe(v);
        self.acted(act);
    }

    /// First half of [`push`](Builder::push): the state at this step. The agent calls this
    /// BEFORE deciding (so the day's decision sees hour 1), then [`acted`](Builder::acted).
    pub fn observe(&mut self, v: StepView) {
        if let Some(p) = &self.prev {
            if v.step == p.step + 1 {
                let draw = town_draw(&p.shops, p.step);
                let mut f = [0.0; 9];
                for i in 0..9 {
                    // own fills: SELL units filled against the shed at step t (only when the quote > 1:
                    // $1 sales add no supply, matching the corpus rival_flow)
                    let fill = self.prev_act.sell[i].min(p.own_shed[i]).max(0.0);
                    let own = if p.prices[i] > 1.0 { fill } else { 0.0 };
                    let buy = if i == P_WHEAT || i == P_FERTILIZER { self.prev_act.buy[i] } else { 0.0 };
                    f[i] = v.inv[i] - p.inv[i] + draw[i] - own + buy;
                }
                self.flows.push_back(f);
                self.flow_steps.push_back(p.step);
                self.poseq.push_back(p.pos_eq);
                if p.step < TURNS_PER_DAY {
                    self.pos0.0 += 1;
                    self.pos0.1 += p.pos_eq as u32;
                }
            } else {
                self.flows.clear();
                self.flow_steps.clear();
                self.poseq.clear();
            }
            while self.flows.len() > TURNS_PER_DAY {
                self.flows.pop_front();
                self.flow_steps.pop_front();
            }
            while self.poseq.len() > TURNS_PER_DAY {
                self.poseq.pop_front();
            }
        }
        if v.step == 1 {
            self.cash_eq1 = Some(v.money[0] == v.money[1]);
        }
        if v.step % TURNS_PER_DAY == 1 {
            self.gaps.push((v.step / TURNS_PER_DAY, v.money[0] - v.money[1]));
        }
        self.prev_act = OwnAct::default();
        self.cur = Some(v.clone());
        self.prev = Some(v);
    }

    /// Second half of [`push`](Builder::push): the own orders applied at the observed step.
    pub fn acted(&mut self, act: OwnAct) {
        self.prev_act = act;
    }

    /// True when the last pushed step is a decision step (hour 1).
    pub fn at_decision(&self) -> bool {
        self.cur.as_ref().map_or(false, |v| v.step % TURNS_PER_DAY == 1)
    }

    /// Decision step with mid-day decisions: hour 1 of every day, and hour 13 of the days in `mid`.
    pub fn at_decision_in(&self, mid: &[usize]) -> bool {
        self.cur.as_ref().map_or(false, |v| is_decision(v.step, mid))
    }

    /// The observation at the last pushed step (meaningful at decision steps).
    pub fn vector(&self) -> Option<[f32; N]> {
        let v = self.cur.as_ref()?;
        let d = v.step / TURNS_PER_DAY;
        let gap = v.money[0] - v.money[1];
        let gap_at = |dd: usize| self.gaps.iter().rev().find(|&&(x, _)| x == dd).map(|&(_, g)| g);
        let mut o = [0f32; N];
        let mut k = 0;
        let mut put = |x: f32| {
            o[k] = x;
            k += 1;
        };
        // a mid-day (hour-13) decision reads as day + 0.5; hour 1 is exactly the old d / 29
        let df = d as f32 + if v.step % TURNS_PER_DAY == MID_HOUR { 0.5 } else { 0.0 };
        put(df / 29.0);
        put((v.money[0] / 1e5) as f32);
        put((v.money[1] / 1e5) as f32);
        put((gap / 1e5) as f32);
        put(if d >= 1 { gap_at(d - 1).map_or(0.0, |g| ((gap - g) / 1e4) as f32) } else { 0.0 });
        put(if d >= 3 { gap_at(d - 3).map_or(0.0, |g| ((gap - g) / 1e4) as f32) } else { 0.0 });
        put(v.hands[0] as f32 / 10.0);
        put(v.hands[1] as f32 / 10.0);
        put(v.quads[0] as f32 / 4.0);
        put(v.quads[1] as f32 / 4.0);
        put(v.hires[0] as f32 / 5.0);
        put(v.hires[1] as f32 / 5.0);
        let t = v.tiles.unwrap_or_default();
        for side in 0..2 {
            let c = &t[side];
            for x in [c.planted, c.growing, c.ripe, c.weed, c.empty, c.pasture, c.coop] {
                put(x as f32 / 64.0);
            }
            for x in c.by_crop {
                put(x as f32 / 64.0);
            }
            for x in c.animals {
                put(x as f32 / 16.0);
            }
        }
        for x in v.own_shed {
            put(slog(x));
        }
        for x in v.prices {
            put(x.max(1e-9).ln() as f32);
        }
        for x in v.inv {
            put(slog(x));
        }
        let mut mask = [0f32; 8];
        for &s in &v.shops {
            if (s as usize) < 8 {
                mask[s as usize] = 1.0;
            }
        }
        for x in mask {
            put(x);
        }
        put(v.shops.len() as f32 / 8.0);
        let mut flow = [0.0; 9];
        for f in &self.flows {
            for i in 0..9 {
                flow[i] += f[i];
            }
        }
        for x in flow {
            put(slog(x));
        }
        put(v.layout_sim.unwrap_or(0.0) as f32);
        let eq = self.poseq.iter().filter(|&&b| b).count();
        put(if self.poseq.is_empty() { 0.0 } else { eq as f32 / self.poseq.len() as f32 });
        put(self.cash_eq1.map_or(0.0, |b| b as u8 as f32));
        // v2
        put(if self.pos0.0 == 0 { 0.0 } else { self.pos0.1 as f32 / self.pos0.0 as f32 });
        let (mut units, mut hour_w, mut early) = (0.0f64, 0.0f64, 0.0f64);
        for (f, &st) in self.flows.iter().zip(&self.flow_steps) {
            let q: f64 = f.iter().map(|x| x.max(0.0)).sum();
            let h = (st % TURNS_PER_DAY) as f64;
            units += q;
            hour_w += q * h;
            if h < 12.0 {
                early += q;
            }
        }
        put(if units > 0.0 { (hour_w / units / 23.0) as f32 } else { 0.0 });
        put(if units > 0.0 { (early / units) as f32 } else { 0.0 });
        debug_assert_eq!(k, N);
        Some(o)
    }

    /// Raw rival flow summed over the last 24 steps (unscaled; the corpus `rival_sold_*`).
    pub fn rival_sold(&self) -> [f64; 9] {
        let mut flow = [0.0; 9];
        for f in &self.flows {
            for i in 0..9 {
                flow[i] += f[i];
            }
        }
        flow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots() {
        let mid = [25, 26, 27, 28, 29];
        assert_eq!(n_slots(&mid), 35);
        assert_eq!(slot_of(24 * 24 + 1, &mid), 24);
        assert_eq!(slot_of(25 * 24 + 1, &mid), 25);
        assert_eq!(slot_of(25 * 24 + 13, &mid), 26);
        assert_eq!(slot_of(26 * 24 + 1, &mid), 27);
        assert_eq!(slot_of(29 * 24 + 13, &mid), 34);
        assert_eq!(slot_of(7 * 24 + 1, &[]), 7);
        assert_eq!(slot_days(&mid).len(), 35);
        assert!(is_decision(25 * 24 + 13, &mid) && !is_decision(24 * 24 + 13, &mid));
    }

    #[test]
    fn names_match_width() {
        assert_eq!(names().len(), N);
    }

    #[test]
    fn rival_flow_from_inventory() {
        let mut b = Builder::new();
        let mut v0 = StepView { step: 1, prices: [5.0; 9], inv: [10.0; 9], own_shed: [3.0; 9], ..Default::default() };
        v0.money = [100.0, 100.0];
        let mut a0 = OwnAct::default();
        a0.sell[0] = 2.0; // we sell 2 carrots, fill 2
        b.push(v0.clone(), a0);
        let mut v1 = v0.clone();
        v1.step = 2;
        v1.inv[0] = 15.0; // +5 on the market: 2 ours -> rival sold 3
        b.push(v1, OwnAct::default());
        assert_eq!(b.rival_sold()[0], 3.0);
        assert!(b.vector().is_some());
    }
}
