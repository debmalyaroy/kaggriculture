//! Episode model: a compact per-step record built from a parsed replay, and the per-day
//! state / behaviour rows computed from it.
//!
//! Alignment (Kaggle replay convention): `steps[t].observation` is the state BEFORE engine
//! step t runs, and the action the engine applies at step t is `steps[t + 1][seat].action`.
//! Here `acts[t]` is that applied action, so `recs[t]` + `acts[t]` -> `recs[t + 1]`.
//!
//! Everything a seat could know at play time is computed from its own private data plus
//! the shared state; the rival's private data (shed, inventories) is never used for the
//! seat's rows.
use crate::action::{parse_action, py_canonical, truthy, OrderKind, ParsedAction};
use crate::consts::*;
use crate::obs::{Replay, SlimDoc};
use crate::row::Row;
use crate::tiles::{parse_tiles, similarity, tile_stats, Tile, TileStats};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Per-step compact state (both seats, public + each seat's own shed).
#[derive(Clone, Debug, Default)]
pub struct StepRec {
    pub money: [f64; 2],
    pub hires: [f64; 2],
    pub n_hands: [u16; 2],
    pub n_quads: [u8; 2],
    /// `farms[0].unlocked_quadrants == farms[1].unlocked_quadrants` (list equality).
    pub quads_eq: bool,
    /// Farmer position and hands list identical between the two farms.
    pub pos_eq: bool,
    pub prices: [f64; 9],
    pub inv: [f64; 9],
    /// Unlocked shop instances in unlock order (SHOPS index; 255 = unknown name).
    pub shops: Vec<u8>,
    /// Each seat's own shed (from its private observation), PRODUCTS order.
    pub shed: [Option<[f64; 9]>; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Episode {
    pub n: usize,
    pub recs: Vec<StepRec>,
    /// `acts[t][seat]` = action applied at engine step t (= steps[t+1][seat].action).
    pub acts: Vec<[ParsedAction; 2]>,
    /// Units sold at step t per product: SELL orders (first MAX_MARKET_ORDERS) filled
    /// against the seat's shed at step t, in queue order.
    pub fills: Vec<[[f64; 9]; 2]>,
    /// BUY_PRODUCT units ordered at step t (valid WHEAT/FERTILIZER orders).
    pub buys: Vec<[[f64; 9]; 2]>,
    /// Decoded tile grids at the steps that need them (daily snapshots, day 15).
    pub tiles: Vec<Option<[Vec<Tile>; 2]>>,
    /// sha256 stream-hash cuts per seat at HASH_CUTS (first 16 hex), None past the end.
    pub stream_hash: [[Option<String>; 4]; 2],
    /// Full-stream sha256 hex per seat (mirror detection).
    pub full_hash: [String; 2],
    /// Shop names of the final town (unlock order).
    pub final_shops: Vec<String>,
}

/// Steps whose tiles are decoded: the daily snapshot (hour 1). Slim records keep tiles only
/// there (and at the last step), so day-15 layout similarity uses step 361 (LAYOUT_SIM_STEP).
pub fn needs_tiles(t: usize) -> bool {
    t % TURNS_PER_DAY == 1
}

/// Step at which the episode-level layout similarity is measured (day 15, hour 1).
pub const LAYOUT_SIM_STEP: usize = 361;

/// The daily snapshot step for day d (hour 1: hour 0 has no hands yet).
pub fn snapshot_step(d: usize) -> usize {
    TURNS_PER_DAY * d + 1
}

fn hex16(h: &Sha256) -> String {
    let d = h.clone().finalize();
    d.iter().take(8).map(|b| format!("{:02x}", b)).collect()
}

impl Episode {
    /// Build from a slim record. Same semantics as [`Episode::build`]: shops carry forward
    /// between `tw` entries; `a` at step t is the action that produced state t.
    pub fn build_slim(doc: &SlimDoc) -> Result<Episode, String> {
        let n = doc.steps.len();
        if n < 2 {
            return Err("fewer than 2 steps".into());
        }
        let mut ep = Episode { n, ..Default::default() };
        ep.recs.reserve(n);
        ep.tiles.reserve(n);
        let mut shops_names: Vec<String> = Vec::new();
        for (t, st) in doc.steps.iter().enumerate() {
            if st.f.len() < 2 {
                return Err(format!("no farms at step {t}"));
            }
            let (f0, f1) = (&st.f[0], &st.f[1]);
            if let Some(tw) = &st.tw {
                shops_names = tw.clone();
            }
            let mut r = StepRec {
                money: [f0.money, f1.money],
                hires: [f0.hires_today, f1.hires_today],
                n_hands: [f0.hands.as_ref().map_or(0, |h| h.len()) as u16, f1.hands.as_ref().map_or(0, |h| h.len()) as u16],
                n_quads: [
                    f0.unlocked_quadrants.as_ref().map_or(0, |q| q.len()) as u8,
                    f1.unlocked_quadrants.as_ref().map_or(0, |q| q.len()) as u8,
                ],
                quads_eq: f0.unlocked_quadrants == f1.unlocked_quadrants,
                pos_eq: f0.farmer == f1.farmer && f0.hands == f1.hands,
                ..Default::default()
            };
            if let Some(m) = &st.m {
                r.prices = m.prices.arr();
                r.inv = m.inventory.arr();
            }
            r.shops = shops_names.iter().map(|s| index_of(&SHOPS, s).map(|i| i as u8).unwrap_or(255)).collect();
            for seat in 0..2 {
                r.shed[seat] = st.p.get(seat).and_then(|p| p.as_ref()).and_then(|p| p.shed.as_ref()).map(|s| s.arr());
            }
            let tiles = if needs_tiles(t) {
                match st.tl.as_ref().map(|v| (v.first().copied().flatten(), v.get(1).copied().flatten())) {
                    Some((Some(a), Some(b))) => Some([parse_tiles(a), parse_tiles(b)]),
                    _ => None,
                }
            } else {
                None
            };
            ep.recs.push(r);
            ep.tiles.push(tiles);
        }
        ep.final_shops = shops_names;
        let mut hashers = [Sha256::new(), Sha256::new()];
        let mut canon = String::with_capacity(512);
        ep.acts.reserve(n - 1);
        for t in 1..n {
            let mut pair: [ParsedAction; 2] = Default::default();
            for (seat, slot) in pair.iter_mut().enumerate() {
                let v: Value = match doc.steps[t].a.get(seat).copied().flatten() {
                    Some(raw) => serde_json::from_str(raw.get()).unwrap_or(Value::Null),
                    None => Value::Null,
                };
                *slot = parse_action(&v);
                canon.clear();
                if truthy(&v) {
                    py_canonical(&v, &mut canon);
                } else {
                    canon.push_str("{}");
                }
                hashers[seat].update(canon.as_bytes());
                hashers[seat].update(b"\0");
                if let Some(k) = HASH_CUTS.iter().position(|&c| c == t) {
                    ep.stream_hash[seat][k] = Some(hex16(&hashers[seat]));
                }
            }
            ep.acts.push(pair);
        }
        for seat in 0..2 {
            let d = hashers[seat].clone().finalize();
            ep.full_hash[seat] = d.iter().map(|b| format!("{:02x}", b)).collect();
        }
        ep.compute_fills();
        Ok(ep)
    }

    fn compute_fills(&mut self) {
        let n = self.n;
        self.fills = vec![[[0.0; 9]; 2]; n - 1];
        self.buys = vec![[[0.0; 9]; 2]; n - 1];
        for t in 0..n - 1 {
            for seat in 0..2 {
                let mut shed = self.recs[t].shed[seat].unwrap_or([0.0; 9]);
                for o in self.acts[t][seat].orders.iter().take(MAX_MARKET_ORDERS) {
                    match (o.kind, o.item, o.qty) {
                        (OrderKind::Sell, Some(i), Some(q)) => {
                            let got = (q as f64).min(shed[i]).max(0.0);
                            shed[i] -= got;
                            self.fills[t][seat][i] += got;
                        }
                        (OrderKind::BuyProduct, Some(i), Some(q)) if i == P_WHEAT || i == P_FERTILIZER => {
                            self.buys[t][seat][i] += q as f64;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    pub fn build(rep: &Replay) -> Result<Episode, String> {
        let n = rep.steps.len();
        if n < 2 {
            return Err("fewer than 2 steps".into());
        }
        let mut ep = Episode { n, ..Default::default() };
        ep.recs.reserve(n);
        ep.tiles.reserve(n);
        for t in 0..n {
            let sh = rep.shared(t).ok_or_else(|| format!("no shared observation at step {t}"))?;
            let farms = sh.farms.as_ref().unwrap();
            let (f0, f1) = (&farms[0], &farms[1]);
            let mut r = StepRec {
                money: [f0.money, f1.money],
                hires: [f0.hires_today, f1.hires_today],
                n_hands: [
                    f0.hands.as_ref().map_or(0, |h| h.len()) as u16,
                    f1.hands.as_ref().map_or(0, |h| h.len()) as u16,
                ],
                n_quads: [
                    f0.unlocked_quadrants.as_ref().map_or(0, |q| q.len()) as u8,
                    f1.unlocked_quadrants.as_ref().map_or(0, |q| q.len()) as u8,
                ],
                quads_eq: f0.unlocked_quadrants == f1.unlocked_quadrants,
                pos_eq: f0.farmer == f1.farmer && f0.hands == f1.hands,
                ..Default::default()
            };
            if let Some(m) = &sh.market {
                r.prices = m.prices.arr();
                r.inv = m.inventory.arr();
            }
            if let Some(town) = &sh.town {
                r.shops = town
                    .unlocked_shops
                    .iter()
                    .map(|s| index_of(&SHOPS, s).map(|i| i as u8).unwrap_or(255))
                    .collect();
                if t == n - 1 {
                    ep.final_shops = town.unlocked_shops.clone();
                }
            }
            for seat in 0..2 {
                r.shed[seat] = rep
                    .obs(t, seat)
                    .and_then(|o| o.private.as_ref())
                    .and_then(|p| p.shed.as_ref())
                    .map(|s| s.arr());
            }
            let tiles = if needs_tiles(t) {
                match (f0.tiles, f1.tiles) {
                    (Some(a), Some(b)) => Some([parse_tiles(a), parse_tiles(b)]),
                    _ => None,
                }
            } else {
                None
            };
            ep.recs.push(r);
            ep.tiles.push(tiles);
        }

        // actions: steps[t][seat].action for t >= 1 (steps[0] is the engine's opening pass)
        let mut hashers = [Sha256::new(), Sha256::new()];
        let mut canon = String::with_capacity(512);
        ep.acts.reserve(n - 1);
        for t in 1..n {
            let mut pair: [ParsedAction; 2] = Default::default();
            for (seat, slot) in pair.iter_mut().enumerate() {
                let v: Value = match rep.cell(t, seat).and_then(|c| c.action) {
                    Some(raw) => serde_json::from_str(raw.get()).unwrap_or(Value::Null),
                    None => Value::Null,
                };
                *slot = parse_action(&v);
                canon.clear();
                if truthy(&v) {
                    py_canonical(&v, &mut canon);
                } else {
                    canon.push_str("{}");
                }
                hashers[seat].update(canon.as_bytes());
                hashers[seat].update(b"\0");
                if let Some(k) = HASH_CUTS.iter().position(|&c| c == t) {
                    ep.stream_hash[seat][k] = Some(hex16(&hashers[seat]));
                }
            }
            ep.acts.push(pair);
        }
        for seat in 0..2 {
            let d = hashers[seat].clone().finalize();
            ep.full_hash[seat] = d.iter().map(|b| format!("{:02x}", b)).collect();
        }

        ep.compute_fills();
        Ok(ep)
    }

    /// Units the town removes from the market at step t (shops every 4, centre every 24),
    /// using the shops unlocked in `recs[t]` (engine `_town_consume`).
    pub fn town_draw(&self, t: usize) -> [f64; 9] {
        town_draw(&self.recs[t].shops, t)
    }

    /// Rival's net units sold at step t as seat `seat` can infer it from public data:
    /// inv[t+1] - inv[t] + town_draw(t) - own fills (when the quote is > 1; $1 sales add no
    /// supply) + own BUY_PRODUCT units. Negative = rival net buying.
    pub fn rival_flow(&self, seat: usize, t: usize) -> [f64; 9] {
        let (a, b) = (&self.recs[t], &self.recs[t + 1]);
        let draw = self.town_draw(t);
        let mut out = [0.0; 9];
        for i in 0..9 {
            let own = if a.prices[i] > 1.0 { self.fills[t][seat][i] } else { 0.0 };
            out[i] = b.inv[i] - a.inv[i] + draw[i] - own + self.buys[t][seat][i];
        }
        out
    }

    pub fn tiles_at(&self, t: usize) -> Option<&[Vec<Tile>; 2]> {
        self.tiles.get(t).and_then(|x| x.as_ref())
    }

    /// Layout similarity at step t from `seat`'s view (None when tiles weren't decoded).
    pub fn layout_sim(&self, t: usize, seat: usize) -> Option<f64> {
        let tl = self.tiles_at(t)?;
        Some(similarity(&tl[seat], &tl[1 - seat], self.recs[t].quads_eq))
    }

    pub fn cash_equal_step1(&self) -> bool {
        self.n > 1 && self.recs[1].money[0] == self.recs[1].money[1]
    }

    /// Both seats submitted identical action streams through the last turn.
    pub fn is_mirror(&self) -> bool {
        self.full_hash[0] == self.full_hash[1]
    }

    /// Number of days with a snapshot (step 24d+1 < n).
    pub fn n_state_days(&self) -> usize {
        (0..30).take_while(|&d| snapshot_step(d) < self.n).count()
    }

    /// Number of days with at least one applied action.
    pub fn n_behaviour_days(&self) -> usize {
        (0..30).take_while(|&d| TURNS_PER_DAY * d < self.n - 1).count()
    }

    fn gap(&self, seat: usize, t: usize) -> f64 {
        self.recs[t].money[seat] - self.recs[t].money[1 - seat]
    }

    /// Per-day state for `seat` at step 24d+1 (hour 1). None when the episode is shorter.
    pub fn daily_state(&self, seat: usize, d: usize) -> Option<Row> {
        let s = snapshot_step(d);
        if s >= self.n {
            return None;
        }
        let riv = 1 - seat;
        let r = &self.recs[s];
        let mut row = Row::with_capacity(160);
        row.i("step", "engine step of the snapshot = 24*day + 1 (hour 1)", Some(s as i64));
        row.f("own_money", "own money at the snapshot", Some(r.money[seat]));
        row.f("rival_money", "rival money at the snapshot", Some(r.money[riv]));
        let g = self.gap(seat, s);
        row.f("gap", "own_money - rival_money", Some(g));
        row.f("gap_d1", "gap - gap one day earlier (null on day 0)",
              (d >= 1).then(|| g - self.gap(seat, s - 24)));
        row.f("gap_d3", "gap - gap three days earlier (null before day 3)",
              (d >= 3).then(|| g - self.gap(seat, s - 72)));
        row.i("own_hires_today", "own farm hires_today at the snapshot", Some(r.hires[seat] as i64));
        row.i("rival_hires_today", "rival farm hires_today at the snapshot", Some(r.hires[riv] as i64));
        row.i("own_hands", "own hired hands alive (len farms[seat].hands)", Some(r.n_hands[seat] as i64));
        row.i("rival_hands", "rival hired hands alive", Some(r.n_hands[riv] as i64));
        row.i("own_quadrants", "own unlocked quadrants (1..4)", Some(r.n_quads[seat] as i64));
        row.i("rival_quadrants", "rival unlocked quadrants (1..4)", Some(r.n_quads[riv] as i64));
        let tl = self.tiles_at(s);
        for (side, who) in [(seat, "own"), (riv, "rival")] {
            let st: Option<TileStats> = tl.map(|g| tile_stats(&g[side], d as i64));
            for (c, crop) in CROPS.iter().enumerate() {
                row.i(format!("{who}_tiles_{}", lower(crop)), "PLANT tiles of this crop",
                      st.map(|x| x.by_crop[c] as i64));
            }
            row.i(format!("{who}_tiles_planted"), "PLANT tiles (growing + ripe)",
                  st.map(|x| x.planted as i64));
            row.i(format!("{who}_tiles_growing"), "PLANT tiles not yet harvestable",
                  st.map(|x| x.growing as i64));
            row.i(format!("{who}_tiles_ripe"),
                  "PLANT tiles harvestable now (yield_units > 0 and day - planted_day >= first_yield_day)",
                  st.map(|x| x.ripe as i64));
            row.i(format!("{who}_tiles_weed"), "WEED tiles", st.map(|x| x.weed as i64));
            row.i(format!("{who}_tiles_empty"), "unlocked empty tiles (null)", st.map(|x| x.empty as i64));
            row.i(format!("{who}_tiles_pasture"), "PASTURE tiles (with or without animal)",
                  st.map(|x| x.pasture as i64));
            row.i(format!("{who}_tiles_coop"), "COOP tiles (with or without animal)",
                  st.map(|x| x.coop as i64));
            for (a, an) in ANIMALS.iter().enumerate() {
                row.i(format!("{who}_animals_{}", lower(an)), "animals of this type on the farm",
                      st.map(|x| x.animals[a] as i64));
            }
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("own_shed_{}", lower(p)), "own shed stock (private observation)",
                  r.shed[seat].map(|x| x[i]));
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("price_{}", lower(p)), "market quote at the snapshot", Some(r.prices[i]));
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("inv_{}", lower(p)), "market inventory at the snapshot", Some(r.inv[i]));
        }
        let mask = r.shops.iter().filter(|&&x| x < 8).fold(0i64, |m, &x| m | (1 << x));
        row.i("shops_mask", "bitmask of unlocked shops (bit i = SHOPS[i], alphabetical)", Some(mask));
        row.i("shops_n", "unlocked shop instances (duplicates count)", Some(r.shops.len() as i64));
        let lo = s.saturating_sub(TURNS_PER_DAY);
        let mut flow = [0.0; 9];
        for t in lo..s {
            let f = self.rival_flow(seat, t);
            for i in 0..9 {
                flow[i] += f[i];
            }
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("rival_sold_{}", lower(p)),
                  "rival net units sold over steps [s-24, s-1] recovered from public market inventory deltas (+town draw - own fills + own buys); negative = net buying",
                  Some(flow[i]));
        }
        row.f("layout_sim", "tile-layout similarity own vs rival at the snapshot (r37 detector)",
              self.layout_sim(s, seat));
        let eq = (lo..s).filter(|&u| self.recs[u].pos_eq).count();
        row.f("pos_equal_frac",
              "share of observations in steps [s-24, s-1] where farmer and hands lists are identical across farms",
              Some(eq as f64 / (s - lo) as f64));
        row.i("cash_equal_step1", "1 if both farms had identical money at step 1", Some(self.cash_equal_step1() as i64));
        Some(row)
    }

    /// What `seat` did during day d (actions applied at steps 24d .. 24d+23).
    pub fn daily_behaviour(&self, seat: usize, d: usize) -> Option<Row> {
        let lo = TURNS_PER_DAY * d;
        if lo >= self.n - 1 {
            return None;
        }
        let hi = (lo + TURNS_PER_DAY).min(self.n - 1); // exclusive, over acts
        let mut units = [0.0; 9];
        let mut rev = [0.0; 9];
        let mut win = [[0.0; 5]; 9];
        let mut strength_num = [0.0; 9];
        let mut strength_den = [0.0; 9];
        let mut buy_p = [0.0; 9];
        let mut buy_seed = [0.0; 5];
        let mut buy_animal = [0.0; 3];
        let mut plants = [0i64; 5];
        let (mut n_orders, mut n_sell, mut sell_slot_sum, mut n_hire, mut n_land) = (0i64, 0i64, 0f64, 0i64, 0i64);
        for t in lo..hi {
            let hour = t % TURNS_PER_DAY;
            let w = HOUR_WINDOWS.iter().position(|&(a, b, _)| hour >= a && hour <= b).unwrap_or(0);
            let r = &self.recs[t];
            let f = &self.fills[t][seat];
            for i in 0..9 {
                if f[i] > 0.0 {
                    units[i] += f[i];
                    rev[i] += f[i] * r.prices[i];
                    win[i][w] += f[i];
                    let a = t.saturating_sub(STRENGTH_WINDOW);
                    let avg = if t == 0 {
                        r.prices[i]
                    } else {
                        (a..t).map(|u| self.recs[u].prices[i]).sum::<f64>() / (t - a) as f64
                    };
                    if avg > 0.0 {
                        strength_num[i] += f[i] * r.prices[i] / avg;
                        strength_den[i] += f[i];
                    }
                }
            }
            let act = &self.acts[t][seat];
            n_orders += act.orders.len() as i64;
            for (slot, o) in act.orders.iter().enumerate() {
                if slot >= MAX_MARKET_ORDERS {
                    break;
                }
                match (o.kind, o.item, o.qty) {
                    (OrderKind::Sell, _, _) => {
                        n_sell += 1;
                        sell_slot_sum += slot as f64;
                    }
                    (OrderKind::BuyProduct, Some(i), Some(q)) => buy_p[i] += q as f64,
                    (OrderKind::BuySeed, Some(i), Some(q)) => buy_seed[i] += q as f64,
                    (OrderKind::BuyAnimal, Some(i), Some(q)) => buy_animal[i] += q as f64,
                    (OrderKind::Hire, _, _) => n_hire += 1,
                    (OrderKind::BuyLand, _, _) => n_land += 1,
                    _ => {}
                }
            }
            for c in 0..5 {
                plants[c] += act.plants[c] as i64;
            }
        }
        let mut row = Row::with_capacity(120);
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("sold_{}", lower(p)),
                  "units sold: SELL orders filled against the own shed at the order's step, queue order",
                  Some(units[i]));
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("revenue_{}", lower(p)), "sold units x the step's quote", Some(rev[i]));
        }
        row.f("sold_total", "sum of sold_*", Some(units.iter().sum()));
        row.f("revenue_total", "sum of revenue_*", Some(rev.iter().sum()));
        for (i, p) in PRODUCTS.iter().enumerate() {
            for (w, &(_, _, wn)) in HOUR_WINDOWS.iter().enumerate() {
                row.f(format!("sold_{}_{wn}", lower(p)), "units sold in this hour window of the day",
                      Some(win[i][w]));
            }
        }
        for (i, p) in PRODUCTS.iter().enumerate() {
            row.f(format!("strength_{}", lower(p)),
                  "unit-weighted mean of quote / mean quote over the previous 12 steps (1 at step 0); null if nothing sold",
                  (strength_den[i] > 0.0).then(|| strength_num[i] / strength_den[i]));
        }
        row.f("buy_product_wheat", "BUY_PRODUCT WHEAT units ordered (valid orders in the first 10)", Some(buy_p[P_WHEAT]));
        row.f("buy_product_fertilizer", "BUY_PRODUCT FERTILIZER units ordered", Some(buy_p[P_FERTILIZER]));
        for (c, crop) in CROPS.iter().enumerate() {
            row.f(format!("buy_seed_{}", lower(crop)), "BUY_SEED units ordered", Some(buy_seed[c]));
        }
        for (a, an) in ANIMALS.iter().enumerate() {
            row.f(format!("buy_animal_{}", lower(an)), "BUY_ANIMAL units ordered", Some(buy_animal[a]));
        }
        let hires = (lo..(lo + TURNS_PER_DAY).min(self.n))
            .map(|u| self.recs[u].hires[seat])
            .fold(0.0, f64::max);
        row.i("hires", "hires made this day from farm state: max hires_today over the day's observations (GM total_hires convention)", Some(hires as i64));
        row.i("hire_orders", "HIRE orders submitted (first 10 per step)", Some(n_hire));
        let q_end = self.recs[(lo + TURNS_PER_DAY).min(self.n - 1)].n_quads[seat] as i64;
        row.i("buy_land", "quadrants gained over the day (state delta, next day's hour 0 - this day's hour 0)",
              Some(q_end - self.recs[lo].n_quads[seat] as i64));
        row.i("buy_land_orders", "BUY_LAND orders submitted (first 10 per step)", Some(n_land));
        for (c, crop) in CROPS.iter().enumerate() {
            row.i(format!("plants_{}", lower(crop)), "PLANT unit actions (farmer + hands) for this crop", Some(plants[c]));
        }
        row.i("market_orders", "market orders submitted (raw list length, all steps of the day)", Some(n_orders));
        row.i("sell_orders", "SELL orders within the first 10 slots", Some(n_sell));
        row.f("sell_slot_mean", "mean 0-based queue slot of SELL orders (null if none)",
              (n_sell > 0).then(|| sell_slot_sum / n_sell as f64));
        Some(row)
    }
}

/// Engine `_town_consume` at step t for the given shop instances.
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
