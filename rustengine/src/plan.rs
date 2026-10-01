//! The ACTION ABSTRACTION for the Track-P search: a day plan.
//!
//! Track P is CLOSED LOOP by operator order -- every decision here is derived
//! from the live state that is passed in. Nothing in this file stores or
//! replays an action sequence across a game; `DayPlan` is rebuilt from the real
//! board every dawn and is discarded at dusk.
//!
//! The raw per-turn action space (a farmer plus up to 10 hands, each choosing
//! one of ~15 ops against 100 tiles, plus 10 ordered market orders) is larger
//! than 1e20. We never search it. We search `DayKnobs` -- ~26 integers naming
//! the day's ECONOMY (labour, purchases, plantings, sell policy) -- and expand
//! it deterministically into 24 turns with `plan_day` + `execute_turn`.
//!
//! Two properties make this the right abstraction and both are load-bearing:
//!   * the rollout expands a candidate with the SAME code that plays the real
//!     turn, so what we search is what we play (the two-pass compiler's
//!     model-drift bug is structurally impossible here);
//!   * a searched PARAMETER vector re-seats on a board that turned out
//!     different, which a searched action sequence cannot.
//!
//! `DayKnobs::skeleton` is the measured field economy -- the one every cohort
//! from rank 1 to rank 400 runs -- so the search's worst case is the field and
//! every accepted move is a measured improvement on it.

use crate::engine::{PlayerAction, UnitAction};
use crate::market;
use crate::rules;
use crate::state::{
    Cell, Farm, State, ANIMAL_NAMES, BOARD, CROP_NAMES, PRODUCTS, SHED_CAP,
    TURNS_PER_DAY,
};

pub const TPD: i64 = TURNS_PER_DAY;
pub const LAST_DAY: i64 = 29;
pub const N_PROD: usize = 9;
pub const N_CROP: usize = 5;

/// Shed-access tiles (the four inner corners), NWSE order.
pub const SHED: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];

/// Product index in `PRODUCTS` order.
pub const P_WHEAT: usize = 0;
pub const P_CARROT: usize = 1;
pub const P_TOMATO: usize = 2;
pub const P_STRAW: usize = 3;
pub const P_MELON: usize = 4;
pub const P_EGG: usize = 5;
pub const P_MILK: usize = 6;
pub const P_WOOL: usize = 7;
pub const P_FERT: usize = 8;

/// Crop index in `CROP_NAMES` / `rules::CROPS` order.
pub const C_WHEAT: usize = 0;
pub const C_CARROT: usize = 1;
pub const C_TOMATO: usize = 2;
pub const C_STRAW: usize = 3;
pub const C_MELON: usize = 4;

/// `DayKnobs::buy` order. NOT `ANIMAL_NAMES` order -- cows and sheep are the
/// field's herd and geese are the option, so they sort by how often they move.
pub const BUY_ORDER: [&str; 3] = ["COW", "SHEEP", "GOOSE"];

/// Liquidation order: the price-impact-sensitive products first, so that when
/// the 10-order queue binds the slots go to the units worth the most.
const SELL_ORDER: [usize; 9] = [
    P_MELON, P_STRAW, P_WOOL, P_MILK, P_FERT, P_EGG, P_TOMATO, P_CARROT,
    P_WHEAT,
];

pub fn base_price(p: usize) -> f64 {
    market::PARAMS[p].base
}

fn crop_to_product(c: usize) -> usize {
    // CROP_NAMES and PRODUCTS agree on their first five entries.
    c
}

// ------------------------------------------------------------------- knobs --

/// The searched day plan. Every field is a decision for ONE day.
#[derive(Clone, Debug, PartialEq)]
pub struct DayKnobs {
    /// Hands to open today (0..=10). Hires sit in the hour-0 market batch.
    pub hire: i8,
    /// Buy the next quadrant today if it is affordable.
    pub buy_land: bool,
    /// Animal purchases today, in `BUY_ORDER` (COW, SHEEP, GOOSE).
    pub buy: [i8; 3],
    /// New tiles planted today, in `CROP_NAMES` order.
    pub plant: [i8; N_CROP],
    /// Units offered today per product; -1 = uncapped.
    pub sell_cap: [i16; N_PROD],
    /// Hold the product while `price < sell_floor/16 * base`.
    pub sell_floor: [u8; N_PROD],
    /// Fertilise strawberry on its production eves.
    pub fert_straw: bool,
    /// Days of feed wheat held back in the shed.
    pub feed_buffer: i8,
    /// Carried units at which a unit detours to the shed mid-tour.
    pub drop_at: i8,
    /// Animal `yield_units` worth walking to.
    pub harvest_at: i8,
}

impl DayKnobs {
    /// The measured field economy for `day`, read against the live board.
    ///
    /// Sources: the top-10 median action skeleton (2 land, ~9 cows, ~5 sheep,
    /// 187 wheat seeds, ~40 strawberry, 12 melon, 17 pastures, 243 plants) and
    /// the closed leak list (feed wheat 142 not 414, fertiliser 62 not 248).
    pub fn skeleton(day: i64, st: &State, me: usize) -> DayKnobs {
        let farm = &st.farms[me];
        let c = census(farm);
        let owned_extra = farm.unlocked_quadrants.len() - 1;

        // Herd: cumulative targets by day, one day early to absorb buy->place.
        let cow_t = sched(&[(0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 6),
                            (10, 7), (13, 8), (16, 9)], day);
        let sheep_t = sched(&[(0, 1), (6, 2), (9, 3), (12, 4), (15, 5)], day);
        let have_cow = c.animals_by_kind[1] + st.private[me].shed.get("COW");
        let have_sheep = c.animals_by_kind[2] + st.private[me].shed.get("SHEEP");
        let mut buy = [0i8; 3];
        if day <= 20 {
            buy[0] = (cow_t - have_cow).clamp(0, 3) as i8;
            buy[1] = (sheep_t - have_sheep).clamp(0, 3) as i8;
        }

        // Standing-tile targets. Counting what STANDS (not a running total)
        // keeps the skeleton a pure function of the observation.
        let mut plant = [0i8; N_CROP];
        if day <= 25 {
            if (4..=12).contains(&day) {
                plant[C_MELON] = (12 - c.crop_tiles[C_MELON]).clamp(0, 2) as i8;
            }
            if (5..=14).contains(&day) {
                plant[C_STRAW] = (38 - c.crop_tiles[C_STRAW]).clamp(0, 4) as i8;
            }
            plant[C_WHEAT] = 14;
        }

        let mut sell_cap = [-1i16; N_PROD];
        // WHEAT and EGG are log-priced (dump-proof); everything else walks the
        // price down, so it is metered. These are the search's STARTING caps,
        // not rules -- the forward model prices the walk-down exactly.
        sell_cap[P_MELON] = 24;
        sell_cap[P_WOOL] = 40;
        sell_cap[P_STRAW] = 60;
        sell_cap[P_MILK] = 40;
        sell_cap[P_CARROT] = 60;
        sell_cap[P_TOMATO] = 60;
        sell_cap[P_FERT] = 60;

        // The CURRENT top-30 hire ramp (360 engine-1.32.7 wins of the 30
        // teams rated 2806-2978, mined 2026-09-03). Leaner than the old
        // 5/7/9 ramp on days 1-7: an idle hand still costs its fibonacci
        // fee, and days 0-7 are a one-quadrant farm with ~21 usable tiles.
        // Kept in step with `DayKnobs`'s twin, src/trackp/build_econ_agent.py's
        // `skeleton_genome()["hires"]`, and rustengine/src/policy.rs SKELETON.
        const HIRE_RAMP: [i8; 30] = [
            5, 4, 4, 5, 4, 5, 8, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        ];
        let hire: i8 = HIRE_RAMP[day.clamp(0, 29) as usize];

        DayKnobs {
            hire,
            // Both quadrants AS EARLY AS CASH ALLOWS. The old `day >= 4`
            // schedule was not the binding constraint -- the purse was -- and
            // the farm sat on NW's 21 usable tiles into day 9. Measured on the
            // official engine: +29% median own bank over 80 gauntlet cells
            // (docs/history/trackp-base-economy-2026-09-03.md).
            buy_land: owned_extra < 2 && day >= if owned_extra == 0 { 0 } else { 5 },
            buy,
            plant,
            sell_cap,
            sell_floor: [0; N_PROD],
            fert_straw: true,
            feed_buffer: 6,
            drop_at: 8,
            harvest_at: 1,
        }
    }
}

fn sched(rows: &[(i64, i64)], day: i64) -> i64 {
    let mut n = 0;
    for (d, v) in rows {
        if day >= *d {
            n = *v;
        }
    }
    n
}

// ------------------------------------------------------------------ census --

#[derive(Default, Clone, Debug)]
pub struct Census {
    pub animals: Vec<((i64, i64), usize)>, // tile, ANIMAL_NAMES index
    pub plants: Vec<((i64, i64), usize)>,  // tile, CROP index
    pub weeds: Vec<(i64, i64)>,
    pub empty: Vec<(i64, i64)>,
    pub structs: Vec<((i64, i64), bool)>, // tile, is_pasture
    pub crop_tiles: [i64; N_CROP],
    pub animals_by_kind: [i64; 3], // ANIMAL_NAMES order: GOOSE, COW, SHEEP
}

pub fn census(farm: &Farm) -> Census {
    let mut c = Census::default();
    for y in 0..BOARD {
        for x in 0..BOARD {
            match &farm.tiles[y as usize][x as usize] {
                Cell::Locked => {}
                Cell::Empty => c.empty.push((x, y)),
                Cell::Weed => c.weeds.push((x, y)),
                Cell::Plant { crop, .. } => {
                    let ci = CROP_NAMES.iter().position(|n| n == crop)
                        .unwrap_or(0);
                    c.crop_tiles[ci] += 1;
                    c.plants.push(((x, y), ci));
                }
                Cell::Structure { kind, animal } => match animal {
                    Some(a) => {
                        let ai = ANIMAL_NAMES.iter()
                            .position(|n| *n == a.animal).unwrap_or(0);
                        c.animals_by_kind[ai] += 1;
                        c.animals.push(((x, y), ai));
                    }
                    None => c.structs.push(((x, y), kind == "PASTURE")),
                },
            }
        }
    }
    c
}

// --------------------------------------------------------------------- ops --

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpKind {
    Plant,
    Water,
    Harvest,
    Fertilize,
    Dig,
    BuildCoop,
    BuildPasture,
    Feed,
    Care,
    CollectFert,
    Place,
    Pickup,
    Drop,
}

#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub k: OpKind,
    /// Crop index for Plant, product index for Pickup, animal index for Place.
    pub arg: u8,
    pub n: i64,
}

impl Op {
    fn new(k: OpKind) -> Op {
        Op { k, arg: 0, n: 1 }
    }
    fn with(k: OpKind, arg: usize) -> Op {
        Op { k, arg: arg as u8, n: 1 }
    }
    fn to_action(self) -> UnitAction {
        let (op, item, n, has_n) = match self.k {
            OpKind::Plant => ("PLANT", CROP_NAMES[self.arg as usize], 1, false),
            OpKind::Water => ("WATER", "", 1, false),
            OpKind::Harvest => ("HARVEST", "", 1, false),
            OpKind::Fertilize => ("FERTILIZE", "", 1, false),
            OpKind::Dig => ("DIG", "", 1, false),
            OpKind::BuildCoop => ("BUILD_COOP", "", 1, false),
            OpKind::BuildPasture => ("BUILD_PASTURE", "", 1, false),
            OpKind::Feed => ("FEED", "", 1, false),
            OpKind::Care => ("CARE", "", 1, false),
            OpKind::CollectFert => ("COLLECT_FERTILIZER", "", 1, false),
            OpKind::Place => ("PLACE", ANIMAL_NAMES[self.arg as usize], 1, false),
            OpKind::Pickup => ("PICKUP", PRODUCTS_OR_ANIMAL[self.arg as usize],
                               self.n, true),
            OpKind::Drop => ("DROP", "", 1, false),
        };
        UnitAction { op: op.to_string(), item: item.to_string(), n, has_n }
    }
}

/// PICKUP namespace: the nine products then the three animals.
pub const PRODUCTS_OR_ANIMAL: [&str; 12] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
    "FERTILIZER", "GOOSE", "COW", "SHEEP",
];

/// Cargo a job needs the unit to be carrying when it arrives.
#[derive(Clone, Copy, Default, Debug)]
pub struct Cargo {
    pub wheat: i64,
    pub fert: i64,
    /// ANIMAL_NAMES order.
    pub animal: [i64; 3],
    /// CROP order; seeds are bought, not carried, but they gate the planting.
    pub seed: [i64; N_CROP],
}

impl Cargo {
    fn carried(&self) -> i64 {
        self.wheat + self.fert + self.animal.iter().sum::<i64>()
    }
}

#[derive(Clone, Debug)]
pub struct Job {
    /// `None` = the shed (a cargo pickup), `Some(t)` = a board tile.
    pub tile: Option<(i64, i64)>,
    pub ops: Vec<Op>,
    pub waits: i32,
}

#[derive(Clone, Debug)]
struct Task {
    pri: u8,
    tile: (i64, i64),
    ops: Vec<Op>,
    cargo: Cargo,
}

// ---------------------------------------------------------------- day plan --

#[derive(Clone, Debug)]
pub struct DayPlan {
    pub day: i64,
    pub knobs: DayKnobs,
    /// Market orders, already chunked into batches of <= 10 (the hard cap).
    pub spill: Vec<Vec<Vec<String>>>,
    pub jobs: Vec<Vec<Job>>,
    pub pool: Vec<(( i64, i64), Vec<Op>, Cargo)>,
    pub keep: [i64; N_PROD],
    pub sold_today: [i64; N_PROD],
    pub is_final: bool,
    /// Diagnostics for the harness: what the dawn planner intended.
    pub n_hire: i64,
    pub n_tasks: usize,
}

fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn shed_dist(p: (i64, i64)) -> i64 {
    SHED.iter().map(|s| dist(p, *s)).min().unwrap_or(0)
}

fn nearest_shed(p: (i64, i64)) -> (i64, i64) {
    *SHED.iter().min_by_key(|s| dist(p, **s)).unwrap()
}

fn move_toward(p: (i64, i64), t: (i64, i64)) -> UnitAction {
    let op = if p.0 < t.0 {
        "EAST"
    } else if p.0 > t.0 {
        "WEST"
    } else if p.1 < t.1 {
        "SOUTH"
    } else {
        "NORTH"
    };
    UnitAction { op: op.to_string(), item: String::new(), n: 1, has_n: false }
}

fn pass() -> UnitAction {
    UnitAction { op: "PASS".to_string(), item: String::new(), n: 1,
                 has_n: false }
}

fn fib_sum(n: i64) -> i64 {
    let (mut a, mut b, mut s) = (1i64, 1i64, 0i64);
    for _ in 0..n {
        s += a;
        let t = a + b;
        a = b;
        b = t;
    }
    s
}

/// Sell orders from the live shed under the day's knobs.
fn sell_orders(
    k: &DayKnobs,
    day: i64,
    shed: &[i64; N_PROD],
    prices: &[f64; N_PROD],
    keep: &[i64; N_PROD],
    sold: &[i64; N_PROD],
    is_final: bool,
) -> Vec<(usize, i64)> {
    let mut out: Vec<(usize, i64)> = Vec::new();
    for &p in SELL_ORDER.iter() {
        let mut have = shed[p];
        if !is_final {
            have -= keep[p];
            if have <= 0 {
                continue;
            }
            let floor = k.sell_floor[p] as f64 / 16.0 * base_price(p);
            if prices[p] < floor {
                continue;
            }
            if k.sell_cap[p] >= 0 {
                have = have.min(k.sell_cap[p] as i64 - sold[p]);
            }
        }
        if have > 0 {
            out.push((p, have));
        }
    }
    let _ = day;
    out.sort_by(|a, b| {
        let va = a.1 as f64 * prices[a.0];
        let vb = b.1 as f64 * prices[b.0];
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn read_shed(st: &State, me: usize) -> [i64; N_PROD] {
    let mut s = [0i64; N_PROD];
    for (i, p) in PRODUCTS.iter().enumerate() {
        s[i] = st.private[me].shed.get(p);
    }
    s
}

fn read_seeds(st: &State, me: usize) -> [i64; N_CROP] {
    let mut s = [0i64; N_CROP];
    for (i, c) in CROP_NAMES.iter().enumerate() {
        s[i] = st.private[me].seeds.get(c);
    }
    s
}

fn read_prices(st: &State) -> [f64; N_PROD] {
    let mut s = [0f64; N_PROD];
    for (i, p) in PRODUCTS.iter().enumerate() {
        s[i] = st.market.prices.get(p) as f64;
    }
    s
}

/// Expand `knobs` against the REAL dawn board into a day's worth of turns.
pub fn plan_day(st: &State, me: usize, knobs: &DayKnobs) -> DayPlan {
    let day = st.step / TPD;
    let is_final = day >= LAST_DAY;
    let farm = &st.farms[me];
    let c = census(farm);
    let shed = read_shed(st, me);
    let seeds = read_seeds(st, me);
    let prices = read_prices(st);
    let money = farm.money;

    // ---- reserves ---------------------------------------------------------
    let n_animals = c.animals.len() as i64;
    let feed_today = if !is_final { n_animals } else { 0 };
    let reserve_w = if feed_today > 0 {
        feed_today * 2 + knobs.feed_buffer as i64
    } else {
        0
    };
    let mut fert_need = 0i64;
    if knobs.fert_straw {
        for ((x, y), ci) in c.plants.iter() {
            if *ci != C_STRAW {
                continue;
            }
            if let Cell::Plant { planted_day, fertilized_until_day, .. } =
                &farm.tiles[*y as usize][*x as usize]
            {
                let age = day - planted_day;
                if (age == 8 || age == 10 || age == 12)
                    && *fertilized_until_day < day
                {
                    fert_need += 1;
                }
            }
        }
    }
    let mut keep = [0i64; N_PROD];
    keep[P_WHEAT] = reserve_w;
    keep[P_FERT] = fert_need;

    // ---- market queue -----------------------------------------------------
    let mut sold = [0i64; N_PROD];
    let mut q: Vec<(u8, Vec<String>)> = Vec::new();
    let sells = sell_orders(knobs, day, &shed, &prices, &keep, &sold, is_final);
    let dawn_sells: Vec<(usize, i64)> = sells.iter().take(6).copied().collect();
    let income: f64 = dawn_sells
        .iter()
        .map(|(p, n)| *n as f64 * prices[*p] * 0.7)
        .sum();
    let mut avail = money + income;

    // feed wheat: the exact shortfall, never a standing order (the 414-vs-142
    // leak was a standing buy that outran the herd)
    let wheat_short = (reserve_w - shed[P_WHEAT]).max(0);
    if wheat_short > 0 && day > 0 {
        let cost = wheat_short as f64 * (prices[P_WHEAT] + 3.0);
        q.push((0, vec!["BUY_PRODUCT".into(), "WHEAT".into(),
                        wheat_short.to_string()]));
        avail -= cost;
    }

    let n_hire_want = knobs.hire.clamp(0, 10) as i64;
    for (p, n) in dawn_sells.iter() {
        q.push((2, vec!["SELL".into(), PRODUCTS[*p].into(), n.to_string()]));
        sold[*p] += n;
    }

    // ---- land -------------------------------------------------------------
    let mut owned_extra = farm.unlocked_quadrants.len() - 1;
    let mut new_empty: Vec<(i64, i64)> = Vec::new();
    if knobs.buy_land && !is_final && owned_extra < 3 {
        if let Some((quad, cost)) = rules::next_land(owned_extra) {
            if avail >= cost as f64 + 500.0 {
                q.push((3, vec!["BUY_LAND".into()]));
                avail -= cost as f64;
                owned_extra += 1;
                for y in 0..BOARD {
                    for x in 0..BOARD {
                        if rules::quadrant_of(x, y, BOARD) == quad
                            && farm.tiles[y as usize][x as usize] == Cell::Locked
                        {
                            new_empty.push((x, y));
                        }
                    }
                }
            }
        }
    }

    // ---- animals ----------------------------------------------------------
    let mut empty = c.empty.clone();
    empty.extend(new_empty.iter().copied());
    empty.sort_by_key(|t| (shed_dist(*t), t.1, t.0));
    let mut buy_n = [0i64; 3]; // ANIMAL_NAMES order
    let room = empty.len() as i64 + c.structs.len() as i64;
    for (bi, name) in BUY_ORDER.iter().enumerate() {
        let ai = ANIMAL_NAMES.iter().position(|n| n == name).unwrap();
        let cost = rules::animal(name).unwrap().cost as f64;
        let want = knobs.buy[bi].max(0) as i64;
        for _ in 0..want {
            if avail < cost + 300.0 || buy_n.iter().sum::<i64>() >= room {
                break;
            }
            buy_n[ai] += 1;
            avail -= cost;
        }
    }
    for (ai, n) in buy_n.iter().enumerate() {
        for _ in 0..*n {
            q.push((4, vec!["BUY_ANIMAL".into(), ANIMAL_NAMES[ai].into(),
                            "1".into()]));
        }
    }

    // ---- tasks from what STANDS on every tile -----------------------------
    let mut tasks: Vec<Task> = Vec::new();
    let mut fert_left = shed[P_FERT];

    for ((x, y), ai) in c.animals.iter() {
        let Cell::Structure { animal: Some(a), .. } =
            &farm.tiles[*y as usize][*x as usize] else { continue };
        let mut ops = Vec::new();
        let mut cargo = Cargo::default();
        if feed_today > 0 && !a.fed_today {
            ops.push(Op::new(OpKind::Feed));
            cargo.wheat += 1;
        }
        if feed_today > 0 && !a.cared_today {
            ops.push(Op::new(OpKind::Care));
        }
        if a.yield_units >= knobs.harvest_at as i64
            || (is_final && a.yield_units > 0)
        {
            ops.push(Op::new(OpKind::Harvest));
        }
        if a.fertilizer_available {
            ops.push(Op::new(OpKind::CollectFert));
        }
        let _ = ai;
        if !ops.is_empty() {
            tasks.push(Task { pri: 0, tile: (*x, *y), ops, cargo });
        }
    }

    let wheat_window_last = 25i64;
    for ((x, y), ci) in c.plants.iter() {
        let Cell::Plant {
            planted_day, watered_today, consecutive_unwatered, yield_units,
            fertilized_until_day, ..
        } = &farm.tiles[*y as usize][*x as usize] else { continue };
        let cd = &rules::CROPS[*ci];
        let age = day - planted_day;
        let mut ops: Vec<Op> = Vec::new();
        let mut cargo = Cargo::default();
        let mut replant = false;

        if cd.ongoing {
            let last_prod = cd.first_yield_day + cd.interval * (cd.max_yield - 1);
            if *yield_units > 0 && age >= cd.first_yield_day {
                ops.push(Op::new(OpKind::Harvest));
            }
            if age >= last_prod && (*yield_units <= 0 || age > last_prod) {
                ops.push(Op::new(OpKind::Dig));
                replant = true;
            } else {
                if *ci == C_STRAW
                    && knobs.fert_straw
                    && (age == 8 || age == 10 || age == 12)
                    && fert_left > 0
                    && *fertilized_until_day < day
                {
                    ops.insert(0, Op::new(OpKind::Fertilize));
                    cargo.fert += 1;
                    fert_left -= 1;
                }
                if !watered_today {
                    ops.push(Op::new(OpKind::Water));
                }
            }
        } else {
            let win0 = (cd.max_yield_day + 1) / 2;
            let ripe = (*yield_units >= cd.max_yield
                        && age >= cd.first_yield_day)
                || age >= cd.max_yield_day;
            if !watered_today
                && !ripe
                && ((win0..=cd.max_yield_day).contains(&age)
                    || *consecutive_unwatered >= 1)
            {
                ops.push(Op::new(OpKind::Water));
            }
            if ripe || (is_final && *yield_units > 0 && age >= cd.first_yield_day)
            {
                if !watered_today
                    && (win0..=cd.max_yield_day).contains(&age)
                    && *yield_units < cd.max_yield
                    && !ops.iter().any(|o| o.k == OpKind::Water)
                {
                    ops.push(Op::new(OpKind::Water));
                }
                ops.push(Op::new(OpKind::Harvest));
                replant = true;
            }
        }
        // Replant only if the crop can still MATURE and be sold. That single
        // condition is the "planted tiles day 28: 30 vs 27" leak, closed.
        if replant && !is_final && day + rules::CROPS[C_WHEAT].first_yield_day
            <= LAST_DAY && day <= wheat_window_last
        {
            ops.push(Op::with(OpKind::Plant, C_WHEAT));
            ops.push(Op::new(OpKind::Water));
            cargo.seed[C_WHEAT] += 1;
        }
        if !ops.is_empty() {
            tasks.push(Task { pri: 1, tile: (*x, *y), ops, cargo });
        }
    }

    for t in c.weeds.iter() {
        let mut ops = vec![Op::new(OpKind::Dig)];
        let mut cargo = Cargo::default();
        if !is_final && day <= wheat_window_last && day + 2 <= LAST_DAY {
            ops.push(Op::with(OpKind::Plant, C_WHEAT));
            ops.push(Op::new(OpKind::Water));
            cargo.seed[C_WHEAT] += 1;
        }
        tasks.push(Task { pri: 3, tile: *t, ops, cargo });
    }

    // ---- placements -------------------------------------------------------
    let mut place: Vec<usize> = Vec::new();
    for (ai, n) in buy_n.iter().enumerate() {
        for _ in 0..*n {
            place.push(ai);
        }
    }
    for (ai, name) in ANIMAL_NAMES.iter().enumerate() {
        for _ in 0..st.private[me].shed.get(name) {
            place.push(ai);
        }
    }
    let mut used: Vec<(i64, i64)> = Vec::new();
    for ai in place.iter() {
        let want_pasture = rules::ANIMALS[*ai].structure == "PASTURE";
        let mut spot = None;
        for (t, is_p) in c.structs.iter() {
            if !used.contains(t) && *is_p == want_pasture {
                spot = Some(*t);
                break;
            }
        }
        if let Some(t) = spot {
            used.push(t);
            let mut cargo = Cargo::default();
            cargo.animal[*ai] += 1;
            tasks.push(Task { pri: 2, tile: t,
                              ops: vec![Op::with(OpKind::Place, *ai)], cargo });
            continue;
        }
        let Some(t) = empty.iter().find(|t| !used.contains(t)).copied() else {
            break;
        };
        used.push(t);
        let mut cargo = Cargo::default();
        cargo.animal[*ai] += 1;
        let build = if want_pasture { OpKind::BuildPasture } else { OpKind::BuildCoop };
        tasks.push(Task {
            pri: 2,
            tile: t,
            ops: vec![Op::new(build), Op::with(OpKind::Place, *ai)],
            cargo,
        });
    }

    // ---- new plantings on the remaining free tiles -------------------------
    let free: Vec<(i64, i64)> =
        empty.iter().filter(|t| !used.contains(t)).copied().collect();
    let mut fi = 0usize;
    let mut seed_free = seeds;
    if !is_final {
        for &ci in &[C_MELON, C_STRAW, C_TOMATO, C_CARROT, C_WHEAT] {
            let cd = &rules::CROPS[ci];
            // Never plant something that cannot mature before the last day.
            if day + cd.first_yield_day > LAST_DAY {
                continue;
            }
            let want = knobs.plant[ci].max(0) as i64;
            for _ in 0..want {
                if fi >= free.len() {
                    break;
                }
                if seed_free[ci] > 0 {
                    seed_free[ci] -= 1;
                } else if avail - cd.seed_cost as f64 >= 20.0 {
                    avail -= cd.seed_cost as f64;
                } else {
                    break;
                }
                let mut cargo = Cargo::default();
                cargo.seed[ci] += 1;
                let pri = if ci == C_WHEAT { 5 } else { 4 };
                tasks.push(Task {
                    pri,
                    tile: free[fi],
                    ops: vec![Op::with(OpKind::Plant, ci),
                              Op::new(OpKind::Water)],
                    cargo,
                });
                fi += 1;
            }
        }
    }

    // ---- labour: sequential nearest-fill, one ordered tour per unit --------
    tasks.sort_by_key(|t| (t.pri, t.tile.1, t.tile.0));
    let n_tasks = tasks.len();
    let budget_f = TPD - 1;
    let budget_h = TPD - 2;
    struct Unit {
        pos: (i64, i64),
        left: i64,
        jobs: Vec<Job>,
        cargo: Cargo,
        n: i64,
    }
    let mut units: Vec<Unit> = Vec::with_capacity(1 + n_hire_want as usize);
    units.push(Unit { pos: SHED[0], left: budget_f, jobs: Vec::new(),
                      cargo: Cargo::default(), n: 0 });
    for i in 0..n_hire_want {
        units.push(Unit { pos: SHED[((i + 1) % 4) as usize], left: budget_h,
                          jobs: Vec::new(), cargo: Cargo::default(), n: 0 });
    }
    let mut pool: Vec<((i64, i64), Vec<Op>, Cargo)> = Vec::new();
    let mut ui = 0usize;
    for mut t in tasks.into_iter() {
        let mut placed = false;
        while ui < units.len() {
            let u = &mut units[ui];
            let extra = (t.cargo.wheat > 0 && u.cargo.wheat == 0) as i64
                + (t.cargo.fert > 0 && u.cargo.fert == 0) as i64
                + (0..3).filter(|i| t.cargo.animal[*i] > 0
                                && u.cargo.animal[*i] == 0).count() as i64;
            let cost = dist(u.pos, t.tile) + t.ops.len() as i64 + extra;
            if u.left >= cost {
                u.n += t.ops.len() as i64;
                u.left -= cost;
                u.pos = t.tile;
                u.cargo.wheat += t.cargo.wheat;
                u.cargo.fert += t.cargo.fert;
                for i in 0..3 {
                    u.cargo.animal[i] += t.cargo.animal[i];
                }
                for i in 0..N_CROP {
                    u.cargo.seed[i] += t.cargo.seed[i];
                }
                u.jobs.push(Job {
                    tile: Some(t.tile),
                    ops: std::mem::take(&mut t.ops),
                    waits: 0,
                });
                placed = true;
                break;
            }
            ui += 1;
        }
        if !placed {
            pool.push((t.tile, t.ops, t.cargo));
        }
    }

    // hires: exactly the hands that got work, cash-checked (fib resets daily)
    let n_used = units.iter().skip(1).filter(|u| u.n > 0).count() as i64;
    let mut n_hire = n_used;
    while n_hire > 0 && fib_sum(n_hire) as f64 > avail - 100.0 {
        n_hire -= 1;
    }
    avail -= fib_sum(n_hire) as f64;
    for _ in 0..n_hire {
        q.push((1, vec!["HIRE".into()]));
    }
    while units.len() > 1 + n_hire as usize {
        let u = units.pop().unwrap();
        for j in u.jobs {
            if let Some(t) = j.tile {
                pool.push((t, j.ops, Cargo::default()));
            }
        }
    }

    // seeds: only for plantings a LIVE unit will execute today (held seed ~0)
    let mut seed_need = [0i64; N_CROP];
    for u in units.iter() {
        for i in 0..N_CROP {
            seed_need[i] += u.cargo.seed[i];
        }
    }
    for ci in 0..N_CROP {
        let short = seed_need[ci] - seeds[ci];
        if short > 0 {
            let pri = if ci == C_WHEAT { 5 } else { 5 };
            q.push((pri, vec!["BUY_SEED".into(), CROP_NAMES[ci].into(),
                              short.to_string()]));
        }
    }

    // cargo pickups become each unit's first job (units spawn at the shed)
    for u in units.iter_mut() {
        let mut pre: Vec<Op> = Vec::new();
        if u.cargo.wheat > 0 {
            pre.push(Op { k: OpKind::Pickup, arg: P_WHEAT as u8,
                          n: u.cargo.wheat });
        }
        if u.cargo.fert > 0 {
            pre.push(Op { k: OpKind::Pickup, arg: P_FERT as u8,
                          n: u.cargo.fert });
        }
        for ai in 0..3 {
            if u.cargo.animal[ai] > 0 {
                pre.push(Op { k: OpKind::Pickup, arg: (9 + ai) as u8,
                              n: u.cargo.animal[ai] });
            }
        }
        if !pre.is_empty() {
            u.jobs.insert(0, Job { tile: None, ops: pre, waits: 0 });
        }
    }

    let fert_want: i64 = units.iter().map(|u| u.cargo.fert).sum();
    let fert_short = (fert_want - shed[P_FERT]).max(0);
    if fert_short > 0 {
        q.push((6, vec!["BUY_PRODUCT".into(), "FERTILIZER".into(),
                        fert_short.to_string()]));
    }

    q.sort_by_key(|(p, _)| *p);
    let orders: Vec<Vec<String>> = q.into_iter().map(|(_, o)| o).collect();
    let spill: Vec<Vec<Vec<String>>> =
        orders.chunks(10).map(|c| c.to_vec()).collect();

    DayPlan {
        day,
        knobs: knobs.clone(),
        spill,
        jobs: units.into_iter().map(|u| u.jobs).collect(),
        pool,
        keep,
        sold_today: sold,
        is_final,
        n_hire,
        n_tasks,
    }
}

// ---------------------------------------------------------------- executor --

struct Ctx<'a> {
    farm: &'a Farm,
    shed: [i64; N_PROD],
    shed_animal: [i64; 3],
    seeds: [i64; N_CROP],
    day: i64,
    h: i64,
}

/// 1 = emit, 0 = skip this op, -1 = wait a turn, -2 = job list was rewritten.
fn validate(op: &Op, tile: Option<&Cell>, pos: (i64, i64), inv: &InvView,
            ctx: &Ctx, jobs: &mut Vec<Job>) -> i32 {
    match op.k {
        OpKind::Pickup => {
            if !SHED.contains(&pos) {
                return 0;
            }
            let have = if (op.arg as usize) < N_PROD {
                ctx.shed[op.arg as usize]
            } else {
                ctx.shed_animal[op.arg as usize - 9]
            };
            if have <= 0 {
                return -1;
            }
            1
        }
        OpKind::Drop => {
            if SHED.contains(&pos) && inv.total > 0 { 1 } else { 0 }
        }
        _ => {
            let Some(tl) = tile else { return -1 };
            match op.k {
                OpKind::Plant => {
                    if *tl != Cell::Empty {
                        return 0;
                    }
                    if ctx.seeds[op.arg as usize] <= 0 {
                        return if ctx.h <= 2 { -1 } else { 0 };
                    }
                    1
                }
                OpKind::Water => match tl {
                    Cell::Plant { watered_today, .. } if !watered_today => 1,
                    _ => 0,
                },
                OpKind::Harvest => match tl {
                    Cell::Plant { crop, planted_day, yield_units, .. } => {
                        if *yield_units <= 0 {
                            return 0;
                        }
                        let ci = CROP_NAMES.iter().position(|n| n == crop)
                            .unwrap_or(0);
                        if ctx.day - planted_day
                            < rules::CROPS[ci].first_yield_day
                        {
                            return 0;
                        }
                        1
                    }
                    Cell::Structure { animal: Some(a), .. } => {
                        if a.yield_units > 0 { 1 } else { 0 }
                    }
                    _ => 0,
                },
                OpKind::Feed => {
                    let Cell::Structure { animal: Some(a), .. } = tl else {
                        return 0;
                    };
                    if a.fed_today {
                        return 0;
                    }
                    if inv.wheat > 0 {
                        return 1;
                    }
                    if ctx.shed[P_WHEAT] > 0 {
                        let n = jobs.iter()
                            .flat_map(|j| j.ops.iter())
                            .filter(|o| o.k == OpKind::Feed)
                            .count()
                            .max(1) as i64;
                        jobs.insert(0, Job {
                            tile: None,
                            ops: vec![Op { k: OpKind::Pickup,
                                           arg: P_WHEAT as u8,
                                           n: n.min(ctx.shed[P_WHEAT]) }],
                            waits: 0,
                        });
                        return -2;
                    }
                    0
                }
                OpKind::Care => match tl {
                    Cell::Structure { animal: Some(a), .. }
                        if !a.cared_today => 1,
                    _ => 0,
                },
                OpKind::CollectFert => match tl {
                    Cell::Structure { animal: Some(a), .. }
                        if a.fertilizer_available => 1,
                    _ => 0,
                },
                OpKind::Fertilize => match tl {
                    Cell::Plant { fertilized_until_day, .. } => {
                        if *fertilized_until_day >= ctx.day {
                            return 0;
                        }
                        if inv.fert > 0 { 1 } else { 0 }
                    }
                    _ => 0,
                },
                OpKind::Dig => match tl {
                    Cell::Empty => 0,
                    Cell::Structure { animal: Some(_), .. } => 0,
                    Cell::Locked => -1,
                    _ => 1,
                },
                OpKind::BuildCoop | OpKind::BuildPasture => {
                    if *tl == Cell::Empty { 1 } else { 0 }
                }
                OpKind::Place => {
                    let ai = op.arg as usize;
                    let want = rules::ANIMALS[ai].structure;
                    let ok = matches!(tl,
                        Cell::Structure { kind, animal: None } if kind == want);
                    if !ok {
                        return 0;
                    }
                    if inv.animal[ai] > 0 {
                        return 1;
                    }
                    if ctx.shed_animal[ai] > 0 {
                        jobs.insert(0, Job {
                            tile: None,
                            ops: vec![Op { k: OpKind::Pickup,
                                           arg: (9 + ai) as u8, n: 1 }],
                            waits: 0,
                        });
                        return -2;
                    }
                    0
                }
                _ => 1,
            }
        }
    }
}

#[derive(Default, Clone, Copy)]
struct InvView {
    total: i64,
    wheat: i64,
    fert: i64,
    animal: [i64; 3],
}

fn read_inv(st: &State, me: usize, ui: usize) -> InvView {
    let mut v = InvView::default();
    if let Some(m) = st.private[me].inventories.get(ui) {
        v.total = m.sum();
        v.wheat = m.get("WHEAT");
        v.fert = m.get("FERTILIZER");
        for (i, n) in ANIMAL_NAMES.iter().enumerate() {
            v.animal[i] = m.get(n);
        }
    }
    v
}

/// Nearest feasible unassigned task for an idle unit.
fn grab(plan: &mut DayPlan, pos: (i64, i64), inv: &InvView, hours_left: i64,
        ctx: &Ctx) -> Option<Job> {
    let mut best: Option<(usize, i64)> = None;
    for (i, (t, ops, cargo)) in plan.pool.iter().enumerate() {
        if cargo.wheat > 0 && inv.wheat <= 0 {
            continue;
        }
        if cargo.fert > 0 && inv.fert <= 0 {
            continue;
        }
        if (0..3).any(|a| cargo.animal[a] > 0 && inv.animal[a] <= 0) {
            continue;
        }
        if (0..N_CROP).any(|s| cargo.seed[s] > 0 && ctx.seeds[s] <= 0) {
            continue;
        }
        let d = dist(pos, *t);
        if d + ops.len() as i64 > hours_left {
            continue;
        }
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    let (i, _) = best?;
    let (t, ops, _) = plan.pool.remove(i);
    Some(Job { tile: Some(t), ops, waits: 0 })
}

/// Play one turn of the committed plan against the live state.
pub fn execute_turn(plan: &mut DayPlan, st: &State, me: usize) -> PlayerAction {
    let day = st.step / TPD;
    let h = st.step % TPD;
    let farm = &st.farms[me];
    let prices = read_prices(st);
    let mut ctx = Ctx {
        farm,
        shed: read_shed(st, me),
        shed_animal: {
            let mut a = [0i64; 3];
            for (i, n) in ANIMAL_NAMES.iter().enumerate() {
                a[i] = st.private[me].shed.get(n);
            }
            a
        },
        seeds: read_seeds(st, me),
        day,
        h,
    };

    // ---- market: spilled dawn orders first, then fresh sells --------------
    let mut market: Vec<Vec<String>> = if plan.spill.is_empty() {
        Vec::new()
    } else {
        plan.spill.remove(0)
    };
    if h > 0 || market.is_empty() {
        let have: Vec<String> = market
            .iter()
            .filter(|o| o[0] == "SELL")
            .map(|o| o[1].clone())
            .collect();
        let fresh = sell_orders(&plan.knobs, day, &ctx.shed, &prices,
                                &plan.keep, &plan.sold_today, plan.is_final);
        for (p, n) in fresh {
            if market.len() >= 10 {
                break;
            }
            if have.iter().any(|x| x == PRODUCTS[p]) {
                continue;
            }
            market.push(vec!["SELL".into(), PRODUCTS[p].into(), n.to_string()]);
            plan.sold_today[p] += n;
        }
    }
    market.truncate(10);

    // ---- units ------------------------------------------------------------
    let n_units = 1 + farm.hands.len();
    let mut acts: Vec<UnitAction> = Vec::with_capacity(n_units);
    for ui in 0..n_units {
        let pos = if ui == 0 { farm.farmer } else { farm.hands[ui - 1] };
        let inv = read_inv(st, me, ui);
        while plan.jobs.len() <= ui {
            plan.jobs.push(Vec::new());
        }
        let a = step_unit(plan, ui, pos, &inv, &mut ctx, h);
        acts.push(a);
    }

    PlayerAction { farmer: acts[0].clone(), hands: acts[1..].to_vec(), market }
}

fn step_unit(plan: &mut DayPlan, ui: usize, pos: (i64, i64), inv: &InvView,
             ctx: &mut Ctx, h: i64) -> UnitAction {
    let hours_left = TPD - h;
    let drop_at = plan.knobs.drop_at as i64;
    for _ in 0..40 {
        if plan.jobs[ui].is_empty() {
            let job = grab(plan, pos, inv, hours_left, ctx);
            match job {
                Some(j) => plan.jobs[ui].push(j),
                None => {
                    if inv.total > 0
                        && (plan.is_final
                            || hours_left <= shed_dist(pos) + 1
                            || inv.total >= drop_at)
                    {
                        if SHED.contains(&pos) {
                            return Op::new(OpKind::Drop).to_action();
                        }
                        return move_toward(pos, nearest_shed(pos));
                    }
                    return pass();
                }
            }
        }
        let (tile_opt, ops_empty) = {
            let j = &plan.jobs[ui][0];
            (j.tile, j.ops.is_empty())
        };
        if ops_empty {
            plan.jobs[ui].remove(0);
            continue;
        }
        let tile_ref: Option<&Cell> = match tile_opt {
            None => {
                if !SHED.contains(&pos) {
                    return move_toward(pos, nearest_shed(pos));
                }
                None
            }
            Some(t) => {
                if pos != t {
                    // mid-tour drop: heavy produce, standing at the shed, and
                    // nothing in the remaining jobs needs the cargo
                    let needs_cargo = plan.jobs[ui].iter().any(|j| {
                        j.ops.iter().any(|o| matches!(o.k,
                            OpKind::Feed | OpKind::Fertilize | OpKind::Place))
                    });
                    if inv.total >= drop_at && shed_dist(pos) == 0
                        && !needs_cargo
                    {
                        return Op::new(OpKind::Drop).to_action();
                    }
                    return move_toward(pos, t);
                }
                Some(&ctx.farm.tiles[t.1 as usize][t.0 as usize])
            }
        };
        let op = plan.jobs[ui][0].ops[0];
        let v = {
            let jobs = &mut plan.jobs[ui];
            validate(&op, tile_ref, pos, inv, ctx, jobs)
        };
        if v == -2 {
            continue;
        }
        if v == -1 {
            plan.jobs[ui][0].waits += 1;
            if plan.jobs[ui][0].waits <= 2 {
                return pass();
            }
            plan.jobs[ui].remove(0);
            continue;
        }
        plan.jobs[ui][0].ops.remove(0);
        if v == 0 {
            continue;
        }
        match op.k {
            OpKind::Plant => {
                ctx.seeds[op.arg as usize] -= 1;
            }
            OpKind::Pickup => {
                let a = op.arg as usize;
                let n = if a < N_PROD {
                    let n = op.n.min(ctx.shed[a]);
                    ctx.shed[a] -= n;
                    n
                } else {
                    let n = op.n.min(ctx.shed_animal[a - 9]);
                    ctx.shed_animal[a - 9] -= n;
                    n
                };
                if n <= 0 {
                    continue;
                }
                return Op { k: OpKind::Pickup, arg: op.arg, n }.to_action();
            }
            _ => {}
        }
        return op.to_action();
    }
    pass()
}

// ------------------------------------------------------------------ policy --

/// Structurally different economies for the transfer test.
///
/// `Field` is what the SEARCH models. Every other variant is an opponent the
/// search has never simulated, which is the only way to tell an exploiter of
/// one fixed policy from a searcher that has found something transferable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// The measured field skeleton (2 land, 9 cows, 5 sheep, 38 strawberry,
    /// 12 melon, wheat everywhere else).
    Field,
    /// Geese and eggs: the log-priced, dump-proof scale play.
    Geese,
    /// Wheat monoculture with wide caps -- the "just dump volume" economy.
    Wheat,
    /// Melon-heavy: rides the quadratic glut curve until it floors.
    Melon,
}

impl Style {
    pub fn parse(s: &str) -> Style {
        match s {
            "geese" => Style::Geese,
            "wheat" => Style::Wheat,
            "melon" => Style::Melon,
            _ => Style::Field,
        }
    }
    fn apply(self, k: &mut DayKnobs, day: i64) {
        match self {
            Style::Field => {}
            Style::Geese => {
                // Half the cow ramp becomes geese; eggs are log-priced, so
                // this economy scales where the field's saturates.
                k.buy[2] = k.buy[0].max(1);
                k.buy[0] = 0;
                if day >= 3 && day <= 20 {
                    k.buy[2] = k.buy[2].max(1);
                }
                k.plant[C_STRAW] = 0;
                k.plant[C_WHEAT] = 18;
            }
            Style::Wheat => {
                k.plant[C_STRAW] = 0;
                k.plant[C_MELON] = 0;
                k.plant[C_WHEAT] = 22;
                for i in 0..N_PROD {
                    k.sell_cap[i] = -1;
                }
            }
            Style::Melon => {
                if (3..=16).contains(&day) {
                    k.plant[C_MELON] = 4;
                }
                k.plant[C_STRAW] = k.plant[C_STRAW].min(1);
                k.sell_cap[P_MELON] = -1;
            }
        }
    }
}

/// A closed-loop policy: re-plans every dawn from the live state, executes the
/// committed plan the rest of the day. This is BOTH the search's prior and its
/// opponent model (see docs/history/trackp-search-design-2026-09-03.md §5).
pub struct SkeletonPolicy {
    pub me: usize,
    plan: Option<DayPlan>,
    /// Scales the opponent's sell caps -- an aggression knob for the opponent
    /// ensemble. 1.0 = the measured field skeleton.
    pub aggression: f64,
    pub style: Style,
}

impl SkeletonPolicy {
    pub fn new(me: usize) -> Self {
        SkeletonPolicy { me, plan: None, aggression: 1.0, style: Style::Field }
    }
    pub fn with_aggression(me: usize, aggression: f64) -> Self {
        SkeletonPolicy { me, plan: None, aggression, style: Style::Field }
    }
    pub fn styled(me: usize, style: Style, aggression: f64) -> Self {
        SkeletonPolicy { me, plan: None, aggression, style }
    }

    pub fn act(&mut self, st: &State) -> PlayerAction {
        let day = st.step / TPD;
        let needs = match &self.plan {
            None => true,
            Some(p) => p.day != day,
        };
        if needs {
            let mut k = DayKnobs::skeleton(day, st, self.me);
            self.style.apply(&mut k, day);
            if (self.aggression - 1.0).abs() > 1e-9 {
                for i in 0..N_PROD {
                    if k.sell_cap[i] >= 0 {
                        k.sell_cap[i] =
                            ((k.sell_cap[i] as f64 * self.aggression)
                                .round() as i16).max(1);
                    }
                }
            }
            self.plan = Some(plan_day(st, self.me, &k));
        }
        let plan = self.plan.as_mut().unwrap();
        execute_turn(plan, st, self.me)
    }
}

/// Shed capacity headroom, for the value function's overflow guard.
pub fn shed_room(st: &State, me: usize) -> i64 {
    (SHED_CAP - st.private[me].shed.sum()).max(0)
}

#[allow(dead_code)]
fn _unused(c: usize) -> usize {
    crop_to_product(c)
}
