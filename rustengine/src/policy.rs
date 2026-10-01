//! `kagg play` -- the compiled closed-loop economy policy behind the Track-P
//! compiled-agent transport (Phase A).
//!
//! This is a LINE-BY-LINE port of the reference Python planner emitted by
//! `src/trackp/build_econ_agent.py --genome skeleton`. That is deliberate and
//! it is the whole point of Phase A: the pure-Python fallback shipped inside
//! `main.py` IS that same planner, so when the subprocess bridge fails the
//! agent degrades in SPEED only, not in strategy. `tests/test_compiled_agent.py`
//! asserts the two produce byte-identical action streams over full episodes --
//! which is the correctness proof for the transport, independent of how strong
//! the policy happens to be.
//!
//! Seat law (unchanged): every decision is derived from the live observation.
//! No tape, no route prefix, no embedded action sequence.
//!
//! Protocol (`kagg play`): one observation JSON per line on stdin, one action
//! JSON per line on stdout. State (today's plan) is kept between lines, so one
//! process serves a whole episode. `RESETP` on a line by itself clears the
//! cached plan, which the harness uses between episodes.
//!
//! ## Phase B: `kagg play --budget-ms N`
//!
//! With a NON-ZERO budget the turn is answered by `search::decide` (the Phase-B
//! amortised day-plan searcher) instead of the skeleton planner above. Zero --
//! the DEFAULT -- keeps the Phase-A path byte for byte, so
//! `tests/test_compiled_agent.py --budget-ms 0` still proves the compiled
//! policy and the inlined Python fallback are one policy. That equivalence is
//! what makes a fallback cost speed and nothing else, and it must survive the
//! search being switched on.
//!
//! The searcher's fallbacks are layered under the bridge's, not instead of
//! them: a panic inside `decide` resets the searcher and answers a legal PASS,
//! a state we cannot reconstruct falls through to the SKELETON planner (which
//! needs only the raw observation), and only then does the Python side's own
//! watchdog/validator/fallback chain come into play.

use crate::json::{parse, Json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufWriter, Write};

// --------------------------------------------------------------- constants --

const TPD: i64 = 24;
const LAST_DAY: i64 = 29;
const MAX_HIRE: usize = 10;
const SHED: [(i64, i64); 4] = [(4, 4), (5, 4), (4, 5), (5, 5)];
const LAND_ORDER: [&str; 3] = ["NE", "SW", "SE"];
const LAND_COST: [f64; 3] = [1000.0, 2000.0, 4000.0];

/// SELLABLE, in the reference planner's order (NOT alphabetical -- the order
/// decides which product gets a market slot when the queue is full).
const SELLABLE: [&str; 9] = [
    "MELON", "STRAWBERRY", "WOOL", "MILK", "EGG", "CARROT", "TOMATO",
    "FERTILIZER", "WHEAT",
];
const VALVE_ORDER: [&str; 9] = [
    "WHEAT", "EGG", "FERTILIZER", "CARROT", "TOMATO", "MILK", "STRAWBERRY",
    "WOOL", "MELON",
];
const PLANT_ORDER: [&str; 3] = ["MELON", "STRAWBERRY", "CARROT"];
const SPECIES: [&str; 3] = ["COW", "SHEEP", "GOOSE"];

fn base_price(item: &str) -> f64 {
    match item {
        "WHEAT" => 25.0,
        "CARROT" => 35.0,
        "TOMATO" => 60.0,
        "STRAWBERRY" => 120.0,
        "MELON" => 250.0,
        "EGG" => 50.0,
        "MILK" => 160.0,
        "WOOL" => 200.0,
        "FERTILIZER" => 100.0,
        _ => 25.0,
    }
}

#[derive(Clone, Copy)]
struct CropDef {
    seed: f64,
    first: i64,
    maxd: i64,
    iv: i64,
    maxy: i64,
    ong: bool,
}

fn crop_def(name: &str) -> CropDef {
    match name {
        "CARROT" => CropDef { seed: 20.0, first: 2, maxd: 3, iv: 0, maxy: 4, ong: false },
        "TOMATO" => CropDef { seed: 50.0, first: 8, maxd: 8, iv: 1, maxy: 4, ong: true },
        "STRAWBERRY" => CropDef { seed: 100.0, first: 10, maxd: 10, iv: 2, maxy: 4, ong: true },
        "MELON" => CropDef { seed: 80.0, first: 10, maxd: 12, iv: 0, maxy: 6, ong: false },
        // WHEAT, and the fallback for an unknown crop (matches the Python
        // `CROPS.get(crop) or CROPS["WHEAT"]`).
        _ => CropDef { seed: 10.0, first: 2, maxd: 4, iv: 0, maxy: 6, ong: false },
    }
}


fn animal_cost(sp: &str) -> f64 {
    match sp {
        "GOOSE" => 300.0,
        "COW" => 400.0,
        _ => 500.0,
    }
}

fn animal_pen(sp: &str) -> &'static str {
    if sp == "GOOSE" { "COOP" } else { "PASTURE" }
}

fn animal_build(sp: &str) -> &'static str {
    if sp == "GOOSE" { "BUILD_COOP" } else { "BUILD_PASTURE" }
}

// ------------------------------------------------------------------ genome --

/// The measured top-10 field skeleton as day-indexed targets. These are the
/// DEFAULTS of `skeleton_genome()` in src/trackp/build_econ_agent.py, frozen
/// here so the compiled agent needs no data file at run time.
pub struct Genome {
    pub land: [(&'static str, i64); 3],
    pub hires: [i64; 30],
    pub herd_cow: &'static [(i64, i64)],
    pub herd_sheep: &'static [(i64, i64)],
    pub herd_goose: &'static [(i64, i64)],
    pub tile_melon: i64,
    pub tile_straw: i64,
    pub tile_carrot: i64,
    pub win_melon: (i64, i64),
    pub win_straw: (i64, i64),
    pub win_carrot: (i64, i64),
    pub win_wheat: (i64, i64),
    pub floor_seed: f64,
    pub floor_straw: f64,
    pub floor_land: f64,
    pub floor_animal: f64,
    pub land_reserve_lead: i64,
    pub pen_lookahead: i64,
    pub feed_buffer: i64,
    pub fert_crop: &'static str,
    pub care: bool,
    pub sell_cap_melon: i64,
    pub sell_cap_wool: i64,
    pub sell_cap_straw: i64,
    pub sell_cap_milk: i64,
    pub sell_cap_egg: i64,
    pub sell_cap_carrot: i64,
    pub sell_cap_tomato: i64,
    pub sell_cap_fert: i64,
    pub fert_keep: i64,
    pub plant_pace: f64,
    pub valve_hi: i64,
    pub valve_lo: i64,
}

pub const SKELETON: Genome = Genome {
    land: [("NE", 0), ("SW", 5), ("SE", -1)],
    hires: [
        5, 4, 4, 5, 4, 5, 8, 8, 10, 11, 11, 11, 9, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11,
    ],
    herd_cow: &[(0, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 6), (10, 7), (13, 8), (16, 9)],
    herd_sheep: &[(0, 1), (6, 2), (9, 3), (12, 4), (15, 5)],
    herd_goose: &[],
    tile_melon: 12,
    tile_straw: 38,
    tile_carrot: 0,
    win_melon: (4, 12),
    win_straw: (5, 14),
    win_carrot: (0, 24),
    win_wheat: (0, 25),
    floor_seed: 20.0,
    floor_straw: 700.0,
    floor_land: 500.0,
    floor_animal: 300.0,
    land_reserve_lead: 0,
    pen_lookahead: 5,
    feed_buffer: 6,
    fert_crop: "STRAWBERRY",
    care: true,
    sell_cap_melon: 24,
    sell_cap_wool: 40,
    sell_cap_straw: 60,
    sell_cap_milk: 40,
    sell_cap_egg: 60,
    sell_cap_carrot: 60,
    sell_cap_tomato: 60,
    sell_cap_fert: 60,
    fert_keep: 12,
    plant_pace: 0.95,
    valve_hi: 55,
    valve_lo: 45,
};

// G2 (2026-09-18): the genome is EXTERNALIZED to genome.json, shared with the
// Python builder (src/kaggriculture/trackp/build_econ_agent.py loads the same
// file). `load_genome()` reads it at runtime and leaks it to `&'static` so the
// Planner signature is unchanged; on any miss it returns `&SKELETON`, which is
// byte-identical to a complete genome.json, so the compiled agent is robust
// whether or not the file was shipped beside it.
fn leak_pairs(arr: &Json, default: &'static [(i64, i64)]) -> &'static [(i64, i64)] {
    if arr.is_null() {
        return default;
    }
    let v: Vec<(i64, i64)> = arr
        .arr()
        .iter()
        .map(|p| (p.idx(0).i64(), p.idx(1).i64()))
        .collect();
    Box::leak(v.into_boxed_slice())
}

fn build_genome(j: &Json) -> Genome {
    let lj = j.get("land");
    let land_of = |k: &str, d: i64| {
        let v = lj.get(k);
        if v.is_null() { d } else { v.i64() }
    };
    let mut hires = SKELETON.hires;
    for (idx, slot) in j.get("hires").arr().iter().enumerate().take(30) {
        hires[idx] = slot.i64();
    }
    let herd = j.get("herd");
    let tj = j.get("tiles");
    let ti = |k: &str, d: i64| {
        let v = tj.get(k);
        if v.is_null() { d } else { v.i64() }
    };
    let wj = j.get("window");
    let win = |k: &str, d: (i64, i64)| {
        let v = wj.get(k);
        if v.is_null() { d } else { (v.idx(0).i64(), v.idx(1).i64()) }
    };
    let sc = j.get("sell_cap");
    let cap = |k: &str, d: i64| {
        let v = sc.get(k);
        if v.is_null() { d } else { v.i64() }
    };
    let f = |k: &str, d: f64| {
        let v = j.get(k);
        if v.is_null() { d } else { v.f64() }
    };
    let i = |k: &str, d: i64| {
        let v = j.get(k);
        if v.is_null() { d } else { v.i64() }
    };
    let b = |k: &str, d: bool| {
        let v = j.get(k);
        if v.is_null() { d } else { v.bool() }
    };
    let fc = j.get("fert_crop");
    let fert_crop: &'static str = if fc.is_null() {
        SKELETON.fert_crop
    } else {
        Box::leak(fc.str().to_string().into_boxed_str())
    };
    Genome {
        land: [
            ("NE", land_of("NE", 0)),
            ("SW", land_of("SW", 5)),
            ("SE", land_of("SE", -1)),
        ],
        hires,
        herd_cow: leak_pairs(herd.get("COW"), SKELETON.herd_cow),
        herd_sheep: leak_pairs(herd.get("SHEEP"), SKELETON.herd_sheep),
        herd_goose: leak_pairs(herd.get("GOOSE"), SKELETON.herd_goose),
        tile_melon: ti("MELON", SKELETON.tile_melon),
        tile_straw: ti("STRAWBERRY", SKELETON.tile_straw),
        tile_carrot: ti("CARROT", SKELETON.tile_carrot),
        win_melon: win("MELON", SKELETON.win_melon),
        win_straw: win("STRAWBERRY", SKELETON.win_straw),
        win_carrot: win("CARROT", SKELETON.win_carrot),
        win_wheat: win("WHEAT", SKELETON.win_wheat),
        floor_seed: f("floor_seed", SKELETON.floor_seed),
        floor_straw: f("floor_straw", SKELETON.floor_straw),
        floor_land: f("floor_land", SKELETON.floor_land),
        floor_animal: f("floor_animal", SKELETON.floor_animal),
        land_reserve_lead: i("land_reserve_lead", SKELETON.land_reserve_lead),
        pen_lookahead: i("pen_lookahead", SKELETON.pen_lookahead),
        feed_buffer: i("feed_buffer", SKELETON.feed_buffer),
        fert_crop,
        care: b("care", SKELETON.care),
        sell_cap_melon: cap("MELON", SKELETON.sell_cap_melon),
        sell_cap_wool: cap("WOOL", SKELETON.sell_cap_wool),
        sell_cap_straw: cap("STRAWBERRY", SKELETON.sell_cap_straw),
        sell_cap_milk: cap("MILK", SKELETON.sell_cap_milk),
        sell_cap_egg: cap("EGG", SKELETON.sell_cap_egg),
        sell_cap_carrot: cap("CARROT", SKELETON.sell_cap_carrot),
        sell_cap_tomato: cap("TOMATO", SKELETON.sell_cap_tomato),
        sell_cap_fert: cap("FERTILIZER", SKELETON.sell_cap_fert),
        fert_keep: i("fert_keep", SKELETON.fert_keep),
        plant_pace: f("plant_pace", SKELETON.plant_pace),
        valve_hi: i("valve_hi", SKELETON.valve_hi),
        valve_lo: i("valve_lo", SKELETON.valve_lo),
    }
}

fn genome_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("TRACKP_GENOME") {
        if !p.is_empty() {
            return Some(std::path::PathBuf::from(p));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("genome.json");
            if p.exists() {
                return Some(p);
            }
        }
    }
    let cwd = std::path::PathBuf::from("genome.json");
    if cwd.exists() {
        return Some(cwd);
    }
    None
}

/// The active genome: genome.json if found (leaked to `&'static`), else the
/// frozen SKELETON default.
pub fn load_genome() -> &'static Genome {
    let g = genome_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| parse(&s).ok())
        .map(|j| build_genome(&j));
    match g {
        Some(g) => Box::leak(Box::new(g)),
        None => &SKELETON,
    }
}

impl Genome {
    fn tile_target(&self, c: &str) -> i64 {
        match c {
            "MELON" => self.tile_melon,
            "STRAWBERRY" => self.tile_straw,
            "CARROT" => self.tile_carrot,
            _ => 0,
        }
    }
    fn window(&self, c: &str) -> (i64, i64) {
        match c {
            "MELON" => self.win_melon,
            "STRAWBERRY" => self.win_straw,
            "CARROT" => self.win_carrot,
            "WHEAT" => self.win_wheat,
            _ => (0, 25),
        }
    }
    fn herd(&self, sp: &str) -> &'static [(i64, i64)] {
        match sp {
            "COW" => self.herd_cow,
            "SHEEP" => self.herd_sheep,
            _ => self.herd_goose,
        }
    }
    fn sell_cap(&self, item: &str) -> i64 {
        match item {
            "MELON" => self.sell_cap_melon,
            "WOOL" => self.sell_cap_wool,
            "STRAWBERRY" => self.sell_cap_straw,
            "MILK" => self.sell_cap_milk,
            "EGG" => self.sell_cap_egg,
            "CARROT" => self.sell_cap_carrot,
            "TOMATO" => self.sell_cap_tomato,
            "FERTILIZER" => self.sell_cap_fert,
            _ => 999, // WHEAT is log-priced: dump-proof, uncapped.
        }
    }
}

// -------------------------------------------------------------- geometry --

fn quad(t: (i64, i64)) -> &'static str {
    if t.1 < 5 {
        if t.0 >= 5 { "NE" } else { "NW" }
    } else if t.0 >= 5 {
        "SE"
    } else {
        "SW"
    }
}

fn shed_dist(t: (i64, i64)) -> i64 {
    SHED.iter().map(|s| (t.0 - s.0).abs() + (t.1 - s.1).abs()).min().unwrap()
}

/// Boustrophedon per quadrant, each starting at its shed corner: a contiguous
/// slice of this order is a cheap walking tour.
fn tile_order() -> Vec<(i64, i64)> {
    let blocks: [([i64; 5], [i64; 5]); 4] = [
        ([4, 3, 2, 1, 0], [4, 3, 2, 1, 0]),
        ([5, 6, 7, 8, 9], [4, 3, 2, 1, 0]),
        ([4, 3, 2, 1, 0], [5, 6, 7, 8, 9]),
        ([5, 6, 7, 8, 9], [5, 6, 7, 8, 9]),
    ];
    let mut out = Vec::with_capacity(96);
    for (xs, ys) in blocks {
        for (i, y) in ys.iter().enumerate() {
            let row: Vec<i64> = if i % 2 == 0 {
                xs.to_vec()
            } else {
                xs.iter().rev().copied().collect()
            };
            for x in row {
                if !SHED.contains(&(x, *y)) {
                    out.push((x, *y));
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------ units --

type Op = Vec<String>;

fn op(tokens: &[&str]) -> Op {
    tokens.iter().map(|t| t.to_string()).collect()
}

struct Unit {
    pos: (i64, i64),
    ops: Vec<Op>,
    cap: usize,
    start: i64,
    loaded: bool,
}

impl Unit {
    fn new(pos: (i64, i64), start: i64, cap: usize) -> Self {
        Unit { pos, ops: Vec::new(), cap, start, loaded: false }
    }
    fn budget(&self) -> i64 {
        self.cap as i64 - self.ops.len() as i64
    }
    fn walk(&mut self, t: (i64, i64)) {
        let (mut ax, mut ay) = self.pos;
        let (bx, by) = t;
        while ax < bx && self.budget() > 0 {
            self.ops.push(op(&["EAST"]));
            ax += 1;
        }
        while ax > bx && self.budget() > 0 {
            self.ops.push(op(&["WEST"]));
            ax -= 1;
        }
        while ay < by && self.budget() > 0 {
            self.ops.push(op(&["SOUTH"]));
            ay += 1;
        }
        while ay > by && self.budget() > 0 {
            self.ops.push(op(&["NORTH"]));
            ay -= 1;
        }
        self.pos = (ax, ay);
    }
    fn act(&mut self, o: Op) -> bool {
        if self.budget() <= 0 {
            return false;
        }
        if o[0] == "HARVEST" || o[0] == "COLLECT_FERTILIZER" {
            self.loaded = true;
        }
        self.ops.push(o);
        true
    }
}

fn cost(pos: (i64, i64), tile: (i64, i64), ops: &[Op]) -> i64 {
    (pos.0 - tile.0).abs() + (pos.1 - tile.1).abs() + ops.len() as i64
}

/// Quadrant index of a tile/pos: `_q` in build_econ_agent.py. NW=0, NE=1,
/// SW=2, SE=3, matching the shed-corner spawn of the four quadrant crews.
fn q_of(p: (i64, i64)) -> usize {
    (if p.1 < 5 { 0 } else { 2 }) + (if p.0 >= 5 { 1 } else { 0 })
}

// ------------------------------------------------------------------ tasks --

struct Task {
    tile: (i64, i64),
    ops: Vec<Op>,
    prio: i64,
    /// Shed withdrawals this task needs. Keys are item names, `AN_<species>`
    /// for an animal, or `SEED_<crop>` (which is a market order, not a
    /// withdrawal, and is therefore excluded from every pickup).
    needs: Vec<(String, i64)>,
}

impl Task {
    fn wants_shed(&self) -> bool {
        self.needs.iter().any(|(k, _)| !k.starts_with("SEED_"))
    }
    fn wants_seed(&self) -> bool {
        self.needs.iter().any(|(k, _)| k.starts_with("SEED_"))
    }
}

// ------------------------------------------------------------------- rows --

// Row / row_json are the track-neutral shell vocabulary; they live in
// crate::core so the bandit lane can share them without depending on this
// economy policy (G0.1).
pub use crate::core::{row_json, Row};

// ----------------------------------------------------------------- helpers --

fn omap(v: &Json) -> BTreeMap<String, i64> {
    let mut m = BTreeMap::new();
    for (k, val) in v.obj() {
        m.insert(k.clone(), val.i64());
    }
    m
}

fn g(m: &BTreeMap<String, i64>, k: &str) -> i64 {
    *m.get(k).unwrap_or(&0)
}

/// Cumulative day-indexed cap: the last `(day, n)` whose day <= d.
fn sched(pairs: &[(i64, i64)], d: i64) -> i64 {
    let mut n = 0;
    for (day, v) in pairs {
        if d >= *day {
            n = *v;
        }
    }
    n
}

/// What stands on one tile, as the planner needs to see it.
enum TileKind<'a> {
    Locked,
    Empty,
    Weed,
    Plant(&'a Json),
    Pen(&'static str),
    Animal(&'a Json),
    Other,
}

fn classify(tl: &Json) -> TileKind<'_> {
    match tl {
        Json::Null => TileKind::Empty,
        Json::Str(s) if s == "LOCKED" => TileKind::Locked,
        Json::Obj(_) => {
            if !tl.get("animal").is_null() {
                TileKind::Animal(tl)
            } else {
                match tl.get("kind").str() {
                    "PLANT" => TileKind::Plant(tl),
                    "WEED" => TileKind::Weed,
                    "PASTURE" => TileKind::Pen("PASTURE"),
                    "COOP" => TileKind::Pen("COOP"),
                    _ => TileKind::Other,
                }
            }
        }
        _ => TileKind::Other,
    }
}

/// The `_pick()` closure: which species a newly available tile should become.
struct Picker<'a> {
    gen: &'a Genome,
    d: i64,
    want: BTreeMap<String, i64>,
    purse: f64,
}

impl Picker<'_> {
    fn open(&self, c: &str) -> bool {
        let (lo, hi) = self.gen.window(c);
        lo <= self.d && self.d <= hi
    }
    fn pick(&mut self) -> Option<String> {
        for c in PLANT_ORDER {
            let tgt = self.gen.tile_target(c);
            if tgt <= 0 || !self.open(c) || g(&self.want, c) >= tgt {
                continue;
            }
            let cd = crop_def(c);
            if self.d + cd.first > LAST_DAY {
                continue;
            }
            let floor = if cd.seed >= 80.0 { self.gen.floor_straw } else { self.gen.floor_seed };
            if self.purse < cd.seed + floor {
                continue;
            }
            self.purse -= cd.seed;
            *self.want.entry(c.to_string()).or_insert(0) += 1;
            return Some(c.to_string());
        }
        let cd = crop_def("WHEAT");
        if self.open("WHEAT")
            && self.d + cd.first <= LAST_DAY
            && self.purse >= cd.seed + self.gen.floor_seed
        {
            self.purse -= cd.seed;
            *self.want.entry("WHEAT".to_string()).or_insert(0) += 1;
            return Some("WHEAT".to_string());
        }
        None
    }
}

// ------------------------------------------------------------------- plan --

pub struct Planner {
    pub gen: &'static Genome,
    order: Vec<(i64, i64)>,
    tidx: BTreeMap<(i64, i64), usize>,
    day: i64,
    plan: Option<Vec<Row>>,
}

impl Planner {
    pub fn new(gen: &'static Genome) -> Self {
        let order = tile_order();
        let tidx = order.iter().enumerate().map(|(i, t)| (*t, i)).collect();
        Planner { gen, order, tidx, day: -1, plan: None }
    }

    pub fn reset(&mut self) {
        self.day = -1;
        self.plan = None;
    }

    fn ti(&self, t: (i64, i64)) -> usize {
        *self.tidx.get(&t).unwrap_or(&0)
    }

    #[allow(clippy::too_many_lines)]
    fn build(&self, d: i64, farm: &Json, priv_: &Json, mkt: &Json) -> Vec<Row> {
        let gen = self.gen;
        let mut money = farm.get("money").f64();
        let tiles = farm.get("tiles");
        let shed = omap(priv_.get("shed"));
        let seeds = omap(priv_.get("seeds"));
        let owned: BTreeSet<String> = {
            let q: Vec<String> = farm
                .get("unlocked_quadrants")
                .arr()
                .iter()
                .map(|v| v.str().to_string())
                .collect();
            if q.is_empty() { ["NW".to_string()].into_iter().collect() } else { q.into_iter().collect() }
        };
        let prices = omap(mkt.get("prices"));
        let sell_all = d >= LAST_DAY - 1;
        let final_day = d >= LAST_DAY;

        fn tl_at<'j>(tiles: &'j Json, t: (i64, i64)) -> &'j Json {
            tiles.idx(t.1 as usize).idx(t.0 as usize)
        }

        // ---- census the REAL board -------------------------------------
        let mut animals: Vec<((i64, i64), &Json)> = Vec::new();
        let mut pens: Vec<((i64, i64), &'static str)> = Vec::new();
        let mut plants: Vec<((i64, i64), &Json)> = Vec::new();
        let mut weeds: Vec<(i64, i64)> = Vec::new();
        let mut empties: Vec<(i64, i64)> = Vec::new();
        let mut n_an: BTreeMap<String, i64> = BTreeMap::new();
        let mut n_cr: BTreeMap<String, i64> = BTreeMap::new();
        for t in &self.order {
            let t = *t;
            if !owned.contains(quad(t)) {
                continue;
            }
            match classify(tl_at(tiles, t)) {
                TileKind::Locked | TileKind::Other => continue,
                TileKind::Empty => empties.push(t),
                TileKind::Animal(tl) => {
                    animals.push((t, tl));
                    *n_an.entry(tl.get("animal").str().to_string()).or_insert(0) += 1;
                }
                TileKind::Plant(tl) => {
                    plants.push((t, tl));
                    let c = tl.get("crop").str();
                    let c = if c.is_empty() { "WHEAT" } else { c };
                    *n_cr.entry(c.to_string()).or_insert(0) += 1;
                }
                TileKind::Weed => weeds.push(t),
                TileKind::Pen(kind) => pens.push((t, kind)),
            }
        }

        // Animals bought on an earlier day sit in the shed: THEY are today's
        // placements. Animals bought today land in the shed after the unit
        // turn, so they are placed (and fed, and cared for) tomorrow.
        let mut pen_need: BTreeMap<String, i64> = BTreeMap::new();
        for sp in SPECIES {
            pen_need.insert(sp.to_string(), g(&shed, sp));
        }

        let mut late: Vec<(i64, Op)> = Vec::new();

        // ---- hires: the whole crew in the hour-0 batch ------------------
        let n_plan = {
            let i = (d.max(0) as usize).min(gen.hires.len() - 1);
            (gen.hires[i].max(0) as usize).min(MAX_HIRE)
        };
        let mut fib: Vec<f64> = vec![1.0, 1.0];
        while fib.len() < n_plan.max(2) {
            let n = fib[fib.len() - 1] + fib[fib.len() - 2];
            fib.push(n);
        }
        let mut n_h = 0usize;
        let mut spend = 0.0f64;
        for i in 0..n_plan {
            let c = fib[i.min(fib.len() - 1)];
            if money - spend - c < 40.0 {
                break;
            }
            spend += c;
            n_h += 1;
        }
        money -= spend;

        // ---- land: 2 buys, on the schedule, cash-gated ------------------
        // land_res is the cost of the NEXT scheduled quadrant, withheld from
        // the herd until it is bought. Tiles, not animals, are the binding
        // constraint early: NW holds 24 usable tiles and the skeleton wants
        // 243 plants.
        let mut land_res = 0.0f64;
        let n_extra = owned.len() - 1;
        if n_extra < 3 && !sell_all {
            let nq = LAND_ORDER[n_extra];
            let ld = gen.land.iter().find(|(k, _)| *k == nq).map(|(_, v)| *v).unwrap_or(-1);
            let need_land = LAND_COST[n_extra] + gen.floor_land;
            if ld >= 0 && d >= ld && money >= need_land {
                late.push((1, op(&["BUY_LAND"])));
                money -= LAND_COST[n_extra];
            } else if ld >= 0 && d >= ld - gen.land_reserve_lead {
                land_res = need_land;
            }
        }

        // ---- what a newly available tile should become ------------------
        let mut pick = Picker { gen, d, want: n_cr.clone(), purse: money };

        // ---- tasks from what STANDS on the board ------------------------
        let mut tasks: Vec<Task> = Vec::new();
        let mut fert_left = g(&shed, "FERTILIZER");
        for (t, tl) in &animals {
            let mut ops: Vec<Op> = Vec::new();
            let mut need: Vec<(String, i64)> = Vec::new();
            if !tl.get("fed_today").bool() && !final_day {
                ops.push(op(&["FEED"]));
                need.push(("WHEAT".into(), 1));
            }
            if tl.get("yield_units").i64() > 0 {
                ops.push(op(&["HARVEST"]));
            }
            if tl.get("fertilizer_available").bool() {
                ops.push(op(&["COLLECT_FERTILIZER"]));
            }
            if gen.care && !tl.get("cared_today").bool() && !final_day {
                ops.push(op(&["CARE"]));
            }
            if !ops.is_empty() {
                tasks.push(Task { tile: *t, ops, prio: 0, needs: need });
            }
        }
        for (t, tl) in &plants {
            let crop = {
                let c = tl.get("crop").str();
                if c.is_empty() { "WHEAT".to_string() } else { c.to_string() }
            };
            let c = crop_def(&crop);
            let age = d - tl.get("planted_day").i64();
            let yld = tl.get("yield_units").i64();
            let mut ops: Vec<Op> = Vec::new();
            let mut need: Vec<(String, i64)> = Vec::new();
            if c.ong {
                let last = c.first + c.iv * (c.maxy - 1);
                if yld > 0 && age >= c.first {
                    ops.push(op(&["HARVEST"]));
                }
                if age < last && !final_day {
                    if crop == gen.fert_crop
                        && fert_left > 0
                        && tl.get("fertilized_until_day").i64() < d
                        && age >= c.first - 1
                        && (age - (c.first - 1)).rem_euclid((2 * c.iv).max(1)) == 0
                    {
                        ops.push(op(&["FERTILIZE"]));
                        need.push(("FERTILIZER".into(), 1));
                        fert_left -= 1;
                    }
                    if !tl.get("watered_today").bool() {
                        ops.push(op(&["WATER"]));
                    }
                } else if age >= last && ops.is_empty() && !sell_all {
                    ops.push(op(&["DIG"]));
                    if let Some(nxt) = pick.pick() {
                        ops.push(op(&["PLANT", &nxt]));
                        ops.push(op(&["WATER"]));
                        need.push((format!("SEED_{nxt}"), 1));
                    }
                }
            } else {
                let ripe = age >= c.maxd || (yld >= c.maxy && age >= c.first);
                if final_day || sell_all {
                    if yld > 0 && age >= c.first {
                        ops.push(op(&["HARVEST"]));
                    } else if !final_day && !tl.get("watered_today").bool() {
                        ops.push(op(&["WATER"]));
                    }
                } else if ripe && yld > 0 {
                    ops.push(op(&["HARVEST"]));
                    if let Some(nxt) = pick.pick() {
                        ops.push(op(&["PLANT", &nxt]));
                        ops.push(op(&["WATER"]));
                        need.push((format!("SEED_{nxt}"), 1));
                    }
                } else if ripe {
                    ops.push(op(&["DIG"]));
                } else if !tl.get("watered_today").bool() {
                    ops.push(op(&["WATER"]));
                }
            }
            if !ops.is_empty() {
                tasks.push(Task { tile: *t, ops, prio: 1, needs: need });
            }
        }

        if !sell_all {
            for t in &weeds {
                tasks.push(Task { tile: *t, ops: vec![op(&["DIG"])], prio: 3, needs: vec![] });
            }
            for (t, kind) in &pens {
                for sp in SPECIES {
                    if animal_pen(sp) == *kind && g(&pen_need, sp) > 0 {
                        *pen_need.get_mut(sp).unwrap() -= 1;
                        let mut ops = vec![op(&["PLACE", sp]), op(&["FEED"])];
                        if gen.care {
                            ops.push(op(&["CARE"]));
                        }
                        tasks.push(Task {
                            tile: *t,
                            ops,
                            prio: 2,
                            needs: vec![(format!("AN_{sp}"), 1), ("WHEAT".into(), 1)],
                        });
                        break;
                    }
                }
            }
            // Tiles the herd will need SOON are held back from the plough: a
            // day-0 board planted wall to wall left every bought animal
            // rotting in the shed until a crop cycle freed a tile.
            let mut soon = 0;
            for sp in SPECIES {
                soon += sched(gen.herd(sp), d + gen.pen_lookahead);
            }
            let pending: i64 = pen_need.values().sum();
            let mut held =
                (soon - (animals.len() as i64 + pens.len() as i64 + pending)).max(0);
            let mut free = empties.clone();
            free.sort_by_key(|t| (shed_dist(*t), self.ti(*t)));
            for t in free {
                let mut placed = false;
                for sp in SPECIES {
                    if g(&pen_need, sp) > 0 {
                        *pen_need.get_mut(sp).unwrap() -= 1;
                        let mut ops =
                            vec![op(&[animal_build(sp)]), op(&["PLACE", sp]), op(&["FEED"])];
                        if gen.care {
                            ops.push(op(&["CARE"]));
                        }
                        tasks.push(Task {
                            tile: t,
                            ops,
                            prio: 2,
                            needs: vec![(format!("AN_{sp}"), 1), ("WHEAT".into(), 1)],
                        });
                        placed = true;
                        break;
                    }
                }
                if placed {
                    continue;
                }
                if held > 0 {
                    held -= 1;
                    continue;
                }
                if let Some(nxt) = pick.pick() {
                    tasks.push(Task {
                        tile: t,
                        ops: vec![op(&["PLANT", &nxt]), op(&["WATER"])],
                        prio: 3,
                        needs: vec![(format!("SEED_{nxt}"), 1)],
                    });
                }
            }
        }

        // ---- herd ramp, from what the CROPS left behind -----------------
        // Capital priority is crops > animals > feed: with the herd funded
        // first, dawn money pinned at $41 through day 11 and a farm with no
        // seed money never recovers.
        money = pick.purse;
        let mut bought: BTreeMap<String, i64> = BTreeMap::new();
        if !sell_all {
            for sp in SPECIES {
                let cap = sched(gen.herd(sp), d);
                let mut have = g(&n_an, sp) + g(&shed, sp);
                let c = animal_cost(sp);
                let floor = gen.floor_animal + land_res;
                while have < cap && money >= c + floor {
                    late.push((2, op(&["BUY_ANIMAL", sp, "1"])));
                    money -= c;
                    *bought.entry(sp.to_string()).or_insert(0) += 1;
                    have += 1;
                }
            }
        }

        // ---- tomorrow's feed, bought today so the dawn pickup finds it --
        let n_beasts = animals.len() as i64;
        let beasts_tomorrow = n_beasts
            + SPECIES.iter().map(|s| g(&shed, s)).sum::<i64>()
            + bought.values().sum::<i64>();
        let reserve_w = if beasts_tomorrow > 0 && !final_day {
            beasts_tomorrow + gen.feed_buffer
        } else {
            0
        };

        // ---- the crew (every hand hired at hour 0 -> deterministic spawn)
        let mut units: Vec<Unit> = vec![Unit::new(SHED[0], 1, (TPD - 1) as usize)];
        for k in 0..n_h {
            units.push(Unit::new(SHED[(k + 1) % 4], 1, (TPD - 1) as usize));
        }

        // ---- pace: expansion never starves the standing farm ------------
        let cap_ops: i64 = units.iter().map(|u| u.cap as i64).sum();
        let tend: i64 = tasks
            .iter()
            .filter(|t| t.prio <= 2)
            .map(|t| t.ops.len() as i64 + 2)
            .sum();
        let room = ((cap_ops as f64 * gen.plant_pace) as i64 - tend).max(0);
        let mut keep: Vec<Task> = Vec::new();
        let mut grow: Vec<Task> = Vec::new();
        for tk in tasks {
            if tk.prio >= 3 && tk.wants_seed() {
                grow.push(tk);
            } else {
                keep.push(tk);
            }
        }
        let mut used = 0i64;
        for tk in grow {
            let c = tk.ops.len() as i64 + 2;
            if used + c > room {
                continue;
            }
            used += c;
            keep.push(tk);
        }
        let mut tasks = keep;
        tasks.sort_by_key(|t| self.ti(t.tile));

        // ---- assign: contiguous tours, shed withdrawals batched up front
        let mut shed_pool: Vec<Task> = Vec::new();
        let mut field_pool: Vec<Task> = Vec::new();
        for tk in tasks {
            if tk.wants_shed() {
                shed_pool.push(tk);
            } else {
                field_pool.push(tk);
            }
        }
        let mut picked: BTreeMap<String, i64> = BTreeMap::new();

        // QUADRANT-MATCHED ASSIGNMENT (port of build_econ_agent.py `_take`/`uq`).
        // Units spawn at the four shed corners (one per quadrant) and TILE_ORDER
        // is a per-quadrant boustrophedon from that corner, so giving each unit
        // ONLY its own quadrant's tasks turns every tour into a tight local
        // sweep from the spawn corner instead of a hand walking across the board
        // to a chunk that happened to fall at its pool position. `uq` groups unit
        // INDICES by quadrant; each `take` call restarts its own cursor over that
        // quadrant's units, so a unit gets its shed tasks (pickup pass) and then
        // its field tasks (field pass).
        let mut uq: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, u) in units.iter().enumerate() {
            uq[q_of(u.pos)].push(i);
        }
        let mut shed_q: [Vec<Task>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for tk in shed_pool {
            let q = q_of(tk.tile);
            shed_q[q].push(tk);
        }
        let mut field_q: [Vec<Task>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for tk in field_pool {
            let q = q_of(tk.tile);
            field_q[q].push(tk);
        }
        for q in 0..4 {
            let qpool = std::mem::take(&mut shed_q[q]);
            take(&mut units, &uq[q], qpool, true, &shed, &mut picked);
        }
        let mut field_left: Vec<Task> = Vec::new();
        for q in 0..4 {
            let qpool = std::mem::take(&mut field_q[q]);
            let mut leftover = take(&mut units, &uq[q], qpool, false, &shed, &mut picked);
            field_left.append(&mut leftover);
        }
        // global sweep: any unit with spare budget takes the nearest leftover
        for u in units.iter_mut() {
            while !field_left.is_empty() && u.budget() > 0 {
                field_left.sort_by_key(|tk| cost(u.pos, tk.tile, &tk.ops));
                if u.budget() < cost(u.pos, field_left[0].tile, &field_left[0].ops) {
                    break;
                }
                let tk = field_left.remove(0);
                u.walk(tk.tile);
                for o in tk.ops {
                    u.act(o);
                }
            }
        }
        if sell_all {
            // bank the day's take before dusk
            for u in units.iter_mut() {
                if u.loaded && u.budget() > 1 {
                    let tgt = *SHED
                        .iter()
                        .min_by_key(|s| (u.pos.0 - s.0).abs() + (u.pos.1 - s.1).abs())
                        .unwrap();
                    u.walk(tgt);
                    u.act(op(&["DROP"]));
                }
            }
        }

        // ---- seed: exactly the plantings labour actually scheduled ------
        let mut planted: BTreeMap<String, i64> = BTreeMap::new();
        for u in &units {
            for o in &u.ops {
                if o[0] == "PLANT" && o.len() > 1 {
                    *planted.entry(o[1].clone()).or_insert(0) += 1;
                }
            }
        }
        let mut seed_orders: Vec<Op> = Vec::new();
        for (sp, n) in &planted {
            let n = n - g(&seeds, sp);
            if n > 0 {
                seed_orders.push(op(&["BUY_SEED", sp, &n.to_string()]));
            }
        }

        // ---- feed wheat for TOMORROW ------------------------------------
        let short = reserve_w - (g(&shed, "WHEAT") - g(&picked, "WHEAT"));
        if short > 0 {
            late.push((4, op(&["BUY_PRODUCT", "WHEAT", &short.to_string()])));
        }

        // ---- sells: on production, above the reserves -------------------
        let mut sold: BTreeMap<String, i64> = BTreeMap::new();
        for item in SELLABLE {
            let mut have = g(&shed, item) - g(&picked, item);
            if item == "WHEAT" && !sell_all {
                have -= reserve_w;
            } else if item == "FERTILIZER" && !sell_all {
                have -= gen.fert_keep;
            }
            if have <= 0 {
                continue;
            }
            if sell_all {
                sold.insert(item.to_string(), have);
                continue;
            }
            let _p = *prices.get(item).unwrap_or(&(base_price(item) as i64));
            let q = have.min(gen.sell_cap(item));
            if q > 0 {
                sold.insert(item.to_string(), q);
            }
        }
        // pressure valve: the shed caps at 100 TOTAL, dusk overflow is DROPPED
        let mut total: i64 = shed.values().sum::<i64>()
            - picked.values().sum::<i64>()
            - sold.values().sum::<i64>();
        if !sell_all && total > gen.valve_hi {
            for item in VALVE_ORDER {
                let mut room2 = g(&shed, item) - g(&picked, item) - g(&sold, item);
                if item == "WHEAT" {
                    room2 -= reserve_w;
                }
                let take_n = room2.min(total - gen.valve_lo);
                if take_n > 0 {
                    *sold.entry(item.to_string()).or_insert(0) += take_n;
                    total -= take_n;
                }
                if total <= gen.valve_lo {
                    break;
                }
            }
        }

        // ---- serialise the market queue (hires FIRST: hour-0 batch) -----
        let mut queue: Vec<Op> = vec![op(&["HIRE"]); n_h];
        queue.extend(seed_orders);
        late.sort_by_key(|(r, _)| *r);
        for (_, o) in late {
            queue.push(o);
        }
        for (item, n) in &sold {
            if *n > 0 {
                queue.push(op(&["SELL", item, &n.to_string()]));
            }
        }

        let mut plan: Vec<Row> = Vec::with_capacity(TPD as usize);
        for h in 0..TPD as usize {
            let lo = (h * 10).min(queue.len());
            let hi = ((h + 1) * 10).min(queue.len());
            let market = queue[lo..hi].to_vec();
            let u0 = &units[0];
            let farmer = if h as i64 >= u0.start
                && ((h as i64 - u0.start) as usize) < u0.ops.len()
            {
                u0.ops[(h as i64 - u0.start) as usize].clone()
            } else {
                op(&["PASS"])
            };
            let hands = units[1..]
                .iter()
                .map(|u| {
                    if h as i64 >= u.start && ((h as i64 - u.start) as usize) < u.ops.len() {
                        u.ops[(h as i64 - u.start) as usize].clone()
                    } else {
                        op(&["PASS"])
                    }
                })
                .collect();
            plan.push(Row { farmer, hands, market });
        }
        plan
    }

    /// Market-only refresh: whatever the shed holds right now, at the live
    /// price. Oversized SELLs partial-fill, so an overlap with the dawn queue
    /// costs a queue slot at most.
    fn fresh_sells(&self, d: i64, farm: &Json, priv_: &Json) -> Vec<Op> {
        let gen = self.gen;
        let shed = omap(priv_.get("shed"));
        let mut n_beasts: i64 = SPECIES.iter().map(|s| g(&shed, s)).sum();
        for row in farm.get("tiles").arr() {
            for tl in row.arr() {
                if tl.is_obj() && !tl.get("animal").is_null() {
                    n_beasts += 1;
                }
            }
        }
        let sell_all = d >= LAST_DAY - 1;
        let reserve_w = if n_beasts == 0 || sell_all { 0 } else { n_beasts + gen.feed_buffer };
        let mut out = Vec::new();
        for item in SELLABLE {
            let mut have = g(&shed, item);
            if item == "WHEAT" && !sell_all {
                have -= reserve_w;
            } else if item == "FERTILIZER" && !sell_all {
                have -= gen.fert_keep;
            }
            if have <= 0 {
                continue;
            }
            let q = if sell_all { have } else { have.min(gen.sell_cap(item)) };
            if q > 0 {
                out.push(op(&["SELL", item, &q.to_string()]));
            }
        }
        out
    }

    /// One turn. `obs` is the official observation; the return is the action
    /// in the interpreter's schema.
    pub fn act(&mut self, obs: &Json) -> Row {
        let (d, h) = {
            let dj = obs.get("day");
            let hj = obs.get("hour");
            if dj.is_null() || hj.is_null() {
                let step = obs.get("step").i64();
                (step / TPD, step % TPD)
            } else {
                (dj.i64(), hj.i64())
            }
        };
        let me = obs.get("player").i64().max(0) as usize;
        let farm = obs.get("farms").idx(me);
        let priv_ = obs.get("private");
        let mkt = obs.get("market");

        if self.day != d || self.plan.is_none() {
            self.plan = Some(self.build(d, farm, priv_, mkt));
            self.day = d;
        }
        let plan = self.plan.as_ref().unwrap();
        let base = plan
            .get(h.max(0) as usize)
            .cloned()
            .unwrap_or(Row { farmer: op(&["PASS"]), hands: vec![], market: vec![] });

        let mut market = base.market;
        if h > 0 {
            let mut keep: Vec<Op> =
                market.into_iter().filter(|o| !o.is_empty() && o[0] != "SELL").collect();
            keep.extend(self.fresh_sells(d, farm, priv_));
            market = keep;
        }
        market.truncate(10);

        let real = farm.get("hands").arr().len();
        let mut hands = base.hands;
        while hands.len() < real {
            hands.push(op(&["PASS"]));
        }
        hands.truncate(real);

        Row { farmer: base.farmer, hands, market }
    }
}

/// `_take`: hand contiguous chunks of `pool` to successive units in `ul` (unit
/// INDICES, one quadrant's crew), with every shed withdrawal for a chunk batched
/// into ONE pickup at the front. The cursor restarts at 0 each call, so the same
/// unit serves its quadrant's pickup pass and then its field pass. Tasks the
/// crew could not fit are returned for the global nearest-first sweep.
///
/// PICKUP executes at the unit's CURRENT tile, so only a unit still standing on
/// a shed tile can withdraw -- a per-tile pickup mid-tour is a silent no-op.
/// That bug once made most FERTILIZEs never fire.
fn take(
    units: &mut [Unit],
    ul: &[usize],
    mut pool: Vec<Task>,
    pickup: bool,
    shed: &BTreeMap<String, i64>,
    picked: &mut BTreeMap<String, i64>,
) -> Vec<Task> {
    let mut ui = 0usize;
    while !pool.is_empty() && ui < ul.len() {
        let uidx = ul[ui];
        ui += 1;
        let (mut pos, mut budget) = {
            let u = &units[uidx];
            (u.pos, u.budget())
        };
        if budget <= 0 {
            continue;
        }
        let mut chunk: Vec<Task> = Vec::new();
        let mut items: BTreeSet<String> = BTreeSet::new();
        while !pool.is_empty() {
            let tk = &pool[0];
            let mut new_items = items.clone();
            if pickup {
                for (k, _) in &tk.needs {
                    if !k.starts_with("SEED_") {
                        new_items.insert(k.clone());
                    }
                }
            }
            let extra = if pickup { (new_items.len() - items.len()) as i64 } else { 0 };
            let c = cost(pos, tk.tile, &tk.ops) + extra;
            if budget - c < 0 {
                break;
            }
            let tk = pool.remove(0);
            pos = tk.tile;
            chunk.push(tk);
            items = new_items;
            budget -= c;
        }
        if chunk.is_empty() {
            continue;
        }
        let u = &mut units[uidx];
        if pickup && !items.is_empty() {
            let mut tot: BTreeMap<String, i64> = BTreeMap::new();
            for tk in &chunk {
                for (k, v) in &tk.needs {
                    if k.starts_with("SEED_") {
                        continue;
                    }
                    let it = k.strip_prefix("AN_").unwrap_or(k).to_string();
                    *tot.entry(it).or_insert(0) += v;
                }
            }
            for (it, want) in &tot {
                let avail = g(shed, it) - g(picked, it);
                let n = (*want).min(avail.max(0));
                if n > 0 {
                    u.act(op(&["PICKUP", it, &n.to_string()]));
                    *picked.entry(it.clone()).or_insert(0) += n;
                }
            }
        }
        for tk in chunk {
            u.walk(tk.tile);
            for o in tk.ops {
                u.act(o);
            }
        }
    }
    pool
}

// ------------------------------------------------------------------ output --

// json_op / row_json now live in crate::core (re-exported above).

// -------------------------------------------------------------------- play --

/// `kagg play`: one observation JSON per line in, one action JSON per line out.
///
/// Every failure path answers with a LEGAL action rather than dying: a bad line
/// yields `{"farmer": ["PASS"], "hands": [...], "market": []}` sized to the
/// hand count we can still read, and the process stays up. The Python bridge
/// has its own fallback on top of that, but a searcher that kills its own
/// process on one malformed byte is not shippable.
pub fn play() {
    play_with(0)
}

/// `kagg play --budget-ms N`. `budget == 0` is the Phase-A skeleton path,
/// unchanged; anything larger routes the turn through the Phase-B searcher.
pub fn play_with(budget_ms: u64) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut p = Planner::new(load_genome());
    let mut sr: Option<crate::search::Searcher> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        if line == "RESETP" {
            p.reset();
            sr = None;
            writeln!(out, "{{\"ok\": true}}").ok();
            out.flush().ok();
            continue;
        }
        let resp = match parse(line) {
            Ok(obs) => {
                if budget_ms == 0 {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || p.act(&obs),
                    ));
                    match r {
                        Ok(row) => row_json(&row),
                        Err(_) => {
                            p.reset();
                            pass_row(&obs)
                        }
                    }
                } else {
                    search_turn(&obs, budget_ms, &mut sr, &mut p)
                }
            }
            Err(_) => "{\"farmer\": [\"PASS\"], \"hands\": [], \"market\": []}".to_string(),
        };
        if writeln!(out, "{resp}").is_err() {
            break;
        }
        if out.flush().is_err() {
            break;
        }
    }
}

/// One searched turn, with every failure answered by something legal.
///
/// Order of degradation, cheapest correct answer first:
///   1. the searcher;
///   2. on a panic inside it -- the searcher is dropped (its committed plan may
///      be half-built) and the turn is answered by the SKELETON planner, which
///      reads the raw observation and needs no reconstruction;
///   3. on an observation we cannot reconstruct into a `State` -- the skeleton
///      planner again;
///   4. on a panic in that too -- a legal all-PASS turn sized to the hand count.
/// The Python bridge's watchdog, validator and pure-Python fallback sit under
/// all four.
fn search_turn(
    obs: &Json,
    budget_ms: u64,
    sr: &mut Option<crate::search::Searcher>,
    p: &mut Planner,
) -> String {
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::obsstate::from_obs(obs)
    }));
    if let Ok(Some((mut st, me))) = built {
        crate::search::seed_opponent_belief(&mut st, me);
        // One searcher per episode. A step that has gone BACKWARDS means a new
        // episode on the same process (the harness reuses one), so rebuild.
        let fresh = match sr.as_ref() {
            None => true,
            Some(s) => s.me != me || st.step == 0,
        };
        if fresh {
            *sr = Some(crate::search::Searcher::new(me));
        }
        let s = sr.as_mut().unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.decide(&st, me, budget_ms)
        }));
        match r {
            Ok(a) => return action_json(&a),
            Err(_) => {
                *sr = None;
            }
        }
    }
    // Fall through to the skeleton; it parses the observation directly.
    p.reset();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.act(obs))) {
        Ok(row) => row_json(&row),
        Err(_) => pass_row(obs),
    }
}

/// A searched `PlayerAction` in the wire format the bridge validates.
///
/// `n` is emitted only when the op actually carries one (`has_n`), so a bare
/// `PLANT WHEAT` does not acquire a spurious count that the validator would
/// then have to strip.
fn action_json(a: &crate::engine::PlayerAction) -> String {
    fn unit(u: &crate::engine::UnitAction) -> String {
        let mut toks: Vec<String> = vec![format!("\"{}\"", u.op)];
        if !u.item.is_empty() {
            toks.push(format!("\"{}\"", u.item));
            if u.has_n {
                toks.push(format!("\"{}\"", u.n));
            }
        }
        format!("[{}]", toks.join(", "))
    }
    let hands: Vec<String> = a.hands.iter().map(unit).collect();
    let market: Vec<String> = a
        .market
        .iter()
        .map(|o| {
            let t: Vec<String> = o.iter().map(|x| format!("\"{x}\"")).collect();
            format!("[{}]", t.join(", "))
        })
        .collect();
    format!(
        "{{\"farmer\": {}, \"hands\": [{}], \"market\": [{}]}}",
        unit(&a.farmer),
        hands.join(", "),
        market.join(", ")
    )
}

fn pass_row(obs: &Json) -> String {
    let me = obs.get("player").i64().max(0) as usize;
    let n = obs.get("farms").idx(me).get("hands").arr().len();
    let hands: Vec<String> = (0..n).map(|_| "[\"PASS\"]".to_string()).collect();
    format!(
        "{{\"farmer\": [\"PASS\"], \"hands\": [{}], \"market\": []}}",
        hands.join(", ")
    )
}

/// `kagg play-selftest`: build one plan from a synthetic day-0 observation and
/// print it. Cheap smoke test for a fresh build, needing no Python.
pub fn play_selftest() {
    let st = crate::state::State::new(1117071212);
    let js = crate::service::json_state(&st, false);
    let full = parse(&js).expect("service json must parse");
    // Reshape the full-state emission into the per-seat observation.
    let obs = format!(
        "{{\"step\": {}, \"day\": 0, \"hour\": 0, \"player\": 0, \"farms\": {}, \
         \"market\": {}, \"town\": {}, \"private\": {}}}",
        0,
        emit(full.get("farms")),
        emit(full.get("market")),
        emit(full.get("town")),
        emit(full.get("private").idx(0))
    );
    let parsed = parse(&obs).expect("observation must parse");
    let mut p = Planner::new(load_genome());
    let r = p.act(&parsed);
    println!("{}", row_json(&r));
}

/// Re-emit a parsed Json value (selftest only; the agent never serialises obs).
fn emit(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Json::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        Json::Arr(a) => {
            format!("[{}]", a.iter().map(emit).collect::<Vec<_>>().join(", "))
        }
        Json::Obj(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, val)| format!("\"{k}\": {}", emit(val)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
