//! CONFIG-DRIVEN MULTI-CHECKPOINT bandit (the release-candidate harness).
//!
//! Same reactive shell as `bandit.rs` (weed-repair, cash-guard, escalation
//! latch, demand-scaled anti-dump sweep, quantity-conserved front-run ledger,
//! final dump) but driven by a MUTABLE route instead of a fixed egg/yarn tape:
//!
//!   * base scaffold route (a .tape), then
//!   * at each ACTIVE checkpoint (D6/D12/D15/D21/D27) it reads the CUMULATIVE
//!     unlocked-shop set, forms the sorted "|"-joined key, and splices in the
//!     searched branch's continuation (route[..cstep] ++ cont). Branches NEST.
//!
//! Everything is configurable from disk (no recompile to reship):
//!   kagg mbandit <config.json> <branches.json> <base.tape>
//! config.json = { checkpoints:[[step,day]...], guardrails:[{name,on,...knobs}] }.
//! Guard rails toggle on/off + tune via config; D6 dispatch and the endgame
//! dump are mandatory (enforced by the Python builder, honoured here as given).
//!
//! The proven `kagg bandit` baseline in bandit.rs is untouched -- this is a new
//! path, measured against that baseline before it can take a seat.

use crate::json::{parse, Json};
use crate::core::{row_json, Row};
use crate::market; // ENGINE-EXACT price model (game rule), not the r37 approximation
use crate::rules; // game cost constants (seed/animal/land/hire) for budget_guard
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};

const MARKET_CAP: usize = 10;
const SELL_PRIORITY: [&str; 7] =
    ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"];
const ALLP: [&str; 9] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
    "FERTILIZER",
];

type Op = Vec<String>;

// ---- _r37 marginal-price scarcity model (ported from pub_flexon_v45) --------
// (base, I0, T, below_func, below_target, above_func, above_target)
//
// G1.2 (config externalization, 2026-09-18): the per-item curve params used to
// be a hardcoded match. They are now DEFAULTS -- `config.tables.r37` may
// override any item -- so a price curve is a config edit, not a recompile. The
// defaults below are exactly the shipped v581 constants, so an absent tables
// section leaves v581 bit-identical.
#[derive(Clone)]
struct R37Curve {
    base: f64,
    i0: f64,
    t: f64,
    bf: String,
    bt: f64,
    af: String,
    at: f64,
}

fn r37_default(item: &str) -> Option<(f64, f64, f64, &'static str, f64, &'static str, f64)> {
    Some(match item {
        "WHEAT" => (25.0, 10000.0, 400.0, "sqrt", 0.8, "log", 0.2),
        "CARROT" => (35.0, 10000.0, 450.0, "hinge", 1.0, "sqrt", 0.7),
        "TOMATO" => (60.0, 10000.0, 200.0, "hinge", 0.4, "sqrt", 0.6),
        "STRAWBERRY" => (120.0, 10000.0, 100.0, "sqrt", 0.7, "linear", 1.6),
        "MELON" => (250.0, 10000.0, 300.0, "log", 0.2, "sq", 3.6),
        "EGG" => (50.0, 10000.0, 332.0, "hinge", 0.4, "log", 0.2),
        "MILK" => (160.0, 10000.0, 122.0, "sqrt", 0.6, "linear", 1.6),
        "WOOL" => (200.0, 10000.0, 105.0, "log", 0.2, "sq", 3.2),
        "FERTILIZER" => (100.0, 10000.0, 200.0, "linear", 0.4, "linear", 0.4),
        _ => return None,
    })
}

// Effective curve for `item`: a config override if present, else the default.
fn r37_curve(tables: &Tables, item: &str) -> Option<R37Curve> {
    if let Some(m) = &tables.r37 {
        if let Some(c) = m.get(item) {
            return Some(c.clone());
        }
    }
    let (base, i0, t, bf, bt, af, at) = r37_default(item)?;
    Some(R37Curve {
        base,
        i0,
        t,
        bf: bf.to_string(),
        bt,
        af: af.to_string(),
        at,
    })
}

fn r37_shape(func: &str, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match func {
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => (1.0 + x).ln(),
        "log10" => (1.0 + x).log10(),
        "hinge" => {
            if t <= 0.0 {
                x
            } else {
                let u = x / t;
                u + 8.0 * (u - 1.0).max(0.0).powi(2)
            }
        }
        _ => x, // linear + fallback
    }
}

fn r37_price(tables: &Tables, item: &str, inventory: i64) -> f64 {
    let c = match r37_curve(tables, item) {
        Some(p) => p,
        None => return 1.0,
    };
    let inv = inventory as f64;
    let price = if inv < c.i0 {
        let amp = c.bt * c.base / r37_shape(&c.bf, c.t, c.t);
        c.base + amp * r37_shape(&c.bf, c.i0 - inv, c.t)
    } else {
        let amp = c.at * c.base / r37_shape(&c.af, c.t, c.t);
        c.base - amp * r37_shape(&c.af, inv - c.i0, c.t)
    };
    price.round().max(1.0)
}

// Largest quantity (<= want) we can sell before the marginal price for `item`
// falls below hold_frac * base -- i.e. don't dump into a floored market.
fn r37_hold_cap(tables: &Tables, item: &str, market_inv: i64, want: i64, hold_frac: f64) -> i64 {
    let base = match r37_curve(tables, item) {
        Some(c) => c.base,
        None => return want,
    };
    let floor = hold_frac * base;
    let mut q = 0i64;
    while q < want && r37_price(tables, item, market_inv + q) >= floor {
        q += 1;
    }
    q
}

// ---- ENGINE-EXACT economics (game rule, market.rs) -------------------------
// The interpreter prices an item by the CURRENT market inventory: below I0
// (10000) it pays a scarcity premium (price > base); above I0 it gluts (price <
// base). Selling adds to inventory, so each extra unit we sell lowers the price
// for our own next unit AND for the opponent. Using the real engine (not the
// r37 flexon approximation) is what "understand the economics from the game
// rule" requires. Strategy KNOBS stay in config; the PARAMS are the game's rule.
fn eng_base(item: &str) -> f64 {
    market::param(item).map(|p| p.base).unwrap_or(1.0)
}
fn eng_price(item: &str, inv: i64) -> f64 {
    match market::param(item) {
        Some(p) => market::price(p, inv as f64) as f64,
        None => 1.0,
    }
}
/// Largest q (<= want) we can SELL while the engine's marginal price stays at or
/// above floor_frac * base. floor_frac ~1.0 = "supply to demand" (sell only the
/// scarcity premium, keep inventory near/under I0, never glut); a LOW floor_frac
/// = "dump" (accept the crash -- only justified when the opponent will dump
/// anyway and we must sell FIRST). Same mechanism, the floor picks the mode.
fn eng_cap(item: &str, market_inv: i64, want: i64, floor_frac: f64) -> i64 {
    if want <= 0 {
        return 0;
    }
    let floor = floor_frac * eng_base(item);
    let mut q = 0i64;
    while q < want && eng_price(item, market_inv + q) >= floor {
        q += 1;
    }
    q
}

// NN feature product order -- MUST match kaggriculture.bandit.nn.extract.PRODUCTS.
const NN_PRODUCTS: [&str; 9] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
];

fn op1(a: &str) -> Op {
    vec![a.to_string()]
}

// ------------------------------------------------------------- parsing --

fn parse_line(line: &str) -> Row {
    let mut parts = line.split('\t');
    let f = parts.next().unwrap_or("PASS");
    let h = parts.next().unwrap_or("");
    let m = parts.next().unwrap_or("");
    let toks = |s: &str| -> Op {
        s.split(' ').filter(|t| !t.is_empty()).map(|t| t.to_string()).collect()
    };
    let ops = |s: &str| -> Vec<Op> {
        if s.is_empty() {
            return vec![];
        }
        s.split(';').filter(|x| !x.is_empty()).map(toks).collect()
    };
    let farmer = {
        let v = toks(f);
        if v.is_empty() { op1("PASS") } else { v }
    };
    Row { farmer, hands: ops(h), market: ops(m) }
}

fn parse_tape(src: &str) -> Vec<Row> {
    src.lines().filter(|l| !l.trim().is_empty()).map(parse_line).collect()
}

/// One JSON token -> a Rust op token. Strings pass through; numeric args
/// (SELL/BUY quantities) render as their integer string, matching the tape.
fn tok(j: &Json) -> String {
    match j {
        Json::Str(s) => s.clone(),
        Json::Num(n) => (*n as i64).to_string(),
        Json::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

fn json_op(j: &Json) -> Op {
    j.arr().iter().map(tok).filter(|t| !t.is_empty()).collect()
}

fn json_row(j: &Json) -> Row {
    let farmer = {
        let v: Op = json_op(j.get("farmer"));
        if v.is_empty() { op1("PASS") } else { v }
    };
    let hands = j.get("hands").arr().iter().map(json_op).collect();
    let market = j.get("market").arr().iter().map(json_op).collect();
    Row { farmer, hands, market }
}

// --------------------------------------------------------------- config --

// A tiny 2-layer MLP (standardise -> Linear -> ReLU -> Linear -> optional
// sigmoid), weights carried INLINE in a guard's `nn` object so the policy ships
// by config alone. Trained offline on the GM corpus (bandit.nn.train_*); torch
// nn.Linear weight is [out, in], exported row-major and flattened here.
#[derive(Clone)]
// One dense layer: `w` is n_out * n_in row-major, bias `b` length n_out.
struct Layer {
    n_in: usize,
    n_out: usize,
    w: Vec<f64>,
    b: Vec<f64>,
}

// Arbitrary-DEPTH MLP. Hidden layers use ReLU; the final layer applies sigmoid
// (default) or is linear. Two config shapes are accepted:
//   * NEW (any depth): {"mu":..,"sd":..,"layers":[{"w":[[..]],"b":[..]}, ...],
//                       "sigmoid": true}
//   * LEGACY (2 layers): {"mu","sd","w1","b1","w2","b2","sigmoid"}  (still parses,
//                       so already-shipped configs keep working unchanged).
struct Mlp {
    n_in: usize,
    mu: Vec<f64>,
    sd: Vec<f64>,
    layers: Vec<Layer>,
    sigmoid: bool, // sigmoid on the FINAL layer's output
}

impl Mlp {
    fn parse(nn: &Json) -> Option<Mlp> {
        let flat = |j: &Json| -> Vec<f64> { j.arr().iter().map(|x| x.f64()).collect() };
        // returns (row-major weights, n_out=rows, n_in=cols)
        let mat = |j: &Json| -> (Vec<f64>, usize, usize) {
            let rows = j.arr();
            let n_out = rows.len();
            let n_in = if n_out > 0 { rows[0].arr().len() } else { 0 };
            let w: Vec<f64> = rows
                .iter()
                .flat_map(|r| r.arr().iter().map(|x| x.f64()).collect::<Vec<_>>())
                .collect();
            (w, n_out, n_in)
        };
        let mu = flat(nn.get("mu"));
        let sd = flat(nn.get("sd"));
        let n_in = mu.len();
        if n_in == 0 || sd.len() != n_in {
            return None;
        }
        let sigmoid = !matches!(nn.get("sigmoid"), Json::Bool(false)); // default true
        let mut layers: Vec<Layer> = Vec::new();
        let layer_list = nn.get("layers").arr();
        if !layer_list.is_empty() {
            for lj in layer_list {
                let (w, n_out, l_in) = mat(lj.get("w"));
                let b = flat(lj.get("b"));
                if n_out == 0 || l_in == 0 || b.len() != n_out || w.len() != n_out * l_in {
                    return None; // malformed -> rail becomes a no-op (safe)
                }
                layers.push(Layer { n_in: l_in, n_out, w, b });
            }
        } else {
            // legacy w1/b1/w2/b2 -> two layers
            let (w1, h, i1) = mat(nn.get("w1"));
            let b1 = flat(nn.get("b1"));
            let (w2, o, i2) = mat(nn.get("w2"));
            let b2 = flat(nn.get("b2"));
            if h == 0 || o == 0 || b1.len() != h || b2.len() != o
                || w1.len() != h * i1 || w2.len() != o * i2 {
                return None;
            }
            layers.push(Layer { n_in: i1, n_out: h, w: w1, b: b1 });
            layers.push(Layer { n_in: i2, n_out: o, w: w2, b: b2 });
        }
        // dimension chain: input -> L0 -> L1 -> ... must line up exactly
        if layers.is_empty() || layers[0].n_in != n_in {
            return None;
        }
        for k in 1..layers.len() {
            if layers[k].n_in != layers[k - 1].n_out {
                return None;
            }
        }
        Some(Mlp { n_in, mu, sd, layers, sigmoid })
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut z: Vec<f64> = (0..self.n_in).map(|i| (x[i] - self.mu[i]) / self.sd[i]).collect();
        let last = self.layers.len() - 1;
        for (li, layer) in self.layers.iter().enumerate() {
            let mut out = vec![0.0f64; layer.n_out];
            for k in 0..layer.n_out {
                let base = k * layer.n_in;
                let mut s = layer.b[k];
                for i in 0..layer.n_in {
                    s += layer.w[base + i] * z[i];
                }
                out[k] = if li < last {
                    if s > 0.0 { s } else { 0.0 } // ReLU on hidden layers
                } else if self.sigmoid {
                    1.0 / (1.0 + (-s).exp())
                } else {
                    s
                };
            }
            z = out;
        }
        z
    }
}

struct Guard {
    on: bool,
    knobs: HashMap<String, f64>,
    lists: HashMap<String, Vec<String>>, // M9: array knobs (e.g. anti_dump.premium)
    nn: Option<Mlp>,                     // optional inline MLP weights
}

// G1.2: strategy tables/curves that used to be hardcoded in this file. All
// optional -- each field is None unless `config.tables` overrides it, and the
// call sites fall back to the historical constants, so the shipped v581 config
// (which carries no `tables`) is bit-unchanged.
#[derive(Default)]
struct Tables {
    sell_priority: Option<Vec<String>>,   // premium-first sell order
    allp: Option<Vec<String>>,            // endgame dump order
    dispatch_window: Option<usize>,       // checkpoint fire window (step..+K)
    sweep: Option<i64>,                   // shed threshold to sweep-sell
    esc_sweep: Option<i64>,               // ditto, under escalation pressure
    endgame_qty: Option<i64>,             // per-item qty in the final dump
    daily_demand: Option<HashMap<String, (Vec<String>, bool)>>, // shop -> (products, single)
    r37: Option<HashMap<String, R37Curve>>,                     // item -> price curve
}

struct Config {
    checkpoints: Vec<(usize, usize)>,     // (step, day)
    guards: HashMap<String, Guard>,
    tables: Tables,
    // G1.4 (ordered rail registry, 2026-09-18): the guardrails LIST drives both
    // the SET and the ORDER. `order` is the guardrail names exactly as listed in
    // config (endgame auto-appended if missing -- it is mandatory), and the
    // dispatcher in `act()` walks it and applies each rail by NAME. Add/reorder/
    // remove a rail = a config edit, no recompile. On/off within the set is still
    // the per-rail `.g()` check, so a listed-but-off rail is skipped (matching the
    // old fixed sequence). The v581 default list therefore stays bit-identical.
    order: Vec<String>,
}

impl Config {
    fn g(&self, name: &str) -> bool {
        self.guards.get(name).map(|g| g.on).unwrap_or(false)
    }
    fn knob(&self, name: &str, k: &str, d: f64) -> f64 {
        self.guards
            .get(name)
            .and_then(|g| g.knobs.get(k).copied())
            .unwrap_or(d)
    }
    // M9: array-valued knob (e.g. anti_dump.premium). Returns None when absent,
    // so the caller keeps its hardcoded default -- never a silent config lie.
    fn list(&self, name: &str, k: &str) -> Option<&Vec<String>> {
        self.guards.get(name).and_then(|g| g.lists.get(k))
    }
    fn nn(&self, name: &str) -> Option<&Mlp> {
        self.guards.get(name).and_then(|g| g.nn.as_ref())
    }
}

fn load_config(src: &str) -> Config {
    let j = parse(src).unwrap_or(Json::Null);
    let mut checkpoints = Vec::new();
    for c in j.get("checkpoints").arr() {
        checkpoints.push((c.idx(0).i64() as usize, c.idx(1).i64() as usize));
    }
    checkpoints.sort_by_key(|c| c.0);
    let mut guards = HashMap::new();
    for g in j.get("guardrails").arr() {
        let name = g.get("name").str().to_string();
        if name.is_empty() {
            continue;
        }
        let on = g.get("on").bool();
        let mut knobs = HashMap::new();
        // M9: previously ONLY scalar numbers were kept, so array knobs
        // (anti_dump.premium, scarcity_i0's sibling lists) were silently
        // dropped and the Rust path used hardcoded values while the config
        // claimed otherwise. Now retain string arrays too and honour them below.
        let mut lists: HashMap<String, Vec<String>> = HashMap::new();
        let mut nn: Option<Mlp> = None;
        if let Json::Obj(m) = g {
            for (k, v) in m {
                match v {
                    Json::Num(n) => {
                        knobs.insert(k.clone(), *n);
                    }
                    Json::Arr(a) => {
                        let items: Vec<String> = a
                            .iter()
                            .filter_map(|x| match x {
                                Json::Str(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect();
                        if !items.is_empty() {
                            lists.insert(k.clone(), items);
                        }
                    }
                    Json::Obj(_) if k == "nn" => {
                        nn = Mlp::parse(v);
                    }
                    _ => {}
                }
            }
        }
        guards.insert(name, Guard { on, knobs, lists, nn });
    }
    // G1.4: the ordered rail registry -- names in config-list order. A drop is a
    // removal from this list; a reorder is a reorder here; toggling `on:false`
    // keeps the name but the rail's own `.g()` skips it (same as the old code).
    let mut order: Vec<String> = Vec::new();
    for g in j.get("guardrails").arr() {
        let name = g.get("name").str().to_string();
        if !name.is_empty() && !order.iter().any(|n| n == &name) {
            order.push(name);
        }
    }
    // endgame is MANDATORY (the config _doc + the Python builder both auto-add it
    // on:true). Belt-and-suspenders here so an omitting config still dumps: add
    // both the ordered slot AND an on:true guard if absent.
    if !order.iter().any(|n| n == "endgame") {
        order.push("endgame".to_string());
    }
    guards.entry("endgame".to_string()).or_insert_with(|| {
        let mut knobs = HashMap::new();
        knobs.insert("final_dump".to_string(), 700.0);
        Guard { on: true, knobs, lists: HashMap::new(), nn: None }
    });
    Config { checkpoints, guards, tables: load_tables(j.get("tables")), order }
}

/// Parse the optional `config.tables` object. Every field is optional; an
/// absent field keeps the caller's hardcoded default (G1.2).
fn load_tables(t: &Json) -> Tables {
    let mut out = Tables::default();
    if t.is_null() {
        return out;
    }
    let strs = |v: &Json| -> Vec<String> {
        v.arr()
            .iter()
            .filter_map(|x| match x {
                Json::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    };
    let sp = strs(t.get("sell_priority"));
    if !sp.is_empty() {
        out.sell_priority = Some(sp);
    }
    let ap = strs(t.get("allp"));
    if !ap.is_empty() {
        out.allp = Some(ap);
    }
    if !t.get("dispatch_window").is_null() {
        out.dispatch_window = Some(t.get("dispatch_window").i64().max(0) as usize);
    }
    if !t.get("sweep").is_null() {
        out.sweep = Some(t.get("sweep").i64());
    }
    if !t.get("esc_sweep").is_null() {
        out.esc_sweep = Some(t.get("esc_sweep").i64());
    }
    if !t.get("endgame_qty").is_null() {
        out.endgame_qty = Some(t.get("endgame_qty").i64());
    }
    // daily_demand: { "<SHOP>": {"products":["A","B"], "single":bool} }
    if let Json::Obj(shops) = t.get("daily_demand") {
        let mut m: HashMap<String, (Vec<String>, bool)> = HashMap::new();
        for (shop, spec) in shops {
            let products = strs(spec.get("products"));
            let single = spec.get("single").bool();
            m.insert(shop.clone(), (products, single));
        }
        if !m.is_empty() {
            out.daily_demand = Some(m);
        }
    }
    // r37: { "<ITEM>": {base,i0,t,below_func,below_target,above_func,above_target} }
    if let Json::Obj(items) = t.get("r37") {
        let mut m: HashMap<String, R37Curve> = HashMap::new();
        for (item, c) in items {
            m.insert(
                item.clone(),
                R37Curve {
                    base: c.get("base").f64(),
                    i0: c.get("i0").f64(),
                    t: c.get("t").f64(),
                    bf: c.get("below_func").str().to_string(),
                    bt: c.get("below_target").f64(),
                    af: c.get("above_func").str().to_string(),
                    at: c.get("above_target").f64(),
                },
            );
        }
        if !m.is_empty() {
            out.r37 = Some(m);
        }
    }
    out
}

/// branches JSON { "<day>": { "<shopkey>": {"cont":[row...]} } }
fn load_branches(src: &str) -> HashMap<usize, HashMap<String, Vec<Row>>> {
    let j = parse(src).unwrap_or(Json::Null);
    let mut out: HashMap<usize, HashMap<String, Vec<Row>>> = HashMap::new();
    if let Json::Obj(days) = &j {
        for (day_s, bykey) in days {
            let Ok(day) = day_s.parse::<usize>() else { continue };
            let mut m = HashMap::new();
            if let Json::Obj(entries) = bykey {
                for (key, entry) in entries {
                    let cont: Vec<Row> =
                        entry.get("cont").arr().iter().map(json_row).collect();
                    if !cont.is_empty() {
                        m.insert(key.clone(), cont);
                    }
                }
            }
            out.insert(day, m);
        }
    }
    out
}

// -------------------------------------------------------------- the agent --

// G1.4: the per-turn mutable state the ordered rail dispatcher threads through.
// The unit/buys rails mutate farmer/hands/buys directly; the sell-phase rails
// record their name into `sell_rails` (in config-list order) for `sells()`.
struct TurnState {
    in_route: bool,          // step < route.len(): a real tape row exists
    farmer: Op,
    hands: Vec<Op>,
    buys: Vec<Op>,           // the tape row's market BUYs (pre-sell overlay)
    sell_rails: Vec<String>, // ordered active sell-phase rails
}

impl TurnState {
    fn new(in_route: bool) -> Self {
        TurnState {
            in_route,
            farmer: op1("PASS"),
            hands: Vec::new(),
            buys: Vec::new(),
            sell_rails: Vec::new(),
        }
    }
}

// One per-step observable snapshot, recorded at the top of each act() so the NN
// feature builder can emit TEMPORAL deltas. Everything here is observable at play
// time (both banks + the public market), so features derived from it are legal in
// the live agent AND reconstructable from a replay in extract.py (must match).
struct Snap {
    opp_money: f64,
    my_money: f64,
    inv: Vec<f64>,   // NN_PRODUCTS order
    price: Vec<f64>, // NN_PRODUCTS order
}

// Env-gated diagnostic logging (KAGG_LOG). OFF by default -- when the var is
// absent every category is false and each check is a single bool test, so it is
// safe to leave in the shipped binary. KAGG_LOG is a comma-separated set of
// categories (or "1"/"all" for everything):
//   dispatch -- every checkpoint: the computed shop key, HIT/miss, rows spliced
//               and how many branches exist for that day; a per-game summary at
//               reset. Proves whether reactive route-selection ever runs.
//   rails    -- every guardrail in the ordered registry, EVERY turn it is ON:
//               whether it CHANGED anything (buys/farmer/hands/press/sell_rails)
//               or was a NO-OP. This is what surfaces a silently dead gate.
//   sells    -- which sell-phase path executed in sells() and how many SELL
//               orders it produced.
//   turn     -- per-step bank / opp / gap trajectory (verbose, ~720 lines/game).
// All output goes to STDERR with a stable "LOG " prefix (stdout is the action
// channel), so it never corrupts the stdio bridge and greps cleanly.
#[derive(Clone, Copy, Default)]
struct LogCfg {
    dispatch: bool,
    rails: bool,
    sells: bool,
    turn: bool,
}

impl LogCfg {
    fn from_env() -> LogCfg {
        let v = std::env::var("KAGG_LOG").unwrap_or_default();
        if v.is_empty() {
            return LogCfg::default();
        }
        let all = v == "1" || v.eq_ignore_ascii_case("all");
        let has = |c: &str| all || v.split(',').any(|t| t.trim().eq_ignore_ascii_case(c));
        LogCfg {
            dispatch: has("dispatch"),
            rails: has("rails"),
            sells: has("sells"),
            turn: has("turn"),
        }
    }
    fn any(&self) -> bool {
        self.dispatch || self.rails || self.sells || self.turn
    }
}

struct MBandit {
    base: Vec<Row>,
    branches: HashMap<usize, HashMap<String, Vec<Row>>>,
    cfg: Config,
    route: Vec<Row>,
    done: Vec<usize>,                              // checkpoint days fired
    pulled: HashMap<usize, HashMap<String, i64>>,  // front-run ledger
    wrep: HashMap<usize, (usize, Op)>,             // weed-repair
    press: bool,
    hist: Vec<Snap>,                               // per-step observable history (NN temporal feats)
    carry: HashMap<String, i64>,                   // supply_cap: volume deferred to a later step
    log: LogCfg,                                   // KAGG_LOG diagnostic toggles
    disp_fires: usize,                             // checkpoints reached this game
    disp_hits: usize,                              // checkpoints that matched a branch
    rail_hits: HashMap<String, usize>,             // per-rail turns-that-changed-state (for the summary)
}

impl MBandit {
    fn new(base: Vec<Row>, branches: HashMap<usize, HashMap<String, Vec<Row>>>, cfg: Config) -> Self {
        MBandit {
            route: base.clone(),
            base,
            branches,
            cfg,
            done: Vec::new(),
            pulled: HashMap::new(),
            wrep: HashMap::new(),
            press: false,
            hist: Vec::new(),
            carry: HashMap::new(),
            log: LogCfg::from_env(),
            disp_fires: 0,
            disp_hits: 0,
            rail_hits: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        // Per-game gate summary (before the counters clear). In a reused process
        // (serve/batch) this prints at the start of each subsequent game; a
        // one-shot env.run never calls reset() twice, so its summary is instead
        // derivable from the per-turn lines.
        if self.log.any() && (self.disp_fires > 0 || !self.rail_hits.is_empty()) {
            let mut rails: Vec<String> = self
                .cfg
                .order
                .iter()
                .map(|n| format!("{}={}", n, self.rail_hits.get(n).copied().unwrap_or(0)))
                .collect();
            rails.sort();
            eprintln!(
                "LOG summary dispatch_fires={} dispatch_hits={} route_was={} | rail_turns_changed: {}",
                self.disp_fires,
                self.disp_hits,
                if self.disp_hits == 0 { "PURE_BASE_TAPE" } else { "spliced" },
                rails.join(" ")
            );
        }
        self.route = self.base.clone();
        self.done.clear();
        self.pulled.clear();
        self.wrep.clear();
        self.press = false;
        self.hist.clear();
        self.carry.clear();
        self.disp_fires = 0;
        self.disp_hits = 0;
        self.rail_hits.clear();
    }

    // Record this step's observable snapshot (called at the top of act()). After
    // this, self.hist.last() is the CURRENT step; deltas look back k entries.
    fn record(&mut self, obs: &Json) {
        let me = Self::me(obs);
        let inv = obs.get("market").get("inventory");
        let pr = obs.get("market").get("prices");
        let inv_v: Vec<f64> = NN_PRODUCTS.iter().map(|p| inv.get(p).f64()).collect();
        let price_v: Vec<f64> = NN_PRODUCTS
            .iter()
            .map(|p| { let x = pr.get(p).f64(); if x > 0.0 { x } else { eng_base(p) } })
            .collect();
        self.hist.push(Snap {
            opp_money: Self::money(obs, 1 - me),
            my_money: Self::money(obs, me),
            inv: inv_v,
            price: price_v,
        });
    }

    fn me(obs: &Json) -> usize {
        obs.get("player").i64().max(0) as usize
    }
    fn money(obs: &Json, seat: usize) -> f64 {
        obs.get("farms").idx(seat).get("money").f64()
    }
    fn shed(obs: &Json, item: &str) -> i64 {
        obs.get("private").get("shed").get(item).i64()
    }

    /// Multi-checkpoint dispatch: at each active checkpoint, key on the sorted
    /// cumulative shop set and splice the matching branch continuation.
    fn dispatch(&mut self, obs: &Json, step: usize) {
        let shops = obs.get("town").get("unlocked_shops");
        let arr = shops.arr();
        if arr.len() < 2 {
            return;
        }
        let window = self.cfg.tables.dispatch_window.unwrap_or(2);
        for (cstep, day) in self.cfg.checkpoints.clone() {
            if step < cstep || step > cstep + window || self.done.contains(&day) {
                continue;
            }
            self.done.push(day);
            let mut keys: Vec<String> =
                arr.iter().take(8).map(|s| s.str().to_string()).collect();
            keys.sort();
            let key = keys.join("|");
            self.disp_fires += 1;
            let branch = self.branches.get(&day).and_then(|m| m.get(&key));
            if self.log.dispatch {
                let n_avail = self.branches.get(&day).map(|m| m.len()).unwrap_or(0);
                eprintln!(
                    "LOG dispatch step={} day={} hit={} spliced_rows={} branches_for_day={} realized_key={}",
                    step, day, branch.is_some() as u8,
                    branch.map(|c| c.len()).unwrap_or(0), n_avail, key
                );
            }
            if let Some(cont) = branch {
                self.disp_hits += 1;
                if cstep < self.route.len() {
                    let mut nr = self.route[..cstep].to_vec();
                    nr.extend(cont.iter().cloned());
                    self.route = nr;
                }
            }
        }
    }

    fn pressure(&mut self, obs: &Json, step: usize) {
        if !self.cfg.g("escalation") {
            return;
        }
        let from = self.cfg.knob("escalation", "esc_from", 168.0) as usize;
        let gap = self.cfg.knob("escalation", "esc_gap", 800.0);
        if step < from || step % 24 != 0 {
            return;
        }
        let me = Self::me(obs);
        self.press = (Self::money(obs, 1 - me) - Self::money(obs, me)) > gap;
    }

    fn weed_repair(&mut self, obs: &Json, farmer: Op, hands: Vec<Op>, step: usize) -> (Op, Vec<Op>) {
        if !self.cfg.g("weed_repair") {
            return (farmer, hands);
        }
        let me = Self::me(obs);
        let farm = obs.get("farms").idx(me);
        let mut positions: Vec<(i64, i64)> = Vec::new();
        let f = farm.get("farmer");
        positions.push((f.idx(0).i64(), f.idx(1).i64()));
        for h in farm.get("hands").arr() {
            positions.push((h.idx(0).i64(), h.idx(1).i64()));
        }
        let mut ops: Vec<Op> = Vec::with_capacity(1 + hands.len());
        ops.push(farmer);
        ops.extend(hands);
        let keys: Vec<usize> = self.wrep.keys().copied().collect();
        for idx in keys {
            let (born, saved) = self.wrep.get(&idx).cloned().unwrap();
            if step.saturating_sub(born) == 1 && idx < ops.len() {
                ops[idx] = saved;
            }
            if step.saturating_sub(born) >= 1 {
                self.wrep.remove(&idx);
            }
        }
        for i in 0..ops.len() {
            if ops[i].is_empty() || i >= positions.len() {
                continue;
            }
            let (x, y) = positions[i];
            let tile = farm.get("tiles").idx(y as usize).idx(x as usize);
            if tile.get("kind").str() != "WEED" {
                continue;
            }
            let head = ops[i][0].as_str();
            if head == "PLANT" || head == "BUILD_PASTURE" || head == "BUILD_COOP" {
                self.wrep.insert(i, (step, ops[i].clone()));
                ops[i] = op1("DIG");
            } else if head == "WATER" || head == "HARVEST" {
                ops[i] = op1("DIG");
            }
        }
        let farmer = ops.remove(0);
        (farmer, ops)
    }

    fn cash_guard(&self, obs: &Json, buys: Vec<Op>) -> Vec<Op> {
        if !self.cfg.g("cash_guard") {
            return buys;
        }
        let floor = self.cfg.knob("cash_guard", "cash_floor", 180.0);
        let me = Self::me(obs);
        if Self::money(obs, me) >= floor {
            return buys;
        }
        buys.into_iter()
            .filter(|o| {
                if o.is_empty() {
                    return true;
                }
                let head = o[0].as_str();
                if head == "SELL" || head == "HIRE" || head == "BUY_ANIMAL" {
                    return true;
                }
                let item = o.get(1).map(|s| s.as_str()).unwrap_or("");
                if item == "WHEAT" {
                    return true;
                }
                !(head == "BUY_SEED" || head == "BUY_PRODUCT")
            })
            .collect()
    }

    // hand_align (flexon repair): a fixed tape's hand slots must align
    // positionally with the LIVE crew, else a differing hire outcome shifts every
    // hand action and desyncs the farm. Pad/truncate to the real crew count.
    // Extracted verbatim from the old inline block so the dispatcher can apply it
    // by name (G1.4); behaviour is unchanged.
    fn hand_align(&self, obs: &Json, hands: &mut Vec<Op>) {
        if !self.cfg.g("hand_align") {
            return;
        }
        let me = obs.get("player").i64().max(0) as usize;
        let farms = obs.get("farms");
        let fa = farms.arr();
        let crew = fa
            .get(me)
            .map(|f| f.get("hands").arr().len())
            .unwrap_or(hands.len());
        while hands.len() < crew {
            hands.push(op1("PASS"));
        }
        hands.truncate(crew);
    }

    fn daily_demand(tables: &Tables, obs: &Json, product: &str) -> i64 {
        let mut n = 0i64;
        for s in obs.get("town").get("unlocked_shops").arr() {
            let name = s.str();
            let hit = if let Some(m) = &tables.daily_demand {
                // config override: shop -> (products, single)
                match m.get(name) {
                    Some((prods, single)) => {
                        prods.iter().any(|x| x == product).then_some(*single)
                    }
                    None => None,
                }
            } else {
                // historical hardcoded shop demand table (v581 default)
                let (demands, single): (&[&str], bool) = match name {
                    "BAKERY" => (&["EGG", "WHEAT"], false),
                    "PIZZA_SHOP" => (&["MILK", "TOMATO", "WHEAT"], false),
                    "BRUNCH_SPOT" => (&["EGG", "WHEAT", "STRAWBERRY"], false),
                    "YARN_STORE" => (&["WOOL"], true),
                    "ICE_CREAM_SHOP" => (&["STRAWBERRY", "MILK", "WHEAT"], false),
                    "PET_CAFE" => (&["CARROT"], true),
                    "SMOOTHIE_SHOP" => (&["STRAWBERRY", "MILK"], false),
                    "FARMERS_MARKET" => (&["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"], false),
                    _ => (&[], false),
                };
                demands.contains(&product).then_some(single)
            };
            if let Some(single) = hit {
                n += if single { 2 } else { 1 };
            }
        }
        let mut d = n * 6;
        if product != "FERTILIZER" {
            d += 1;
        }
        d
    }

    // G1.4: `sell_rails` is the ordered subset of the config guardrails that act
    // in the sell phase (endgame / scarcity_sell / anti_dump / front_run), in
    // config-list order. On/off is still each rail's own `.g()` check; the LIST
    // drives (a) which sell rails exist and (b) the ORDER the per-item modifiers
    // (anti_dump, scarcity_sell) fold into the sweep amount. With the v581 list
    // (anti_dump absent, scarcity present) this is bit-identical to the old code.
    /// Suffix-sum: total quantity the TAPE still plans to SELL of `item` from
    /// `from` onward. Anything in the shed beyond this is "dead stock" the
    /// economy was never going to sell -- safe to liquidate without touching the
    /// cash-flow sells. (their _dead_stock / future_sells.)
    fn future_sells(&self, item: &str, from: usize) -> i64 {
        let mut s = 0i64;
        let mut t = from;
        while t < self.route.len() {
            for o in &self.route[t].market {
                if o.len() >= 3 && o[0] == "SELL" && o[1] == item {
                    s += o[2].parse::<i64>().unwrap_or(0).max(0);
                }
            }
            t += 1;
        }
        s
    }

    /// Planned PURCHASE budget + protected item reserves for tape steps
    /// [start,end) -- what the block will spend on hires/land/seeds/animals/
    /// products, and how much of each item it will consume (must not be sold).
    /// (their _block_requirements.)
    fn block_requirements(&self, obs: &Json, start: usize, end: usize) -> (f64, HashMap<String, i64>) {
        let me = Self::me(obs);
        let farm = obs.get("farms").idx(me);
        let mut quadrants = farm.get("unlocked_quadrants").arr().len();
        let hires_today = farm.get("hires_today").i64();
        let mut budget = 0.0f64;
        let mut need: HashMap<String, i64> = HashMap::new();
        let mut seed_bal: HashMap<String, i64> = HashMap::new();
        let mut item_bal: HashMap<String, i64> = HashMap::new();
        let mut hires_by_day: HashMap<usize, i64> = HashMap::new();
        let tpd = 24usize;
        let end = end.min(self.route.len());
        for t in start..end {
            let row = &self.route[t];
            let mut units: Vec<&Op> = vec![&row.farmer];
            for h in &row.hands {
                units.push(h);
            }
            for u in units {
                if u.is_empty() {
                    continue;
                }
                let op = u[0].as_str();
                let arg = if u.len() > 1 { u[1].as_str() } else { "" };
                if op == "PLANT" && rules::crop(arg).is_some() {
                    let b = seed_bal.entry(arg.to_string()).or_insert(0);
                    *b -= 1;
                    let n = need.entry(arg.to_string()).or_insert(0);
                    *n = (*n).max(-*b);
                } else if op == "FEED" {
                    let b = item_bal.entry("WHEAT".into()).or_insert(0);
                    *b -= 1;
                    let n = need.entry("WHEAT".into()).or_insert(0);
                    *n = (*n).max(-*b);
                } else if op == "FERTILIZE" {
                    let b = item_bal.entry("FERTILIZER".into()).or_insert(0);
                    *b -= 1;
                    let n = need.entry("FERTILIZER".into()).or_insert(0);
                    *n = (*n).max(-*b);
                }
            }
            for o in &row.market {
                if o.is_empty() {
                    continue;
                }
                let op = o[0].as_str();
                let item = if o.len() > 1 { o[1].as_str() } else { "" };
                let qty: i64 = if o.len() > 2 { o[2].parse().unwrap_or(1).max(1) } else { 1 };
                match op {
                    "HIRE" => {
                        *hires_by_day.entry((t - start) / tpd).or_insert(0) += 1;
                    }
                    "BUY_LAND" => {
                        let extra = quadrants.saturating_sub(1);
                        if extra < rules::LAND_PRICES.len() {
                            budget += rules::LAND_PRICES[extra] as f64;
                            quadrants += 1;
                        }
                    }
                    "BUY_SEED" => {
                        if let Some(c) = rules::crop(item) {
                            budget += (c.seed_cost * qty) as f64;
                            *seed_bal.entry(item.to_string()).or_insert(0) += qty;
                        }
                    }
                    "BUY_PRODUCT" if item == "WHEAT" || item == "FERTILIZER" => {
                        let minv = obs.get("market").get("inventory").get(item).i64();
                        budget += eng_price(item, minv) * qty as f64;
                        *item_bal.entry(item.to_string()).or_insert(0) += qty;
                    }
                    "BUY_ANIMAL" => {
                        if let Some(a) = rules::animal(item) {
                            budget += (a.cost * qty) as f64;
                        }
                    }
                    _ => {}
                }
            }
        }
        for (day, n) in hires_by_day {
            let first = if day == 0 { hires_today } else { 0 };
            for k in 0..n {
                budget += rules::hire_cost((first + k) as u32, rules::FARM_HAND_COST_MULT) as f64;
            }
        }
        (budget, need)
    }

    /// budget_guard: at a block boundary, guarantee cash + the value of stock the
    /// block already plans to sell covers the block's purchases; cover any
    /// shortfall with extra SELLs of UNPROTECTED shed stock (highest price first)
    /// moved in FRONT of the buys. This is the CASH-FLOW BACKSTOP that lets the
    /// price rails hold/defer safely without starving the economy.
    fn budget_guard(&self, obs: &Json, buys: &mut Vec<Op>, step: usize, block: usize) {
        let me = Self::me(obs);
        let (budget, need) = self.block_requirements(obs, step, step + block);
        let inv = obs.get("market").get("inventory");
        let prices = obs.get("market").get("prices");
        let shed = obs.get("private").get("shed");
        let px = |p: &str| -> f64 {
            let v = prices.get(p).f64();
            if v > 0.0 { v } else { eng_price(p, inv.get(p).i64()) }
        };
        let mut existing: HashMap<String, i64> = HashMap::new();
        for o in buys.iter() {
            if o.len() >= 3 && o[0] == "SELL" {
                *existing.entry(o[1].clone()).or_insert(0) += o[2].parse().unwrap_or(0).max(0);
            }
        }
        let mut cash = Self::money(obs, me);
        for p in NN_PRODUCTS {
            let planned = existing.get(p).copied().unwrap_or(0)
                .max(self.future_sells(p, step) - self.future_sells(p, step + block));
            cash += (shed.get(p).i64().min(planned) as f64) * px(p);
        }
        let mut short = budget - cash;
        if short <= 0.0 {
            return;
        }
        let mut cand: Vec<(f64, &str, i64)> = Vec::new();
        for p in NN_PRODUCTS {
            let price = px(p);
            if price < 1.0 {
                continue;
            }
            let protected = need.get(p).copied().unwrap_or(0).max(0);
            let avail = shed.get(p).i64() - protected - existing.get(p).copied().unwrap_or(0);
            if avail > 0 {
                cand.push((price, p, avail));
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut added = false;
        for (price, p, avail) in cand {
            if short <= 0.0 {
                break;
            }
            let qty = avail.min((short / price).ceil() as i64).max(0);
            if qty < 1 {
                continue;
            }
            let mut merged = false;
            for o in buys.iter_mut() {
                if o.len() >= 3 && o[0] == "SELL" && o[1] == p {
                    let cur: i64 = o[2].parse().unwrap_or(0);
                    o[2] = (cur + qty).to_string();
                    merged = true;
                    break;
                }
            }
            if !merged {
                if buys.len() >= MARKET_CAP {
                    continue;
                }
                buys.push(vec!["SELL".into(), p.into(), qty.to_string()]);
            }
            short -= qty as f64 * price;
            added = true;
        }
        if added {
            let (sells, others): (Vec<Op>, Vec<Op>) =
                buys.drain(..).partition(|o| !o.is_empty() && o[0] == "SELL");
            buys.clear();
            buys.extend(sells);
            buys.extend(others);
        }
    }

    /// NN feature vector -- MUST match kaggriculture.bandit.nn.extract._feat:
    /// per product [inv/I0, price/base, shed/50] then
    /// [day/30, step/720, money_me/1e5, money_opp/1e5, gap/1e5].
    // Feature vector for the NN heads. Layout MUST match extract.py _feat exactly.
    //   BASE (32): per-product [inv/1e4, price/base, shed/50] (27) +
    //              [day/30, step/720, me/1e5, opp/1e5, gap/1e5] (5)
    //   TEMPORAL (33): opp_money_delta over 1/4/8 steps (÷1e4) -- the DUMP signal
    //              (opp bank jumps when they sell) -- + my_money_delta_4 (÷1e4) +
    //              town phase [step%4 /4, step%24 /24] + per-product
    //              inv_delta_4 (÷1e4), price_delta_4 (÷base), scarcity_margin
    //              (price-0.9*base)/base.  Total NFEAT = 65.
    // Requires self.hist to hold the current step's snapshot as its LAST entry
    // (record() is called at the top of act()).
    fn nn_feats(&self, obs: &Json, step: usize) -> Vec<f64> {
        let me = Self::me(obs);
        let inv = obs.get("market").get("inventory");
        let pr = obs.get("market").get("prices");
        let shed = obs.get("private").get("shed");
        let mut f: Vec<f64> = Vec::with_capacity(65);
        for p in NN_PRODUCTS {
            f.push(inv.get(p).f64() / 10000.0);
            let price = pr.get(p).f64();
            let price = if price > 0.0 { price } else { eng_base(p) };
            f.push(price / eng_base(p));
            f.push(shed.get(p).f64() / 50.0);
        }
        let mem = Self::money(obs, me);
        let opp = Self::money(obs, 1 - me);
        f.push(obs.get("day").f64() / 30.0);
        f.push(step as f64 / 720.0);
        f.push(mem / 1e5);
        f.push(opp / 1e5);
        f.push((mem - opp) / 1e5);
        // --- temporal features from history (last entry == current step) ---
        // ORDER MUST MATCH extract.py _feat exactly. NFEAT = 32 base + 75 = 107.
        let n = self.hist.len();
        let cur = self.hist.last();
        let back = |k: usize| -> Option<&Snap> {
            if n == 0 { None } else { self.hist.get(n.saturating_sub(1 + k)) }
        };
        let dmoney = |k: usize, sel: fn(&Snap) -> f64| -> f64 {
            match (cur, back(k)) { (Some(c), Some(p)) => sel(c) - sel(p), _ => 0.0 }
        };
        // opponent money deltas over 1/2/4/8/16 (their bank jumping = them selling)
        for k in [1usize, 2, 4, 8, 16] {
            f.push(dmoney(k, |s| s.opp_money) / 1e4);
        }
        f.push(dmoney(4, |s| s.my_money) / 1e4);
        // town-consume phase
        f.push((step % 4) as f64 / 4.0);
        f.push((step % 24) as f64 / 24.0);
        // opponent SELL CADENCE: fraction of the last k steps where their bank rose
        let cadence = |k: usize| -> f64 {
            if n < 2 { return 0.0; }
            let lo = n.saturating_sub(k);
            let mut c = 0usize; let mut d = 0usize;
            for i in (lo + 1)..n {
                d += 1;
                if self.hist[i].opp_money > self.hist[i - 1].opp_money { c += 1; }
            }
            if d == 0 { 0.0 } else { c as f64 / d as f64 }
        };
        f.push(cadence(8));
        f.push(cadence(16));
        // our capacity: labour on hand + total shed we could sell
        f.push(obs.get("farms").idx(me).get("hands").arr().len() as f64 / 8.0);
        let shed_total: f64 = NN_PRODUCTS.iter().map(|p| shed.get(p).f64()).sum();
        f.push(shed_total / 300.0);
        // market inventory deltas over 4/8/16 (supply-pressure trend), per product
        for k in [4usize, 8, 16] {
            for i in 0..NN_PRODUCTS.len() {
                let d = match (cur, back(k)) { (Some(c), Some(p)) => c.inv[i] - p.inv[i], _ => 0.0 };
                f.push(d / 1e4);
            }
        }
        // price deltas over 4/8/16, per product
        for k in [4usize, 8, 16] {
            for (i, p) in NN_PRODUCTS.iter().enumerate() {
                let d = match (cur, back(k)) { (Some(c), Some(pp)) => c.price[i] - pp.price[i], _ => 0.0 };
                f.push(d / eng_base(p));
            }
        }
        // scarcity margin per product
        for p in NN_PRODUCTS {
            let price = pr.get(p).f64();
            let price = if price > 0.0 { price } else { eng_base(p) };
            f.push((price - 0.9 * eng_base(p)) / eng_base(p));
        }
        f
    }

    fn sells(&mut self, obs: &Json, buys_used: usize, step: usize, sell_rails: &[String]) -> Vec<Op> {
        let mut out: Vec<Op> = Vec::new();
        if buys_used >= MARKET_CAP {
            return out;
        }
        let slots = MARKET_CAP - buys_used;
        // present-in-list check: dropping a sell rail from `guardrails` removes it
        // from `sell_rails`, so it no longer fires (SET driven by the list). `.g()`
        // still gates on/off. In v581 all four are present, so this is inert.
        let has = |n: &str| sell_rails.iter().any(|s| s.as_str() == n);
        let final_dump = self.cfg.knob("endgame", "final_dump", 700.0) as usize;
        // terminal_rescue: PROGRESSIVE endgame liquidation, superseding the crude
        // `endgame` full-dump when on. Measured vs tschinkel we bled the whole
        // endgame (e.g. -5137 on day 29 alone) because we sold AFTER the opponent
        // flooded the market: our fixed tape spread its dump late into prices its
        // own earlier sells + the opponent's dump had already crushed. This starts
        // EARLIER (tr_from, ~day 27.5) and, every step, sells each shed item up to
        // the qty that keeps the marginal (r37) price >= tr_floor*base -- so it
        // front-loads high-value liquidation into the best price each step and
        // beats the opponent's step-700 flood, without crashing the market. Near
        // the very end (>= tr_hard) it hard-dumps the rest (no future to protect).
        // Deterministic, config-gated, only ever sells shed -> never contradicts
        // the tape's production. On when listed+`on`; supersedes `endgame`.
        if has("terminal_rescue") && self.cfg.g("terminal_rescue") {
            let tr_from = self.cfg.knob("terminal_rescue", "tr_from", 660.0) as usize;
            if step >= tr_from {
                let tr_floor = self.cfg.knob("terminal_rescue", "tr_floor", 0.5);
                let tr_hard = self.cfg.knob("terminal_rescue", "tr_hard", 712.0) as usize;
                let items: Vec<&str> = match &self.cfg.tables.allp {
                    Some(v) => v.iter().map(|s| s.as_str()).collect(),
                    None => ALLP.to_vec(),
                };
                for p in &items {
                    if out.len() >= slots {
                        break;
                    }
                    let have = Self::shed(obs, p);
                    if have < 1 {
                        continue;
                    }
                    let q = if step >= tr_hard {
                        have // last steps: dump everything, nothing left to time
                    } else {
                        let minv = obs.get("market").get("inventory").get(*p).i64();
                        eng_cap(p, minv, have, tr_floor) // engine-exact liquidation
                    };
                    if q >= 1 {
                        out.push(vec!["SELL".into(), (*p).into(), q.to_string()]);
                    }
                }
                return out;
            }
        }
        if has("endgame") && self.cfg.g("endgame") && step >= final_dump {
            let dump_qty = self.cfg.tables.endgame_qty.unwrap_or(1000).to_string();
            let allp: Vec<&str> = match &self.cfg.tables.allp {
                Some(v) => v.iter().map(|s| s.as_str()).collect(),
                None => ALLP.to_vec(),
            };
            for p in allp {
                if out.len() >= slots {
                    break;
                }
                if Self::shed(obs, p) > 0 {
                    out.push(vec!["SELL".into(), p.into(), dump_qty.clone()]);
                }
            }
            return out;
        }
        // mlp_sell: the learned SALE head owns selling in its window (tape SELLs
        // were suppressed structurally). Per product, sell round(pred_fraction *
        // shed), then cap by eng_cap at a HARD safety floor so a bad prediction
        // can never crash the market below nn_floor*base. Config carries the
        // weights inline (ship-by-config); absent/malformed weights -> no-op.
        if has("mlp_sell") && self.cfg.g("mlp_sell") {
            if let Some(mlp) = self.cfg.nn("mlp_sell") {
                let nn_floor = self.cfg.knob("mlp_sell", "nn_floor", 0.2);
                let feats = self.nn_feats(obs, step);
                let pred = mlp.forward(&feats);
                for (i, p) in NN_PRODUCTS.iter().enumerate() {
                    if out.len() >= slots || i >= pred.len() {
                        break;
                    }
                    let have = Self::shed(obs, p);
                    if have < 1 {
                        continue;
                    }
                    let frac = pred[i].clamp(0.0, 1.0);
                    let mut q = (frac * have as f64).round() as i64;
                    if q < 1 {
                        continue;
                    }
                    let minv = obs.get("market").get("inventory").get(*p).i64();
                    q = q.min(eng_cap(p, minv, have, nn_floor)).min(have);
                    if q >= 1 {
                        out.push(vec!["SELL".into(), (*p).into(), q.to_string()]);
                    }
                }
            }
            return out;
        }
        // supply_demand DRAIN: owns selling in its window (the tape's SELLs were
        // suppressed structurally). Every step, sell each shed item up to the
        // engine-exact qty that keeps price >= sd_floor*base -- a smooth
        // supply-to-demand stream into scarcity, never a burst/glut. Returns so
        // the sweep/scarcity/front_run rails don't also fire (this IS the sell).
        if has("supply_demand") && self.cfg.g("supply_demand") {
            let floor = self.cfg.knob("supply_demand", "sd_floor", 0.9);
            let items: Vec<&str> = match &self.cfg.tables.sell_priority {
                Some(v) => v.iter().map(|s| s.as_str()).collect(),
                None => SELL_PRIORITY.to_vec(),
            };
            for p in items {
                if out.len() >= slots {
                    break;
                }
                let have = Self::shed(obs, p);
                if have < 1 {
                    continue;
                }
                let minv = obs.get("market").get("inventory").get(p).i64();
                let q = eng_cap(p, minv, have, floor);
                if q >= 1 {
                    out.push(vec!["SELL".into(), p.into(), q.to_string()]);
                }
            }
            return out;
        }
        let sweep: i64 = if self.press {
            self.cfg.tables.esc_sweep.unwrap_or(24)
        } else {
            self.cfg.tables.sweep.unwrap_or(40)
        };
        let sell_priority: Vec<&str> = match &self.cfg.tables.sell_priority {
            Some(v) => v.iter().map(|s| s.as_str()).collect(),
            None => SELL_PRIORITY.to_vec(),
        };
        let anti_dump = has("anti_dump") && self.cfg.g("anti_dump");
        let scarcity = has("scarcity_sell") && self.cfg.g("scarcity_sell") && !self.press;
        let hold_frac = self.cfg.knob("scarcity_sell", "hold_frac", 1.0);
        // G1.4: the per-item sweep modifiers fold in the ORDER they appear in the
        // config guardrails list. v581 lists scarcity_sell (anti_dump absent), so
        // mod_order = ["scarcity_sell"] and the result matches the old fixed
        // anti_dump-then-scarcity sequence exactly. With both present, swapping
        // their order in the config swaps how they compose (a positive control).
        let mod_order: Vec<&str> = sell_rails
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s == "anti_dump" || *s == "scarcity_sell")
            .collect();
        for p in sell_priority {
            if out.len() >= slots {
                break;
            }
            let have = Self::shed(obs, p);
            if have >= sweep {
                let base_amt = have - 8;
                let mut amt = base_amt;
                for m in &mod_order {
                    match *m {
                        // M9: honour a configured anti_dump.premium set (else the
                        // historical hardcoded default) + the anti_dump.scarcity_i0
                        // gate. Default i0 = +inf, so absent config == prior. This
                        // OVERWRITES amt from base_amt (as the old code did), so its
                        // position relative to scarcity_sell is observable.
                        "anti_dump" if anti_dump => {
                            let is_premium = match self.cfg.list("anti_dump", "premium") {
                                Some(prem) => prem.iter().any(|x| x.as_str() == p),
                                None => matches!(p, "MILK" | "WOOL" | "STRAWBERRY" | "MELON"),
                            };
                            if is_premium {
                                let i0 = self.cfg.knob("anti_dump", "scarcity_i0", f64::INFINITY);
                                let minv = obs.get("market").get("inventory").get(p).i64();
                                amt = if (minv as f64) >= i0 {
                                    0 // market not yet stressed -> hold the premium sweep
                                } else {
                                    base_amt.min(Self::daily_demand(&self.cfg.tables, obs, p).max(8))
                                };
                            }
                        }
                        // _r37 marginal-price hold: don't dump into a floored market.
                        "scarcity_sell" if scarcity => {
                            let inv = obs.get("market").get("inventory").get(p).i64();
                            amt = amt.min(r37_hold_cap(&self.cfg.tables, p, inv, base_amt, hold_frac));
                        }
                        _ => {}
                    }
                }
                if amt > 0 {
                    out.push(vec!["SELL".into(), p.into(), amt.to_string()]);
                }
            }
        }
        // price_sell: pull the REST-OF-DAY's premium SELLs FORWARD into a
        // favourable (r37) price. The tape schedules premium items (MILK/WOOL/...)
        // at a fixed step mid-day; measured vs tschinkel we realise a LOWER price
        // because we sell later/fragmented into a consumed market. This looks a
        // whole window ahead (not front_run's 1 step -- which never even SEES the
        // later sells) and, only while the r37 marginal price stays >= ps_floor*base,
        // sells now up to r37_hold_cap so it never crashes the market below the
        // floor. Pulled qty is booked in the front-run ledger, so the original
        // step's tape SELL is suppressed (no double-sell). Off unless listed+on.
        if has("price_sell") && self.cfg.g("price_sell") && step < final_dump {
            let ps_from = self.cfg.knob("price_sell", "ps_from", 145.0) as usize;
            let ps_look = self.cfg.knob("price_sell", "ps_look", 12.0) as usize;
            let ps_floor = self.cfg.knob("price_sell", "ps_floor", 0.95);
            let items: Vec<String> = match self.cfg.list("price_sell", "items") {
                Some(v) => v.clone(),
                None => ["MILK", "WOOL", "STRAWBERRY", "MELON", "EGG"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            };
            if step >= ps_from {
                let end = (step + 1 + ps_look).min(self.route.len());
                let mut already: Vec<String> =
                    out.iter().filter(|o| o.len() >= 2).map(|o| o[1].clone()).collect();
                for item in &items {
                    if out.len() >= slots {
                        break;
                    }
                    let p = item.as_str();
                    if already.iter().any(|x| x == p) {
                        continue;
                    }
                    let have = Self::shed(obs, p);
                    if have < 1 {
                        continue;
                    }
                    let minv = obs.get("market").get("inventory").get(p).i64();
                    // remaining planned SELLs of this item in (step+1 .. end)
                    let mut plan_steps: Vec<(usize, i64)> = Vec::new();
                    let mut planned: i64 = 0;
                    for t in (step + 1)..end {
                        for o in &self.route[t].market {
                            if o.len() >= 3 && o[0] == "SELL" && o[1] == *item {
                                let want: i64 = o[2].parse().unwrap_or(0);
                                let prev =
                                    self.pulled.get(&t).and_then(|m| m.get(p)).copied().unwrap_or(0);
                                let rem = (want - prev).max(0);
                                if rem > 0 {
                                    planned += rem;
                                    plan_steps.push((t, rem));
                                }
                            }
                        }
                    }
                    if planned < 1 {
                        continue;
                    }
                    // sell only up to where the ENGINE marginal price stays >=
                    // ps_floor*base (supply-to-demand; never glut the market)
                    let cap = eng_cap(p, minv, planned.min(have), ps_floor);
                    if cap < 1 {
                        continue; // market already at/below floor -> hold, sell on schedule
                    }
                    out.push(vec!["SELL".into(), p.into(), cap.to_string()]);
                    already.push(p.to_string());
                    // book the pulled qty against the earliest future steps
                    let mut rem = cap;
                    for (t, q) in plan_steps {
                        if rem <= 0 {
                            break;
                        }
                        let take = q.min(rem);
                        let e = self.pulled.entry(t).or_default();
                        *e.entry(p.to_string()).or_insert(0) += take;
                        rem -= take;
                    }
                }
            }
        }
        // endgame_liquidate: D27-onward, PRICE-ADAPTIVE liquidation of DEAD stock
        // (shed beyond the tape's future planned sales). If the price is floored on
        // D27 we HOLD (sell nothing this turn) and try again as it recovers across
        // D28; on/after el_hard_day (D29) everything not already queued is dead and
        // is dumped regardless of price -- nothing is left unsold. Additive and
        // volume-safe: only surplus the economy never planned to sell. Highest
        // engine-value lots first.
        if has("endgame_liquidate") && self.cfg.g("endgame_liquidate") {
            let el_from = self.cfg.knob("endgame_liquidate", "el_from", 648.0) as usize;
            if step >= el_from {
                let floor = self.cfg.knob("endgame_liquidate", "el_floor", 0.6);
                let hard_day = self.cfg.knob("endgame_liquidate", "el_hard_day", 29.0) as i64;
                let day = (step / 24) as i64;
                let items: Vec<&str> = match &self.cfg.tables.sell_priority {
                    Some(v) => v.iter().map(|s| s.as_str()).collect(),
                    None => SELL_PRIORITY.to_vec(),
                };
                let already: Vec<String> =
                    out.iter().filter(|o| o.len() >= 2).map(|o| o[1].clone()).collect();
                let mut cand: Vec<(f64, String, i64)> = Vec::new();
                for p in items {
                    if already.iter().any(|x| x == p) {
                        continue;
                    }
                    let have = Self::shed(obs, p);
                    if have < 1 {
                        continue;
                    }
                    let minv = obs.get("market").get("inventory").get(p).i64();
                    let surplus = if day >= hard_day {
                        have // day29+: everything not queued is dead
                    } else {
                        (have - self.future_sells(p, step + 1)).max(0)
                    };
                    if surplus < 1 {
                        continue;
                    }
                    let q = if day >= hard_day {
                        surplus // must liquidate -- dump regardless of price
                    } else if eng_price(p, minv) >= floor * eng_base(p) {
                        eng_cap(p, minv, surplus, floor) // price OK -> sell up to the floor
                    } else {
                        0 // floored -> HOLD, retry as price recovers
                    };
                    if q >= 1 {
                        cand.push((eng_price(p, minv) * q as f64, p.to_string(), q));
                    }
                }
                cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for (_, p, q) in cand {
                    if out.len() >= slots {
                        break;
                    }
                    out.push(vec!["SELL".into(), p, q.to_string()]);
                }
            }
        }
        // opp_front_run: the learned OPPONENT-DUMP head. If P(opponent dumps a
        // premium item within the horizon) >= opp_thresh, sell OUR premium
        // holdings now -- but ONLY up to eng_cap at opp_floor*base, so even a
        // wrong prediction never sells into a crashed market (the fear of
        // front-running an unpredictable opponent is bounded by the price floor).
        // Additive: reduces shed the tape would later dump, front-running the dump.
        if has("opp_front_run") && self.cfg.g("opp_front_run") {
            if let Some(mlp) = self.cfg.nn("opp_front_run") {
                let thresh = self.cfg.knob("opp_front_run", "opp_thresh", 0.6);
                let floor = self.cfg.knob("opp_front_run", "opp_floor", 0.85);
                let pred = mlp.forward(&self.nn_feats(obs, step));
                // dump prob = the OPP output. Standalone opp net -> its single output;
                // JOINT (sale9+opp) net -> the LAST output (opp is column 9). last()
                // is correct for both, so one config field feeds either net.
                if pred.last().copied().unwrap_or(0.0) >= thresh {
                    // config-driven premium set (opp_front_run.premium), else default.
                    let prem_owned: Vec<String>;
                    let premium: Vec<&str> = match self.cfg.list("opp_front_run", "premium") {
                        Some(v) => { prem_owned = v.clone(); prem_owned.iter().map(|s| s.as_str()).collect() }
                        None => vec!["MILK", "WOOL", "STRAWBERRY", "MELON"],
                    };
                    let mut already: Vec<String> =
                        out.iter().filter(|o| o.len() >= 2).map(|o| o[1].clone()).collect();
                    for p in premium {
                        if out.len() >= slots {
                            break;
                        }
                        if already.iter().any(|x| x == p) {
                            continue;
                        }
                        let have = Self::shed(obs, p);
                        if have < 1 {
                            continue;
                        }
                        let minv = obs.get("market").get("inventory").get(p).i64();
                        let q = eng_cap(p, minv, have, floor);
                        if q >= 1 {
                            out.push(vec!["SELL".into(), p.into(), q.to_string()]);
                            already.push(p.to_string());
                        }
                    }
                }
            }
        }
        if has("front_run") && self.cfg.g("front_run") {
            let fr_from = self.cfg.knob("front_run", "fr_from", 145.0) as usize;
            let look_n = self.cfg.knob("front_run", "fr_look", 1.0) as usize;
            let look_p = self.cfg.knob("front_run", "fr_look_press", 2.0) as usize;
            let fr_look = if self.press { look_n.max(look_p) } else { look_n };
            if step >= fr_from {
                let look: Vec<(usize, Vec<Op>)> = {
                    let end = (step + 1 + fr_look).min(self.route.len());
                    (step + 1..end).map(|t| (t, self.route[t].market.clone())).collect()
                };
                let mut already: Vec<String> =
                    out.iter().filter(|o| o.len() >= 2).map(|o| o[1].clone()).collect();
                for (t, market) in &look {
                    let t = *t;
                    for o in market {
                        if out.len() >= slots {
                            break;
                        }
                        if o.len() < 3 || o[0] != "SELL" {
                            continue;
                        }
                        let p = o[1].clone();
                        if p == "WHEAT" || p == "FERTILIZER" || already.contains(&p) {
                            continue;
                        }
                        let want: i64 = o[2].parse().unwrap_or(0);
                        let prev = self.pulled.get(&t).and_then(|m| m.get(&p)).copied().unwrap_or(0);
                        let q = (want - prev).min(Self::shed(obs, &p));
                        if q < 1 {
                            continue;
                        }
                        out.push(vec!["SELL".into(), p.clone(), q.to_string()]);
                        already.push(p.clone());
                        self.pulled.entry(t).or_default().insert(p, prev + q);
                    }
                }
            }
        }
        out
    }

    /// G1.4: apply ONE guardrail by name to the turn state. This is the uniform
    /// shim the ordered dispatcher calls; the rails have heterogeneous signatures
    /// underneath but present one `apply(name, &mut ts)` face here. Adding,
    /// reordering or dropping a rail is a config edit -- the only compile-time
    /// fact is which names are wired. Unknown names are ignored (forward-compat).
    /// The unit/buys rails apply only inside the tape (`in_route`), exactly as the
    /// old `if step < route.len()` block did; escalation and the sell-phase rails
    /// run every turn. Each rail still self-gates on its `.g()` on/off flag.
    fn apply_rail(&mut self, name: &str, ts: &mut TurnState, obs: &Json, step: usize) {
        match name {
            "escalation" => self.pressure(obs, step),
            "weed_repair" => {
                if ts.in_route {
                    let farmer = std::mem::take(&mut ts.farmer);
                    let hands = std::mem::take(&mut ts.hands);
                    let (f, h) = self.weed_repair(obs, farmer, hands, step);
                    ts.farmer = f;
                    ts.hands = h;
                }
            }
            "hand_align" => {
                if ts.in_route {
                    self.hand_align(obs, &mut ts.hands);
                }
            }
            "cash_guard" => {
                if ts.in_route {
                    let buys = std::mem::take(&mut ts.buys);
                    ts.buys = self.cash_guard(obs, buys);
                }
            }
            // herd_gate: WORLD-CONDITIONED production right-sizing. The world's
            // shops are unknown at day 0, so the EARLY herd (which also yields
            // FERTILIZER) is committed blind and kept. But from D6 (from_step) the
            // first shops are known, so we stop GROWING a premium the world can't
            // absorb: suppress NEW SHEEP when there is no YARN (WOOL floors to ~$6),
            // and NEW COW when there is no MILK shop. Only NEW purchases past
            // from_step are dropped; the pre-D6 animals stay. A dropped buy leaves
            // the tape's later FEED/HARVEST as legal no-ops and banks the saved
            // BUY_ANIMAL cost -- no position desync. Booleans, config-tunable
            // (default off), so the rule is a rule, not an NN. EGG (dump-proof) is
            // the intended substitute, added by a branch, not here.
            "herd_gate" => {
                if ts.in_route {
                    let from = self.cfg.knob("herd_gate", "from_step", 146.0) as usize;
                    if step >= from {
                        let shops: Vec<String> = obs.get("town").get("unlocked_shops")
                            .arr().iter().map(|s| s.str().to_string()).collect();
                        let has = |names: &[&str]| shops.iter().any(|s| names.contains(&s.as_str()));
                        let no_milk = !has(&["PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"]);
                        let no_wool = !has(&["YARN_STORE"]);
                        let stop_sheep = no_wool && self.cfg.knob("herd_gate", "stop_sheep_noyarn", 0.0) >= 0.5;
                        let stop_cow = no_milk && self.cfg.knob("herd_gate", "stop_cow_nomilk", 0.0) >= 0.5;
                        if stop_sheep || stop_cow {
                            ts.buys.retain(|o| {
                                if o.len() >= 2 && o[0] == "BUY_ANIMAL" {
                                    if o[1] == "SHEEP" && stop_sheep {
                                        return false;
                                    }
                                    if o[1] == "COW" && stop_cow {
                                        return false;
                                    }
                                }
                                true
                            });
                        }
                    }
                }
            }
            // scarcity_hold: DEFER a tape SELL scheduled into a glutted market
            // (engine price < floor*base) -- drop it this turn; the stock stays in
            // the shed for a later step when price recovers, and endgame_liquidate
            // guarantees it still clears by D29. Safe because budget_guard (below in
            // the list) re-adds any minimum selling the block's purchases need.
            // Time-gated: never holds past the endgame window.
            "scarcity_hold" => {
                if ts.in_route {
                    let until = self.cfg.knob("scarcity_hold", "until", 640.0) as usize;
                    if step < until {
                        let floor = self.cfg.knob("scarcity_hold", "floor", 0.9);
                        let inv = obs.get("market").get("inventory");
                        ts.buys.retain(|o| {
                            if o.len() >= 3 && o[0] == "SELL" {
                                let item = o[1].as_str();
                                return eng_price(item, inv.get(item).i64()) >= floor * eng_base(item);
                            }
                            true
                        });
                    }
                }
            }
            // supply_cap: VOLUME-PRESERVING anti-glut (replaces the broken
            // scarcity_hold, which DROPPED sells and stranded volume -> -$80k).
            // For each tape SELL, sell only up to eng_cap(floor) -- the qty that
            // keeps the marginal price >= floor*base -- and CARRY the excess to
            // later steps (town consume refills demand every 4 steps), draining
            // the carry when the market has room. Nothing is lost: the physical
            // stock stays in the shed and the endgame dump clears any residue, so
            // total volume is preserved while price never crashes below the floor.
            // budget_guard (listed AFTER this) still re-adds any selling the
            // block's purchases need, so deferral never starves cash.
            "supply_cap" => {
                if ts.in_route {
                    // ALL supply_cap behaviour is config-tunable (build once, tune by
                    // config): floor = price floor for eng_cap; bound_future = never
                    // defer more than the tape still plans to sell later (prevents
                    // runaway hoarding); carry_max = hard cap on per-item carry, the
                    // excess is force-sold now rather than dumped at a crashed endgame.
                    let floor = self.cfg.knob("supply_cap", "floor", 0.9);
                    let bound_future = self.cfg.knob("supply_cap", "bound_future", 0.0) >= 0.5;
                    let carry_max = self.cfg.knob("supply_cap", "carry_max", 0.0) as i64;
                    let inv = obs.get("market").get("inventory");
                    let mut nb: Vec<Op> = Vec::with_capacity(ts.buys.len());
                    for o in std::mem::take(&mut ts.buys) {
                        if o.len() >= 3 && o[0] == "SELL" {
                            let item = o[1].as_str();
                            let want: i64 = o[2].parse().unwrap_or(0);
                            let owed = self.carry.get(item).copied().unwrap_or(0);
                            let total = want + owed; // fold in prior deferrals
                            let minv = inv.get(item).i64();
                            let mut sell = eng_cap(item, minv, total, floor).min(total).max(0);
                            let mut new_carry = (total - sell).max(0);
                            // escape valves so volume clears ACROSS the window:
                            if bound_future {
                                let fut = self.future_sells(item, step + 1);
                                if new_carry > fut {
                                    sell += new_carry - fut; // force-sell the un-planned excess
                                    new_carry = fut;
                                }
                            }
                            if carry_max > 0 && new_carry > carry_max {
                                sell += new_carry - carry_max;
                                new_carry = carry_max;
                            }
                            self.carry.insert(item.to_string(), new_carry.max(0));
                            if sell > 0 {
                                nb.push(vec!["SELL".into(), item.into(), sell.to_string()]);
                            }
                        } else {
                            nb.push(o);
                        }
                    }
                    // drain carry for items with no tape SELL this step
                    let pend: Vec<String> = self
                        .carry
                        .iter()
                        .filter(|(_, &v)| v > 0)
                        .map(|(k, _)| k.clone())
                        .collect();
                    for item in pend {
                        if nb.iter().any(|o| o.len() >= 2 && o[0] == "SELL" && o[1] == item) {
                            continue;
                        }
                        let owed = self.carry.get(&item).copied().unwrap_or(0);
                        let minv = inv.get(&item).i64();
                        let mut cap = eng_cap(&item, minv, owed, floor);
                        // carry_max force-drain: never let deferred stock exceed the cap
                        // (it would otherwise pile up to a crashed endgame dump).
                        if carry_max > 0 && owed - cap > carry_max {
                            cap = owed - carry_max;
                        }
                        if cap > 0 {
                            nb.push(vec!["SELL".into(), item.clone(), cap.to_string()]);
                            self.carry.insert(item.clone(), (owed - cap).max(0));
                        }
                    }
                    ts.buys = nb;
                }
            }
            // budget_guard: the CASH-FLOW BACKSTOP (block-boundary purchase coverage).
            "budget_guard" => {
                if ts.in_route {
                    let block = self.cfg.knob("budget_guard", "block", 72.0) as usize;
                    if block > 0 && step % block == 0 {
                        let mut buys = std::mem::take(&mut ts.buys);
                        self.budget_guard(obs, &mut buys, step, block);
                        ts.buys = buys;
                    }
                }
            }
            // supply_demand: OWN selling in the active window. Measured vs
            // tschinkel we SELL MORE units (+5645) yet BANK LESS (-9868): the tape
            // dumps its full scheduled qty in BURSTS, gluttoning the market and
            // cratering the price for every unit. This (a) SUPPRESSES the tape's
            // own SELLs in the window and (b) registers a sell-phase drain that,
            // every step, sells each shed item up to the engine-exact qty that
            // keeps the marginal price >= sd_floor*base -- a SMOOTH stream into
            // scarcity (supply to demand), never a pile. The market refills via
            // town consumption between steps, so steady eng_cap selling clears
            // production without a glut. Outside the window the tape sells
            // normally; terminal_rescue owns the endgame. Never sells more than
            // shed -> never contradicts production.
            "supply_demand" => {
                if ts.in_route {
                    let sd_from = self.cfg.knob("supply_demand", "sd_from", 145.0) as usize;
                    let sd_until = self.cfg.knob("supply_demand", "sd_until", 640.0) as usize;
                    if step >= sd_from && step < sd_until {
                        // (a) suppress the tape's own SELLs (the drain owns selling)
                        ts.buys.retain(|o| !(o.len() >= 1 && o[0] == "SELL"));
                        // (b) let sells() run the eng_cap drain this step
                        ts.sell_rails.push("supply_demand".to_string());
                    }
                }
            }
            // mlp_sell: same structural handoff as supply_demand, but the learned
            // SALE head (sells() block) owns the quantities. Suppress the tape's
            // SELLs in the window and enable the drain; the head decides per item.
            "mlp_sell" => {
                if ts.in_route {
                    let from = self.cfg.knob("mlp_sell", "nn_from", 145.0) as usize;
                    let until = self.cfg.knob("mlp_sell", "nn_until", 640.0) as usize;
                    if step >= from && step < until {
                        ts.buys.retain(|o| !(o.len() >= 1 && o[0] == "SELL"));
                        ts.sell_rails.push("mlp_sell".to_string());
                    }
                }
            }
            // sell-phase rails: recorded in list order, consumed by sells().
            "endgame" | "scarcity_sell" | "front_run" | "anti_dump" | "price_sell"
            | "terminal_rescue" | "opp_front_run" | "endgame_liquidate" => {
                ts.sell_rails.push(name.to_string());
            }
            _ => {} // unknown rail: no-op (a future rail declared before wiring)
        }
    }

    fn act(&mut self, obs: &Json) -> Row {
        let step = obs.get("step").i64().max(0) as usize;
        if step == 0 {
            self.reset();
        }
        self.record(obs); // per-step observable snapshot -> NN temporal features
        if std::env::var("KAGG_DUMP_FEATS").is_ok() {
            // parity harness: dump the live feature vector so a Python check can
            // confirm nn_feats matches extract.py _feat exactly. Gated by env.
            let f = self.nn_feats(obs, step);
            eprintln!("FEATS {} {}", step,
                f.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>().join(","));
        }
        self.dispatch(obs, step);

        // Structural pre-rail turn state: the route row (or a PASS default past
        // the route end) plus the front-run ledger fold-back onto THIS step's tape
        // SELLs. The ordered rails then mutate this in config-list order (G1.4).
        let mut ts = TurnState::new(step < self.route.len());
        if ts.in_route {
            let base = self.route[step].clone();
            ts.farmer = base.farmer;
            ts.hands = base.hands;
            ts.buys = base.market;
            if let Some(mut pulled) = self.pulled.remove(&step) {
                let taken = std::mem::take(&mut ts.buys);
                let mut adj: Vec<Op> = Vec::with_capacity(taken.len());
                for o in taken {
                    if o.len() >= 3 && o[0] == "SELL" {
                        if let Some(r) = pulled.get_mut(&o[1]) {
                            if *r > 0 {
                                let want: i64 = o[2].parse().unwrap_or(0);
                                let take = want.min(*r);
                                *r -= take;
                                let left = want - take;
                                if left > 0 {
                                    adj.push(vec!["SELL".into(), o[1].clone(), left.to_string()]);
                                }
                                continue;
                            }
                        }
                    }
                    adj.push(o);
                }
                ts.buys = adj;
            }
        }

        // G1.4 ORDERED RAIL REGISTRY: apply each guardrail by NAME in config-list
        // order. escalation/weed_repair/hand_align/cash_guard mutate `ts` here; the
        // sell-phase rails (endgame/scarcity_sell/front_run/anti_dump) record into
        // ts.sell_rails for sells() to consume in that same order. Each rail's own
        // `.g()` gates on/off, so a listed-but-off rail is a no-op (as before).
        for name in self.cfg.order.clone() {
            // rails log: snapshot the mutable turn state + press, apply, then
            // report whether THIS rail (when ON) actually changed anything or was
            // a silent no-op. A rail listed on:true that never reports "changed"
            // is a dead gate.
            let snap = if self.log.rails {
                Some((ts.farmer.clone(), ts.hands.clone(), ts.buys.clone(), ts.sell_rails.len(), self.press))
            } else {
                None
            };
            self.apply_rail(&name, &mut ts, obs, step);
            if let Some((f0, h0, b0, sr0, p0)) = snap {
                let on = self.cfg.g(&name);
                let changed = f0 != ts.farmer || h0 != ts.hands || b0 != ts.buys
                    || sr0 != ts.sell_rails.len() || p0 != self.press;
                if changed {
                    *self.rail_hits.entry(name.clone()).or_insert(0) += 1;
                    eprintln!(
                        "LOG rail step={} name={} on={} CHANGED (buys {}->{} sell_rails {}->{} press {}->{})",
                        step, name, on as u8, b0.len(), ts.buys.len(), sr0, ts.sell_rails.len(), p0 as u8, self.press as u8
                    );
                } else if on {
                    eprintln!("LOG rail step={} name={} on=1 no-op", step, name);
                }
            }
        }
        if ts.in_route {
            ts.buys.truncate(MARKET_CAP);
        }

        let TurnState { farmer, hands, buys, sell_rails, .. } = ts;
        let mut market = self.sells(obs, buys.len(), step, &sell_rails);
        if self.log.sells && !market.is_empty() {
            eprintln!(
                "LOG sells step={} active_sell_rails=[{}] produced={} orders={:?}",
                step, sell_rails.join(","), market.len(), market
            );
        }
        market.extend(buys);
        market.truncate(MARKET_CAP);
        market.retain(|o| {
            if o.is_empty() {
                return false;
            }
            let head = o[0].as_str();
            if !matches!(
                head,
                "SELL" | "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL" | "BUY_LAND" | "HIRE"
            ) {
                return false;
            }
            let needs_qty =
                matches!(head, "SELL" | "BUY_SEED" | "BUY_PRODUCT" | "BUY_ANIMAL");
            if needs_qty {
                if o.len() < 3 {
                    return false;
                }
                matches!(o[2].parse::<i64>(), Ok(n) if n > 0)
            } else {
                true
            }
        });
        if self.log.turn {
            let me = Self::me(obs);
            let (m, o) = (Self::money(obs, me), Self::money(obs, 1 - me));
            eprintln!(
                "LOG turn step={} day={} me={:.0} opp={:.0} gap={:+.0} orders={} route_len={}",
                step, step / 24, m, o, m - o, market.len(), self.route.len()
            );
        }
        let n_hands = obs.get("farms").idx(Self::me(obs)).get("hands").arr().len();
        let mut hands = hands;
        hands.truncate(n_hands);
        while hands.len() < n_hands {
            hands.push(op1("PASS"));
        }
        Row { farmer, hands, market }
    }
}

/// `kagg mbandit <config.json> <branches.json> <base.tape>`: the stdio bridge.
pub fn play(args: &[String]) {
    // positional after the subcommand; ignore any --flag tokens (e.g. main.py's
    // --budget-ms, which this path has no search to spend).
    let pos: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
    let cfg_path = pos.first().map(|s| s.as_str()).unwrap_or("config.json");
    let br_path = pos.get(1).map(|s| s.as_str()).unwrap_or("branches.json");
    let base_path = pos.get(2).map(|s| s.as_str()).unwrap_or("base.tape");
    let cfg = load_config(&std::fs::read_to_string(cfg_path).unwrap_or_default());
    let branches = load_branches(&std::fs::read_to_string(br_path).unwrap_or_default());
    let base = parse_tape(&std::fs::read_to_string(base_path).unwrap_or_default());

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut b = MBandit::new(base, branches, cfg);
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
            b.reset();
            writeln!(out, "{{\"ok\": true}}").ok();
            out.flush().ok();
            continue;
        }
        let resp = match parse(line) {
            Ok(obs) => {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.act(&obs)));
                match r {
                    Ok(row) => row_json(&row),
                    Err(_) => "{\"farmer\": [\"PASS\"], \"hands\": [], \"market\": []}".to_string(),
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
