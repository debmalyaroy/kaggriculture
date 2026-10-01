//! Compiled BANDIT seat: the proven v56y tape-in-shell, ported to Rust so it
//! ships as a static musl binary (no Python-env fragility on the ladder).
//!
//! Faithful port of agents/v56y_trackp.py at its shipped defaults: two embedded
//! 719-step tapes (EGG / YARN), a t145 world fork on YARN_STORE, and the reactive
//! shell -- weed-repair (a tape op that would no-op on a WEED becomes DIG, a
//! displaced PLANT/BUILD replays next turn), a premium-first sell overlay with a
//! quantity-conserved one-turn front-run ledger, an escalation-pressure latch,
//! and a bankruptcy cash-guard. Validated action-for-action against the Python
//! agent (a faithful port of agents/v56y_trackp.py).
//!
//! Protocol identical to `kagg play`: one observation JSON per line on stdin,
//! one action JSON per line on stdout. Every failure answers with a legal PASS.

use crate::json::{parse, Json};
use crate::core::{row_json, Row};
use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};

// Economy tapes are DATA, read at run time (they used to be include_str!'d; since the 2026-10-01 data purge the
// repo carries no tapes, so the engine must build without them). Folder: $KAGG_TAPE_DIR, default rustengine/tapes
// (relative to the working directory). A missing tape loads as an empty economy (the shell then plays PASS rows).
//   v56y family (bandit seat: backbone+premium EGG/YARN fork): bandit_egg.tape, bandit_yarn.tape
//   v57 family (trackp seat: the v51-rebased strongest balanced economy): bandit_v57_egg.tape, bandit_v57_yarn.tape
fn tape_src(name: &str) -> String {
    let dir = std::env::var("KAGG_TAPE_DIR").unwrap_or_else(|_| "rustengine/tapes".to_string());
    std::fs::read_to_string(std::path::Path::new(&dir).join(name)).unwrap_or_default()
}

// shipped defaults (env knobs in the Python resolve to these)
const MARKET_CAP: usize = 10;
const FINAL_DUMP: usize = 700;
const SWEEP: i64 = 40;
const ESC_SWEEP: i64 = 24;
const ESC_GAP: f64 = 800.0;
const ESC_FROM: usize = 168;
const ESC_LOOK: usize = 2;
const FR_FROM: usize = 145;
const FR_LOOK: usize = 1;
const CASH_FLOOR: f64 = 180.0;

const SELL_PRIORITY: [&str; 7] =
    ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"];
const ALLP: [&str; 9] = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL",
    "FERTILIZER",
];

type Op = Vec<String>;

fn op1(a: &str) -> Op {
    vec![a.to_string()]
}

/// Parse one embedded tape line "farmer\thands\tmarket" into a Row. Each field
/// is `;`-separated ops, each op space-separated tokens.
fn parse_line(line: &str) -> Row {
    let mut parts = line.split('\t');
    let f = parts.next().unwrap_or("PASS");
    let h = parts.next().unwrap_or("");
    let m = parts.next().unwrap_or("");
    let toks = |s: &str| -> Op { s.split(' ').filter(|t| !t.is_empty()).map(|t| t.to_string()).collect() };
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

// ------------------------------------------------------------------ config --
//
// G1.1 (optional config surface, 2026-09-18): `kagg bandit` was 100% compile
// time -- include_str! tapes, const thresholds, a hardcoded D6 fork at 144/145.
// The ~6 live measurement consumers (gate_rustbandit/gate_basetape/
// loss_consensus/game_insight/test_basetape/densify_tape) invoke it WITHOUT a
// config and must stay bit-identical, so the whole surface is OPTIONAL: with no
// `KAGG_BANDIT_CONFIG` env path, `BanditCfg::default()` resolves to exactly the
// shipped constants below and nothing changes. A provided config overrides only
// the fields it names (tape / thresholds / fork); absent fields keep the const.
struct BanditCfg {
    final_dump: usize,
    sweep: i64,
    esc_sweep: i64,
    esc_gap: f64,
    esc_from: usize,
    esc_look: usize,
    fr_from: usize,
    fr_look: usize,
    cash_floor: f64,
    fork_dispatch_step: usize, // D6 economy dispatch (default 144)
    fork_early_step: usize,    // early yarn/egg latch (default 145)
    egg_override: Option<Vec<Row>>,  // replaces the ACTIVE family's EGG tape
    yarn_override: Option<Vec<Row>>, // replaces the ACTIVE family's YARN tape
}

impl Default for BanditCfg {
    /// The shipped compiled defaults -- the exact constants above. This is the
    /// path every current consumer takes (no config), so it MUST stay identical.
    fn default() -> Self {
        BanditCfg {
            final_dump: FINAL_DUMP,
            sweep: SWEEP,
            esc_sweep: ESC_SWEEP,
            esc_gap: ESC_GAP,
            esc_from: ESC_FROM,
            esc_look: ESC_LOOK,
            fr_from: FR_FROM,
            fr_look: FR_LOOK,
            cash_floor: CASH_FLOOR,
            fork_dispatch_step: 144,
            fork_early_step: 145,
            egg_override: None,
            yarn_override: None,
        }
    }
}

/// Load an OPTIONAL bandit config from `KAGG_BANDIT_CONFIG`. Returns the default
/// (compiled constants) when the env var is unset, the file is missing, or the
/// JSON is unparseable -- a bad config never silently ships a broken seat, it
/// just plays today's proven defaults. Schema (every field optional):
///   { "thresholds": { final_dump,sweep,esc_sweep,esc_gap,esc_from,esc_look,
///                     fr_from,fr_look,cash_floor },
///     "fork": { dispatch_step, early_step },
///     "egg_tape": "path.tape", "yarn_tape": "path.tape" }
fn load_bandit_cfg() -> BanditCfg {
    let mut c = BanditCfg::default();
    let path = match std::env::var("KAGG_BANDIT_CONFIG") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return c,
    };
    let Ok(src) = std::fs::read_to_string(&path) else { return c };
    let Ok(j) = parse(&src) else { return c };
    let th = j.get("thresholds");
    let u = |v: &Json, d: usize| if v.is_null() { d } else { v.i64().max(0) as usize };
    let i = |v: &Json, d: i64| if v.is_null() { d } else { v.i64() };
    let f = |v: &Json, d: f64| if v.is_null() { d } else { v.f64() };
    c.final_dump = u(th.get("final_dump"), c.final_dump);
    c.sweep = i(th.get("sweep"), c.sweep);
    c.esc_sweep = i(th.get("esc_sweep"), c.esc_sweep);
    c.esc_gap = f(th.get("esc_gap"), c.esc_gap);
    c.esc_from = u(th.get("esc_from"), c.esc_from);
    c.esc_look = u(th.get("esc_look"), c.esc_look);
    c.fr_from = u(th.get("fr_from"), c.fr_from);
    c.fr_look = u(th.get("fr_look"), c.fr_look);
    c.cash_floor = f(th.get("cash_floor"), c.cash_floor);
    let fk = j.get("fork");
    c.fork_dispatch_step = u(fk.get("dispatch_step"), c.fork_dispatch_step);
    c.fork_early_step = u(fk.get("early_step"), c.fork_early_step);
    // Tape overrides: parse like the embedded tapes; require a full 700+ row tape
    // (a header-only or truncated file is ignored, never half-shipped).
    let load_tape = |p: &str| -> Option<Vec<Row>> {
        let src = std::fs::read_to_string(p).ok()?;
        let rows: Vec<Row> = src.lines().filter(|l| l.contains('\t')).map(parse_line).collect();
        if rows.len() > 700 { Some(rows) } else { None }
    };
    if !j.get("egg_tape").is_null() {
        c.egg_override = load_tape(j.get("egg_tape").str());
    }
    if !j.get("yarn_tape").is_null() {
        c.yarn_override = load_tape(j.get("yarn_tape").str());
    }
    c
}

/// One economy = an EGG-fork tape + a YARN-fork tape. Openings are shared across
/// economies (verified desync-free), so the D6 dispatch can switch the tail.
struct Econ {
    name: &'static str,
    egg: Vec<Row>,
    yarn: Vec<Row>,
}

pub struct Bandit {
    econs: Vec<Econ>,     // full economy registry (all economies)
    base: usize,          // seat's default economy index
    active: usize,        // economy chosen at D6 (== base until dispatched)
    fork: Option<bool>,                          // Some(true)=YARN, Some(false)=EGG
    dispatched: bool,                             // D6 economy chosen yet
    pulled: HashMap<usize, HashMap<String, i64>>, // front-run ledger, keyed by tape step
    wrep: HashMap<usize, (usize, Op)>,            // weed-repair: hand idx -> (born_step, op)
    press: bool,
    amode: i8, // compat animal routing (KAGG_COMPAT): 0 hedge, 1 wool/sheep, 2 milk/cow
    cfg: BanditCfg, // G1.1: OPTIONAL config surface; default == compiled constants
}

impl Bandit {
    /// `family` names the seat's DEFAULT economy ("v57" trackp, else "v56y"
    /// bandit). All economies are always loaded; the D6 dispatch picks among them.
    /// Uses the compiled-constant defaults (the path all current consumers take).
    pub fn new_family(family: &str) -> Self {
        Self::new_family_cfg(family, BanditCfg::default())
    }

    /// G1.1: as `new_family`, but with an explicit (optionally config-loaded)
    /// `BanditCfg`. With `BanditCfg::default()` this is byte-for-byte the old
    /// constructor, so `new_family`/the 6 consumers are unaffected.
    pub fn new_family_cfg(family: &str, mut cfg: BanditCfg) -> Self {
        let mut econs = vec![
            Econ { name: "v56y", egg: parse_tape(&tape_src("bandit_egg.tape")), yarn: parse_tape(&tape_src("bandit_yarn.tape")) },
            Econ { name: "v57", egg: parse_tape(&tape_src("bandit_v57_egg.tape")), yarn: parse_tape(&tape_src("bandit_v57_yarn.tape")) },
        ];
        // KAGG_BASETAPE: replace the base economy's EGG tape with a candidate tape
        // from disk (a real mined LB frontier tape) to validate it through our shell.
        // Rows lacking a tab (headers) are dropped. Off-by-default test hook.
        if let Ok(path) = std::env::var("KAGG_BASETAPE") {
            if let Ok(src) = std::fs::read_to_string(&path) {
                let egg: Vec<Row> = src.lines().filter(|l| l.contains('\t')).map(parse_line).collect();
                if egg.len() > 700 {
                    econs[0].egg = egg;
                }
            }
        }
        let base = econs.iter().position(|e| e.name == family).unwrap_or(0);
        // G1.1: optional config tape overrides apply to the ACTIVE family only.
        if let Some(egg) = cfg.egg_override.take() {
            econs[base].egg = egg;
        }
        if let Some(yarn) = cfg.yarn_override.take() {
            econs[base].yarn = yarn;
        }
        Bandit {
            econs,
            base,
            active: base,
            fork: None,
            dispatched: false,
            pulled: HashMap::new(),
            wrep: HashMap::new(),
            press: false,
            amode: 0,
            cfg,
        }
    }

    pub fn new() -> Self {
        Self::new_family("v56y")
    }

    /// The active economy's tape for the current (egg/yarn) sub-fork.
    fn tape_for(&self, is_yarn: bool) -> &Vec<Row> {
        let e = &self.econs[self.active];
        if is_yarn { &e.yarn } else { &e.egg }
    }

    fn reset(&mut self) {
        self.fork = None;
        self.active = self.base;
        self.dispatched = false;
        self.pulled.clear();
        self.wrep.clear();
        self.press = false;
        self.amode = 0;
    }

    /// COMPAT (KAGG_COMPAT): relabel COW<->SHEEP consistently in an op vec so the
    /// post-D6 animal economy matches the world's demand (wool world -> all sheep;
    /// milk world -> all cow). Both are pasture animals with the same op grammar,
    /// so the relabel keeps buy/place/pickup internally consistent; the reactive
    /// shell absorbs the small cash/production difference. Applied only when the
    /// D6 shop read is unambiguous (amode != 0), leaving the pre-D6 hedge intact.
    fn compat_swap(amode: i8, ops: &mut [Op]) {
        let (from, to) = match amode {
            1 => ("COW", "SHEEP"),
            2 => ("SHEEP", "COW"),
            _ => return,
        };
        for op in ops.iter_mut() {
            for tok in op.iter_mut() {
                if tok == from {
                    *tok = to.to_string();
                }
            }
        }
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

    fn pressure(&mut self, obs: &Json, step: usize) {
        if step < self.cfg.esc_from || step % 24 != 0 {
            return;
        }
        let me = Self::me(obs);
        let mine = Self::money(obs, me);
        let theirs = Self::money(obs, 1 - me);
        self.press = (theirs - mine) > self.cfg.esc_gap;
    }

    /// weed-repair: mirrors agents/v56y_trackp.py `_weed_repair`.
    fn weed_repair(&mut self, obs: &Json, farmer: Op, hands: Vec<Op>, step: usize) -> (Op, Vec<Op>) {
        let me = Self::me(obs);
        let farm = obs.get("farms").idx(me);
        // positions: [farmer] + hands, each [x, y]
        let mut positions: Vec<(i64, i64)> = Vec::new();
        let f = farm.get("farmer");
        positions.push((f.idx(0).i64(), f.idx(1).i64()));
        for h in farm.get("hands").arr() {
            positions.push((h.idx(0).i64(), h.idx(1).i64()));
        }
        let mut ops: Vec<Op> = Vec::with_capacity(1 + hands.len());
        ops.push(farmer);
        ops.extend(hands);
        // pass 1: reinstate a displaced PLANT/BUILD one step after the DIG, then expire
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
        // pass 2: any op that would no-op on a WEED tile becomes DIG
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

    /// cash-guard: drop luxury BUY_SEED/BUY_PRODUCT (not WHEAT) below the floor.
    fn cash_guard(&self, obs: &Json, buys: Vec<Op>) -> Vec<Op> {
        let me = Self::me(obs);
        if Self::money(obs, me) >= self.cfg.cash_floor {
            return buys;
        }
        buys
            .into_iter()
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

    /// Daily market consumption of `product` = shops draining it (6/day each,
    /// x2 for single-product shops) + town center (1/day, except fertilizer).
    /// Selling above this rate grows inventory -> price falls (glut). Premium
    /// (base>$100) hits the $1 floor fastest, so we cap premium sells to demand.
    fn daily_demand(obs: &Json, product: &str) -> i64 {
        let mut n = 0i64;
        for s in obs.get("town").get("unlocked_shops").arr() {
            let (demands, single): (&[&str], bool) = match s.str() {
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
            if demands.contains(&product) {
                n += if single { 2 } else { 1 };
            }
        }
        let mut d = n * 6; // 6 consumption ticks per day
        if product != "FERTILIZER" {
            d += 1; // town center, once/day
        }
        d
    }

    /// sell overlay: final dump, priority sweep, quantity-conserved front-run.
    fn sells(&mut self, obs: &Json, buys_used: usize, is_yarn: bool, step: usize) -> Vec<Op> {
        let mut out: Vec<Op> = Vec::new();
        if buys_used >= MARKET_CAP {
            return out;
        }
        let slots = MARKET_CAP - buys_used;
        if step >= self.cfg.final_dump {
            for p in ALLP {
                if out.len() >= slots {
                    break;
                }
                if Self::shed(obs, p) > 0 {
                    out.push(vec!["SELL".into(), p.into(), "1000".into()]);
                }
            }
            return out;
        }
        let sweep = if self.press { self.cfg.sweep.min(self.cfg.esc_sweep) } else { self.cfg.sweep };
        let demand_scale = std::env::var("KAGG_DEMANDSELL").map(|v| v != "0").unwrap_or(true);
        for p in SELL_PRIORITY {
            if out.len() >= slots {
                break;
            }
            let have = Self::shed(obs, p);
            if have >= sweep {
                // Demand-scaled cap: never sweep-sell a premium (base>$100) faster
                // than the shops can consume it (a day's demand), so the sweep
                // can't dump premium to the $1 floor. Non-premium (wheat/egg/fert,
                // dump-proof) keep the original have-8 sweep.
                let base_amt = have - 8;
                let amt = if demand_scale
                    && matches!(p, "MILK" | "WOOL" | "STRAWBERRY" | "MELON")
                {
                    base_amt.min(Self::daily_demand(obs, p).max(8))
                } else {
                    base_amt
                };
                out.push(vec!["SELL".into(), p.into(), amt.to_string()]);
            }
        }
        // one-turn front-run of the tape's own upcoming SELLs (quantity-conserved)
        let fr_look = if self.press { self.cfg.fr_look.max(self.cfg.esc_look) } else { self.cfg.fr_look };
        if step >= self.cfg.fr_from {
            // snapshot the lookahead markets first (drops the self borrow so the
            // pulled-ledger mutable borrow below is legal).
            let look: Vec<(usize, Vec<Op>)> = {
                let tape = self.tape_for(is_yarn);
                let end = (step + 1 + fr_look).min(tape.len());
                (step + 1..end).map(|t| (t, tape[t].market.clone())).collect()
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
        out
    }

    pub fn act(&mut self, obs: &Json) -> Row {
        let step = obs.get("step").i64().max(0) as usize;
        if step == 0 {
            self.reset();
        }
        let shops = obs.get("town").get("unlocked_shops");
        let yarn_present = shops.arr().iter().any(|s| s.str() == "YARN_STORE");
        if self.fork.is_none() && (step >= self.cfg.fork_early_step || yarn_present) {
            self.fork = Some(yarn_present);
        }
        // D6 3-WAY ANIMAL TILT (t144), on the shared crop base:
        //   MILK-demand world (SMOOTHIE/ICE_CREAM/PIZZA) -> COW economy (v57), to
        //     capture milk value where the shops hold milk price up;
        //   WOOL-demand world (YARN_STORE)             -> SHEEP tilt (yarn fork);
        //   otherwise                                  -> GEESE/EGG (dump-proof
        //     default) -- log-priced egg can't be crashed, so it is the robust
        //     anchor for a fixed tape that can't adaptively dodge dumps.
        // Decided once at D6 from the OBSERVED shops (both seats).
        // D6 tilt (tested-optimal, 2026-09-15): the GEESE/EGG economy is DUMP-PROOF
        // (egg glut-price is log-shaped), so it holds value under a contesting
        // opponent -- measured to BEAT a cow/milk switch by +22k in milk worlds.
        // So: WOOL demand -> sheep (yarn fork); everything else -> dump-proof egg
        // default. A cow/milk switch was tested and DROPPED (crashes under
        // contention). Seat base stays the seat's family; only the sub-fork tilts.
        if !self.dispatched && step >= self.cfg.fork_dispatch_step {
            let wool = shops.arr().iter().filter(|s| s.str() == "YARN_STORE").count();
            self.fork = Some(wool > 0); // true = SHEEP/WOOL, false = GEESE/EGG (dump-proof)
            self.dispatched = true;
            // COMPAT (KAGG_COMPAT): route the post-D6 pasture herd to the world's
            // demand. YARN present & no milk shop -> wool (all sheep); milk shop
            // (PIZZA/ICE_CREAM/SMOOTHIE) present & no YARN -> milk (all cow); mixed
            // or neither -> hedge (no swap). Anticipatory: leaves the pre-D6 herd.
            if std::env::var("KAGG_COMPAT").map(|v| v != "0" && !v.is_empty()).unwrap_or(false) {
                let has = |name: &str| shops.arr().iter().any(|s| s.str() == name);
                let milk = has("PIZZA_SHOP") || has("ICE_CREAM_SHOP") || has("SMOOTHIE_SHOP");
                // WOOL-ONLY (full-panel gated 2026-09-16): the cow/milk tilt was net
                // NEGATIVE (FARMERS_MARKET|SMOOTHIE -6, PIZZA|PET_CAFE -2) -- one milk
                // shop gluts, so forcing cow beats neither the dump-proof egg default
                // nor keeping sheep. Only the wool tilt is a clean strict win (YARN
                // world: single-product shop consumes 2x, wool holds; milk worthless).
                self.amode = if wool > 0 && !milk {
                    1 // clear wool world (YARN, no milk shop) -> all sheep
                } else {
                    0 // everything else -> keep the proven hedge
                };
            }
        }
        let is_yarn = self.fork.unwrap_or(false);
        self.pressure(obs, step);

        let tape_len = self.tape_for(is_yarn).len();
        let (farmer, hands, mut buys): (Op, Vec<Op>, Vec<Op>);
        if step < tape_len {
            let base = self.tape_for(is_yarn)[step].clone();
            let (f2, h2) = self.weed_repair(obs, base.farmer, base.hands, step);
            farmer = f2;
            hands = h2;
            buys = base.market;
            // apply the front-run ledger for THIS step: reduce the tape's SELLs
            if let Some(mut pulled) = self.pulled.remove(&step) {
                let mut adj: Vec<Op> = Vec::with_capacity(buys.len());
                for o in buys.into_iter() {
                    if o.len() >= 3 && o[0] == "SELL" {
                        let rem = pulled.get_mut(&o[1]);
                        if let Some(r) = rem {
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
                buys = adj;
            }
            buys = self.cash_guard(obs, buys);
            buys.truncate(MARKET_CAP);
        } else {
            farmer = op1("PASS");
            hands = vec![];
            buys = vec![];
        }
        let mut market = self.sells(obs, buys.len(), is_yarn, step);
        market.extend(buys);
        // Truncate to the cap FIRST (exactly as v56y's `(...)[:MARKET_CAP]`),
        // THEN drop engine no-ops the bridge validator would reject: an
        // arg-taking op (SELL/BUY_*) with qty <= 0. Doing it in this order keeps
        // the emitted queue action-for-action identical to v56y-through-validator
        // (0 diffs) while clearing the transport gate (repairs 0). The tape's
        // "SELL X 0" placeholders are engine no-ops, so dropping them is free.
        market.truncate(MARKET_CAP);
        market.retain(|o| {
            if o.is_empty() {
                return false;
            }
            let head = o[0].as_str();
            // must be a real market op (drops tape "PASS"/no-op placeholders)
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
                match o[2].parse::<i64>() {
                    Ok(n) => n > 0,
                    Err(_) => false,
                }
            } else {
                true
            }
        });
        // Normalise hands to the LIVE crew count exactly as the bridge validator
        // would: trim to n_hands, pad the rest with PASS. The engine treats an
        // unspecified hand as PASS, so this changes no outcome -- it only means
        // the binary emits already-aligned actions (transport gate: repairs 0).
        let n_hands = obs.get("farms").idx(Self::me(obs)).get("hands").arr().len();
        let mut hands = hands;
        hands.truncate(n_hands);
        while hands.len() < n_hands {
            hands.push(op1("PASS"));
        }
        let mut farmer = farmer;
        // COMPAT: relabel COW<->SHEEP in the emitted ops per the D6-decided mode
        // (no-op when amode==0, i.e. KAGG_COMPAT off or an ambiguous world).
        Self::compat_swap(self.amode, std::slice::from_mut(&mut farmer));
        Self::compat_swap(self.amode, &mut hands);
        Self::compat_swap(self.amode, &mut market);
        Row { farmer, hands, market }
    }
}

impl Default for Bandit {
    fn default() -> Self {
        Self::new()
    }
}

/// `kagg bandit [family]`: the stdio bridge, mirroring `policy::play_with`.
pub fn play(family: &str) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    // G1.1: an OPTIONAL config (KAGG_BANDIT_CONFIG); unset -> compiled defaults,
    // i.e. bit-identical to the old `Bandit::new_family(family)` for the 6
    // measurement consumers that call `kagg bandit`/`kagg trackp` with no config.
    let mut b = Bandit::new_family_cfg(family, load_bandit_cfg());
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
