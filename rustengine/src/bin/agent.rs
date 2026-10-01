//! v51 COMPILED agent — trackp lane (closed-loop, quantity-conserved front-run
//! + cash-reserve floor + F-B weed-repair validator + F-D opponent-pressure
//! escalation, on the rebased 2026-09-11 economy). Reads one observation JSON
//! per line on stdin, writes one action JSON per line on stdout. main.py
//! spawns this once per episode and degrades to the inlined Python fallback on
//! any failure. This is a faithful port of agents/v51_trackp.py;
//! tests/test_agent_equiv.py asserts action-for-action equivalence.
//!
//! Build (native, for equivalence testing):  cargo build --release --bin agent
//! Build (Kaggle Linux):  x86_64-unknown-linux-musl via scripts/build_v50_linux.ps1

use kaggriculture_engine::json::{self, Json};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

// Economy tables are DATA, read at run time (they used to be include_str!'d; since the 2026-10-01 data purge the repo
// carries none). Folder: $KAGG_AGENTDATA_DIR, default rustengine/agentdata (relative to the working directory).
fn agentdata(name: &str) -> String {
    let dir = std::env::var("KAGG_AGENTDATA_DIR").unwrap_or_else(|_| "rustengine/agentdata".to_string());
    std::fs::read_to_string(std::path::Path::new(&dir).join(name)).unwrap_or_else(|_| "[]".to_string())
}

const MARKET_CAP: usize = 10;
const FINAL_DUMP: i64 = 700;
const CASH_FLOOR: f64 = 180.0;
const SWEEP: i64 = 40;
const FR_FROM: i64 = 145;
const FR_LOOK: i64 = 1;
// F-D escalation (KAGG_ESC defaults on in the Python; constants mirror it)
const ESC_GAP: f64 = 800.0;
const ESC_FROM: i64 = 168;
const ESC_SWEEP: i64 = 24;
const ESC_LOOK: i64 = 2;
const SELL_PRIORITY: [&str; 7] = ["MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"];
const ALLP: [&str; 9] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];

/// One tape turn. farmer/hands are passed through verbatim; market is the raw
/// list of order arrays (each order is a Vec<Json>).
struct Turn {
    farmer: Json,
    hands: Json,
    market: Vec<Vec<Json>>,
}

fn load_tape(src: &str) -> Vec<Turn> {
    let j = json::parse(src).expect("tape parse");
    j.arr()
        .iter()
        .map(|t| Turn {
            farmer: t.get("farmer").clone(),
            hands: t.get("hands").clone(),
            market: t
                .get("market")
                .arr()
                .iter()
                .map(|o| o.arr().to_vec())
                .collect(),
        })
        .collect()
}

fn ord_op(o: &[Json]) -> &str {
    o.get(0).map(|x| x.str()).unwrap_or("")
}
fn ord_item(o: &[Json]) -> &str {
    o.get(1).map(|x| x.str()).unwrap_or("")
}
fn ord_qty(o: &[Json]) -> i64 {
    o.get(2).map(|x| x.i64()).unwrap_or(0)
}

/// Serialize a Json value, emitting integral numbers as integers (Python emits
/// ints as ints; the engine wants int qtys).
fn ser(j: &Json, out: &mut String) {
    match j {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 9.0e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{}", n));
            }
        }
        Json::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    _ => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Arr(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ser(v, out);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            out.push('{');
            for (i, (k, v)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                ser(&Json::Str(k.clone()), out);
                out.push(':');
                ser(v, out);
            }
            out.push('}');
        }
    }
}

fn sell_order(item: &str, qty: i64) -> Vec<Json> {
    vec![
        Json::Str("SELL".to_string()),
        Json::Str(item.to_string()),
        Json::Num(qty as f64),
    ]
}

struct Agent {
    egg: Vec<Turn>,
    yarn: Vec<Turn>,
    fork_yarn: Option<bool>,
    pulled: HashMap<i64, HashMap<String, i64>>,
    press: bool,
    wrep: HashMap<usize, (i64, Json)>,
}

impl Agent {
    fn new() -> Self {
        Agent {
            egg: load_tape(&agentdata("trackp_egg.json")),
            yarn: load_tape(&agentdata("trackp_yarn.json")),
            fork_yarn: None,
            pulled: HashMap::new(),
            press: false,
            wrep: HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.fork_yarn = None;
        self.pulled.clear();
        self.press = false;
        self.wrep.clear();
    }

    fn shed_map(obs: &Json) -> HashMap<String, i64> {
        let mut m = HashMap::new();
        for (k, v) in obs.get("private").get("shed").obj() {
            m.insert(k.clone(), v.i64());
        }
        m
    }

    fn money(obs: &Json) -> f64 {
        let me = obs.get("player").i64() as usize;
        let farms = obs.get("farms").arr();
        if me < farms.len() {
            farms[me].get("money").f64()
        } else {
            1e9
        }
    }

    fn cash_guard(obs: &Json, buys: Vec<Vec<Json>>) -> Vec<Vec<Json>> {
        if CASH_FLOOR <= 0.0 || Self::money(obs) >= CASH_FLOOR {
            return buys;
        }
        buys.into_iter()
            .filter(|o| {
                if o.is_empty() {
                    return true;
                }
                let op = ord_op(o);
                if op == "SELL" || op == "HIRE" || op == "BUY_ANIMAL" {
                    return true;
                }
                let item = ord_item(o);
                if item == "WHEAT" {
                    return true;
                }
                !(op == "BUY_SEED" || op == "BUY_PRODUCT")
            })
            .collect()
    }

    fn sells(&mut self, obs: &Json, buys_used: usize, use_yarn: bool) -> Vec<Vec<Json>> {
        let slots = MARKET_CAP as i64 - buys_used as i64;
        if slots <= 0 {
            return vec![];
        }
        let slots = slots as usize;
        let shed = Self::shed_map(obs);
        let step = obs.get("step").i64();
        let mut out: Vec<Vec<Json>> = vec![];
        let get = |p: &str| *shed.get(p).unwrap_or(&0);

        if step >= FINAL_DUMP {
            for p in ALLP.iter() {
                if out.len() >= slots {
                    break;
                }
                if get(p) > 0 {
                    out.push(sell_order(p, 1000));
                }
            }
            return out;
        }
        // SWEEP: sell product the tape stranded above the safety threshold;
        // under F-D pressure the threshold tightens.
        let sweep = if self.press { std::cmp::min(SWEEP, ESC_SWEEP) } else { SWEEP };
        for p in SELL_PRIORITY.iter() {
            if out.len() >= slots {
                break;
            }
            let have = get(p);
            if have >= sweep {
                out.push(sell_order(p, have - 8));
            }
        }
        // CLOSED-LOOP front-run (quantity-conserved); pressure widens lookahead
        let fr_look = if self.press { std::cmp::max(FR_LOOK, ESC_LOOK) } else { FR_LOOK };
        if step >= FR_FROM {
            let tape = if use_yarn { &self.yarn } else { &self.egg };
            let mut already: Vec<String> = out.iter().map(|o| ord_item(o).to_string()).collect();
            let tmax = std::cmp::min(step + 1 + fr_look, tape.len() as i64);
            let mut t = step + 1;
            'outer: while t < tmax {
                for o in tape[t as usize].market.iter() {
                    if out.len() >= slots {
                        break 'outer;
                    }
                    if ord_op(o) != "SELL" || o.len() < 3 {
                        continue;
                    }
                    let p = ord_item(o).to_string();
                    if p == "WHEAT" || p == "FERTILIZER" || already.contains(&p) {
                        continue;
                    }
                    let pulled_prev = *self.pulled.get(&t).and_then(|m| m.get(&p)).unwrap_or(&0);
                    let q = std::cmp::min(ord_qty(o) - pulled_prev, get(&p));
                    if q < 1 {
                        continue;
                    }
                    out.push(sell_order(&p, q));
                    already.push(p.clone());
                    self.pulled.entry(t).or_default().insert(p, pulled_prev + q);
                }
                t += 1;
            }
        }
        out
    }

    /// F-D: the opponent farm is public; latch pressure at day boundaries
    /// from day 7 while their bank outpaces ours by ESC_GAP.
    fn pressure(&mut self, obs: &Json) {
        let step = obs.get("step").i64();
        if step < ESC_FROM || step % 24 != 0 {
            return;
        }
        let me = obs.get("player").i64() as usize;
        let farms = obs.get("farms").arr();
        if farms.len() < 2 || me >= 2 {
            return;
        }
        let mine = farms[me].get("money").f64();
        let theirs = farms[1 - me].get("money").f64();
        self.press = (theirs - mine) > ESC_GAP;
    }

    /// F-B: a tape op that would silently no-op on a WEED tile becomes DIG;
    /// a displaced PLANT/BUILD is replayed next turn.
    fn weed_repair(&mut self, obs: &Json, farmer: Json, hands: Json, step: i64) -> (Json, Json) {
        let me = obs.get("player").i64() as usize;
        let farms = obs.get("farms").arr();
        if me >= farms.len() {
            return (farmer, hands);
        }
        let farm = &farms[me];
        let mut positions: Vec<(i64, i64)> = vec![];
        {
            let fp = farm.get("farmer").arr();
            if fp.len() >= 2 {
                positions.push((fp[0].i64(), fp[1].i64()));
            } else {
                positions.push((0, 0));
            }
        }
        for p in farm.get("hands").arr() {
            let pa = p.arr();
            if pa.len() >= 2 {
                positions.push((pa[0].i64(), pa[1].i64()));
            } else {
                positions.push((-1, -1));
            }
        }
        let mut ops: Vec<Json> = vec![farmer];
        for h in hands.arr() {
            ops.push(h.clone());
        }
        // one-turn replay of displaced ops
        let keys: Vec<usize> = self.wrep.keys().cloned().collect();
        for k in keys {
            let (born, op) = self.wrep.get(&k).cloned().unwrap();
            if step - born == 1 {
                if k < ops.len() {
                    ops[k] = op;
                }
            }
            if step - born >= 1 {
                self.wrep.remove(&k);
            }
        }
        let tiles = farm.get("tiles").arr();
        for i in 0..ops.len() {
            let opa = ops[i].arr();
            if opa.is_empty() || i >= positions.len() {
                continue;
            }
            let (x, y) = positions[i];
            if x < 0 || y < 0 || (y as usize) >= tiles.len() {
                continue;
            }
            let row = tiles[y as usize].arr();
            if (x as usize) >= row.len() {
                continue;
            }
            let tile = &row[x as usize];
            if tile.get("kind").str() != "WEED" {
                continue;
            }
            let op0 = opa.get(0).map(|v| v.str()).unwrap_or("");
            if op0 == "PLANT" || op0 == "BUILD_PASTURE" || op0 == "BUILD_COOP" {
                self.wrep.insert(i, (step, ops[i].clone()));
                ops[i] = Json::Arr(vec![Json::Str("DIG".to_string())]);
            } else if op0 == "WATER" || op0 == "HARVEST" {
                ops[i] = Json::Arr(vec![Json::Str("DIG".to_string())]);
            }
        }
        let farmer = ops.remove(0);
        (farmer, Json::Arr(ops))
    }

    fn act(&mut self, obs: &Json) -> String {
        let step = obs.get("step").i64();
        if step == 0 {
            self.reset();
        }
        let shops = obs.get("town").get("unlocked_shops").arr();
        let has_yarn = shops.iter().any(|s| s.str() == "YARN_STORE");
        if self.fork_yarn.is_none() && (step >= 145 || has_yarn) {
            self.fork_yarn = Some(has_yarn);
        }
        let use_yarn = self.fork_yarn == Some(true);
        self.pressure(obs);

        let (farmer, hands, mut buys): (Json, Json, Vec<Vec<Json>>) = {
            let tape = if use_yarn { &self.yarn } else { &self.egg };
            if (step as usize) < tape.len() {
                let base = &tape[step as usize];
                let mut buys: Vec<Vec<Json>> = base.market.clone();
                // honour front-run pulls: subtract this step's already-pulled SELLs
                if let Some(mut pmap) = self.pulled.remove(&step) {
                    let mut adj: Vec<Vec<Json>> = vec![];
                    for o in buys.into_iter() {
                        let it = ord_item(&o).to_string();
                        let pv = *pmap.get(&it).unwrap_or(&0);
                        if ord_op(&o) == "SELL" && o.len() >= 3 && pv > 0 {
                            let take = std::cmp::min(ord_qty(&o), pv);
                            pmap.insert(it.clone(), pv - take);
                            let left = ord_qty(&o) - take;
                            if left > 0 {
                                adj.push(sell_order(&it, left));
                            }
                        } else {
                            adj.push(o);
                        }
                    }
                    buys = adj;
                }
                (base.farmer.clone(), base.hands.clone(), buys)
            } else {
                (
                    Json::Arr(vec![Json::Str("PASS".to_string())]),
                    Json::Arr(vec![]),
                    vec![],
                )
            }
        };
        let (farmer, hands) = if (step as usize) < self.egg.len() {
            self.weed_repair(obs, farmer, hands, step)
        } else {
            (farmer, hands)
        };
        if (step as usize) < self.egg.len() {
            buys = Self::cash_guard(obs, buys);
            buys.truncate(MARKET_CAP);
        }
        let buys_used = buys.len();
        let mut market = self.sells(obs, buys_used, use_yarn);
        market.extend(buys);
        market.truncate(MARKET_CAP);

        // serialize action
        let mut s = String::from("{\"farmer\":");
        ser(&farmer, &mut s);
        s.push_str(",\"hands\":");
        ser(&hands, &mut s);
        s.push_str(",\"market\":[");
        for (i, o) in market.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            ser(&Json::Arr(o.clone()), &mut s);
        }
        s.push_str("]}");
        s
    }
}

fn main() {
    let mut agent = Agent::new();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let obs = json::parse(&line).unwrap_or(Json::Null);
            agent.act(&obs)
        })) {
            Ok(r) => r,
            Err(_) => "{\"farmer\":[\"PASS\"],\"hands\":[],\"market\":[]}".to_string(),
        };
        if writeln!(out, "{}", reply).is_err() {
            break;
        }
        let _ = out.flush();
    }
}
