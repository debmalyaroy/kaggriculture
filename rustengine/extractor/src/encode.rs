//! Byte-exact Rust port of `kaggriculture.data.trackp_corpus.encode_tokens` /
//! `encode_action` / `split_of` / `world_family` (TOKEN_LAYOUT_VERSION = 1).
//!
//! Parity is the whole point: every value produced here must equal the Python
//! encoder's on the same replay, packed identically as little-endian int32.
//! `tests/test_trackp_rust_parity.py` gates any drift.
use serde_json::Value;
use sha1::{Digest, Sha1};

// --- frozen vocab (mirror of trackp_corpus.py) -------------------------------
pub const CROPS: [&str; 5] = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"];
pub const ANIMALS: [&str; 3] = ["COW", "GOOSE", "SHEEP"];
pub const PRODUCTS: [&str; 9] = [
    "CARROT", "EGG", "FERTILIZER", "MELON", "MILK", "STRAWBERRY", "TOMATO", "WHEAT", "WOOL",
];
pub const SHOPS: [&str; 8] = [
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP",
    "SMOOTHIE_SHOP", "YARN_STORE",
];
pub const MOVER_VERBS: [&str; 18] = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST", "FEED", "CARE",
    "PLACE", "PICKUP", "DROP", "DIG", "FERTILIZE", "COLLECT_FERTILIZER", "BUILD_PASTURE",
    "BUILD_COOP",
];
pub const MARKET_VERBS: [&str; 7] =
    ["PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"];

pub const TOK_W: usize = 14;
pub const MAX_TOKENS: usize = 160;
pub const MAX_HANDS: usize = 12;
pub const MAX_MARKET: usize = 10;
pub const ACT_W: usize = 2 + MAX_HANDS * 2 + MAX_MARKET * 3; // 56

pub const VAL_FRACTION: f64 = 0.15;
pub const LOSER_RATING_MIN: f64 = 2100.0;

// token type tags (match trackp_corpus.py)
const T_SHED: i32 = 6;
const T_MARKET: i32 = 7;
const T_HAND: i32 = 8;
const T_FARMER: i32 = 9;
const T_GAME: i32 = 10;

const NULL: Value = Value::Null;

/// Python `int(x)`: truncate toward zero; null/missing/non-number -> 0.
/// Public so the extractor's row builder matches Python's `int()` on
/// day/step/player (a JSON float like 6.0 truncates, it does not become 0).
pub fn as_int(v: &Value) -> i64 {
    ji(v)
}

/// Python `int(x)`: truncate toward zero; null/missing/non-number -> 0.
#[inline]
fn ji(v: &Value) -> i64 {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if let Some(u) = n.as_u64() {
                u as i64
            } else if let Some(f) = n.as_f64() {
                f.trunc() as i64
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Python truthiness.
#[inline]
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[inline]
fn pos(arr: &[&str], v: &Value) -> i32 {
    // Python `LIST_ID.get(name, -1) + 1`: found -> idx+1, else 0.
    match v.as_str() {
        Some(s) => arr.iter().position(|&c| c == s).map(|i| i as i32 + 1).unwrap_or(0),
        None => 0,
    }
}

#[inline]
fn verb_id(arr: &[&str], s: &str) -> i32 {
    arr.iter().position(|&c| c == s).map(|i| i as i32).unwrap_or(0)
}

/// VERB-AWARE mover arg (layout v2): PLANT->crop, PLACE->animal, PICKUP->product.
/// 1:1 within each verb -> no cross-item collisions.
#[inline]
fn mover_arg(verb: &str, v: &Value) -> i32 {
    match verb {
        "PLANT" => pos(&CROPS, v),
        "PLACE" => pos(&ANIMALS, v),
        "PICKUP" => pos(&PRODUCTS, v),
        _ => 0,
    }
}

/// VERB-AWARE market arg: BUY_SEED->crop, BUY_ANIMAL->animal, BUY_PRODUCT/SELL->product.
#[inline]
fn market_arg(verb: &str, v: &Value) -> i32 {
    match verb {
        "BUY_SEED" => pos(&CROPS, v),
        "BUY_ANIMAL" => pos(&ANIMALS, v),
        "BUY_PRODUCT" | "SELL" => pos(&PRODUCTS, v),
        _ => 0,
    }
}

/// Python `int(m[2]) if str(m[2]).lstrip("-").isdigit() else 0`.
#[inline]
fn qty(v: &Value) -> i32 {
    match v {
        Value::Number(n) => {
            // integers map through; floats stringify with "." -> not isdigit -> 0
            if n.is_i64() || n.is_u64() {
                ji(v) as i32
            } else {
                0
            }
        }
        Value::String(s) => {
            let t = s.strip_prefix('-').unwrap_or(s);
            if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
                s.parse::<i64>().unwrap_or(0) as i32
            } else {
                0
            }
        }
        _ => 0,
    }
}

#[inline]
fn inv_sum(o: &Value) -> i32 {
    match o.as_object() {
        Some(m) => m.values().map(|v| ji(v)).sum::<i64>() as i32,
        None => 0,
    }
}

#[inline]
fn push_tok(out: &mut Vec<i32>, row: &[i32]) {
    out.extend_from_slice(row);
    for _ in row.len()..TOK_W {
        out.push(0);
    }
}

#[inline]
fn geti(v: &Value, k: &str) -> i32 {
    ji(v.get(k).unwrap_or(&NULL)) as i32
}

/// Encode one seat-specific observation into a flat little-endian-ready int32
/// token buffer (len = n_tokens * TOK_W). Mirrors `encode_tokens(obs, seat)`.
pub fn encode_tokens(obs: &Value, seat: usize) -> Vec<i32> {
    let farms = obs.get("farms").and_then(|v| v.as_array());
    let me = farms.and_then(|f| f.get(seat)).unwrap_or(&NULL);
    let opp = farms.and_then(|f| f.get(1 - seat)).unwrap_or(&NULL);
    let priv_ = obs.get("private").unwrap_or(&NULL);
    let shed = priv_.get("shed").unwrap_or(&NULL);
    let invs = priv_.get("inventories").and_then(|v| v.as_array());
    let market = obs.get("market").unwrap_or(&NULL);
    let prices = market.get("prices").unwrap_or(&NULL);
    let minv = market.get("inventory").unwrap_or(&NULL);
    let day = ji(obs.get("day").unwrap_or(&NULL)) as i32;
    let hour = ji(obs.get("hour").unwrap_or(&NULL)) as i32;
    let step = ji(obs.get("step").unwrap_or(&NULL)) as i32;

    let mut out: Vec<i32> = Vec::with_capacity(MAX_TOKENS * TOK_W);
    let mut n = 0usize;

    // tiles (ours)
    if let Some(tiles) = me.get("tiles").and_then(|v| v.as_array()) {
        'outer: for (y, rowv) in tiles.iter().enumerate() {
            if let Some(rowa) = rowv.as_array() {
                for (x, t) in rowa.iter().enumerate() {
                    if !t.is_object() {
                        continue;
                    }
                    let kind = t.get("kind").and_then(|v| v.as_str()).unwrap_or("");
                    let tt = match kind {
                        "PLANT" => 1,
                        "PASTURE" => 2,
                        "COOP" => 3,
                        "WEED" => 4,
                        _ => continue,
                    };
                    let crop = pos(&CROPS, t.get("crop").unwrap_or(&NULL));
                    let anim = pos(&ANIMALS, t.get("animal").unwrap_or(&NULL));
                    let yld = geti(t, "yield_units");
                    let wat = truthy(t.get("watered_today").unwrap_or(&NULL)) as i32;
                    let care = truthy(t.get("cared_today").unwrap_or(&NULL)) as i32;
                    let fed = truthy(t.get("fed_today").unwrap_or(&NULL)) as i32;
                    let cunw = geti(t, "consecutive_unwatered");
                    let cunf = geti(t, "consecutive_unfed");
                    let fert = truthy(t.get("fertilizer_available").unwrap_or(&NULL)) as i32;
                    let pday = match t.get("planted_day") {
                        Some(v) if !v.is_null() => ji(v) as i32,
                        _ => day,
                    };
                    let plnt = (day - pday).max(0);
                    let cday = match t.get("placed_day") {
                        Some(v) if !v.is_null() => ji(v) as i32,
                        _ => day,
                    };
                    let plcd = (day - cday).max(0);
                    push_tok(
                        &mut out,
                        &[tt, x as i32, y as i32, crop, anim, yld, wat, care, fed, cunw, cunf,
                          fert, plnt, plcd],
                    );
                    n += 1;
                    if n >= MAX_TOKENS - 30 {
                        break 'outer;
                    }
                }
            }
        }
    }

    // shed (one token, 9 product counts)
    {
        let mut row = vec![T_SHED, 0, 0, 0, 0];
        for p in PRODUCTS.iter() {
            row.push(ji(shed.get(*p).unwrap_or(&NULL)) as i32);
        }
        push_tok(&mut out, &row);
        n += 1;
    }

    // market: one token per product (price, inventory)
    for (pi, p) in PRODUCTS.iter().enumerate() {
        let price = ji(prices.get(*p).unwrap_or(&NULL)) as i32;
        let inv = ji(minv.get(*p).unwrap_or(&NULL)) as i32;
        push_tok(&mut out, &[T_MARKET, 0, 0, pi as i32, 0, price, inv]);
        n += 1;
    }

    // farmer
    let fp = me.get("farmer").and_then(|v| v.as_array());
    let fx = fp.and_then(|a| a.get(0)).map(|v| ji(v) as i32).unwrap_or(0);
    let fy = fp.and_then(|a| a.get(1)).map(|v| ji(v) as i32).unwrap_or(0);
    let fcar = invs.and_then(|a| a.get(0)).map(inv_sum).unwrap_or(0);
    push_tok(&mut out, &[T_FARMER, fx, fy, 0, 0, fcar]);
    n += 1;

    // hands
    if let Some(hands) = me.get("hands").and_then(|v| v.as_array()) {
        for (hi, hp) in hands.iter().take(MAX_HANDS).enumerate() {
            let ha = hp.as_array();
            let hx = ha.and_then(|a| a.get(0)).map(|v| ji(v) as i32).unwrap_or(0);
            let hy = ha.and_then(|a| a.get(1)).map(|v| ji(v) as i32).unwrap_or(0);
            let car = invs.and_then(|a| a.get(hi + 1)).map(inv_sum).unwrap_or(0);
            push_tok(&mut out, &[T_HAND, hx, hy, hi as i32, 0, car]);
            n += 1;
        }
    }

    // game-state
    let money = (ji(me.get("money").unwrap_or(&NULL)) / 16) as i32;
    let oppmoney = (ji(opp.get("money").unwrap_or(&NULL)) / 16) as i32;
    let nhands =
        me.get("hands").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0) as i32;
    let land = me
        .get("unlocked_quadrants")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0) as i32;
    let hires = geti(me, "hires_today");
    let mut shopmulti = 0i32;
    if let Some(shops) = obs
        .get("town")
        .and_then(|t| t.get("unlocked_shops"))
        .and_then(|v| v.as_array())
    {
        for s in shops {
            if let Some(ss) = s.as_str() {
                if let Some(i) = SHOPS.iter().position(|&x| x == ss) {
                    shopmulti |= 1 << i;
                }
            }
        }
    }
    push_tok(
        &mut out,
        &[T_GAME, 0, 0, 0, 0, day, hour, step / 24, money, oppmoney, nhands, land, hires,
          shopmulti],
    );
    n += 1;

    if n > MAX_TOKENS {
        out.truncate(MAX_TOKENS * TOK_W);
    }
    out
}

/// Encode a composite action into a fixed 56-int32 array. Mirrors `encode_action`.
pub fn encode_action(act: &Value) -> [i32; ACT_W] {
    let mut out = [0i32; ACT_W];
    if let Some(fm) = act.get("farmer").and_then(|v| v.as_array()) {
        if !fm.is_empty() {
            let verb = fm[0].as_str().unwrap_or("");
            out[0] = verb_id(&MOVER_VERBS, verb);
            if fm.len() > 1 {
                out[1] = mover_arg(verb, &fm[1]);
            }
        }
    }
    let mut o = 2;
    if let Some(hands) = act.get("hands").and_then(|v| v.as_array()) {
        for h in hands.iter().take(MAX_HANDS) {
            if let Some(ha) = h.as_array() {
                if !ha.is_empty() {
                    let verb = ha[0].as_str().unwrap_or("");
                    out[o] = verb_id(&MOVER_VERBS, verb);
                    if ha.len() > 1 {
                        out[o + 1] = mover_arg(verb, &ha[1]);
                    }
                }
            }
            o += 2;
        }
    }
    let mut o = 2 + MAX_HANDS * 2; // 26
    if let Some(mkt) = act.get("market").and_then(|v| v.as_array()) {
        for m in mkt.iter().take(MAX_MARKET) {
            if let Some(ma) = m.as_array() {
                if !ma.is_empty() {
                    let verb = ma[0].as_str().unwrap_or("");
                    out[o] = verb_id(&MARKET_VERBS, verb);
                    if ma.len() > 1 {
                        out[o + 1] = market_arg(verb, &ma[1]);
                    }
                    if ma.len() > 2 {
                        out[o + 2] = qty(&ma[2]);
                    }
                }
            }
            o += 3;
        }
    }
    out
}

// --- hashing (must match hashlib.sha1 hexdigest prefixes) --------------------
fn sha1_hex_prefix(s: &str, nchars: usize) -> String {
    let mut h = Sha1::new();
    h.update(s.as_bytes());
    let d = h.finalize();
    let mut out = String::with_capacity(40);
    for b in d.iter() {
        out.push_str(&format!("{:02x}", b));
        if out.len() >= nchars {
            break;
        }
    }
    out.truncate(nchars);
    out
}

/// Deterministic 15%-val split; matches `split_of` (1 = val, 0 = train).
pub fn split_of(eid: &str) -> i32 {
    let v = u64::from_str_radix(&sha1_hex_prefix(eid, 8), 16).unwrap_or(0);
    if ((v % 10000) as f64) / 10000.0 < VAL_FRACTION {
        1
    } else {
        0
    }
}

/// day-6 shop signature -> family id; matches `world_family` (empty sig -> 0).
pub fn world_family(sig: &str) -> i32 {
    if sig.is_empty() {
        return 0;
    }
    let v = u64::from_str_radix(&sha1_hex_prefix(sig, 6), 16).unwrap_or(0);
    1 + (v % 4095) as i32
}

/// Which worker (of njobs) owns this episode; matches the Python hash-split.
pub fn worker_of(eid: &str, njobs: u64) -> u64 {
    u64::from_str_radix(&sha1_hex_prefix(eid, 8), 16).unwrap_or(0) % njobs
}
