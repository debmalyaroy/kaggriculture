//! Byte-exact Rust mirror of `trackp_corpus.encode_tokens` (TOKEN_LAYOUT_VERSION=2).
//! Emits a seat's observation tokens straight from the engine `State`, so RL
//! rollouts skip the Python obs->token conversion (the last per-step Python hot
//! loop). PARITY-GATED: `kagg toktest` prints tokens that must equal the Python
//! encoder's for the same state, or the RL trains on garbage.

use crate::state::{Cell, State};

pub const TOK_W: usize = 14;
pub const MAX_TOKENS: usize = 160;
const MAX_HANDS: usize = 12;

// vocab order MUST match src/kaggriculture/data/trackp_corpus.py exactly
const CROPS: [&str; 5] = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"];
const ANIMALS: [&str; 3] = ["COW", "GOOSE", "SHEEP"];
const PRODUCTS: [&str; 9] = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
                             "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"];
const SHOPS: [&str; 8] = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
                          "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"];

const T_PLANT: i32 = 1;
const T_PASTURE: i32 = 2;
const T_COOP: i32 = 3;
const T_WEED: i32 = 4;
const T_SHED: i32 = 6;
const T_MARKET: i32 = 7;
const T_HAND: i32 = 8;
const T_FARMER: i32 = 9;
const T_GAME: i32 = 10;

fn crop_id(c: &str) -> i32 {
    CROPS.iter().position(|&x| x == c).map(|i| i as i32 + 1).unwrap_or(0)
}
fn anim_id(a: &str) -> i32 {
    ANIMALS.iter().position(|&x| x == a).map(|i| i as i32 + 1).unwrap_or(0)
}

#[inline]
fn push_tok(toks: &mut Vec<[i32; TOK_W]>, tt: i32, x: i32, y: i32, ca: i32, cb: i32,
            vals: &[i32]) {
    let mut row = [0i32; TOK_W];
    row[0] = tt; row[1] = x; row[2] = y; row[3] = ca; row[4] = cb;
    for (i, &v) in vals.iter().enumerate() {
        if 5 + i < TOK_W {
            row[5 + i] = v;
        }
    }
    toks.push(row);
}

/// Observation tokens for `seat` -- byte-exact vs Python encode_tokens.
pub fn tokens_for(s: &State, seat: usize) -> Vec<[i32; TOK_W]> {
    let me = &s.farms[seat];
    let opp = &s.farms[1 - seat];
    let pv = &s.private[seat];
    let day = s.step / 24;
    let hour = s.step % 24;
    let mut toks: Vec<[i32; TOK_W]> = Vec::new();

    // tiles (ours): tiles[y][x]
    'outer: for (y, row) in me.tiles.iter().enumerate() {
        for (x, c) in row.iter().enumerate() {
            let (tt, crop, anim, vals): (i32, i32, i32, [i32; 9]) = match c {
                Cell::Plant { crop, planted_day, watered_today,
                              consecutive_unwatered, yield_units, .. } => (
                    T_PLANT, crop_id(crop), 0,
                    [*yield_units as i32, if *watered_today { 1 } else { 0 }, 0, 0,
                     *consecutive_unwatered as i32, 0, 0,
                     (day - planted_day).max(0) as i32, 0],
                ),
                Cell::Structure { kind, animal } => {
                    let tt = if kind == "PASTURE" { T_PASTURE }
                             else if kind == "COOP" { T_COOP }
                             else { continue };
                    match animal {
                        Some(a) => (tt, 0, anim_id(&a.animal),
                            [a.yield_units as i32, 0,
                             if a.cared_today { 1 } else { 0 },
                             if a.fed_today { 1 } else { 0 },
                             0, a.consecutive_unfed as i32,
                             if a.fertilizer_available { 1 } else { 0 },
                             0, (day - a.placed_day).max(0) as i32]),
                        None => (tt, 0, 0, [0; 9]),
                    }
                }
                Cell::Weed => (T_WEED, 0, 0, [0; 9]),
                _ => continue, // Empty | Locked -> not a tile token
            };
            push_tok(&mut toks, tt, x as i32, y as i32, crop, anim, &vals);
            if toks.len() >= MAX_TOKENS - 30 {
                break 'outer;
            }
        }
    }
    // shed: one token, first 9 products
    let shed: Vec<i32> = PRODUCTS.iter().map(|p| pv.shed.get(p) as i32).collect();
    push_tok(&mut toks, T_SHED, 0, 0, 0, 0, &shed);
    // market: one token per product (price, inventory)
    for (pi, p) in PRODUCTS.iter().enumerate() {
        push_tok(&mut toks, T_MARKET, 0, 0, pi as i32, 0,
                 &[s.market.prices.get(p) as i32, s.market.inventory.get(p) as i32]);
    }
    // farmer + hands (carry = sum of that unit's inventory)
    let fcar = pv.inventories.first().map(|m| m.sum()).unwrap_or(0) as i32;
    push_tok(&mut toks, T_FARMER, me.farmer.0 as i32, me.farmer.1 as i32, 0, 0, &[fcar]);
    for (hi, hp) in me.hands.iter().take(MAX_HANDS).enumerate() {
        let car = pv.inventories.get(hi + 1).map(|m| m.sum()).unwrap_or(0) as i32;
        push_tok(&mut toks, T_HAND, hp.0 as i32, hp.1 as i32, hi as i32, 0, &[car]);
    }
    // game-state
    let mut shopmulti = 0i32;
    for sh in &s.town.unlocked_shops {
        if let Some(i) = SHOPS.iter().position(|&x| x == sh) {
            shopmulti |= 1 << i;
        }
    }
    push_tok(&mut toks, T_GAME, 0, 0, 0, 0, &[
        day as i32, hour as i32, (s.step / 24) as i32,
        (me.money as i64 / 16) as i32, (opp.money as i64 / 16) as i32,
        me.hands.len() as i32, me.unlocked_quadrants.len() as i32,
        me.hires_today as i32, shopmulti]);

    if toks.len() > MAX_TOKENS {
        toks.truncate(MAX_TOKENS);
    }
    toks
}

/// Flatten to `n_tokens t0_0 t0_1 ... t0_13 t1_0 ...` space-separated ints, prefixed
/// by the token count -- the compact wire form vecserve sends to Python.
pub fn tokens_flat(s: &State, seat: usize) -> String {
    let toks = tokens_for(s, seat);
    let mut out = String::with_capacity(toks.len() * 40 + 8);
    out.push_str(&toks.len().to_string());
    for row in &toks {
        for v in row {
            out.push(' ');
            out.push_str(&v.to_string());
        }
    }
    out
}
