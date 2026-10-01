//! `_View`: the per-step snapshot the layers read, over the compact [`Obs`].
use crate::act::Cmd;
use crate::obs::{qget, FarmObs, Obs, Qty, Tile};

pub const PRODUCTS: [&str; 9] =
    ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];
pub const LAST_ACT_STEP: i64 = 718;

pub fn seed_price(crop: &str) -> Option<i64> {
    Some(match crop {
        "WHEAT" => 10,
        "CARROT" => 20,
        "TOMATO" => 50,
        "STRAWBERRY" => 100,
        "MELON" => 80,
        _ => return None,
    })
}
pub fn animal_cost(a: &str) -> Option<i64> {
    Some(match a {
        "GOOSE" => 300,
        "COW" => 400,
        "SHEEP" => 500,
        _ => return None,
    })
}
pub fn animal_structure(a: &str) -> Option<&'static str> {
    Some(match a {
        "GOOSE" => "COOP",
        "COW" | "SHEEP" => "PASTURE",
        _ => return None,
    })
}
pub const LAND_PRICES: [i64; 3] = [1000, 2000, 4000];
pub fn move_delta(op: &str) -> Option<(i64, i64)> {
    Some(match op {
        "NORTH" => (0, -1),
        "SOUTH" => (0, 1),
        "EAST" => (1, 0),
        "WEST" => (-1, 0),
        _ => return None,
    })
}
pub fn is_move(op: &str) -> bool {
    move_delta(op).is_some()
}

/// Python `_fib`: 1, 1, 2, 3, 5, ...
pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    for _ in 0..n.max(0) {
        let c = a + b;
        a = b;
        b = c;
    }
    a
}

pub fn shed_adjacent(pos: (i64, i64), board: i64) -> bool {
    let half = board / 2;
    (pos.0 == half - 1 || pos.0 == half) && (pos.1 == half - 1 || pos.1 == half)
}

static EMPTY: Qty = Vec::new();

pub struct View {
    pub obs: Obs,
    pub me: usize,
    pub step: i64,
    pub board: i64,
    /// `shed` with negatives clipped to 0, in observation order.
    pub shed: Qty,
    pub positions: Vec<(i64, i64)>,
}

impl View {
    /// None when Python's `Chassis.act` would raise before its layer block (fewer than two
    /// farms) — the entry then falls back to the tape action.
    pub fn new(obs: Obs) -> Result<View, Obs> {
        let me = obs.player.max(0) as usize;
        if obs.farms.len() < 2 || me >= obs.farms.len() || obs.farms[me].farmer.is_none() {
            return Err(obs);
        }
        let shed: Qty = obs.shed.iter().map(|(k, v)| (*k, (*v).max(0))).collect();
        let f = &obs.farms[me];
        let mut positions = vec![f.farmer.unwrap()];
        positions.extend(f.hands.iter().copied());
        let board = if f.rows == 0 { 10 } else { f.rows as i64 };
        let step = obs.step();
        Ok(View { obs, me, step, board, shed, positions })
    }
    pub fn farm(&self) -> &FarmObs {
        &self.obs.farms[self.me]
    }
    pub fn rival(&self) -> &FarmObs {
        &self.obs.farms[1 - self.me.min(1)]
    }
    pub fn money(&self) -> f64 {
        self.farm().money
    }
    pub fn hires_today(&self) -> i64 {
        self.farm().hires_today
    }
    pub fn quadrants(&self) -> i64 {
        self.farm().quadrants.len() as i64
    }
    pub fn shed(&self, item: &str) -> i64 {
        qget(&self.shed, item)
    }
    pub fn shed_total(&self) -> i64 {
        self.shed.iter().map(|(_, n)| n).sum()
    }
    pub fn price(&self, item: &str) -> i64 {
        qget(&self.obs.prices, item)
    }
    pub fn seeds(&self, crop: &str) -> i64 {
        qget(&self.obs.seeds, crop)
    }
    pub fn inv(&self, i: usize) -> &Qty {
        self.obs.invs.get(i).unwrap_or(&EMPTY)
    }
    pub fn invs(&self) -> &[Qty] {
        &self.obs.invs
    }
    pub fn in_hands(&self, item: &str) -> i64 {
        self.obs.invs.iter().map(|m| qget(m, item).max(0)).sum()
    }
    /// `_tile_at` ("LOCKED" off the board).
    pub fn tile(&self, pos: (i64, i64)) -> &Tile {
        self.farm().tile(pos.0, pos.1)
    }
    pub fn shops(&self) -> &[&'static str] {
        &self.obs.shops
    }
}

/// `_is_noop`: true when the engine will certainly ignore `act` (mirrors `_apply_unit_action`).
pub fn is_noop(act: &Cmd, tile: &Tile, inv: &Qty, v: &View, pos: (i64, i64)) -> bool {
    if act.is_empty() {
        return true;
    }
    let op = act.op();
    let board = v.board;
    if let Some((dx, dy)) = move_delta(op) {
        let (x, y) = (pos.0 + dx, pos.1 + dy);
        return !((0..board).contains(&x) && (0..board).contains(&y));
    }
    if op == "PASS" {
        return true;
    }
    let adjacent = shed_adjacent(pos, board);
    match op {
        "DROP" => return !adjacent || inv.is_empty(),
        "PICKUP" => return !adjacent,
        "PLACE" => {
            let item = act.s(1);
            if let Some(sk) = animal_structure(item) {
                if tile.is_dict() && tile.kind == sk && !tile.has_animal() {
                    return qget(inv, item) <= 0;
                }
            }
            return !adjacent || qget(inv, item) <= 0;
        }
        _ => {}
    }
    if tile.is_locked() {
        return true;
    }
    let animal = tile.is_dict() && tile.has_animal();
    match op {
        "PLANT" => !tile.is_none() || v.seeds(act.s(1)) <= 0,
        "WATER" => tile.kind != "PLANT" || tile.watered_today,
        "HARVEST" => !tile.is_dict() || tile.yield_units <= 0,
        "FERTILIZE" => tile.kind != "PLANT" || qget(inv, "FERTILIZER") <= 0,
        "DIG" => tile.is_none() || animal,
        "BUILD_COOP" | "BUILD_PASTURE" => !tile.is_none(),
        "FEED" => !animal || tile.fed_today || qget(inv, "WHEAT") <= 0,
        "COLLECT_FERTILIZER" => !animal || !tile.fertilizer_available,
        "CARE" => !animal || tile.cared_today,
        _ => true,
    }
}
