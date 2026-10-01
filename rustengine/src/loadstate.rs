//! STATE INJECTION: parse the `json_state` format back into a `State`.
//!
//! The big-trackp runtime searcher (2026-09-05) needs rollouts FROM THE
//! CURRENT MID-GAME STATE, not from a seed: the ladder does not expose the
//! seed to agents (configuration.seed is None) and seeds are uniform 31-bit,
//! unrecoverable in budget. This module is the exact inverse of
//! `service::json_state` / `json_farm` / `json_tile` / `json_private`, so
//!
//!     state_from_json(parse(&json_state(&st, false))).digest() == st.digest()
//!
//! is the round-trip law tests enforce. The one field JSON does not carry is
//! the RNG seed: rollouts accept it via an optional "seed" key (default 0)
//! and therefore see a RESAMPLED random future (weed spawns, shop unlocks).
//! Over sell-planning horizons the market math is deterministic in the
//! actions, so this is the accepted approximation -- the gates measure it.

use crate::json::Json;
use crate::state::{AnimalTile, Cell, Farm, Market, OMap, Private, State, Town};

fn omap_from(j: &Json) -> OMap {
    let mut v: Vec<(String, i64)> = Vec::new();
    if j.is_obj() {
        for (k, val) in j.obj() {
            v.push((k.clone(), val.i64()));
        }
    }
    OMap(v)
}

fn pair_from(j: &Json) -> (i64, i64) {
    (j.idx(0).i64(), j.idx(1).i64())
}

fn cell_from(j: &Json) -> Result<Cell, String> {
    if j.is_null() {
        return Ok(Cell::Empty);
    }
    if !j.is_obj() {
        // the only bare-string tile is "LOCKED"
        return match j.str() {
            "LOCKED" => Ok(Cell::Locked),
            other => Err(format!("unknown tile literal {other:?}")),
        };
    }
    let kind = j.get("kind").str().to_string();
    match kind.as_str() {
        "WEED" => Ok(Cell::Weed),
        "PLANT" => Ok(Cell::Plant {
            crop: j.get("crop").str().to_string(),
            planted_day: j.get("planted_day").i64(),
            watered_today: j.get("watered_today").bool(),
            consecutive_unwatered: j.get("consecutive_unwatered").i64(),
            yield_units: j.get("yield_units").i64(),
            max_lifespan_step: j.get("max_lifespan_step").i64(),
            fertilized_until_day: j.get("fertilized_until_day").i64(),
        }),
        _ => {
            let animal = if j.get("animal").is_null() {
                None
            } else {
                Some(AnimalTile {
                    animal: j.get("animal").str().to_string(),
                    placed_day: j.get("placed_day").i64(),
                    yield_units: j.get("yield_units").i64(),
                    consecutive_unfed: j.get("consecutive_unfed").i64(),
                    fed_today: j.get("fed_today").bool(),
                    cared_today: j.get("cared_today").bool(),
                    fertilizer_available: j.get("fertilizer_available").bool(),
                    pending_care_bonus: j.get("pending_care_bonus").i64(),
                })
            };
            Ok(Cell::Structure { kind, animal })
        }
    }
}

fn farm_from(j: &Json) -> Result<Farm, String> {
    let mut tiles: Vec<Vec<Cell>> = Vec::new();
    for row in j.get("tiles").arr() {
        let mut r: Vec<Cell> = Vec::new();
        for c in row.arr() {
            r.push(cell_from(c)?);
        }
        tiles.push(r);
    }
    Ok(Farm {
        money: j.get("money").f64(),
        farmer: pair_from(j.get("farmer")),
        hands: j.get("hands").arr().iter().map(pair_from).collect(),
        hires_today: j.get("hires_today").i64(),
        unlocked_quadrants: j
            .get("unlocked_quadrants")
            .arr()
            .iter()
            .map(|q| q.str().to_string())
            .collect(),
        tiles,
    })
}

fn private_from(j: &Json) -> Private {
    Private {
        shed: omap_from(j.get("shed")),
        seeds: omap_from(j.get("seeds")),
        inventories: j.get("inventories").arr().iter().map(omap_from).collect(),
    }
}

pub fn state_from_json(j: &Json) -> Result<State, String> {
    let farms_j = j.get("farms");
    let priv_j = j.get("private");
    let seed = if j.get("seed").is_null() { 0 } else { j.get("seed").i64() };
    Ok(State {
        step: j.get("step").i64(),
        seed,
        farms: [farm_from(farms_j.idx(0))?, farm_from(farms_j.idx(1))?],
        private: [private_from(priv_j.idx(0)), private_from(priv_j.idx(1))],
        market: Market {
            inventory: omap_from(j.get("market").get("inventory")),
            prices: omap_from(j.get("market").get("prices")),
        },
        town: Town {
            unlocked_shops: j
                .get("town")
                .get("unlocked_shops")
                .arr()
                .iter()
                .map(|s| s.str().to_string())
                .collect(),
        },
    })
}
