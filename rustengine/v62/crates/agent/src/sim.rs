//! Bridge from the agent's compact observation to the bit-exact engine's state types, for
//! layers that simulate unit actions (the payload's `_apply_unit_action` == the engine's).
use crate::act::Cmd;
use crate::obs::{FarmObs, Qty, Tile};
use kagg_engine::engine::{apply_unit_action, UnitAction};
use kagg_engine::state::{AnimalTile, Cell, Farm, OMap, Private};

pub fn omap(q: &Qty) -> OMap {
    OMap(q.iter().map(|(k, v)| (kagg_engine::state::name(k), *v)).collect())
}

pub fn cell(t: &Tile) -> Cell {
    match t.kind {
        "" => Cell::Empty,
        "LOCKED" => Cell::Locked,
        "WEED" => Cell::Weed,
        "PLANT" => Cell::Plant {
            crop: kagg_engine::state::name(t.crop),
            planted_day: t.planted_day,
            watered_today: t.watered_today,
            consecutive_unwatered: t.consecutive_unwatered,
            yield_units: t.yield_units,
            max_lifespan_step: t.max_lifespan_step,
            fertilized_until_day: t.fertilized_until_day,
        },
        k => Cell::Structure {
            kind: kagg_engine::state::name(k),
            animal: (!t.animal.is_empty()).then(|| AnimalTile {
                animal: kagg_engine::state::name(t.animal),
                placed_day: t.placed_day,
                yield_units: t.yield_units,
                consecutive_unfed: t.consecutive_unfed,
                fed_today: t.fed_today,
                cared_today: t.cared_today,
                fertilizer_available: t.fertilizer_available,
                pending_care_bonus: t.pending_care_bonus,
            }),
        },
    }
}

pub fn farm(f: &FarmObs) -> Farm {
    Farm {
        money: f.money,
        farmer: f.farmer.unwrap_or((0, 0)),
        hands: f.hands.clone(),
        hires_today: f.hires_today,
        unlocked_quadrants: f.quadrants.iter().map(|s| s.to_string()).collect(),
        tiles: (0..f.rows).map(|y| (0..f.cols).map(|x| cell(&f.tiles[y * f.cols + x])).collect()).collect(),
    }
}

pub fn private(shed: &Qty, seeds: &Qty, invs: &[Qty]) -> Private {
    Private { shed: omap(shed), seeds: omap(seeds), inventories: invs.iter().map(omap).collect() }
}

pub fn unit(c: &Cmd) -> UnitAction {
    UnitAction {
        op: c.op().to_string(),
        item: c.s(1).to_string(),
        n: if c.len() >= 3 { c.n(2) } else { 1 },
        has_n: c.len() >= 3,
    }
}

/// `_apply_unit_action(farm, private, idx, c, 10, day, 24, 100)`.
pub fn apply(f: &mut Farm, p: &mut Private, idx: usize, c: &Cmd, day: i64) {
    if c.is_empty() {
        return;
    }
    apply_unit_action(f, p, idx, &unit(c), day);
}
