//! B1.4: multi-day rolling economic model. Consumes the engine-derived
//! absorption table (`data/absorption.json`, from tools/absorption.py) and a
//! realized market scenario (`data/scenario_seed42.json`, from tools/season_tape.py),
//! rolls a per-day shed + cash ledger, optimises each day's sells with the
//! staircase-PWL LP (solve_day), and reports predicted Cash720 vs the actual
//! Cash720 the tape banked on the real engine. Also runs a small HiGHS MIP
//! (land-tier selection) to exercise integrality — the piece microlp can't do.

use crate::market::{Params, CARROT, MELON, STRAWBERRY, TOMATO, WHEAT};
use crate::model::{solve_day, Item};
use good_lp::{constraint, variable, variables, Solution, SolverModel};
use std::fs;
use std::path::Path;

fn data_path(name: &str) -> String {
    // resolve relative to the crate dir regardless of CWD
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join(name);
    here.to_string_lossy().into_owned()
}

fn crop_for(name: &str) -> Params {
    match name {
        "CARROT" => CARROT,
        "MELON" => MELON,
        "STRAWBERRY" => STRAWBERRY,
        "TOMATO" => TOMATO,
        _ => WHEAT,
    }
}

/// Print the engine-derived A_c profiles the model builds deliveries from.
pub fn show_absorption() {
    let raw = match fs::read_to_string(data_path("absorption.json")) {
        Ok(s) => s,
        Err(_) => {
            println!("(absorption.json not found — run tools/absorption.py first)");
            return;
        }
    };
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    println!("--- B1.2-full: engine-derived T-absorption table A_c (data/absorption.json) ---");
    for crop in ["WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY"] {
        let c = &v[crop];
        let prof: Vec<String> = c["profile"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| format!("d{}:{}", p["deliver_day"], p["units"]))
            .collect();
        println!(
            "  {:11} total {:>2}  occ {:>2}d  profile [{}]",
            crop,
            c["total_units"],
            c["occupancy_days"],
            prof.join(", ")
        );
    }
    println!();
}

/// Roll the realized-scenario multi-day model and compare to actual Cash720.
pub fn run_season() {
    show_absorption();

    let raw = match fs::read_to_string(data_path("scenario_seed42.json")) {
        Ok(s) => s,
        Err(_) => {
            println!("(scenario_seed42.json not found — run tools/season_tape.py first)");
            return;
        }
    };
    let scen: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let crop_name = scen["crop"].as_str().unwrap();
    let p = crop_for(crop_name);
    let m0 = scen["m0"].as_f64().unwrap();
    let seed_cost = scen["seed_cost_total"].as_f64().unwrap();
    let actual_cash = scen["actual_cash"].as_f64().unwrap();
    let sells = scen["sells"].as_array().unwrap();

    println!("--- B1.4: multi-day rolling model over realized scenario (seed {}) ---",
        scen["seed"]);
    println!("crop {crop_name}, {} daily sell batches; per-day PWL LP over a rolling shed+cash ledger",
        sells.len());

    // Rolling ledger: each day, that day's harvested units are the shed supply;
    // the LP chooses sells (carrots never floor => sells all, but the LP proves
    // it and would withhold on a fragile product). Cash accrues; must stay >= 0.
    let mut cash = m0 - seed_cost; // seeds already paid in the realized run
    let mut predicted_rev = 0.0f64;
    let mut sold_units = 0i64;
    for s in sells {
        let v0 = s["v0"].as_i64().unwrap();
        let qty = s["qty"].as_i64().unwrap();
        // one-item single-day allocation solved by the actual LP backend
        let items = vec![Item { p, v0, supply: qty }];
        let res = solve_day(&items, /*shed_cap*/ 100).expect("day solve");
        let rev = res.total_revenue as f64;
        predicted_rev += rev;
        sold_units += res.shed_used;
        cash += rev;
        assert!(cash >= 0.0, "cash ledger went negative");
    }
    let predicted_cash = m0 - seed_cost + predicted_rev;

    println!("  units sold (model)  : {sold_units}");
    println!("  predicted revenue   : ${predicted_rev:.0}");
    println!("  predicted Cash720   : ${predicted_cash:.0}");
    println!("  ACTUAL Cash720      : ${actual_cash:.0}   (tape on the real engine)");
    let gap = actual_cash - predicted_cash;
    println!("  GAP (actual-pred)   : ${gap:.0}");
    if gap.abs() < 0.5 {
        println!("  => Rust multi-day model reproduces the engine-validated Cash720 to the dollar.");
    }
    println!();

    mip_land_demo(cash);
}

/// Small HiGHS MIP: pick which land tiers to unlock (order-forced NE->SW->SE,
/// costs 1000/2000/4000) to maximise (tiles*value_per_tile - cost) under a cash
/// budget. Integer/binary vars — enforced by HiGHS (microlp relaxes them, noted).
pub fn mip_land_demo(cash_budget: f64) {
    // value of one extra 25-tile quadrant over the remaining season, net of
    // labor to work it (illustrative $/quadrant); the point is the integrality.
    let value_per_quadrant = 1500.0;
    let costs = [1000.0, 2000.0, 4000.0]; // NE, SW, SE

    let mut vars = variables!();
    let y: Vec<_> = (0..3).map(|_| vars.add(variable().binary())).collect();

    let objective = value_per_quadrant * (y[0] + y[1] + y[2])
        - (costs[0] * y[0] + costs[1] * y[1] + costs[2] * y[2]);

    #[cfg(feature = "highs")]
    let mut model = vars.maximise(objective).using(good_lp::highs);
    #[cfg(all(feature = "microlp", not(feature = "highs")))]
    let mut model = vars.maximise(objective).using(good_lp::microlp);

    // order-forced: SW only after NE, SE only after SW
    model = model.with(constraint!(y[1] <= y[0]));
    model = model.with(constraint!(y[2] <= y[1]));
    // cash budget
    model = model.with(constraint!(costs[0] * y[0] + costs[1] * y[1] + costs[2] * y[2] <= cash_budget));

    let sol = model.solve().expect("land MIP");
    let pick: Vec<&str> = ["NE", "SW", "SE"]
        .iter()
        .enumerate()
        .filter(|(i, _)| sol.value(y[*i]) > 0.5)
        .map(|(_, n)| *n)
        .collect();
    println!("--- B1.3/B5 (integrality): HiGHS MIP land-tier decision ---");
    println!("  cash budget ${cash_budget:.0}, value/quadrant ${value_per_quadrant:.0}, costs {costs:?}");
    println!("  unlock: {:?}  (order-forced NE->SW->SE, binary vars)", pick);
    #[cfg(not(feature = "highs"))]
    println!("  (note: default microlp is LP-only; build --features highs for a true MIP)");
}
