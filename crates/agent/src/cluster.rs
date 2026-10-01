//! Opponent cluster key for the chassis router (v63.12 chassis): the rival's farm economy at the route switch.
//!
//! key = FNV hash of the rival's bucketed contents -- per PLANT crop / PASTURE animal the tile count bucket
//! (0 | 1-2 | 3-5 | 6-10 | 11-20 | 21+), weeds counted as empty -- plus its hand count and unlocked quadrants.
//! Positions are ignored (weed spawns move crops around). The same function labels tapes offline
//! (chassis-screen, from Obs::from_state) and routes at runtime (router, from the observation), so the two agree.
use crate::obs::FarmObs;

fn bucket(n: u32) -> u32 {
    match n {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        6..=10 => 3,
        11..=20 => 4,
        _ => 5,
    }
}

pub fn econ_key(f: &FarmObs) -> u64 {
    let mut cnt: std::collections::BTreeMap<String, u32> = Default::default();
    for t in &f.tiles {
        if !t.is_dict() || t.crop == "WEED" || t.kind == "WEED" {
            continue;
        }
        let sub = if t.has_animal() { t.animal } else { t.crop };
        *cnt.entry(format!("{}:{sub}", t.kind)).or_default() += 1;
    }
    let s: String = cnt.iter().map(|(k, n)| format!("{k}={};", bucket(*n))).collect();
    let s = format!("{s}|h{}|q{}", f.hands.len(), f.quadrants.join(","));
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Finer keys for the repair loop: level 2 = exact tile counts (no buckets), level 3 = the exact layout (every tile in
/// place, weeds as empty). Same hands / quadrants tail as `econ_key`.
pub fn exact_key(f: &FarmObs) -> u64 {
    let mut cnt: std::collections::BTreeMap<String, u32> = Default::default();
    for t in &f.tiles {
        if !t.is_dict() || t.crop == "WEED" || t.kind == "WEED" {
            continue;
        }
        let sub = if t.has_animal() { t.animal } else { t.crop };
        *cnt.entry(format!("{}:{sub}", t.kind)).or_default() += 1;
    }
    let s: String = cnt.iter().map(|(k, n)| format!("{k}={n};")).collect();
    let s = format!("{s}|h{}|q{}", f.hands.len(), f.quadrants.join(","));
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

pub fn layout_key(f: &FarmObs) -> u64 {
    let mut s = String::new();
    for t in &f.tiles {
        if !t.is_dict() || t.crop == "WEED" || t.kind == "WEED" {
            s.push_str(if t.is_locked() { "L;" } else { "_;" });
        } else {
            let sub = if t.has_animal() { t.animal } else { t.crop };
            s.push_str(&format!("{}:{sub};", t.kind));
        }
    }
    let s = format!("{s}|h{}|q{}", f.hands.len(), f.quadrants.join(","));
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// The three keys, most specific first: ["l<layout>", "x<exact>", "<econ>"] (the router's lookup order).
pub fn keys(f: &FarmObs) -> [String; 3] {
    [format!("l{:x}", layout_key(f)), format!("x{:x}", exact_key(f)), format!("{:x}", econ_key(f))]
}
