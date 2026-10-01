//! Farm tile grids: compact decoding, per-state counts, and layout similarity.
use crate::consts::{index_of, ANIMALS, CROPS, CROP_FIRST_YIELD_DAY};
use serde_json::value::RawValue;
use serde_json::Value;

/// Code for "field absent / not a known name" in `crop` / `animal`.
pub const NONE: u8 = 255;
/// Code for a present but unknown name.
pub const UNKNOWN: u8 = 254;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Locked,
    Empty,
    Plant,
    Weed,
    Pasture,
    Coop,
    OtherObj,
    OtherScalar,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    pub kind: Kind,
    /// `tile["crop"]` for dict tiles (CROPS index / UNKNOWN / NONE).
    pub crop: u8,
    /// `tile["animal"]` for dict tiles (ANIMALS index / UNKNOWN / NONE).
    pub animal: u8,
    pub yield_units: f64,
    pub planted_day: i64,
}

fn code(list: &[&str], v: Option<&Value>) -> u8 {
    match v {
        None | Some(Value::Null) => NONE,
        Some(Value::String(s)) => index_of(list, s).map(|i| i as u8).unwrap_or(UNKNOWN),
        Some(_) => UNKNOWN,
    }
}

fn decode(v: &Value) -> Tile {
    let mut t = Tile { kind: Kind::OtherScalar, crop: NONE, animal: NONE, yield_units: 0.0,
                       planted_day: 0 };
    match v {
        Value::Null => t.kind = Kind::Empty,
        Value::String(s) if s == "LOCKED" => t.kind = Kind::Locked,
        Value::Object(o) => {
            t.kind = match o.get("kind").and_then(|k| k.as_str()) {
                Some("PLANT") => Kind::Plant,
                Some("WEED") => Kind::Weed,
                Some("PASTURE") => Kind::Pasture,
                Some("COOP") => Kind::Coop,
                _ => Kind::OtherObj,
            };
            t.crop = code(&CROPS, o.get("crop"));
            t.animal = code(&ANIMALS, o.get("animal"));
            t.yield_units = o.get("yield_units").and_then(|x| x.as_f64()).unwrap_or(0.0);
            t.planted_day = o.get("planted_day").and_then(|x| x.as_f64()).unwrap_or(0.0) as i64;
        }
        _ => {}
    }
    t
}

/// Decode a `tiles` grid, flattened row-major (y then x), exactly as Python's
/// `[x for row in tiles for x in row]`.
pub fn parse_tiles(raw: &RawValue) -> Vec<Tile> {
    let grid: Vec<Vec<Value>> = serde_json::from_str(raw.get()).unwrap_or_default();
    grid.iter().flat_map(|row| row.iter().map(decode)).collect()
}

impl Tile {
    /// Python `(a.get("crop"), a.get("animal")) if isinstance(a, dict) else (None, None)`.
    #[inline]
    pub fn key(&self) -> (u8, u8) {
        match self.kind {
            Kind::Plant | Kind::Weed | Kind::Pasture | Kind::Coop | Kind::OtherObj => {
                (self.crop, self.animal)
            }
            _ => (NONE, NONE),
        }
    }

    /// A crop that could be harvested now: PLANT with yield and past first_yield_day.
    pub fn ripe(&self, day: i64) -> bool {
        self.kind == Kind::Plant
            && self.yield_units > 0.0
            && (self.crop as usize) < CROPS.len()
            && day - self.planted_day >= CROP_FIRST_YIELD_DAY[self.crop as usize]
    }
}

/// Layout similarity (r37 detector, `.local/winplan/forensics_opp.py::similarity`):
/// 0 unless both farms list the same unlocked quadrants; else the share of occupied
/// coordinates (either side has a crop/animal) where (crop, animal) match, 0 below 8.
pub fn similarity(a: &[Tile], b: &[Tile], quadrants_equal: bool) -> f64 {
    if !quadrants_equal {
        return 0.0;
    }
    let (mut m, mut t) = (0u32, 0u32);
    for (x, y) in a.iter().zip(b.iter()) {
        let (sa, sb) = (x.key(), y.key());
        if sa != (NONE, NONE) || sb != (NONE, NONE) {
            t += 1;
            if sa == sb {
                m += 1;
            }
        }
    }
    if t >= 8 {
        m as f64 / t as f64
    } else {
        0.0
    }
}

/// Counts over one farm grid.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TileStats {
    pub by_crop: [u32; 5],
    pub planted: u32,
    pub growing: u32,
    pub ripe: u32,
    pub weed: u32,
    pub empty: u32,
    pub pasture: u32,
    pub coop: u32,
    pub animals: [u32; 3],
}

pub fn tile_stats(tiles: &[Tile], day: i64) -> TileStats {
    let mut s = TileStats::default();
    for t in tiles {
        match t.kind {
            Kind::Plant => {
                s.planted += 1;
                if (t.crop as usize) < 5 {
                    s.by_crop[t.crop as usize] += 1;
                }
                if t.ripe(day) {
                    s.ripe += 1;
                } else {
                    s.growing += 1;
                }
            }
            Kind::Weed => s.weed += 1,
            Kind::Empty => s.empty += 1,
            Kind::Pasture => s.pasture += 1,
            Kind::Coop => s.coop += 1,
            _ => {}
        }
        if (t.animal as usize) < 3 {
            s.animals[t.animal as usize] += 1;
        }
    }
    s
}
