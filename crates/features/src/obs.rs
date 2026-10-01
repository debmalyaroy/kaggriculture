//! Borrowing serde structs for the Kaggle replay / observation JSON.
//!
//! Only the fields the features need are materialised; everything else is skipped by
//! serde without allocation, and the bulky `tiles` grids stay as borrowed raw JSON until a
//! feature actually needs them (see [`crate::tiles::parse_tiles`]).
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;

/// Number that may be null / missing -> 0.0.
fn nz<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(0.0))
}

/// Per-product numbers (prices, inventory, shed). Extra keys (animals in the shed) ignored.
#[derive(Deserialize, Default, Clone, Copy, Debug, PartialEq)]
pub struct Items {
    #[serde(rename = "CARROT", default, deserialize_with = "nz")]
    pub carrot: f64,
    #[serde(rename = "EGG", default, deserialize_with = "nz")]
    pub egg: f64,
    #[serde(rename = "FERTILIZER", default, deserialize_with = "nz")]
    pub fertilizer: f64,
    #[serde(rename = "MELON", default, deserialize_with = "nz")]
    pub melon: f64,
    #[serde(rename = "MILK", default, deserialize_with = "nz")]
    pub milk: f64,
    #[serde(rename = "STRAWBERRY", default, deserialize_with = "nz")]
    pub strawberry: f64,
    #[serde(rename = "TOMATO", default, deserialize_with = "nz")]
    pub tomato: f64,
    #[serde(rename = "WHEAT", default, deserialize_with = "nz")]
    pub wheat: f64,
    #[serde(rename = "WOOL", default, deserialize_with = "nz")]
    pub wool: f64,
}

impl Items {
    /// In `consts::PRODUCTS` order.
    pub fn arr(&self) -> [f64; 9] {
        [
            self.carrot, self.egg, self.fertilizer, self.melon, self.milk, self.strawberry,
            self.tomato, self.wheat, self.wool,
        ]
    }
}

#[derive(Deserialize, Default, Debug)]
pub struct Market {
    #[serde(default)]
    pub prices: Items,
    #[serde(default)]
    pub inventory: Items,
}

#[derive(Deserialize, Default, Debug)]
pub struct Town {
    #[serde(default)]
    pub unlocked_shops: Vec<String>,
}

#[derive(Deserialize, Default, Debug)]
pub struct Private {
    #[serde(default)]
    pub shed: Option<Items>,
}

#[derive(Deserialize, Default, Debug)]
pub struct Farm<'a> {
    #[serde(default, deserialize_with = "nz")]
    pub money: f64,
    #[serde(default, deserialize_with = "nz")]
    pub hires_today: f64,
    #[serde(default)]
    pub farmer: Option<Vec<f64>>,
    #[serde(default)]
    pub hands: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub unlocked_quadrants: Option<Vec<String>>,
    #[serde(default, borrow)]
    pub tiles: Option<&'a RawValue>,
}

/// One seat's observation. Seat 1's replay observation may omit shared keys.
#[derive(Deserialize, Default, Debug)]
pub struct Obs<'a> {
    #[serde(default, borrow)]
    pub farms: Option<Vec<Farm<'a>>>,
    #[serde(default)]
    pub market: Option<Market>,
    #[serde(default)]
    pub town: Option<Town>,
    #[serde(default)]
    pub private: Option<Private>,
}

/// `steps[t][seat]`.
#[derive(Deserialize, Default, Debug)]
pub struct Cell<'a> {
    #[serde(default, borrow)]
    pub observation: Option<Obs<'a>>,
    #[serde(default, borrow)]
    pub action: Option<&'a RawValue>,
}

#[derive(Deserialize, Default, Debug)]
pub struct Info {
    #[serde(rename = "EpisodeId", default)]
    pub episode_id: Option<serde_json::Value>,
    #[serde(rename = "TeamNames", default)]
    pub team_names: Option<Vec<Option<String>>>,
    #[serde(default)]
    pub seed: Option<serde_json::Value>,
}

/// The replay as served by the episode CDN (also `replay_json` in the GM shards).
#[derive(Deserialize, Default, Debug)]
pub struct Replay<'a> {
    #[serde(default)]
    pub info: Option<Info>,
    #[serde(default)]
    pub module_version: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub rewards: Option<Vec<Option<f64>>>,
    #[serde(default, borrow)]
    pub steps: Vec<Vec<Option<Cell<'a>>>>,
}

/// One step of a slim record (see crates/corpus/src/slim.rs): `a` actions, `m` market,
/// `tw` shops only when changed, `f` farms without tiles, `p` privates, `tl` tiles at hour 1.
#[derive(Deserialize, Default, Debug)]
pub struct SlimStep<'a> {
    #[serde(default, borrow)]
    pub a: Vec<Option<&'a RawValue>>,
    #[serde(default)]
    pub m: Option<Market>,
    #[serde(default)]
    pub tw: Option<Vec<String>>,
    #[serde(default, borrow)]
    pub f: Vec<Farm<'a>>,
    #[serde(default)]
    pub p: Vec<Option<Private>>,
    #[serde(default, borrow)]
    pub tl: Option<Vec<Option<&'a RawValue>>>,
}

/// A slim record (`slim` column of the slim corpus).
#[derive(Deserialize, Default, Debug)]
pub struct SlimDoc<'a> {
    #[serde(default)]
    pub info: Option<Info>,
    #[serde(default)]
    pub module_version: Option<String>,
    #[serde(default)]
    pub rewards: Option<Vec<Option<f64>>>,
    #[serde(default, borrow)]
    pub steps: Vec<SlimStep<'a>>,
}

impl<'a> SlimDoc<'a> {
    pub fn parse(s: &'a str) -> serde_json::Result<SlimDoc<'a>> {
        serde_json::from_str(s)
    }
}

impl<'a> Replay<'a> {
    pub fn parse(s: &'a str) -> serde_json::Result<Replay<'a>> {
        serde_json::from_str(s)
    }

    pub fn cell(&self, t: usize, seat: usize) -> Option<&Cell<'a>> {
        self.steps.get(t).and_then(|c| c.get(seat)).and_then(|c| c.as_ref())
    }

    pub fn obs(&self, t: usize, seat: usize) -> Option<&Obs<'a>> {
        self.cell(t, seat).and_then(|c| c.observation.as_ref())
    }

    /// Shared state (farms/market/town) at step t: seat 0's observation, or seat 1's when
    /// seat 0's lacks the shared keys.
    pub fn shared(&self, t: usize) -> Option<&Obs<'a>> {
        let ok = |o: &&Obs<'a>| o.farms.as_ref().map(|f| f.len() >= 2).unwrap_or(false);
        self.obs(t, 0).filter(ok).or_else(|| self.obs(t, 1).filter(ok))
    }
}
