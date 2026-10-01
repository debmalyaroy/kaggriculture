//! Kaggriculture macro (per-day) features and behaviour.
//!
//! Pure and deterministic: no file I/O, no clocks, no randomness. `corpus-extract` uses it
//! to build the training corpus and the Rust agent reuses it at play time, so train and
//! serve compute features with the same code.
//!
//! * [`obs`]: borrowing serde structs for replay / observation JSON.
//! * [`episode::Episode`]: compact per-step record + `daily_state` / `daily_behaviour` rows.
//! * [`tiles`]: tile decoding, per-state counts, layout similarity.
//! * [`action`]: engine-faithful order parsing and Python-canonical JSON (stream hashes).
pub mod action;
pub mod consts;
pub mod dayobs_adapter;
pub mod episode;
pub mod obs;
pub mod row;
pub mod tiles;

pub use episode::{Episode, StepRec};
pub use obs::Replay;
pub use row::{Col, Row, Val};

/// Bump when any computed value changes meaning.
pub const FEATURES_VERSION: u32 = 1;
