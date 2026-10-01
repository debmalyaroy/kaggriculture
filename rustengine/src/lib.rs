//! Library face of the bit-exact Kaggriculture engine plus the Track-P search.
//!
//! `main.rs` keeps its own `mod` declarations (Phase A owns that file); this
//! lib target exists so the offline search harness in `src/bin/` can link
//! against the same modules without touching it. The engine modules
//! (`engine`, `market`, `mt19937`, `rules`, `state`) are BIT-EXACT against the
//! official interpreter and are also the bandit lane's pre-ranker -- they are
//! re-exported here unchanged and must never be edited to suit the search.

pub mod json;
pub mod engine;
pub mod market;
pub mod mt19937;
pub mod plan;
pub mod rules;
pub mod search;
pub mod state;
pub mod obstoken;
pub mod value;

// G0.2 (harness separation, 2026-09-18): the track-neutral CORE shell and the
// two SEAT lanes are exposed here so the separate `kagg-bandit` / `kagg-trackp`
// binaries (src/bin/) can link them without pulling in the engine search or the
// trackp economy policy. `core`/`bandit`/`mbandit` depend only on `json`+`core`
// -- neither seat binary carries the other lane's strategy.
pub mod core;
pub mod bandit;
pub mod mbandit;

// G0.2 (3-way split completion, 2026-09-18): the engine SERVICE layer (mass
// `batch` rollouts + the stdio `serve` env) re-exported so a lean `kagg-engine`
// binary (src/bin/kagg_engine.rs) can carry the engine/pre-ranker path with
// NEITHER seat's strategy. `service` depends only on `engine`+`state` (both
// bit-exact, above); main.rs keeps its own private `mod service` unchanged.
pub mod loadstate;
pub mod service;
