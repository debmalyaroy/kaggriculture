//! `kagg-trackp` -- the standalone compiled TRACK-P seat (v57).
//!
//! G0.2 (harness separation, 2026-09-18). This ALSO fixes the mislabel audited
//! in `harness-audit-2026-09-18`: on the monolith, `kagg trackp` was literally
//! `bandit::play("v57")` -- a bandit family under a trackp name. Here the naming
//! is honest: `kagg-trackp` is the binary for the v57 seat, and it links only
//! the track-neutral core + bandit lane (no cross-lane strategy).
//!
//! Subcommands:
//!   kagg-trackp                          the v57 tape-in-shell seat (default)
//!   kagg-trackp mbandit <cfg> <br> <base>    the config-driven RC harness

use kaggriculture_engine::{bandit, mbandit};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("mbandit") => mbandit::play(&args),
        // default and explicit: the v57 seat
        _ => bandit::play("v57"),
    }
}
