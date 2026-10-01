//! `kagg-bandit` -- the standalone compiled BANDIT seat.
//!
//! G0.2 (harness separation, 2026-09-18): the bandit seat is now its own binary
//! rather than a subcommand of the monolithic `kagg`. It links ONLY the
//! track-neutral core + the bandit lane (`json`, `core`, `bandit`, `mbandit`) --
//! it carries none of the trackp economy policy or the engine search. The
//! monolithic `kagg` retains a `bandit`/`mbandit` subcommand for backward
//! compatibility, but a fresh ship should build and package THIS binary.
//!
//! Subcommands:
//!   kagg-bandit                         the fixed v56y tape-in-shell (default)
//!   kagg-bandit v56y | v57              a named embedded family
//!   kagg-bandit mbandit <cfg> <br> <base>   the config-driven RC harness

use kaggriculture_engine::{bandit, mbandit};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        // config-driven multi-checkpoint RC harness (what ships today)
        Some("mbandit") => mbandit::play(&args),
        // fixed embedded families
        Some("v57") | Some("trackp") => bandit::play("v57"),
        None | Some("v56y") | Some("bandit") => bandit::play("v56y"),
        Some(other) => {
            eprintln!(
                "kagg-bandit: unknown subcommand '{other}'\n\
                 usage: kagg-bandit [v56y|v57|mbandit <cfg> <branches> <base>]"
            );
            std::process::exit(2);
        }
    }
}
