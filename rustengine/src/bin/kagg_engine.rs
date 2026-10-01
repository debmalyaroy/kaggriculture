//! `kagg-engine` -- the standalone engine / pre-ranker binary (G0.2, 3-way split).
//!
//! The bit-exact engine plus its two service entry points, carrying NEITHER
//! seat's strategy (no `bandit`/`mbandit`, no trackp `policy`/`search`). This is
//! the shared substrate the two seat lanes and the RL loop run ON:
//!
//!   kagg-engine batch <jobs.tsv> [threads]   mass tape-pair rollouts -> banks
//!   kagg-engine serve                        the stdio env (RL self-play / opp)
//!
//! The monolithic `kagg` retains `batch`/`serve` for backward compatibility;
//! this binary is the clean 3-way peer of `kagg-bandit` / `kagg-trackp`, so a
//! deploy can ship exactly the lane it needs. It is additive: no existing
//! consumer of `kagg` changes.

use kaggriculture_engine::service;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("batch") if args.len() >= 3 => {
            let threads = args
                .get(3)
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                });
            service::batch(&args[2], threads);
        }
        Some("serve") => service::serve(),
        _ => {
            eprintln!(
                "kagg-engine: the engine/pre-ranker substrate (no seat strategy)\n\
                 usage:\n  \
                 kagg-engine batch <jobs.tsv> [threads]\n  \
                 kagg-engine serve"
            );
            std::process::exit(2);
        }
    }
}
