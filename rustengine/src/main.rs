//! Kaggriculture engine port.
//!
//! The RNG (bit-exact against CPython) AND the full step function are ported and
//! bit-identical to the official engine (see tests/test_rust_engine.py). The
//! compiled agent seats run on it: `kagg bandit` (v56y) and `kagg trackp` (v57)
//! are the shipped tape-in-shell agents; `serve`/`batch`/`episode`/`bench` drive
//! offline evaluation. The Phase-B `play`/searcher path is a PRE-RANKER only.
//!
//! NON-NEGOTIABLE, wherever this ends up: this binary is a PRE-RANKER. Every
//! release decision -- crown, playoff, gates, A/Bs -- stays on the official
//! Python interpreter. Drift then costs ranking quality, not correctness. The
//! repo already holds this line (`src/fastsim.py` refuses to reimplement the
//! rules; `src/parity.py` measures forecast error against real episodes), and
//! the reason is on record: a Kaggle image once shipped a divergent
//! interpreter -- startingMoney 2000 against 3000, COW 600 against 400,
//! SELL FERTILIZER silently dropped -- and nothing raised.
//!
//! Usage:
//!   kagg rng-probe <seed> <day>     reference vectors, for the D1 probe
//!   kagg weeds <seed> <day> <n>     the weed-spawn draw sequence
//!   kagg selftest                   state scaffold sanity + digest shape
//!   kagg prices <inv>               all 9 quoted prices at that inventory
//!   kagg price-sweep <item> <lo> <hi> <step>   raw+quoted, for diffing
//!   kagg rules                      pure rule tables + functions, for diffing

mod engine;
mod core;
mod bandit;
mod mbandit;
mod json;
mod loadstate;
mod market;
mod mt19937;
mod obsstate;
mod obstoken;
mod plan;
mod policy;
mod rules;
mod search;
mod service;
mod state;
mod value;

use mt19937::MT;
use std::env;
use std::io::{BufRead, BufWriter, Write};

/// `sorted(SHOPS)` from the interpreter. Order AND LENGTH matter: `choice`
/// indexes this list, so a wrong length changes `randbelow(n)` and the whole
/// draw. An earlier version had 6 entries (taken from one replay's observed
/// unlocked_shops) while the engine defines 8 -- the D1 test hardcoded the
/// same wrong list on both sides and so validated against itself.
const SHOPS: [&str; 8] = [
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
];

fn rng_probe(seed: i64, day: i64) {
    let mut a = MT::for_day(seed, day);
    // Transmit the IEEE-754 bit pattern, not a decimal rendering. `{:.17}` is
    // 17 digits AFTER THE POINT, so a value like 0.0154... loses significant
    // digits and no longer round-trips -- which looked exactly like an RNG
    // divergence on the first run of the D1 test.
    let randoms: Vec<String> = (0..8).map(|_| a.random().to_bits().to_string()).collect();
    let mut b = MT::for_day(seed, day);
    let bits: Vec<String> = (0..4).map(|_| b.getrandbits(32).to_string()).collect();
    let mut c = MT::for_day(seed, day);
    let choices: Vec<String> = (0..4)
        .map(|_| format!("\"{}\"", c.choice(&SHOPS)))
        .collect();
    println!(
        "{{\"day\": {}, \"first8_random\": [{}], \"getrandbits32_first4\": [{}], \"choice_first4\": [{}]}}",
        day,
        randoms.join(", "),
        bits.join(", "),
        choices.join(", ")
    );
}

/// The weed draw for one day: one `random()` per empty tile, per player, in the
/// interpreter's iteration order. Emitting the raw sequence lets the harness
/// compare against Python without needing the board state.
fn weeds(seed: i64, day: i64, n: usize) {
    let mut r = MT::for_day(seed, day);
    let vals: Vec<String> = (0..n).map(|_| r.random().to_bits().to_string()).collect();
    println!("[{}]", vals.join(", "));
}

/// Scaffold sanity: the state model constructs and digests. This is NOT a
/// correctness claim about the engine -- the step function is not ported.
fn selftest() {
    let st = state::State::new(1117071212);
    let d = st.digest();
    println!("{{\"board\": {}, \"steps\": {}, \"shed_cap\": {}, \"max_orders\": {}, \"digest_len\": {}, \"starting_money\": {}}}",
        state::BOARD, state::EPISODE_STEPS, state::SHED_CAP,
        state::MAX_MARKET_ORDERS, d.len(), st.farms[0].money);
}

/// Every product's quoted price at one inventory level.
fn prices(inv: f64) {
    let parts: Vec<String> = market::PARAMS
        .iter()
        .map(|p| format!("\"{}\": {}", p.item, market::price(p, inv)))
        .collect();
    println!("{{{}}}", parts.join(", "));
}

/// Raw and quoted price across a range, for the differential sweep. Raw is
/// emitted as a bit pattern so float comparison is exact.
fn price_sweep(item: &str, lo: f64, hi: f64, step: f64) {
    let p = match market::param(item) {
        Some(p) => p,
        None => {
            eprintln!("unknown item {item}");
            std::process::exit(2);
        }
    };
    let mut rows: Vec<String> = Vec::new();
    let mut inv = lo;
    while inv <= hi {
        rows.push(format!(
            "[{}, {}, {}]",
            inv,
            market::price_raw(p, inv).to_bits(),
            market::price(p, inv)
        ));
        inv += step;
    }
    println!("[{}]", rows.join(", "));
}

/// Dump every pure rule function/table so the differential test can compare
/// them against the interpreter without re-implementing anything in Python.
fn rules_dump() {
    let fibs: Vec<String> = (0..25).map(|n| rules::fib(n).to_string()).collect();
    let hires: Vec<String> = (0..15)
        .map(|n| rules::hire_cost(n, rules::FARM_HAND_COST_MULT).to_string())
        .collect();
    let mut quads: Vec<String> = Vec::new();
    for y in 0..10i64 {
        for x in 0..10i64 {
            quads.push(format!("\"{}\"", rules::quadrant_of(x, y, 10)));
        }
    }
    let shed: Vec<String> = rules::shed_access_tiles(10)
        .iter()
        .map(|(x, y)| format!("[{x}, {y}]"))
        .collect();
    let crops: Vec<String> = rules::CROPS
        .iter()
        .map(|c| format!(
            "\"{}\": {{\"seed\": {}, \"first_yield_day\": {}, \"max_yield_day\": {}, \"interval\": {}, \"max_yield\": {}, \"ongoing\": {}}}",
            c.name, c.seed_cost, c.first_yield_day, c.max_yield_day,
            c.interval, c.max_yield, c.ongoing))
        .collect();
    let animals: Vec<String> = rules::ANIMALS
        .iter()
        .map(|a| format!(
            "\"{}\": {{\"cost\": {}, \"structure\": \"{}\", \"first_yield_day\": {}, \"interval\": {}, \"max_held\": {}, \"product\": \"{}\"}}",
            a.name, a.cost, a.structure, a.first_yield_day, a.interval,
            a.max_held, a.product))
        .collect();
    println!(
        "{{\"fib\": [{}], \"hire_cost\": [{}], \"quadrants\": [{}], \"shed_access\": [{}], \"land_order\": [\"{}\"], \"land_prices\": [{}], \"max_shop_instances\": {}, \"crops\": {{{}}}, \"animals\": {{{}}}}}",
        fibs.join(", "), hires.join(", "), quads.join(", "), shed.join(", "),
        rules::LAND_ORDER.join("\", \""),
        rules::LAND_PRICES.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        rules::MAX_SHOP_INSTANCES,
        crops.join(", "), animals.join(", "));
}

/// Run a full episode from an action tape and print one digest per step.
///
/// Tape format (written by tests/test_rust_engine.py):
///   SEED <int>
///   then, per step, one line per player:
///     <farmer tokens> TAB <hand;hand;...> TAB <order;order;...>
///   tokens inside an action are space-separated; empty sections are empty.
///
/// Output: `<step> <digest>` per step, digest taken AFTER the step applies --
/// the same instant the Python framework records its post-interpreter state.
fn episode(path: &str) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("cannot open {path}: {e}");
        std::process::exit(2);
    });
    let mut lines = std::io::BufReader::new(file).lines();
    let seed_line = lines.next().expect("empty tape").unwrap();
    let seed: i64 = seed_line
        .strip_prefix("SEED ")
        .expect("tape must start with 'SEED <n>'")
        .trim()
        .parse()
        .expect("bad seed");

    let parse_player = |line: &str| -> engine::PlayerAction {
        let mut parts = line.split('\t');
        let farmer = parts.next().unwrap_or("PASS");
        let hands = parts.next().unwrap_or("");
        let market = parts.next().unwrap_or("");
        engine::PlayerAction {
            farmer: engine::UnitAction::parse(
                &farmer.split(' ').filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()),
            hands: engine::positional(hands)
                .into_iter()
                .map(|h| engine::UnitAction::parse(
                    &h.split(' ').filter(|t| !t.is_empty())
                        .collect::<Vec<_>>()))
                .collect(),
            market: engine::positional(market)
                .into_iter()
                .map(|o| o.split(' ').filter(|t| !t.is_empty())
                    .map(str::to_string).collect())
                .collect(),
        }
    };

    let mut st = state::State::new(seed);
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    loop {
        let Some(l0) = lines.next() else { break };
        let Some(l1) = lines.next() else { break };
        let actions = [parse_player(&l0.unwrap()), parse_player(&l1.unwrap())];
        engine::step(&mut st, &actions);
        writeln!(out, "{} {}", st.step, st.digest()).unwrap();
    }
    writeln!(out, "FINAL {} {}", st.farms[0].money.to_bits(),
             st.farms[1].money.to_bits()).unwrap();
}

/// Throughput measurement: replay one tape N times, report steps/second.
fn bench(path: &str, reps: usize) {
    let raw = std::fs::read_to_string(path).unwrap();
    let mut lines = raw.lines();
    let seed: i64 = lines.next().unwrap()
        .strip_prefix("SEED ").unwrap().trim().parse().unwrap();
    let body: Vec<&str> = lines.collect();
    let t0 = std::time::Instant::now();
    let mut total_steps = 0u64;
    let mut sink = 0.0f64;
    for _ in 0..reps {
        let mut st = state::State::new(seed);
        for pair in body.chunks(2) {
            if pair.len() < 2 {
                break;
            }
            let parse = |line: &str| -> engine::PlayerAction {
                let mut parts = line.split('\t');
                engine::PlayerAction {
                    farmer: engine::UnitAction::parse(
                        &parts.next().unwrap_or("PASS").split(' ')
                            .filter(|t| !t.is_empty()).collect::<Vec<_>>()),
                    hands: engine::positional(parts.next().unwrap_or(""))
                        .into_iter()
                        .map(|h| engine::UnitAction::parse(
                            &h.split(' ').filter(|t| !t.is_empty())
                                .collect::<Vec<_>>()))
                        .collect(),
                    market: engine::positional(parts.next().unwrap_or(""))
                        .into_iter()
                        .map(|o| o.split(' ').filter(|t| !t.is_empty())
                            .map(str::to_string).collect())
                        .collect(),
                }
            };
            let actions = [parse(pair[0]), parse(pair[1])];
            engine::step(&mut st, &actions);
            total_steps += 1;
        }
        sink += st.farms[0].money;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{{\"steps\": {}, \"seconds\": {:.4}, \"steps_per_sec\": {:.0}, \
         \"episodes_per_sec\": {:.2}, \"sink\": {}}}",
        total_steps, dt, total_steps as f64 / dt,
        reps as f64 / dt, sink);
}

/// Per-turn search budget for `kagg play`, in milliseconds.
///
/// `--budget-ms N` (or a bare trailing integer), overridden by
/// `TRACKP_BUDGET_MS`. Anything unparseable is 0, i.e. the skeleton: a typo in
/// a build script must not silently ship a searcher, and it must not silently
/// ship a budget that blows the per-turn watchdog either.
fn budget_arg(args: &[String]) -> u64 {
    if let Ok(v) = env::var("TRACKP_BUDGET_MS") {
        return v.trim().parse().unwrap_or(0);
    }
    let mut it = args.iter().skip(2);
    while let Some(a) = it.next() {
        if a == "--budget-ms" {
            return it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        }
        if let Some(v) = a.strip_prefix("--budget-ms=") {
            return v.parse().unwrap_or(0);
        }
        if let Ok(v) = a.parse::<u64>() {
            return v;
        }
    }
    0
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let usage = "usage: kagg <bandit|trackp|play|serve|mbandit <cfg> <branches> <base>|\n              batch <jobs.tsv> [threads]|episode <id>|bench <id> <n>|rules|\n              selftest|rng-probe <seed> <day>|weeds <seed> <day> <n>|\n              prices <inv>|price-sweep <item> <lo> <hi> <step>>";
    if args.len() < 2 {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "rng-probe" if args.len() == 4 => {
            rng_probe(args[2].parse().unwrap(), args[3].parse().unwrap())
        }
        "weeds" if args.len() == 5 => weeds(
            args[2].parse().unwrap(),
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
        ),
        "selftest" => selftest(),
        "rules" => rules_dump(),
        "episode" if args.len() == 3 => episode(&args[2]),
        "bench" if args.len() == 4 => {
            bench(&args[2], args[3].parse().unwrap())
        }
        // Track P service modes (P4.0 / P3.1). PRE-RANKER ONLY, as above.
        "batch" if args.len() >= 3 => {
            let threads = args
                .get(3)
                .and_then(|t| t.parse().ok())
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                });
            service::batch(&args[2], threads)
        }
        "serve" => service::serve(),
        "vecserve" => service::vecserve(),
        "toktest" if args.len() >= 3 => {
            // parity probe: play <nsteps> PASS/PASS from <seed>, print seat's tokens
            let seed: i64 = args[2].parse().unwrap_or(0);
            let nsteps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let seat: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            let mut s = crate::state::State::new(seed);
            let empty = crate::engine::PlayerAction::default();
            for _ in 0..nsteps {
                crate::engine::step(&mut s, &[empty.clone(), empty.clone()]);
            }
            println!("{}", crate::obstoken::tokens_flat(&s, seat));
        }
        // The compiled-agent transport (Track P Phase A). AGENT mode, not a
        // pre-ranker: `play` is what ships inside submission.tar.gz.
        //
        //   kagg play                    the skeleton policy (Phase A)
        //   kagg play --budget-ms 100    the Phase-B searcher at 100 ms/turn
        //
        // The DEFAULT is 0 = skeleton. A search budget must be asked for
        // explicitly: `kagg play` bare is the path whose byte-for-byte
        // equivalence with the Python fallback is a shipped guarantee, and a
        // default that quietly searched would retire that guarantee for every
        // caller that has not been updated. TRACKP_BUDGET_MS overrides the
        // flag, so a measurement can sweep budgets without repacking.
        "play" => policy::play_with(budget_arg(&args)),
        "bandit" => bandit::play("v56y"),
        "trackp" => bandit::play("v57"),
        // config-driven multi-checkpoint RC harness: reads config + branches +
        // base from disk (positional args), so a daily reship is package-only.
        "mbandit" => mbandit::play(&args),
        "play-selftest" => policy::play_selftest(),
        "prices" if args.len() == 3 => prices(args[2].parse().unwrap()),
        "price-sweep" if args.len() == 6 => price_sweep(
            &args[2],
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
        ),
        _ => {
            eprintln!("{usage}");
            std::process::exit(2);
        }
    }
}
