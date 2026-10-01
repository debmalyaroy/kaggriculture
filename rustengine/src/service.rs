//! Track P service modes: `kagg batch` (P4.0) and `kagg serve` (P3.1).
//!
//! batch: play a list of tape-PAIRS at engine speed across all cores and
//! report final banks. This is the generic screening primitive -- the funnel,
//! the P2 generated-route pre-rank, the P4 search and the parked P6 rehab all
//! consume it. PRE-RANKER ONLY: nothing here decides a release.
//!
//! serve: a step-wise environment protocol over stdio for RL rollouts. The
//! observation is emitted as JSON in the OFFICIAL interpreter's observation
//! schema, so the same Python agent code parses both.
//!
//! Single-seat tape format (written by src/trackp/common.py):
//!   SEED <int>
//!   then ONE line per step: `farmer\thand;hand;...\torder;order;...`
//! (The two-seat `kagg episode` format interleaves two such lines per step.)

use crate::engine::{self, PlayerAction, UnitAction};
use crate::state::{Cell, Farm, OMap, Private, State, EPISODE_STEPS};
use std::io::{BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// ------------------------------------------------------------------ tapes --

pub struct Tape {
    pub seed: i64,
    pub steps: Vec<PlayerAction>,
}

pub fn parse_action_line(line: &str) -> PlayerAction {
    let mut parts = line.split('\t');
    let farmer = parts.next().unwrap_or("PASS");
    let hands = parts.next().unwrap_or("");
    let market = parts.next().unwrap_or("");
    PlayerAction {
        farmer: UnitAction::parse(
            &farmer.split(' ').filter(|t| !t.is_empty()).collect::<Vec<_>>()),
        hands: crate::engine::positional(hands)
            .into_iter()
            .map(|h| UnitAction::parse(
                &h.split(' ').filter(|t| !t.is_empty()).collect::<Vec<_>>()))
            .collect(),
        market: crate::engine::positional(market)
            .into_iter()
            .map(|o| o.split(' ').filter(|t| !t.is_empty())
                .map(str::to_string).collect())
            .collect(),
    }
}

pub fn load_tape(path: &str) -> Result<Tape, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut lines = raw.lines();
    let seed: i64 = lines
        .next()
        .ok_or("empty tape")?
        .strip_prefix("SEED ")
        .ok_or("tape must start with 'SEED <n>'")?
        .trim()
        .parse()
        .map_err(|_| "bad seed".to_string())?;
    let steps = lines.map(parse_action_line).collect();
    Ok(Tape { seed, steps })
}

/// Play one full episode from two single-seat tapes. Returns final banks.
pub fn play_pair(seed: i64, a: &Tape, b: &Tape) -> (f64, f64) {
    let mut st = State::new(seed);
    let n = EPISODE_STEPS as usize;
    let empty = PlayerAction::default();
    for i in 0..n {
        let a0 = a.steps.get(i).unwrap_or(&empty).clone();
        let a1 = b.steps.get(i).unwrap_or(&empty).clone();
        engine::step(&mut st, &[a0, a1]);
        if st.step >= EPISODE_STEPS {
            break;
        }
    }
    (st.farms[0].money, st.farms[1].money)
}

// ------------------------------------------------------------------ batch --

/// Jobs file: one job per line, `seed \t tapeA \t tapeB`. seed `-` means
/// "use tapeA's SEED header". Output (stdout), in INPUT order:
/// `idx \t seed \t bank0 \t bank1`, or `idx \t ERR \t <msg>`.
pub fn batch(jobs_path: &str, threads: usize) {
    let raw = std::fs::read_to_string(jobs_path).unwrap_or_else(|e| {
        eprintln!("cannot read {jobs_path}: {e}");
        std::process::exit(2);
    });
    let jobs: Vec<(String, String, String)> = raw
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut p = l.split('\t');
            (
                p.next().unwrap_or("-").to_string(),
                p.next().unwrap_or("").to_string(),
                p.next().unwrap_or("").to_string(),
            )
        })
        .collect();

    let results: Mutex<Vec<Option<String>>> =
        Mutex::new(vec![None; jobs.len()]);
    let next = AtomicUsize::new(0);
    let nthreads = threads.max(1).min(jobs.len().max(1));

    std::thread::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (seed_s, pa, pb) = &jobs[i];
                let line = match (load_tape(pa), load_tape(pb)) {
                    (Ok(ta), Ok(tb)) => {
                        let seed = if seed_s == "-" {
                            ta.seed
                        } else {
                            seed_s.parse().unwrap_or(ta.seed)
                        };
                        let (b0, b1) = play_pair(seed, &ta, &tb);
                        format!("{i}\t{seed}\t{b0}\t{b1}")
                    }
                    (Err(e), _) | (_, Err(e)) => format!("{i}\tERR\t{e}"),
                };
                results.lock().unwrap()[i] = Some(line);
            });
        }
    });

    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for line in results.into_inner().unwrap().into_iter().flatten() {
        writeln!(out, "{line}").unwrap();
    }
}

// ------------------------------------------------------------- JSON emit --

fn json_escape(s: &str) -> String {
    // Engine strings are ASCII tokens; quotes/backslashes never occur, but
    // escape them anyway so the emitter is total.
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_omap(m: &OMap) -> String {
    let rows: Vec<String> = m
        .0
        .iter()
        .map(|(k, v)| format!("\"{}\": {}", json_escape(k), v))
        .collect();
    format!("{{{}}}", rows.join(", "))
}

fn json_tile(c: &Cell) -> String {
    match c {
        Cell::Empty => "null".to_string(),
        Cell::Locked => "\"LOCKED\"".to_string(),
        Cell::Weed => "{\"kind\": \"WEED\"}".to_string(),
        Cell::Plant {
            crop, planted_day, watered_today, consecutive_unwatered,
            yield_units, max_lifespan_step, fertilized_until_day,
        } => format!(
            "{{\"kind\": \"PLANT\", \"crop\": \"{}\", \"planted_day\": {}, \
             \"watered_today\": {}, \"consecutive_unwatered\": {}, \
             \"yield_units\": {}, \"max_lifespan_step\": {}, \
             \"fertilized_until_day\": {}}}",
            crop, planted_day, watered_today, consecutive_unwatered,
            yield_units, max_lifespan_step, fertilized_until_day),
        Cell::Structure { kind, animal: None } =>
            format!("{{\"kind\": \"{kind}\"}}"),
        Cell::Structure { kind, animal: Some(a) } => format!(
            "{{\"kind\": \"{}\", \"animal\": \"{}\", \"placed_day\": {}, \
             \"yield_units\": {}, \"consecutive_unfed\": {}, \
             \"fed_today\": {}, \"cared_today\": {}, \
             \"fertilizer_available\": {}, \"pending_care_bonus\": {}}}",
            kind, a.animal, a.placed_day, a.yield_units, a.consecutive_unfed,
            a.fed_today, a.cared_today, a.fertilizer_available,
            a.pending_care_bonus),
    }
}

fn json_farm(f: &Farm) -> String {
    let hands: Vec<String> = f
        .hands
        .iter()
        .map(|(x, y)| format!("[{x}, {y}]"))
        .collect();
    let tiles: Vec<String> = f
        .tiles
        .iter()
        .map(|row| {
            let cells: Vec<String> = row.iter().map(json_tile).collect();
            format!("[{}]", cells.join(", "))
        })
        .collect();
    let quads: Vec<String> = f
        .unlocked_quadrants
        .iter()
        .map(|q| format!("\"{q}\""))
        .collect();
    format!(
        "{{\"money\": {}, \"farmer\": [{}, {}], \"hands\": [{}], \
         \"hires_today\": {}, \"unlocked_quadrants\": [{}], \"tiles\": [{}]}}",
        f.money, f.farmer.0, f.farmer.1, hands.join(", "), f.hires_today,
        quads.join(", "), tiles.join(", "))
}

fn json_private(p: &Private) -> String {
    let invs: Vec<String> = p.inventories.iter().map(json_omap).collect();
    format!(
        "{{\"shed\": {}, \"seeds\": {}, \"inventories\": [{}]}}",
        json_omap(&p.shed), json_omap(&p.seeds), invs.join(", "))
}

/// Full-state observation in the official schema (both seats' private
/// included -- the trainer owns both sides).
pub fn json_state(st: &State, done: bool) -> String {
    let shops: Vec<String> = st
        .town
        .unlocked_shops
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect();
    format!(
        "{{\"step\": {}, \"day\": {}, \"hour\": {}, \"done\": {}, \
         \"farms\": [{}, {}], \
         \"market\": {{\"inventory\": {}, \"prices\": {}}}, \
         \"town\": {{\"unlocked_shops\": [{}]}}, \
         \"private\": [{}, {}]}}",
        st.step, st.day(), st.step % crate::state::TURNS_PER_DAY,
        done,
        json_farm(&st.farms[0]), json_farm(&st.farms[1]),
        json_omap(&st.market.inventory), json_omap(&st.market.prices),
        shops.join(", "),
        json_private(&st.private[0]), json_private(&st.private[1]))
}

// ------------------------------------------------------------------ serve --

/// Line protocol on stdin; one JSON observation per response on stdout.
///
///   RESET <seed>                         both seats driven by the caller
///   RESET <seed> OPP <seat> <tapepath>   seat <seat> scripted from a tape
///   STEP <line>                          caller's action for the free seat
///   STEP2 <line0>\x1e<line1>             both seats (0x1e separates)
///   QUIT
///
/// <line> is the single-seat tape format `farmer\thands\tmarket`. Responses:
/// after RESET the initial observation; after STEP/STEP2 the post-step
/// observation (with "done" true once the episode ends).
/// Batched (vectorized) serve for on-policy RL: hold N games and step them ALL
/// in ONE stdio round-trip. Kills the N-round-trip-per-game-step IPC tax that
/// caps Python self-play (24 sequential STEP2 -> 1 VSTEP). Protocol:
///   VRESET <n> <seed0>   -> create N games (seeds seed0..seed0+n-1)
///   VSTEP <g0a0>\x1e<g0a1>\x1f<g1a0>\x1e<g1a1>\x1f...  -> step all N one step
/// Each reply is a JSON array [state0, state1, ...] (one per game). The engine
/// steps in Rust; only the tiny obs JSON crosses to Python for the NN forward.
fn write_vstates<W: std::io::Write>(out: &mut W, games: &[State]) {
    // Emit TOKENS per seat (Rust does the v2 encoding -> Python skips encode_tokens)
    // plus banks (reward) + step. t0/t1 are "ntok v v ..." flat strings.
    write!(out, "[").unwrap();
    for (i, s) in games.iter().enumerate() {
        if i > 0 {
            write!(out, ",").unwrap();
        }
        write!(out, "{{\"s\":{},\"m0\":{},\"m1\":{},\"nh0\":{},\"nh1\":{},\"t0\":\"{}\",\"t1\":\"{}\"}}",
               s.step, s.farms[0].money as i64, s.farms[1].money as i64,
               s.farms[0].hands.len(), s.farms[1].hands.len(),
               crate::obstoken::tokens_flat(s, 0),
               crate::obstoken::tokens_flat(s, 1)).unwrap();
    }
    writeln!(out, "]").unwrap();
    out.flush().unwrap();
}

pub fn vecserve() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut games: Vec<State> = Vec::new();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        if let Some(rest) = line.strip_prefix("VRESET ") {
            let mut p = rest.splitn(2, ' ');
            let n: usize = p.next().unwrap_or("0").trim().parse().unwrap_or(0);
            let seed0: i64 = p.next().unwrap_or("0").trim().parse().unwrap_or(0);
            games = (0..n).map(|i| State::new(seed0 + i as i64)).collect();
            write_vstates(&mut out, &games);
            continue;
        }
        if let Some(rest) = line.strip_prefix("VSTEP ") {
            for (i, ch) in rest.split('\x1f').enumerate() {
                if i >= games.len() {
                    break;
                }
                let mut halves = ch.split('\x1e');
                let a0 = parse_action_line(halves.next().unwrap_or(""));
                let a1 = parse_action_line(halves.next().unwrap_or(""));
                engine::step(&mut games[i], &[a0, a1]);
            }
            write_vstates(&mut out, &games);
            continue;
        }
        writeln!(out, "{{\"error\": \"unknown vec command\"}}").unwrap();
        out.flush().unwrap();
    }
}


pub fn serve() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut st: Option<State> = None;
    let mut opp: Option<(usize, Tape)> = None;

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        if let Some(rest) = line.strip_prefix("LOADSTATE ") {
            // STATE INJECTION (2026-09-05): continue an episode from a
            // caller-supplied mid-game state (json_state format, optional
            // "seed" for the rollout RNG). The runtime searcher's entry
            // point -- the ladder never exposes the seed, so rollouts start
            // HERE, not at RESET.
            match crate::json::parse(rest)
                .and_then(|j| crate::loadstate::state_from_json(&j))
            {
                Ok(s) => {
                    writeln!(out, "{}", json_state(&s, false)).unwrap();
                    out.flush().unwrap();
                    st = Some(s);
                }
                Err(e) => {
                    writeln!(out, "{{\"error\": \"{}\"}}", json_escape(&e))
                        .unwrap();
                    out.flush().unwrap();
                }
            }
            opp = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("ROLLOUT ") {
            // ONE-SHOT ROLLOUT (2026-09-07): snapshot + both seats' action
            // streams + horizon in a single message; steps internally and
            // returns only the final observation. Kills the per-step IPC
            // that made deep tree search unaffordable (24 round-trips ->
            // 1). Format:
            //   ROLLOUT <H> <snapjson>\x1e<lines0>\x1e<lines1>
            // where <linesN> is that seat's H tape lines joined by \x1f.
            let mut head = rest.splitn(2, ' ');
            let h: usize = head.next().unwrap_or("0").parse().unwrap_or(0);
            let rest2 = head.next().unwrap_or("");
            let mut parts = rest2.split('\x1e');
            let snap = parts.next().unwrap_or("");
            let lines0: Vec<&str> =
                parts.next().unwrap_or("").split('\x1f').collect();
            let lines1: Vec<&str> =
                parts.next().unwrap_or("").split('\x1f').collect();
            match crate::json::parse(snap)
                .and_then(|j| crate::loadstate::state_from_json(&j))
            {
                Ok(mut s) => {
                    let mut done = s.step >= EPISODE_STEPS;
                    for i in 0..h {
                        if done {
                            break;
                        }
                        let a0 = parse_action_line(
                            lines0.get(i).copied().unwrap_or(""));
                        let a1 = parse_action_line(
                            lines1.get(i).copied().unwrap_or(""));
                        engine::step(&mut s, &[a0, a1]);
                        done = s.step >= EPISODE_STEPS;
                    }
                    writeln!(out, "{}", json_state(&s, done)).unwrap();
                    out.flush().unwrap();
                }
                Err(e) => {
                    writeln!(out, "{{\"error\": \"{}\"}}", json_escape(&e))
                        .unwrap();
                    out.flush().unwrap();
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("GENGAME ") {
            // WHOLE-GAME GENERATION (2026-09-07): seed + both tapes in one
            // message; plays the full episode internally and returns the
            // day-boundary observations plus final state. Kills the
            // 719-round-trip IPC tax on self-play generation (~1 s/game
            // -> engine-bound ~10 ms/game). Format:
            //   GENGAME <seed>\x1e<lines0>\x1e<lines1>
            // where <linesN> is seat N's tape lines joined by \x1f.
            // Response: {"days": [obs...], "final": obs}
            let mut parts = rest.split('\x1e');
            let seed: i64 = parts
                .next()
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0);
            let lines0: Vec<&str> =
                parts.next().unwrap_or("").split('\x1f').collect();
            let lines1: Vec<&str> =
                parts.next().unwrap_or("").split('\x1f').collect();
            let mut s = State::new(seed);
            let mut days = String::new();
            // ckpts: full-state snapshots at the DISPATCH phase (day*24+2), the
            // exact step mbandit's dispatch() reads unlocked_shops. Additive and
            // step-tagged (json_state carries "step"), so a consumer keys by the
            // real checkpoint step instead of the day*24-phased, day-8-onward
            // `days` list (which never contained the D6 step 146). Existing
            // `days`/`final` consumers are unaffected.
            let mut ckpts = String::new();
            let mut done = false;
            for i in 0..(EPISODE_STEPS as usize) {
                if done {
                    break;
                }
                let a0 =
                    parse_action_line(lines0.get(i).copied().unwrap_or(""));
                let a1 =
                    parse_action_line(lines1.get(i).copied().unwrap_or(""));
                engine::step(&mut s, &[a0, a1]);
                done = s.step >= EPISODE_STEPS;
                let st = s.step as usize;
                if st % 24 == 0 && st > 191 && st < 700 {
                    if !days.is_empty() {
                        days.push(',');
                    }
                    days.push_str(&json_state(&s, done));
                }
                if st % 24 == 2 && (146..=650).contains(&st) {
                    if !ckpts.is_empty() {
                        ckpts.push(',');
                    }
                    ckpts.push_str(&json_state(&s, done));
                }
            }
            writeln!(
                out,
                "{{\"days\": [{}], \"ckpts\": [{}], \"final\": {}}}",
                days,
                ckpts,
                json_state(&s, done)
            )
            .unwrap();
            out.flush().unwrap();
            continue;
        }
        if let Some(rest) = line.strip_prefix("PLANSEARCH ") {
            // REAL-SIMULATION SEARCH (2026-09-07): from a snapshot, roll
            // EACH candidate plan (our seat) to TERMINAL against a fixed
            // opponent continuation, and return each plan's final banks.
            // No value net -- the leaf is the actual banked score. One IPC
            // for the whole plan set. Format:
            //   PLANSEARCH <me_seat> <snapjson>\x1e<opp>\x1e<plan0>\x1e<plan1>..
            // where <opp>/<planN> are step lines joined by \x1f, indexed
            // from the snapshot's step. Response: {"banks": [[my,op],...]}
            let mut head = rest.splitn(2, ' ');
            let me: usize = head.next().unwrap_or("0").parse().unwrap_or(0);
            let me = me.min(1);
            let rest2 = head.next().unwrap_or("");
            let mut parts = rest2.split('\x1e');
            let snap = parts.next().unwrap_or("");
            let opp_lines: Vec<&str> =
                parts.next().unwrap_or("").split('\x1f').collect();
            let plans: Vec<Vec<&str>> = parts
                .map(|p| p.split('\x1f').collect())
                .collect();
            match crate::json::parse(snap)
                .and_then(|j| crate::loadstate::state_from_json(&j))
            {
                Ok(base) => {
                    let mut banks = String::new();
                    for (pi, plan) in plans.iter().enumerate() {
                        let mut s = base.clone();
                        let mut done = s.step >= EPISODE_STEPS;
                        let mut i = 0usize;
                        while !done {
                            let mine = parse_action_line(
                                plan.get(i).copied().unwrap_or(""));
                            let theirs = parse_action_line(
                                opp_lines.get(i).copied().unwrap_or(""));
                            let acts = if me == 0 {
                                [mine, theirs]
                            } else {
                                [theirs, mine]
                            };
                            engine::step(&mut s, &acts);
                            done = s.step >= EPISODE_STEPS;
                            i += 1;
                        }
                        if pi > 0 {
                            banks.push(',');
                        }
                        banks.push_str(&format!(
                            "[{}, {}]",
                            s.farms[me].money as i64,
                            s.farms[1 - me].money as i64));
                    }
                    writeln!(out, "{{\"banks\": [{}]}}", banks).unwrap();
                    out.flush().unwrap();
                }
                Err(e) => {
                    writeln!(out, "{{\"error\": \"{}\"}}", json_escape(&e))
                        .unwrap();
                    out.flush().unwrap();
                }
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("RESET ") {
            let mut parts = rest.splitn(4, ' ');
            let seed: i64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
            opp = None;
            if parts.next() == Some("OPP") {
                let seat: usize =
                    parts.next().unwrap_or("1").parse().unwrap_or(1);
                let path = parts.next().unwrap_or("");
                match load_tape(path) {
                    Ok(t) => opp = Some((seat.min(1), t)),
                    Err(e) => {
                        writeln!(out, "{{\"error\": \"{}\"}}", json_escape(&e))
                            .unwrap();
                        out.flush().unwrap();
                        continue;
                    }
                }
            }
            let s = State::new(seed);
            writeln!(out, "{}", json_state(&s, false)).unwrap();
            out.flush().unwrap();
            st = Some(s);
            continue;
        }
        let (a0, a1): (PlayerAction, PlayerAction) =
            if let Some(rest) = line.strip_prefix("STEP2 ") {
                let mut halves = rest.split('\x1e');
                (
                    parse_action_line(halves.next().unwrap_or("")),
                    parse_action_line(halves.next().unwrap_or("")),
                )
            } else if let Some(rest) = line.strip_prefix("STEP ") {
                let ours = parse_action_line(rest);
                match (&st, &opp) {
                    (Some(s), Some((seat, tape))) => {
                        let idx = s.step as usize;
                        let scripted = tape
                            .steps
                            .get(idx)
                            .cloned()
                            .unwrap_or_default();
                        if *seat == 0 {
                            (scripted, ours)
                        } else {
                            (ours, scripted)
                        }
                    }
                    _ => (ours, PlayerAction::default()),
                }
            } else {
                writeln!(out, "{{\"error\": \"unknown command\"}}").unwrap();
                out.flush().unwrap();
                continue;
            };
        let Some(s) = st.as_mut() else {
            writeln!(out, "{{\"error\": \"no episode; RESET first\"}}")
                .unwrap();
            out.flush().unwrap();
            continue;
        };
        engine::step(s, &[a0, a1]);
        let done = s.step >= EPISODE_STEPS;
        writeln!(out, "{}", json_state(s, done)).unwrap();
        out.flush().unwrap();
    }
}
