//! Replay a real ladder game's WORLD (engine seed = replay `info.seed`) against the opponent's
//! recorded actions (open-loop tape), with either our recorded actions (`--verify`: must reproduce
//! the replay's banks exactly) or a Rust agent in our seat (counterfactual: "what would X have
//! banked against what that opponent actually did?").
//!
//!     tapeplay --tapes DIR --verify
//!     tapeplay --tapes DIR --profiles configs/profiles/v2.json --pa 13 [--base configs/bases/v61.1]
//!     tapeplay --tapes DIR --profiles P --policy W.bin --shield configs/shield/v1.json --guarded
//!                                            (a learned candidate vs the band tapes: python/band_gate.py)
//!     tapeplay --tapes DIR --profiles P --pa 19 --guarded   (the opponent's recorded stream is played
//!                                             through the chassis guards -- hand alignment, weed repair,
//!                                             budget/room guards, sell clamps, dead stock, liquidation --
//!                                             so it stays legal and funded against a different us)
//!     tapeplay --tapes DIR --profiles P --pa 35 --group 35,35,36 --guarded   (per-group profile from step 25)
//!     tapeplay --tapes DIR --obs-dump FILE   (agent mode: our seat's dayobs vectors, one line per
//!                                             decision day: id seat day v0..v89, tab-separated)
//!
//! Tapes come from python/replay_to_tape.py. One line per tape:
//! `id<TAB>seat<TAB>replay_us<TAB>replay_them<TAB>sim_us<TAB>sim_them`.
//! Caveat: the opponent is open-loop — a reactive opponent would have reacted to a different us.
use agent::act::Action;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::{self, Json};
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;

/// The seat's observation straight from the engine state (Obs::from_state: bit-identical to parsing the
/// JSON observation, ~25x cheaper; see ../docs/history/fast-tournaments-2026-09-26.md). KAGG_OBS_JSON=1 = the JSON path.
fn obs_of(st: &State, seat: usize) -> Result<agent::obs::Obs, String> {
    if std::env::var_os("KAGG_OBS_JSON").is_some() {
        agent::obs::Obs::parse(&seat_obs_json(st, seat))
    } else {
        Ok(agent::obs::Obs::from_state(st, seat))
    }
}


fn act_of(j: &Json) -> PlayerAction {
    if j.is_null() {
        return runner::to_engine(&Action::pass());
    }
    runner::to_engine(&Action::from_json(j))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let dir = get("--tapes").expect("--tapes DIR");
    let verify = args.iter().any(|a| a == "--verify");
    let mut base = if verify { None } else { Some(agent::base::Base::load(&get("--base").unwrap_or_else(|| "configs/bases/v61.1".into())).expect("base")) };
    if let (Some(b), Some(p)) = (base.as_mut(), get("--profiles")) {
        b.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("profiles");
        b.chain.clone_profile = get("--ca").and_then(|s| s.parse().ok());
        b.chain.afr_profile = get("--fa").and_then(|s| s.parse().ok());
        b.chain.afr_trigger = get("--ftrig").and_then(|s| s.parse().ok()).unwrap_or(1);
    }
    // a learned candidate: --policy W.bin (the GRU picks each day's profile greedily; needs --profiles)
    // under --shield F (the opponent-group mask); the same wiring as agent-stdio / ppo-rollout
    if let (Some(b), Some(w)) = (base.as_mut(), get("--policy")) {
        let net = policy::Net::load(&w).expect("policy weights");
        let n = b.chain.profiles.len();
        assert_eq!(net.n_act, n, "policy action count != --profiles table");
        b.obs_track = Some(dayobs::Builder::new());
        b.policy = Some(agent::base::PolicyCtl::new(std::sync::Arc::new(net), vec![true; n], None));
    }
    if let (Some(b), Some(f)) = (base.as_mut(), get("--shield")) {
        let j = json::parse(&std::fs::read_to_string(&f).expect("shield file")).expect("shield json");
        let (sh, tol) = agent::layers::group::Shield::from_json(&j).expect("shield");
        b.chain.group_ctl = Some(agent::layers::group::GroupCtl::shield_only(sh, tol));
    }
    // --group P_DIFF,P_PARTIAL,P_COPY: the opponent-group controller plays one fixed profile per day-0
    // group from the day-1 boundary on (day 0 keeps --pa); the same controller as selfplay --group-a
    if let (Some(b), Some(g)) = (base.as_mut(), get("--group")) {
        let v: Vec<usize> = g.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        assert_eq!(v.len(), 3, "--group P_DIFF,P_PARTIAL,P_COPY");
        b.chain.group_ctl = Some(agent::layers::group::GroupCtl::new([v[0], v[1], v[2]], [0, 0, 0]));
    }
    // --force-route R: route R in every world (screening); --route-table FILE: per-world route overrides
    if let (Some(b), Some(r)) = (base.as_mut(), get("--force-route").and_then(|s| s.parse::<i64>().ok())) {
        b.chassis.router.force(r);
    }
    // --disguise: the stream disguise (crates/agent/src/disguise.rs)
    if let (Some(b), true) = (base.as_mut(), args.iter().any(|a| a == "--disguise")) {
        b.disguise = Some(agent::disguise::Disguise::default());
    }
    // --opening-route K: play route K's actions before day 6 instead of route 0 (a different opening line)
    if let (Some(b), Some(k)) = (base.as_mut(), get("--opening-route").and_then(|s| s.parse::<i64>().ok())) {
        b.chassis.router.opening = k;
    }
    if let (Some(b), Some(f)) = (base.as_mut(), get("--route-table")) {
        b.chassis.router.override_worlds(&json::parse(&std::fs::read_to_string(&f).expect("route table")).expect("route table json"));
    }
    // --shell FILE: the learned sales shell (crates/agent/src/shell.rs) after the chain
    if let (Some(b), Some(f)) = (base.as_mut(), get("--shell")) {
        let m = agent::shell::ShellModel::load(&f).expect("shell model");
        b.shell = Some(agent::shell::ShellCtl::new(std::sync::Arc::new(m)));
    }
    // --rshell F / --chain-off S / --knob-over F: reactive shell v2, whole-game stage switches, knob overrides
    if let Some(b) = base.as_mut() {
        runner::apply_extras(b, &args);
    }
    let guarded = args.iter().any(|a| a == "--guarded");
    // --record DIR: write each game as played (both seats' actions) as a tape, for DAgger labelling
    let record = get("--record");
    let takeover: usize = get("--takeover").and_then(|s| s.parse().ok()).unwrap_or(0);
    if let Some(rd) = record.as_ref() {
        std::fs::create_dir_all(rd).expect("record dir");
    }
    // the guarded opponent (runner::guard_router / guarded_tape_base): its tape as the only route of a
    // chassis with only the guards that keep a recorded stream legal against a different us
    let guard_router = runner::guard_router(&get("--base").unwrap_or_else(|| "configs/bases/v61.1".into()));
    // --money-dump FILE: both seats' money after every step (id, step, ours, theirs), for loss anatomy
    let mut money_out = get("--money-dump").map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("money-dump file")));
    let mut obs_out = get("--obs-dump").map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("obs-dump file")));
    if let (Some(b), true) = (base.as_mut(), obs_out.is_some()) {
        b.obs_track = Some(dayobs::Builder::new());
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let (mut n, mut exact, mut w_rep, mut w_sim) = (0, 0, 0, 0);
    for f in files {
        let j = json::parse(&std::fs::read_to_string(&f).unwrap()).expect("tape json");
        let seed = j.get("seed").i64();
        let seat = j.get("seat").i64() as usize;
        let rewards = j.get("rewards").arr().iter().map(|x| x.f64()).collect::<Vec<_>>();
        let acts = j.get("actions").arr();
        let mut st = State::new(seed);
        let mut me = base.as_ref().map(|b| b.fresh());
        let mut opp = guarded.then(|| {
            let stream: Vec<Action> = acts.iter().map(|p| p.arr().get(1 - seat).map(Action::from_json).unwrap_or_else(Action::pass)).collect();
            let t = runner::Tape { id: String::new(), seed, seat, band: String::new(), stream: std::sync::Arc::new(stream) };
            runner::guarded_tape_base(&t, &guard_router)
        });
        // --verify + --obs-dump: both seats' dayobs vectors along the exact recorded trajectory
        let mut vb: Option<[dayobs::Builder; 2]> = (verify && obs_out.is_some()).then(Default::default);
        let mut vrows: Vec<(usize, usize, [f32; dayobs::N])> = vec![];
        let mut t = 0usize;
        let mut money: Vec<(usize, f64, f64)> = vec![];
        let mut rec_acts: Vec<String> = vec![];
        let mut agree = 0usize;
        loop {
            let pair = acts.get(t).map(|p| p.arr()).unwrap_or(&[]);
            let tape = |s: usize| pair.get(s).map(act_of).unwrap_or_else(|| act_of(&Json::Null));
            if let Some(bs) = vb.as_mut() {
                for (s, b) in bs.iter_mut().enumerate() {
                    let text = seat_obs_json(&st, s);
                    if let Ok(o) = agent::obs::Obs::parse(&text) {
                        if let Ok(v) = agent::view::View::new(o) {
                            b.observe(agent::dayview::step_view(&v));
                            if b.at_decision() {
                                if let Some(x) = b.vector() {
                                    vrows.push((s, t / dayobs::TURNS_PER_DAY, x));
                                }
                            }
                        }
                    }
                    let own = pair.get(s).map(Action::from_json).unwrap_or_else(Action::pass);
                    b.acted(agent::dayview::own_act(&own));
                }
            }
            let mut ours_a: Action = match me.as_mut() {
                Some(b) => match obs_of(&st, seat) {
                    Ok(o) => b.act(o).0,
                    Err(_) => Action::pass(),
                },
                None => pair.get(seat).map(Action::from_json).unwrap_or_else(Action::pass),
            };
            // --takeover T: the recorded player's own actions until turn T (our agent follows along, so its
            // state is built from the same history), then our agent plays their seat from the same position
            if t < takeover {
                let rec = pair.get(seat).map(Action::from_json).unwrap_or_else(Action::pass);
                if me.is_some() && runner::action_json(&rec) == runner::action_json(&ours_a) {
                    agree += 1;
                }
                ours_a = rec;
            }
            let theirs_a: Action = match opp.as_mut() {
                Some(b) => match obs_of(&st, 1 - seat) {
                    Ok(o) => b.act(o).0,
                    Err(_) => pair.get(1 - seat).map(Action::from_json).unwrap_or_else(Action::pass),
                },
                None => pair.get(1 - seat).map(Action::from_json).unwrap_or_else(Action::pass),
            };
            if record.is_some() {
                let (x, y) = (runner::action_json(&ours_a), runner::action_json(&theirs_a));
                rec_acts.push(if seat == 0 { format!("[{x},{y}]") } else { format!("[{y},{x}]") });
            }
            let ours = if me.is_some() { runner::to_engine(&ours_a) } else { tape(seat) };
            let theirs = if opp.is_some() { runner::to_engine(&theirs_a) } else { tape(1 - seat) };
            let a: [PlayerAction; 2] = if seat == 0 { [ours, theirs] } else { [theirs, ours] };
            t += 1;
            // KRL_GROUP_ONLY: print our seat's rival group (decided at step 25) and stop the game at step 30
            if t == 30 && std::env::var_os("KRL_GROUP_ONLY").is_some() {
                let g = me.as_ref().and_then(|b| b.chain.group_ctl.as_ref()).and_then(|g| g.group).map(|g| g as i64).unwrap_or(-1);
                println!("GROUP	{}	{g}", f.file_stem().and_then(|x| x.to_str()).unwrap_or("?"));
                break;
            }
            let alive = engine::step(&mut st, &a);
            if money_out.is_some() {
                money.push((t, st.farms[seat].money, st.farms[1 - seat].money));
            }
            if !alive {
                break;
            }
        }
        let (us, them) = (st.farms[seat].money, st.farms[1 - seat].money);
        // KRL_END_STOCK=1: unsold stock at the end (shed + carried), per seat, valued at the final quotes (worth $0)
        if std::env::var_os("KRL_END_STOCK").is_some() {
            let id = f.file_stem().unwrap().to_string_lossy();
            for (who, s) in [("us", seat), ("them", 1 - seat)] {
                let mut parts = vec![];
                let mut val = 0.0;
                for (k, n) in st.private[s].shed.0.iter() {
                    let carried: i64 = st.private[s].inventories.iter().map(|m| m.get(k)).sum();
                    let tot = *n + carried;
                    if tot > 0 {
                        let px = kagg_engine::market::param(k).map(|p| kagg_engine::market::price(p, st.market.inventory.get(k) as f64) as f64).unwrap_or(0.0);
                        val += px * tot as f64;
                        parts.push(format!("{k}:{tot}"));
                    }
                }
                eprintln!("[endstock] {id} {who} value {val:.0} {}", parts.join(","));
            }
        }
        if let Some(rd) = record.as_ref() {
            // the game as played (our seat = the agent under test, the other = the guarded opponent): a tape
            // `branch` can label (DAgger); "opp_team" carries over from the source tape
            let stem = f.file_stem().unwrap().to_string_lossy().to_string();
            let opp_team = j.get("opp_team");
            let ot = if opp_team.is_null() { "null".to_string() } else { format!("\"{}\"", opp_team.str().replace('"', "")) };
            let body = format!("{{\"id\":\"{stem}__r\",\"seed\":{seed},\"seat\":{seat},\"rewards\":[{},{}],\"us\":\"cand\",\"opp_team\":{ot},\"actions\":[{}]}}",
                if seat == 0 { us } else { them }, if seat == 0 { them } else { us }, rec_acts.join(","));
            std::fs::write(std::path::Path::new(rd).join(format!("{stem}__r.json")), body).expect("record tape");
        }
        let (rus, rthem) = (rewards.get(seat).copied().unwrap_or(0.0), rewards.get(1 - seat).copied().unwrap_or(0.0));
        n += 1;
        exact += (us == rus && them == rthem) as usize;
        w_rep += (rus > rthem) as usize;
        w_sim += (us > them) as usize;
        let id = f.file_stem().unwrap().to_string_lossy();
        if let Some(w) = obs_out.as_mut() {
            use std::io::Write;
            for (s, d, v) in &vrows {
                let xs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                writeln!(w, "{id}	{s}	{d}	{}", xs.join("	")).unwrap();
            }
        }
        if let (Some(w), Some(b)) = (obs_out.as_mut(), me.as_ref()) {
            use std::io::Write;
            for (d, v) in &b.day_obs {
                let xs: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
                writeln!(w, "{id}	{seat}	{d}	{}", xs.join("	")).unwrap();
            }
        }
        if let Some(w) = money_out.as_mut() {
            use std::io::Write;
            for (t, a, b) in &money {
                writeln!(w, "{id}\t{t}\t{a}\t{b}").unwrap();
            }
        }
        if takeover > 0 {
            // extra column: turns before the takeover where our agent would have played exactly the recorded action
            println!("{id}\t{seat}\t{rus}\t{rthem}\t{us}\t{them}\t{agree}");
        } else {
            println!("{id}\t{seat}\t{rus}\t{rthem}\t{us}\t{them}");
        }
    }
    eprintln!("[tapeplay] {n} tapes; replay wins {w_rep}, sim wins {w_sim}; exact-bank reproductions {exact}/{n}");
}
