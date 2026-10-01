//! Replay a real ladder game's WORLD (engine seed = replay `info.seed`) against the opponent's
//! recorded actions (open-loop tape), with either our recorded actions (`--verify`: must reproduce
//! the replay's banks exactly) or a Rust agent in our seat (counterfactual: "what would X have
//! banked against what that opponent actually did?").
//!
//!     tapeplay --tapes DIR --verify
//!     tapeplay --tapes DIR --profiles configs/profiles/v2.json --pa 13 [--base configs/bases/v61.1]
//!
//! Tapes come from python/replay_to_tape.py. One line per tape:
//! `id<TAB>seat<TAB>replay_us<TAB>replay_them<TAB>sim_us<TAB>sim_them`.
//! Caveat: the opponent is open-loop — a reactive opponent would have reacted to a different us.
use agent::act::Action;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::{self, Json};
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;

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
    let other = args.iter().any(|a| a == "--other");
    let mut base = if verify { None } else { Some(agent::base::Base::load(&get("--base").unwrap_or_else(|| "configs/bases/v61.1".into())).expect("base")) };
    if let (Some(b), Some(p)) = (base.as_mut(), get("--profiles")) {
        b.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("profiles");
        b.chain.clone_profile = get("--ca").and_then(|s| s.parse().ok());
        b.chain.afr_profile = get("--fa").and_then(|s| s.parse().ok());
        b.chain.afr_trigger = get("--ftrig").and_then(|s| s.parse().ok()).unwrap_or(1);
        // --group P,P,P [--endgame E,E,E] [--jitter J,J,J] [--jitter-end J,J,J]: the bandit group layer
        let tri = |k: &str| -> Option<Vec<i64>> {
            let v: Vec<i64> = get(k)?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (v.len() == 3).then_some(v)
        };
        if let Some(p) = tri("--group") {
            let j = tri("--jitter").unwrap_or(vec![0, 0, 0]);
            let e = tri("--endgame").unwrap_or(vec![-1, -1, -1]);
            let je = tri("--jitter-end").unwrap_or(j.clone());
            b.chain.group_ctl = Some(
                agent::layers::group::GroupCtl::new([p[0] as usize, p[1] as usize, p[2] as usize], [j[0], j[1], j[2]])
                    .with_endgame([0, 1, 2].map(|i| (e[i] >= 0).then_some(e[i] as usize)), [je[0], je[1], je[2]]),
            );
            if let Some(g) = b.chain.group_ctl.as_mut() {
                g.mirror_tol = get("--mirror-tol").and_then(|s| s.parse().ok());
            }
        }
        // --sched a,b,...: explicit per-day profile ids (day d uses entry d; the last repeats)
        if let Some(sc) = get("--sched") {
            let mut v: Vec<usize> = sc.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            while !v.is_empty() && v.len() < 30 {
                v.push(*v.last().unwrap());
            }
            b.chain.schedule = Some(v);
        }
    }
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let (mut n, mut exact, mut w_rep, mut w_sim) = (0, 0, 0, 0);
    for f in files {
        let j = json::parse(&std::fs::read_to_string(&f).unwrap()).expect("tape json");
        let seed = j.get("seed").i64();
        // --other: our agent takes the OTHER seat (e.g. a team base replaying its own player's games)
        let seat = if other { 1 - j.get("seat").i64() as usize } else { j.get("seat").i64() as usize };
        let rewards = j.get("rewards").arr().iter().map(|x| x.f64()).collect::<Vec<_>>();
        let acts = j.get("actions").arr();
        let mut st = State::new(seed);
        let mut me = base.as_ref().map(|b| b.fresh());
        let mut t = 0usize;
        loop {
            let pair = acts.get(t).map(|p| p.arr()).unwrap_or(&[]);
            let tape = |s: usize| pair.get(s).map(act_of).unwrap_or_else(|| act_of(&Json::Null));
            let ours = match me.as_mut() {
                Some(b) => {
                    let text = seat_obs_json(&st, seat);
                    match agent::obs::Obs::parse(&text) {
                        Ok(o) => runner::to_engine(&b.act(o).0),
                        Err(_) => runner::to_engine(&Action::pass()),
                    }
                }
                None => tape(seat),
            };
            let theirs = tape(1 - seat);
            let a: [PlayerAction; 2] = if seat == 0 { [ours, theirs] } else { [theirs, ours] };
            t += 1;
            if !engine::step(&mut st, &a) {
                break;
            }
        }
        let (us, them) = (st.farms[seat].money, st.farms[1 - seat].money);
        let (rus, rthem) = (rewards.get(seat).copied().unwrap_or(0.0), rewards.get(1 - seat).copied().unwrap_or(0.0));
        n += 1;
        exact += (us == rus && them == rthem) as usize;
        w_rep += (rus > rthem) as usize;
        w_sim += (us > them) as usize;
        let id = f.file_stem().unwrap().to_string_lossy();
        println!("{id}\t{seat}\t{rus}\t{rthem}\t{us}\t{them}");
    }
    eprintln!("[tapeplay] {n} tapes; replay wins {w_rep}, sim wins {w_sim}; exact-bank reproductions {exact}/{n}");
}
