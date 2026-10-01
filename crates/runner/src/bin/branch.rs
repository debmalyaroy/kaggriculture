//! Price sale decisions on the exact engine, and compare the recorded player with our live agent.
//!
//!     branch --tapes DIR --out FILE [--k 40] [--other-seat] [--profiles P --pa 35 --group 35,35,36] [--base B] [--shell M]
//!
//! For each tape (seat = the player we study: a top-50 player, or our own agent in a recorded DAgger game):
//!   1. replays the recorded game exactly;
//!   2. runs our live agent in SHADOW on the same history (it sees the seat's observations every turn;
//!      its action is recorded, never applied), so each turn has "what we would have done here";
//!   3. picks up to --k (turn, item) sale decisions where the seat holds stock of a product: every turn
//!      where the seat or our shadow sells it, plus a random sample of holds (seeded by the tape id);
//!   4. for each decision plays three alternatives for that item at that turn -- hold (0), sell half
//!      (rounded up), sell all -- then both players continue with their recorded actions to the end,
//!      and records the seat's final margin for each.
//! Output (tab-separated, one row per decision):
//!   id seat step item their_cls our_cls stock f0..f19 (crate::shell features) m_hold m_part m_all m_real
//! The model trained on these (python/top50/train_shell.py) learns the best of the three, whoever
//! played it: where the top player's choice wins it is kept, where ours wins ours is kept.
use agent::act::Action;
use agent::shell::{class_of, qty_of, set_sell, ShellMem, NF, SHELL_ITEMS};
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;
use std::io::Write;

/// The seat's observation straight from the engine state (Obs::from_state: bit-identical to parsing the
/// JSON observation, ~25x cheaper; see ../docs/history/fast-tournaments-2026-09-26.md). KAGG_OBS_JSON=1 = the JSON path.
fn obs_of(st: &State, seat: usize) -> Result<agent::obs::Obs, String> {
    if std::env::var_os("KAGG_OBS_JSON").is_some() {
        agent::obs::Obs::parse(&seat_obs_json(st, seat))
    } else {
        Ok(agent::obs::Obs::from_state(st, seat))
    }
}


fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn finish(mut st: State, from: usize, a0: &[PlayerAction], a1: &[PlayerAction], first: [PlayerAction; 2]) -> State {
    let mut alive = engine::step(&mut st, &first);
    let mut t = from + 1;
    while alive {
        let pair = [a0.get(t).cloned().unwrap_or_default(), a1.get(t).cloned().unwrap_or_default()];
        alive = engine::step(&mut st, &pair);
        t += 1;
    }
    st
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let dir = get("--tapes").expect("--tapes DIR");
    let out = get("--out").expect("--out FILE");
    let other_seat = args.iter().any(|a| a == "--other-seat");
    let k_max: usize = get("--k").and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut base = agent::base::Base::load(&get("--base").unwrap_or_else(|| "configs/bases/v61.1".into())).expect("base");
    if let Some(p) = get("--profiles") {
        base.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("profiles");
    }
    if let Some(g) = get("--group") {
        let v: Vec<usize> = g.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        assert_eq!(v.len(), 3, "--group P_DIFF,P_PARTIAL,P_COPY");
        base.chain.group_ctl = Some(agent::layers::group::GroupCtl::new([v[0], v[1], v[2]], [0, 0, 0]));
    }
    if let Some(f) = get("--shell") {
        base.shell = Some(agent::shell::ShellCtl::new(std::sync::Arc::new(agent::shell::ShellModel::load(&f).expect("shell"))));
    }
    runner::apply_extras(&mut base, &args);
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).expect("out"));
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let (mut n_games, mut n_rows) = (0usize, 0usize);
    for f in files {
        let Ok(j) = json::parse(&std::fs::read_to_string(&f).unwrap()) else { continue };
        let id = f.file_stem().unwrap().to_string_lossy().to_string();
        let seed = j.get("seed").i64();
        // --other-seat: study the tape's opponent (band_tapes.py tapes store OUR seat; the top player is the other one)
        let seat = if other_seat { 1 - j.get("seat").i64() as usize } else { j.get("seat").i64() as usize };
        let acts = j.get("actions").arr();
        let rec = |s: usize| -> Vec<Action> { acts.iter().map(|p| p.arr().get(s).map(Action::from_json).unwrap_or_else(Action::pass)).collect() };
        let (r0, r1) = (rec(0), rec(1));
        let eng = |v: &[Action]| -> Vec<PlayerAction> { v.iter().map(runner::to_engine).collect() };
        let (e0, e1) = (eng(&r0), eng(&r1));
        let mine = if seat == 0 { &r0 } else { &r1 };
        // pass 1: exact replay with the shadow agent and the shell features; collect candidate decisions
        let mut st = State::new(seed);
        let mut shadow = base.fresh();
        let mut mem = ShellMem::default();
        let mut states: Vec<State> = Vec::with_capacity(720);
        let mut cands: Vec<(usize, usize, usize, usize, i64, [f32; NF], bool)> = vec![]; // t, item, their, our, stock, x, "sells"
        let mut t = 0usize;
        loop {
            states.push(st.clone());
            if let Ok(o) = obs_of(&st, seat) {
                if let Ok(v) = agent::view::View::new(o.clone()) {
                    let ours = shadow.act(o).0;
                    if v.step < agent::view::LAST_ACT_STEP {
                        if let Some(theirs) = mine.get(t) {
                            for (k, item) in SHELL_ITEMS.iter().enumerate() {
                                let stock = v.shed(item);
                                if stock <= 0 {
                                    continue;
                                }
                                let (tc, oc) = (class_of(theirs, item, stock), class_of(&ours, item, stock));
                                cands.push((t, k, tc, oc, stock, mem.features(&v, k), tc > 0 || oc > 0 || tc != oc));
                            }
                        }
                    }
                    mem.observe(&v);
                }
            }
            let pair = [e0.get(t).cloned().unwrap_or_default(), e1.get(t).cloned().unwrap_or_default()];
            let alive = engine::step(&mut st, &pair);
            t += 1;
            if !alive {
                break;
            }
        }
        let m_real = st.farms[seat].money - st.farms[1 - seat].money;
        // choose decisions: all "active" ones (someone sells / they differ) first, then holds, seeded shuffle
        let mut h = fnv(&id);
        let mut rnd = || {
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            h
        };
        let (mut act, mut hold): (Vec<_>, Vec<_>) = cands.into_iter().partition(|c| c.6);
        for v in [&mut act, &mut hold] {
            for i in (1..v.len()).rev() {
                let j = (rnd() % (i as u64 + 1)) as usize;
                v.swap(i, j);
            }
        }
        let n_act = act.len().min(k_max * 3 / 4);
        let mut pick: Vec<_> = act.into_iter().take(n_act).collect();
        let n_hold = k_max.saturating_sub(pick.len());
        pick.extend(hold.into_iter().take(n_hold));
        // pass 2: branch each decision
        for (t, k, tc, oc, stock, x, _) in pick {
            let item = SHELL_ITEMS[k];
            let mut m = [0f64; 3];
            for (c, mc) in m.iter_mut().enumerate() {
                let mut a = mine[t].clone();
                set_sell(&mut a, item, qty_of(c, stock));
                let pa = runner::to_engine(&a);
                let other = if seat == 0 { e1.get(t) } else { e0.get(t) }.cloned().unwrap_or_default();
                let first = if seat == 0 { [pa, other] } else { [other, pa] };
                let fin = finish(states[t].clone(), t, &e0, &e1, first);
                *mc = fin.farms[seat].money - fin.farms[1 - seat].money;
            }
            let xs = x.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\t");
            let _ = writeln!(w, "{id}\t{seat}\t{t}\t{item}\t{tc}\t{oc}\t{stock}\t{xs}\t{}\t{}\t{}\t{m_real}", m[0], m[1], m[2]);
            n_rows += 1;
        }
        n_games += 1;
    }
    eprintln!("[branch] {n_games} games, {n_rows} decisions -> {out}");
}
