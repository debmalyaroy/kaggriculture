//! All-Rust games: `kagg_engine` state + one `agent::base::Base` per seat.
//!
//! Each step builds the official per-seat observation (`obsjson::seat_obs_json`), the agent acts
//! on it, and the action is handed to `engine::step` positionally (empty market orders keep
//! their slot). Actions are solicited for steps 0..=718 and the step-719 state is scored, as the
//! official runner does.
use agent::act::{Action, Cmd, Tok};
use agent::base::Base;
use agent::obs::Obs;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;

fn tok_str(t: &Tok) -> String {
    match t {
        Tok::S(s) => s.to_string(),
        Tok::I(i) => i.to_string(),
        Tok::F(f) => f.to_string(),
        Tok::Null => String::new(),
    }
}

/// Our action -> the engine's positional player action.
pub fn to_engine(a: &Action) -> PlayerAction {
    PlayerAction {
        farmer: agent::sim::unit(&a.farmer),
        hands: a.hands.iter().map(agent::sim::unit).collect(),
        market: a.market.iter().map(|o: &Cmd| o.0.iter().map(tok_str).collect()).collect(),
    }
}

#[derive(Clone, Debug, Default)]
pub struct GameResult {
    pub seed: i64,
    pub banks: [f64; 2],
    /// Worst agent turn per seat, microseconds.
    pub max_us: [f64; 2],
    /// Per-step agent time per seat, microseconds (index = step).
    pub step_us: Vec<[f32; 2]>,
    /// Slow turns (with `play_prof`): (step, seat, total us, [pre, core, post] us, top-5 (stage, us)).
    pub slow: Vec<(usize, usize, f32, [f32; 3], Vec<(usize, f32)>)>,
    /// The REALIZED world: first two unlocked shops after day 6 ("A|B").
    pub world: String,
    /// Where the time went (microseconds, whole game): building the observation JSON, parsing it,
    /// the agents' act (excluding the parse), the engine step; per seat, every chain stage and the
    /// pre / core (chassis + terminal) / post phases.
    pub prof: Prof,
}

#[derive(Clone, Debug, Default)]
pub struct Prof {
    pub json_us: f64,
    pub parse_us: f64,
    pub act_us: f64,
    pub engine_us: f64,
    pub stage_us: [Vec<f64>; 2],
    pub phase_us: [[f64; 3]; 2],
}

/// Play one full game; `agents[s]` plays seat `s`.
pub fn play(seed: i64, agents: &mut [Base; 2]) -> GameResult {
    play_prof(seed, agents, None)
}

/// `play`, also recording every turn slower than `slow_us` with its phase/stage breakdown.
pub fn play_prof(seed: i64, agents: &mut [Base; 2], slow_us: Option<f32>) -> GameResult {
    play_with(seed, agents, slow_us, &|_, _, _, _| {})
}

/// `play_prof` with a hook that may rewrite a seat's action after its agent chose it
/// (`hook(step, seat, &state, &mut action)`): the oracle / counterfactual experiments.
pub fn play_with(seed: i64, agents: &mut [Base; 2], slow_us: Option<f32>, hook: &dyn Fn(i64, usize, &State, &mut Action)) -> GameResult {
    let mut slow = vec![];
    let mut st = State::new(seed);
    let mut max_us = [0f64; 2];
    let mut step_us: Vec<[f32; 2]> = Vec::with_capacity(720);
    let mut world = String::new();
    let mut prof = Prof::default();
    let obs_json = std::env::var_os("KAGG_OBS_JSON").is_some();
    let obs_check = std::env::var_os("KAGG_OBS_CHECK").is_some();
    loop {
        let mut row = [0f32; 2];
        let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
        for (s, ag) in agents.iter_mut().enumerate() {
            // the observation straight from the state (KAGG_OBS_JSON: via JSON as before;
            // KAGG_OBS_CHECK: build both and panic on any difference)
            let tj = std::time::Instant::now();
            let text = if obs_json || obs_check { seat_obs_json(&st, s) } else { String::new() };
            prof.json_us += tj.elapsed().as_secs_f64() * 1e6;
            let t0 = std::time::Instant::now();
            let parsed = if obs_json { Obs::parse(&text) } else { Ok(Obs::from_state(&st, s)) };
            if obs_check {
                let a = format!("{:?}", Obs::parse(&text).ok());
                let b = format!("{:?}", Some(Obs::from_state(&st, s)));
                assert_eq!(a, b, "obs mismatch seed {seed} step {} seat {s}", st.step);
            }
            let tp = t0.elapsed().as_secs_f64() * 1e6;
            prof.parse_us += tp;
            let mut a = match parsed {
                Ok(o) => ag.act(o).0,
                Err(_) => Action::pass(),
            };
            hook(st.step, s, &st, &mut a);
            let us = t0.elapsed().as_secs_f64() * 1e6;
            prof.act_us += us - tp;
            if prof.stage_us[s].len() < ag.chain.turn_us.len() {
                prof.stage_us[s].resize(ag.chain.turn_us.len(), 0.0);
            }
            for (i, x) in ag.chain.turn_us.iter().enumerate() {
                prof.stage_us[s][i] += *x as f64;
            }
            for i in 0..3 {
                prof.phase_us[s][i] += ag.phase_us[i] as f64;
            }
            max_us[s] = max_us[s].max(us);
            row[s] = us as f32;
            if slow_us.is_some_and(|t| us as f32 > t) {
                let mut top: Vec<(usize, f32)> = ag.chain.turn_us.iter().copied().enumerate().filter(|x| x.1 > 0.0).collect();
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                top.truncate(5);
                slow.push((step_us.len(), s, us as f32, ag.phase_us, top));
            }
            acts.push(to_engine(&a));
        }
        step_us.push(row);
        let acts: [PlayerAction; 2] = [acts.remove(0), acts.remove(0)];
        let te = std::time::Instant::now();
        let alive = engine::step(&mut st, &acts);
        prof.engine_us += te.elapsed().as_secs_f64() * 1e6;
        if !alive {
            break;
        }
        if st.step == 145 {
            world = st.town.unlocked_shops.iter().take(2).cloned().collect::<Vec<_>>().join("|");
        }
    }
    GameResult { seed, banks: [st.farms[0].money, st.farms[1].money], max_us, step_us, slow, world, prof }
}
