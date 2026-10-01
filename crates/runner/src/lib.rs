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
/// An agent action as the replay JSON object {"farmer": [...], "hands": [[...]], "market": [[...]]}
/// (tapes written by `tapeplay --record` read back with Action::from_json).
pub fn action_json(a: &Action) -> String {
    fn tok(t: &agent::act::Tok) -> String {
        match t {
            agent::act::Tok::S(s) => format!("\"{s}\""),
            agent::act::Tok::I(i) => i.to_string(),
            agent::act::Tok::F(f) => f.to_string(),
            agent::act::Tok::Null => "null".into(),
        }
    }
    let cmd = |c: &Cmd| format!("[{}]", c.0.iter().map(tok).collect::<Vec<_>>().join(","));
    let list = |v: &[Cmd]| format!("[{}]", v.iter().map(cmd).collect::<Vec<_>>().join(","));
    format!("{{\"farmer\":{},\"hands\":{},\"market\":{}}}", cmd(&a.farmer), list(&a.hands), list(&a.market))
}

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
    /// Unlocked shops at the end, in unlock order (the realized world = the first two).
    pub shops: Vec<String>,
}

/// `--rshell F --chain-off S --knob-over F --endg F [--endg-force K]` from an argument list, applied to `b`.
pub fn apply_extras(b: &mut Base, args: &[String]) {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let force = get("--endg-force").and_then(|s| s.parse().ok());
    // --cut NAME: run the chain only through stage NAME (layers::CUTS); `--cut chassis` = the chassis alone, no layers
    if let Some(c) = get("--cut") {
        b.cut = agent::base::cut_index(&c).unwrap_or_else(|| panic!("unknown cut {c}"));
    }
    // --tpp NET.json: the top-player policy plays every turn (crates/agent/src/tpp.rs)
    // --group-knobs FILE [--group-knobs-for 0,1]: knob overlay only against these rival groups (0 DIFFERENT, 1 PARTIAL, 2 COPY)
    if let Some(f) = get("--group-knobs") {
        let j = kagg_engine::json::parse(&std::fs::read_to_string(&f).expect("group-knobs file")).expect("group-knobs json");
        let gs: Vec<usize> = get("--group-knobs-for").unwrap_or_else(|| "0,1".into()).split(',').filter_map(|x| x.trim().parse().ok()).collect();
        b.chain.group_over = Some((gs, j));
        b.chain.group_over_lineage = args.iter().any(|a| a == "--group-knobs-lineage");
    }
    if let Some(f) = get("--preempt") {
        b.preempt = Some(agent::preempt::Preempt::new(std::sync::Arc::new(agent::preempt::PreCfg::load(&f).expect("preempt"))));
    }
    if let Some(f) = get("--dispatch") {
        b.dispatch = Some(agent::dispatch::Dispatch::new(std::sync::Arc::new(agent::dispatch::DispatchCfg::load(&f).expect("dispatch"))));
    }
    if let Some(f) = get("--gt") {
        b.gtl = Some(agent::gt::GtLayer::new(std::sync::Arc::new(agent::gt::GtCfg::load(&f).expect("gt"))));
    }
    if let Some(f) = get("--prem-sell") {
        b.prem = Some(std::sync::Arc::new(agent::base::PremSell::load(&f).expect("prem-sell")));
    }
    if let Some(fb) = get("--force-buy") {
        b.force_buys = fb.split(',').filter_map(|x| { let (a, q) = x.split_once(':')?; Some((a.trim().parse().ok()?, q.trim().parse().ok()?)) }).collect();
    }
    // --tpp-mkt NET.json [--tpp-ops SELL,BUY_PRODUCT] [--tpp-from STEP] [--tpp-to STEP]: the hybrid
    if let Some(f) = get("--tpp-mkt") {
        b.tpp_mkt = Some(agent::base::TppMkt {
            net: std::sync::Arc::new(agent::tpp::Net::load(&f).expect("tpp net")),
            from: get("--tpp-from").and_then(|s| s.parse().ok()).unwrap_or(0),
            to: get("--tpp-to").and_then(|s| s.parse().ok()).unwrap_or(719),
            ops: get("--tpp-ops").unwrap_or_else(|| "SELL".into()).split(',').map(|s| s.to_string()).collect(),
            add: args.iter().any(|a| a == "--tpp-add"),
        });
    }
    if let Some(f) = get("--tpp") {
        b.tpp = Some(std::sync::Arc::new(agent::tpp::Net::load(&f).expect("tpp net")));
    }
    b.apply_extras2(get("--rshell").as_deref(), get("--chain-off").as_deref(), get("--knob-over").as_deref(), get("--endg").as_deref(), force)
        .expect("rshell / chain-off / knob-over / endg");
}

/// Configure a base from a flag list (selfplay --a-args / --b-args): --profiles P [--pa K] [--policy W]
/// [--shield F] [--group D,P,C] [--shell F] [--disguise] + the apply_extras flags.
pub fn configure(b: &mut Base, args: &[String]) {
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    if let Some(p) = get("--profiles") {
        b.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("profiles");
    }
    if let Some(w) = get("--policy") {
        let net = policy::Net::load(&w).expect("policy weights");
        let n = b.chain.profiles.len();
        assert_eq!(net.n_act, n, "policy action count != --profiles table");
        b.obs_track = Some(dayobs::Builder::new());
        b.policy = Some(agent::base::PolicyCtl::new(std::sync::Arc::new(net), vec![true; n], None));
    }
    if let Some(f) = get("--shield") {
        let j = kagg_engine::json::parse(&std::fs::read_to_string(&f).expect("shield file")).expect("shield json");
        let (sh, tol) = agent::layers::group::Shield::from_json(&j).expect("shield");
        b.chain.group_ctl = Some(agent::layers::group::GroupCtl::shield_only(sh, tol));
    }
    if let Some(g) = get("--group") {
        let v: Vec<usize> = g.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        assert_eq!(v.len(), 3, "--group P_DIFF,P_PARTIAL,P_COPY");
        b.chain.group_ctl = Some(agent::layers::group::GroupCtl::new([v[0], v[1], v[2]], [0, 0, 0]));
    }
    if let Some(f) = get("--shell") {
        let m = agent::shell::ShellModel::load(&f).expect("shell model");
        b.shell = Some(agent::shell::ShellCtl::new(std::sync::Arc::new(m)));
    }
    if args.iter().any(|a| a == "--disguise") {
        b.disguise = Some(agent::disguise::Disguise::default());
    }
    // --route-table FILE: per-world route overrides {"SHOP1|SHOP2": route id} (route library screening)
    if let Some(f) = get("--route-table") {
        b.chassis.router.override_worlds(&kagg_engine::json::parse(&std::fs::read_to_string(&f).expect("route table")).expect("route table json"));
    }
    apply_extras(b, args);
}

/// Play one full game; `agents[s]` plays seat `s`.
pub fn play(seed: i64, agents: &mut [Base; 2]) -> GameResult {
    play_prof(seed, agents, None)
}

/// `play`, also recording every turn slower than `slow_us` with its phase/stage breakdown.
pub fn play_prof(seed: i64, agents: &mut [Base; 2], slow_us: Option<f32>) -> GameResult {
    let mut slow = vec![];
    let mut st = State::new(seed);
    let mut max_us = [0f64; 2];
    let mut step_us: Vec<[f32; 2]> = Vec::with_capacity(720);
    loop {
        let mut row = [0f32; 2];
        let mut acts: Vec<PlayerAction> = Vec::with_capacity(2);
        for (s, ag) in agents.iter_mut().enumerate() {
            let text = seat_obs_json(&st, s);
            let t0 = std::time::Instant::now();
            let a = match Obs::parse(&text) {
                Ok(o) => ag.act(o).0,
                Err(_) => Action::pass(),
            };
            let us = t0.elapsed().as_secs_f64() * 1e6;
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
        if !engine::step(&mut st, &acts) {
            break;
        }
    }
    GameResult { seed, banks: [st.farms[0].money, st.farms[1].money], max_us, step_us, slow, shops: st.town.unlocked_shops.clone() }
}

/// A real ladder player's recorded game (python/band_tapes.py / slim_to_tape.py format).
#[derive(Clone)]
pub struct Tape {
    pub id: String,
    pub seed: i64,
    /// OUR seat in that game (the recorded opponent sat in the other one)
    pub seat: usize,
    pub band: String,
    pub stream: std::sync::Arc<Vec<Action>>,
}

/// Load every tape under `dir` (recursively), keeping the opponent seat's stream.
pub fn load_tapes(dir: &str) -> Vec<Tape> {
    fn walk(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let q = e.path();
                if q.is_dir() {
                    walk(&q, out);
                } else if q.extension().is_some_and(|x| x == "json") {
                    out.push(q);
                }
            }
        }
    }
    let mut files = vec![];
    // several pools: "DIR[,DIR...]" (ppo.py --tapes-shards: the leaders pool + one shard of every top-player game)
    for d in dir.split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    files
        .iter()
        .filter_map(|f| {
            let j = kagg_engine::json::parse(&std::fs::read_to_string(f).ok()?).ok()?;
            let seat = j.get("seat").i64() as usize;
            let stream: Vec<Action> = j.get("actions").arr().iter().map(|p| p.arr().get(1 - seat).map(Action::from_json).unwrap_or_else(Action::pass)).collect();
            Some(Tape { id: j.get("id").str().to_string(), seed: j.get("seed").i64(), seat, band: j.get("band").str().to_string(), stream: std::sync::Arc::new(stream) })
        })
        .collect()
}

/// The router for a guarded tape: v61.1's chassis settings reduced to the guards that keep a recorded
/// stream legal against a different us (hand_align, weed_repair, clamp_sells: 60/60 real games exact
/// when nothing is off; budget_guard broke 41/60), no route switching, no opening override.
/// `KRL_TAPE_GUARDS=k1,k2` overrides the kept set.
pub fn guard_router(base_dir: &str) -> kagg_engine::json::Json {
    use kagg_engine::json::Json;
    let txt = std::fs::read_to_string(format!("{base_dir}/router.json")).expect("router.json");
    let Json::Obj(kv) = kagg_engine::json::parse(&txt).expect("router json") else { panic!("router.json is not an object") };
    let mut kv: Vec<(String, Json)> = kv.into_iter().filter(|(k, _)| k != "opening_step0_market" && k != "select_step" && k != "endgame_step").collect();
    let keep = std::env::var("KRL_TAPE_GUARDS").unwrap_or_else(|_| "hand_align,weed_repair,clamp_sells".into());
    let all = ["hand_align", "weed_repair", "sell_lead", "front_run", "budget_guard", "room_guard", "clamp_sells", "dead_stock", "terminal_liquidation", "r36", "racepx"];
    let mut st: Vec<(String, Json)> = match kv.iter().find(|(k, _)| k == "settings") { Some((_, Json::Obj(o))) => o.clone(), _ => vec![] };
    st.retain(|(k, _)| !all.contains(&k.as_str()));
    for g in all {
        st.push((g.into(), Json::Bool(keep.split(',').any(|x| x.trim() == g))));
    }
    kv.retain(|(k, _)| k != "settings" && k != "shop_rescue");
    kv.push(("settings".into(), Json::Obj(st)));
    kv.push(("shop_rescue".into(), Json::Bool(false)));
    kv.push(("select_step".into(), Json::Num(1e9)));
    kv.push(("endgame_step".into(), Json::Num(1e9)));
    Json::Obj(kv)
}

/// The opponent for a tape: its stream as the only route of a chassis (cut `chassis`: no layers).
pub fn guarded_tape_base(tape: &Tape, router: &kagg_engine::json::Json) -> Base {
    let mut b = Base::from_parts("tape", vec![("0".into(), (*tape.stream).clone())], router).expect("guarded tape base");
    b.cut = agent::base::cut_index("chassis").expect("cut chassis");
    b
}
