//! A base agent = routes + router + chassis settings, loaded from `configs/bases/<name>/`
//! (`routes.json`, `router.json`). Hot-swappable: `Agent::swap_base` replaces it between turns.
use crate::act::{Action, Cmd};
use crate::chassis::{Chassis, Diagnostics, Route, Settings};
use crate::router::ShopRouter;
use crate::obs::Obs;
use crate::layers;
use crate::view::*;
use kagg_engine::json::{self, Json};

pub struct Base {
    pub name: String,
    pub chassis: Chassis,
    /// The `_SHOP` terminal rescue (agent lines 995-1009) — layer 1 of the v61.1 chain.
    pub shop_rescue: bool,
    /// Last chain stage to run (index into [`CUTS`]).
    pub cut: usize,
    pub chain: layers::Chain,
    pub terminal: crate::terminal::Terminal,
    /// Route 0 as loaded (before any PIPE install).
    pub pristine_route0: Option<crate::chassis::Route>,
    /// Last turn's (pre, terminal+chassis, post) time, us.
    pub phase_us: [f32; 3],
    pub disguise: crate::disguise::Disguise,
}

impl Base {
    pub fn from_json(name: &str, routes: &Json, router: &Json) -> Result<Base, String> {
        let parsed: Vec<(String, Vec<Action>)> =
            routes.obj().iter().map(|(k, tape)| (k.clone(), tape.arr().iter().map(Action::from_json).collect())).collect();
        Base::from_parts(name, parsed, router)
    }

    pub fn from_parts(name: &str, routes: Vec<(String, Vec<Action>)>, router: &Json) -> Result<Base, String> {
        let mut rs: Vec<Route> = Vec::new();
        for (k, mut t) in routes {
            let id: i64 = k.parse().map_err(|_| format!("route id {k:?}"))?;
            // _R42_OPENING: every route's step-0 market is replaced by the opening orders.
            let op = router.get("opening_step0_market");
            if op.is_arr() && !t.is_empty() {
                t[0].market = Action::from_json(&Json::Obj(vec![("market".into(), op.clone())])).market;
            }
            rs.push(Route::new(id, t));
        }
        if rs.is_empty() {
            return Err("no routes".into());
        }
        let mut shop_router = ShopRouter::from_json(router);
        if router.get("mode").str() == "trie" {
            shop_router.trie = Some(std::sync::Arc::new(crate::trie::Trie::from_json(router.get("trie"), &rs).ok_or("trie: bad or empty \"trie\" block")?));
        }
        let chassis = Chassis {
            routes: rs,
            router: shop_router,
            cfg: Settings::from_json(router.get("settings")),
            opponent_plan: None,
            players: vec![],
            diagnostics: Diagnostics::default(),
        };
        let shop_rescue = router.get("shop_rescue").is_null() || router.get("shop_rescue").bool();
        let pristine_route0 = chassis.route_idx(0).map(|i| chassis.routes[i].clone());
        // router.json "cut": a stage name (layers::CUTS; "chassis" = tapes only, "full" = every layer)
        let cut = match router.get("cut").str() {
            "" => CUTS.len() - 1,
            c => layers::cut_index(c).ok_or(format!("unknown cut {c:?}"))?,
        };
        Ok(Base { name: name.to_string(), chassis, shop_rescue, cut, chain: Default::default(), terminal: Default::default(), pristine_route0, phase_us: [0.0; 3], disguise: Default::default() })
    }

    /// A fresh copy (same routes/router/settings/cut, new per-game state) — cheap way to get
    /// many independent agents from one loaded base.
    pub fn fresh(&self) -> Base {
        let ch = &self.chassis;
        let mut routes = ch.routes.clone();
        // PIPE mutates route 0 in place; a fresh base restarts from the pristine tape.
        if let Some(p) = self.pristine_route0.as_ref() {
            if let Some(i) = ch.route_idx(0) {
                routes[i] = p.clone();
            }
        }
        Base {
            name: self.name.clone(),
            chassis: Chassis {
                routes,
                router: ch.router.clone(),
                cfg: ch.cfg.clone(),
                opponent_plan: ch.opponent_plan.clone(),
                players: vec![],
                diagnostics: Diagnostics::default(),
            },
            shop_rescue: self.shop_rescue,
            cut: self.cut,
            chain: layers::Chain {
                profiles: self.chain.profiles.clone(),
                schedule: self.chain.schedule.clone(),
                clone_profile: self.chain.clone_profile,
                clone_strict: self.chain.clone_strict,
                afr_profile: self.chain.afr_profile,
                afr_trigger: self.chain.afr_trigger,
                group_ctl: self.chain.group_ctl.as_ref().map(|g| layers::group::GroupCtl::new(g.profiles, g.jitter).with_endgame(g.endgame, g.jitter_end).with_mirror(g.mirror_tol)),
                dispatch: self.chain.dispatch.fresh(),
                ..Default::default()
            },
            terminal: Default::default(),
            pristine_route0: self.pristine_route0.clone(),
            phase_us: [0.0; 3],
            disguise: Default::default(),
        }
    }

    /// Install a profile table (`configs/profiles/*.json`) and optionally a fixed profile for the
    /// whole game (applied at every day boundary; day 0 hour 0 is always profile 0).
    pub fn set_profiles(&mut self, path: &str, fixed: Option<usize>) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let j = json::parse(&text)?;
        self.chain.profiles = layers::knobs::load_profiles(&j)?;
        // sync list from the pristine tapes (route 0 as loaded, before any PIPE rewrite)
        let mut ch0 = Chassis { routes: self.chassis.routes.clone(), router: self.chassis.router.clone(), cfg: self.chassis.cfg.clone(), opponent_plan: None, players: vec![], diagnostics: Diagnostics::default() };
        if let (Some(p), Some(i)) = (self.pristine_route0.as_ref(), ch0.route_idx(0)) {
            ch0.routes[i] = p.clone();
        }
        self.chain.dispatch = layers::dispatch::Dispatch::with_sets(layers::dispatch::load_sets(j.get("dispatch"))?, &ch0);
        if let Some(bad) = self.chain.profiles.iter().find(|(_, k)| k.dispatch_set >= self.chain.dispatch.sets.len() as i64) {
            return Err(format!("profile {:?}: dispatch_set {} out of range", bad.0, bad.1.dispatch_set));
        }
        if let Some(k) = fixed {
            if k >= self.chain.profiles.len() {
                return Err(format!("profile {k} out of range"));
            }
            self.chain.schedule = Some(vec![k; 30]);
        }
        Ok(())
    }

    pub fn load(dir: &str) -> Result<Base, String> {
        let rd = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).map_err(|e| format!("{dir}/{f}: {e}"));
        let routes = crate::obs::parse_routes(&rd("routes.json")?)?;
        let router = json::parse(&rd("router.json")?)?;
        let name = std::path::Path::new(dir).file_name().and_then(|s| s.to_str()).unwrap_or(dir).to_string();
        Base::from_parts(&name, routes, &router)
    }

    /// Run the chain through stage `self.cut` (see [`layers::CUTS`]): pre phases outermost
    /// first, the chassis (+ `_SHOP` rescue), then post phases innermost first.
    pub fn act(&mut self, obs: Obs) -> (Action, Option<View>) {
        let step = obs.step();
        let v = match View::new(obs) {
            Ok(v) => v,
            Err(obs) => {
                // Python: Chassis.act raises -> make_agent falls back to the tape action.
                let (a, _) = self.chassis.entry(obs);
                if self.shop_rescue && step >= LAST_ACT_STEP {
                    self.chassis.diagnostics.terminal_rescue_errors += 1;
                }
                return (a, None);
            }
        };
        let t0 = std::time::Instant::now();
        self.chain.pre(self.cut, &v, &mut self.chassis);
        let t1 = std::time::Instant::now();
        let shop = self.shop_rescue;
        let pre = move |ch: &mut Chassis, v: &View| -> Action {
            let a = ch.act(v);
            if shop && v.step >= LAST_ACT_STEP {
                terminal_rescue(ch, v)
            } else {
                a
            }
        };
        let action = if self.cut >= 1 {
            let p = v.obs.player;
            let committed = self.chain.v219.info(p).committed || self.chain.v233.committed(p);
            let obs = v.obs.clone();
            let k = &self.chain.knobs;
            self.terminal.cfg = Some(crate::terminal::TermCfg {
                on: k.term_on,
                start: k.term_start.clamp(crate::terminal::FINAL - 23, crate::terminal::FINAL),
                sims: k.term_sims.max(0) as usize,
                passes: k.term_passes.max(1) as usize,
                props: k.term_props.max(1) as usize,
            });
            self.terminal.act(&mut self.chassis, &v, crate::terminal::Ctx { committed_project: committed, obs: &obs }, &pre)
        } else {
            pre(&mut self.chassis, &v)
        };
        let t2 = std::time::Instant::now();
        let mut action = self.chain.post(self.cut, action, &v, &mut self.chassis);
        // disguise (after every layer; appended orders only, see crate::disguise)
        crate::disguise::Disguise::money(&mut action, &v, self.chain.knobs.disguise_money);
        if self.chain.knobs.disguise_stream {
            self.disguise.stream(&mut action, &v);
        }
        self.phase_us = [(t1 - t0).as_secs_f32() * 1e6, (t2 - t1).as_secs_f32() * 1e6, t2.elapsed().as_secs_f32() * 1e6];
        (action, Some(v))
    }
}

pub use layers::{cut_index, CUTS};
/// `_SHOP` rescue at step >= 718: every unit next to the shed with stock DROPs, the rest
/// PASS, and the market sells the whole projected shed, largest value first.
pub fn terminal_rescue(ch: &Chassis, v: &View) -> Action {
    let units: Vec<Cmd> = v
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| if shed_adjacent(*p, v.board) && !v.inv(i).is_empty() { Cmd::new("DROP") } else { Cmd::pass() })
        .collect();
    let mut a = Action { farmer: units[0].clone(), hands: units[1..].to_vec(), market: vec![] };
    let projected = ch.projected_shed(&a, v);
    let mut m: Vec<Cmd> = PRODUCTS
        .iter()
        .filter(|p| crate::obs::qget(&projected, p) > 0)
        .map(|p| Cmd::order("SELL", p, crate::obs::qget(&projected, p)))
        .collect();
    m.sort_by_key(|o| -v.price(o.s(1)) * o.n(2));
    a.market = m;
    a
}
