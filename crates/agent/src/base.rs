//! A base agent = routes + router + chassis settings, loaded from `configs/bases/<name>/`
//! (`routes.json`, `router.json`). Hot-swappable: `Agent::swap_base` replaces it between turns.
use crate::act::{Action, Cmd};
use crate::chassis::{Chassis, Diagnostics, Route, Settings};
use crate::router::ShopRouter;
use crate::obs::Obs;
use crate::layers;
use crate::view::*;
use kagg_engine::json::{self, Json};

#[derive(Clone)]
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
    /// Macro observation builder (crates/dayobs); None = not tracked (the v61.x submissions).
    pub obs_track: Option<dayobs::Builder>,
    /// (day, vector) at each decision step while tracking.
    pub day_obs: Vec<(usize, [f32; dayobs::N])>,
    /// Macro policy driving the day-boundary profile choice (requires `obs_track`).
    pub policy: Option<PolicyCtl>,
    /// Learned sales shell (crate::shell), applied after the chain; None = off.
    pub shell: Option<crate::shell::ShellCtl>,
    /// Reactive shell v2 (crate::rshell), applied after the chain and before the v1 shell; None = off.
    pub rshell: Option<crate::rshell::RShellCtl>,
    /// Stream disguise (crate::disguise): no-op orders that change our action stream, not the game.
    pub disguise: Option<crate::disguise::Disguise>,
    /// Learned endgame controller (crate::endg): patches the endgame knobs from its decision step on; None = off.
    pub endg: Option<crate::endg::EndgCtl>,
    /// Layer activity log (KRL_LAYER_LOG = per-game summary file, KRL_LAYER_TRACE = per-step file; default off).
    pub lay: LayerLog,
    /// Top-player policy (crate::tpp): when set, the cloned per-turn network plays every turn; None = off.
    pub tpp: Option<std::sync::Arc<crate::tpp::Net>>,
    /// Hybrid (operator 28 Sep): the chassis keeps the labour, the top-player policy sets the market orders of
    /// `ops` (e.g. SELL / BUY_PRODUCT) from step `from` on; the chassis keeps its other orders. None = off.
    pub tpp_mkt: Option<TppMkt>,
    /// `--force-buy STEP:Q,...` (28 Sep, from the exact counterfactual search on v63.7's real games): append
    /// `BUY_PRODUCT WHEAT Q` at exactly these steps when a market slot is free. Empty = off.
    pub force_buys: Vec<(i64, i64)>,
    /// Premium seller (28 Sep; `--prem-sell F`): right after the town's drain ticks, sell each product only down
    /// to a price floor on the engine's own curve, holding the rest for the next tick. None = off.
    pub prem: Option<std::sync::Arc<PremSell>>,
    /// Pre-emption of the rival's forecast sales (crate::preempt; `--preempt F`). None = off.
    pub preempt: Option<crate::preempt::Preempt>,
    /// Opponent State Tracker (crate::gt, v63.12 G3): the rival's unsold stock, updated once per turn, read by the
    /// market layers that opt in (preempt `stock_src: "gt"`, rshell tranching).
    pub gt: crate::gt::RivalTracker,
    /// Game-theoretic liquidation layer (crate::gt::GtLayer, `--gt F`, v63.12 G4-G7). None = off.
    pub gtl: Option<crate::gt::GtLayer>,
    /// Specialist dispatcher (crate::dispatch, `--dispatch F`, v63.12 A1/A3). None = off.
    pub dispatch: Option<crate::dispatch::Dispatch>,
    /// SALE manager: demand-aware batching + small-batch front-running (crate::managers::sale). None = off.
    pub sale: Option<crate::managers::sale::Sale>,
    /// MARKET GUARD manager's final check (crate::managers::market_guard::FinalCheck). None = off.
    pub final_check: Option<crate::managers::market_guard::FinalCheck>,
    /// Cash floor (1 Oct live losses 116094939 / 116107754 / 116055277): while our money is below `.0` and the step
    /// is before `.1`, the sale shells (rshell, shell, sale manager) may not cut a route SELL -- holding early wheat
    /// with no cash left made the next day's hires fail, the crops die and the farm never recover. None = off.
    pub cash_floor: Option<(f64, i64)>,
}

/// `--prem-sell FILE`: {"from":144,"to":660,"every":4,"phase":1,"floor":{"WHEAT":1.6,...},"reserve":{"WHEAT":30}}
/// floor = fraction of the product's base price; reserve = shed units never sold by this layer.
#[derive(Clone, Debug, Default)]
pub struct PremSell {
    pub from: i64,
    pub to: i64,
    pub every: i64,
    pub phase: i64,
    pub floor: Vec<(&'static str, f64)>,
    pub reserve: Vec<(&'static str, i64)>,
}

impl PremSell {
    pub fn load(path: &str) -> Result<PremSell, String> {
        let j = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let num = |k: &str, d: i64| if j.get(k).is_null() { d } else { j.get(k).i64() };
        let map = |k: &str| -> Vec<(&'static str, f64)> { j.get(k).obj().iter().map(|(n, v)| (crate::act::intern(n), v.f64())).collect() };
        Ok(PremSell {
            from: num("from", 144), to: num("to", 660), every: num("every", 4).max(1), phase: num("phase", 1),
            floor: map("floor"), reserve: map("reserve").into_iter().map(|(n, v)| (n, v as i64)).collect(),
        })
    }

    /// Extra SELL orders for this step (appended after the chain's own orders).
    pub fn orders(&self, v: &View, market: &[crate::act::Cmd]) -> Vec<crate::act::Cmd> {
        let step = v.step;
        let mut out = vec![];
        if step < self.from || step > self.to || step.rem_euclid(self.every) != self.phase {
            return out;
        }
        for &(item, fl) in &self.floor {
            let Some(p) = kagg_engine::market::param(item) else { continue };
            // already sold by the chain this step: leave it alone
            if market.iter().any(|c| c.is_sell3() && c.s(1) == item && c.n(2) > 0) {
                continue;
            }
            let reserve = self.reserve.iter().find(|r| r.0 == item).map(|r| r.1).unwrap_or(0);
            let avail = v.shed(item) - reserve;
            if avail <= 0 {
                continue;
            }
            let inv = crate::obs::qget(&v.obs.mkt_inventory, item);
            let floor = fl * p.base;
            let mut n = 0i64;
            while n < avail && (kagg_engine::market::price(p, (inv + n) as f64) as f64) >= floor {
                n += 1;
            }
            if n > 0 {
                out.push(crate::act::Cmd::order("SELL", item, n));
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct TppMkt {
    pub net: std::sync::Arc<crate::tpp::Net>,
    pub from: i64,
    pub to: i64,
    pub ops: Vec<String>,
    /// true = keep the chassis's own orders of `ops` too (the policy only adds orders)
    pub add: bool,
}

/// The learned controller: at each decision step the dayobs vector goes through the GRU and the
/// chosen profile becomes the chain's request for that day.
#[derive(Clone, Debug)]
pub struct PolicyCtl {
    pub net: std::sync::Arc<policy::Net>,
    pub h: [f32; policy::H],
    /// Allowed profile ids (others masked).
    pub allow: Vec<bool>,
    /// None = argmax (deploy); Some(seed) = sample (rollouts).
    pub rng: Option<u64>,
    /// Branch oracle: on day `.0` play profile `.1` instead of choosing (the rest of the game is the
    /// policy's own play). Every game is deterministic given its seed and the choices, so replaying a
    /// game with one forced day is an exact counterfactual branch.
    pub force: Option<(usize, usize)>,
    /// Per decision: (day, action, logp, value logit, allowed-profile bitmask used). The learner
    /// recomputes logp under the same mask, so a shield restriction never biases the ratio.
    pub trace: Vec<(usize, usize, f32, f32, u64)>,
}

impl PolicyCtl {
    pub fn new(net: std::sync::Arc<policy::Net>, allow: Vec<bool>, rng: Option<u64>) -> Self {
        PolicyCtl { net, h: [0.0; policy::H], allow, rng, force: None, trace: vec![] }
    }
    fn next_u(&mut self) -> f64 {
        let s = self.rng.get_or_insert(1);
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    /// The profile played on the previous decision (0 before the first).
    pub fn last(&self) -> usize {
        self.trace.last().map(|t| t.1).unwrap_or(0)
    }

    /// Choose today's profile from the observation. `shield` (from the opponent-group controller)
    /// further restricts `allow`; if the two leave nothing, the shield wins.
    pub fn decide(&mut self, day: usize, x: &[f32], shield: Option<&[bool]>) -> usize {
        let o = self.net.step(&mut self.h, x);
        let mut allow: Vec<bool> = match shield {
            Some(sh) => self.allow.iter().zip(sh).map(|(&a, &b)| a && b).collect(),
            None => self.allow.clone(),
        };
        if !allow.iter().any(|&a| a) {
            allow = shield.map(|sh| sh.to_vec()).unwrap_or_else(|| vec![true; self.allow.len()]);
        }
        let bits = allow.iter().enumerate().fold(0u64, |b, (i, &a)| if a && i < 64 { b | (1 << i) } else { b });
        let m = o.pi.iter().zip(&allow).filter(|(_, &a)| a).map(|(&v, _)| v).fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = o.pi.iter().zip(&allow).map(|(&v, &a)| if a { ((v - m) as f64).exp() } else { 0.0 }).collect();
        let z: f64 = e.iter().sum();
        let a = if self.rng.is_some() {
            let mut u = self.next_u() * z;
            let mut k = 0;
            for (i, &p) in e.iter().enumerate() {
                if p > 0.0 {
                    k = i;
                    if u < p {
                        break;
                    }
                    u -= p;
                }
            }
            k
        } else {
            (0..e.len()).max_by(|&i, &j| e[i].total_cmp(&e[j])).unwrap_or(0)
        };
        // the forced branch choice (the draw above still ran, so the rng stream is unchanged)
        let a = match self.force {
            Some((d, k)) if d == day && k < e.len() => k,
            _ => a,
        };
        let logp = (e[a] / z).max(1e-30).ln() as f32;
        self.trace.push((day, a, logp, o.v, bits));
        a
    }
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
        let chassis = Chassis {
            routes: rs,
            router: ShopRouter::from_json(router),
            cfg: Settings::from_json(router.get("settings")),
            opponent_plan: None,
            players: vec![],
            diagnostics: Diagnostics::default(),
            race_signal: None,
        };
        let shop_rescue = router.get("shop_rescue").is_null() || router.get("shop_rescue").bool();
        let pristine_route0 = chassis.route_idx(0).map(|i| chassis.routes[i].clone());
        Ok(Base { name: name.to_string(), chassis, shop_rescue, cut: CUTS.len() - 1, chain: Default::default(), terminal: Default::default(), pristine_route0, phase_us: [0.0; 3], obs_track: None, day_obs: vec![], policy: None, shell: None, rshell: None, disguise: None, endg: None, lay: Default::default(), tpp: None, tpp_mkt: None, force_buys: vec![], prem: None, preempt: None, gt: Default::default(), gtl: None, dispatch: None, sale: None, final_check: None, cash_floor: None })
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
                diagnostics: Diagnostics { log: ch.diagnostics.log, ..Default::default() },
                race_signal: None,
            },
            shop_rescue: self.shop_rescue,
            cut: self.cut,
            chain: layers::Chain {
                profiles: self.chain.profiles.clone(),
                schedule: self.chain.schedule.clone(),
                clone_profile: self.chain.clone_profile,
                clone_strict: self.chain.clone_strict,
                game_off: self.chain.game_off,
                afr_profile: self.chain.afr_profile,
                afr_trigger: self.chain.afr_trigger,
                group_over: self.chain.group_over.clone(),
                group_over_lineage: self.chain.group_over_lineage,
                precedence: self.chain.precedence.clone(),
                group_ctl: self.chain.group_ctl.as_ref().map(|g| {
                    let mut c = layers::group::GroupCtl::new(g.profiles, g.jitter).with_endgame(g.endgame, g.jitter_end).with_mirror(g.mirror_tol).with_shield(g.shield.clone());
                    c.fallback = g.fallback;
                    c
                }),
                ..Default::default()
            },
            terminal: Default::default(),
            pristine_route0: self.pristine_route0.clone(),
            phase_us: [0.0; 3],
            obs_track: self.obs_track.as_ref().map(|_| dayobs::Builder::new()),
            day_obs: vec![],
            policy: self.policy.as_ref().map(|p| PolicyCtl::new(p.net.clone(), p.allow.clone(), p.rng)),
            shell: self.shell.as_ref().map(|s| crate::shell::ShellCtl::new(s.model.clone())),
            rshell: self.rshell.as_ref().map(|s| {
                let mut r = crate::rshell::RShellCtl::new(s.cfg.clone());
                r.want_buy_x = s.want_buy_x;
                r
            }),
            disguise: self.disguise.as_ref().map(|_| crate::disguise::Disguise::default()),
            endg: self.endg.as_ref().map(|e| crate::endg::EndgCtl::new(e.cfg.clone())),
            lay: Default::default(),
            tpp: self.tpp.clone(),
            tpp_mkt: self.tpp_mkt.clone(),
            force_buys: self.force_buys.clone(),
            prem: self.prem.clone(),
            preempt: self.preempt.as_ref().map(|p| crate::preempt::Preempt::new(p.cfg.clone())),
            gt: crate::gt::RivalTracker::new(self.gt.window),
            gtl: self.gtl.as_ref().map(|g| crate::gt::GtLayer::new(g.cfg.clone())),
            dispatch: self.dispatch.as_ref().map(|d| crate::dispatch::Dispatch::new(d.cfg.clone())),
            sale: self.sale.as_ref().map(|m| crate::managers::sale::Sale::new(m.cfg.clone())),
            final_check: self.final_check.clone(),
            cash_floor: self.cash_floor,
        }
    }

    /// Install a profile table (`configs/profiles/*.json`) and optionally a fixed profile for the
    /// whole game (applied at every day boundary; day 0 hour 0 is always profile 0).
    pub fn set_profiles(&mut self, path: &str, fixed: Option<usize>) -> Result<(), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let j = json::parse(&text)?;
        self.chain.profiles = layers::knobs::load_profiles(&j)?;
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
        self.gt.update(&v);
        if self.chassis.cfg.lead_price {
            // price-checked lead selling: allowed items = the rival races it, or selling now >= the expected later price
            let c = self.chassis.cfg.clone();
            let days = v.step / 24;
            let mut ok: Vec<&'static str> = vec![];
            for it in crate::view::PRODUCTS.iter().copied() {
                let stock = self.gt.stock(it);
                let race = stock >= c.lead_signal_stock && (1..=c.lead_signal_look).any(|k| self.gt.sale_forecast(it, v.step + k, days).0 >= c.lead_signal_p);
                let idx = crate::market::item_index(it);
                let inv0 = crate::obs::qget(&v.obs.mkt_inventory, it);
                let now = crate::market::price_i(idx, it, inv0) as f64;
                let mut x = inv0 as f64;
                for t in 0..c.lead_price_h {
                    if t < c.lead_price_rival_h {
                        x += stock as f64 / c.lead_price_rival_h as f64;
                    }
                    x = (x - crate::managers::sale::drain(it, &v.obs.shops, v.step + t, 1) as f64).max(0.0);
                }
                let later = crate::market::price_i(idx, it, x.round() as i64) as f64;
                if race || now >= later * (1.0 - c.lead_price_tol) {
                    ok.push(it);
                }
            }
            self.chassis.race_signal = Some(ok);
        } else if self.chassis.cfg.lead_signal {
            // the market guard's race signal: items the rival is forecast to sell within the look-ahead
            let c = &self.chassis.cfg;
            let days = v.step / 24;
            let sig: Vec<&'static str> = crate::gt::ITEMS
                .iter()
                .copied()
                .filter(|it| self.gt.stock(it) >= c.lead_signal_stock && (1..=c.lead_signal_look).any(|k| self.gt.sale_forecast(it, v.step + k, days).0 >= c.lead_signal_p))
                .collect();
            self.chassis.race_signal = Some(sig);
        }
        if let Some(d) = self.dispatch.as_mut() {
            let group = self.chain.group_ctl.as_ref().and_then(|g| g.group);
            if let Some(p) = d.decide(&v, group, &self.gt) {
                if let Some(c) = p.gt {
                    self.gtl = Some(crate::gt::GtLayer::new(c));
                }
                if let (Some(n), Some(pc)) = (p.policy, self.policy.as_mut()) {
                    if n.n_act == pc.net.n_act {
                        pc.net = n;
                    }
                }
                if let (Some(c), Some(rs)) = (p.rshell, self.rshell.as_mut()) {
                    rs.cfg = c;
                }
                if let (Some(c), Some(pe)) = (p.preempt, self.preempt.as_mut()) {
                    pe.cfg = c;
                }
                if let Some(c) = p.endg {
                    match self.endg.as_mut() {
                        Some(e) => e.cfg = c,
                        None => self.endg = Some(crate::endg::EndgCtl::new(c)),
                    }
                }
            }
        }
        if let Some(n) = self.tpp.as_ref() {
            let a = n.act(&v);
            if std::env::var_os("KRL_TPP_TRACE").is_some() {
                let f = v.farm();
                eprintln!("[tpp] step {step} money {:.0} hands {} seeds {:?} shed {:?} -> {}", f.money, f.hands.len(), v.obs.seeds, v.shed, a.dump());
            }
            return (a, Some(v));
        }
        // the group controller sees this step before the hour-1 decision (it decides the group at
        // step 25, the day-1 decision step); the chain's own track call on this step is then a no-op
        if let Some(g) = self.chain.group_ctl.as_mut() {
            g.track(&v);
        }
        let mut mid_pick: Option<usize> = None;
        if let Some(b) = self.obs_track.as_mut() {
            b.observe(crate::dayview::step_view(&v));
            // mid-day decisions (hour 13 of the policy's `mid` days; ppo2 runs): slot = chronological index
            let mid: &[usize] = self.policy.as_ref().map(|p| p.net.mid.as_slice()).unwrap_or(&[]);
            if b.at_decision_in(mid) {
                if let Some(x) = b.vector() {
                    let s = step.max(0) as usize;
                    let day = s / dayobs::TURNS_PER_DAY;
                    let slot = dayobs::slot_of(s, mid);
                    let at_mid = s % dayobs::TURNS_PER_DAY == dayobs::MID_HOUR;
                    if let Some(p) = self.policy.as_mut() {
                        let n = self.chain.profiles.len().max(1);
                        let sh = self.chain.group_ctl.as_ref().and_then(|g| g.mask(day as i64, n, p.last()));
                        let k = p.decide(slot, &x, sh.as_deref());
                        if at_mid {
                            mid_pick = Some(k);
                        } else {
                            self.chain.requested = Some(k);
                        }
                    }
                    self.day_obs.push((slot, x));
                }
            }
        }
        if let Some(k) = mid_pick {
            self.apply_profile_now(k, &v);
        }
        let t0 = std::time::Instant::now();
        self.chain.lin_matched = self.preempt.as_ref().map(|p| p.matched(step)).unwrap_or(false);
        self.chain.pre(self.cut, &v, &mut self.chassis);
        if let Some(e) = self.endg.as_mut() {
            // endgame model: decide once at its step (inputs = the reactive shell's inputs of the previous turn),
            // then replace the endgame knobs of whatever profile is active on every later step
            let p = v.obs.player;
            if step >= e.cfg.from && e.chosen(p).is_none() {
                let x = match self.rshell.as_ref() {
                    Some(rs) => crate::endg::features(&rs.last_g, &rs.last, &self.chain.knobs),
                    None => vec![0.0; crate::endg::NF],
                };
                e.decide(p, x);
            }
            if let Some(k) = e.patch(p, step, &self.chain.knobs) {
                self.chain.knobs = k;
            }
        }
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
        let on = self.lay.on();
        if on && step % 24 == 1 {
            self.lay.profiles.push(self.chain.profile);
        }
        let fired0 = if on { self.chain.fired.clone() } else { vec![] };
        let mut action = self.chain.post(self.cut, action, &v, &mut self.chassis);
        if on {
            let names: Vec<&'static str> = (0..self.chain.fired.len()).filter(|&i| self.chain.fired[i] > fired0.get(i).copied().unwrap_or(0)).map(|i| layers::CUTS[i]).collect();
            for n in names {
                self.lay.hit_name(&format!("chain:{n}"), step);
            }
        }
        let mut before = if on { Some(action.clone()) } else { None };
        if on {
            self.lay.add_us("chain_post", t2.elapsed().as_secs_f64() * 1e6);
        }
        let money = v.obs.farms.get(v.obs.player as usize).map(|f| f.money).unwrap_or(0.0);
        let floor_snap = self.cash_floor.filter(|(m, until)| step < *until && money < *m).map(|_| action.market.clone());
        let tq = std::time::Instant::now();
        if self.rshell.is_some() {
            let sig = self.chain_sig(&v);
            let tq2 = self.lay.tick("rshell:chain_sig", tq);
            let _ = tq2;
            if let Some(rs) = self.rshell.as_mut() {
                rs.apply(&mut action, &v, &self.chassis, &sig);
            }
        }
        self.lay.mark("rshell", &mut before, &action, step);
        let tq = self.lay.tick("rshell", tq);
        if let Some(sh) = self.shell.as_mut() {
            let group = self.chain.group_ctl.as_ref().and_then(|g| g.group);
            sh.apply(&mut action, &v, group);
        }
        self.lay.mark("shell", &mut before, &action, step);
        let tq = self.lay.tick("shell", tq);
        if let Some(sm) = self.sale.as_mut() {
            sm.apply(&mut action, &v, &self.chassis, &self.gt);
        }
        self.lay.mark("sale", &mut before, &action, step);
        if let Some(snap) = floor_snap {
            // restore every route SELL the shells cut (never more than the route itself asked for)
            for o in snap.iter().filter(|o| o.is_sell3()) {
                let (item, want) = (o.s(1), o.n(2).max(0));
                if let Some(c) = action.market.iter_mut().find(|c| c.is_sell3() && c.s(1) == item) {
                    if c.n(2) < want {
                        c.set_n(2, want);
                    }
                } else if action.market.len() < 10 {
                    action.market.push(o.clone());
                }
            }
        }
        self.lay.mark("cash_floor", &mut before, &action, step);
        if let Some(m) = self.tpp_mkt.as_ref() {
            if step >= m.from && step <= m.to {
                let owned = |c: &crate::act::Cmd| m.ops.iter().any(|o| o == c.op());
                let mut market: Vec<crate::act::Cmd> = action.market.iter().filter(|c| m.add || !owned(c)).cloned().collect();
                market.extend(m.net.market_orders(&v).into_iter().filter(|c| owned(c)));
                market.truncate(crate::tpp::MAXM);
                action.market = market;
            }
        }
        self.lay.mark("tpp_mkt", &mut before, &action, step);
        if let Some(pe) = self.preempt.as_mut() {
            pe.group = self.chain.group_ctl.as_ref().and_then(|g| g.group);
            pe.gt_stock = crate::gt::ITEMS.map(|it| self.gt.stock(it));
            pe.apply(&mut action, &v, &self.chassis);
        }
        self.lay.mark("preempt", &mut before, &action, step);
        if let Some(gl) = self.gtl.as_mut() {
            gl.group = self.chain.group_ctl.as_ref().and_then(|g| g.group);
            gl.apply(&mut action, &v, &self.gt);
        }
        self.lay.mark("gt", &mut before, &action, step);
        let tq = self.lay.tick("preempt", tq);
        if let Some(pm) = self.prem.as_ref() {
            for c in pm.orders(&v, &action.market) {
                if action.market.len() < crate::tpp::MAXM {
                    action.market.push(c);
                }
            }
        }
        for &(fs, fq) in &self.force_buys {
            if fs == step && action.market.len() < crate::tpp::MAXM {
                action.market.push(crate::act::Cmd::order("BUY_PRODUCT", "WHEAT", fq));
            }
        }
        self.lay.mark("prem+force_buy", &mut before, &action, step);
        if let Some(fc) = self.final_check.as_ref() {
            fc.apply(&mut action, &v, &self.chassis);
        }
        self.lay.mark("final_check", &mut before, &action, step);
        if let Some(d) = self.disguise.as_mut() {
            d.apply(&mut action, &v);
        }
        self.lay.mark("disguise", &mut before, &action, step);
        self.lay.tick("tail", tq);
        if on {
            self.lay.add_us("chain_pre", self.phase_us_pre(t0, t1));
            self.lay.add_us("chassis+terminal", (t2 - t1).as_secs_f64() * 1e6);
        }
        if on {
            let p = v.obs.player;
            if self.endg.as_ref().is_some_and(|e| e.chosen(p).is_some_and(|k| k != 0 && step >= e.cfg.starts[k])) {
                self.lay.hit_name("endg_patch", step);
            }
            if step == crate::terminal::FINAL - 23 || step == LAST_ACT_STEP {
                let t = self.terminal.diag;
                if t[4] > 0 && !self.lay.hits.contains_key("terminal") {
                    self.lay.hit_name("terminal", step);
                }
            }
            self.lay.step_done(step);
            if step >= LAST_ACT_STEP {
                self.layer_summary(&v);
            }
        }
        if let Some(b) = self.obs_track.as_mut() {
            b.acted(crate::dayview::own_act(&action));
        }
        // KRL_ACT_LOG=FILE (C1 tape-sync check): per step the physical part of the action (farmer + hands), keyed by
        // the opening shops and our seat, so two builds' games can be diffed step by step
        if let Ok(path) = std::env::var("KRL_ACT_LOG") {
            use std::io::Write;
            let mut phys = action.clone();
            phys.market.clear();
            let line = format!("{}\t{}\t{}\t{}\n", v.obs.shops.join("|"), v.obs.player, step, phys.dump());
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = f.write_all(line.as_bytes());
            }
        }
        self.phase_us = [(t1 - t0).as_secs_f32() * 1e6, (t2 - t1).as_secs_f32() * 1e6, t2.elapsed().as_secs_f32() * 1e6];
        (action, Some(v))
    }
}

impl Base {
    /// One JSON line per game to KRL_LAYER_LOG: every installed layer, the steps it changed the action (count, first,
    /// last), plus the decision layers' own state (rival group, lineage match, group overlay hits, endgame pick and its
    /// best alternative utility, terminal planner outcome, policy profile per day). `dead` = installed layers that never
    /// changed a single action in the game: a layer that is on and reads 0 is not working.
    fn layer_summary(&mut self, v: &View) {
        if self.lay.done {
            return;
        }
        self.lay.done = true;
        let Ok(path) = std::env::var("KRL_LAYER_LOG") else { return };
        let p = v.obs.player;
        let mut installed: Vec<&str> = vec![];
        if self.rshell.is_some() {
            installed.push("rshell");
        }
        if self.shell.is_some() {
            installed.push("shell");
        }
        if self.tpp_mkt.is_some() {
            installed.push("tpp_mkt");
        }
        if self.preempt.is_some() {
            installed.push("preempt");
        }
        if self.prem.is_some() || !self.force_buys.is_empty() {
            installed.push("prem+force_buy");
        }
        if self.disguise.is_some() {
            installed.push("disguise");
        }
        if self.sale.is_some() {
            installed.push("sale");
        }
        if self.final_check.is_some() {
            installed.push("final_check");
        }
        if self.endg.is_some() {
            installed.push("endg_patch");
        }
        if self.terminal.cfg.is_some_and(|c| c.on) {
            installed.push("terminal");
        }
        let q = |k: &str| format!("\"{k}\"");
        let hits: Vec<String> = self.lay.hits.iter().map(|(k, (n, a, b))| format!("{}:[{n},{a},{b}]", q(k))).collect();
        let inst: Vec<String> = installed.iter().map(|k| q(k)).collect();
        let dead: Vec<String> = installed.iter().filter(|k| !self.lay.hits.contains_key(**k)).map(|k| q(k)).collect();
        let off: Vec<String> = (0..layers::CUTS.len()).filter(|i| (self.chain.knobs.off | self.chain.game_off) >> i & 1 == 1).map(|i| q(layers::CUTS[i])).collect();
        let endg = match self.endg.as_ref() {
            Some(e) => match e.seats.iter().find(|s| s.0 == p) {
                Some(s) => {
                    let best = s.3.iter().skip(1).cloned().fold(f32::MIN, f32::max);
                    let zero = s.3.iter().skip(1).filter(|u| **u == 0.0).count();
                    format!("{{\"pick\":{},\"name\":{},\"u_best_alt\":{:.4},\"u_exact_zero\":{zero},\"gate\":{}}}", s.1, q(&e.cfg.names[s.1]), if s.3.is_empty() { 0.0 } else { best }, e.cfg.gate)
                }
                None => "{\"pick\":-1}".into(),
            },
            None => "null".into(),
        };
        let group = self.chain.group_ctl.as_ref().and_then(|g| g.group).map(|g| g as i64).unwrap_or(-1);
        let lin = match self.preempt.as_ref() {
            Some(pe) => format!("{{\"matched\":{},\"weight\":{:.3},\"fired\":{}}}", pe.matched(v.step), pe.lineage_weight(), pe.fired),
            None => "null".into(),
        };
        if let Some(rs) = self.rshell.as_ref() {
            for (i, n) in ["rs:observe+lineage", "rs:projected_shed", "rs:global", "rs:items"].iter().enumerate() {
                self.lay.add_us(n, rs.prof[i]);
            }
        }
        let t = self.terminal.diag;
        let line = format!(
            "{{\"player\":{p},\"money\":{:.0},\"installed\":[{}],\"dead\":[{}],\"hits\":{{{}}},\"stages_off\":[{}],\"group\":{group},\"group_over_hits\":{},\"lineage\":{lin},\"endg\":{endg},\"terminal\":{{\"planned\":{},\"skip_shadow\":{},\"skip_baseline\":{},\"no_plan\":{},\"changed_steps\":{}}},\"profiles\":{:?},\"ms\":{{{}}},\"chain_ms\":{{{}}}}}",
            v.farm().money,
            inst.join(","),
            dead.join(","),
            hits.join(","),
            off.join(","),
            self.chain.group_over_hits,
            t[0],
            t[1],
            t[2],
            t[3],
            t[4],
            self.lay.profiles,
            self.lay.us.iter().map(|(k, v)| format!("{}:{:.1}", q(k), v / 1e3)).collect::<Vec<_>>().join(","),
            {
                let mut st: Vec<(f64, usize)> = self.chain.stage_us.iter().enumerate().map(|(i, x)| (x.0, i)).filter(|x| x.0 > 0.0).collect();
                st.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                st.iter().take(10).map(|(u, i)| format!("{}:{:.1}", q(layers::CUTS[*i]), u / 1e3)).collect::<Vec<_>>().join(",")
            }
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{line}");
        }
    }

    fn phase_us_pre(&self, t0: std::time::Instant, t1: std::time::Instant) -> f64 {
        (t1 - t0).as_secs_f64() * 1e6
    }

    /// Switch the chain to profile `id` at once (the hour-13 decision of a mid day). Same effect as the
    /// chain's day-boundary switch (layers::Chain::switch_profile): the profile's knobs, the per-day group
    /// jitter, and the knobs mirrored into the chassis / CA / OR2. Profiles only move market and timing
    /// knobs, so a mid-day switch cannot desync the tape.
    pub fn apply_profile_now(&mut self, id: usize, v: &View) {
        let ch = &mut self.chain;
        let Some((_, k)) = ch.profiles.get(id) else { return };
        let mut k = k.clone();
        if let (Some((groups, j)), Some(g)) = (ch.group_over.as_ref(), ch.group_ctl.as_ref().and_then(|g| g.group)) {
            if groups.contains(&g) && (!ch.group_over_lineage || ch.lin_matched) {
                if let Ok(k2) = k.with(j) {
                    if k2 != k {
                        ch.group_over_hits += 1;
                    }
                    k = k2;
                }
            }
        }
        if let Some(g) = ch.group_ctl.as_ref() {
            let o = g.offset(v.step / 24);
            if o != 0 {
                k.rsa_look = (k.rsa_look + o).max(1);
                k.race_clone = (k.race_clone + o).max(1);
                k.race_escalated = (k.race_escalated + o).max(1);
                k.race_mirror = (k.race_mirror + o).max(1);
            }
        }
        ch.profile = id;
        self.chassis.cfg.racepx_margin = k.racepx_margin;
        ch.ca.margin = Some(k.ca_margin);
        ch.or2.margin = Some(k.or2_slot_margin);
        ch.knobs = k;
    }

    /// Reactive shell v2 + whole-game stage switches + knob overrides on every profile.
    ///   rshell     crate::rshell config JSON
    ///   chain_off  comma-separated stage names (market or economy stages, whole game)
    ///   knob_over  JSON object of knob overrides applied on top of EVERY profile (after --profiles)
    pub fn apply_extras(&mut self, rshell: Option<&str>, chain_off: Option<&str>, knob_over: Option<&str>) -> Result<(), String> {
        self.apply_extras2(rshell, chain_off, knob_over, None, None)
    }

    /// `apply_extras` + the endgame model (`endg` config path; `endg_force` = force proposal K, for the labeller).
    pub fn apply_extras2(&mut self, rshell: Option<&str>, chain_off: Option<&str>, knob_over: Option<&str>, endg: Option<&str>, endg_force: Option<usize>) -> Result<(), String> {
        if let Some(f) = rshell {
            let c = crate::rshell::RConfig::load(f)?;
            self.rshell = Some(crate::rshell::RShellCtl::new(std::sync::Arc::new(c)));
        }
        if let Some(s) = chain_off {
            self.chain.game_off = layers::knobs::stage_bits(s, true)?;
        }
        if let Some(f) = endg {
            let c = crate::endg::EndgCfg::load(f)?;
            if let Some(m) = endg_force {
                let mut c = c;
                c.mode = crate::endg::Mode::Force(m);
                self.endg = Some(crate::endg::EndgCtl::new(std::sync::Arc::new(c)));
            } else {
                self.endg = Some(crate::endg::EndgCtl::new(std::sync::Arc::new(c)));
            }
            if self.rshell.is_none() {
                // the endgame model reads the shell's inputs: an observe-mode shell (inputs recorded, action unchanged)
                let rc = crate::rshell::RConfig { on: false, ..Default::default() };
                self.rshell = Some(crate::rshell::RShellCtl::new(std::sync::Arc::new(rc)));
            }
        }
        if let Some(f) = knob_over {
            let j = json::parse(&std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?)?;
            if self.chain.profiles.is_empty() {
                self.chain.profiles = vec![("v61.1".into(), Default::default())];
            }
            for (_, k) in self.chain.profiles.iter_mut() {
                *k = k.with(&j)?;
            }
        }
        Ok(())
    }

    /// The chain's trackers as the reactive shell v2 reads them.
    pub fn chain_sig(&self, v: &View) -> crate::rshell::ChainSig {
        let p = v.obs.player;
        let ch = &self.chain;
        let k = &ch.knobs;
        let cx = ch.ctx.iter().find(|(q, _)| *q == p).map(|(_, c)| c.clone()).unwrap_or_default();
        let v92 = if k.v92_on {
            ch.v92.predicted_rival(p, v.step).map(|cmds| {
                let mut q: crate::obs::Qty = vec![];
                for c in cmds {
                    if let Some(it) = PRODUCTS.iter().copied().find(|x| *x == c.s(1)) {
                        crate::obs::qadd(&mut q, it, c.n(2).max(0));
                    }
                }
                q
            })
        } else {
            None
        };
        let pl = self.chassis.players.iter().find(|(q, _)| *q == p).map(|(_, s)| s);
        let mut debts: crate::obs::Qty = vec![];
        if let Some(s) = pl {
            if s.sell.due_step > v.step {
                for (it, n) in &s.sell.suppress {
                    crate::obs::qadd(&mut debts, it, *n);
                }
            }
            for (t, q) in &s.sell.r36_debts {
                if *t > v.step {
                    for (it, n) in q {
                        crate::obs::qadd(&mut debts, it, *n);
                    }
                }
            }
        }
        crate::rshell::ChainSig {
            race_h: ch.race.horizon(p),
            r37_h: cx.r37_horizon.unwrap_or(2),
            race_lost: ch.race.level(p) >= k.race_escalated && k.race_escalated > k.race_clone,
            clone: ch.race.clone_signal(v),
            similarity: layers::r37::similarity(v),
            group: ch.group_ctl.as_ref().and_then(|g| g.group),
            afr_24: ch.afr.recent(p, v.step, 24),
            v92,
            route: pl.and_then(|s| s.route),
            k_race_clone: k.race_clone,
            k_v92_on: k.v92_on,
            k_afr_on: k.afr_on,
            k_rsa_look: k.rsa_look,
            term_start: k.term_start,
            profile: ch.profile,
            fert_committed: ch.v219.fert_dedicated(p, &v.obs.invs) + ch.r51.undelivered(p),
            debts,
        }
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

/// Per-game layer activity (Base::layer_summary). `hits[layer] = (steps it changed the action, first step, last step)`.
#[derive(Default, Clone, Debug)]
pub struct LayerLog {
    pub hits: std::collections::BTreeMap<String, (u64, i64, i64)>,
    pub step_hits: Vec<String>,
    pub profiles: Vec<usize>,
    /// microseconds per layer, whole game
    pub us: std::collections::BTreeMap<String, f64>,
    pub done: bool,
    state: u8, // 0 unknown, 1 on, 2 off
}

impl LayerLog {
    pub fn on(&mut self) -> bool {
        if self.state == 0 {
            self.state = if std::env::var_os("KRL_LAYER_LOG").is_some() || std::env::var_os("KRL_LAYER_TRACE").is_some() { 1 } else { 2 };
        }
        self.state == 1
    }

    fn trace(&self) -> bool {
        std::env::var_os("KRL_LAYER_TRACE").is_some()
    }

    pub fn add_us(&mut self, name: &str, us: f64) {
        *self.us.entry(name.to_string()).or_insert(0.0) += us;
    }

    /// Adds the time since `t` to `name` (when the log is on) and returns now.
    pub fn tick(&mut self, name: &str, t: std::time::Instant) -> std::time::Instant {
        let now = std::time::Instant::now();
        if self.state == 1 {
            self.add_us(name, (now - t).as_secs_f64() * 1e6);
        }
        now
    }

    pub fn hit_name(&mut self, name: &str, step: i64) {
        let e = self.hits.entry(name.to_string()).or_insert((0, step, step));
        e.0 += 1;
        e.2 = step;
        self.step_hits.push(name.to_string());
    }

    /// Record `name` when the action changed since `before`; `before` then becomes the current action.
    pub fn mark(&mut self, name: &str, before: &mut Option<crate::act::Action>, now: &crate::act::Action, step: i64) {
        if let Some(b) = before.as_mut() {
            if b != now {
                self.hit_name(name, step);
                if self.trace() {
                    // the market diff is what a sale layer changes; farmer / hand changes are marked as such
                    let mk = |a: &crate::act::Action| {
                        let d = a.dump();
                        d.find("\"market\": ").map(|i| d[i + 10..d.len() - 1].to_string()).unwrap_or(d)
                    };
                    let what = if b.market != now.market { format!("{} -> {}", mk(b), mk(now)) } else { "farmer/hands".to_string() };
                    if let Some(last) = self.step_hits.last_mut() {
                        *last = format!("{name}: {what}");
                    }
                }
                *b = now.clone();
            }
        }
    }

    /// KRL_LAYER_TRACE: one line per step on which any layer changed the action: `step<TAB>layer,layer,..`.
    pub fn step_done(&mut self, step: i64) {
        if !self.step_hits.is_empty() {
            if let Ok(path) = std::env::var("KRL_LAYER_TRACE") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = writeln!(f, "{step}\t{}", self.step_hits.join(","));
                }
            }
        }
        self.step_hits.clear();
    }
}
