//! The wrapper chain after the chassis, in Python call order (docs/chain_scan.txt).
//!
//! A Python wrapper is `pre(obs)` -> `parent(obs)` -> `post(obs, action)`. The chain runs the
//! pre phases outermost first, then the chassis, then the post phases innermost first — the
//! exact order Python executes them. Cross-layer state that an OUTER layer writes before an
//! INNER one reads it (e.g. `_R37_HORIZONS`, read by the R36 reserve) lives in [`Ctx`].
pub mod basic;
pub mod ca;
pub mod cha;
pub mod dispatch;
pub mod group;
pub mod hd2;
pub mod knobs;
pub mod or2;
pub mod race;
pub mod tail;
pub mod afr;
pub mod v44y;
pub mod r36;
pub mod r37;
pub mod r51;
pub mod r85;
pub mod v219;
pub mod v231;
pub mod v233;
pub mod v9;
pub mod v92;

use crate::act::Action;
use crate::chassis::Chassis;
use crate::view::View;

/// Chain stages in Python call order; a cut name = the chain THROUGH that stage
/// (python/ref/truncate.py uses the same names).
pub const CUTS: [&str; 66] = [
    "chassis", "terminal", "room", "v28", "v219", "experiment", "order", "v31", "v231", "r36", "r37", "release",
    "v233", "r46", "r51_input", "r51_warehouse", "r53", "r70",
    "r85", "r95", "r97", "courier", "carrot", "herd", "fert", "opening", "race", "racepx", "racegate",
    "ctrtable", "overflow", "ca", "or2", "ch", "sr", "hd2", "cs",
    "race", "r127", "pg", "v44y", "y", "e335", "v11", "v13v",
    "e343_wl", "adv", "t62a", "pipe", "ma", "wb3", "fx", "dp", "mp", "bd", "mpx", "sm",
    "cxd", "e410", "e402", "mg", "ig", "rsa", "afr", "tsell", "hfeed",
];

pub fn cut_index(name: &str) -> Option<usize> {
    if name == "full" {
        return Some(CUTS.len() - 1);
    }
    CUTS.iter().position(|c| *c == name)
}

/// State written by outer layers' pre phases and read by inner layers.
#[derive(Default, Debug, Clone)]
pub struct Ctx {
    /// `_R37_HORIZONS[player]` (None = never set: readers default to 2).
    pub r37_horizon: Option<i64>,
    /// `_V9_ITEM_HZ[player]` (None = absent).
    pub v9_item_hz: Option<Vec<(&'static str, i64)>>,
    /// `_RACE_STATE[player]['horizon']` for this step (0 = off).
    pub race_horizon: i64,
}

#[derive(Default)]
pub struct Chain {
    pub ctx: Vec<(i64, Ctx)>,
    pub v219: v219::V219,
    pub v231: v231::V231,
    pub r37: r37::R37,
    pub v233: v233::V233,
    pub r51: r51::R51,
    pub r85: r85::R85,
    pub v9: v9::V9,
    pub ca: ca::Ca,
    pub or2: or2::Or2,
    pub v92: v92::V92,
    pub herd: hd2::Herd,
    pub race: race::Race,
    pub y: v44y::Y,
    pub cha: cha::Cha,
    pub afr: afr::Afr,
    /// Steps on which each stage changed the action (coverage for parity runs).
    pub fired: Vec<u64>,
    /// Per-stage (total us, max us) — latency attribution.
    pub stage_us: Vec<(f64, f64)>,
    /// This turn's time per stage (pre + post), us — reset at the start of each `pre`.
    pub turn_us: Vec<f32>,
    /// Active knobs (the current profile's).
    pub knobs: knobs::Knobs,
    /// Profile table (entry 0 = v61.1); empty = profile 0 only.
    pub profiles: Vec<(String, knobs::Knobs)>,
    /// Fixed per-day schedule (profile id per day 0..29), used when no controller request.
    pub schedule: Option<Vec<usize>>,
    /// A controller's choice for the next day boundary.
    pub requested: Option<usize>,
    /// Active profile id.
    pub profile: usize,
    /// Heuristic controller: switch to this profile on days a clone is detected (else 0).
    pub clone_profile: Option<usize>,
    /// Strict clone gate for the controller: a day counts as a clone day only if >= 20 of the
    /// last 24 steps had our units on exactly the rival's squares and layout similarity >= .95
    /// (no step-1 cash mirror, which fires on much of the non-clone field).
    pub clone_strict: bool,
    /// Front-run-triggered escalation: at a day boundary, switch to this profile when AFR saw
    /// >= `afr_trigger` pre-emptions in the past day (else the base schedule applies).
    pub afr_profile: Option<usize>,
    pub afr_trigger: usize,
    /// Per player: last 24 position-equality observations (strict gate).
    pub ctrack: Vec<(i64, Vec<bool>)>,
    /// Opponent-group controller (layers/group.rs): per-group profile + secret timing jitter.
    pub group_ctl: Option<group::GroupCtl>,
    /// D6 dispatch (layers/dispatch.rs): cluster tracker + rule sets from the profile table.
    pub dispatch: dispatch::Dispatch,
}

impl Chain {
    pub fn ctx(&mut self, player: i64) -> &mut Ctx {
        let i = match self.ctx.iter().position(|(p, _)| *p == player) {
            Some(i) => i,
            None => {
                self.ctx.push((player, Ctx::default()));
                self.ctx.len() - 1
            }
        };
        &mut self.ctx[i].1
    }

    /// Day-boundary profile switch: every game starts on profile 0; at hour 1 of each day the
    /// controller's request (else the fixed schedule, else the current profile) takes effect and
    /// holds for the day.
    fn clone_day(&self, v: &View) -> bool {
        if !self.clone_strict {
            return self.race.clone_signal(v);
        }
        let Some((_, h)) = self.ctrack.iter().find(|(p, _)| *p == v.obs.player) else { return false };
        h.len() >= 24 && h.iter().filter(|b| **b).count() >= 20 && r37::similarity(v) >= 0.95
    }

    fn track_clone(&mut self, v: &View) {
        if !self.clone_strict || self.clone_profile.is_none() {
            return;
        }
        let (own, rival) = (v.farm(), v.rival());
        let eq = own.farmer == rival.farmer && own.hands == rival.hands;
        let p = v.obs.player;
        let i = match self.ctrack.iter().position(|(q, _)| *q == p) {
            Some(i) => i,
            None => {
                self.ctrack.push((p, Vec::with_capacity(25)));
                self.ctrack.len() - 1
            }
        };
        let h = &mut self.ctrack[i].1;
        h.push(eq);
        if h.len() > 24 {
            h.remove(0);
        }
    }

    pub fn switch_profile(&mut self, v: &View, ch: &mut Chassis) {
        let step = v.step;
        let next = if step == 0 {
            Some(0)
        } else if step % 24 == 1 {
            self.requested
                .take()
                .or_else(|| self.afr_profile.and_then(|k| (self.afr.recent(v.obs.player, step, 24) >= self.afr_trigger.max(1)).then_some(k)))
                .or_else(|| self.clone_profile.map(|k| if self.clone_day(v) { k } else { 0 }))
                .or_else(|| self.group_ctl.as_ref().and_then(|g| g.profile(step / 24)))
                .or_else(|| self.schedule.as_ref().and_then(|s| s.get((step / 24) as usize).copied()))
                .or_else(|| self.dispatch.knobs(step / 24).map(|_| self.profile))
        } else {
            None
        };
        if let Some(id) = next {
            let k = self.profiles.get(id).map(|(_, k)| k.clone()).unwrap_or_default();
            self.profile = if id < self.profiles.len() { id } else { 0 };
            let mut k = k;
            if let Some(g) = self.group_ctl.as_ref() {
                // secret per-game, per-day timing jitter (market-side knobs only: sync-safe)
                let o = g.offset(step / 24);
                if o != 0 {
                    k.rsa_look = (k.rsa_look + o).max(1);
                    k.race_clone = (k.race_clone + o).max(1);
                    k.race_escalated = (k.race_escalated + o).max(1);
                    k.race_mirror = (k.race_mirror + o).max(1);
                }
            }
            if let Some(j) = self.dispatch.knobs(step / 24) {
                // D6 dispatch overrides on top of the day's profile (validated at load)
                k = k.with(j).unwrap_or(k);
            }
            ch.cfg.racepx_margin = k.racepx_margin;
            self.ca.margin = Some(k.ca_margin);
            self.or2.margin = Some(k.or2_slot_margin);
            self.knobs = k;
        }
    }

    /// Pre phases, outermost first (stage `cut` down to 1).
    pub fn pre(&mut self, cut: usize, v: &View, ch: &mut Chassis) {
        if let Some(g) = self.group_ctl.as_mut() {
            g.track(v);
        }
        let set = self.knobs.dispatch_set;
        self.dispatch.pre(v, ch, set);
        self.switch_profile(v, ch);
        self.track_clone(v);
        if self.turn_us.len() < CUTS.len() {
            self.turn_us = vec![0.0; CUTS.len()];
        }
        self.turn_us.iter_mut().for_each(|x| *x = 0.0);
        for stage in (1..=cut).rev() {
            if (self.knobs.skip >> stage) & 1 == 1 {
                continue;
            }
            let t0 = std::time::Instant::now();
            match stage {
                8 => self.v231.pre(v),
                14 => self.r51.pre(v),
                45 => self.cha.wl_pre(v),
                48 => {
                    if self.cha.pipe_pre(v, ch) {
                        self.v9.ct_disabled = true;
                    }
                }
                49 => self.cha.ma_pre(v),
                51 => self.cha.fx_pre(v),
                54 => self.cha.bd_pre(v),
                55 => self.cha.mpx_pre(v),
                37 => {
                    let h = self.race.pre(v, ch, &self.knobs);
                    self.ctx(v.obs.player).race_horizon = h;
                }
                21 => self.v9.courier_pre(v),
                22 => self.v9.carrot_pre(v),
                26 => {
                    let hz = self.v9.race_pre(v, ch, &self.knobs);
                    self.ctx(v.obs.player).v9_item_hz = Some(hz);
                }
                10 => {
                    let h = self.r37.pre(v);
                    self.ctx(v.obs.player).r37_horizon = Some(h);
                }
                _ => {}
            }
            self.turn_us[stage] += t0.elapsed().as_secs_f32() * 1e6;
        }
    }

    /// Post phases, innermost first (stage 1 up to `cut`).
    pub fn post(&mut self, cut: usize, mut action: Action, v: &View, ch: &mut Chassis) -> Action {
        if self.fired.len() < CUTS.len() {
            self.fired = vec![0; CUTS.len()];
            self.stage_us = vec![(0.0, 0.0); CUTS.len()];
        }
        for stage in 1..=cut {
            if (self.knobs.skip >> stage) & 1 == 1 {
                continue;
            }
            let before = action.clone();
            let t0 = std::time::Instant::now();
            action = match stage {
                1 => action, // terminal closure planner: wraps the chassis in Base::act
                2 => basic::room_guard(action, v),
                3 => action, // v28 entry guard: Rust layers do not raise
                4 => self.v219.apply(action, v, ch),
                5 => action, // APPLY_TIMING = False
                6 if v.step >= 144 => basic::sales_first(action),
                7 => action, // v31 entry guard
                8 => self.v231.post(action, v, ch),
                9 => {
                    let cx = self.ctx(v.obs.player).clone();
                    let a = r36::reserve(action, v, ch, &cx, self.knobs.racegate_margin);
                    if v.step >= 288 {
                        basic::sales_first(a)
                    } else {
                        a
                    }
                }
                10 => self.r37.post(action, v, ch),
                12 => self.v233.post(action, v, ch),
                14 if !self.knobs.r51_input_on => action,
                14 => {
                    let p = v.obs.player;
                    let parents = [self.v219.info(p), self.v233.info(p)];
                    self.r51.post(action, v, ch, parents)
                }
                15 => r51::warehouse(action, v, ch),
                18 => {
                    let p = v.obs.player;
                    let dedicated = self.v219.fert_dedicated(p, &v.obs.invs) + self.r51.undelivered(p);
                    self.r85.post(action, v, ch, dedicated, self.knobs.r85_feed_on)
                }
                19 => self.r85.r95(action, v, ch),
                20 => r85::r97(action, v, ch),
                21 => self.v9.courier(action, v, ch),
                22 => self.v9.carrot(action, v, ch),
                24 if self.knobs.v9_fert_on => v9::V9::fert(action, v, ch),
                25 => {
                    // v92 PREDICT sits right after the V9 opening (ca25 order); inert unless knobs.v92_on
                    let a = v9::V9::opening(action, v, ch);
                    self.v92.post(a, v, ch, &self.v9, &self.knobs)
                }
                26 => self.v9.race_post(action, v, ch),
                29 => self.v9.ctrtable(action, v),
                30 => v9::overflow(action, v),
                31 => self.ca.post(action, v, ch),
                32 => self.or2.post(action, v, ch),
                33 => self.or2.ch(action, v, ch),
                34 => hd2::sr(action, v, ch),
                35 => self.herd.hd2(action, v, ch),
                36 => self.herd.cs(action, v, ch),
                37 => {
                    self.race.post(&action, v);
                    action
                }
                38 => self.race.r127(action, v, ch),
                39 => v44y::preguard(action, v),
                40 if self.knobs.v44y_on => v44y::v44y(action, v, ch),
                41 => self.y.post(action, v, ch),
                42 => v44y::e335(action, v),
                45 => self.cha.wl(action, v, ch),
                46 => {
                    let a = if self.knobs.adv_on { cha::adv(action, v, ch, self.knobs.adv_look) } else { action };
                    self.race.set_prev_action(&a, v);
                    a
                }
                47 => cha::t62a(action, v),
                48 => self.cha.pipe_post(action, v),
                50 => self.cha.wb3(action, v),
                51 => {
                    let ev = self.knobs.ev_on.then_some(self.knobs.ev_h);
                    self.cha.fx_post(action, v, ch, ev)
                }
                52 if self.knobs.dp_on => self.cha.dp(action, v, ch, self.knobs.dp_h),
                53 if self.knobs.mp_on => self.cha.mp(action, v, ch, self.knobs.mp_h),
                54 => self.cha.bd(action, v),
                55 if self.knobs.mpx_on => self.cha.mpx(action, v, ch),
                56 => cha::sm(action, v),
                57 if self.knobs.cxd_on => {
                    // rival models for counter D (empty = the copy assumption, v61.1)
                    let models: Vec<Vec<crate::act::Cmd>> = match (self.knobs.cxd_model, self.v92.predicted_rival(v.obs.player, v.step)) {
                        (1, Some(p)) => vec![p],
                        (2, Some(p)) => vec![action.market.clone(), p],
                        _ => vec![],
                    };
                    tail::cxd_models(action, v, ch, &models)
                }
                58 if !self.knobs.e410_on => action,
                58 => {
                    let reactive = self.r51.worker_actors(v.obs.player);
                    tail::e410(action, v, ch, &reactive)
                }
                59 => {
                    let (a, cut) = tail::e402(action, v, ch);
                    if cut > 0 {
                        self.ca.cut_spare_carrot(v.obs.player, cut);
                    }
                    a
                }
                60 => tail::mg(action),
                61 => tail::ig(action, v, ch),
                62 if self.knobs.rsa_on => tail::rsa(action, v, ch, self.knobs.rsa_look, self.knobs.rsa_min_frac),
                63 if self.knobs.afr_on => {
                    let k = self.knobs.clone();
                    self.afr.post(action, v, ch, &k)
                }
                // end-game sale-timing search (ours, knobs tsell_*; off = v61.1)
                64 if self.knobs.tsell_on => self.v92.tsell(action, v, ch, &self.knobs),
                // shepherds_ledger feed-risk pickup (knob hfeed_on; off = v61.1)
                65 if self.knobs.hfeed_on => tail::hfeed(action, v, ch),
                _ => action, // 11 RELEASE, 13 R46, 16 R53, 17 R70, 23 HERD, 27 RACEPX, 28 RACEGATE, 43 V11, 44 V13V, 49 MA: no-op post
            };
            let us = t0.elapsed().as_secs_f64() * 1e6;
            self.stage_us[stage].0 += us;
            self.stage_us[stage].1 = self.stage_us[stage].1.max(us);
            if let Some(t) = self.turn_us.get_mut(stage) {
                *t += us as f32;
            }
            if action != before {
                self.fired[stage] += 1;
            }
        }
        action
    }
}
