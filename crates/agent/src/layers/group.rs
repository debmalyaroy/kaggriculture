//! Opponent-group controller: classify the opponent from public day-0 signals, then play each
//! group's own lever profile, optionally with secret per-game timing jitter.
//!
//! Groups (from clone forensics on 324 real ladder games, 2026-09-25):
//!   0 DIFFERENT  our units stood on the rival's squares on < 10% of day 0's turns (the best-scoring
//!                threshold on 324 real games; identification by turn: 70% at turn 1, 76% at turn 3,
//!                77% at turn 24, 81% by day 1.5 -- .local/ident_timing.py)
//!                (a different farm plan; typically weaker, won by the economy);
//!   1 PARTIAL    same squares, but different money at step 1 (a copy of our farm plan with a
//!                different opening trade);
//!   2 COPY       same squares AND identical money at step 1 (same opening: a near copy, the pure
//!                market race; the group that plans against us from our public replays).
//! The group is fixed at the day-1 boundary (step 25) and never changes.
//!
//! Sync rule: a group only selects a PROFILE (market-side knobs) and jitters market-side timing
//! knobs; the structural tape and farm actions are untouched, so every branch stays in sync with the
//! base. Jitter: each day, `rsa_look` and the RACE horizons move by an offset in [-J, +J] drawn from a
//! hash of a build salt, this game's opening state (shops, market inventory, rival money at the
//! first observation) and the day -- so replays show the distribution, never the next game's draw.
use crate::view::View;

const SALT: u64 = 0x9a0f_e1d3_2026_0925;
/// First end-game day (the course change).
pub const END_DAY: i64 = 24;

fn fnv(h: u64, s: &str) -> u64 {
    let mut h = h;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[derive(Clone, Debug, Default)]
pub struct GroupCtl {
    /// Profile id per group [DIFFERENT, PARTIAL, COPY].
    pub profiles: [usize; 3],
    /// Max jitter offset per group (0 = none), days 1..23.
    pub jitter: [i64; 3],
    /// End-game profile per group for days >= END_DAY (None = keep the day-1..23 profile).
    pub endgame: [Option<usize>; 3],
    /// Max jitter offset per group for days >= END_DAY.
    pub jitter_end: [i64; 3],
    /// Decided group (from step 25 on).
    pub group: Option<usize>,
    /// Day-0 position-equality observations, and step-1 cash equality.
    pos_eq: Vec<bool>,
    cash_eq1: Option<bool>,
    /// Per-game hash of the opening state (set at the first observation).
    h: Option<u64>,
    /// D24 mirror guard: Some(tol) keeps the day-1..23 profile at END_DAY when the rival is an exact
    /// mirror on day 23 (same squares on >= 90% of its turns AND |money gap| <= tol at its last step).
    /// v622 round 1: a D24 switch lost the v62.1 mirror 14-0 but won near copies (herd_safe) 4-0.
    pub mirror_tol: Option<f64>,
    pos_eq23: Vec<bool>,
    pub mirror24: Option<bool>,
    /// RL shield (kaggriculture-rl only): per-group profile masks for the learned policy.
    pub shield: Option<Shield>,
    /// False for a shield-only controller: it masks a policy's choice and never plays a profile of its
    /// own. (A shield-only controller once supplied profile 0 as a fallback that outranked fixed
    /// schedules, so every fixed-profile opponent in PPO played v61.1.)
    pub fallback: bool,
    /// Last step `track` consumed: the RL agent tracks before its hour-1 decision and the chain's
    /// pre phase tracks again on the same step; the second call is a no-op.
    last_step: Option<i64>,
}

/// RL shield config (`--shield FILE`, JSON): which profiles the learned policy may pick per group,
/// and whether an exact day-23 mirror pins the day-1..23 choice from day 24 on (the D24 mirror
/// guard applied to a learned policy). A group without a list allows every profile.
///     {"DIFFERENT": [0, 13, 19], "PARTIAL": null, "COPY": [19, 31], "never": [9, 11], "mirror_keep": true, "mirror_tol": 0}
#[derive(Clone, Debug, Default)]
pub struct Shield {
    pub allow: [Option<Vec<usize>>; 3],
    pub mirror_keep: bool,
    /// Profiles never allowed, on any day or group (the league profile audit's losers).
    pub never: Vec<usize>,
}

impl Shield {
    pub fn from_json(j: &kagg_engine::json::Json) -> Result<(Shield, Option<f64>), String> {
        let list = |k: &str| -> Result<Option<Vec<usize>>, String> {
            let v = j.get(k);
            if v.is_null() {
                return Ok(None);
            }
            v.arr().iter().map(|x| usize::try_from(x.i64()).map_err(|_| format!("shield {k}: bad profile id"))).collect::<Result<Vec<_>, _>>().map(Some)
        };
        let s = Shield {
            allow: [list("DIFFERENT")?, list("PARTIAL")?, list("COPY")?],
            mirror_keep: !j.get("mirror_keep").is_null() && j.get("mirror_keep").bool(),
            never: list("never")?.unwrap_or_default(),
        };
        let tol = (!j.get("mirror_tol").is_null()).then(|| j.get("mirror_tol").f64());
        Ok((s, tol))
    }
}

impl GroupCtl {
    pub fn new(profiles: [usize; 3], jitter: [i64; 3]) -> Self {
        GroupCtl { profiles, jitter, jitter_end: jitter, fallback: true, ..Default::default() }
    }

    /// Add the end-game course change (days >= END_DAY).
    pub fn with_endgame(mut self, endgame: [Option<usize>; 3], jitter_end: [i64; 3]) -> Self {
        self.endgame = endgame;
        self.jitter_end = jitter_end;
        self
    }

    /// A shield-only controller: masks the policy, never plays a fallback profile.
    pub fn shield_only(shield: Shield, mirror_tol: Option<f64>) -> Self {
        let mut g = GroupCtl::new([0, 0, 0], [0, 0, 0]).with_mirror(mirror_tol).with_shield(Some(shield));
        g.fallback = false;
        g
    }

    /// Add the RL shield (None = off).
    pub fn with_shield(mut self, shield: Option<Shield>) -> Self {
        self.shield = shield;
        self
    }

    /// Add the D24 mirror guard (None = off).
    pub fn with_mirror(mut self, tol: Option<f64>) -> Self {
        self.mirror_tol = tol;
        self
    }

    /// Called every step before the day-boundary switch.
    pub fn track(&mut self, v: &View) {
        let step = v.step;
        if self.last_step == Some(step) {
            return;
        }
        self.last_step = Some(step);
        let (own, rival) = (v.farm(), v.rival());
        if self.h.is_none() {
            let mut h = fnv(0xcbf2_9ce4_8422_2325 ^ SALT, &format!("{}", v.obs.player));
            for s in &v.obs.shops {
                h = fnv(h, s);
            }
            for (it, q) in &v.obs.mkt_inventory {
                h = fnv(h, &format!("{it}{q}"));
            }
            h = fnv(h, &format!("{:.0}", rival.money));
            self.h = Some(h);
        }
        if (0..24).contains(&step) {
            self.pos_eq.push(own.farmer == rival.farmer && own.hands == rival.hands);
        }
        if step == 1 {
            self.cash_eq1 = Some(own.money == rival.money);
        }
        if self.mirror_tol.is_some() && ((END_DAY - 1) * 24..END_DAY * 24).contains(&step) {
            self.pos_eq23.push(own.farmer == rival.farmer && own.hands == rival.hands);
            if step == END_DAY * 24 - 1 {
                let eq = self.pos_eq23.iter().filter(|b| **b).count();
                let gap = (own.money - rival.money).abs() as f64;
                let m = eq * 10 >= self.pos_eq23.len() * 9 && gap <= self.mirror_tol.unwrap_or(-1.0);
                self.mirror24 = Some(m);
                if std::env::var("KAGG_GROUP_LOG").is_ok() {
                    eprintln!("[group] seat {} group {:?} d23_pos_eq {}/{} money_gap {:.0} mirror {}", v.obs.player, self.group, eq, self.pos_eq23.len(), gap, m);
                }
            }
        }
        if step >= 25 && self.group.is_none() {
            let eq = self.pos_eq.iter().filter(|b| **b).count();
            let same_squares = !self.pos_eq.is_empty() && eq * 10 >= self.pos_eq.len();
            self.group = Some(if !same_squares { 0 } else if self.cash_eq1 == Some(true) { 2 } else { 1 });
            if std::env::var("KAGG_GROUP_LOG").is_ok() {
                eprintln!("[group] seat {} step {} decided {:?} mirror_tol {:?}", v.obs.player, step, self.group, self.mirror_tol);
            }
        }
    }

    /// The profile to request at a day boundary (None until the group is known).
    pub fn profile(&self, day: i64) -> Option<usize> {
        if !self.fallback {
            return None;
        }
        self.group.map(|g| {
            if day >= END_DAY && self.mirror24 != Some(true) { self.endgame[g].unwrap_or(self.profiles[g]) } else { self.profiles[g] }
        })
    }

    /// The shield's mask for the learned policy's choice on `day` (None = no restriction).
    /// `last` is the profile the policy played yesterday: after an exact day-23 mirror it is the
    /// only one allowed from day 24 on (`mirror_keep`). Before the group is known (day 0) nothing
    /// is masked.
    pub fn mask(&self, day: i64, n_prof: usize, last: usize) -> Option<Vec<bool>> {
        let sh = self.shield.as_ref()?;
        let never = |mut m: Vec<bool>| {
            for &i in &sh.never {
                if i < n_prof {
                    m[i] = false;
                }
            }
            m
        };
        let Some(g) = self.group else {
            // day 0 (group not yet known): only the never-list applies
            return (!sh.never.is_empty()).then(|| never(vec![true; n_prof]));
        };
        if sh.mirror_keep && day >= END_DAY && self.mirror24 == Some(true) {
            let mut m = vec![false; n_prof];
            if last < n_prof {
                m[last] = true;
            }
            return Some(m);
        }
        let base = match sh.allow[g].as_ref() {
            Some(ids) => {
                let mut m = vec![false; n_prof];
                for &i in ids {
                    if i < n_prof {
                        m[i] = true;
                    }
                }
                m
            }
            None => vec![true; n_prof],
        };
        Some(never(base))
    }

    /// This day's jitter offset for the decided group.
    pub fn offset(&self, day: i64) -> i64 {
        let (Some(g), Some(h)) = (self.group, self.h) else { return 0 };
        let j = if day >= END_DAY { self.jitter_end[g] } else { self.jitter[g] }.max(0);
        if j == 0 {
            return 0;
        }
        let x = fnv(h, &format!("d{day}"));
        (x % (2 * j as u64 + 1)) as i64 - j
    }
}
