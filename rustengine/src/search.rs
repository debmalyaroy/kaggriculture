//! The Track-P searcher: amortised day-plan search on the bit-exact engine.
//!
//! CLOSED LOOP, by operator order. Every decision below is a function of the
//! `State` handed to `decide` -- which is itself reconstructed from the live
//! observation. Nothing is stored across a game except (a) today's expanded
//! plan and (b) the parameter vector the search currently likes for tomorrow,
//! both of which were derived from observations and are re-derived against the
//! real board every dawn. No tape, no route prefix, no router, no embedded
//! action sequence, not even as a fallback: the fallback is `PASS` plus the
//! standing sell queue.
//!
//! Structure (docs/history/trackp-search-design-2026-09-03.md):
//!   * §2  action abstraction  -> plan.rs (DayKnobs -> 24 turns)
//!   * §3  horizon             -> K = 6 days, roll to 720 from day 24
//!   * §4  value function      -> value.rs
//!   * §5  opponent model      -> plan::SkeletonPolicy (the measured field)
//!   * §7  the amortised loop  -> `Searcher::decide` below

use std::cell::RefCell;
use std::time::Instant;

use crate::engine::{self, PlayerAction};
use crate::plan::{
    self, DayKnobs, DayPlan, SkeletonPolicy, C_MELON, C_STRAW, C_WHEAT,
    LAST_DAY, N_CROP, N_PROD, P_EGG, P_FERT, P_MELON, P_MILK, P_STRAW, P_WHEAT,
    P_WOOL, TPD,
};
use crate::state::{State, EPISODE_STEPS};
use crate::value;

/// Substitute seed for rollouts. The episode seed is NOT in the observation,
/// so weed spawns and the 3-daily shop draw are not reproducible. Every
/// candidate in one search shares the same substitute, so the RANKING is
/// unbiased even though the rollout world is not the real one (§6).
const SEED_SUBS: [i64; 4] = [1_000_003, 2_718_281, 4_099_991, 9_181_819];

#[derive(Clone, Debug)]
pub struct SearchCfg {
    /// Lookahead in days. Rolls to step 720 instead when the horizon reaches
    /// the end of the game.
    pub k_days: i64,
    /// (1+lambda) hill-climb width.
    pub lambda: usize,
    /// Substitute-seed ensemble size (>1 only matters across a shop unlock).
    pub ensemble: usize,
    /// Hard cap on rollouts PER `decide` CALL, independent of the clock.
    /// Setting this (with a large `budget_ms`) makes a run reproducible, which
    /// a wall-clock budget can never be -- and `src/determinism.py`'s rule is
    /// that a paired A/B over a non-reproducible agent is invalid, not just
    /// noisy. The ms budget is the realistic measurement; the rollout budget
    /// is the one the paired tests use.
    pub max_rollouts: usize,
    /// Fraction of a dawn call's budget spent repairing TODAY's plan before
    /// the rest goes to tomorrow.
    pub repair_frac: f64,
    /// Opponent-model aggression used in rollout (1.0 = the field skeleton).
    pub opp_aggression: f64,
}

impl Default for SearchCfg {
    fn default() -> Self {
        SearchCfg {
            k_days: 6,
            lambda: 8,
            ensemble: 1,
            max_rollouts: 100_000,
            repair_frac: 0.5,
            opp_aggression: 1.0,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub rollouts: u64,
    pub sim_steps: u64,
    pub accepted: u64,
    pub decides: u64,
    /// Nanoseconds, split by phase.
    pub ns_root: u64,
    pub ns_rollout: u64,
    pub ns_exec: u64,
    pub ns_total: u64,
}

pub struct Searcher {
    pub me: usize,
    pub cfg: SearchCfg,
    pub stats: Stats,
    committed: Option<DayPlan>,
    pending: Option<(DayKnobs, f64)>,
    rng: u64,
    /// `stats.rollouts` at the start of the current `decide`, so `max_rollouts`
    /// is a per-turn cap rather than a per-game one.
    roll_start: u64,
}

impl Searcher {
    pub fn new(me: usize) -> Self {
        Searcher {
            me,
            cfg: SearchCfg::default(),
            stats: Stats::default(),
            committed: None,
            pending: None,
            rng: 0x9E37_79B9_7F4A_7C15 ^ (me as u64 + 1),
            roll_start: 0,
        }
    }

    pub fn with_cfg(me: usize, cfg: SearchCfg) -> Self {
        let mut s = Searcher::new(me);
        s.cfg = cfg;
        s
    }

    fn next_rand(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn rollouts_used(&self) -> usize {
        (self.stats.rollouts - self.roll_start) as usize
    }

    fn pick(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_rand() % n as u64) as usize }
    }

    /// One turn. Emits from the committed day plan in microseconds, then
    /// spends whatever budget remains amortising TOMORROW's plan.
    pub fn decide(&mut self, st: &State, me: usize, budget_ms: u64)
        -> PlayerAction
    {
        let t0 = Instant::now();
        self.me = me;
        self.stats.decides += 1;
        self.roll_start = self.stats.rollouts;
        let day = st.step / TPD;
        let dawn = self.committed.as_ref().map_or(true, |p| p.day != day);

        if dawn {
            // The plan we searched yesterday was searched against a PREDICTED
            // board. Re-seat the parameters on the real one -- which is the
            // whole reason the abstraction is parameters and not a tape.
            let knobs = match self.pending.take() {
                Some((k, _)) => k,
                None => DayKnobs::skeleton(day, st, me),
            };
            let repair_budget =
                (budget_ms as f64 * self.cfg.repair_frac) as u64;
            let knobs = self.repair(st, knobs, t0, repair_budget);
            self.committed = Some(plan::plan_day(st, me, &knobs));
        }

        let te = Instant::now();
        let action = {
            let plan = self.committed.as_mut().unwrap();
            plan::execute_turn(plan, st, me)
        };
        self.stats.ns_exec += te.elapsed().as_nanos() as u64;

        // Amortise: plan tomorrow across all 24 of today's turns.
        if st.step + 1 < EPISODE_STEPS {
            self.amortise(st, t0, budget_ms);
        }
        self.stats.ns_total += t0.elapsed().as_nanos() as u64;
        action
    }

    // ------------------------------------------------------------- search --

    /// Evaluate one candidate for the day that begins at `root` (which must be
    /// a dawn state): the candidate plays day 0 of the rollout, then BOTH
    /// seats continue on the field skeleton to the horizon.
    ///
    /// Judging the candidate under a fixed continuation is what makes a
    /// 26-dimensional daily search stable: it measures the LASTING value of
    /// tomorrow's decision instead of rewarding a plan that defers its cost to
    /// a day the search also controls.
    fn evaluate(&mut self, root: &State, knobs: &DayKnobs, sub: i64) -> f64 {
        let me = self.me;
        let t0 = Instant::now();
        let mut s = root.clone();
        s.seed = sub;
        let day0 = s.step / TPD;
        let end_day = day0 + self.cfg.k_days;
        let roll_to_end = end_day > LAST_DAY;
        let stop_step = if roll_to_end {
            EPISODE_STEPS
        } else {
            end_day * TPD
        };

        let mut my_plan = plan::plan_day(&s, me, knobs);
        let mut opp = SkeletonPolicy::with_aggression(
            1 - me, self.cfg.opp_aggression);
        let mut steps = 0u64;
        while s.step < stop_step {
            let d = s.step / TPD;
            if my_plan.day != d {
                let k = DayKnobs::skeleton(d, &s, me);
                my_plan = plan::plan_day(&s, me, &k);
            }
            let a_me = plan::execute_turn(&mut my_plan, &s, me);
            let a_op = opp.act(&s);
            let acts = if me == 0 { [a_me, a_op] } else { [a_op, a_me] };
            engine::step(&mut s, &acts);
            steps += 1;
        }
        self.stats.rollouts += 1;
        self.stats.sim_steps += steps;
        self.stats.ns_rollout += t0.elapsed().as_nanos() as u64;
        value::objective(&s, me)
    }

    fn evaluate_ens(&mut self, root: &State, knobs: &DayKnobs) -> f64 {
        let n = self.cfg.ensemble.clamp(1, SEED_SUBS.len());
        let mut acc = 0.0;
        for i in 0..n {
            acc += self.evaluate(root, knobs, SEED_SUBS[i]);
        }
        acc / n as f64
    }

    /// Repair TODAY's knobs against the real dawn board.
    fn repair(&mut self, st: &State, knobs: DayKnobs, t0: Instant,
              budget_ms: u64) -> DayKnobs
    {
        if budget_ms == 0 {
            return knobs;
        }
        let root = st.clone();
        let mut best = knobs;
        let mut best_v = self.evaluate_ens(&root, &best);
        let deadline = budget_ms as u128 * 1_000_000;
        while t0.elapsed().as_nanos() < deadline
            && self.rollouts_used() < self.cfg.max_rollouts
        {
            let cand = self.mutate(&best, st, self.me);
            let v = self.evaluate_ens(&root, &cand);
            if v > best_v {
                best_v = v;
                best = cand;
                self.stats.accepted += 1;
            }
        }
        best
    }

    /// Spend the rest of the turn's budget on TOMORROW's knobs.
    fn amortise(&mut self, st: &State, t0: Instant, budget_ms: u64) {
        let deadline = budget_ms as u128 * 1_000_000;
        if t0.elapsed().as_nanos() >= deadline {
            return;
        }
        // Root: simulate the rest of today from the REAL current state, our
        // seat on the committed plan, the opponent on the field skeleton. It
        // costs <= 24 steps and gets strictly more accurate as the day runs.
        let tr = Instant::now();
        let root = match self.roll_to_dawn(st) {
            Some(r) => r,
            None => return,
        };
        self.stats.ns_root += tr.elapsed().as_nanos() as u64;
        if root.step >= EPISODE_STEPS {
            return;
        }
        let day = root.step / TPD;

        let (mut best, mut best_v) = match self.pending.take() {
            Some((k, v)) => (k, v),
            None => {
                let k = DayKnobs::skeleton(day, &root, self.me);
                let v = self.evaluate_ens(&root, &k);
                (k, v)
            }
        };
        // The incumbent's value was measured against YESTERDAY's root; the
        // root has moved, so re-score it before comparing anything to it.
        best_v = self.evaluate_ens(&root, &best);

        let lambda = self.cfg.lambda.max(1);
        'outer: while t0.elapsed().as_nanos() < deadline {
            for _ in 0..lambda {
                if t0.elapsed().as_nanos() >= deadline
                    || self.rollouts_used() >= self.cfg.max_rollouts
                {
                    break 'outer;
                }
                let cand = self.mutate(&best, &root, self.me);
                let v = self.evaluate_ens(&root, &cand);
                if v > best_v {
                    best_v = v;
                    best = cand;
                    self.stats.accepted += 1;
                }
            }
        }
        self.pending = Some((best, best_v));
    }

    /// Simulate the remainder of today under the committed plan.
    fn roll_to_dawn(&mut self, st: &State) -> Option<State> {
        let me = self.me;
        let plan = self.committed.as_ref()?;
        let mut s = st.clone();
        s.seed = SEED_SUBS[0];
        let mut p = plan.clone();
        let mut opp = SkeletonPolicy::with_aggression(
            1 - me, self.cfg.opp_aggression);
        let target = ((st.step / TPD) + 1) * TPD;
        while s.step < target && s.step < EPISODE_STEPS {
            let a_me = plan::execute_turn(&mut p, &s, me);
            let a_op = opp.act(&s);
            let acts = if me == 0 { [a_me, a_op] } else { [a_op, a_me] };
            engine::step(&mut s, &acts);
            self.stats.sim_steps += 1;
        }
        Some(s)
    }

    // ------------------------------------------------------------- moves ---

    /// One or two typed perturbations of a day plan. The move table is the
    /// search's inductive bias: it moves along the axes the economy actually
    /// has (labour, land, herd, tiles, sell policy), never along raw ops.
    fn mutate(&mut self, k: &DayKnobs, st: &State, me: usize) -> DayKnobs {
        let mut c = k.clone();
        let n_moves = 1 + (self.next_rand() % 2) as usize;
        for _ in 0..n_moves {
            match self.pick(9) {
                0 => {
                    let d = if self.next_rand() & 1 == 0 { 1 } else { -1 };
                    c.hire = (c.hire + d).clamp(0, 10);
                }
                1 => c.buy_land = !c.buy_land,
                2 => {
                    let i = self.pick(3);
                    let d = if self.next_rand() & 1 == 0 { 1 } else { -1 };
                    c.buy[i] = (c.buy[i] + d).clamp(0, 4);
                }
                3 => {
                    // tiles: move in fours, the size of a wheat replant wave
                    let i = [C_WHEAT, C_STRAW, C_MELON][self.pick(3)];
                    let d: i8 = if self.next_rand() & 1 == 0 { 4 } else { -4 };
                    c.plant[i] = (c.plant[i] + d).clamp(0, 24);
                }
                4 => {
                    let i = self.pick(N_CROP);
                    let d: i8 = if self.next_rand() & 1 == 0 { 1 } else { -1 };
                    c.plant[i] = (c.plant[i] + d).clamp(0, 24);
                }
                5 => {
                    // sell cap: the LOW-price lever. -1 means uncapped.
                    let i = SELLABLE[self.pick(SELLABLE.len())];
                    let cur = if c.sell_cap[i] < 0 { 200 } else { c.sell_cap[i] };
                    let step = (cur / 4).max(4);
                    let up = self.next_rand() & 1 == 0;
                    let nv = if up { cur + step } else { cur - step };
                    c.sell_cap[i] = if nv >= 200 { -1 } else { nv.max(0) };
                }
                6 => {
                    // sell floor in sixteenths of base price: hold for a
                    // better quote, or dump regardless
                    let i = SELLABLE[self.pick(SELLABLE.len())];
                    let d: i16 = if self.next_rand() & 1 == 0 { 1 } else { -1 };
                    c.sell_floor[i] =
                        (c.sell_floor[i] as i16 + d).clamp(0, 24) as u8;
                }
                7 => c.fert_straw = !c.fert_straw,
                _ => {
                    if self.next_rand() & 1 == 0 {
                        let d = if self.next_rand() & 1 == 0 { 2 } else { -2 };
                        c.feed_buffer = (c.feed_buffer + d).clamp(0, 24);
                    } else {
                        let d = if self.next_rand() & 1 == 0 { 2 } else { -2 };
                        c.drop_at = (c.drop_at + d).clamp(2, 40);
                    }
                }
            }
        }
        // Cheap legality trims so the rollout is not wasted on a plan the
        // planner would discard anyway.
        let owned_extra = st.farms[me].unlocked_quadrants.len() - 1;
        if owned_extra >= 3 {
            c.buy_land = false;
        }
        c
    }
}

/// Products whose sell policy is worth searching. WHEAT and EGG are log-priced
/// (dump-proof) so their caps are left alone by the cap/floor moves; they are
/// still sold every turn above the feed reserve.
const SELLABLE: [usize; 6] = [P_MELON, P_STRAW, P_WOOL, P_MILK, P_FERT, P_EGG];

// ------------------------------------------------------------ convenience --

thread_local! {
    static TLS: RefCell<Option<Searcher>> = const { RefCell::new(None) };
}

/// Stateless entry point over a thread-local `Searcher` -- exactly the
/// signature Phase A's `kagg play` was asked to assume. Prefer holding a
/// `Searcher` yourself when you can; this exists so the bridge does not have
/// to.
pub fn decide(st: &State, me: usize, budget_ms: u64) -> PlayerAction {
    TLS.with(|c| {
        let mut b = c.borrow_mut();
        if b.as_ref().map_or(true, |s| s.me != me) || st.step == 0 {
            *b = Some(Searcher::new(me));
        }
        b.as_mut().unwrap().decide(st, me, budget_ms)
    })
}

/// Serialise a `PlayerAction` to the tape line format `service.rs` parses:
/// `farmer \t hand;hand;... \t order;order;...`.
pub fn action_to_line(a: &PlayerAction) -> String {
    fn unit(u: &crate::engine::UnitAction) -> String {
        if u.item.is_empty() {
            u.op.clone()
        } else if u.has_n {
            format!("{} {} {}", u.op, u.item, u.n)
        } else {
            format!("{} {}", u.op, u.item)
        }
    }
    let hands: Vec<String> = a.hands.iter().map(unit).collect();
    let orders: Vec<String> = a.market.iter().map(|o| o.join(" ")).collect();
    format!("{}\t{}\t{}", unit(&a.farmer), hands.join(";"), orders.join(";"))
}

/// Fill a plausible opponent `private` block from their PUBLIC board.
///
/// The interpreter hides the opponent's shed, seeds and carried inventories.
/// Phase A's bridge cannot observe them, so the rollout would otherwise model
/// an opponent who can never feed an animal or place a purchase. This credits
/// their shed with enough feed wheat and fertiliser to run the herd standing on
/// their board. It is a BELIEF, not an observation, and it biases us toward
/// under-valuing the opponent (§4).
pub fn seed_opponent_belief(st: &mut State, me: usize) {
    let opp = 1 - me;
    if st.private[opp].shed.sum() > 0 {
        return;
    }
    let c = plan::census(&st.farms[opp]);
    let herd: i64 = c.animals.len() as i64;
    st.private[opp].shed.add("WHEAT", (herd * 3).min(40));
    st.private[opp].shed.add("FERTILIZER", (herd / 2).min(12));
    for ci in 0..N_CROP {
        let _ = ci;
    }
    st.private[opp].seeds.add("WHEAT", 20);
}

/// Diagnostics for the harness: the knobs the searcher currently commits to.
impl Searcher {
    pub fn committed_knobs(&self) -> Option<DayKnobs> {
        self.committed.as_ref().map(|p| p.knobs.clone())
    }
    pub fn pending_value(&self) -> Option<f64> {
        self.pending.as_ref().map(|(_, v)| *v)
    }
}

#[allow(dead_code)]
const _ASSERT_PROD: usize = N_PROD;
#[allow(dead_code)]
const _ASSERT_WHEAT: usize = P_WHEAT;
