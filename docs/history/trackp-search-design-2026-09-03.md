# Track P Phase B — the search (2026-09-03)

> **SUPERSEDED AS A STRENGTH CLAIM (2026-09-03).** Phase A and Phase B were
> joined and measured against real opponents on the official interpreter:
> **0 wins in 40 cells**, and significantly worse than the seat it would
> replace. Read `docs/history/trackp-phase-ab-integration-2026-09-03.md` before
> believing any number in this document as evidence of ladder strength.
> The transport and engineering claims here still hold.


Lane: `trackp`, closed-loop. **SEAT LAW (operator, absolute): every decision is
derived from the live observation. No tape, no route prefix, no router, no
embedded action sequence, not even as a fallback.** Nothing in this document
relaxes that; the search is a function of the observed state and nothing else.

Phase A (a concurrent agent) owns the transport: `kagg play` in
`rustengine/src/main.rs`, `rustengine/src/service.rs`, the `main.py` ctypes /
stdio bridge and the Linux build. Phase B — this document — owns the SEARCH:
`rustengine/src/plan.rs`, `rustengine/src/value.rs`, `rustengine/src/search.rs`,
`rustengine/src/lib.rs`, `rustengine/src/bin/search_eval.rs`, and the offline
harness under `.local/trackp_search/`.

---

## 0. The entry point (the contract Phase A calls)

```rust
// rustengine/src/search.rs
pub struct Searcher { /* persists across turns; holds the committed day plan */ }

impl Searcher {
    pub fn new(me: usize) -> Self;

    /// Decide one turn. `st` is the CURRENT engine state as reconstructed from
    /// the observation; `budget_ms` is the wall-clock this call may burn.
    /// Emits the turn from the committed day plan (microseconds) and spends
    /// whatever budget is left amortising tomorrow's plan.
    pub fn decide(&mut self, st: &State, me: usize, budget_ms: u64) -> PlayerAction;
}

/// Stateless convenience over a thread-local Searcher, exactly the signature
/// the mission proposed. Phase A may call either.
pub fn decide(st: &State, me: usize, budget_ms: u64) -> PlayerAction;

/// Serialise to the tape line format service.rs already parses
/// (`farmer \t hand;hand;... \t order;order;...`).
pub fn action_to_line(a: &PlayerAction) -> String;

/// Fill a plausible opponent `private` block when the observation cannot.
pub fn seed_opponent_belief(st: &mut State, me: usize);
```

Three properties of `st` that Phase A must know, because the search is built
around them:

1. **`st.private[1 - me]` is a BELIEF, not an observation.** The opponent's
   shed, seeds and carried inventories are hidden by the interpreter. The
   bridge should leave them empty and call `seed_opponent_belief`, which
   credits the opponent's shed from what is standing on their (public) board.
2. **`st.seed` is a placeholder.** The episode seed is not in the observation,
   so weed spawns and the 3-daily shop draw are not reproducible in rollout.
   §6 says what the search does instead.
3. **Everything else is exact.** Both farms' tiles, both moneys, the shared
   market inventory and prices, and the unlocked shop list are all public, so
   the forward model of the *market* — the thing that decides these games — is
   exact for every already-unlocked shop.

`decide` never panics: every path has a fallback to `PASS` + the standing sell
queue, and the budget is checked with `Instant` after every rollout.

---

## 1. The objective — the single most important decision

### It is not "maximise own bank", and the data says so three ways

**(a) The whole field banks the same.** Median bank at day 29: top-10 88.6k,
ours 87.3k, ranks 200-400 89.4k. Median action skeleton is identical from rank
1 to rank 400 (2 land, ~9 cows, ~5 sheep, 187 wheat seeds, ~40 strawberry, 12
melon, ~290 hires, 17 pastures, 243 plants). A search that maximises own bank
is optimising the coordinate on which rank 1 and rank 400 are indistinguishable.

**(b) Bank is a shared-world quantity.** corr(own bank, opp bank) = 0.73-0.80;
P(opp < 75k | own < 75k) = 0.83-0.89. Most of the variance in own bank is
*world*, not skill. Formally, if own and opp have comparable variance sigma^2
and correlation rho,

    Var(own - opp) = 2 sigma^2 (1 - rho) = 0.40 .. 0.54 sigma^2
    Var(own)       =   sigma^2

so the **difference** carries the same decision content with 1.9-2.5x less
variance per rollout. Under a fixed compute budget that is a straight
2x improvement in how many candidate plans we can rank correctly. This is not
a stylistic preference; it is the reason the search can work at all at 300 ms.

**(c) The ladder pays wins, and the top-10 edge is entirely pairwise, and it is
concentrated where margins are thin.**

| own bank bin | ranks 1-10 win% | ranks 200-400 win% |
|---|---|---|
| < 90k  | **61.5%** | **48.0%** |
| 90-110k | 67-69% | 60% |
| >= 120k | 73% | 76% |

Independently confirmed: worlds with wool<50 & strawberry<50 at step 576 are
20.4% of games; the winner banks 67.4k there versus 95.3k elsewhere, and
P(margin < $3,000) is **0.435 in LOW-price worlds versus 0.352 in HIGH**. The
edge lives in the bin where the two banks are closest together.

### The surrogate

The true objective is `P(own_final > opp_final)`. With a deterministic forward
model and one opponent model we get one sample, not a probability, so we use a
smooth surrogate that has the right shape:

    V(s) = sigmoid( ( A_me(s) - A_opp(s) ) / s_t )

where `A(s)` is the horizon asset valuation of §4 and `s_t` is the residual
uncertainty in the FINAL margin, seen from time t.

`s_t` is calibrated from the observed margin distribution, not invented. If the
final margin is approximately zero-mean with scale sigma, then
`P(|margin| < 3000) = 2*Phi(3000/sigma) - 1`, so

    HIGH worlds: P = 0.352 -> 3000/sigma = 0.4565 -> sigma ~= $6,570
    LOW  worlds: P = 0.435 -> 3000/sigma = 0.5757 -> sigma ~= $5,210

We use `s_0 = $6,000` at game start and shrink it with the square root of the
remaining horizon, `s_t = s_0 * sqrt(days_left / 30)`, floored at $400.

Why this shape is the right one, in words:

* **Far ahead:** the sigmoid saturates, so an extra $10k of bank is worth
  almost nothing. The search stops taking risk to grow a lead it already has.
  A bank-maximiser would keep taking it — and in a shared world that means
  dumping product, which crashes the price for *both* seats and converts a
  comfortable win into a coin flip.
* **Near even:** the gradient is maximal. Exactly the LOW-price, thin-margin
  regime where the top-10 edge lives (61.5% vs 48.0%). Every dollar is priced
  at its true win value there, which is the `win_metric.flips()` insight
  expressed as a search objective rather than as a post-hoc report.
* **Far behind:** the gradient is small again and *convex on the way back*, so
  plans with more spread are preferred over plans with a better mean. That is
  correct — from 20k down, only variance wins.

Late in the game `s_t -> $400` and the sigmoid becomes a step: play to win the
comparison, not to bank. That is the endgame the leak table describes (top-10
carry 24 units of shed stock into day 29 and sell 3x more on the last day).

**One numerical guard, added after the first run measured it.** Past ~40 sigmas
the logistic is exactly 1.0 in f64, every candidate ties, the hill climb
accepts nothing, and the searcher silently degrades to the plain skeleton for
the rest of the game. That is not a hypothetical: on seed 4000 the objective
pinned at 1.000 from day 7 and the committed knobs from day 18 onward were the
untouched skeleton. `value.rs` therefore adds `1e-9` per dollar of edge as a
TIE-BREAKER. It is not a second objective: near an even game the logistic's own
gradient is ~4e-5 per dollar, four orders of magnitude larger, so the
tie-breaker can never outvote the win-probability term where that term has any
resolution at all. It only speaks where the primary term has gone numerically
silent.

### What this rejects

* `max own_bank` — rejected by (a), (b), (c).
* `max (own - opp)` linear — rejected by (c): it prices the 200,000th dollar
  the same as the dollar that flips a $500 loss into a win. It also
  systematically prefers dumping, because in the shared market a dump costs
  the opponent more revenue than it costs us only in the *mean*, never in the
  *win indicator*.
* Any objective computed on own bank alone with a "world difficulty" control.
  We tried the moral equivalent for two weeks (regime-conditioned bank
  targets) and the top-10 comparison in §1(a) says the level is not the lever.

---

## 2. The action abstraction

The raw per-turn action space is a farmer plus up to 10 hands, each choosing
one of ~15 ops against a 10x10 board, plus an ordered queue of 10 market
orders: well over 10^20 legal turns. **We never search it.** Three previous
closed-loop attempts (planner_v0 0-64, econ planner 0-6 vs seats) failed at
least partly by trying to be clever per-turn; per-turn judgment is retired as
a seat route and this design does not reopen it.

What we search is the **day plan**, expressed as a compact parameter vector
that a deterministic planner expands into 24 turns of concrete unit actions.

```rust
pub struct DayKnobs {
    pub hire: i8,              // hands to open today, 0..=10
    pub buy_land: bool,        // BUY_LAND this dawn
    pub buy: [i8; 3],          // COW, SHEEP, GOOSE purchases today
    pub plant: [i8; 5],        // new tiles today: WHEAT CARROT TOMATO STRAWBERRY MELON
    pub sell_cap: [i16; 9],    // per-product units offered today (-1 = uncapped)
    pub sell_floor: [u8; 9],   // hold below floor/16 of base price
    pub fert_straw: bool,      // fertiliser -> strawberry production eves
    pub feed_buffer: i8,       // days of feed wheat kept in the shed
    pub drop_at: i8,           // carried units before a unit detours to the shed
    pub harvest_at: i8,        // animal yield_units threshold to bother harvesting
}
```

~26 integers per day. The planner (`plan.rs::plan_day`) turns `DayKnobs` +
the real dawn board into:

* a priority-ordered market queue, spilled 10 orders per turn (the hard cap);
* one ordered job list per unit, assigned by a sequential nearest-fill over
  travel + op cost, with cargo (feed wheat, fertiliser, animals) resolved to a
  single batched `PICKUP` at the shed before the tour;
* a task pool that idle units grab from.

`plan.rs::execute_turn` then plays the plan one turn at a time, re-validating
every op against the tile the unit is actually standing on (a failed buy or a
late hire degrades to a skipped op, never to a drifting tour).

Three reasons this abstraction is the right one:

1. **What we search is what we play.** The rollout expands `DayKnobs` with the
   *same* `plan_day`/`execute_turn` code that plays the real turn. There is no
   plan/execute mismatch — the class of bug that made the two-pass compiler
   diverge from the world its own decisions created.
2. **Parameters re-seat; tapes do not.** At real dawn the board will differ
   from the one we predicted (the opponent did something else). A searched
   *parameter vector* is simply re-expanded against the real board. A searched
   *action sequence* would have to be repaired or abandoned. This is also why
   the abstraction does not violate the seat law: nothing is stored across the
   turn except numbers that were themselves derived from live observations.
3. **Neighbouring plans have correlated value**, so a cheap local search works.
   Raw op sequences do not have that property.

### The skeleton is the prior, not the answer

`DayKnobs::skeleton(day, board)` returns the measured field economy — the one
every cohort from rank 1 to rank 400 runs. The search starts there and hill-
climbs. That matters: it means the *worst* case of the search is the field
economy, and every accepted move is a measured improvement on it under the
objective of §1. It also means the search inherits the closed leak list (feed
wheat 142 not 414, fertiliser 62 not 248, 17 pastures, 9 cows, 5 sheep, PASS
turns ~520 not 926) as its starting point instead of rediscovering it.

---

## 3. Horizon, and why not full-game MCTS

A rollout to the horizon costs `(H - t)` engine steps. Measured on this box the
Rust engine runs 103k steps/s bare; with two policies in the loop the offline
harness measures the real figure (§8) and it is roughly 3-6 us/step for the
engine plus the planner amortised over the day. Kaggle's 1.6 vCPU core is
assumed **2x slower**, so budget ~50k policy-steps/s and keep total per-turn
work under 300 ms of a 1,000 ms `actTimeout`.

Per-turn full-game MCTS is not affordable *early*: at day 0 a single rollout is
720 steps and a tree of any width needs thousands. It is also not *desirable*,
for a reason that has nothing to do with speed: 25 days of rollout against an
approximate opponent model is fiction, and the deeper it goes the more
confidently wrong it is. The previous lane's own finding says the same thing
from the other side — games are decided on days 5-7 (per-turn win-prob AUC is
already 0.86 at day 6), so the decision-relevant horizon is short.

So: **K-day lookahead with a value function at the horizon, K = 6 by default,
adaptive at the ends.**

    day 0..1   K = 4    (the board is nearly empty; little to distinguish)
    day 2..23  K = 6    (>= one wheat replant cycle, 3 cow productions,
                         3 strawberry productions, 2 shop-unlock boundaries)
    day 24..29 roll to step 720 exactly, no value function

K = 6 is chosen from the game's own periods, not tuned into the seat: WHEAT
matures in 4 days, STRAWBERRY produces every 2 from day 10, COW every 2 from
day 8, SHEEP every 3 from day 6, shops unlock every 3. Six days is the shortest
horizon in which every one of those cycles closes at least once, so a plan
cannot look good merely by deferring cost past the horizon.

The last six days roll to the true end. That is deliberate: the value function
is least trustworthy exactly where the leak table says money is left on the
table (day-28 plantings that never sell, day-29 shed stock), and in that window
a full rollout is cheap (<= 144 steps).

Cost accounting, Kaggle-pessimistic at 20 us per policy-step:

    K = 6  ->  144 steps  ->  ~2.9 ms per candidate evaluation
    300 ms/turn budget    ->  ~100 evaluations per turn
    x 24 turns amortised  ->  ~2,400 evaluations per day-plan decision

A (1+lambda) hill climb with lambda = 8 gets ~300 generations over 26
dimensions. That is a real search, not a token one.

---

## 4. The value function at the horizon

`value.rs::asset_value(st, seat, day)` returns dollars:

    A = money
      + liquidation_value(shed)
      + standing_crop_value
      + standing_animal_value
      + realisable_seed_value

**`liquidation_value`** is the only honest way to price stock, and it is exact
rather than modelled: pricing is per-unit lockstep off shared inventory, so
selling `n` units of item `i` earns `sum_{k=0..n-1} price_i(inv_i + k)`. We
evaluate that sum with the engine's own `market::price`. A shed of 40 wool is
not worth 40 x quote; the search sees the walk-down and will refuse the dump
that a naive valuation invites. This is where the memory file's microstructure
(WHEAT/EGG log-priced and dump-proof; WOOL sq T=105, STRAWBERRY linear T=100,
MILK linear T=122 saturating fast; MELON quadratic above I0 and in no shop) is
*derived* rather than hard-coded — we price with the real curve and let the
search discover which products tolerate volume.

**`standing_crop_value`**: remaining productions the tile can still deliver
*before day 29*, times the marginal liquidation price, minus the water/labour
ops it will consume. A strawberry planted on day 26 yields nothing and is
valued at zero. This structurally kills the "planted tiles day 28: 30 vs 27"
leak without a rule.

**`standing_animal_value`**: remaining production days given `placed_day`,
`interval`, `max_held` and the days left, times marginal price, plus the
pending care bonus, minus 1 WHEAT of feed per remaining day. An animal at
`consecutive_unfed = 1` is one missed feed from escaping and is discounted by
the probability the plan actually feeds it. This structurally kills the
"starving animals day 10: 2 vs 0" leak.

**`realisable_seed_value`**: seeds are worth their crop's value only if days
AND free tiles remain to plant them; otherwise ~0. Kills "unplanted seed held
d15-d24: 20-35 vs 3".

Structures (empty coop/pasture) are valued at a nominal $0 — they save an op,
not money.

The objective of §1 is then `sigmoid((A_me - A_opp) / s_t)`.

### Honest statement of approximation error

* `A_opp` uses the *simulated* opponent's private block, which descends from
  our opponent model, not from observation. Its absolute level is wrong.
* It is wrong in *approximately the same way for every candidate plan*, because
  candidates differ mainly in our own actions. What survives the comparison is
  the part of `A_opp` that our plan actually moves — the shared market prices.
  That coupling is the part we want and the part the forward model gets right.
* The residual bias is toward under-valuing the opponent (their belief shed
  starts thin), which makes us slightly complacent. §7 mitigates.

---

## 5. The opponent model

The rollout steps BOTH seats — the engine has no choice, and that is the point:
one seat's selling moves the other's prices. The opponent seat is played by the
same `plan.rs` skeleton with `DayKnobs::skeleton`.

The justification is unusually strong and it is measured, not assumed: **the
whole field runs one economy.** The median action skeleton is identical for
ranks 1-10, ranks 200-400, and us. Modelling the opponent as the field skeleton
is modelling them as the measured median of every cohort on the ladder.

Where it is wrong, stated plainly:

* Real elite opponents are **open-loop tapes with world-specific branch points**
  (shop-unlock order at steps 88/120/153/160/216). They sell into shops that
  are actually open; our model sells on a price floor. So our model
  *under-captures demand* relative to a real top-10 tape, which makes prices in
  rollout slightly higher than they will really be, which makes us slightly too
  willing to hold stock.
* Self-play-style modelling — assuming the opponent is us — is an
  approximation, and a known-biased one: it makes the market look like a
  mirror when the real distribution is a mixture over families. We do not claim
  otherwise.
* The model is *reactive to prices*, which is the property that matters for the
  coupling. It is not reactive to *us* specifically (no opponent-of-opponent
  modelling, no recursion).

Mitigation available and implemented as a knob: `opp_aggression` scales the
opponent's sell caps and floors. `--opp-ensemble` evaluates each candidate
against a pessimistic and a neutral opponent and scores it by the worse of the
two. It costs 2x per candidate; the harness reports whether it is worth it.

---

## 6. The two things we cannot observe

**The seed.** Weed spawns (0.5% per empty tile per day) and the 3-daily shop
draw both come from `MT::for_day(seed, day)` and the seed is not in the
observation. Rollout therefore substitutes a fixed synthetic seed derived from
the current step. All candidates in one search share the same substitute, so
the *ranking* is unbiased by it even though the absolute rollout is a different
world. When the horizon crosses a shop-unlock boundary the searcher evaluates
over a small ensemble of substitute seeds (default 2) and averages, because
which shop unlocks genuinely changes which product is worth holding.

**The shop draw specifically — and why the weed fingerprint does not work.**
Weeds and shop draws share one stream: `for_day` draws one `random()` per empty
tile of farm 0, then farm 1, then `choice(SHOPS)`. So the number of draws
consumed before the choice depends on *both* players' empty-tile counts, and
competitors do use rival weed geometry as a fingerprint. We looked at whether
the search can exploit it and the answer is **no**, for an information-theoretic
reason worth recording so nobody re-opens it:

* Each tile's observed outcome is a Bernoulli(0.005) draw. Its entropy is
  H(0.005) = 0.045 bits. About 200 empty tiles across both farms per day gives
  ~9 bits of information about the stream per day.
* Identifying the draw requires either the seed (>= 32 bits, so >= 4 days of
  perfect weed observation, by which time 1-2 shops have already unlocked and
  are simply *visible*), or MT19937 state recovery (624 x 32 bits from
  consecutive raw outputs, which 0.5% threshold tests do not provide).
* Brute-forcing the seed at runtime is ~2^32 x 200 draws inside a 1-second
  turn. Not affordable by four orders of magnitude.

What *is* free and is exploited: **the unlocked shop list is public the moment
it changes**, and every already-unlocked shop's consumption (1 of each listed
product every 4 steps, doubled for single-product shops) is exact in the
forward model. A tape has to guess at compile time; the search reads it. That
is the concrete, defensible closed-loop advantage over an open-loop tape, and
it is the reason the lane is worth building at all.

---

## 7. The search loop (amortised)

Per game-day, the searcher decides ONE `DayKnobs` — tomorrow's — using all 24
of today's turns.

```
on decide(st, me, budget_ms):
  t0 = now()
  if day(st) != committed_day:                  # dawn
      committed = plan_day(st, me, pending_best or skeleton(day))
      committed_day = day(st)
      pending_best = None
      repair_search(st, budget * 0.5)           # the real board != the predicted one
  action = execute_turn(&mut committed, st, me) # microseconds
  amortised_search(st, me, remaining budget)
  return action
```

`amortised_search`:

1. **Root construction.** Simulate the rest of today from the *real current*
   state: our seat on `committed`, opponent on the skeleton. Cost <= 24 steps
   (~0.5 ms). Re-done every turn, so the root gets strictly more accurate as
   the day proceeds and by turn 20 it is nearly the true dawn board.
2. **Evaluate the incumbent** (`pending_best`, else `skeleton(day+1)`) from the
   root: expand it for day d+1, then run the skeleton for both seats through
   day d+K, then value.
3. **(1+lambda) hill climb.** Sample `lambda` neighbours by perturbing 1-2
   coordinates from a typed move table (a hire +/-1, a land buy toggled, an
   animal added, +/-4 tiles of one crop, a sell cap raised or lowered by one
   step, a sell floor moved one notch, fertiliser toggled). Keep the best if it
   beats the incumbent. Repeat until the budget is gone.
4. **Store** the incumbent as `pending_best`.

Judging a candidate: **only day d+1 uses the candidate knobs**; days d+2..d+K
use the skeleton for both seats. This is deliberate. It measures the *lasting*
value of tomorrow's decision under a fixed continuation, instead of rewarding a
plan that merely defers its cost to a day the search also controls. It is the
same discipline as a one-step policy-improvement operator over a fixed base
policy, and it is what makes a 26-dimensional daily search stable.

Search-quality curve, profile, and the time split are reported by the harness
(§8), so "is the search actually using the compute" is a measurement, not a
claim.

---

## 8. The offline harness

`rustengine/src/bin/search_eval.rs` + driver `.local/trackp_search/run_eval.py`.
Headless: no Kaggle, no transport, no Python agent in the loop.

    kagg-search-eval --seeds 4000:4032 --budget-ms 150 --k 6 --lambda 8 --both-seats

Reports, per configuration:

* own bank, opp bank, win / draw / loss, score (draws 0.5);
* the **LOW/HIGH price-regime split**, computed inside the harness with
  `src/kaggriculture/measure/band_panel.py`'s frozen spec: the statistic is
  `mean over {MILK, STRAWBERRY, EGG, WOOL} of median(price / base)` over the
  window's turns; PRIMARY window d9-12, threshold **1.063**; EARLY window d3-5,
  threshold **1.1065** (from `.local/band_panel/regime.json`, calibrated on
  8,000 ladder traces, d9-12 AUC 0.832 / Cohen d 1.35, median bank 67.3k LOW vs
  95.0k HIGH). Win% is reported per regime, and the harness refuses to report a
  regime win% when a regime has fewer than 6 cells;
* `--budget-sweep 10,50,150,300` for the search-quality curve;
* `--profile` for the time split (root construction / rollout steps / planning /
  valuation) and rollouts-per-turn.

Opponents available in Phase B: the field skeleton, three structurally
different skeleton economies (`--real-style geese|wheat|melon`) that the search
has never simulated, a sell-aggression knob, and the searcher against itself at
an asymmetric budget. **`data/gauntlet/pub_v16rc5.py` and the other Python
gauntlet agents cannot be driven from Phase B** — they need a Python agent and
a Rust searcher in the same match, which is exactly what Phase A's `kagg play`
provides. That is the first Phase-A integration test, not a Phase-B claim, and
this document does not report a number against them.

### Two limitations of this substrate, both measured

**1. The absolute d9-12 LOW regime never occurs here.** Across 32 control cells
(skeleton mirror) and every searcher configuration, ZERO cells fall below the
frozen d9-12 threshold of 1.0630. The reason is structural, not a bug: the
threshold is calibrated on ladder traces whose games bank an 88k median, and a
skeleton mirror banks 60k, so the four tracked products are never glutted to
ladder depth. Worse for this particular statistic, the field skeleton buys no
geese at all, so EGG sits permanently on its (1.32.7 hinge) SCARCITY side with
a ratio above 1 and drags the four-product mean above the cut in every cell.
A regime readout with 0 LOW cells is not a LOW result, it is no result. The
harness therefore reports three splits and the report names which one it is
using: absolute d9-12 (no coverage here), absolute d3-5 (does split, ~31% LOW,
and `band_panel.py` documents it as usable at Cohen d 1.36 / AUC 0.734), and
the within-run relative split at this run's own median, exactly as
`band_panel.py`'s "regime REL" row does. **The LOW-regime gate is therefore
answered on d3-5 and REL, not on d9-12, and that is a weaker instrument than
the ladder panel.**

**2. Wall-clock budgets are not reproducible.** `budget_ms` means the number of
rollouts depends on machine load, so two runs of the same seed differ (measured:
seed 4000 at 50 ms banked 81,552 on one run and 105,108 on the next).
`src/kaggriculture/measure/determinism.py`'s rule is that a paired A/B over a non-reproducible agent
is invalid, not merely noisy. The harness therefore also has `--rollouts N`, a
fixed per-turn evaluation budget that IS reproducible; paired tests use that,
and the wall-clock curve is reported as the realistic-but-noisy measurement it
is.

---

## 9. What we will NOT do, and why

* **No per-turn MCTS over raw unit ops.** Branching factor >10^20 and the lane
  already measured per-turn judgment at 0-64 and 0-6.
* **No full-game rollouts from early days.** Cost, and 25 days against an
  approximate opponent is confident fiction.
* **No learned value function or learned policy.** `train_gates.policy_allowed()`
  is sign-tested and there is no fresh positive evidence; a GBM/NN here would
  also make the searcher unauditable. The value function is arithmetic in the
  engine's own price curve, so every number it produces can be checked by hand.
* **No RNG inversion / seed recovery / weed-geometry shop prediction.** §6.
* **No hard-coded product rules** ("never dump wool"). The forward model prices
  the walk-down exactly; a hard rule would be a second, driftable source of
  truth. The memory file's microstructure is used to *check* the search's
  behaviour, not to constrain it.
* **No engine changes.** `engine.rs`, `market.rs`, `rules.rs`, `mt19937.rs`,
  `state.rs` are bit-exact and are also the bandit lane's PRE-RANKER. Phase B
  adds files; it does not edit those. `tests/test_rust_engine.py` is run to
  prove it.
* **No sell-timing "adaptive" layer as a separate module.** `_ADAPT_SELL` is
  RETIRED (2026-08-18, five measurements, zero wins). Sell timing here is not a
  layer bolted onto a tape; it is a coordinate of the searched day plan,
  measured by the same paired test as everything else. If it does not pay, the
  search will not select it.
* **No claim that this beats anything until §8 has been run.** Three previous
  closed-loop attempts failed. The gate is in the report, and a negative with
  numbers is the useful outcome.

---

## 9b. Throughput actually measured

Measured by the harness on this box (24 logical cores, 8 harness threads):

    13.6 - 15.8 us per SIMULATED step  (engine + BOTH policies replanned)
    -> ~2.0 ms per K=6 rollout (144 steps)
    -> 6.5 rollouts/turn at 10 ms, 31.9 at 50 ms
    execute_turn (the part that plays the real turn): 0.005 - 0.008 ms

The bare engine benches at 103k steps/s (9.7 us/step), so the two closed-loop
policies cost ~5 us/step on top -- the day planner amortised over 24 turns is
cheap relative to the engine itself. Assuming Kaggle's 1.6 vCPU core is 2x
slower, **dev 150 ms is the right stand-in for a Kaggle 300 ms budget**, and
that is how the report below reads the curve.

## 10. Gate

The searcher graduates from Phase B when, at a Kaggle-realistic budget
(<= 300 ms/turn, single core):

1. it beats `DayKnobs::skeleton` head-to-head on a fixed seed set, both seats,
   sign-tested (`win_metric.paired_test` semantics: McNemar exact on discordant
   pairs); **and**
2. it holds up in LOW-price worlds specifically — LOW-regime score >= its
   HIGH-regime score is not required, but LOW-regime score must be >= 0.60,
   the measured top-10 level; **and**
3. the search-quality curve is monotone in budget over 10 -> 300 ms. A flat
   curve means the compute is not being used and the result is luck.

Failing any of the three is reported as a failure with the numbers.

---

## 11. Measured, 2026-09-03

All numbers from `rustengine/target/release/search_eval`, 16 seeds x both
seats = 32 paired cells per row, 8 harness threads, K = 6, lambda = 8,
ensemble = 1. Raw output in `.local/trackp_search/`.

### 11.1 The budget curve vs the field skeleton

| configuration | cells | score | W-D-L | own median | opp median | edge (mean) | rollouts/turn |
|---|---|---|---|---|---|---|---|
| CONTROL skeleton vs skeleton | 32 | 0.500 | 12-8-12 | 59,933 | 59,933 | 0 | - |
| searcher @ 10 ms | 32 | 0.938 | 30-0-2 | 79,368 | 56,453 | +25,207 | 6.5 |
| searcher @ 50 ms | 32 | 0.938 | 30-0-2 | 74,234 | 45,216 | +29,786 | 31.9 |
| searcher @ 150 ms | 32 | 0.938 | 30-0-2 | 81,773 | 38,964 | +42,061 | 65.2 |
| searcher @ 300 ms | 32 | **1.000** | **32-0-0** | **86,850** | 38,416 | **+47,578** | 142.7 |

Paired, same seeds and seats: 50 vs 10 ms 2+/2- (p 1.0000); 150 vs 50 ms
2+/2- (p 1.0000); 300 vs 150 ms **2+/0-** (p 0.5000, one-directional but
underpowered on 2 discordant pairs).

Reading it honestly:

* **The search is being used, and more of it buys more edge.** The dollar edge
  is strictly monotone in budget (+25.2k -> +29.8k -> +42.1k -> +47.6k),
  rollouts/turn scale linearly with the clock (6.5 -> 31.9 -> 65.2 -> 142.7),
  and accepted hill-climb moves per game go 7,001 -> 20,567 -> 29,231 ->
  35,881. Gate condition 3 (a monotone curve) **passes on the edge**.
* **The win rate is saturated over most of the range** (30-0-2 at 10, 50 and
  150 ms) and only moves at the top (32-0-0 at 300 ms). A score that is already
  0.938 at 10 ms cannot show much of a curve, which is why §8's harness also
  runs searcher(B) vs searcher(10 ms).
* **Own bank is not monotone** (79.4k -> 74.2k -> 81.8k -> 86.9k) and the dip
  is the objective working as designed, not a defect: the search buys
  `P(own > opp)`, and at 50 ms it took a line that cut the opponent from 56.5k
  to 45.2k while giving up 5k of its own. A bank-maximiser would have refused
  that trade and won the same 30 games.
* **Time is going where it should.** At 300 ms: 288.2 ms in rollouts, 0.3 ms
  building the root, and 0.005 ms actually playing the turn. Nothing is being
  wasted on bookkeeping.

### 11.2 What it does not yet show

* **The 300 ms row reaches the field's economy; the cheaper rows do not.**
  86.9k median bank against the ladder's 88.6k field median is level; 79-82k at
  10-150 ms is not. And the baseline is soft either way: this Rust skeleton
  banks 60k in mirror where a ladder game banks 88k, so beating it is a lower
  bar than beating the field, and no claim about ladder strength follows from
  these rows.
* **Seed sensitivity is real and the 16-seed window is kind.** The skeleton
  mirror control banks a 59.9k median on seeds 4000-4015 but only 53.1k on
  4000-4039. Sixteen seeds is not enough to quote a bank.

### 11.3 Wider seed set, 40 seeds x both seats = 80 cells, 50 ms

| configuration | cells | score | W-D-L | own median | opp median |
|---|---|---|---|---|---|
| CONTROL skeleton vs skeleton | 80 | 0.500 | 32-16-32 | 53,118 | 53,118 |
| searcher @ 50 ms | 80 | **0.988** | 79-0-1 | 74,290 | 37,512 |

Regime split (see §8 limitation 1 for why d9-12 is not the readout here):

| split | LOW cells | LOW score | LOW median bank | HIGH cells | HIGH score | HIGH median bank |
|---|---|---|---|---|---|---|
| d9-12 absolute | 1 | (no coverage) | - | 79 | 0.987 | 74,534 |
| d3-5 absolute (thr 1.1065) | 33 | **0.970** | 65,014 | 47 | 1.000 | 77,549 |
| within-run REL (thr 1.2092) | 40 | **1.000** | 64,187 | 40 | 0.975 | 80,755 |

The relative split is doing real work: on the CONTROL run its LOW half banks
47.5k against the HIGH half's 65.8k, so it is separating genuinely poorer
worlds even though none of them reach the ladder's absolute cut.

**Gate condition 2 (LOW-regime score >= 0.60) passes on both usable splits**
— 0.970 on d3-5 (33 cells) and 1.000 on REL (40 cells) — against this
opponent, on a weaker instrument than the ladder panel.

### 11.4 TRANSFER: opponents the search never simulates

The searcher's internal opponent model is the field skeleton. Beating exactly
the policy you model is the classic exploiter result and proves nothing, so
each row below is a structurally different economy the search has never
simulated. 16 seeds x both seats = 32 cells, 50 ms.

| real opponent | score | W-D-L | own median | opp median | edge (mean) |
|---|---|---|---|---|---|
| Field (the MODELLED one, 80 cells) | 0.988 | 79-0-1 | 74,290 | 37,512 | +33,969 |
| Geese / egg economy | 1.000 | 32-0-0 | 99,111 | 21,834 | +74,673 |
| Melon-heavy | 1.000 | 32-0-0 | 83,114 | 36,581 | +41,795 |
| Wheat monoculture, uncapped sells | **0.875** | 28-0-4 | 63,125 | 49,098 | +13,831 |

**It transfers.** The searcher is not an exploiter of one fixed policy: it
beats three economies it has never modelled, two of them cleanly.

**And the informative row is the last one.** The wheat monoculture with every
sell cap removed is the opponent that gluts the market, and it is by far the
hardest: our score falls 0.988 -> 0.875 and our edge collapses from +34.0k to
+13.8k, with the opponent banking 49.1k instead of 37.5k. That is exactly the
thin-margin, low-price world the whole 2800 plan says the top-10 edge lives in
(61.5% vs 48.0% below 90k). **The search is weakest precisely where the
competition is decided**, and closing that is the obvious next work item.

### 11.5 The search-quality curve, measured against a fixed searcher

Against the skeleton the score saturates, so the curve is measured as
searcher(B) vs a fixed searcher at 10 ms/turn. 8 seeds x both seats = 16 cells.

| budget | score | W-D-L | own median | opp median | rollouts/turn | accepted moves/game |
|---|---|---|---|---|---|---|
| 50 ms | 0.562 | 9-0-7 | 64,160 | 56,301 | 23.4 | 9,145 |
| 150 ms | 0.688 | 11-0-5 | 75,388 | 63,507 | 77.2 | 15,241 |
| 300 ms | 0.750 | 12-0-4 | 80,497 | 69,031 | 147.2 | 17,779 |

**Strictly monotone in budget on the win rate, and gate condition 3 passes.**
The paired McNemar tests between adjacent rows are individually underpowered
(150 vs 50: 5+/3-, p 0.7266; 300 vs 150: 4+/3-, p 1.0000) — 16 cells cannot
resolve a step this size, and the honest statement is that the LEVEL of each
row is measured and the STEP between adjacent rows is not. The trend across
three rows is one-directional and the mean bank deltas are +11.5k and +4.9k.

This is also the sober number to carry forward: 5x the compute buys 0.562 and
30x buys 0.750 head-to-head, not the 0.99 the skeleton comparison suggests.

### 11.6 Profile

At 300 ms/turn: 288.2 ms in rollouts, 0.3 ms building the root, **0.005 ms
actually playing the turn**. 17-19 us per simulated step including BOTH
policies replanning; the bare engine benches at 9.7 us, so the two closed-loop
day planners cost ~8 us/step between them. Nothing measurable goes anywhere
else. On a Kaggle core assumed 2x slower this is ~70 K=6 rollouts inside a
300 ms budget, against the ~147 measured here — i.e. **dev 150 ms is the
Kaggle-300 ms row**, and that row is 0.688 head-to-head and 0.938 / 81.8k
against the skeleton.

### 11.7 Out-of-window spot checks

Ten seeds outside the tuning window, played through the Phase-A entry point
(`search::decide`, seat 0, 300 ms) against the field skeleton, with every
action serialised and re-parsed through the tape grammar before it reaches the
engine: **10 wins, 0 losses**, own banks 38.4k / 46.1k / 66.6k / 78.0k / 83.0k
/ 95.7k / 106.1k / 107.7k / 109.8k / 110.7k (median ~89k). Seat 0 only, so
this carries no seat-bias information; the paired tables above are the
measurement.

### 11.8 Verdict against the gate

| gate | result |
|---|---|
| 1. beats the deterministic skeleton head-to-head, fixed seeds, both seats | **PASS** — 0.988 (79-0-1) on 80 cells at 50 ms; 1.000 (32-0-0) at 300 ms on 32 cells |
| 2. holds up in LOW-price worlds | **PASS on the available instrument** — 0.970 (d3-5, 33 cells) and 1.000 (REL, 40 cells). NOT tested on the ladder-calibrated d9-12 cut, which has zero coverage in this substrate |
| 3. monotone search-quality curve 10 -> 300 ms | **PASS** — 0.562 / 0.688 / 0.750 head-to-head, and the dollar edge vs the skeleton is strictly monotone (+25.2k / +29.8k / +42.1k / +47.6k) |

**What this does NOT establish, stated plainly.** None of it is a ladder claim.
The opponents are Rust policies, the strongest of which banks 53-60k in mirror
against a ladder median of 88k. The searcher's own 74-87k median bank only
reaches field level at the top budget. The one opponent that plays the
market-glutting style the real low-price worlds are made of took our score down
to 0.875. And Phase B cannot drive `pub_v16rc5.py` or any other real gauntlet
agent — that is Phase A's first integration test and the number that will
actually matter.

### 11.9 Next work items, in priority order

1. **The glutted-market case.** Score 0.875 and edge +13.8k against the wheat
   dumper versus 0.988 / +34.0k against the field. Candidate causes: the
   horizon value function prices shed stock at a liquidation walk-down that
   assumes WE are the only seller, and the opponent model never dumps. Fixing
   the first is arithmetic (debit both liquidations against one inventory);
   fixing the second is the opponent ensemble (`--opp-ensemble`, already
   implemented, not yet measured).
2. **A stronger baseline.** The Rust skeleton banks 53k in mirror; the Python
   `skeleton_template.py` it is modelled on is better tuned. Until the baseline
   is at field level, "beats the baseline" understates how far there is to go.
3. **Power.** 16 cells cannot resolve adjacent steps of the budget curve. The
   paired tests need 60+ cells, and the reproducible `--rollouts` mode (not the
   wall clock) should be the substrate for them.
4. **The Phase-A integration test**: the searcher against the Python gauntlet,
   then the band panels. Nothing here substitutes for it.
* **The opponent in 11.1 IS the search's internal model.** That is the classic
  exploiter trap and the reason §8 added structurally different opponents
  (`--real-style geese|wheat|melon`) that the search never simulates.
* **The ladder-calibrated LOW regime has no coverage here** (§8, limitation 1),
  so the LOW gate is answered on the weaker d3-5 and relative splits.
