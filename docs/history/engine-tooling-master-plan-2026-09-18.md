# Master plan: harness, OR tape-generator, crown tournament (2026-09-18)

**Scope:** a single detailed plan (NO IMPLEMENTATION — planning only) covering four
requested items and how they interlock:
- **A. Harness revamp** (detail in `docs/history/harness-revamp-plan-2026-09-18.md`; summarized here)
- **B. World Generator + Tape Generator + Simulator** — a Rust OR/MILP pipeline that
  solves for a max-Cash₇₂₀ production tape
- **C. Crown tournament** — is it needed, how to improve, does a Rust port help
- **Solver decision** for B (Rust SCIP `russcip` vs alternatives)

Deadline context: ~1 week; submissions are well short of optimal (we rate ~2140,
top-10 is ~2900). Everything below is prioritized against that.

---

## 0. The through-line: fidelity is the meta-bottleneck

All four items share one root problem this project keeps hitting: **we cannot
trust a local number.** The Rust *engine* is bit-exact, but the *reactive
observation* served to agents is a hand-rolled subset, the *panel* is saturated
and desyncs, and the *crown* runs on that substrate. So: the OR tape-generator
would produce candidates we can't rank; the crown can't discriminate; and every
"improvement" is a coin flip (v58 read 0.947 offline, lost 0-16 live).

**Therefore the plan's spine is: make the substrate faithful (A) → make the
selector trustworthy (C) → then the OR generator (B) has something that can prove
it.** Speed is NOT the bottleneck (serve is already 12× official); fidelity is.

```
   B. World Gen (Rust+solver) ─► Tape Gen (Rust) ─► base_tape ─┐
        max Cash₇₂₀, exact economy    → .tape (TSV)            ▼
                                                      mbandit reactive shell
   Simulator = rustengine (DONE) ◄─ VALIDATE tape, real Cash₇₂₀ ─┤ (config-driven)
        bit-exact ground truth                                   ▼
                                              A. Faithful serve harness
                                                         ▼
                                              C. Crown tournament (selector) ─► ship
```

---

## 0.5 TOP-10 TARGET — the two-slot hybrid (the spine for shipping)

Goal: **2140 → ~2900 (top 10)** by 2026-09-23. The two research docs
(`research/Advanced Algorithms and Final Agent Architecture...`,
`research/Dynamic Labor Routing and Tape Synchronization...`) converge on a
**two-slot hybrid**, which fits our exactly-2 active submission slots and the
Bradley-Terry ladder (one stable floor + one high-variance contender):

**SLOT 1 — "Sparse Closed-Loop" Anchor** (target: stable 2400-2500 floor)
- = **our EXISTING architecture** (proven/OR tape + reactive market shell). The
  doc's Slot 1 (kaitofukami *v41 sparse-closed-loop*, boatlee *hysteresis* — both
  in our Task-1 public panel) is precisely what we already ship → direction confirmed.
- Upgrades: (a) stronger **OR base economy** (Item B); (b) micro-routing as
  **PC-TAPF** = Hungarian-assignment + Conflict-Based-Search (formalizes Item B2:
  precedence DAG till→plant→water→harvest, flowtime-min, conflict-free); (c)
  **dependency-directed plan-repair** shell — annotate the tape with causal
  dependencies so a stochastic shock (weed spawn, shop unlock, land buy, mirror
  front-run) patches only the fractured tasks locally (regression search +
  localized Hungarian-CBS) instead of ad-hoc guardrails or a full replan. This is
  the principled generalization of our reactive rails and directly attacks the
  "trajectory-stitching" fragility. Role: the rating floor that cannot make an
  unforced biological/labor error.

**SLOT 2 — "Adaptive Predator" = BC-warmup → self-play RL** (target: 2800+)
This is **the leaderboard topper's proven method** (Michal1337/pkmn-kaggle) AND the
field's confirmed medal recipe (§0.7: SNORLAX reached silver with BC→RL). Our earlier
"BC-seat dead (0.000)" was our OWN weak attempt on the broken static substrate — not
evidence the method fails. **CRITICAL (per §0.7 field evidence): the RL operates at
the MACRO / daily-plan level (~30 decisions), driving the SHARED deterministic
micro-executor (tape + PC-TAPF routing + plan-repair) — NOT per-turn** (per-turn RL
plateaus dumb at 80k). Full pipeline (task list, Workstream E):
- **Foundation = train==test.** RL is only valid if the training env is byte-identical
  to the ladder. Our Rust engine + faithful obs (Track A) IS that foundation — the
  topper's "C-native encoder byte-identical to Python" is exactly our fidelity work.
- **Featurization:** flatten the documented obs (both farms, market inv+prices, town,
  private) + per-tile / per-option encodings + world-signature (shop unlocks); a
  byte-identical Rust encoder with a parity-tested Python reference.
- **Action policy = masked pointer/autoregressive transformer** over the LEGAL
  (verb, target, qty) option list per unit + market queue (~10–20M params, as topper).
  Legality mask from engine rules (cash, shed, tile, adjacency/routability, seeds,
  ≤10 orders) applied pre-softmax.
- **BC warmup:** DuckDB-distill top-100/200 replays → per-decision (obs, action,
  return, world-sig) for winner/high-rated seats; reconstruct obs via deterministic
  replay; masked cross-entropy to imitate expert options. Feasible on RTX 4060 / Kaggle GPU.
- **Self-play PPO fine-tune:** two-sided (both seats learn); reward = terminal
  win/loss (zero-sum) + optional bank-margin shaping (reward discipline: who gets paid
  on both sides); **opponent pool** = frozen self-checkpoints + top-100/200 replay
  agents + public tape agents (v58, the 2 strong tapes), rating-weighted; **teacher-KL**
  to prior checkpoint; **gated promotion** (only if it beats the pool on a
  frozen-reference battery); GAE / trajectory-stitching for stochastic recovery.
- **Eval:** frozen-reference batteries (≥N games, replicated seeds, banded panel);
  **live ladder = primary signal.** Gate vs Slot 1 + strong references before shipping.
- **Packaging:** export weights + masked CPU inference runtime, <1s/turn (small model).
- **Compute is the real tradeoff (operator decides):** topper used 4×H200 / 11B steps;
  we have RTX 4060 8GB + a fast Rust engine. Options: (i) scaled local self-play to a
  feasible step budget, (ii) Kaggle/cloud GPU bursts, (iii) BC-only (no RL) as a floor.
- **Cheaper learned variants (NOT instead of — additional options):** economy/opening
  DISPATCHER (pick OR-economy per world) and market-timing predictor — smaller, and
  usable even if full RL is descoped.
- **Fallback:** if the learned Slot 2 doesn't beat Slot 1 on the faithful gate by the
  deadline, Slot 2 ships a diversified OR `route2` (second-slot rule) — but this is a
  fallback, not the plan.

**Data distillation (feeds Slot 2 + the panel):** DuckDB out-of-core over the
**already-downloaded top-100/200 parquet replays** → filter winner seats →
`(state, action, return)` corpus → BC / dispatcher training. We already hold the
replays in parquet (10,576+), so this is a query, not a 10 GB JSON crunch.
Hardware reality (15.7 GB RAM, RTX 4060 8 GB): BC (supervised) is feasible on
Kaggle's free GPU; **IMPALA RL fine-tune is a STRETCH goal, not a pillar.**

**Action masking (both docs):** a legality mask from the deterministic state
(cash, shed cap, tile occupancy, routability) applied to any learned component —
prevents illegal/fatal moves, shrinks the action space, reuses the engine's
legality rules, ties into the harness.

**Deliberately NOT adopted from the docs (over-engineering / off-discipline for a
1-week window):** the ArcWake async engine rewrite, Kolmogorov/MDL motif
compression, generalized-planning-program control flow, and the full IMPALA RL
cluster. The CORE useful ideas we DO take: PC-TAPF routing, dependency-directed
plan-repair, the two-slot split, DuckDB distillation, and action masking.

**Top-10 causal chain:** the rating gap is **ECONOMY** (production volume). Slot 1's
stronger OR economy + robust plan-repair is the **primary, certain** lever; Slot 2's
diversity + optional learned dispatcher is the **variance** push. Both are only
trustworthy because of the faithful harness (A) + crown (C). Nothing ships unvalidated.

---

## 0.6 BASELINE CORRECTION — v56y is NOT the reference; distrust static numbers

**Operator correction (2026-09-18).** Our prior "measured evidence" was taken
against **STATIC opponents / the mis-ranking serve substrate**, so it is unreliable
and produced **false negatives**:
- **v58 OUTPERFORMS v56y on the real ladder.** The earlier "v58 0-8 vs v56y, weak
  economy" was a measurement artifact. The ONLY real v58 defect was the `__file__`
  load crash (fixed); the economy is NOT weak.
- **Two public TAPE-based agents outperform v56y** (operator-tested). The tape-based
  Slot-1 ceiling is well ABOVE v56y.
- Therefore the "paid-for lessons" from the same static substrate — **"BC-seat dead
  (0.000)", "offline≠ladder", "v58 regressed"** — are SUSPECT and must be RE-TESTED
  on the faithful harness, never carried as settled fact.

**Root incoherence (the mistake):** the plan says serve mis-ranks reactive agents and
that "bank in isolation" (v56y's 145k) is a static metric — yet I quoted those same
static numbers as ground truth, anchoring everything on v56y. The **LADDER (contested
market) is the oracle**; isolated bank is not.

**Corrections applied throughout this plan:**
1. **Gate reference is no longer v56y.** It is the strongest LADDER-VALIDATED set:
   **v58 + the two public tape agents + the top of the public panel**, banded. Every
   "beat v56y" below now means "beat these strong references, under CONTESTED play on
   the faithful harness." (v56y stays only as a weak floor sanity-check.)
2. **RE-BASELINE FIRST.** The moment A1/A2 give a faithful harness, re-measure
   {v56y, v58, the 2 public tapes, top panel} head-to-head to establish the TRUE
   ranking, and re-test every prior negative verdict before acting on it.
3. **The OR MILP's isolated-market objective is itself a static assumption** — it can
   mis-rank under contention (v56y's exact failure). The OR tape is validated against
   REACTIVE contesters; the two public tapes are both the GATE and a concrete
   production TARGET / warm-start (better than v56y — extract their tapes).
4. **BC-seat is UNPROVEN, not dead** — pending faithful re-measurement. Dispatcher
   stays the safe default; a faithfully-gated BC seat is NOT foreclosed.

**The strong references (identified from live submissions, 2026-09-18):** the two
LIVE (latest-2) submissions are public tapes —
`submission_competitive_v46.tar.gz` (**2517.4**) and
`submission_k0006_open10_h24_frontload_advance2_v43.tar.gz` (**2493.6**). These are
the primary gate references + panel opponents + OR targets. (Not staged locally —
pull their source kernels or obtain the tarballs.)
**Ladder-score caveat:** the ladder shows v58 (56293043) = **463.3** and v57/v56y
(56260608) = **2072.0** — so v58 is NOT better than v56y on the ladder (contra an
earlier note); anchor on the **~2500 public tapes**, not v58/v56y. v56y (2072) is a
mid floor, the 2500 tapes are the bar, top-10 ≈ 2900.

---

## 0.7 FIELD INTELLIGENCE — what actually works right now (competition discussions)

From four competition discussion threads (read 2026-09-18; deadline Sept 23, 13 days):

- **[741743, SNORLAX, 450th → SILVER]** THE FEASIBLE MEDAL RECIPE, confirmed:
  **"behavior cloning for warm-up, then RL," at the MACRO level (daily-plan
  policies, NOT per-turn), ~300k games total.** This is the topper's BC→RL pattern
  at a *tractable* scale (300k games, not 11B steps) — reachable on our fast Rust
  engine in hours.
- **[741258, KKY, +27]** PER-TURN RL is a trap: PPO evaluating every turn plateaus
  at **~80k money on 300M steps, "still a dumb policy," 2 weeks / 100+ experiments,
  compute-bound (DGX Spark too slow).** The field consensus in the thread
  (BillDoser, HyperNeonByte, aisormo) is to **move the RL to the DAILY-PLAN level
  and hand execution to a deterministic process — 30 decisions, not 720** — because
  per-turn credit assignment is brutal and the action space is huge.
- **[741730, KKY self-play]** self-play RL **still can't beat strong open-source
  (TAPE) opponents**; resume-with-Adam-optimizer-state matters; dynamic opponent
  pool is the planned fix. Common learned-policy weakness: **animals & fertilizers.**
- **[741414, Ops Lab]** economy is **net P&L** (revenue − seeds − feed − labor), not
  gross — the exact objective the OR-MILP already models.

**What this CHANGES in the plan (major):**
1. **RL is MACRO-level, not per-turn.** The unifying architecture (also research
   doc 2's macro/micro split) is a **shared deterministic MICRO-executor** (our tape
   + PC-TAPF routing + plan-repair) driven by a **MACRO planner** that makes ~30
   daily decisions (what to plant/buy/build/hire, when/what to sell). Slot 1's macro
   planner = the OR/rule policy; Slot 2's macro planner = a **BC-warmed macro RL
   policy**. Same executor, swappable brain. This fixes both credit-assignment and
   compute.
2. **Compute is feasible.** ~300k games (SNORLAX) on our Rust engine (143 ep/s/core,
   parallelizable) is minutes-to-hours, not the topper's 11B steps. BC warmup +
   macro action space is what makes it tractable on an RTX 4060.
3. **Strong TAPE agents remain the bar** — RL doesn't beat them yet, so Slot 1's OR
   economy is genuinely competitive AND the right gate reference (with v58).
4. **Animals + fertilizer are a field-wide weak spot** — the OR-MILP models them
   explicitly (an edge), and any learned policy must be checked on them.
5. Add the **Ops-Lab-style net-P&L per-crop/animal accounting** as an analysis lens
   on our replays (cost-leak detection), and consider the tool for eyeballing.

**Net:** the feasible top-10 path the field validates = **BC warmup → macro-level RL
over a deterministic executor**, gated against the strong tape agents + v58 on the
faithful harness. Per-turn RL and 11B-step self-play are NOT the route for our window.

---

## A. Harness revamp (summary; full detail in the companion doc)

**Diagnosis (grounded):** engine is bit-exact; seat-swap + mirror-dedupe + holdout
already exist; serve is 12× faster than official. The ONE defect is the reactive
**observation content** on `kagg serve`. The competition docs settle the contract:
it is **single-arg `agent(obs)`** with `step` INSIDE obs (not the two-arg
`(obs, configuration)` form the code-mapping assumed), so serve's call shape is
right — the risk is what's in obs:
- serve's `obs_for` must reproduce the documented schema exactly — every field
  incl. `step` and **both** `market.inventory` and `market.prices` — with correct
  numeric types; any drift changes reactive decisions;
- market dicts emitted in **insertion order**, official uses **PRODUCTS order** —
  order-dependent agents drift;
- two-arg agents (kaggle-environments also accepts `agent(obs, config)` via arity
  inspection) need a `configuration` only when declared — low risk;
- the Python fallback harness shows the symptom directly (a reactive agent sells
  1434 vs the Rust binary's 7198 WOOL on the same worlds ⇒ it reads a different
  market obs).
- **Gate blind spot:** `serve_equiv` (12/12) only certified *tape-shell* agents —
  the very agents that can't expose the obs bug — so `serve_allowed()` reads green
  while reactive ranking is broken.

**Plan:** Phase 0 per-turn obs/action diff (serve vs official) to name the exact
divergent fields → Phase 1 faithful reactive obs contract (fix signature, byte-
identical obs incl. dict ordering, one canonical obs builder, extend the parity
gate to ≥6 *reactive* agents) → Phase 2 (optional, post-deadline) PyO3 embedding →
Phase 3 warm-up + replay-agent opponents. **Phase 1 is the deadline priority.**

---

## B. World Generator + Tape Generator + Simulator (OR pipeline)

### B.0 Reframing what actually needs building
- **Simulator: already done.** `rustengine` is the bit-exact engine. Its role in
  this pipeline is the **ground-truth validator** — play each generated tape,
  measure real Cash₇₂₀, and close the loop a pure solver cannot.
- **World Generator: new.** A Rust module encoding the exact economy as an
  optimization model + a solver backend.
- **Tape Generator: new (thin).** Serialize the solver's action schedule into our
  existing `.tape` TSV, byte-compatible with the bandit base tapes.

### B.1 The exact model to encode (from the engine, not the sketch)
The Python sketch you gave is an aspatial approximation with wrong numbers
(`sell×25`, "simplified labor"). The faithful model uses the real rules:

- **Objective:** maximize `money` at the last acted step (718); money is f64.
- **Price (per product):** `quote = max(1, round_half_even(base ± amp·shape(x,T)))`,
  piecewise around market inventory I0=10000. 1.32.7 params (base, T, below/above
  shape+target) for WHEAT/CARROT/TOMATO/STRAWBERRY/MELON/EGG/MILK/WOOL/FERTILIZER;
  shapes ∈ {linear, sq, sqrt, log, log10, hinge(gain 8)}. **Banker's rounding.**
- **Sale dynamics:** a SELL of q walks the price DOWN unit-by-unit (each unit
  re-quoted at current market inventory, +1 inventory per non-$1 unit; $1-floor
  units don't add supply). Prices refresh per order-index. **Revenue for selling q
  from inventory `inv` is a concave, piecewise-linear staircase in cumulative
  units** — this is the key structure that makes the nonlinearity tractable (B.2).
- **Town drain:** every 4 steps single-product shops (YARN_STORE→WOOL,
  PET_CAFE→CARROT) drain 2, multi-product shops drain 1 each listed product;
  every 24 steps all products except FERTILIZER drain 1. Drain below I0 raises
  quotes (scarcity) — a *tailwind* the model should exploit (sell into scarcity).
- **Production:** crops {WHEAT,CARROT,TOMATO,STRAWBERRY,MELON} with
  (seed_cost, first_yield_day, max_yield_day, interval, max_yield, ongoing);
  water cadence (planting day = unwatered day 1; **weed at 2 consecutive unwatered**);
  yield accrues by watering in the maturity window (non-ongoing) or daily schedule
  (ongoing); MELON caps 6 units days 10–12. Animals {GOOSE 300→EGG, COW 400→MILK,
  SHEEP 500→WOOL} with (first_yield_day, interval, max_held); **FEED consumes 1
  WHEAT/day, escape at 2 consecutive unfed**; CARE banks +1 applied at next fed
  production; no age decay.
- **Costs/caps:** land NE/SW/SE = 1000/2000/4000 (NW free); COOP/PASTURE **free**;
  hire cost = Fibonacci(kth hire that day) {1,1,2,3,5,8,…}, **resets daily**; shed
  cap 100 (overflow discarded); start money 3000.
- **Labor:** 1 farmer + N hands, each **one op/turn**; hands re-hired daily,
  farmer+hand positions reset daily.
- **Spatial (the omission that matters):** every tile op requires the unit to be
  ADJACENT to that tile; DROP/PICKUP require shed-adjacency; units MOVE one step/turn.
  The sketch's "action budget ≤ 1+hands" ignores movement entirely.

### B.1b Strategic structure the rules impose (from the README) — what the optimizer must exploit
The competition README is not just rules; it hands us the economic structure the
solver should be built around:
- **`T` = one 5×5 field's 24-day production capacity** (at optimal watering, no
  fertilizer; animal totals pre-discounted 30% for wheat-feed). This is the market's
  absorption yardstick: **selling ~T units past I0 moves price by `target × base`.**
  The published price points make the ceiling concrete:

  | resource | base | T | P(I0−T) | P(I0+T) | P(I0+2T) | selling behavior |
  |---|---|---|---|---|---|---|
  | Wheat | 25 | 400 | 45 | 20 | 19 | staple — absorbs gluts (log above); volume-friendly |
  | Egg | 50 | 332 | 70 | 40 | 39 | staple — hinge scarcity, log glut; volume-friendly |
  | Carrot | 35 | 450 | 70 | 10 | 1 | hinge scarcity — spikes when shops drain supply |
  | Tomato | 60 | 200 | 84 | 24 | 9 | hinge scarcity; ongoing crop |
  | Fertilizer | 100 | 200 | 140 | 60 | 20 | free animal byproduct — pure side income |
  | Strawberry | 120 | 100 | 204 | 1 | 1 | **premium — floors at +T; meter sales** |
  | Milk | 160 | 122 | 256 | 1 | 1 | **premium — floors at +T; meter sales** |
  | Wool | 200 | 105 | 240 | 1 | 1 | **premium — floors at +T; meter sales** |
  | Melon | 250 | 300 | 300 | 1 | 1 | **premium — barely reacts to scarcity, crashes on glut** |

  ⇒ **The optimizer's core tension is production-vs-absorption, and it is
  product-specific:** dump staples (wheat/egg/carrot) but *meter* premium goods
  (strawberry/melon/milk/wool) in small bundles spread across many turns, because
  each unit re-quotes the price down and the premium curves floor after ~one field's
  glut. This is the single most important thing the MILP's revenue model must get
  right — and it argues for **diversified production across several products** so total
  revenue isn't capped by any one market's absorption.
- **Sell into scarcity (hinge goods).** Carrot/tomato/egg use `hinge` below I0, and
  the town shops that consume them (pet cafe→carrot 2×, pizza/farmers-market→tomato,
  bakery/brunch→egg) drain supply below I0 → prices rise sharply past the knee. So a
  timing lever exists: **produce hinge goods and sell when shop consumption has
  drained the market** (prices above base). The reactive market shell and/or a
  scenario-aware MILP should exploit this; it's a tailwind, not just noise.
- **Yield/tile/day efficiency** (README Object Types): Goose/Egg 1.00, Wheat 0.80,
  Carrot 0.75, Melon 0.55, Cow/Milk 0.50, Tomato 0.33, Sheep/Wool 0.33,
  Strawberry 0.24. Combined with base price this ranks **revenue density per tile**,
  the prior for the crop/animal MIX under the 25-tile-per-quadrant land constraint
  (buy NE/SW/SE at 1000/2000/4000 to add fields — each new field adds ~T of
  absorption headroom, another reason land expansion and product diversity compound).
- **Fertilizer is free and sells at base 100** (linear both sides, T=200): every
  surviving animal yields 1/day regardless of feeding → a steady side-income the
  optimizer should always collect and sell, plus it doubles crop watering bonuses.
- **No arbitrage:** buy-then-sell nets exactly zero (buy at post-buy inv, sell at
  pre-sell inv), so BUY_PRODUCT is only for wheat-as-animal-feed / fertilizer inputs.

These turn the objective from "maximize output" into "**maximize output SUBJECT TO
per-product market absorption**", which is exactly what the staircase-revenue MILP
(B.2.1) encodes — the `T`-scaled breakpoints ARE the absorption limits above.

### B.2 The three hard modeling problems (honest)
1. **Nonlinear/MINLP revenue.** Price is nonlinear in inventory and own-sale
   revenue is bilinear (`q × price(inv)`). **Resolution:** the per-unit re-quote
   makes revenue a *concave separable staircase* in cumulative units sold per
   product per selling-window → model it with an **incremental (SOS2 / lambda)
   piecewise-linear formulation**. That converts the hard MINLP revenue into exact
   MILP columns (one breakpoint per unit, or bucketed). No approximation of the
   curve shape is needed — the staircase is exact given a target market-inventory
   trajectory.
2. **Spatial routing explosion.** A faithful position+movement model (10×10 × 719
   turns × ≤16 units) is intractable monolithically. **Resolution — two-layer
   decomposition:**
   - **Layer 1 (economic MILP):** solve *what/when* — plant/water/harvest/feed/care,
     buy seeds/animals/land, hire count, and sell/buy schedule — under an
     **action-budget relaxation** (Σ ops ≤ 1 + hands per turn) and all economic/
     biological constraints. Gives an upper bound and a target schedule.
     Decompose by **rolling horizon / per-day** (warm-started from a proven tape)
     rather than a monolithic 720-step model.
   - **Layer 2 (routing realizer) = PC-TAPF** (Precedence-Constrained Target-
     Assignment & Path-Finding, per `research/Dynamic Labor Routing...`): a per-day
     **Hungarian-assignment** (laborers→tasks, flowtime-min cost matrix) +
     **Conflict-Based Search** (conflict-free grid paths, vertex/edge constraints),
     over a precedence DAG (till→plant→water→harvest, animal place→feed→harvest).
     Places farmer+hands to execute Layer-1's ops (adjacency, one-op/turn, movement,
     daily position reset). Where routing can't realize the schedule, tighten
     Layer-1's per-turn budget and re-solve. **MVP fallback:** a greedy nearest-task
     router + our existing reactive guardrails; upgrade toward Hungarian-CBS as time allows.
   - **Robustness layer = dependency-directed plan-repair** (same doc): annotate the
     tape with causal-dependency metadata; on a stochastic shock (weed spawn, shop
     unlock, land buy, mirror front-run) excise ONLY the fractured causal links and
     patch locally (regression search + localized Hungarian-CBS) rather than replan
     from scratch. This is the principled form of the reactive shell — what makes a
     rigid tape survive the realized (non-deterministic) world. Rust memory arenas
     give near-instant rollback. (We do NOT adopt the doc's ArcWake async rewrite or
     motif-compression — out of scope for the window.)
3. **Opponent & RNG coupling.** The market is SHARED (opponent sales move prices)
   and shop unlocks/weeds are RNG. The solver optimizes an **isolated market
   projection** — exactly the wall this project already hit ("offline panels don't
   predict the ladder"). **Resolution:** (a) keep the reactive market shell
   (front-run/scarcity-sell) around the tape — the whole field does this; (b)
   optionally solve against a **distribution of shop-unlock scenarios** (robust/
   scenario MILP) or per **realized-world cells** (as `sell_search.batch_eval_cells`
   already groups); (c) **validate on the engine** and treat MILP-cash as an upper
   bound, not a promise.

### B.3 Architecture — a principled upgrade of `sell_search`, engine-in-the-loop
The repo already has the right skeleton: `sell_search.py` runs (1+λ) evolution on
the MARKET channel against the bit-exact engine with paired-holdout gating;
`tape_gen.py`/`factory.py` do farm-plan hillclimbing. The OR pipeline **replaces
the blind evolutionary search over the farm+economic plan with a solver**, and
keeps their proven engine-validation + paired sign-test gate:

```
proven tape (v56y egg/yarn)  ── warm start ──►  Layer-1 economic MILP (per-day rolling)
                                                         │ target schedule
                                                         ▼
                                              Layer-2 routing realizer (CP/assignment)
                                                         │ playable tape (TSV)
                                                         ▼
                                   rustengine `kagg batch/episode`  ── real Cash₇₂₀
                                                         │ gap vs MILP-predicted
                              ┌── feedback: tighten budgets / re-linearize ──┘
                                                         ▼
                              paired-holdout gate vs incumbent (win_metric)  ── ship base_tape
```

Output feeds the **config-driven `base_tape`** for `mbandit` + reactive shell — i.e.
this is also the principled fix for the v58 "no strong config-driven base" problem.

### B.4 Solver decision (you noted a Rust SCIP exists)
| Backend | Rust crate | Handles | Build/License | Verdict |
|---|---|---|---|---|
| **SCIP** | **`russcip`** | MILP **and** MINLP/nonlinear, constraint handlers | links libscip; **SCIP is Apache-2.0 since v8/9** (the old non-commercial restriction is gone) | **Recommended primary** — can take nonlinear price expressions directly if we ever skip linearization; strongest on our hard structure |
| **HiGHS** | `highs` / `good_lp` | MILP/LP only (linear) | permissive (MIT), **easiest Cargo build**, very fast | **Recommended for Layer-1** once revenue is piecewise-linearized (B.2.1) — simpler + faster than SCIP for pure MILP |
| CBC | `coin_cbc` | MILP | permissive, older/slower | fallback only |
| OR-Tools CP-SAT | (C++ via FFI; no clean pure-Rust) | integer scheduling/routing | Apache-2.0 | **best fit for the Layer-2 routing** subproblem, if we want a strong CP solver there |

**Recommendation:** model behind a **solver-agnostic Rust trait** (variables/
constraints/objective) with two backends: **HiGHS for the linearized Layer-1 MILP**
(fast, permissive, easy build) and **`russcip`/SCIP as the fallback** for any
nonlinear pieces we choose not to linearize. Use a **CP-SAT-style formulation for
Layer-2 routing** (russcip can also do it). Do NOT hand-roll a solver — no
production-grade pure-Rust MINLP solver exists; "pure Rust" here means the *model*
is Rust, calling a linked solver, which is the standard and correct interpretation.

### B.5 Tape output (byte-compatible)
Emit per turn `{farmer, hands[], market[]}` → TSV `farmer\thands\tmarket`, 719 rows,
semicolons within hands/market, spaces within an op, integer quantities, ≤10 market
orders/turn, optional `SEED <n>` header. This is exactly the bandit base-tape schema
(`bandit_v57_egg.tape` etc.), so the generated tape drops straight into the
config-driven `mbandit` base.

### B.6 Honest risk on B
Highest ceiling, lowest certainty. The nonlinear revenue is linearizable (tractable),
but the **spatial routing** and the **isolated-market gap** are where offline-optimal
tapes have historically failed to beat the proven reactive economy. **B must earn
its place by validating on the engine and passing the paired-holdout gate vs the
v56y incumbent — nothing ships on MILP-predicted cash alone.** Realistically, within
a week a *simplified* Layer-1 (single-region, linearized revenue, greedy router) can
produce a *candidate* base to test; a full faithful generator is a multi-week R&D
effort.

---

## C. Crown tournament — needed? improve? Rust port?

### C.1 Is it needed? YES.
It is the selection mechanism in the daily cycle (`refresh_cycle.py`:
mine→diagnose→**crown**→arms→build→gate). The crowned candidate must beat the
incumbent by ≥ `CROWN_GATE=10` percentage points on a referee roster (two rounds:
screen subset → full roster, two seed sets), else **HOLD**; reserved **holdout**
tapes validate the crown without selecting it (overfit alarm); referee pruning
(`referee_power.py`) drops non-discriminating tapes; overlap exclusion prevents a
candidate being refereed by its own team/episode. Without it there is no principled
"which candidate ships" — so it stays.

### C.2 What's wrong with it now
- **Saturated panel** — `CROWN_GATE` is frozen at HOLD because referees no longer
  discriminate (memory: `crown-gate-saturated`).
- **Runs on the mis-ranking substrate** — the crown uses `serve` when
  `serve_allowed()`, which mis-ranks reactive agents (Item A).
- **Referees are loss TAPES** that desync for reactive candidates — the same panel-
  fidelity failure as Pillar A of the bandit plan.
So the crown currently cannot tell good reactive candidates from bad.

### C.3 How to improve it (mostly the other tasks paying off)
1. **Faithful substrate (Item A Phase 1)** → crown matches become trustworthy for
   reactive candidates; re-cert `serve_equiv` on reactive agents before trusting it.
2. **Real, banded referee panel** — replace desyncing loss-tapes with the
   **high-scoring public agents (Task 1)** + **top-100/200 replay opponents**
   (the download), each labeled by real ladder rating, filling the empty 2400–2600
   rung where matchmaking actually pairs us. This de-saturates the panel.
3. **Recalibrate the gate** — a 10-pp threshold on a saturated panel is meaningless;
   move to **paired McNemar per rating band + ladder-weighted aggregate**
   (`win_metric.paired_test`), as the bandit Pillar-A gate specifies. Keep the
   holdout overfit alarm.
4. **Keep referee pruning** (`referee_power`) but feed it the new panel.

### C.4 Does porting the crown to Rust help? Mostly NO.
- The **matches already run in Rust** — `kagg batch` replays tape-pairs fully
  natively (no per-turn IPC); `kagg serve` runs reactive matches. The crown's
  **orchestration** (`evaluate.py`, `ProcessPoolExecutor`, BT/Elo ranking) is thin
  Python glue over native matches; porting it to Rust saves little because the
  compute is already native — the cost is scheduling/I/O, not Python arithmetic.
- The **only** meaningful Rust win for the crown is removing per-turn stdio for
  **reactive** candidates by **PyO3-embedding** the agent in the engine — but that
  is exactly **Item A Phase 2**, and it's optional/high-effort.
- **Verdict:** do NOT port the crown orchestration to Rust. Invest the same effort
  in fidelity (A) + panel (Task 1 + replays) + gate recalibration, which is what
  actually unsticks it. Revisit PyO3 only if reactive-tournament wall-clock becomes
  the limiter.

---

## D. Combined sequencing for ~1 week (priority = certainty of ladder impact)

Organized around the two shipping slots (§0.5):

1. **Item A Phase 1 — faithful serve obs contract + reactive parity gate.**
   Highest certainty. Unblocks the crown and every gate. (Days.)
2. **Item C — crown de-saturation:** wire the Task-1 public panel + top-100/200
   replay opponents in as banded referees; recalibrate the gate to paired/banded.
   Rides directly on A. (Days, overlaps.)
3. **SLOT 1 = Item B — OR economy + PC-TAPF routing + plan-repair shell:** start
   with linearized Layer-1 + greedy router + engine validation on a *single*
   proven-world segment; upgrade toward Hungarian-CBS + plan-repair as time allows.
   Only a tape that validates AND passes the holdout gate vs v56y ships. This is the
   **primary top-10 lever** (economy). (Week+; simplified candidate possible in window.)
4. **SLOT 2 = predator track (parallel):** DuckDB-distill top-100/200 replays → BC
   **dispatcher** (not a per-turn seat) + market-timing predictor; action-masked;
   gate vs Slot 1. If it doesn't beat Slot 1, Slot 2 ships a DIVERSE OR-economy
   (`route2`). RL fine-tune only if everything else lands.
5. **Item A Phase 2 (PyO3) — only if** reactive-tournament throughput limits after 1–4.

**Certain wins = A + C + Slot 1** (a trusted substrate/selector + a stronger,
robust OR economy — the economy gap IS the rating gap). **Slot 2 is the variance
bet**; it only ships if it beats Slot 1 on the faithful gate, else it ships as a
diversified second economy.

## E. Scope & discipline (operator: EVERYTHING is in scope, 2026-09-18)

Nothing below is excluded. The previous "non-goals" that *excluded* components
(BC-seat, IMPALA/DT alternatives, ArcWake/motif/generalized-planning, etc.) are
**withdrawn** — all are in scope (see §F). What remains are **discipline lessons**
(HOW to build, not WHAT to exclude):
- **Don't rewrite the bit-exact engine** (extend it; keep parity).
- **Fidelity before throughput** — a fast mis-ranking harness still mis-ranks; do A first.
- **Nothing ships on static/isolated numbers or predicted cash** — engine-validate
  under CONTESTED play + paired gate vs the strong references (the two ~2500 tapes).
- **Don't trust `serve_equiv` green until it contains reactive agents.**
- **The research docs' / topper's confidence is not the bar** ("cannot err", "97%")
  — the paired gate on the faithful engine is the only arbiter.
- **"In scope" ≠ "equally urgent."** Sequencing (§D) protects a shippable critical
  path (A+C+Slot 1) so the advanced items (IMPALA, DT, ArcWake, motif compression,
  generalized-planning, PyO3/native-inference variants) can land as they land without
  blocking a submission.

## F. Full component register (all IN — operator decides sequencing, not inclusion)

Every requested/derived component, first-class:
- **Foundation:** A faithful harness · C crown de-saturation · D data (replays,
  top-200, 125 public agents, DuckDB distillation) · re-baseline on faithful harness.
- **Slot 1 (Anchor):** OR/MILP economy (net-P&L) · PC-TAPF routing (Hungarian+CBS) ·
  dependency-directed plan-repair · reactive market shell (hysteresis, price-gated tranches).
- **Slot 2 (Predator):** BC warmup · **macro-level** self-play RL · opponent pool +
  teacher-KL + gated promotion · economy/opening dispatcher · market-timing predictor.
- **Algorithm choices to EVALUATE (not pick blind):** PPO **and** IMPALA;
  pointer-transformer **and** Decision-Transformer; per-turn RL (low priority, field
  says it plateaus) **and** macro RL.
- **Inference/runtime:** native Rust inference for our policy via `candle`/`tract`/`ort`
  (ONNX) — the PyO3 alternate, and better for the submission; stdio serve + optional
  shared-memory IPC for Python opponents; PyO3 embedding also available if wanted.
- **Advanced engine/RL optimizations:** ArcWake async macro/micro loop · Kolmogorov/MDL
  motif compression for fast plan-repair · generalized-planning (conditional-goto tapes
  that branch on shop-unlocks without full repair).
- **Analysis:** Ops-Lab-style net-P&L per-crop/animal accounting (cost-leak detection);
  the Ops-Lab tool itself for eyeballing.
- **Solvers:** HiGHS (primary), russcip/SCIP (MINLP), CP-SAT (routing).

## G. Harness architecture refactor — bandit ⊥ trackp, fully hot-swappable

**Operator directive:** the bandit and trackp harnesses must be **completely
separate**, **highly configurable**, and **everything hot-swappable** — guardrails,
policies, base tape/routes, adaptive plan. Both agents are **Rust, musl-compiled in
Linux Docker**, with a **thin `main.py` driver**; learned policies use **native Rust
inference** (the PyO3 alternate), not PyO3. **Reference pattern = the shipped v46
`Chassis`** (config-driven layers + safe raw-tape fallback).

**Audit (2026-09-18) — what's actually there** (full detail: tasklist Workstream G,
memory `harness-audit-2026-09-18`; CLAUDE.md corrected):
- The CLAUDE.md **"Track P shares zero code with bandit" claim is FALSE.** Three
  coupling knots: (1) ONE `kagg` binary holds all subcommands and **`kagg trackp` is
  literally `bandit::play("v57")`**; (2) bandit BUILD tooling lives under
  `trackp/harness/`; (3) bandit BRANCH data lives under `models/trackp/`.
- Bandit has TWO paths: **`bandit.rs`** (`kagg bandit`/`trackp`) is 100% compile-time
  (hardcoded tapes/thresholds/D6 fork); **`mbandit.rs`** (`kagg mbandit`) is ~80%
  config-driven (the release candidate).
- **Dead config (config lies):** `anti_dump.premium`/`scarcity_i0` arrays are IGNORED
  by the Rust path (only the Python fallback honors them); r37 curves, demand tables,
  dispatch window, sweep constants are hardcoded regardless of config.
- **Trackp seat is 100% hardcoded** (skeleton genome + policy.rs consts; no config/tape
  at ship time).
- **musl bug:** `build_rust_bandit.py` packages the LOCAL Windows `kagg`, not the musl
  binary; there is no dedicated bandit musl script.
- v58's 463 = a **stale-tape BUILD bug** (old v45 fallback), NOT a weak architecture —
  rebuild with the correct `base_tape`.

**Refactor (Workstream G, task-level in the tasklist):** G0 decouple (separate
`--bin`/crates, extract a track-neutral `harness-core`, move bandit tooling/data out of
the trackp namespace, drop dead imports) · G1 bandit fully config-driven (retire
`bandit.rs`, externalize every table/curve/constant, FIX the dead array config,
data-driven ordered rail registry) · G2 trackp externalize genome→JSON that both the
builder and `policy.rs` load · G3 unified per-track config schema + validation (error on
unconsumed keys — no silent dead config) · G4 separate musl builds + thin drivers per
track (fix the Windows-binary bug) · G5 native ONNX inference (`candle`/`tract`/`ort`) ·
G6 parity + dead-config tests + keep memory/docs/CLAUDE.md synced with findings.

**Sequencing note:** G0 (decouple) is FOUNDATIONAL and early — the config-driven bandit
(feeds Slot 1 / the v58 rebuild), the OR-tape base (B), and native inference (F/G5) all
sit on top of it. G is planning-only here; no code until go.
