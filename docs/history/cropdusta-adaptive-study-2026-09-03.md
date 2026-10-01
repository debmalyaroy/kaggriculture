# Crop Dusta — reverse-engineering the runtime production rule (2026-09-03)

Read-only analysis. No agent, model, gate or engine file touched; nothing
submitted; no kernel pushed. Scratch scripts live in the session scratchpad
(`cd_*.py`, listed in the appendix). Motivated by
`docs/closed-loop-intel-2026-09-03.md`, which flagged Crop Dusta as the only
one of the top 15 teams whose *production* allocation changes between
same-build games.

**Crop Dusta = `rishigottumukkala`, public LB rank 2, rating 2956.3, submitted
2026-09-01 02:01 UTC** (`.local/lb2/kaggriculture-publicleaderboard-2026-09-03T12_07_24.csv`).

---

## 0. Headline

The adaptive finding is **confirmed and fully resolved**. It is not resync
noise, not partial fills, not RNG cascading through a fixed policy, and not
item availability. Crop Dusta runs a **fixed opening tape through day 6**, then
a small number of **discrete decision points at fixed clock times**, each of
which reads **one public market observable — how many units of a product the
town consumes in a single turn — and dispatches on it.**

The single most important result:

> **At step 145 (day 6, hour 1) the herd branch is a deterministic function of
> the town's one-turn consumption of WOOL and MILK. A depth-2 rule on that
> observable reproduces Crop Dusta's choice in 134 of 134 games (100.00%),
> across two concurrently-live builds, on three different days.**

The rule is *literally*:

```
at step 145 (day 6, hour 1):
    dWOOL = market_inventory[WOOL] (t-1)  - market_inventory[WOOL] (t)
    dMILK = market_inventory[MILK] (t-1)  - market_inventory[MILK] (t)
    if dWOOL >= 2:  run the SHEEP economy      # a wool-consuming shop is unlocked
    elif dMILK >= 2: run the COW economy       # a milk-consuming shop is unlocked
    else:            run the GOOSE economy     # default
```

Baseline consumption of every product is exactly 1 unit/turn (the town
centre). A value of 2 or 3 means an unlocked shop lists that product. So the
rule is equivalently, and more implementably: **look at the town's unlocked
shop set at day 6 and buy the animal whose product those shops buy.**

Two further decision points follow the same "produce what the town eats"
principle, on the same instrument:

| when | decision | observable | fit |
|---|---|---|---|
| step 145 (day 6 h1) | herd: SHEEP / COW / GOOSE | 1-turn WOOL, then MILK consumption | **134/134 = 100%** (Sep 1-3); 186/190 = 97.9% as a COW/not-COW binary back to Aug 21 |
| step 385 (day 16 h1) | plant TOMATO at all: yes/no | 1-turn TOMATO consumption ≥ 3 | 182/190 = **95.8%** across 9 build-days |
| step 505 (day 21 h1) | CARROT commitment (size **and** timing) | 1-turn CARROT consumption | seeds ≈ **11.2 × tick − 3.2**, r = 0.769, n = 172; tick ≥ 4 ⇒ start day 21, ≤ 3 ⇒ wait to day 24 (92.3% on the Sep-1 build) |
| ~step 241 (day 10 h1) | top up with GEESE in a non-goose world | 1-turn EGG consumption ≥ 2 | 71/93 = 76.3% — real but not exact |

---

## 1. Data and method

- 212 Crop Dusta routes in `data/routes/index.json`; **190 on engine 1.32.7
  with a `data/turntrace/<episode>.npz`** (dates 2026-08-21 … 2026-09-03; the
  three biggest days are 09-01 n=78, 09-02 n=44, 08-21 n=21).
- Actions: `data/routes/<rid>.json.gz` (720 turns of `{farmer, hands, market}`).
- State: `data/turntrace/<episode>.npz`, the 49-field per-turn observation
  (`src/kaggriculture/data/turn_features.py::feature_names()`), seat-correct.
- **No replays were pulled.** `data/sameday/quota_ledger.json` showed **2019
  events in the trailing 24 h against a cap of 2000 — headroom −19** when this
  study started. The whole analysis is therefore index + trace only, and the
  quota ledger is unchanged by this work. (Everything below would have been
  *nicer* to confirm against `town.shops` in a raw replay; see §7.)
- Production ops = `PLANT / BUILD_PASTURE / BUILD_COOP / FERTILIZE` (unit) and
  `BUY_SEED / BUY_ANIMAL / BUY_LAND` (market). Logistics
  (`WATER/FEED/CARE/PICKUP/PLACE/DROP/DIG/HIRE/SELL/BUY_PRODUCT`) excluded, as
  in the prior study.
- Verdicts use `src/kaggriculture/measure/win_metric.py` (`summarise`, `min_detectable`) plus Fisher
  exact on 2×2 win/loss tables.

### Build partition (important — "same day" is not "same build")

Grouping by episode date is wrong here: **two Crop Dusta builds were live
simultaneously** (the competition allows 2 active submissions).

| build | signature | dates seen | n (1.32.7) |
|---|---|---|---|
| **A** | no day-5 animal purchase; herd decision fires at step 145 | 08-21 … 09-03 | 96 in Sep |
| **B** | unconditional `BUY_ANIMAL SHEEP ×2` at steps 121/134 (day 5), herd decision fires at steps 153-155 | 08-28 … 09-03 | 38 in Sep |

Both builds obey the **same** day-6 rule; build B just prepends two
unconditional sheep. Grouping by date instead of build is what produced the
apparent "misclassifications" in an intermediate pass — a labelling artefact,
not a rule failure.

---

## 2. WHEN the fork happens

### 2a. Fork-step distribution (task question 1)

Cumulative production vectors, per pair, first step at which the L1 distance
exceeds 5 and never returns below it:

| date | eps | pairs | min | p10 | median | p90 | max |
|---|---|---|---|---|---|---|---|
| 2026-09-01 | 78 | 3003 | 114 (d4.8) | 124 (d5.2) | **145 (d6.0)** | 198 (d8.2) | 471 (d19.6) |
| 2026-09-02 | 44 | 946 | 113 (d4.7) | 118 (d4.9) | **122 (d5.1)** | 155 (d6.5) | 278 (d11.6) |

Fork-day histogram, 09-01: day 4: 76, **day 5: 855, day 6: 1704**, day 7: 34,
day 8: 259, day 9+: 75. (The 09-02 mass at day 5 is the build-B prefix, not a
decision.)

Per-key first-permanent-divergence (median step, 09-01, 3003 pairs) shows the
decision points as sharp clusters, not a smear:

```
BUY_ANIMAL:COW        med 145  (day  6.0)   p10 145
BUY_ANIMAL:SHEEP      med 153  (day  6.4)   p10 145
BUY_ANIMAL:GOOSE      med 153  (day  6.4)   p10 152
BUILD_COOP            med 155  (day  6.5)   p10 154
BUY_SEED:TOMATO       med 385  (day 16.0)   p10 385
PLANT:TOMATO          med 399  (day 16.6)   p10 389
BUY_SEED:CARROT       med 523  (day 21.8)   p10 512
PLANT:CARROT          med 526  (day 21.9)   p10 515
```

**It is several decision points, not one**, and they sit at fixed clock times
(day 6 h1, day 16 h1, day 21 h1), not at data-dependent moments.

### 2b. The information arrives *exactly* at day-6 dawn

The sharpest single result in this study. A depth-2 decision tree on the full
49-feature observation, fitted at read-step *s* to predict the herd branch
(09-01, n=78, 5-fold stratified CV):

```
step 138 … 144   train 0.692   cv 0.693
step 145         train 1.000   cv 1.000
step 146 … 150   train 1.000   cv 1.000
```

Nothing at step 144 predicts the branch; **everything at step 145 does.** The
day-6 shop unlock happens at dawn (step 144) and its first consumption tick
lands during 144→145. Crop Dusta reads it and acts in the same turn.

---

## 3. WHAT is observable at the fork (task question 2)

### 3a. The literal split

The one-turn inventory drop across the day-6 dawn tick, `inv(144) − inv(145)`,
tabulated against the branch actually taken (09-01, n=78):

| (dWOOL, dMILK, dEGG) | COW | SHEEP | GOOSE |
|---|---|---|---|
| (1, 1, 1) | 0 | 0 | 6 |
| (1, 1, 2) | 0 | 0 | 12 |
| (1, 1, 3) | 0 | 0 | 8 |
| (1, 2, 1) | **16** | 0 | 0 |
| (1, 2, 2) | **7** | 0 | 0 |
| (1, 3, 1) | **10** | 0 | 0 |
| (3, 1, 1) | 0 | **5** | 0 |
| (3, 1, 2) | 0 | **6** | 0 |
| (3, 2, 1) | 0 | **8** | 0 |

Perfectly separable, with the priority order **WOOL > MILK > default GOOSE**.
`dEGG` is present in both COW and GOOSE rows and never flips anything — egg
demand does *not* trigger the goose branch at day 6; **goose is the
fall-through when no wool- or milk-consuming shop is unlocked.**

The same in accumulated form (units below the 10,000 equilibrium at step 145)
shows why: a value of 7 is the baseline drain, 8-9 is a shop unlocked *this
morning*, and 26/27/45 is a shop that unlocked on day 3 and has been eating for
three days. Both the fresh and the aged version of a wool shop send Crop Dusta
to sheep; both versions of a milk shop send it to cows.

Equivalent price-space form (same 100% fit on the Sep-1 build), if you would
rather read prices than inventories:

```
mkt_inv[WOOL] <= 0.9992  ->  SHEEP        (price_WOOL >= ~1.10)
price[MILK]   >  1.15    ->  COW
otherwise                ->  GOOSE
```

### 3b. Nothing else is used

Fitted on the full 49-feature vector at step 145 the tree picks
`mkt_inv_WOOL` then `price_MILK` and stops at 100%. Explicitly checked and
**not** used:

- **The opponent's farm** — `them_money / them_hands / them_quads / them_crops
  / them_animals / them_dry / them_starving` never enter a tree that already
  has the market features, and a tree restricted to `them_*` is at chance.
  Whatever Crop Dusta is doing, **it is not reading the opponent.**
- **Own farm state** — `us_*` likewise adds nothing once the market features
  are in.
- **Own money** — money at the decision is $50-110 in all three branches; they
  are broke in every world and it does not separate.
- **Day / hour** — the decision time is fixed, so time carries no information.

### 3c. Is the observable exogenous? (leakage check)

Yes, cleanly.

- Before step 145 Crop Dusta's *only* market activity is `SELL WHEAT` and
  `SELL FERTILIZER`, in identical amounts across all three branch groups
  (mean 40.5 / 40.7 / 40.3 wheat and 25.0 / 25.0 / 25.0 fertilizer for the
  WOOL / MILK / GOOSE groups, 12 episodes each). **They touch no wool, milk or
  egg before the read.** The signal is town consumption plus the opponent, not
  their own footprint.
- Whole-game they buy only `WHEAT` (33,615 units over 20 episodes) and a
  trickle of `FERTILIZER` from the market — **never TOMATO or CARROT**, and
  they hold no tomato or carrot to sell before day 16 / day 21. The day-16 and
  day-21 ticks are exogenous too.
- Inventory trajectories confirm the mechanism: all products drain at exactly
  1.0 unit/day through day 3 in every episode, then the groups separate from
  day 4 — one day after the day-3 shop unlock. Extra drain from day 4 to day 6
  is ≈ 6.1 units/day for wool in the wool group, ≈ 4.6 for milk in the milk
  group, ≈ 4.7 for egg in the "neither" group. That is shop consumption
  (`.local/memory/cosim-compiler-and-market-truth.md`: each shop instance
  consumes 1 of each listed product every 4 steps = 6/day, doubled for
  single-product shops such as YARN_STORE and PET_CAFE).

### 3d. Mundane explanations, ruled out one by one

The prior study asked for these to be checked honestly. None survives:

1. **Weed-collision resync / pathing drift.** Would produce timing jitter in a
   fixed action stream. It cannot produce a *different item* in a market order
   at a *fixed step*, and it certainly cannot make that item 100% predictable
   from market inventory. Ruled out.
2. **Market partial fills.** `SELL`/`BUY` partially fill per unit; they do not
   substitute COW for SHEEP for GOOSE. Ruled out.
3. **Engine RNG cascading through an identical policy.** The shop draw *is*
   the RNG — but a fixed policy would buy the same herd regardless of the
   draw. The whole point is that the policy's output is a function of the
   draw. Ruled out (indeed, it is the mechanism).
4. **Item availability — "they always ask for COW, and it silently no-ops".**
   This was the most serious alternative. Ruled out two ways: (a) the *issued*
   orders differ (a COW-world episode issues `BUY_ANIMAL COW` at 145; a
   goose-world episode issues `BUY_ANIMAL SHEEP` at 145 and `GOOSE` at 152 —
   they do not spray all three and let the engine choose); (b) every animal is
   purchased in every world type later in the game — in wool worlds they still
   buy 123 cows and 22 geese, in milk worlds 340 sheep and 113 geese, in
   default worlds 57 cows and 163 sheep. Nothing is gated.
5. **Different build.** Controlled: the rule is 100% within each of the two
   concurrently-live builds separately.

**The adaptive finding stands. It is a genuine runtime read of public market
state.**

---

## 4. HOW MANY distinct behaviours (task question 3)

**It is not a dispatch table with N tapes, and it is not free-form per-turn
search. It is a fixed opening plus 3-4 conditioned decision points, two of
which are discrete and one of which is a continuous dose.**

- **Day 0-6: one fixed tape.** Median production in steps 0-143 is *identical*
  across all three branch groups: 3 COW + 2 SHEEP, 5 `BUILD_PASTURE`, 1
  `BUY_LAND` (step 121), 21 wheat seed, 11 strawberry seed, 5 melon seed, 20-21
  wheat planted, 11 strawberry, 5 melon, 0 fertilize. Production similarity in
  the day-0-6 window over all 3,003 same-day pairs: **96.2%.**
- **Day 6: a 3-way discrete branch** (herd + housing).
- **Day 16: a binary branch** (tomato or no tomato).
- **Day 21/24: a continuous dose** (carrot seeds, roughly linear in demand),
  plus a binary timing choice.
- **Day 10: a weaker discrete top-up** (add geese or not).

Hierarchical clustering of whole-game production vectors does **not** resolve
into 3, 6 or 8 clean clusters aligned with the branches — because the carrot
dose is continuous and dominates the L1 vector. Nominal cluster sizes at k=3
are 45/26/7 with branches mixed throughout. That is the answer: **the tail of
the game is parameterised, not enumerated.**

Production similarity over the 3,003 same-day pairs (09-01), which also
**re-validates the prior study's metric on a 200×-larger pair sample**:

| window | all pairs | same herd branch | same branch + same tomato call |
|---|---|---|---|
| day 0-6 | 96.2 | 96.1 | 96.0 |
| day 6-18 | 77.9 | 79.5 | 82.4 |
| day 18-30 | 75.4 | 75.2 | 78.1 |

(Prior study, 15 pairs: 94.2 / 75.0 / 69.9. **Same numbers, same shape — the
production-only metric holds up.**) Conditioning on the branch recovers only
~2-5 points, which is the quantitative statement that the discrete branch is
*not* the whole story.

---

## 5. Does the adaptation pay? (task question 4)

**Honest answer: not measurable from their games, and the parts that *are*
measurable are flat.**

The rule is deterministic given the observable. Crop Dusta therefore **never
takes the wrong branch**, so there is no counterfactual in the corpus and no
treatment effect to estimate. Every branch-conditional statistic below is
mostly measuring *world type*, not the value of adapting.

All 190 1.32.7 games, `win_metric.summarise`:

| world (branch fired) | n | W-L | score | own bank med | opp bank med | margin med | share of banked money |
|---|---|---|---|---|---|---|---|
| COW (milk shop) | 87 | 58-29 | 0.667 | 96,069 | 90,942 | +4,145 | 0.522 |
| SHEEP (wool shop) | 57 | 42-15 | 0.737 | 104,769 | 95,451 | +8,530 | 0.523 |
| GOOSE (neither) | 46 | 40-6 | **0.870** | 84,764 | 73,510 | +8,982 | 0.535 |
| **all** | **190** | **140-50** | **0.737** | | | | |

- Fisher exact: COW vs GOOSE **p = 0.024**; COW vs SHEEP p = 0.42; SHEEP vs
  GOOSE p = 0.33. `min_detectable(190, 0.74) = 0.126`, so only the largest gap
  is resolvable at all.
- **Market share is flat: 0.522 / 0.523 / 0.535.** Whatever branch fires, they
  take essentially the same slice of the money banked. That is the cleanest
  reading available: *the branch does not change their edge; it tracks the
  world.* Milk worlds are simply harder ones (opponent banks 91k vs 74k).
- Tomato yes (n=63) 0.841 vs no (n=127) 0.685, Fisher **p = 0.023**. Same
  confound: a tomato-shop world is a richer world.

The one place where a **quasi-counterfactual does exist** is the carrot dose,
because the fit is r = 0.77, not 1.0 — within a carrot-demand stratum they
plant materially different amounts. Stratifying on the day-21 tick and
splitting each stratum at its own median dose:

| tick | n | win rate, above-median dose | below-median dose |
|---|---|---|---|
| 1 | 21 | 0.700 | 0.818 |
| 2 | 33 | 0.500 | 0.579 |
| 3 | 44 | 0.800 | 0.792 |
| 4 | 37 | 0.706 | 0.750 |
| 5 | 23 | 0.636 | 0.583 |
| 6 | 24 | 0.917 | 0.917 |

**No signal.** The residual carrot dose beyond the demand read buys nothing
measurable. If we copy this, copy the *coarse* demand response, not the fine
dose.

---

## 6. A specification we could build against

Everything below is inferred from actions, not read from code (see §7). It is
stated at the level of detail needed to implement and A/B it.

### 6.1 Fixed opening (steps 0-143) — identical in every game

```
step 0     HIRE ×4 ; BUY_ANIMAL COW ×2, SHEEP ×2 ; BUY_SEED MELON 5, WHEAT 9
           farmer: BUILD_PASTURE
step 1     BUY_PRODUCT WHEAT 4 ; BUY_ANIMAL COW ×1        (byte-identical
           across every 09-01 episode inspected -> 3 COW + 2 SHEEP by step 1)
steps 0-143  5 × BUILD_PASTURE ; 1 × BUY_LAND (step 121)
             ~21 WHEAT seed, ~11 STRAWBERRY seed, ~5 MELON seed, all planted
             sell WHEAT + FERTILIZER only; buy WHEAT as feed
             no FERTILIZE before day 6
```

### 6.2 The day-6 dispatch — the thing worth copying

```python
# at step 145 == day 6, hour 1  (build B: the same read, acted on at 153-155)
d_wool = prev_market_inv["WOOL"] - market_inv["WOOL"]   # or: WOOL in unlocked shops
d_milk = prev_market_inv["MILK"] - market_inv["MILK"]   # or: MILK in unlocked shops

if d_wool >= 2:      economy = SHEEP
elif d_milk >= 2:    economy = COW
else:                economy = GOOSE
```

Median build-out in the 3 days following the decision (steps 144-192, then
144-288), Sep 1-3, n=134:

| | COW (n=60) | SHEEP (n=33) | GOOSE (n=41) |
|---|---|---|---|
| day 6-8: BUY_ANIMAL | COW ×4 | SHEEP ×4 | SHEEP ×1 + GOOSE ×3 |
| day 6-8: housing | BUILD_PASTURE ×4 | BUILD_PASTURE ×4 | BUILD_PASTURE ×1 + **BUILD_COOP ×3** |
| day 6-8: seed | WHEAT 11, STRAWBERRY 6, MELON 2 | WHEAT 10, STRAWBERRY 7, MELON 1 | WHEAT 8, STRAWBERRY 9, MELON 1 |
| day 6-12 cumulative | COW 5, SHEEP 2, PASTURE 8 | SHEEP 9, COW 1, PASTURE 10 | GOOSE 5, SHEEP 1, COOP 5, PASTURE 1 |
| whole game (09-01 medians) | COW 10, SHEEP 4, PASTURE 14, WHEAT 112, STRAWBERRY 32 | SHEEP 12, COW 4, PASTURE 20, WHEAT 144, STRAWBERRY 21 | GOOSE 7, SHEEP 4, COW 3, COOP 7, PASTURE 10, WHEAT 129, STRAWBERRY 30 |

Note the second-order consequence, which is where most of the day-6-18
divergence actually lives: the housing choice re-budgets tiles, so the SHEEP
economy plants ~30% more wheat and ~35% less strawberry than the GOOSE
economy. `BUY_LAND` is **always exactly 2** (steps 121 and ~198-200) in every
game and every branch — the land plan is not adaptive.

### 6.3 The two later reads

```python
# step 385 == day 16, hour 1
if (prev_inv["TOMATO"] - inv["TOMATO"]) >= 3:
    open a TOMATO block   # median 17 seeds; otherwise zero tomato all game
# 95.8% of 190 games across 9 build-days

# step 505 == day 21, hour 1
t = prev_inv["CARROT"] - inv["CARROT"]
carrot_seeds = round(11.2 * t - 3.2)          # r = 0.769, n = 172
if t >= 4: start buying now (day 21)
else:      wait until step 577 (day 24, hour 1)   # 92.3% on the Sep-1 build

# ~step 241 == day 10, hour 1  (weakest of the four)
if economy != GOOSE and (prev_inv["EGG"] - inv["EGG"]) >= 2:
    add geese + coops    # 76.3%
```

Observed carrot dose by demand, all builds:

| day-21 tick | n | carrot seeds, median [IQR] |
|---|---|---|
| 1 | 14 | 8 [4-10] |
| 2 | 29 | 15 [9-28] |
| 3 | 41 | 26 [20-35] |
| 4 | 33 | 36 [27-53] |
| 5 | 23 | 65 [55-76] |
| 6 | 24 | 76 [54-83] |
| 8 | 5 | 74 [73-92] |

### 6.4 Why this is cheap for us to try

- The observable is **one subtraction on public market inventory**, or a
  membership test on the town's unlocked shop list. Cost per turn is
  negligible; it cannot threaten the 1 s `actTimeout`.
- It is a **route-family selector**, not a per-turn planner. Everything the
  project has failed at (BC policy −0.2885, arm commits −$52,940, planner
  0-64, compiled per-turn search 0-40) has been per-turn judgment. This is
  three `if` statements over pre-authored economies — precisely the shape our
  chassis already supports, and precisely the shape
  `docs/top10-plan-2026-09-03.md` §1.1 says dominates (base selection).
- The natural first experiment is the **day-16 tomato read alone** on an
  existing base: one binary, one threshold, one pre-authored block, and the
  largest single-decision effect in Crop Dusta's corpus (0.841 vs 0.685, though
  see the confound in §5).
- The natural second is the **day-6 3-way herd branch**, which needs three
  authored economies rather than one — a real build cost, and the reason to do
  tomato first.

**Do not port the fine carrot dose.** §5 shows the residual dose buys nothing.
The coarse "if carrot demand is high, commit to carrot early" is the part with
evidence behind it.

---

## 7. What is inference, and what would settle it

**Directly measured, no inference:**
- The action streams, the fork steps, the branch labels, the market inventories
  and prices at every step, the 100%/95.8%/r=0.769 fits, the win/loss records,
  and the leakage and availability checks in §3c-3d.

**Inferred:**
- **That the trigger is the unlocked shop set** rather than the inventory/price
  it produces. Those two are near-collinear at day-6 dawn (one turn of
  consumption is exactly the shop count), so this data cannot separate them.
  Both give the same implementation, so it does not matter much — but the
  paper claim "they read `town.shops`" is a hypothesis, not a measurement.
- **That the thresholds are 2 and 3.** They are the *observed* separating
  values with no episode in between; the true constants could be anything in
  the gap, and could be expressed in price or revenue space rather than unit
  space.
- **That the day-16 and day-21 reads are single-shot at those steps** rather
  than a "first day on which condition X holds" loop that happened to fire
  there. The day-16 tomato read is 89.7% accurate as early as day 12-13, so
  the underlying demand is persistent and an earlier polling loop would look
  much the same.
- **Anything about their code.** No Crop Dusta kernel is public
  (`docs/competitor-intel-2026-09-03.md`'s appendix has none). Every rule here
  is a correlate that happens to be perfect on 134-190 games; a different
  internal quantity that is a deterministic function of the shop draw would
  fit identically.

**The measurement that would settle it**, if quota allows: pull 3-4 replays
spanning the three branch groups on the 09-01 build and read
`town.shops` at step 144. If the unlocked set at day 6 lists WOOL exactly in
the sheep games and MILK exactly in the cow games, the "shop set" reading is
confirmed over the "price" reading. **This was not done: the rolling 24 h
episode quota was already exceeded (2019/2000) when the study began.** Cost is
~4 pulls, ~128 MB, and the analysis script is already written.

**Where the rule is known to be weaker:**
- The 3-way rule is 100% on all builds from 2026-09-01 onward and 95.2% on
  08-21, but only 43-67% on the 08-28…08-31 builds — because **those builds had
  no day-6 goose branch at all** (geese first appear at step 193+ there). As a
  COW/not-COW binary the same rule is 186/190 = 97.9% back to 08-21. So the
  *milk read* is at least two weeks old in their agent and the *goose branch*
  is a recent addition. Expect the specific branch set to keep moving; the
  instrument (one-turn town consumption) is the durable part.

---

## Appendix — artefacts

All scratch, session scratchpad only; nothing written into the repo except this
document.

```
cd_load.py       route loading + production-event extraction
cd_summary.py    per-episode whole-game production vectors      -> cd_prod.json
cd_fork.py       pairwise fork-step distributions (§2a)
cd_branch.py     branch labelling + trace join                  -> cd_rows*.json, cd_arrs.npy
cd_tree.py       depth-1..4 trees on the 49-feature vector (§3b)
cd_rule.py       confusion matrix for the depth-2 rule
cd_step.py       rule accuracy vs read step — the 144->145 jump (§2b)
cd_145.py        the (dWOOL,dMILK,dEGG) tick table (§3a)
cd_shop.py       inventory trajectories + own-sell leakage check (§3c)
cd_tom.py/cd_tc.py/cd_dose.py   day-16 tomato and day-21 carrot reads (§6.3)
cd_clust.py      production-similarity windows + clustering (§4)
cd_wide.py/cd_wide2.py/cd_bin.py/cd_build.py   all-1.32.7 validation, build partition
cd_pay.py/cd_stat.py/cd_last.py  payoff, Fisher tests, availability check (§5, §3d)
```

Related: `docs/closed-loop-intel-2026-09-03.md` (which this resolves),
`docs/competitor-intel-2026-09-03.md`, `docs/top10-plan-2026-09-03.md`,
`.local/memory/cosim-compiler-and-market-truth.md` (shop consumption rates and
the price curves used to interpret the ticks).
