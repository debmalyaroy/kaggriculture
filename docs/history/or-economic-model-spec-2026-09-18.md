# OR / MILP Economic-Model Specification — Kaggriculture Tape Generator

**Task:** B0.1 of the engine-tooling master plan (`docs/history/engine-tooling-master-plan-2026-09-18.md` §B.1/B.1b; `docs/history/implementation-tasklist-2026-09-18.md` row B0.1).
**Status:** SPEC ONLY — no code, no solver run. This is the model a Rust implementer builds from (Workstream B1–B3).
**Date:** 2026-09-18.
**Grounded in:** `vendor/kaggle_environments/envs/kaggriculture/{kaggriculture.py, kaggriculture.json, README.md, AGENTS.md}`. The interpreter `kaggriculture.py` is the source of truth wherever the README prose disagrees (see §11, Confirmed vs Unconfirmed).

---

## 0. Purpose and scope

Produce a **single owned open-loop schedule** (`.tape`) for one seat that maximises the seat's **bank balance at the end of the season** (`Cash720`), assuming a fixed/expected opponent and town-demand environment. The tape is then replayed on the faithful harness (`kagg serve` / `kagg batch`) and its predicted `Cash720` is checked against the realised one (B3.2). The bar to clear: a live ~2500 tape scores **0.917 win-rate** on our faithful harness against `ref_v46` (`.local/memory/rebaseline-faithful-2026-09-18.md`).

The model is **economic-first**: the spatial farm (10×10 tiles, farmer movement) is abstracted into a *field-day production-absorption* layer (§6) so the core is a tractable MILP. Spatial routing (which tile, which mover, walk distance) is a downstream sub-problem (§7.4, CP-SAT), not part of the Layer-1 economic MILP.

### 0.1 The single most important modeling lever

From F3.1 net-P&L over 789 top-100 games (`.local/memory/pnl-cost-leak-2026-09-18.md`): **winners bank 102.8k vs losers 85.6k by realising a HIGHER $/unit on 7 of 9 products while selling LESS volume on 6 of 9.** Losers over-produce cheap staples (WHEAT +238 u/game, FERTILIZER +182 u, EGG +30 u). The market re-quotes each sold unit **downward within the turn** (marginal pricing, §3), so revenue per product is a **concave staircase** and dumping past the knee is nearly free volume at ~$1. The MILP must therefore be able to *choose to withhold supply*. Section 3 encodes this as a piecewise-linear concave revenue function whose later breakpoints have ~zero slope — the optimiser restricts supply on scarcity products **for free**, which is exactly the F3.1 edge. This is the whole reason to prefer an optimiser over a greedy heuristic.

---

## 1. Objective

Maximise the seat's cash at the terminal recorded step.

```
maximise   Cash_720
```

**Semantics (verified in `interpreter`, lines 894–965):** the episode has `episodeSteps = 720`. Actions are applied on steps `s = 0 … 718` (**719 action rows**); DONE fires when `step ≥ episodeSteps − 2 = 718` and `reward = farm["money"]` at that point. So the objective is the bank after the step-718 turn resolves. **Unsold inventory and shed contents score $0** (README "Reward"; `_commit_unit` only moves money on SELL/BUY). Cash is a `float` internally but every price is an integer dollar (`market_price` returns `int`), and land/hire/seed/animal costs are integers, so `Cash_720` is integer-valued.

Money is moved by exactly three interpreter paths — `_commit_unit` (SELL/BUY_*), `_do_hire`, `_do_buy_land`. `BUILD_COOP`/`BUILD_PASTURE`/`PLANT`/`WATER`/`HARVEST`/`FEED`/`CARE`/`DIG`/movement are all **free**. The cash ledger therefore closes exactly (confirmed empirically in `.local/memory/trackp-marginal-unit-2026-09-04.md`).

---

## 2. Indices, sets, constants

### 2.1 Time
| Symbol | Meaning | Value |
|---|---|---|
| `S` | steps (turns) in season | 720; actions on `s ∈ {0..718}` |
| `H` | turns per day | 24 |
| `D` | days | 30, `d ∈ {0..29}` (0-indexed) |
| `d(s)` | day of step | `s // 24` |
| `h(s)` | hour of step | `s % 24` |

End-of-day refresh fires after the step with `(s+1) % 24 == 0` (i.e. `h(s)=23`); day `d` spans steps `24d … 24d+23`.

### 2.2 Economy constants (from `kaggriculture.json` + `kaggriculture.py`)
| Symbol | Meaning | Value |
|---|---|---|
| `M0` | starting money | 3000 |
| `maxOrders` | market orders/turn | 10 (extras silently dropped) |
| `shedCap` | non-seed shed capacity | 100 (end-of-day overflow discarded) |
| `I0` | market start inventory / anchor, every product | 10000 |
| `PRICE_FLOOR` | price floor | 1 |
| `LAND_ORDER` | unlock order | NE, SW, SE (NW free) |
| `LAND_PRICES` | quadrant costs | [1000, 2000, 4000] |
| `fib` hire | n-th hire cost/day | `farmHandCostMult · fib(n)`, `fib = 1,1,2,3,5,8,13,21,…`, resets daily; mult = 1 |
| `weedSpawnChance` | per empty unlocked tile/day | 0.005 |
| `townShopUnlockInterval` | days between shop unlocks | 3 (first unlock end of day 2 → visible day 3) |
| `townShopSellInterval` | turns between shop consumption ticks | 4 |
| `townCenterSellInterval` | turns between town-center ticks | 24 (once/day) |
| `MAX_SHOP_INSTANCES` | cap on unlocked shops | 8 |

**Land tiles:** NW gives 25 tiles; buying NE/SW/SE adds 25 each → cumulative usable tiles {25, 50, 75, 100}. Shed sits at the 4 center tiles `(4,4),(5,4),(4,5),(5,5)`, one per quadrant (§5.5).

### 2.3 Crops (`CROPS`, lines 11–17)
| Crop | seed $ | first_yield_day | max_yield_day | interval | max_yield | ongoing | water window `[⌈mday/2⌉, mday]` |
|---|---|---|---|---|---|---|---|
| WHEAT | 10 | 2 | 4 | 0 | 6 (4 w/o fert) | no | days 2–4 |
| CARROT | 20 | 2 | 3 | 0 | 4 (3 w/o fert) | no | days 2–3 |
| TOMATO | 50 | 8 | 8 | 1 | 4 | yes | (ongoing; production days 8,9,10,11) |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | yes | (ongoing; production days 10,12,14,16) |
| MELON | 80 | 10 | 12 | 0 | 6 | no | days 6–12; cap 6 reached day 10 (day 8 w/ fert) |

### 2.4 Animals (`ANIMALS`, lines 19–23)
| Animal | cost $ | structure (free build) | first_yield_day | interval | max_held | product |
|---|---|---|---|---|---|---|
| GOOSE | 300 | COOP | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 | 3 | 6 | WOOL |

Animals produce **indefinitely while fed** (feed = 1 WHEAT/day; escape at 2 consecutive unfed days). `max_held` caps *unharvested* product on the tile, not lifetime. Every surviving animal yields **1 FERTILIZER/day** regardless of feed/care (`COLLECT_FERTILIZER`). CARE banks +1 (fed AND cared) paid on the next fed production tick (§4.3).

### 2.5 Products and market parameters — **USE THE CODE, NOT THE README TABLE**
`MARKET_PARAMS` (lines 41–51). The README "Price Function" table's *Below func / Below target* columns are **stale** for CARROT, TOMATO, EGG. The **above-I0 (glut / SELL) side matches** the README and is what drives sell-revenue once supply crosses `I0`; the below-I0 side governs realised price while a product sits in scarcity (the empirical steady state — STRAWBERRY/MILK/WOOL/CARROT/EGG/TOMATO stay below `I0` all season, `.local/memory/trackp-market-contest-2026-09-04.md`), so it must also be taken from the code.

| Product | base | T | below_func | below_target | above_func | above_target |
|---|---|---|---|---|---|---|
| WHEAT | 25 | 400 | sqrt | 0.80 | log | 0.20 |
| CARROT | 35 | 450 | **hinge** | **1.00** | sqrt | 0.70 |
| TOMATO | 60 | 200 | **hinge** | 0.40 | sqrt | 0.60 |
| STRAWBERRY | 120 | 100 | sqrt | 0.70 | linear | 1.60 |
| MELON | 250 | 300 | log | 0.20 | sq | 3.60 |
| EGG | 50 | 332 | **hinge** | 0.40 | log | 0.20 |
| MILK | 160 | 122 | sqrt | 0.60 | linear | 1.60 |
| WOOL | 200 | 105 | log | 0.20 | sq | 3.20 |
| FERTILIZER | 100 | 200 | linear | 0.40 | linear | 0.40 |

Only WHEAT and FERTILIZER are buyable (`BUY_PRODUCT`); all nine are sellable.

---

## 3. The market price curve and the staircase revenue PWL (core of the model)

### 3.1 Exact engine price function (`market_price`, `_shape`, lines 61–206)
```
_shape(f, x, T):        # x clamped to ≥0
  linear: x        sq: x²        sqrt: √x
  log: ln(1+x)     log10: log10(1+x)
  hinge(x,T): u=x/T; u + 8.0·max(0,u−1)²      # HINGE_GAIN = 8.0; f(T)=1 by construction

market_price(item, inv):
  if inv < I0:  amp = below_target·base / _shape(below_func, T, T)
                p   = base + amp · _shape(below_func, I0−inv, T)
  else:         amp = above_target·base / _shape(above_func, T, T)
                p   = base − amp · _shape(above_func, inv−I0, T)
  return max(1, int(round(p)))          # banker's rounding, then $1 floor
```

**Banker's rounding (round-half-to-even).** Python `round()` breaks .5 ties to the even integer (`round(0.5)=0, round(1.5)=2, round(2.5)=2`). The implementer MUST replicate round-half-even, not round-half-up, or breakpoint unit values drift by $1 exactly at half-dollar quotes. `max(1, int(round(p)))` applies the floor after rounding.

### 3.2 Per-unit re-quote (marginal pricing) — `_process_market` (lines 544–628)
Market orders resolve **one unit at a time, in lockstep across both players**. For a SELL, the unit is priced at the **pre-sell inventory**; then (only if `price > 1`) inventory increases by 1 and the price is re-quoted downward for the next unit. **A unit sold at the $1 floor does NOT add to inventory** (`_commit_unit`, line 659) — so floor units neither move the price nor add "supply value": they are pure filler worth $1 each. A BUY_PRODUCT unit is priced at **post-buy** inventory (`inv−1`) so a buy/sell round-trip nets zero.

Consequence: **the revenue from selling `q` units of product `p`, starting from market inventory `v₀`, is a concave, integer-valued staircase**:
```
Rev_p(q | v₀) = Σ_{m=0}^{q−1} price_p( v₀ + m )        # for the seat's own units, ignoring
                                                        # concurrent opponent/town units this tick
```
where the sum's summand is non-increasing in `m` (price is monotone non-increasing in inventory above the floor). Once `price_p = 1`, every further unit adds exactly $1 and never advances `v₀`, so the staircase becomes a flat line of slope 1.

### 3.3 PWL encoding for the MILP (B1.1)
For each product `p` and each **selling window** `w` (§3.4), precompute the **cumulative revenue breakpoint curve** `R_{p,w}(q)` for `q = 0,1,…,Qmax_{p,w}` using the exact §3.1–3.2 recursion seeded at the window's expected opening inventory `v0_{p,w}` (§7.2). Then model revenue as a **concave PWL maximisation**:

- Continuous/integer variable `sell_{p,w} ∈ ℤ≥0` = units sold of `p` in window `w`.
- Revenue `rev_{p,w}` modelled as the lower envelope of the secant/segment lines of `R_{p,w}`:
  `rev_{p,w} ≤ R_{p,w}(k) + Δ_{p,w}(k)·(sell_{p,w} − k)` for each breakpoint `k`,
  where `Δ_{p,w}(k) = R_{p,w}(k+1) − R_{p,w}(k) = price_p(v0+k)` is the marginal price at unit `k` (non-increasing ⇒ concave ⇒ these tangent/secant constraints are valid for a **maximisation**, no SOS2/binary needed).
- Objective term `+ rev_{p,w}`.

Because `R` is concave, this "cutting-plane from above" formulation is exact at integer `q` and needs **no binaries** — the classic advantage of concave PWL in a maximisation. Provide the full breakpoint list up to `Qmax` (choose `Qmax` = documented `T` plus the floor-width so the flat $1 tail is represented, e.g. STRAWBERRY 62, WOOL 59, MELON 158; see §3.5). Represent the flat $1 tail as one final segment of slope 1 to keep the column count small.

**Banker's-rounding at breakpoints:** compute each `Δ_{p,w}(k)` with the exact `max(1,int(round(·)))` pipeline (§3.1). Do NOT interpolate the smooth curve — the integer staircase is what the engine pays, and a smooth approximation systematically over-values the middle of the ramp.

### 3.4 Selling windows
Discretise the season into selling windows to keep breakpoint tables finite while capturing that price **persists across days** (inventory is not reset). Recommended: **one window per day** (`w = d`), 30 windows/product, with `v0_{p,d+1}` = `v0_{p,d}` + own sells − town/opponent drain (§7.2 schedule). A per-day granularity matches the town-drain tick structure (§3.6) and the harvest-to-shed dusk delay (harvest reaches the shed only at end-of-day; `.local/memory/trackp-market-contest-2026-09-04.md`). Finer (per-turn) windows are unnecessary for an open-loop economic plan and blow up columns.

### 3.5 Worked staircases (verified against `market_price`)
Starting from `v₀ = I0` (glut onset), units-until-$1-floor and sample marginal prices `price(I0), price(I0+10), +20, +30, +50`:

| Product | floor at `I0 +` | marginal $ at [+0,+10,+20,+30,+50] | P(I0−T) scarcity |
|---|---|---|---|
| STRAWBERRY | 62 | [120, 101, 82, 62, 24] | 204 |
| WOOL | 59 | [200, 194, 177, 148, 55] | 240 |
| MELON | 158 | [250, 249, 246, 241, 225] | 300 |
| MILK | 76 | [160, 139, 118, 97, 55] | 256 |
| CARROT | 842 | [35, 31, 30, 29, 27] | **70** |
| TOMATO | 529 | [60, 52, 49, 46, 42] | 84 |
| EGG | ~50000 (never floors realistically) | [50, 46, 45, 44, 43] | 70 |
| WHEAT | ~50000 | [25, 23, 22, 22, 22] | 45 |

Reads: premium fragile products (STRAWBERRY/WOOL/MILK, and MELON on `sq`) **collapse to $1 within ~60–160 units** — the PWL's near-flat tail is exactly why the optimiser will cap their volume, matching F3.1 winners. Staples (WHEAT/EGG) barely decay: dumping them is not punished, which is precisely why losers over-produce them and gain nothing. Note CARROT scarcity price is **$70** (code's `hinge/1.00` below-curve), not the README table's $42 — this materially raises CARROT's early-season value.

Illustrative realised average: selling 80 STRAWBERRY starting from `I0−50` (a scarce opening) nets **$10,773 = $134.7/u** — consistent with the F3.1 winner realisation of STRAWBERRY at ~$152/u when volume is kept near ~350 and the market stays scarce.

### 3.6 Town demand as the scarcity engine (`_town_consume`, lines 728–749)
Town consumption is the **demand sink that keeps scarcity products valuable** by draining inventory back below `I0`:
- **Town center:** every 24 turns (once/day, flat all season) removes **1 of each non-fertilizer product** (`TOWN_CENTER_PRODUCTS`).
- **Shops:** every 4 turns, each unlocked shop *instance* removes 1 of each product it demands (single-product shops remove 2×). With interval 4 and 24 turns/day → **6 ticks/day**, so a shop demanding wheat drains 6 wheat/day per copy.
- Shops unlock at end of day `d` where `(d+1) % 3 == 0`, drawn **uniformly with replacement** from 8 shop types (`rng.choice(sorted(SHOPS))`), capped at 8 instances. Duplicates possible; **WOOL demand exists only if a YARN_STORE unlocks** — WOOL "pays only in YARN worlds" (F3.1; `.local/memory/trackp-marginal-unit-2026-09-04.md`).

Shop table (`SHOPS`, lines 103–112): BAKERY[EGG,WHEAT] · PIZZA_SHOP[MILK,TOMATO,WHEAT] · BRUNCH_SPOT[EGG,WHEAT,STRAWBERRY] · YARN_STORE[WOOL×2] · ICE_CREAM_SHOP[STRAWBERRY,MILK,WHEAT] · PET_CAFE[CARROT×2] · SMOOTHIE_SHOP[STRAWBERRY,MILK] · FARMERS_MARKET[WHEAT,CARROT,TOMATO,STRAWBERRY].

Because shop identity is random, `v0_{p,w}` is a **scenario quantity** (§7.2), not a constant. The model uses an **expected drain schedule** in the base solve and re-solves per realised shop scenario in rolling-horizon mode (§7).

---

## 4. Production & biological constraints (B1.2)

Model production at the **field-day absorption** level (§6). The constraints below define the per-tile biology the absorption table must respect; the MILP consumes the absorption table, not the raw per-tile dynamics.

### 4.1 One-time crops (WHEAT, CARROT, MELON) — `WATER`/`HARVEST`/`_decay_plants`
- `yield_units` initialises to **1** (`_new_plant`).
- Watering **on the planting day is mandatory**: `consecutive_unwatered` starts at 1; two consecutive unwatered end-of-days → WEED. Every-other-day watering keeps it alive, but **daily** watering during the yield window maximises yield.
- During the window `[⌈max_yield_day/2⌉, max_yield_day]`, each **watered** day adds `+1` (or `+2` if fertilised that day) to `yield_units`, capped at `max_yield` (WATER action, lines 431–444). Fertiliser bonus applies only on watered days.
- Harvestable once `age ≥ first_yield_day`; HARVEST moves `yield_units` to inventory and clears the tile.
- **MELON caps at 6 on day 10** (base 1 + 1/watered-day over days 6–10), so occupying past day 10 wastes the tile; fertiliser reaches cap day 8.
- Decay: `max_lifespan_step = (planted_day + max_yield_day + 1)·24`; then `−1 yield` every other step until WEED (`_decay_plants`).

### 4.2 Ongoing crops (TOMATO, STRAWBERRY) — `_daily_refresh_plants` (lines 769–803)
- Production accrues at **end-of-day**, not on WATER: on scheduled days `days_since_first % interval == 0` add `+1` (or `+2` if watered AND fertilised that day), capped at `max_yield` and at `max_held`=`max_yield`.
- TOMATO fires days 8,9,10,11 (4 productions); STRAWBERRY fires days 10,12,14,16.
- After `production_count == max_yield`, `max_lifespan_step` set → decay to WEED. So a strawberry tile is productive ~days 10–16 then must be re-planted; the MILP should treat each ongoing plant as a fixed **yield profile over its lifespan**, not an infinite source.
- Must be watered daily (weed@2-unwatered) throughout.

### 4.3 Animals — `_daily_refresh_animals` (lines 805–834)
- Require structure (`BUILD_COOP`/`BUILD_PASTURE`, free, on an empty owned tile) then `PLACE animal` (consumes 1 from inventory).
- `FEED` consumes **1 WHEAT** from the mover's inventory/day; 2 consecutive unfed days → **escape** (tile reverts to empty structure). Newly placed animal has `consecutive_unfed = 0` (survives first day unfed).
- Production at end-of-day on scheduled ticks (`first_yield_day`, `interval`): base 1, **+ banked `pending_care_bonus` if fed that tick**, capped at `max_held`.
- **CARE bank:** at end-of-day, if fed AND cared, `pending_care_bonus += 1`. Paid in full on the next fed production tick, then resets. Unfed production tick: base 1 only, bank discarded.
- **1 FERTILIZER/animal/day** available at end-of-day for every surviving animal regardless of feed/care; `COLLECT_FERTILIZER` grabs it (does not accumulate).
- **Terminal-value caveat (`.local/memory/trackp-marginal-unit-2026-09-04.md`):** an animal only contributes to `Cash720` via product SOLD before step 718. Its remaining productive value depends on **age** and remaining days, and its output has *negative* marginal revenue once its product market is floored (WOOL past ~59 units). The MILP must value animals by their **sellable, non-floored** output, net of cumulative WHEAT feed cost — not by a flat per-animal figure. Feed is a real WHEAT sink competing with WHEAT sales.

### 4.4 Fertiliser
Two sources: buy at market (base $100, buyable) or `COLLECT_FERTILIZER` from animals (free, 1/animal/day). Uses: FERTILIZE a plant (doubles the water/production bonus for 3 days) or SELL. Fertiliser has its own price curve (base 100, linear/linear 0.40) and is a legitimate product to sell, but F3.1 losers over-produce it (+182 u) — treat its sale revenue with the same concave PWL (floors slowly) and let the optimiser decide FERTILIZE-vs-SELL by marginal value.

---

## 5. Economy constraints (B1.3)

For each turn/day as appropriate:

### 5.1 Cash non-negativity
`cash_s ≥ 0` for all `s`, where `cash_s = M0 + Σ(sell revenue up to s) − Σ(buys, hires, land up to s)`. The engine stops an order mid-fill when money runs out (`_commit_unit` returns False), and hire/land are no-ops if unaffordable. The MILP enforces `cash_s ≥ 0` as a hard constraint at every turn boundary (or at least per-day, matching window granularity), because the plan cannot spend money it does not yet have. Cash flow is the historically binding constraint on expansion (`.local/memory/trackp-base-economy-2026-09-03.md`: LAND buys were cash-gated to days 9/14).

### 5.2 Shed capacity
`Σ_p shed_p ≤ 100` (non-seed items). End-of-day inventory drop discards overflow; mid-day PLACE/BUY into a full shed is refused. Enforce `shed_p` stock ≤ 100 total at each end-of-day; since unsold product scores $0 and occupies scarce shed slots, this constraint couples production timing to selling timing — you must sell to make room.

### 5.3 Land
Binary unlock vars `y_NE, y_SW, y_SE ∈ {0,1}` with unlock-day vars; cost 1000/2000/4000 charged when unlocked; **order-forced** NE→SW→SE (engine `_do_buy_land` unlocks in `LAND_ORDER` regardless of which quadrant you "want"). Usable tiles at day `d` = `25·(1 + Σ unlocked-by-d)`. Tiles constrain the total field-area available to the absorption layer (§6).

### 5.4 Hire (Fibonacci, daily reset)
Per day `d`, integer `hire_d ∈ {0,1,2,…}` hands. Cost of `hire_d` hands = `Σ_{n=0}^{hire_d−1} fib(n)` where `fib(0)=1, fib(1)=1, fib(2)=2,…` (`_hire_cost`, `_fib`). Cumulative daily hire cost = `[0,1,2,4,7,12,20,33,54,…]` for `0,1,2,…` hands. Resets each day. Model as a small SOS1/indicator selection over precomputed cumulative costs (≤ ~10 breakpoints suffices; the 8th hand already costs $21 for one day).

### 5.5 Labor budget
Each **mover** (1 farmer + `hire_d` hands) performs **exactly one op/turn**. Per-day op budget = `24 · (1 + hire_d)`. Every biological action costs one op: PLANT, WATER (1/plant/day), HARVEST, FEED (1/animal/day), CARE, COLLECT_FERTILIZER, FERTILIZE, DROP/PICKUP/PLACE, plus **movement** to reach the tile. The absorption table (§6) must debit the op cost of a full crop/animal cycle including a movement allowance. Historically labor is *slack*, not binding (10 hands PASSed 150–180 turns/day, `.local/memory/trackp-base-economy-2026-09-03.md`) — but the constraint must exist so the model cannot water 200 plants with one farmer. Note movers spawn at the shed each dawn and drop inventory at dusk; shed adjacency = the 4 center tiles (one per quadrant, 3 locked until bought).

### 5.6 Market-order cap
**≤ 10 market orders per turn**, extras silently dropped. An "order" is one `[op,item,n]` entry (any `n`). So per turn ≤ 10 distinct (op,item) sell/buy/HIRE/BUY_LAND lines; a single SELL can carry a large `n`. Constraint: number of distinct market lines emitted on any turn ≤ 10. At per-day window granularity, spread a day's sells across its 24 turns so no turn exceeds 10 lines — a **packing constraint on the tape emitter** (§9), rarely binding economically since one SELL line drains a whole day's stock of a product.

---

## 6. The T-absorption table (tractability device, master §B.1b)

To avoid modeling all 100 tiles × 720 turns, abstract the farm as a set of **field-slots** each running a **crop/animal program** with a fixed, precomputed **production-absorption profile**.

### 6.1 Definition
For crop `c` planted on day `d0`, define `A_c(d0, ·)` = the vector of units delivered to inventory per day over the plant's life, under a stated care policy (daily water; optional fertilise), together with its **op-cost profile** (ops/day) and **tile-occupancy** (days the tile is held). One "field" = a batch of identical tiles started together; a field's profile = tiles × single-tile profile.

The anchor `T` in `MARKET_PARAMS` is, by the engine's own definition (README + code comment), *"the production capacity of one 5×5 field over a 24-day window at optimal watering, no fertilizer"* — animal `T` pre-discounted 30% for wheat-feed overhead and 1 build day. So `Σ_{24 days} A_c ≈ T` for the product's anchoring crop is the calibration target. This ties the absorption table directly to the price curve: **producing ~T units of a product over a field-cycle is exactly the volume that moves its price by `target·base`.**

### 6.2 Per-tile-day throughput (README "Yield/tile/day", cross-checked)
WHEAT 0.80 · CARROT 0.75 · TOMATO 0.33 · STRAWBERRY 0.24 · MELON 0.55 · EGG(goose) 1.00 · MILK(cow) 0.50 · WOOL(sheep) 0.33. A 25-tile field at steady state ≈ `25 · rate` units/day.

### 6.3 What is and isn't confirmable
The documented `T` values are exact; the **per-day shape** of `A_c` within a cycle (how the 24-day total is distributed, and the exact "setup-day" discount that makes WHEAT `T=400` rather than a naive `25·0.80·24 = 480`) is **not decomposed in the vendor docs**. The implementer should **generate `A_c` by simulating one tile through the interpreter** (or the Rust port) under the stated care policy — this is exact and removes the guesswork — and verify `Σ_{24d} A_c` reproduces `T` to within the setup discount. Do NOT hand-fit the intra-cycle curve; derive it from the engine. Flag any residual >5% between simulated 24-day sum and documented `T` for review (it encodes the setup-day discount, §11).

### 6.4 MILP variables over fields
- `plant_{c,d} ∈ ℤ≥0` — tiles of crop `c` started on day `d` (bounded by free tiles that day and by op budget to service them).
- Product delivered to shed on day `d`: `deliver_{p,d} = Σ_{c→p, d0 ≤ d} plant_{c,d0} · A_c(d0, d)`.
- Shed balance: `shed_{p,d} = shed_{p,d−1} + deliver_{p,d} − sell_{p,d}`, `0 ≤ Σ_p shed_{p,d} ≤ 100`.
- Tile balance: occupied tiles on day `d` ≤ usable tiles (§5.3); a tile is released `occupancy_c` days after start.
- Animals modeled as long-lived field-slots with a placement day, a per-day product/fertilizer profile, and a per-day WHEAT-feed debit.

This reduces the decision space from ~72k tile-turn cells to ~(#crops × 30 days) planting integers + selling integers + a handful of land/hire binaries — solvable as an LP-relaxation-tight MILP.

---

## 7. Decomposition & tractability (B1.4 / master §B structure)

### 7.1 Rolling horizon
Solve a **full-season LP/MILP** for the schedule skeleton, then re-solve a **per-day (or per-3-day) MILP** as shop scenarios realise, warm-started from the previous solution. Open-loop tape means one final pass fixes all 719 rows, but the plan is *built* by rolling forward so `v0_{p,w}` uses observed shop unlocks where available.

### 7.2 Town-drain + shop-unlock as expected/scenario schedule
- **Deterministic** part: town-center drain (1/product/day) and, once a shop is unlocked, its 6 ticks/day are deterministic.
- **Stochastic** part: which shops unlock (with replacement, 8 types, up to 8 instances, at days 3,6,9,…). Build **S scenarios** (e.g. Monte-Carlo draws of the unlock sequence, or the empirical distribution from mined top-100 worlds) and either (a) optimise expected `Cash720` over scenarios (two-stage stochastic MILP with first-stage plant/land, second-stage sells), or (b) optimise the base tape against the **expected** drain and rely on the runtime seat (not this generator) to adapt. Recommendation for B1: **expected-value single scenario first**, then scenario-robust once the base solve is validated. `v0_{p,w}` per scenario = `I0 − Σ(drain up to w) − Σ(opponent sells up to w) + Σ(own+opp sells)`; opponent sells taken from a fixed opponent model (the ~2500 tape being beaten).

### 7.3 Warm-start
Seed the solver with the mined top-30 economy day-indexed schedule (`.local/memory/trackp-base-economy-2026-09-03.md`) as an incumbent; this both accelerates the MILP and guarantees the optimiser never returns worse than a known-good tape.

### 7.4 Spatial routing sub-problem (out of Layer-1)
Given the chosen per-day action multiset (plant/water/harvest/feed/care counts + sells), assign actions to movers and sequence movement so every action is reachable within 24 turns — a **CP-SAT routing/scheduling** feasibility check (§8). If infeasible, add a labor/movement cut back to the MILP and re-solve. In practice labor is slack, so this is a feasibility guard, not an optimiser.

---

## 8. Solver stack & recommendation (B0.2)

| Role | Engine | Rust crate | Integration risk |
|---|---|---|---|
| **Primary** linear MILP | **HiGHS** | `highs` (C-API bindings) + `good_lp` (modeling) | **Low** — pure Rust build of HiGHS via `highs-sys`; mature; handles the concave-PWL MILP (§3.3) which needs no binaries for revenue, only for land/hire. |
| Fallback MINLP | SCIP | `russcip` | **Medium** — SCIP license (ZIB academic) + C dependency; use only if a genuinely nonlinear term (e.g. exact `sq`/`log` price instead of PWL) is required. The PWL avoids this; keep russcip as escape hatch. |
| Routing sub-problem | OR-Tools CP-SAT | `cp_sat` / `cp-sat` (protobuf wrapper) or FFI | **High** — OR-Tools is C++; Rust bindings are thin/less-maintained. Mitigate by making routing a *feasibility* check callable as a subprocess, or reimplement as a small greedy/exact assignment in native Rust since labor is slack. |

**Recommendation:** ship Layer-1 on **HiGHS via `good_lp`**. The entire economic model (§3–6) is a **linear MILP** once revenue is PWL and price scenarios are fixed — no MINLP needed. Reserve `russcip` for research on exact-nonlinear pricing. Treat CP-SAT routing as optional; prefer a native Rust assignment check given labor slack. Prototype-build all three crates (B0.2) with a toy solve to confirm Cargo integration before committing.

---

## 9. Output contract: the `.tape` (B3)

- **Format:** 719 rows (steps 0..718), one row per turn, tab-separated three columns `farmer \t hands \t market`, each a JSON/serialized action list, integer args only.
  - `farmer`: `[op, ...]` (one op).
  - `hands`: `[[op,...], …]` positionally aligned to that day's hired hands (§ house rule 5).
  - `market`: ordered list, **≤ 10 orders/turn** (extras dropped), sells ordered by priority.
- **Emitter obligations (translate §5–6 solution → turns):**
  1. Schedule market lines so no turn exceeds 10; spread a day's sells across its turns; SELL a whole product's day-stock in one line (large `n` is harmless — oversized SELL partially fills per-unit and costs only a queue slot).
  2. Emit HIRE early in the day before hands are needed; hands only exist after HIRE resolves.
  3. Emit BUY_SEED/BUY_ANIMAL before the PLANT/PLACE that consumes them; PLANT is atomically dropped if same-turn demand exceeds seeds.
  4. Respect the harvest→shed dusk delay: product is sellable the day AFTER it lands in the shed for end-of-day-dropped inventory (mid-day DROP/PLACE can make it same-day-sellable).
  5. Water each live plant once/day within its window; feed each animal once/day; never plant past labor capacity (an unwatered plant is a WEED in 2 days + a DIG to clear).
- **Validation (B3.2):** replay the tape on `kagg batch`/`serve`; compare **actual `Cash720` vs MILP-predicted**. The gap = linearization error (PWL vs exact staircase, should be ~$0 at integer sells) + market-assumption error (expected vs realised town/opponent drain). Report both components. A large gap on the market side is a signal to move to scenario-robust (§7.2); a large gap on the production side means the absorption table (§6.3) is mis-calibrated.

---

## 10. Model summary (one screen)

```
max  Σ_{p,w} rev_{p,w}                                   (Cash720; unsold = $0)
s.t. rev_{p,w} ≤ R_{p,w}(k) + price_p(v0+k)(sell_{p,w}−k)  ∀k   (concave staircase PWL, §3)
     shed_{p,d} = shed_{p,d−1} + deliver_{p,d} − sell_{p,d}     (flow)
     deliver_{p,d} = Σ plant_{c,d0}·A_c(d0,d) + animal profiles   (absorption, §6)
     Σ_p shed_{p,d} ≤ 100                                 (shed cap)
     occupied_tiles_d ≤ 25·(1+Σ unlocked)                 (land, order-forced)
     ops_used_d ≤ 24·(1+hire_d)                            (labor)
     cash_s ≥ 0  ∀s                                        (cash-flow, buys/hire/land ledger)
     #market_lines_turn ≤ 10                               (order cap, emitter)
     feed: WHEAT debit 1/animal/day; escape if 2 unfed     (biology, §4)
     plant/water/harvest/decay/CARE per §4 absorbed into A_c
 vars sell_{p,w}∈ℤ≥0, plant_{c,d}∈ℤ≥0, hire_d∈ℤ≥0, y_{NE,SW,SE}∈{0,1}, animal placements∈ℤ≥0
```

Objective bar: beat a ~2500 tape at **0.917** win-rate on the faithful harness vs `ref_v46`.

---

## 11. Confirmed vs unconfirmed constants

**Confirmed from vendor files (exact):** all of §2 (json + py), all `CROPS`/`ANIMALS`/`SHOPS`/`MARKET_PARAMS`, the price function and `_shape`/`hinge` (HINGE_GAIN=8.0), per-unit lockstep marketing, $1-floor-adds-no-supply rule, banker's rounding, Fibonacci hire + daily reset, land order/prices, shed cap + overflow discard, town-drain cadence, shop table + with-replacement unlock + 8-cap, CARE bank mechanics, decay rules, action-cost model, 719 action rows / DONE at step 718.

**Could NOT confirm from vendor files (flagged, do not invent):**
1. **Intra-cycle shape of the absorption table `A_c(d0,·)`** and the exact **setup-day discount** that makes documented `T` lower than the naive `25·rate·24` (e.g. WHEAT `T=400` vs `480`). The docs state `T` "bakes in setup-heavy opening days" but give no per-day decomposition. → **Derive by simulating one tile through the engine** (§6.3), not by assumption.
2. **Realised opponent sell schedule and shop-unlock distribution** for `v0_{p,w}` — these are episode/opponent-specific; take opponent sells from the fixed ~2500 tape and shop unlocks from mined top-100 worlds or Monte-Carlo (§7.2), not from any constant in the vendor files.
3. **README "Below func/target" for CARROT/TOMATO/EGG** is **wrong vs code**; §2.5 uses the code. This is a documented divergence, not an unknown.
4. Melon README "bonus window 6–12" vs code cap-at-day-10 — reconciled: base 1 + 1/watered-day hits cap 6 on day 10 (verified); days 11–12 add nothing. Use day-10 effective harvest.
```
