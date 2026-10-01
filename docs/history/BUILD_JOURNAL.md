# BUILD_JOURNAL.md

What was built, in order, and what each step measured. Negative results are
kept deliberately -- most of the cost of this project has been re-testing ideas
that were already disproved.

---

## 2026-08-04 -- v0 to v4

| Build | Idea | Measured |
|---|---|---|
| `v0_baseline` | plant, water, harvest, sell | ~$10k |
| `v1_heuristic` | every job priced in dollars, Hungarian assignment | ~$51k, 100% vs v0 |
| `v2_tuned` | coordinate descent over 28 knobs | ~$83k, 100% vs v1 |
| `v3` | `travel_weight` 5.0 + Voronoi zoning + fertilizer weight | 81% vs v2 over 16 matches |
| `v4_optimal` | max-weight matching instead of greedy pairs | submitted |

Rating reached **683.7**, which is the field median. Local win rates against
our own lineage kept improving while the public rating did not move. That
disconnect is the whole story of the next day.

### Disproved on 2026-08-04 (do not retry without new evidence)

* `water_window_priority` 3.0 -> 0% win rate. Ported from another notebook's bug
  table; their bug needs a priority *ladder*, and we price jobs in dollars.
* `drop_stack_value` 900 -> 25%. Inventories auto-drop nightly.
* `cost_per_animal_day` 3.5 -> 80% over 10 matches, 56% over 16. **Always run a
  third independent seed set.**
* strawberry/melon price swap -> 0%, -$10,056.
* Rescue pens for shed-stranded animals -> 62.5% vs 81%.
* `travel_weight` 8.0 -> 0%. The optimum is ~5.

---

## 2026-08-05 -- the diagnosis

### The ladder said we were average; local tests said we were winning

Pulled our own 10 most recent ladder replays. No timeouts, no errors, `DONE` in
every seat: **5W-5L, banks $57k-$99k against opponents banking $39k-$95k.** The
agent was not broken, it was ordinary.

Pulled three of the highest-rated episodes of the day (avg score 2,996) and
profiled them with a new tool (`tools/replay_profile.py`) that reconstructs each
product's cash flow exactly from the shared market inventory trail.

| | our v4 | top of ladder |
|---|---|---|
| final bank | 57.5k | 110-131k |
| sheep | **0** | 6 |
| wool revenue | $0 | **$40k/player** |
| fertilizer sold | ~0 | 216 units, ~$12k |
| first animal | day 11 | **day 0** |
| wheat tiles | up to **72** | 0-11 |
| strawberry tiles | 8-24 | 40 |
| unit-turns worked | 4,972 | 7,321 |

Movement efficiency -- the thing v3 spent its effort on -- was *worse* on the
winning side (56% of unit-turns walking against our 44%). The difference was
never motion. It was the asset mix.

### Then the notebooks

`kaggle kernels pull` on the ranked competitors' public notebooks, and
`tools/notebook_extract.py` to decode the base85+zlib payloads inside them. The
agents turned out to be **719-turn action recordings** with a slip-recovery
layer, a mirror-probability detector and a 7-feature logistic model. Two
independent leaders shipped the same architecture and nearly the same opening:
3 cows and a sheep on turn zero, ten wheat and seven melon in the ground,
fertilizer on the market from day 2, two land purchases, 12-15 hands.

### The tape bug

Building sparring partners from those recordings produced agents that banked
$24k instead of $130k. Cause: in a Kaggle replay `steps[i]["action"]` is the
action that *produced* state i. Every tool in the repo that paired an action
with an observation was one turn out. After the fix a taped game reproduces to
within the weed seed -- $124.8k/$134.4k against a recorded $130.7k/$131.3k --
and we finally have an opponent worth losing to.

Second bug found the same way: `obs["step"]` exists only for seat 0. Anything
keyed on it silently read 0 forever in seat 1, which reset nothing and made the
first tapes replay their opening move 720 times.

---

## 2026-08-05 -- the rebuild

Changes to `agents/v1_heuristic.py`, each traceable to a number above.

| Change | Why |
|---|---|
| `SELLABLE` includes FERTILIZER | one free unit per animal per day, ~$20k a season, never sold |
| market **outlook** pricing | value a tile against the market it will be sold into: standing supply on both farms, minus the town's published appetite |
| **mirror prior** | on turn zero neither farm has planted, so a census says melon is worth $250; assume the opponent grows what we grow |
| **marginal** tile pricing | price tile eleven against the ten already planned, which caps melon at its demand instead of at a hand-set number |
| animal value includes fertilizer + care | a cow pays for itself in three days, so buy on turn zero |
| feed is **bought**, not grown | frees the land v4 was filling with wheat |
| runway = days until income | a herd bought against a field of melons has no income for ten days |
| explicit **cash crop** | the fast crop is not competing on rate, it is buying the days in between |
| livestock before seed and land | and animals can be bought with no pen standing -- BUY_ANIMAL fills the *shed* |
| hire to 12-15 | twelve hands cost $376 a day against thousands in revenue |

**Measured, 4 seeds x 2 seats:** 100% win rate against `agent_v4_optimal`,
$92,538 against its $69,090. Against the ladder tapes: still 0%, but the margin
went from -$100.7k to -$52.4k over the CEM run.

### Tooling built the same day

* `tools/replay_profile.py` -- exact per-product cash flow from a replay
* `tools/notebook_extract.py` / `tools/notebooks.py` -- harvest and decode public notebooks
* `tools/tape_analysis.py` -- read an opponent's recorded trajectory day by day
* `tools/optimize.py` -- CEM policy search scored against chosen opponents
* `tests/test_contract.py` -- the submission contract, checked mechanically
* registry now stores params, file hash, per-opponent measurements and submissions

### Still open

* 0% against the top tapes. The gap is concentrated in days 0-10.
* Worst-case turn latency rose to ~90 ms in the full test (limit 150 ms, Kaggle
  allows 1,000 ms on slower hardware). Watch it; do not let the outlook
  recompute per ensemble member.

---

## 2026-08-06 -- reading their code, not their prose

The write-ups say the market channel carries the edge. Reading the *decoded
agent* says what the market channel does. Two mechanics came out of
`_projected_shed` and `_front_run_market`; both are interpreter facts we had
simply never used.

### same-turn selling

`kaggriculture.py:904` applies every unit action, `:910` then processes the
market. Produce dropped this turn is sellable this turn. Every build to date
read `private["shed"]` as it stood at the start of the turn, so **every harvest
reached the market a turn late**, and a unit standing on the shed about to drop
twelve wool contributed nothing to the sell plan. This is the mechanism behind
the leak audit's 19%-of-turns against the ladder's 47%.

### sell-first ordering

`_process_market` iterates by order *index*, pairing our i-th order against
their i-th and quoting both against the same pre-commit inventory. The lower
index sells into a market the other player has not moved yet. Our queue led
with HIRE -- 43% of every slot we spend -- so our premium lines were routinely
quoted after theirs. Theirs is literally `action["market"] = (sells + remainder)[:10]`.

| change | seed set A | seed set B |
|---|---|---|
| same-turn selling | 75%, +$9,659 | 83%, +$4,512 |
| + sell-first | 100%, +$7,268 | 90%, +$5,675 |
| **v14 vs v9_cem** | **100%, +$19,934** | |
| v14 vs the live submission | **100%, +$49,205** | |

Local Elo **944 over 62 matches (98%)**, against v9_cem's 854. Bank against a
2,900-rated tape rises from ~$77k to ~$102k, though it still wins none of them.

### Implemented from their code and rejected on measurement

* their sell-slot score, `(1 + opponent_exposure) x glut_weight x price x
  log1p(qty)` -- **25%, -$2,158**. Good for a tape that knows its own volumes;
  on a planner that re-prices every turn it fights the valuation.
* town-schedule hour bias, from 304 mined episodes showing 11.6% of the field's
  volume in hour 0 and 8.8% in hour 1 -- **50%, +$505**. That concentration is
  the engine's overnight auto-drop refilling the shed, not price timing.
* terminal sells-only ("final step 718: replace useless BUY/HIRE with all
  SELLs") -- **$0 at 6 turns, -$596 at 24**. We already avoid terminal waste.

Three of five imported mechanics were worth nothing. The two that paid were
both *interpreter* facts rather than strategy, which is the pattern worth
remembering: read the engine, and read their code against the engine.

---

## 2026-08-06 (evening) -- the meta moved, and six things were quietly wrong

### Where we actually stand

Rank **301 of 2,425** at **2,269.9** -- the copied v21.1 notebook, 22 episodes
in and still climbing (it was 1,502.4 the same morning). Our own `v14_market`
sits at 873.7 over 24 episodes. Rank 200 is 2,539; rank 100 is 2,751; the top
20 occupy 2,933-3,031.

### Routes are perishable, and that is the whole game

Reading the current public notebooks rather than our own lineage:

| Complete route, no market intervention | Wins |
|---|---:|
| old public v21 / Dennis medoid | **19/46** |
| current Konstantin medoid | 40/46 |
| current Richard Silence medoid | 41/46 |

Same agent, same safety layers, different route. The top three current families
differ at ~110 channel moments and sit more than 1,200 from the stale one.
Freshness beats cleverness.

Two corroborating facts from our own mined data. The top and bottom deciles of
300 winner-seats run **identical farms** at day 20 (39.9 vs 39.8 strawberry,
8.0 vs 8.0 cows); the whole 1.9x bank spread opens between days 18 and 29. And
day 29 alone carries **11.8% of the season's sell volume**. It is a selling
difference, not a build difference.

### Opponent-move prediction: a published negative result

`fleongg` reached a verified **2830.4** by taking the v21 hazard-table
preemption and *disabling* it -- `_PREEMPT_THRESHOLD` 0.55 -> 2.0, a threshold a
probability can never cross. Fukami's own ablation agrees: creating new early
SELLs collapsed at ~28 interventions/game, while **re-ordering** already
scheduled sells scored 53/53. Predicting a sale does not establish the value of
taking it early. Do not re-test this without new evidence.

### Six defects, each of which silently corrupted a measurement

| Where | What | Consequence |
|---|---|---|
| `dashboard/serve.py:905` | `/\/g` for `/\/g` | **the entire dashboard UI was dead** -- a parse-time JS error kills every handler in the one script block, while `/api/snapshot` kept answering perfectly, so it read as a stale browser cache |
| `tests/test_contract.py` | globbed `agents/v9_*.py` | v10/v13/v14 were **never contract-tested**, including the agent on the ladder |
| `tools/mine_top.py:182` | sampled hour 0 | `top_trajectory.csv:hands` was 0.0 in all 9,000 rows; `money` read ~13% low |
| `tools/notebooks.py:126` | diffed a glob | re-runs reported `decoded=0` for everything already decoded |
| `tools/notebook_extract.py` | needed a *sequence* of literals | single-line b85 payloads were skipped; two notebooks carrying real agents now decode |
| `dashboard/serve.py:298` | env dict built twice | `PYTHONUNBUFFERED`/`KAGG_VERBOSE` never reached a child |

Plus four `running` rows in `runs.json` orphaned by a killed server, with no UI
path to clear them -- `reap_orphans()` now runs at startup.

### A correction we had been carrying

**`obs["step"]` works in both seats.** It is `"shared": true` in the base
schema and `core.py:__get_shared_state` copies it before the agent is called; a
two-seat probe returns 0..718 in each. What is seat-0-only is the **stored
replay**, where shared properties are stripped from non-first agents. The
original bug was real, but it lived in replay parsing, not in the live
observation -- so an open-loop tape keyed on `obs["step"]` is safe, which is
exactly what every scoring agent does.

### `tools/engine_check.py`

A public probe (`muneeb2405`) measured a **Kaggle notebook image running a
different engine than the ladder**: `startingMoney` 2000 vs 3000, `COW` 600 vs
400, `farmHandCostMult` 10 vs 1, `MELON above_target` 0.90 vs 3.60, and
`SELL FERTILIZER` silently dropped. None of that raises; it just answers a
different question with total confidence.

So: 14 named constants that have been seen to drift, a sha256 fingerprint over
all 121 economy constants, and a behavioural probe that actually sells a
fertilizer and watches the money. `require()` is now called by `evaluate`,
`elo`, `optimize`, `trajectory` and `submit`, cached per process and inherited
by pool workers through `KAGG_ENGINE_CHECKED`. This box records fingerprint
`95854b693b4183cc`; the baseline is committed at `data/engine_baseline.json`.

`tests/test_system.py` now asserts both directions -- that this box passes and
that a COW-at-600 drift is rejected by name. A guard that cannot fail is
decoration.

---

## 2026-08-06 (late) -- the route pipeline, top-200, and one button

### `tools/routes.py` -- fresh routes off the public archive

Streams: download a ~27 MB replay, extract the 719-turn route, **delete it**,
move on. Disk stays flat against a 21 GB/day archive. Three disciplines are
baked in, and each came from a failure the first test run produced.

**The slice is the top 200, not the top 30.** The rating floor at rank 200 is
~2,590 against ~2,900 at rank 30, so the extra 170 teams are still far above the
687 median. And a 30-team panel is small enough that one replay-heavy family
dominates it -- exactly what the medoid rule exists to stop.

`leaderboard_top(200)` returned **20**. `leaderboard --show` is capped at 20
rows however many you ask for, so every request for a deeper slice had been
silently answered with the same 20 teams. Now reads the full CSV from the
download endpoint: 2,472 teams, and the rating floor is *derived* from the
score at rank `--top` rather than hardcoded -- this field grew 517 teams in 22
hours, so a constant would already be wrong.

### Three bugs the first real run found

| | |
|---|---|
| **`?_s0.json.gz`** | `str(A or B if cond else "?")` binds the ternary across the `or`, so every file without a `-` in its name -- which is every file downloaded straight from the daily dataset -- got episode id `"?"`, and the mine died several hundred megabytes in on an invalid Windows filename. Fixed, plus `route_path()` now sanitises whatever it is handed. |
| **contamination** | A test run over `.local/mine` indexed 12 routes banking $39k-$99k -- **our own mid-field games** -- into what was labelled a top-30 route mine. There is now a rating floor, defaulted to the live score at rank `--top`. |
| **zero fit routes** | The published 6/3/rest window split assumes ~17 routes per team. With 6 it puts all six in the outer holdout and leaves **nothing to fit on**, while reporting a successful mine. The split now scales below the threshold and says when it had to. |

### Efficiency: reject on the header

Borrowed from `llccqq624/kaggriculture-replay-data-miner`: read the first
~128 KB and pull `TeamNames`/`rewards` out of the bytes rather than
`json.load`-ing 27 MB. Measured **1 ms against 1,651 ms**.

The first version only rejected teams outside the slice. But the top of the
manifest is dominated by a handful of very active teams, so most of the budget
went on fully parsing replays for teams already at their per-team cap. Adding
the cap to the header check took the rate from **20 routes per 1.33 GB to 39
per 1.43 GB** -- and the first number was still falling.

### `tools/tape_runtime.py` -- what a route ships inside

Our old tape template had a position-repair layer and nothing else. Fine for a
sparring partner, which only has to be faithful; not enough to score with. The
new runtime carries the four layers every scoring agent has, each an
interpreter fact rather than a strategy:

* **projected shed** -- `:904` applies unit actions, `:910` processes the
  market, so produce dropped this turn is sellable this turn;
* **clamping** -- `_commit_unit` returns False the moment the shed empties and
  the *whole order* is abandoned, having burned one of ten slots;
* **sell-first ordering** -- `_process_market` pairs order i against order i
  against the same pre-commit inventory. Order only, never inventory;
* **terminal liquidation** -- unsold inventory scores nothing and day 29 is
  11.8% of the field's season volume.

First build: `89923171_s1` (Subin An, rank 12), recorded **$154,615 against
$151,458**, 22 KB, and it passes `conformance.py` clean.

### `tools/autopilot.py` -- fetch, train, build, gate, submit

One command and one dashboard button. Five stages, strictly sequential, sized
in *iterations* from the core count so a 4-core Kaggle notebook does fewer
trials rather than truncated ones. Promotion requires beating the incumbent on
seeds `--seed0 + 1000*cycle`, which the search never saw. Nothing uploads
without a typed `SUBMIT`, and `scripts\autopilot.ps1 -Schedule` **refuses** to
schedule the submit stage at all -- only the latest two submissions stay
active, so an unattended upload is a way to evict a better agent overnight.

The architecture does not change between cycles. `agents/v1_heuristic.py` is
the only policy source; a cycle rewrites its `PARAMS` and `SELL_SCHEDULE`
blocks. That is what makes two cycles comparable.

### v15_route -- the first mined route, measured

`89981837_s0` (GUAM, rank 40), chosen from the **fit window only** -- `--build
best` now refuses to look at the outer or validation windows, because a route
picked for scoring well in them cannot then be measured by them.

| opponent | win% | our bank | theirs | margin |
|---|---:|---:|---:|---:|
| `v14_market` (our best) | **100%** | 162,304 | 107,979 | **+54,325** |
| `v9_cem` | **100%** | 159,472 | 96,128 | **+63,344** |
| tape 90036815 (2,900-rated) | 17% | 95,826 | 98,303 | -2,478 |
| tape 90041552 (2,900-rated) | 17% | 95,526 | 100,208 | -4,682 |

Two things worth reading carefully.

**The $162k is not the real number.** Against a peer, both players sell into the
same shared inventory and the price curve does the rest -- banks compress to
~$95k for both. The $160k figure is what a strong route earns against weak
opposition, which is why local bank against our own lineage is such a poor
guide to ladder strength.

**The tape result is the real one, and it is the largest single move this
project has measured.** `v14_market` against that same tape was **0% and
-$54,149**. v15_route is 17% and **-$2,478**. A ~$52k swing in head-to-head
margin, from "never competitive" to "loses narrowly".

It also banks **$160,888 against the source route's recorded $151,824**, so the
safety stack is adding value over a raw replay rather than merely preserving it.

**Latency: mean 0.1 ms, worst 0.3 ms**, against v14's 4.7 ms mean and a worst
that has read anywhere from 125 to 234 ms. A tape does no planning, so the
whole latency risk on the submission contract disappears. Size drops from
109 KiB to 22 KiB. Self-play validation: DONE, 94,229 / 94,229 -- identical
banks, which is what a deterministic tape against itself should produce and a
useful check that the runtime has no seat-dependent behaviour.

---

## 2026-08-07 -- selection by measurement, and the first win over a top tape

### Bank is a bad selector, and the tournament proved it

`--build best` ranked routes by the bank they recorded. Eight fresh candidates
plus two incumbents, 2 seeds x 2 seats, 36 games each:

| route | rank | team | win% | mean bank |
|---|---:|---|---:|---:|
| `90551213_s1` | 7 | Dmitry Larko | **89%** | 120,855 |
| `90555869_s1` | 3 | Raj Aryan | 89% | 120,623 |
| `90557412_s0` | 5 | Just a moroccan | 78% | 120,438 |
| `90543557_s1` | 6 | Akhil Chinta | 72% | 120,337 |
| `90556633_s1` | 8 | Chloe | 61% | 120,365 |
| `90288945_s1` | 6 | Konstantin03 | 44% | 111,049 |
| `90288225_s0` | 5 | Konstantin03 | 33% | 111,088 |
| `90288253_s1` | 6 | Konstantin03 | 22% | 107,517 |
| **v15_route** | - | incumbent | **11%** | **142,753** |
| v14_market | - | incumbent | 0% | 92,217 |

**v15_route has the highest mean bank in the field and the second-worst win
rate.** That is the whole argument in one row: a route that banks $142k against
weak opposition loses to one that banks $120k against strong opposition,
because the recorded bank measures the *game*, not the trajectory. Selection is
now by playing the candidates (`tools/routes.py --select`), and the fit window
is the only pool it draws from.

**Konstantin03's three routes came 6th, 7th and 8th** at 22-44%, despite that
team ranking 5-6 when the routes were captured. Their live rank is now 49.
Route decay, measured on our own data rather than quoted from a write-up.

### Mining the freshest day is what changed the ceiling

The 2026-08-06 daily dataset published overnight: median episode rating 2,973
against 2,767 for the days already mined. One 2 GB pass over it returned routes
from teams at **live ranks 2, 3, 4, 6, 7, 8** -- scores 3,021 to 3,102. The
earlier mine, over older days, had topped out at rank 40.

### `agents/v16_route.py` -- and the tape finally falls

`90551213_s1` (Dmitry Larko, live rank 7, score 3,039.7). Gated on seed set
77000, which the tournament never touched:

| opponent | win% | our bank | theirs | margin |
|---|---:|---:|---:|---:|
| `v15_route` | **100%** | 127,996 | 113,729 | +14,268 |
| `v14_market` | **100%** | 161,307 | 94,861 | +66,446 |
| **tape 90036815 (2,900-rated)** | **100%** | 128,303 | 112,589 | **+15,714** |

Against that one tape, across the project:

```
v9_cem       0%     (margin about -54k)
v14_market   0%     -54,149
v15_route   17%      -2,478
v16_route  100%     +15,714
```

From never winning a single game to winning every game. Local Elo **1112 over
132 matches, 100%**. 20 KiB, mean turn 0.11 ms, worst 2.3 ms, self-play
validation DONE at 120,182 / 120,182.

### The contract test now scopes its latency budget

Widening the agent glob pulled in eleven legacy agents that are over the turn
budget -- the v5 ensembles run 21-58 ms mean and up to 410 ms worst, which is
precisely why that line was abandoned. Legality and imports are still asserted
for all of them; the *latency* assertion is scoped to shippable agents, with
the legacy set named and printed. A dead experiment failing the suite costs the
signal from every other check, which is how a real regression gets waved
through.

---

## 2026-08-07 (later) -- price impact, read out of v22

### The decomposition that made it worth implementing

Kaito Fukami's v22 publishes the ablation rather than only the code, and the
decomposition is the whole reason this was worth our time:

```text
old route + old market memory       1/46
current route, market unchanged    36/46
current route + price impact       44/46
```

Route refresh repaired the basin; the order controller added eight more wins
without changing a single production or inventory decision. We had already done
the first half. The second half is one function.

### The mechanic

Score a SELL by **what it does to its own price**:

```
impact = quantity x (quote at current inventory - quote after selling that quantity)
```

Highest first. Every unit sold raises that product's inventory and lowers its
quote, and the curves are convex above equilibrium and differ sharply by
product -- melon and wool are `sq`, milk and strawberry `linear`, wheat only
`log`. So a large melon order landing after a large wool order is quoted into a
market our own wool already moved.

The constraint is what makes it safe: SELLs are permuted **within the slots they
already occupy**. No order created, none resized, and every non-SELL keeps its
index. This is the same order-only invariant that scored 53/53 where the
variant that *created* early sells collapsed.

### Their transcription is exact, and we checked

They reimplement the price curve inside the agent rather than importing it.
Against our vendored 1.32.4 interpreter: **9/9 constants identical, and maximum
price difference zero across 2,835 points** spanning inventory 9,000-11,200.
Worth verifying rather than trusting -- a transcribed constant is exactly the
kind of thing that silently answers a different question.

### Measured on our own agent

An otherwise byte-identical control (`_IMPACT_SLOTS = False`, 4 lines differ):

| seed set | games | win | margin |
|---|---:|---:|---:|
| 88000 | 10 | 100% | +$1,565 |
| 123000 | 24 | 100% | +$1,811 |
| **total** | **34** | **34-0** | **+$1,811** |

Small, and never negative. Our own discipline calls anything under ~$3k noise,
which is why the third independent seed set mattered: the sign held over 34
paired games.

**It also corroborates their number almost exactly.** Their reported mean
margins were -7,839 (old route), +1,838 (current route), +3,509 (current +
impact) -- so their price-impact contribution was **+1,671**. We measure
**+1,565 and +1,811** from an independent reimplementation. That agreement is
better evidence than either number alone.

### `agents/v17_route.py`

v16's route plus impact ranking. Gated on seed set 31000:

| opponent | win% | margin |
|---|---:|---:|
| `v16_route` (same route, no impact) | 100% | +1,335 |
| `v14_market` | 100% | +72,302 |
| tape 90036815 (2,900-rated) | 100% | +12,779 |
| tape 90041552 (2,900-rated) | 100% | +11,910 |

### The other notebook

`anasriaz/kaggriculture` is a fork of the v22 analysis -- same 1/46 - 36/46 -
44/46 table, plus price-curve plots. No independent contribution, but a useful
model for how to present one, and its EDA framing is what our own submission
notebook now uses.

### Against the top-100, on games selection never opened

`tools/panel.py` builds one opponent per top-N team from the **outer and
validation** windows only, so a route cannot look good here merely because it
was chosen here. 42 opponents, 3 seeds, both seats, 252 episodes:

```
OVERALL  238/252 = 94.4%
         beaten outright: 40/42 opponents
         mean margin +13,151
```

| rank | team | win% | margin |
|---:|---|---:|---:|
| 1 | THUNDER THUNDER | 100% | +8,836 |
| 3 | Raj Aryan | 100% | +2,148 |
| 6 | Konstantin03 | 100% | +7,582 |
| 6 | Akhil Chinta | 67% | +379 |
| **7** | **Dmitry Larko** | **17%** | **+0** |
| 8 | Chloe | 100% | +2,172 |
| **9** | **Seb (allegedly)** | **17%** | **-5,146** |
| 76 | Kaito Fukami | 100% | +23,379 |
| 95 | fle3n | 100% | +22,004 |

Two honest notes.

**The margin against Dmitry Larko is exactly zero.** That is our own route's
team: we are replaying their trajectory against a held-out game of theirs, and
in the seats where both sides run the same route the episode is a perfect
mirror. It is not a loss so much as a tie we cannot break, and it is a
structural ceiling of replaying someone else's route -- we can never beat the
player we are copying.

**Seb (allegedly) at rank 9 genuinely beats us**, -$5,146 over six games. One
real opponent on this panel is simply stronger.

One outlier worth flagging rather than averaging away: Winter Lamb (rank 50)
reads 179,805 against 23,736. A margin of +156k is not a policy result, it is
an opponent whose recorded route fell apart on a different weed seed. Its
presence lifts the mean margin; the 40/42 count does not depend on it.

---

## 2026-08-07 -- splicing fails, and why we could not beat 42/42

### The question

Against the top-100 held-out panel, `v17_route` beat 40 of 42 opponents. The
two it did not:

* **Dmitry Larko, rank 7 -- margin exactly $0.** That is the team whose route we
  replay. Both sides run the same 719 actions, so the episode is a perfect
  mirror. Not a loss: a tie we cannot break.
* **Seb (allegedly), rank 9 -- -$5,146.** A genuinely stronger opponent.

The mirror looked like a structural ceiling of shipping one team's route
verbatim, so the obvious fix was to make the route nobody's: take the field
channel from one donor and the market channel from another.

### Splicing does not work, and the failure is informative

`tools/routes.py --splice-search`, 3 field x 3 market donors + the incumbent,
2 seeds x 2 seats, 36 games each:

| field donor | market donor | win% | mean bank |
|---|---|---:|---:|
| THUNDER THUNDER | THUNDER THUNDER *(not a splice)* | **100%** | 146,497 |
| v17_route | *(incumbent)* | 89% | 139,760 |
| Raj Aryan | Raj Aryan *(not a splice)* | 72% | 140,190 |
| **THUNDER THUNDER** | **Raj Aryan** | **33%** | **36,843** |
| **Raj Aryan** | **THUNDER THUNDER** | **6%** | **12,900** |

Every genuine cross-splice collapses -- banks fall from ~$146k to **$12,900 -
$36,843**, a 4-11x destruction. The search correctly picked a non-spliced route.

The reason is that "market channel" is not a synonym for "selling". The market
queue also carries `BUY_SEED`, `BUY_PRODUCT` (the wheat feed pump, ~980 units a
season), `BUY_ANIMAL`, `HIRE` and `BUY_LAND`. Bolting one route's purchasing on
to another's field play buys seed for crops that are never planted, feed for a
herd of a different size, and land on a schedule the labour curve does not
match. The shed clamp keeps it *legal*; nothing can keep it *solvent*.

So prvsiyan's "the farmer and hand tapes contribute almost nothing; the market
tape carries the frontier gain" means **optimise the market channel**, not
**the channels are independent**. Worth having tested, and worth not testing
again.

### On the mirror, reconsidered

A tie against the team we replay is less bad than it first reads. Faithfully
replaying a rank-7 route should converge on roughly a rank-7 rating; the mirror
is the *floor* of that, not a defect. What lifts us above the donor is the part
that is ours -- the safety stack and price-impact ranking, worth +$1,811 over
the raw route -- and that is why 40 of 42 fell rather than 20.

The real answer to "why not 42/42" is that **42/42 is not the right target**.
No agent on this ladder beats every other; the rank-1 team does not. The
measurement that matters is where our win rate sits against the same panel that
the best routes face, which is why `tools/panel.py` now also gets pointed at
the panel's own members.

### `agents/v18_route.py`

The splice search surfaced something better than it was looking for: a
**rank-1 route** (THUNDER THUNDER, `90521287_s0`, mined 2026-08-06, recorded
$139,451 vs $137,951). On fresh seeds:

| opponent | win% | margin |
|---|---:|---:|
| `v17_route` | 100% | +2,049 |
| tape 90036815 (2,900-rated) | 100% | +8,956 |

### Panel coverage

The first top-100 panel had 42 opponents because **65 of the top 100 had never
been mined** -- we only had whoever happened to surface at the top of the
manifest. A targeted `--top 100` mine took the index from 255 to 359 routes and
the panel from 42 to **53 opponents**.

### The calibration that answers "why not 42/42"

Point `tools/panel.py` at one of the panel's own members and the question
answers itself. `opp_r001_90559891_s1` -- the **rank-1 team's** route, chosen by
rank alone -- against the same 53-opponent panel:

```
OVERALL  56/212 = 26.4%
         beaten outright: 4/53 opponents
         mean margin -98
```

Distribution: **4 wins, 20 ties, 29 losses.** And the losses cluster tightly --
-3,820, -3,702, -3,680, -3,703, -3,575, -4,450 -- a systematic ~$3,700 deficit
against most of the field.

Two conclusions, and the second is the useful one.

**No route beats the whole panel.** The rank-1 team's own route beats 4 of 53.
42/42 was never the target; 20 of its 53 results are ties, because the top of
this ladder is a cluster of near-identical routes that mostly draw with each
other. That is the same fact as the top-20 occupying a 3.9% band, seen from the
inside.

**A team's rank tells you almost nothing about any individual route of theirs.**
Rank comes from their *submitted* agent. A given recorded episode might be a
weak game, the losing seat, or an older version. Same team, two routes:

| route | team | how it was chosen | panel |
|---|---|---|---:|
| `90559891_s1` | THUNDER THUNDER (rank 1) | by rank | **26.4%** |
| `90551213_s1` | Dmitry Larko (rank 7) | by **playing** (`--select`) | **94.4%** |

Selection by playing is worth roughly seventy points of win rate over selection
by rank. Rank is a weak prior for the shortlist and nothing more -- which is
exactly what the tournament already said about recorded bank, now confirmed for
the other obvious proxy.

### v18_route against all 53: 52 of 53

```
OVERALL  310/318 = 97.5%
         beaten outright: 52/53 opponents
         mean margin +7,558
```

Beats rank 1 (67%), 3, 4, 4, 6, 6, **7 (100%, +1,669 -- the mirror is gone)**
and 9 (Chloe). 23 KiB, worst turn 1.2 ms, self-play DONE at 113,392 / 113,392.

**The natural experiment lands exactly as predicted.** Two routes, same team:

| route | chosen by | wins | ties | losses |
|---|---|---:|---:|---:|
| `90559891_s1` | rank | 4 | 20 | 29 |
| `90521287_s0` | **playing** | **52** | 0 | 1 |

Same team, same runtime, same panel, same seeds. Seventy-one points of win rate
decided by nothing but which route was picked. Rank is a shortlist prior;
playing is the selector.

### The one loss is a market squeeze, and it is the next real problem

`Seb (allegedly)`, rank 9: **0% over 6 games, -$42,006**, consistent across both
seats and every seed -- our bank collapses from its usual ~$133k to ~$78k while
theirs holds at ~$112-116k. Sell volumes explain it:

| product | v18 | Seb | combined | units to the $1 floor |
|---|---:|---:|---:|---:|
| STRAWBERRY | 295 | 268 | **563** | ~62 |
| FERTILIZER | 208 | 328 | 536 | -- |
| MILK | 204 | 218 | **422** | ~76 |
| WOOL | 152 | 253 | **405** | ~59 |

Two high-volume sellers meet and the shared market floors. Both routes were fit
against a *typical* opponent; neither is robust to the other. Price-impact
ranking cannot help here -- it permutes our ten slots, and the problem is that
the market is saturated regardless of order.

The fix would have to change *what* or *how much* we sell when the opponent's
standing supply says the market will floor. That is an inventory decision, and
every published ablation on this ladder says inventory changes collapse. So it
is the next real problem rather than an obvious win: the honest framing is that
**97.5% is against a panel where most opponents do not collide this hard**, and
one that does costs $42k.

---

## 2026-08-07 -- forks: what is possible, what is not, and what pays

### A fork of our agent cannot be beaten. That is a theorem, not a gap.

`v18_route` against an exact copy of itself: **tie on every seed**, exactly --
91,075/91,075, 125,093/125,093, 118,735/118,735, 138,247/138,247,
110,284/110,284. Two identical deterministic programs in a symmetric game with
a shared seed have no state either side holds that the other does not.

We built the tie-break anyway to find out where it fails, because "it cannot
work" is worth more when it is measured. `_mirror_tiebreak` detects an exact
public-state match and pulls a slice of the *next* turn's premium SELL forward,
repaying it the turn after so net inventory is unchanged. Instrumented over a
full mirror episode:

```
agent calls                                    719
mirror detected                                719
armed (streak >= 24, step >= 240)              456
  of which no premium sale planned next turn   405
turns where an order was actually created        0
```

Two reasons it never fires, and the second is the interesting one. Most armed
turns have nothing scheduled to borrow. And on the turns that do, the shed is
empty of that product -- **a good route already sells everything on the turn it
becomes available**, so there is no slack to pull forward. The route is
already at the physical frontier of early selling.

Shipped `_MIRROR_TIEBREAK = False` and left the code in place with this note.
Do not rebuild it.

The honest ways to be ahead of a fork are both outside the agent:

* **Publish the write-up, keep improving past it.** A route decays in days --
  we measured a rank-5/6 family drop to 22-44% -- so a fork of today's notebook
  is stale by the weekend. This is a real edge and it needs no trickery.
* **Do not claim bytes you did not submit.** Publishing a notebook while
  submitting a refinement is ordinary practice; publishing one that asserts a
  SHA-256 it is not is not. `tools/build_notebook.py` embeds and verifies the
  real hash, and that should stay true.

### `tools/forks.py` -- the insight channel that does work

There is no telemetry in the agent and there cannot be: a submission runs with
no network, and a callback hidden in a published notebook would be a serious
breach. The channel that *is* available is the one Kaggle gives everybody --
the public episode archive -- and it is enough.

Compare our shipped route against every mined route, per channel, and the turns
that differ are somebody's edit, in the open, with their banks attached. First
run against 505 mined routes at >=75% agreement:

| route | rank | team | overall | field | market | bank |
|---|---:|---|---:|---:|---:|---:|
| `90559730_s1` | 24 | lemon13418 | 78.3% | 69.0% | **87.5%** | 102,493 |
| `90560529_s0` | 24 | lemon13418 | 78.3% | 69.0% | 87.5% | 154,625 |
| `90525850_s0` | 2 | THUNDER THUNDER | 77.9% | 69.0% | 86.8% | 153,388 |
| `90542882_s1` | 1 | THUNDER THUNDER | 75.7% | 65.8% | 85.6% | 114,569 |

`lemon13418` (rank 24) is running a sibling of our donor family and their edit
is legible: 90 market turns changed, concentrated on days 8-25, rebalancing
**WOOL -38/+32, MILK -30/+24, STRAWBERRY -22/+22** with a `BUY_PRODUCT WHEAT`
adjustment -- volume moved between products and days, not created. All three of
their games carry the identical diff, which is itself the tell that they ship a
tape.

The per-channel split is what makes this actionable. 87.5% market agreement
with 69% field agreement says the edit is mostly in the farm plan; a fork that
differed on 4 market turns and none in the field would be a surgical sale-timing
change and deserve far closer reading.

### Why 44 top-100 teams looked unreachable: an absolute floor on a drifting field

The `--top 100` mine ran to exhaustion -- **all 3,010 candidate episodes across
five archived days** -- and still reached only 56 of the current top 100. The
cause was a filter I introduced, not the archive.

`--min-rating` defaulted to the live score at rank `--top`: **2,843**. Applied
to five days of history, that keeps 1,290 of 3,761 indexed episodes -- **34%** --
and the rejection is almost entirely chronological:

| day | episodes clearing the floor |
|---|---:|
| 2026-08-03 | 47 |
| 2026-08-04 | 117 |
| 2026-08-05 | 263 |
| 2026-08-06 | 683 |

The whole field's rating level climbs daily, so a game that was excellent on the
3rd fails a bar set from the 7th's leaderboard. That is calendar drift wearing a
quality filter's clothes, and it systematically hid every team whose archived
games were older than their current rank -- exactly the recently-risen teams
worth studying.

Fixed: each episode now carries its **percentile within its own day**
(`_day_pct`), and `--min-pct` (default 0.35) thresholds on that. Ranking within
a day is stable, so an older game is judged against the field it actually played
in. `--rescan` re-considers episodes already seen, for use after loosening a
filter. `--min-rating` remains available but is no longer the default gate.

Worth stating plainly: the earlier claim that "a handful of teams simply haven't
played a high-`min_score` game in the archived window" was wrong. They had; our
filter discarded them.

---

## 2026-08-07 -- what the ladder says, which is not what the panel said

### `tools/ourgames.py`: 67 real games, 77.6%

Every other measurement here is a proxy. This one is not -- it downloads the
episodes our submitted agent actually played, against the opponents it actually
drew, and deletes the replays as it reads them.

```
67 ladder games: 52W 14L 1T = 77.6% win rate
```

Against a **95.2%** held-out panel. The panel over-predicts by ~18 points, and
it should: its opponents replay fixed trajectories and do not react. Believe the
ladder.

### The losses have one shape, and it is not ours

|  | our sales/game | opponent's sales/game |
|---|---:|---:|
| in wins | 1,275 | 1,811 |
| in losses | **1,261** | **1,854** |

**Our volume barely moves.** An open-loop route sells the same basket whatever
happens -- 1,275 against 1,261 is noise. What varies is the opponent. Per
product, what they sell in our losses against our wins:

| product | opp in wins | opp in losses |
|---|---:|---:|
| STRAWBERRY | 316 | **410** |
| MELON | 144 | **212** |
| MILK | 280 | **320** |
| WOOL | 192 | **228** |
| WHEAT | 633 | 441 |

They shift volume *into* the premium products our revenue depends on and out of
wheat. The shared price curve does the rest. We keep selling into the collapse
because the route cannot see it. That is the whole loss mechanism, and it is the
same one that costs $48k against Seb.

### `_floor_guard`: correct, adaptive, and far too small

The obvious response, and the first adaptive layer with ladder evidence behind
it: before sending a SELL, price the marginal unit; if it would fetch the $1
floor, hold it and release it when the price recovers. Public market inventory
only -- no opponent prediction, no acting on a prediction, both of which have
collapsed everywhere they have been tried.

| seed set | games | win | margin |
|---|---:|---:|---:|
| 60000 | 16 | 100% | +$196 |
| 150000 | 28 | 86% | +$159 |

Consistent sign over 44 paired games and **far below the $3k noise bar**.
Against the two squeeze opponents it recovers +$324 and +$113 -- it does not
fix the squeeze at all.

Why it cannot: holding inventory does not recover the good prices, because the
opponent keeps selling and the price never recovers to release into. The
squeeze is a *rate* problem, not a floor problem. Fixing it needs the route to
sell **different products** when a premium market is being flooded, and that is
a genuine inventory decision of the kind that has collapsed in every published
ablation. It stays the open problem.

### On "win every game"

67 ladder games, 14 losses, to 14 *different* opponents -- no repeat. Combined
with the earlier calibration, where the rank-1 team's own route beat 4 of 53
panel opponents, the position is clear: **no agent on this ladder beats every
other, and none of the top ones does either.** 77.6% against live opposition is
the number to improve, not 100%.

### Knowledge graphs, enforced rather than requested

CLAUDE.md has said since the start that an agent shipped without a current graph
is incomplete. When it was finally checked, **25 of 31 agents had none** -- the
rule was there, the enforcement was not. Added `model_graph.py --missing`, which
rebuilds any graph older than its agent, and wired it into `autopilot.py`'s
BUILD stage so a candidate cannot reach the gate ungraphed. The submission
notebook now regenerates the shipped agent's graph from its own embedded bytes,
so it describes what is in the notebook rather than what was current when the
notebook was written.

### Where the losses actually open, and why the obvious fix is wrong

Re-mined our 67 ladder games with a cash-by-day trace. The deficit opens
**late**: median day 27, range 18-29, with 8 of 12 losses breaking in days
25-29. The mean cash gap tells it more clearly:

| day | gap in wins | gap in losses |
|---:|---:|---:|
| 18 | -2,317 | -3,413 |
| 21 | +9,781 | +1,451 |
| 24 | +13,899 | **+39** |
| 27 | +13,842 | **-2,909** |

Through day 18 the two groups are indistinguishable. We win by pulling ahead
across days 21-24; in the games we lose, that surge simply never happens.

The obvious reading was a timing problem: days 25-29 are the field's
liquidation stampede -- day 29 alone is 11.8% of season volume -- so sell ahead
of the crowd. `_endgame_pull` does exactly that, bounded to stock already in
the shed, in a window where every unit gets liquidated at step 718 anyway.

**It is badly wrong.**

| opponent | win% | margin |
|---|---:|---:|
| `v18_route` (control) | 0% | **-19,054** |
| squeeze opponent A | 17% | -19,014 |
| squeeze opponent B | 0% | -20,033 |

Mean bank falls from ~$131k to ~$112k. Pulling sales forward floods the market
*ourselves*: the route's schedule was already spreading the endgame load to
avoid self-glut, and the accelerator undoes that. Shipped `False`, and this is
the fourth independent confirmation of the same rule -- **change the order,
never the inventory.**

It also forces a re-reading of the diagnosis. In losses we do not *lose* the
endgame, we *fail to gain* it: the gap stays flat rather than reversing. Across
14 losses to 14 *different* opponents with no repeat, the simpler explanation is
that in those games the opponent was stronger, not that we have a fixable
scheduling bug. A late-opening gap is what being second-best looks like when
both sides liquidate at the same time.

Three adaptive layers now built and measured against the same route:

| layer | effect | shipped |
|---|---|---|
| price-impact SELL ranking | +$1,811 over 34 games | **yes** |
| floor guard | +$170 over 44 games (under the noise bar) | no |
| mirror tie-break | arms 456 turns, fires 0 | no |
| endgame accelerator | **-$19,054** | no |

## 2026-08-07 -- archive coverage, measured instead of assumed

"Have we downloaded everything?" deserved a number rather than a claim. The
answer, cross-referencing our mined-episode set against every daily manifest:

| day | episodes | processed | routes kept | median `min_score` |
|---|---:|---:|---:|---:|
| 2026-07-30 | 864 | 0 | 0 | 616 |
| 2026-07-31 | 928 | 0 | 0 | 1,149 |
| 2026-08-01 | 829 | 0 | 0 | 1,327 |
| 2026-08-02 | 793 | 790 (99.6%) | 12 | 2,293 |
| 2026-08-03 | 787 | 787 (100%) | 8 | 2,708 |
| 2026-08-04 | 755 | 755 (100%) | 32 | 2,746 |
| 2026-08-05 | 743 | 742 (99.9%) | 112 | 2,819 |
| 2026-08-06 | 683 | 683 (100%) | **242** | 2,952 |

Two things fall out of this table.

**The archive is 6,382 episodes / 152.6 GB and we have processed 3,757 of
them, but coverage of the days that matter is 100%.** The three untouched days
are not a gap: their median player is rated 616-1,327, so every route in them
would be rejected on arrival. Downloading them costs 52.7 GB to learn nothing.

**The `routes kept` column is the decay curve, measured on our own mine.** The
same filter keeps 8 routes from 2026-08-03 and 242 from 2026-08-06 -- a 30x
difference across three days -- because the whole ladder's rating floor rises
daily. This is why a route is a perishable asset, and it is now visible without
running a single episode.

Corollary for the miner: an absolute rating floor drifts against the calendar
and silently discards older days wholesale. The per-day percentile floor
(`--min-pct`) is not a refinement, it is the only correct form.

## 2026-08-07 -- a crash that looked like an empty result set

`tools/ourgames.py` stopped at 20 of 116 episodes for one submission and
reported no error. The cause was `UnicodeEncodeError`: team names are arbitrary
Unicode, a Windows console is cp1252, and the progress `print` sat *inside* the
download loop. One opponent with a non-Latin-1 name killed the whole mine.

The failure mode is what makes this worth writing down. It did not look like a
crash -- it looked like "that submission has only 20 episodes", which is a
perfectly plausible number. The measurement it corrupted (a per-model win rate)
would have been read as fact. Fixed with an `_ascii()` display guard on every
path that prints a team name.

Same family as the engine-fingerprint problem: nothing raises, the tool just
answers a different question confidently.

## 2026-08-07 -- the ladder is 11 route families, and two dead ends off it

Clustering all 525 mined routes by action signature (cosine >= 0.995,
single-linkage) collapses 89 teams into **11 families**:

| routes | teams | rank range | note |
|---:|---|---:|---|
| 264 | **47 teams** | 3-99 | the crowd |
| 97 | 22 teams | 11-194 | |
| 70 | 20 teams | 21-190 | |
| 28 | 7 teams | 33-88 | |
| 15 | **SOLO** | **1-10** | Seb (allegedly) |
| 12 | 2 teams | 1-24 | THUNDER THUNDER + lemon13418 (**ours**) |
| 3 | SOLO | 26 | David Schindler15 |

Half the visible ladder runs one lineage. The three opponents v18 could not
beat map exactly onto this: **Seb** and **David Schindler15** each run a solo
family, and **lemon13418** shares *our* family -- which is why that loss is
only -$961, a near-mirror.

### Dead end 1: build from the rank-1 outlier

We held 15 Seb routes, 6 in the `fit` window, and had never built one. The
archive said Seb beat our lineage **4/4** head to head (no seat bias in the
mine: 47.1% vs 48.4%). Built `v19_seb_route` from `90285497_s0`.

Head to head it beat v18 **12-0, +$30,503**. On the held-out panel it lost:

| | v19_seb | v18 |
|---|---:|---:|
| mean win% over 51 common opponents | 72.2% | **93.5%** |
| mean bank | $113,726 | **$133,025** |
| better / worse / equal | **3 / 45 / 3** | |

It genuinely fixes both unbeatable opponents (Seb 0% -> 50%, margin -48,160 ->
+10,868; David Schindler15 0% -> 67%) and regresses against 45 others.

**Seb's route is a counter to our lineage, not a better route.** v18 is adapted
to the crowd, and matchmaking means you meet the crowd. Trading 45 opponents
for 2 is a bad trade even when the 2 include the one that humiliates you.

**A 12-0 head-to-head against the incumbent is not evidence of superiority.**
Only the fixed-roster panel is. Non-transitivity is real here and it is strong.

### Dead end 2: a per-turn consensus route

"Take the modal action at each of the 720 turns across all top routes" -- voted
over all **198 fit-window routes**, rendered on the same runtime, played:

**$12,085 against v18's $179,117. 0% of 8 games, -$167,032.** The worst agent
this project has produced, and it lands on the same floor as the channel splice
($12,900), by the same mechanism.

Why, measured:

| | |
|---|---:|
| modal whole-turn action share | mean **35.5%**, median 31.3% |
| turns with a true majority (>50%) | **105 / 720** |
| unit actions that are position-dependent | **87.8%** |
| unit actions that are pure movement | 48.4% |

At two turns in three there is no consensus to find, so the winner is a
plurality of about a third and you are splicing at the finest possible
granularity. And when two routes both say `WEST` at turn 400 they are not
agreeing -- different farmers, different tiles, different destinations. The
label matches; the plan does not.

**A route is a plan carrying state, not a sequence of independent decisions.**
ML can choose between plans. It cannot average them.

## 2026-08-07 -- win rate is not the objective, and we had it backwards

Mining our own games for *every* submission rather than only the live one
(`tools/ourgames.py --all`) produced 284 games and a result that reverses a
standing assumption:

| submission | public | games | W-L-T | win% | median opponent score |
|---|---:|---:|---|---:|---:|
| 55299462 (copied v21.1) | **2459.2** | 117 | 43-55-19 | **36.8%** | **2,557** |
| 55314513 (our v18) | 2342.9 | 70 | 53-16-1 | **75.7%** | **2,275** |
| 55296785 (v14) | 847.8 | 39 | 19-20-0 | 48.7% | 954 |
| 55288369 (v9) | 820.9 | 24 | 13-11-0 | 54.2% | 815 |
| 55269409 (v4) | 668.4 | 34 | 16-18-0 | 47.1% | 762 |

**Our agent wins 2.1x more often than the copy and is rated 116 points lower.**
The last column is why: the copy is matched against opponents with a median
rating of 2,557, ours against 2,275. A 282-point difference in the
neighbourhood, and the ladder pays for beating strong opponents, not for
winning often.

The tie column is the second tell. The copy draws **19** games; v18 draws 1.
A public notebook route is shipped by many people at once, and two identical
deterministic programs in a symmetric game tie every time. The copy is
partly playing itself.

This makes `CLAUDE.md`'s rule -- *"the ladder scores wins, not coin margin;
rank and select on win rate"* -- correct against coin margin but **incomplete**.
An unweighted win rate is measured against whatever opponents matchmaking
happened to supply, and matchmaking supplies opponents near your current
rating. So win rate is partly a readout of the rating you already have, which
makes it circular as a selection key. 75.7% against 2,275 is a *weaker* result
than 36.8% against 2,557.

Two consequences worth holding on to:

1. **v18's 75.7% will fall as its rating climbs**, without the agent changing
   at all. Do not read that fall as a regression.
2. **Selection needs opponent strength in it.** The held-out panel already has
   this property -- it plays a fixed roster of top-100 tapes regardless of our
   rating -- which makes the panel the better *comparator* even though section
   3b showed it a poor *predictor*. Those are different jobs and it is good at
   one of them.

## 2026-08-08 -- the base route was not the best one in its own family

The 2026-08-07 archive day had still not been published as of the 04:00 fetch
(the mine indexed 08-04..08-06 and got 0 new routes), so freshness could not be
improved today. The budget went to two questions instead: is v18's route the
best base in our own family, and can the squeeze be answered inside the market
channel.

### Family tournament: play the siblings, same seeds, same roster

Five fit-window candidates on an 11-opponent mini-panel (the three opponents
v18 cannot beat -- Seb `r001`, lemon13418 `r024`, David Schindler15 `r026` --
plus the TT-sibling tape that beats v18, plus seven crowd tapes), 2 seeds x 2
seats, then the whole thing again on an independent seed set:

| base route | recorded bank | set 60000 | set 150000 |
|---|---:|---|---|
| `90521287_s0` (v18) | $139,451 | 8/11 at 100% | crowd 100%, lemon 0% |
| **`90525850_s0`** | $153,388 | 8/11 at 100%, lemon 100% | **10/11 at >=75%, lemon 100%** |
| `90559730_s1` (lemon fit) | $102,493 | crowd 100%, ties own tape $0 | crowd 100%, ties own tape $0 |
| `90518963_s1` | $123,711 | **9/11 sweep, +$9.7k margins** | **collapse: 0% vs the crowd** |
| `90196844_s1` (Kaito) | $162,617 | crowd 0% | -- |

Two lessons, both re-confirmations. `90518963_s1` is the third-seed-set rule
in action: a 9/11 sweep on one seed set, a wholesale collapse on the next.
And `90196844_s1` is "bank is a bad selector" again -- the highest recorded
bank in the candidate set, 0% against the crowd.

**`90525850_s0` is the aggregate winner**: against v18's base it is equal or
better on 10 of 11 opponents across both seed sets -- it beats the lemon tape
in all four measurements (v18: one win, one loss on margin -$1,262), does
better against Schindler (0%/75% vs 0%/25%) -- and slightly worse only against
Seb. Crowd margins run ~$1-1.5k higher than v18's throughout. lemon's own fit
route ties their tape at exactly $0 -- same program, perfect mirror -- which
caps it at a draw and rules it out as a base.

### The glut guard: the fifth confirmation, and the cheapest one

The floor guard's trigger sits at the $1 floor, and a strawberry quoting $20
off a $120 base is a squeeze the guard never sees. So: `_GUARD_RATIO` (hold
the part of a SELL whose marginal unit quotes below ratio x base) and
`_SWAP_ADVANCE` (while holding, pull forward an already-scheduled SELL of a
healthy product, value-capped by the running deferred-minus-advanced balance).
Both in `tape_runtime.py` behind flags, byte-neutral when off (default build
ties v18 exactly, $111,796/$111,796).

Against a byte-identical control on the new base, 3 seeds x 2 seats:

| variant | vs control | vs Seb | vs Schindler | vs crowd |
|---|---:|---:|---:|---:|
| ratio 0.25 | **0%, -$13,403** | 33%, -$4,331 | 0%, -$11,673 | 0%, -$2,916 |
| ratio 0.25 + swap | **0%, -$19,094** | 33%, -$13,023 | 0%, -$24,233 | 0%, -$6,331 |
| ratio 0.15 + swap | **0%, -$10,376** | 33%, -$5,672 | 0%, -$18,016 | 67%, -$2,976 |

The mechanism is wrong, not the knob. A ratio trigger cannot tell an
opponent's squeeze from the route's *own* scheduled price impact -- the
schedule deliberately sells into sub-ratio marginal quotes on its big days --
so the guard holds back the route's planned volume and shifts season revenue
into the terminal dump. The swap-advance then self-floods the healthy markets,
which is the endgame accelerator's failure repeated at smaller scale. The
gradient across 0.25 -> 0.15 -> 0 points straight back to the pure floor
trigger (+$170, under noise). Shipped off; the code stays as the measured
record. **Change the order, never the inventory -- fifth independent
confirmation, and this one cost an afternoon rather than a submission.**

The per-turn portfolio idea ("simulate the top 100 and pick the best action
each turn") was not re-tested: the consensus-route experiment already measured
it at $12,085 / -$167,032, and the channel splice at $12,900. A route is a
plan carrying state. Selection between whole plans, judged by playing -- which
is what the family tournament above is -- remains the only form of "choose
from the top 100" that survives measurement.

## 2026-08-08 (later) -- v19 replayed against every opponent v18 actually met

Question asked directly: would v19 have done better in v18's real ladder games?
Refreshed the mine (v18 now: **107 real games, 70W-36L-1T = 65.4%**, down from
75.7% as matchmaking feeds it stronger opponents -- the predicted fall, not a
regression). Built one tape per unique opponent (96 of 96, preferring the
episode v18 LOST, so the rerun is biased toward the hard cases), then played
v18 and v19 against all 96 on identical seeds, both seats -- 768 games, fully
paired.

| | v18 | v19 |
|---|---:|---:|
| paired record | 292W-92L (76.0%) | **304W-80L (79.2%)** |
| opponents better against | **0** | **95 of 96** (1 within noise) |
| strict win-count flips | -- | 6 (incl. two 0/4 -> 2/4, three 2/4 -> 4/4) |
| win-count regressions | -- | **0** |

v19's margin improves against every single opponent, including all 19 tapes
neither agent beats (Winter Lamb: -43,321 -> **-8,165**). The 19 unbeaten
tapes lose by small margins (mostly -$200..-$4k) and cluster in the strong
route families; per the standing result no route beats every other, and the
v19_seb dead end already showed per-opponent counter-building trades the crowd
for the corner. The fix for the residue is the standing loop -- fresher mine,
family tournament, panel -- not per-opponent patching.

Caveat kept honest: tapes replay each opponent's recorded game *against v18*
and do not react; treat the paired delta (+3.2 points, universal margin gain)
as the finding, not the absolute win rates.

## 2026-08-08 (evening) -- the per-turn idea rebuilt from scratch, and measured

Two agents built today at the user's direction, in fresh code, deliberately
NOT reusing any prior experiment's implementation.

### v20_pool -- per-turn selection over the top-100's sell-moves

`tools/pool_agent.py`. Field channel and every BUY/HIRE from the base route
(90525850_s0, v19's); the SELL channel chosen per turn by argmax live-market
value over 2,721 recorded top-100 sell-moves (483 turns), feasibility-clamped
to the shed and bounded by the pool's 75th-percentile cumulative envelope.
Base route wins ties and ignores the envelope, so the floor is v19.

* First build let the pool borrow SELL WHEAT / SELL FERTILIZER: **$33,023 vs
  $171,741** -- it sold the herd's feed. The per-turn failure mode reproduced
  in new code, cause identified in one match.
* Guarded build: **bit-identical to v19 across all 80 paired games** (10
  opponents x 2 seeds x 2 seats x 2 seed sets). The selector never once found
  a feasible sell-set worth more than the base route's own. A frontier route
  sells everything the turn it reaches the shed; there is nothing better to
  choose *from the same turn of other plans*.

### v21_counter -- identify the opponent's program, forecast, counter

`tools/counter_agent.py`. Opponent cumulative sales reconstructed exactly from
the shared inventory (town consumption transcribed from the interpreter and
subtracted; only $1-floor sales are invisible, per A4), matched against 32
distinct programs clustered from 90 fit-window routes, with a cross-team
runner-up gate (recordings of the same opponent must not veto each other).
When decisive: forecast their next sells; advance our own scheduled stock
(borrow/repay) and put our sell of a threatened product first in the queue on
the exact turn their dump is forecast (pure permutation).

* **Identification works.** It recognizes Seb's program from seat play alone,
  and produced two genuine intel finds: ladder team "Winter Lamb" runs Seb's
  program, and "chocolat" runs David Schindler15's. The clone economy is
  directly observable from the market trail mid-game.
* **The counter is worth ~$0.** Borrow-forward fired zero times -- the shed is
  empty between harvests, the same physics that muted the mirror tie-break.
  The threat-first permutation fired 6-26 times a game and moved margins by
  +$6..+$138 against squeeze opponents and -$3..-$159 elsewhere; no game's
  outcome changed across 80 paired games (mean bank equal to the dollar).

### The session's net answer

"Choose the best of the top-100's moves each turn" -- built safely, it
converges to the best whole route (v19); built unsafely, it liquidates its own
working capital. "Predict the opponent's next action and counter it" -- the
prediction is achievable and cheap; the counter has no profitable action to
take because order-book position is already maximised and inventory changes
are the measured-fatal category. Both conclusions now rest on today's code,
not on prior experiments. v19 remains the ship candidate, unchanged.

The one durable asset from v21 is the identifier itself: opponent-program
recognition from public state could label our ladder losses by family in
`ourgames.py` (which opponents run which program, measured live), guiding the
next mine. Filed as an analysis tool idea, not an agent layer.

## 2026-08-09 -- v22: the first adaptive layer that survived validation

Directed build: use all our gameplay data to fix v19, per-turn optimization
over the top-100 space, in-game RL. What shipped is what survived four rounds
of paired measurement; the cuts are the story.

**Diagnosis** (all 293 live games, losses labeled by nearest program):
v19's losses are 62% our own lineage's fresher editions (23/37, -$80k), then
tao wu11 and Raj Aryan. The 18 opponents that beat v19/twin live were taped
and became the training targets -- our own losses as the objective.

**Training** (tools/train_arms.py): 28-dim sell-retiming genomes from turn
192, (1+lambda) hill-climb, full-episode fitness. Two hard lessons paid for
in the first attempt: recombining elites on a hostile landscape walks the
mean into a cliff (gen-1 best -$384k), and overflowing a turn's 10-order
market cap silently drops the BUYs (sells sort first) at -$1.4M/candidate.
Slot-aware placement + monotone hill-climb fixed both.

**Validation cuts** (416 paired games x 4 rounds): tt arm (+97k in training)
does not generalize -- half the ladder looks TT-ish mid-game and false
commits cost more than true ones earn; tao arm flat; threat-first ordering
was the hidden regression (-$1.8k vs out-of-library programs -- it reorders
on ANY match, decisive or not); feed-pull byte-identical null. All four are
in the code, off.

**What ships**: identifier (exact sales reconstruction, 36-program library,
cross-team gate) + bandit (2 decisive reads to commit, 24-turn
non-confirmation to abandon) + raj/seb arms. Certification: 124/208 vs
v19's 123/208, margin +6,739 vs +6,636, ZERO regressions; Schindler
3/8->4/8, Seb-panel margin -225->+79, fmind +630, crowd +140..+265.

The honest summary: v22 is v19 plus small measured gains plus the adaptive
machinery, with every non-generalizing layer amputated. The per-turn and
in-game-RL forms of adaptivity keep converging to the same result: the value
lives in WHICH schedule you play (freshness) and a small, well-guarded
opponent-conditioned switch -- not in turn-level improvisation.

## 2026-08-09 (night) -- the freshness wall, measured exactly

Directed: rebase v22 on the latest data to fix its live losses. Tournament of
ALL 26 buildable candidates (every 08-07 fit route + every fit route of a
team that has beaten us live), refereed by tapes of v22's own losses from the
last 12 hours plus a crowd guard.

**Result: nobody wins.** Best candidate (kevin park r6): 6/16 vs the referee
roster, 0% against both fresh THUNDER-edition tapes. v19's base: 0% against
all three loss referees. The teams beating v22 under new names (Epiphany_Thu,
Saisree) run THUNDER-family editions played on 08-08/09 -- and the public
archive ends at 08-07. The ladder's frontier leads the archive by the
publication lag, permanently. No base available today fixes yesterday's
losses; shipping the least-bad candidate would regress the crowd and
own-family columns.

Decision: hold v22+v19 active; monitor armed for the 08-08 archive day
(04:00 fetch), which contains the exact episodes of the current frontier
editions -- then re-run this same referee-scored tournament and ship v22.1
on the first base that beats the loss tapes. The daily rhythm IS the
strategy; nothing else measured survives contact with the lag.

## 2026-08-10 -- v19.1 + v22.1: the fresh-base double ship

The 08-08 AND 08-09 archive days landed within hours of each other (the
08-08 dataset had been up since 00:09 on the 9th; our 04:00 fetch missed it
on a stale index -- worth a look at the index-refresh path). Mined both
(85 + 14 routes; two operator-error restarts: PYTHONUTF8 required for
worker file reads, --clean-stage is exclusive not a modifier).

**Tournament, refereed by tapes of OUR OWN live losses** (7 loss tapes + 5
hard-panel tapes, 2 seed sets, both seats, 96 games each):

| base | day | overall | vs v22's killers |
|---|---|---:|---:|
| v19 (08-06) | -- | 45/96 (47%) | 29% |
| best 08-08 (M. Labrador r19) | 08-08 | 83/96 (86%) | 86% |
| **91468035_s1 (Valmorlee r5)** | **08-09** | **96/96 (100%), +285,924** | **100%** |

One day of freshness: 47% -> 86% -> 100% on the same games. The freshness
decay curve, measured at daily resolution against our own failures.

**Shipped** (both from the same notebook, versions 7 and 8):
* `v19_1_route.py` -- ref 55396717, sha 1d15176475110f21. Validation passed.
* `v22_1_bandit.py` -- ref 55397322, sha 57b5b9ddd21d0804. Same base +
  identifier (43 programs) + seb2/frontier arms retrained on this base
  against the loss tapes ((1+lambda) ES, both positive on top of a base
  that already sweeps: +40,871 and +36,648 training gain). Paired
  validation: both 96/96; v22.1 +$12,478 total margin (Seb +3,509, lemon
  +4,274, frontier +4,790), zero regressions, NOVEL referees byte-equal --
  the lookalike lesson enforced.

Active pair: **v22.1 + v19.1**. Cross-checked the field discussion thread
(732450): every discrepancy it lists was already engine-derived here; the
one new fact is the hand-spawn bugfix implying the live engine is 1.32.5/6
vs our vendored 1.32.4 -- filed: re-vendor + refresh engine_check baseline.

## 2026-08-10 (later) -- the commit audit, and an attribution corrected

v22.1's first 39 real games audited replay-by-replay for arm commits:
no-commit 36 games 89% (+17,277); seb2 committed 3 games 3/3 (+14,292);
frontier never fired. Meanwhile v19.1 (same base, no adaptive layer):
**45W-4L = 92%**. The twins are statistically identical live; the 1,100-point
rating gap during convergence was matchmaking path, not the bandit -- an
attribution error made in the heat of the climb and corrected here.

What the double ship actually proves: **loss-refereed fresh-base selection
is the engine.** Both twins run ~91% against live opposition on a base
chosen by beating tapes of our own losses from the newest archive day. The
adaptive layer is live-measured as safe and mildly positive (8% fire rate,
3/3), worth keeping, not worth crediting for the surge.

Priority reorder: (1) automate mine -> loss-tapes -> tournament -> arms ->
build -> gate as the daily cycle; (2) arms only where the audit shows fire
opportunities; (3) both slots on the freshest base every refresh.

## 2026-08-10 -- discussion sweep (732450, 731635, 733383, 733392, 733173)

* 731635: the hand-spawn-on-locked-tiles fix was "merged, rolling out" as of
  ~08-03 -- second confirmation the live engine moved past our vendored
  1.32.4. Re-vendor urgency raised. Pre-fix routes under-use the trapped SE
  hand; post-fix routes (our fresh 08-09 base) already internalize the fix.
* 733383: field consensus lands where our measurements did -- pure RL (PPO/
  SAC) thrashes on delayed rewards + zero-tolerance mechanics; deterministic
  backbone + rule-based market logic wins. Independent validation, no action.
* 733392: competitors are building fast simulators; one (rank 483) claims a
  CUDA env at 50k steps/s x 1024 envs. The brute-force trajectory-search arms
  race is starting. Action: evaluate tools/fastsim.py for the training loops.
* 733173: wenyuan Guo (rank 110, on our panel) publicly describes their
  target design: "static production backbone with selective hysteresis-based
  adaptation" -- the meta is converging on backbone+adaptive architectures.
  Raises the value of the signature-obfuscation defense; also means fresh
  routes will increasingly embed adaptive edits, which the clustering's
  cross-episode-consistency feature will expose.

## 2026-08-10 (night) -- Phase 0+1 delivered; the hold that proved the doctrine

Balance change absorbed end to end: engine re-vendored to 1.32.6 (baseline
refreshed; the fingerprint did NOT change -- the rebalance altered logic and
config defaults, not the 121 fingerprinted constants, which is why
engine_check could not see it; version-drift alarm from archive metadata is
the fix). Consumption model corrected in the agent template (centre 1x/day
flat, duplicate-aware shops). All pre-1.32.6 local numbers voided.

Re-crown under the new engine with referees cut from our newest losses:
**the incumbent base held** (61% vs every challenger <=47%) -- the crown
gate's first live save. Arms retrained under 1.32.6 with novel-guards in the
objective: +32k each on training seeds, ~0 out-of-sample, and frontier3
REGRESSED -$5.4k on its own training tape in paired validation. Sixth
confirmation: schedule-retiming arms do not generalize off their seeds.
**v23_bandit built and gated but NOT submitted** -- shipping it would evict
v19.1 (93% live, 62W-5L, still converging), the exact trade the
champion+challenger doctrine forbids. First HOLD, by the book.

Infrastructure landed today: tools/refresh_cycle.py (daily orchestrator,
scheduled 05:00 as KaggricultureRefreshCycle), tools/sink.py (every local
episode -> training row, engine-tagged), tools/features.py (283-dim
behavioral signature), families.py --broad (883 routes -> 7 mega-families,
all tape-like at consistency <=0.004), commit_audit.py, public EDA notebook
(kaggriculture-behavioral-families-eda, sanitized: no strategy exposure),
docs/architecture.md + submit strategy in docs/history/submission.md.

## 2026-08-10 (close) -- everything implementable today, implemented

* refresh_cycle upgraded: arms + bandit as gated stages, broad-signature
  candidate clustering, engine version-drift alarm (archive metadata vs
  vendored logic marker). Scheduled daily 05:00.
* Identifier v2 is a trained MODEL: multinomial logistic over behavioral
  clusters, prefix rows manufactured from 901 routes with dropout
  augmentation for floored-sale blindness. 94.3% on the held-out newest
  day; stdlib export equivalence 2.1e-07. Weights in
  .local/v22/identifier/; integration lands with the next bandit build.
* Docs: environment-rules updated for the 1.32.6 rebalance (stale sections
  flagged); generation doc for v19.1/v22.1/v23; architecture + submit
  strategy already in.
* Public EDA notebook v2 re-pushed with robust dataset discovery (v1
  errored on mount paths, 0 episodes found).
* Remaining, deliberately data- or session-gated: decision-theoretic commit
  gates (audit rows accumulating), surrogate-assisted search (sink
  filling), Rust step engine + trajectory optimization (multi-day block),
  signature obfuscation (timing-jitter class -- needs fresh-session A/B
  discipline, six overfit/regression confirmations say do not rush it).

## 2026-08-11 (early) -- the identifier as a model, and its granularity law

Integrated the trained identifier into the bandit template (route library
dropped from the shipped file entirely; -28KB). Three findings paid for:

1. **Observable-features-only**: the first training used buy totals, which
   an in-game observer cannot reconstruct (opponent buys conflate with
   consumption in the public inventory). Retrained on sell curves +
   endgame + t only: 94.5% held-out-day.
2. **The granularity law**: fine clusters (link 0.025, 18 classes)
   discriminate the loss-family teams but COLLAPSE across days (held-out
   30.9% -- program drift exceeds fine-cluster tolerance). Coarse clusters
   (0.055, 7 classes) are day-stable at 94.5% but merge most top teams
   into one mega-class. Shipped coarse; per-team arms need either
   day-adaptive labels or the finer model retrained daily -- filed with
   the data-gated work.
3. **Own-class exclusion**: our base's class may never map to an arm --
   the unguarded build committed an arm against its own twin (-$683).
   Guard added at build time; smoke now near-ties the twin correctly.

The 05:00 cycle now runs the complete modern stack unattended: model
identifier, own-class guard, arms trained with novel-guards in-objective,
paired bandit gate that refuses regressions. Tomorrow's crown is the
submit; tonight's hold stands.

## 2026-08-11 -- "build everything, let the data catch up": done

* tools/train_gates.py -- decision-theoretic commit gate estimator;
  self-gating (needs >=25 judged commits; currently 0; refuses honestly).
* tools/train_surrogate.py -- search surrogate on the episode sink;
  self-gating (397/2000 current-engine rows and filling daily).
* Both wired into refresh_cycle's daily model-refresh block alongside the
  identifier retrain.
* rust/step_engine -- the fast-engine port begun with the market kernel:
  quote() with banker's-rounding mirror + paired sell resolution (A4 floor
  rule). **Parity level 1 GREEN: 270,009 price points, zero mismatches**;
  5x through unbatched FFI (PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 needed on
  Python 3.14). Field-phase port = mechanical continuation of a green
  pattern; parity harness in tools/parity_rust.py is the standing gate.
* tools/self_identify.py -- the obfuscation acceptance gate, with the
  baseline measured: EVERY submission we have fielded classifies to one
  behavioral class at 100% consistency, 0.79 confidence. We are fully
  identifiable; obfuscation has its target number (chance) and its cost
  gate (zero, paired).

## 2026-08-12/13 -- v24 era (condensed; full detail in docs/history/issues-and-improvements.md)

* v24.0 and v24.1 pairs shipped (v24.1_bandit peaked 2380.1, then-best).
* Pipeline hardened: 04:30 release + hourly scrape; kaggleusercontent fast
  fetch; playoff crown gate on WINS (margin criterion removed everywhere);
  win_metric.py became the only currency; ARM GUARD rebuilt sign-tested
  after the volume-only gate shipped arms and cost -2,971/game.
* GC latency bug found: gc.freeze() + thresholds in every generated agent
  (245-330 ms collector pauses -> 27 ms worst).

## 2026-08-14 -- the big day: backfills, full Rust engine, CROWN-2, v25 pair, publishing

* HISTORICAL BACKFILL (operator order): 2,952 episodes traced on-disk;
  staged-replay backlog ingested (+5,768 routes); enrichment re-fetch of
  6,087 old episodes. Trace corpus 22 -> 12,427 episodes; policy corpus
  31,680 -> 4,240,800 transitions.
* RUST ENGINE COMPLETE: full step function ported; 50/50 differential
  episodes bit-identical (chaos + real replays, full state per step);
  103,260 steps/s = 143.6 ep/s one core (~400x). rust_prerank bridge;
  daily parity self-check that revokes the funnel on mismatch.
* HOURLY FETCH FAMINE root-caused: SDK naive-UTC datetimes read as IST aged
  episodes 5.5 h; every hourly sip since 08-12 returned 0 ids silently.
  Fixed + regression suite; budgets raised to safety-valve levels.
* CROWN-2 executed with kill-switch discipline: Rust funnel ADOPTED
  (200 -> top-24 tickets, 4-seed finals, trailing recall self-audit);
  overlap exclusion replaced strict-future; rating-band panel REJECTED by
  its pre-registered backtest (wins ordering 0.396 vs 0.128, loses MAE);
  portfolio overlays REJECTED (0/32 raw; retimed worse -- third death of
  cross-farm schedule transfer); per-turn BC policy head judged 0/52 -->
  POLICY GUARD holds a definitive NO.
* E5: GRU won held-out 0.950 vs logistic 0.910 (43 classes) -> embedded in
  v25.1 via --gru-auto; paired panel shows no win conversion yet (its
  consumer, arms, stays gated).
* FIELD STUDY (22,575 plays): prices are a shared tide (r~0 with WINNING);
  the GAP decides (lead share r=0.69); elite signature = faster first-10k,
  leaner crew, cleaner watering, deeper early investment, mid-game surge.
  All future rewards gap-shaped.
* SHIPPED: v25.0 pair (route hit 2406 -- project best), then v25.1_bandit
  (first GRU deployment); live pair = v25.0_route + v25.1_bandit.
  UNATTENDED PUBLISHING ENABLED in the 04:30 release (operator order) with
  gate/already-live/quota/sha rails.
* Stream-hash opening exposure shipped into every release build: our base
  shares its day-3 opening with 607 routes / 18 teams; unique by day 10.
* TRACK P plan published (closed-loop planner lineage, v5): three-layer
  agent, PPO+PFSP league, IQL warm start, ExIt upgrade, model zoo with
  per-model metrics/gates, go/no-go calendar. Ship strategy: {bandit,route}
  now -> {bandit, closed-loop} on graduation.
* Generation-scoped guards: rehabilitated arms/policy generations can open
  gates on their own positive evidence; historical negatives are history,
  not a veto. Midday rehab job built then DELETED same evening by operator
  decision (sell-side ceiling measured near zero); machinery parked as
  Track P item P6.
* Memory relocated to .local/memory/ (operator order); ~/.claude project
  folder holds only a redirect stub.

## 2026-08-15 -- Track P implemented end to end (isolated planner lineage)

Operator order: implement ALL of Track P -- pipeline, agent, everything --
sharing ZERO Python code with bandit/route. Delivered as `src/trackp/`
(21 modules), `rustengine` service modes, `agents/planner_v0.py`, a daily
scheduled task, and a 40-case test suite. All measured on day one:

* Rust service modes: `kagg batch` (tape pairs -> banks, all cores) and
  `kagg serve` (stdio env, official obs schema). Verified: 6/6 sampled
  1.32.6 ladder replays reproduce recorded final banks EXACTLY through
  batch; serve == batch bit-for-bit; 8.5 ep/s aggregate with the Python
  planner in the loop (777 steps/s single stream).
* Found while verifying: pre-rebalance replays ((data/mine, module 1.32.4,
  townCenterSellInterval 12 + demand schedule) do NOT reproduce on the
  current engine -- the divergence that looked like an engine bug was the
  Aug-7 rebalance. Current ladder config (1.32.6, center interval 24)
  confirmed from staged replays; the vendored engine is the ladder's.
* Trace v2: 208 dims/turn (shop draw, both farms, both privates, full
  action summaries). Recaptured 5,580 episodes (5,570 on 1.32.6) from the
  staged-replay cache before ingest deletes them; 45 KB/episode.
* Insight run (11,140 plays, four passes): NEW #1 elite marker =
  demand_match_mean, Cohen's d +1.49 -- production-mix alignment with the
  live shop draw beats every gap marker. First direct field validation of
  shop-conditional planning (P1.2). Interaction pass: 0 FDR survivors yet.
* Families: 651 opening-hash families cover 93.4% of seats (>=5 members);
  persistence gate 1.0 (anchored hashing is stable by construction).
* L2 projector PASSES: 7/9 products beat the current-price naive at
  2/4/6-day horizons (10,920 predictions/product).
* Planner v0 (fresh 3-layer single file, imports math only, 1 ms/turn):
  CMA-ES (own implementation, 24 params, 60 gens on serve) improved league
  fitness -0.78 -> -0.679; the searched build carries to the official
  engine ($37k -> $53k vs pass, $19-27k -> $45k vs v25 route).
* IQL warm start: 0.878 mean held-out decision accuracy on 68,970 elite
  macro decisions. PPO+PFSP league (anchors/selves/current/exploiters,
  annealed gap reward, anchor-only checkpoint selection) runs 16 iters
  clean; neural L1 ties the hand rules -- L0 executor throughput is the
  binding constraint (route incumbents bank 2-3x more). Exploiter round:
  0% edge vs frozen best at this budget. Export equivalence 5.7e-6.
* P4.1 validity: Spearman 0.347 open-loop vs closed-loop -- WEAK, recorded;
  official confirmation stays mandatory everywhere.
* P2 generator (return+shop-conditioned GRU decoder, elite fine-tune 0.48
  loss) + funnel: 24 plans -> 3 jointly-novel (hash AND sell-curve) -> 1
  officially confirmed (not competitive yet, honestly logged).
* Gates: PLANNER GUARD (>=25 judged, sign test, generation-scoped) at 12/25
  gathering; sim-to-real drop 0.0 (PASS); P1.6 panel 16 games across 4
  shop-draw regimes (0/16 -- the honest day-one verdict); graduation 1-4 =
  false, condition 5 (operator) never automated.
* Ops: `KaggricultureTrackP` daily 13:30 (3h limit), off the 04:30 release
  path; hourly-ingest concurrency made crash-proof in discover().
* Tests: tests/test_trackp.py -- 40/40 incl. an ISOLATION test (no
  bandit/route imports, suite-pinned) and serve==batch parity; full fast
  run_all green (obs-schema linter auto-covers the new agent; one wrapped-
  call false positive restructured away).

## 2026-08-15 (day) -- v26 pair shipped through a quota-exhaustion gauntlet

The 04:30 release HELD: every loss-tape fetch 429'd (overnight backlog drain
exhausted the ~24h replay quota; throttle lasted 6+ hours on both fetch
paths). Manual re-run at 10:15 with the hourly job paused: the new stale-tape
fallback reused yesterday's 12 tapes, crown said HOLD (challenger 6% vs
incumbent 62%) -> pair rebuilt on base 92513718_s1 with the day's retrained
models -- and then the outer 4h cap killed the run after the build, before
publish. The new --resume-publish flag finished the job: both agents gated
(route 4.2 ms / bandit 21.6 ms worst), kernels pushed (bandit v9, route
v16), both submitted 14:27 IST sha-verified. Five hardenings shipped (cache-
first tape fetch, backoff ladder, global-quota short-circuit, stale-tape
fallback, resume-publish + 5h cap). Track P's first daily run executed
manually after (schedule restored). Full incident detail in
docs/history/issues-and-improvements.md.

## 2026-08-15 (night) -- 1.32.7 hinge rebalance: staged, verified, armed

Kaggle announced (discussion 735311 / PR #1399): CARROT/TOMATO/EGG scarcity
pricing becomes a hinge (quadratic runaway past T). Ladder still on 1.32.6,
so the whole migration is STAGED behind switches and armed on a watcher:

* 1.32.7 wheel staged + exact-diffed; Rust hinge behind ENGINE_1327 const
  (verified EXACT vs the 1.32.7 interpreter over 137 inventories x 9
  products); trackp/common + planner template + counter_agent + leak_check
  all hinge-capable behind models/engine_version.json / build-time flags.
* Parity harness now filters real replays by the ACTIVE engine version --
  the false-alarm that would have closed the funnel at every rebalance is
  designed out.
* scripts/engine_swap_1327.py --check rides the hourly scrape: on the first
  1.32.7 replay it swaps vendor, flips the flags, rebuilds kagg, rebuilds
  planner_v0, re-baselines engine_check, revokes pre-hinge funnel evidence,
  and runs parity -- atomically, logged to data/logs/engine_swap.log.
* Strategy note: the hinge is a direct payout multiplier on the demand-match
  edge (our #1 elite marker, d=+1.49). The planner's L2 prices the spike
  automatically once flipped: produce carrot/tomato/egg into drained
  markets nobody else supplies. Test pins: 47/47.

## 2026-08-16 (night) -- v27 go-live plan: second-slot rule, intraday gate, adaptive sell-timing

Diagnosis that drove it: the "bandit decline" (2339 -> 2099 -> 2041 -> 1979)
is (a) window artifacts -- v25.0_bandit was retired at FOUR HOURS by the
untested v25.1; ratings converge with games -- and (b) real: since v25 every
adaptive layer is guard-locked (arms audited NEGATIVE-VALUE, 47-class GRU has
no consumer but 3 relay schedules), so the bandit is the route + 420 KB of
dead weight and cannot out-rate its twin. Win%-up-rating-down is matchmaking
(house rule 6), not improvement.

Shipped tonight, live in the 04:30 v27 release:

* SECOND-SLOT RULE (refresh_cycle.stage_bandit -> 3-tuple + stage_second +
  models/second_slot.json + daily_release.newest_pair): the bandit keeps
  slot 2 only on a sign-tested paired win over the route (McNemar on
  (seed set x opponent) cells); otherwise slot 2 = best tournament candidate
  from a DIFFERENT day-3 opening family (v{x}.{y}_route2.py). The marker
  file is authoritative; a bandit that lost its seat is never published
  just for existing.
* INTRADAY-FIX GATE (src/intraday_gate.py + submit.py step 1b): a v{x}.{y>0}
  build refuses to ship without a fresh (<24h) sign-tested PASS verdict vs
  the submission it retires. Override: --allow-untested-fix (operator only).
* ADAPTIVE SELL-TIMING (tape_runtime): opponent dump reconstruction from
  public inventory deltas (lower bound: delta minus our fills), phase-mod-24
  recurrence, pull scheduled SELLs forward to quote BEFORE a predicted
  collision; _adaptive_repay ledgers every pull so volume is timing-pure.
  MEASURED twice paired vs 8-opponent panel: 16 cells diff -1.6pp p=1.0;
  ledger version 24 cells / 192 games diff -1.3pp, discordant 1-3, p=0.625.
  NO EDGE on the open-loop panel -> flagship ships _ADAPT_SELL=False; the
  slot-2 diversity route ships it ON (experimental seat, operator order;
  panel-vs-live validity is WEAK rho=0.347, live ladder decides).
* Tests: test_second_slot (5 checks) + test_intraday_gate (7 checks); all
  14 fast suites + test_agents green. 04:30 task verified pointing at
  scripts/daily_release.bat in this repo -- no action needed to pick up.

## 2026-08-16 01:26 -- ENGINE SWAPPED TO 1.32.7 (operator-forced)

Operator order: v27 trains/gates/crowns entirely on 1.32.7. Evidence basis:
PyPI latest = 1.32.7; staff "rolls out shortly" (19h prior) + docs-updated
confirmation (2h prior); our 1.32.6 ladder reading was 25h stale and replay
fetches were 429-throttled until the 05:30 IST quota reset, so the hourly
watcher was blind through the 04:30 build window; Kaggle upload validation
runs the latest engine regardless. scripts/engine_swap_1327.py --swap: all
stages green (vendor install, 6 flag flips, rust rebuild + hinge probe
531/3252/758 @9000, vendored-python probe matches exactly, cross-check PASS,
engine_check re-baselined fp 2dd3d49b983445fc, planner rebuilt, preranker
recall revoked, parity 2/2 bit-identical). requirements.txt >=1.32.7.
14 fast suites green post-swap. Pre-hinge conclusions void; the adaptive
sell-timing verdict is being re-measured on hinge economics.

## 2026-08-16 (morning) -- "why is every morning broken": the audit and the fix

The recurring death spiral, reconstructed from three days of logs:
02:35 hourly run also ran a 30+12 GB archive mine ending minutes before the
release -> the ROLLING replay quota was spent exactly at 04:30 -> the release
found loss tapes 429d (Aug 15 HELD; Aug 16 tape-fallback) -> any unrelated
crash forced a rerun -> the hourly track deliberately skips while a release
runs, so index mtime aged -> the rerun saw "stale" and launched the FULL
in-line fetch (ourgames + fresh_data --lb-gb 12, up to 4h) on a dead quota ->
hours of throttled crawl, more quota burned, repeat tomorrow.

Fixes (all tested, tests pinned in tests/test_quota_guard.py):
* run() TIMEOUT SEMANTICS in BOTH refresh_cycle.py and daily_release.py: a
  hung/overrun child returns code 124 and the run continues (the Aug-16
  04:30 killer was one uncaught TimeoutExpired from seq_dataset.py).
* REPLAY-QUOTA LEDGER at the single chokepoint (sameday._quota_allow, used
  by sameday/ourgames/routes/cycle tapes): rolling 24h budget (2,000
  downloads, env-overridable) + a 03:00-06:00 quiet window for non-release
  callers; KAGG_RELEASE=1 (set by daily_release ENV) bypasses but records.
  QUOTA-HOLD is never retried through the CLI transport (same quota, slower).
* ARCHIVE MINE MOVED 02:xx -> 21:xx and shrunk (30->12, 12->8 GB): 7h+ of
  quota recovery before the release instead of zero.
* HEARTBEAT FRESHNESS: the hourly script writes data/sameday/heartbeat.txt
  after every successful run (even empty ones); the release fetch-skip keys
  on it instead of index mtime, which lied whenever a sip found nothing new.
* RELEASE FETCH TIME-BUDGETED: ourgames 25 min, fresh_data 60 min with
  --lb-gb 4 (freshness is the release's need; completeness is the hourly
  track's job).

## 2026-08-16 (mid-morning) -- demand-capture steering + rating curves + a third timeout site

* RATING CURVES (task 87 DONE): src/rating_track.py snapshots the active
  pair hourly (wired into sameday_scrape.bat) -> data/lb/rating_track.jsonl;
  dashboard Models tab renders rating-vs-time sparklines. Kaggle exposes only
  the current score, so curves accumulate from today -- no backfill exists.
  Verified rendering in Chrome (2 SVGs, no console errors).
* DEMAND-CAPTURE STEERING (task 86, implemented; measurement queued behind
  the release's 3-wide compute cap): audited v1_heuristic's demand model
  against the vendored 1.32.7 interpreter and found it was still the 1.32.4
  engine -- town centre "every 12 steps, 1/2/4 by decade" vs the true ONCE A
  DAY, FLAT 1 (up to 8x late-game drain overestimate -> outlook prices
  inflated -> overplanting into gluts), and "every never-seen shop at half
  weight" vs the true DRAWS WITH REPLACEMENT (uniform 1/8, every 3 days,
  8 instances cap). town_demand_per_day now: realized unlocked_shops
  (duplicates count) + replacement expectation for remaining draws + flat
  centre. This makes the outlook PER-GAME demand-matched -- exactly what the
  1.32.7 hinge pays (demand_match is the #1 elite marker).
  Same 1.32.4 centre model found and fixed in counter_agent.py's opponent-
  sell reconstruction (phantom consumption on 30 steps/game inflating every
  opponent dump estimate; ships in TONIGHT's bandit build).
* Third uncaught-timeout site fixed: fresh_data.py _run (killed this
  morning's fetch child mid identifier retrain); trackp/pipeline.py _run
  hardened the same way. episodes.py already converts to KaggleError; the
  interactive submit path is human-attended and left as-is.

## 2026-08-16 (09:15) -- demand-model fix MEASURED: sign-tested positive

Panel measurement was uninformative (v1 lineage scores ~0 vs elite mined
routes on 1.32.7 -- saturated referees measure nothing; referee-power lesson
re-confirmed). Head-to-head, where the demand model is the only difference:
NEW beats OLD 22-10 over 32 paired games (68.8%, binomial p~0.03), mean
margin +3.3k/+4.1k per seed set. v1 hinge pins verified exact (531/3252/758)
before trusting the number. The engine-true demand model stays; v2
regeneration (tune.py on the corrected model) is routine follow-up in the
params line; the counter_agent reconstruction fix ships in tonight bandit.

## 2026-08-16 (10:45) -- CROWN-ON-SERVE: tournament substrate swap, audit-gated

The crown took 2-3h because every cell ran the official pure-Python engine,
3-wide (BSOD cap). The Rust engine already existed; what was missing was a
crowning-grade harness. Built and shipped:

* src/serve_match.py -- real Python agents (every closed-loop layer live) on
  `kagg serve`; MEASURED 0.63s/game vs 10-15s official; also an
  evaluate.py-compatible eval mode (same CLI, same RESULT lines), so the
  tournament swap is a command substitution, not a parser change. Also a
  standalone tool: serve_match.py A B --seed N [--compare-official].
* SUBSTRATE EQUIVALENCE AUDIT: 12/12 pairings (closed-loop candidates vs
  loss tapes + panel, incl. the adaptive layer) EXACT-BANK matches ->
  models/serve_equiv.json.
* refresh_cycle.serve_allowed() -- gate: >=12 exact matches, 0 mismatches,
  current engine, <7d old. stage_tourney logs the substrate decision; an
  empty serve cell replays on the official engine (defensive).
* serve_spot_check() -- after the halving rounds, two finalist pairings
  replay on the official engine; ANY bank mismatch increments the audit's
  mismatch count = self-revocation (funnel adopt-with-kill-switch pattern).
* tests/test_serve_substrate.py (8 checks); 16 fast suites green.

Effect: tomorrow's 04:30 tournament runs ~16-25x faster (~2,300 cells x
0.63s ~ 25 min single-file; today's in-flight cycle predates the wiring and
finishes on the official path). The pre-ranker funnel remains separate and
still gated on recall.

## 2026-08-16 (10:55) -- 06:44 release tree died externally mid-tournament; relaunched

The 06:44 run completed screening (11 -> 6) and was three cells into round 2
when the entire tree (cmd + daily_release + cycle + evaluates) vanished
between 10:24 and 10:44 -- no traceback, no exit banner, scheduler reports
exit 0, no crash events, ExecutionTimeLimit PT10H not reached, not a battery
stop. Timing correlates with a Claude Code session restart (the second
session-boundary tree death; the first was the v24-era detached run). Cause
unproven; noted as a pattern. Relaunched 10:55 via Start-ScheduledTask --
the new process loads the serve tournament substrate, so the redone
tournament costs ~25 min instead of 2-3h. First release running with ALL of
today's rails: budgeted fetch, quota ledger + KAGG_RELEASE bypass,
timeout-as-failed-child everywhere, second-slot rule, serve substrate with
official spot-check.

## 2026-08-16 (11:00) -- ROOT CAUSE of the 10:24 tree death: BSOD 0x133 again

System log: bugcheck 0x133 (DPC_WATCHDOG_VIOLATION) at 10:23:23 -- the same
driver watchdog crash as the two 2026-08-13 incidents -- then reboot 10:34 +
a servicing restart. The kill window matches the release log cutoff to the
second. Contributing load: the tournament 3-wide PLUS session-run
measurement games (demand A/B, serve-equivalence audit officials) in the
same window. THE 3-WIDE CAP IS MACHINE-WIDE, NOT PER-JOB -- measurement
games must never run while a cycle tournament is active. The serve
substrate shrinks this whole risk class (one Rust process + light drivers
instead of N pure-Python engines); today's relaunched run is on it.

## 2026-08-16 (13:25) -- v27 PAIR SHIPPED; every new rail exercised in one run

Uploaded 13:24 IST (refs 55547047 route / 55547042 route2), first pair
crowned, gated and validated wholly on 1.32.7. The run itself proved the
day's engineering end to end:
* seq_dataset overran AGAIN (yesterday's release-killer) -> "TIMED OUT
  (treated as failed, cycle continues)" -- one log line, zero damage.
* First RUST-SERVE tournament: 6 MINUTES (13:02->13:08) for the halving
  rounds that took 2-3h official; spot-check 2/2 exact banks -> substrate
  stays adopted.
* Crown: HOLD (best fresh candidate 33% vs incumbent 50%+10) -- expected on
  transition day; HOLD-rebuild shipped the incumbent base with fresh models.
* SECOND-SLOT RULE first verdict: bandit 43/72 vs route 43/72, 18 cells,
  0 discordant, diff +0.000 -- the guard-locked bandit is cell-for-cell
  IDENTICAL to the route; seat handed to the diversity route
  (episode-91579667-replay_s1, different day-3 opening, ADAPTIVE SELL ON).
* Fetch ran inside its new budget (~45 min); quota guard + KAGG_RELEASE
  bypass live; tape fallback carried the referees again.
Watch items: v27 ratings converge over ~24h (judge route-vs-route2 on EQUAL
windows -- rating_track curves now exist for exactly this); tomorrow's crown
gets the first hinge-native candidates; NVIDIA driver update is on the
operator (both 08-13 dumps bucket 0x133_ISR_nvlddmkm).

## 2026-08-18 -- adaptive retirement, A/A pair, convergence hold, 429 breaker

Context: v27/v28 slot-2 (diversity family + `_ADAPT_SELL=True`) sat 478-760
rating points under the flagship on equal windows -- the fifth and sixth
no-win measurements for adaptive sell-timing. Kaggle's episode endpoints
429'd for 30+ hours while every hourly run fired ~2,000 attempts into the
wall. An UNKNOWN submission (ref 55578702, 18:23 IST 2026-08-17, "v1:
three-layer heuristic (observer/conductor/executor)", publicScore 561.6) did
not come from this pipeline and retired v28.0_route2 from the active pair.

| Change | Where | Measured / rationale |
|---|---|---|
| Adaptive sell-timing RETIRED (flip removed from `stage_second`) | `src/refresh_cycle.py` | 5 measurements, 0 wins: 3 paired panels (-1.3..-1.6pp, p>=0.625) + 2 live windows (-760, -478) |
| Operator force marker (dated, self-expiring) | `second_slot_force()` + `models/second_slot_force.json` | ships route+bandit 2026-08-18 as an A/A pair; bandit is a verified route clone (18 cells, 0 discordant) so the converged gap = the ladder's noise floor |
| Release convergence hold | `daily_release.release_held()` + `models/release_hold.json` (19th-20th) | cycle trains/crowns, publish chain skipped; ratings converge with games (v27 retired at 18h still climbing) |
| 429-storm circuit breaker | `src/sameday.py` `_BREAKER_TRIP=40`, probe 25, cooldown 55 min | stops the 48k requests/day self-perpetuating throttle; release (KAGG_RELEASE=1) never probe-capped |
| Tests | `test_second_slot` (7 checks: flip ABSENT, force dates, hold window), `test_quota_guard` (7: breaker trip/stale/clear) | 16/16 fast suites green |

Also journaled late (fixed 2026-08-17 evening): the quota ledger recorded
ATTEMPTS in `_quota_allow()` -- an afternoon of real 429s burned the 2,000
budget with zero data and self-locked the guard. `_quota_allow` is now
check-only; `_quota_record()` counts only bytes-on-disk downloads.

## 2026-08-18 -- phase 2: two levers resolved on evidence, the factory goes live

| Question | Measured | Consequence |
|---|---|---|
| L4: do winners demand-match the public shop draw? | 280 paired winner-vs-loser episodes (clean day-3 draw from obsfeat `shop_profile`): 116 vs 128, pooled p=0.48, flat across all 8 draws | DROPPED. Caveat: 1.32.6-era evidence; re-test on hinge-era volume (`models/shop_study.json`) |
| L6: does within-turn SELL reorder pay? | Engine verified (`_process_market`): per-unit lockstep, marginal price per unit -- own reorder/split is revenue-NEUTRAL per product. What pays is queue-INDEX priority: order i pairs against opponent's order i; 4,214 exact-tie collisions in 73 games, net race value +2,218/game (contested_dumps) | Already shipped -- `tape_runtime._sell_first` has been in every route agent. Residual (contested-first among sells) = second-order, parked |
| L5: can we search routes the ladder cannot mine back? | `src/sell_search.py`: farm plan FIXED (legality-safe), SELL schedule mutated (move/split/merge/resize), scored on `kagg batch` vs 6-team elite panel, common seeds, (1+16) elitist, FINAL GATE = fresh-seed paired sign test vs base | First run live: base 0.2708 on the panel, 0.3021 by gen 9. A PASSED agent enters the tournament as `factory::<path>`; `stage_build` copies a factory winner. Crown still decides |

Also: the research doc's "price-impact ranking" (sort your payload by marginal
revenue) is now engine-DISPROVED for this game -- lockstep marginal pricing
makes it a no-op. Keep the receipt; do not rebuild it.

## 2026-08-20 evening -- the performance gap diagnosed and the fix wired

**Diagnosis (own-games forensics, 733 games):** the flagship sells ZERO
CARROT/TOMATO/EGG -- the 1.32.7 scarcity trio -- while opponents' egg volume
tracks their wins (31 EGG in their wins vs 10 in their losses). Deficit
opens median day 17 and compounds steadily: we are out-earned mid-game by
hinge-priced sales we do not participate in. No market layer can patch it
(BUY_PRODUCT is WHEAT/FERTILIZER only) -- it is a farm-plan gap.

**Screen (src/hinge_screen.py, obsfeat sell-mix ranking + kagg batch):**
786 hinge-selling fresh routes in the pool; of the top 10, FIVE sign-tested
better than the incumbent on the elite panel -- best: 94034947_s0 (Hanaro)
0.875 vs 0.451, +42pp, discordant 37-3, p~0. These routes were mined and
indexed all along; the medoid/rank funnel never gave them a ticket, which
is why the crown HELD on a stale base for 5 days while the ladder marked
it down to ~1650.

**Fix:** HINGE_SLOTS=4 reserved tournament tickets in stage_tourney for the
top hinge-trio sellers (obsfeat share x bank). Same referees, same overlap
exclusion, same +10pp crown bar -- the quota only guarantees they get
MEASURED. First exercised by the 2026-08-21 04:30 cycle (post-hold, so it
also publishes). 16/16 suites green.

## 2026-08-21 (evening) — v32 root cause + shell fixes + gauntlet instrument

**v32 root cause CLOSED (mechanical, ours):** `_safe_market` clamped SELLs to
`_projected_shed`, whose deposit model only credited `DROP`. The engine has a
second deposit path — `PLACE item n` beside the shed — and Izzoudine's route
deposits that way (48 PLACE vs 9 DROP; kss: 49 DROP). The projection read ~0,
the clamp cancelled essentially every sale, and the agent banked 2,146 on the
same seed/opponent where the bare tape banked 58,292. On the ladder that was
the 411/320 crater. The engine's market is per-unit lockstep partial fill
(`_commit_unit`), so the clamp's premise ("oversized orders are abandoned")
was false anyway.

Fixes shipped in `src/tape_runtime.py` (+ `src/v22_agent.py` projection):
1. `_safe_market` no longer clamps or cancels — malformed-order filter only
   (bandit template keeps its clamp because additive layers rely on it, but
   its projection now credits PLACE deposits too).
2. `_sell_first` split into two flags: `_FEED_PIN` (turn-0 wheat buy to slot
   0 — feed-denial defense, Rayk C94) and `_SELL_FIRST` (blanket reorder,
   helps one family, hurt another by -16k on one seed / won by +20k on
   another — per-route flag decided by the gate, never assumed).
3. `_weed_repair` also repairs weed-blocked `PLACE` (C92's exact case).
4. `_terminal_market` over-asks (projected + carried) since partial fill
   makes over-asking free.
5. `routes.py` refuses `endgame_pull`/`swap_advance` builds — their
   conservation depended on the removed clamp (double-sell hazard).

New instruments: `src/gauntlet.py` (round-robin vs data/gauntlet reactive
public agents: V16-RC5, Kaito v27, Rayk C94/C95, Adaptive-R1 + our proven
builds — serve-substrate, paired seats), `src/strict_future.py` (chronological
veto gate, freeze recorded at episode 95803242). Majority-vote reconstructions
built for the three stable top teams: mv_saikushal_0821 (rank 12, field-sim
0.992), mv_izzoudine_0821 (rank 6, 0.926), mv_peikopon_0821 (rank 7, 0.926).
Top-5 teams measured ADAPTIVE (field-sim 0.05-0.30) — not copyable, ever.

## 2026-08-21 (night) — v33.0 pair shipped: the two-lineage experiment

Gauntlet cross-matrix (6 seeds, both seats, serve substrate): V16-RC5 0.767
> our v25/kss 0.667 > Adaptive-R1 0.633 > C95 0.533 >> Kaito v27 = our
v22_2/val 0.200. All fresh-mined top-team reconstructions FAILED the veto
(sai 0.25, izz 0.25, pei 0.04): the top-5 teams are adaptive (field-sim
0.05–0.30, uncopyable), and even the stable rank-6–12 routes fold against
reactive elites. Fresh-tape copying is closed as a lane.

Ship decision (evidence): flagship v33.0_route = Valmorlee 91464294_s1 (our
only converged own-ladder evidence, 2776 @ 71.8%/103 games; fixed shell
sweeps the old build 12-0 mirror; strict-future PASS 33-17 p=0.033).
Slot 2 v33.0_route2 = kss 92513718_s1 (different opening family, best of
ours on the elite panel 0.667, strict-future PASS 34-16 p=0.015,
uncorrelated with the V16-RC5 clone flood). Bandit lost the seat (ab_test:
no sign-tested difference vs route). Feed-pin: inert for kss (already
wheat@slot0), for val cuts feed-denial loss margins −21k→−4.3k at equal W/L.
Review catch: weed repair no longer fires on PLACE-as-deposit (only
PLACE-as-animal-placement); both agents deterministic; self-play validation
seed varies per submit.py run (bank levels not comparable across runs).

This pair is a DESIGNED ladder experiment: two lineages converge in
parallel; judge at 100+ games (~48h); the winner anchors slot 1 and the
next experiment takes slot 2 (winning-plan Phase 3, <=2/week).
