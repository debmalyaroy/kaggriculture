# agent v9 — the economic rebuild

`agents/v9_cem.py` is `agents/v1_heuristic.py` (rebuilt on 2026-08-05) with a
`PARAMS` block found by cross-entropy-method policy search against recorded
top-ladder trajectories. Knowledge graph: `agents/v9_cem.html`.

**Measured, paired, both seats:**

| against | win rate | mean bank | opponent | margin |
|---|---|---|---|---|
| `agent_v4_optimal` (last submitted) | **100%** (10) | $116,498 | $64,263 | +$52,235 |
| `v1_heuristic` (rebuilt, pre-search) | **100%** (10) | $102,064 | $80,870 | +$21,194 |
| itself with the market model off | **100%** (12) | $96,584 | $75,637 | +$20,948 |
| top-ladder tapes | 0% | ~$77k | ~$160k | −$52k |

The last row is the honest one: this agent beats everything we have ever built
and still loses to a recorded 2,900-rated game. See §6.

---

## 1. Why v4 was stuck at the field median

v4's rating was 683.7, rank 957 of 1,908, against a field median of 685.7. It
was not broken — ten downloaded ladder replays show `DONE` in every seat, no
timeouts, 5W-5L. It was **ordinary**, and the reason is visible the moment you
put its cash flow next to a top game's:

| | v4 | top of ladder |
|---|---|---|
| sheep | 0 | 6 |
| wool revenue | $0 | ~$40k |
| fertilizer sold | ~0 | 216 units (~$12k) |
| first animal | day 11 | **day 0** |
| wheat tiles | up to 72 | 0–11 |
| strawberry tiles | 8–24 | 40 |
| unit-turns worked | 4,972 | 7,321 |

Four of those are one mistake each, and each is fixed below.

## 2. Fertilizer is a product

`_daily_refresh_animals` sets `fertilizer_available = True` on every animal
every day. `COLLECT_FERTILIZER` turns that into one unit of FERTILIZER for one
unit-turn. Its base price is $100 and **nothing in the town consumes it**, so
it starts at exactly base and falls only as players sell.

Every build up to v4 defined `SELLABLE = [p for p in PRODUCTS if p != "FERTILIZER"]`
and sold none of it. Fourteen animals over a season is ~400 free units; the
first ~250 into the market still fetch $50–100.

Two consequences in the code:

* `SELLABLE` now includes FERTILIZER, with `fert_stock` units held back for the
  fields;
* `animal_lifetime_profit` counts the fertilizer stream, which is what turns a
  $400 cow into a three-day payback and makes buying on turn zero obvious.

## 3. Price against the market you will sell into

`crop_value_per_tile_day` used the **spot** price. A planting decision is made
ten days before the sale, and melon is the trap: the town centre is its only
buyer (~140 units a season) and its glut curve is *squared*, so two farms each
planting a dozen melon tiles destroy the price before either harvests.

The new `market_outlook()` computes, per product,

```
effective_inventory = now
                    + horizon x (standing supply on both farms
                                 - town demand over the sale window)
```

and prices everything through it. Three details earn their keep:

* **the mirror prior.** On turn zero neither farm has planted, so a census of
  standing tiles says the market is empty and melon is worth $250 — right up to
  the moment both farms harvest sixty of them. `mirror_weight` assumes the
  opponent grows what we grow unless their own tiles already say otherwise.
  CEM raised it from 1.0 to **1.46**, independently confirming it matters.
* **a demand window, not a season.** A harvest lands in a burst; the town has
  not drained a season's worth by the afternoon we sell.
* **marginal pricing.** `plan_tiles` prices tile eleven against the ten already
  planned, so a crop caps itself at the size of its demand instead of at a
  hand-set tile count.

Turning this whole layer off (`outlook_weight = 0`) costs **0% win rate and
$20,948 a game**. It is the largest single component in the rebuild.

## 4. Cash flow is one decision, not six knobs

v4 held a **twelve-day** feed runway in cash — $6,700 at a fourteen-head herd —
through exactly the window where livestock and land are the highest-return
things money can buy. That is why the first animal arrived on day 11.

* the runway is now measured in **days until the farm earns anything**
  (`income_gap`), not a constant;
* `BUY_ANIMAL` fills the *shed* — a pen is only needed to `PLACE` — so the
  purchase is no longer gated on an empty pen already standing. That gate is
  what stalled turn zero: no pens exist yet, so no animal could be bought, so
  the $3,000 went into seed;
* livestock is served **before** seed and land in the order queue;
* an explicit **cash crop** (`cash_crop_*`) reserves ~10 tiles for the fastest
  profitable crop whenever the income gap is open. It is not competing on rate
  — melon beats it five to one — it is buying the days in between. Without it
  the farm buys three cows on turn zero and watches two starve on day three.

## 5. Feed is bought, not grown

`crop_target("WHEAT")` used to track the herd (`feed_self_frac` 0.85), and
`filler_crop` was pinned to WHEAT, so every spare tile on every quadrant became
the cheapest crop in the game. A wheat tile returns about what a wheat unit
costs; the same land under strawberry has four times the price ceiling.
`feed_self_frac` is now 0.25 — a hedge against the wheat price running away —
and the filler is whatever the market pays most for.

## 6. What still loses

Against the recorded top-ladder trajectories this agent banks ~$77k to their
~$160k and has not won a game. The gap is concentrated in days 0–10, where they
are already selling fertilizer and wheat while we are still planting.

Their submissions are not policies. They are 719-turn recordings with a
slip-recovery layer (see `BUILD_JOURNAL.md`), which is viable because the
environment is deterministic apart from weed spawns and shop-unlock order.
Closing the gap means either out-producing that trajectory in the first ten
days, or optimising and shipping our own.

## 7. Parameters that changed, and what the search said

CEM ran 14 generations, population 14, 3 seeds x 2 seats x 2 tape opponents =
168 matches a generation, and moved the margin against the tapes from
−$75,055 to −$51,968. Notable moves off the hand-set defaults:

| knob | hand-set | CEM | reading |
|---|---|---|---|
| `mirror_weight` | 1.0 | **1.46** | assume the opponent out-grows the mirror |
| `min_tile_rate` | 6.0 | **14.2** | leave a tile empty rather than farm it badly |
| `target_strawberry` | 40 | 46 | more of the best ongoing crop |
| `target_melon` | 10 | 16 | with marginal pricing capping it in play |
| `sell_orders` | 4 | 2 | fewer, larger sales — order slots are scarce |
| `fert_stock` | 4 | **0** | sell every unit; applying it is priced per-tile anyway |
| `outlook_horizon` | 0.5 | 0.39 | discount projected supply a little |

## 8. Learned layers that were measured and *not* shipped

| layer | result | why |
|---|---|---|
| committee of 3 + gradient-boosted arbiter (151 trees, valid logloss 0.4095 vs 0.693 coin flip, export verified to 4e-07) | 50% win, −$3,724 | members are jittered copies of one policy, so voting adds variance, not information |
| opponent counter-play (`opponent_mode=counter`) | 56% over 16, +$374 | inside noise |
| score-aware risk (`adaptive_mode=risk`) | 42% over 12, +$283 | inside noise |
| Monte-Carlo control on `ASSET_BIAS` (80 episodes) | 17% over 12, −$8,383 | episodic win/loss credit over 128 states x 5 multipliers is far too high-variance at this budget, and the CEM search had already found a good allocation. Resumable; needs 200+ episodes |

The adaptive play that *does* pay is in §3: reading the opponent's public farm
and pricing our own crops against their future supply.


## 9. What counts as the RL here

The layer that worked is `src/kaggriculture/train/optimize.py`: cross-entropy-method policy
search. It samples whole parameter vectors, plays each one against recorded
top-ladder trajectories in both seats, keeps the elite fraction and refits the
sampling distribution. That is derivative-free policy optimisation on episodic
returns -- and it is what produced the +$21,194 over the hand-fixed engine.

The tabular Monte-Carlo control (`src/kaggriculture/train/train_rl.py`) is the more literal
reading of "RL" and it measured *negative* at the budget available. Both are in
the repo; only the first is in the agent, and the table above says why.

## 10. Two bugs worth remembering

* **`src/kaggriculture/pipeline/params.py` wrote agents in cp1252.** Every generated agent goes
  through it. A single em-dash in a module docstring produced a file Python
  refused to parse, and the agent scored exactly $3,000 -- its starting money --
  because the engine substituted a default action every turn. Now explicit
  UTF-8 on read and write.
* **`obs["step"]` is fine in both seats.** *(Corrected 2026-08-06.)* It is
  `"shared": true` in the base schema and the core copies shared properties
  into every seat before calling the agent, so a probe reads 0..718 in seat 1
  too. What is seat-0-only is the **stored replay**, which strips shared
  properties from non-first agents — so a tool reading a saved episode still
  needs `_step_index(obs)` (day*24 + hour), and that is where the original bug
  actually was.

## 11. Six fixes tried against the measured failures — all rejected

The public notebooks were read properly on 2026-08-06 (`src/kaggriculture/data/notebooks.py`
now harvests them; `.local/dbg/meta_findings.txt` has the extracted text). Two
independent top-ten write-ups reach the same conclusion:

> "The farmer and hand tapes contribute almost nothing. **The market tape
> carries the frontier gain.**" — prvsiyan, *The Soil Remembers Rain*

> "c18 changes only 20 pre-terminal field turns from c16 but **112 market
> turns**. This is the clearest evidence in the diary that the current edge
> comes from **inventory-sale timing** rather than another change in herd
> composition." — raykkretzschmar, *Findings from Zero to Top Meta*

They also publish the converged farm: **12 hands by day 10, 3 quadrants, 8
cows, 6 sheep, a 40-tile strawberry plateau, a late wheat refill.**

`src/kaggriculture/measure/leak_check.py` audits a replay for the five ways produce becomes
nothing. Against `tape_90036815_s1` on seed 33, v9_cem versus that tape:

| | v9_cem | the tape |
|---|---|---|
| SELL orders | 194 on 134 turns (**19%**) | 545 on 336 turns (**47%**) |
| milk sold | 96 | 181 |
| cows | **3** | 8 |
| unbanked at the end | **$7,007** | $106 |
| price timing (realised / peak) | 65-99% | 52-99% |

Per-unit timing is already comparable. The gaps are **volume** and **cows**.

Six changes aimed at exactly those gaps, each A/B'd over 10-16 paired matches
against v9_cem:

| change | result |
|---|---|
| reserve 3 tiles from crops for the planned herd | 30%, −$4,563 |
| reserve 8 tiles | 33%, −$2,145 |
| cash crop triggers on an empty bank, not just the income gap | 0%, −$16,003 |
| sell throughput 5 orders x 14 units (from 2 x 7) | 0%, −$14,316 |
| terminal haul window 8 turns instead of 24 | 0%, −$11,040 |
| terminal haul window 34 turns | 0%, −$14,253 |
| mirror front-run: prioritise the line they are about to dump | 56% over 16, −$416 (noise) |

**v9_cem is a genuine local optimum.** Two of these have real mechanisms behind
them — the front-run is a published +$1,865 result in a mirror match, and three
cows against a meta of eight is $14k of milk — and neither survives contact
with our actual production rate. Holding land for cows costs more strawberry
than the cows return; selling faster walks our own price down a convex curve.

### A measurement bug found on the way, worth more than any of the six

The first three of those tests were meaningless when run. Variants are built by
writing a parameter dict into the master, and `pen_reserve_max` had a **code
default of 8 that disagreed with its PARAMS default of 0** — so any agent
generated from a parameter dict predating that knob silently reserved eight
tiles. The "control" and the "pen fix" were the same agent.

Two fixes, both permanent:

* `src/kaggriculture/pipeline/params.py` now **completes** a caller's dict from the base file's
  PARAMS, so a generated agent can never be missing a knob;
* the A/B protocol starts with a control built the same way as the variant.
  A correct control now ties **exactly**: $101,422 to $101,422, margin $0.

Any A/B in this repo run before 2026-08-06 that generated its candidate from an
older parameter dict is suspect for the same reason.

## 12. v13 — reading their code, not just their write-ups

The write-ups all say the market channel carries the edge. Their **code** says
what the market channel actually does. Decoding a top-10 agent's `main.py`
(`.local/kernels/_agents/`) and reading `_projected_shed`, `_opponent_exposure`
and `_ranked_sells` produced two mechanisms. One of them is worth real money.

### Same-turn selling — the bug this exposed

The interpreter resolves every unit action **before** it processes the market:

```
kaggriculture.py:904   _apply_unit_action(...)   # DROP lands here
kaggriculture.py:910   _process_market(...)      # SELL runs here
```

So produce dropped this turn is sellable this turn. Every build up to v9_cem
read `private["shed"]` as it stood at the *start* of the turn, so **every
harvest reached the market a turn late** — and, worse, a unit standing on the
shed about to drop 12 wool contributed nothing to the sell plan. That is the
mechanism behind the number from the leak audit: we sold on 19% of turns where
the tape sold on 47%.

`projected_shed()` now adds this turn's DROP and PLACE deposits (respecting the
100-unit cap) before the sell plan is built. `assign()` already runs before
`market_orders()`, so the ops are known.

| | win rate | margin |
|---|---|---|
| vs v9_cem, seed set A (8 matches) | **75%** | +$9,659 |
| vs v9_cem, seed set B (12 matches) | **83%** | +$4,512 |
| vs `agent_v4_optimal` (8 matches) | **100%** | +$41,767 |
| vs `tape_90036815_s1` (8 matches) | 0% | −$50,459, but bank **$77k → $105.6k** |

Three independent seed sets, both seats. This is the first change since the
CEM run to beat the incumbent.

### Their sell-slot scoring — measured, rejected

`_ranked_sells` scores each line as

```
(1 + opponent_exposure[item]) * glut_weight[item] * price * log1p(quantity)
```

where `glut_weight` is the interpreter's own `above_target` — melon 3.6 and
wool 3.2 collapse violently when glutted, wheat 0.2 barely moves — and
`opponent_exposure` counts every standing tile of that line on their farm, not
just the harvest-ready ones.

Implemented faithfully as `sell_priority()` and measured **25%, −$2,158**. It is
a good rule for a scheduled tape that already knows its own volumes; layered on
a planner that re-prices every turn it just fights the valuation. Kept behind
`glut_priority`, default off.

### What is still unread

`bruceqdu/my-2026-08-04-high-score-pipeline`,
`beicicc/kaggriculture-c22-exact-reproducibility-control` and the v18/v19
Kaito notebooks have not been mined for mechanisms yet. Given that one function
in one decoded agent was worth +$9,659, that backlog is the highest-value
reading left.
