# Kaggriculture — environment reference

Distilled from `kaggle_environments/envs/kaggriculture/kaggriculture.py`
(v1.32.3) and the official README. **Where the official README and the
interpreter disagree, this file follows the interpreter** — that is what
actually scores the match. Discrepancies are listed in
[issues-and-improvements.md](history/issues-and-improvements.md).

## Format

| | |
|---|---|
| Type | 2-player simulation (agent vs. agent), not a CSV competition |
| Submission | `main.py` exporting `agent(obs)` — or a `submission.tar.gz` with `main.py` at the root |
| Episode | 720 turns = 24 turns/day × 30 days |
| Per-turn budget | `actTimeout` 1 s (`remainingOverageTime` 60 s for the whole episode) |
| Starting bank | $3,000 |
| Score | Bank balance at the final step. Unsold inventory counts for **nothing** |

## Action shape

```py
{
  "farmer": [op, *args],          # the one permanent farmer
  "hands":  [[op, *args], ...],   # one entry per hired hand, in `farms[me]["hands"]` order
  "market": [[op, *args], ...],   # ordered; only the first 10 are processed
}
```

Unit ops: `NORTH/SOUTH/EAST/WEST`, `PASS`, `PICKUP <item> [n]`, `PLACE <item> [n]`,
`DROP`, `PLANT <crop>`, `WATER`, `HARVEST`, `FERTILIZE`, `BUILD_COOP`,
`BUILD_PASTURE`, `DIG`, `FEED`, `CARE`, `COLLECT_FERTILIZER`.

Market ops: `BUY_SEED`, `BUY_PRODUCT` (wheat & fertilizer only), `BUY_ANIMAL`,
`SELL`, `HIRE`, `BUY_LAND`. **Illegal actions are silent no-ops** — there is no
error signal, so a broken agent looks like a lazy one.

## Turn order

1. Unit actions for both players (simultaneous)
2. Market queue, one unit at a time, both players in lockstep
3. Town shops + town centre drain market inventory
4. Plant decay tick
5. End of day (every 24th turn): watering/feeding checks, production, weed
   spawn, inventories dropped to the shed, hands dismissed, farmer respawned at
   the shed, a new shop possibly unlocked

## Assets

| Asset | Capital | First yield | Cadence | Cap | Units/day (played well) | Base $ | $/tile-day |
|---|---|---|---|---|---|---|---|
| Wheat | $10 | day 2 | one-time | 6 | 1.00 (1.50 fert) | 25 | **25** |
| Carrot | $20 | day 2 | one-time | 4 | 1.00 (1.33 fert) | 35 | **35** |
| Tomato | $50 | day 8 | daily | 4 | 0.31 (0.62 fert) | 60 | **18** |
| Strawberry | $100 | day 10 | 2-daily | 4 | 0.24 (0.47 fert) | 120 | **28** |
| Melon | $80 | day 10 | one-time | 6 | 0.60 | 250 | **150** |
| Goose (egg) | $300 + coop | day 4 | daily | 4 | 2.00 | 50 | **100** |
| Cow (milk) | $400 + pasture | day 8 | 2-daily | 6 | 1.50 | 160 | **240** |
| Sheep (wool) | $500 + pasture | day 6 | 3-daily | 6 | 1.33 | 200 | **266** |

Derivations that are easy to get wrong:

* **Watering bonus window** for one-time crops starts at `ceil(max_yield_day/2)`
  and each watered day inside it adds **+1** unit (+2 if fertilized).
  Melon therefore reaches its cap of 6 on **day 10**, not day 12 — harvest then;
  waiting only risks decay.
* **A plant must be watered on its planting day.** `_new_plant` sets
  `consecutive_unwatered = 1`, so one more missed day turns it into a weed.
  After that, watering every *other* day keeps a plant alive; daily watering
  only pays inside the bonus window.
* **`CARE` banks +1 per day, not +2.** The interpreter does
  `pending_care_bonus += 1` when the animal was both fed and cared for. The
  bank is paid out on the next scheduled production, so steady state is
  `(1 + interval) / interval` units per day: goose 2.0, cow 1.5, sheep 1.33.
* **Feeding costs 1 wheat per animal per day** and the wheat must be in a
  unit's *inventory* (`PICKUP` from the shed first). Two consecutive unfed days
  and the animal is gone permanently; the empty pen remains.
* **`FERTILIZE` covers `day`, `day+1`, `day+2`** and only pays on days the
  plant is also watered.

## Labour

* `HIRE` costs `fib(n)` where `n` is the number of hires already made *that day*:
  1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987…
* Cumulative: 8 hands = **$54/day**, 12 hands = **$376/day**, 16 hands = **$2,583/day**.
* Hands are dismissed at end of day and re-spawn at the shed. Each acts every turn.
* Only 10 market orders are processed per turn, so >10 hires need a second turn.

Labour is by far the cheapest resource in the game and is almost never the
thing to economise on — the first ten hands cost less than a single melon seed
crop.

## Land

`BUY_LAND` unlocks quadrants in the fixed order NE → SW → SE for
**$1,000 / $2,000 / $4,000**, 25 tiles each. At the observed $50–150 per
tile-day, a quadrant bought on day 5 returns its price many times over.

## Market

```
price(inv) = base ± amp · f(|inv − I0|),   I0 = 10,000,  amp = target·base / f(T)
```

Sale price is quoted **before** the sale; sales at the $1 floor do not add to
inventory. Buy price is quoted at post-buy inventory, so a buy/sell round trip
nets exactly zero.

Units of glut needed to drive each product to the $1 floor:

| Product | Units above I0 to reach $1 | Character |
|---|---|---|
| Wheat | (unreachable — log curve) | glut-proof staple |
| Egg | (unreachable — log curve) | glut-proof staple |
| Carrot | ~866 | tolerant |
| Tomato | ~537 | tolerant |
| Melon | ~158 | fragile |
| Milk | ~76 | very fragile |
| Wool | ~59 | very fragile |
| Strawberry | ~62 | very fragile |

## Town demand — the real revenue ceiling

Shops unlock on days 3, 6, 9, …, 24 (random order, all 8 by day 24) and each
consumes 1 of every product it wants every 4 turns (6/day; single-product shops
12/day). The town centre consumes 1 of every non-fertilizer product every 12
turns, doubling after day 10 and doubling again after day 20.

Expected units drained over a full season, and what that is worth at base price:

| Product | Expected demand | Value at base |
|---|---|---|
| Strawberry | ~536 | $64k |
| Milk | ~437 | $70k |
| Wool | ~338 | $68k |
| Melon | ~140 | $35k |
| Tomato | ~338 | $20k |
| Egg | ~338 | $17k |
| Wheat | ~635 | $16k |
| Carrot | ~437 | $15k |

Total ≈ **$305k of demand shared between two players**. Every match we have
instrumented ends with market inventory *300–500 units below* I0 on the premium
goods — i.e. both agents combined are producing well under half of what the
town wants, and prices spend the whole game *above* base. See
[strategy.md](history/strategy.md) for what that implies.

## Observation notes

* `farms` is public for both players (tiles, money, positions, quadrants).
  Only `private` (shed, seeds, per-unit inventories) is hidden.
* `tiles[y][x]` — y is the row. Farmer position is `[x, y]`.
* The shed holds **100 non-seed items**; the end-of-day inventory drop discards
  anything that does not fit. Seeds live in their own uncapped slot.
* `config` is passed as the agent's optional second argument; `obs["step"]` is
  supplied by the framework, `obs["day"]`/`obs["hour"]` by the environment.
* `step` is `"shared": true` in the base schema, so the core copies it into
  **every** seat's observation before calling the agent — verified by probe,
  0..718 in both. It is absent only from the **stored replay**, because shared
  properties are stripped from non-first agents when the episode is serialised.
  Read `obs["step"]` in an agent; use day*24 + hour when parsing a replay.

---

## 2026-08-07 balance change (engine >= 1.32.6) — CURRENT RULES

Confirmed via discussion 733431 and kaggle-environments PR #1394; vendored
locally 2026-08-10. **Any note above that contradicts this section is stale.**

* **Town Center**: buys ONCE per day (interval 24), flat 1 of each product
  except FERTILIZER. The old 2x/day schedule with 2x (day 10+) and 4x
  (day 20+) multipliers is REMOVED. Late-game demand is ~8x lower; markets
  floor much faster under sell pressure; late selling is structurally worse.
* **Shops**: sampled WITH replacement, up to `MAX_SHOP_INSTANCES = 8` of the
  same shop. `unlocked_shops` lists duplicates; each instance consumes on
  the same 4-turn schedule. Per-game demand composition is now random and
  observable — a game may have no yarn store (wool demand collapses) or
  several smoothie shops (strawberry/milk demand doubles+).
* **Hand spawning**: the spawn-on-locked-tiles bug is fixed (the SE trap
  hand at (5,5) no longer occurs). Routes recorded pre-fix under-use that
  hand.
* Fingerprinted market constants are UNCHANGED — engine_check cannot see
  this class of change; the refresh cycle's version-drift alarm (archive
  engine metadata vs vendored) is the guard.

## 2026-08-15 balance change (engine >= 1.32.7) — ANNOUNCED, rollout pending

Discussion 735311 + PR #1399: CARROT, TOMATO and EGG's scarcity side becomes
the **hinge** shape — `u = x/T; f = u + 8·max(0, u−1)²` (f(T)=1, so `target`
keeps its meaning). Params: CARROT below log/0.20 → hinge/**1.00** (changes
below-knee pricing too — mild scarcity pays ~2× the old log curve), TOMATO
linear→hinge (0.40), EGG linear→hinge (0.40). T unchanged (450/200/332).
Above-I0 unchanged. Everything else unchanged. Staff: "should be the last
change."

Measured magnitudes (verified vs the 1.32.7 interpreter): CARROT quotes
$531 at 1,000 units drained, $3,513 at 2,000; TOMATO $3,252 at 1,000;
EGG $758 at 1,000. Activation odds given NO production (staff): tomato 50%,
carrot 26%, egg 22% of games — keyed to the shop draw.

**Rollout discipline**: the ladder still ran 1.32.6 at announcement time.
`scripts/engine_swap_1327.py --check` rides every hourly scrape and performs
the atomic swap (vendor, Rust ENGINE_1327, counter_agent flag,
engine_version.json, planner rebuild, re-baseline, funnel-evidence
revocation, parity) within ~1h of the first 1.32.7 replay. Until then every
component stays on 1.32.6 — the 1.32.4 lesson.
