# Agent v1 — heuristic planner

**File:** `agents/v1_heuristic.py` (self-contained; submittable as `main.py`
unchanged) · **Typical bank:** ~$50k · **Beats v0:** 100% of matches, by ~$40k.

## Design principle

Everything is priced in dollars. Watering a melon, milking a cow, digging a
weed and hauling a crate to the shed all produce a dollar figure, so unrelated
jobs compete on a single axis and no hand-written priority ladder is needed.
Units are then matched to jobs by **value density** — dollars per unit-turn,
including the walk.

## Pipeline (once per turn, ~6 ms)

```
obs
 ├─ census            what the farm holds: crops by type, live animals, empty
 │                    pens, weeds, empty tiles
 ├─ plan_tiles        what each empty tile should become, subject to cash,
 │                    labour capacity and portfolio targets
 ├─ build_tasks       every actionable job, each with a dollar value
 ├─ assign            greedy value-density matching of units to jobs
 ├─ market_orders     hire → sell → feed → seed → land → livestock
 └─ action dict
```

### census / plan_tiles

`plan_tiles` walks the empty tiles **sorted by distance from the shed** and
hands the nearest ones to livestock (every feeding trip starts at the shed) and
the outer ring to crops. It is bounded by three things:

* **portfolio targets** — a per-asset tile cap (`target_cow`, `target_melon`, …)
  that encodes the demand ceilings from [strategy.md](strategy.md)
* **capital** — a pen is only committed if the animal to fill it is already in
  the shed or affordable *right now*, after the feed reserve
* **labour capacity** — `(1 + hands) × 24 × capacity_util` unit-turns, spent at
  `cost_per_crop_day` per crop tile and `cost_per_animal_day` per animal.
  Planting past this is negative value: an unwatered plant is a weed in two days.

Crops are ranked by live market price, not base price, so the portfolio drifts
toward whatever is currently scarce.

### build_tasks — the valuation rules

| Job | Value |
|---|---|
| `WATER` (plant would die today) | full remaining value of the plant |
| `WATER` (inside bonus window) | 1 unit × price (2 if fertilized) |
| `WATER` (safe, outside window) | *no task* — every-other-day watering suffices |
| `HARVEST` crop | units × price + a fraction of the next crop's run-rate (the tile is freed) |
| `HARVEST` animal | units × price, doubled in urgency at `max_held` |
| `FEED` (animal would die) | full remaining value of the animal |
| `FEED` (safe) | `feed_safe_discount` × remaining value |
| `CARE` | `care_weight` × product price (it banks +1 unit) |
| `FERTILIZE` | units it will actually add × price, capped by headroom |
| `PLANT` / `BUILD_*` | expected lifetime profit of what goes there |
| `DIG` weed / surplus pen | run-rate of the crop that will replace it |
| `COLLECT_FERTILIZER` | fraction of the fertilizer price |

Jobs needing an item (`FEED` needs wheat, `PLACE` needs the animal,
`FERTILIZE` needs fertilizer) carry a `need` field; if the unit is not carrying
it, the assignment adds the shed detour into the distance so the job's density
is scored honestly against jobs that need no detour.

### assign

All (job, unit) pairs are scored `value / (1 + distance + detour)` and consumed
greedily. Two properties make this stable:

* moving toward a job *raises* its own density, so units do not oscillate
  between targets across turns
* jobs are keyed by (tile, op), so two units may work the same tile in one turn
  when the ops are independent (one `FEED`s while the other `CARE`s) but never
  duplicate the same op

`PLANT` requests are counted against the seed stock before being emitted,
because the interpreter drops *all* plant requests for a crop if the turn's
total exceeds the seeds held.

### market_orders

Order matters — only 10 orders per turn are processed:

1. **HIRE** (hours 0–2; cheapest asset in the game)
2. **SELL** — for each product, the number of units that can go out before the
   marginal price drops below `sell_floor × base`, capped at `sell_chunk` to
   spread sales across the day so town demand can refill the curve. All floors
   are dropped from `dump_day`, and immediately whenever the shed passes
   `shed_pressure` (overflow past 100 items is destroyed).
3. **BUY_PRODUCT wheat** — feed. Runs ahead of every capital purchase.
4. **BUY_SEED** — bounded by `plantable_slots`, i.e. by labour capacity.
5. **BUY_LAND** — gated on day window, on ≥`land_min_used` of current land
   being in use, and on keeping the reserve intact.
6. **BUY_ANIMAL** — last, from surplus cash only, and re-checking the feed
   runway for the post-purchase herd size after every head.

## Four bugs that cost more than any tuning

Recorded because they are the interesting part of the development.

1. **Produce stranded in farmers' hands (+$25k).** `SELL` draws from the shed
   only. Goods harvested on day 29 are auto-dropped to the shed at end of day —
   which *is* the end of the game, so they score nothing. Making hauling
   outrank every other job on the last day was the single largest improvement
   in v1's development.
2. **The herd that could not be fed (+$8k).** A flat per-animal cash reserve let
   the agent buy 20 animals it could not feed; the whole herd starved around day
   18. Replaced by a feed *runway* reserve sized in wheat-days at the live
   wheat price, discounted by the farm's own wheat production.
3. **Runaway seed buying (+$14k).** The filler crop's target was modelled as
   "infinite", so the seed-demand routine bought a fresh field's worth of seed
   *every turn* until the bank was empty — which then blocked hiring, which
   collapsed maintenance, which turned the farm into a weed patch. Seed demand
   is now bounded by plantable slots.
4. **Mature animals valued at zero (+$5k, and a subtle one).** Remaining animal
   value was computed as `days_left − first_yield_day`, which goes negative in
   the last week — so nothing fed the herd and it starved two days before
   scoring. It now accounts for the animal's age.

## Known weaknesses

Carried into [issues-and-improvements.md](issues-and-improvements.md).

## Reproducing

```bash
python -m kaggriculture.engine.run_match  agents/v1_heuristic.py agents/v0_baseline.py --seed 3
python -m kaggriculture.measure.evaluate   agents/v1_heuristic.py --vs agents/v0_baseline.py starter -n 4
python -m kaggriculture.measure.analyze    agents/v1_heuristic.py --vs pass --seed 3 --every 4 --board
```
