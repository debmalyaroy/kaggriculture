# Agent v0 — baseline

**File:** `agents/v0_baseline.py` · **Typical bank:** ~$10k · **Beats:** `pass`,
`random`, `starter` (the built-in baselines) every time.

v0 exists to prove the action plumbing works end to end and to give v1/v2 a
non-trivial punching bag. It is deliberately simple and every simplification is
listed below.

## How it works

```
carrot-only monoculture on the starting NW quadrant
  + 4 hired hands per day
  + static round-robin tile ownership
  + sell everything, every turn
```

1. **Tile ownership.** The 25 NW tiles are listed in row-major order and tile
   `i` is assigned to unit `i % n_units`. Because the list is row-major, unit
   *u* ends up owning column *u* — a compact vertical strip, which keeps
   walking distances short by accident rather than by design.
2. **Per-unit decision.** Each unit scans only its own tiles, scores each one
   with `_needs_work`, and takes the highest-priority job, breaking ties by
   Manhattan distance. If it is not standing on the target it takes one step
   toward it.
3. **Priorities.** `WATER (in bonus window) > HARVEST > PLANT ≈ WATER > DIG`.
4. **Market.** Four `HIRE` orders at hour 0, then sell the entire shed and top
   the seed stock back up to one seed per empty tile.

## The one non-obvious detail

The first draft harvested carrots as soon as `age >= max_yield_day` and scored
$7.2k. Carrot's bonus-watering window is days 2–3, so on day 3 the tile needs
*both* a `WATER` (worth +1 unit) and a `HARVEST` — and a unit only gets one
action. Harvesting first threw away the last watering bonus: 2 units per plant
instead of 3, a third of all revenue.

Giving in-window watering a higher priority than harvesting took v0 from $7.2k
to $9.4k against the same opponent. The same rule, generalised, is in v1's
`build_tasks`.

## Deliberate limitations

| Limitation | Cost | Addressed in |
|---|---|---|
| One crop, chosen a priori | ignores that carrot is 7th of 8 assets by $/tile-day | v1 portfolio planner |
| Never buys land | caps the farm at 25 of 100 tiles | v1 land rules |
| No livestock | forgoes the three highest-value assets in the game | v1 |
| Fixed 4 hands | under-hires when there is work, over-hires when there isn't | v1 labour model |
| Sells the whole shed every turn at any price | walks fragile prices to the floor | v1 market engine |
| No fertilizer | forgoes ~30% yield on one-time crops | v1 |
| Static tile ownership | a unit idles while its neighbour is swamped | v1 global assignment |

## Reproducing

```bash
python -m kaggriculture.engine.run_match agents/v0_baseline.py starter --seed 3
python -m kaggriculture.measure.evaluate agents/v0_baseline.py --vs pass random starter -n 4
```
