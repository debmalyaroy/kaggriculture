# Agent v2 — self-play tuned

**File:** `agents/v2_tuned.py` — **generated**, not hand-written. It is
`agents/v1_heuristic.py` with a different `PARAMS` block, produced by
`src/kaggriculture/train/tune.py`. Never hand-edit it; fix logic in v1 and regenerate.

**Result:** mean bank **$82,588**, **100%** win rate against v1, v0 and the
built-in baselines.

| Opponent | Win % | v2 mean | opponent mean | margin |
|---|---|---|---|---|
| `agents/v1_heuristic.py` | 100% | $74,810 | $21,558 | $53,252 |
| `agents/v0_baseline.py` | 100% | $85,323 | $10,471 | $74,852 |
| `starter` | 100% | $87,632 | $3,508 | $84,124 |

(4 seeds x both seats per opponent.) Note v1 falls from its usual ~$50k to
$21.6k when it plays v2 — they compete for the same finite town demand, so a
stronger opponent takes revenue off you as well as adding it to itself. That is
exactly why the leaderboard is a skill rating and why the tuner optimises
margin.

## Method

Coordinate descent over 28 knobs, scored by paired self-play:

* opponents: a **frozen copy of v1** plus `v0_baseline`
* 3 fixed seeds, **both seats** played for every pairing, so seat and seed noise
  largely cancels
* objective: `mean(margin) + 0.25 x mean(bank)` — margin decides matches, the
  bank term breaks ties toward absolute strength
* one pass over the space; each knob's candidates tried against the current
  best, accepted greedily

Why coordinate descent and not CMA-ES or Bayesian optimisation: each evaluation
costs ~40 s of wall clock, the knob groups (labour / cash / portfolio / market)
are largely separable, and the search has to stay reproducible and
interruptible. It is a hill climber — it finds a local optimum, not the global
one.

Reproduce or extend:

```bash
python -m kaggriculture.train.tune --seeds 4 --passes 2 --budget-min 150
python -m kaggriculture.train.tune --only max_herd target_cow target_goose --seeds 6
```

The full trial log is in `.local/tune/history.json`.

## What the search changed, and what it means

| Knob | v1 | v2 | Score after |
|---|---|---|---|
| `hands_max` | 16 | **10** | 30,806 |
| `capacity_util` | 0.82 | **0.95** | 33,094 |
| `reserve_base` | 260 | **600** | 34,390 |
| `reserve_per_tile` | 9.0 | **0.0** | 36,928 |
| `feed_runway_days` | 7.0 | **12.0** | 42,864 |
| `max_herd` | 22 | **12** | **73,272** |
| `animal_cash_buffer` | 400 | **1200** | 73,407 |
| `target_cow` | 26 | **8** | 74,952 |
| `target_sheep` | 18 | **12** | 75,811 |
| `target_goose` | 6 | **12** | 75,954 |
| `feed_buffer_days` | 3.0 | **2.0** | 79,587 |
| `seed_lookahead` | 8 | **4** | 81,288 |

Baseline (v1's parameters) scored 24,358. Three findings are worth more than
the numbers:

### 1. A small, well-fed herd beats a large starving one (+$30k, the single biggest jump)

Halving `max_herd` from 22 to 12 nearly doubled the score. v1's parameters kept
buying livestock until cash ran out, and the herd then died in waves — every
death is permanent and takes the animal's whole remaining production with it.
Twelve animals that survive the season out-earn twenty-two that do not. The
same instinct shows up in `animal_cash_buffer` (400 → 1200) and
`feed_runway_days` (7 → 12): the search consistently paid for herd *survival*
over herd *size*.

### 2. Geese up, cows down — the opposite of the naive $/tile-day ranking

Base-price economics rank sheep ($266/tile-day) > cow ($240) > goose ($100).
The tuner went the other way: goose 6 → 12, cow 26 → 8. Two reasons, both in
[strategy.md](strategy.md):

* **Lead time.** A goose pays from day 4; a cow not until day 8. Over a 30-day
  season with a cash-starved opening, four extra productive days compound.
* **Glut.** Milk floors at only +76 units above I0 and wool at +59, while eggs
  sit on a log curve that mathematically never reaches the floor. A goose's
  revenue is *reliable*; a cow's headline rate assumes a milk price you destroy
  by selling into it.

The naive ranking was measuring the wrong thing. This is the clearest case in
the project of the search correcting the analysis.

### 3. Fewer hands, worked harder

`hands_max` 16 → 10 and `capacity_util` 0.82 → 0.95. Hands are cheap in
absolute terms, but the 11th through 16th cost $89–$987 each on the Fibonacci
curve, and early on that cash is worth more as seed and feed. Meanwhile raising
`capacity_util` means the crew is trusted to cover more tiles each — the
walking-overhead estimate baked into v1 was too pessimistic once the farm is
laid out with livestock nearest the shed.

`reserve_per_tile` going to 0 fits the same story: per-tile working capital was
double-counting the feed runway, which is the reserve that actually matters.

## Honest limitations

* **One pass, 3 seeds, 2 opponents.** Anything under ~$3k of score is inside
  the noise. Several knobs (`dump_day`, `feed_self_frac`, `liquid_target`)
  showed no measurable effect at all, which may mean they do not matter or may
  mean the search lacked resolution.
* **Over-fitting risk.** The tuner self-plays against a frozen v1. A real
  leaderboard field topping 2900 will be far more varied, and a parameter set
  sharpened against one opponent can be brittle against many. Treat $82k as
  "clearly stronger than v1", not as a leaderboard prediction.
* **Local optimum.** Coordinate descent cannot discover a change that requires
  two knobs to move together — for example the melon-rush opening described in
  [issues-and-improvements.md](issues-and-improvements.md) D, which needs
  `target_melon`, `land_min_day` and the reserve settings to move at once.
* **The search cannot fix logic.** Every large gain in this project came from a
  *bug*, not a parameter. Tuning added ~$30k; the four bugs listed in
  [agent-v1.md](agent-v1.md) were worth ~$52k between them.
