# Kaggriculture — strategy analysis

This is the reasoning the agents are built on. Numbers come from the
interpreter (see [environment-rules.md](environment-rules.md)) and from
instrumented matches (`src/kaggriculture/measure/analyze.py`).

## 1. The market is under-supplied, so this is a production race

The single most important empirical fact: **every match we have run ends with
market inventory 300–500 units *below* the 10,000 starting level on the premium
goods.** Prices therefore drift *upward* all game — a typical day-29 board
shows milk at $340 (base $160), strawberry at $300 (base $120), wool at $250
(base $200), wheat at $50 (base $25).

The town wants roughly $305k of produce per season and two competent agents
between them supply a fraction of it. Consequences:

* Clever *selling* is worth much less than more *producing*. Price-timing logic
  earns single-digit percentages; another twenty productive tiles earns tens of
  thousands.
* Glut-avoidance rules (don't dump 150 melons at once) matter only for the
  fragile goods — melon, milk, wool, strawberry — and only if you get big.
* Under-production means the *scarcity* side of the price curve is where you
  live, and that side is concave: the first units you sell are the dearest.
  Diversifying across products beats maximising one.

Both agents therefore optimise for throughput, and the market engine exists
mostly to avoid self-harm, not to extract alpha.

## 2. What a tile is worth

Ranking by dollars per tile per day at base prices (see the table in
environment-rules.md):

```
sheep 266 > cow 240 > melon 150 > goose 100 > carrot 35 > strawberry 28 > wheat 25 > tomato 18
```

At the *observed* end-game prices the spread widens further: a cow at milk-$340
is worth $510/tile-day, roughly twenty times a wheat tile.

But raw rate is not the whole story:

* **Melon has no shop demand at all.** Only the town centre eats it — about 140
  units for the whole season. Selling 150 melons at once walks the price from
  $250 down to ~$25 (its curve is quadratic in glut, floor at +158 units).
  Peak extractable revenue from melon is ≈$26k, so ~12 tiles is the sweet spot;
  a whole field of them is self-defeating.
* **Milk and wool floor at +76 and +59 units.** They are the two most valuable
  products *and* the two most fragile. Their saving grace is that three shops
  drink milk and the yarn store eats wool at double rate, so the town keeps
  draining the glut back out — but only at ~18/day and ~12/day.
* **Wheat and eggs are mathematically glut-proof** (log curves — the price
  never reaches the floor). They are the safe dumping ground for surplus
  capacity and the reason wheat doubles as animal feed and as a cash crop.

## 3. The compounding problem

Ranked by return *per dollar per day*, which is what matters while the bank is
the binding constraint:

| Investment | Outlay | Payback | Multiple | Compounded |
|---|---|---|---|---|
| Wheat seed | $10 | 4 days | 11× | **1.82×/day** |
| Carrot seed | $20 | 3 days | 5.5× | 1.77×/day |
| Melon seed | $80 | 10 days | 19.5× | 1.34×/day |
| Cow | $400 | 8 days to first milk | ~20× over the season | 1.14×/day |
| NE quadrant | $1,000 | immediate (25 tiles) | very large | — |

Wheat is the best *compounder* and cows are the best *terminal asset*. That is
the whole shape of the game: grow cash fast on short-cycle crops, then convert
it into livestock and land before the season runs out. Buying a herd on day 0
(v1's first draft did exactly this) leaves nothing to feed it with, and a
starved animal is gone permanently.

## 4. Labour is nearly free — and is the second constraint

Hiring the first ten hands costs **$143 for the day**; they deliver 230
extra unit-turns. Even at wheat prices that is a ~50× return. The Fibonacci
curve only starts to bite past ~13 hands ($609/day) and turns brutal at 16
($2,583/day).

What labour buys, per day:

* a crop tile costs ≈2.2 unit-turns/day (watering, plus amortised plant/harvest
  and walking)
* an animal costs ≈4.5 unit-turns/day (feed, care, periodic harvest, the walk
  to the shed for wheat, and fertilizer collection)

So a crew of 1 farmer + 10 hands ≈ 264 raw unit-turns ≈ 215 effective, which
supports roughly 60 crop tiles or 45 animals. **Planting more than the crew can
water is strictly negative value** — an unwatered plant becomes a weed in two
days and you then pay a `DIG` to get the tile back. Both v1 and v2 compute this
capacity explicitly and refuse to over-plant.

## 5. Feed is a hard constraint, not a soft cost

Each animal eats 1 wheat per day, and the wheat must be carried from the shed
by a unit. Two missed days and the animal is gone.

* One wheat *tile* yields ≈1 wheat/day, so it takes roughly one wheat tile per
  animal to be self-sufficient.
* Buying feed instead frees that tile for something better, but wheat's price
  climbs all season (the town drains ~30/day once the wheat shops are open),
  reaching $50–60 by day 25. At 20 animals that is $1,000+/day.
* Milk is worth ~$300/unit at the same moment, so buying feed is *always*
  correct on unit economics — provided the cash is there on the day.

This is why both v1 and v2 hold a **feed runway reserve**: cash that capital
purchases may not touch, sized as
`runway_days × (herd − own wheat supply) × wheat price`. Without it the agent
buys a herd it cannot feed and the entire herd dies around day 18. That failure
cost about $25k per match before it was fixed.

## 6. The endgame

Money in the bank is the only thing that scores. Produce in a farmer's hands at
the final bell scores nothing, and produce that reaches the shed on the last
day can still be sold only if there are turns left to sell it.

Concretely, on the last day the pipeline is harvest → carry → `DROP` at the
shed → `SELL`, which is three-plus turns. Making hauling outrank every other
job on day 29, and dropping all price floors from day 28, was worth **~$25k per
match** — the single largest improvement in v1's development, larger than any
portfolio change.

Related endgame rules both agents follow:

* stop planting a crop once `days_left < harvest_day + 1`
* stop buying livestock once `days_left < first_yield_day + 3`
* keep feeding mature animals to the very last day — their remaining value is
  proportional to `days_left`, not to `days_left − first_yield_day` (getting
  this wrong made the herd starve out in the final week)

## 7. What is deliberately *not* modelled

* **Opponent modelling.** Both farms are public, so denying the opponent a
  market is possible in principle — flooding wool to the floor before they
  harvest theirs, for instance. Under current production levels neither agent
  gets big enough for this to pay, so it is left as future work.
* **Search / rollout.** The 1-second turn budget is generous relative to the
  ~6 ms the planner takes, so a shallow lookahead is affordable. See
  [issues-and-improvements.md](issues-and-improvements.md).
