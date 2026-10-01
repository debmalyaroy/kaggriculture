# Loss analysis — live ladder episodes

First submission (`main.py` = v2_tuned, submitted 2026-08-04). Record after
~35 minutes on the ladder: **5 wins, 1 loss**, rating **600 → 748**.

| when | opponent | result |
|---|---|---|
| 3m  | Uriel Johnson | WIN |
| 7m  | **shi_koyo** | **LOSS** |
| 11m | Yilan Zhu | WIN |
| 22m | momoon | WIN |
| 26m | Dr.Abdulbaset Musleh | WIN |
| 31m | LeiYang | WIN |
| 33m | (self) | validation episode |

## The loss — episode 89965738, seed 406294494

```
[Win]  shi_koyo  820 (+7)   $116,318
[Loss] Debmalya  754 (-44)  $ 69,811      gap $46,507 (-40%)
```

### Final market prices (both players' selling combined)

| product | base | final | read |
|---|---|---|---|
| Strawberry | 120 | **272** | wildly under-supplied — the town was starving for it |
| Milk | 160 | **267** | under-supplied |
| Wool | 200 | **242** | under-supplied |
| Tomato | 60 | **103** | under-supplied |
| Egg | 50 | 66 | mildly under-supplied |
| Wheat | 25 | 51 | drained by shops, as expected |
| Carrot | 35 | 42 | near base |
| **Melon** | **250** | **189** | **over-supplied — the only product below base** |

### What actually went wrong

1. **We were out-produced, not out-traded.** $116k vs $70k with the same rules
   and the same 30 days. Every premium good except melon ended *above* base,
   so there was unmet demand the whole game — this was a production race and we
   ran slower.

2. **Melon was the one product in glut, and we grow 12 tiles of it.** Melon has
   no shop demand at all (only the town centre, ~140 units/season) and its
   price curve is quadratic in glut. Selling into a falling melon market while
   strawberry sat at $272 is a straight misallocation.

3. **Two cows finished the game unplaced in the shed** — $800 of capital bought
   and never deployed, plus all the milk they would have produced. Livestock
   purchase and pen construction can get out of step.

4. **Nine fertilizer finished unused.** Fertilizer is collected free from
   livestock and doubles every scheduled production on ongoing crops
   (strawberry, tomato) — exactly the two goods that ended most under-supplied.

5. **Their animals sat in one contiguous block; ours were scattered.** Matches
   the measurement that 50.6% of our unit-turns were spent walking.

### Fixes applied in v3

| finding | fix | measured |
|---|---|---|
| 50.6% of turns walking | `travel_weight` 5.0 + Voronoi zoning | movement → 44.6%, productive 41.9% → 46.5% |
| fertilizer unused | `fert_weight` 2.5 | +$5,784 margin vs v2 |
| stranded livestock | rescue pens for shed animals | see agent-v3 |
| melon glut / strawberry shortage | portfolio rebalance | pending |

Overall v3 vs v2: **81% win rate over 16 matches**, both seats, three seed sets.
