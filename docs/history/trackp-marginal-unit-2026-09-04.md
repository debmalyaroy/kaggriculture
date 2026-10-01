# Track P — the marginal-unit ledger: why a $500 sheep costs this planner $1,191

**The question, as handed over:** every production expansion this planner can
make is significantly negative on its own bank, yet
`.local/trackp_market/counterfactual.py` prices those same units, delivered
free, at +$40,367 (WOOL) / +$27,235 (STRAWBERRY) / +$22,626 (MILK) of margin.
*Why does a $500 sheep that the market pays ~$2,760 for cost this planner more
than $2,760 elsewhere?*

**The answer, in one line: the market does not pay $2,760 for THIS planner's
marginal sheep. It pays MINUS $80 a unit for the wool, because the opponent has
already filled the wool market and the engine's above-I0 wool curve is
quadratic.** The $2,760 came from a counterfactual run on seed 3 — the single
most wool-scarce draw in the set. Over 16 seeds our existing 5-sheep herd plus
v43's rigid 138 units already pin the shared wool inventory at the $1 price
floor in six of them.

Everything else in the ledger — cash, feed, labour, tiles — is real, is
measured below, and is second order to that one fact.

**And the fix the ledger names was built, measured, and does NOT clear the
bar.** A live-observation herd gate (buy sheep past one only if the town's
shop draw actually eats wool) is worth +$1,572 to +$3,363 of own bank, is
14-of-15 worlds on the gauntlet at p = 0.0010 — and is **not resolved against
`v43.0_bandit` out of sample (+948 margin, 8 of 13 worlds, p = 0.5811), raises
the OPPONENT's bank in both measurements (+624 / +431), and flips 0 of 80
cells.** That is the sixth large, real, reproducible economic gain in this lane
that converts to zero wins.

---

## 0. The instrument: a cash ledger that closes to the dollar

Every dollar that enters or leaves a seat's bank passes through exactly three
engine functions. `_commit_unit` (SELL / BUY_PRODUCT / BUY_SEED / BUY_ANIMAL),
`_do_hire` (the fibonacci fee) and `_do_buy_land`. `BUILD_PASTURE` and
`BUILD_COOP` are **free** — they cost a unit-op and nothing else. So

```
bank = 3000 + SELL - BUY_PRODUCT - BUY_SEED - BUY_ANIMAL - HIRE - BUY_LAND
```

is an exact identity. `.local/trackp_exec/ledger.py` monkeypatches those three
plus `_apply_unit_action`, `_daily_refresh_animals`, `_daily_refresh_plants`
and `_drop_inventories_to_shed`, and records per day, per seat:

| group | fields |
|---|---|
| cash | the six lines above, by day, at the price actually paid |
| revenue | units and dollars per product, so a realised $/unit falls out |
| labour | every unit-op by name (incl. PASS and the four moves), unit-turns |
| board | tiles by crop, animals by species, pens, weeds, fallow, quadrants |
| physical | **shed-overflow discards**, **animal escapes**, production events, units lost to the `max_held` cap, care bonus banked, fed/cared days, animal-days stuck in the shed, crops that died unwatered, units harvested into inventory |

**The identity closed to $0.00 on every one of the ~1,100 cells played for this
report**, and the instrument reproduces the earlier documents to the dollar
(seed 3 vs v43: bank 63,943, sell revenue 82,294, wool 76 units at $241.8 —
`docs/history/trackp-market-contest-2026-09-04.md` §1.2 exactly; gauntlet seeds 3-6
baseline own median 63,857 — `docs/history/trackp-base-economy-2026-09-03.md` §5.1
exactly).

Files: `.local/trackp_exec/{ledger,mledger,daytable,summary,woolcheck,bothseats}.py`.

**One trap worth recording.** `kaggle_environments` rebuilds the observation
objects every step, so a farm dict's `id()` is only meaningful *within one
interpreter call*. `_apply_unit_action` runs before `_process_market`, so the
first version of the probe attributed 3 of 807 harvests and silently reported
zero. The fix is to buffer and attribute inside `_process_market`. Any future
probe that keys on object identity across steps is wrong.

---

## 1. THE LEDGER — 16 sheep against the 5-sheep skeleton

Official vendored interpreter, `agents/v43.0_bandit.py`, **12 seeds
(11-16, 31-36) × both seats = 24 cells**, paired cell for cell. The genome is
`s_herd.json`'s `A2_sheep16` — the row the handover quotes as 53,915 → 43,427.
At 24 cells it is bigger: **dBank −15,333, dOpp +1,281, dMargin −16,614.**

12.88 extra sheep were actually bought (the schedule asks for 11; cash-gating
and the escape/rebuy loop move it).

### 1.1 The cash ledger (sums to dBank, exactly)

| line | total | per marginal sheep |
|---|---|---|
| BUY_ANIMAL | **−6,788** | −527 |
| BUY_PRODUCT (feed wheat, +91.9 units) | **−3,876** | −301 |
| BUY_SEED | +337 | +26 |
| HIRE | 0 | 0 |
| BUY_LAND | 0 | 0 |
| SELL (all products) | **−5,006** | −389 |
| **= dBank** | **−15,333** | **−1,191** |

### 1.2 The revenue line, product by product — this is the finding

| product | units | $ | realised price |
|---|---|---|---|
| **WOOL** | **+28.1** | **−2,238** | **$165.8 → $100.0** |
| STRAWBERRY | −31.1 | −2,630 | $168.7 → $202.3 |
| WHEAT | −51.8 | −1,095 | $29.7 → $30.7 |
| MILK | −5.0 | −117 | $108.9 → $112.0 |
| MELON | 0.0 | +22 | — |
| FERTILIZER | +65.9 | +1,052 | $46.4 → $38.2 |
| **total** | | **−5,006** | |

**We sold 28 more wool units and took $2,238 LESS money for wool.** The wool
line went from 77 units at $165.8 ($12,767) to 105 units at $100.0 ($10,510).
The marginal revenue of a wool unit at this planner's production level is
**−$80**.

That single row is the answer to the question. The sheep is not expensive; its
output is worthless. Everything else is the cost of producing it.

### 1.3 Labour — saturated, and the substitution is 1:1

Unit-turns are fixed at `(1 + hands) × 23`, so an extra animal op is exactly a
missing crop op. Measured:

| animal ops added | | crop ops lost | |
|---|---|---|---|
| FEED | +73.5 | WATER | −129.6 |
| CARE | +73.5 | HARVEST | −33.7 |
| COLLECT_FERTILIZER | +66.0 | PLANT | −27.2 |
| PICKUP | +21.7 | FERTILIZE | −4.9 |
| PLACE | +13.8 | DIG | −5.0 |
| BUILD_PASTURE | +11.0 | movement | −63.6 |
| **+259.5** | | **−264.0** | |

**PASS moved +0.9.** There is no slack: the crew is fully committed and the
herd is paid for out of watering. The board shows it — WHEAT tile-days −106.7,
STRAWBERRY tile-days −27.9, **weed tile-days +57.5**. The crop revenue that
buys is −$3,725 (wheat + strawberry), 24% of the loss.

### 1.4 Physical losses — the herd outgrows the crew

| | base (5 sheep) | delta at 16 sheep |
|---|---|---|
| sheep animal-days | 73.2 | **+68.3** (i.e. 5.3 standing days per extra sheep) |
| sheep production events | 17.9 | +8.2 |
| **sheep ESCAPES** (2 unfed days) | **0.0** | **+6.4** |
| cow escapes | 0.0 | +1.9 |
| animal-days stuck in the shed (bought, never placed) | 14.0 | +25.1 |
| STRAWBERRY discarded by dusk shed overflow | 0.8 | **+15.3** |
| units lost to the `max_held` cap | 12.9 (cows) | +0.7 |
| crops died unwatered | 16.5 | +2.7 |

**6.4 of the 12.88 sheep starve to death** — $3,200 of capital destroyed, 21%
of the loss — and the ones that survive stand for 5.3 days each, so they reach
one or two production events. A sheep placed on day 24 produces *nothing*
(first yield is 6 days out, the season ends at 29).

### 1.5 The four candidate explanations, ranked

The handover asked which term dominates. In order:

1. **Wool's marginal revenue is negative (−$80/unit).** Not a cost at all — the
   asset's *output* is worth less than nothing. §2 explains why with the
   engine's own price function.
2. **The animals never live long enough to produce.** 5.3 animal-days each,
   6.4 escapes. This is downstream of the cash curve (§3), not of tending: at
   the *base* herd there are **zero escapes**, 95% fed-days and 100%
   cared-days. The crew tends 5 sheep and 9 cows perfectly; it cannot tend 16
   and 9.
3. **Labour, 1:1 against watering** (−$3,725 of crop revenue). Real,
   mechanical, and it also explains the +57.5 weed tile-days.
4. **Feed wheat does get dearer with our own buying** — $32.2 → $35.4 on the
   average, and the 91.9 marginal units cost **$42.2 each** (wheat's below-I0
   branch is `sqrt` with target 0.80 on T = 400). Worth −$3,876, 25% of the
   loss, but it is a consequence of the herd size, not the reason it fails.
5. **The pasture build displaces nothing worth naming.** BUILD_PASTURE +11 and
   PLACE +13.8 out of +259.5 extra animal ops — 9%, and both are 1-op jobs.

---

## 2. WHY the marginal wool unit is worth −$80

`market_price` is a pure function of the SHARED inventory. WOOL's above-I0
branch is **quadratic**: `above_func "sq"`, `above_target 3.2`, `T = 105`, so
`amp = 3.2 × 200 / 105² = 0.0581` and

| inventory | $ | inventory | $ |
|---|---|---|---|
| I0 −200 | 245 | I0 +20 | 177 |
| I0 −100 | 240 | I0 +30 | 148 |
| I0 −20 | 226 | I0 +40 | 107 |
| **I0 +0** | **200** | I0 +50 | 55 |
| I0 +10 | 194 | **I0 +59** | **1 (floor)** |

The whole above-I0 range is **59 units wide.** Past that the price is $1, and
`_commit_unit`'s "sales at $1 do not increase market supply" rule pins the
inventory there for the rest of the season.

The marginal revenue of one more unit, given `n` units still to sell after it:

| inventory | price | MR (n=10) | MR (n=30) | MR (n=60) |
|---|---|---|---|---|
| I0 −20 | 226 | +226 | +226 | +226 |
| I0 +0 | 200 | +200 | +200 | +200 |
| I0 +20 | 177 | +147 | +87 | **−3** |
| I0 +30 | 148 | +108 | +28 | **−92** |
| I0 +40 | 107 | +57 | **−43** | **−193** |
| I0 +50 | 55 | **−5** | **−125** | **−305** |

### 2.1 Where the wool market actually sits — measured, 16 seeds

Shared WOOL inventory relative to I0 at the end of each day, our seat vs v43:

```
seed |  d8  d10  d12  d14  d16  d18  d20  d22  d24  d26  d28 | our $/u  units
   3 | -28  -46  -60  -70  -88 -102 -104 -141 -161 -177 -203 | $241.8    76
  13 | -64  -82  -96 -106 -124 -138 -140 -153 -153 -156 -157 | $242.0    58
  11 |  +8  +14  +12   +2  -16  -30  -32  -45  -37  -29  -39 | $225.6    80
  12 |  +8  +14  +12   +2  -16  -30  -32  -45  -41  -31  -37 | $225.8    78
  36 |  +8  +14  +12   +2  -16  -30  -32  -45  -37  -29  -27 | $225.6    80
  15 |  +8  +14  +12   +2  -16  -30  -32  -45  -45  -35  -33 | $226.1    74
  32 |  +8  +14  +24  +38  +44  +42  +40   +3  -16  -30  -64 | $204.0    79
  14 |  +8  +14  +24  +38  +20   +6   +4   -9   -5   +5   -1 | $205.3    78
  34 |  +8  +14  +24  +38  +20   +6   +4   -9   -5   +5   -1 | $205.3    78
  35 |  +8  +14  +24  +38  +20   +6   +4   -9  -10   +0   +2 | $208.9    73
   5 |  +8  +14  +24  +38  +44  +54  +59  +46  +51  +53  +43 | $ 75.6    79
   4 |  +8  +14  +24  +38  +44  +54  +59  +59  +59  +59  +59 | $ 36.3    78
   6 |  +8  +14  +24  +38  +44  +54  +59  +59  +59  +59  +59 | $ 39.2    72
  16 |  +8  +14  +24  +38  +44  +54  +59  +59  +59  +59  +59 | $ 38.2    74
  31 |  +8  +14  +24  +38  +44  +54  +59  +59  +59  +59  +59 | $ 36.3    78
  33 |  +8  +14  +24  +38  +44  +54  +59  +59  +59  +59  +59 | $ 35.0    81
```

**In 6 of 16 seeds the wool market is pinned at the $1 floor from day 20 and we
sell 72-81 units for $35-76 each.** In the rest it stays below I0 and we get
$205-242. The split is not luck of our play; it is the **shop draw**. A
`YARN_STORE` eats 2 wool every 4 steps — **12/day, ×2 the multiplier for a
single-product shop** — against the town centre's 1/day. One YARN_STORE is
worth 360 units of wool demand over a season; none is worth 30.

### 2.2 And the opponent fills it before we do

Both seats' sell books, base skeleton vs v43, 24 cells (mean per cell):

| product | our units | our $ | $/u | their units | their $ | $/u |
|---|---|---|---|---|---|---|
| WHEAT | 475.0 | 14,099 | 29.7 | 344.8 | 11,261 | 32.7 |
| CARROT | 0.0 | 0 | — | 64.0 | 3,470 | 54.2 |
| STRAWBERRY | 109.0 | 18,392 | 168.7 | 268.7 | 50,788 | 189.0 |
| MELON | 72.0 | 12,175 | 169.1 | 71.9 | 17,422 | 242.3 |
| MILK | 142.8 | 15,544 | 108.9 | 266.1 | 35,259 | 132.5 |
| **WOOL** | **76.8** | **12,730** | **165.8** | **138.0** | **22,807** | **165.3** |
| FERTILIZER | 180.2 | 8,358 | 46.4 | 328.1 | 20,300 | 61.9 |
| **total** | | **81,299** | | | **161,308** | |

**v43 puts 138 wool units on the market in every world** (it is a rigid tape —
`docs/history/trackp-market-contest-2026-09-04.md` §1.3 measured 138 to the unit
against three completely different opponents). We add 77. Combined supply is
215 against a town that eats 30 without a YARN_STORE. **The 59-unit window
above I0 is consumed by the opponent alone before our first sheep is placed.**

### 2.3 So the counterfactual was not wrong; it was seed 3

`docs/history/trackp-market-contest-2026-09-04.md` §4.1 priced +168 free WOOL units at
**+$40,367** and noted the market was "still 36 units short of I0". That is
true — **on seed 3**, which the table above shows is the most wool-scarce draw
in the set (−203 at season end, our realised price $241.8). The document said
so itself ("the wool prize is seed-dependent on the shop draw") and the number
was carried forward anyway.

The same arithmetic on seeds 4, 6, 16, 31, 33 prices the marginal wool unit at
approximately **zero** (the price is already $1) and the marginal *sheep* at
minus its purchase price, its feed and its labour.

**Rule for this lane: never price a marginal unit off one seed's
counterfactual when the product's price shape is quadratic and its demand is
drawn at random.**

---

## 3. The cash curve — why the sheep is always LATE

Per-day economy of the shipped skeleton, 24 cells (means):

```
day   money quad crew PASS walk WATR PLNT | cow shp shedA | wheatT strawT weed  sellRev
  0    3000  1.0  5.0   73   30   18   18 | 0.0 0.0   0.0 |    0.0    0.0  0.0        0
  2     725  2.0  4.0   17   59   31    3 | 1.0 1.0   0.0 |   27.9    0.0  0.1        0
  5     119  2.0  5.0    8   73   33    8 | 1.0 1.0   0.0 |   38.0    0.0  0.2    1,430
  8    1067  2.0 10.0   99   75   37   18 | 1.0 1.0   0.0 |   26.0    0.0  0.2    1,327
 11    4043  2.0 10.0  140   69   37    0 | 1.0 1.0   0.0 |   16.3    8.5  0.6      952
 12    1254  3.0  9.0   28  100   41   21 | 1.0 1.0   3.5 |   16.3    8.5  0.7      946
 13    1354  3.0 10.0   28  131   58   16 | 4.5 1.0   0.0 |   16.5   12.8  0.8    1,205
 17     943  3.0 10.0   19  123   51   10 | 8.0 2.8   0.5 |   28.3   17.7  0.5    2,571
 20   14014  3.0 10.0   27  100   44   16 | 9.0 5.0   0.0 |   31.8   15.4 11.1    5,745
 24   30201  3.0 10.0   24  102   41   16 | 9.0 5.0   0.0 |   30.3   15.2  9.5    5,194
 29   50531  3.0 10.0  191   32    0    0 | 9.0 5.0   0.0 |    0.0    5.6  1.2    4,479
```

Read the herd columns. **The genome asks for 6 cows by day 7 and 5 sheep by
day 15. The board has ONE cow and ONE sheep until day 12.** The purse is
$119-$2,377 from day 2 to day 10, and `land_reserve_lead: 0` correctly
withholds SW's $2,500 from the herd from day 5 until the buy lands on day 12 —
so no animal is affordable at all in that window (a sheep needs
$500 + $300 floor + $2,500 reserve = $3,300).

The consequence, from the engine's own production rule (sheep:
`first_yield_day` 6, `interval` 3):

| placed on day | production events left in the season |
|---|---|
| 2 | **8** |
| 13 | 5 |
| 17 | 3 |
| 20 | 2 |
| 24 | **0** |

The base's 5 sheep score exactly 19 production events, and 8 of them come from
the ONE sheep bought on day 0. (Measured: 17.9 events per cell.)

**And from day 19 the farm sits on idle cash it never redeploys** — $7,699 at
day 19 rising to $50,531 at day 29, i.e. essentially the entire final bank is
accumulated in the last ten days and spent on nothing. That is where the
marginal sheep *could* be bought, and it is exactly the window where a sheep is
worth 2 production events.

**So the sheep is unaffordable precisely while it would be valuable, and
affordable precisely once it is worthless.** That is the cash-timing term the
handover asked about, and it is structural: the money that would buy an early
sheep is the money that buys the second quadrant, and the land is worth
+19.6k (`docs/history/trackp-base-economy-2026-09-03.md` §3).

---

## 4. The fix the ledger names, built and measured

If the marginal wool unit is worth −$80 in the worlds with no YARN_STORE and
+$226 in the worlds with one, then **the herd should be sized to the observed
market, not to a fixed schedule** — and both inputs are in the live
observation. `town.unlocked_shops` gives the exact per-day consumption rate
(`_demand()` already reproduces the engine's shop table), and
`market.prices` gives the live price.

New genome knob `herd_gate` in `src/kaggriculture/trackp/build_econ_agent.py`, **default OFF
(empty)**, seat law intact (no tape, no prefix, live observation only):

```json
"herd_gate": {"SHEEP": {"min_rate": 6, "floor": 1}}
```

caps a species at `floor` unless the town's consumption rate for its product
and the live price both clear a bar. Verified: the skeleton at defaults plays a
full official episode to the **same bank to the dollar** as before the change
(seed 3: 93,732 / 159,813; seed 5: 85,090 / 147,925, vs `pub_v16rc5`).

### 4.1 Development, 12 seeds (11-16, 31-36) × both seats vs v43.0_bandit

Worlds are mirror-deduped (both seats of one seed are ONE observation); only
the worlds where the gate actually fires enter the sign test.

| variant | dBank | dOpp | dMargin | worlds better | p | wins |
|---|---|---|---|---|---|---|
| `SHEEP {min_rate 6, floor 1}` | +1,767 | **−414** | +2,180 | 5/5 | 0.0625 | 0 |
| `SHEEP {min_rate 6, floor 2}` | +1,515 | **−767** | +2,282 | 5/5 | 0.0625 | 0 |
| `SHEEP` + `COW {min_rate 6, floor 2}` | +2,543 | +271 | +2,272 | 6/6 | 0.0312 | 0 |
| `SHEEP {min_price 0.95, floor 1}` | +2,095 | +966 | +1,130 | 5/8 | 0.7266 | 0 |
| `COW {min_rate 6, floor 2}` alone | −18 | +79 | −97 | 1/2 | 1.0000 | 0 |
| `floor_animal 0` (a cash-priority control) | +2,829 | −2,038 | +4,867 | 9/12 | 0.1460 | 0 |

The mechanism does what the ledger predicted: it buys 1.5 fewer sheep, sells
17 fewer wool units, and the wool price goes **up** $165.8 → $192.4, so the
wool line barely moves while the $750 and the labour go into strawberry
(+5.8 units, +$1,418).

### 4.2 Confirmation on 20 seeds nobody had looked at (3-6, 21-24, 41-52)

40 cells each, vs `v43.0_bandit`, both seats, mirror-deduped:

| variant | dBank | **dOpp** | dMargin | worlds better | **p** | wins |
|---|---|---|---|---|---|---|
| `SHEEP {min_rate 6, floor 1}` | +1,572 | **+624** | +948 | 8/13 | **0.5811** | 0 |
| `SHEEP` + `COW` | +1,503 | **+430** | +1,073 | 8/13 | **0.5811** | 0 |
| `floor_animal 0` | **−3,029** | −2,021 | **−1,008** | 10/20 | 1.0000 | 0 |

`floor_animal 0` — the biggest dev-seed margin row — went **negative** out of
sample. The kind-window trap, paid for the third time in this lane.

### 4.3 The gauntlet, seeds 3-6, five opponents, 40 cells

| variant | dBank | **dOpp** | dMargin | worlds better | p | wins |
|---|---|---|---|---|---|---|
| `SHEEP {min_rate 6, floor 1}` | **+3,363** | **+431** | +2,932 | **14/15** | **0.0010** | 0 |

(Baseline own median 63,857 — identical to `docs/history/trackp-base-economy-2026-09-03.md`
§5.1, which is the instrument's cross-check.)

### 4.4 Verdict on the fix

**It is a correct decision and it is not a win.**

* It is significant on the gauntlet (14 of 15 worlds, p = 0.0010) and **not
  resolved on the gate the mission named** (v43 alone, 8 of 13, p = 0.5811).
  Same shape as `drop_daily` yesterday.
* **The opponent's bank goes UP in both out-of-sample measurements** (+624 vs
  v43, +431 on the gauntlet). Selling less wool raises the wool price, and the
  opponent sells 138 units of it to our 77 — so on the coordinate that decides
  these games, the fix is mildly *counterproductive*. This is the same
  mechanism `docs/history/trackp-base-economy-2026-09-03.md` §6.3 recorded: improving
  our own economy on this board grows the pie and hands the opponent a share.
* **It flips 0 of 80 cells.** Base 0 wins, variant 0 wins, every arm.

It is left in the genome, documented, **defaulted OFF**. Nothing was rebuilt:
`src/kaggriculture/trackp/compiled/main.py`, `rustengine/src/policy.rs` and
`submission.tar.gz` (sha256 `21a6a024…`) are untouched, so
`models/release_2026-09-03_trackp.json` stays valid and the fork/exec beacon
experiment the live ship exists for is not muddied.

---

## 5. Blunt — the go/no-go this feeds

The handover asked for a clean answer on whether the closed-loop lane survives
to the freeze. Here it is.

**The executor diagnosis is now closed with the same finality as the other
four.** The marginal sheep does not cost $2,760 elsewhere. It returns **minus
$80 a unit** of wool, because:

1. the engine's above-I0 wool curve is quadratic and only 59 units wide;
2. the town eats wool only if the shop draw gave a YARN_STORE, which happens in
   about half of worlds;
3. `v43.0_bandit` alone supplies 138 units in **every** world, filling that
   window before our herd exists;
4. and the herd cannot be bought early anyway — the purse is $119-$2,377 from
   day 2 to day 11 because the second quadrant (worth more) has the money, so
   the animals arrive on days 13-20 and get 2-5 production events instead of 8.

**There is no executor fix behind this.** The `_plan` day ledger is not leaking:
zero escapes at the base herd, 95% fed-days, 100% cared-days, the cash identity
closes to the dollar, and the crew's ops go 1:1 into the highest-priority work
available. The planner is executing correctly against a market that will not
pay for more of what it makes.

The one lever the ledger did name was built and measured in full. It improves
our own bank by $1,572-$3,363, it is significant on the gauntlet, and it
**raises the opponent's bank and wins nothing**.

That makes this the **sixth** distinct, well-powered, negative result in this
lane: per-turn search (−14,755 own bank, p = 0.0064), the base economy (+45%
own bank, 0 of 120 cells), joint liquidation (not resolved at 96 cells), a
bigger herd (significantly negative), market contest (a real −9,520 on the
opponent that flips 0 of 48 cells, with a computed ceiling of −21,838 given
FREE supply), and now the marginal-unit ledger (the asset's marginal revenue is
negative, and the correct fix moves the opponent's bank the wrong way).

**Recommendation: NO-GO on the closed-loop Track-P seat for the freeze.**
`docs/history/plan-2800-2026-09-03.md`'s contingency — two bandit-lane agents from
different day-3 opening families — is the option that remains. The live
evidence agrees independently: submission 55992307 banks a median 86,799 (the
ladder median) against opponents rated 600-720 and holds them to 87,583, while
`v43.1_bandit` in the same band banks 118,432 and holds them to 43,938. **The
economy is competent; the pairwise edge is zero, and six separate attempts to
create one have now failed.**

Ship nothing from this pass.

---

## 6. Gates

| gate | result |
|---|---|
| `python tests/test_trackp.py` | **65 passed, 0 failed** |
| `python tests/test_rust_engine.py` | **6 episodes bit-identical, 0 diverged** |
| skeleton at new defaults vs the shipped one, full official episodes | **same bank to the dollar** — 93,732 / 159,813 (seed 3), 85,090 / 147,925 (seed 5), vs `pub_v16rc5` |
| `submission.tar.gz` sha256 | **`21a6a02469202041d01ad6c30a9f13087fe02793d616fb02651d9bbdf4c1fab9`** — unchanged, matches `models/release_2026-09-03_trackp.json` |
| `policy.rs` / `compiled/main.py` | **untouched** |
| cash identity on every measured cell | **closes to $0.00** |
| anything submitted or pushed | **no** |

### 6.1 A documentation bug found in passing (not fixed, no behaviour change)

`skeleton_genome()` documents `land_reserve_lead: -1` as "off". The code reads
`d >= ld - int(g.get("land_reserve_lead", 6))`, so −1 means the reserve starts
**one day later**, not off. A variant built with `-1` measured **byte-identical
to the baseline on all 24 cells**, which is how it was found. The shipped value
is `0` and is unaffected; anyone using −1 expecting "off" will get the reserve.

---

## Appendix — reproducing every number

```powershell
# the day-by-day economy that shows the herd frozen to day 12 and $50k idle at 29
python .local\trackp_exec\daytable.py .local\trackp_exec\base.py

# season totals + physical losses (escapes, discards, capped yield, unwatered)
python .local\trackp_exec\summary.py .local\trackp_exec\base.py

# THE LEDGER: paired P&L per genome variant, cash lines sum to dBank
python .local\trackp_exec\mledger.py .local\trackp_exec\s_diag.json `
  --seeds 11,12,13,14,15,16,31,32,33,34,35,36 --workers 8

# where the wool market sits, seed by seat, and the marginal-revenue table
python .local\trackp_exec\woolcheck.py .local\trackp_exec\base.py `
  --seeds 3,4,5,6,11,12,13,14,15,16,31,32,33,34,35,36

# who supplies the animal-product markets
python .local\trackp_exec\bothseats.py .local\trackp_exec\base.py

# the herd gate: development, then the out-of-sample confirmation
python .local\trackp_exec\mledger.py .local\trackp_exec\s_gate.json `
  --seeds 11,12,13,14,15,16,31,32,33,34,35,36 --workers 8
python .local\trackp_exec\mledger.py .local\trackp_exec\s_confirm.json `
  --seeds 3,4,5,6,21,22,23,24,41,42,43,44,45,46,47,48,49,50,51,52 --workers 8
python .local\trackp_exec\mledger.py .local\trackp_exec\s_gaunt.json --seeds 3,4,5,6 `
  --vs data\gauntlet\pub_v16rc5.py data\gauntlet\kaito_v48.py `
      data\gauntlet\pub_rayk_c94.py agents\v43.0_bandit.py agents\v42.1_trackp.py
```

**Nothing was submitted and no kernel was pushed.**
