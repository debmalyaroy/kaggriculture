# Track P — contesting the market: the mechanism, the ceiling, and why it fails

**Mission:** make the closed-loop Track-P seat beat `agents/v43.0_bandit.py`
head to head. The diagnosis handed over
(`docs/history/trackp-compiled-2026-09-03.md` §0.4) was that our own farm output has
caught up with the live seat and the entire remaining gap is **market share** —
the opponent banks ~135,000 against us and ~79,000 against a seat that
contests, in the same world. The instruction was to build the contest, judge on
the OPPONENT's bank first, and to say so plainly if it does not move.

**Verdict: the diagnosis is CORRECT, the lever is REAL and it MOVED — and it
still does not come close to the bar. Nothing here should ship.** Ten
market-contest mechanisms were built and measured on the official engine, 42
configurations in all. **Zero wins in every one of them.**

**The opponent's bank DID move, and that is the honest headline.** One
mechanism — same-day marketing (`drop_daily`) — takes **9,520 off the
opponent's bank** across 24 out-of-sample worlds against all five gauntlet
opponents (median 135,090 → 116,206) while our own median falls only
62,644 → 60,303, improving the margin by **+5,334, better on
17 of 24 worlds, p = 0.0639** (on the 20-world gauntlet alone: +7,127, 16 of
20, **p = 0.0118**, opponent 135,602 → 116,206). Every one of the five
opponents banks less against it. **And it flips 0 of 48 cells**, because the
gap is 66,000 and the effect is 5,300. On the primary gate — `v43.0_bandit`
alone over 14 worlds — it is **not resolved** (+2,947, 9 of 14, p = 0.4240),
and on the four reserved seeds it went the *wrong* way. §3 has the split.

Everything else that moved the opponent's bank down moved ours down at least
as far.

And the honest ceiling is now *computed*, not guessed: with an exact
re-simulation of the market against v43's own (fixed) order tape, a **free,
costless** doubling of our STRAWBERRY and MILK supply — 168 extra units of each,
conjured from nothing — moves the margin from −80,553 to **−21,838 on seed 3.
Still a loss.** Market contest alone cannot beat this opponent from this
economy. `kaito_v48`, the one gauntlet agent that ever beats v43, already
executes that flood and measures −17,817.

---

## 0. What was measured first: nobody beats v43

Before building anything, the bar was calibrated. Official vendored
interpreter, seeds 3,4,5,6, both seats, 8 cells each, `.local/econ/pyduel.py`:

| our seat | W-D-L | our median bank | **v43's median bank** |
|---|---|---|---|
| `data/gauntlet/kaito_v48.py` | **2-0-6** | 51,950 | **64,510** |
| `data/gauntlet/pub_v16rc5.py` | 0-0-8 | 64,477 | 79,808 |
| `data/gauntlet/pub_rayk_c94.py` | 0-0-8 | 70,566 | 89,812 |
| **trackp compiled (budget 0)** | 0-0-8 | 55,700 | 124,800 |
| `agents/v42.1_trackp.py` | 0-0-8 | 0 | 138,301 |

Two things fall out of this table and they set up everything below.

**(a) The ranking by wins is EXACTLY the ranking by the opponent's bank, and it
is not the ranking by our own bank.** `kaito_v48` banks the *least* of the four
real agents (51,950 — less than we do) and is the only one that ever wins,
because it is the only one that holds v43 under 70k. This is the mission's
thesis, confirmed on the primary gate's own instrument.

**(b) Our own bank is already competitive with the field.** 55,700 against
kaito's 51,950. The farm is not the thing losing these games.

The trackp row is the budget-0 skeleton rendered as a plain Python agent by
`src/kaggriculture/trackp/build_econ_agent.py --genome skeleton`, and it reproduces the
shipped artefact's §0.4 numbers **to the dollar** (55,700 own / 124,800
opponent, seeds 3-6, both seats). That equivalence — asserted by
`tests/test_compiled_agent.py` and re-checked here — is what makes a
one-minute sweep a legitimate measurement of the compiled seat.

---

## 1. The instrument: who sold what, when, and at what price

Three new read-only probes monkeypatch the vendored interpreter's
`_commit_unit` / `_process_market` and log every committed market unit;
a fourth re-simulates the market alone, and a fifth is the sweeper.
Nothing about the agent or the engine is changed by any of them.

| file | what it answers |
|---|---|
| `.local/trackp_market/probe.py` | per-player revenue and units by product; per-day revenue beside the market inventory AND price of all nine products; the SELL order-**slot** histogram |
| `.local/trackp_market/impact.py` | realised price per unit vs the price that stood at the START of that market step — i.e. how much each seat walked its own price down |
| `.local/trackp_market/supply.py` | per product: units supplied by each seat, units the town consumed, and **the day the shared inventory crossed I0** |
| `.local/trackp_market/counterfactual.py` | an exact re-simulation of the market alone, with v43's order tape held fixed and OUR supply varied (§4) |
| `.local/trackp_market/msweep.py` | genome sweep ranked on **MARGIN and the opponent's bank**, mirror-deduped to worlds, with an exact paired sign test |

`msweep.py` is the methodological point. **Every previous sweep in this lane
(`.local/econ/sweep.py`, `genome_ga`) ranked candidates by median OWN bank** —
the one coordinate the integration report says we do not lose on. This one
ranks on `own − opp` and prints the opponent's bank next to ours.

### 1.1 The market is in SCARCITY, not glut — the brief's framing was wrong

The handover said "shared-inventory lockstep pricing means buying more raises
prices for whoever sells most." The measured board says something else. Seed 3,
our seat vs `v43.0_bandit`, market inventory relative to I0 = 10,000 with the
price beside it, at the last step of each day:

```
day        WHEAT      CARROT       TOMATO   STRAWBERRY       MELON        MILK        WOOL   FERTILIZER
 10      +57/$22    -107/$43     -11/$61      -23/$160    +49/$226     +7/$145    -46/$233     +66/$87
 20      +34/$22    -273/$56     -21/$63      -61/$186   +123/$99     -20/$199   -104/$240    +196/$61
 29     +235/$20    -506/$79     -30/$64      +49/$26    +114/$120     +50/$55   -212/$246    +436/$13
```

For most of the season **STRAWBERRY, MILK, WOOL, CARROT, EGG and TOMATO sit
BELOW I0 — the price is ABOVE base and rising**, because the town eats more
than both farms together supply. WOOL ends 212 units short of I0 at $246
against a $200 base; CARROT ends 506 short at $79 against $35. Only WHEAT,
MELON and FERTILIZER ever glut.

That inverts the plan. In a scarce market you cannot suppress anyone by
selling: the town simply absorbs it. **There is nothing to deny until the
combined supply crosses I0.**

### 1.2 Where v43's 135,000 actually comes from

Same episode, seed 3, `impact.py`:

| product | us: units / revenue / avg | v43: units / revenue / avg |
|---|---|---|
| WHEAT | 473 / 10,480 / $22.2 | 345 / 7,658 / $22.2 |
| CARROT | 0 / 0 | 64 / 4,904 / $76.6 |
| STRAWBERRY | 142 / 14,079 / $99.1 | 269 / **38,970** / $144.9 |
| MELON | 72 / 12,163 / $168.9 | 72 / **17,440** / $242.2 |
| MILK | 138 / 19,104 / $138.4 | 266 / **40,737** / $153.1 |
| WOOL | 76 / 18,380 / $241.8 | 138 / 32,867 / $238.2 |
| FERTILIZER | 170 / 8,088 / $47.6 | 328 / 20,494 / $62.5 |
| **TOTAL** | **1,071 / 82,294 / $76.8** | **1,482 / 163,070 / $110.0** |

They sell **38% more units at a 43% better price**. The price half is not
mostly within-step walk-down: measured against the price standing at the start
of each market step, we lose 7% to our own dumping and they lose 3%. The price
gap is *when* and *what*, not *how*.

### 1.3 The same opponent, against a seat that contests

The decisive comparison. Identical seed, identical opponent file, only our seat
changes. v43's own volumes barely move across all three worlds (336-345 WHEAT,
**64 CARROT, 72 MELON and 138 WOOL to the unit**, 253-269 STRAWBERRY, 266-268
MILK) — **it is a rigid schedule and essentially only its prices move.**

| v43's realised price | vs our seat | vs `pub_v16rc5` | vs `kaito_v48` |
|---|---|---|---|
| STRAWBERRY | $144.9 | $21.3 | **$15.9** |
| MILK | $153.1 | $77.2 | **$31.0** |
| WOOL | $238.2 | $243.9 | $219.7 |
| WHEAT | $22.2 | $37.9 | $43.2 |
| **v43's total sell revenue** | **163,070** | 108,625 | **91,374** |
| **v43's bank** | **135,030** | 79,808 | **62,791** |

`kaito_v48` sells **311 STRAWBERRY for $5,726** (an average of $18.4 — it earns
almost nothing on them) and **264 MILK for $2,308** ($8.70). It ends up worth
*less* than we are on those two products — our STRAWBERRY+MILK revenue is
33,183 against its 8,034 — and it takes **$67,000** off v43 for it. On this
seed it still loses (44,974 to 62,791); over the four seeds that trade is what
turns 0-8 into 2-6.

That is the whole mechanism, and it is a *volume* mechanism: STRAWBERRY and
MILK both price linearly above I0 (amp $1.92 and $2.10 per unit), so ~60 and
~76 units past the crossing take them from $150-200 to the $1 floor.

Crossing day, from `supply.py` and the per-day inventory trace:

| world | STRAWBERRY supplied (both) | town consumed | first day inventory ends above I0 | v43's straw revenue |
|---|---|---|---|---|
| ours | 142 + 269 = 411 | 362 | **day 24** | 38,970 |
| pub_v16rc5 | 286 + 253 = 539 | 485 | day 16 | 5,396 |
| kaito_v48 | 311 + 267 = 578 | ~516 | day 16 | 4,234 |

MILK is the same story told differently: in our world the shared inventory
*oscillates* around I0 from day 8 to day 23 (−20 to +10, price $145-199) and
only settles above it at day 24; in kaito's world it is pinned at **+76 from
day 15**, which is exactly the point the linear glut branch reaches the $1
floor, and it stays there for the rest of the season.

Eight days earlier over the crossing is worth ~$34,000 of the opponent's bank
on strawberry alone. **We are short roughly 150 STRAWBERRY and 130 MILK units
of being able to do it.**

---

## 2. What was built — ten mechanisms, all closed-loop, all defaulted OFF

Everything is a genome knob in `src/kaggriculture/trackp/build_econ_agent.py`, decided from
the live observation. No tape, no route, no prefix, no embedded schedule. The
skeleton with every new knob at its default **banks identically to the
dollar** to the pre-change one on a full official episode, so the shipped
policy is unchanged unless a knob is deliberately turned on.

| knob | mechanism | why it should have worked |
|---|---|---|
| `demand_crops` | tile target for CARROT/TOMATO sized from the town's REAL consumption rate, read off `town.unlocked_shops` (`_demand()` reproduces the engine's shop table exactly) | CARROT and TOMATO are the two products *nobody* supplies; CARROT ran a 506-unit deficit at $79 |
| `sell_order: 1` | biggest-dollar SELL into the earliest market slot | slot index is a hard price ladder in `_process_market`; v43 puts 84% of its sells in slot 0, we spread ours over slots 0-9 |
| `sell_order: 2` + `sell_prio` | explicit race order, MELON and FERTILIZER first | those two have (near) zero town demand, so every unit ever sold stays in inventory — a one-shot pool split by who sells first, and we lose both ($169 vs $242, $48 vs $63) |
| `hour0_sells` | reserve market slots at hour 0 for sells, paid for by hiring fewer hands | `farm["hands"]` is cleared nightly, so the crew is re-hired from scratch every dawn; from day 8 the ramp asks for ten hands and **ten HIRE orders fill all ten of hour 0's market slots, so this seat cannot place a sell order on the first turn of any day for two thirds of the season** |
| `drop_daily` | a loaded unit walks back to the shed and DROPs at the end of its tour | harvest reaches the shed only at dusk, so we sell *yesterday's* crop; v43's melons reached the market **during** day 10, which is how it takes the top of the melon curve |
| `hold_price` | do not sell a product below $X/unit unless under shed pressure or liquidating | selling at the $1 floor banks $1 and, by the engine's own rule, does not even raise the inventory |
| `window.MELON [0,2]` | plant melon day 0 so it first-yields day 10 and enters the race | MELON is in no shop; the town eats 1/day; the ~158-unit pool above I0 is worth ~$26,000 and is decided on one day |
| herd reallocation | sheep-heavy / cow-heavy / goose ramps | §4 shows +168 units of WOOL is worth +$40,367 of margin and +168 MILK is worth +$22,626 |
| `floor_straw`, `window.STRAWBERRY` | more strawberry, earlier | the crossing-day analysis above |
| max dump (`sell_cap` off, `fert_keep 0`) | no reserves at all | maximum inventory pressure |

---

## 3. The measurements

Official vendored interpreter, `agents/v43.0_bandit.py`, **dev seeds 11-16**,
both seats — 12 cells = **6 worlds** (both seats of one seed are ONE world;
`msweep.py` dedupes before every p-value). Report seeds 3-6 and the reserved
set 21-24 were kept clean during development. `dMargin` is the paired
per-world change in `own − opp`; `oppDown` is the paired mean fall in the
opponent's bank (positive = we suppressed them).

Baseline: **0-0-12, own 53,915, opp 123,629, margin −69,684.** The 25 most
informative of the 42 configurations run:

| variant | own | opp | margin | dMargin | better | p | oppDown |
|---|---|---|---|---|---|---|---|
| **`drop_daily` (same-day marketing)** — *did not replicate, §3.1* | 48,781 | 112,593 | **−62,922** | **+5,934** | **6/6** | **0.0312** | **+7,378** |
| `drop_daily` + `melon0` + hour0 + prio | 38,823 | 103,937 | −65,114 | **+6,186** | 5/6 | 0.2188 | **+21,892** |
| `drop_daily` + `melon0` | 43,059 | 111,252 | −68,193 | +4,618 | 4/6 | 0.6875 | +19,190 |
| `melon0` (day-0 melon) | 47,491 | 118,296 | −70,804 | **+3,977** | 4/6 | 0.6875 | **+17,191** |
| `melon0` + hour0 + prio | 38,956 | 110,508 | −68,844 | **+4,272** | 4/6 | 0.6875 | **+18,058** |
| `melon0` + maxdump + straw300 | 46,731 | 114,101 | −67,370 | +1,080 | 2/6 | 0.6875 | +3,155 |
| `floor_straw 300` | 63,966 | 135,245 | −70,214 | +1,166 | 3/6 | 1.0000 | −5,367 |
| maxdump | 58,998 | 125,876 | −68,688 | +940 | 4/6 | 0.6875 | +9,863 |
| `sell_order 1` (slot) | 53,891 | 123,654 | −69,712 | −29 | 1/6 | 0.2188 | −15 |
| `hold_price` | 53,915 | 123,629 | −69,684 | −6 | 0/2 | 0.5000 | −3 |
| `hour0_sells 1` | 57,256 | 131,348 | −73,425 | −3,786 | 1/6 | 0.2188 | −3,700 |
| `hour0_sells 2` | 52,908 | 129,652 | −78,694 | −10,755 | 0/6 | **0.0312** | −9,776 |
| carrot 20 tiles | 53,797 | 124,061 | −67,844 | −4,330 | 1/6 | 0.2188 | −466 |
| carrot 40 tiles | 51,965 | 157,026 | −106,492 | −30,280 | 0/6 | **0.0312** | −31,646 |
| strawberry wide window | 61,197 | 138,562 | −79,951 | −11,851 | 1/6 | 0.2188 | −18,008 |
| **18 cows** (milk flood) | **27,098** | 108,808 | −81,148 | −9,022 | 1/6 | 0.2188 | **+11,174** |
| 18 cows + more strawberry | 39,281 | 139,309 | −97,824 | −26,907 | 0/6 | **0.0312** | −15,098 |
| full flood (cows + straw + melon0 + dump + slot) | 45,340 | 119,048 | −76,812 | −4,433 | 1/6 | 0.2188 | +6,578 |
| 10 sheep | 59,858 | 137,157 | −77,300 | −3,979 | 2/6 | 1.0000 | −3,646 |
| 16 sheep | 43,427 | 129,222 | −85,796 | −11,725 | 0/6 | 0.1250 | −475 |
| 24 sheep | 31,975 | 123,297 | −92,562 | −17,000 | 1/6 | 0.6250 | +2,992 |
| 9 sheep by day 9 | 49,188 | 120,014 | −70,827 | −3,632 | 3/6 | 1.0000 | +61 |
| 13 sheep by d11, cows cut to 5 | 54,549 | 121,180 | −73,682 | −4,676 | 2/6 | 0.6875 | −320 |
| 13 sheep by d11, cows kept | 41,478 | 128,085 | −80,386 | −11,246 | 2/6 | 0.6875 | +2,121 |
| 9 sheep + `floor_animal 100` | 46,991 | 138,245 | −87,340 | −18,564 | 0/6 | **0.0312** | −12,577 |
| 8 geese | 44,284 | 148,819 | −101,508 | −32,927 | 0/6 | **0.0312** | −22,179 |
| 16 geese | 20,764 | 112,586 | −90,824 | −27,207 | 0/6 | **0.0312** | +5,806 |
| 24 geese | 11,997 | 112,371 | −98,864 | −36,673 | 0/6 | **0.0312** | −36,673 |

**Wins: 0 of 12, in every single row.**

### 3.0 The primary gate, stated the way the mission asked for it

Head-to-head vs `agents/v43.0_bandit.py`, official vendored interpreter, both
seats, **14 seeds** (3-6, 11-16, 21-24) = 28 cells = 14 worlds:

| seat | W-L | median margin | **opponent's median bank** | our median bank |
|---|---|---|---|---|
| trackp skeleton (the shipped policy, budget 0) | **0-28** | −66,270 | **116,284** | 55,003 |
| + `drop_daily`, the best surviving candidate | **0-28** | −66,716 | 110,898 | 47,063 |

Paired over the 14 worlds: **dMargin +2,947, better on 9, p = 0.4240; the
opponent's bank falls 4,495 on the mean. Not resolved, and no cell flips.**

Secondary gate — the integration gauntlet (`pub_v16rc5`, `kaito_v48`,
`pub_rayk_c94`, `v43.0_bandit`, `v42.1_trackp`), seeds 3-6, both seats,
40 cells = 20 worlds:

| seat | W-L | median margin | **opponent's median bank** | our median bank |
|---|---|---|---|---|
| trackp skeleton | **0-40** | −66,072 | **135,602** | 63,857 |
| + `drop_daily` | **0-40** | −56,797 | **116,206** | 60,303 |

Paired: **dMargin +7,127, better on 16 of 20 worlds, p = 0.0118, opponent's
bank down 12,441.** No cell flips. §3.1 has the full replication record and
why this is suggestive rather than established.

### 3.1 The one significant mechanism, and its full replication record

`drop_daily` — same-day marketing — was the only mechanism to reach
significance on the dev seeds: **+5,934 of margin, better on 6 of 6 worlds,
p = 0.0312.** Per the seed-hygiene rule it was then re-measured on the
integration report's own seeds (3-6), on a set reserved before any of this
work started (21-24), and on the whole five-opponent gauntlet. All of it:

| measurement | worlds | our bank | opp bank | dMargin | better | p | oppDown |
|---|---|---|---|---|---|---|---|
| v43 only, dev 11-16 (**tuned on**) | 6 | 48,781 | 112,593 | +5,934 | 6/6 | 0.0312 | +7,378 |
| v43 only, report 3-6 | 4 | 45,712 | 110,898 | +5,043 | 2/4 | 1.0000 | +9,750 |
| v43 only, **HELD-OUT 21-24** | 4 | 57,286 | 125,432 | **−3,629** | **1/4** | 0.6250 | **−5,084** |
| **v43 only, POOLED (the primary gate)** | **14** | 47,063 | 110,898 | **+2,947** | **9/14** | **0.4240** | +4,495 |
| **5-opponent gauntlet, seeds 3-6** | **20** | 60,303 | **116,206** | **+7,127** | **16/20** | **0.0118** | **+12,441** |
| **all non-development evidence** | **24** | 60,303 | **116,206** | **+5,334** | **17/24** | **0.0639** | **+9,520** |
| everything incl. the dev seeds | 30 | 54,787 | 116,206 | +5,454 | 23/30 | 0.0052 | +9,092 |

Per opponent on the gauntlet (4 worlds each), the opponent's bank falls in
**every** row:

| opponent | dMargin | better | oppDown |
|---|---|---|---|
| `pub_v16rc5` | +11,546 | 4/4 | **+20,637** |
| `v42.1_trackp` | +8,393 | 4/4 | +14,300 |
| `kaito_v48` | +8,506 | 3/4 | +9,841 |
| `v43.0_bandit` | +5,043 | 2/4 | +9,750 |
| `pub_rayk_c94` | +2,146 | 3/4 | +7,677 |

**Read this carefully, because it cuts both ways.**

* The effect is **real and consistent in direction**: five out of five
  opponents bank less, the pooled out-of-sample sign test is 17 of 24, and the
  mechanism is a plain engine fact (harvest reaches the shed only at dusk, so
  this seat sells yesterday's crop; v43's melons reach the market *during* day
  10).
* It is **not resolved on the gate the mission actually named.** Against
  `v43.0_bandit` alone over 14 worlds it is +2,947 at p = 0.4240, and on the
  four seeds nobody had ever looked at it went **negative**, with the
  opponent's bank going UP 5,084.
* `drop_daily` was **selected out of 42 configurations on the dev seeds**, so
  a p = 0.0639 out-of-sample replication is suggestive, not established.
  Quoting the 6/6 dev row alone would have been the exact mistake
  `docs/history/trackp-base-economy-2026-09-03.md` §5.1 records.
* **It flips 0 of 48 cells.** A 9,520 dent in a 135,090 opponent bank is 7% of
  their bank and 8% of the gap.
* Its two combinations (`+melon0`, `+melon0+hour0+prio`) go *negative* on the
  held-out set (−13,843 and −14,144) and are null on the gauntlet (+331 and
  +1,483). Only the bare mechanism survives.

Read the shape of the rest:

* **The only two mechanisms that reliably lower the opponent's bank are the
  day-0 melon race (+17-18k) and flooding milk with more cows (+11k).** Both
  cost us more than they cost them.
* **Every production expansion is significantly negative.** Sheep, cows and
  geese all collapse our own bank — geese from 53,915 to 11,997. The
  bottleneck is not the market; it is that this executor cannot add an asset
  without paying more in cash, tiles and labour than the asset returns.
* **The slot mechanisms are null.** `sell_order` moved the margin by −29 and
  `hold_price` by −6. They are correct descriptions of the engine and they buy
  nothing, which §4 independently confirms.
* **Nothing is significant in our favour.** The best row is +4,272 at
  p = 0.6875 against a −69,684 gap.

---

## 4. The ceiling, computed rather than swept

Because v43 is a rigid schedule — identical volumes across three completely
different opponents — its sell orders can be lifted from one real episode and
held fixed while the market is re-simulated exactly. `market_price` and
`_town_consume` are pure functions of inventory and the unlocked-shop list, so
`counterfactual.py` reproduces the whole price path from the two seats' orders
alone. That converts "can we contest the market" from a sweep into arithmetic.

**Seed 3. Extra STRAWBERRY and MILK put on the market each day from day 8,
free of any production cost:**

| extra/day | extra units each | our sell revenue | **their sell revenue** | margin |
|---|---|---|---|---|
| 0 (today) | 0 | 79,944 | 160,497 | −80,553 |
| +2 | 42 | 75,861 | 139,999 | −64,138 |
| +4 | 84 | 73,600 | 124,830 | −51,230 |
| +6 | 126 | 72,490 | 112,063 | −39,573 |
| +8 | 168 | 72,205 | 102,897 | −30,692 |
| +12 | 252 | 68,978 | 91,981 | −23,003 |
| +16 | 336 | 65,872 | 87,710 | **−21,838** |
| +24 | 504 | 64,462 | 86,695 | −22,233 |

**Two readings, and they are the report.**

**(a) The trade is genuinely favourable and the mission's thesis is right.**
Going from 0 to +8/day costs us $7,739 of revenue and costs v43 **$57,600**.
Contesting the market is worth roughly seven times what it costs — *if the
supply is free*.

**(b) It saturates, below zero.** Unlimited free strawberry and milk bottoms
out at a margin of **−21,838**. There is no amount of dumping those two
products that beats this opponent. The residual is WOOL, WHEAT, FERTILIZER,
CARROT and MELON revenue that v43 keeps regardless.

And the model is validated by the one agent that does execute the flood:
`kaito_v48` measures **−17,817** of margin on this seed. The computed ceiling
is −21,838 with our (smaller) production of everything else. The arithmetic and
the real episode agree.

### 4.1 Per-product contest value

`+8 units/day of ONE product from day 8, free`, against the fixed tape:

| flood | our revenue | **their revenue** | margin | end inventory vs I0 |
|---|---|---|---|---|
| **WOOL** | **+38,156** | −2,211 | **+40,367** | −36 — *still scarce* |
| STRAWBERRY | +2,345 | **−24,890** | +27,235 | +51 |
| MILK | −10,084 | **−32,710** | +22,626 | +68 |
| CARROT | +8,378 | −1,018 | +9,396 | −330 — *still scarce* |
| FERTILIZER | +4,293 | −4,117 | +8,410 | +493 |
| EGG | +7,660 | 0 | +7,660 | +146 |
| TOMATO | +7,180 | 0 | +7,180 | +146 |
| MELON | +2,430 | −1,443 | +3,873 | +158 |
| WHEAT | +3,518 | −43 | +3,561 | +769 |
| **going FIRST in the slot ladder, same supply** | +1,387 | −1,387 | **+2,774** | |

Three things worth carrying forward:

1. **WOOL is the largest single number on the board and it needs no contest at
   all.** 168 extra units are worth +$38,156 of our own revenue and leave the
   market *still* 36 units short of I0 — the town's wool demand (~426 units in
   seed 3) is roughly double what both farms together supply, and the below-I0
   curve is logarithmic, so it does not sag. Wool is 10-11 extra sheep. **Ten
   sheep placed by day 10 would be worth more than every market mechanism in
   this document combined.** Sweeps A and G say this executor cannot buy them
   without losing more elsewhere; that is an executor problem, not a market
   one.
2. **STRAWBERRY and MILK are pure suppression** — they take $25-33k off v43 for
   $2k of our own gain (strawberry) or a $10k loss (milk). This is what
   `kaito_v48` does.
3. **The slot ladder is worth $2,774.** It is a real, correctly-described
   engine mechanic and it is a rounding error. That closes `sell_order`,
   `hour0_sells` and collision timing as a line of work: the measured null in
   §3 was not a bug.

Seed 11 reproduces every ordering (STRAWBERRY +31,949, MILK +21,906, slot
+1,238), with WOOL lower (+17,289) because that seed drew more YARN_STOREs and
the wool market was closer to saturation. The *ranking* is stable; the wool
prize is seed-dependent on the shop draw.

---

## 5. Blunt conclusion

**The mission's diagnosis was right, the mechanism is real, the opponent's
bank DID move — and it moved about a seventh of the way, which is not close.**

* The opponent's bank IS the coordinate that decides these games — it ranks the
  gauntlet exactly (§0), and moving it is worth ~7× what it costs (§4).
* **We moved it.** Same-day marketing takes **9,520** off the opponent across
  24 out-of-sample worlds and **every one of the five gauntlet opponents banks
  less** (7,677 to 20,637 each), for 2,341 of our own bank. Day-0 melon takes
  17-18k off v43 on the dev seeds; flooding milk with 18 cows takes 11k off.
* **And it buys nothing.** 9,520 against a 66,000 gap flips **0 of 48 cells**.
  Every other mechanism that moved the opponent's bank down moved ours down at
  least as far. **Zero wins in all 42 configurations, in every cell played.**
* The one significant effect is **not resolved on the gate the mission named**
  (`v43.0_bandit` alone: +2,947, 9 of 14 worlds, p = 0.4240) and went the
  wrong way on the four seeds nobody had looked at.
* And the ceiling is below zero anyway: a **free** doubling of strawberry and
  milk supply lands at −21,838, which is where `kaito_v48` already sits.

**The binding constraint is not the market and it is not per-turn compute
(measured −14,755 yesterday). It is that this executor cannot add production.**
Sheep, cows and geese are all significantly negative on own bank; the
counterfactual says the same units, supplied free, would be worth +$40,367
(wool), +$27,235 (strawberry) and +$22,626 (milk) of margin. The entire gap is
the distance between "the market will pay for these units" and "this planner
cannot produce them without going backwards."

That is the fifth distinct thing this lane has proven cannot close the gap:
per-turn search (−14,755), the base economy (+45% own bank, 0 wins), joint
liquidation (not resolved at 96 cells), a bigger herd (significantly negative),
and now market contest (a real, consistent −9,520 on the opponent's bank that
flips 0 of 48 cells, with a computed ceiling of −21,838 even given free
supply).

### Recommendation

**Ship nothing from this pass**, and specifically do **not** flip `drop_daily`
on the 16-of-20 gauntlet row alone. It fails the primary gate, it went negative
on the reserved seeds, it was selected out of 42 configurations, it costs 2,341
of own bank, and it wins zero cells — and turning it on would mean rebuilding
the artefact, which would end the fork/exec beacon experiment the current ship
was made for. **If it is ever adopted it needs its own pre-registered
measurement**: widen the reactive band per
`docs/history/instrument-repair-2026-09-03.md` §6, then judge `drop_daily` alone
(not its combinations) on held-out worlds, on MARGIN, with the win column
reported next to it.

Every knob is left in the genome, documented,
**defaulted OFF**, and the skeleton at defaults is verified to play a full
official episode to the same bank *to the dollar* as before the change.
`src/kaggriculture/trackp/compiled/main.py` and `rustengine/src/policy.rs` were deliberately
**not** regenerated, so the frozen artefact record in
`models/release_2026-09-03_trackp.json` (sha256 `21a6a024…`) stays valid and
the fork/exec beacon experiment the ship was made for is not muddied. Anything
that ships from here must port the knob to `policy.rs`, add the field to
`.local/econ/gen_rust.py`'s `emit()`, rebuild, and re-run the identity gate —
**and check that the reported bank moved**, per the stale-stage trap in
`docs/history/trackp-base-economy-2026-09-03.md` §7.

If the lane continues, the next experiment is the one this document points at
and does NOT solve: **why does adding a $500 sheep that the market will pay
$2,760 for cost this planner more than $2,760 elsewhere?** The counterfactual
prices the units; the sweeps show the planner cannot make them. Instrument
`_plan`'s day ledger (`.local/econ/selfprofile.py`, and
`economy.compile_cosim(..., debug=True)`) on a sheep-heavy genome and find
which of cash, pasture tiles, feed wheat or unit-ops actually binds. That is a
day-plan question. **It is not a market question and it is not a search
question, and both of those have now been closed with numbers.**

---

## 6. Gates

| gate | result |
|---|---|
| `python tests/test_trackp.py` | **65 passed, 0 failed** |
| `python tests/test_rust_engine.py` | **6 episodes bit-identical, 0 diverged** (no engine file touched) |
| skeleton at new defaults vs the previous skeleton, full official episodes | **same bank to the dollar** on both seeds tried — 93,732 / 159,813 (seed 3) and 85,090 / 147,925 (seed 5), vs `pub_v16rc5` |
| skeleton reproduces the shipped artefact's §0.4 numbers | **yes, to the dollar** — 55,700 own / 124,800 opponent over seeds 3-6, both seats |
| compiled artefact / `policy.rs` / `compiled/main.py` | **untouched** — `submission.tar.gz` still hashes to `21a6a02469202041d01ad6c30a9f13087fe02793d616fb02651d9bbdf4c1fab9`, the sha256 in `models/release_2026-09-03_trackp.json` |
| anything submitted or pushed | **no** |

The six deliberate-breakage cases and the latency bar were not re-run: no
artefact was rebuilt, so the numbers in
`docs/history/trackp-compiled-2026-09-03.md` §0.2-0.3 still describe the file on disk.

---

## Appendix — reproducing every number

```powershell
# the bar: what the field does against v43
python .local\econ\pyduel.py --me data\gauntlet\kaito_v48.py `
  --vs agents\v43.0_bandit.py --seeds 3,4,5,6 --workers 8 --quiet

# who sold what, when, at what price, and in which market slot
python .local\trackp_market\probe.py .local\trackp_market\base.py `
  agents\v43.0_bandit.py --seed 3 --seat 0
python .local\trackp_market\impact.py  <A.py> <B.py> --seed 3 --seat 0
python .local\trackp_market\supply.py  <A.py> <B.py> --seed 3 --seat 0

# the ceiling: exact market re-simulation against v43's fixed order tape
python .local\trackp_market\counterfactual.py --seed 3
python .local\trackp_market\counterfactual.py --seed 11

# a genome sweep ranked on MARGIN and the opponent's bank (dev seeds)
python .local\trackp_market\msweep.py .local\trackp_market\s_comb.json `
  --seeds 11,12,13,14,15,16 --workers 8

# the replication, on the report seeds AND the reserved set, then the split
python .local\trackp_market\msweep.py .local\trackp_market\s_confirm.json `
  --seeds 3,4,5,6,21,22,23,24 --workers 8 `
  --json .local\trackp_market\confirm.json
python .local\trackp_market\split.py .local\trackp_market\confirm.json --seeds 3,4,5,6
python .local\trackp_market\split.py .local\trackp_market\confirm.json --seeds 21,22,23,24

# the gauntlet (secondary gate)
python .local\trackp_market\msweep.py .local\trackp_market\s_confirm.json `
  --seeds 3,4,5,6 --workers 8 `
  --vs data\gauntlet\pub_v16rc5.py data\gauntlet\kaito_v48.py `
      data\gauntlet\pub_rayk_c94.py agents\v43.0_bandit.py agents\v42.1_trackp.py

# the skeleton is unchanged in play with every new knob at its default
python src\trackp\build_econ_agent.py --genome skeleton --out <tmp>.py
```

**Nothing was submitted and no kernel was pushed.**
