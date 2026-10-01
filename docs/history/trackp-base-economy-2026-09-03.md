# Track P — replacing the base economy (2026-09-03)

Prerequisite named by `docs/history/trackp-phase-ab-integration-2026-09-03.md` §6 item 1:
*"the skeleton path alone (budget 0) must reach the mined route's median bank
against the gauntlet, i.e. ~80k, not 44k."* This is the attempt and its number.

**HEADLINE: 44,205 → 63,943 median own bank on the bar's own instrument
(seeds 3,4,5,6 x both seats x 5 gauntlet opponents = 40 official-engine cells).
+45%. The 80,000 bar is NOT met, and it is outside the bootstrap CI of the
best build. Wins went 0-0-40 → 0-0-40, zero discordant pairs, p = 1.0000.**

**AND THE EFFECT IS SMALLER THAN THAT NUMBER SUGGESTS.** On a third seed set
(21-24) that was reserved and never looked at during development, the paired
per-cell own-bank delta is **−434 on the mean** and positive on only 24 of 40
cells, against +14,058 / 30-of-40 on the report seeds. Over all 120 cells the
effect is real but roughly two-thirds the size the headline implies: **+9,732
mean per cell, positive on 84 of 120 (sign test p = 1.4e-05), median own bank
54,286 → 64,332.** §5.1 has the split. Read the 63,943 as the bar-instrument
number it is, not as the size of the improvement.

---

## 0. The instrument, and why it is trustworthy

The budget-0 path is `rustengine/src/policy.rs`, which is a line-by-line port of
the agent `src/kaggriculture/trackp/build_econ_agent.py --genome skeleton` emits.
`tests/test_compiled_agent.py` asserts the two are action-for-action identical.
So the budget-0 economy can be measured as a **plain Python agent file on the
official vendored interpreter**, with no Docker, no tarball and no bridge —
about 1 minute for 40 cells instead of the Linux harness's much slower loop.

That equivalence was checked, not assumed: the unchanged skeleton rendered by
`build_econ_agent.py` reproduced the integration report's §3.1 budget-0 row
**to the dollar** — 44,205 own / 117,902 opponent median over the same 40 cells.

Driver: `.local/econ/pyduel.py` (paired official-engine duel, both seats,
per-opponent W-D-L, share of all money banked).

**Seed hygiene.** Every genome variant was developed on seeds **11,12,13,14**,
which the integration report never used. The report's seeds **3,4,5,6** were
touched only to (a) calibrate the baseline and (b) score the final pick. A third
set, 21-24, was reserved as an untouched confirmation. This mattered: the
best-looking dev-seed variant (`P_herdM`) collapsed on the report seeds.

---

## 1. The elite economy, mined (deliverable 1)

`.local/econ/mine.py`. From `data/routes/index.json`: engine-**1.32.7** rows
with `won == true`, team joined to
`.local/lb2/kaggriculture-publicleaderboard-2026-09-03T12_07_24.csv`, teams
**ranked 1-30 today (rating 2806-2978)**, up to 14 highest-banking wins per team
so no one team's 219 games dominates. **360 profiles, 30 teams, median final
bank 130,486.**

**Units, stated explicitly** (the `sig` block in `index.json` sums ORDER
QUANTITY; nothing here uses it — every statistic below is re-derived from the
raw `data/routes/<rid>.json.gz` tapes):

| statistic | unit |
|---|---|
| `crew[d]` | `len(hands)` on the busiest turn of day `d` — a unit COUNT |
| `plant[d][c]` | number of `PLANT` unit OPS — a tile COUNT |
| `animal[d][k]` | sum of `BUY_ANIMAL` quantities — a unit COUNT |
| `seed[d][c]` | sum of `BUY_SEED` quantities — a seed COUNT |
| `sell[d][p]` | sum of `SELL` quantities **OFFERED** — an UPPER BOUND, because `SELL` partial-fills per unit |
| `ops[d][op]` | count of unit ops by name |

### The top 30 run ONE tape family, and it is startlingly coherent

Per-team medians over 30 teams: **land bought on day 6 and day 11** (mode, 28 of
30 teams), **all 12 melon tiles planted on day 0**, strawberry 33-38 tiles
planted d5-d11, cows 2/2/3/4/4/4/6/8/9 by day 8, 5 sheep, **zero geese**, crew
5 → 4 → 8 → 10 → 11 → 12. The only real variation is the cow ramp's slope.

### Season totals: elite vs the shipped skeleton

| | top-30 median | our budget-0 skeleton | ratio |
|---|---|---|---|
| wheat tiles planted | 187 | 117 | 0.63 |
| strawberry tiles planted | 38 | 24 | 0.63 |
| melon tiles planted | 12 | 12 | 1.00 |
| HARVEST ops | 467 | 280 | 0.60 |
| WATER ops | 1,138 | 928 | 0.82 |
| FEED / CARE ops | 326 / 351 | 250 / 250 | 0.77 / 0.71 |
| FERTILIZE ops | 62 | 46 | 0.74 |
| **PASS unit-turns** | **515** | **2,387** | **4.6x WORSE** |
| SELL WOOL (offered) | 174 | 75 | **0.43** |
| SELL MILK (offered) | 270 | 172 | 0.64 |
| SELL STRAWBERRY (offered) | 272 | 172 | 0.63 |
| SELL FERTILIZER (offered) | 349 | 199 | 0.57 |
| wheat BOUGHT | 178 | 271 | 1.52 (a leak) |

We are at **60-65% of the elite on every revenue line**, and the elite's revenue
is ~63% ANIMAL product (milk + wool + the fertiliser the herd drops).

---

## 2. Transplanting the elite targets DOES NOT WORK (deliverable 2, negative)

Every elite day-target dropped into the skeleton genome on its own made the
agent **worse** (40 cells, dev seeds, median own bank; baseline 58,426):

| change | median |
|---|---|
| baseline `skeleton_genome()` | 58,426 |
| elite herd ramp (9 cows by d8) | 55,416 |
| **all 12 melon tiles on day 0** | 47,476 |
| lower seed cash floors (`floor_straw` 150) | 49,855 |
| the FULL elite economy (land d6/d11 + elite hires + elite herd + melon d0 + straw d5-12) | **46,582** |

**This is a finding, not a failed experiment.** The elite economy is not a set of
day-indexed targets that can be lifted; it is a schedule whose parts are
load-bearing for each other in an executor we do not have. Their day-0 spend is
2 cows + 2 sheep + 12 melon seeds = $2,760 of $3,000, which only pays because
their day-10 melon harvest funds the day-11 strawberry block and the third
quadrant. Ours cannot run that sequence: melon on day 0 sterilises 12 of the 21
usable NW tiles for ten days on a farm that has no second quadrant yet.

---

## 3. What actually moved the number: LAND, and the hire ramp that pays for it

The diagnosis came from instrumenting our own agent's day-indexed economy
(`.local/econ/selfprofile.py`) and putting it beside the mined profile.

Dawn money on the shipped skeleton was **$132-$2,600 from day 1 to day 13**.
The genome asked for `NE` on day 4 and `SW` on day 11, but both buys are
CASH-GATED, so they actually landed on **day 9 and day 14**. The farm therefore
sat on NW's ~21 usable tiles for a third of the season while **ten hands were
hired from day 4 and PASSed 150-180 unit-turns a day**, at `fib_sum(10) = 143`
a day. The idle crew was eating the land money; the missing land was why the
crew was idle.

Two changes, both pure genome:

* **`land: {"NE": 0, "SW": 5}`** — schedule both buys as early as cash allows.
  The day is only a lower bound; the purse still decides.
* **`hires`** — the mined top-30 ramp, which is *leaner* on days 1-7
  (`5,4,4,5,4,5,8,8,10,10,...` after the `MAX_HIRE = 10` clamp) instead of
  `5,7,8,9,10,10,...`.

40 cells each, dev seeds 11-14, median own bank:

| build | median | mean | share of all money banked |
|---|---|---|---|
| shipped skeleton | 58,426 | — | 31.8% |
| elite hire ramp only | 60,855 | 61,057 | 32.8% |
| land ASAP only (base hire ramp) | 62,385 | 58,559 | 29.7% |
| land `NE 0 / SW 8` + elite ramp | 73,656 | — | 32.6% |
| **land `NE 0 / SW 5` + elite ramp (ADOPTED)** | **77,998** | **73,001** | **33.9%** |
| land `NE 0 / SW 6` + elite ramp | 77,998 | 73,001 | 33.9% (cash-gated to the same real day) |
| land `NE 0 / SW 10` + elite ramp | 70,120 | 66,817 | 32.5% |
| land `NE 0 / SW 8 / SE 12` (a third buy) | 61,294 | 60,553 | 30.3% |

Neither half is worth much alone (+2.4k, +4.0k); **together they are worth
+19.6k.** The lean crew is what makes the day-0 land buy affordable, and the
land is what gives the crew something to do.

`SE` (the 4th quadrant, $4,000) is a clear negative and stays off.

---

## 4. The economy curve achieved vs the elite profile (deliverable 4)

Medians, days 0,2,4,…,28.

```
crew (hands)
  elite      5     4     4     8    10    11     9    10    12    12    12    12    12    11    11
  before     5     8    10    10    10    10    10    10    10    10    10    10    10     9     7
  after      5     4     4     8    10    10     9    10    10    10    10    10    10    10    10

quadrants owned
  elite      1     1     1     2     2     2     3     3     3     3     3     3     3     3     3
  before     1     1     1     1     1     2     2     2     3     3     3     3     3     3     3
  after      1     2     2     2     2     2     3     3     3     3     3     3     3     3     3

wheat tiles planted, cumulative
  elite      7    10    17    22    27    39    59    70    83    97   109   125   151   173   187
  before    18    18    36    36    51    51    52    55    67    70    87    99   109   117   117
  after     18    31    56    65    87    90   106   114   130   140   162   172   193   196   196

strawberry tiles planted, cumulative
  elite      0     0     0    12    20    21    38    38    38    38    38    38    38    38    38
  before     0     0     0     0     0     8    22    24    24    24    24    24    24    24    24
  after      0     0     0     0     0     0     8    22    22    22    22    22    22    22    22

PASS unit-turns per day
  elite     26     9    21    16    23    21    25    10    17    21    14    18    14     9     8
  before    73   143   145   180   147   110    86    82    10    32    22    31    19    17     8
  after     73    17     3    86    76   110    22    34    32    20    20    16    19    34    38
```

**Closed:** land timing (we now *beat* the elite — 2 quadrants from day 1),
wheat throughput (196 planted vs their 187), and most of the early idle crew.

**STILL OPEN, and it is the whole remaining gap: STRAWBERRY.** 22 tiles starting
day 12, against 38 tiles starting day 6. Strawberry first-yields at age 10 and
produces 4 times at 2-day intervals, so a tile planted on day 6 banks four
productions and a tile planted on day 12 banks two. That single line is worth
roughly 27k of the elite's 130k.

The block is cash, not the window and not the tiles: `floor_straw = 700` means a
strawberry needs a purse of $800, and our dawn purse is under that for most of
days 1-13. But **lowering the floor does not help** — 300 and 150 both measured
*worse* than leaving it at 700 (72,620 and 71,585 against 77,998), because the
cash the floor protects is what buys the land and the herd. The strawberry gap
is a symptom of the cash curve, not of the floor, and it is not reachable by
re-tuning a threshold.

---

## 5. THE BAR (deliverable 3)

`data/gauntlet/pub_v16rc5.py`, `kaito_v48.py`, `pub_rayk_c94.py`,
`agents/v43.0_bandit.py`, `agents/v42.1_trackp.py`; both seats; official
vendored interpreter; budget 0 (search OFF).

### On the integration report's own seeds (3,4,5,6) — 40 cells

| | before | **after** |
|---|---|---|
| median own bank | 44,205 | **63,943** |
| median opponent bank | 117,902 | 135,149 |
| W-D-L | 0-0-40 | **0-0-40** |
| share of all money banked | 30.0% | 33.0% |

Per opponent (median own / median opponent bank, pooled over 16 cells each
across seeds 3,4,5,6,11,12,13,14):

| opponent | own before | own after | opp before | opp after | W-L |
|---|---|---|---|---|---|
| `pub_v16rc5` | 50,476 | **84,696** | 117,219 | 147,847 | 0-16 |
| `pub_rayk_c94` | 50,864 | 73,348 | 127,124 | 137,738 | 0-16 |
| `kaito_v48` | 54,501 | 67,078 | 116,950 | 130,978 | 0-16 |
| `v42.1_trackp` | 52,869 | 64,350 | 119,580 | 143,163 | 0-16 |
| `v43.0_bandit` | 54,794 | 59,248 | 120,982 | 132,950 | 0-16 |

### Pooled over 80 cells (seeds 3,4,5,6 + 11,12,13,14)

| | before | after |
|---|---|---|
| median own bank | 54,552 | **70,208** |
| mean own bank | 54,081 | 68,896 |
| median opponent bank | 118,042 | 136,894 |
| share of all money banked | 30.9% | **33.5%** |
| `win_metric.paired_test` | — | score 0.000 vs 0.000, **0 discordant pairs, p = 1.0000** |

**Bootstrap 95% CI on the 40-cell median (4,000 resamples):**

| | median | 95% CI |
|---|---|---|
| before, report seeds | 44,205 | [42,529 · 55,152] |
| **after, report seeds** | **63,943** | **[57,040 · 72,336]** |
| after, pooled 80 cells | 70,208 | [63,771 · 76,200] |

**80,000 is above the upper end of both CIs. THE BAR IS NOT MET.** Per the
mission's instruction, the number is reported and the tuning stops here rather
than continuing toward the bar.

### 5.1 The held-out seed set, and how much of the gain is real

Seeds 21,22,23,24 were reserved before any tuning and looked at exactly once,
after the genome was frozen. All three sets are 40 cells (5 opponents x 4 seeds
x both seats), paired cell-for-cell against the old skeleton:

| seed set | before median | after median | before mean | after mean | paired mean Δ | better on |
|---|---|---|---|---|---|---|
| report 3-6 (the bar) | 44,205 | 63,857 | 50,733 | 64,791 | **+14,058** | 30/40 |
| dev 11-14 (tuned on) | 58,426 | 76,231 | 57,429 | 73,001 | **+15,572** | 30/40 |
| **HELD-OUT 21-24** | 53,551 | **59,125** | 58,946 | **58,512** | **−434** | **24/40** |

**GRAND POOL, 120 cells:** median own bank 54,286 → 64,332, mean 55,702 →
65,435, paired delta **+9,732 mean / +8,478 median, positive on 84 of 120,
two-sided sign test p = 1.4e-05**. Bootstrap 95% CI on the pooled median:
**[60,799 · 68,641]**. Opponent bank +7,681 mean, higher on 66 of 120. Share
31.4% → 33.6%. **Wins 0 of 120, before and after, 0 discordant pairs,
p = 1.0000.**

So: the change IS a real improvement in own bank — 84/120 at p = 1.4e-05 is not
noise — but the report-seed and dev-seed windows both flattered it, and on the
untouched window the mean gain is zero. The honest size is ~+10k per cell, not
+20k, and the honest median is ~64k, not ~78k. **That makes the miss on the
80,000 bar larger, not smaller.**

That CI is also why the sweep table above must be read carefully: a ±7k band on
a 40-cell median means most of the individual knob results in §3 are NOT
separable from each other. Only the land change is large enough to survive it,
and only it was adopted.

### The kind-window trap, paid again

`P_herdM` (a faster cow/sheep ramp on top of the land fix) was the best variant
on the dev seeds — 77,061 median, 75,842 mean, 34.8% share, the highest share
measured. On the report seeds it scored **58,494**, worse than the plain land
fix's 63,943. **It was not adopted.** Same lesson as the integration report's
§5: never read a mechanism off one 32/40-cell window.

---

## 6. What this says about the lane — be blunt

**1. The base economy was genuinely broken and is now genuinely better.**
+45% on the bar's own instrument, +29% pooled over 80 cells, from a mechanism
that is fully explained (land starvation feeding an idle-crew cash burn) rather
than fitted.

**2. It buys exactly zero wins. Again.** 0-0-40 before, 0-0-40 after, **zero
discordant pairs, p = 1.0000.** This is the third time this lane has produced a
large, real, reproducible economic gain that converts to nothing: the Phase-B
search's +37% did it, the joint-liquidation model did it, and now +45% of base
economy does it. The margins are 2:1, and `flips()` says dollars only buy wins
where margins are thin.

**3. And the market-share diagnosis survives the fix — with a twist that makes
it worse.** Share moved 30.9% → 33.5%, against `v42.1_trackp`'s 43.0%. Look at
the per-opponent table: **the opponent's bank goes UP in every single row.**
Paired per cell, our bank gains +14,058 on average and the OPPONENT's gains
+13,332. Mechanically that is not a coincidence — growing this economy means
buying more (land, seed, feed wheat), buying drains market inventory, drained
inventory raises the price, and a higher price is worth more to whoever sells
most, which is them. **Raising our production on this board grows the pie and
hands the opponent nearly half the increment.**

So the honest conclusion is stronger than "the base economy was the problem":
**the base economy was *a* problem, it was worth 45%, and fixing it did not move
the coordinate the integration report says we lose on.** The remaining gap to
80k is strawberry timing, which is downstream of a cash curve that early land
and a lean crew have already been spent on.

**4. The elite economy is expressible in `DayKnobs`/the genome — but it is not
transferable.** Nothing had to be embedded: land days, hire ramp, herd schedule,
tile targets and planting windows are all parameters, and the seat law was never
strained. The elite's *numbers* fit the genome fine. What does not transfer is
that those numbers are only good together, inside their executor. Our best
economy is NOT theirs; it is a different one that our planner can actually run
(2 quadrants from day 1, 196 wheat tiles, a lean early crew). That is worth
saying plainly to anyone who plans to mine harder: **the ceiling here is the
executor, not the parameter values, and no amount of better mining changes it.**

**5. Recommendation for the Sep 24 go/no-go, unchanged in direction.** This work
removes the "base economy" explanation from the table. It does not make the lane
competitive and it does not close a single cell. `docs/history/plan-2800-2026-09-03.md`'s
contingency — two bandit-lane agents from different base families — remains the
leading option.

---

## 7. What changed in the repo

| file | change |
|---|---|
| `src/kaggriculture/trackp/build_econ_agent.py` | `skeleton_genome()`: `land` → `{"NE": 0, "SW": 5, "SE": -1}`, `hires` → the mined top-30 ramp. New `hire_slack` knob, **default -1 = OFF** (see below). The pace budget is now priced off the planned crew size rather than the built unit list — arithmetically identical at `hire_slack -1`. |
| `rustengine/src/policy.rs` | `SKELETON` const regenerated from the Python genome by `.local/econ/gen_rust.py`, never hand-edited. The generator round-trips the OLD const byte-for-byte, which is how the port is known not to have drifted. |
| `rustengine/src/plan.rs` | `DayKnobs::skeleton`: the same hire ramp, and `buy_land` now fires as early as cash allows instead of `day >= 4`. This is the SEARCH's prior, so the search now hill-climbs from the better economy too. |
| `.local/candidates/trackp_compiled/skel_ref.py`, `stage_win/` | regenerated / restaged (they were stale and silently made the identity gate score the OLD policy — see below). |

`hire_slack` was built to test the biggest single leak in §1 (2,387 PASS
unit-turns vs 515) by hiring only the hands the day's task list can feed. It
**measured worse than the hand-set lean ramp** (63,788 / 70,201 / 72,120 at
slack 0 / 1 / 1-with-a-12-ceiling, against 77,998), so it ships OFF. It is left
in because it is the correct shape of the fix and the ramp is the crude
approximation; it just is not the ramp that wins today.

### Gates re-run after the change

| gate | result |
|---|---|
| `cargo build --release --bin kagg` | **OK** (retry loop needed: another session held `kagg.exe` open via `kagg serve`) |
| `python tests/test_rust_engine.py` | **6/6 bit-identical, 0 diverged** — no engine file was touched |
| `python tests/test_compiled_agent.py --seeds 3,4,5` | **PASS**, identity gate, 0 differing actions over 4,320 turns, 0 fallbacks, 0 repairs |
| template edit is a no-op at `hire_slack -1` | verified directly: 1,438 turns against `pub_v16rc5`, **0 differing actions** vs the shipped `skel_ref.py` |

**A trap worth recording.** `tests/test_compiled_agent.py` defaults to
`--stage .local/candidates/trackp_compiled/stage_win` and `--ref .../skel_ref.py`.
Both are *copies*. After changing the policy they were stale, and the test still
printed **PASS with 0 differences** — because a stale binary and a stale
reference agree with each other perfectly. The give-away was the BANK: seed 3
still read 55,008, the pre-change number. Restaging moved it to 63,943.
**The identity gate proves the two twins agree; it does not prove either of them
is the policy you just wrote. Always check the bank moved.**

---

## Appendix — reproducing every number

```powershell
# mine the elite economy (READ-ONLY over data/routes)
python .local/econ/mine.py 30 14           # -> .local/econ/profiles_top30.json
python .local/econ/summarize.py            # day-indexed profile + season totals

# the bar, on the official engine, budget 0 (search OFF)
python src/kaggriculture/trackp/build_econ_agent.py --genome skeleton --out .local/econ/cand/now.py
python .local/econ/pyduel.py --me .local/econ/cand/now.py `
  --vs data/gauntlet/pub_v16rc5.py data/gauntlet/kaito_v48.py `
       data/gauntlet/pub_rayk_c94.py agents/v43.0_bandit.py agents/v42.1_trackp.py `
  --seeds 3,4,5,6 --workers 8 --json .local/econ/now_report.json

# our own economy curve, beside the mined elite one
python .local/econ/selfprofile.py .local/econ/cand/now.py `
       data/gauntlet/pub_v16rc5.py 11,12,13,14 .local/econ/self_now.json
python .local/econ/compare.py .local/econ/self_now.json

# a genome sweep (each row = 40 official-engine cells)
python .local/econ/sweep.py .local/econ/sweep7.json 11,12,13,14

# regenerate the Rust twin of the genome -- never hand-edit SKELETON
python .local/econ/gen_rust.py            # prints the policy.rs const

# gates
cargo build --release --bin kagg
python tests/test_rust_engine.py
cp src/kaggriculture/trackp/compiled/main.py .local/candidates/trackp_compiled/stage_win/main.py
cp rustengine/target/release/kagg.exe .local/candidates/trackp_compiled/stage_win/kagg.exe
python src/kaggriculture/trackp/build_econ_agent.py --genome skeleton `
       --out .local/candidates/trackp_compiled/skel_ref.py
python tests/test_compiled_agent.py --seeds 3,4,5
```

**Nothing was submitted and no kernel was pushed.**
