# Instrument repair — the panels could not see (2026-09-03)

What broke, what was changed, what the repaired instrument measures, and the
part that is still unmeasurable. Companion to
`docs/history/gate-calibration-2026-09-03.md` (which now carries the standing rules)
and `docs/top10-plan-2026-09-03.md` section B3.

Everything below is measured. Raw run:
`.local/band_panel/groundtruth_2026-09-03.log` (2,110 cells, 856s, 6 workers,
serve substrate green at 12/12 exact-bank). Analysis:
`.local/band_panel/groundtruth.py`.

---

## 1. What was broken

Three failures, all on 2026-09-03:

**(a) The panels anti-predicted.** `v44_single` won the panel that selected it
**77-11** (the base scores 64-24) and then lost **84 of 86 discordant held-out
cells**, p < 0.0001. Selecting and scoring on the same cells returns the
selection, not the strength.

**(b) The panels went blind.** Every arm returned 0.975-0.988; the entire v22
chassis moved **4 cells out of 652** against the raw tape (p = 0.50). The
reactive gauntlet was worse — 96-0 for the candidate AND 96-0 for the
incumbent, a test nothing can fail.

**(c) Paired tests were computed over cells, not worlds.** Both seats of one
(opponent, seed) are the same world played from opposite sides; with
deterministic agents on identical starting farms the swap reproduces the cell
to the dollar. Counting both doubled every sample. Measured on a real
comparison: the reactive-band margin test read **p = 0.0227 on 24 cells and
p = 0.1460 on the 12 real worlds**. Every earlier "significant" panel claim
computed this way is overstated.

---

## 2. The world distribution — the core repair

### The diagnosis in the brief was right about the symptom, wrong about the cause

Symptom, confirmed at scale (2,110 cells, seed 501, both seats):

| band | cells | median shared bank | % sub-90k | vs ladder |
|---|---|---|---|---|
| ladder reference (8,000 traces) | — | **85,000** | **58%** | — |
| sub2000 (tapes) | 480 | 63,056 | 96% | NOT ladder-like |
| mid (tapes) | 480 | 55,714 | 99% | NOT ladder-like |
| top100 (tapes) | 996 | 55,546 | 93% | NOT ladder-like |
| **reactive (live agents)** | 154 | **88,656** | **53%** | **LADDER-LIKE** |
| all tape cells | 1,956 | 56,178 | 95% | NOT ladder-like |

**Option (a) works. The reactive band's worlds match the ladder's.** That is
the answer to task 2: add reactive opponents, and the distribution is fixed —
88.7k / 53% against a ladder at 85k / 58%.

### The stated cause — "tapes never unlock shops" — is FALSE

It was worth checking rather than repeating, and it does not survive:

| band | shops unlocked (mean) | cells with NO shop unlocked | first unlock step (median) |
|---|---|---|---|
| sub2000 | 8.0 | 0% | 72 |
| mid | 8.0 | 0% | 72 |
| top100 | 8.0 | 0% | 72 |
| reactive | 8.0 | 0% | 72 |

(Every shop, in every cell, in every band. The readout used to be truncated to
the first six unlocks, which made all four bands read "6.0" and hid the very
comparison it existed for; `play()` now stores the whole sequence.)

Identical. The shop gate is not the mechanism. What actually happens is that
**both banks collapse in a tape world**, in the same seeds:

| band | our median bank | opponent median bank |
|---|---|---|
| sub2000 | 64,569 | 55,968 |
| mid | 58,110 | 52,957 |
| top100 | 61,014 | 51,333 |
| **reactive** | **90,521** | **93,368** |

Our own agent — unchanged, same seed — banks a 61k median against a tape and
**90.5k against a reactive opponent**. So a tape panel is not "the same game
against a weaker opponent"; it is a **different economy**, one with only a
single real trader in it. That reframes the bias: the tape bands do not merely
deflate the level, they change what winning is made of, which is exactly why
mechanisms with large measured effects there (nearest-neighbour dump
prediction, the per-turn searcher's +37% bank) convert to zero wins.

`shop_stats()` and the per-band world readout are now printed on every run so
this is checked rather than assumed.

---

## 3. What changed in the tooling

Files: `src/kaggriculture/measure/band_panel.py`, `src/kaggriculture/engine/serve_gate.py`, `src/kaggriculture/measure/elite_panel.py`,
`src/kaggriculture/data/dual_bar_miner.py` (one call site), `tests/test_instrument_repair.py`.
The CLI is unchanged — `--build`, `--score`, `--seeds`, `--workers`, `--bands`
all behave as before.

1. **MARGIN everywhere.** Every band, every agent, and every split now reports
   cells W-L-D, ratio, `clean_ratio` (dead-opponent-excluded), the **median
   paired margin per game**, and — when two agents are scored — the count of
   cells where the **margin moved but the winner did not**. `serve_gate` and
   `elite_panel` print the same block. This restores resolution to data we
   already collect: on the saturated gauntlet, v43.0 vs v42.0 is 48-0 vs 48-0
   with **zero** discordant cells and **24 worlds where the margin moved**.
2. **A recorded hold-out split, by default.** Each manifest carries a
   rating-stratified, deterministic, RNG-free `selection` / `holdout`
   assignment (`assign_splits`; both halves span the band, so the split is not
   confounded with strength). Splits are frozen into the manifests: sub2000
   20/20, mid 20/20, top100 42/41, reactive 7/6. Scoring reports both halves,
   the **bar is judged on the held-out half**, and the paired agent comparison
   prints a held-out row marked DECISION. Seeds are splittable too
   (`--seed-holdout`, needs >= 2 seeds; a held-out cell then shares neither
   opponent nor seed with anything selection saw). `band_panel.panel_tapes
   (band, "selection")` is the API for any selecting caller; `dual_bar_miner`
   S3 now screens on the selection half only, so the complement stays clean.
3. **A `reactive` band**, 13 live agents (the `serve_gate` roster plus our
   current bandit), **frozen as copies** under `.local/band_panel/reactive/` so
   another lane cannot change an opponent mid-comparison. Self-play cells are
   detected by CONTENT hash, not path, and skipped.
4. **Mirror dedupe before every p-value.** Paired tests now run over distinct
   worlds and print `n=12/24 worlds` so the discount is visible.
5. **World distribution against the ladder** — per band, per opponent kind,
   per agent, with a LADDER-LIKE verdict and the shop-unlock readout.
6. **Tests**: `tests/test_instrument_repair.py`, 16 pure-logic cases (margin
   arithmetic, dead-opponent split, stratified/deterministic holdout, seed
   splits, world stats, shop stats, mirror dedupe, exact sign test, spearman,
   panel-halves disjointness). No episodes; `tests/run_all.py --fast` is green
   (19 suites).

---

## 4. Validation against ground truth — the blunt part

Six agents with live ladder outcomes, scored on the OLD panel (tape bands) and
the NEW one (reactive band), 2,110 cells, one seed, both seats.

| agent | live | OLD tape ratio | OLD tape margin | NEW reactive ratio | NEW reactive margin |
|---|---|---|---|---|---|
| v43.0_bandit | 2519 | 0.988 | +21,856 | 0.833 | +15,174 |
| v42.0_bandit | 2545 / 1959 | 0.804 | +1,777 | 1.000 | +16,086 |
| v42.1_trackp | 1975 | 0.791 | +2,176 | 1.000 | +15,554 |
| v42.1_bandit | 1733 | 0.672 | +1,471 | 0.667 | +2,869 |
| v41.1_trackp | 2323 | 0.479 | -302 | 1.000 | +14,931 |
| v41.0_bandit | 1818 | 0.521 | +289 | 0.833 | +8,506 |

Spearman rank correlation with the live outcome, with the **exact** two-sided
permutation p over all 720 orderings:

| measure | rho | p | rho using v42.0's re-ship (1959) |
|---|---|---|---|
| OLD tape, all bands, ratio | +0.543 | 0.297 | +0.314 |
| OLD sub2000 ratio | +0.618 | 0.244 | +0.530 |
| OLD mid ratio | +0.464 | 0.372 | +0.348 |
| OLD top100 ratio | +0.657 | 0.175 | +0.486 |
| OLD tape HELD-OUT ratio | +0.657 | 0.175 | +0.486 |
| NEW reactive ratio | +0.617 | 0.233 | +0.463 |
| **NEW reactive MARGIN** | **+0.829** | **0.058** | +0.486 |
| NEW reactive HELD-OUT margin | +0.771 | 0.103 | +0.429 |

**Say it plainly: no offline measure ranks these six agents at p < 0.05.**
The n=6 two-sided 0.05 critical value is rho = 0.886. The best measure we have
— the reactive band's median paired margin — reaches +0.829 (p = 0.058), and
that is the *only* one that even approaches it.

Three things follow, and they matter more than the ranking:

**(a) The repair moved the instrument in the right direction, on every axis
we can see.** Every reactive measure beats its tape counterpart, and the
margin channel beats the cell channel everywhere it is measured. The tape
panel's single worst error is concrete and disappears under the repair:
**v41.1_trackp, our second-best live outcome (2323), is DEAD LAST on the tape
panel at 0.479 and -$302/game, and top of the reactive band at 1.000 and
+$14,931/game.** The old panel did not under-rate it; it inverted it.

**(b) The ground truth is not solid enough to certify anything.**
`v42.0_bandit` is one file with two ladder results 586 points apart (2545,
then 1959). Swap that single row and the best rho falls from +0.829 to +0.486.
A rank correlation over six n=1 rows, one of which contradicts itself, can
**refute** a panel (a strong negative would be decisive) but cannot certify
one. It did not refute the repaired panel; it also cannot bless it.

**(c) The reactive band's CELL channel is already saturating too.** Three of
six agents score 1.000 on it. With 13 opponents and one seed there are only 12
worlds; the cells stop discriminating almost immediately and only the margin
column separates the field (+16,086 / +15,554 / +14,931 against +2,869). This
is the same failure the gauntlet had, arriving early. **Widen the reactive
roster and the seed count before the cell channel is trusted again.**

---

## 5. What still cannot be measured

- **Whether any offline number predicts a ladder rating.** Six n=1 rows, one
  self-contradictory, is not a validation set. This is not a tooling gap we
  can close by building more instrument; it needs more shipped-and-converged
  agents, or it stays open. Until then offline gating **ranks candidates and
  vetoes regressions; it does not forecast a rating**, and no shipping
  decision should be justified by a panel number alone.
- **The absolute level of any band.** The tape bands measure a different
  economy (section 2); the reactive band measures a 13-agent field, not the
  ladder's distribution. Both rank; neither forecasts a win rate.
- **Anything on the mid/top100 bands at high resolution**, until they are
  re-mined with reactive opponents or replaced. Their held-out halves are
  clean now, but their worlds are still 95% sub-90k.
- **Non-transitivity.** Every band is an average over a fixed roster. It
  cannot see "loses specifically to the family that will be seated opposite
  us", which is what the second-slot rule exists for.

## 6. What to do next with this instrument

1. Score every candidate on `--bands sub2000,mid,top100,reactive` and read the
   **HELD-OUT** line and the **margin**, in that order.
2. Widen the reactive roster (more live agents, more seeds) — its cell channel
   is one build away from being as blind as the gauntlet was.
3. Re-mine the tape bands from reactive opponents' play if that is possible at
   all; otherwise treat every tape band as a ranking aid with a known economy
   bias, never as a level.
4. Never quote a p-value computed over paired cells again. Worlds, or nothing.
