# Gate calibration — what our panels have and have not predicted (2026-09-03)

Written during the v43 ship audit. This is the answer to "we scored X
offline, what does that mean on the ladder?" — checked against our own
ship history rather than assumed. Read this before believing any single
panel number again.

## The record

| agent | elite panel | mid-field gauntlet | live outcome |
|---|---|---|---|
| chassis_wool_rice = v42.1_bandit | 0.803 (swept-by 1/77) | 0.917 (66-6) | **STALLED 1733** @60g |
| chassis_router = v42.0_bandit | 0.706 (swept-by 12) | 1.000 (72-0) | **converged 2545** |
| v42.0_bandit re-ship (same file) | 0.681 | 1.000 | 1959 @131g, flat |
| chassis_medoid = v42.0_trackp | 0.519 (swept-by 22) | 1.000 | 1905 |
| 104516013_s0 = v42.1_trackp | 0.667 | — | 1932 @120g, climbing |
| v41.1_trackp (specialist router) | 0.351 | — | 2323 |
| v41.0_bandit | 0.468 | — | 1818 |

## Three conclusions

**1. The elite panel alone does not rank ladder outcomes.** It inverts
twice in seven rows. The wool bandit scored the HIGHEST elite number we
have ever measured (0.803, swept by only 1 of 77 teams) and converged
812 points BELOW an agent that scored 0.706. Never ship on an elite
number alone.

**2. The failure mode is a trade, not a level.** Every inversion above is
a candidate that bought elite strength with mid-field concessions. The
wool family conceded 6 gauntlet cells; that stopped the burst before the
elite band, and K decays to ~8.5 by game 80, so the elite ceiling it was
built for became unreachable. Convergence is TWO-DIMENSIONAL: pass-through
speed while K is hot (mid-field ~1.000) AND the ceiling (elite high).
Hence the standing spec: **elite >= 0.75 AND mid-field ~1.000**.

**3. The mid-field bar is now blind.** The reactive gauntlet returns 96-0
for the candidate AND 96-0 for the incumbent. A test nothing fails cannot
discriminate, so "mid-field 1.000" today means only "not measurably worse
than v42.0". Since conclusion 2 says the mid-field bar is what kills
convergence, this is the most dangerous blind spot we have. The rating-band
panels (`src/kaggriculture/measure/band_panel.py`, sub2000 / mid) replace it and must be run on
every candidate before ship.

## Known biases of the elite panel (it stays useful, but for ranking only)

- **Open-loop degradation.** Each panel opponent is an elite team's TAPE
  replayed in a world it was not compiled for; the tape degrades, our
  reactive chassis does not. Absolute level is inflated for every
  candidate. It ranks; it does not forecast a win rate.
- **Selection overlap / winner's curse.** `dual_bar_miner.py` screens
  candidates on S1 = 8 and S2 = 24 cells of this same panel, so a mined
  candidate's full-panel score is partly the selection it survived.
  Report a HELD-OUT split (teams never used in selection) as the decision
  number.
- **Own-team cells.** A candidate built on team T's tape scores against
  team T's tape in the panel. Report with and without the own-team cell.
- **Path luck dominates any single ladder row.** Byte-identical
  submissions diverge 300-1400 rating; every "live outcome" above is n=1.
  The table can support "high elite alone does not imply a high level";
  it cannot support a regression or a conversion factor.

## STANDING RULES after the instrument repair (added 2026-09-03 evening)

Written after the panels did two things on the same day: they ANTI-predicted
(`v44_single` won its selection panel 77-11, then lost 84 of 86 discordant
held-out cells, p < 0.0001) and they went BLIND (the whole v22 chassis moved 4
of 652 cells, p = 0.50). Full account and measurements:
`docs/history/instrument-repair-2026-09-03.md`. These four rules are now enforced in
the tooling, not remembered:

1. **Report MARGIN alongside cells, always.** `band_panel`, `serve_gate` and
   `elite_panel` print cells, ratio, `clean_ratio`, the **median paired margin
   per game**, and — when two agents are scored — the count of cells where the
   margin moved but the winner did not. A band that reads 0.988 for every arm
   can still be moving $15k/game; the win column had thrown that away.
2. **The HELD-OUT half is the decision number.** Every panel manifest carries a
   rating-stratified `selection` / `holdout` split. Selection tools screen on
   `band_panel.panel_tapes(band, "selection")` (`dual_bar_miner` S3 now does);
   scoring reports both halves and judges the bar on the held-out one. Never
   quote a panel number produced by the cells that chose the candidate.
3. **Both seats of one world are ONE observation.** With deterministic agents
   and identical starting farms the seat swap reproduces the cell to the
   dollar. Every paired test now dedupes exact mirrors before computing a
   p-value. This is not cosmetic: a reactive-band comparison read
   p = 0.0227 on 24 cells and p = 0.1460 on the 12 real worlds. **Any earlier
   claim of significance computed over paired cells is overstated by roughly a
   factor of sqrt(2) in the test statistic.**
4. **Report the world distribution against the ladder.** Ladder reference,
   from 8,000 real per-turn traces: median shared bank ~85k, 58% of worlds
   sub-90k. Every band prints its own median / sub-90k share, the gap, and a
   LADDER-LIKE verdict, plus the shop-unlock rate that explains it.

## The rule going forward

A candidate ships when it is at the top of BOTH bars at once — never
because one number is a record. Report, for every candidate: elite
(overall + held-out + own-team-excluded), band panels sub2000 / mid /
**reactive** with the LOW/HIGH price-regime split, the **median paired
margin** on each, the leak audit, and the local `submit.py --dry-run`
gate. Read the HELD-OUT line first. The final Bradley-Terry runs on
post-lock games only, so true pairwise strength is the objective and the
live rating is seeding plus our measurement instrument.

## What the repaired panel is and is not validated to do (2026-09-03)

All six agents in the table above were re-scored on the tape bands and on
the new reactive band (2,110 cells; `.local/band_panel/groundtruth.py`).
Rank correlation with the live outcome, exact permutation p:

| measure | rho | p |
|---|---|---|
| OLD tape bands, ratio | +0.543 | 0.297 |
| OLD top100 ratio (= the elite panel) | +0.657 | 0.175 |
| NEW reactive ratio | +0.617 | 0.233 |
| **NEW reactive median MARGIN** | **+0.829** | **0.058** |

**No offline measure ranks these six at p < 0.05** (the n=6 critical value
is rho = 0.886). The repair improved every axis — the reactive band beats
the tape bands, and margin beats cells — and it removed one flat
inversion: v41.1_trackp (live 2323) is last on the tape panel at 0.479
and first on the reactive band at 1.000. But `v42.0_bandit` alone has two
live results 586 apart for the same bytes, and swapping that single row
drops the best rho to +0.486.

**So: offline gating RANKS candidates and VETOES regressions. It does not
forecast a rating, and no ship is justified by a panel number alone.**
The reactive band's CELL channel is already saturating (three of six
agents at 1.000 over only 12 distinct worlds) — widen the roster and the
seeds before trusting it, and read the margin column meanwhile.
