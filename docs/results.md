# Results and lessons (final, 2026-10-01)

## Final releases
| Release | Submission | Change | Live record |
|---|---|---|---|
| v63.12_rl_c4 / c5 | 56693053 / 56693063 | v63.11 + strawberry dump (gt) / learned endgame selector | c4: 54 W / 52 L (106 games) |
| v63.13_rl_f898 | 56717454 | f898 chassis (64 worlds routed) + reactive sale timing, preempt / front-run | 22 W of 29 |
| v63.14_rl_f898 | 56718979 | + land_repair, milk-world routes, endgame + economy managers, racepx −10 (0 vs COPY) | 41 W / 12 L / 1 D (54) |
| v63.16_rl_f898 | 56720196 | + 4 world routes, `wb3` only on signal with cash guard | 53 W / 11 L / 1 D (65) |
| v63.17_rl_f898 | — (built after the deadline) | + sale cash floor, 10 re-screened world routes | 343 / 356 replayed live games |

Board at close of the last check (1 Oct 09:00 IST): team score 2179.0, rank 388 of 10,246. Bradley-Terry projection
from v63.16's own games against each opponent submission's rating: 2262 ± 59 → rank ~300 (90%: 224–403)
(`python/release/bt_live.py`).

## v63.17 gates
| Gate | Baseline | v63.17 |
|---|---|---|
| All 356 live games of 5 submissions (replayed) | v63.16: 328 W | **343 W** (same world +10 / −0) |
| Held-out tapes, 7 re-routed worlds | cash-floor build 1,833 / 1,933 | **1,910** (+87 / −10, p ≈ 2e-16) |
| Held-out tapes, 3 round-2 worlds | — | +17 / −4 |
| vs v63.13 (field) | — | 128 / 128 |
| vs top-20 public agents (field, 2,104 games) | — | 0.904 (+19 / −1 vs the cash-floor build) |

## What lost games, and the fixes
| Mechanism | Example | Fix |
|---|---|---|
| Missed BUY_LAND after a cash dip → builds/plants on LOCKED land fail silently, animals stranded | 5 v63.13 losses | `land_repair` guard |
| `wb3` reordered wheat buys first unconditionally in BRUNCH worlds → cash-tight routes starved | 115981405 | `wb3_mode` 2 (signal + cash guard) |
| Sales shell zeroed the route's early wheat sales at ~$10 cash → hires failed, crops died, farm never recovered (−$88k to −$131k) | 116094939, 116107754, 116055277, 116142413 | `cash_floor` |
| Route under-produces vs a bigger herd / better late-game economy in a world | 115978938 (YARN), 116130574 (ICE\|ICE) | per-world route screens with held-out validation |
| Mirror race against our own route family | 116107754 | none — one side of a mirror must lose |

## Lessons worth keeping
1. **Same-world rule.** A tape A/B is valid only where the realized world is unchanged; 7 of 9 early "wins" of the
   cash floor came from changed worlds. Split every comparison.
2. **Layers must react on a signal and never delete route actions they did not create.** Both late regressions
   (`wb3`, the sales shell) were unconditional edits to the route's own orders.
3. **Find the layer, not the symptom.** Manager ablation → stage ablation → `KRL_LAYER_TRACE` on one game pinned
   each cause exactly; reading code first misled twice.
4. **Silent no-ops hide bugs.** Illegal or impossible actions do nothing, so a broken plan looks like a lazy one; trace
   farm state per day (`KRL_TRACE_DIR`).
5. **Validate route picks on held-out tapes and live games.** A pick that wins its screening sample or a live game
   can still lose held-out (ICE|PET 1788: +1 live, −4 held-out).
6. **Live replays are the best regression set**: with the submitted config they reproduce the ladder banks exactly.
7. **Windows process pools**: `max_tasks_per_child` deadlocks after N tasks per worker; reap idle child processes.
8. **Measure in wins, paired.** Mean margins and single games never decided anything.
