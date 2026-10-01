# agent v19 — `agents/v19_route.py`

Built 2026-08-08. An open-loop 719-turn route on the standard tape runtime
(`src/kaggriculture/engine/tape_runtime.py`), identical recipe to v18: weed repair, projected-shed
clamping, sell-first ordering, price-impact SELL ranking, terminal liquidation.
All other adaptive layers (mirror tie-break, endgame pull, floor guard, glut
guard, swap-advance) exist in the runtime and are **off**, each with a measured
negative or null result behind it.

## What changed vs v18

Only the base route: `90521287_s0` → **`90525850_s0`** (same team, THUNDER
THUNDER, rank 2 at capture; fit window; recorded bank $153,388; mined
2026-08-06 — still the freshest archive day available on 2026-08-08).

Selected by a same-family tournament — five fit-window candidates played on an
11-opponent mini-panel (the three opponents v18 could not beat, the TT-sibling
tape that beats v18, and seven crowd tapes), 2 seeds × 2 seats, repeated on an
independent seed set. `90525850_s0` was equal or better than v18's base against
10 of 11 opponents across both sets. Two candidates re-confirmed old rules:
`90518963_s1` swept one seed set and collapsed on the next (third-seed-set
rule), and the highest-recorded-bank candidate went 0% against the crowd (bank
is a bad selector).

## Measured

| | |
|---|---|
| Top-100 held-out panel (71 opponents, 426 games, seeds ×3, both seats) | **97.2%, 70/71 beaten outright, mean margin +$8,796** |
| v18 same panel (for reference, 52-opponent roster) | 93.6%, 49/52, +$7,740 |
| Seb (allegedly), rank 1 — v18's −$48k squeeze | **67%, +$1,534** |
| lemon13418 (rank 24, our-family sibling) | **67%, +$483** |
| THUNDER sibling tape that beat v18 | **67%, +$6,575** |
| David Schindler15 (rank 26) — still unbeaten | 0%, −$3,099 |
| Self-play validation | DONE, banks 146,248 / 146,248 |
| Latency | mean 0.22 ms, worst 82.26 ms (budget 1 s) |
| Size | 41,406 bytes |

## Negative results shipped off in the same pass

`_GUARD_RATIO` (floor guard generalised to a price-ratio trigger) and
`_SWAP_ADVANCE` (value-hedged pull-forward of an already-scheduled healthy
SELL while holding a depressed one): both lose 0% to a byte-identical control
at −$10k to −$19k over 3 seeds × 2 seats. The trigger cannot distinguish an
opponent's squeeze from the route's own scheduled price impact. Fifth
independent confirmation of *change the order, never the inventory*. Details:
BUILD_JOURNAL 2026-08-08; code remains in `tape_runtime.py` behind flags as
the measured record.

## Graph

`agents/v19_route.html`, regenerated 2026-08-08 by `src/kaggriculture/agentbuild/model_graph.py`.
