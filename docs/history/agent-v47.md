# v46.2_bandit + v47.0_trackp — the evolved-router / value-net-tree pair (2026-09-06/07)

## Shared chassis (both seats)
Base economy 105367892_s0 (guarded re-screen winner on the fresh 84-team
band) + race-first flush (residue flush opens step 704; measured: opponents
fired the settlement race first 15/20 in lost worlds) + opp-dump pre-empt
(observable-features MLP, val AUC 0.807, heads >= 0.70 only, volume-
conserving via the p2_adv ledger) + **62-world tail router**: at step
145/146 the route swaps to the world's tail — donor tails whole-prefix-
exact from the clone family, selected by a mirror-fit tournament (donors +
tail-vs-tail games on manufactured seeds) and improved by (1+lambda)
evolution in 47 worlds — + **mirror counter-hardening** (board-probe
detection at steps 73/97/121; a detected clone gets pre-empt threshold
0.75 and flush from 700).

## v47.0_trackp adds the value-net game tree
LOADSTATE searcher at day boundaries (d8+, fine-grained d21+): plan
variants (retime/hold/defer) rolled H steps on the embedded Rust engine,
then **2-ply expansion of the top plans** with opponent-model branching
(family-book schedule vs mirror; min-over-opponents backup), leaves valued
as **predicted final margin** by the value net (27-dim day-boundary state,
trained on 33.6k synthetic states across all 64 shop-pair worlds via
Colab; sign-acc 0.770). Confidence gate $1500 ($500 in the d26+ settlement
window). Tar carries main.py + the kagg musl binary.

## Gate record (paired vs the live v46.1 seats)
| instrument | v46.2_bandit | v47.0_trackp |
|---|---|---|
| hard band 96 | 46/96 vs 45 (2-1 discordant) | 46/96 vs 45 (2-1) |
| live-loss gauntlet 22 | 3 flips vs 0 | 3 flips vs 0 |
| elite >=2500 (40) | 24/40 +3,956 | 23/40 +3,546 |
| in-family BT (24 seeds) | 78/96 (beats v45.0-arms 43) | 44/96 vs 30 |
| reactive panel (78 cells) | exact W/L tie | W/L tie (p=1.0 decision) |

Value net v2 (126k rows, better MAE) lost the play head-to-head 0-5 to v1
(better sign-acc) — v1 ships; scale-up path is the Drive-resumable Colab
notebook.

---

# v47.1 pair — the all-improvements release (2026-09-07, IN BUILD)

Operator freeze order: the pair ships with ALL fixes and improvements;
the full gate battery runs ONCE on the frozen build. Status below is the
build-time truth; the gate table is filled by the end-battery.

## The improvement checklist (operator-mandated, all-or-nothing)

| # | improvement | status | enabled in |
|---|---|---|---|
| 1 | Day-3 fork — per-first-shop economy continuation, splice t=74 | DONE, REALIZED-grouped rerun: 6 branches (BAKERY .812→.958 … ICE_CREAM .896→1.000) | bandit |
| 2 | Family-killer counter — mirror-only splice t=122, evolved vs the clone family | DONE, fitness 0.979, prefix byte-exact through 122 | bandit |
| 3 | Dual-tail runtime selection — ROLLOUT 48-step value pick, primary vs secondary tail | DONE (34 secondary tails) | bandit |
| 4 | Softened mirror hardening — pre-empt 0.80, flush 702 (collapse audit: 6/14 own-collapses had hardening active) | DONE | bandit |
| 5 | Evolved-tails filter fix — EVOLVED teams no longer dropped by the bank>opp filter | DONE (62 tails embed) | bandit |
| 6 | Day-6 fork — per-WORLD continuation, splice t=146, parent-tagged (d3/base), fires only on matching prefix | DONE, REALIZED-grouped: 13 branches (FARMERS_MARKET\|BAKERY .656→.938) | bandit |
| 7 | Value net — v5 (39-dim, 704k rows) and p1 (sibling-rank, 0.977 pair-acc) both REFUTED in duels (12-20; 1-23). Lane CLOSED; v1 ships. Depth-2 CLOSED (measured catastrophic on every net) | CLOSED BY MEASUREMENT | trackp keeps v1 |
| 8 | Realized-world conditioning — worlds are realized by play, not seed (shop RNG shared with weeds); all per-world evolution regrouped by realized.py + batch_eval_cells | DONE (day-3/day-6 rerun on it) | evolution instruments |
| 9 | Owned opening — full-tape evolution off the clone commons | RUNNING (gen 120/300, 0.283→0.367); ships only if fitness gate passes | bandit (candidate) |

## Runtime precedence (bandit route ownership)
day-3 fork (t=74) > family-killer counter (t=122, mirror-only) >
day-6 fork (t=146) > world pair-tails (t=145/146, dual-tail ROLLOUT
select). Any committed owned route silences the arms
(wtail/d3branch/d6branch/counter_on guards).

## v47.1_trackp
Chassis + rollout searcher + family-book opponent conditioning +
value net v1 (27-dim; the only net that survives duels), 2-ply,
$1500/$500 confidence gate, 650 ms budget. Tar = main.py + kagg musl
binary (GENGAME/ROLLOUT verbs included).

## Generation/search instruments added this cycle
GENGAME (whole game, one IPC, 17.8 ms/game) and ROLLOUT (48-step, one
IPC) verbs in rustengine service.rs — 16k games in ~25 min on this box;
sibling_gen.py (searcher-distribution dataset); value_train_pair.py
(pairwise trainer); realized.py (realized-world map);
sell_search.batch_eval_cells (seat-explicit cells).

## Gate record — single battery on the frozen pair, 2026-09-07 11:23-12:55 IST
Frozen shas: bandit `bdbc651c` (391,446 B), trackp `89724114` (544,240 B).
Legality suite: all checks passed.

| instrument | v47.1_bandit | v47.1_trackp |
|---|---|---|
| upset gauntlet (17 certified live-loss cells) | **2/17 credited flips** (incumbents: 0 by construction) | 0/17 |
| elite band (≥2500 recorded, 120 cells) | **72/120 credited (+26 uncredited), median +$3,250** — best ever posted | 64/120 (+23), median +$2,299 |
| 4-way BT (24 games/pair, both seats) | **strength 1.713, 51/72, rank 1** (13-11 over live v46.2) | strength 0.645, 30/72, rank 3 (18-6 over live v47.0 p=0.023; 7-17 to live v46.2) |
| reactive band panel (26 cells, bar 0.90 held-out) | **PASS — 26-0, held-out 12-0 (1.000), med margin +$12,995** (first perfect sweep) | FAIL — 22-4, held-out 0.833; swept by kaito_v48 + v43.0_bandit (seed 501) |
| paired bandit-vs-trackp | margin 11b/2w p=0.0225 (bandit better) | — |

BT strengths → Elo vs the 4-agent mean: v47.1_bandit +94, live v46.2 +43,
v47.1_trackp −76, live v47.0 −178. Both new seats beat the seat they retire;
the bandit beats everything.

## Honest boundaries (from the same battery)
* The 15 unflipped upset cells replay near-identically to the recorded
  losses — no owned layer fires there (no mirror, no matching realized
  branch). Upsets are a BASE-ROUTE problem; growing realized day-6
  coverage is the lever, not more routing.
* v47.1_trackp fails the 0.90 reactive bar (0.833) and loses to the live
  bandit head-to-head; it ships (if it ships) as the hedge seat on the
  strength of 18-6 over the incumbent it retires and 64/120 elite.
* Owned opening: measured and REJECTED (0.138 vs base 0.867 paired,
  3/101 cells, p<0.0001) — base route retained.
