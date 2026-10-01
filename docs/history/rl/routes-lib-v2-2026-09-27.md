# Route library v2 (2026-09-27)

Goal: more strong routes per world than v61.1's 41, each world's route chosen by paired evidence.
Table: `configs/route_tables/lib_v2.json` (per-world overrides, `--route-table`), ONLY valid with the base
`configs/bases/v61.1r2` (= the v61.1 library + the chosen routes, ids kept; v61.1r = stage 1). Agent = v63.5_rl (PPO i790 + big1
shell + chain-off r127,sm,r95). Everything paired vs the current route on the same seeds and seats.

## Pipeline
1. Mine (A1). A route must play route 0's exact farm actions (farmer + hands) on steps 1..143 and the same opening
   purchases, so the step-144 switch keeps the tape in sync. Source 1: the top-100 local extract (6,674 seats:
   1,438 match route 0's farm actions, 1,330 also the purchases). Kept wins; no late land / tomato plantings (V219
   scans the WHOLE library for those and one such route would switch V219 off in every game). Batch a: rated >= 2500
   (160, 57 worlds); batch b: every other win (643, 64 worlds). Source 2 (running): the full GM dataset via its
   stream hashes (`python/routes/gm_routes.py`, laptop): 20,184 seats share route 0's first 136 steps exactly; 1,563 distinct winning streams rated >= 2400; 1,284 new and without late land/tomato (batch c, screened in 3 base chunks v61.1c1..c3).
2. Lineage screen (A3): each candidate forced in its realized world, 6 bank seeds (train split, index 100..105) x
   both seats x 6 lineage opponents (v63, v62.1, v62, v61.1, v63.1_rl, random-knob clones; mirror excluded).
   `python/routes/screen_routes.py`. Most candidates LOSE: median net -44 (batch a), -30 (batch b); foreign tapes
   under our layers play worse (as the main repo's transplant test found). 14 candidates net >= +4.
3. Public screen (finalists only): the top-25 public agents closed loop on the same world seeds
   (`python/routes/screen_public.py`).
4. Confirm (A4): fresh seeds (bank train 20..49 + the 3 held-out seeds) x both seats x 6 lineage opponents, and the
   991-tape real-player band gate vs v63.1_rl (`python/routes/confirm.py`).

## Per-world log
| world | route (team, key) | lineage screen | public-25 screen | held-out / fresh lineage | band (<2500 losses) | verdict |
|---|---|---|---|---|---|---|
| BRUNCH_SPOT\|BRUNCH_SPOT | 4256 (dataset, 108857546_0) | +6/-0 | +0/-0 (train), +0/-0 (held-out) | +22/-10 (33 seeds) | 27 vs 28 (with 2604: 27, +18/-6 vs v63.1_rl) | ADDED (stage 2, 16:20 IST) |
| PIZZA_SHOP\|PIZZA_SHOP | 109023754_1 (dataset) | +8/-0 | +2/-24 | +33/-24 | - | rejected (public) |
| PIZZA_SHOP\|PIZZA_SHOP, YARN_STORE\|BRUNCH_SPOT (x2) | 108794516_0, 107877287_1, 107863236_0 | +6..+8 | - | +24/-46, +12/-25 | - | rejected (fresh seeds) |
| PET_CAFE\|FARMERS_MARKET | 2604 (Navier-stokes, 111253030_0) | +10/-0 | +6/-0 | +51/-38 (33 seeds) | 28 vs 28 | ADDED (stage 1, 15:12 IST) |
| BRUNCH_SPOT\|PET_CAFE | 109237719_0 (DECEM) | +10/-0 | +20/-2; held-out +12/-10 | +35/-35 (33 seeds) | 29 vs 28 | rejected (flat lineage, band -1) |
| PET_CAFE\|FARMERS_MARKET (2nd) | 108419076_1 (dataset) | +10/-0 | - | +34/-120 | - | rejected (fresh seeds) |
| PET_CAFE\|PIZZA_SHOP (x2), ICE_CREAM_SHOP\|PET_CAFE, YARN_STORE\|BRUNCH_SPOT | 108776890_1, 108638190_0, 108740485_0, 108041846_1 | +4..+10 | - | +24/-128, +14/-149, +28/-55, +12/-25 | - | rejected (fresh seeds) |
| PIZZA_SHOP\|BRUNCH_SPOT, YARN_STORE\|ICE_CREAM_SHOP | 108336856_1, 108754362_0 | +6, +4 | - | +8/-203, +5/-67 | - | rejected (fresh seeds) |
| YARN_STORE\|BAKERY (x2) | 107857197_1, 107876284_0 | +8/-0 | +0/-34, +0/-41 | +34/-9, +33/-11 | - | rejected (public) |
| YARN_STORE\|PET_CAFE | 1021 (111446955_0) | +14/-0 | +8/-0 | +4/-8 | - | rejected (held-out) |
| PET_CAFE\|BAKERY | 109389905_0 (kigasudayooo) | +14/-0 | +2/-0 | +4/-4 | - | rejected (held-out flat) |
| PET_CAFE\|PIZZA_SHOP | 111341303_0 (offhand) | +8/-2 | 0/0 | +4/-12 | - | rejected (held-out) |
| YARN_STORE\|ICE_CREAM_SHOP | 107154725_1 | +6/-0 | 0/0 | 0/0 | - | rejected (no effect) |
| PIZZA_SHOP\|PIZZA_SHOP | 108794967_1 (kigasudayooo) | +8/-0 | +27/-52 | - | - | rejected (public) |
| ICE_CREAM_SHOP\|PET_CAFE | 110092680_1 | +6/-2 | +0/-40 | - | - | rejected (public) |
| PET_CAFE\|YARN_STORE | 111791196_0 | +4/-0 | +6/-22 | - | - | rejected (public) |
| PIZZA_SHOP\|BRUNCH_SPOT, SMOOTHIE_SHOP\|BAKERY, YARN_STORE\|SMOOTHIE_SHOP (x2) | | +4..+10 | 0/-2 .. +2/-6 | - | - | rejected (public) |

Note: a 6-route table (all six lineage+public survivors) scored 29 band losses < 2500 vs 28 and mostly failed the
held-out lineage check: the screen's per-world winners are partly selection noise at 6 seeds x 6 opponents, so every
route needs the 33-seed confirmation before it goes in.

## Result (27 Sep 17:10 IST)
2,087 candidate routes screened (160 + 643 + 1,284), all 64 worlds covered by candidates. Lineage-screen median net
-30 to -54: tapes recorded by other players (even with our exact opening) lose under our layers in most worlds. 25
finalists re-tested on 33 fresh seeds: 5 held; 3 of those lost to the public-25 field. Kept 2 worlds:

  configs/route_tables/lib_v2.json = {"PET_CAFE|FARMERS_MARKET": 2604, "BRUNCH_SPOT|BRUNCH_SPOT": 4256}
  base: configs/bases/v61.1r2 (required: the ids exist only there)

Combined table, v63.5_rl agent: band 27 losses < 2500 (v63.5_rl 28), 2500+ win rate 0.940 (= 0.940), paired vs v63.1_rl
+18/-6 (v63.5_rl +18/-7). Lineage on fresh seeds: PET_CAFE|FARMERS +51/-38, BRUNCH|BRUNCH +22/-10 (both of the
33-seed runs); public-25 neutral-to-positive. The other 62 worlds keep the v61.1 routes (byte-identical play: a
route that is never selected is never read).

Use: agent / tapeplay / selfplay `--base configs/bases/v61.1r2 --route-table configs/route_tables/lib_v2.json`.
Not done: letting PPO choose the route (the route is fixed at step 144, before PPO's first decision on day 6).

## RC pre-test (27 Sep 17:50 IST, Q130) -- lib_v2 REJECTED for the release

Paired vs v63.6_rl (PPO i910, big1 shell, chain-off r127,sm,r95) on the held-out 64-world lineage panel (mirror excluded):
`--base configs/bases/v61.1r2 --route-table configs/route_tables/lib_v2.json` = **+4/-36 (p < 0.001)**, worse against
v63 (0/7), v62.1 (0/5), v62 (0/8), v61.1 (0/8), rand (0/5); only v63.1_rl +4/-3. Public-25 losses +0/-0; real tapes
ladder +3/-0, band +1/-0; band gate 60 vs 61 below 2500 (rebuilt 1,100-tape band set). The screen's positives were measured with
the v63.5_rl stack on 6-33 seeds per world; on i910 over the full held-out panel the change loses. Not shipped.
