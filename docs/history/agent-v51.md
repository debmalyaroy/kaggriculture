# v51 pair — rebased economy + closed production loop (2026-09-11)

## Why v51

The live v50.1_bandit (56156111) and v50_trackp (56147229) converged at
~2224 / ~2469: identical 2000-2500-band failure (62.7% / 57.7%), losses
out-banked from day 7, 8 / 26 upsets. Root cause: the closed loop only
controlled SELL TIMING; production was open-loop replay of an economy that
loses to the current 2000+ field. Operator order: do NOT retire the closed
loop — fix it.

## The instrument: live-loss panel (`.local/livepool/`)

All 91 losses of the live pair were re-fetched; each winner's action stream
became an engine tape in its recorded world (`info.seed`), certified by
replaying both recorded tapes on the Rust engine to the dollar: **39
certified cells** (27 no-YARN / 12 YARN, 26 from the 2000-2500 band, 14
upsets). `build_panel.py` (builder), `panel_gate.py` (closed-loop gate),
`rank_bases.py` (base selection, search/holdout split), `fresh_gate.py`
(public gate with per-game fresh opponent loading), `record_pub.py` (record
a public agent into an engine tape), `evolve.py` (packages evolution).

## What was measured

1. **Packages evolution over the v50 tapes: zero.** 41+ generations, no
   improving mutation; marginal plant/land packages cannot close a −31k
   structural gap and the self-play empties map mislabels real worlds.
2. **Base ranking (search half → holdout half):** the recorded
   **yhay81/shop-router-0911-simple** schedule (a public kernel that plays
   ONE fixed world-independent schedule; recordings from 5 different
   worlds/opponents byte-identical) scores **0.833 / +34.5k** on held-out
   no-YARN loss cells vs v50_egg's **0.038 / −15.4k** (p = 0.002). On YARN
   cells it is score-equal to v50_yarn (0.40 vs 0.35, p = 1.0) with +22k
   better margin. Both v51 fork tapes = this schedule (fork machinery kept).
3. **Packages evolution over the NEW base: zero again** (egg already 0.909
   on the search half — no headroom this panel can see).

## v51_trackp (compiled lane)

`agents/v51_trackp.py` — v50 layer stack (F1 cash floor 180, sweep ≥40,
quantity-conserved one-turn front-run from step 145, final dump 700, day-6
fork) plus:

- **F-B weed-repair validator** (`KAGG_WREPAIR=0` disables): a tape op that
  would silently no-op on a WEED tile (WATER/HARVEST/PLANT/BUILD_*) becomes
  DIG; a displaced PLANT/BUILD replays next turn. Port of the bandit's gated
  `_weed_repair` pattern.
- **F-D opponent-pressure escalation** (`KAGG_ESC=0` disables): the opponent
  farm is public; at day boundaries from day 7, if their bank leads ours by
  > `KAGG_ESCGAP` (800), the sweep threshold tightens to 24 and the
  front-run lookahead widens to 2.
- Ablation: F-B and F-D are score-neutral on the panel (0.769 all configs,
  margins within noise) — shipped as measured-zero-cost insurance.

Rust port: `rustengine/src/bin/agent.rs` (tapes in
`rustengine/agentdata/trackp_{egg,yarn}.json`, v50 backups kept as
`*.v50.bak`). **Equivalence: 15,099 steps, 0 mismatches** (panel opponents +
self-play). musl static-pie binary via Docker rust:latest; tarball
`.local/candidates/v51_trackp_compiled/submission.tar.gz` (main.py transport
+ binary + full Python fallback).

## v51_bandit (route lane)

`agents/v51_bandit.py` = v50_bandit_fixed.py with `_ROUTE` swapped to the
rebased tape and every TAPE-SWAPPING layer baked off (d3_fork, d6_fork,
world_tail, family_counter, arms, dispatcher, tree_dispatch, sim_search —
they would swap play back to wool branches). All overlays kept: ML relay,
adaptive F2 flush (mirror-aware), weed_repair, F1 cash guard, market-timing
stack. Ships as Python (`.local/candidates/v51_bandit_python/`) — the relay
ML web remains uncompilable (v50 finding).

## Gate results (2026-09-11)

| gate | v51_trackp | v51_bandit | v50 baseline |
|---|---|---|---|
| live-loss panel (39 cells) | **0.769** (+37,422), upset-risk lost 6/28 | **0.705** (+36,969), lost 6/28 | 0.115 / 0.141, lost 26 / 24 of 28 |
| paired vs v50 | +0.654, p<0.0001 | +0.564, p<0.0001 | — |
| publics+clones (in-process gate, 11 opp) | 176-0 | — | v50 lost 0-16 to v49.1-bandit clone |
| publics fresh-loaded per game | 80-0 (5 opp, 0 opp-errors) | 136-24 (10 opp) | — |
| official engine self-play | DONE/DONE 106,108 | DONE/DONE 105,020 | 103,372 |
| Rust≡Python | 15,099 steps 0 mismatch | n/a (Python) | 30,198 / 0 |

**Known weakness (documented, accepted):** v51_bandit loses the near-mirror
4-12 vs yhay81-0911-simple itself (−951/game; relay-off is worse 2-14,
flush-rate irrelevant) and 4-12 vs v51_trackp (−483). v51_trackp beats that
family 16-0 — the pair covers it. Opening prefix = a public kernel's, so
clone/mirror exposure is by construction; mirrors measure as ties/near-ties,
not upsets.

## Shipping

Pushed as **v2** of the two PRIVATE notebooks (`is_private:true` verified
before push): `debmalya84/kaggriculture-private-submission-bandit`,
`debmalya84/kaggriculture-private-submission-trackp`; submitted to the
competition from those notebooks (operator-ordered 2026-09-11, bandit first).
