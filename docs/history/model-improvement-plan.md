# Model improvement plan (agreed 2026-08-12)

The post-v24-release workstream for every trainable model. One variable at a
time; every change passes the three-tier gauntlet before it ships.

## Verdicts

| # | Component | Today | New algorithm | Trains on |
|---|---|---|---|---|
| 1 | Family clustering | greedy single-linkage, fixed 0.055 | HDBSCAN (fallback: agglomerative + silhouette-chosen cut) | local |
| 2 | Family registry | none — class ids shift every retrain | Hungarian matching on centroids vs yesterday, distance ceiling ⇒ "new family" | local |
| 3 | Identifier | multinomial logistic, raw GD, no scaling | GBDT bake-off: LightGBM vs XGBoost vs CatBoost, soft labels (cluster-distance targets), class weights; L-BFGS logistic (scaled, C-swept) stays as calibrated baseline | local or cloud CPU |
| 3b | Identifier DL challenger | — | tiny GRU / temporal-CNN (hidden ≤32) over the LIVE-OBSERVABLE event stream; challenger only — must beat #3 on walk-forward to proceed | local GPU (CUDA + PyTorch on this box) |
| 4 | Identifier calibration | none | temperature scaling fit on held-out day | local |
| 5 | relay_config (M2) | unweighted consensus, hardcoded support/qty | recency × final-bank weighted median per dump event + IQR confidence; per-product dump-size threshold | local |
| 6 | Gates threshold | one global mean-based threshold | empirical-Bayes shrinkage per family (Beta-Binomial correctness, Normal-Normal gain/cost) | local |
| 7 | Gates in-game commit | posterior ≥0.85 + streak | SPRT on per-turn posteriors | local |
| 8 | Market-duel RL | 1-param lead adaptation (relay_lead_rl) | offline Q-learning on discretized duel state → pure-python table; fitted Q-iteration with LGBM if the table plateaus, distilled back | local CPU |
| 9 | Surrogate | ridge on filename hashes (placeholder) | LGBM quantile regression on real genome/schedule + opponent-family features; rank by upper quantile | local |
| 10 | End-to-end deep RL / NN policy | — | REJECTED: no cross-episode memory in-game, combinatorial action space, mined routes already encode near-optimal play vs the actual meta | — |

Build order: 1→2 (stabilizes every downstream label), 3→4, 3b (optional), 5,
6→7, 8; 9 waits for the arms program to reactivate.

## Why no deep learning elsewhere

Tabular, ~100k rows × 276 dims: GBDTs are the empirically stronger and more
debuggable fit; a NN adds infrastructure without expected gain. The single
defensible DL slot is 3b, because the identifier's true input is a turn-by-turn
event SEQUENCE and hand-engineered prefix features may miss temporal patterns.
A 32-unit GRU is a few thousand multiplies per inference — exportable to pure
python within the ~20 ms turn budget (and can run every k-th turn).

## The three-tier gauntlet (how "better" is decided)

1. **Walk-forward offline** — train on days ≤ d, test on d+1, slide across the
   whole index. Metrics: macro-F1 (imbalanced families), log-loss (what the
   commit gate consumes), accuracy-by-prefix-turn (early > late). Challenger
   must win macro-F1 AND log-loss.
2. **Paired-seed panel** — two byte-identical agents differing only in the
   embedded weights; fixed opponent panel + relay-decision episode subset,
   both seats, same seeds. Noise floor $3k/game. No worse overall, better on
   the relay subset.
3. **Field A/B** — ship as an intra-day bandit version (v{x}.{y}_bandit) from
   the same notebook slug. Judge the bandit-minus-route rating differential
   (the route agent is the built-in control; differencing removes day
   effects) after 24h minimum — fresh submissions start ~600 provisional.
   Promote if the differential ≥ the incumbent bandit's; revert after 24h if
   clearly worse. Rollback = one submission from the preserved previous
   notebook version (last_release.json holds the pair).

One variable per submission. Never judge on unweighted ladder win rate
(house rule: matchmaking reads back the rating you already have).

## Verifying the adaptive-play (RL) layer

Five levels, cheapest first; each must pass before the next is worth running:

1. **Scripted-opponent scenario tests** — synthetic tapes with controlled dump
   schedules (shifted, missing, resized). Assert: adaptation fires after
   `_RELAY_OBS_MIN` observations, contested sells land ahead of the observed
   timing, non-contested products untouched. Permanent contract tests.
2. **Ablation A/B, paired seeds** — adaptation ON vs OFF, same seeds/seats, vs
   three pools: clones (reproduce ~+202/game), families whose real timing
   differs from M2 priors (adaptation must beat static M2 here), non-relay
   opponents (must measure ~$0 — do-no-harm).
3. **Decision-trace audit** — per-turn log of observed dumps / chosen lead /
   race decisions in local sims; every pulled sell justified by its own
   arithmetic (model-graph discipline applied to the layer).
4. **Counterfactual replay on real games** — rebuild off-schedule opponents as
   tapes from real episodes, replay ON vs OFF (the v23.1 diagnosis method).
5. **Field forensics** — mine our own replays for contested dump events
   (both sides dumped the same product within a few turns): who sold first,
   at what price delta. Plus the bandit-minus-route differential over 24-48h.

For the offline-trained duel policy add, before all of the above: **regret vs
oracle best response** on held-out family schedules — near-oracle timing on
schedules it never saw, beating static M2 within k observed dumps.

## Where things run

**The local box has CUDA + PyTorch (confirmed 2026-08-12), so ALL training —
including the GRU challenger — runs locally by default.** Cloud notebooks are
overflow capacity only. Kaggle notebook images have shipped a different
interpreter than the ladder (wrong prices, dropped actions — see
engine_check.py); simulation results from cloud notebooks are untrusted unless
engine_check passes in that image.

| Environment | Used for | Never for |
|---|---|---|
| Local (CPU + CUDA GPU) | everything: all training incl. GRU, panel sims (evaluate.py), export-equivalence tests, tests/, agent builds | — |
| Kaggle private notebook (GPU 30h/wk) | overflow training only; training bundle attached as private Kaggle Dataset; weights.json notebook output pulled back into models/ | any measurement without engine_check |
| Colab (GPU) | overflow training only, bundle via Drive | same |

Mechanics to build:
- `src/export_training_bundle.py` — pack X / y / dates / family labels into
  one compressed npz (~100 MB float32); useful locally for fast experiment
  iteration and required for any cloud overflow run.
- Every weight import path re-runs the export-equivalence check (stdlib
  inference == framework inference to 1e-6, as train_identifier.py already
  does) plus the walk-forward harness BEFORE any panel time is spent.

## Addendum 2026-08-13: trajectory rating scorer (item 11)

Operator proposal, accepted: a model that reads game trajectories (per-turn /
per-checkpoint features; the GRU seq_dataset infrastructure) and predicts the
LADDER RATING of the play, trained on archived episodes x the players' known
leaderboard ratings (win/loss as auxiliary head). Uses:
  a) candidate pre-ranking before the tournament (the surrogate's job, done
     with real signal),
  b) tie-break DIAGNOSTIC on saturated panels, reported beside the playoff,
  c) live collapse detection: score our own pair's hourly-scraped games; a
     predicted-rating drop flags a v23.1-style failure hours before the
     ladder rating develops.
GUARDRAIL (Goodhart): never the gate. Wins (playoff) + ladder A/B decide
what ships; the scorer ranks and warns. Validation: walk-forward Spearman
vs actual team ratings + retrospective must-pass (v23.1 trajectories at 1056
must score below v23_route at 2208).
