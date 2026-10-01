# Trackp BC+RL — cloud (vast.ai) run guide

End-to-end BC→RL on a rented GPU box, synced via Google Drive, fully hands-off
and crash-resilient (≤~3 min loss on any failure).

## Pieces (all built + smoke-validated)
- **Corpus (v2)**: `data/trackp_corpus_v2` — verb-aware action encoding (no
  item collisions), Rust-extracted (byte-parity gate passed), + self-play (16
  agents). Built locally by `pipeline.local_prep_v2`, pushed to `gdrive:trackp/corpus`.
- **BC trainer**: `trackp.bc_train` — token-transformer (~2.3M), parallel loader,
  AMP, resumable (`weights_only=False`, checkpoint every ~500 steps + quartiles),
  self-stop time budget.
- **RL self-play**: `trackp.rl_selfplay` — PPO actor-critic over N `kagg serve`
  envs, GAE, teacher-KL to BC, opponent pool (PASS + 7 gate killers + past self),
  reactive rails applied (train==deploy), resumable (~150s checkpoints).
- **Reactive rails**: `trackp.reactive_rails` — 9 config-driven levers
  (weed_repair, water_guard, budget_guard, endgame_liquidate, scarcity_sell,
  sell_premium, harvest_ready, feed_care, collect_fertilizer) + a mandatory
  `sanitize` last (desync-proof). Config: `configs/trackp_rails.json`. Add a
  gate with `@rail("name")`.
- **Gate**: `measure.tournament_gate` — enforces the profile ≈100% <2500 (any
  loss flagged) / 70–90% ≥2500; accepts `.pt` policies.
- **Orchestrator**: `pipeline.cloud_train` (BC→gate→RL500M→gate→RL1B→deploy) +
  `scripts/cloud_run.sh` (supervisor: restart-on-crash → resume).

## Local prep (this box, before renting) — automated
`python -m kaggriculture.pipeline.local_prep_v2` waits for the Rust re-extract,
finalizes, runs self-play, combines, and pushes corpus+code to `gdrive:trackp`.
Prereq for the push: `rclone config` a Google Drive remote locally.

## When to rent
Rent **after** `local_prep_v2` prints `COMPLETE` (corpus+code on gdrive).

## On the VM — your ONLY job
```bash
git clone <repo> && cd kaggriculture      # or: rclone copy gdrive:trackp/code .
rclone config                              # add a Google Drive remote named 'gdrive'
GDRIVE=gdrive:trackp WORKERS=16 bash scripts/cloud_run.sh
```
That single command: installs deps, builds the Linux `kagg` engine, pulls the
corpus + latest checkpoints, then runs the supervised BC→RL pipeline to
completion, syncing checkpoints to `gdrive:trackp/ckpt` every ~150s. On any
crash it auto-resumes from the last checkpoint. Pull the final `policy_rl.pt`
from `gdrive:trackp/ckpt` when it reports COMPLETE.

## Box spec (vast.ai)
RTX 3090 (24GB) · **≥16 CPU cores** (the RL/loader lever) · ≥48GB RAM ·
≥150GB NVMe · ≥200Mbps · reliability ≥0.98 · on-demand · CUDA 12.4+ / torch 2.x.

## Tuning rails without retraining
RL trains with `configs/trackp_rails.json`. To test knob combinations, re-gate
a trained `.pt` with a different rails file:
`python -m kaggriculture.measure.tournament_gate ckpts/policy_rl.pt` (edit the
config), or randomize rail params during RL for robustness.
