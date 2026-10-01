# Repository layout

One repository since 2026-10-01 (the former `kaggriculture-rl/` sub-project was merged into the root; its stale
`harness/` copy of the Python package was dropped). The repo holds **code, small configs and docs only** — every
dataset, tape, replay, route table, weight file, build and generated agent is regenerated on demand (see
[data.md](data.md)) and is gitignored.

```
.
├── AGENTS.md / CLAUDE.md      agent working rules (CLAUDE.md points to AGENTS.md)
├── Cargo.toml / Cargo.lock    Rust workspace: members crates/*; rustengine*/research crates are excluded
├── pyproject.toml             the `kaggriculture` Python package (src/)
├── requirements.txt           kaggle-environments + scientific stack
├── crates/                    Rust workspace (the v63.x agent)
│   ├── agent/                 agent library + `agent-stdio` (the submitted binary): chassis, router, managers, layers,
│   │                          shells, dispatcher, endgame model. data/v92_lib.bin is a compile-time asset (include_bytes!)
│   ├── engine/                `kagg-engine`: bit-exact Rust port of the official engine (kaggle-environments 1.32.x)
│   ├── runner/                all-Rust game tools: chassis-vs-tapes, tapeplay, trace, chassis-screen/-repair/-grid,
│   │                          league-rand, selfplay, ppo-rollout, stdio-match, labelers (endg/gt/buy), ...
│   ├── policy/                pure-Rust forward pass of the macro policy (policy-check)
│   ├── dayobs/ features/      per-day observation vector + features shared by corpus and agent
│   ├── corpus/                corpus-extract library (replays -> per-day Parquet corpus)
│   └── tools/                 slim-check (engine parity vs real replays)
├── rustengine/                legacy `kagg` engine port + CLI (serve, batch, play, prerank, bandit/trackp lanes);
│   ├── v62/                   used by src/kaggriculture (serve_match, prerank). Standalone cargo projects:
│   └── extractor/             rustengine/, rustengine/v62 (v62 agent port), rustengine/extractor (trackp-extract)
├── src/kaggriculture/         Python package (`pip install -e .`, absolute imports, ROOT from kaggriculture.paths)
│   ├── engine/                official-engine bridge, serve_match, parity / conformance / engine_check
│   ├── data/                  ladder + dataset fetch, ingest, features, mining, registry
│   ├── measure/               evaluate, win_metric (paired tests), gates, analysis
│   ├── train/ trackp/ bandit/ training, Track P planner, bandit lane
│   ├── agentbuild/            build agents / notebooks / submissions, chassis tools, model graphs
│   ├── pipeline/              orchestration, autopilot, release, submit (never auto-submits)
│   └── winplan/ experiments/  study code
├── python/                    RL-era Python tools (run from the repo root; RL = repo root)
│   ├── release/               pack_submission.py, bt_live.py (BT rank projection), live_loss_classify.py
│   ├── top50/                 top-team crawl/fetch (EpisodeService), shell training/eval, route/opening screens
│   ├── rshell/ v6312/         reactive-shell training, public-agent field harness (public25, field_chassis), panels
│   ├── routes/ ref/ endg/     route library builds/screens, base-data extraction, endgame model training
│   ├── learn/ tpp/            BC / PPO / oracle learners, top-player policy
│   └── *.py                   ladder_pull, replay_to_tape, loss_tapes/gates, opponent clustering, delta, ...
├── configs/                   agents/ (agent.json candidates), bases/, profiles/, rshell/, shield/, opp/, endg/,
│                              lineage/, fixes/, bandit/ ... (generated multi-MB tables are not kept)
├── scripts/                   build_submission.ps1 (stage + Docker cross-build), schedulers, download/dashboard
│   ├── bandit/                bandit-lane builders
│   └── trackp/                Track P box/cloud scripts
├── kaggle/                    notebook + kernel templates
│   ├── private_kernel/        the private submission notebook (payload sha check + self-play validation)
│   ├── payload_dataset/       dataset-metadata.json for debmalya84/kaggriculture-rl-payload
│   ├── submission/            main_config.py (the v63.x --config bridge), main_template.py (older flag bridge)
│   └── gate/ delta/ league/ kernels/ bin_dataset/   templates for Kaggle-side jobs
├── ops/ aws/                  task queue runner + checks; AWS box bootstrap/sync scripts
├── agents/                    hand-written source agents only (v1_heuristic.py); built agents are generated here
├── dashboard/                 local control-panel server
├── tests/                     test suites (tests/run_all.py)
├── research/                  research notes; rustengine-or/ (standalone MILP tape-generator crate + tools)
└── docs/                      documentation; history/ = dated notes, journals, version write-ups
```

## Conventions
- **Generated output** goes to `data/`, `models/`, `weights/`, `target*/` or `.local/` — all gitignored, all created
  on demand by the code that writes them.
- **Scratch** (logs, candidate configs, experiment output) goes to `.local/`. Assistant memory is `.local/memory/`.
- Python tools in `python/` compute the repo root as `RL` from their own location; run them from the repo root.
- New Rust code goes into a `crates/` member; new Python package code into `src/kaggriculture/<subpackage>`;
  one-off study scripts into `python/` (or `.local/` if throwaway).
