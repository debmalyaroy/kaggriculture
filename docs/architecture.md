# Architecture

This is the system view: the game, the components, how data flows between them and which tool does what. The agent
itself is described in depth in [agent-architecture.md](agent-architecture.md).

## 1. The game in one paragraph
Two farmers, 720 turns (30 days × 24 hours). Each turn an agent sends a farmer action, one action per hired hand
(positionally aligned with `farms[me]["hands"]`) and up to 10 market orders. Crops need watering, animals need
structures and care, products go to the shed and are sold into a shared market whose prices react to both players'
sales and to the town's drain. Shops unlock during the game and change demand; the **world** (first two shops) is
realized by play — the shop draw shares an RNG with weeds, so both action streams change it (first shop stable for
splits ≥ 74, both for ≥ 146). The final bank balance decides the game; the ladder rates win/loss/draw. Rules:
[game-rules.md](game-rules.md).

## 2. Components

```
 ┌──────────────────────────────────────────────── Kaggle ────────────────────────────────────────────────┐
 │  main.py bridge ──JSON lines──▶ agent-stdio (crates/agent, static musl)  ·  private kernel validation   │
 └───────────────▲─────────────────────────────────────────────────────────────────────────────────────────┘
                 │ submission.tar.gz (python/release/pack_submission.py, scripts/build_submission.ps1)
 ┌───────────────┴────────────────────────── Rust workspace (crates/) ─────────────────────────────────────┐
 │ agent    chassis + router + managers + 65-stage layer chain + shells + rival/endgame layers            │
 │ engine   kagg-engine: bit-exact port of the official interpreter (state, step, market, RNG, worlds)    │
 │ runner   all-Rust games: tape tournaments, screens, traces, leagues, labelers, PPO rollouts             │
 │ policy · dayobs · features · corpus · tools (slim-check)                                               │
 └───────────────▲──────────────────────────────────────────────────────────────▲──────────────────────────┘
                 │ tapes, routes, configs, labels                               │ kagg serve / batch / prerank
 ┌───────────────┴──────── python/ (RL-era tools) ────────┐   ┌────────────────┴──── src/kaggriculture ─────┐
 │ ladder pulls, replay → tape, top-team crawl, route     │   │ official-engine bridge, data fetch/ingest,  │
 │ screens, shell/endgame/policy training, field harness, │◀──│ measurement (win_metric), pipeline,          │
 │ release helpers (pack, BT projection, loss classes)    │   │ agent builds, Track P / bandit lanes         │
 └────────────────────────────────────────────────────────┘   └──────────────────▲──────────────────────────┘
                                                                                  │
                                                 rustengine/ — legacy `kagg` engine port + CLI
```

## 3. Rust workspace (`crates/`)

### `engine` (`kagg-engine`)
A byte-identical port of the Kaggle interpreter (kaggle-environments 1.32.x):

| Module | Role |
|---|---|
| `state.rs` | engine state mirroring the Python observation exactly (farms, tiles, private shed/seeds/inventories, market, town) |
| `engine.rs` | the step function, a line-by-line port of the interpreter's mutation (unit actions, market lockstep, decay, production) |
| `market.rs` | the price model (base, inventory curve, hinges) |
| `rules.rs` | pure rule functions (crops, animals, land, hires, shed access) |
| `mt19937.rs` | CPython's `random.Random` reproduced exactly (weeds, shop draws) |
| `world.rs` | realized worlds: which shops an episode actually unlocks |
| `obsjson.rs`, `obsstate.rs`, `loadstate.rs` | observation JSON ↔ state (per seat, full state, state injection) |
| `tape.rs` | the tape action format and whole-episode rollouts |
| `features.rs`, `policies.rs`, `json.rs` | generic seat features, synthetic test policies, dependency-free JSON |

Parity: `crates/tools` `slim-check` replays real games and requires exact banks.

### `runner` — all-Rust games (`target/release/<bin>`)
| Binary | Purpose |
|---|---|
| `chassis-vs-tapes` | our agent (or a bare chassis / forced route) in one seat of every recorded game vs the recorded opponent; traces via `KRL_TRACE_DIR`, `KRL_LAYER_TRACE` |
| `tapeplay` | replay a real game's world against the opponent's recorded stream; `--verify` reproduces the replay exactly |
| `trace` | per-turn state traces of recorded games (both streams) |
| `chassis-screen`, `chassis-repair`, `chassis-grid` | per-world route search, per-(world, rival key) route overrides, guard-settings grid |
| `selfplay`, `stdio-match` | base vs base on the engine; tournaments between external `agent-stdio` builds |
| `league-rand`, `ppo-rollout`, `search-eval` | macro-policy training data, PPO rollouts, play-time search evaluation |
| `branch`, `branch2`, `buylabel`, `endg-label`, `gt-label` | exact-engine labels for the sales shell, reactive shell, buy head, endgame controller, game-theory layer |
| `bcdump`, `tppcheck` | top-player policy dataset and train/serve equivalence |
| `lineage-table` | farm layouts of every library route (lineage matching) |

### Other crates
`policy` (pure-Rust forward pass of the macro policy, `policy-check`), `dayobs` (per-day observation vector shared by
corpus and agent), `features` (per-day features from parsed observations), `corpus` (`corpus-extract`: replays →
per-day Parquet corpus), `tools` (`slim-check`).

## 4. Legacy engine CLI (`rustengine/`)
The older port that the Python package drives: `kagg serve` / `vecserve` (stdio environment for Python agents —
full-fidelity matches, exact vs official), `kagg play` (planner skeleton/search), `kagg bandit` / `trackp` / `mbandit`
(compiled seat lanes, tapes read at run time from `$KAGG_TAPE_DIR` / `$KAGG_AGENTDATA_DIR`), `selftest`, `rules`.
`rustengine/v62` is the v62 agent port, `rustengine/extractor` the trackp replay extractor. Standalone cargo projects
(excluded from the root workspace).

## 5. Python package (`src/kaggriculture`)
`pip install -e .`; absolute imports; the repo root comes from `kaggriculture.paths.ROOT`.

| Subpackage | Key modules |
|---|---|
| `engine` | `engine_check`, `conformance`, `ladder_parity`, `reactive_parity`, `serve_match`, `run_match`, `rust_prerank`, `official_eval`, `forced_env` |
| `data` | `episodes`, `sameday` (quota-guarded replay fetch), `ourgames`, `routes`, `registry`, `features`, `turn_features`, `stream_hashes`, `leaderboard_harvest`, `policy_dataset` |
| `measure` | `win_metric` (paired tests, flips), `evaluate`, `ab_test`, `elo`, `gauntlet`, `intraday_gate`, `loss_analysis`, `loss_autopsy`, `determinism`, `referee_power` |
| `pipeline` | `autopilot`, `pipeline`, `refresh_cycle`, `release`, `submit` (never auto-submits), `live_board`, `dashboard` |
| `agentbuild` | `build_agent`, `build_submission`, `build_notebook`, `model_graph`, router agents |
| `train` | `tune`, `optimize`, `cmaes`, BC / RL / supervised trainers, `train_gates` |
| `trackp`, `bandit` | the Track P planner lineage and the bandit seat lane (separate by design) |
| `winplan`, `experiments` | public-agent harvest and study code |

## 6. RL-era tools (`python/`)
Run from the repo root; each computes `RL` = repo root from its own path.

| Area | Modules |
|---|---|
| Live games | `ladder_pull.py`, `ladder_analyze.py`, `ladder_study.py`, `replay_to_tape.py`, `loss_tapes.py`, `loss_rca.py` |
| Top teams | `top50/fetch.py` (EpisodeService crawl + tapes), `top50/route_screen.py`, `opening_screen.py`, `layer_screen.py` |
| Routes | `routes/gm_routes.py`, `build_bases.py`, `screen_routes.py`, `confirm.py`, `final_base.py` |
| Shells / endgame / policy | `top50/train_shell.py`, `rshell/train.py`, `endg/train.py`, `learn/*` (BC, PPO, oracle), `tpp/*` |
| Field harness | `rshell/public25.py`, `v6312/field_chassis.py`, `field80.py`, `panel_gate.py` |
| Opponents | `opp_cluster2.py`, `opp_lineage.py`, `opp_corpus_sig.py` |
| Release | `release/pack_submission.py`, `release/bt_live.py`, `release/live_loss_classify.py`, `tarball_check.py` |
| Kaggle jobs | `delta.py`, `gate_kaggle.py`, `league_kaggle.py` (+ `kaggle/` templates) |

## 7. Data flow

```
 Kaggle replays ──ladder_pull / top50 fetch──▶ tapes (data/ladder, data/tapes)
        │                                            │
        └──corpus-extract──▶ per-day corpus ──▶ route library (python/routes) ──▶ chassis base (routes.json, router.json)
                                                     │                                   │
                                     labels (branch*, endg-label, gt-label)       chassis-screen / chassis-repair
                                                     │                                   │
                                        training (python/top50, rshell, endg, learn) ──▶ configs (shell, rshell, endg, gt,
                                                                                          policy, profiles, preempt)
                                                                                                  │
                                       candidate = agent.json + base + configs ◀───────────────────┘
                                                     │
                  gates: chassis-vs-tapes (all tapes, losses, held-out) · live replays · field (public25 / field_chassis)
                                                     │  paired sign test, same-world split
                                                     ▼
                         pack_submission ──▶ payload dataset ──▶ private kernel (validation) ──▶ submit
```

## 8. Measurement model
- **Tape tournament**: our agent replaces one seat; the other seat's recorded stream is replayed (guarded). Fast and
  exact, but open-loop: valid only while the realized world is unchanged (same-world rule).
- **Live replays**: our own ladder games; with the submitted config they reproduce the live banks exactly — the
  strongest regression set.
- **Field**: Python opponents on the Rust serve engine; both sides react, worlds are real. The final arbiter for
  changes that alter the world (e.g. earlier sales).
- **Decisions**: paired, in wins, with the exact sign test; per-world keep/revert on held-out data.
Details: [workflows.md](workflows.md).

## 9. Release path
`pack_submission.py` stages `main.py` (`kaggle/submission/main_config.py`) + `agent-stdio` (Linux, Docker cross-built)
+ the candidate's config files into `data/builds/<name>/submission.tar.gz`; the payload goes to the private dataset,
the private kernel validates it with a self-play episode, and the submission is made from the kernel output.
Details: [release.md](release.md).
