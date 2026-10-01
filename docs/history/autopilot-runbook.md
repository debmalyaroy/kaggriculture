# Autopilot & pipeline — architecture + complete working guide

The autopilot is the continuous, self-improving Slot-2 pipeline. It trains one neural
policy, **builds BOTH harnesses (bandit + trackp) from it**, **gates** them against a
held-out reference panel across diverse worlds, **locks the winning config**, and emits
a **release candidate** + submit checklist. It **never submits** — you submit from the
private notebooks. It resumes each cycle, grows a self-play league, generates its own
gameplays, and can fold in downloaded data — so it keeps improving, then **stops when
it converges**.

- Entry point: `python -m kaggriculture.pipeline.autopilot`
- Setup: `PYTHONPATH="src;vendor"`, `PYTHONUTF8=1`, the `llm` env python
  (`C:/ProgramData/anaconda3/envs/llm/python.exe`) — torch 2.11+cu128, onnxruntime.

---

# PART 1 — ARCHITECTURE

## 1.1 The one idea
There is **one learned economy** (a neural macro-policy) and **two ways to ship it**.
The CODE is stable; the daily job finds the OPTIMAL **config** via train→test→improve,
locks it on a passing gate, and ships it. The target is the measured ~100k ladder
economy: a **wheat-fed cow herd** (milk $264) + land day ~6 + ~10-12 crew + strawberry/
carrot cash crops + a few melon (don't-dump).

## 1.2 The full pipeline (data → policy → two seats → gate → release)

```
 DATA                         TRAIN (one NN)                         SHIP (two seats)
 ─────                        ──────────────                         ────────────────
 download_supervisor ─┐       bc_warmup (BC)                         BANDIT seat
   (top-200, CDN)     ├─► corpus ──► return-conditioned  ─┐          compile policy→tape
 bc_corpus            │    macro    Transformer            │          + rails/dispatch
   (440k macro rows)  │    parquet  (models/rl/bc_policy)  │          D6/D12/D15/D21/D27
 opponent_pool ───────┘                                    │          → build_rust_bandit
   (786 ≥2500 tapes)          macro_rl (PPO self-play) ────┤            → musl tarball
 macro_selfplay ◄──┐         ├ SeasonCbsController exec    │          TRACK-P seat
   (self-distill)  │         ├ BatchRoller rollouts        │          policy→ONNX/npz
                   │         │   (7.7ms/game, kagg batch)  │          + native_infer
 world_freq +      │         ├ WorldSampler (T2/T3)        │            (numpy==torch)
 seed_bank ────────┼───────► ├ self-play LEAGUE (T1)       │          → weights bundle
 (diverse worlds)  │         ├ learned critic (A5)         │
                   │         └ teacher-KL + gated promote ─┘                 │
 self-play games ──┘              → models/rl/rl_policy.pt                    ▼
   (tape+world gen)                        │                        ══ GATE (both) ══
                                           └──────────────────────► full_gate vs the
                                                                     crown panel (15
                                                                     banded REACTIVE
                                                                     refs) on Rust serve
                                                                     across diverse worlds
                                                                            │
                                                     ship? ◄─────────────────┘
                                                       │ pass                │ fail
                                                       ▼                     ▼
                                              release.lock_config      keep training
                                              + RELEASE_CANDIDATE      (convergence ladder)
                                              + SUBMIT_INSTRUCTIONS
                                              (you submit; never auto)
```

## 1.3 Components (module → what it does)

**Data**
- `data.bc_corpus` → `data/bc_corpus/macro.parquet` (+ `macro_destbreso.parquet`,
  `macro_selfplay.parquet`): the BC targets. **Rows = games × ~30 days.** Baseline was
  **440k rows ≈ 14.7k games** (top-100 800 + destbreso ≥2300 13,876). A strong field
  entrant used **300k+ games for BC**, so scaling the corpus matters. Levers to ~300k:
  (1) full destbreso (45,404 games avail — lower `--min-rating`); (2) drain the download
  (72,115 episodes queued ≈ 144k game-seats, via `--fetch`); (3) self-distillation
  (`macro_selfplay`, unlimited); (4) more public datasets.
- `models/opponent_pool_full.json`: 786 ≥2500 real opponent tapes (RL league seeds).
- `train.worlds`: `build_world_freq` → `world_freq.json` (which shop-worlds the ladder
  visits); `WorldSampler` (sample a target world for the policy); `build_seed_bank` →
  `seed_bank.json` (32 seeds across 35 worlds — the diverse gate + rollout worlds).
- `.local/scratch/gm/download_supervisor.sh`: CDN-aware; drains new top-200 replays
  on the daily 429 reset (started by `--fetch`).

**Train (one NN)**
- `train.bc_warmup`: return-conditioned masked-causal Transformer; supervised on the
  macro corpus (incl. self-play), BF16 on the 4060; val-split early-stop; writes
  `models/rl/bc_policy.pt`.
- `train.macro_env`:
  - `SeasonCbsController` — the executor (persistent-cluster PC-TAPF routing + wheat-fed
    husbandry for cow/goose/sheep + land expansion + fertilizer + don't-dump sells).
  - `BatchRoller` — compile a plan → tape (once) and replay vs many opponents via Rust
    `kagg batch` (~7.7 ms/game); **bounded parallel compile** (persistent pool, cap 6);
    `rollout_many` = one all-cores batch for many plans.
  - backends: `serve` (bit-exact, for compile + the gate), `batch` (fast rollouts).
- `train.macro_rl`: PPO self-play — autoregressive plan sampling, learned critic (A5),
  teacher-KL to the frozen BC prior, gated promotion, the **self-play league** (T1,
  `models/rl/league/*.tape`), world-conditioning (T2). Writes `models/rl/rl_policy.pt`
  (+ `rl_policy_best.pt`).
- `train.selfplay_corpus`: generate gameplays (tape+world gen) → keep the ones that
  **WIN vs real opponents** → distil into `macro_selfplay.parquet` (self-distillation).
- `train.macro_actions.sanitize_plan`: plan guardrails — cash-feasible land, wheat feed
  scaled to the herd (CO2), early-income crop.

**Ship — two seats (both built every cycle)**
- BANDIT: `slot2_loop.compile_base_tape` (policy→tape) → `bandit.build.build_rust_bandit`
  (musl `kagg` + config.json + branches.json + base.tape → tarball). Runtime dispatches
  at D6/D12/D15/D21/D27 + 6 reactive rails.
- TRACK-P: `train.package_policy` (ONNX + parity) + `train.native_infer` (pure-numpy
  forward == torch; `save_bundle` → `trackp_weights.npz`). Runtime runs the NN live.

**Gate + release**
- `measure.eval_harness.full_gate`: smoke + banded seat-swap vs the crown panel's 15
  REACTIVE refs (tapes excluded — inert off-world) on **Rust serve** (bit-exact,
  3.5× faster) across diverse `seed_bank` worlds; ladder-weighted aggregate + per-band
  non-regression → ship/hold.
- `pipeline.release`: versioning (v59 major/day, minor/intraday), `lock_config` (the
  locked config IS the release), `submit_checklist` (per-harness private notebook).

**Orchestration**
- `pipeline.slot2_loop` (stages: BC / RL / package / gate / compile-tape / build).
- `pipeline.daily_slot2` (one day: data → train → bandit → trackp → gate → lock → report).
- `pipeline.autopilot` (the continuous, self-improving, self-stopping loop — below).

---

# PART 2 — THE AUTOPILOT CONTROL FLOW

## 2.1 One cycle
```
[optional] data refresh (every --refresh-every)   fold new downloads → re-warm BC
[optional] self-distill (every --distill-every)    generate gameplays → grow BC corpus → re-warm
TRAIN   stage_rl (resume; league grows)            → rl_policy.pt (+ best)
GATE    package + build BOTH seats + full_gate     per-harness verdict/agg/bands
RELEASE lock configs on pass; write candidate      RELEASE_CANDIDATE_* + SUBMIT_INSTRUCTIONS
STATUS  write AUTOPILOT_STATUS.{json,md}           after every phase
```

## 2.2 The convergence ladder (how it improves, then stops)
Every rung is tied to the **held-out gate** (the overfit guard):
1. **Improving** (new best gate > `--min-gain`) → continue; save `rl_policy_best.pt`.
2. **Stalled → KNOBS** (`--escalate-after` cycles no gain) → escalate budget, lower
   `--kl-coef`, more exploration; every 3rd escalation a bigger fresh BC. Up to
   `--max-escalations`.
3. **Stalled → DATA** (escalations done + `--patience` no gain) → **DATA BOOST**:
   self-play distill (generate gameplays) + fold downloads (`--fetch`) + re-warm BC →
   **RESUME**. Up to `--max-data-boosts`.
4. **Converged** (even data doesn't help) → **STOP** (`state: converged`), best
   checkpoint kept.

**Overfit watch:** `train_win − gate_agg > --overfit-gap` → `overfit_warn` in status;
counts as no-gain (can't run away); ship checkpoint is always the best-gate one.

---

# PART 3 — WORKING GUIDE

## 3.1 RUN
Foreground: `$PY -m kaggriculture.pipeline.autopilot`
Detached (true autopilot):
```
# bash:  nohup $PY -m kaggriculture.pipeline.autopilot --fetch --refresh-every 6 > .local/scratch/autopilot.log 2>&1 &
# PS:    Start-Process -WindowStyle Hidden $PY -ArgumentList "-m","kaggriculture.pipeline.autopilot","--fetch" -RedirectStandardOutput ".local/scratch/autopilot.log" -RedirectStandardError ".local/scratch/autopilot.err"
```
Key flags: `--iters/--games-per-iter/--games-per-plan` (budget), `--bar 0.9` (ship bar),
`--gate-seeds 2` (fast per-cycle gate), `--fetch` (download), `--refresh-every N`,
`--distill-every N`, `--patience/--max-escalations/--max-data-boosts` (stopping),
`--cpu`. Compile parallelism hard-capped at 6.

## 3.2 MONITOR — "how do I know?"
`$PY -m kaggriculture.pipeline.autopilot --status` prints `AUTOPILOT_STATUS.md`:
- **State / Phase / Cycle** (e.g. `running / gate / 7`).
- **Release candidates** — a **version** here (e.g. `BANDIT: v59`) means a submit is
  ready. Also drops `.local/candidates/RELEASE_CANDIDATE_<ver>_<harness>.md`.
- **Last gate result per harness** — `verdict` (ship/hold), `aggregate`, `bands`.
- Training curve, league size, escalations, data boosts, overfit warnings, heartbeat.
Raw: `models/rl/AUTOPILOT_STATUS.json`, `.../AUTOPILOT_HEARTBEAT` (stale > 15 min ⇒
check), `.local/scratch/autopilot.log`.

## 3.3 FIX
- `state: error` → status has `error` + `traceback`; `autopilot.err` full trace.
  OOM → lower `--d-model`/`--games-per-iter` or `--cpu`. Docker down → bandit build
  skipped (loop continues). Missing binary → ensure `rustengine/kagg.exe` + the
  isolated `kagg-engine.exe`.
- Heartbeat stale, no `done/stopped` → process died; fix + **re-run** (resumes from
  `loop_state.json` + `rl_policy_best.pt`; no progress lost).
- Gate stuck at 0/hold → training, not a bug (policy not yet competitive). The ladder
  escalates then data-boosts automatically.

## 3.4 STOP
Graceful: `$PY -m kaggriculture.pipeline.autopilot --stop` (STOP flag; exits at the
next phase, `state: stopped`). Hard: kill the process. Re-run to resume.

## 3.5 GET RESULTS / SUBMIT
When `--status` shows a release candidate:
1. Read `.local/candidates/SUBMIT_INSTRUCTIONS.md` (locked config + notebook + version).
2. **BANDIT** → `kaggriculture-private-submission-bandit`: artifact
   `.local/candidates/slot2_bandit_*.tar.gz` (self-contained: `kagg`+config+tape).
3. **TRACK-P** → `kaggriculture-private-submission-trackp`: `models/rl/trackp_weights.npz`
   + numpy runtime + executor.
4. Set version **v59** (major=day, minor=intraday), Save & Run, submit that version.
5. Latest-2 rule: a submit evicts the older active agent — keep two strong agents.

## 3.6 TIMING (measured)
Per cycle ≈ train (~7 min, 10k games) + gate BOTH seats (dominant: ~9 min @ gate-seeds 4,
~4.5 min @ 2) + build (~1.5 min) = **~13-20 min/cycle**. Convergence ≈ **6-12 h**.
Fastest lever: `--gate-seeds 2` per cycle; wider gate only near a candidate.

## 3.7 GUARANTEES
One heavy job at a time; compile workers ≤ 6; never overwrites `rustengine/kagg.exe`
(isolated builds); the gate is bit-exact to the official engine; **never submits**.
