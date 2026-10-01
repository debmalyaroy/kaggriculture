# Trackp training runbook

Everything for the cloud RL flywheel: start / stop / resume / logs / metrics /
debug. The box is a vast.ai single-3090. **Automation never submits to the Kaggle
competition** — building artifacts, uploading the ckpt dataset, and pushing a
PRIVATE kernel are the authorized automated steps; the final *Submit* is your click.

## 0. Connection + layout

```
ssh -p <port> -i ~/.ssh/vast_knee root@<box-ip>        # (banner lines are noise)
```

| where | what |
|---|---|
| `/root/run_pipeline.sh` | crash-resilient supervisor (`while true` → `cloud_train`). `WORKERS=12 ENVS=256` |
| `/root/kaggriculture/` | the repo (code scp'd from local) |
| `…/ckpts/policy_bc.pt`, `policy_rl.pt` | latest checkpoints (full: weights+optimizer, ~70 MB) |
| `…/ckpts/policy_*_ep*_q*.pt` | milestone copies (25/50/75/100 %) |
| `…/ckpts/cloud_train.log` | stage START/DONE |
| `…/ckpts/pipeline.log` | trainer stdout (`[bc] …` / `[rl] …`) |
| `…/ckpts/metrics.jsonl` | **learning trace** (loss / meanR / win-rate / eps-s) written at every checkpoint |
| `…/ckpts/stages/` | stage completion markers |
| `gdrive:kaggriculture/ckpt` | **upload** (box, stage-end) / download base weights. `policy_bc.pt`, `policy_rl.pt` |
| `gdrive:kaggriculture/ckpt_archive` | **upload** (box, 30M). `policy_rl_step{N}_{UTC}.pt`, `policy_bc_final_{UTC}.pt` (never overwritten) |
| `gdrive:kaggriculture/delta` | **download** (box/laptop) — you upload `bc_corpus_delta_{epoch}.tar.gz` |
| `gdrive:kaggriculture/top_agents` | **download** (box) — you upload `top_agents_{epoch}.tar.gz` (roster + agent payloads incl. binaries) |

`epoch = YYYYMMDD_HHMMSS`. `ckpt`/`ckpt_archive` = box→gdrive (rclone on the box).
`delta`/`top_agents` = you upload from local (no rclone locally); the box pulls.

## 1. Start / stop / resume

**Start (single clean supervisor — always kill first so you don't double-run):**
```bash
pkill -9 -f run_pipeline.sh; pkill -9 -f cloud_train; pkill -9 -f bc_train; pkill -9 -f rl_selfplay
sleep 3
setsid bash /root/run_pipeline.sh >/dev/null 2>&1 </dev/null &   # detached
```

**Stop (fully):**
```bash
pkill -9 -f run_pipeline.sh    # kill the SUPERVISOR first, or it respawns the trainer
pkill -9 -f cloud_train; pkill -9 -f bc_train; pkill -9 -f rl_selfplay
```
> ⚠️ Killing only `cloud_train`/`bc_train` while `run_pipeline.sh` lives → the
> supervisor restarts a second trainer → **two trainings split the GPU** (symptom:
> throughput halves, `pgrep -c` doubles). Always kill `run_pipeline.sh` first.

**Resume:** just start again. `bc_train` and `rl_selfplay` auto-resume from their
own latest checkpoint (`global_step` restored). **Yes — stopping RL mid-run is
safe:** local checkpoints are written every ~150 s (and at the 1000-step / quartile
marks), so a stop loses ≤ ~2–3 min of steps and only the in-flight rollout buffer,
never trained weights. gdrive only holds the 30M-step archives (to save egress),
but a *same-box* resume uses the local checkpoint, so that's irrelevant to restarts.

## 2. Status + how to tell it's learning

```bash
# from local:
python scripts/trackp/box_status.py                 # BC%: seen/ETA ; RL: gstep/target + meanR/win/eps-s
python scripts/trackp/box_status.py --rl-target 1000000000

# on the box, the learning trace:
python -m kaggriculture.trackp.trainmetrics ckpts --kind rl --last 20
python -m kaggriculture.trackp.trainmetrics ckpts --kind bc --last 20
```
- **BC learning** = `loss` in `metrics.jsonl` (online CE on freshly-streamed shards; it
  tracks shard difficulty, not a val curve — judge it by *not diverging*, and by the
  BC gate after completion).
- **RL learning** = `meanR` / `winrate` trending **up** over gstep (the policy learning
  to win self-play), with `loss`, and `eps` = env-steps/s (throughput health).
- **The real verdict is always the gate**, not the loss (`tournament_gate <ckpt> --seeds N`).

## 3. The continuous flywheel (daily)

| step | run | command |
|---|---|---|
| build delta | local | `python scripts/trackp/delta_build.py` → `bc_corpus_delta_{epoch}.tar.gz`; **you upload** to `gdrive:.../delta/` |
| fetch top agents | local | `python scripts/trackp/agent_fetch.py --top 16` (or `--url <slug> --version N --score S`) → `.local/top_agents/` + `index.json` |
| pack top agents | local | `python scripts/trackp/league_refresh.py --pack` → `top_agents_{epoch}.tar.gz`; **you upload** to `gdrive:.../top_agents/` |
| generate self-play data | local | `python scripts/trackp/self_play_data.py --pairs 30 --seeds 8` → BC-format shards in `data/trackp_selfplay/` |
| archive ckpts | box (always-on) | `nohup setsid bash scripts/trackp/ckpt_upload_loop.sh >ckpts/archiver.log 2>&1 &` |
| league consume | box (always-on) | `nohup setsid bash scripts/trackp/league_refresh_loop.sh >ckpts/league.log 2>&1 &` (pulls+extracts the pack) |
| daily drop | local | `python scripts/trackp/submission_build.py --weights policy_rl.pt` (add `--push` when green) |
| laptop lineage | laptop | `python scripts/trackp/laptop_pipeline.py --pull-delta` → prints weights path |

**Agent types:** `agent_fetch` handles single-`.py` agents AND agents that pack a
**binary** (C++/Rust exe, `.so`, or a `submission.tar.gz`) — each agent's folder
holds its full payload and `index.json` records `type` (py|binary|tar|bundle) +
`entry` so the box knows how to invoke it (`--pack` preserves exec bits; the box
`--from-tar` restores them). The RL model is torch weights on the ckpt path.

## 4. Submission builder — proof it works (no PASS flags)

`submission_build.py` does NOT trust a PASS boolean. It plays the built agent vs a
copy of itself and vs a PASS baseline across seeds/worlds (both seats) and gates on
**measured numbers**, refusing to push unless ALL hold:

- **completed_all** — every episode ran with no engine error (status DONE),
- **beats_PASS** — win vs the do-nothing agent ≥ 90 % (a working agent crushes PASS),
- **actually_acts** — non-PASS turns ≥ 30 % (catches the degenerate all-PASS policy),
- **latency_ok** — worst turn < 900 ms (under the 1 s `actTimeout`),
- **banks_nontriv** — self-play mean bank > 100 (not banking ~0),
- **deterministic** — same obs → same action (reproducible submission).

**Every validation game is logged** to `.local/submissions/<stamp>/replays/*.jsonl`
(per turn: day/hour, both actions, running banks). Review it:
```bash
python scripts/trackp/replay_view.py .local/submissions/<stamp>/replays/selfplay_seed3.jsonl
python scripts/trackp/replay_view.py <replay.jsonl> --actions        # every non-PASS move
```
Then the panel gate (`tournament_gate`, surfaces win-rates <2500 / ≥2500) and the
intraday gate vs the last submission. **First `--push`**: it prints the exact
`kaggle datasets version` (refresh the ckpt dataset) + `kaggle kernels push`
(is_private=true, `configs/trackp_bc_kernel-metadata.json`). Wire the dataset/kernel
dirs once; never `kaggle competitions submit` from here.

## 5. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| GPU 0 % / power idle, checkpoint frozen, procs alive `S` (not `T`) | dataloader deadlock (often after a SIGSTOP/SIGCONT of the main proc) | full stop (§1) then restart; resumes from checkpoint |
| throughput halved, `pgrep -c -f bc_train` ~ doubled | two supervisors (killed trainer but not `run_pipeline.sh`) | `pkill -9 -f run_pipeline.sh` then clean restart (§1) |
| cumulative `X/s` low but GPU 100 % | the log's `seen/elapsed` avg was dragged by a pause; instantaneous is fine | sample twice 30 s apart: `(seen2-seen1)/30` |
| CUDA OOM in RL | mb/envs too high for free VRAM (esp. if another proc holds memory) | lower `--envs` or PPO `mb`; ensure no stray proc holds VRAM (`nvidia-smi`) |
| bench needs a clean GPU while BC runs | — | pause atomically: `PIDS=$(pgrep -f bc_train); trap 'for p in $PIDS;do kill -CONT $p;done' EXIT; for p in $PIDS;do kill -STOP $p;done; <bench>` |
| resume seems to start from 0 | `global_step` is restored from ckpt; the `[bc] step…` counter is **session-relative** | check `torch.load(...)['global_step']`, not the log's step |
| box died entirely | local ckpt gone | pull newest `gdrive:kaggriculture/ckpt_archive/policy_*_stepN_*.pt`, scp to `ckpts/`, restart |

**Pause BC for a clean benchmark** (safe SIGSTOP with guaranteed resume) — the `trap`
pattern in the table is mandatory; a bare SIGSTOP that isn't resumed looks exactly
like the deadlock row above.

## 5a. Self-play data generation

`self_play_data.py` makes fresh BC training data by having pool agents play each
other across many worlds/seeds (clone-vs-clone allowed), on the **official
engine**, encoded through the **exact BC pipeline** (`trackp_corpus.episode_rows`
— same tokenizer, same winner-keep filter: winner always, loser kept if
rating ≥ 2100).

- **Pool** (`--pool`, de-duped by path; default `all`) draws from the in-repo
  faithful reactive roster — no fetch step required:
  - `crown` — top reactive kernels from `models/crown_panel.json` with **real
    ladder ratings** (`selfplay_corpus.top_reactive_agents`, ~11 agents),
  - `killers` — the **5 agents we keep losing to** (`loss_forensics.KILLERS`:
    tschinkel 2945, alperen_rhythm, k0006 2494, nathanjacob, tetsutani),
  - `pub` — `agents/pub_*.py` (curated public head-to-head agents),
  - `top_agents` — reactive `type=py` agents in `.local/top_agents/index.json`
    (from `agent_fetch`; binaries/tars skipped — not callable),
  - plus any policy checkpoints via `--policies` (play vs the field or a clone).
  `all` yields ~31 distinct agents out of the box.
- **Output** = `data/trackp_selfplay/` — a self-contained corpus
  (`shard_*.parquet` + `norm.json` + `index.parquet` + `manifest.json`),
  **byte-compatible with BC's loader** (verified: `bc_train.shard_list` finds it).
  Re-runs APPEND.

```bash
python scripts/trackp/self_play_data.py --pairs 30 --seeds 8               # sample 30 matchups (pool=all)
python scripts/trackp/self_play_data.py --pool crown,killers --pairs 40    # only the strong reactive field
python scripts/trackp/self_play_data.py --all-pairs --seeds 4              # every combination incl. self
python scripts/trackp/self_play_data.py --policies ckpts/policy_rl.pt --pairs 20  # our policy vs the field
```
Uses: extra BC data, a **delta** source (`--corpus data/trackp_selfplay`), or a
teacher-refresh base. One game ≈ 1,440 rows (2 seats × 720 turns) when both
agents are ≥ 2100. Binary/tar agents are skipped (not runnable as a callable).

## 5b. Sample-efficiency levers: refreshed teacher + distillation

Running RL at `--ppo-epochs 1` is ~1.5× faster (1B in ~6 d vs ~9 d) but each
transition drives one gradient pass instead of two → less sample-efficient and
noisier. These levers buy that back. All are flags on `rl_selfplay`; the recipe
for the **main autopilot run** is set via `set_rl_recipe.sh` (env → persisted →
restart); a **one-off polish** uses `consolidate.py`.

### Teacher levers (stability / consolidation)
| flag | effect |
|---|---|
| `--kl-c 1.0` | firmer teacher-KL anchor (default 0.5). Raise at 1 epoch for stability. |
| `--teacher-mode bc` | (default) anchor to the frozen BC — imitation prior. |
| `--teacher-mode ema --teacher-ema-decay 0.999` | teacher = slow EMA of SELF (target-net) → recovers a 2nd epoch's *consolidation*, lets policy climb past BC. |
| `--teacher-mode refresh --teacher-refresh-every 200` | hard-copy self into the teacher every N rollouts (coarser EMA). |
| `--teacher-ckpt <path>` | load the teacher from a SEPARATE checkpoint (e.g. an improved BC), independent of the policy warm-start. Must share model dims. |

**Switch the running RL to a recipe (box), durably:**
```bash
# 1-epoch, firmer anchor, EMA teacher:
PPO_EPOCHS=1 KL_C=1.0 TEACHER_MODE=ema bash scripts/trackp/set_rl_recipe.sh
# anchor to an improved BC you uploaded:
TEACHER_CKPT=/root/kaggriculture/ckpts/policy_bc_v2.pt bash scripts/trackp/set_rl_recipe.sh
```
It writes `/root/rl_recipe.env` (sourced by `run_pipeline.sh`, survives restarts)
and cleanly restarts (RL resumes from its checkpoint). Verify:
`grep '\[rl\] SELF-PLAY' ckpts/pipeline.log` → shows `ppo_epochs= kl_c= teacher=`.

### Distillation polish (add epochs back on the FINAL policy)
Run the bulk at 1 epoch, then before a deadline drop, polish the latest checkpoint
with more gradient passes — recovers 2-epoch quality only where it matters, cheaply:
```bash
python scripts/trackp/consolidate.py --budget 10000000 --epochs 4 --kl-c 0.8
python scripts/trackp/consolidate.py --teacher ema --teacher-ckpt ckpts/policy_bc_v2.pt
```
Writes `…_polished.pt` (separate file — main lineage untouched); gate it vs the raw
checkpoint (`submission_build.py`) before submitting.

### Using your local BC updates (they ARE useful)
Keep improving BC locally (more epochs / delta via `laptop_pipeline.py` or
`bc_train --corpus data/trackp_delta`). An improved BC helps the running RL in three ways:
1. **Refreshed teacher** — upload it, then `TEACHER_CKPT=<improved-bc> bash set_rl_recipe.sh`.
   The RL *policy* keeps its own progress; only the KL anchor upgrades to the better
   prior (esp. valuable when the delta-BC knows new meta the frozen teacher doesn't).
2. **Distillation anchor** — `consolidate.py --teacher-ckpt <improved-bc>` distills the
   improved BC's knowledge into the RL policy during the polish.
3. **Warm-start for new lineages** — a better BC → a stronger fresh RL (the
   `laptop_pipeline.py` side-lineage), gated and pooled.
Keep the BC **dims identical** (384/4/1024) so it can slot in as a teacher; a
different-sized BC can still warm-start a new lineage but can't be a KL teacher.
Caveat: BC imitates the ladder; RL has learned to *win*. Use the better BC as an
*anchor/prior*, not a target to regress onto — don't over-raise `--kl-c`.

## 6. Config knobs (`/root/run_pipeline.sh` env)

`WORKERS` (BC dataloaders, 12) · `ENVS` (RL self-play envs, 256) · `RL_STEPS`
(target, 1B) · `GDRIVE` (`gdrive:kaggriculture`).

**RL recipe lives in `/root/rl_recipe.env`** (sourced by `run_pipeline.sh`, so it
overrides the defaults and survives restarts). Set/change it with
`set_rl_recipe.sh` (see §5b). **CURRENT recipe: `PPO_EPOCHS=1 KL_C=1.0`** — RL runs
**256 envs, 1 PPO epoch, KL_C 1.0, teacher-cache, fp16 rollout, torch.compile**
→ ~1,897 env-steps/s (**500M ≈ 3 d, 1B ≈ 6 d** — inside the 30 Sep deadline).
Planned polish: `consolidate.py` on the final checkpoint to buy back 2-epoch
quality (§5b). Archiver egress ≈ $0.03 over the run.

To change the recipe (env → persisted → clean restart, RL resumes from checkpoint):
```bash
PPO_EPOCHS=1 KL_C=1.0 TEACHER_MODE=ema bash scripts/trackp/set_rl_recipe.sh
```

## 7. Local v2 delta pipeline (`local_pipeline.py`)

The laptop-side flywheel that feeds fresh BC data into `data/trackp_corpus_v2/`
and produces `policy_bc_<YYYYMMDD_HHmmss>.pt` deltas. Five stages (run one, or
`all`). Params: `--top-n` (public agents, 10) · `--seeds` (worlds, 10) · `--clone`
(add clone-vs-clone) · `--jobs` (serve workers, 12) · `--limit` (cap delta eps) ·
`--epochs`/`--bs`/`--workers` (BC).

```bash
python scripts/trackp/local_pipeline.py all --top-n 10 --seeds 10 --clone   # end-to-end
python scripts/trackp/local_pipeline.py fetch                                # 1
python scripts/trackp/local_pipeline.py delta                               # 2
python scripts/trackp/local_pipeline.py selfplay --top-n 10 --seeds 10 --clone --jobs 12
python scripts/trackp/local_pipeline.py train                               # 4 + 5
```

| stage | does | output |
|---|---|---|
| **fetch** | `kaggle datasets download georgymamarin/kaggriculture-episodes` — **only replay files missing** locally (name-diff; byte-budget safe) | new `replays_*.parquet` in `D:\gm_dataset` |
| **delta** | new episodes (engine-keep − already-ingested in base ∪ delta) → v2 via `trackp_corpus._stream_to_shards` | `data/trackp_corpus_v2/delta/shard_<ts>_NNNN.parquet` + `delta/index.json` |
| **selfplay** | top-N public agents × M worlds on the **fast serve path** (+`--clone`) via `selfplay_corpus` | `data/trackp_corpus_v2/self_play/shard_<ts>_<tag>_NNNN.parquet` + `self_play/index.json` |
| **train** | BC on the **untrained** batches from BOTH indices (skips already-trained), resuming the **latest** ckpt in `.local/checkpoints`, base-norm pinned | `.local/checkpoints/policy_bc_<ts>.pt`; marks batches `trained_by` |
| **upload** | push the new weight to `gdrive:kaggriculture/ckpt` | (rclone-optional — see below) |

**Naming + index.** Every delta/self-play batch is stamped `YYYYMMDD_HHmmss` in the
shard names and recorded in that dir's `index.json`: `{ts, kind, source, shards,
episodes, rows, trained_by:[...]}`. `train` reads BOTH indices, trains only batches
with empty `trained_by`, then appends the produced policy name — so **re-runs never
re-train the same data**. Both v2 shard sets match the base schema (`shard_*.parquet`,
`TOKEN_LAYOUT_VERSION=2`); `bc_train` globs them directly.

**Checkpoints ↔ gdrive.** BC resumes from `.local/checkpoints`, kept in sync with
`gdrive:kaggriculture/ckpt` (which holds the remote `policy_bc.pt`). Delta runs chain
off the **latest** `policy_bc_<ts>.pt` (else the base `policy_bc.pt`). Dims are pinned
to the 6M model (`384/4/1024`) to match the base.

**⚠ No rclone on this laptop.** `train`'s down-sync and `upload` are **rclone-optional**:
if `rclone` is installed they run automatically; otherwise the pipeline prints the exact
`rclone`/manual command and expects **you** to (a) place `policy_bc.pt` into
`.local/checkpoints` before the first `train`, and (b) upload the new `policy_bc_<ts>.pt`
to `gdrive:kaggriculture/ckpt` afterward. Nothing else needs you.

---

## 8. RL checkpoint → agent → panel eval → build gate (hands-off)

Turn an archived RL checkpoint into a playable agent, prove it actually plays,
and measure it against the top-agents panel on the **fast serve path**. Every
step here needs **torch**, so run with the GPU env, not the default python:

```
PY="C:/ProgramData/anaconda3/envs/llm/python.exe"
```

### 8.1 Build agents from checkpoints

RL archives land in `.local/checkpoints/self_play/` as
`policy_rl_step<GS>_<stamp>.pt` (pull from `gdrive:kaggriculture/ckpt_archive`).
For each checkpoint you want to test, drop a one-line wrapper **next to the
weight** (see `agent_rl_30M.py` / `agent_rl_60M.py` / `agent_rl_90M.py` for the
template): it finds the repo root, imports `policy_agent.make_agent`, and
exports `agent`. It hard-codes the `.pt` path and loads
`configs/trackp_rails.json` (the exact rails RL trained against). To make one for
a new milestone, copy a wrapper and change the `_CKPT` filename + the docstring
step. Nothing else.

### 8.2 Agent architecture (what the wrapper builds)

```
obs
 └─ TC.encode_tokens            verb-aware tokens, TOKEN_LAYOUT_VERSION=2
 └─ set-transformer policy      d_model 384 / 4 layers / 8 heads / ff 1024  (~6M)
 └─ factorized heads            fverb·farg | hverb·harg×hands | mverb·marg·mqty×market
 └─ verb-aware decode           argmax (sample=False) for eval/deploy
 └─ REACTIVE RAILS  ◄────────── configs/trackp_rails.json, applied in order:
        weed_repair · water_guard · budget_guard · endgame_liquidate  (ON)
        scarcity_sell · sell_premium · harvest_ready · feed_care · collect_fertilizer (OFF)
 └─ sanitize (MANDATORY)        legal op, hands aligned positionally, ≤10 market orders
```

The **reactive rails** (`src/kaggriculture/trackp/reactive_rails.py`) are the
plug-in layer: the learned policy proposes, enabled rails override
(weed-repair, water-guard, budget-guard, endgame-liquidate…), and `sanitize`
runs **last** so the emitted action is always legal/desync-proof. Add a rail
with `@rail("name")` and enable it in the JSON — no policy retrain needed. Keep
the deployed rail set == the trained one unless you re-gate.

### 8.3 Prove they PLAY (not PASS) — do this before trusting any score

An untrained/degenerate policy can silently emit PASS every turn. Verify verb
distribution over a full game — farmer should be **>90% non-PASS** with real
verbs (PLANT/WATER/CARE/BUILD/PICKUP/BUY/SELL/HIRE):

```
$PY - <<'PY'
import sys,json; from collections import Counter
sys.path.insert(0,'src'); sys.path.insert(0,'vendor')
from kaggriculture.trackp.policy_agent import make_agent
from kaggriculture.engine import serve_match as SM
rails=json.load(open('configs/trackp_rails.json'))
ag=make_agent('.local/checkpoints/self_play/policy_rl_step<GS>_<stamp>.pt', rail_config=rails)
opp=SM.load_agent(r'data\kernels\_agents\kaggriculture-baseline__AGENT_B85.py')
fc=Counter(); srv=SM.Serve()
def c(o,cfg=None): a=ag(o,cfg); fc[a['farmer'][0]]+=1; return a
b0,b1=SM.run_match(c,opp,3,srv); srv.close()
print('bank',b0,'verbs',dict(fc.most_common()))
PY
```

`rl_panel_eval.py` also runs a built-in verify (each candidate vs PASS) and
prints `OK (bank … vs PASS 3000)` before the panel loop; a candidate that
fails to load or emits illegal actions is dropped, not silently scored 0.

### 8.4 Run the panel (fast serve path)

```
$PY scripts/trackp/rl_panel_eval.py \
    --candidates .local/checkpoints/self_play/agent_rl_30M.py \
                 .local/checkpoints/self_play/agent_rl_60M.py \
                 .local/checkpoints/self_play/agent_rl_90M.py \
    --seeds 24 --panel-n 24 --out .local/checkpoints/self_play/panel_eval.json
```

- **Panel** = `selfplay_corpus.top_reactive_agents(N)` — the real ladder-rated
  reactive kernels (currently **11**, ratings 2172–2837).
- **Seeds** = worlds; seat is **alternated** across seeds so each candidate sees
  both seats on the same (opponent, seed, seat) grid → cells are comparable.
- **Efficiency:** the neural candidate is loaded **once** (stateless per game);
  each opponent is reloaded **per game** (reactive agents carry per-episode
  state). Output: per-opponent W-D-L / score / mean bank, a per-candidate panel
  score, a ranked scoreboard, and a JSON report.
- **Cost:** each neural game ≈ **15–20 s** (per-turn batch-1 GPU inference, NOT
  the engine). So 3×11×24 ≈ **~4 h**. For a cheap tracking pass use
  `--seeds 6 --panel-n 24` (~1 h) or fewer opponents.

### 8.5 When is a checkpoint worth BUILDING for submission? (the gate)

Building a *deployable* agent works at any checkpoint (8.1). Building a
*competitive* one is a **measured** decision, not a step count:

1. **Track the curve** — run 8.4 with `--seeds 6` on each new archived milestone
   (100M/200M/250M…). Watch panel score + mean bank climb. (Baseline 2026-09-22
   @ ≤95M: score **0.000**, bank **~1k vs ~172k** — the agents play but are ~100×
   short; SELL/game 3→81 across 30M→90M is the live signal RL is learning.)
2. **Build trigger** — the first checkpoint that starts **winning games** vs the
   low-rated panel (2172–2230) and banks **tens of thousands**.
3. **Then, and only then:** run the full held-out panel gate
   (`reactive_parity` → `evaluate`/`win_metric.paired_test`), pick the
   **best-measured** checkpoint (usually — not always — the latest; RL is not
   monotonic), package, and submit **manually**. This runbook never submits.

> Deadlines: RL 1B ETA ≈ 6 days (~28 Sep) at ~1800 eps/s; final submission
> 30 Sep. Competitiveness by 1B is **plausible but not assured** given the slow
> bank curve — keep the milestone eval running so the build moment is data-driven.

---

## 9. Box panel evaluator — automatic "does it beat the panel?" + best-ckpt feedback

Instead of running §8.4 by hand, a daemon on the box does it continuously. The
RL loop trains by fast NN self-play and **never checks the policy against the
real panel**; this evaluator closes that gap **without touching training**
(`scripts/trackp/box_panel_eval.py`).

**What it does, every `--every` steps (default 30M, checked each 10-min poll):**
1. snapshots the live `policy_rl.pt` (retries until a clean, non-mid-write copy);
2. plays it vs the **pinned panel** on `kagg serve`, **on CPU** (default 6
   threads) → **zero GPU contention**, ~**2.4 s/game** on the box → a full
   11×6 eval in **~3 min**;
3. appends `{gstep, panel_score, per-opp}` to `ckpts/panel_eval.jsonl` (the
   curve) and uploads that curve to gdrive;
4. **feedback:** if the score improved, saves `ckpts/policy_rl_best.pt` +
   `.panel_best.json` (the best-by-REAL-metric checkpoint — RL is not
   monotonic, so latest ≠ best). It mirrors the 70 MB best to gdrive **only
   once the score > 0** (a 0.0 baseline stays local — no wasted egress); all
   rclone calls are **timeout-bounded** so slow gdrive never wedges the loop.

It **only reads** `policy_rl.pt` (the trainer owns it) and never submits.

**Pinned panel = identical on box and laptop.** `configs/panel_roster.json`
lists the 11 opponents (name / repo-relative path / rating, 2172–2837) built
from `top_reactive_agents`. `_panel()` prefers this roster (resolved against
ROOT, cross-OS) over `top_reactive_agents`, because the box's own
`crown_panel.json` resolves to 0 files. **If you change the panel:** regenerate
the roster locally and ship it + any new agent files to the box:
```
python -c "import sys,json,pathlib; sys.path.insert(0,'src'); from kaggriculture.data import selfplay_corpus as SC; R=pathlib.Path('.').resolve(); json.dump([{'name':n,'path':pathlib.Path(p).resolve().relative_to(R).as_posix(),'rating':r} for n,p,r in SC.top_reactive_agents(24)], open('configs/panel_roster.json','w'), indent=2)"
FILES=$(python -c "import json;print(' '.join(r['path'] for r in json.load(open('configs/panel_roster.json'))))")
tar -czf - configs/panel_roster.json $FILES | ssh -p <port> -i ~/.ssh/vast_knee root@<box> 'cd /root/kaggriculture && tar -xzf -'
```

**Launch / stop / restart on the box** (detached, like the archiver):
```
cd /root/kaggriculture
nohup setsid /venv/main/bin/python scripts/trackp/box_panel_eval.py >>ckpts/panel_eval.log 2>&1 </dev/null &
pkill -f box_panel_eval.py            # stop
# knobs: --every 30000000  --seeds 6  --panel-n 24  --poll-min 10  --threads 6  --once
```

**Watch it** — the curve is in `ckpts/panel_eval.jsonl` (+ gdrive) and the best
in `ckpts/.panel_best.json` (+ gdrive `policy_rl_best.pt` once >0). `box_status.py`
(and therefore the hourly report) now prints a **`PANEL:`** line — last
`gstep/score/bank` and the running best. That line **is** the §8.5 build-gate
signal: when `best score` starts climbing off 0.0 and mean bank reaches the tens
of thousands, pull `policy_rl_best.pt` and run the full held-out gate.
