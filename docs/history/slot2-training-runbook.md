# Slot-2 training runbook (BC → macro-RL predator) — RTX 4060

Everything below is **CODE-ready and smoke-tested on CPU**. The only thing left is
to run the training on the GPU. Nothing here submits to Kaggle — it builds and
gates the artefact; the operator submits.

Field-validated method (master plan §0.6/§0.7): **BC warmup → MACRO-level self-play
RL** (~30 daily decisions over a shared deterministic micro-executor), NOT per-turn
RL (per-turn plateaus). SNORLAX BC→RL reached silver in ~300k games.

## 0. One-time setup

```bash
# env: the anaconda 'llm' env has torch 2.11 + CUDA 12.8, onnxruntime 1.29.
C:/ProgramData/anaconda3/envs/llm/python.exe -c "import torch; print(torch.cuda.is_available())"  # True on the 4060
# run everything with the package + vendored engine on the path:
#   PowerShell:  $env:PYTHONPATH="src;vendor"
#   bash:        PYTHONPATH="src;vendor"
```

Corpus (already built; rebuild only if stale):
```bash
python -m kaggriculture.data.bc_corpus --limit 0                 # top-100 (both seats when --both)
python -m kaggriculture.data.bc_corpus --source destbreso        # 416k macro rows, rating≥2300
# -> data/bc_corpus/macro.parquet (+ macro_destbreso.parquet), 440k macro rows total
```

## 1. Full pipeline, one command

```bash
python -m kaggriculture.pipeline.slot2_pipeline \
    --stages corpus,bc,rl,package,gate \
    --epochs 12 --d-model 384 --layers 8 --heads 8 --batch-size 128 \
    --iters 400 --games-per-iter 96 --ppo-epochs 4 --battery 16
```
Stages run in order; each writes its own artefact so you can resume from any stage
with `--stages rl,package,gate` etc.

## 2. Stage by stage (what to watch)

### 2a. BC warmup — `models/rl/bc_policy.pt`
```bash
python -m kaggriculture.train.bc_warmup --epochs 12 --d-model 384 --layers 8 --heads 8
# ~10-16M params at this width; return-conditioned; BF16 autocast on CUDA.
```
- Return-conditioned: each day sees a return-to-go token + the imitated seat's
  rating; at RL/inference we condition on **winning** (rtg=1).
- Watch: `val` (total), `val_cls` (7-way daily-class CE), `val_vec` (Smooth-L1 on
  the 26-dim action). Overfit ⇒ raise dropout / lower `--d-model`.
- Corpus is ~440k macro rows; a 12-epoch run is minutes on the 4060.

### 2b. Macro-RL (PPO self-play) — `models/rl/rl_policy.pt`
```bash
python -m kaggriculture.train.macro_rl --iters 400 --games-per-iter 96 \
    --ppo-epochs 4 --kl-coef 0.1 --ent-coef 0.001 --init-std 0.3 --battery 16
```
- Policy = diagonal Gaussian over the per-day action vector; plans sampled
  **autoregressively** (day t updates day t+1's cumulative features).
- **Opponent pool** = 786 real ≥2500 destbreso tapes + frozen-self (self-play).
- **Teacher-KL** anchors to the frozen BC prior (`--kl-coef`); **gated promotion**
  saves a checkpoint only if it beats the last promoted one on the frozen battery.
- Watch: `battery` (win rate vs rated tapes) trending up; `std` shrinking as the
  policy sharpens; `PROMOTED` markers. If `battery` stalls at 0, the micro-executor
  is the ceiling — see §4.
- **Throughput (B4, wired 2026-09-18): ~7.7 ms/game → 300k games in ~38 min.**
  `--batch` (default in `slot2_pipeline`) compiles each plan ONCE on the fast
  serve engine (field ops are seed-independent) and replays that tape vs many
  opponents/seeds via Rust `kagg batch`. Per-game cost = compile/`--games-per-plan`
  + ~5 ms; at `--games-per-plan 64` that's **7.7 ms/game** (bit-exact to the
  closed-loop engine, validated). Higher `--games-per-plan` = faster/game but
  fewer distinct plans/iter — 32–64 is the sweet spot.
  Backends: `serve` (closed-loop, 0.26 s/game, bit-exact) for compile + battery;
  `batch` (open-loop, ~5–8 ms/game) for the rollout mass. Legacy `vendored`
  (Python, ~2 s/game) is the cross-check only.

### 2c. Package — `models/rl/policy.onnx` + `policy_manifest.json`
```bash
python -m kaggriculture.train.package_policy          # RL ckpt; --bc for BC only
```
- Exports at fixed T=30 (causal ⇒ pad future days harmlessly), **asserts onnxruntime
  == torch** (parity gate; refuses to ship on mismatch).
- Manifest carries norm stats + the exact feature recipe so the native-Rust runtime
  (F1.1 / G5.1, `candle`/`tract`/`ort`) builds identical inputs.

### 2d. Gate — ship-or-fallback
```bash
python -m kaggriculture.measure.eval_harness --gate --seeds 6
```
- Smoke + seat-swapped vs the **strong refs** (the two ~2500 public tapes).
- **Ship iff aggregate ≥ 0.9 vs v46** (the B4.0 bar). Else Slot-2 falls back to a
  diversified OR `route2` (E5.3) — a fallback, not the plan.
- No submission happens here. On SHIP, hand the artefact to the operator.

## 3. Smoke (always green before a real run)
```bash
python -m kaggriculture.pipeline.slot2_pipeline --smoke   # ~2 min CPU, all stages
```

## 4. Known ceiling & the upgrade path (honest)
- The default micro-executor is the **greedy B2.0 realizer** — legal, real
  production, but under-routes dense economies (this is exactly the B4.2 finding:
  the gap to top play is production VOLUME). It is a pluggable hook:
  `MacroEnv(executor=...)`. Swapping in **PC-TAPF / plan-repair** (Workstream B2,
  `rustengine-or/season_cbs`) raises the ceiling without touching the RL loop.
- Throughput upgrade: an **obs-free executor** lets rollouts run on Rust
  `kagg batch` (seed⇥tapeA⇥tapeB → banks), the path to ~300k games.
- Native inference (F1.1/G5.1): the ONNX + manifest already export the exact
  feature recipe for a no-Python submission runtime.

## 5. Guardrails
- One heavy job at a time (16.9 GB box; OOM history). BF16 keeps the BC model small.
- Never overwrite `rustengine/kagg.exe`; RL rollouts only READ the engine.
- Gate before shipping; the pipeline never submits. Operator submits.
