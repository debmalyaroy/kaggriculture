# Slot-2 operator runbook — run / improve / loop / submit

Hands-free BC→macro-RL predator: **train → test → gate → loop → ship-prep**, then
YOU submit. The pipeline never calls Kaggle. Everything runs on this box (4060 for
the net, Rust `kagg batch` for rollouts).

**Setup (once):** `PYTHONPATH="src;vendor"` and use the `llm` env python
(`C:/ProgramData/anaconda3/envs/llm/python.exe`), which has torch 2.11+cu128,
onnxruntime 1.29. Corpus + 786-opponent pool are already on disk.

---

## 1. How to RUN (one command, hands-free)

```bash
python -m kaggriculture.pipeline.slot2_loop --rounds 6
```

That is the whole loop. It:
1. **BC warmup** (skipped if a checkpoint exists; `--fresh-bc` forces it).
2. For each round: **RL self-play** (budget escalates each round) → **package**
   (ONNX + parity gate) → **gate** vs the strong refs (v46).
3. When a round clears the bar (default `--bar 0.9`): **compile the policy into a
   bandit base tape**, **build + parity-gate the tarball**, write
   `.local/candidates/SUBMIT_INSTRUCTIONS.md`, and **stop**.
4. If no round clears the bar in the budget: leaves the BEST artifact + a HOLD
   verdict in `SUBMIT_INSTRUCTIONS.md`.

**Resume** later (adds more rounds to where it stopped):
```bash
python -m kaggriculture.pipeline.slot2_loop --rounds 4 --resume
```
State lives in `models/rl/loop_state.json`; best policy in `models/rl/rl_policy_best.pt`.

**Re-gate + build the current best without training:**
```bash
python -m kaggriculture.pipeline.slot2_loop --ship-only
# add --force-ship to build the tarball for inspection even if below the bar
```

**Watch it:** `tail -f .local/scratch/slot2_loop.log` (or point the loop's stdout wherever).

### Individual stages (if you want manual control)
```bash
python -m kaggriculture.train.bc_warmup --epochs 12 --d-model 384 --layers 8   # -> models/rl/bc_policy.pt
python -m kaggriculture.train.macro_rl  --batch --iters 300 --games-per-iter 256 --games-per-plan 64
python -m kaggriculture.train.package_policy                                    # -> policy.onnx + parity
python -m kaggriculture.measure.eval_harness --gate --seeds 6                   # ship/fallback verdict
```

---

## 2. How the LOOP works (train → test → gate → loop → submit)

```
BC (once) ─► [ RL round r ─► package(ONNX+parity) ─► gate vs v46 ]
                   ▲                                      │
                   └──────── escalate budget ◄── score < bar
                                                          │ score ≥ bar
                                                          ▼
              compile policy ─► base tape ─► build_rust_bandit ─► parity gate
                                                          ▼
                              write SUBMIT_INSTRUCTIONS.md  (STOP — you submit)
```

- **Reward** = win/draw/loss on the bit-exact engine (batch rollouts, ~7.7 ms/game).
- **Gate** = seat-swapped paired vs the two ~2500 public tapes; **ship iff ≥ 0.9 vs v46**.
- **Never submits.** Ship-prep only builds the artifact + instructions.

---

## 3. How to IMPROVE (the levers, in priority order)

Read the last gate score in `models/rl/loop_state.json` / `SUBMIT_INSTRUCTIONS.md`.

1. **Executor quality (biggest lever if banks stay low).** The default greedy
   micro-executor realizes ~57% of a plan; if the policy's banks plateau far below
   the opponents' (140–172k) no matter how long it trains, the executor is the wall.
   Port the persistent-cluster router properly (season_cbs, 86–87%) into
   `macro_env.PersistentClusterController`, then run with that executor. (The naive
   port currently regresses — it needs the husbandry/threshold logic.) This is the
   #1 non-GPU improvement.
2. **More rounds / bigger budget:** `--rounds 8` or raise `--iters` / `--games-per-iter`.
   The loop already escalates iters each round.
3. **Corpus:** when the download drains (CDN 429 resets daily), rebuild the corpus
   for more high-rated data: `python -m kaggriculture.data.bc_corpus --limit 0`.
4. **Knobs:** `--kl-coef` (lower = policy drifts further from BC), `--init-std`
   (exploration), `--games-per-plan` (higher = lower-variance reward, faster/game),
   `--margin-shaping` (reward bank margin, not just win).
5. **BC width:** `--fresh-bc --d-model 384 --layers 8` for a stronger warm start.

**Throughput sanity:** ~7.7 ms/game at `--games-per-plan 64` → 300k games ≈ 38 min.
If rollouts feel slow, confirm `--batch` is on (it is, by default) and that the lean
`kagg-engine.exe` built (isolated `cargo build --release --bin kagg-engine`).

---

## 4. How to SUBMIT (you do this — never the pipeline)

When `SUBMIT_INSTRUCTIONS.md` says **SHIP**:

1. Check the currently-active pair (latest-2 rule — a new submit evicts the older):
   ```bash
   kaggle competitions submissions -c kaggriculture
   ```
2. Confirm you keep two strong agents after the swap.
3. Submit the built tarball (`.local/candidates/slot2_bandit_*.tar.gz`) via its
   private kernel, or run the repo's gate-and-ask submit:
   ```bash
   python -m kaggriculture.pipeline.submit      # re-checks self-play, then asks
   ```
4. Never submit an agent that has not passed the local self-play + gate.

---

## 5. Two shipping vehicles for the SAME learned policy

The loop ships the **bandit** vehicle (ready). Track-P is the higher-ceiling second seat.

| | Bandit (loop default) | Track-P |
|---|---|---|
| Runtime | policy → base **tape** + reactive rails (`kagg-bandit`) | policy → ONNX → **native Rust inference** (`kagg-trackp`), reactive |
| Build | ready (`build_rust_bandit`) | needs native inference (F1.1/G5.1) — not built |
| Risk / ceiling | lower / good | higher / higher |

Ship the bandit first for a competitive seat; add the trackp reactive seat once
native inference lands. Two different seats = your two-slot diversification.

---

## 6. Guardrails
- One heavy job at a time (16.9 GB box). The loop is one job.
- Never overwrite `rustengine/kagg.exe`; rollouts only READ the engine; builds are isolated.
- The gate is the truth, not predicted cash. Nothing ships below the bar.
- The pipeline never submits. You submit.
```
