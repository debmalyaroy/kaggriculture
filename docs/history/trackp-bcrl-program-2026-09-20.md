# Trackp (Slot-2) BC→RL Program — Plan of Record (2026-09-20)

**Status:** approved architecture + tasklist; awaiting operator "go" to start T0.1.
**Scope:** the Slot-2 "Adaptive Predator" (trackp) — a reactive, end-to-end
neural policy shipped as the *closed-loop* second seat alongside the bandit tape
(Slot 1). Modelled on the field-validated BC→self-play-PPO pipeline
(Michal1337/pkmn-kaggle, 12th/6807, transiently #1), adapted to Kaggriculture.

Load on: "trackp bcrl program", "slot2 plan".

---

## 0. Objective & definition of done
DONE = a compiled trackp agent that:
1. passes `reactive_parity` / `serve_allowed`;
2. wins a **banded paired-McNemar vs the tape floor** on the crown panel with
   **no band regressed AND ladder-weighted aggregate > 0**;
3. holds its result on reserved holdout tapes;
4. fits the compiled tarball (**worst turn < 1 s**, target < 200 ms);
5. gets an explicit operator go. Never submits from the session.

## 1. Resolved decisions
- **DF1** one end-to-end per-turn policy; RETIRE the macro-plan + hand-written
  executor (that split is the documented $0–20k executor wall). Keep the macro
  lineage only as a fallback.
- **DF2** deploy = compiled Rust tarball, policy baked to Rust (`candle`);
  1 s/turn budget (torch/ONNX only as the parity oracle).
- **DF3** NN runs IN the rollout loop via a Rust binary-obs bridge → reactive
  AND fast.
- **DF4** small transformer (~2M params, compact ~100-token obs). Train in
  PyTorch; Rust supplies vec-envs; bake to Rust for deploy.
- **Venue** BC → Kaggle T4×2 (epochs 1–2) + laptop (epoch 3); RL → laptop
  (4060 + Rust envs); data prep + deploy bake + all gating → local.
- **Gate** existing crown panel (`models/crown_panel.json` +
  `measure/crown_gate.py` via `eval_harness.crown_refs`). No new gate.
- **Streaming everywhere;** nothing loads gm_dataset whole.
- **Dataset** `D:\gm_dataset` refreshed to the 2026-09-20 Kaggle version
  (1.32.7 episode count grew by ~5.7k; corpus build picks these up).

## 2. Audit of the existing trackp code (why we replace, what we keep)
Canonical Slot-2 path today: `data/bc_corpus.py → train/bc_warmup.py →
train/macro_env.py → train/macro_rl.py → train/package_policy.py →
measure/eval_harness.py`. (`train/bc_*`, `trackp/harness/bc_*` are older dupes;
`trackp/*` is a second, MLP-macro lineage.)

**Architectural findings (drive DF1):**
- A1 trackp is a *macro-plan generator + fixed hand-written executor*
  (`macro_env.SeasonCbsController`), not an adaptive per-turn policy. RL can only
  tune the plan, never the execution → the executor wall.
- A2 reactivity vs throughput are mutually exclusive today: reactive serve path
  ~0.63 s/game; the fast `BatchRoller` compiles a plan to ONE fixed tape vs PASS
  at a ref seed then replays it → not reactive, weed-desync-prone.
- A3 BC input has NO board/market/opponent state (`bc_warmup._features`) — only
  day/world/RTG/own-cumulative-plan. The richer per-turn features exist in
  `data/bc_corpus.MICRO_FEATURES` (incl. `opp_money`) but are discarded.
- A4 deploy gap: `package_policy` exports only the network; the 300-line
  executor must run every turn and isn't ported.
- A5 corpus source mismatch: BC trains from `.local/scratch/gm/...` + destbreso,
  NOT the full `D:\gm_dataset` (240k @1.32.7).

**Correctness bugs:**
- B1 (HIGH) `bc_warmup._world_bucket` uses builtin `hash(str)` → salted per
  process → train/serve skew (correct sha1 pattern already in
  `data/bc_corpus.split_of`). Also baked into the `package_policy` manifest.
- B2 (HIGH) `macro_rl.sample_plan` runs `sanitize_plan` AFTER sampling but PPO
  stores the pre-sanitize action → credit assigned to overwritten actions.
- B3 (MED) `macro_rl.frozen_self_cells` builds a `PlanController` then never
  uses it; spawns a fresh serve process per call.
- B4 (MED-HIGH) `MacroAction.to_vector` buckets sell/hire lossily → coarse
  sell-quantity target (price realization is the #1 ladder lever).
- B5–B8 (LOW-MED) herd hard-cap 12; norm-over-train+val; dead `limit_eps`;
  open-loop tape ignores weed RNG (desync).

**Keep (the reusable spine):** `data/bc_corpus.py` (deterministic sha1 split,
streaming, MICRO+MACRO, `world_sig`); `MacroEnv(serve)` + `serve_match` +
`reactive_parity` + `eval_harness` (4 faithful evals + crown gate);
`OpponentPool` (+destbreso prewarm, league); `package_policy` ONNX parity
discipline; Rust `kagg serve`/`batch`.

**Salvage/rebuild:** adapt `data/bc_corpus.py`; keep the gate/opponent spine;
adapt `package_policy`; retire `bc_warmup`/`macro_actions`/`macro_env`-executor/
`macro_rl`/`trackp/export_policy` to the fallback lane (fix B1/B2/B4 there only
if we keep it).

---

## 3. Architecture (the entire thing)

### 3.0 One network, three lives
One policy network: (1) BC-trained in PyTorch on Kaggle, (2) RL-fine-tuned in
PyTorch on the laptop with the Rust engine feeding it, (3) deployed as Rust code
in the tarball. Identical weights across all three.

### 3.1 Per-turn data flow (deploy)
```
engine obs → [Rust token encoder, byte-exact to the BC one]
  → ~100 tokens {tiles · animals · shed · market(9) · hands · game-state · RTG}
  → Transformer encoder (ONE forward, ~2M params, order-invariant)
  → { value head (CLS; train-only) ; pointer decoder (cheap) }
  → autoregressive composite decode → {farmer, hands[k], market[≤10]}
```
Key: the transformer encodes the board ONCE per turn; the action decoder is a
cheap pointer module over those embeddings — this preserves ~10–25k steps/s in RL.

### 3.2 Obs → tokens (≤128 tokens; compact)
| group | count | per-token features |
|---|---|---|
| our tiles (non-LOCKED) | ~25–60 | x,y,kind(one-hot),crop,age,watered,yield,animal,fed,cared |
| opponent observable tiles | few (aggregated) | same, summarized |
| shed | 1–2 | ~13-item inventory |
| market | 9 | per product: price, inventory (shared/observable → reactivity) |
| hands | ≤~15 | x,y, carried inventory |
| game-state | 1 | money, opp_money, day, hour, unlocked-shops(multi-hot), n_hands, land, step |
| return-to-go | 1 | conditioning target (set to "win" at inference) |

Token embed = zone-type embedding + projection of numeric fields (+ categorical
lookups). NO positional embedding (order-invariant; spatial info in tile x,y).
Two byte-exact extractors, gate-locked: Python (BC corpus) + Rust (`obstoken.rs`).

### 3.3 Encoder + heads
~2M params, d_model≈128, 3–4 layers, 4 heads, GELU, pre-norm, no inference
dropout. Value head on pooled CLS (PPO/GAE + promotion; ignored at deploy).
Pointer policy head: each legal action = an option token (source+target+verb+qty
embeddings), scored by dot-product with a query; legality = which tokens exist +
−∞ mask.

### 3.4 Composite action decode
`{farmer op, one op per hand (positional), ≤10 ordered market orders}`, decoded
autoregressively over the already-encoded tokens: farmer → each hand (in index
order, conditioned on prior picks so movers don't collide) → market loop (pick
order or SUBMIT, cap 10, decode order = engine priority). Legality-masked each
sub-step by the shared Rust rules. **v1 fallback if hard to learn:** factorized
single-shot heads + deterministic conflict-resolver by hand index.

### 3.5 BC training (Kaggle → laptop)
```
gm_dataset (1.32.7) → Python token extractor (winner + high-rated loser,
  return-conditioned, per-episode split) → sharded parquet (~100M dec, ~12 GB)
  → upload Kaggle Dataset
Kaggle T4×2: memmap IterableDataset → TokenTransformer → CE over legal option
  tokens (+opt value regression); epochs 1–2 (~11 h, one session) → ckpt
  (weights+optimizer+LR-sched+step) → download
laptop 4060: resume → epoch 3 (~17 h), continue LR tail, constant effective
  batch (grad-accum) → models/rl/bc_policy.pt
```
Return-conditioning makes including loser-seats safe (RTG token says
winning/losing). Never load the corpus whole.

### 3.6 RL self-play (laptop)
```
512–2048 Rust envs (lockstep, auto-reset) --binary token buffers--> torch (4060)
  ONE batched encoder forward → batched decode → actions back to Rust
  trajectories → PPO minibatch update (GPU); single process (PyO3) = no weight sync
```
PPO + GAE, one net both seats (negamax GAE), teacher-KL to frozen BC prior,
entropy, gated promotion @0.53. Opponents: growing self-checkpoint league +
transcribed ≥2500 real economies, diversity-weighted by opening/economy family.
Reward = terminal score (win/draw 0.5/loss); illegal=forfeit; anti-stall.
**Budget: 500M milestone → gate → extend to 1B cap only if the aggregate is
still climbing.** LR/entropy scheduled against the 1B cap; full-state
checkpoints (policy+opt+sched+step+league+pool+RNG) so 500M→1B is seamless.

### 3.7 Deploy (compiled tarball)
Export weights → Rust `candle` forward matching 3.3–3.4; Rust obs encoder (3.2)
already exists. Parity: Rust == torch action-for-action over ≥26 cells. Package
`main.py` + static `kagg` (engine + baked policy) → `submission.tar.gz`. Latency
one encoder + cheap decode, single sample CPU → target worst < 200 ms.

### 3.8 Gate
`eval_harness` seat-swap vs `crown_refs()` (banded reactive panel) on the
faithful serve engine → `crown_gate` banded McNemar + ladder-weighted aggregate.
Ship iff no band significantly regressed AND aggregate > 0, + reserved holdout +
`reactive_parity`.

---

## 4. Phased tasklist
Runs: [L]=local · [R]=Rust · [K]=Kaggle. Effort: S≤½d · M≈1–2d · L≈3–5d.

### Phase 0 — Design lock (G0)
- T0.1 [L,S] obs token-layout spec → `docs/history/trackp-token-spec.md` (every
  `obsstate.rs` field mapped).
- T0.2 [L,S] composite action + pointer/decode + legality spec (matches `rules.rs`).
- T0.3 [L,S] confirm gate wiring (`eval_harness.crown_refs`→`crown_gate`).
- **G0:** specs signed off.

### Phase 1 — Streaming corpus + Kaggle dataset + bug-fixes (G1)
- T1.1 [L,S] fix B1 (deterministic sha1 world bucket + manifest recipe).
- T1.2 [L/R,M] streaming gm_dataset reader (1.32.7, join rating/bank, winner +
  high-rated loser, RTG); bounded RSS; ~100M decisions.
- T1.3 [L,M] per-turn token extractor → sharded parquet + manifest; flush-and-
  clear; legality invariant (mask-rate 0); ~12 GB reported.
- T1.4 [L,S] fix B4 (precise sell-qty target).
- T1.5 [K,S] Kaggle Dataset packaging + upload.
- **G1:** corpus streams (bounded RAM), legal, split leak-free, uploaded.

### Phase 2 — Rust binary-obs + vec-env (G2)
- T2.1 [R,M] token exporter (flat buffer, no JSON) → `rustengine/src/obstoken.rs`.
- T2.2 [R/L,M] parity test Rust==Python extractor (non-skippable) → 0 mismatch
  ≥40 eps/≥40k obs.
- T2.3 [R/L,L] `kagg vecenv` + zero-copy torch bridge + **throughput benchmark**
  (measured steps/s locks the RL wall-clock).
- T2.4 [R,S] Rust legal-mask generator (shared RL+deploy) matches Python mask.
- **G2:** parity green + measured throughput.

### Phase 3 — Policy + BC (G3 = KILL GATE)
- T3.1 [L,M] `TokenTransformer` (~2M; encoder-once + pointer decoder) →
  `train/policy_net.py`.
- T3.2a [K,L] streaming BC trainer + Kaggle notebook epochs 1–2 (one session,
  ~11 h); resumable ckpt.
- T3.2b [L,M] download; local epoch 3 as true continuation (resume LR tail,
  constant effective batch via grad-accum, ~17 h) → `bc_policy.pt`.
- T3.3 [L/R,M] offline top-1/k accuracy + closed-loop playout vs crown panel +
  tape floor.
- **G3 (KILL):** BC imitates AND realizes competitive bank in closed loop (no
  $0/20k collapse) AND no panel regression → RL; else STOP, write negative,
  fall back to tape pair.

### Phase 4 — Self-play RL fine-tune, local (G4)
- T4.1 [L/R,L] PPO+GAE self-play on Rust vec-env (batched bf16, compiled), BC
  init, teacher-KL, entropy; legality via masking (fixes B2).
- T4.2 [R,M] gated promotion (@0.53) + league + ≥2500 opponents + diversity
  weighting; fix B3.
- T4.3 [L,S] terminal-score reward; illegal=forfeit; anti-stall; **500M
  milestone → 1B cap**, extend only if aggregate still climbing; LR/entropy
  scheduled vs 1B.
- T4.4 [L,S] full-state checkpoint/resume (policy+opt+sched+step+league+pool+RNG).
- **G4:** RL beats BC head-to-head AND beats/parities tape floor on the panel.

### Phase 4b — Per-world two-net specialization (G4b)
- T4b.1 [L,S] target economy/world families (day-6 shops) from corpus.
- T4b.2 [L/R,L] two-net alternating finetune (A=target family, B=meta pool).
- T4b.3 [L/R,M] assemble behind day-6 dispatch; band-gate.
- **G4b:** specialists beat the generalist, no band regressed.

### Phase 5 — Deploy: compiled tarball (G5)
- T5.1 [R,L] bake weights → Rust `candle` forward → `policy.rs`.
- T5.2 [R/L,M] parity Rust==torch action-for-action (extend
  `tests/test_compiled_agent.py`) → 0 diffs ≥26 cells.
- T5.3 [R,S] legal masking + ≤10 market budget + hands positional alignment;
  `tests/test_agents.py` green.
- T5.4 [R,M] tarball build + latency run → worst turn < 1 s (target < 200 ms),
  0 fallbacks.
- **G5:** parity + latency + legal full episode.

### Phase 6 — Gate, holdout, ship (G6 = operator go)
- T6.1 [R,M] banded McNemar vs tape floor + crown panel.
- T6.2 [R,S] reserved-holdout replay (never in selection).
- T6.3 [R,S] `reactive_parity`/`serve_allowed` with trackp in roster.
- T6.4 [L,S] second-slot/intraday-fix rule compliance.
- T6.5 [L,S] model graph + docs refreshed.
- **G6:** explicit operator go. Never submits.

### Phase 7 — Optional decision-time search (G7)
- T7.1 [R,L] PUCT (policy priors + value leaf) on the bundled Rust engine; PIMC
  determinization.
- T7.2 [R,M] top-k prior pruning to stay < 1 s/turn.
- T7.3 [R/L,M] gate search-ON vs OFF on the panel — must beat OFF or stays off.
- **G7:** search ships only if it beats the cheaper search-off policy.

---

## 5. Compute budget & milestones
| Stage | Venue | Time |
|---|---|---|
| Data prep (100M corpus) | local CPU | ~3–8 h (one-time) |
| Upload (~12 GB) | local→Kaggle | ~1.3 h (one-time) |
| BC epochs 1–2 | Kaggle T4×2 | ~11 h (one session) |
| ckpt download + BC epoch 3 | local 4060 | ~17 h |
| RL to 500M | local 4060 | ~5–14 h (locked at T2.3) |
| RL 500M→1B (if climbing) | local | +~5–14 h |
| deploy bake + gate | local | ~2–3 d |

Gate ladder: G0→G1→G2→**G3 (KILL)**→G4→G4b→G5→G6→G7(opt). Hard stop at G3 →
tape-pair fallback.

## 6. Risks / mitigations
- Executor wall recurs → end-to-end policy + G3 kill gate.
- Composite action hard → encoder-once + pointer decoder + masking + BC-primary
  + return-conditioning; v1 factorized fallback.
- Rust↔torch drift → non-skippable parity (T2.2, T5.2).
- RL throughput below estimate → measured T2.3; compact obs + small model +
  compiled bf16; gate early-stop.
- Deploy latency > 1 s → ~2M model + baked Rust; measured T5.4.
- Overfit to selection panel → reserved holdout + banded gate.

## 7. Next action
On operator "go": start **T0.1** (token spec) + **T0.2** (action spec). No
pipeline code executes before G0 sign-off.
