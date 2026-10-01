# Codebase audit + detailed forward tasklist (2026-09-19)

Verified by import/symbol audit + gate runs, not by claim. Status:
✅ done+verified · 🔄 partial · ▶ not started · ⚠️ built-but-rejected · ⛔ external.

---

## PART A — AUDIT: what's actually complete

### Verified present + working (import audit + gates)
- **14/14 Slot-2 modules import clean**; all claimed symbols + `BatchRoller`
  methods present. Docs (7) + artifacts (bc_policy.pt, trackp_weights.npz,
  world_freq.json, seed_bank.json, release.py) present. 3-way Rust bins
  (kagg-engine/bandit/trackp) build. **Bandit parity gate PASS** (ship==measure).

### Infrastructure — COMPLETE (~95%)
| Area | Items | Status |
|------|-------|--------|
| Faithful harness | serve S5/S3/S4 fix, reactive parity gate | ✅ |
| Crown panel | 28 banded refs, paired-McNemar gate | ✅ |
| Corpus + pool | 440k macro rows, 786 ≥2500 opponents, prewarm | ✅ |
| **Executor** | X1 season_cbs, X2 cow/goose, X3 land, X5 fertilizer | ✅ |
| **Plan quality** | P1 guardrails (0/6 bank-0) | ✅ |
| **Training dynamics** | T1 league, T2 world-cond, T3 world-gen, T4 teacher-KL, A5 critic | ✅ |
| **Speed** | S1 batch (7.7ms), S2 serve gate (3.5x, bit-exact), S3 gpp, S4 battery-batch, S6 bounded-parallel-compile (1.3x, cap 6) | ✅ |
| **Bandit seat** | B1 compile, B2 build, B3 dispatch D6/D12/D15/D21/D27+rails, B5 parity, B6 vs-v46, B7 route2 | ✅ |
| **Track-P seat** | R1 ONNX, R2 numpy inference (==torch), weight bundle packs | ✅ |
| **Gating/evals** | G1-G5, full_gate (banded release gate, both harnesses) | ✅ |
| **Release** | versioning v59 (major/day, minor/intraday), config-lock, private notebooks, submit checklist | ✅ |
| **Orchestration** | slot2_pipeline, slot2_loop (resumable), daily_slot2 (build+gate+lock+release) | ✅ |

### NOT complete — the honest gaps
| Gap | Why it matters |
|-----|----------------|
| **The actual TRAINING RUN to convergence** | The loop WORKS (train_win climbs) but has NOT been run to produce a competitive policy. **This is the #1 gap — everything else is infra waiting for a trained NN.** |
| **B4 branch re-evolution** | bandit reuses egg-tape branches; needs policy-conditioned branches per checkpoint-world |
| **R3-R6 trackp bundle + gates** | weights pack (R3 core done); the self-contained main.py assembly + latency/competitive gates remain |
| **A2/A3 algorithm evals** | IMPALA/DT not run (need trained baselines to compare) |
| **P2-P5, T5, O6** | plan-quality refinements, curriculum, corpus delta |
| **Kaggle discussions** | 9 URLs could NOT be fetched (JS/auth SPA); need browser-session or paste |
| **D1/D4 data** | download drain + ourgames — CDN 429 blocked |

**Bottom line:** the two-harness config-driven pipeline is built, gated, versioned,
release-ready, and memory-safe. What's missing is (1) running it to train a
competitive policy, and (2) a handful of optimizations (B4, R3-R6) + evals. No
core piece is broken.

---

## PART B — DETAILED FORWARD TASKLIST (explicit descriptions)

Effort: S ≤½day · M ~1day · L 2-3day. Each task states WHAT it does, HOW, EXIT.

### TR — Training (the #1 gap)
- **TR1 · Full training run to convergence** · L · ▶
  WHAT: run `slot2_loop --rounds N` (or `daily_slot2`) on the 4060 with the
  now-working loop (P1+X2-X5+T1+T2+T3+A5) until the gate clears a band vs the
  crown panel. HOW: BC warmup once, then RL rounds with league growth + world
  sampling; resume via loop_state; escalate games/round. EXIT: `full_gate`
  aggregate ≥ 0.9 with no band regressed, i.e. a real release candidate.
- **TR2 · Training telemetry + early-stop** · S · ▶
  WHAT: log per-round bank distribution, per-band gate, league size, std; stop
  when the gate plateaus. HOW: extend loop_state history + a plot/JSON. EXIT:
  a readable training curve; auto-stop on no-improvement.
- **TR3 · Hyperparameter sweep (config search)** · M · ▶
  WHAT: sweep the RL knobs (kl_coef, init_std, gpp, league_frac, margin_shaping,
  d_model) to find the OPTIMAL CONFIG (the daily job's real purpose). HOW: a
  small grid/random search calling `macro_rl.train`, ranked by `full_gate`.
  EXIT: the best config recorded per day; fed to the release lock.

### PQ — Plan quality (raise the economy toward ~100k)
- **PQ1 (P3) · Bootstrap-recipe extraction** · M · 🔄
  WHAT: replay 5-10 ≥2900 opponents on their own seed; extract their opening
  sequence (buy/hire/plant/land cadence) that banks ~100k from $3000. HOW:
  parse `opponent_actions`, tabulate day-0..6 spend + product mix. EXIT: a
  documented bootstrap recipe the BC/guardrails bias toward.
- **PQ2 (P2) · Bank-0 edge-case audit** · S · ▶
  WHAT: sweep sampled plans, catch any residual bank-0 (e.g. all-slow-crop),
  tighten `sanitize_plan`. EXIT: 0 bank-0 across 100 sampled plans.
- **PQ3 (P4/P5) · Reward + return-conditioning check** · S · ▶
  WHAT: confirm reward variance > 0 post-P1 and rtg=1 plans out-bank rtg=0.
  EXIT: measured monotonicity; else fix the conditioning feature.

### EX — Executor refinements
- **EX1 · Right-size heuristic in the executor** · M · ▶
  WHAT: cap animals/land to what the realized cash bootstrap supports (the
  executor currently trusts the plan; naive animals/land are net-negative).
  HOW: an internal cash-aware clamp mirroring `sanitize_plan`. EXIT: animal/land
  plans no longer under-bank a crop-only plan unless genuinely profitable.
- **EX2 (X4/S5) · Obs-free executor** · L · ⚠️REJECTED
  WHAT: dead-reckon the board to compile with 0 serve. STATUS: built, fidelity
  failed (72-86% low). Re-open only with an exact field-mechanics re-port.

### BA — Bandit seat completion
- **BA1 (B4) · Policy-conditioned branches** · M · ▶
  WHAT: make the bandit world-adaptive from the SAME NN — at each checkpoint
  (D6/D12/D15/D21/D27) compile the policy conditioned on each common world
  (`world_freq`) → that plan is the branch for that world. HOW: loop
  `compile_base_tape(world_bucket=W)` per world; wire into `build_multi_ckpt`'s
  branch/splice format; dispatch picks by observed shops. EXIT: branches match
  the learned base; gate ≥ base-only.
- **BA2 · Bandit guardrail config surface** · S · 🔄
  WHAT: expose the rail set/order + knobs as the LOCKED config (already
  config-driven in mbandit; surface it in the release config). EXIT: a rail
  toggle/knob change is a config edit, no rebuild.

### TP — Track-P seat completion
- **TP1 (R3) · Self-contained trackp submission bundle** · M · 🔄
  WHAT: assemble the submittable trackp agent = `main.py` that loads
  `trackp_weights.npz` (packed ✅), runs `native_infer.numpy_forward` to sample
  the plan, then `SeasonCbsController` to realize it. HOW: a bundle dir (main.py
  + weights.npz + inlined executor + native_forward) tar'd, OR a package-import
  main.py. EXIT: the bundle runs `agent(obs)` standalone, 0 exceptions.
- **TP2 (R4) · Latency gate** · S · ▶
  WHAT: assert worst-turn < 1s for the trackp bundle (numpy infer ~30 calls/game
  is cheap; verify). EXIT: measured worst-turn.
- **TP3 (R5) · Competitive gate + release** · S · ▶
  WHAT: run `full_gate` on the trackp bundle, lock config on SHIP. EXIT: trackp
  release candidate produced. (Wired in `daily_slot2.stage_release`; needs TP1.)
- **TP4 (R6) · Guardrail overlay for trackp** · M · ▶
  WHAT: reactive guardrails on the live NN (e.g. don't-dump SELL clamp, cash
  floor) as a config surface, mirroring the bandit rails. EXIT: config-driven
  trackp guardrails.

### AL — Algorithm evals
- **AL1 (A2) · PPO vs IMPALA** · L · ▶
  WHAT: add a V-trace off-policy variant; train both on a fixed ~20k-game budget;
  compare sample-efficiency + wall-clock; keep the winner. EXIT: measured verdict.
- **AL2 (A3) · pointer vs Decision-Transformer** · L · ▶
  WHAT: compare the current head vs a full return-conditioned DT on BC-fit +
  RL-finetune. EXIT: measured verdict.

### SP — Speed (further, all memory-bounded)
- **SP1 · Parallel gate** · M · ▶
  WHAT: run the crown-panel ref matchups across a BOUNDED process pool (cap 4-6).
  EXIT: 13-min crown gate → ~2-3 min, no memory blowup.
- **SP2 · Corpus delta (O6)** · M · ▶
  WHAT: `bc_corpus` folds only NEW episode ids instead of re-processing all
  parts. EXIT: a daily corpus update is incremental.

### DA — Data (external-blocked)
- **DA1 (D1) · Download drain** · S · ⛔ CDN 429 — supervisor drains on reset.
- **DA2 (D4/C3.1) · ourgames → corpus + ladder calibration** · M · ⛔ CDN 429.

### CO — Competition intel (READ via browser 2026-09-19; folded into tasks below)
Read discussions 741935, 741907, 741797, 741792 (others wouldn't render). Findings:
- **741907 (Marcelo)**: DAILY-RESET TOLL — farmer+hands teleport to shed (4,4) daily,
  hands deleted+re-hired, HIRE cost resets → ~15pp walking toll. Gains SATURATE ~5
  cells (the walk-back, NOT lack of land). Farm hands often DON'T pay for themselves
  (toll + Fibonacci). MELON banks ~24.2k = 3× wheat with FEWER visits (17 vs 61 harv).
  FERTILIZER (+$2,725) beats the animal product (25 eggs ~$1,250). Traps: 2 units
  PLANT w/ 1 seed = silent no-op; FEED needs wheat in UNIT inv; melon max_yield_day=12.
- **741935 (Matheus)**: routing look-ahead — score tiles `action_weight/(distance+1)`,
  Top-10, DEPTH-2 look-ahead (c1 then c2), avoid map-crossing.
- **741797 (Young Uk Song, ~1221)**: BC→RL; "offline imitation good, closed-loop
  UNSTABLE" (our exact pain). Stabilizers: KL-to-clone, critic warmup, low LR (we do).
- **741792 (Sayaka Miki, 13th)**: "**Yes it's RL**"; the field converged to a HACKABLE
  local-optimum copied from the public route notebook. Krzysztof: shop-dependent play
  needs an "online solver / continuation per variant" (= our world-cond + branches).

**CORRECTED by the georgymamarin 2600-farm analysis (silver medal) — SUPERSEDES the
Marcelo/7858th near-shed/melon-only/hire-light advice. The ~100k economy = a WHEAT-FED
ANIMAL HERD (milk $264) + land day 6 + ~12 crew. Our architecture is right; we UNDER-SCALED.**

- **CO2 · Scale to the wheat-fed-herd economy (THE ~100k recipe)** · M · ▶ **(HIGHEST VALUE)**
  WHAT: bias the plan/executor to the measured top profile — **first land day 6, peak
  crew ~12 (~10/day hires), WHEAT-HEAVY (~160 plantings) feeding a BIG COW herd → MILK
  $264 + WOOL + EGG + fertilizer, plus STRAWBERRY(~33)+CARROT(~31) cash crops, few
  MELON(~12) don't-dump.** HOW: revise `sanitize_plan` defaults + BC target weighting +
  the config-sweep priors to this profile; scale X2 husbandry from 2-3 to a big herd
  fed by the wheat. EXIT: policy plans bank toward ~90-100k (median win 92,792), not ~18k.
- **CO3 · Milk-first husbandry (cows > sheep)** · S · ▶ **(HIGH VALUE)**
  WHAT: MILK ($264, cow) is the top product; scale COWS fed by wheat. Bias animal buys
  to cows; ensure the herd is fed (wheat inventory sized to herd). EXIT: milk revenue is
  the dominant line.
- **CO4 · Land day-6 + crew-12 config** · S · ▶
  WHAT: the top profile buys first land ~day 6 and hires to peak crew ~12. Set these as
  config/sweep priors (our defaults hire 8-10, buy land later). EXIT: sweep centers on
  land-d6 / crew-12.
- **CO5 · Fertilizer-priority economy** · S · ▶
  WHAT: fertilizer collection is a real revenue line (by-product of the herd). Keep X5's
  COLLECT_FERTILIZER prioritized. EXIT: fertilizer revenue realized alongside milk/wool.
- **CO6 · Depth-2 routing look-ahead** · M · ▶
  WHAT: replace within-cluster nearest-first with `action_weight/(distance+1)` Top-10
  + depth-2 (c1→c2) look-ahead so a mover doesn't cross the farm for one action.
  EXIT: measured routing efficiency up vs nearest-first.
- **CO7 · Exploit the copy-cat meta (self-play edge)** · M · ▶
  WHAT: the field is a hackable local optimum (13th place says so). Ensure the league
  (T1) + world-conditioning (T2) push the policy to a strategy the copied route
  baseline doesn't counter; add the public route baseline (haideptry) as a gate ref +
  a league seed to explicitly train the counter. EXIT: our policy beats the public
  route baseline decisively on the gate.
- **CO8 · PLANT-collision guard in the executor** · S · ▶
  WHAT: two movers issuing PLANT against a 1-seed pool = silent no-op (documented
  trap). Guard: don't assign PLANT to more movers than available seed of that crop.
  EXIT: no wasted PLANT turns.

---

## PART C — recommended sequence (dependency, not priority — operator decides)
1. **TR1** (train — unblocks everything; the loop is ready).
2. **BA1** (policy branches) + **TP1** (trackp bundle) — complete both seats from the trained NN.
3. **CO1** (discussions) — fold competition intel in (needs browser/paste).
4. **PQ1-3, EX1** — push the economy toward ~100k.
5. **SP1-2, AL1-2** — speed + algorithm evals.
