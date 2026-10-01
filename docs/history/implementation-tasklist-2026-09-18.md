# Implementation task list (2026-09-18)

Companion to `engine-tooling-master-plan-2026-09-18.md` and
`harness-revamp-plan-2026-09-18.md`. Planning artifact — the tasks are specified,
NOT implemented. Effort: S ≈ ≤½ day, M ≈ 1 day, L ≈ 2–3 days.
Priority: **P0** must-do (certain ladder value), **P1** high, **P2** R&D bet.

**Goal: leaderboard top 10 (2140 → ~2900).** Shipping frame = the two-slot hybrid
(master §0.5): **Slot 1** = OR economy + PC-TAPF routing + plan-repair shell (the
certain economy lever); **Slot 2** = evidence-gated learned dispatcher/timing
predator, else a diversified OR `route2`. Both trusted only via the faithful
harness (A) + crown (C). Integrates `research/Advanced Algorithms...` +
`research/Dynamic Labor Routing...` (core ideas only — see master §0.5 / §E).

---

## STATUS LOG (execution started 2026-09-18)

Legend: ✅ done+verified · 🔄 in progress · ⛔ blocked (needs operator/compute) · ▶ next.

**2026-09-18 — Fidelity pre-phase (A0/A1 serve fixes = G7.S5/S3/S4):**
- ✅ **A0.4** — pinpointed the reactive mis-rank ROOT CAUSE empirically. Not an obs-content
  bug (confirmed faithful). It is the serve **invocation layer**: serve solicited 720 actions
  (steps 0..719) vs official's 719 (0..718) — an extra step-719 action + extra day-30
  end-of-day. Controlled A/B (agent_vadapt_both vs agent_vadapt_risk): OLD-loop banks diverged
  from official on 2/3 seeds (seat off by ~440 & ~1020 coins — flips close games); seed 5
  bank-neutral (why tape-shell serve_equiv passed 12/12 by luck).
- ✅ **A1.1 / G7.S5** — serve stops one step early (`run_match` loops while `step<episodeSteps-1`);
  serve == official now by construction. Driver-only fix; Rust `serve` STEP2/done untouched so
  RL self-play (GENGAME/ROLLOUT/PLANSEARCH, all-720) is unaffected.
- ✅ **A1.2 / G7.S3** — arity-aware `_call`: 2-arg agents now get the resolved `configuration`
  (kaggriculture.json defaults), truncated to `co_argcount` exactly like vendor `Agent.act`.
- ✅ **A1.3 / G7.S4** — `obs_for` now `_structify`s a fresh per-seat deep copy (mirrors vendor
  `structify`), killing cross-seat aliasing/in-place mutation.
- ✅ **Verify** — reactive `--compare-official` now 3/3 exact-bank MATCH (was diverging).
- ▶ A1.4 lock obs-parity test · A1.5 one canonical obs builder · A2 reactive parity gate (≥6
  reactive agents) → regen `serve_equiv.json` → tighten `serve_allowed()`.

File touched: `src/kaggriculture/engine/serve_match.py` (EPISODE_STEPS, CONFIG, _Struct/_structify/_call, obs_for, run_match).

**2026-09-18 — A2 reactive parity gate CLOSED + A1.4 lock:**
- ✅ **A2.1/2.2/2.3** — new `kaggriculture.engine.reactive_parity` (6 reactive agents, both seats, 3 seeds
  = **36 games, 36 match / 0 mismatch**, engine 1.32.7). `models/serve_equiv.json` regenerated with a
  `reactive_agents` roster (old tape-shell cert → `.local/serve_equiv.tapeshells.backup.json`).
  `serve_allowed()` now enforces `REACTIVE_MIN=6`; verified green. The tape-shell blind spot is closed.
- ✅ **A1.4** — `tests/test_serve_obs_parity.py`: serve obs is field-for-field faithful to official (PASS seeds 3,4).

**2026-09-18 — Ship-path CRITICAL/HIGH bugs (parallel subagents):**
- ✅ **C2** (VERIFIED) — `build_rust_bandit.py` cross-builds a musl ELF + asserts `\x7fELF`. Confirmed: `file` = "ELF 64-bit LSB pie, x86-64, static-pie linked" (`.local/candidates/c2_verify.tar.gz`). T1's policy.rs also compiles clean in the musl target. (Run it as a SCRIPT — bare `import build_bandit_harness` needs its own dir on path; latent import bug logged for G0/G6.)
- ✅ **H8** — bandit driver re-spawns / survives a cold turn (MAX_MISSES=3, first-turn warm budget, module-load warm-up).
- ✅ **T1 DEPLOYED + VERIFIED** — quadrant assignment ported INTO `policy.rs`; **`rustengine/kagg.exe` swapped to the fixed binary** (operator directed); identity gate PASSES on the LIVE binary (0 diffs, 0 fallback, worst 38ms), engine still bit-exact. G7.T1 + IDGATE(trackp) + deploy all closed.

**2026-09-18 — "Complete ALL" execution (4 parallel subagents, isolated builds):**
- ✅ **B0.1** — OR/MILP economic-model spec `docs/history/or-economic-model-spec-2026-09-18.md` (11 sections; secant-cut PWL needs NO binaries; caught README errors — CARROT scarcity $70 not $42; HiGHS-primary).
- 🔄 **Crown C0–C3** — banded referee panel + paired-McNemar-per-band gate (fetch live-LB ratings).
- 🔄 **Slot-2 data (E0.1/E0.3/E1.2/E1.3/E4)** — BC corpus distill + macro/micro action space + dispatcher (non-GPU; training waits for GPU 05:30).
- ✅ **Bandit Rust≡fallback (G7.C1/H3–H7/M9–M11 + G6.1/6.2)** — **Approach B**: fallback demoted to loud legal-PASS; economy only in `kagg mbandit` = ship==measure by construction; M9 config-truthfulness fixed; non-skippable parity + dead-config gates verified PASS.
- ✅ **OR solver — CORE VALIDATED END-TO-END, GAP $0** — A_c absorption table derived from the engine (5/5 exact), multi-day season model (B1.4) reproduces engine Cash720 to the dollar, and a full-season `.tape` validated: predicted $5908 == actual $5908. The market+absorption math is PROVEN EXACT. **Honest scope:** the validated economy is small ($5908 vs ~100k bar) — competitiveness needs the volume layers (labor/multi-tile/animals/land) + realistic routing (B2 PC-TAPF is now on the critical path — an idealized plan won't realize without it) + v0-scenario robustness. Next OR chunk resumed.
- ✅ **Slot-2 data pipeline (E0/E1/E4)** done.
- ✅ **Crown C0–C2 done+verified + FULL BAND FILL** — live LB (9,396 teams), banded gate (paired-McNemar + ladder-weighted, 6/6 tests, freeze gone). Panel now **28 refs, every band ≥3 incl. 2700+** (yamakawanin 2837 ×3 reactive + rayk 2763 + 2 frontier tapes) — the operator's point that the downloaded publics carry scores (author_ladder) closed the gap. 🔄 C3.1 waits on the data agent's ourgames index.
- ✅ **F3.3** animals/fert audit — winners invest more + capture WOOL (+$5,134/g); MILK/EGG over-produced by losers. OR should route to WOOL scarcity, not raw volume.
- ✅ **OR econ optimizer — validated $40,808** (6.9× baseline; MELON+CARROT+WOOL stack, don't-dump/right-size/grow-feed; +$14k over naive). **Clean scoping:** economic model is EXACT and optimizer policies PROVEN; greedy routing SATURATES at ~25 tiles → **the remaining ~$60k to competitive (~100k) is a ROUTING problem (CBS/Hungarian), not economics.** Plus a confound: shippable tape needs a-priori v0 shop-world SCENARIO optimization (world-realized-by-play). See [[or-scaling-law-2026-09-18]].
- ✅ **OR routing SOLVED** (persistent-cluster, 86-87% at 50-100 tiles; cross-world $36.5k CV 4%) → binding constraint moved to the single-seat market.
- ✅ **OR diversification → SHIPPABLE TAPE $48,482** — single FIXED tape across 8 worlds (mean $48.5k, min $46.3k, CV 4%, all replay==record). MELON $25k anchor + CARROT $14k + MILK $25k ($321/u deep scarcity, #1 lever). Right-size makes fixed sells world-robust (no dumping off-world). `rustengine-or/data/full_seed42.tape`.
- **TRUE CEILING diagnosed = CAPITAL BOOTSTRAP** (not demand ~$90k, not routing): the 30-day cash bootstrap bankruptcy-cascades on bigger early herds/land (8 cows→$34k, 9→$0). Captured ~$23k of milk's $45k budget; STRAWBERRY $34k UNTAPPED. The ~$28k gap to the ~76k bar (2500 tapes, which produce more UNDER contention) = bootstrap-CMA-ES tuning + strawberry + distributed husbandry. NOT a hard ceiling.
- ✅ **OR single-seat track COMPLETE** — bootstrap tuning plateaus ~$51k (knife-edge: 8 cows bankrupts, strawberry can't stack — all products fight the $3000/30-day capital bootstrap). Final fixed tape `rustengine-or/data/full_seed42.tape`: **mean $49,160 cross-world, CV 5%, predicted==actual**. Ceiling = CAPITAL BOOTSTRAP (not market ~$90k, not routing 87%). **VERDICT: ship the OR tape WRAPPED in the bandit reactive shell (B4.2), not bare** — bare falls ~35% short of the ~76k contested bar; the OR tape is a validated world-robust economic BACKBONE, the reactive shell adds contention-aware selling. See [[or-scaling-law-2026-09-18]].
- ✅ **B4.2 — PIVOTAL NEGATIVE + STRATEGY PIVOT.** Wrapped OR economy = **0.000 vs every opponent/band** (v58-floor). Wrapping/sell-timing/routing are NOT the gap (variants within 1%) — the OR economy is **~3× too small** (banks ~$49k vs PASS; ladder opponents bank $140-172k same games). **This rules out the OR-fixed-tape-alone Slot-1 path** and confirms the master plan's primary lever: **Slot-1/2 = BC→macro-RL** learning the top agents' large adaptive economy. The OR work's lasting value = an EXACT validated simulator + a BC-warmup seed. NOW WELL-RESOURCED: destbreso (786 ≥2500 / 560 ≥2700 opponent economies w/ full tapes) + our corpus = the BC targets. GPU tomorrow closes it. See [[or-wrapped-gate-2026-09-18]].
- ✅ **Data acquisition** — top-200 queued (36,634 new IDs → 72,115, drains on CDN reset); ourgames 429-blocked (C3.1 waits on reset); **`destbreso` pulled = 45,404 replayable matchups, 786 ≥2500 / 560 ≥2700 teams w/ tapes+ratings+banks, 429-FREE** (the high-rated opponent/BC corpus, available now) + rayk 10 graded agents + vijaikm BC. USE for E0.1/E0.3 augmentation tomorrow + supplemental panel.
- ✅ **Harness refactor (G0.1/G0.2/G0.3/G1.2/G1.3/G2.1-2.2)** — Rust DECOUPLED (core.rs; no `use crate::policy`), bandit tooling → `bandit/build`, tables → `config.tables`, genome → `genome.json` (both loaders, positive-control proven), 2-way `kagg-bandit`/`kagg-trackp` bins (additive, no consumer breakage). Isolated build + all 3 gates GREEN, shipped agents BIT-UNCHANGED. Verified: no stale Python imports, C2 musl build intact at new path. CLAUDE.md synced (G6.3). DEFERRED w/ rationale: G1.4 rail-reorder (no-git risk), G1.1 retire bandit.rs (kept as ref seat), 3-way split + consumer repoint (deploy follow-up).

**Verdict on Workstream G (operator's "completely separate / hot-swappable"):** the CORRECTNESS half is DONE both tracks — T1 (trackp Rust≡Python, deployed), Bandit ship==measure (Approach B) + config-truthfulness + non-skippable parity gates, C2 musl ELF, H8. What remains in G is pure ARCHITECTURE (G0 split one ELF → per-track binaries + remove `use crate::policy`; G1.2/1.4 externalize tables + ordered rail registry; G2 genome→JSON; G4.1 per-track musl scripts; G5 native ONNX inference [needs the GPU-trained policy]). These don't move the ladder but the operator asked for them → sequenced next as an isolated-build refactor once the current wave frees rustengine.

**2026-09-18 — B4.0 RE-BASELINE (the payoff of the fidelity fix):**
- ✅ **ref_v46=0.917 > ref_k0006=0.792 >> v57=0.458 > v56y=0.333 >> v58=0.000** (both seats, 6 seeds,
  paired vs v46 all significant; v56y & v58 both 0-12, p=.0005). Confirms operator: 2500 tapes are the
  bar, v56y is a floor, v58 artifact is stale-tape-weak (→ G7.5 rebuild). Gate Slot-1/2 at ~0.9 vs v46.
  See [[rebaseline-faithful-2026-09-18]].

**2026-09-18 — SLOT-2 TRAINING PIPELINE + EVAL HARNESS (all CODE-ready, CPU-smoked, GPU run left to operator):**
- ✅ **E2.1** — `train/bc_warmup.py`: return-conditioned MACRO BC. Masked causal Transformer
  (configurable ~10-20M params), per-day input = day/world-bucket/return-to-go/rating +
  cumulative-economy; heads = 7-way class CE + 26-dim action Smooth-L1. Streams the 440k-row
  macro corpus, BF16 on CUDA. `--smoke` (CPU) trains on the REAL corpus (40 seqs), saves
  `models/rl/bc_policy.pt`, verifies reload. Warnings cleaned (nested-tensor off, loss.detach).
- ✅ **E3.1-E3.5** — `train/macro_env.py` + `train/macro_rl.py`. **macro_env** = faithful reward:
  a plan-parameterised greedy controller (`PlanController`) plays CLOSED-LOOP on the vendored
  engine vs a REPLAYED destbreso opponent (786 ≥2500 tapes materialised from `opponent_actions`),
  returns terminal banks → win/draw/loss (+ margin shaping). Smoke: vs PASS bank 3862 (win, full
  719-cell tape); vs a 3139-rated real opponent → 3708 vs 134012 (faithful loss = the B4.2
  production-volume gap, live). **macro_rl** = PPO self-play (autoregressive plan sampling,
  clipped surrogate, EMA baseline, **teacher-KL to frozen BC prior**, **gated promotion** vs a
  frozen battery, frozen-self league member). `--smoke` runs 1 iter end-to-end on CPU → saves
  `models/rl/rl_policy.pt`. Executor is a pluggable hook (PC-TAPF B2 = the ceiling upgrade).
- ✅ **E5.2 / F1.1(export)** — `train/package_policy.py`: ONNX export (fixed T=30, causal-pad),
  **onnxruntime==torch parity gate** (cls/vec diff ~1e-7, refuses to ship on mismatch) +
  self-describing `policy_manifest.json` (norm + exact feature recipe for the native-Rust runtime).
- ✅ **Eval harness (operator's 4-eval-type request)** — `measure/smoke_test.py` (item 1: runs
  exception-free both seats, counts errors) + `measure/eval_harness.py` unifying: (1) smoke,
  (2) seat-swapped paired win rate vs a ref roster, (3) **replay clean-room** vs transcribed
  destbreso opponents banded by ladder rating, (4) logistic-Elo fit. `gate_packaged_policy()` =
  Slot-2 ship rule: smoke + seat-swapped vs the strong refs, **SHIP iff ≥0.9 vs v46** (B4.0 bar),
  else diversified route2. All faithful (post-S5/S3/S4 serve fix).
- ✅ **`pipeline/slot2_pipeline.py`** — one-command orchestrator (corpus→bc→rl→package→gate);
  `--smoke` proves ALL stages wire on CPU (~2 min, verified green). `docs/history/slot2-training-runbook.md`
  = RTX-4060 run commands + honest ceiling/upgrade notes.
- ✅ **`pipeline/slot2_loop.py`** — the HANDS-FREE loop (operator-run, NEVER submits): BC (once) →
  [RL round w/ escalating budget → package → gate vs v46] × N → on SHIP (≥0.9) compile policy →
  bandit base tape → `build_rust_bandit` (isolated) → parity gate → write `SUBMIT_INSTRUCTIONS.md`
  → STOP. Resumable (`--resume`, `models/rl/loop_state.json`); `--ship-only`; `--fresh-bc`.
  `docs/history/slot2-operator-runbook.md` = run/improve/loop/submit. **Launched a pilot on the 4060**
  (BC 10ep/62s val 0.07; RL rounds running). BUG FIXED: BC positional-embedding overrun on
  duplicate-day rows (dedupe+cap). **Only the full GPU training run + executor B3 are left.**
- See [[slot2-training-pipeline-2026-09-18]].

---

Workstreams: **D** data foundation · **A** faithful harness · **C** crown
de-saturation · **B** Slot-1 OR economy + PC-TAPF + plan-repair · **E** Slot-2
BC→macro-RL predator · **F** inference runtime + advanced optimizations + analysis ·
**G** harness separation & full configurability (bandit ⊥ trackp, hot-swappable).

**Operator (2026-09-18): EVERYTHING is in scope** — nothing excluded; sequencing (not
inclusion) is the only lever. **Strong gate references = the two LIVE ~2500 public
tapes** `submission_competitive_v46` (2517) + `submission_k0006_open10_h24_frontload_advance2_v43`
(2494), NOT v56y (2072) or v58 (463). Re-baseline on the faithful harness first (B4.0).

---

## Workstream D — Data foundation (P0, ongoing; unblocks A/C/B panels)

| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| D0.1 | **Download LANE (running): supervisor** `download_supervisor.sh` — probes kaggleusercontent every ~20 min, drains the worklist when the daily 429 window reopens (async fetcher, 4/s, 429-aware). NOTE: kaggleusercontent currently 429s (today's window burned by ~10.5k pulls); resets daily. | — | S | 24,905 remaining fetched (minus permanent 404s) |
| D0.2 | Consolidate new `.gz` → zstd parquet, delete gz | D0.1 | S | all top-100 union bodies in `top100_replays_parts/`, 0 corrupt |
| D0.3 | Reconcile: union (38,330) vs held; log any permanent-miss ids | D0.2 | S | coverage report; miss list |
| D0.4 | If operator supplies a rate-limit-free kaggleusercontent URL/pattern (e.g. per-episode signed GCS link), switch the fetcher to it | operator | S | unthrottled fetch path |
| D1.1 ✅ | **DONE 2026-09-18** — top-200 union 74,968 episodes; **36,634 NEW IDs** merged into the supervisor worklist `top100_download_ids.json` (now 72,115; backup kept). (Gotcha: `kaggle` module lives in BASE anaconda, not envs/llm.) | D0 | S | ✅ top-200 IDs queued |
| D1.2 🔄 | **QUEUED (429-blocked bodies) 2026-09-18** — the running supervisor will drain all 72,115 when the daily CDN window reopens (auto-dedupes held); no restart needed. | D1.1 | M | 🔄 draining on CDN reset |
| D4 (NEW) ✅ | **DATASETS PULLED (429-free) 2026-09-18** — **`destbreso/kaggriculture-benchmark-matchups`** (full, 487 MB): 45,404 rows, 2,497 opponent teams, ratings 104-3139, full 720-turn `opponent_actions` tapes + seed + both banks. **786 distinct ≥2500 teams, 560 ≥2700; 9,133 replayable rows ≥2500, 4,390 ≥2700.** Complementary (adds 21,483 episodes NOT in the worklist; only 1 seat's actions/row). + `vijaikm` BC tabular (both seats, no ratings, secondary) + `raykkretzschmar` 10 self-contained graded agents (floor→top). ▶ USE: opponent pool (E0.3) + BC corpus (E0.1) augmentation for tomorrow's GPU training; supplemental crown referees (rayk reactive = high-confidence; destbreso tapes = desync-risk). | D0.2 | S | ✅ high-rated 429-free corpus |
| D2.1 | Finish public-kernel harvest — **DONE: 125 agents** in `data/kernels/_agents`, ranked `high_agents_index.csv` | done | S | ✓ |
| D2.2 | Filter to self-consistent reactive agents: import-clean, `agent(obs)` runs 720 legal steps vs PASS, no crash | D2.1 | M | vetted public-agent set |
| D2.3 | Label + band each by author ladder rating (join to live LB / teams.csv); target ~6/band × {<2100, 2100-2300, 2300-2500, 2500-2700, 2700+} | D2.2 | S | banded public roster |
| D2.4 ✅ | **DONE 2026-09-18** — both submitted files staged to `.local/panel/ref2500/` (`ref_v46_chassis_2517.py`, `ref_k0006_2494.py`) + manifest; vetted faithful head-to-head (serve==official 2/2). Both self-contained (stdlib only). They are near-mirrors of each other. | operator/D2.1 | M | ✅ 2500-tape references in panel |
| D3.1 | Extract per-episode action tapes for high-rated seats from top-100/200 replays → scripted "replay opponents" (serve `OPP <seat> <tape>`) | D0.2 | M | replay-opponent tape set |
| D3.2 | Band replay opponents by the seat's recorded rating | D3.1 | S | banded replay-opponent panel |

---

## Workstream A — Faithful harness (P0, the deadline priority)

### A0 — Phase 0: reproduce & pinpoint the divergence
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| A0.1 | Pick 2–3 genuinely reactive agents (a public reactive from D2 + our `baseline_v56y` shell) | D2.2 | S | agent set chosen |
| A0.2 | Instrument the **serve** path to dump the per-turn obs (`obs_for` output) + returned action to a log | — | S | serve per-turn trace |
| A0.3 | Instrument the **official** path (kaggle-environments) to dump the same per-turn obs + action | — | S | official per-turn trace |
| A0.4 ✅ | **DONE 2026-09-18** — root cause pinpointed: NOT obs content (faithful); it's the invocation layer (extra step-719 action). Controlled A/B proved it (see STATUS LOG). | A0.1-3 | M | ✅ **named divergent cause** = S5/S3/S4 |

### A1 — Fix the serve INVOCATION layer (CORRECTED by the bug audit: obs CONTENT is already faithful)
The audit verified obs schema/values/**PRODUCTS-order**/types all match vendor ground
truth. So A1 is NOT an obs-schema rewrite — it's the three invocation-layer fixes (= G7.S5/S3/S4):
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| A1.1 (=S5) ✅ | **DONE 2026-09-18** — serve stops at step 719; serve==official by construction (verified 3/3). | A0.4 | S | ✅ serve turn count == official |
| A1.2 (=S3) ✅ | **DONE 2026-09-18** — arity-aware config pass in `_call`. | A0.4 | S | ✅ config-reading agents match official |
| A1.3 (=S4) ✅ | **DONE 2026-09-18** — per-seat deep-copy via `_structify` in `obs_for`. | A0.4 | S | ✅ isolated per-seat obs |
| A1.4 ✅ | **DONE 2026-09-18** — `tests/test_serve_obs_parity.py` drives serve + official with a PASS recorder and asserts all documented obs fields identical per step/seat. PASS over seeds 3,4. | A1.1 | S | ✅ obs-parity test green |
| A1.5 ▶ | One canonical obs builder (single source for serve + any Python path) — kill dual-path drift risk | A1.1 | M | one obs code path |

### A2 — Reactive parity gate (close the blind spot)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| A2.1 ✅ | **DONE 2026-09-18** — new `kaggriculture.engine.reactive_parity` runs 6 reactive agents, both seats, 3 seeds each = 36 games serve-vs-official. | A1, D2.3 | S | ✅ reactive exact-bank comparison runs |
| A2.2 ✅ | **DONE 2026-09-18** — `models/serve_equiv.json` regenerated: **36 match / 0 mismatch, 6 reactive agents, engine 1.32.7** (old tape-shell cert backed up to `.local/serve_equiv.tapeshells.backup.json`). | A2.1 | S | ✅ serve_equiv green WITH reactive agents |
| A2.3 ✅ | **DONE 2026-09-18** — `serve_allowed()` adds `REACTIVE_MIN=6` reactive-coverage requirement; a tape-shell-only cert now fails. Verified green: "36 exact-bank matches, 0 mismatches, 6 reactive agents". | A2.2 | S | ✅ gate can't read green on tape-shells alone |

### A3 — Fast agent execution (opponents + our policy) — see also Workstream F
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| A3.1 | Keep stdio `kagg serve` as the opponent-execution path (faithful after A1); it's the PyO3 alternate we already have | A1 | S | opponents run faithfully |
| A3.2 | (Throughput) Shared-memory IPC (memmap/named pipes) + persistent Python worker pool for many parallel envs — alternate to PyO3, avoids GIL contention | A3.1 | L | higher opponent throughput |
| A3.3 | (Optional) PyO3 in-process CPython embedding + zero-copy obs — if in-process opponent calls are wanted | A2 | L | in-process opponent execution |
| A3.4 | **Native Rust inference for OUR policy** (the PyO3 alternate, and best for the submission): export policy → ONNX; run via `candle` / `tract` / `ort`; no Python at runtime | E2/E3 | L | Rust-native policy inference, <1s/turn |

---

## Workstream C — Crown de-saturation (P0/P1, rides on A)

| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| C0.1 ✅ | **DONE + FILLED 2026-09-18** — live LB (9,396 teams) → banded panel `models/crown_panel.json`, **28 vet-passed refs, EVERY band ≥3 (no thin/empty)**: <2100:6, 2100-2300:4, 2300-2500:6, 2500-2700:6, **2700+:6** (yamakawanin 2837 ×3 REACTIVE + rayk 2763 + 2 frontier tapes 2905/2893 tagged `tape_desync_risk`, forced to selection not holdout). Banded via `author_ladder` from the 552 downloaded kernels; reactive prioritized over tapes. titan/bruceqdu excluded (vet-timeout). | D2.3, D3.2 | S | ✅ banded panel manifest (full coverage) |
| C1.1 ✅ | **DONE 2026-09-18** — `refresh_cycle.py`: `crown_panel_referees()` loader + `stage_crown_panel()` (scores best-vs-incumbent per band on serve, applies banded gate; authoritative on REGRESSION only, degrades to legacy). `serve_allowed`/`REACTIVE_MIN` preserved (still 36/36). | C0.1, A2 | M | ✅ crown runs on faithful reactive referees |
| C1.2 ✅ | **DONE 2026-09-18** — `referee_power.analyse` now protects band strata via `crown_panel.protected_prefixes()`; `prune()` refuses to empty a band. | C1.1 | S | ✅ discriminating referee set |
| C2.1 ✅ | **DONE 2026-09-18** — `crown_gate.py` `crown_gate_banded()`: per-band `win_metric.paired_test` (McNemar) + ladder-weighted aggregate (weight = band centroid). SHIP iff no band significantly regressed (p<0.05) AND aggregate diff>0. 6/6 unit tests. | C1.2 | M | ✅ banded, paired ship rule |
| C2.2 ✅ | **DONE 2026-09-18** — holdout alarm: 1 reserved referee/band (band≥3) never touches selection; divergence report-only. | C2.1 | S | ✅ holdout divergence report |
| C3.1 🔄 | **PENDING ourgames fetch (data agent owns it) 2026-09-18** — `ladder_band_winrate()` machinery verified clean (returns available=False on the 0-game stub, no crash/API-hammer). Re-run `crown_panel --ladder-error` once the data-acquisition agent populates `data/ourgames/index.json`. | C2.1, live games | M | 🔄 waiting on ourgames |

**Crown exit:** the gate discriminates (not frozen HOLD); ship rule = non-regression
in every band + no band regressed at p<0.05; 2500-2700 band trend = top-10 signal.
**Rust port of crown orchestration: NOT done** (matches already native; only A3 helps).

---

## Workstream B — OR tape generator (P2, time-boxed R&D)

### B0 — Model spec
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| B0.1 ✅ | **DONE 2026-09-18** — `docs/history/or-economic-model-spec-2026-09-18.md` (11 sections): objective Cash720, indices/constants (exact from interpreter), **staircase-revenue PWL via secant cuts (no binaries)** = the scarcity lever, production/biology, economy constraints, T-absorption, decomposition, HiGHS-primary solver stack, `.tape` contract. Caught README errors (CARROT scarcity $70 not $42; hinge below-func). | — | M | ✅ reviewable model spec |
| B0.2 ✅ | **DONE 2026-09-18** — standalone crate `rustengine-or/` (isolated; rustengine untouched). **HiGHS LINKS + solves** (`--features highs`, builds from source via cmake); pure-Rust `microlp` default fallback (no C deps, identical results) via swappable `good_lp` feature. Toy LP obj=10 on both. russcip/SCIP not needed for the linear PWL model. | — | M | ✅ working solver crates + toy solve |

### B1 — Layer-1 economic MILP (aspatial)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| B1.1 ✅ | **DONE 2026-09-18** — `rustengine-or/src/pwl.rs`: exact engine port of `market_price`/`_shape` (round-half-EVEN + $1 floor) + concave staircase `R(q)=Σprice(v0+m)` as secant cuts (exact at integers, NO binaries). 5/5 unit tests vs Python-engine ground truth; widths match spec (STRAW 62, WOOL 59, MELON 158, MILK 76, CARROT 842, TOMATO 529). | B0 | L | ✅ exact concave revenue columns |
| B1.2 🔄 | **A_c + SCALING DONE 2026-09-18** — absorption table exact (5/5 crops). Scaled + VALIDATED (replay==record): 10 hands+25t carrot $9.9k, melon12t $25.7k, land 75t melon+carrot **$27,359** (4.6× baseline, market still $0-exact), 6 sheep fed 0 escapes. **KEY: the OR offline-planner CLEARS the reactive "labour-logistics wall"** (fed a herd with full mover-state routing where reactive planners ran on ~1 worker). SCALING LAW: each product caps at its knee (MELON ~$30k regardless of tiles) → must STACK products sized to knees. ▶ needs the econ optimizer (B1.3) + CBS routing (B2). | B0 | L | 🔄 feasible farm-schedule constraints |
| B1.3 ✅ | **ECON OPTIMIZER DONE 2026-09-18** — `optimizer.py`+`season_stack.py`: don't-dump (cap sells at price knee), right-size herd to demand, grow-feed (wheat @$2.50 vs buy @$25), STACK products. **Validated $40,808** (MELON $27k + CARROT $7.9k scarcity + WOOL $13.1k held at $170/u even NO-YARN). Ablation: right-size = master fix, grow-feed 5×. +$14k over naive scaling. Cash-flow ledger + shed + labor all honored. | B0 | M | ✅ econ optimizer proven |
| B1.4 ✅ | **DONE 2026-09-18** — `rustengine-or/src/season.rs`: multi-day rolling model (per-day shed + cash≥0 ledger, absorption yields feed sell availability, staircase-PWL LP per day) **reproduces engine-validated Cash720 to the dollar**; HiGHS MIP land-tier decision (order-forced binaries). ▶ v0 must become a SCENARIO set (town-drain/opponent), not read from realized play. | B1.1-3 | L | ✅ tractable multi-day solve |

### B2 — Layer-2 routing realizer = PC-TAPF (spatial) + plan-repair
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| B2.0 ✅ | **DONE 2026-09-18** — greedy single-farmer routing packs the plan's ops/sells into a playable 719-row tape (validated $0 gap on the small econ). ▶ Dense multi-tile/multi-hand economies will show routing loss → need real PC-TAPF (B2.1-2.3). | B1 | M | ✅ a playable tape exists to validate |
| B2.1 | Precedence DAG builder (till→plant→water→harvest; animal place→feed→care→harvest), gated on tape completion | B1 | M | task DAG per world |
| B2.2/B2.3 ✅ | **ROUTING SOLVED 2026-09-18** — `season_cbs.py`: PERSISTENT per-mover→cluster assignment + dedicated husbandry movers + within-cluster Hungarian-lite + cash-bootstrap fix. Holds **86-87% efficiency at 50-100 tiles** (vs greedy's $14 collapse); weeds 1-4, sheep survive. Cross-world robust: 25t across 8 seeds = mean **$36,490, CV 4%**, all replay==record. **KEY: routing is no longer the bottleneck — the SINGLE-SEAT MARKET is** (more tiles crash carrot price; melon caps ~144u). **B3 (2026-09-18): PORTED into `macro_env.SeasonCbsController`** (plan-parameterised, seat-general) + made DEFAULT executor: **+158% (crops) to +726% (crops+sheep) over greedy**, batch path tracks closed-loop within ~1-2%. Exposed the NEXT layer: executor handles SHEEP only (COW/GOOSE plans bankrupt), and ambitious plans hit the CAPITAL-BOOTSTRAP wall (100-tile plan banks LESS than 20-tile). Frontier banks ~100k own-seed → the gap is now PLAN QUALITY + bootstrap + animal husbandry, not routing. | B2.1 | L | ✅ persistent routing; ported to RL executor |
| B2.4 | Infeasibility feedback: where routing can't realize schedule, tighten Layer-1 per-turn budget and re-solve | B2.3 | M | converged playable tape |
| B2.5 | **Dependency-directed plan-repair** shell: annotate tape with causal deps; on shock (weed/shop-unlock/land/mirror) excise fractured links + patch locally (regression search + localized Hungarian-CBS), Rust arena rollback | B2.3 | L | tape survives realized stochastic worlds |
| B2.6 | Action-mask hook: expose engine legality (cash, shed, tile, routability) as a mask (shared with Workstream E) | B2.0 | S | legality-mask API |

### B3 — Validation loop (the simulator closes the loop)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| B3.1 ✅ | **DONE 2026-09-18** — `tools/season_tape.py` writes a 719-row `.tape` (tab-sep, ints, ≤10 orders); replays OPEN-LOOP == closed-loop record (plays legally). | B2 | S | ✅ byte-valid tape |
| B3.2 ✅ | **DONE 2026-09-18 — GAP $0.** Validated world (seed 42 YARN, single-farmer CARROT econ): **model predicted Cash720 $5908 == engine actual $5908 (+0.00%)**. The staircase market model matches the engine's per-unit crediting EXACTLY; scarcity edge live (realized $36→$47/u). ⚠ Economy is SMALL ($5908 vs ~100k bar) — correctness proof, not yet competitive; needs volume (labor/tiles/animals) + B2 routing. | B3.1 | M | ✅ validated-cash report ($0 gap) |
| B3.3 ✅ | **VACUOUSLY COMPLETE 2026-09-18** — B3.2 already validated the projection == engine actual to the DOLLAR ($0 gap, seed 42), i.e. there is no projection error to fold back and iterate away. The staircase market model IS the engine's per-unit crediting. (Any future world with a gap re-opens this; today the exit criterion "gap shrinks" is met at gap=0.) | B3.2 | M | ✅ $0 gap, nothing to iterate |

### B4 — Gate & ship (needs A + C)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| B4.0 ✅ | **DONE 2026-09-18** — `kaggriculture.measure.rebaseline`, faithful serve, both seats, 6 seeds. **RANKING: ref_v46=0.917 > ref_k0006=0.792 >> v57=0.458 > v56y=0.333 >> v58=0.000.** All paired vs v46 significant (v56y & v58 both 0-12, p=.0005). CONFIRMS operator: 2500 tapes are the bar, v56y is a FLOOR not a ceiling, v58 artifact is stale-tape-weak (→ G7.5 rebuild). Slot-1/2 must clear ~0.9 vs v46. Full JSON `.local/rebaseline_2026-09-18.json`. | A2 | M | ✅ ranked strong references |
| B4.1 | Paired-holdout gate vs the **strong references** (the two ~2500 public tapes, contested) on the faithful panel (`win_metric.paired_test`) | B3, B4.0, C2 | M | significant-winner verdict |
| B4.2 ✅ | **DONE 2026-09-18 — PIVOTAL NEGATIVE.** Wrapped the OR backbone in the bandit shell (2 variants: OR-sells-stand w/ intrinsic sweep killed via `tables.sweep=1e6`, and full-rails). **Both score 0.000 / 0-0-8 vs EVERY opponent & band** (incl. <2100); paired vs v46 p≈0.0078. Variants within 1% → wrapping/sell-timing is NOT the gap. Wrap FIDELITY confirmed (vs PASS $51.7k = the validated economy). **THE GAP IS PRODUCTION VOLUME: OR banks ~$49k vs PASS while ladder opponents bank $140-172k (3-4×).** Artifacts `.local/candidates/or_wrapped_{A,C}/`. | B4.1 | M | ✅ built + gated (0.000, reframes Slot-1) |
| B4.3 | Official-engine confirmation before any submit | B4.2 | S | non-regression vs strong refs on official |

**OR exit:** a base_tape that validates ≈ predicted AND beats the **strong references**
(the two ~2500 public tapes, contested) on the faithful gate + official confirm. Nothing ships
on predicted cash or on static/isolated-bank numbers (§0.6).
**Slot 1 = A2-gated OR economy + B2 PC-TAPF routing + B2.5 plan-repair shell.**

---

## Workstream E — Slot 2 "Adaptive Predator" = BC-warmup → MACRO self-play RL

The topper's method (Michal1337/pkmn-kaggle) AND the field's confirmed medal recipe
(§0.7: SNORLAX BC→RL → silver, ~300k games). **RL is MACRO / daily-plan level (~30
decisions) over the SHARED deterministic micro-executor (tape + PC-TAPF + plan-repair),
NOT per-turn** (per-turn RL plateaus dumb at 80k — KKY). Nothing here is dropped; the
smaller variants (dispatcher, timing) are ADDITIONAL, cheaper options, not substitutes.
Everything gates against the strong references (the two ~2500 public tapes) on the faithful harness.

### E0 — Corpus (BC data + opponent pool)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E0.1 ✅ | **DONE + AUGMENTED 2026-09-18** — top-100 subsample (576k micro / 24k macro) **+ destbreso `--source destbreso` (rating≥2300): 416,280 macro + 9,990,720 micro rows** (13,876 tapes, rating med 2594/max 3139, incl. high-rated LOSSES for return-conditioned BC). **Total: 440,280 macro + 10,566,720 micro** for tomorrow's GPU. Concat-superset verified. ▶ full top-100 `--limit 0` re-run pending (needs a streaming ParquetWriter + a quiet box — a concurrent-heavy OOM truncated it once). | D0.2/D1.2 | M | ✅ training corpus (high-rated) |
| E0.2 ✅ | **COMPLETE TO DATA-PHYSICS LIMIT 2026-09-18** — **MACRO (the RL/BC train level) is EXACT for every source** (derived from actions), split by episode. Top-100 replays also give full MICRO obs (both seats reconstructable). destbreso MICRO obs is fields-NaN and this is a HARD PHYSICS LIMIT, not a TODO: it stores one seat's actions and the world is realized by BOTH seats, so the per-turn obs is unreconstructable in principle. RL trains on macro → unaffected; micro-option BC is top-100-only by design. | E0.1 | M | ✅ macro exact; micro obs = physics limit |
| E0.3 ✅ | **DONE + AUGMENTED 2026-09-18** — `models/opponent_pool_full.json` (814 entries): **786 real-rated destbreso_tape opponents ≥2500 (560 ≥2700, 197 ≥2900, span 2500-3139), rating-weighted** + 24 replay + 2 ref + v57 + frozen-self. A rich RL league of the actual frontier economies. | D2.3/D3.2 | S | ✅ pool manifest (frontier-rated) |

### E1 — Featurization + action space (shared with executor)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E1.1 | Byte-identical Rust obs encoder + Python reference (parity-tested) — the train==test foundation | A1 | L | encoder w/ parity test |
| E1.2 ✅ | **DONE 2026-09-18** — `train/macro_actions.py`: 26-dim `MacroAction` vector + 7 primary classes + `day_macro()` distiller; vocab fixed by game rules. Corpus class balance: PLANT_EXPAND 49.5%, HIRE 27.1%, BUY_ANIMAL 13.6%, BUY_LAND 6.3%, BUILD 2.2%, SELL 1.2%. | E0.2 | M | ✅ macro action schema |
| E1.3 ✅ | **DONE 2026-09-18** — `train/micro_options.py`: 58 canonical (verb,target) options + encode/decode + exact `market_legal_mask` (cash/shed/seed/≤10-order) + field-op prereqs. (E1.1 Rust encoder = later hook; this is its reference oracle.) | B2.6 | M | ✅ masked option encoder |

### E2 — BC warmup
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E2.1 ✅ | **DONE 2026-09-18** — `train/bc_warmup.py`, return-conditioned masked causal Transformer (~10-20M configurable), 7-way class + 26-dim action heads, on the 440k macro corpus. CPU-smoked on the real corpus → `models/rl/bc_policy.pt`. GPU run = operator. | E0.2, E1 | L | ✅ BC checkpoint code ready |
| E2.2 ✅ | **DONE 2026-09-18** — `measure/eval_harness.py` `_battery_winrate` (in macro_rl) + `replay_clean_room` establish BC-vs-pool win rates banded by rating; run post-BC via `--stages bc,gate`. | E2.1, E0.3 | M | ✅ BC-vs-pool battery |

### E3 — Self-play MACRO RL fine-tune
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E3.1 ✅ | **DONE 2026-09-18** — `train/macro_rl.py` PPO self-play (autoregressive plan sampling, clipped surrogate) + `train/macro_env.py` faithful reward (vendored engine, replayed real opponents). CPU-smoked → `models/rl/rl_policy.pt`. | E2.1, A1 | L | ✅ training loop runs |
| E3.2 ✅ | **DONE 2026-09-18** — reward = win/draw/loss terminal + optional `margin_shaping` (bank diff / 1e5), both seats measured via `our_seat`. | E3.1 | M | ✅ reward spec |
| E3.3 ✅ | **DONE 2026-09-18** — pool sampling (786 rated destbreso tapes, weighted) + **teacher-KL** to frozen BC prior + **gated promotion** vs frozen battery (saves only on improvement). | E3.1, E0.3 | M | ✅ non-forgetting training |
| E3.4 ✅ | **RL-SIDE DONE 2026-09-18** — episodic terminal-return advantage (return-minus-EMA-baseline over the 30 macro days); weed count exposed as a plan-quality signal; animals/fert are first-class in the macro action space so reward routes there via banks. The ONLY remaining piece — explicit off-path trajectory-STITCHING for weed/shop shocks — is an EXECUTOR capability (dependency-directed plan-repair, B2.5), NOT an RL-loop gap; it rides on B2.5 when built. | E3.1 | M | ✅ RL advantage/shaping done; stitching = B2.5 |
| E3.5 ✅ | **THROUGHPUT WIRED 2026-09-18 (B4)** — `macro_env.BatchRoller`: compile each plan ONCE on the seed-independent serve engine, replay vs many opponents via Rust `kagg batch`. **Measured 7.7 ms/game @ games_per_plan=64 → 300k games in ~38 min single-proc** (batch uses all cores), bit-exact to closed-loop (validated all seeds). Opponents prewarmed in ONE parquet pass. `macro_rl --batch`, `slot2_pipeline --games-per-plan`. The 300k RUN itself = operator GPU job, now affordable. | E3.1 | L | ✅ ~10ms/game path; run = GPU |

### E4 — Cheaper learned variants (additional, not substitutes)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E4.1 ✅ | **DONE 2026-09-18 (HONEST NEGATIVE)** — `train/dispatcher.py`: world day-6 shop-signature does NOT predict winning economy (5-fold CV lift **−0.039**). Consistent with repo's documented "dispatch gains nothing at any pool size." **Do NOT ship a runtime shop→economy router.** Slot-2 value = learned economy/execution, not selection. | E0.2 | M | ✅ trained (negative result recorded) |
| E4.2 ✅ | **DONE 2026-09-18** — winner sell-timing: WHEAT dominates volume (peak d29), MELON earliest (mean d17/peak d10 = the day-10 cap), staples dump late (d29). Feeds shell sell-gate tuning. | E0.2 | M | ✅ timing model/gates |

### E5 — Gate + package
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E5.1 ✅ | **DONE 2026-09-18** — `eval_harness.gate_packaged_policy()`: smoke + seat-swapped vs strong refs (paired both-seats) + banded replay clean-room; **SHIP iff ≥0.9 vs v46** else diversified route2. Faithful serve. | A2, C2, E2/E3 | M | ✅ ship-or-fallback verdict |
| E5.2 ✅ | **DONE 2026-09-18** — `package_policy.py`: ONNX export + onnxruntime==torch parity gate + manifest (feature recipe for the CPU/native runtime). `build_policy_agent()` = compile-then-execute runtime. Worst-turn latency measured by `smoke_test` (mean ms/turn). Native-Rust <1s/turn = F1.1. | E5.1 | M | ✅ submittable-policy artefact |
| E5.3 | Fallback if Slot 2 ≤ Slot 1: diversified OR `route2` (different opening family) | E5.1 | M | diversified Slot 2 |

### E6 — Algorithm choices to EVALUATE (all in scope; pick by measured result, not assumption)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| E6.1 | RL algorithm: **PPO** (on-policy, simpler, SNORLAX/topper) vs **IMPALA** (V-trace off-policy, high async-actor throughput) — run both on a small budget, compare sample-efficiency & wall-clock | E3.1 | L | measured PPO-vs-IMPALA verdict |
| E6.2 | Policy model: **masked pointer-transformer** (topper) vs **Decision-Transformer** (return-conditioned, offline/trajectory-stitching, research doc 1) — compare BC fit + RL fine-tune | E2.1 | L | measured pointer-vs-DT verdict |
| E6.3 | Decision granularity: **macro/daily** (field-validated) vs **per-turn** (field says plateaus) — keep macro primary, log per-turn as a data point | E3.1 | M | granularity verdict |

**Slot 2 exit:** a BC-warmed macro-RL policy (or dispatcher) that beats Slot 1 on the
faithful gate → ships; else diversified OR `route2`.

---

## Workstream F — Inference runtime, advanced optimizations, analysis (all in scope)

### F1 — Inference runtime (submission + RL rollouts)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| F1.1 | Native Rust policy inference (ONNX via `candle`/`tract`/`ort`) — see A3.4; the submission path (no Python, CPU <1s/turn) and fast in-process RL inference | E2/E3 | L | Rust-native inference |
| F1.2 | Shared-memory / PyO3 opponent execution options (see A3.2/A3.3) | A3 | L | throughput options available |

### F2 — Advanced engine/RL optimizations (research doc 2; later-phase, not blockers)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| F2.1 | ArcWake async macro/micro loop: suspend micro-execution for the 3-day macro inference without desync (Rust `ArcWake`) | B2.5, E3 | L | clean async macro/micro |
| F2.2 | Kolmogorov/MDL **motif compression**: detect recurring labor motifs, compress to macro-actions → faster Phase-3 plan-repair | B2.5 | L | repair speedup |
| F2.3 | **Generalized-planning** tape: conditional-goto / parameterized branches so shop-unlocks trigger a branch instead of full plan-repair | B2.5 | L | tape adapts w/o full repair |
| F2.4 | Rust memory-arena rollback for instant Phase-3 recovery | B2.5 | M | O(1) state rollback |

### F3 — Analysis (net-P&L / cost-leak; research 741414)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| F3.1 ✅ | **DONE 2026-09-18** (first cut, sell-side) — `kaggriculture.measure.pnl_replay` over 789 decided top-100 games. Winner bank 102.8k vs loser 85.6k. **Winners realize higher $/u on 7/9 products & less volume on 6/9** (scarcity capture); losers over-produce WHEAT +238u/FERT +182u/EGG +30u. OR targets: max STRAWBERRY/WOOL(→YARN)/MELON/MILK, min wheat/fert/egg over-prod. `.local/pnl_replay_2026-09-18.json`. ▶ Refine: exact realized $/u via market re-sim + buy-side costs. | D0.2 | M | ✅ per-run P&L breakdown |
| F3.2 | (Optional) use the community Ops-Lab dashboard for eyeballing candidate replays | — | S | visual P&L review |
| F3.3 ✅ | **DONE 2026-09-18** — `kaggriculture.measure.animals_audit` (591 games). Winners invest MORE in animals (buy +1.19, pasture +1.25, feed/care +30/+27, fert collect +20) and CAPTURE WOOL (+$5,134/g); but MILK/EGG rev ≈equal-or-higher for LOSERS (they over-produce cheap-glutted products). Verdict: OR model SHOULD build animals but route selectively to **WOOL scarcity (YARN)**, not raw volume — matches the PWL. `.local/animals_audit_2026-09-18.json`. | F3.1 | M | ✅ animals/fertilizer profitability check |

---

## Workstream G — Harness separation & full configurability (bandit ⊥ trackp, hot-swappable)

**Target (operator):** two COMPLETELY SEPARATE, highly configurable Rust harnesses
(bandit, trackp); EVERYTHING hot-swappable via config — guardrails, policies, base
tape/routes, adaptive plan; both **musl-compiled in Docker**, thin `main.py` driver;
native inference (no PyO3). **Reference pattern = the shipped v46 `Chassis`** (config-
driven layers, safe raw-tape fallback). Grounded in the 2026-09-18 audit
([[harness-audit-2026-09-18]]). Findings are recorded in memory + CLAUDE.md (done)
and this plan (G6 keeps them synced).

### G0 — Decouple bandit ↔ trackp (the 3 coupling knots the audit found)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G0.1 ✅ | **DONE 2026-09-18** — new track-neutral `rustengine/src/core.rs` (`Op`/`Row`(Clone)/`row_json`); `bandit.rs`+`mbandit.rs` now `use crate::core` not `crate::policy`. Zero bandit↔trackp Rust coupling. Gates green. | — | L | ✅ no bandit↔trackp Rust dep |
| G0.2 ✅ | **3-WAY SPLIT DONE 2026-09-18** — added the lean `kagg-engine` bin (`src/bin/kagg_engine.rs`, batch+serve only, 407KB vs the 560-568KB seat bins = carries NO seat strategy) alongside `kagg-bandit`/`kagg-trackp`; `service`+`loadstate` re-exposed via `lib.rs` (deps engine+state only, main.rs private mods untouched). All 4 bins build isolated; **parity re-verified after the lib change — banks 88520/70492 bit-identical, G6.1/G6.2 PASS**. Monolith `kagg` RETAINED (additive; all ~25 consumers unchanged). Repointing consumers off the monolith = a no-ladder-value deploy step, parked (the monolith is correct, not stale). | G0.1 | L | ✅ 3 lean per-track binaries |
| G0.3 ✅ | **DONE 2026-09-18** — bandit build tooling moved `trackp/harness/` → `src/kaggriculture/bandit/build/`; bare imports → absolute; consumers updated (verified: no stale old-path imports; new path imports clean w/ C2 `_cross_build_musl` intact). Branch DATA left under models/trackp (gitignored blobs; code separation complete). | — | M | ✅ bandit code in bandit namespace |
| G0.4 ✅ | **DONE 2026-09-18** — removed the vestigial `from day3_evolve import our_route` from build_rust_bandit.py + build_bandit_harness.py (only comments/help referenced it). Both compile. | — | S | ✅ no unused cross-imports |

### G1 — Bandit: full config-driven hot-swap
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G1.1 ✅ | **DONE 2026-09-18** — `bandit.rs` given an OPTIONAL config surface (`BanditCfg`/`load_bandit_cfg` via `KAGG_BANDIT_CONFIG`: thresholds/fork/tapes); Default == shipped consts → the 6 measurement consumers stay BIT-IDENTICAL (kept working, not deleted). Live-proof: final_dump 650 vs 700 changes endgame. | G0.2 | L | ✅ config-ified (non-destructive) |
| G1.2 ✅ | **DONE 2026-09-18** — optional `config.tables` (parsed by `load_tables`→`Config.tables`): sell_priority, allp, dispatch_window K, sweep/esc_sweep, endgame_qty, daily_demand shop table, r37 price curves. Every call site defaults to today's constant → v581 (no tables key) BIT-IDENTICAL; G6.2 still PASS. | G1.1 | L | ✅ no strategy constants baked in Rust |
| G1.3 ✅ | **DONE 2026-09-18** (via G7.M9) — `load_config` reads+honors array knobs (`anti_dump.premium`, `scarcity_i0`). | G1.1 | M | ✅ config edits apply on Rust |
| G1.4 ✅ | **DONE 2026-09-18** (isolated build, git-safe via `.local/backup/src_snapshot`) — ordered rail REGISTRY in mbandit.rs: `Config.order: Vec<String>`, uniform `apply_rail(name, &mut TurnState, obs, step)`, `for name in cfg.order.clone() { apply_rail(...) }`. Rails now reorder/toggle/add/remove via config; v581 (no order key) BIT-IDENTICAL (banks 88520/70492), G6.2 PASS. See [[bandit-rail-registry-2026-09-18]]. | G1.2 | L | ✅ rails hot-swappable incl. order |
| G1.5 ✅ | **DONE 2026-09-18** — base tape + branches + rail set/order all config-driven; router strategy = the ordered rail registry (G1.4). | G1.4 | M | ✅ pluggable router via config |

### G2 — Trackp: externalize genome/policy for hot-swap
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G2.1 ✅ | **DONE 2026-09-18** — `src/kaggriculture/trackp/genome.json`; BOTH sides load at runtime (build_econ_agent.py::skeleton_genome() loads it w/ inline fallback; policy.rs `load_genome()`/`build_genome()` via Box::leak keeping the `&'static Genome` signature). **Positive control proven** (TRACKP_GENOME tweak changed play-selftest 5→2 HIREs = genuinely loading). T1 quadrant assignment intact (identity gate 0-diff). | G0.2 | L | ✅ genome swappable without recompile |
| G2.2 ✅ | **DONE 2026-09-18** — planner targets now come from genome.json (loaded both sides). | G2.1 | M | ✅ plan hot-swappable |
| G2.3 ⏸ | Pluggable skeleton-vs-ONNX policy — needs the GPU-trained policy (G5). Deferred to Slot-2 training. | G2.1, G5 | M | ⏸ needs trained policy |

### G3 — Unified per-track config schema (hot-swap EVERYTHING)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G3.1 | Define ONE config schema per track covering: base_tape/routes, router/policy(name+params), guardrails(ordered; toggle+knobs+arrays), adaptive_plan/opponent_plan, checkpoints — two instances (bandit, trackp), no cross-refs | G1, G2 | M | documented schema |
| G3.2 | Config VALIDATION: error on unknown/unconsumed keys (kill silent dead config, e.g. old anti_dump arrays); versioned configs fail loudly (e.g. bandit_config.json w/o base_tape) | G3.1 | M | no dead knobs possible |

### G4 — Separate musl build + thin driver per track
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G4.1 | Separate musl build scripts: `build_bandit_linux.ps1` / `build_trackp_linux.ps1`, each cross-building its own `--bin` for `x86_64-unknown-linux-musl` in Docker | G0.2 | M | per-track musl binaries |
| G4.2 | **Fix the Windows-binary bug:** build_rust_bandit.py packages the LOCAL `kagg.exe` (:288) — package the MUSL `kagg-bandit` for real Linux submission | G4.1 | S | Linux-correct bandit tar |
| G4.3 | Thin `main.py` driver per track (load binary+config+tapes, spawn, stdio, watchdog); keep a minimal safe fallback; submission = per-track members only | G4.1 | M | thin drivers, correct tars |

### G5 — Native inference (the PyO3 alternate)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G5.1 | For any learned policy (trackp NN, Slot-2 BC/RL): export → ONNX, run in Rust via `candle`/`tract`/`ort` inside the track binary — no PyO3, no Python at runtime | E2/E3 | L | Rust-native policy inference |

### G6 — Hygiene, tests, doc sync (operator: findings → memory/docs/CLAUDE.md)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G6.1 ✅ | **DONE 2026-09-18** — `tests/test_bandit_parity.py`: stages the artefact around an isolated binary, plays 2 seeds × both seats on `kagg serve`, asserts Rust bridge answered EVERY turn (fallback==0, sells>0 = economy ran) AND `_py_fallback` is legal-PASS. Non-skippable: FAILs with no binary, REJECTS the stale `rustengine/kagg.exe`. Verified PASS. Added to `run_all.py` slow set. | G1 | M | ✅ byte-parity guards |
| G6.2 ✅ | **DONE 2026-09-18** — config round-trip test parses `mbandit.rs` `g()`/`knob()`/`list()` sites, asserts every v581 knob consumed (incl. new arrays). Positive-control verified (injected dead knob → FAIL). | G3.2 | M | ✅ dead-config test green |
| G6.3 ✅ | **DONE 2026-09-18** — CLAUDE.md Track-P note updated to the REAL post-refactor separation (core.rs decoupling, bandit/build move, genome.json, config.tables, additive kagg-bandit/kagg-trackp bins, what's still runtime-shared). Memory + tasklist current throughout. | ongoing | S | ✅ docs match reality |

### G7 — Bug audit & remediation (codebase-wide correctness sweep)
| ID | Task | Dep | Eff | Exit criterion |
|----|------|-----|-----|----------------|
| G7.1 | Bug audit — BANDIT ship path (mbandit.rs, bandit.rs, build_rust_bandit.py, build_bandit_harness.py): config-not-applied, tape/dispatch off-by-one, splice desync, market ≤10 cap/priority, rounding, DRIVER `__file__`/watchdog/fallback, Rust↔Python divergence, packaging | — | M | ranked bug list (RUNNING 2026-09-18) |
| G7.2 | Bug audit — TRACKP + OBS/serve path (policy.rs, build_econ_agent.py, compiled/bridge.py, serve_match.py, service.rs, obsstate.rs): obs-fidelity vs documented schema (the serve mis-rank root cause), planner illegal-action edge cases, fallback parity, from_obs belief misuse | — | M | ranked bug list (RUNNING) |
| G7.3 | Consolidate → `docs/history/bug-findings-2026-09-18.md` (bandit DONE; trackp/obs pending); rank CRITICAL→LOW; record in memory/CLAUDE.md per operator | G7.1, G7.2 | M | bug-list doc + fix tasks |
| **G7.C2** ✅ | **DONE + VERIFIED 2026-09-18** — `build_rust_bandit.py` `_cross_build_musl()` Docker cross-build to `x86_64-unknown-linux-musl`, packages `<out>/kagg`, asserts `\x7fELF`. VERIFIED: `file` reports **"ELF 64-bit LSB pie executable, x86-64, static-pie linked, stripped"** (magic `7f454c46`); full tar at `.local/candidates/c2_verify.tar.gz`. Bonus: T1's `policy.rs` also compiles clean in the musl target. NOTE: `build_rust_bandit.py` must be run as a script (bare `import build_bandit_harness` needs its own dir on path) — latent absolute-import bug (G6/G0). | G4.2 | M | ✅ Kaggle actually runs the Rust binary |
| **G7.C1** ✅ | **DONE 2026-09-18 (Approach B)** — fallback DEMOTED to a loud legal-PASS net (`_py_fallback` in the DRIVER; stops claiming "action-faithful"); `main()` no longer appends the embedded economy (that WAS C1). Economy lives ONLY in `kagg mbandit` = ship==measure by construction. Dissolves C1/H3-H7/M10/M11. Isolated build ✅; parity gate PASS. | G1.1 | L | ✅ honest fallback; one true economy |
| **G7.H8** ✅ | **DONE 2026-09-18** — `build_rust_bandit.py` DRIVER: `_kill`→`_retire(soft)` with `MAX_MISSES=3` (re-spawn next turn, desync-safe fresh stream), `FIRST_BUDGET` held until the bridge first ANSWERS (not just step 0), and module-load warm-up so the branches.json load is off the per-turn clock. py_compile + AST-extract compile verified. | — | M | ✅ bridge survives a slow/cold turn |
| G7.H3-H7 ✅ | **DONE 2026-09-18 (via C1 Approach B)** — the fallback no longer runs an economy, so there are no rails to diverge (H3/H4/H5/H6/H7 dissolved). The only economy is `kagg mbandit`. | G1.4 | L | ✅ no divergent second economy |
| G7.M9-M11 | **M9 ✅ / M10-M11 dissolved 2026-09-18** — M9 FIXED in `mbandit.rs`: `load_config` now parses+honors array knobs (`anti_dump.premium`, `scarcity_i0`); config no longer lies on the ship path (no-op for v581, anti_dump off). M10/M11 dissolved (no Python economy to diverge from). | G1.3 | M | ✅ config truthful; rounding moot |
| **G7.T1** ✅ | **DONE + DEPLOYED + VERIFIED 2026-09-18** — quadrant assignment in `policy.rs`; `rustengine/kagg.exe` swapped to the fixed binary; identity gate PASSES on the LIVE binary (0 diffs, 0 fallback), engine bit-exact preserved. | G2.1 | L | ✅ Rust≡fallback assignment, gate runs |
| G7.T2 ✅ | **DONE (guarded) 2026-09-18** — the two Python-only blocks (`drop_daily`, `hire_slack`) are NO-OP for the skeleton default; a `assert_rust_honored(genome)` guard now runs in `build_econ_agent.render()` and REFUSES to render loudly if a genome sets either to non-default (would diverge from policy.rs). Tested: skeleton passes, `drop_daily=1`/`hire_slack=3` raise. Silent Rust↔fallback divergence is now impossible; porting the knob lifts the guard. | G2.1 | M | ✅ genome divergence asserted |
| **G7.S5** ✅ | **DONE 2026-09-18** — serve stops at step 719 (`run_match` loops while `step<episodeSteps-1`). Verified: reactive A/B diverged 2/3 seeds pre-fix, 3/3 exact MATCH post-fix. | A0 | S | ✅ serve turn-count matches official |
| G7.S3 ✅ | **DONE 2026-09-18** — arity-aware `_call` passes resolved `configuration` to 2-arg agents (truncated to `co_argcount`, mirrors vendor `Agent.act`). | A1 | S | ✅ config-reading agents match official |
| G7.S4 ✅ | **DONE 2026-09-18** — `obs_for` `_structify`s a fresh per-seat deep copy (mirrors vendor `structify`), no cross-seat aliasing. | A1 | S | ✅ isolated per-seat obs |
| G7.IDGATE 🔄 | **TRACKP DONE + DEPLOYED 2026-09-18** — `test_compiled_agent.py` no longer skips (no binary → FAIL); PASSES on the live deployed binary. BANDIT parity gate (G6.1) = in progress (bandit-unify subagent). | G7.C1, G7.T1 | M | 🔄 trackp enforced+deployed; bandit in progress |
| G7.5 ✅ | **BUILT + GATED 2026-09-18** — rebuilt the bandit: `build_rust_bandit --config bandit_config_v581.json --base-tape rustengine/tapes/bandit_v57_egg.tape --build --tar` → static musl ELF (1.38MB, verified `\x7fELF`) + tarball `.local/candidates/bandit_v581_0918.tar.gz`; `rustengine/kagg.exe` UNTOUCHED (isolated build). **Parity gate PASS** (`test_bandit_parity.py`, isolated win binary): bridge 719/fallback 0, sells 373/456, banks 88520/70492 bit-identical, config round-trip PASS. **Faithful competitive gate vs v46: bandit 61,578 vs v46 65,498 → LOSS (~0.0), does NOT clear the ~0.9 bar** (v57-family, consistent with B4.0). VERDICT: valid diversified/FALLBACK seat, ready as an artifact — **NOT a primary submission** (the ≥0.9 competitive seat = the Slot-2 predator, GPU-pending). No submission from here. | G7.C2, G7.C1, G1.3 | M | ✅ rebuilt + fully gated (fallback-quality) |
| G7.6 | Each CRITICAL/HIGH fix gets a regression test (esp. a Rust≡fallback parity test + an ELF-only packaging assert) | G7.C1 | M | criticals guarded |

---

## Sequencing (everything in scope; deadline Sept 23 entry / Sept 30 final)

Critical path first (A+C+Slot 1), Slot 2 (BC→macro-RL) in parallel, F advanced items
ride on top. "In scope" ≠ "urgent" — this order protects a shippable pair even if the
advanced tracks slip.

| Phase | Critical path (fidelity + Slot 1) | Parallel (data + Slot 2 + F) |
|-------|-----------------------------------|------------------------------|
| 1 | A0 diff → divergence fields | D0 supervisor draining; D2.2-2.4 vet + add 2500 tapes; E0.1 DuckDB distill |
| 2 | A1 obs-contract fix | D3 replay opponents; B0 spec+solver; E0.2-0.3 corpus+pool; E1.1 obs encoder |
| 3 | A2 reactive parity gate; **B4.0 RE-BASELINE** {2500 tapes, v58, v56y} | C0/C1 panel→crown; B1.1-1.2; E1.2-1.3 action space+mask; E2.1 BC warmup |
| 4 | C2/C3 gate recalibrate + ladder-calibrate | B1.3-1.4; B2.0 MVP router; E2.2 BC eval; E4 dispatcher/timing |
| 5 | Slot 1: B2.1-2.5 PC-TAPF + plan-repair; B3.1-3.2 validate | E3.1-3.4 self-play macro-RL; E6.1-6.2 algo eval; F3.1 P&L |
| 6 | Slot 1: B3.3 iterate + B4.1 gate vs **2500 tapes** | E3.5 RL scale; E5.1 Slot 2 gate; F1.1 native inference; D1 top-200 |
| 7 | Ship pair: Slot 1 (B4.2-4.3) + Slot 2 (E5.2 winner or E5.3 route2) | F2 advanced opt / E6 remaining as they land |

**Stop rules (top-10, without-mistake guardrails):**
- **Fidelity gates everything.** Nothing ships (OR tape, plan-repair, RL policy,
  dispatcher) without A2 (reactive parity) + a paired gate **on the faithful engine
  under contested play**. The docs'/field's confidence is not the bar; the gate is.
- **Gate against the ~2500 public tapes** (the real bar), NOT v56y (2072) or v58
  (463). Re-baseline (B4.0) before trusting any prior ranking (§0.6).
- **No static/isolated-bank verdicts.** Prior negatives (BC-dead, offline≠ladder)
  are re-tested on the faithful harness, not carried forward.
- **Slot 2 = BC→macro-RL** (field-validated; SNORLAX→silver). Per-turn RL is logged
  as a data point only (field says it plateaus). If Slot 2 doesn't beat Slot 1 by the
  deadline, it ships a diversified OR `route2` — a fallback, not the plan.
- **Submission mechanics.** Only latest-2 stay active (currently the two ~2500 tapes);
  a new submit evicts the older of the two — confirm the pair keeps two strong agents
  before each submit.
- **PC-TAPF/plan-repair/advanced-F are upgrades, not blockers.** If they aren't ready,
  ship Slot 1 on the B2.0 greedy router + existing guardrails.
