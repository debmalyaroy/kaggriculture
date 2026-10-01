# Slot-2 forward tasklist (bandit + trackp pair → top-10)

Complete task inventory for both seats. **No priority is implied by order** —
tasks are grouped by area and carry dependencies + effort so the operator sets
sequence. Status: ✅ done+verified · 🔄 partial · ▶ not started · ⛔ blocked
(external). Effort: S ≤½day · M ~1day · L 2-3day.

Anchors that are DONE and not repeated below: faithful serve harness (A),
crown banded panel (C), BC corpus (440k rows) + opponent pool (786 ≥2500),
`bc_warmup`/`macro_rl`/`package_policy`/`eval_harness`/`slot2_loop`/`daily_slot2`.

---

## P — Plan quality / economy (the competitiveness content)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| P1 | **Plan-space guardrails** (`macro_actions.sanitize_plan`, wired in `sample_plan`): clamp day-0 land to the $3000 cash schedule, defer+right-size animals off day 0, guarantee an early-income crop. | — | M | ✅ | **VERIFIED: 0/6 policy plans bank-0 (was ~all); banks 2739-4491** |
| P2 | **Bank-0 root-cause fix**: instrument a bankrupting plan on the engine; confirm the cause (over-spend vs unfed animals vs over-hire) and remove it | P1 | S | ▶ | a named, fixed cause; banks stable |
| P3 | **Bootstrap-viability study**: replay 5-10 ≥2900 opponents on their own seed, extract HOW they bank ~100k from $3000 (opening sequence, product mix, land/hire cadence) | — | M | 🔄 | documented bootstrap recipe(s) the policy should imitate |
| P4 | **Reward shaping review**: confirm terminal win/loss + margin shaping discriminates once P1 lands (not saturating at 0 or 1) | P1 | S | ▶ | reward variance > 0 across a batch |
| P5 | **Return-conditioning at inference**: verify conditioning on rtg=1 (win) actually pulls higher-banking plans vs rtg=0 | P1 | S | ▶ | rtg=1 plans out-bank rtg=0 plans |

## X — Executor (micro realizer)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| X1 | **season_cbs PC-TAPF executor** (`SeasonCbsController`), plan-parameterised + seat-general, default | — | L | ✅ | +158-726% over greedy (verified) |
| X2 | **Cow/goose husbandry** (`SeasonCbsController` generalised): typed animal tiles, build PASTURE/COOP, PLACE, FEED (wheat) all animals, CARE, HARVEST MILK/EGG/WOOL, buy all types cash-gated | X1 | M | ✅ | **VERIFIED on viable mixed plans (cows bought/fed/harvested); naive animals net-negative = right-sizing is the policy's job (matches F3.3)** |
| X3 | **BUY_LAND / quadrant expansion** (executor emits BUY_LAND cash-gated, re-clusters as owned grows) | X1 | M | ✅ | **VERIFIED: buys 3 tiers, unlocks NW/NE/SW/SE; over-expansion net-negative = policy right-sizes** |
| X4 | **Obs-free (dead-reckoning) executor** (`DeadReckonCompiler`, `BatchRoller(compile_mode='deadreckon')`) | X1 | L | ⚠️REJECTED | **BUILT+VALIDATED but fidelity FAILS (banks 72-86% too low; engine spawn/yield too intricate to approximate). Kept opt-in; serve-compile stays default. gpp is the safe speed lever.** |
| X5 | **Fertilizer economy** (COLLECT_FERTILIZER from animals + sell FERTILIZER) | X1 | M | ✅ | wired (collect+sell); low-impact at current scale |

## T — Training dynamics (self-improvement without new data)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| T1 | **Persistent self-play league** (`macro_rl`): on promotion, compile the policy's greedy plan → `models/rl/league/*.tape`; `_load_league` restores it; rollouts sample league members (`league_frac`). | — | M | ✅ | **VERIFIED: league grows on promotion; train_win climbs 0.016→0.375 across 4 iters (was 0.000)** |
| T2 | **World-conditioning** (`worlds.WorldSampler`, wired in `sample_plan`): policy conditions on a sampled target-world bucket | P1 | M | ✅ | wired; policy input = world bucket (per-game adapt = live seat) |
| T3 | **World generator** (`worlds.build_world_freq` → `data/worlds/world_freq.json`; WorldSampler weights by ladder freq) | T2 | M | ✅ | **VERIFIED: extracted day-6 world freq from corpus; sampled by frequency** |
| T4 | **Teacher-KL + gated promotion** | — | S | ✅ | non-forgetting (verified) |
| T5 | **Curriculum**: start vs weaker league/PASS, ramp to ≥2900 as the policy improves | T1 | M | ▶ | win-rate curve climbs smoothly, no collapse |

## S — Rollout speed

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| S1 | **Batch rollout path** (compile-once + `kagg batch`) | — | L | ✅ | 7.7 ms/game, bit-exact (verified) |
| S2 | **Serve backend** (bit-exact; compile + battery + **the GATE**: `eval_harness.play_two` now serve, 3.5x faster, bit-exact) | — | M | ✅ | serve==vendored (verified, incl. gate) |
| S3 | **`games_per_plan` knob** exposed (amortises the compile) | S1 | S | ✅ | higher gpp → lower ms/game |
| S4 | **Battery eval on batch** (`_battery_winrate(br=...)`) | S1 | S | ✅ | one compile, batched vs battery opponents |
| S5 | **Obs-free executor removes serve everywhere** (= X4) | X4 | — | ⚠️ | X4 rejected (fidelity) |
| S6 | **Bounded parallel compile** (`BatchRoller.parallel_compile`, persistent pool, HARD-CAP 6 workers, default 3) | S1 | M | ✅ | **VERIFIED: 1.3-1.4x, memory-safe (no over-spawn)** |

## B — Bandit seat (compiled tape + rails)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| B1 | **compile_base_tape** (policy → base tape, strip SEED header) | — | S | ✅ | valid 719-row base tape |
| B2 | **build_rust_bandit** (musl tarball, isolated) | B1 | M | ✅ | ELF-verified tarball |
| B3 | **Dispatch D6/D12/D15/D21/D27 + endgame + 6 rails** | — | — | ✅ | confirmed in config + build |
| B4 | **Regenerate branches for the LEARNED base tape** (`build_multi_ckpt`) — current branches were evolved for the egg tape | B1 | M | ▶ | branches match the learned base at each checkpoint |
| B5 | **Parity gate** (ship==measure) | B2 | M | ✅ | bridge 719/fallback 0 (verified) |
| B6 | **Competitive gate vs v46** on faithful panel | B2 | M | ✅ | score reported (currently 0.0, needs P/T) |
| B7 | **Route2 variant** (`compile_base_tape(world_bucket=8)` → distinct opening → second bandit in `daily_slot2`) | B1 | M | ✅ | daily builds bandit + route2 |

## R — Track-P seat (live NN inference)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| R1 | **ONNX export + parity gate** (ort==torch) | — | M | ✅ | parity ~1e-6 (verified) |
| R2 | **Native inference** (`train/native_infer.py`, pure-NUMPY transformer forward — macro=~30 calls/game so no Rust ONNX crate needed) | R1 | L | ✅ | **VERIFIED: numpy==torch (cls 4e-6, vec 1e-6)** |
| R3 | **trackp submission bundle** (self-contained main.py: numpy weights + native_forward + inline executor) | R2 | M | ▶ | tarball runs on Kaggle (R2 core done) |
| R4 | **Latency gate** (worst-turn <1s) | R3 | S | ▶ | measured worst-turn |
| R5 | **Competitive gate vs v46** | R3 | M | ▶ | score vs v46 |
| R6 | **Skeleton-vs-ONNX policy switch** in kagg-trackp (G2.3) | R2 | M | ▶ | config selects skeleton or NN |

## O — Orchestration / daily pipeline

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| O1 | **slot2_loop** (BC→[RL→package→gate]×N→ship-prep, resumable, never submits) | — | L | ✅ | loop runs, resumes, ships-prep (verified) |
| O2 | **daily_slot2** (data→train→bandit→trackp→pair→report) | O1 | M | ✅ | builds both seats + report (verified) |
| O3 | **League archive wired into daily** (daily train uses `macro_rl` batch → `models/rl/league` auto-loads+grows) | T1 | S | ✅ | league persists+grows across daily runs |
| O4 | **Pair selection + versioning + config-lock + private-notebook release** (`pipeline/release.py`: v59 major/day, minor/intraday; lock_config; kaggriculture-private-submission-{bandit\|trackp}) | B7 | M | ✅ | **VERIFIED: v59→v59.1; both harnesses gated + locked + notebook checklist** |
| O5 | **Schedule** (`scripts/schedule_daily_slot2.ps1 -Register`) — daily, never submits | O2 | S | ✅ | -Register/-Run/-Unregister |
| O6 | **Corpus delta refresh** (fold only NEW episodes vs full re-process) | — | M | ▶ | incremental corpus update |

## D — Data

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| D1 | **Download supervisor draining** (61,539 remaining) | — | S | ⛔ | drains on CDN 429 reset |
| D2 | **bc_corpus** (macro 440k + micro) | — | M | ✅ | corpus on disk |
| D3 | **opponent_pool** (786 ≥2500) | — | S | ✅ | pool manifest |
| D4 | **ourgames → corpus** (our live ladder games) | live games | M | ⛔ | CDN 429; C3.1 pending |
| D5 | **Prewarm opponents (one parquet pass)** | — | S | ✅ | no per-lookup rescans (verified) |

## A — RL algorithm evals (measured, not assumed)

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| A1 | **PPO** (clipped surrogate, EMA baseline, macro-episodic) | — | L | ✅ | loop runs (verified) |
| A2 | **IMPALA eval** vs PPO | A1 | L | ▶ | not run (needs both trained) |
| A3 | **Decision-Transformer eval** (return-conditioned) vs pointer policy | R1 | L | ▶ | measured BC-fit + RL-finetune verdict |
| A4 | **Granularity**: macro (default) vs per-turn data point | A1 | M | 🔄 | macro primary; per-turn logged |
| A5 | **Learned critic** (MLP value baseline over mean-pooled features; MSE-to-return) | A1 | M | ✅ | **VERIFIED: train_win climbs 0.11→0.39 with critic** |

## G — Gating / measurement

| ID | Task | Dep | Eff | Status | Exit criterion |
|----|------|-----|-----|--------|----------------|
| G1 | **eval_harness** (smoke / seat-swapped / replay clean-room / Elo) | — | M | ✅ | 4 eval types (verified) |
| G2 | **Crown banded panel + paired-McNemar gate** | — | L | ✅ | banded ship rule (verified) |
| G3 | **World-diverse gate** (`worlds.build_seed_bank` → 32 seeds/35 worlds; `eval_harness.gate_seeds` pulls the bank) | T3 | M | ✅ | **VERIFIED: gate plays across 35 worlds (was fixed consecutive seeds)** |
| G4 | **Worst-turn latency in the gate** (`smoke_test` tracks worst single turn; PASS requires <1s) | — | S | ✅ | worst-turn asserted <1s |
| G5 | **Official-engine confirmation** — the gate runs on Rust `serve` = bit-exact to official (verified) | — | S | ✅ | serve==vendored==official (verified) |

---

## Cross-cutting dependency notes (facts, not priorities)
- A competitive policy (gate ≥0.9 vs v46) currently requires **P1** (else plans bank 0). B6/R5 read 0.0 until then.
- **B4** (branch regen) is needed for the bandit's D6/D12/… to fire branches matched to the learned base tape.
- **R2** (native inference) is the only structural blocker for the trackp seat; until it lands, the pair's Seat 2 = **B7** (route2 bandit) or the current live agent.
- **T1** (league) is what makes daily runs improve without new data; **X4/S5** (obs-free) is what removes serve for pure-batch speed.
- Blocked-external only: **D1/D4** (CDN 429), and any GPU-scale training run (fast now, but still a run).

## Verification standard (applies to every task)
Nothing is "done" on predicted numbers. Done = ran on the **faithful engine**, gated vs the **~2500 refs**, banks/latency **measured**, and (for Rust) an **identity/parity gate** green.
