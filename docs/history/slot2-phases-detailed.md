# Slot-2 pipeline — every phase in detail, with fix-points

Companion to `slot2-forward-tasklist.md`. This is the **data flow through each
phase**, sub-step by sub-step, with `[FIX: <id>]` marking exactly where each
open item slots in. `▶` not started · `🔄` partial · `✅` done. Module names are
real. No priority implied.

Legend for a sub-step: `<module>  IN → OUT`.

---

## PHASE 1 — DATA  (build the corpus + opponents)

```
1.1 download_supervisor.sh        ladder/top-200 ids → new .gz episodes      ⛔ CDN 429
1.2 consolidate_gz.py             .gz → top100_replays_parts/*.parquet        ✅
1.3 data.bc_corpus                replays → macro.parquet (440k) + micro      ✅
        + --source destbreso      + macro_destbreso.parquet                   ✅
1.4 data...opponent_pool          replays/destbreso → opponent_pool_full.json ✅ (786 ≥2500)
1.5 data.ourgames                 our LIVE ladder games → corpus             ⛔ CDN 429 (D4)
```
**Fixes that live in DATA:**
- `[FIX: O6]` **Corpus delta refresh** — today `bc_corpus` re-processes the WHOLE
  parquet set; make it fold only NEW episode ids (incremental), so a daily run
  isn't a full re-scan. `IN`: new part files `OUT`: appended macro/micro rows.
- `[FIX: T3-prep]` **World-distribution extraction** — scan the corpus for the
  day-6 `world_sig` frequency (which shop-worlds the ladder actually visits).
  `OUT`: `data/worlds/world_freq.json`. This is what the TRAIN world generator
  samples from (so we don't over-train on worlds real games never see).
- `[FIX: D4]` **ourgames unblock** — drains on CDN reset; feeds real self-eval data.

**DATA output that TRAIN consumes:** `macro.parquet` (BC targets),
`opponent_pool_full.json` (RL opponents), `world_freq.json` (world generator).

---

## PHASE 2 — TRAIN  (the meat; where most fixes live)

### 2A. BC warmup  (`bc_warmup`, GPU)
```
2A.1 _load_sequences     macro.parquet → per-(episode,seat) day sequences   ✅ (dedup by day)
2A.2 build_model+train   sequences → models/rl/bc_policy.pt                  ✅ (return-conditioned)
```
BC trains on the **complete** corpus each run (small). `[FIX: O6]` makes the
corpus incremental upstream; BC itself stays full-fit.

### 2B. RL self-play loop  (`macro_rl`, GPU net + Rust rollouts) — per iteration:
```
2B.1 sample_plan (autoregressive)   policy → N plans (30-day MacroActions)
        [FIX: P1] cash-aware DECODE so day-0 spend respects $3000 (no bank-0)
        [FIX: P5] condition on rtg=1 (win) — verify it pulls higher-banking plans
        [FIX: T2] feed realized day-6 world_sig → world-ADAPTIVE plans, re-plan at D6

2B.2 COMPILE plan → tape            BatchRoller.compile_plan_tape
        current: closed-loop kagg serve (261 ms/plan)                        ✅
        [FIX: X4 / S5] OBS-FREE executor: dead-reckon movers+tiles+shed in
        Python → tape with ZERO serve calls (compile → microseconds). This is
        the "completely obs-free mechanism" — it removes the last serve
        dependency so EVERYTHING is pure batch.

2B.3 sample opponents               pool.sample → (entry, seed)
        [FIX: T1] draw from the GROWING self-play LEAGUE archive (past promoted
        selves), not just the current policy / static pool
        [FIX: T3] WORLD GENERATOR: choose the rollout SEED weighted by
        world_freq.json (ladder world distribution) instead of rng.randint
        [FIX: T5] curriculum: weak→strong opponents as the policy improves

2B.4 batch-play tapes vs opponents  kagg batch → banks → reward (7.7 ms/game) ✅
        executor realises the economy on the engine:
        [FIX: X2] COW/GOOSE husbandry (today sheep only → animal plans starve)
        [FIX: X3] BUY_LAND quadrant expansion (NE/SW/SE) re-cluster + plant
        [FIX: X5] FERTILIZER economy (winners invest — F3.3)

2B.5 PPO update                     clipped surrogate + advantage             ✅
        [FIX: A5] learned CRITIC (replace EMA baseline → lower-variance adv)
        [FIX: A2] IMPALA (V-trace) eval as an alternative to PPO
        [FIX: A3] Decision-Transformer eval as an alternative policy class

2B.6 gated promotion                battery win↑ → models/rl/rl_policy.pt     ✅
        [FIX: T1] on promotion, SNAPSHOT to models/rl/league/ (archive grows)
```
**TRAIN is where these live:** P1, P5, T2 (plan/world inputs) · X4 (obs-free
compile) · T1, T3, T5 (opponents + world generator + curriculum) · X2, X3, X5
(executor completeness) · A5, A2, A3 (algorithm).

**TRAIN output:** `rl_policy.pt` (+ `rl_policy_best.pt`, `loop_state.json`, the
growing `league/`).

---

## PHASE 3 — TEST / GATE  (is the policy shippable?)

```
3.1 package_policy         rl_policy.pt → policy.onnx + manifest (ort==torch) ✅
3.2 smoke_test             agent runs 720 turns, 0 exceptions, both seats     ✅
3.3 seat_swapped           vs the two ~2500 refs (v46, k0006), both seats     ✅ (bar ≥0.9)
3.4 replay_clean_room      vs banded destbreso opponents (rating bands)       ✅
3.5 crown_gate_banded      paired-McNemar per band + ladder-weighted          ✅
3.6 latency                worst-turn ms
        [FIX: G4] assert worst-turn < 1 s (both seats), not just mean
3.7 world-coverage gate
        [FIX: G3] weight gate cells by world_freq.json (real world mix)
3.8 official confirm
        [FIX: G5] replay the candidate on the OFFICIAL engine before any submit
```
**Fixes in TEST:** G3 (world-weighted), G4 (worst-turn latency), G5 (official
confirm). **Output:** ship/HOLD verdict + the gate scores.

---

## PHASE 4 — BUILD / SHIP  (per seat; only on a passing policy)

### 4A. BANDIT seat  (compiled tape + rails)
```
4A.1 slot2_loop.compile_base_tape  policy → base.tape (strip SEED)            ✅
4A.2 regenerate branches
        [FIX: B4] build_multi_ckpt: evolve branches FOR THE LEARNED base tape
        at D6/D12/D15/D21/D27 (current branches were evolved for the egg tape)
4A.3 build_rust_bandit             base+config+branches → musl tarball        ✅
        dispatch D6/D12/D15/D21/D27 + endgame + 6 rails                       ✅ (confirmed)
4A.4 gates                         parity (ship==measure) + vs-v46            ✅
4A.5 route2 variant
        [FIX: B7] a SECOND bandit with a DIFFERENT day-3 opening family
        (second-slot rule: the pair must differ)
```

### 4B. TRACK-P seat  (live NN inference)
```
4B.1 package_policy                policy.onnx + manifest                     ✅
4B.2 native inference
        [FIX: R2] rustengine/src/infer.rs (candle/tract/ort): load onnx, build
        features per manifest, run the NN each macro decision (Rust==torch)
4B.3 kagg-trackp bundle
        [FIX: R3] musl kagg-trackp + onnx + main.py → tarball
4B.4 gates
        [FIX: R4] worst-turn < 1 s   [FIX: R5] vs-v46   [FIX: R6] skeleton|NN switch
```
**R2 is the only structural blocker for the trackp seat.** Until it lands, the
pair's Seat 2 = the B7 route2 bandit (or the current live agent).

---

## PHASE 5 — PAIR / REPORT  (`daily_slot2`)

```
5.1 gate both seats banded                                                    ✅
5.2 pair selection
        [FIX: O4] enforce TWO DISTINCT agents (second-slot rule) + honour
        second_slot_force / release_hold overrides
5.3 release_report.md + SUBMIT_INSTRUCTIONS.md   (NEVER submits)              ✅
```
**Output:** the two tarballs + a report telling the operator exactly what to
submit. The operator submits.

---

## PHASE 6 — LOOP / SCHEDULE  (make it daily + self-improving)

```
6.1 resume            loop_state.json + rl_policy_best.pt → continue          ✅
6.2 league growth
        [FIX: O3] each daily run grows models/rl/league/ (self-improve w/o new data)
6.3 schedule
        [FIX: O5] Windows Task Scheduler: daily_slot2 after the data fetch,
        never submit
6.4 submit            operator, per SUBMIT_INSTRUCTIONS                        manual
```

---

## The two fixes you named, pinpointed
- **World generator in TRAIN** = `[FIX: T3]` at **2B.3** (seed selection weighted
  by `world_freq.json` from `[FIX: T3-prep]` in DATA) + `[FIX: T2]` at **2B.1**
  (feed `world_sig` into sampling so plans ADAPT per world). Today: uniform random
  seeds, world-blind policy.
- **Completely obs-free mechanism** = `[FIX: X4 / S5]` at **2B.2** — a
  dead-reckoning executor that compiles a plan to a tape with **no engine call**,
  removing the last `kagg serve` dependency so the whole loop is pure `kagg batch`.

## Where fixes cluster (fact, not priority)
- **TRAIN** carries the most: P1, P5, T1, T2, T3, T5, X2, X3, X4, X5, A2, A3, A5.
- **BUILD** carries the seat-completion: B4, B7 (bandit), R2, R3, R4, R5, R6 (trackp).
- **TEST**: G3, G4, G5. **DATA**: O6, T3-prep, D4. **LOOP**: O3, O5, O4.
- Competitiveness (gate ≥0.9) gates on **P1 first** (else 2B produces bank-0 plans
  and everything downstream reads 0.0).
