# Slot-2 per-phase status + ETA + speedup research (both seats)

All times are **measured on this box** (RTX 4060 + Rust `kagg` on CPU), not guessed.
Status: ✅ done · 🔄 partial · ▶ not started · ⚠️ built-but-rejected · ⛔ external-blocked.

## Measured unit costs (the basis for every ETA)
| Op | Measured | Note |
|----|----------|------|
| BC epoch | ~6 s (10 ep = 62 s) | GPU, 4.77M params, 14.5k seqs |
| Batch game (PASS opp) | 7.7 ms | pure `kagg batch`, all cores |
| **RL game (real opp, incl. compile+battery+PPO)** | **~51 ms** | pilot: 7680 games / 390 s, single process |
| Plan compile (serve) | 261 ms/plan | amortised /gpp; the serial bottleneck |
| Gate game (2 reactive agents) | ~2.3 s | ref-agent Python per-turn dominates |
| Bandit musl build | ~35 s | isolated Docker cross-build |
| numpy policy inference | <5 ms/call | ~30 calls/game → trivial |

---

## PHASE 1 — DATA
| Task | Seat | Status | ETA |
|------|------|--------|-----|
| bc_corpus (440k macro) | both | ✅ | done |
| opponent_pool (786 ≥2500) | both | ✅ | done |
| world_freq + seed_bank (35 worlds) | both | ✅ | ~1 min build |
| O6 corpus delta (incremental) | both | ▶ | minutes when built |
| D1/D4 download drain + ourgames | both | ⛔ CDN | — |
**Phase ETA:** ~0 (corpus exists); refresh is minutes.

## PHASE 2 — TRAIN (shared trunk → one policy for both seats)
| Task | Status | ETA |
|------|--------|-----|
| BC warmup | ✅ | **~1 min** (GPU) |
| RL loop (batch, season_cbs exec, league, world-cond, critic) | ✅ | see below |
| P1 guardrails / X2-X5 executor / T1 league / T2-T3 world / A5 critic | ✅ | — |
| T5 curriculum · A2 IMPALA · A3 DT · P2-P5 refinements | ▶ | evals need runs |
**RL ETA (single process, measured 51 ms/game):**
| Games | Wall time | Purpose |
|-------|-----------|---------|
| 10k | ~9 min | quick signal |
| 50k | ~43 min | first competitive read |
| 300k (SNORLAX-scale) | **~4.3 hr** | full run |
**This is the slowest phase and the main speedup target (see below).**

## PHASE 3 — TEST / GATE (both seats, on Rust serve = bit-exact)
| Task | Status | ETA |
|------|--------|-----|
| smoke_test (0 errors, worst-turn <1s) | ✅ | ~10 s |
| seat-swapped vs 2 strong refs × 6 world-seeds | ✅ | **~55 s** (24 games) |
| clean-room banded (destbreso tapes) | ✅ | ~30 s |
| crown banded panel (28 refs) | ✅ | ~13 min (336 games) |
| **Full vetted-roster gate (58 reactive agents)** | ▶ | ~27 min — the "all public agents" gate |
| G3 world-diverse seeds · G4 latency · G5 official | ✅ | — |
**Phase ETA:** 55 s (iteration gate) → 13-27 min (ship gate).

## PHASE 4 — BUILD (per seat)
### Bandit seat
| Task | Status | ETA |
|------|--------|-----|
| compile base tape (policy→tape) | ✅ | ~0.3 s |
| build_rust_bandit (musl tarball) | ✅ | **~35 s** |
| dispatch D6/D12/D15/D21/D27 + endgame + 6 rails | ✅ | — |
| parity gate + vs-v46 gate | ✅ | ~1 min |
| B7 route2 (distinct opening) | ✅ | +35 s |
| **B4 regenerate branches for learned base** | ▶ | heavy search (branch evolution) |
### Track-P seat
| Task | Status | ETA |
|------|--------|-----|
| R1 ONNX export + parity | ✅ | ~5 s |
| **R2 native numpy inference (== torch)** | ✅ | validated |
| R3 self-contained bundle (numpy weights + exec) | ▶ | ~½ day build |
| R4 latency · R5 vs-v46 · R6 skeleton/NN switch | ▶ | — |
**Phase ETA:** bandit ~2 min; trackp seat needs R3 packaging (R2 core done).

## PHASE 5 — PAIR / SHIP
| Task | Status | ETA |
|------|--------|-----|
| daily_slot2 (build+gate both, report) | ✅ | ~3 min (minus train) |
| O4 pair report (bandit + route2/trackp) | ✅ | — |
| O5 schedule (never submits) | ✅ | — |
| operator submits | manual | — |

## PHASE 6 — LOOP (daily, self-improving)
| Task | Status | ETA/day |
|------|--------|---------|
| resume + T1 league growth | ✅ | — |
| 1 daily RL round (e.g. 30k games) + build + gate | ✅ | **~30-40 min/day** |

---

## HOW TO RUN MUCH FASTER (ranked by payoff)

The bottleneck is **TRAIN** (51 ms/game single-process). Ranked levers:

1. **Parallel rollout workers (biggest win).** `kagg batch` already uses all cores
   for the *game plays*, but the **plan COMPILE (serve) is single-process serial**.
   Run N compile+batch workers across processes → near-linear speedup.
   *Expected:* 300k games **4.3 hr → ~35-45 min on 6-8 cores.* ▶ build: a
   multiprocessing pool in `rollout_batch_fast`.
2. **Kill the compile with a correct obs-free executor.** X4 (dead-reckon) was
   REJECTED (fidelity). The correct-but-costly path: port the field mechanics
   exactly. *Expected:* removes the 261 ms/plan compile → ~2× on top of #1. ▶ (L).
3. **Fewer RL games via a stronger BC warm start.** A better BC init (bigger
   `--d-model`, more epochs, world-conditioned BC) needs fewer RL games to reach
   the same level. *Expected:* 300k → 50-100k games. ✅ knobs exist.
4. **Cheaper mid-training battery, full gate only at ship.** Battery every iter
   is on batch now (S4); drop `--battery` size mid-run, run the 28/58-ref gate
   only at promotion. *Expected:* -10-20% train time. ✅ knob (`--battery`).
5. **Raise `games_per_plan`** (amortise compile further). At gpp=128 the compile
   share halves again. ✅ knob (trade-off: fewer distinct plans/iter).
6. **Parallel gate.** Run the ref matchups across processes.
   *Expected:* 13-min crown gate → ~2-3 min on 6 cores. ▶ build.
7. **Cache opponent tapes** (destbreso prewarmed in one pass). ✅ done.

**Target after #1+#3+#4:** a competitive-read training round (50k games) in
**~5-8 min**; a full 300k run in **~35-45 min**; daily loop **~15-20 min**.

---

## The two gaps you flagged (actions)
- **Gate roster:** only reactive agents transfer (tapes are inert off-world), so the
  usable pool is the **58 vetted reactive agents**. Action: wire the **full 58-ref
  gate** as the pre-submit ship gate (the 28-ref crown panel is the mid-tier;
  2-ref is per-iteration). ▶
- **Algorithm evals (A2 IMPALA / A3 DT):** not run — they need both variants trained
  on a small budget and compared. ▶ (each ~1-2 hr of runs).
