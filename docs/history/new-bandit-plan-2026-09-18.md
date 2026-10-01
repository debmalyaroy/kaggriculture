# Plan: build a new bandit (2026-09-18)

Two questions, kept separate because they have very different answers:

* **(A) Build *a* new bandit** — the mechanical config→generate→build→gate→package
  pipeline. **Ready today**, ~30–60 min, no blockers.
* **(B) Build a *competitive* bandit** — one that clears the v46 bar (B4.0 ≥ ~0.9).
  **Blocked on a bigger base economy**; the current OR/v57 family loses to v46
  (measured 61,578 vs 65,498). The lever is production VOLUME (B4.2), which the
  GPU-trained Slot-2 policy supplies.

---

## What a bandit IS (the pieces a build assembles)

The shipped bandit is the config-driven `kagg mbandit` harness. Everything is
hot-swappable via one `configs/bandit_config_*.json`:

| Piece | Where | Swappable |
|-------|-------|-----------|
| **base_tape** | `--base-tape *.tape` (e.g. `rustengine/tapes/bandit_v57_egg.tape`) | yes — the economy backbone |
| **branches** | generated per active checkpoint (D6 mandatory; D12/D15/D21/D27 optional) | yes — `config.checkpoints` |
| **rails/guardrails** | `config.order` (ordered registry, G1.4) + toggles/knobs | yes — reorder/add/remove |
| **tables** | `config.tables` (sell_priority, dispatch K, sweep, endgame, r37 curves) | yes (G1.2) |
| **genome/knobs** | `config` array knobs (anti_dump, scarcity_i0, …) | yes (G1.3/M9) |

Ship path == measure path (Approach B, G7.C1): the economy lives ONLY in
`kagg mbandit`; the driver fallback is a loud legal-PASS. So whatever the parity
gate measures is exactly what ships.

---

## (A) Build *a* new bandit — the ready pipeline

1. **Author/choose the config** — `configs/bandit_config_v581.json` is the current
   template. Edit checkpoints, rail order, tables, knobs. Validation (G3.2/G6.2)
   fails loudly on an unconsumed/dead knob.
2. **Choose a base tape** — today: `rustengine/tapes/bandit_v57_egg.tape` (or
   `_yarn`). This is the economy ceiling of the whole build (see B).
3. **(If checkpoints changed) regenerate matching branches** — the tape generator
   evolves branches for exactly the active checkpoint set:
   `python -m kaggriculture.bandit.build.build_multi_ckpt …` /
   `build_bandit_harness.py` (branches nest under D6→D12→…).
4. **Build (isolated, musl)** — never touch `rustengine/kagg.exe`:
   ```
   python -m kaggriculture.bandit.build.build_rust_bandit \
       --config configs/bandit_config_vNNN.json \
       --base-tape rustengine/tapes/<base>.tape --build --tar
   # -> .local/candidates/<name>.tar.gz  (static ELF + main.py + config + tapes)
   ```
5. **Gate 1 — parity (ship==measure), non-skippable:**
   ```
   CARGO_TARGET_DIR=.local/scratch/bandit_target cargo build --release \
       --bin kagg --manifest-path rustengine/Cargo.toml
   KAGG_BIN=.local/scratch/bandit_target/release/kagg.exe \
       python tests/test_bandit_parity.py     # bridge 719/fallback 0, sells>0, config truthful
   ```
6. **Gate 2 — faithful competitive, vs the strong refs (THE ship decision):**
   ```
   python -m kaggriculture.measure.eval_harness \
       .local/scratch/bandit_parity_stage/main.py \
       --refs .local/panel/ref2500/ref_v46_chassis_2517.py --seeds 6
   # SHIP as a PRIMARY seat iff aggregate ≥ ~0.9 vs v46 (B4.0). Else = fallback/route2.
   ```
7. **Hand off** — the pipeline NEVER submits. On a green Gate 2, the operator
   submits the tarball; latest-2 rule means it evicts the older active seat, so
   confirm the pair keeps two strong agents first.

**Status:** every step above is built and was exercised this session
(`bandit_v581_0918.tar.gz`, parity PASS). What that build lacks is only economy.

---

## (B) Build a *competitive* bandit — the real work

**Diagnosis (B4.2, measured):** wrapping / sell-timing / routing are NOT the gap
(variants within 1%). The bandit's economy banks ~49–61k while ladder opponents
bank 140–172k in the same games — it is **~3× too small**. The base_tape is the
ceiling, and every base_tape we have (OR $49k, v57 family) is ~3× short.

So a competitive bandit needs a **base_tape with a ~3× larger realized economy.**
Ranked by expected payoff:

### B1 (primary) — compile the trained Slot-2 policy into the base tape
The Slot-2 BC→macro-RL policy learns the large adaptive economy of the 786 ≥2500
destbreso tapes. Once trained (GPU), turn its plan into a bandit base_tape:
1. Train: `python -m kaggriculture.pipeline.slot2_pipeline --stages bc,rl,package,gate`.
2. Roll the policy's greedy plan and emit a tape:
   `macro_env.MacroEnv().rollout(plan, seed).tape_cells` →
   `macro_env.cells_to_batch_tape(cells, seed)` → `*.tape`.
   (This wiring exists; add a thin `--emit-base-tape` entry to `package_policy`.)
3. Use that tape as `--base-tape`; the bandit rails add contention-aware selling on
   top (the exact "OR backbone + reactive shell" idea, but with a *learned*, large
   backbone instead of the too-small OR one).
4. Gate 2 vs v46. **This is the unification of the two tracks and the most likely
   path to ≥0.9.**
   *Prereq:* the GPU training run (E3.5) + a better micro-executor if the greedy
   B2.0 realizer caps production (see B3).

### B2 (parallel) — raise the OR base tape toward the bar
OR peaks ~$49k (capital-bootstrap ceiling, not market/routing). Untapped:
STRAWBERRY (~$34k budget), distributed husbandry, bootstrap-CMA-ES. Realistic
gain is maybe +$20–30k — still short of 3×, so OR is a **fallback base**, not the
primary competitive one. Worth doing only as route2 diversification.

### B3 (enabler for B1) — lift the micro-executor
The reward/executor ceiling is the greedy B2.0 realizer (under-routes dense
economies). PC-TAPF / plan-repair (B2.2/2.3 `season_cbs` + B2.5) is the pluggable
upgrade (`MacroEnv(executor=…)`); it raises BOTH the RL reward signal and the
quality of a compiled base tape. Needed if B1's greedy tape banks too low.

### B4 (throughput) — Rust-native rollouts
For the ~300k-game RL budget, wire the obs-free executor to `kagg-engine batch`
(the lean bin built today) so rollouts run on Rust instead of the vendored engine.

---

## Critical path to a competitive bandit

```
GPU train Slot-2 (E3.5)  ──►  compile policy → base_tape (B1.2)  ──►  build_rust_bandit
        │                                                                    │
        └─(if greedy tape too low)─► PC-TAPF executor (B3) ──► retrain ───────┤
                                                                              ▼
                                              parity gate ──► vs-v46 gate ≥0.9 ──► operator submits
```

**Blockers:** only the GPU training run (E3.5) and, if the greedy base tape banks
too low, the PC-TAPF executor (B3). The bandit build/gate/package machinery itself
is 100% ready. Nothing here submits; the operator makes the final call.
