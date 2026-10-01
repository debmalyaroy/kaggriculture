# Daily pipeline — bandit + trackp pair, and the road to top-10

The single source of truth for how the two seats are built every day and what it
takes to reach top-10 (~2900). No hedging: one strategy, one pipeline, one gap.

## The strategy in one paragraph

There is **one learned economy** and **two ways to ship it**. A competitive plan
can't be found (recorded tapes are inert off-world) or hand-built (OR caps ~$50k);
it must be **learned** (BC→macro-RL from the 786 ≥2500-rated opponents, whose plans
bank ~100k). The **bandit seat** ships that policy as a compiled base tape + reactive
rails; the **trackp seat** ships the same policy as live NN inference. Top-10 needs
one seat banking a **~100k world-robust economy** — the entire competition reduces
to closing the gap from our current ~18k realization to the ~100k the corpus proves
is reachable.

## The daily pipeline (one command, never submits)

```bash
python -m kaggriculture.pipeline.daily_slot2
```

```
 1. DATA     fold new high-rated games into the BC corpus (best-effort; CDN-gated)
 2. TRAIN    BC warm-start (once) + a few RL rounds, RESUMING yesterday's best
                (models/rl/loop_state.json + rl_policy_best.pt)   [~30-60 min, 4060]
 3. BANDIT   compile policy → base tape + rails → build_rust_bandit → tarball
                → parity gate (ship==measure) + faithful gate vs v46
 4. TRACKP   export ONNX (+ native inference when F1.1 lands) → tarball
 5. PAIR     gate both seats banded (crown panel); pick the pair
                (bandit + trackp, else bandit + diversified route2)
 6. REPORT   release_report.md + SUBMIT_INSTRUCTIONS.md   ← YOU submit
```

Each stage is skippable (`--skip train bandit …`). Training is incremental and
**resumes** — every daily run makes the policy a little better, and the pair is
rebuilt + re-gated from the current best. Nothing submits; the report tells you the
exact `kaggle` commands and which seat cleared the bar.

**Schedule it** (Windows Task Scheduler / the repo's `scripts/`): run `daily_slot2`
once a day after the data fetch. It writes artifacts + a report; it never calls Kaggle.

### How the two seats map to the existing daily jobs
- The repo already runs `KaggricultureRefreshCycle` (05:00) and `KaggricultureTrackP`
  (13:30). `daily_slot2` is the Slot-2 learned-predator equivalent; wire it as a third
  daily job (or fold its train/build/gate into the refresh cycle) once the policy
  clears the bar. Until then it runs on demand and the live pair stays the current
  gated agents.

## What "done" looks like per seat

| | Bandit seat | Track-P seat |
|---|---|---|
| Train | shared BC→RL (done, fast) | shared BC→RL (done, fast) |
| Package | compile→tape→`build_rust_bandit` ✅ | ONNX ✅ → **native inference (F1.1) ▶** |
| Gate | parity ✅ + vs-v46 ✅ | vs-v46 (once packaged) |
| Runtime | tape replay + rails (low risk) | live NN (reactive, higher ceiling) |
| Blocker | none (economy quality) | native inference build |

## Road to top-10 — the ONLY gap left is plan quality

Everything below the policy is built and fast:
- **Rollouts:** 7.7 ms/game, 300k games ≈ 38 min (B4, bit-exact).
- **Executor:** season_cbs PC-TAPF, +158–726% over greedy (B3).
- **Corpus + pool:** 440k macro rows, 786 ≥2500 opponents.
- **Gate + package + daily pipeline:** done.

The gap is that the learned **plans** don't yet bank ~100k. Three concrete, non-GPU
fixes, in order — this is the whole remaining competitiveness roadmap:

1. **Plan-space guardrails (highest value).** The policy's raw plans BANKRUPT (bank 0):
   they over-commit the $3000 day-0 bootstrap and mix animals the executor can't feed.
   Constrain the action decode so day-0 spend respects cash and commitments are
   coherent. Turns the RL reward from mostly-zeros into a real gradient.
2. **Cow/goose husbandry** in `SeasonCbsController` (today: sheep only). Then
   animal-bearing plans (the frontier uses them) stop starving.
3. **Train to climb.** With (1)+(2), run the daily loop; the policy learns the
   bootstrap-viable ~100k plans its corpus demonstrates. Add **world-conditioning**
   (feed day-6 `world_sig` + re-plan) for per-world adaptation.

Then **native inference (F1.1/G5.1)** unlocks the trackp reactive seat as the
higher-ceiling second slot.

**Definition of success:** the daily gate shows a seat ≥ 0.9 vs v46 on the faithful
banded panel → `daily_slot2` builds its tarball → you submit → the ladder confirms.

## Why this stopped being confusing
The economic walls are now all named and mostly cleared: routing (B3 ✅), throughput
(B4 ✅), fidelity (A ✅), gating (C ✅), packaging (both seats). The only open work is
**plan economics** (guardrails + husbandry + training) and the **trackp native-inference**
build. That's it — no more discovery, just execution of those items via the daily loop.
