# Which ML to use here, and why — including whether RL makes sense

## What the problem actually is

| property | value | consequence |
|---|---|---|
| Action space | ~18 ops × up to 14 units, plus an ordered 10-slot market queue | ~10¹⁷ joint actions per turn — cannot be enumerated or softmaxed |
| Horizon | 720 turns | credit assignment spans ~200 turns (a melon planted day 0 pays day 10) |
| Reward | final bank, **win/loss only** for rating | maximally sparse |
| Simulator | local, deterministic per seed, **~5 s/episode** | unlimited data, but *slow* per sample |
| Inference budget | 1 s/turn, 1.6 vCPU, 100 MiB | no GPU, no big model |

Those five rows decide everything below.

---

## Tier 1 — CMA-ES over the policy parameters ← **do this first**

**Implemented: `src/kaggriculture/train/cmaes.py`.**

The agent already *is* a parameterised policy: 28 continuous knobs that fully
determine behaviour. That makes this a low-dimensional, expensive, noisy
black-box optimisation — exactly what CMA-ES is for.

Why it beats the existing coordinate descent: CMA-ES adapts a full covariance
matrix, so it finds optima that require **two knobs to move together**. This
project has already hit that wall — `travel_weight` paid off, and then every
capacity knob measured flat when moved *alone* afterwards. That is the signature
of a correlated optimum a one-at-a-time search cannot reach.

Design choices that matter more than the optimiser:

* **Common random numbers** — every candidate in a generation sees the same seed
  set, in both seats. Comparisons are paired, so seed noise mostly cancels.
* **Opponent league** — candidates play a *pool* (v3, v2, v1), not one frozen
  copy. Tuning against a single opponent over-fits to it.
* **Checkpoint every generation** to `.local/cmaes/state.json`, so it survives
  being killed and resumes with `--resume`.

```powershell
python tools\cmaes.py --generations 20 --popsize 8 --seeds 4      # hours; run overnight
python tools\cmaes.py --resume
```

Expected: this is where the next few points of win rate come from, and it needs
no downloaded data.

---

## Tier 2 — Learn the *valuations*, keep the rules ← **the real ML win**

The planner's structure is sound; its weakness is that every task value comes
from a hand-derived formula, e.g. `crop_value_per_tile_day` and
`remaining_animal_value`. Those are guesses about a quantity that genuinely
depends on day, current prices, labour slack, shed pressure and the opponent's
board — far too many interactions for a closed form.

**Method: fitted value regression on simulated rollouts.**

1. Play N episodes, logging at each decision `(features of the tile/state, the
   op chosen, the realised discounted contribution to final bank)`.
2. Train a small model — gradient-boosted trees or even ridge regression — to
   predict that contribution.
3. Replace *only* the leaf estimators inside `build_tasks`. The safety rules
   (feed before starvation, water before weeding, haul before the final bell)
   stay hand-written.

Why this shape: inference is microseconds and megabytes, it fits the 1 s budget
trivially, and a mispredicted *value* degrades gracefully — whereas a
mispredicted *action* can lose an animal permanently.

---

## Tier 3 — Imitation from ladder replays ← **needs the download**

The daily top-episode datasets give `(observation, action)` pairs from the best
agents. Kaggle published them for exactly this.

**Do not clone the whole policy.** This game punishes rare mistakes permanently:
a policy that is 95% right still misses ~36 feeds per episode, and each one is a
dead animal. Clone the *allocation* decisions only — what to plant on an empty
tile, when to buy land, how many hands to hire — and keep rules for survival.

`src/kaggriculture/measure/action_diff.py` is the diagnostic that tells you whether this is even
worth it: it replays a top player's trajectory, asks our agent what it would do
in each identical state, and reports the systematic skews.

---

## Tier 4 — Actual reinforcement learning

**Short answer: yes, but not end-to-end, and not on this hardware.**

Why naive deep RL (PPO/DQN on the raw board) fails here:

* **Sample cost.** PPO on a problem this size needs ~10⁶–10⁷ episodes. At 5 s an
  episode on 2 cores that is measured in decades. It only becomes feasible with a
  rewritten vectorised simulator (the Python interpreter is the bottleneck, not
  the algorithm) and a cluster.
* **Action space.** ~10¹⁷ joint actions per turn. You would need autoregressive
  or factored action heads — a serious engineering project.
* **Sparse reward over 720 steps.** Needs heavy reward shaping, and the shaped
  reward is just the hand-written value function again.
* **Inference budget.** Whatever you train must run in 1 s on 1.6 vCPU.

Where RL *does* fit:

* **Evolution strategies over the knobs** — CMA-ES (Tier 1) *is* a policy-search
  RL method. It is the version of RL that fits this compute budget, and it is
  already implemented.
* **Per-tile Q-learning on a restricted decision.** Scope it to "what to do with
  *this* tile today" — 6 actions, local features, ~30-step horizon. Credit
  assignment is short and the action space is tiny. This is genuinely tractable
  and would replace the hand-written tile priorities.
* **Bandit over market timing.** "Sell now or hold" is a small, repeated,
  well-instrumented decision with a measurable counterfactual.

**Recommended order: Tier 1 → Tier 2 → Tier 3 → scoped Tier 4.** Anything that
promises to replace the whole planner with a learned policy is, on this compute
budget, a way to lose to a well-tuned heuristic.

---

## The rule that governs all of it

Four "obvious" signals have already measured negative in this project
(strawberry scarcity, rescue pens, cheaper animal labour, bigger crews). Whatever
a model suggests is a **hypothesis for self-play**, and nothing is believed until
it clears **three independent seed sets** in `src/kaggriculture/measure/evaluate.py`.
