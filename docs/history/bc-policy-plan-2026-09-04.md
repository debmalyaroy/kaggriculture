# Learning a policy from the top 100 — plan (2026-09-04)

## Why this and not more of what we did today

Everything in the harness lane is closed, on measurement:

* every adaptive layer is quantity-conserving (`_net_due` borrows a future
  sell and repays it), so none can change what we produce;
* six layer-level fixes built today, five measured negative or inert, one
  ($15/episode) rejected at p = 0.0002;
* dispatch is dead at every scale: headroom PEAKS at ~20 economies and
  shrinks (0.111 -> 0.095 going 20 -> 40), and an honest dispatcher is below
  best-single at pool sizes 2, 5, 10, 20, 30 and 40;
* search cannot rescue the runtime policy: 600 generations of genome GA on
  the right objective moved margin -76k -> -50,595 with score stuck at 0.000.

And the reason the tape lane is capped is now measured, not argued: **rank 1's
mined tape scores 0.071 on real 2300+ worlds** -- worse than our own base and
worse than the factory default. A replay is a trace of decisions, not the
decision rule. The better the agent, the less its tape carries. We cannot copy
them by mining them, and we cannot search our way to them.

**Behavioural cloning is the one route that transfers a policy rather than a
trajectory, and it skips the search that has failed all day.**

## What we hold (measured, not estimated)

    TOP 100:  3,994 episodes | 3,983 fully re-simulatable | 99 distinct teams
              2,766 winning seats | ~2,870,000 state-action pairs
    TOP 500: 10,049 episodes | 10,016 re-simulatable | 467 teams | ~7.2M pairs

Every route record carries its `seed` and both seats are indexed, so every
episode replays to the dollar locally -- no downloads, no quota.

State substrate already exists (`src/kaggriculture/data/turn_features.py`): day/hour/step,
9 prices, 9 market inventories, 7 own-farm stats, 7 OPPONENT-farm stats,
9 shed, 5 seeds. That is exactly "market + economy, both sides".

## WHERE the policy starts -- measured, not assumed (2026-09-04)

Same team, same DAY (one submission, so version churn cannot inflate it):
fraction of that team's games sharing byte-identical actions per segment.

| team | rank | d0-2 | d3-5 | d6-11 | d12-17 | d18-23 | d24-29 |
|---|---|---|---|---|---|---|---|
| Crop Dusta | 1 | 0.93 | **0.21** | 0.07 | 0.07 | 0.07 | 0.07 |
| Jesse Bullard | 2 | 1.00 | 1.00 | **0.20** | 0.20 | 0.20 | 0.20 |
| keiz | 3 | 0.75 | 0.50 | 0.08 | 0.08 | 0.08 | 0.08 |
| MtN | 6 | 1.00 | 1.00 | 1.00 | **0.14** | 0.07 | 0.07 |
| Andrey Tikhomirov | 7 | 0.93 | 0.93 | 0.93 | **0.36** | 0.07 | 0.07 |
| Wang H2O | 14 | 0.93 | 0.86 | 0.86 | 0.86 | **0.21** | 0.57 |
| mid-field (yy, kaguramena, Kirderf, ...) | 100+ | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Every top team runs a FIXED opening segment and then decides at runtime.
The switch point is team-specific: day 3, day 6, day 12, day 18.** The
mid-field is scripted end to end -- and so are we, for all 720 steps.

This is why rank 1's mined tape scores 0.071 (docs/history/hard-band-2026-09-04.md):
93% of their games share only d0-2; everything after is a live decision, so
one recorded sample transplanted into another world is noise.

CONSEQUENCE FOR THIS PLAN: BC must be trained on steps AFTER each team's
script ends, and the opening should be taken as a fixed prefix, not learned.
Per-team script length is computed by this same segment test and stored with
the corpus.

## Architecture -- four stages, each independently gated

### Stage 0 -- corpus (compute, no risk)
Re-simulate 3,983 episodes from seed + both action tracks; capture the state
vector each turn; pair it with the action THAT TEAM played. Label every row
with episode id, team, team rating, seat, and final outcome.
Split BY TEAM as well as by episode -- a policy that has seen a team's other
games is not being tested on generalisation.

### Stage 1 -- the "what should we do" question, answered BEFORE training
Fit an interpretable model (gradient boosting + permutation importance) to
predict the top-100 action from state, and compare it against OUR action in
the same state. This is cheap and it derisks the whole project:

* if top-100 actions are NOT predictable from state, they are partly
  open-loop and BC will fail -- we stop here and save the compute;
* if they ARE, the importance surface tells us WHICH decisions differ from
  ours, and that alone is actionable even without shipping a network.

This is the step that answers "based on the combined game play, what should
we do?" and it is the next thing to run.

### Stage 2 -- policy head over MARKET decisions only
The harness keeps the tape for field ops (plant/water/harvest are heavily
scripted and our tape executes them correctly) and the learned policy decides
SELLS: which product, how much, when. Rationale: this is the smallest action
space, it is where every measured leak lives (race edge $860 against a $1,300
median close-loss deficit; share 46.6% in poor worlds), and it is reversible
-- the tape still plays if the head abstains.

### Stage 3 -- extend to production
Plant/animal/land decisions. Today proved production, husbandry and selling
are ONE coupled schedule, so this stage must move them together or not at
all. Only attempted if Stage 2 clears.

### Stage 4 -- full policy, no tape
The 2800 architecture. Every action from live state, which is what the top of
the leaderboard demonstrably runs.

## How the HARNESS changes, concretely

Today's harness = replay a tape + ~15 repair/re-timing layers. It is replaced
from the market end inward:

| stage | tape decides | policy decides |
|---|---|---|
| now | everything | nothing |
| 2 | field ops, production | sells |
| 3 | field ops | sells, production |
| 4 | nothing | everything |

Each stage DELETES layers rather than adding them: at stage 2, `_relay`,
`_pull_sells`, `_plead`, `_premium_lead`, `_market_timing`, `_impact_slots`
and `_sell_first` all become dead code, because the head decides directly what
they were re-timing. That is the harness improvement -- fewer moving parts,
each replaced by a decision rather than a correction.

## Gates (unchanged bars, no new instruments)

1. Hard band CREDITED wins > 19/56 and median margin > -1,418 (v44.0 live).
2. `train_gates.policy_allowed()` -- the existing sign-tested gate that
   refused the last policy head at score_delta -0.2885.
3. Reactive band: no regression on the held-out median margin.
4. Latency: mean turn < 20 ms, worst < 100 ms (house rule; the real limit is
   1 s/turn).
5. Single file, `math` only -- weights embed as base85+zlib with a pure-python
   forward pass, as the GRU and logistic already do.

## Risks, stated up front

* **Distribution shift.** BC is trained on states the expert reached; once our
  actions differ we leave that distribution. Mitigation: Stage 2 keeps the
  tape driving the field, so the state stays near the demonstrated one.
* **The experts may be partly open-loop too.** Stage 1 detects this before we
  spend the compute.
* **BC lands below its demonstrator.** They rate 3013 and we rate 2314, so
  there is a large margin to give away and still gain.
* **Embedding size.** A large network may not fit the single-file constraint;
  the fallback is a distilled small net or a decision table.

## Cost

Stage 0-1: a few hours of local compute, no downloads, no submissions.
Decision point after Stage 1: proceed to training, or stop with the
interpretable findings and spend the slots elsewhere.
