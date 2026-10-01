# Submitting to Kaggriculture

Competition: <https://www.kaggle.com/competitions/kaggriculture>

| | |
|---|---|
| Entry / team-merger deadline | **23 September 2026** |
| Final submission deadline | **30 September 2026** |
| Leaderboard final | ~15 October 2026 (games keep running for ~2 weeks) |
| Prizes | $5,000 each for places 1-10 ($50,000 total) |

## The rules that actually bite

Straight from the Overview page's Evaluation and FAQ sections:

* **5 submissions per day**, per team.
* **Only your latest 2 submissions stay active** — and the final leaderboard is
  computed from those two. Submitting a weaker agent can *push a stronger one
  out of the active pair*. This is the single easiest way to lose rank by
  accident.
* **A Validation Episode runs on upload**: your agent plays *against a copy of
  itself*. If it errors, the submission is marked Error and the slot is spent.
  `src/kaggriculture/pipeline/submit.py` runs exactly this check locally first.
* **Rating is win/loss only.** "The actual coin difference in a match does not
  affect the rating change — only the win, loss, or tie outcome matters." Rank
  candidates by **win rate**, not by bank.
* **Submission size limit 100 MiB**; your files land in
  `/kaggle_simulations/agent/`, so imports must resolve from there.
* **Runtime resources: 1.6 vCPU, 6.5 GiB RAM, 8 GiB disk** — slower than a dev
  box, so keep a wide margin under the 1-second `actTimeout`.

This is a **simulation competition**: you submit an *agent*, not a predictions
CSV. Two agents play a 30-day farming season head to head and the one with the
larger bank balance wins.

## What the submission must look like

A single `main.py` at the root that defines `agent(obs)` (an optional second
`config` argument is passed if the function accepts one). All three agents in
`agents/` are self-contained and can be renamed to `main.py` as-is — there are
no imports beyond `math`.

For a multi-file agent, tar it with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py weights.pkl
```

## Steps

```bash
# 0. one-time: install the CLI and save your API token
pip install kaggle
mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/access_token   # paste token from
#     https://www.kaggle.com/settings/api  ->  "Generate New Token"

# 1. ACCEPT THE RULES IN A BROWSER FIRST -- the CLI cannot do this.
#    https://www.kaggle.com/competitions/kaggriculture  ->  "Join Competition"
kaggle competitions list --group entered        # verify you have joined

# 2 + 3. rank every agent by win rate, run Kaggle's own validation checks,
#        show your daily quota, then ask before uploading
python -m kaggriculture.pipeline.submit

#    or, to prepare without uploading:
python -m kaggriculture.pipeline.submit --dry-run

# 4. watch it play
kaggle competitions submissions kaggriculture           # note the submission id
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay  <EPISODE_ID> -p ./replays
kaggle competitions logs    <EPISODE_ID> 0 -p ./logs
kaggle competitions leaderboard kaggriculture -s
```

`src/kaggriculture/agentbuild/build_submission.py` copies the chosen agent to `build/main.py` and
re-runs the contract tests against it first, so a submission that would be
rejected for a malformed action never leaves the machine.

## Before every submission

```bash
python tests/test_agents.py            # legality, latency, beats all builtins
python -m kaggriculture.measure.evaluate agents/v2_tuned.py --vs agents/v1_heuristic.py agents/v0_baseline.py -n 6
```

Watch the reported worst turn time. Kaggle's runners are slower than a dev box
and `actTimeout` is **1 second per turn**; a timeout forfeits the episode.

## Things that silently cost you the match

* Illegal ops are **no-ops with no error**. A typo in an op name looks exactly
  like an idle farmer. `tests/test_agents.py` validates every emitted action.
* Only the first **10 market orders** each turn are processed — the rest are
  dropped without warning. Order the queue by priority.
* `hands` must line up positionally with `farms[me]["hands"]`.
* Produce left in a farmer's inventory at the final step scores **nothing**.

---

# Submit strategy (2026-08-10, the operating doctrine)

## The constraints that shape it

* Only the **latest 2** submissions stay active; the final leaderboard is
  computed from them. 5/day budget. A new submission starts at 600 and takes
  ~1 day of episodes to converge. Route freshness decays measurably per day
  (47% -> 86% -> 100% across three days of base age, measured 2026-08-10).
* Final submission deadline **2026-09-30**; entry/merger 09-23.

## Steady state: champion + daily challenger

1. **One submission per day**, in the morning right after the archive-day
   crown, so it converges while fresh. It evicts the older/weaker slot;
   the champion's converged rating is never sacrificed.
2. The pair is therefore: **yesterday's proven champion + today's crown.**
   If today's crown fails the >=10-point crown gate, submit nothing — a
   stale-but-converged champion beats a fresh-but-unproven sidegrade.
3. Slot A/B (config variants twinned on one base) only while a question
   needs live evidence, judged on W-L records at >=50 games, never on
   early ratings.

## Event-driven exceptions (submit outside cadence)

* **Balance change / engine bump** (like 2026-08-07's): all recorded routes
  and local numbers are suspect -> re-crown under the new engine and refresh
  BOTH slots as soon as gated candidates exist.
* **Both slots under water** (win rate < 50% at their level): double refresh.

## Endgame protocol (the week of 09-23 -> 09-30)

* Freeze experiments. Daily crown continues; only clear crown-gate winners
  ship.
* **The final pair must be submitted no later than ~09-28**, so both
  converge before scoring closes. Reserve the last 48h for convergence, not
  for changes.
* The final pair composition: the two highest-confidence agents on the
  freshest base available — by then the pipeline's daily rhythm IS the
  strategy; the endgame is just refusing to break it late.

## Selection keys, ranked

1. Referee tournament vs our own live losses (freshest signal we control).
2. Held-out panel (fixed roster, comparator).
3. Live W-L after ~50 games (ground truth, slow).
* Never: unweighted ladder win rate (circular), early ratings (path noise),
  recorded bank (measured misleading).
