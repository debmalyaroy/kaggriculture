# Learning from the ladder — dataset + model

Kaggle publishes the highest-rated episodes every day and says outright they are
for *"IL/BC, bootstrapping RL, or just gathering statistics"*
(competition discussion 731215). That is a labelled corpus of what the best
agents actually do.

## Pipeline

```
src/kaggriculture/data/build_dataset.py   ->  data/episodes.csv   ->  src/kaggriculture/train/learn_params.py
   download + featurise         1 row per player        what predicts winning
   + delete (streaming)         per episode             -> priors for tune.py
```

### 1. Build the dataset

```bash
python -m kaggriculture.data.build_dataset --days 3 --per-day 50      # ~150 episodes
python -m kaggriculture.data.build_dataset --days 5 --per-day 100     # ~500 episodes
```

Needs the `kaggle` CLI, authenticated, **on a machine with network**.

Replays are ~27 MB each and a daily dataset is ~21 GB, so nothing is bulk
downloaded. Each episode is fetched, reduced to ~60 numbers, and **deleted**
before the next one — disk high-water mark stays at one replay. The run is
resumable: episode ids already in the CSV are skipped, so you can stop and
restart freely.

Roughly 55 features per player: peak and cumulative tiles per asset, hiring
curve, land-purchase days, first-animal day, units sold per product, and the
distribution of unit actions (including `move_frac`, the share of turns spent
walking).

### 2. Learn what predicts winning

```bash
python -m kaggriculture.train.learn_params --top 25
```

Two views, deliberately:

* **Paired winner-minus-loser deltas.** Both players in an episode faced the
  same seed, town and market, so differencing within the episode removes all of
  it. This is the robust view — trust it first.
* **Logistic regression** (pure numpy, no sklearn) on standardised features.
  Disentangles correlated features the paired view reads one at a time — e.g.
  "more hands" vs "more hands *because* more land".

Output ends with suggested priors mapped onto real `PARAMS` knobs.

## How to use the output — and how not to

The priors go into `src/kaggriculture/train/tune.py`; **self-play decides**. This project has a
running list of "obvious" signals that measured negative:

| signal | looked like | measured |
|---|---|---|
| strawberry $272 vs base $120, melon in glut | grow strawberry, drop melon | **0% win rate, −$10,056** |
| 2 cows stranded in the shed at game end | build them a pen | **62.5% vs 81%** |
| zoning frees labour, so raise capacity | more tiles per hand | no gain on any knob |
| clustered animals are cheaper to service | `cost_per_animal_day` 4.5→3.5 | 80% on two seed sets, **56% over three** |

That last one is the cautionary tale: it looked like a clear win after 10
matches and evaporated at 16. **Always run a third independent seed set before
believing a result.**

Neither view is causal. A feature can predict winning because it causes winning
or because strong agents happen to do it. Treat the output as a hypothesis
generator for the search, never as a conclusion.

## Behaviour cloning — the bigger, riskier option

The same replays give `(observation, action)` pairs. A policy trained to imitate
winners has a higher ceiling than copying aggregate statistics, but:

* it must run in **< 1 s/turn on 1.6 vCPU** inside a **100 MiB** submission, so
  gradient-boosted trees or a small MLP, not a deep net;
* this game punishes rare mistakes permanently — one missed feed loses an animal
  for the rest of the season — so a policy that is 95% right can still be much
  worse than a rule that is always right;
* a hybrid is probably the sweet spot: keep the heuristic planner for survival
  rules (feeding, watering, the endgame haul) and learn only the *allocation*
  decisions.

Not attempted yet. See `docs/history/issues-and-improvements.md` C5.
