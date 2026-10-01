# Public notebook review — what was taken, what was not, and why

Reviewed 2026-08-04. The most useful source by far was
**raykkretzschmar — "Findings from Zero to Top Meta"**, because it is itself an
aggregation: its "Public work I used" table states what it took from each of the
other notebooks, which gives coverage of the whole list.

| Notebook | Contribution (per its own or the meta write-up's framing) |
|---|---|
| bovard — Getting Started | agent contract |
| georgymamarin — Visualized: what every crop pays | mechanics and market charts |
| romantamrazov — Hamburger | staged mixed herds, clone-aware market timing |
| pilkwang — Structured Economic Policy | economic scheduler lineage |
| prvsiyan — Frontier Lab | cross-play gates |
| degnonguidi — Agent Builder | scaffolding |
| tetsutani — Adaptive Farming Strategy | adaptive scheduling |
| saitejabandaruin — Pure Architecture (2600+ Elo) | speed-over-ensemble argument |
| raykkretzschmar — Zero to Top Meta | bug table, evaluation protocol, build ladder |

---

## Incorporated

### 1. Elo as the scoreboard — `src/kaggriculture/measure/elo.py` ✅ built
> *"Optimize only mean bank vs starter → high local bank, mediocre Elo."*

Exactly the failure mode this project was drifting into. There is now a local
Elo ladder over every agent in `agents/`, round-robin, both seats, persisted in
`.local/elo/ladder.json` so each iteration is rated against every predecessor.
First run:

```
agent_v3_20260804_160019.py   655   80% over 10 games
v2_tuned.py                   629
v1_heuristic.py               581
ml_rl.py / ml_ridge.py        578 / 577   (untrained, as expected)
```

### 2. The evaluation protocol ✅ already matched, now explicit
Their §8 list, against what we do:

| their step | ours |
|---|---|
| compile/import the exact packaged main.py | `src/kaggriculture/pipeline/submit.py` imports `build/main.py` |
| self-play validation, both DONE | `src/kaggriculture/pipeline/submit.py` runs Kaggle's own check |
| previous best on several seeds, **both seats** | `src/kaggriculture/measure/evaluate.py` default |
| at least one different public family | `src/kaggriculture/train/cmaes.py` uses an opponent **pool** |
| keep results including losses | `docs/history/loss-analysis.md`, `.local/elo/ladder.json` |

> *"Shared-market games are not symmetric. A one-seat test can reverse the
> apparent winner."* — confirms the both-seats design.

### 3. Diversify the second submission slot ✅ adopted as policy
> *"Two near-identical active submits → meta shift kills both."*

Our two active slots are v2 and v3, which differ by two knobs. Recorded in
`MEMORY.md`: the next submission should be a *different family* (the RL or ridge
bot once trained), not another v3 variant.

### 4. Speed is strategy ✅ already enforced, now understood
> *"When an agent times out, it defaults to `PASS`, completely ruining its
> economy."*

We knew the 1 s limit; we did not know the failure is a silent `PASS` rather
than an error. That makes our worst-turn gate (250 ms, well under 1000) more
valuable than it looked. Measured worst turn: **3.8 ms**.

---

## Tested and rejected — with numbers

Two entries from their bug table were implemented and **measured worse**. Both
are real bugs *in their architecture* and non-bugs in ours, which is the most
interesting finding of the review.

| Their bug | Our result | Why it does not transfer |
|---|---|---|
| *"CARE ranked above melon WATER → day 9 waters only part of the field; yield ~70 not ~96"* | `water_window_priority 3.0` → **0% win rate, −$6,181** | Their fix presumes a priority **ladder**, where CARE can sit above WATER. This agent prices every job in **dollars**, so the trade-off is already encoded — a marginal watering is worth `bonus × price` and competes on that. Tripling it just makes units cross the farm for a $250 job, undoing the spatial-efficiency win. |
| *"No DROP after HARVEST → produce stuck in unit inventory"* | `drop_stack_value 900` → **25% win rate, −$874** | Real risk, already covered: inventories auto-drop to the shed every night, so only the **final** day's load is genuinely at risk — and the endgame haul (worth +$25k when added) already handles that. Interrupting a busy unit mid-job costs more than the deferred banking. |

Both knobs are kept in `PARAMS` at neutral values with the measurement recorded
inline, so the tuner can revisit them if the architecture changes.

---

## Not incorporated, and why

| Idea | Why not |
|---|---|
| **The 2600-Elo agent's actual heuristics** | Its logic ships as a **compressed base85 payload** — deliberately opaque. Nothing readable to learn from, and copying an encoded blob would be both unusable and against the spirit of the work. |
| **Massive multi-agent ensembles** | The same notebook argues against them: they hit the 1 s timeout and default to `PASS`. Our planner runs in ~1 ms. Agreed and already avoided. |
| **"Fixed 10 cows forever → mirror banks collapse toward ~40k"** | Our herd is already mixed (cow 8 / sheep 12 / goose 12) and `max_herd` was tuned to 12 by search. Their prescribed fix — *"add sheep/strawberry/opponent routing"* — is two-thirds already done; **opponent routing** is the missing third and is the top open item in `docs/history/issues-and-improvements.md` B1. |
| **"Clone-aware market timing"** (Hamburger) | Genuinely new and probably valuable: detect a near-mirror opponent and sell *before* they flood the shared market. Requires reading `obs["farms"][1-me]`, which we do not do at all yet. Not implemented because it is a real feature, not a tweak — logged as the next structural item. |
| **"Protect melons until sold" / "reserve wheat before animal spam"** | Already true here: `DIG` never targets a living plant, and the feed-runway reserve (a fix worth ~$8k) predates this review. |
| Visualisation notebooks (Mamarin, prvsiyan) | Their crop-economics charts duplicate `docs/history/environment-rules.md`, which is derived from the interpreter source rather than from observation — strictly more reliable. |

---

## Net effect

Nothing from the review raised the agent's win rate directly. What it changed:

1. A **scoreboard** that matches how the competition scores (Elo, not bank).
2. Independent confirmation that the evaluation protocol is right.
3. A submission-slot **diversity** policy.
4. One concrete named feature to build next: **clone-aware / opponent-aware
   market timing**.

And a reusable lesson: a bug report from another architecture is a hypothesis
about ours, not a patch. Two of them measured negative.
