# Runbook

Everything you can run, what each option does, and what to do when it breaks.

All commands assume:

```powershell
cd D:\codebase\kaggriculture
```

Every script writes a timestamped log to `.local\logs\<name>-<yyyyMMdd-HHmmss>.log`
and never writes outside the project.

---

## 0. One-time setup

```powershell
python -m pip install --upgrade kaggle          # the CLI
.\scripts\download.ps1 -Check                   # verify CLI, auth, entry, disk
```

If PowerShell refuses to run the scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download.ps1 -Check
```

or, once per machine:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Your Kaggle credentials live in `%USERPROFILE%\.kaggle\` and are never read,
copied or logged by anything in this project — the tools shell out to the
`kaggle` CLI, which reads them itself.

---

## 1. The scripts at a glance

| Script | What it does | Typical run time |
|---|---|---|
| `scripts\download.ps1` | Pull top-20 ladder replays + all your own games | 2–20 min |
| `scripts\test.ps1` | Legality, latency, agents still win | 30 s (`-Full`: ~5 min) |
| `scripts\simulate.ps1` | Win rate, official episode, Elo, diff vs top-20 | 2–15 min |
| `scripts\train.ps1` | The search / ML algorithms that produce a new agent | 10 min – hours |
| `scripts\pipeline.ps1` | All of the above, in order | 30 min – hours |
| `scripts\submit.ps1` | Validate exactly as Kaggle will, then optionally upload | 3–8 min |

Each one prints a `Next:` block suggesting what to run afterwards. Each accepts
`-?` for full help:

```powershell
Get-Help .\scripts\train.ps1 -Detailed
```

### The one command — autopilot

```powershell
python tools\autopilot.py                   # fetch -> train -> build -> gate
python tools\autopilot.py --quick           # small budgets, ~15 min
python tools\autopilot.py --submit          # adds the submit stage (still asks)
python tools\autopilot.py --status
python tools\autopilot.py --emit-notebook   # a Kaggle/Colab notebook of the same

.\scripts\autopilot.ps1 -Schedule           # daily at 03:00, unattended
.\scripts\autopilot.ps1 -Status
.\scripts\autopilot.ps1 -Schedule -Remove
```

Five stages, always in that order, **never overlapping**:

| Stage | What it does |
|---|---|
| FETCH | `routes.py --mine` over the top-200, then `mine_top.py` for the derived tables |
| TRAIN | `schedule.py --derive` (sale timing from the new data), then `optimize.py` on the parameter vector |
| BUILD | `build_submission.py`, then `test_contract.py` **and** `conformance.py` over the packaged file |
| GATE | `evaluate.py` against the incumbent on seeds `--seed0 + 1000*cycle` — seeds the search never saw |
| SUBMIT | only with `--submit`, and `submit.py` still demands a typed `SUBMIT` |

**One architecture, configuration per cycle.** There is no new model each run.
`agents/v1_heuristic.py` is the only policy source; a cycle rewrites its
`PARAMS` and `SELL_SCHEDULE` blocks against the new data. That is precisely
what makes two cycles comparable — if the architecture moved too, a gate result
would not tell you which change did the work.

**Promotion needs a win on unseen seeds** (>55% and a positive margin).
Not promoting is the common outcome and it is the point.

Budgets come from the core count, not the clock, so a Kaggle notebook (~4
cores) runs the same stages with fewer trials rather than truncated ones:

| | mine GB | generations | popsize | seeds | gate seeds |
|---|---|---|---|---|---|
| this box (24 cores) | 12 | 8 | 12 | 3 | 6 |
| a small box / Colab | 4 | 4 | 8 | 2 | 4 |
| `--quick` | 2 | 2 | 6 | 2 | 3 |

**Scheduled runs refuse the submit stage outright.** Five submissions a day and
only the latest two active mean an unattended upload is a way to evict a better
agent overnight.

### Run these two before you trust any number

```bash
python -m kaggriculture.engine.engine_check     # is this the engine the ladder scores with?
python -m kaggriculture.engine.conformance      # do our agents match AGENTS.md / README.md?
```

**`engine_check`** exists because a Kaggle notebook image has shipped a
*different* engine than the ladder — `startingMoney` 2000 against 3000, `COW`
600 against 400, `farmHandCostMult` 10 against 1, and `SELL FERTILIZER`
silently dropped. None of that raises. It answers a different question with
total confidence, which is worse than crashing. The check compares 14 named
constants, a sha256 fingerprint over all 121 economy constants, and a
behavioural probe that actually sells a fertilizer and watches the money move.
`require()` runs inside `evaluate`, `elo`, `optimize`, `trajectory` and
`submit`, cached per process and inherited by pool workers.

This box records fingerprint `95854b693b4183cc`; the baseline lives in
`data/engine_baseline.json`. When Kaggle ships a legitimate engine update the
fingerprint will change — re-baseline **deliberately** with
`--update-baseline`, and say so in `BUILD_JOURNAL.md`, because every
measurement taken before that point was taken against a different game.

**`conformance`** audits our agents against the competition's own
`AGENTS.md` and `README.md`, which ship as competition *data*
(`kaggle competitions files kaggriculture`) and do get revised. It separates
two things deliberately: **contract** rules an agent can violate (illegal ops,
misaligned `hands`, more than 10 market orders — all of which the interpreter
punishes *silently*), and **coverage**, the documented observation fields we
never read. Ignoring a field is not a violation; it is a strategy gap with a
citation.

---

## 2. Download

```powershell
.\scripts\download.ps1                                  # defaults
.\scripts\download.ps1 -PerDay 200 -Days 7 -Jobs 12     # a lot more data
.\scripts\download.ps1 -Check                           # preflight only
.\scripts\download.ps1 -Diagnose                        # why top-N found no names
.\scripts\download.ps1 -Schedule                        # hourly Windows task
```

| Option | Default | Meaning |
|---|---|---|
| `-Days` | 3 | How many daily top-episode datasets to mine. Kaggle publishes one per day. |
| `-PerDay` | 40 | Episodes taken from each day. **This is why you only got 40** — it is a per-day cap, so the default total is 40 × 3 = 120. Raise it freely. |
| `-Own` | 0 | Your own episodes. `0` means **all** of them. |
| `-TopN` | 20 | Restrict ladder episodes to the top-N leaderboard teams. |
| `-Jobs` | 8 | Parallel downloads. See "Why it used to be slow" below. |
| `-AnyTop` | off | Skip the team filter entirely and take the highest-rated episodes. |
| `-Check` | off | Preflight only: CLI, auth, competition entry, disk. |
| `-Diagnose` | off | Print raw leaderboard CLI output and what gets parsed from it. |
| `-Verbose2` | off | Echo every Kaggle CLI call and its duration. (`-Verbose` is reserved by PowerShell.) |
| `-Report` | off | Rebuild `docs\eda-report.html` afterwards. |
| `-Schedule` / `-Remove` | off | Install / remove the hourly scheduled task. |

### Why it used to be slow

Each replay is ~27 MB **and** each one spawned a fresh `kaggle.exe`: SDK import,
TLS handshake, auth round-trip, then the transfer. That is 1–2 s of dead time
per file before any payload moves, strictly serial. Downloads now run in a
thread pool (`-Jobs`, default 8) with featurising overlapped on the main
thread, so network and CPU stop taking turns. On a 16-episode batch this was a
~6× wall-clock improvement in testing.

If your link is fast, `-Jobs 16` helps further. If Kaggle starts returning
429s, drop back to 4.

### Disk

Nothing accumulates. Each replay is fetched, reduced to ~60 numbers, and
deleted. `data\episodes.csv` grows by two rows per episode. State lives in
`data\fetch_state.json`, keyed by episode id, so an interrupted run costs
nothing and a double run duplicates nothing.

---

## 3. Tests

```powershell
.\scripts\test.ps1                  # fast: compile, import, legality, latency
.\scripts\test.ps1 -Full            # adds the beats-the-built-ins matches
.\scripts\test.ps1 -Agent agents\agent_v3_20260804_160019.py
```

Four layers, cheapest first:

1. **compile** — every `.py` parses.
2. **import** — each agent loads and exposes `agent(obs, config)`.
3. **contract** — legal action dicts, legal ops, and per-turn latency against
   the real 1-second `actTimeout`. This is what silently kills a submission:
   Kaggle does not tell you your agent timed out, it just loses.
4. **baseline** (`-Full`) — each agent must beat `pass`, `random` and `starter`.
   Slow, because these are full 720-turn matches, and it is the only layer that
   catches a change that runs fine but plays worse.

Run the fast layers after every edit; run `-Full` before submitting.

---

## 4. Simulation

```powershell
.\scripts\simulate.ps1                                   # standard set
.\scripts\simulate.ps1 -Mode quick -Matches 16
.\scripts\simulate.ps1 -Mode all                         # everything
.\scripts\simulate.ps1 -Mode match -Vs agents\v2_tuned.py -Replay
```

| `-Mode` | Tool | What it tells you |
|---|---|---|
| `standard` (default) | quick + official + elo | The three numbers worth checking after a change |
| `quick` | `evaluate.py` | Win rate over N seeds × both seats |
| `official` | `official_eval.py` | One episode under stock config with the real 1 s `actTimeout` — what Kaggle actually runs |
| `elo` | `elo.py` | Round-robin rating, accumulated in `.local\elo\ladder.json` across iterations |
| `match` | `run_match.py` | A single match, optionally with a replay file |
| `diff` | `diff_sim.py` | Replays downloaded top-20 games and asks what *our* agent would have done from the identical state |
| `graph` | `model_graph.py` | Regenerates `agents\<model>.html`, the per-model knowledge graph |
| `all` | everything | |

Other options: `-Agent` (default: newest `agents\agent_v*.py`), `-Vs`,
`-Matches` (seeds per opponent, each played twice — once per seat), `-Rounds`
(Elo), `-Seed0`, `-Workers`, `-Replay`, `-IncludeBuiltins`.

**Read win rate, not coin margin.** The Kaggle leaderboard is a skill rating: a
10-coin win and a 10,000-coin win score identically. Margin is only useful as a
tie-break between two agents with the same record.

---

## 5. Training

```powershell
.\scripts\train.ps1 -Algo learn                          # start here
.\scripts\train.ps1 -Algo cmaes -Generations 12 -Seeds 6
.\scripts\train.ps1 -Algo tune -Minutes 45
.\scripts\train.ps1 -Algo rl -Episodes 400
.\scripts\train.ps1 -Algo all -Minutes 20
```

| `-Algo` | Tool | What it is | When to use it |
|---|---|---|---|
| `learn` | `learn_params.py` | Attribution over the downloaded ladder data: which state features predict winning. Produces a ranking, not an agent. | First. It tells you which knobs are worth searching. |
| `tune` | `tune.py` | Coordinate descent over `PARAMS`, one knob at a time, paired both-seat matches on common random numbers. | Cheap, interpretable, gets stuck in local optima. |
| `cmaes` | `cmaes.py` | CMA-ES over all 28 knobs at once. | The right optimiser here: correlated, noisy, no gradients. Checkpointed — `-Resume` continues. |
| `ridge` | `train_supervised.py` | Ridge regression on decision-level outcomes, written into the `ASSET_BIAS` hook. | When you have a lot of episodes and want a learned prior. |
| `rl` | `train_rl.py` | Monte-Carlo control over the same hook. | Slowest, highest variance. Needs a few hundred episodes before it beats the heuristic it started from. |

Options: `-Base` (default: newest agent), `-Minutes` (tune budget),
`-Generations` / `-Popsize` (CMA-ES), `-Episodes` (ridge/rl), `-Seeds`,
`-Seed0`, `-Workers`, `-Top` (learn), `-Resume`, `-NoGraph`.

Every run writes a **new** `agents\agent_v<algo>_<timestamp>.py` rather than
overwriting its input, so a bad run costs nothing. Afterwards the script rates
the new agent on the Elo ladder and regenerates `agents\<model>.html`, so the
knowledge graph always matches the model.

### Things already measured as worse — do not retry without new evidence

| Change | Result |
|---|---|
| `strawberry 32 / melon 6` | 0% win rate, −$10,056 |
| rescue pens for stranded animals | 62.5% vs 81% |
| `cost_per_animal_day` 4.5→3.5 | 80% over 10 matches, **56% over 16** |
| `water_window_priority` 3.0 | 0%, −$6,181 |
| `drop_stack_value` 900 | 25%, −$874 |
| `cost_per_crop_day` 1.6 | 25% |
| `capacity_util` 1.0 | 50% |
| `max_herd` 16 | 50% |

The `cost_per_animal_day` row is the cautionary one: it looked like an
improvement over 10 matches and reversed over 16. Treat anything under ~16
paired matches as noise.


---

## 5a. The improvement loop (use this instead of raw tuning)

```powershell
python tools\sprt.py                       # what a decision actually costs
python tools\improve.py --minutes 180      # propose -> test -> keep only real gains
python tools\improve.py --status           # what it has tried and decided
python tools\improve.py --resume
```

**Why this replaced `tune.py`/`cmaes.py` as the default.** Those compare
candidates on 8-16 seeds and keep whichever scored higher. At that sample size
the standard error swamps the effect being measured, so a run of
"improvements" is a random walk that ratchets on noise — `cost_per_animal_day`
measured 80% over 10 matches and 56% over 16. More search on a broken accept
rule produces more noise, not more Elo.

`improve.py` changes the accept rule, not the search: every candidate is played
in paired both-seat games against the incumbent and judged by a sequential
probability ratio test (`src/kaggriculture/measure/sprt.py`), the method computer-chess engines use
for the same problem. Only a statistically significant gain is kept. SPRT stops
as soon as the evidence is decisive, so a clearly bad candidate costs ~30 games
instead of a fixed budget.

| Option | Meaning |
|---|---|
| `--minutes` | wall-clock budget for the whole run |
| `--elo1` | smallest gain worth adopting (default 25) |
| `--max-games` | cap per candidate before "inconclusive" |
| `--base` | starting agent (default: newest by *filename*) |
| `--resume` / `--status` | checkpointed after every game |

**Pick `--elo1` for your compute budget, not your ambition.** Run
`python tools\sprt.py` for the measured table. At elo1=25 a +100 Elo change
decides in ~165 games and a +15 Elo change never decides at all inside 400 —
which is the honest answer, because a 15-Elo change will not move a placing.

Three proposal families: `perturb` (small subsets of numeric knobs),
`structural` (discrete design flips — `assign_mode`, `ensemble_k`), and
`recombine` (blend with an earlier accepted point). Structural is the family
that actually moves a plateaued agent.

### Structural changes measured so far

| Change | Result |
|---|---|
| `assign_mode` greedy -> optimal | **ACCEPTED** 74W-29L over 103 paired games, +163 Elo [+94, +247] |
| `ensemble_k` 0 -> 12 | preliminary 11W-5L over 16 games, +137 Elo [-28, +412] — **not decided**, interval still includes zero |

**Optimal assignment.** Task allocation was greedy: score every (unit, task)
pair, sort, take what is free. That is at best a 1/2-approximation. Two hands,
two jobs: A scores 10 on job 1 and 9 on job 2, B scores 9.5 on job 1 and 0 on
job 2 — greedy takes A->1 and strands B for a total of 10, optimal is 18.5. The
agent now solves the real max-weight matching (Hungarian, pure Python, verified
against brute force on 400 random instances) at 0.6 ms per turn.

### The ensemble

```
measured on agent_v4, 240-turn match, against a 1000 ms actTimeout

  ensemble_k    mean      p95      headroom at p95
       0       0.29 ms   0.47 ms      2126x
       4       1.32 ms   2.21 ms       453x
      12       3.32 ms   5.74 ms       174x
```

The agent uses well under 1% of its per-turn budget, so an ensemble is close to
free. `ensemble_k` committee members each propose an op per unit and the ops are
voted on; below `ensemble_consensus` support the incumbent's op stands, so a
diffuse split cannot outvote the tuned policy.

**Market orders are deliberately not voted.** They carry budget invariants —
feed runway, working reserve, order priority — and mixing two members' orders
can spend the same cash twice. Averaging structured plans is how ensembles
produce illegal states.

`improve.py` now proposes `ensemble_k` as a structural flip, so the loop will
settle the open question on its own given enough games.

---

## 6. The whole pipeline

```powershell
.\scripts\pipeline.ps1                                        # full loop, no upload
.\scripts\pipeline.ps1 -Algo cmaes -Minutes 45 -PerDay 120 -Days 5
.\scripts\pipeline.ps1 -SkipDownload -Algo tune -Submit       # validate, do not upload
.\scripts\pipeline.ps1 -Submit -Live -Message "cma gen12"     # can upload
```

Stages, in order:

1. **download** — fresh ladder replays + your own games *(optional: warns and continues)*
2. **analyse** — attribution, EDA report, diff vs the top-20 *(optional)*
3. **train** — the chosen algorithm, producing a new agent
4. **simulate** — win rate, official episode, Elo
5. **test** — legality, latency, beats-the-built-ins
6. **submit** — only if you pass `-Submit`; only uploads if you *also* pass `-Live`

A required stage that fails stops the run rather than feeding a broken artefact
to the next one. The run ends with a summary table of stage / result / duration.

`-Native` runs `tools\pipeline.py` instead — the Python orchestration with its
own `refine` step and interactive menu. It never auto-submits either.

---

## 7. Submitting

```powershell
.\scripts\submit.ps1                                     # DRY RUN: validates, uploads nothing
.\scripts\submit.ps1 -Live                               # can upload
.\scripts\submit.ps1 -Agent agents\agent_v3.py -Live -Message "cma gen12"
```

**Two independent gates.** Nothing uploads unless you pass `-Live` *and* type
`SUBMIT` at the prompt inside `tools\submit.py`. Both are deliberate: a
submission is irreversible, you get 5 per day, and only the latest 2 stay
active.

What validation covers, in Kaggle's own order:

1. **rank** — every agent in `agents\` by win rate, unless you named one
2. **size** — under 100 MiB
3. **latency** — per-turn time against the real 1-second `actTimeout`
4. **self-play** — the Validation Episode Kaggle runs on upload, run locally
   first so a failure costs seconds instead of one of your five daily slots
5. **approval** — stats printed, you type `SUBMIT`

After a submission, the resulting games appear in your episode list within about
an hour — `.\scripts\download.ps1` pulls them back in for the next iteration.

---

## 7a. Choosing which bot to submit

```powershell
python tools\dashboard.py                 # -> docs\elo-dashboard.html
.\scripts\submit.ps1 -List                # the same table, in the terminal
```

The dashboard lists every bot with its Elo, games played, win rate and a
confidence bound, and states which one to submit — or refuses to name one and
says why.

**The rules it applies, in order:**

1. **Contract tests first.** `.\scripts\test.ps1 -Full`. An agent that emits one
   illegal action or overruns the 1-second `actTimeout` loses on Kaggle without
   telling you why. No rating means anything until this passes.
2. **Rank by Elo, never by bank.** The leaderboard is a skill rating: a 10-coin
   win and a 10,000-coin win score identically. Coin margin is a tie-break only.
3. **Ignore any rating under 16 games.** The confidence column is `400/sqrt(n)`
   in rating points — 2 games is +/-283, 32 games is +/-71. Two bots whose
   intervals overlap are *tied*, however different the point ratings look.
4. **Confirm head-to-head against whatever is currently on Kaggle.**
   `python tools\evaluate.py <new> --vs <current> -n 16`. Ladder Elo is
   transitive and can hide a bad matchup; the direct result cannot. Want >=55%
   over >=16 paired matches.
5. **Distrust anything measured on fewer than 16 matches.** Worked example from
   this project: `cost_per_animal_day` 4.5->3.5 looked like 80% over 10 matches
   and was 56% over 16.
6. **Then spend the slot.** Five submissions a day, only the latest two active —
   an unsure submission costs an active slot as well as a daily one.

### Submitting a specific bot

```powershell
.\scripts\submit.ps1 -List                              # what am I choosing between
.\scripts\submit.ps1 -Agent v2_tuned                    # dry run, named bot
.\scripts\submit.ps1 -Agent agent_v3_20260804_160019 -Live -Message "cma gen12"
.\scripts\submit.ps1 -Best                              # top-rated bot, dry run
.\scripts\submit.ps1 -Best -Live                        # top-rated bot, can upload
```

`-Agent` accepts a bare name (`v2_tuned`), a file name (`v2_tuned.py`), a
repo-relative path or an absolute path. An unmatched name fails with the list of
what is available rather than a bare "not found".

`-Best` reads the ladder and **refuses** when the leader's interval overlaps a
rival — that is a tie, not a lead. It prints the reason; `-Force` overrides once
you have read it.

| Option | Meaning |
|---|---|
| `-Agent <name>` | Submit exactly this bot |
| `-Best` | Take the top-rated bot off the Elo ladder |
| `-Force` | Let `-Best` proceed on an unsettled lead |
| `-List` | Print the ladder and exit |
| `-Live` | Actually upload (still requires typing `SUBMIT`) |
| `-Seeds` | Validation seeds per pairing (default 6) |
| `-Message` | Submission description shown on Kaggle |

### Building up the ladder

```powershell
python tools\elo.py --rounds 4                    # full round-robin, both seats
python tools\elo.py --only agents\<new>.py        # rate one newcomer, ~12 matches
python tools\elo.py --show                        # table only, no matches
```

Ratings accumulate in `.local\elo\ladder.json` across runs, so each iteration's
agent is rated against every previous one. Elo starts at 600, matching Kaggle's
initial rating for a new submission — but it is relative to *this field only* and
is not the Kaggle number.

---

## 7b. The control panel

```powershell
.\dashboard.bat                      # double-click works too
.\scripts\dashboard.ps1              # same thing, with -Port / -Stop
.\scripts\dashboard.ps1 -AllowSubmit # arms the live submit button
.\scripts\dashboard.ps1 -Stop        # kill whatever is holding the port
python dashboard\serve.py             # the raw server, no wrapper
```


```powershell
python dashboard\serve.py                  # -> http://127.0.0.1:8787
python dashboard\serve.py --allow-submit   # enables the live submit button
```

Standard library only -- `http.server` and threads, no pip install standing
between you and your own dashboard. It binds to **127.0.0.1 only** and is not
configurable to `0.0.0.0`: the process runs arbitrary commands by design and
must not be reachable from the network.

Every button shells out to the same tool the CLI uses, so there is no second
implementation to drift out of sync.

| Tab | What it does |
|---|---|
| Models | Elo **and public score** side by side, games, confidence, worst-turn ms, eval count; per-model Evaluate / Elo / Losses / Params / Graph / **Download** / Validate / Submit |
| Data | episode counts, last fetch, download with day/per-day/jobs, preflight and both diagnostics |
| Opponents | **mine fresh top-30 routes**, build tapes and clones, then play the whole set **or one named topper** |
| Losses | post-mortems ranked by how much money each cause is worth |
| Configs | the model factory: build any config into an agent, with a smoke test |
| Runs | every job ever launched, with live streaming logs and a stop button |

**Submit keeps both gates.** The server must be started with `--allow-submit`
*and* the browser must send the literal string `SUBMIT`; without the flag the
button is disabled and the endpoint returns 403. Five submissions a day and only
the latest two active make an accidental click expensive.

### Downloading the artefact

```
GET /api/artifact?model=v14_market              -> main.py
GET /api/artifact?model=v14_market&format=tar   -> submission.tar.gz
```

The **Download** button on each model row. `/api/file` deliberately allowlists
only `.html/.md/.json/.csv`, which meant the one file this whole project exists
to produce could not be fetched at all. This is a separate endpoint rather than
a wider allowlist: it resolves a *registered model name*, never a
caller-supplied path, so what can be downloaded widens without widening where it
can be read from. `format=tar` produces the `main.py`-at-root tarball
`AGENTS.md` documents for multi-file submissions.

### What the Models table now shows, and why

Local Elo next to the **public score** is the comparison that says whether local
measurement is tracking the ladder at all. It was invisible before: the registry
stored Elo only, and the public score lived in the Kaggle web UI. Refresh with:

```bash
python -m kaggriculture.data.registry --sync-kaggle    # pull public scores onto the models
python -m kaggriculture.pipeline.submit --measure-all      # latency for every agent, no upload
```

`--measure-all` runs the same self-play validation episode as a real submission
but uploads nothing, which is how every model gets a latency figure without
spending one of the day's five slots. Treat a single latency reading with
suspicion — the same agent measured 234 ms and 125 ms on this box depending on
load.

### Model factory

```powershell
python tools\build_agent.py --list
python tools\build_agent.py configs\v5_ensemble.json --smoke
python tools\build_agent.py "configs\*.json" --smoke
python tools\build_agent.py --new v6 --base "agents\agent_v4_*.py" --set ensemble_k=16
python tools\build_agent.py --params            # every knob the source declares
```

A config names a base (a glob, so it does not go stale), overrides, an optional
`asset_bias` copied from another agent, and tags. **Unknown knob names are a hard
error**, with a did-you-mean: a typo otherwise builds cleanly, runs cleanly, and
silently tests a null edit.

### Routes -- mining the current top-30

```powershell
python tools\routes.py --mine --top 30 --per-team 6 --max-gb 8 --jobs 8
python tools\routes.py --report
python tools\routes.py --list
python tools\routes.py --clean-stage      # after an interrupted run
```

**A route is a perishable asset.** The same agent with the same safety layers
measures 19/46 on last week's route and 40-41/46 on a current one; the top three
current families sit more than 1,200 channel moments from the stale one. So this
is a thing you run on a cadence, not once.

The daily `manifest.csv` carries no team, so team identity has to be read from
inside the replay (`info.TeamNames`). That means downloading, which is why this
streams: fetch one ~27 MB replay, extract the 719-turn route, **delete it**,
move on. Disk stays flat against a 21 GB/day archive. Ranking is by `min_score`
rather than `avg_score` — a high average can be one strong player farming a weak
one, and half of that game is a route you do not want.

Two disciplines are baked in. **One medoid route per submission**, so a team
that replays heavily cannot dominate the panel by sheer count. And **disjoint
windows**: the newest 6 per team are an outer holdout never opened during
selection, the next 3 validate, the rest are fit-only donors.

Provenance: this reads Kaggle's own published episode archive, which is what
the whole field mines and credits. Payloads decoded out of competitors'
*notebooks* stay data-only under `SECURITY.md` — never imported, never run,
never shipped.

### Opponents

```powershell
python tools\opponents.py --build-tapes
python tools\opponents.py --build-clones --top 5
python tools\opponents.py --play "agents\agent_v4_*.py" --n 4
python tools\opponents.py --play agents\v14_market.py --vs tape_90041552_s0.py
```

Tapes replay a topper's recorded 720 actions with a repair layer for states that
have drifted -- the architecture of the 2600-Elo public agent. They are faithful
but do not react, so **beating a tape is necessary, not sufficient**. Clones fit
our own heuristic's spatial and valuation knobs to reproduce their recorded
decisions, giving an opponent that responds to new states but can only express
strategies our policy can represent.

`--vs` picks specific opponents; without it you play the whole set. `play()`
always accepted an opponent list, but the CLI never exposed one, so "simulate
against this particular leaderboard topper" was not expressible.

Tape quality is bounded by replay quality: build them after a real download —
or better, after `tools\routes.py --mine`, which targets named teams rather than
whatever happened to rank highly that day.

### Loss analysis

```powershell
python tools\loss_analysis.py --agent "agents\agent_v4_*.py" --vs agents\v2_tuned.py --scan 8
python tools\loss_analysis.py --replay .local\episodes\mine\12345.json --seat 0
```

Output is ordered by how much money each cause is worth, not by when it
happened. A loss is rarely one bad turn; it is usually a structural gap -- fewer
hands, a herd that starved on day 18, produce stranded in inventories -- that a
per-turn diff buries.

### Registry

```powershell
python tools\registry.py            # models, data, runs at a glance
python tools\registry.py --json     # what the dashboard consumes
```

Single source of truth under `.local\registry\`. It reads ratings from
`.local\elo\ladder.json` rather than copying them -- a registry that keeps its
own copy of a number another tool owns is a registry that goes stale.

---

## 7c. Ensembles, the arbiter, and the forecaster

### Selecting an ensemble from the ladder

```powershell
python tools\select.py --show                    # who would be picked, and why
python tools\select.py --build --k 4
python tools\select.py --build --spread-guard    # refuse near-duplicate members
```

Picks the top of the Elo ladder, but only models whose rating is
*distinguishable*: a 750 on 4 games and a 720 on 40 is one measured number and
one rumour. Members whose interval sits entirely under the leader's are dropped.
The chosen parameter sets are written into the agent as an explicit
`ENSEMBLE_MEMBERS` committee, so the vote is between policies that each earned
their place rather than between jitter around one of them.

Latency is checked before the result is accepted — an ensemble that thinks too
long is a loss, not a stronger agent. The current 3-member selection measured
0.98 ms mean, 1.71 ms p95 (586x headroom).

**`--spread-guard` earns its keep.** Run it today and it rejects both members:
our top three models have *identical* voting knobs and differ only in
`assign_mode` and their learned `ASSET_BIAS`. A committee of clones costs time
and adds no information. Diversity has to be manufactured — different training
runs, different seeds, different objectives — not assumed because the files have
different names.

### Learning the arbiter (RL)

```powershell
python tools\train_arbiter.py --base "agents\v5_ensemble_2*.py" --minutes 60
python tools\train_arbiter.py --status
python tools\train_arbiter.py --resume
```

Counting votes assumes every member is equally trustworthy in every state, which
is false: a policy that over-invests early is right on day 2 and wrong on day 27.
`MEMBER_WEIGHTS` is a small table — 3 day bands x 3 cash bands x one weight per
member — learned by **cross-entropy method** over paired self-play.

CEM rather than a gradient method for reasons specific to this problem: the
objective is a noisy non-differentiable Monte-Carlo win rate, the table is small
enough that CEM is at its best, and every generation's elite set is inspectable
while the run is happening.

This is also the answer to "why not just use PPO on `ml_rl.py`". That trainer
credits one scalar episode return to 128 state buckets; the signal is thin at
the source, and no algorithm fixes a signal problem. The arbiter is a *smaller,
better-shaped* decision, exercised 720 times an episode instead of once.

**Keep `ensemble_k` small when training the arbiter.** k=24 means 225 weights,
which needs far more games than a sane budget allows. k=4–8 is the useful range.
The CEM winner is put through a full SPRT before it is allowed to become an
agent — a lucky generation cannot promote itself.

### The forecaster and its parity number

```powershell
python tools\fastsim.py --demo
python tools\parity.py --matches 4 --horizon 72
```

`fastsim.py` is a value function, **not** a simulator, and that is deliberate.
Re-implementing the game is how you end up tuning an agent for a game nobody is
playing: the copy drifts, silently, and every number afterwards is about the
copy. We vendor the official interpreter and it stays the ground truth. What
search and RL actually need is narrower and safely measurable: how much is this
position worth?

`parity.py` plays real episodes and reports the error. **Measured today:**

```
median absolute error : 141%
rank accuracy         : 75%
verdict: biased but usable for ranking. Use it to compare plans, never to
         predict a bank.
```

That is the honest state: it orders positions correctly three times in four,
which is what a search needs, and its dollar figures are badly biased, which
means they must never be reported as a prediction. Re-run parity after any
change to the crop or animal tables.

---

## 7ca. Adaptive play

```powershell
python tools\adaptive.py --list
python tools\adaptive.py --agent "agents\agent_v4_*.py" --mode both
python tools\adaptive.py --agent "agents\agent_v4_*.py" --mode risk --ab
python tools\adaptive.py --off "agents\agent_vadapt_*.py"
```

Also on the dashboard: **Models -> Adaptive play**, which writes a new agent
rather than editing one in place, so switching a mode never silently changes a
model you already rated.

| Mode | What changes |
|---|---|
| `off` | plays the same policy whatever the score (default) |
| `bandit` | Exp3 over the committee, reweighted from an in-game wealth signal |
| `risk` | score-aware: presses when behind late, protects a lead when ahead |
| `both` | both |

**Why Exp3 and not UCB.** The reward here is non-stationary by construction --
what pays on day 3 is wrong on day 27 -- and UCB's confidence bounds assume a
fixed arm quality. Exp3 is the standard choice for a drifting or adversarial
reward. The weight floor (`bandit_floor`) keeps every member exploring: a bandit
that commits fully cannot notice the game has moved.

Credit assignment matters as much as the algorithm. Members are credited when
the op they proposed is the op that actually got played, not when their vote
wins -- blaming a member for being outvoted punishes it for a decision that was
not its own. Weights update once per in-game day, because a single turn's wealth
change is mostly market noise.

**Why risk mode is not just "play well".** On a skill ladder a narrow loss and a
wide loss score identically, so protecting a losing position is strictly worse
than gambling out of it. Behind late, it releases reserves and extends the
planting horizon; ahead late, it holds cash and stops starting things that
mature after the horizon.

**Measured cost** (240-turn match, 1000 ms actTimeout):

```
mode     mean      p95      note
off      2.27 ms   3.91 ms  committee of 8
bandit   2.29 ms   3.90 ms  Exp3 adds ~0.02 ms
risk     0.31 ms   0.54 ms  no committee needed
both     2.30 ms   3.90 ms
```

Adaptation is free at this scale. Whether it *helps* is a separate question --
`--ab` runs a full SPRT against the static build it came from, because
"adaptive" is a claim like any other.

---

## 7cb. Population-Based Training, XGBoost, and Kaggle-exact runs

### PBT -- the fix for a population of clones

```powershell
python tools\pbt.py --population 6 --minutes 120
python tools\pbt.py --status
python tools\pbt.py --harvest 4
```

`tune.py`, `cmaes.py` and `improve.py` all optimise **one** policy, which is why
`select.py --spread-guard` rejects every committee member we have: every
generation is a small step from the same parent, so the survivors are
near-copies by construction. PBT runs a league of workers; the weak ones
**exploit** (copy a random strong worker, not the single best -- that is what
stops the population collapsing onto one lineage) and **explore** (perturb).

It prints a **diversity** number each generation: mean relative spread across
the voting knobs. Below 0.05 the population has collapsed and the ensemble has
nothing to work with. That number, not the win rate, is what PBT is for.

### XGBoost arbiter

```powershell
pip install xgboost
python tools\train_xgb.py --collect 30 --base "agents\v5_ensemble_*.py"
python tools\train_xgb.py --train --rounds 200
python tools\train_xgb.py --build --ab
```

Learns P(win | state, member) and uses it as the vote weight -- a learned
generalisation of the bucketed `MEMBER_WEIGHTS` table. Trees interpolate between
states; a table has to observe each band separately, and every training row
costs part of a 7-second match.

**XGBoost is a training-time dependency only.** The model is exported to nested
Python lists and injected into `XGB_MODEL`, so the submission stays one file.
The exporter is checked against `booster.predict(output_margin=True)` on every
build and agreement is ~1e-6.

One detail that is easy to get wrong and silent when you do: **XGBoost splits in
float32.** Parsing the JSON dump into float64 and comparing there sends roughly
one comparison in 700 down the wrong branch -- measured, not guessed: 7 of 200
rows disagreed before the fix, 0 after. Thresholds are stored as float32 and the
feature vector is cast before comparing.

### Kaggle-exact episodes

```powershell
python tools\kaggle_env.py --show      # the competition's real configuration
python tools\kaggle_env.py --audit     # every call site, and how it differs
python tools\kaggle_env.py --verify --agent "agents\agent_v4_*.py"
```

Kaggle runs `episodeSteps 720, actTimeout 1, runTimeout 1200`. Most harnesses
here raise `actTimeout` to 60 so a debug run is not killed mid-episode -- which
is fine for bulk search and means **those runs cannot detect an agent that would
time out on Kaggle**. An overrun is not slow there, it is *wrong*: the engine
substitutes a default action and you lose games with nothing in the logs.

`--audit` lists every place this repo builds an episode and what it changed:
currently **3 exact, 21 relaxed**. `strict_env()` overrides nothing but the seed,
and `--verify` plays a real episode at the real timeout. Verified so far: v4,
the 24-member ensemble and the adaptive build all finish DONE at `actTimeout 1`.

---

## 7d. The system, end to end

```
configs/*.json          declarative model definitions
   |  src/kaggriculture/agentbuild/build_agent.py            (validates knob names, smoke-tests)
   v
agents/*.py             every generation, each with a knowledge graph
   |  src/kaggriculture/measure/elo.py + src/kaggriculture/pipeline/improve.py (SPRT-gated; only real gains kept)
   v
.local/elo/ladder.json  the running scoreboard
   |  src/kaggriculture/data/registry.py               (single source of truth)
   v
dashboard/serve.py      run anything, see everything, submit behind two gates
```

Supporting: `src/kaggriculture/measure/opponents.py` (topper tapes and clones),
`src/kaggriculture/measure/loss_analysis.py` (why a game was lost), `src/kaggriculture/train/select.py` (best-of
ensemble), `src/kaggriculture/train/train_arbiter.py` (learned vote weights),
`src/kaggriculture/engine/fastsim.py` + `src/kaggriculture/engine/parity.py` (value function and its measured error).

```powershell
python -m pytest tests -q          # both suites
python tests\test_agents.py        # legality, latency, beats-the-built-ins
python tests\test_system.py        # registry, factory, opponents, dashboard, SPRT
```

`test_system.py` covers what we punish ourselves with rather than what Kaggle
punishes: a registry that goes stale, a config typo that builds a null edit, a
tape opponent that plays an illegal move, an SPRT that accepts noise, a
dashboard endpoint that runs something it should have refused.

### What was taken from the reference repo, and what was not

[deepeshumrao/kaggriculture-agent](https://github.com/deepeshumrao/kaggriculture-agent)
is a contract-first build of the same competition.

**Adopted.** Its parity discipline — prove the local model matches the official
interpreter rather than assuming it — is why `parity.py` exists and publishes a
number instead of a claim. Its habit of isolating unconfirmed mechanics in one
place is why `fastsim.py` imports the agent's own `CROPS`/`ANIMALS` tables
instead of restating them.

**Not adopted.** Its `local_env.py` as ground truth. A hand-written simulator
that silently diverges from the real rules invalidates every measurement taken
against it; we run the vendored official interpreter instead and pay the
wall-clock cost. Also not adopted: its single-file-submission parity test — our
agents *are* single files by construction, so there are no two copies to drift.

---

## 7e. The ladder gap (read this before tuning anything else)

Measured 2026-08-05 from 238 winning top-20 ladder games against our own v4.

| | ladder winners | our v4 | |
|---|---|---|---|
| final bank (median) | **115,070** | 68,078 | we earn 59% of what they do |
| wheat sold per wheat tile-day | **5.36** | 1.19 | they sell 4.5x more wheat than they grow |
| fertilizer sold | 270 | **0** | an income stream we do not touch |
| first animal day | **0** | 11 | they buy livestock on day zero |
| hands held late | **11.85** | 7.55 | we shed labour late, they do not |
| tile-days STRAWBERRY | 648 | 399 | |
| tile-days WHEAT | 125 | **619** | our crop mix is inverted |
| crop tiles worked | 54 | **75** | we work more land for less money |

**The obvious fix was tested and it is wrong.** `configs/v6_ladder.json` copies
their end-state: 13 hands, herd 16, strawberry as filler, land on day 1, cheaper
animal buffer. On the same seed it banked **51,382 against v4's 68,078**, and
`peak_hands` *fell* to 7 despite `hands_max` rising to 13.

The mechanism is cash. Switching the filler crop off wheat removed the early
income that funds hiring, so the agent could not afford the hands the config
asked for, and everything downstream shrank with it. Their twelve hands and
day-zero livestock are a **consequence of an economic engine, not a set of
targets**. Asserting the targets without the engine makes things worse.

**Where the engine probably is.** 5.36 units of wheat sold per tile-day of wheat
grown cannot come from farming it -- our own yield is 1.19 and the tables do not
allow four times that. The arithmetic says they are *buying wheat and reselling
it*, which fits the measured 32% price swings the market can produce. They also
sell 270 fertilizer a game; we sell none.

**Consequence for the ML work.** RL, XGBoost arbiters and ensembles all optimise
*which of our policies to run*. None of them can invent an income stream the
policy family does not contain. Tuning harder here refines a farm that earns
half as much as the ones it is being scored against. Build the missing revenue
first, then let the learned machinery -- which is finished and waiting -- tune it.

---

## 7f. The fertilizer loop -- and why the dose decided it

`docs/history/environment-rules.md` line 64: a watered day inside the yield window adds
**+1 unit, or +2 if the tile is fertilized**. Fertilizing *doubles* yield, and
fertilizer is produced free by livestock via `COLLECT_FERTILIZER`. That is the
compounding loop behind the ladder gap: early animals -> free fertilizer ->
double yields across the whole farm -> the cash for more hands and land.

It also explains the single strangest number in the gap analysis. Ladder winners
sell **5.36 wheat per wheat tile-day**; our v4 sells 1.19. No amount of trading
explains that, but a doubled yield does.

### The dose, not the direction

| build | change | result vs v4 |
|---|---|---|
| `v7_fertilizer` | fert_weight **5.0**, collect 2.0 | **2W-18L** |
| animals only | earlier animals, weights unchanged | 40% -- neutral |
| **`v8_fertilizer`** | fert_weight **3.5**, collect 1.0 | **31W-19L (62%), +85 Elo [-10, +195]** |

Pushing fertilizer harder made it dramatically *worse*. At 5.0 the crew spends
its unit-turns fertilizing tiles instead of harvesting them, and a doubled yield
you never collect is worth nothing. The optimum sits between v4's 2.5 and 5.0.

The animals-only row is the one that stops a wrong conclusion: buying livestock
earlier, on its own, does nothing. The gain is the *fertilizer*, not the herd --
the herd is only how you get it.

### An earlier trap worth remembering

`v7` banked **$81,752 against v2 where v4 banked $68,078** -- a 20% improvement
on the metric everyone reaches for first. Head-to-head against v4 it went
2W-18L. **Banking more money and winning more games are different questions**,
and only the second one is scored. Any candidate judged on mean bank against a
weak opponent will mislead you.

### Also measured: round-trip trading does not work

Buying 94 wheat moved the price from 25 to 35 (+40%) and selling it back pushed
it to 27. A pure round trip on a 3,000 stake returned **+30 over 60 turns**. Our
own market impact eats the spread, so "buy low, sell high" is not the missing
income stream. The same measurement matters in the other direction though:
dumping a large harvest crashes the price we are selling into, so sale *timing*
is worth real money.

---

## 8. Scheduling

```powershell
.\scripts\download.ps1 -Schedule                 # hourly, starting in 2 minutes
.\scripts\download.ps1 -Schedule -Remove         # uninstall
```

Each run resumes from `data\fetch_state.json`, so the hourly task only fetches
what is new. Log: `.local\logs\fetch.log`.

Kaggle publishes the top-episode dataset once a day, so most of the hourly value
is your own agents' new games. The daily pull is skipped automatically until a
genuinely new date appears.

---

## 9. Troubleshooting

**`kaggle` is not recognised / `No module named kaggle.__main__`**
`python -m kaggle` can never work — the package ships no `__main__.py`. The
resolver in `tools\episodes.py` tries, in order: PATH → the interpreter's
`Scripts` dir → the **per-user** `Scripts` dir (`%APPDATA%\Python\Python313\Scripts`,
which is where `pip install` puts it when it says "Defaulting to user
installation") → `python -m kaggle.cli` → the `kaggle` console-script entry
point via `importlib.metadata`. If all five fail:

```powershell
python -c "import kaggle;print(kaggle.__file__)"    # is it this interpreter's?
python -m kaggle.cli --version
```

**"0 team names resolved"**
Not fatal. Run `.\scripts\download.ps1 -Diagnose` — it prints the raw CLI output
for each invocation tried and saves it to `.local\diag\leaderboard_raw.txt`.
Meanwhile `-AnyTop` gets you effectively the same episodes, because the daily
dataset is already ordered by agent rating.

**"cannot list your submissions" / `TypeError: ... unexpected keyword argument 'page_number'`**
A bug in the `kaggle` package itself, not in this project. In 1.8.3,
`cli.py` calls `KaggleApi.competition_submissions(page_number=...)` and that
build's API method does not take `page_number`, so
`kaggle competitions submissions` dies every time. Only the CLI wrapper is
wrong — the data is fine — so the project now calls the API in-process instead,
passing only the keywords the installed build actually declares. Ladder
downloads were never affected; only your own games were.

Check which path is working:

```powershell
python download_data.py --submissions
```

If both the CLI and the in-process API fail, pin a build without the bug:

```powershell
python -m pip install "kaggle==1.7.4.5"
```

**The run looks frozen**
It should not any more: every line is timestamped and flushed, and downloads
report per-episode progress with an ETA. If a stage really is silent, add
`-Verbose2` (download) to echo each CLI call, or check the log under
`.local\logs\`.

**"exited 1" with no obvious cause**
The full output is in the log file named at the end of the run.

**Nothing gets written to my disk**
Check you are running from `D:\codebase\kaggriculture`. Every script resolves
the project root from its own location, so a wrong working directory should not
matter — but a second copy of the repo would.

---

## 10. File map

```
scripts\           PowerShell entry points (this runbook)
  _lib.ps1         shared: python resolution, logging, timed stage runner
  download.ps1     data
  test.ps1         correctness
  simulate.ps1     evaluation
  train.ps1        search + ML
  pipeline.ps1     all of it
  submit.ps1       validation and upload

tools\             the Python that does the work
agents\            every generation, plus its <model>.html knowledge graph
data\              episodes.csv, fetch_state.json, fetch_log.csv
docs\              strategy, environment rules, EDA, this runbook
.local\            scratch: logs, replays, Elo ladder, diagnostics, checkpoints
```

`.local\` is disposable. Everything else is the project.

---

## 7g. Two submission paths, and the description

**Path A — bare `main.py`.** Fastest, no write-up.

```powershell
python tools\submit.py --agent agents\v17_route.py --dry-run   # validate only
python tools\submit.py --agent agents\v17_route.py             # validate, then ask
```

**Path B — a notebook that explains itself.** Slower, and what the top of this
board does.

```powershell
python tools\build_notebook.py --agent agents\v17_route.py            # write it
python tools\build_notebook.py --agent agents\v17_route.py --private  # keep it unlisted
python tools\build_notebook.py --agent agents\v17_route.py --push     # upload the kernel
kaggle competitions submit kaggriculture -k <user>/<slug> -v <n> -m "..."
```

`--private` publishes the kernel unlisted — the agent still runs and still
scores, you simply do not share the write-up. Public is the default because
sharing is the point of the format; make it private when the work is not ready
to be cited.

### The description writes itself

Kaggle shows the description next to the score forever, and it is the only
context you get when you come back in a week to four rows that all say
`main.py`. So `src/kaggriculture/pipeline/submit.py` assembles it from the registry rather than
asking you to type it — architecture, route provenance (team, rank at capture,
episode, seat, window), every measured result, local Elo, the latency the
validation episode just recorded, and the notebook link if you pass one.

```powershell
python tools\submit.py --agent agents\v17_route.py --describe
python tools\submit.py --agent agents\v17_route.py --describe --notebook user/slug
```

It is printed, wrapped, immediately before the confirmation prompt, so you read
exactly what will be attached before you type `SUBMIT`. `--message` still
overrides it entirely.

Or the **Notebook** button on each model row in the dashboard.

`AGENTS.md` documents the notebook path (`submit -k <user>/<slug> -v <n>`), and
it is worth using: a bare `main.py` tells nobody anything, including us in a
week. The mechanisms that actually moved our score arrived through other
people's write-ups — v22's price-impact ranking reached us because its author
published the *ablation that isolated it*, and we reproduced their effect size
to within a few hundred dollars because they published the number and not only
the code.

The generated notebook carries, in order:

1. the headline result and how it was measured, **including what it lost**;
2. a machine-readable contribution card — upstream sources, route provenance,
   what is new, what was tried and rejected — so a fork can cite precisely;
3. EDA that *motivates* the mechanism rather than illustrating it afterwards:
   the official price curve per product, the freshness decay we measured, and
   the tournament scatter showing recorded bank does not predict head-to-head
   strength;
4. the exact agent bytes, reconstructed and **verified by SHA-256**, compiled,
   and tarred as `main.py` at the root;
5. credit for every borrowed mechanism, and a plain caveat that counterfactual
   replay results are not a leaderboard score.

The payload round-trips byte-for-byte — the builder checks that the embedded
blob decompresses to the same SHA-256 as the file on disk. **Pushing the kernel
and entering it in the competition are separate steps**, and the second is never
automated: a submission is irreversible and only the latest two stay active.
