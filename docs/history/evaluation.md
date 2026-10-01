# How Kaggriculture is evaluated, and how to reproduce it

## 1. How Kaggle actually scores you

Kaggriculture is a **simulation competition**, so there is no held-out test set
and no "model" to download. Scoring works like this:

1. You submit an agent. Kaggle runs it in the *same* `kaggle-environments`
   package you can `pip install` — `kaggle_environments/envs/kaggriculture/`.
   There is no private variant of the rules.
2. Your agent is repeatedly paired against **other people's live submissions**
   in 720-turn episodes.
3. Each episode's winner is whoever has more money at the end. Results feed a
   **skill rating** (a TrueSkill-style number), and *that* is the leaderboard
   score — not your bank balance.

As of August 2026 the top of the leaderboard sits around **2921**, the gold
cutoff around **2790**, and the official starter notebook scores **415**. A new
submission starts near 600 and climbs or falls as it plays.

**The practical consequence:** absolute bank size is not the objective —
*beating the opponent* is. An agent that banks $60k while its opponent banks
$65k ranks below one that banks $30k against an opponent's $25k. Both
`src/kaggriculture/train/tune.py` and `src/kaggriculture/measure/evaluate.py` therefore optimise and report
**margin**, not just bank.

## 2. Reproducing the official evaluation locally

Because the scoring environment is the public package, local evaluation is
*exact* — same interpreter, same rules, same market curves. The only thing you
cannot reproduce locally is the pool of opponents.

```bash
# stock competition configuration: 720 steps, real 1-second actTimeout,
# default market curves, no overrides -- the same code path Kaggle uses
python -m kaggriculture.engine.official_eval --agent agents/v2_tuned.py

# with replay artefacts you can open in a browser
python -m kaggriculture.engine.official_eval --agent agents/v2_tuned.py --replay
```

This is deliberately different from `src/kaggriculture/measure/evaluate.py`, which raises
`actTimeout` so dozens of matches can be batched in parallel. **Always run
`official_eval.py` before submitting** — it is the run that would catch a
turn-time forfeit.

Under the hood it is the same three lines as the official getting-started
notebook:

```py
from kaggle_environments import make
env = make("kaggriculture", debug=True)      # no configuration overrides
env.run(["main.py", "starter"])
print([(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
```

### The three local opponent tiers

| Tier | What it tells you |
|---|---|
| `pass`, `random`, `starter` (built-in) | legality and a floor. The starter scores 415 on the leaderboard, so beating it means very little |
| `agents/v0_baseline.py`, `agents/v1_heuristic.py` | real signal — a competent opponent that competes for the same market |
| self-play vs a frozen copy | what `src/kaggriculture/train/tune.py` optimises; the closest proxy for a strong opponent |

### Turn-time budget

`actTimeout` is **1 second per turn**, with 60 seconds of overage for the whole
episode. Kaggle's runners are slower than a dev box. Current measurements:

```
mean  ~6 ms/turn      worst  ~25 ms/turn
```

`tests/test_agents.py` fails the build if the worst turn exceeds 500 ms.

## 3. Setting up your Kaggle API key

You do this on your own machine — **do not paste the key into this chat or into
any file in the repo.** Nothing in this project needs the key except the final
`kaggle competitions submit` call, which you run yourself.

**Step 1 — generate the token.** Sign in, go to
<https://www.kaggle.com/settings/api>, and click **"Generate New Token"** under
*API*. Keep the value that appears / the file that downloads.

**Step 2 — install the CLI and authenticate.** Pick whichever fits:

*Easiest — browser OAuth, no key handling at all:*

```powershell
pip install kaggle
kaggle auth login          # opens a browser, done
```

*Token file (Windows):*

```powershell
mkdir "$env:USERPROFILE\.kaggle" -Force
# paste the token string into this file and save:
notepad "$env:USERPROFILE\.kaggle\access_token"
```

*Token file (macOS / Linux):*

```bash
mkdir -p ~/.kaggle && nano ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

*Environment variable (good for CI):*

```powershell
setx KAGGLE_API_TOKEN "xxxxxxxxxxxxxx"
```

If your CLI version predates token auth it will want the older
`kaggle.json` (a `{"username": ..., "key": ...}` file) in the same
`.kaggle` folder — the downloaded file goes there as-is.

**Step 3 — verify, and accept the rules.** Rules acceptance is browser-only;
the CLI cannot do it:

```powershell
kaggle competitions list -s kaggriculture     # CLI works?
# then in a browser: https://www.kaggle.com/competitions/kaggriculture -> "Join Competition"
kaggle competitions list --group entered      # confirms you joined
```

**Step 4 — build and submit.**

```powershell
python -m kaggriculture.agentbuild.build_submission --agent agents/v2_tuned.py
kaggle competitions submit kaggriculture -f build/main.py -m "v2 tuned"
```

**Step 5 — watch it play.**

```powershell
kaggle competitions submissions kaggriculture      # note the submission id
kaggle competitions episodes <SUBMISSION_ID>       # episodes it has played
kaggle competitions replay  <EPISODE_ID> -p .local/replays
kaggle competitions logs    <EPISODE_ID> 0 -p .local/logs
kaggle competitions leaderboard kaggriculture -s
```

## 4. Closing the loop with real opponents

Once you have submitted, the episodes your agent plays against live leaderboard
agents are the only true measurement available. The useful workflow:

1. `kaggle competitions episodes <SUBMISSION_ID> -v` → CSV of results
2. download the replays of the **losses**
3. feed them through `src/kaggriculture/measure/analyze.py`-style inspection to see *how* the
   winner out-produced you — the replay JSON contains both farms' full tile
   state every turn, so an opponent's portfolio, hiring curve and sell timing
   are all visible

That is the highest-value next step after the first submission, and it is
currently not automated. See `docs/history/issues-and-improvements.md` (B1).
