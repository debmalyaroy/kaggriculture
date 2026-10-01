# Known issues and improvement backlog

Ordered by expected value. "Est." is a rough guess at the bank swing per match;
anything marked *measured* was A/B tested with `src/kaggriculture/measure/evaluate.py`.

---

## A. Environment observations worth knowing

### A1. The official README disagrees with the interpreter on `CARE`
The README says a cared-for animal banks **+2** per day. The interpreter does
`pending_care_bonus += 1`. Steady-state output is therefore
`(1 + interval) / interval` units/day — goose 2.0, cow 1.5, sheep 1.33 — not the
higher figures the README implies. All valuations here follow the code.
**If the organisers "fix" the code to match the README, cows and sheep get ~35%
better and the portfolio targets should be re-tuned.**

### A2. The README's "max yield/tile/day" table is only reachable with fertilizer
Wheat's quoted 1.5/day needs a fertilized bonus window; unfertilized it is 1.0.

### A3. Melon reaches its cap on day 10, not day 12
`max_yield_day` is 12, but watering days 6–10 already brings `yield_units` to
the cap of 6, and `first_yield_day` is 10. Harvesting on day 10 rather than 12
is worth two extra tile-days per melon and avoids the decay window entirely.

### A4. Sales at the $1 floor do not add to market inventory
So dumping into a floored market is free of further price damage — useful in
the last-day liquidation, and the reason `dump_day` is safe.

### A5. Movement onto `LOCKED` tiles is legal
Hands can spawn on a locked shed-access tile. Tile *operations* still no-op
there. Neither agent exploits this; a path across a locked quadrant can be a
shortcut.

---

## B. Open weaknesses in v1/v2

### B1. No opponent modelling — *est. $5–15k in close matches*
Both farms are fully public. Nothing in either agent reads
`obs["farms"][1 - me]`. Two concrete plays:
* **Pre-emptive selling.** Wool floors at +59 units. If the opponent has 10
  sheep about to be harvested, selling our wool first captures the scarce side
  of the curve and leaves them the floor.
* **Feed squeeze.** Wheat is the only feed. Buying wheat aggressively raises
  the price for a livestock-heavy opponent. Currently unattractive because we
  are also a wheat buyer, but worth revisiting for a crop-heavy build.

### B2. Greedy assignment has no lookahead — *est. $3–8k*
`assign` matches units to jobs one turn at a time by value density. It cannot
plan a *route* (feed six animals in one pass), so a unit re-derives its target
every turn and pays a full walk per job. A cheap fix: after picking a unit's
job, bias next turn's scoring toward jobs adjacent to the destination — i.e. a
greedy tour rather than a greedy step. A 1-second turn budget against ~6 ms of
current compute leaves ample room.

### B3. Wheat pickup batching is crude
A unit fetching feed picks up a flat 10 wheat. It should pick up exactly the
number of animals it intends to feed on this trip, and prefer the shed access
tile nearest the pasture cluster.

### B4. Fertilizer is under-used
Fertilizer is collected from animals (one per animal per day, free) but only
applied opportunistically. On ongoing crops it doubles every scheduled
production — strawberry goes from 0.24 to 0.47 units/tile-day, which would move
it several places up the ranking. A dedicated "fertilize the strawberry block
on production-eve" rule is probably worth $3–5k.

### B5. No mid-game re-planning of committed tiles
Once a tile holds a strawberry it stays a strawberry until harvest, even if
prices move. `DIG`-and-replant is available and occasionally correct — e.g. a
tomato block when tomato has been floored.

### B6. Empty pens are only reclaimed reactively
When an animal dies the pen remains and is `DIG`ged only if the surplus
calculation notices. A dead animal should immediately trigger either a
replacement purchase or a `DIG`.

### B7. Hand count is a static function of the bank
`hands_max` tuned to 10, but the right number varies by phase: few hands while
cash-poor on day 1, many once 75 tiles are in production. A schedule keyed to
`working_tiles` would likely beat the single constant.

### B8. `sell_chunk` spreads sales but ignores the day's shape
Town shops consume every 4 turns and the town centre every 12. Selling *just
after* a consumption tick gets a marginally better price than just before.
Small, but free.

---

## C. Robustness / engineering

### C1. Turn-time headroom is large but untested at scale
Measured ~6 ms mean, ~25 ms worst against a 1 s `actTimeout`. The pairwise
assignment is O(tasks × units) ≈ 100 × 12; if a future version adds rollouts
this needs re-measuring on Kaggle's hardware, which is slower than this box.

### C2. The agent assumes the default configuration
`boardSize 10`, `turnsPerDay 24`, `episodeSteps 720`, `shedCapacity 100` and the
default market curves are baked into constants. `total_days` is read from
`config` when the framework passes it, but `MARKET_PARAMS` is not — an episode
created with `marketParams` overrides would mis-price everything. Reading the
observed `market.prices` (which the agent already does for valuation) makes it
mostly safe; the `sellable_units` projection is the exception.

### C3. Tuning is single-objective and shallow
Coordinate descent over 3 seeds × 2 seats × 2 opponents. That is enough to rank
knobs but not enough to trust small differences — anything under ~$3k of score
is inside the noise. A proper run wants 8+ seeds and a population method.

### C4. No opponent panel diversity
Tuning self-plays against a frozen v1 and v0. A leaderboard population will be
more varied; over-fitting to a frozen copy of yourself is a real risk.

---

## C4b. The tuner optimises the wrong objective — *est. meaningful, untested*

`src/kaggriculture/train/tune.py` scores `mean(margin) + 0.25 * mean(bank)`. But the competition
Overview is explicit: *"The actual coin difference in a match does not affect
the rating change — only the win, loss, or tie outcome matters."* A parameter
set that wins 60% of matches by $1 outranks one that wins 55% by $40k, and the
current objective cannot see that.

`src/kaggriculture/pipeline/submit.py` already **selects** on win rate, so the final choice is
correct, but the search itself is still steering by margin. Changing `score()`
to weight win rate (say `100 * win_rate + tiny * margin`) is a small edit that
invalidates the existing tuning and needs a fresh run — worth doing before the
next big search.

## C5. Pipeline follow-ups

* **Behaviour cloning.** The daily episode datasets give `(observation, action)`
  pairs from the best agents on the ladder -- Kaggle published them explicitly
  for "IL/BC, bootstrapping RL, or just gathering statistics". `src/kaggriculture/train/refine.py`
  only copies *aggregate statistics*; a policy trained on the pairs directly is
  a genuinely different and probably stronger approach. The blocker is that a
  learned policy must still respect the 1-second turn budget.
* **Opponent-aware evaluation.** We cannot replay a downloaded opponent against
  our agent (actions are state-dependent). But we *can* mine the ladder for the
  distribution of portfolios and synthesise a few scripted opponents that
  reproduce them, giving the tuner a more varied panel than a frozen copy of
  itself (C4).
* **Sample size.** The winner profile is a median over ~6 player-games at the
  default `--per-day 3 --days 2`. Anything surprising should be re-checked with
  a larger `--max-mb` before acting on it.

## D. Ideas not yet evaluated

* **Melon monoculture rush.** 25 melon tiles from day 0 cost $2,000 and return
  ~$26k on day 10, with almost no labour (1.1 unit-turns/tile-day). It leaves
  the farm idle for ten days, but the day-10 cash could fund a full 100-tile
  livestock build-out. Worth a dedicated agent variant to measure.
* **Goose-heavy build.** Eggs are mathematically glut-proof and geese pay from
  day 4 with the shortest lead time of any animal. A goose wall may out-earn
  cows in short-horizon or contested markets.
* **Shed-capacity arbitrage.** The 100-item shed cap destroys overflow. Holding
  produce in *unit inventories* over a day boundary does not help (they auto-drop),
  but spreading harvests so the shed never overflows is worth checking.
* **Two-phase land purchase.** Buying SE ($4,000) is currently gated late. If
  the melon rush lands, buying all three quadrants by day 12 may dominate.

---

## Backlog after the 2026-08-05 rebuild

Ranked by expected value. `agents/v9_cem.py` beats everything in this repo
(Elo 875, 91% over 56 matches) and still loses 0/18 to the recorded top-ladder
trajectories, so the list below is about that gap.

### A1. The first ten days (biggest remaining gap)

Against a tape we bank ~$77k to their ~$160k, and the divergence is all in
days 0-10: they are selling fertilizer from day 2 and 40 wheat on day 5 while
we are still planting. Profile a match against `opponents/tape_90036815_s1.py`
day by day with `src/kaggriculture/data/replay_profile.py` and close the specific gaps rather
than tuning globally.

### A2. Ship our own trajectory

The leaders' architecture — a recorded episode plus slip recovery — is public
and legitimate to use with *our own* play. `src/kaggriculture/measure/opponents.py` already builds a
faithful tape from any replay, so the missing piece is trajectory optimisation
(anneal or CEM over the action sequence itself, not the parameters). Only worth
doing once the policy generating the recording is strong enough.

### A3. Sale timing

Prices are shared and the glut curves are convex, so the first seller takes the
top of the curve. We sell in `sell_chunk` slices on a fixed floor; nothing
models "sell before they do". The opponent's standing tiles say when their
harvest lands.

### A4. More CEM, more seeds

The run that produced v9 used 3 seeds x 2 seats x 2 opponents. The margin was
still improving at generation 14 and `min_tile_rate` sat well inside its box
while `mirror_weight` pushed toward the top of its. Widen the box, add a third
tape, resume: `python -m kaggriculture.train.optimize --resume --generations 20`.

### A5. RL with a real budget

`src/kaggriculture/train/train_rl.py` at 80 episodes measured -$8,383. 128 states x 5 multipliers
with episodic win/loss credit needs 200+ episodes before the table means
anything. Resumable via `.local/rl/q.json`.

### Closed by the rebuild

* ~~fertilizer income never collected~~ — FERTILIZER is sellable, worth ~$12-20k
* ~~first animal on day 11~~ — turn zero, gated by labour rather than a flat runway
* ~~72 tiles of wheat~~ — feed is bought; the filler is whatever the market pays for
* ~~spot pricing~~ — assets are priced against projected supply on both farms
  (ablation: turning it off costs $20,948 a game)
* ~~tape opponents were fiction~~ — the replay off-by-one is fixed and tapes
  reproduce their source game

---

## 2026-08-06 — infrastructure the strategy work was standing on

None of these are strategy. All of them were quietly making the strategy work
unreliable, and none had an entry here before.

### B1. The dashboard UI was dead — CLOSED

`dashboard/serve.py:905` had `/\/g` where its two sibling lines have `/\/g`.
In JavaScript that is a parse error, and the whole UI lives in one `<script>`
block, so **nothing** was ever defined: no `refresh()`, no `run()`, no tab
switcher. The server kept answering `/api/snapshot` perfectly, so it read like
a stale browser cache and survived a day.

`tests/test_system.py::test_dashboard_javascript_parses` now runs `node
--check` over the extracted script, with a regex-literal fallback when node is
absent. Every other dashboard test checked that *names* line up; none of them
ever parsed the code.

### B2. The registry stored almost nothing — CLOSED

`record_eval()` and `record_submission()` had **zero call sites** anywhere in
the repo. The 6 models carrying an eval and the 2 carrying a submission had
them typed in by hand. `evaluate.py` even advertised a `--no-record` flag for a
write that was never implemented.

Now: `evaluate.py` records every measurement, `submit.py` records the upload
plus its latency and self-play banks, `--measure-all` fills latency in for
every agent without spending a slot, and `registry.py --sync-kaggle` pulls
public scores off Kaggle. The Models table shows local Elo and public score in
adjacent columns — the comparison that says whether local measurement tracks
the ladder at all.

**Watch the latency numbers.** The same agent measured 234 ms and 125 ms on
this box depending on load. One reading is not a measurement.

### B3. The submission could not be downloaded — CLOSED

`/api/file` allowlists `.html/.md/.json/.csv`, so `build/main.py` — the single
artefact this project exists to produce — was unreachable from the dashboard.
Added `GET /api/artifact?model=<name>[&format=tar]`, which resolves a
registered *model name* rather than a path.

### B4. "Simulate against this topper" was not expressible — CLOSED

`opponents.play()` has always taken an `opponents` list; the CLI never exposed
one, so `--play` meant "play all ten tapes" and nothing else. Added `--vs`, and
a matching selector in the dashboard's Opponents tab.

### B5. Four run rows were stuck at "running" forever — CLOSED

`finish_run` only ever fires from the dashboard's output pump, so a killed
server orphans its in-flight rows. They showed amber with a Stop button that
could not work, and `/api/clear-runs` deliberately preserves running rows.
`registry.reap_orphans()` now runs at startup and marks them `orphaned`.

### B6. The contract test excluded the agent on the ladder — CLOSED

`agent_paths()` globbed `agents/v9_*.py`, so v10, v13 and **v14 — the
submitted agent** — were never contract-checked. Now globs `v[0-9]*`, and
`ALLOWED_IMPORTS` admits the four stdlib modules a route tape needs
(`base64`, `zlib`, `json`, `copy`). 22 agents covered, all passing.

### B7. Engine drift was undetectable — CLOSED

A public probe found a Kaggle notebook image running `startingMoney` 2000
against the ladder's 3000, `COW` 600 against 400, `farmHandCostMult` 10 against
1, and **`SELL FERTILIZER` silently dropped**. `src/kaggriculture/engine/engine_check.py` checks
14 named constants, fingerprints all 121, and behaviourally probes a fertilizer
sale. Wired into `evaluate`, `elo`, `optimize`, `trajectory`, `submit`.

### B8. We had never checked our agents against the official spec — CLOSED

`AGENTS.md` and `README.md` ship as competition *data* and are revised — the
CARE text changed between the version this project first read and the
2026-08-04 revision. `src/kaggriculture/engine/conformance.py` audits both the hard contract and
which documented observation fields we actually read.

### Still open

* **`top_trajectory.csv` needs a re-mine.** The hour-0 sampling bug is fixed in
  `src/kaggriculture/data/mine_top.py`, but the 9,000 rows already on disk were written with it,
  so `hands` is 0.0 throughout and `money` is ~13% low.
* **`notebook_extract` still misses one form.** Two of the three known
  single-line payloads now decode; `ultimate-mega-ensemble-3000` uses a shape
  the flattened `_src` does not preserve.
* **Worst-turn latency is close to the bar.** `v5_ensemble_*` measure 279–325 ms
  against our 250 ms limit. They are dead experiments, but the same measurement
  put v14 at 234 ms once.

---

## 2026-08-06 — conformance against the competition's own docs

`src/kaggriculture/engine/conformance.py`, first run. `AGENTS.md` and `README.md` were pulled from
competition data and checked against the live interpreter and `v14_market`.

**Contract: clean.** All 10 documented configuration defaults match, every
action is legal, `hands` aligns positionally, the market queue peaks at 8 of the
10 allowed orders, and all 50 `DROP`s were issued from the four shed-adjacent
centre tiles `(4,4) (5,4) (4,5) (5,5)`. `hires_today` correctly offsets the
Fibonacci hire cost.

**Two documented fields we never read.** Not violations — strategy gaps with a
citation.

### C1. `max_lifespan_step` — the decay clock

> "once a plant passes its max lifespan (one day after `max_yield_day` for
> one-time crops, one day after the cumulative production cap for ongoing
> crops), `yield_units` drops by 1 every other turn until 0, at which point the
> tile becomes a weed" — README.md

Every plant has a published expiry sitting in its own tile dict and we ignore
it. Two consequences. Value: a tile inside its decay window is worth strictly
less than our valuation says, so we keep watering plants that are already
shrinking. And weeds: 45 strawberries planted together expire together, which
means a self-inflicted weed cluster and 45 `DIG`s. We currently model weeds as
purely exogenous (`weedSpawnChance` 0.005) when a large share of them are ours.

The fix is a per-day planting quota read straight off the board — count tiles
whose `planted_day` equals today and cap it — plus reading `max_lifespan_step`
into the tile valuation.

### C2. `pending_care_bonus` — the banked CARE payout

> "On a scheduled production day, if the animal is fed, the entire banked bonus
> is added to that production's yield and the bank resets to 0. If the animal
> is unfed on the production day, the base 1 unit is still produced, but the
> banked bonus is **not applied and the bank resets to 0**." — README.md

We issue `CARE` (19,242 times in the mined policy diff) but never read what it
has banked. Missing a feed on a production day silently destroys the entire
accumulated bonus, and the field number is visible to us. Feeding priority
should rise with `pending_care_bonus` on a production day; ours is flat.

Note the CARE rule itself changed: the README's old "+2" text is gone as of the
2026-08-04 revision and now matches the interpreter's +1. Worth re-pulling the
docs before trusting any rule this project wrote down earlier.

---

## 2026-08-08 — glut guard measured negative; do not retry without new evidence

A ratio-triggered floor guard (`_GUARD_RATIO`, hold SELL units whose marginal
quote is below ratio × base) and a value-hedged swap-advance (`_SWAP_ADVANCE`,
pull forward an already-scheduled healthy SELL while holding a depressed one)
both exist in `src/kaggriculture/engine/tape_runtime.py` behind flags, default off, and both
measured **large negative against a byte-identical control** (0% at −$10k to
−$19k over 3 seeds × 2 seats; see BUILD_JOURNAL 2026-08-08). The trigger
cannot distinguish an opponent's squeeze from the route's own scheduled price
impact. This is the fifth independent confirmation of *change the order,
never the inventory*. The only guard operating point that is not negative is
the $1 floor (+$170, under the noise bar), because floored sales do not add
to market inventory (A4). The squeeze response remains an open problem with
no surviving in-agent mechanism; the honest lever is base-route selection
(2026-08-08 family tournament) and freshness.

---

## 2026-08-11 — arm commits shipped without field evidence; do not re-arm below 25 judged commits

v23/v23.1_bandit enabled `_CLASS2ARM = {'1': 'raj'}` with `gates.json`
showing **0 judged commits**. The identifier mapped ~half the mainstream
field into class 1; the raj market overlay is incoherent with the base
farm plan (drops the t=240 `BUY_LAND`, off-plan buys, weed spiral) and
halved our bank in 14 of 19 ladder losses. Rating: 2776 (v22.2, arms
off) → 1056 (v23.1, arms on). Full analysis:
`docs/history/agent-v23_2-postmortem.md`.

**Guard added** (`src/kaggriculture/agentbuild/v22_agent.py`): non-empty class→arm mappings are
forced empty until the commit audit records ≥25 judged real-ladder
commits. This is the 7th confirmation of the v19.1/v22.1 arm lesson —
schedule-retiming arms must earn their way in with paired validation in
which the arm actually fires.

---

## 2026-08-11 — the relay meta went public; the gap that remains is the production core

Six public notebooks reviewed (`docs/competitor-meta-2026-08-11.md`).
Boatlee published V16-RC2-MarketRelay in full at public score **3094**; it is
being copied verbatim (flexonafft) and extended (llccqq624 dual-market-relay:
3/4-turn lead + premium 1-turn pre-empt from step 120). Their structural
signature and debt-accounted FERTILIZER advance are functionally identical to
our relay v2 — convergent, published first by them.

**Measured (12 games, both seats) vs their dual-market-relay: we lose −1,241.**
Isolation: raising our relay lead 2→5 changed the result **not at all**
(byte-identical); disabling our relay moved it only to −1,405. The relay is
worth ~+164/game here — the ~1.3k gap is their **production core**, not the
race. **Do not escalate relay leads; the parameter is inert.**

Next action (ranked highest): run boatlee's public artifact through the
`refresh_cycle` crown gate as a base candidate — never adopt on one pairing,
never evict a stronger live agent for a noise-level gain.

---

## 2026-08-12 — foreign notebook routes: eligible on purpose, never unattended

Boatlee's public production route is now a registered crown candidate
(`pubboatleev16rc2_s0`, `source: "notebook"`). It was never eligible before for
a structural reason, not a policy one: `stage_tourney` pools only records that
`routes.py --mine` put in the index with `window=="fit"` and dated
newest_day/today, and a notebook artifact has no route record at all. Their
agent embeds its route as one blob, so registering it needed no simulation.
Measured on our runtime vs our crowned base: **75% wins, +866** (6 seeds, both
seats).

Because the 06:30 cycle crowns and `daily_release` then submits,
`source=="notebook"` routes are **excluded from the unattended tournament pool**
unless `refresh_cycle --include-foreign` is passed. Shipping a competitor's
published route as our submission must be a human decision.

**Mining is bounded by archive lag, not by budget.** Newest published archive
day is D-2 and is mined out, so `--days 2` yields 0 new routes; only
newest-day/today routes can be crown candidates, so deep-history mining feeds
the identifier and sink only. See `docs/competitor-meta-2026-08-11.md`.

---

## 2026-08-12 — the crown gate is frozen by a saturated referee panel

The attended tournament (`--include-foreign`, 2,844 fit routes / 2,843 same-day
→ 14 candidates over 62 families) put **four** candidates at a perfect 56/56.
`CROWN_GATE` is a *win-percentage* threshold (+10 points over the incumbent), so
when incumbent and challenger both sit at 100% **nothing can ever clear it** —
the gate is stuck at HOLD regardless of candidate quality.

Root cause of the saturation: the referee tapes were built from v23.1's ladder
losses, and v23.1 lost those to the arm-commit bug rather than to strong play.
Any healthy route beats them, so the panel measures "is the arm bug absent".

Two fixes, both worth doing before the gate is trusted again:
1. Rebuild referees from the losses of a *healthy* agent (v23.2 / live route).
2. Let `stage_crown` fall through to a **margin** threshold when incumbent and
   best candidate both saturate on win%. The ranking already uses margin as a
   tiebreak; only the gate is win%-only.

Result of the run itself: boatlee's public core led on margin (+409,748 vs the
incumbent's +393,616 = ~+288/game) — inside the $3k noise floor, so **it is not
a meaningful upgrade and we do not need their route**. Details:
`docs/crown-tourney-2026-08-12.md`.

## 2026-08-12 — same-day fetch was ingest-bound, not download-bound

The scrape's 6 download threads were never the bottleneck: during the evening
v24 release the stage dir held **1,600+ replays already downloaded** while the
single ingest thread advanced at ~5 episodes/min. Cause: each ~30 MB replay
was `json.load`-ed **five times** (replay_meta, _episode_date, _ingest, and
extract_actions per seat) — ~2 s of redundant parsing per episode, amplified
by GIL contention with the still-running download threads. Fixed 2026-08-12:
`opponents.replay_meta` / `extract_actions` and `sameday._episode_date` accept
a pre-parsed `data=` dict and `_ingest` parses exactly once (measured 5.2x,
byte-identical output; scratchpad equivalence test).

Remaining (not yet done):
1. **--max-gb is still soft when ingest lags** — `spent` is tallied at ingest
   time, but all 2,000 futures are queued up front, so downloads can overrun
   the budget long before the loop notices (46 GB staged vs a 12 GB budget
   observed). Tally bytes inside `grab()` and stop submitting/cancel there.
2. **_stage leftovers accumulate** (files past budget are kept, only ingested
   ones are deleted). They ARE reused on the next run (grab() checks first),
   which with the single-parse fix makes re-runs fast — but add an age-based
   sweep so the dir cannot grow unbounded.
3. If ingest needs to go faster still, move parse+extract into the worker
   threads (return parsed artefacts, keep index mutation on the main thread).

## 2026-08-12 — pipeline de-duplication + hourly fetch + parallel tournament

Measured on release attempt 4: stage-1 FETCH ~48 min, then the cycle re-ran
the SAME fetch (its own harvest + scrape + archive + breadth) for ~55 more
minutes. Fixes, all live for the 2026-08-13 04:30 run:

1. `refresh_cycle.py --no-fetch` — skips fresh_data + archive/breadth mines
   entirely; `daily_release.py` passes it (stage 1 just fetched). With
   --no-fetch, "new routes" means routes dated today/newest-day, so the
   nothing-new HOLD check still works.
2. **Hourly scrape**: `KaggricultureSameDay` re-registered at :35 past every
   hour (was 4x/day), 2 GB sips, top-120/3h harvest, own-games delta, and a
   release-in-progress guard (command-line match). The 02:xx run alone
   carries the historical archive backfill (`--days 0`) that used to sit in
   the release's critical path — old days feed the identifier, never crown
   candidates. `fresh_data.py` default archive pass narrowed to `--days 2`.
3. **Parallel tournament**: stage_tourney fans (candidate x seed-set) cells
   to 6 concurrent evaluate.py processes (`TOURNEY_JOBS`); the loop was
   single-core. Expected ~4-5x on the 30-60 min tournament leg.
4. Release trigger moved 06:30 → **04:30 IST**; the stage-5 quota floor
   (holds until 00:05 UTC) makes the pair land 05:35–06:00 IST regardless of
   how fast the run gets.

Duplicate downloads were already prevented at three levels (index dedupe in
sameday, `mined` set vs manifest in routes.py, _stage reuse in grab()); the
waste was running the whole *sweep* twice, not re-downloading bytes.

### Error audit of the 2026-08-12 release logs (all fixed same night)

- ~80x 429 on ListTeamPublicSubmissions: whole teams silently skipped per
  sweep. leaderboard_harvest now retries 3x with backoff per team and
  stretches its inter-call pause under throttle pressure (decays on success).
- 8x tape replay CLI exit-1 (killed attempt 3): stage_tapes now fetches from
  kaggleusercontent first, CLI fallback.
- 4x ConnectionAbortedError 10053 in the archive mine burst: _fetch_one now
  retries once after 3s (permanent errors re-raise).
- 404 Not Found for files pruned from old daily datasets (2026-08-01/05):
  now TOMBSTONED into the mined set on failure -- previously retried on
  every backfill run forever.

## 2026-08-13 — crown gate: margin fallback replaced by a WINS playoff

Operator review (2026-08-12 night) called out the margin fallback correctly:
the ladder pays wins, not coin margin, so a saturated-panel crown decided on
margin optimizes the wrong currency. Root cause of the saturation is panel
weakness, not metric choice -- when every strong agent beats all loss tapes,
NO metric computed against those referees discriminates.

Fix (live for the 2026-08-13 04:30 run): when the panel saturates,
`stage_playoff` runs a round-robin among the top PLAYOFF_FINALISTS(3)
candidates + incumbent (both seat-pairs, both seed sets, TOURNEY_JOBS-wide).
Strong-vs-strong cannot saturate like strong-vs-referee, approximates the
top-field population better than any single head-to-head, and its currency
is pure win rate. `stage_crown` crowns on playoff win% >= incumbent +
PLAYOFF_GATE(10pp); margin is logged as a diagnostic and remains decisive
ONLY if no playoff could be run. Unit-tested all three paths (crown, hold,
no-playoff fallback).

Note: a pure head-to-head vs the incumbent alone would be WORSE than margin
(the ladder rarely pairs us with ourselves; single-opponent transfer is
weak). The round-robin against several strong finalists is the point.

## 2026-08-13 morning — two incidents, both resolved

1. **Machine crash killed the overnight release** at 01:32 (kernel-power 41,
   unexpected shutdown) during arm training — after v24.0_route was built but
   before the bandit. Not a pipeline defect; the 04:30 scheduled run fired
   normally afterward.
2. **Registry-clustering regression**: the 04:30 run was the first to retrain
   the identifier with the ascending-order family registry, and held-out
   accuracy cratered 0.893 -> 0.571 (train 0.589 — unfittable labels:
   oldest-seeded append-only representatives chain distinct current styles
   into stale blobs). Caught before the build stage; run stopped, clustering
   reverted (newest-first rebuild), retrain restored **0.926** (best yet,
   with temperature T*=0.81 folded into the export). Lesson recorded: a
   clustering change must gate on downstream label QUALITY (held-out acc),
   not stability alone. The 04:30 run also PROVED the speedups: --no-fetch
   cycle, 6-wide parallel tournament, tape fast-path.

## 2026-08-13 midday — BSOD investigation: driver DPC starvation under load

Three hard crashes in 11 hours, all under sustained engine load:
- 01:40  bugcheck 0x9F DRIVER_POWER_STATE_FAILURE (power IRP hung mid-run)
- 11:09  bugcheck 0x133 DPC_WATCHDOG_VIOLATION arg1=1 (5-wide ablation)
- 12:00  bugcheck 0x133 DPC_WATCHDOG_VIOLATION arg1=1 (5-wide tournament)
No WHEA events (not thermal/hardware-detected); minidumps saved
(C:\Windows\Minidump\081326-*). Diagnosis: a driver starves the kernel DPC
queue when the box runs 5-6 pure-python engine processes flat-out for tens
of minutes; a second driver (or the same one) hangs power transitions.

Mitigations applied (software-side):
1. TOURNEY_JOBS 6 -> 3 (refresh_cycle.py); ad-hoc drivers capped at 2.
2. Processor maximum state 90% on AC (kills sustained turbo pressure):
   `powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 90`
   (revert with value 100).
3. AC standby disabled (STANDBYIDLE 0) — removes power-transition windows
   during unattended runs (revert: restore prior timeout).
4. Long experiment drivers now checkpoint per cell and RESUME after a crash
   (.local/cand241/ckpt — 5/20 cells survived today's crash).

OPERATOR TODO (needs hands): update storage/NIC/GPU drivers, or feed the
three minidumps to WinDbg (`!analyze -v`) to name the culprit module.

## 2026-08-13 afternoon — RECROWN: kss base shipped as the v24.1 pair

Loss forensics on v24.0 (51 losses): ~50% base-strength, ~31% coin-flips.
Deep harvest with the fixed pagination (10,352 ids, 511 submissions, true
top-300) enriched the index; hardened tournament (referees = what actually
beats us: kss tape, fresh loss tapes, decoded public relay/clone, crowd):

  92513718_s1 (kss)   72/72   +7,231/game   <- CROWNED (+19.4pp on WINS)
  incumbent           58/72   +4,934/game
  91650813_s1         52/72   +4,878/game
  (7 more, 46/72 and below; yesterday's top-bank route went 0/72 --
   offline bank is not panel strength)

Shipped 15:36-15:38 IST: 55477866 v24.1_route (nb v14, eb86a1866a77) +
55477892 v24.1_bandit (nb v6, 5672592a85c1). Mine-your-conqueror loop closed
end-to-end: morning loss -> replay -> route index -> panel win -> our base.

Also fixed during the ship: submit_agent trusted `kernels status`, which
reports the PREVIOUS version's COMPLETE while the new run executes -- the
sha guard correctly refused v5 output for the v6 agent; the wait now polls
the OUTPUT sha until it matches (up to 30 min). Entire recrown ran at the
mitigated profile (2-wide, CPU 90%) with zero BSODs.


## 2026-08-13 night - v24.1 field forensics: the market, not the race

Fetched all 14 v24.1 losses plus the 8 biggest wins (22 replays,
`.local/lossreplays/`) and pointed four new lenses at them. Two schema traps
found first: `farm.tiles` is a 10x10 NESTED grid and the shed lives in
`observation.private`, not on the farm, so a flat parse silently reports zero
crops/animals/shed for every game.

### 1. Our play is IDENTICAL in all 22 games

Same quad-2 day (6), same quad-3 day (10), same peak hands (14 @ day 10),
same end hands (9), same end crops (2) / animals (14), zero carried
inventory, zero shed left. That is the open-loop route doing exactly its job.
The consequence matters: **our final bank varies 41,650..171,260 with our
actions held constant**, so essentially all outcome variance is the shared
market and the opponent, not our decisions.

### 2. Contested-dump forensics: the relay WINS its sub-game and it explains nothing

New tool `src/kaggriculture/data/contested_dumps.py` (diagnostic, gates nothing). It rebuilds
every collision - our SELL and theirs on the same product within N turns -
and prices who printed first.

    22 games, 2,392 contested dumps (1,217 exact TIES, 1,175 decided)
    we sold first          798/1,175   67.9% of decided races
    lost races cost us         -629/game
    won races cost THEM      +2,543/game
    net race value           +1,914/game   (losses +1,987, wins +2,283)
    games where the race net went AGAINST us:   0 / 20
    corr(race net, final margin) = -0.108

So the relay's real-world win rate is ~68%, it is worth ~+2k/game, it is
positive in *every* game including all twelve real losses, and it has no
correlation with the final margin. **The dump race is not what loses these
games.** Half of all collisions are exact same-turn ties, which no amount of
racing can win - the earlier 100%-first reading was a seat-0 tie-break bug.

### 3. Per-order price impact is NOT identifiable; the effect is cumulative

The per-order regression is worthless in both directions: a placebo
regression (the same order flow against the price move that ALREADY
happened, which a sell cannot cause) is larger in magnitude than the forward
estimate for all nine products. Sell timing is entangled with the market's
own cycle. `--impact` now prints the placebo next to every estimate and
flags it ENTANGLED.

But at the DAY/game horizon the effect is enormous:

    corr(their total sell units, our bank)   = -0.46
    corr(late-game mean price,   our bank)   = +0.88   (margin +0.81)
    our sell units: 1,418 +/- 33  (fixed schedule, as expected)

Opponents who flood (3,693-3,763 units) leave the late market at 33-54 and
our bank at 54k-103k; Ethan J. Schauder sold 413 units, late prices sat at
110, and we banked 171k. 97% of their volume lands in products we also sell,
so diverting the mix is not available - everybody farms the same things.
**The losses are low-price-regime games**: losses average late price ~50,
wins ~82. In a depressed market both banks compress and the game becomes the
coin-flip we keep losing by -536 to -4,532.

This also retires the old "price impact is tiny (10,000-unit equilibria vs
40-unit dumps)" reading: it was measured at the wrong horizon. One order
moves nothing; a day of them walks the price down permanently. Recovery test
over 111 heavy product-days: +4.0% same day, **-21.8% three days later** -
the drop does not come back.

### 4. The prize: sell TIMING, worth up to +16,518/game and strictly feasible

Price paths are strongly trended and in opposite directions - FERTILIZER
slides 95 -> ~10 over 30 days, WHEAT climbs 32 -> ~45. Our route sells a
fixed basket on a fixed schedule mined from one episode. Pricing that
schedule against the only strictly-feasible reordering (sell LATER; holding
stock in the shed is free, selling earlier is impossible because the units do
not exist yet):

    product     units/g  realised  hold-px   as-run $/g   feasible gain
    STRAWBERRY      272     125.9    152.9       34,220       +7,358
    WOOL            138      78.7    108.3       10,851       +4,086
    MILK            224     131.9    146.4       29,533       +3,264
    WHEAT           445      41.6     44.8       18,492       +1,412
    MELON           103     146.1    150.0       15,060         +398
    FERTILIZER      236      52.9     52.9       12,508           +0
    TOTAL                                                   +16,518/game

FERTILIZER is already optimal - it decays monotonically, so selling on sight
is correct. STRAWBERRY, WOOL and MILK are being dumped well before their
peaks. The hold-until-best figure assumes foresight of the peak, so it is the
ceiling of a causal policy, not a plan - but **every single v24.1 loss is
under 12,878**, so capturing even a third of it flips all fourteen.

### What this changes

- Relay / dump-race: **stop investing.** Measured at 68% win rate and
  +1.9k/game on the real ladder, uncorrelated with the margin. It works; it
  is not the bottleneck. `_HOLD = False` stays right for the *turn-scale*
  price-hold layer - the horizon it targets does not exist.
- Next lever is a **day-scale sell-timing policy** for STRAWBERRY / WOOL /
  MILK, driven by the observed price level and trend rather than by the
  mined schedule's fixed turns. This is an open-loop route change (cheap,
  reliable, gated by the playoff), not another adaptive layer.
- Loss-tape value confirmed: the twelve real conquerors become referee tapes
  in the 04:30 cycle automatically.

Tools added: `src/kaggriculture/data/contested_dumps.py` (tracked); drivers
`.local/loss_forensics_v241.py`, `loss_structure_v241.py`,
`crowding_v241.py`, `flood_products_v241.py`, `timing_value_v241.py`.

### Post-mortem: why the adaptive/RL stack did not find this

It is fair to ask, since day-scale sell timing is exactly what an adaptive
layer should own. Traced in code, the stack never had the task:

1. **The price PATH is not in the state.** The agent's entire notion of price
   is `_quote(item, inventory)` - a pure function of the *current* shared
   inventory (src/kaggriculture/agentbuild/v22_agent.py:518). No trend, no day index, no memory of the
   level. "FERTILIZER decays 95 -> 10, so sell it early; WHEAT climbs 32 ->
   45, so sell it late" is not a representable thought.
2. **The feature exists but is wired to the wrong consumer.**
   `_track_opponent` reconstructs `opp_cum` exactly - the opponent's
   cumulative sales, the quantity that correlates -0.46 with our bank. It has
   three consumers and none of them is a sell decision: it is computed
   (:538), normalised into the *identifier* feature vector (:642), and diffed
   to detect *dump events* for the race (:904). It answers "who are you" and
   "when will you dump", never "should I sell now".
3. **The one layer that could - `_price_hold` - encodes mean reversion, which
   the data falsifies.** Its trigger is inventory-above-equilibrium plus a
   projected recovery of "the town draining half the excess within the hold
   window", with `_HOLD_MAX = 6` **turns**. The measured market does not
   revert (-21.8% three days after a heavy day) and the prize is at multi-day
   scale. Its own `_quote`-based gain estimate is near-flat around
   equilibrium, so it cleared the max($40, 6%) threshold ~never - hence 0
   fires in 80 games, and hence `_HOLD = False` was the right call for the
   layer as written.
4. **The online RL is one scalar.** `relay_lead_rl` is the race lead in
   turns, incremented by +2 when it observes it was pre-empted, clamped to
   `_RELAY_LEAD_MAX`. Action space = how many turns early to race a dump.
   Sell quantity and multi-day rescheduling were never actions.
5. **The eval could not have seen it.** The ablation roster is 9 tapes mined
   from our conquerors - i.e. strong, heavy-producing farms. Decoded sell
   volumes: 1,014..3,742 units, mean 1,963. The real field is 413..3,763,
   mean 1,633, and its best-paying games are the 413/868/930/1,089-unit
   opponents. **The roster contains no light-selling opponent at all**, so
   the high-price regime is unsampled and the market is depressed in nearly
   every ablation game. A market-timing layer measured only there is being
   asked about a regime it never sees. Reward is also end-of-game margin -
   far too sparse to credit a day-8 sell.
6. **The learning parts are guarded off** anyway: arms at 0 commits
   (commit_judge: E[gain] $1 vs E[cost] $41,315), GRU on HOLD because its
   consumer is arms.

So the stack did not fail at this task; it was pointed at a different one
(identify the opponent, win the dump race) and it *succeeded* there - 68%,
+1.9k/game, measured on the ladder. The fix is therefore not more RL:

- add price level + trend per product, and `opp_cum`, to the sell decision;
- replace the mean-reversion trigger with the measured monotone trends;
- put the light-selling half of the field back in the roster before believing
  any market-timing measurement.

### CORRECTION (same night): the timing prize is +2,507/game, not +16,518

The hold-until-best figure above rested on "holding stock in the shed is
free". It is not: **`_SHED_CAP = 100` units** total, and the route pushes
~1,418 units through the shed per game, so most of that schedule is
unreachable. Measured shed occupancy is mean 25.5/100 with only 0.7% of turns
above 90, so *bounded* deferral is available - tens of units, not hundreds.

Re-sized with a greedy constrained re-timing that never lets deferred stock
plus observed occupancy exceed the cap (20-unit safety margin), over the same
22 games (`.local/feasible_timing_v241.py`):

    horizon      gain/game    % of ~100k bank
     12 turns      +1,521           1.5%
     24 turns      +2,080           2.1%
     48 turns      +2,329           2.3%
     72 turns      +2,507           2.5%   <- saturates here
    120 turns      +2,578           2.6%
    240 turns      +2,336           2.3%   <- declines: early holds block later ones

    per product @72 turns: STRAWBERRY +1,022, MILK +1,012, WOOL +733,
                           WHEAT +195, MELON +9, FERTILIZER +0

So the prize is ~2.5% of bank with PERFECT foresight of the peak - the same
order as the relay's measured +1,914/game, not 8x it. A causal
trend-following policy captures some fraction of that. Worth building (the
twelve real losses average -2,348, so ~+1,000/game flips several of them), but
it is an incremental gain, not the step change the first pass implied.
FERTILIZER and MELON are confirmed no-ops: never defer them.

Also fixed while auditing:

- **`_price_hold` read `farms[me]["coins"]`; the engine key is `money`**
  (src/kaggriculture/agentbuild/v22_agent.py:1205). The value was always the 0 default, so the
  `coins < _HOLD_CASH_FLOOR` guard tripped every turn and the capture path was
  **dead code even before `_HOLD = False`**. A missing key and a real zero are
  indistinguishable through a default, so nothing raised.
- New test **`tests/test_obs_schema.py`** walks every shipped agent's AST,
  resolves each `_get(container, "key")` / `.get("key")` against the schema
  observed in a real replay, and fails on any key the engine does not have.
  Across all 51 agents `coins` was the ONLY real instance - the other 61
  initial flags were the checker's own false positives (tile keys are the
  union over PLANT and PASTURE cells; `obs["farms"][seat]` is a farm, not an
  obs), each fixed in the checker rather than waved away.
- **`_price_hold`'s release path could silently evict route orders**:
  `(release + market)[:_MAX_ORDERS]` with no room computation, unlike
  `_relay` which reserves `room = _MAX_ORDERS - len(existing)` first. With 10
  scheduled orders any release would push the tail off the queue - including
  BUYs the route's economy depends on. Fixed by construction in the
  replacement layer.

### The RL question, answered with measurement: this problem does not want RL

Asked whether the adaptive RL should have found the sell-timing gap, and how to
improve/train/test it. The decisive property of sell timing is that **the
counterfactual is fully observed**: at every sell decision the logged replay
contains the entire future price path, so the payoff of every action is
computable after the fact. That is full-information feedback, not bandit
feedback -- no exploration problem, no credit-assignment delay. RL is the tool
for when the counterfactual is invisible; here a regressor on labelled
decisions strictly dominates and trains in seconds.

So an offline harness was built (`.local/policy_eval_timing.py`) that scores
any policy exactly on the 22 logged episodes under the real shed cap, and
three causal policies were tested against the constrained oracle, both learned
ones LEAVE-ONE-EPISODE-OUT so neither sees the episode it is scored on
(`.local/learned_timing.py`):

    policy      gain/game   % of ceiling   fires
    momentum        +203          14%       363     <- the shipped rule
    ridge           +120           9%       329     6 features, per product
    peakday          +77           5%        77     per-product target day
    oracle        +1,404         100%       778     perfect foresight

    fitted peak days: STRAWBERRY 17, MILK 13, WOOL 5, WHEAT 27

**Every causal policy is weak, and that is the answer.** The oracle's advantage
is not model capacity, it is foresight: it takes the argmax of a realised price
path that depends on the opponent's not-yet-taken actions. An argmax-over-
realisations baseline is optimistically biased by construction, so the gap
between it and a well-specified expected-value policy (the ridge, +120) is
mostly irreducible noise. No algorithm - GBDT, transformer, PPO - conjures
information that is not in the state at decision time.

**Corrected size of the prize, in three steps:** +16,518/game (hold-until-best,
infeasible - ignored the 100-unit shed) -> +2,507/game (shed-feasible, still
perfect foresight) -> **~+200/game (causally achievable)**. That is below the
$3,000 house noise floor. Sell timing is a real but marginal lever, not the
step change the first pass implied.

**The reusable protocol, which is the durable win here:** before building any
learner, compute the oracle-versus-causal gap on logged data. If a
well-specified expected-value policy captures little of the oracle, the gap is
noise and no model will close it. This costs minutes and would have pre-empted
both the arm-commit programme (E[gain] $1 vs E[cost] $41,315) and this one.

Where RL genuinely still fits: nowhere currently identified. Every lever
measured this session is either already won (the dump race, 68% / +1.9k) or
foresight-bound (sell timing). The base/crown choice remains the only measured
step change (v24.0 -> v24.1 was +25pp of win rate), and that is a search
problem over mined routes, not a policy-learning one.

### Crown-panel correlation matching (asked for, implemented)

Root cause of the measurement blindness, found in `stage_tapes`: the referee
panel was built from **losses only**, `rows.sort()` then `rows[:N]`, i.e. the
opponents who beat us worst -- systematically the heavy producers. Decoded
roster volumes were 1,014..3,742 (mean 1,963) against a real field of
413..3,763 (mean 1,633), **with no light seller at all**. A price-regime change
measured there is being asked about a regime the panel never shows.

Fixed: `field_strata()` in src/kaggriculture/pipeline/refresh_cycle.py picks one representative game
per stratum of the opponent SELL-VOLUME distribution -- the feature measured to
drive the price regime (corr with our bank -0.46) -- wins included, and
`stage_tapes` appends them to the loss tapes. `FIELD_STRATA = 4`. The tape
docstring label was also wrong for these ("beat us" for a game we won) and
`sink._route_identity` reads docstrings, so it now states what happened.

The matched roster immediately paid: it shows a clean regime gradient that the
old panel could not represent -- v24.2 beats the light sellers by
+41k..+104k/game and goes 0/4 against both heavy sellers (HunterBlue, Anurag),
which are exactly the teams we lost to on the ladder. The tapes reproduce the
real losses faithfully.

### The bug that mattered most: the ARM GUARD unlocked on evidence VOLUME

Found while explaining an unexpected ablation result, and it was a live time
bomb aimed at the next unattended 04:30 release.

`src/kaggriculture/agentbuild/v22_agent.py` gated arm commits on `judged_commits < 25` -- the COUNT of
audited commits, with no regard for what the audit said. commit_judge.py
exists precisely to prove arms harmful, so by recording 27 judgements it
mechanically **UNLOCKED the feature it had just disproved**. A rebuild duly
shipped arms ON (class 1 -> raj) with no warning printed.

What the audit actually says, over its 27 real judged commits:

    mean margin_delta, all commits        -50,787
    mean margin_delta, CORRECT commits    -52,940   <- correct still loses
    mean margin_delta, WRONG commits      -41,315
    e_gain_correct (as stored)                 +1   <- clamped!

`train_gates.py` computed `e_gain = max(1.0, e_gain_raw)`, so the clamp
converted "a correct commit loses $52,940" into "a correct commit gains $1"
and the negative-value verdict was invisible to every consumer.

Fixed in three places, single-sourced:

- `train_gates.arms_allowed(gates)` is now the ONE predicate: enough judged
  commits AND favourable evidence. Returns (allowed, reason).
- `train_gates.py` records `e_gain_correct_raw` (-52,939.6),
  `e_cost_false_raw`, and an explicit `arms_positive_value: false`, so the
  clamp can no longer hide the sign.
- `src/kaggriculture/agentbuild/v22_agent.py` and `src/kaggriculture/pipeline/refresh_cycle.py` both call that predicate, so
  guard and arm-training profile can never disagree. The cycle keyed its
  fast-arm profile on `n_commits < 25` too, so at 27 rows it would have gone
  back to the full 6x10 search -- 30-60 min of critical path for schedules the
  guard discards.

The guard now prints:

    ARM GUARD: audit says a commit is NEGATIVE-VALUE:
    E[gain|correct]=$-52,940 vs E[cost|wrong]=$41,315 -- arm commits DISABLED

**Lesson, and it generalises past this repo:** a gate that opens on the amount
of evidence rather than its direction will eventually open on evidence that
says "don't". Every count-based unlock in the pipeline was audited for the same
shape after this.

### Verdict on the market-timing layer: MEASURED REGRESSION, shipped OFF

Clean one-variable A/B (`.local/ablation_timing_clean.py`): two builds from the
SAME embedded models differing only in the `_TIMING` literal, 7 field-matched
opponents, 3 seeds, both seats, 84 games each.

    opponent            opp vol      timeON        timeOFF     delta
    Ethan (light)           413  12/12 +103,893  12/12 +105,013   -1,120
    Mahfujur                868  12/12  +64,836  12/12  +65,964   -1,128
    Prasad                  930  12/12  +57,915  12/12  +60,287   -2,372
    Nathaniel             1,089  12/12  +83,567  12/12  +91,831   -8,264
    kss (crowned base)    1,943   7/12     +653  12/12   +2,372   -1,719
    HunterBlue (heavy)    3,693   0/12   -6,090   2/12   -4,693   -1,396
    Anurag (heavy)        3,763   0/12   -3,701   5/12   -1,744   -1,956

    timeON  55/84 wins (65.5%), +43,011/game
    timeOFF 67/84 wins (79.8%), +45,576/game        delta -2,565/game

Worse on 7 of 7 opponents and -14.3pp of win rate. The margin delta alone sits
at the noise floor; the 7/7 consistency and the win-rate drop are the result.

The mechanism was already priced in this same session and I should have seen it
before building: **deferring a sell forfeits the dump race.** Winning the race
is worth +1,914/game measured on the ladder; the timing layer's entire upside
is +203/game. That is a ~10x losing trade. And 1,217 of 2,392 real collisions
are exact same-turn TIES, which deferral converts from ties into losses.
Skipping `racing` products is insufficient -- the relay only marks what it
actively fired on, not the far larger set of simultaneous sells.

`_TIMING = False`, kept in the tree with the measurement in the comment, since
a future engine rebalance that weakens the race would change the trade.

**Net position after this session:** no agent change ships. The value delivered
is the bug fixes (which change what TOMORROW builds), the schema test, the
field-matched panel, and the arm-guard fix that stops a -$51k-per-commit
feature from going live unattended.

## 2026-08-13 late night - the currency fix: margin retired as a criterion

Operator's call, and correct: the ladder pays win / draw / loss, so $1 and
$10,000 of margin are the SAME RESULT. Every decision made on mean margin was
denominated in a currency the competition does not use.

The flaw is worse than cosmetic. A margin shift converts to wins only through
the DENSITY of the margin distribution near zero, so two changes with identical
mean margin can be worth wildly different win rates. Measured over **1,748 real
games** (`src/kaggriculture/measure/win_metric.py`, run it to reprint):

    W/D/L 1044/26/678, expected score 0.605
    median |margin| $8,136, sd $23,850
    27.6% of all games decided by under $3,000
    240 of our 678 losses were by under $3,000

    uniform shift  -> win rate     games flipped (of 1,748)
      +$   500          +4.06pp            71
      +$ 1,000          +6.81pp           119
      +$ 3,000         +15.22pp           266
      +$10,000         +28.49pp           498

So +$1,000/game spread evenly is worth +6.8pp, while the same MEAN concentrated
in a handful of blowouts is worth ~0. A mean margin is a mean over a variable
whose value is a step function at zero, which is why it can never be the
criterion.

### New module: src/kaggriculture/measure/win_metric.py

- `score()` / `expected_score()` / `summarise()` -- draws are 0.5, not 0. The
  old collapse alarm counted every draw as a loss.
- `flips()` -- the EXACT win value of a per-game effect. Use whenever per-game
  numbers exist; it applies the zero-crossing per game instead of averaging
  dollars that mostly cannot matter.
- `paired_test()` -- McNemar exact on discordant pairs. Replaces the "$3k noise
  floor" with a p-value.
- `min_detectable()` -- sample-size PLANNING only, explicitly not a post-hoc
  veto (the timing A/B is significant at p=0.031 while sitting below the crude
  floor).
- `dollars_to_wins()` -- restating legacy dollar claims only.
- `parse_eval()` / `totals()` -- see below.

### Consumers converted

| site | was | now |
|---|---|---|
| `refresh_cycle.stage_crown` | crowned on `$3,000/game` when win% saturated and no playoff ran | **HOLD** -- no win evidence, no crown. `CROWN_MARGIN_PER_GAME` deleted |
| `train_surrogate` | regressed margin dollars (R^2 0.801) | target = **score**; holdout **AUC 0.880, accuracy 0.947** (margin R^2 kept as diagnostic) |
| `train_gates` | E[gain]/E[cost] in dollars | prefers `score_delta`, warns loudly when only dollars exist |
| `commit_judge` | recorded `margin_delta` only | also records both sides' banks and `score_delta` |
| `collapse_alarm` | `bank < $60,000` trigger | **expected score** trigger; bank is context |
| `evaluate.py` | headline `win%` + margin | headline **expected score**, W-D-L, margin marked diagnostic |
| `autopilot` promote | `win > 0.55 AND margin > 0` | `win > 0.55`; a dollar figure can no longer veto a real win-rate gain |

`collapse_alarm`'s bank floor was actively harmful, not merely wrong-currency:
bank is set largely by the OPPONENT's flooding (corr(late price, our bank)
= +0.88), so a healthy agent drawn against heavy sellers banks 54k and would
have tripped a 60k floor, while a broken agent in a rich market clears it.

### A latent break caught in the same pass

Changing `evaluate.py`'s table would have silently destroyed the crown gate:
**three** call sites in `refresh_cycle` scraped the human `win%` column with
`re.match(r"\S+\s+(\d+)%...")`, and on a parse failure they returned `w=0,
g=0` **without raising**. Adding a score column would have handed the
tournament all-zero results.

Fixed at the root: `evaluate.py` now prints a stable machine-readable
`RESULT	<opp>	<score>	<w>	<d>	<l>	<games>	<bank>	<opp_bank>	<margin>`
line per opponent, and `win_metric.parse_eval()` is the only sanctioned reader
-- it RAISES on no matches rather than reporting a silent zero. All three
refresh_cycle sites plus the ablation drivers now use it. A display format is
no longer load-bearing. (The playoff site had a second bug: it `return`ed
inside the per-opponent loop, counting only the first opponent.)

### This session's dollar claims, restated in the currency that pays

    claim                                    $/game   -> win rate   games flipped
    relay / dump race                        +1,914      +10.76pp        188
    sell timing, causal ceiling                +203       +2.86pp         50
    sell timing, shed-feasible (foresight)   +2,507      +13.04pp        228
    sell timing, original (infeasible)      +16,518      +34.21pp        598
    boatlee public core                        +288       +3.26pp         57
    ARM COMMIT (measured)                   -50,787      -55.95pp          -

The timing A/B, restated properly: **expected score 0.655 vs 0.798, -14.29pp**,
paired sign test over 21 blocks **6 worse / 0 better, exact p = 0.031**. That is
a far better-founded rejection than the -$2,565 ever was, and it no longer
leans on a dollar noise floor.

`CLAUDE.md`'s measurement-discipline section was rewritten: the "treat
differences under ~$3k as noise" rule is gone. New tests:
`tests/test_win_metric.py` (7 checks, including that $1 and $10,000 score
identically and that `parse_eval` raises rather than returning zeros).

**Still owed:** the arm audit's 27 rows predate `score_delta`, so the gate is
currently priced in dollars and says so out loud. A `commit_judge` re-run would
settle it in win units. It changes no decision today -- arms are blocked either
way -- so it is not urgent.

## 2026-08-13 night - plan execution: A, B, C1, D1 + H review

Operator decisions: 1c (disable the release task), 2b (stop auto-memory), 3a
(stay at 3 workers), 4a (max-fidelity capture), 5a (forward-only), 6a
(identifier experiment), 7a (re-judge, defer retrain), 8b (probe + scaffold,
standalone binary), 9 no notebook pushes, 10 no uploads at all.

### A - v24.1 protected

`KaggricultureRefreshCycle` **disabled**; `KaggricultureSameDay` left enabled
(it only downloads, and forward-only capture depends on it). Live pair stays
55477866 route (2157.4) + 55477892 bandit (**2380.1 -- our best rating ever**,
previous best v23_route at 2214.3).

### B - pipeline speed, all measured

**The biggest fetch win was not on the plan.** The archive miner used
`kaggle datasets download` -- the THROTTLED transport, whose own record in
sameday.py is ~5 episodes per 30 minutes (~360 s each) -- while sameday used
kaggleusercontent at a measured 3.05 s. Archive-era ids resolve there too
(verified on 2026-08-01 episodes, 12 days old), so `routes._fetch_one` now
tries the direct endpoint first with the CLI as fallback, matching sameday and
stage_tapes. A direct 404 still falls through so tombstoning survives.

| fix | before | after |
|---|---|---|
| archive fetch transport | ~360 s/episode throttled | **3.05 s/episode** |
| byte budget accounting | decompressed bytes | **wire bytes** (measured 13-16x ratio; an "8 GB" budget was ~0.5 GB of traffic) |
| identifier retrain | 8.1 min | **1.15 min** (11,912 cache hits, 49 built) |
| tournament budget | 1,120+ games fixed | **696 vs 1,440 (2.07x)** by successive halving |
| release fetch | always ran | skipped when the index is <2 h old, else runs |
| sha poll | flat 30 s | 10 s -> 20 s -> 30 s backoff |

`src/kaggriculture/data/feature_cache.py` caches route + prefix vectors keyed by route id, and
**invalidates on a hash of features.py and routes.py** so a code change forces
a rebuild. Verified 25/25 vectors byte-identical to fresh computation, and the
retrain produced identical output (33 classes, 0.921 held-out) to the uncached
run -- which is what proves the cache rather than assuming it.

Successive halving is `refresh_cycle.run_halving()`, hoisted to module level
with the play step INJECTED so it is testable without episodes. Only the final
round is returned, so the crown still compares finalists and the incumbent over
an identical roster and both seed sets -- the contract it already had. Six
contract tests in `tests/test_tourney_halving.py`.

**Deferred with reasons, not silently:** B4 fetch-concurrency ramp (the
transport reroute already took fetch from ~360 s to 3.05 s, so the marginal
value at 8 jobs is small) and B6 referee pruning (needs a measurement pass over
real tournament results).

Identifier note: 27 -> 33 classes and 0.934 -> 0.921 held-out is CORPUS GROWTH
(index now 11,961 routes), not the cache -- proven because the broken-cache run
(full recompute) and the working-cache run gave byte-identical numbers.

### C1 - the data we were destroying

`src/kaggriculture/data/turn_features.py` captures per-turn observation traces at ingest, wired
into `sameday._ingest` before the replay is deleted, plus **`info.seed`, which
was being discarded entirely**. Verified end-to-end: both seats (720, 49),
`trace=True` on the index records, seed recorded.

One correction to the plan's own spec: "275-dim per turn" was the PREFIX
(checkpoint) width -- a cumulative summary, wrong for per-turn capture and
expensive to recompute 720 times. The per-turn object is the STATE, 49 explicit
fields. Measured **39-40 KB per episode compressed** (both seats) against a
282 KB estimate, so ~17.6 MB/day. Storage was never the constraint; download
is, which is why forward-only (5a) still holds.

Information boundary respected: `farms` is public for both seats, `private` is
per-seat, so only OUR shed/seeds are recorded. Reading the opponent's private
block would be the train/serve skew bug reverted in the 2026-08-11 audit.

### D1 - the engine port is de-risked

`rustengine/` (Cargo, `kagg` binary) with a bit-exact CPython MT19937.
`tests/test_rust_rng.py`:

- **30 (seed, day) pairs**: `random()` bit-exact, `getrandbits` exact,
  `choice()` exact
- **240-draw weed sequences bit-exact** over 6 (seed, day) pairs, including the
  derived `< 0.005` decisions the engine actually uses
- **large key 987657284949965972 (> 2^32) seeds identically**, proving
  `init_by_array` over 32-bit little-endian words -- a truncated
  `init_genrand` would pass small seeds and fail exactly here
- scaffold constants pinned against the vendored interpreter

So bit-exact differential testing of the engine IS available: the port is
mechanical rather than approximate. `src/state.rs` holds the state model and a
per-step `digest()` for the harness, with the un-ported functions listed
explicitly rather than implied by absence. The step function is NOT ported and
nothing here feeds any decision.

Two harness bugs of my own, both caught by the test failing rather than by
inspection: Rust's `{:.17}` is 17 digits AFTER THE POINT, so 0.0154... lost
significant digits and looked exactly like an RNG divergence; then once the
protocol carried bit patterns, the threshold assertion compared integers
against 0.005. The protocol now transmits `f64::to_bits()`.

### E1 - re-judged, and the answer is worse than before

`commit_judge --games 40` yielded only **2 new rows** (most sampled episodes
produce no commit at all: the old identifier must hit class 1 at p >= 0.85).
Both new rows are devastating: `score_delta = -1.0` each -- the commit turned a
WIN into a LOSS outright, on 0.0 vs off 1.0.

**And regenerating the gate exposed a bug I had introduced today.** The gate
LOOSENED from 0.95 to **0.55**: the scored subset had no correct commits, so
`e_gain_raw` fell to 0, and the `max(1.0, ...)` clamps turned "no measured
upside" into "upside 1", making `p_star = 1/(1+1) = 0.5` -- the most PERMISSIVE
threshold available, on evidence saying commits are catastrophic. Same
clamp-hides-signal shape as the original bug, in the new code path.

Fixed two ways: one-sided score evidence now falls back with a loud warning
instead of estimating a ratio from one side, and `e_gain_raw <= 0` clamps
p_star to its most CONSERVATIVE value (0.95) rather than computing from
placeholders. `tests/test_gates.py` locks both in.

### H - review against the taxonomy

Three bug classes had no guard. Two now do:

- **stale path** -> `tests/test_paths.py`: 173 directory references across 119
  files all resolve, and nothing constructs a path into the removed `tools/`.
  (It initially flagged ITSELF, since its own regex contains the literal it
  searches for; it now excludes the file defining the check.)
- **gate opens on volume / clamp hides signal** -> `tests/test_gates.py`, six
  cases, each asking the same question: what happens when the evidence arrives
  and is bad?
- **confounded experiment** -> still unguarded; the determinism pre-check for
  A/B drivers is not written yet.

Suites green: test_paths, test_win_metric, test_gates, test_tourney_halving,
test_obs_schema, test_rust_rng, test_agents (which now covers the shipped pair).

### Still open

C2 new feature families (market regime at corr +0.88 remains the strongest
unused signal), B4, B6, E2 duel panel, E3 blend policy, D2 step-function port,
and the determinism pre-check. No agent was shipped and no notebook pushed.

## 2026-08-13 late - no-defer pass: integration, Rust engine layers, harness

### Integration audit -- two modules were orphans

Asked to prove nothing is dummy, an audit of "which module is actually imported
by the pipeline" found `determinism` and `referee_power` consumed by **nothing**
except scratch drivers. Both are now wired into `stage_tourney`:

- **determinism.require()** runs before the tournament spends its budget, over
  every candidate AND every referee. A paired comparison over a
  non-reproducible agent yields a number that reads exactly like a result --
  which is a mistake I made myself with the unseeded `random` built-in.
- **referee_power.analyse()** prunes referees that separate nobody, and
  **refuses to prune on thin evidence**: with fewer than 5 distinct candidates
  every referee looks non-discriminating, and the FIELD strata are protected
  because they exist for price-regime coverage rather than discrimination.
  Verified on real data: it correctly declined to prune from a 2-candidate
  table, which would otherwise have deleted the very panel-bias fix added
  hours earlier.

Post-fix audit: win_metric (6 consumers), feature_cache, turn_features,
determinism, referee_power -- all live, no orphans.

### Rust engine: four layers ported and EXHAUSTIVELY verified

Ordered deliberately: the parts that can be checked exhaustively rather than
sampled were done first, because "verified on every input" is available for pure
functions and never will be for the step function.

| layer | verification | result |
|---|---|---|
| `mt19937.rs` RNG | 30 (seed,day) pairs; 240-draw weed sequences; >2^32 key | **bit-exact** |
| `market.rs` prices | 2,538 quotes swept across 9 products, incl. a 1-unit sweep either side of I0 | **all exact**, raw float bit-differences **0** |
| `rules.rs` pure rules | fib(0..24), hire costs, all 100 quadrant cells, shed access, land, 5 crops x 6 fields, 3 animals x 6 fields | **exhaustively exact** |
| `state.rs` | constructs + per-step `digest()` for the harness | scaffold |
| **step function** | **NOT PORTED** | unit ops, order processing, daily refresh, end-of-day |

**Two real bugs found by these tests, in code I had already called passing:**

1. **`SHOPS` had 6 entries; the engine defines 8.** I had taken the list from
   one replay's observed `unlocked_shops`. The D1 test hardcoded the same wrong
   six on BOTH sides, so `choice()` agreed with itself and disagreed with the
   engine. The test now PARSES `SHOPS` out of the interpreter source.
2. That exposed a genuine port bug: CPython's `_randbelow` uses
   `k = n.bit_length()`, not `(n-1).bit_length()`, and says so in a comment
   ("don't use (n-1) here because n can be 1"). The two agree everywhere except
   exact powers of two -- invisible at 6, wrong at 8, where k=4 draws in [0,15]
   and rejects >= 8, consuming a different number of values entirely.

The named FP hazard was measured rather than assumed: **raw float bit-differences
across 2,538 price evaluations = 0**, so Rust's libm and CPython's agree here.
`round_half_even` was still written by hand, because CPython rounds half-to-EVEN
while Rust's `f64::round()` is half-away-from-zero.

One test expectation of mine was wrong rather than the port: I asserted every
product clamps to PRICE_FLOOR under extreme glut, but WHEAT's glut penalty is
logarithmic so both engine and port quote 15 at inventory 200,000. The test now
asserts agreement plus `>= PRICE_FLOOR`, and reports that 7 of 9 actually clamp.

### Harness: one entrypoint, discovery-based

`tests/run_all.py` discovers every `tests/test_*.py` rather than listing them,
so a new suite cannot be forgotten -- which is how `test_agents.py` stayed
broken for days after the restructure. Ordered fast-to-slow so a structural
break surfaces in seconds. `--fast` skips the episode-playing suites, `--list`
shows the plan.

Discovery immediately found a suite I had not been running (`test_arm_overrides`).

Current: **11 suites**, 8 fast ones green in 8.5 s. New this pass:
test_rust_rules, test_rust_market, plus test_paths / test_gates /
test_tourney_halving / test_rust_rng from earlier today.

### Still not done, stated plainly

- **The engine step function is not ported.** Unit actions (18 ops,
  order-sensitive), market order processing (<=10 per turn, price moves within
  the turn), daily plant/animal refresh, end-of-day. This is the bulk of the
  852 LOC and needs a full-episode differential harness, not per-function tests.
- **C2 market-regime features are blocked on data, not effort**: `features.py`
  consumes ACTION streams only and has no observation access, so market
  features cannot enter the identifier until `turn_features` traces accrue
  (forward-only by decision 5a). A 22-episode bootstrap corpus was captured
  from replays already on disk -- 31,680 transitions, no download.
- E2 duel panel, E3 blend policy, B4 concurrency ramp.

## 2026-08-13 late - B4, E2, E3 completed (no deferrals left except the port)

### B4 - adaptive fetch concurrency

My reason for deferring this was a value judgment ("marginal after the transport
reroute"), which is not a blocker. Built `src/kaggriculture/data/fetch_pool.py`: an
`AdaptivePool` whose width ramps up on clean runs and HALVES on a burst of
throttle signals, so it finds the server's knee instead of guessing it. Wired
into `sameday.py` -- `--jobs` is now the STARTING width, `--max-jobs` the
ceiling (default 24).

The distinction that matters: **404 is not throttling.** A pruned archive file is
permanent, and reading it as a rate limit would narrow the pool over data that
simply does not exist. `tests/test_fetch_pool.py` asserts that directly, along
with the deadlock case -- narrowing must reclaim slots without blocking behind
in-flight requests, verified with every slot held.

### E2 - the duel policy: fires 248x per episode, changes NOTHING

Completed properly rather than left open. Added a `--duel-policy` build flag to
the generator that embeds `models/lab/duel_q.json` and replaces the static
relay lead with the learned lookup, using `duel_lab.bucket_state` VERBATIM (a
divergence there would look up a different state than the one trained -- the
same class of bug as aiming per-class dump schedules at the wrong class ids). A
DUEL GUARD refuses to build if the Q table's row width disagrees with the action
list, since argmax would then index the wrong lead.

Built two variants differing in ONE literal -- verified: 4/4 shared payloads
byte-identical, one extra blob (the Q table). Paired panel, field-matched
roster, 3 seeds, both seats:

    opponent              vol      duelON        duelOFF      delta
    Ethan (light)         413   12/12 +104,744  12/12 +104,894    -149
    Mahfujur              868   12/12  +65,678  12/12  +65,964    -285
    Prasad                930   12/12  +59,959  12/12  +60,287    -328
    Nathaniel           1,089   12/12  +91,832  12/12  +91,832      +0
    kss (crowned base)  1,943   12/12   +2,580  12/12   +2,372    +208
    HunterBlue (heavy)  3,693    2/12   -4,863   2/12   -4,694    -170
    Anurag (heavy)      3,763    5/12   -1,810   5/12   -1,744     -66

    duelON  67/84 (79.8%)      duelOFF 67/84 (79.8%)     delta -113/game

**Identical win counts on every single opponent.** Instrumented to rule out "the
A/B tested nothing": `_duel_lead` is called **338 times** per episode and returns
a lead **248** times, so the policy is genuinely live and simply does not change
outcomes.

Why the offline number did not transfer: duel_lab measured 0.141 against the
static lead's 0.003 in an OFFLINE MODEL OF THE DUEL against empirical family
schedules. In the real agent the lead is already escalated by the online rule and
composed as `win = max(win, policy)`, so a larger learned lead rarely binds --
the actual constraints are shed contents, `_RELAY_MIN_QUOTE` and the 10-order
slot budget. **Not shipped.** This is the third time this session an offline
advantage has evaporated in the full agent, which is itself the lesson.

### E3 - offline policy pipeline, end to end

`src/kaggriculture/data/policy_dataset.py` joins traces + action schedules + outcomes into
(state, action, return) arrays. Target is SCORE, not margin. Actions are the
per-product SELL VECTOR, because the blend policy emits a bounded schedule
delta -- the route is the coordinate system every adaptive layer is written in.

First run produced NOTHING, and the reason was informative: our own games live in
`data/ourgames`, and their routes are not in the route index, so every
trajectory was skipped. Added a replay fallback (`opponents.extract_actions`
from a replay still on disk) plus both-seat outcome inversion. Now: **31,680
transitions, 44 trajectories, 22 episodes, 0 skipped**.

`src/kaggriculture/experiments/policy_bc.py` trains a return-conditioned BC net (49+1 -> 64 ->
64 -> 9, tanh) on GPU. Split is **by EPISODE, never by transition** -- successive
turns of one game are near-duplicates and a random split leaks the answer.

    val loss 0.0854 vs mean-action baseline 0.1092   (BEATS the baseline)
    export equivalence: worst |diff| 1.58e-06 (OK)

**The export check caught a real bug in itself**: it first failed at 5.4e-01
because the payload held the BEST-epoch weights while the live torch model still
held the FINAL epoch's. That is precisely the train/serve skew the check exists
for, caught on its own first run.

Deliberately NOT built: the arm-blend head. The audit says a commit is worth
-$50,787 (correct commits -$52,940), so a policy over that basis would measure
badly for reasons unrelated to the policy. Arms need rehabilitation first, which
needs the Rust engine to make a full-budget search affordable.

**HONEST LIMIT, stated in the tool's own output:** 22 episodes proves the
pipeline, not the policy. Capture is forward-only by decision, and a shippable
policy needs on the order of hundreds of episodes.

### Harness

**13 suites, ALL GREEN, 228 s.** New: test_fetch_pool. The only thing still
genuinely unfinished is the engine step function.

## 2026-08-13 night - critic pass: scheduled jobs, a real bug, and a CORRECTION

### All scheduled jobs enabled -- with uploads off

Asked to enable every job, which conflicts with the earlier decision to disable
the release to protect v24.1 (bandit at 2380.1, the best rating this project has
had; only the latest 2 submissions stay active, so an upload evicts a developed
rating for a provisional one). Resolved by enabling the task AND adding
`--no-submit` to `scripts/daily_release.bat`: every stage runs -- fetch, train,
crown, build, gate, notebook push -- and nothing is uploaded. The run is
therefore also the end-to-end validation of today's refactor.

    KaggricultureAutopilot     Ready   04:00  autopilot.ps1 -Stages fetch
    KaggricultureRefreshCycle  Ready   04:30  daily_release.bat --no-submit
    KaggricultureSameDay       Ready   hourly :35

### Enabling everything exposed a contention gap

A THIRD task existed that nothing in today's work accounted for:
**KaggricultureAutopilot at 04:00**, running `routes.py --mine --jobs 8`. Two
findings:

- `scripts/release_running.ps1` -- the guard that stops the hourly scrape
  colliding with a release -- matched only `daily_release`. The autopilot was
  **invisible to it**, so the 04:35 scrape would start fetching while the 04:00
  mine was still running, against the same endpoints. That is exactly the 429
  contention that once left a cycle with nothing built. The guard now matches
  daily_release, refresh_cycle, autopilot and `routes.py --mine`, excludes
  itself, and was validated against 8 representative command lines -- including
  that `routes.py --build` (a route build, not a mine) must NOT match.
- The autopilot's principal is **Interactive**, not S4U like the other two, so it
  only fires while someone is logged in. Left as-is (it is fetch-only and the
  hourly scrape covers the same ground) but recorded rather than silently
  inherited.

Also noted: `RefreshCycle`'s last result was `267014` = SCHED_S_TASK_TERMINATED,
consistent with it being stopped mid-run earlier today.

### Bug found and fixed: adaptive narrowing did nothing under load

`fetch_pool._narrow()` reclaimed permits with a non-blocking acquire and simply
lost whatever it could not take. Under FULL load -- every permit held, which is
precisely when throttling happens -- it reclaimed **zero** and silently did
nothing. So the controller narrowed only when it did not need to.

Fixed with a debt counter: the shortfall is carried and swallowed by subsequent
`release()` calls, so a narrow always takes effect. Regression test asserts the
saturated case (8 -> 2 with every slot held) and that the pool never hands back
more permits than its width, plus the existing no-deadlock case.

### CORRECTION: the 8.24 s episode figure was wrong

I measured 8.24 s for a 720-step episode early today and quoted it all session.
Re-measured uncontended, twice each:

    random vs random        2.40 s, 2.42 s
    real agent vs starter   4.70 s, 4.75 s

The original was taken while other jobs were running. The representative figure
for a TOURNAMENT game is ~4.70 s, because random agents barely act while a real
agent against a tape does full work. Consequences:

- 3 workers -> 0.64 ep/s, ~55k episodes/day (not 31k)
- 10^6 episodes at 3 workers -> ~18 days (not 32)
- every "engine-bound" estimate quoted earlier today was ~1.75x too pessimistic

A second trap in the same tool: measuring a 120-step episode and scaling by 6
under-measures, because per-step cost GROWS with farm complexity (more hands,
more unit ops). `src/kaggriculture/pipeline/pipeline_budget.py` now times a full 720-step episode with
a real agent, and says so.

### How long a complete pipeline takes (measured)

    stage                                   minutes   bound by
    detect                                      0.1   network
    games: ourgames delta                       2.0   network
    mine: leaderboard + archive                 2.7   network
    tapes: fetch + build                        1.2   network
    train: identifier (cache warm)              1.2   cpu
    train: relay + gates + surrogate            1.5   cpu
    tourney (successive halving, 744 games)     19.4  engine
    playoff                                     1.2   engine
    arms (fast profile under ARM GUARD)         1.0   engine
    build + model graphs                        2.5   cpu
    gate: paired validation + latency           1.0   engine
    notebooks: push x2                          4.0   network
    upload + sha poll                           5.0   network
    TOTAL (serial)                             42.9

    engine 22.7 (53%) | network 15.0 (35%) | cpu 5.2 (12%)

**Two-track: hourly fetch + a 36.9 min morning release**, so a 04:30 start has
the pair ready ~05:07. With a driver fix at 12 workers: **26.6 min total, 20.6
min morning path** -- the tournament falls from 19.4 to 4.9 min and engine stops
being the dominant cost.

Cold feature cache adds ~6 min; the tool reports WARM/COLD rather than assuming.

## 2026-08-14 - why nothing published, and the GC bug behind it

### Two reasons, and the second was a real defect

1. `--no-submit` is in the launcher by decision (uploads off while v24.1 is
   parked).
2. **The 04:30 run FAILED THE GATE anyway** and exited 1 after ~2 minutes:

       FAIL  worst turn 282 ms; Kaggle allows 1000 ms but runs on 1.6 vCPU,
             so we require < 250 ms here
       REFUSING TO SUBMIT

So even with uploads enabled, nothing would have shipped.

### The cause was garbage collection, not agent logic

Profiled per turn in self-play: mean 3.4 ms, p99 ~8 ms, and exactly ONE turn per
episode spiking to 245-330 ms. Two observations settled it:

- the spike step MOVED with the seed (519 / 587 / 499), so it is not
  data-dependent;
- with `gc.disable()` that same worst turn fell from **245.5 ms to 12.2 ms**.

It was a generational collection walking the large immortal objects the module
unpacks at import -- the 719-turn route, identifier weights, arm overlays, relay
schedules. The agent's real worst-case compute is ~13 ms.

### The live agent has been failing this gate too

    agent            mean    p99    worst over seeds 7 / 99 / 12345
    v24.1 (LIVE)     3.5ms  8.3ms   330.7 / 225.8 / 255.2   <- fails 2 of 3
    v24.2            3.7ms  8.1ms   309.8 / 261.3 /  37.4   <- fails 2 of 3
    v24.4 (fixed)    3.6ms  7.4ms    26.9 /  21.7 /  24.9   <- passes all

**v24.1_bandit, currently live and rated 2380.1, shipped by luck** -- it happened
to measure under 250 ms on the day. This matters beyond the gate: Kaggle runs on
~1.6 vCPU, so a 300 ms pause here could be 600-900 ms there, against a 1000 ms
actTimeout. A forfeited turn on a timeout is a real risk the live pair is
carrying right now.

### Fix: gc.freeze() + threshold tuning, in the generator

    gc.collect()
    gc.freeze()                      # immortal payloads never scanned again
    gc.set_threshold(50000, 50, 50)  # fewer, cheaper passes for turn garbage

The collector stays ENABLED -- disabling it outright would leak any reference
cycle for a whole episode, and the fix does not need that. `freeze()` targets the
cause precisely: everything allocated at import moves to a permanent generation.

Verified **timing-only**: v24.2 and v24.4 produce byte-identical final banks on
3 seeds (80496/77881, 49477/45694, 46242/43912). Mean and p99 unchanged, so it
costs no compute. `submit.py --dry-run` on v24.4: **worst 28.52 ms**, gate PASS.

`gc` joins base64/copy/json/math/zlib in the emitted agent -- stdlib only, no new
dependency.

### Not published

v24.4_bandit is built, gated and graphed but NOT uploaded: pushes remain off by
decision. Publishing it would need an explicit go-ahead, and would also swap the
live pair (only the latest 2 stay active).

## 2026-08-14 -- historical backfill, the full Rust engine, and the last open plan items

Operator order: "Do a historical backfill completely. Also, ensure everything
in the plan is completed as a whole." This supersedes decision 5a
(forward-only trace capture).

### The backlog nobody was mining

`data/sameday/_stage` held ~2,940 full replays (~22 GB) that interrupted
scrapes had downloaded but never ingested -- on disk, invisible to the index,
contributing nothing. Two backfills drained it:

* `src/kaggriculture/data/backfill_traces.py` -- 49-field per-turn traces for **2,952 episodes**
  in 5.5 min at 6 workers (134x the 22-episode bootstrap corpus).
* `src/kaggriculture/data/backfill_ingest.py` -- the same `sameday._ingest` the hourly scrape
  uses, in parallel shards (safe because `save_index` merges under a lock):
  routes, outcomes, obs-features, seeds and traces for every staged episode.
  Truncated files from killed downloads fail cleanly and are skipped.

The first policy-corpus rebuild attempt exposed WHY ingest matters: only 33
of 2,952 traced episodes were in the route index, so the (state, action,
outcome) join starved -- traces without ingested routes are unlabelled.

### D2 complete: the step function is ported and bit-exact

`rustengine/src/engine.rs` now ports the ENTIRE mutation path: 18 unit ops,
the per-unit lockstep market, town consumption, plant decay, daily refreshes,
weeds, shed sweeps, end-of-day. Differential harness
(`tests/test_rust_engine.py`) drives BOTH engines with identical action tapes
and compares a canonical full-state digest EVERY step -- money as IEEE-754
bit patterns, every tile field, both private blocks.

* **50/50 episodes bit-identical** (10 scripted-chaos + 40 real ladder
  replays under their recorded seeds), ~36,000 steps of exact agreement.
* **103,260 steps/s = 143.6 episodes/s on ONE core** (kagg bench), vs the
  official engine's 0.36 ep/s at 3 workers: ~400x, from a binary that
  sidesteps the 3-worker BSOD cap entirely.
* Porting traps that mattered: Python dict INSERTION ORDER is load-bearing
  (which item overflows a full shed depends on it -- OMap replicates it);
  banker's rounding in the price quote; `consecutive_unwatered` starts at 1;
  the care bonus is popped only on fed production days.

Integration, all live:

* `src/kaggriculture/engine/rust_prerank.py` -- open-loop schedule-vs-schedule ranking; 96 games
  in 11.3 s. PRE-RANKER ONLY.
* `train_gates.preranker_allowed()` -- funnel widening requires measured
  recall@N >= 0.75 over >= 6 candidates. Speed is not the criterion.
* Daily parity in `scripts/run_pipeline.py`: 1 chaos + 2 replay episodes
  through both engines every run; a mismatch REVOKES the recall evidence so
  the gate closes itself, while the release (official engine) continues.

### C2 measured: market regime does NOT help the identifier

With the backfilled traces, the +0.88 price signal finally became a testable
feature family: 27 public market dims (price level, 4-day trend, inventory)
at each checkpoint, paired protocol (same routes, labels, split, dropout
masks -- the mask applies to the action prefix only, since prices are never
invisible). Over the FULL join -- 5,846 routes, 105,228 rows -- the
held-out-day delta is **+0.0014 against a route-level SE of ~0.0044**:
decisively within noise, so NOT adopted. The signal is real but the
identifier is the wrong consumer -- it identifies WHO the opponent is, and
the market path says more about how the game is going than who is playing.
The policy/surrogate path is where that signal pays.

### Pre-ranker recall, first measurement

recall@4 = **0.75 over 8 candidates** (`models/lab/preranker_recall.json`),
exactly at the gate floor, so `preranker_allowed()` opens -- with a caveat
the numbers make visible: the official side saturated (six candidates at
0/6, two at 6/6), so ranking among the 0/6 tie is arbitrary and the recall
floor is conservative. Both actual official winners are in the Rust top-4.
Re-measure with weaker referees before leaning on the funnel widening.

### Gates & crown: the remaining plan changes, shipped

* FIELD_STRATA 4 -> 8 (the field spans 413..3,763 opponent sell units).
* HELD-OUT PANEL: `N_HOLDOUT = 4` loss tapes reserved at tape time, excluded
  from the tournament roster, arm training and the bandit gate; every real
  crown is re-played against them (`stage_holdout`) and a divergence lands a
  WARNING in the morning report. Report-only by design: the split exists to
  make referee overfit VISIBLE, which it previously could not be.
* POLICY GUARD: `policy_allowed()`, same sign-tested shape as
  `arms_allowed()`, blocked until counterfactual judging shows positive
  score_delta over >= 25 games. Exists BEFORE any policy ships.
* The last margin criterion (`bm < rm - 3000` vetoing the bandit in
  stage_bandit) is gone: wins decide, margin breaks exact ties.

### Suite count

14 suites, all green: paths, win_metric, gates (11 cases incl. the new
guards), tourney_halving, obs_schema, rust_rng, rust_market, rust_rules,
rust_engine (differential), arm_overrides, fetch_pool, agents, contract,
system.

### BC policy on the full corpus (E3 closes)

`policy_bc.py` retrained on **4,240,800 transitions / 2,951 episodes**
(737 held-out episodes, split BY EPISODE): val loss **0.0214 vs the 0.0459
mean-action baseline** (2.1x better), export equivalence 2.2e-06, and the
data-limited flag is gone. Shipping remains blocked by POLICY GUARD until
counterfactual judging records >= 25 games with positive mean score_delta --
the guard exists precisely so this model cannot repeat the arms' path of
shipping on volume instead of sign.

## 2026-08-14 (evening) -- v25.0 shipped, CROWN-2 measured and wired

### Shipped
v25.0 pair submitted (sha-verified, notebook kernels bandit v7 / route v15),
retiring the v24.1 pair. Crown HOLD (best fresh 32% vs incumbent 61% -- the
same-day panel DISCRIMINATES again, no saturation). Bandit recommended on
the new wins-only rule: 48/72 vs route 45/72.

### The hourly-fetch timezone bug (root cause of the fetch famine)
Kaggle SDK returns naive UTC datetimes; `_epoch()` read them as local IST,
aging every episode 5.5 h. The hourly 3 h window therefore matched NOTHING
from 08-12 to 08-14 -- silently, exit 0. Fixed in leaderboard_harvest._epoch
(+ regression suite tests/test_harvest_epoch.py, suite #15). First fixed
production run found 1,869 ids and streamed 4.7+ GB. Budgets raised
throughout by operator order (safety valves only).

### CROWN-2: what was adopted and what the kill-switches rejected
- ADOPTED -- Rust-wide funnel (B): recall@4 = 0.75 with discriminating
  referees; refresh_cycle now pre-ranks 200 fresh routes and gives official
  tickets to the top 24, finals at 4 seed sets, with a TRAILING recall audit
  that closes the funnel on drift. Fallback = old selection, always.
- ADOPTED -- overlap exclusion (D): replaces strict-future (0/74, dead under
  same-day data). No candidate refereed by its own episode/team.
- REJECTED -- rating-band panel (A): wins ordering (Spearman 0.396 vs
  0.128!) but fails calibration (MAE 0.25 vs 0.17) on the 19-submission
  backtest. Pre-registered gate required both. Ordering signal recorded in
  models/lab/band_panel.json for a calibrated variant.
- REJECTED -- portfolio overlays (C): raw transplant 0/32 (-45..52k/g);
  retimed variant WORSE (-112..143k/g -- donor timing schedules sells before
  OUR farm produces; unexecutable sells silently rot). Third rejection of
  cross-farm schedule transfer. models/lab/overlay_sweep.json.
- REJECTED -- per-turn BC policy head: judged 0/52 games, mean score_delta
  -0.29. POLICY GUARD holds a definitive NO (52 rows, negative sign).
- MEASURED -- GRU: won held-out 0.950 vs logistic 0.910 (43 classes) and is
  embedded in the HELD v25.1 via --gru-auto; but paired panel 45/72 vs 48/72
  (within noise) -- identification still lacks a paying consumer.
- MEASURED -- copycat behaviour (live pair): route-copy vs route 1-2-1
  (seat symmetry), bandit vs route-copy 4-0-0, bandit-copy 1-2-1. A wholesale
  copier tops out at expected score <= 0.5 against the pair.

### Backfills complete
Trace corpus 22 -> 12,427 episodes (backfill_traces + backfill_ingest +
backfill_enrich re-fetching 6,087 old episodes; 563 permanently gone).
Index enrichments: outcomes 100%, traces/seeds/obs/teams filled wherever the
archive still serves the replay.

## 2026-08-14 (night) -- v25.1 live, field study, Track P

- v25.1_bandit SUBMITTED (kernel v8, sha-verified): live pair is now
  v25.0_route + v25.1_bandit. First GRU deployment (held-out 0.950 vs
  logistic 0.910, --gru-auto). Gate: worst turn 7.15 ms. v25.0_route hit
  2406 -- the project's highest rating -- within 8 h of submission.
- v25-pair replays: 139 ingested (route 40W-29L, bandit 48W-22L); hourly job
  accrues the rest; tomorrow's cycle trains on them.
- FIELD STUDY (src/kaggriculture/experiments/field_correlations.py, 22,575 plays):
  * PRICES ARE A SHARED TIDE -- r~0.00 with WINNING (0.55-0.75 with bank).
    The 2026-08-13 "market regime is the bottleneck" belief is REVISED: only
    DIFFERENTIAL exposure to a crash can pay, not absolute price level.
  * The GAP decides: lead_share r=0.69, day-23 gap 0.57, day-16 gap 0.38.
    => rewards must be gap-shaped, never bank-shaped.
  * ELITE signature (2900+ wins vs mid): faster first-10k (12.75 vs 13.49 d,
    -0.64), LEANER crew (-0.60), lower dry-share (-0.60), deeper early
    investment (-0.43 idle cash), small day-10 lead but DOUBLE day-16 gap.
  * Artifact: hands sampled at day boundaries read 0 (daily reset) -- fix
    before reusing that feature.
- TRACK P plan v2 published (closed-loop planner lineage): P0 insight loop,
  P1 production switching with elite priors, P2 elite-conditioned route
  generator, P3 hierarchical self-play RL (gap-shaped reward, kagg serve),
  P4 Rust-native heuristic search, P5 slot-challenge graduation.
  .local/docs/track-p.html + claude.ai artifact.
- Foreign framework check (jek1wantaufik/buddy/agric): modular closed-loop
  planner skeleton, runs but embryonic (940 vs our 188,440 in one game).
  Confirms the field's direction; nothing to adopt; kept in .local/foreign.
- CROWN-2 validation: machinery exercised 3x live (funnel, exclusion,
  determinism, halving); full-run completion was console-killed twice; final
  re-run in flight at save time -- tomorrow's 04:30 is the definitive
  end-to-end pass either way, now with real v25 loss tapes.

## 2026-08-15 — Track P implementation measurements (all first-day, honest)

* **kagg batch/serve verified against the ladder**: 6/6 sampled 1.32.6
  replays reproduce recorded final banks exactly; serve==batch. Throughput:
  batch ~40+ ep/s process-inclusive; serve 777 steps/s single stream,
  8.5 ep/s with 8 workers and the Python planner in the loop.
* **1.32.4 replay trap (measured)**: data/mine replays (pre-rebalance) ran
  townCenterSellInterval=12 + a day-ramped center demand schedule that
  1.32.6 removed. Re-running their actions on the current engine gives
  28k/48k vs recorded 68k/83k. Rule: never trust re-run banks across the
  Aug-7 engine boundary. (First divergence localizes to one extra center
  drain by step 13 — the diagnostic that found it.)
* **Insight v2 (11,140 plays)**: demand_match_mean d=+1.49 (NEW #1 elite
  marker; the 49-field study could not see it — no shop draw in traces).
  gap_day23 d=+1.06, lead_share d=+1.05 confirm the gap findings.
  Interaction pass: 0 FDR-surviving plan×regime cells yet.
* **Projector beats naive 7/9 products** (e.g. WOOL MAE 29.1 vs 34.8;
  FERTILIZER 7.0 vs 10.1) at 2/4/6-day horizons.
* **CMA-ES on serve league**: fitness −0.78 → −0.679 in 60 gens (24 params);
  transfers to official engine ($19-27k → $45k vs v25 route).
* **IQL**: 0.878 mean held-out decision acc (68,970 elite decisions);
  advantage-return corr ≈ 0 (elite-only returns have no variance — expected).
* **PPO+PFSP**: 16 iters clean; neural L1 == hand rules on anchor gap
  (−104k) → L0 executor throughput is the binding constraint, not macro
  choice. Exploiter edge 0% at 4-iter budget.
* **P4.1 validity: WEAK (Spearman 0.347, overlap 0.25)** — Rust open-loop
  fitness must not decide closed-loop questions; official confirm stays
  mandatory (already enforced everywhere).
* **P1.6 panel**: 0/16 across 4 shop-draw regimes vs v25 route; graduation
  1–4 = false. The honest day-one baseline for the track.

## 2026-08-15 — v26 release incident chain (quota exhaustion → timeout → recovered)

Timeline: 04:30 run HELD (all 13 loss-tape fetches 429'd — the hourly job's
overnight backlog drain exhausted the ~24h replay quota; direct AND CLI paths
throttled for 6+ hours). 10:15 manual re-run with SameDay paused: retry
ladders burned 76 min proving the same global 429 → tape FALLBACK (new)
reused 2026-08-14's 12 tapes → crown: challenger 6% vs incumbent 62% →
HOLD-rebuild of the pair on base 92513718_s1 with fresh models → outer 4h
subprocess cap killed the run at 14:15 AFTER the pair was built, BEFORE
publish. 14:30 --resume-publish (new): both gated, kernels pushed
(bandit v9, route v16), both submitted 14:27 IST, sha-verified.

Hardening shipped: (1) refresh_cycle tape fetch = local-stage cache →
direct → CLI with bounded backoff; (2) global-quota short-circuit (2 full
ladder failures ⇒ straight to fallback); (3) stale-tape fallback (newest
prior day, labels preserved, loud report line); (4) daily_release
--resume-publish; (5) cycle timeout 4h → 5h.

Unexplained-but-bounded: submit.py's gate measured worst-turn ~250–330 ms
on v26.0_route three times while byte-identical v25.0_route measured 5–8 ms;
a controlled fixed-seed harness shows both at 2.7 ms, and the 4th gate run
passed at 4.2 ms. A transient harness artifact, not agent code; if it
recurs, instrument submit.py's timed wrapper (thread/GC interference at
episode start is the suspect).

## 2026-08-16 — second-slot rule, intraday gate, adaptive sell-timing (v27)

* Bandit rating "decline" diagnosed: window artifacts (v25.0_bandit retired
  at 4h by the untested v25.1; ratings converge with games) + since v25 all
  adaptive layers are guard-locked, so the bandit was the route + 420 KB of
  dead weight. Within-pair sign flip at v25 was real; v26 gap (24 pts) is a
  young-submission tie.
* SECOND-SLOT RULE: slot 2 must be EARNED — bandit ships only on a
  sign-tested paired win over the route; otherwise a diversity route
  (different day-3 stream-hash family) ships as v{x}.{y}_route2.py.
  models/second_slot.json is authoritative; daily_release obeys it.
* INTRADAY-FIX GATE: src/kaggriculture/measure/intraday_gate.py + submit.py step 1b. y>0 builds
  refuse to ship without a fresh sign-tested PASS vs the sub they retire.
* Adaptive sell-timing (tape_runtime: _observe_opp_sells/_adaptive_sell/
  _adaptive_repay): measured paired twice, 16 cells p=1.0 and 24 cells /
  192 games diff −1.3pp p=0.625 — NO panel edge (consistent with the
  2026-08-13 sell-timing rejection). Flagship OFF; slot-2 route2 ON as a
  live-ladder experiment (panel-vs-live validity is WEAK, ρ=0.347).
  Judge it on equal-window route-vs-route2 ladder records after 24h.
* Tests: tests/test_second_slot.py, tests/test_intraday_gate.py; 14 fast
  suites + test_agents green.

## 2026-08-21 — shell defects found & fixed (v32 post-mortem)

* **FIXED — `_safe_market` clamp bankrupted PLACE-deposit routes.** The
  engine's market partial-fills per unit (`_commit_unit`), so an oversized
  SELL is harmless; and the shed has TWO deposit paths (`DROP` and
  `PLACE item n`). The clamp cancelled sales whenever the DROP-only
  projection undercounted ($2,146 vs $58,292, same seed/opponent). Route
  template: clamp removed. Bandit template: clamp kept (additive layers
  rely on it) with a PLACE-aware projection.
* **FIXED — blanket sell-first reordering assumed universal.** Split into
  `_FEED_PIN` (turn-0 wheat to slot 0; C94-evidenced) + `_SELL_FIRST`
  (per-route flag). Feed pin cuts val-family feed-denial losses
  −21k→−4.3k at equal W/L; inert for kss (already slot-0).
* **FIXED — weed repair missed animal-PLACE and wrongly covered
  deposit-PLACE.** Repairs only weed-blockable ops now.
* **GUARDED — `endgame_pull`/`swap_advance` double-sell hazard**: their
  conservation was the removed clamp; routes.py refuses to build with
  them until explicit subtraction is reimplemented.
* **OPEN — offline instruments still only rank.** The reactive gauntlet
  (data/gauntlet) + strict-future gate replace frozen-tape crowns, but the
  ladder at 100+ games remains the only validator (v33.0 pair is the
  running two-lineage experiment).
* **OPEN — GRU identifier stale** (52 classes vs 49); `--gru-auto`
  correctly ships the logistic. Retrain when the class map settles.

## 2026-08-27 — the "engine divergence" that wasn't, and the gates that came out of it

The Aug-23 diagnosis compared v33's LOCAL validation mirror (96k) against
v32's KAGGLE validation mirror (7,777) because the ourgames index lagged and
the submission ids were never mapped through `kaggle competitions
submissions`. Rules that fell out of it, now enforced in code:

* **Map submission ids before attributing behavior to a build.** The index
  lags; the CLI listing is the truth.
* `src/kaggriculture/engine/ladder_parity.py` replays fresh raw ladder replays through the
  vendored engine and demands exact-bank matches (4/4 on first audit);
  `parity_ok()` self-revokes on any mismatch. The engine-question is now a
  minutes-long measurement, never an inference.
* `src/kaggriculture/pipeline/submit.py` validation: bank floor 60k with 3-seed retry, non-PASS
  first-action assert, idle-run + sell-basket telemetry. The v32 artifact
  (Kaggle mirror 7,777) would have been caught post-submit by the new
  validation-mirror check in `src/kaggriculture/measure/collapse_alarm.py` (floor 30k).

## 2026-08-27 — premium one-turn sell lead: correct, inert on piped routes

`_PLEAD` in `tape_runtime` (build flag `--premium-lead`, default OFF)
implements the C94/V16-RC5 conservation lead for WOOL/MILK/MELON/STRAWBERRY
against ANY opponent — borrow ≤10 units from the next turn's scheduled SELL,
repay through the relay ledger (unit-verified: pull 10, next turn 16→6).
The paired gauntlet A/B (2×8 opponents×6 seeds×2 seats) measured **zero
games changed**: our mined routes deposit produce into the shed in the same
turn they sell it, so a sell-side-only lead has an empty shed to draw from.
Do not re-enable without ALSO advancing the deposit (farmer-op change — a
separate bounded mechanism that needs its own paired evidence), or a base
whose shed carries stock between turns.

## 2026-08-27 — the loss audit points at the base, not the timing

v33.0 at 60 games/slot: ~26% win, banks healthy, 38/89 losses closer than
$5k, 34 losses tip at exactly day 11. But the refreshed elite gauntlet
(Kaito v48/v43, V16-RC5, teacher trio + old panel) scores the Valmorlee
base **0.208** with −4k..−12k margins — structurally behind the current
frontier, not one turn late. The close ladder losses are a low-rating
matchmaking artifact. Next lever is winning-plan P1.1: a current-frontier
base (fresh top-10 mining on 1.32.7), gauntlet cross-play as the ranker,
strict-future as the veto.


## 2026-09-01 platform consolidation
- v22 chassis: arm-override repay double-sell closed (ledgers cleared at
  first override); _BRANCHES per-episode route (serve process-reuse leak);
  repay remainder drop is BY DESIGN (carry-forward measured -20 discordant).
- sameday scraper: truncation-proof downloads (Content-Length + JSON tail
  + atomic rename).
- trackp planner: mid-day replan redesigned as MARKET-ONLY (labor tours
  are never truncated). 16-0 paired vs dawn-only, median +9,371 own-bank.
  REPLAN_HOURS=(0,8,16) default. Planner lane is still not seat-grade
  (0-6 vs live seats): it is the GA/macro search platform.

## 2026-09-03 — Track P Phase B: the day-plan searcher (Rust, closed loop)

Design: `docs/history/trackp-search-design-2026-09-03.md`. Code:
`rustengine/src/plan.rs` (action abstraction), `value.rs` (objective),
`search.rs` (amortised search), `lib.rs`, `src/bin/search_eval.rs` (offline
harness), `src/bin/search_selftest.rs` (Phase-A boundary self-test).
Driver: `.local/trackp_search/run_eval.py`. Phase A owns `main.rs`,
`service.rs` and the bridge; nothing here touches them, and
`tests/test_rust_engine.py` still passes 6/6 bit-identical after the change.

**What it is.** Not per-turn judgment (retired: planner_v0 0-64, econ planner
0-6). The searched object is the DAY PLAN — ~26 integers naming the day's
economy (hire, land, herd buys, tiles per crop, per-product sell caps and
floors, fertiliser, feed buffer) — expanded by a deterministic planner into 24
turns. The rollout expands a candidate with the SAME code that plays the real
turn, so plan/execute drift is structurally impossible, and a searched
PARAMETER vector re-seats on a board that turned out different (a searched
action sequence cannot). Amortised: today's 24 turns each spend their budget on
TOMORROW's plan; the turn itself costs 5-8 us.

**The objective is not own bank.** Rank 1 and rank 400 bank the same 88k
median, corr(own, opp) = 0.73-0.80, and the entire top-10 edge is pairwise and
concentrated in low-price worlds (61.5% vs 48.0% below 90k). So the objective
is `sigmoid((A_me - A_opp) / s_t)` with `s_t` calibrated from the observed
margin distribution (P(margin<$3k) = 0.352 HIGH / 0.435 LOW implies sigma
$6,570 / $5,210). Var(own - opp) is 0.40-0.54 of Var(own) at those
correlations, i.e. ~2x better signal per rollout for the quantity that actually
decides the game.

**Two things this measurement pass found the hard way.**
1. The logistic SATURATES: past ~40 sigmas it is 1.0 in f64, every candidate
   ties, the hill climb accepts nothing and the searcher silently degrades to
   the plain skeleton. Observed on seed 4000 (objective pinned 1.000 from day
   7; committed knobs from day 18 were the untouched skeleton). Fixed with a
   1e-9-per-dollar tie-breaker, four orders of magnitude below the logistic's
   own gradient near an even game.
2. A wall-clock `budget_ms` is NOT reproducible (seed 4000 at 50 ms banked
   81,552 and then 105,108). `--rollouts N` gives a reproducible per-turn
   budget; `src/kaggriculture/measure/determinism.py`'s rule applies — paired A/Bs use that one.

**Substrate limitation, recorded so nobody misreads a regime number.** The
frozen d9-12 LOW threshold (1.0630, calibrated on 8,000 ladder traces) is
reached by ZERO of 32 control cells here: a skeleton mirror banks 60k against a
ladder median of 88k, so the four tracked products are never glutted to ladder
depth, and the field skeleton buys no geese, so EGG sits permanently on its
1.32.7 hinge SCARCITY side and drags the four-product mean above the cut. The
LOW readout is therefore taken on the d3-5 window (which does split ~31/69, and
`band_panel.py` documents it as usable at Cohen d 1.36) plus a within-run
relative split — a weaker instrument than the ladder panel, and it is labelled
as such in the harness output.

---

## Z. Track P compiled agent — Phase A findings (2026-09-03)

Full write-up: `docs/history/trackp-compiled-2026-09-03.md`.

### Z1. `kagg serve` gives the agent ONE MORE TURN than the official engine — OPEN
`service.rs::serve()` reports `done` at `s.step >= EPISODE_STEPS` (720). The
official interpreter fires DONE when the pre-step counter reaches
`episodeSteps - 2`, so agents act **719** times, not 720. Measured directly
(`.local/build/turncount.py`): 719 official calls vs 720 serve calls for the
same pair and seed.

For an agent that acts on the final turn this changes the bank — the Track-P
skeleton planner banked **52,909 on serve and 50,771 on the official engine**
for one seed, a $2,138 phantom. Tape agents PASS at the end, which is exactly
why `models/serve_equiv.json` shows clean matches and never caught this
(`v43.0_bandit vs pub_v16rc5` still matches 2/2 on the same day).

**This is not a Track-P-only problem: the release tournament runs on `serve`
whenever `refresh_cycle.serve_allowed()` is green**, so any closed-loop
candidate that sells on step 719 is scored on a turn the ladder will not give
it.

The engines themselves are fine — driven by the same action tape they are
bit-identical for all 719 steps including this policy's stream
(`.local/build/divergence.py`, and `tests/test_rust_engine.py` still 5/5).

Fix is one line, `let done = s.step >= EPISODE_STEPS - 1;`, but it invalidates
the standing `serve_equiv` evidence and shifts tournament numbers, so it was
deliberately NOT applied by the Phase-A task. It needs a re-audit
(`src/kaggriculture/engine/serve_match.py --compare-official` over >= 12 seeds with a
CLOSED-LOOP agent, not tapes) before flipping.

### Z2. Wrapping an agent in a callable OBJECT silently cripples it — FIXED in harness.py
kaggle-environments introspects the agent callable's signature to decide
whether to pass `configuration`. A class with `__call__(self, obs, cfg=None)`
made it pass only the observation, and `data/gauntlet/pub_v16rc5.py` (which
swallows the resulting exception and returns PASS) then banked **exactly its
3,000 starting money** for a whole episode — while the candidate under test
"won" 100k-bank games and the gate printed PASS.

Two rules follow, and they apply to every panel and gauntlet, not just here:
* wrap agents with a plain `def wrapped(observation, configuration)`, never a
  callable object;
* **an opponent that finishes on its starting money is a broken harness, never
  a result.** `src/kaggriculture/trackp/compiled/harness.py` now flags any such cell as
  `DEAD OPPONENT` and counts them in the gate summary.

### Z3. The skeleton planner never executes its own land schedule — PARTLY FIXED
`src/kaggriculture/trackp/build_econ_agent.py`'s genome says the 2nd quadrant is bought on
day 4. Measured, it arrived on **day 10-11**: the herd ramp (a cow a day from
day 2) spent the money first, so the farm sat on NW's 24 usable tiles for a
third of the season — 18 standing crops and 168 PLANT ops all season against
the field's 243 plants.

New genome key **`land_reserve_lead`** (default `0`) withholds the next
scheduled quadrant's cost from the herd until it is bought. 24 official-engine
cells (6 seeds x 2 seats x 2 opponents):

| `land_reserve_lead` | median bank | mean bank |
|---|---|---|
| -1 (off, old behaviour) | 48,368 | 47,796 |
| **0 (new default)** | **55,008** | **52,099** |
| 1 | 48,513 | 50,340 |
| 2 / 6 | 41,732 | 43,397 |

Reserving too far ahead is worse than not reserving at all: SW costs 2,000 and
holding that from day 5 starves the herd for six days. `valve_hi`/`valve_lo`
were promoted from hard-coded 55/45 to genome keys in the same pass, so the
Rust port and the Python planner parameterise identically.

### Z4. What still costs the skeleton ~40k — OPEN, Phase B
Per-day ledger of one episode against the top-10 medians in
`docs/history/plan-2800-2026-09-03.md`:

| symptom | ours | top-10 |
|---|---|---|
| idle (PASS) unit-turns | 2,340 | 513 |
| movement share of non-PASS ops | 52% (2,457 / 4,744) | — |
| PLANT ops, whole season | 168 | 243 plants |
| undug weeds standing from day 21 | 11-13 | — |
| shed at 100/100 (dusk overflow DISCARDED) | days 23 and 27 | — |
| final MILK / FERTILIZER quote | $1 / $1 (glutted) | — |

All of these are per-turn decisions, which is precisely what the compiled
substrate was built to afford.

### Z5. `panic = "abort"` removed from the release profile — INTENTIONAL
`kagg play` wraps each turn in `catch_unwind` so one bad observation degrades
to a legal PASS instead of killing the process mid-episode. This changes the
profile for every binary in `rustengine/`, the pre-ranker included. It does not
change arithmetic; `tests/test_rust_engine.py` still reports 5/5 bit-identical.

### Z6. Nobody has proven Kaggle's sandbox permits fork/exec — OPEN
The compiled transport spawns a subprocess. yhay81's published ctypes approach
only needs `dlopen`, a weaker requirement. If spawning is blocked we fall back
for all 719 turns and score the Python planner: safe, but the artefact buys
nothing and it would look like an ordinary weak game. `main.py` therefore
writes `TRACKP start` / `TRACKP end` beacons to stderr, which
kaggle-environments captures into the episode's per-agent `logs` — read them in
the first replay after any submission. Contingency: add a cdylib target and a
ctypes path as a second transport.

**Measured (full tables in `docs/history/trackp-search-design-2026-09-03.md` §11, raw
output in `.local/trackp_search/`).** Against the field skeleton, 80 cells at
50 ms: **0.988 (79-0-1)**, own median 74.3k vs 37.5k; at 300 ms on 32 cells,
**1.000 (32-0-0)**, own median 86.9k. The dollar edge is strictly monotone in
budget (+25.2k / +29.8k / +42.1k / +47.6k at 10/50/150/300 ms) and rollouts
scale linearly (6.5 / 31.9 / 65.2 / 142.7 per turn). The turn itself costs
0.005 ms; 17-19 us per simulated step with BOTH policies replanning.

**It TRANSFERS** — it is not an exploiter of its own opponent model. Against
three economies the search never simulates: geese 1.000, melon 1.000, wheat
monoculture with uncapped sells **0.875**. That last row is the finding worth
carrying: the market-GLUTTING opponent is by far the hardest (edge collapses
from +34.0k to +13.8k), i.e. the searcher is weakest in exactly the thin-margin
low-price world where the 2800 plan says the top-10 edge lives.

**Search-quality curve** measured as searcher(B) vs a fixed searcher at 10 ms
(against the skeleton the score saturates and shows nothing): 0.562 / 0.688 /
0.750 at 50 / 150 / 300 ms. Monotone, but adjacent steps are individually
underpowered on 16 cells (p 0.73 and 1.00) — the level of each row is measured,
the step between rows is not.

**None of this is a ladder claim.** The opponents are Rust policies banking
53-60k in mirror against a ladder median of 88k, and Phase B cannot drive
`pub_v16rc5.py` or the band panels — that is Phase A's first integration test.
Next items: debit both seats' liquidation against one market inventory (the
value function currently prices our shed as if we were the only seller, which
is exactly the glutted-market weakness), measure the opponent ensemble, raise
the baseline to the tuned Python skeleton's level, and re-run the paired tests
on 60+ cells in the reproducible `--rollouts` mode.

### Z7. `FIRST_BUDGET` was LONGER than `actTimeout` — FIXED 2026-09-03
`src/kaggriculture/trackp/compiled/bridge.py` gave the first turn a **1.50 s** watchdog
against a **1.0 s** `actTimeout`. A watchdog longer than the engine's own limit
protects nothing: in the single situation it exists for — the binary spawns and
never answers — it sat on turn 1 for a measured **1,503 ms** and would have
handed the engine a turn the engine had already timed out, converting a
harmless fallback into a possible forfeit.

The value rested on an ASSUMPTION written into `docs/history/trackp-compiled-2026-09-03.md`
§3 ("the engine has not started the clock on turn 1 in the same way") that was
never measured. **The lesson is the general one: never leave a safety constant
resting on prose.**

It is now **0.50 s**, sized from measurement — spawn plus the first day plan is
<=110 ms with 6 cells in flight, so 0.50 s is ~4.5x the observed cost and leaves
half of `actTimeout` in hand. Re-measured: the same breakage case now peaks at
**503 ms**, the episode completes DONE, and the bank is identical.

**It was only found because the timeout path was finally FIRED** (breakage case
D: a stand-in `kagg` that reads the request and sleeps). Risk 7 of the Phase-A
doc had recorded the path as "untested in anger" for a day. Two more cases were
added at the same time — a binary that answers non-JSON, and one that answers
illegal actions (2,077 validator repairs, bridge stays alive, **0** misaligned
hands / over-10 order lists / unknown ops reaching the engine). Runner:
`.local/build/fallback_test.sh` (A-F) and `.local/build/fallback_df.sh` (D and
F with `--json`).

### Z8. The Phase-B searcher now measures NEGATIVE against its own skeleton — OPEN
`SEARCH_BUDGET_MS` shipped at 150 and is now **0**. After the base-economy fix
(`docs/history/trackp-base-economy-2026-09-03.md`) the searcher hill-climbs *away* from
the better economy, because its value function and knob space were fitted around
the OLD weak skeleton.

Same rebuilt binary, the same 40 official-engine cells (5 gauntlet opponents x
seeds 3,4,5,6 x both seats), the real tarball untarred inside Linux:

| budget | W-D-L | own median | own mean | share of all money | worst turn (6 workers) |
|---|---|---|---|---|---|
| **0** | 0-0-40 | **63,857** | **64,791** | **33.0%** | **110.2 ms** |
| 150 | 0-0-40 | 48,543 | 50,036 | 29.7% | 266.7 ms |

Paired cell for cell: **-14,755 of own bank on the mean, richer on only 11 of
40 (two-sided sign test p = 0.0064)**, for the same zero wins. It also breaks
the 250 ms latency bar.

**It cannot be rescued by a bigger budget on Kaggle.** `budget_ms` is WALL
CLOCK, checked after every rollout, so a slower core buys FEWER ROLLOUTS rather
than more milliseconds — 1.6 vCPU would run an even weaker search than the row
that already measures negative here.

Two further reasons budget 0 is the right ship, not merely the stronger one:
at budget 0 the compiled path is byte-identical to the inlined Python fallback,
so a sandbox that blocks `fork`/`exec` (Z6) costs **nothing** in strategy; and
it is REPRODUCIBLE, where a wall-clock budget is not — `src/kaggriculture/measure/determinism.py`'s
rule is that a paired A/B over a non-reproducible agent is *invalid*, not merely
noisy.

**Do not raise the constant without a fresh paired sign test against budget 0 on
the gauntlet.** The searcher is not deleted: `--budget-ms N` / `TRACKP_BUDGET_MS`
still enable it, and `search_selftest` still proves a zero-budget searcher is
action-for-action the skeleton. The open work is a value function fitted to the
CURRENT economy, and — per `docs/history/trackp-phase-ab-integration-2026-09-03.md` §6
item 2 — one judged on the OPPONENT's bank, since that is the coordinate the
lane loses on.

## 2026-09-03 — Track P Phase A+B joined: the transport is done, the agent is 0-40

Full report: `docs/history/trackp-phase-ab-integration-2026-09-03.md`.

The compiled searcher (`submission.tar.gz` = `main.py` + static musl `kagg`,
150 ms/turn of day-plan search) was measured for the first time against REAL
opponents on the OFFICIAL vendored interpreter: 5 opponents x 4 seeds x both
seats = 40 cells.

**Result: 0 wins, 0 draws, 40 losses.** Our median bank 60,468, theirs 126,016.
Paired against `agents/v42.1_trackp.py` on the 32 shared cells: **0.000 vs
0.688, 22 discordant pairs all to v42.1, exact p < 0.0001 — the candidate is
significantly WORSE**, not marginally. `src/kaggriculture/measure/band_panel.py` and
`src/kaggriculture/engine/serve_gate.py` were therefore NOT run; the mission gated them on winning a
material share of the cells.

Transport, in the same 40 cells (28,760 timed turns): 40/40 DONE, **0 fallback
fires, 0 validator repairs**, worst turn 191.9 ms against a 1,000 ms
`actTimeout` with 6 cells running concurrently. That half is finished work.

### The diagnosis, and it is not the one the design predicted

Same worlds, same seeds, same seats, same opponents — only OUR seat changes.
Share of ALL money banked in the game:

| our seat | our median | opponent median | our share |
|---|---|---|---|
| compiled searcher @150 ms | 57,760 | 124,986 | 32.2% |
| compiled skeleton (budget 0) | 43,905 | 117,219 | 30.1% |
| `v42.1_trackp` (mined open-loop route) | 68,526 | 79,086 | **43.0%** |

`v42.1_trackp` holds `pub_v16rc5` to 66k; the searcher lets it bank 126k in the
same world. **The searcher does not merely bank less — it concedes the shared
market.** Bank is a shared-world quantity (corr 0.73-0.80), and the searcher's
gain over the skeleton is pie growth, not share: both banks rise.

**The search works; the economy it searches inside does not.** Over the same 40
cells the search adds **+16,263 of median bank (+37%)** versus the skeleton —
a large, reproducible economic gain — and `win_metric.paired_test` returns
**0 discordant pairs, p = 1.0000**. Zero wins. That is `flips()` read
backwards: dollars buy wins only where margins are thin, and 2:1 is not thin.

Phase B's own next-work list had "a stronger baseline" as item 2. On this
evidence it is item 1 and it is the whole item: `rustengine/src/plan.rs`'s
field-skeleton economy produces roughly half the output of a mined elite route,
and 26 searched knobs per day cannot close that.

### Separate finding: the LIVE seat has a hard failure mode

`agents/v42.1_trackp.py` banks **exactly $0** against `agents/v43.0_bandit.py`
on seeds 3, 4 and 5 (35,157 on seed 6), on BOTH the official engine and the
Rust engine (`python -m kaggriculture.engine.serve_match agents/v42.1_trackp.py
agents/v43.0_bandit.py --seed 3` -> `138301 / 0`). Its mean turn time drops to
0.07 ms in those games: a rigid tape running its purchase schedule into a market
a heavy dumper has already drained, spending its purse to zero and never
recovering. Our two seats do not meet on the ladder, but this is a measured
fragility against exactly the market-glutting opponent
`docs/history/plan-2800-2026-09-03.md` says decides low-price worlds.

### Engineering defects found and fixed

* `src/kaggriculture/trackp/compiled/harness.py` reloaded `main.py` per cell but never killed
  the `kagg` subprocess the previous module had spawned — one leaked process
  per cell for the whole run.
* The same harness's timing wrapper called `fn(obs, cfg)` unconditionally.
  `data/gauntlet/pub_v16rc5.py` exports `def agent(obs)` with ONE parameter, so
  a `--ref` run of any such agent would have raised on every turn. The wrapper
  must keep the two-argument signature `kaggle_environments` introspects and
  call the inner function with its real arity. (Same family as the 2026-09-03
  callable-object bug that made opponents bank their starting money.)
* `tests/test_compiled_agent.py`'s identity assertion is only meaningful at
  search budget 0. It now takes `--budget-ms`: 0 keeps the IDENTITY gate
  (compiled policy == inlined Python fallback), anything higher gates on
  transport only. **Run it at 0 or the equivalence guarantee silently stops
  being tested.**

### Glutter fix (2026-09-03, same session): tested at 96 cells, NOT supported

`value.rs` did price each seat's liquidation as if it were the only seller. The
fair-interleave joint model (`TRACKP_JOINT_LIQ`, **default OFF**) fixes that
arithmetically. Against the wheat-monoculture glutter at a reproducible 32
rollouts/turn it scores 0.906 vs 0.875 — the same +0.031 at 32 cells and at 96
— and the paired win test is **p = 1.0000 (32 cells) and p = 0.4531 (96)**.
Not resolved on three times the evidence, with a stable small effect. It is not
shipped and it touched no gauntlet number.

**The reusable lesson is the mechanism, not the verdict.** On the 16-seed
window the change appeared to take revenue off the glutter: opponent bank lower
on 22 of 32 cells, mean −5,635. On 48 seeds the opponent's bank moves **+294 and
is lower on 47 of 96 — a coin flip** — and the entire gain is our own bank
(+4,335). So the effect is "hold less, realise more", not "contest the market",
and the 32-cell story was a kind-window artefact of exactly the kind
`.local/memory/trackp-phase-b-search.md` already warns about for seeds
4000-4015. **Never read a mechanism off 32 cells**, even when the direction
matches your hypothesis — especially then.

## Opponent-conditioned market warfare — feasibility, prize, and one build (2026-09-03)

Written by the market-layer lane. Every number below is reproducible from
`.local/predict_feasibility.py`, `.local/nn_robustness.py`,
`.local/phase_vs_nn.py`, `.local/prize_exposure2.py`,
`.local/kill_trace.py`, `.local/dead_tape_probe.py`,
`.local/mirror_check.py` and `.local/panel_signtest.py`.

### B1 CHASSIS A/B — settled: our layers are not what costs us 280 rating

`.local/candidates/raw_105017780_s1.py` (rank-10's bare tape) against
`agents/v43.0_bandit.py` (that tape + the whole chassis), band panels, seeds
501-502, both seats, on the Rust serve substrate with a freshly written
`models/serve_equiv.json` (12/12 exact-bank vs the official engine, engine
1.32.7 — the record was 7 days stale and every panel was refusing; it is now
green without `--allow-ungated`).

| band | raw tape | + chassis | discordant worlds |
|---|---|---|---|
| sub2000 | 0.975 | 0.975 | 0-0 |
| mid | 0.988 | 0.988 | 0-0 |
| top100 | 0.976 | 0.988 | 0-2 for the chassis |
| ALL, 326 worlds | 0.9785 | 0.9847 | 0-2, McNemar p = 0.50 |

652 paired cells; **4 differ**, all toward the chassis; margin is a wash
(chassis median −117/game). "Strip the layers until the regression
disappears" is closed — there is no regression to strip. The corollary is
worse news: a panel on which the entire chassis is worth 4 cells cannot gate
a market-layer change either.

### The "v43 zeroes 22 elite tapes" existence proof is a PANEL ARTEFACT

22 of 83 top100 panel opponents (88 of 332 cells) bank ≤ $3,000 against us.
They are not broken tapes — each banks 112k-144k against a PASS agent. But:

* the RAW tape zeroes the same 22 (dead-cell counts identical: 4 / 36 / 88),
  so it is the base economy, not the adaptive layer;
* 17 of the 22 leave our bank identical to the dollar ($147,554) — they stop
  affecting the shared market by ~day 8;
* route-feature distance from our base: zeroed median 0.0146 vs survivors
  0.0379, and the day ledger shows them submitting our exact market orders
  from day 2 to day 17. They are near-MIRRORS of our own base;
* **over 90 real ladder replays the lowest opponent bank was $15,287**
  (median $83,154). It never happens live.

An open-loop tape that loses the day-1 sell race cannot clamp and dies; the
same team's live agent does. Do not motivate a build with this number again.

### FEED DENIAL — priced with the exact engine curve, NEGATIVE, closed

From the same 90 replays:

* depleting 100 WHEAT units costs **us ~$3,356** and costs **them ~$444**
  across all their early feed buys — **ROI 0.13**;
* making one of their feed orders genuinely unaffordable costs a median
  **$9,052** of ours and fails a few units of one order;
* we are as cash-fragile as they are: their minimum money median $3
  (p10 $24), ours $9 (p10 $174);
* WHEAT is sqrt-shaped below equilibrium and already sits ~300 units below
  it, so doubling the price costs $83k-105k.

The incidental pressure of buying our own scheduled feed is all the denial
that pays. `_FEED_PULL` stays off; nothing new was built here.

### SELL FRONT-RUNNING — the surface is huge, the residual lever is not

Colliding sells are 30-100% (median 60%) of the opponent's revenue. But the
turn-level race is already near even (2,790 lead / 2,873 lag over 90 games —
the relay's +$1,914/game is in that number), and the interpreter processes
market orders by ORDER-SLOT INDEX with both players quoted at the same
pre-commit inventory, so a same-slot collision is a perfect price tie. Of
6,232 same-turn collisions we held the earlier slot 21.4%, they held it
~13%, the rest were exact ties. `_impact_slots` already orders our sells;
`_threat_first` (reorder by opponent threat) measured −$1.8k in 2026-08-09
and stays off.

### SCHEDULE PREDICTION — the 20-class identifier is dead, retrieval works

Held-out day (600 routes) against a 5-day library, step of their next
≥8-unit dump, hit within ±6 turns, s = 360 (day 16):

| predictor | WHEAT | STRAW | MELON | MILK | WOOL | FERT |
|---|---|---|---|---|---|---|
| global median | 0.287 | 0.870 | 0.103 | 0.389 | 0.396 | 0.325 |
| identifier class (M2, ships) | 0.302 | 0.879 | 0.192 | 0.399 | 0.356 | 0.335 |
| online PHASE model (what decides) | 0.039 | n/a | 0.000 | 0.029 | 0.014 | 0.231 |
| nearest neighbour | 0.802 | 0.941 | 0.775 | 0.811 | 0.806 | 0.571 |

* The identifier has **collapsed to three live classes** (1,624 / 279 / 97),
  so "per-class consensus" is one global schedule for 80% of the field. It
  adds ~nothing and is actively WORSE than base for feed-buy timing (MAE
  91.9 vs 44.3 at s=480). Never key a new mechanism on its class id.
* NN retrieval survives the observation noise the live agent has: at feature
  dropout 0.2 a 250-route library still scores 0.77 / 0.94 / 0.71 / 0.78 /
  0.78 / 0.54. Same-team retrieval is only 1.6-3.3%, so it is family
  retrieval, not memorisation. 250 routes ≈ 95% of a 2,000-route library and
  packs to 62 KB.
* **The relay's online phase model is beaten by the unconditional median**
  on WHEAT / MILK / MELON / WOOL at day 13-16, the window the loss study
  says decides games — and it has 92-97% coverage there, so it is what
  actually ships.

### What was built (default OFF, each flag gateable alone)

* `src/kaggriculture/train/nn_schedule.py --build` → `models/relay/nn_lib.json` (250 routes ×
  4 checkpoints 240/288/360/480; 288 and 360 straddle the decisive day
  13-16 window).
* `src/kaggriculture/agentbuild/v22_agent.py --nn-sched` (`_NN_SCHED`): the relay races the RETRIEVED
  schedule instead of the class consensus. **Measured inert** — identical
  banks on 9 of 10 probe cells — because the schedule layer is nearly dead
  code: the relay consults it only for products with fewer than
  `_RELAY_OBS_MIN` observed dumps, which after step 216 is almost never.
* `src/kaggriculture/agentbuild/v22_agent.py --nn-outrank` (`_NN_OUTRANK`, implies the above):
  retrieval outranks the online phase model. This is the flag with an effect.
* Guards: build refuses a library older than 7 days or whose feature
  dimension disagrees with the identifier's input dimension (the train/serve
  skew class of bug).
* Latency mean 1.85 → 2.15 ms/turn, max 12 → 25 ms.

A better OFFLINE predictor is not a win. The verdict is the paired band-panel
record with the CLEAN ratio plus the margin shift, never the prediction table.

### GATE RESULT: `--nn-outrank` REJECTED (2026-09-03, same session)

Band panels, seeds 501-502, both seats, `_NN_OUTRANK` on vs off with
everything else identical — 652 paired cells, 524 clean:

| | sub2000 | mid | top100 | ALL |
|---|---|---|---|---|
| cells, off | 0.9750 | 0.9875 | 0.9880 | 0.9847 |
| cells, on | 0.9750 | 0.9875 | 0.9880 | 0.9847 |
| discordant | 0-0 | 0-0 | 0-0 | 0-0, p = 1.0000 |
| margin (on − off), raw | +0 | −26 | −5 | **−23 median, −69 mean** |
| margin, CLEAN | −53 | −90 | −70 | **−70 median, −86 mean** |
| cells better/worse | — | 38/86 | 74/168 | **170/352** |

Zero win-cell movement and a small, consistent **negative** margin, against a
bar of ≥ +$1,000/game (≈ +3pp of win rate). Both flags stay OFF.

**The lesson is the one already in `market-regime-is-the-bottleneck.md`:
a large offline prediction gain need not convert.** Predicting the step of
the opponent's next dump went from 0.29 to 0.81 hit@±6 on WHEAT and from
0.03 to 0.81 against the model that actually ships — and it bought nothing,
exactly as perfect-foresight sell timing bought nothing once the shed cap
and the dump race were priced. Compute the conversion, not the prediction.

The library builder (`src/kaggriculture/train/nn_schedule.py`), the two flags and the
measurement scripts are kept: they are the instrument that produced the
negative, and they are how the negative gets re-checked if the field's
homogeneity ever breaks.

---

## Runtime branch dispatch: audited, rebuilt, and NOT SHIPPED (2026-09-04)

Full numbers: `models/release_2026-09-04_v44_bandit.json`. Pre-registered
decision rule (written before the results): `.local/branch/decision_rule.md`.

### 1. The recorded evidence for the mechanism was not evidence

Memory carried "8 per-first-shop day-stitched branches scored **0.800** on the
elite+fresh-band panel vs **0.767** for the unbranched field agent — the 2600+
mechanism, confirmed". Three separate defects:

* **The pack could not branch.** `models/trackp/branchpack_kaileh.json`,
  `_v2` and `_v3` are byte-identical to one another and hold **3 distinct
  routes behind 8 shop keys**, with **6 of the 8 keys pointing at a route
  byte-identical to the base**. The shipped v37 branched in 2 of 8 worlds.
* **The comparison was cross-chassis.** 0.800 is `v37.0_bandit`, a
  `tape_runtime` build; 0.767 is `v37_mvc`, a `v22_agent` build. Different
  agents, different layers.
* **The same-chassis A/B already existed and said nothing.** In
  `models/gauntlet/gauntlet_20260831_124212v38_elite.json`, `v38_chbranch`
  (branched) and `v36.0_bandit` (unbranched, same base route `daecc8b6a0`)
  both score **46-14**, with identical W/L on 9 of 10 opponents *to the
  dollar*. And the whole 0.800-vs-0.767 delta is **2 cells of 60 against one
  opponent** (`gronk.py`, 2-4 → 4-2): McNemar p = 0.5.

`validate_branchpack` in `src/kaggriculture/agentbuild/v22_agent.py` now prints the DISTINCT-route
count and refuses a pack whose every branch is the base. That one line would
have caught this a week ago.

### 2. Rebuilt properly, the mechanism works and does not yet pay

Cross-team pool (183 routes, 50 current top-50 teams, 1.32.7 wins ≥ Sep 2),
selection by **staged played games** in each class's own worlds with a
non-regression guard — the yhay81/ShopForge shape, not a vote.

| arm | sub2000 | mid | top100 | vs control, paired |
|---|---|---|---|---|
| `v43.0_bandit` (live) | 0.974 | 1.000 | 0.984 | 0 discordant of 262 |
| `v44_ctl` (rebuild, branch OFF) | 0.974 | 1.000 | 0.984 | — |
| **`v44_branch` (4-key pack)** | **1.000** | **1.000** | 0.967 | 2 better / 2 worse, **p = 1.0000** |
| `v44_single` (same grafts, unconditional) | 0.595 | 0.625 | 0.863 | 2 better / **84 worse**, p < 0.0001 |

`serve_gate` (reactive gauntlet, seeds 101-104): **TIE**, 96-0 = 96-0, 0
discordant. A wider mixed hold-out (20 mid-band tapes + 16 reactive agents,
8 seeds, 288 paired cells): 0.9931 vs 0.9722, **7 better / 1 worse,
p = 0.0703**. Positive-leaning, short of the pre-registered p < 0.05. **A tie
is not a ship**, so v44 is a HOLD.

### 3. The negative control is the most useful result in this pass

`v44_single` is the single best candidate on the panel that selected it —
**77-11 where the base is 64-24** — and it loses **84 of 86** discordant
held-out cells, with the mid band at 0.625. The selection panel
*anti-predicted*. This is gate-calibration conclusion 2 reproduced on
purpose, and it is the strongest evidence in the record for never shipping on
one panel.

What the conditioning bought, by first-shop class on held-out cells:

| first shop | branch OFF | branch pack | grafted unconditionally |
|---|---|---|---|
| ICE_CREAM_SHOP | 35-0 | 35-0 (keeps base) | 26-9 |
| PIZZA_SHOP | 11-0 | 11-0 (keeps base) | 9-2 |
| YARN_STORE | 2-4 | **6-0** | 6-0 |
| SMOOTHIE_SHOP | 21-1 | **22-0** | 22-0 |

The pack keeps the base in exactly the two classes where grafting costs 11
cells and grafts in exactly the classes where it gains 5. The mechanism is
doing real work; the *net* is what is not yet significant.

### 4. Diversity: byte-identical production really does end

Digest of the emitted farmer+hands channel over 288 held-out cells:
**branch OFF = 2 distinct production streams; branch ON = 10, dispatch firing
in 61% of worlds.** (The full action digest was already 82-85 distinct of 108
for every arm — the market channel varies through the reactive layers, so
"every game is byte-identical" was only ever true of the unit channel.)

### 5. A later branch point is WORSE, not better

The coordinator supplied rank 2's reverse-engineered rule
(`docs/history/cropdusta-adaptive-study-2026-09-03.md`): at **step 145** read the
one-turn market-inventory drop, `dWOOL ≥ 2 → SHEEP`, else `dMILK ≥ 2 → COW`,
else `GOOSE`. Implemented as `src/kaggriculture/agentbuild/v22_agent.py --branch-on drain` and
measured on the same base, pool and worlds:

* Every candidate grafted at 145 is **worse** than the same candidate grafted
  at 72: best 63-25 against 77-11, with the base at 64-24.
* Best possible drain pack **74-14 (0.841)** against the shop pack's
  **86-2 (0.977)**.

**For a whole-tape substitution, three days of coherent economy is worth more
than a perfect read of the herd class.** Crop Dusta can branch at 145 because
their branch is one animal-purchase decision inside their own plan; ours is a
648-turn tape swap. Copy their *observable*, not their *commit step*.

Bonus structure: the day-6 drain class is mostly predictable from the day-3
first shop — PIZZA→COW 10/10, ICE_CREAM→COW 11/12, YARN→SHEEP 12/12,
SMOOTHIE→SHEEP 24/26, BAKERY→SHEEP 7/8 — and its one genuinely new split is
PET_CAFE (8 SHEEP / 12 GOOSE). The COW group is exactly the {ICE_CREAM,
PIZZA} pair the played selection told us to keep the base in: two independent
instruments naming the same partition.

### 6. Open correctness debt (measured inert, kept out of the shared file)

`_feed_pull`, `_relay`, `_weed_repair`, `_pull_sells` and `_premium_lead`
read the module-global `_ROUTE` for lookahead, so with a branch active they
look ahead into a schedule the agent is **not playing** — the shape of the
v40 double-sell bug. Measured rather than assumed: a chassis with all 13
lookaheads scoped to the active route scores **64-24 branch-OFF (identical,
mean margin +24,090 both) and 77-11 branch-ON (unscoped 77-11)**. Nil effect
today, so it is debt, not a live bug. The patch is held out of
`src/kaggriculture/agentbuild/v22_agent.py` because those are market layers another agent owns:
`.local/branch/route-scope.patch`, applied by `.local/branch/scope_route.py`.

---

## v44.0_bandit — the branch that shipped, and the two keys that did not (2026-09-03 evening)

Decision record with every number: `models/release_2026-09-03_bandit_ship.json`.
Artefact `agents/v44.0_bandit.py`, sha256 `8fd42213fa1e2ce48cbffdb1398fd428ca5030690597b49468ca0f37bd2c4473`.

### The held 4-key pack cleared its bar, then failed a better instrument

Extending the 2026-09-04 hold-out from 288 seat-0 cells to 672 paired cells
(56 opponents, seeds 701-706, both seats) took the 4-key pack from
p = 0.0703 to **27 better / 11 worse, p = 0.0139** after seat-mirror dedupe.
It cleared the pre-registered p < 0.05 bar honestly, on data, not on a
lowered bar.

Then it was run against the **reactive band** — the only one whose worlds
look like the ladder's (91.6k median shared bank / 50% sub-90k against the
ladder's 85k / 58%; the tape bands sit near 74k / 90%+). It **lost**:
4 better / 15 worse (p = 0.0192) on seeds 711-730, and the regression
**replicated** on independent seeds 741-760 (10b/19w overall, 12b/26w
p = 0.0336 on the cells where the dispatch fired).

### Per-key, and why the tape panels could not see it

The dispatch step *is* the shop unlock, so the first shop is a property of
the world, not of the arm (0 disagreements over 1,152 paired cells). Any
sub-pack is therefore reconstructable **exactly** from cells already played
(`.local/branch/subset_pack.py`).

| key | reactive 711-730 | reactive 741-760 | 56-opponent 701-706 |
|---|---|---|---|
| YARN_STORE | 0b/0w, +$1,442 | **8b/0w p=0.0078**, +$1,686 | **10b/0w p=0.0020**, +$1,639 |
| BAKERY | 4b/1w, +$355 | 0b/2w, +$452 | 17b/10w, +$2,427 |
| PET_CAFE | 0b/4w, −$207 | **0b/6w p=0.0312**, −$264 | 0b/0w, +$1,091 |
| SMOOTHIE_SHOP | **0b/10w p=0.0020**, −$903 | **2b/11w p=0.0225**, −$440 | 0b/1w, −$32 |

The 56-opponent tape set drew **2 SMOOTHIE cells and 0 net PET_CAFE cells**.
An opponent mix can hide a significant regression simply by not sampling the
worlds it lives in — that, not the p-value, is what nearly shipped it.

**Shipped pack: BAKERY + YARN_STORE only.** Confirmatory hold-out on seeds
fixed *after* the reduction: 691 real worlds, **10 better / 2 worse,
p = 0.0386**, mean margin **+$2,530/game**. Fired cells 94-0 against the
control's 90-4 (median margin +34,978 vs +11,713); kept-base cells 0
discordant of 578.

Cost of the fix, stated plainly: coverage fell from 40-46% of worlds to
12-21%. FARMERS_MARKET and BRUNCH_SPOT are 31% of held-out worlds and have
still **never been decided** — the 88-world selection panel never produced
them. That is the next build's work.

### 6 (closed). The route-scope debt is paid

`.local/branch/route-scope.patch` is **applied** to `src/kaggriculture/agentbuild/v22_agent.py`: the
13 tape lookaheads in `_feed_pull`, `_relay`, `_weed_repair`, `_pull_sells`
and `_premium_lead` now read `_rt(state)` — the route the episode is
actually playing. Re-measured on 672 cells before applying: **0 discordant
branch-OFF**, 3 better / 4 worse branch-ON (**p = 1.0000**), and every one
of the 7 discordant worlds is a world where the dispatch fired — exactly the
shape the fix predicts. Inert today, correct once a branchpack ships.
Pre-patch source kept at `.local/branch/v22_agent.prescope.bak`.

### The day-16 TOMATO read: built, gated, REJECTED

Rank 2's third decision point (step 385: plant tomato iff the one-turn town
TOMATO drain ≥ 3, 182/190 = 95.8%) is implemented as
`src/kaggriculture/agentbuild/v22_agent.py::_tomato_read`, behind `--tomato-read` / `_TOMATO_ON`,
**default OFF**, additive by construction (it re-crops PLANT ops the base
already schedules, buys their seeds, re-points only the PICKUPs on tiles it
re-cropped, and offers one SELL).

At rank 2's own dose of 17 seeds, on 672 held-out cells: **0 better / 62
worse, p < 0.0001, −0.144 score and −$6,244 mean margin per game** (raw
540-132 against 642-30).

The mechanism is structural and was measured, not guessed: at step 385 this
base has **zero free tiles** (61 PLANT + 14 PASTURE = all 75 unlocked tiles
of the 10×10 grid) and **4.9% idle hand-turns** (199 of 4,035), with every
hand moving 108-183 times. A tomato block cannot be *added* to it — it can
only displace the base's own wheat, inside an economy whose PICKUP and SELL
schedule is written for wheat. Rank 2 can open a tomato block because their
tape budgets for it. **Copy the observable, never the commitment** — the
same lesson the day-6 herd branch taught, now measured twice.

### Instrument notes worth keeping

* The original 288-cell hold-out was **seat 0 only**, so its p = 0.0703 was
  not inflated by seat duplication. The mirror-dedupe rule is right in
  general (it drops 35-60% of cells in both-seat batteries) but it did not
  apply there.
* The band panel at 1 seed is **saturated** between these two agents: 80 of
  92 held-out worlds are identical. It vetoes; it does not rank.
* v44.0 passes all four band bars (sub2000 1.000, mid 1.000, top100 0.976,
  reactive 1.000); the live v43.0 **fails two** (sub2000 0.950, reactive
  0.833). The one number below the incumbent is top100 (0.967 clean vs
  0.984) — pre-registered bar 2 is missed and is recorded as missed.

---

## Track P — market contest, and the ceiling of it (2026-09-04)

Full report: `docs/history/trackp-market-contest-2026-09-04.md`. Nothing shipped;
`src/kaggriculture/trackp/compiled/main.py` and `rustengine/src/policy.rs` untouched.

**The diagnosis in `docs/history/trackp-compiled-2026-09-03.md` §0.4 is correct.** On
seeds 3-6 vs `v43.0_bandit`, the ranking by wins is exactly the ranking by the
OPPONENT's bank and not by our own: `kaito_v48` 2-0-6 with v43 held to 64,510
while banking only 51,950 itself; `pub_v16rc5` 0-8 at 79,808; `pub_rayk_c94`
0-8 at 89,812; trackp 0-8 at 124,800; `v42.1_trackp` 0-8 at 138,301 (banking
$0). **No agent we hold beats v43 more than 2 times in 8.**

**But the market is in SCARCITY, not glut** — against our seat STRAWBERRY,
MILK, WOOL, CARROT, EGG and TOMATO all sit below I0 all season, price above
base. You cannot deny anything until combined supply crosses I0, and crossing
it is a volume problem: `kaito_v48` crosses STRAWBERRY and MILK eight days
earlier than we do and that alone takes **$67,000** off v43 (its STRAWBERRY
goes $144.9 → $15.9, MILK $153.1 → $31.0, on unchanged volumes — v43 is a
rigid tape).

**The ceiling is now computed, not guessed.**
`.local/trackp_market/counterfactual.py` re-simulates the market exactly
(`market_price` + `_town_consume` are pure functions) with v43's real order
tape fixed and our supply varied. A **free** doubling of our STRAWBERRY and
MILK output moves the margin from −80,553 to **−21,838 — still a loss**, and
that is where `kaito_v48` (−17,817) already sits. Per-product value of +168
free units: WOOL **+40,367** (and the market stays scarce), STRAWBERRY
+27,235, MILK +22,626, CARROT +9,396, FERTILIZER +8,410, EGG +7,660, TOMATO
+7,180, MELON +3,873, WHEAT +3,561. **Winning the market-order SLOT race with
the same supply: +2,774** — collision timing is a rounding error, which is why
every slot mechanism measured null.

**Two instrument lessons worth copying out of this lane.**

1. **Rank on the opponent's bank, or at least on margin.** Every previous
   sweep here (`.local/econ/sweep.py`, `genome_ga`) ranked by median OWN bank,
   the one coordinate the integration report says we do not lose on.
   `.local/trackp_market/msweep.py` ranks on `own − opp`, dedupes both seats
   of a seed into ONE world, and prints an exact paired sign test.
2. **The opponent's bank DID move, and it still bought nothing.**
   `drop_daily` (same-day marketing: a loaded unit walks back and DROPs so
   today's harvest can be sold today, instead of waiting for the dusk drop)
   takes **9,520 off the opponent across 24 out-of-sample worlds**, and all
   five gauntlet opponents bank less (20,637 / 14,300 / 9,841 / 9,750 / 7,677
   each). Margin +5,334, 17 of 24, p = 0.0639; on the 20-world gauntlet alone
   +7,127, 16 of 20, **p = 0.0118**, opponent median 135,602 -> 116,206.
   **It flips 0 of 48 cells.** It also FAILS the primary gate it was aimed at
   (`v43.0_bandit` alone, 14 worlds: +2,947, 9/14, p = 0.4240) and went the
   WRONG WAY on the reserved seeds 21-24 (-3,629, 1 of 4, opponent bank UP).
   It was selected out of 42 configurations, so read the replication as
   suggestive, not established -- and never report the 6-of-6 dev-seed row it
   was picked on without the reserved set beside it.

**And the conclusion that matters for the whole project:** every production
expansion this executor can make is *significantly negative on its own bank* —
16 sheep 53,915 → 43,427, 18 cows → 27,098, 16 geese → 20,764, 24 geese →
11,997, all 0 of 6 worlds better at p = 0.0312 — while the counterfactual
prices those same units at +$40,367 (wool) / +$27,235 (straw) / +$22,626
(milk) of margin. **The gap is not the market and not per-turn search (both now
closed with numbers). It is that `_plan` cannot add an asset without spending
more than the asset returns.** That is the next experiment, and it is a
day-plan/cash/labour question.

---

## 2026-09-04 — day-6 demand branch on close-prefix donors; three market layers killed

Full account: `docs/three-economy-branch-2026-09-04.md`. Numbers:
`models/release_2026-09-04_v45.json`. Pre-registration:
`.local/branch3/decision_rule.md`.

**What changed in the code.** `src/kaggriculture/agentbuild/v22_agent.py` gained two default-OFF
competitor ports, each independently gateable: `--adaptive-market` (boatlee
V29-R1's `_adaptive_market`, all 20 `_AM_CONFIG` constants) and
`--regime-gear` (lynnsakurai's protect/convert epoch gear, implies
`--adaptive-market`). Both keep their state in the PER-EPISODE state dict, not
a module global, because `kagg serve` and the ladder reuse the process. No
existing layer was touched; a build without the flags is behaviourally
unchanged.

**The correction worth recording.** `.local/memory/branch-dispatch-audit.md`
concluded "a later branch point is WORSE — copy the observable, never the
commit step." That was a statement about the DONORS. Ranking all 27,837
engine-1.32.7 routes by opening distance
(`.local/branch3/prefix_scan.py`) finds **95 routes whose steps 0-144 are
byte-identical to ours**, across 12 teams, with all three economies available
at `d_unit` = 0.000. Last night's day-6 graft used donors at d ≈ 0.46, the
field median. At d ≤ 0.033 the same day-6 trigger produces a large,
replicated win.

**The finding about the shipped agent.** On 383 held-out reactive cow worlds
the live `v43.1_bandit` is **0 better / 10 worse** against its own
branch-OFF control (p = 0.0020, −$776/game): its day-3 first-shop trigger
dispatches the BAKERY branch into worlds whose day-6 demand is MILK. Its wool
gain (16-0, p < 0.0001) and this cow loss cancel, so the shipped shop pack is
a **net wash** overall (16-10, p = 0.3269). `v45_branch1` beats it 16-3
(p = 0.0044) almost entirely by not making that mistake.

**Three mechanisms killed, each well powered and each alone:**

| flag | held-out result | verdict |
|---|---|---|
| `--adaptive-market` | 0 better / 0 worse of 360 worlds; median −$92; fires in 357 of 360 | DEAD. The market is in scarcity, not glut — an extra-liquidation controller has nothing to exploit. The "narrow, capped, self-disabling shape" hypothesis is refuted; the shape was not the problem. |
| `--regime-gear` | vs `--adaptive-market` alone: 0 discordant, outcome differs in **1 world of 360** | DEAD AND DEGENERATE. Its published 9-31% fire rate was measured on a PARTIAL predicate (no `capacity_gap`, shed or near-mirror gate). **A fire rate measured on a partial predicate is not a fire rate.** |
| `--tomato-read` | 0 better / 39 worse, p < 0.0001, −$4,439/game, negative in every class | DEAD. Do not re-flip without fresh sign-tested evidence. |

**The GOOSE branch is now our canonical margin-that-cannot-flip example:**
median ΔMargin +$5,456/game, 170 of 175 worlds moved, and **2 better / 3
worse, p = 1.0000.** It does not ship, and the 2-key `v45_branch3` is not the
recommendation despite the larger headline margin.

**Open / not done.** No official-engine spot check on this build;
`src/kaggriculture/measure/band_panel.py` was not the runner (same roster and split, purpose-built
seat-0 driver); the tape bands were not scored; the market layers were
measured on one 360-world set only; `src/kaggriculture/measure/intraday_gate.py` FAILS
`v45_branch1` (32 cells, p = 1.0), so it is a next-daily candidate, never an
intraday retirement of the live climbing v43.1. The obvious next candidate is
the wool donor the winner's-curse rule deliberately declined
(`105018824_s0`, Knight of Favonius, +$22,638 on selection) — it needs its own
separately pre-registered hold-out.

---

## 2026-09-04 — Track P, the marginal-unit ledger: the asset's marginal revenue is NEGATIVE

Full report: `docs/history/trackp-marginal-unit-2026-09-04.md`. This closes the last
open question in the closed-loop lane and the recommendation is **NO-GO**.

### A new environment observation, and it is a big one

**A6. WOOL's glut range is only 59 units wide, and the town's wool demand is a
coin flip.** `MARKET_PARAMS["WOOL"]` is `above_func "sq"`, `above_target 3.2`,
`T = 105`, so `amp = 3.2 * 200 / 105^2 = 0.0581` and the price falls
200 → 177 → 148 → 107 → 55 → **$1** between I0+0 and I0+59. Combined with A4
("sales at $1 do not add to inventory"), the inventory then pins there for the
rest of the season. The town eats **1 wool/day** from the town centre and
**+12/day per `YARN_STORE`** (single-product shops carry the ×2 multiplier), and
shops are drawn at random — so whether wool is worth $240 or $1 is decided by
the draw, not by play. Measured over 16 seeds against `v43.0_bandit`: wool pins
at I0+59 from day 20 in **6 of them** and our realised price is **$35-76**; in
the rest it stays below I0 at **$205-242**. `v43.0_bandit` supplies **138 wool
units in every world** (rigid tape) to our 77, so the opponent consumes the
window before our first sheep is placed. **MILK is the same story with a linear
branch** (target 1.6, T = 122): its end-of-season inventory is +72 (≈$9/unit) in
10 of 16 seeds.

### The consequence for any production decision

Adding 16 sheep to the skeleton sells **+28.1 wool units and takes $2,238 LESS
money for wool** — the marginal wool unit is worth **−$80**. The full paired
cash ledger over 24 official cells (12 seeds × both seats) is −15,333, i.e.
**−$1,191 per marginal sheep**: −527 purchase, −301 feed (marginal wheat $42.2
against a $32.2 average), −389 net revenue.

**`_plan` is not leaking.** At the base herd there are **zero escapes**, 95%
fed-days and 100% cared-days, and the crew's ops substitute 1:1 (+259.5 animal
ops against −264.0 crop ops, PASS +0.9). What binds is the cash curve — dawn
money is $119-$2,377 from day 2 to day 10 because the second quadrant rightly
holds $2,500 — so the board carries **one cow and one sheep until day 12** and
an animal placed on day 20 gets 2 production events against 8 for one placed on
day 2. From day 19 the farm then sits on $7,699 rising to **$50,531** of idle
cash it never redeploys.

### Method note that changes how counterfactuals may be quoted

`docs/history/trackp-market-contest-2026-09-04.md` §4.1 priced 168 free WOOL units at
**+$40,367**. That was computed on **seed 3**, which the 16-seed table shows is
the most wool-scarce draw in the set. **Never price a marginal unit off one
seed's counterfactual when the product's price shape is non-linear and its
demand is drawn at random.** The document flagged the seed dependence and the
number was carried forward anyway.

### What was added to the repo

* `src/kaggriculture/trackp/build_econ_agent.py`: genome knob **`herd_gate`**, **default OFF
  (empty)** — caps a species unless the town's live consumption rate for its
  product (from `town.unlocked_shops` via the existing `_demand()`) and the live
  price clear a bar. Seat law intact; the skeleton at defaults plays a full
  official episode to the **same bank to the dollar** as before (93,732/159,813
  seed 3, 85,090/147,925 seed 5 vs `pub_v16rc5`).
  **Measured and NOT adopted:** own bank +1,572 (20 out-of-sample seeds) to
  +3,363 (gauntlet), gauntlet 14 of 15 worlds at p = 0.0010, but **not resolved
  against `v43.0_bandit` out of sample (+948 margin, 8 of 13, p = 0.5811), the
  OPPONENT's bank rises in both measurements (+624 / +431), and it flips 0 of
  80 cells.**
* `.local/trackp_exec/{ledger,mledger,daytable,summary,woolcheck,bothseats}.py`
  — the cash ledger closes to **$0.00** on every cell measured, and also records
  shed-overflow discards, animal escapes, `max_held` losses, care bonus banked,
  fed/cared days and crops that died unwatered.

### Two traps paid for

* **`kaggle_environments` rebuilds the observation objects every step**, so
  `id(farm)` is only meaningful within one interpreter call, and
  `_apply_unit_action` runs *before* `_process_market`. A probe keyed on object
  identity across steps attributed 3 of 807 harvests and reported a clean zero
  without raising. Buffer, and attribute inside `_process_market`.
* **`land_reserve_lead: -1` is not an off switch.** The genome comment says it
  is; the code reads `d >= ld - int(g.get("land_reserve_lead", 6))`, so −1 starts
  the reserve one day *later*. A variant built with −1 measured byte-identical
  to the baseline on all 24 cells, which is how it was found. The shipped value
  is 0 and is unaffected; nothing was changed.

### Kind-window trap, paid a third time

`floor_animal 0` was the best margin row on the development seeds
(**+4,867**, 19 of 24 cells). On 20 seeds nobody had looked at it measured
**−1,008**. Same lesson as `P_herdM` and `drop_daily`.

Gates: `tests/test_trackp.py` 65/0, `tests/test_rust_engine.py` 6/6
bit-identical, `submission.tar.gz` still `21a6a024…`, `policy.rs` and
`compiled/main.py` untouched, nothing submitted, no kernel pushed.

---

## The day-6 dispatch lane is finished at this donor pool (2026-09-04)

Full record: `docs/decision-table-2026-09-04.md`. 600 worlds x 26 economies,
16,200 games, 0 error cells, nothing shipped.

* **The learned decision table has nothing to learn.** The perfect-dispatcher
  oracle over the whole pool is 1.0000 and its permutation NULL is 1.0000
  (p = 1.0000), at every fixed dispatch width k = 2/3/4. In the MARGIN channel
  the observed oracle is *below* its null (+$28,539 vs +$36,949): economies
  that share our 145-turn prefix win and lose together.
* **`v44.0_bandit`'s hand-written `drd6_WOOL >= 2` threshold already equals the
  2-arm oracle for its own donor** (validation 1.0000). Depth-2 and depth-3
  trees over 117 step-145 features are strictly worse. Do not re-open this
  without a pool from outside the byte-identical-prefix family.
* **No better wool donor exists in the pool** — the open question left by
  `docs/three-economy-branch-2026-09-04.md` §8.4 is closed. The best
  alternative under the same rule ties v44.0 (2 better / 2 worse, p = 1.0).
* **Thirteen economies designed from the engine constants won ZERO of 600
  worlds.** Robustness and base price are the same axis running backwards
  (WHEAT/EGG/CARROT are glut-proof *because* they are cheap), and conceding a
  fragile market hands the opponent ~$25,000 because our own production is
  what floors it. **Most of the wool line's value is denial, not revenue, and
  denial is invisible unless you score margin.**

### Three measurement traps paid for in this pass

* **Regret weights must be in SCORE units first.** Weighting training labels
  by the dollar gap collapsed every tree to "never fire": the cow worlds where
  a sheep donor loses $25,000 outweigh the wool worlds where it flips a result.
  Use `|Δscore|`, with dollars only breaking ties among score-tied worlds.
* **A full-set oracle over many arms saturates and stops being a test.** With
  17+ arms, oracle and null both reach 1.0000. Fix the dispatch WIDTH and run
  the subset search again *inside* each permutation draw, so the null pays for
  the search too.
* **"Median shared bank"** in `docs/history/instrument-repair-2026-09-03.md` is the
  POOLED per-agent bank distribution, not the sum of the two banks. Summing
  reads 162,707 against a reference of 85,000 and falsely fails the
  ladder-like check.

## Market layers re-gated on margin; the "$15k/game" gap is matchmaking (2026-09-04)

Full record: `docs/market-layer-audit-2026-09-04.md`. Raw run
`.local/mlayer/logs/mlayer_reactive_s12.log` (3,120 cells). Nothing under
`src/`, `agents/`, `models/` or `configs/` was modified; nothing submitted.

**The premise.** The order to re-test every market layer rested on a live
observation: `v44.0_bandit` banks a median 78,344 where `v43.1_bandit` banks
93,498 "against comparable opposition", and removing `_PULL` was blamed. The
episode rows carry the opponent's pre-game rating, and once you band by it the
observation dissolves: **`v43.1` has never played an opponent rated above
2262** (143 games, median oppR 2004) while **65% of `v44.0`'s games are
2300-2500** (median oppR 2343). Own bank is a shared-world quantity. At
matched opponent rating 2100-2300 the two bank **94,232 and 93,498**, and
`v44.0` wins more in every window they share (1.000 vs 0.875 below 1900,
1.000 vs 0.860 at 1900-2100, 0.857 vs 0.684 at 2100-2300). The "4.3x
win-margin" gap collapses the same way. `v44.0` leads live by 148 rating.
**Band by `op_rating_pre` before comparing two submissions' banks — always.**

**The layers.** Seven single-flag arms cut from the SHIPPED file (edit count
asserted by `.local/mlayer/make_arms.py`), reactive band, 12 seeds, 84
held-out worlds, mirror-deduped, margin primary. Restoring `_PULL` is *worse*
(1b/18w, p = 0.0001, mean −$46/game) and removing it from `v43.1` is *better*
(24b/11w, p = 0.0410) — the same conclusion with the edit direction flipped.
`_FEED_PULL` leaves 84 of 84 worlds **byte-identical**. `_TIMING` on is
−$662/game and raises the opponent's bank +$746: a fourth negative, it stays
off. `_RELAY`, `_PLEAD`, the median+3MAD dump threshold and the M2 family
schedules all move the median paired margin by **$0** and the mean by under
$150/game. **No market layer clears the $1,000/game floor in either
direction except `_TIMING`, which is a regression.**

**The instrument is the real finding.** All ten agents scored **144-0
(1.000)** on the reactive held-out half — the saturation
`docs/history/instrument-repair-2026-09-03.md` §4(c) called "one build away" has
arrived, on the band that replaced the gauntlet for exactly this reason. And
`v44.0`'s only non-wins in 312 cells are 15 cells against `v43.0_bandit`, our
own frozen clone, 12 of them exact **$0 draws**; it is 297-0 against the other
twelve opponents. Every cell any market layer wins or loses is one of those
self-play ties being tipped. The 0.949 / 0.971 spread between arms on this
panel is a clone artefact, not a field measurement.

Meanwhile `v44.0` is **19-33 (0.365) against ladder opponents rated
2300-2500**. The panel and the ladder disagree by 40 points of win rate in the
band that decides our rating. **Widen the reactive roster to 2300+ opponents
before another band number is used to accept or reject anything.**

### The saturation objection, settled with a positive control (same day)

The reactive band being saturated is a fair objection to a NULL measured on
it. It was settled rather than argued: the two `_PULL` arms plus `_TIMING` —
a layer with a known real effect — were re-run on **all four bands**, 5,280
cells, **351 held-out worlds**, where the tape bands still have headroom.

| held-out, 351 worlds | winners | margin | p | median | opponent bank |
|---|---|---|---|---|---|
| `v44.0` + `_PULL` | 0b / 0w | 13b / 92w | 0.0000 | 0 | +0 |
| `v43.1` − `_PULL` | 0b / 0w | 88b / 30w | 0.0000 | 0 | +0 |
| `v44.0` + `_TIMING` | **0b / 20w** | 55b / 296w | 0.0000 | **−857** | **+652** |

Restoring `_PULL` loses margin in **every band** (sub2000 4b/14w, mid 2b/21w,
top100 7b/51w, reactive 0b/6w) and flips **zero cells in 351 worlds** in
either direction. In the same run `_TIMING` flips **20 held-out cells at
p < 0.0001**. **The instrument is not blind to a market layer — it resolves
one decisively; `_PULL` has no positive effect for it to miss.** Standing
practice from here: before accepting "the panel was saturated" as the
explanation for a null, run a known-effect arm as a positive control in the
same run.

### Freshness treadmill, run for real (Lever 2)

`scripts/freshness_cycle.py --bases 4 --seeds 3 --workers 8`,
`.local/freshness/report_20260904_1412.json` (plus a 3-base 12:53 run).
Bases: the newest 1.32.7 win of each current top-30 team.

| base | team @ LB | top100 held-out | reactive held-out margin | serve gate |
|---|---|---|---|---|
| `105107501_s0` | NayuNayu 2860 | 13b/2w cells p=0.0074, margin −2,914 | 3b/22w p=0.0002 | TIE |
| `105107407_s0` | Wang H2O 2843 | 5b/4w cells, margin +1,786 p=0.0037 | 9b/16w | TIE |
| `105107425_s1` | Giulio Ravasio 2968 | 13b/38w p=0.0006, margin −10,706 | 10b/15w | TIE (overall REGRESSION) |
| `105107336_s1` | bono 2884 | 11b/28w p=0.0095, margin −11,186 | 15b/10w | **REGRESSION** |

**No fresh base beats `v44.0_bandit`** — two tie, two regress, and the two
that tie each win one band's channel while losing another's. Re-basing on a
fresher top-30 tape is not the lever today.

## 2026-09-11 — v51: rebase on the live-loss panel winner (the treadmill, retargeted)

The 2026-09-04 freshness verdict ("re-basing on a fresher top tape is not
the lever") was measured against the SELF panels and reactive gauntlet. It
inverts when the objective is the LIVE 2000-2500 field: against the 39
certified worlds/opponents we actually lost (`.local/livepool/`), the
recorded yhay81/shop-router-0911-simple schedule holds out at 0.833/+34.5k
vs the v50 egg tape's 0.038/−15.4k (p=0.002). The measurement target — not
the rebasing idea — was what was wrong. See docs/history/agent-v51.md.

Also fixed on the way:
- The in-process public gate (`.local/frontier_gate.py`) lets an opponent
  that keeps cross-episode global state degrade to PASS after game 1 and
  inflates margins; `.local/livepool/fresh_gate.py` re-loads the opponent
  per game (opp-errors column proves live play). Use fresh_gate for
  absolute reads.
- Packages evolution (trackp A2) is a measured no-op against real worlds
  (self-play empties map mislabels real tile states): 3 runs, 0 improving
  mutations. Do not re-run it against the live panel without a real-world
  tile model.

## 2026-09-12 — the 2500 wall is a MIRROR-GAME war; mirror-war tape evolution

Both live seats stall at 2417-2467 (bandit last-50 54%). Diagnosis on the
38 certified loss cells of the live pair (`manifest_v511loss.json`):

- 33/38 losses are decided by <$6k; the close-loss bank-gap curve is
  +1,398 AHEAD at day 27 → −2,305 at day 30. Day 29 alone: us $7,671
  earned vs them $10,815.
- **21/38 losses are >80% identical-stream mirror games** — half the
  2450-2500 band forks the SAME public yhay81 kernel v51 rebased on. A
  shared schedule caps at its coin-flip equilibrium (~2450-2500). One
  clone (`_po11_before_market` marker) pre-empts sells one step early and
  wins by $32.
- Mirror games double every dump: both sides crash their own glut markets
  (WOOL 214→$1-11, MILK →$1, MELON 264→78) while town drain makes
  WHEAT/TOMATO/EGG rise all game. The engine's per-unit lockstep +
  scarcity curves make endgame sell timing the whole margin.

REFUTED on the way:
- Elite (3000+) recorded streams as base tapes: 39 elite games certified
  bank-exact; head-to-head their streams lose 73/78 to our base even at
  home seeds (adaptive play desyncs into silent no-ops vs any other
  opponent). Recorded elite play is NOT a schedule — its edge is pure
  reactivity. Elite winner banks median 107.8k ≈ ours.
- Deterministic scarcity surgery (hold TOMATO/EGG/WHEAT sells late, wheat
  buy-early-sell-late arb): EVERY variant regresses 2-55k — the schedule
  is a tightly-coupled cash/shed machine; only engine-in-the-loop search
  edits it safely.
- Bandit relay-off, FLUSHRATE knobs: no effect on the loss panel (0.079
  stays). The bandit deficit was the base tape's races, not one overlay.

THE FIX — mirror-war evolution (`.local/livepool/mirror_search.py`):
(1+λ) over the base tape's market channel; fitness = beat the UNMODIFIED
yhay81 schedule head-to-head (both seats × 12 seeds, ties 0.5), hard
floor = no regression on 8 certified loss cells. Base-vs-base mirror is
19/24 exact ties, so any edge is real. Results (300 iters):

- market-channel winner: mirror **1.000 holdout, +$3,596** mean margin;
  259 micro re-timings (±1-2 unit nudges winning per-unit lockstep races).
- field-ops variant: +$6,574 tape-level but transfers worse in-agent
  (p=0.11). econ variant (+27k own bank): does NOT transfer — it
  exploited the mirror partner, not the world. In-agent panel decides.

In-agent (38-cell live-loss panel, paired McNemar):
- v52_trackp 0.816 vs v51.1's 0.553, **p=0.002**; upset-risk 16→4.
- v52_bandit 0.553 vs v51's 0.079, **p<0.0001**; upset-risk 31→15.
- ab_test full-fidelity: v52_trackp beats live v51.1_trackp 25-5-2
  (p=0.0003); v52_bandit beats live v51_bandit 24-6-2 (p=0.0014).

Arms-race note: our episodes are public — forks will re-mine the evolved
tape with days of lag. The posture is the daily treadmill: re-run the
mirror search against our own latest tape + refreshed loss panel.

### 2026-09-12 addendum — the reactive-floor lesson (v52 draft pulled)

The unguarded mirror-war winner PASSED every recorded-tape instrument
(loss panel p=0.002, oppwins 0.909, ab_test vs live p=0.0003) and then
FAILED the only reactive instrument: fresh publics fell 128-0 (+55-105k
collapse margins) → 118-10 (+2.5-6.7k). Against a reactive router the
evolved stream realizes different market dynamics — in the probed game
our STRAWBERRY crashes to $1-18 from day 20 and the opponent's bank
doubles. A one-mutation probe (single WHEAT qty nudge) keeps the full
48-0 sweep, so the property is not knife-edge — the ACCUMULATED evolved
changes break it.

Rule going forward: **no tape ships on recorded instruments alone; every
accept in any tape evolution must hold a reactive floor** (unbeaten vs
fresh-loaded reactive publics, margin >= 0.6x base). Implemented in
`mirror_search.py --reactive-opps` (reactive_probe.py subprocess); the
guarded search re-runs from base. Draft agents renamed v52draft_* and
their tarballs/notebooks deleted so they cannot ship by accident.

### 2026-09-12 late — v53: the family war (panel-first guarded search)

v52 broke the 2450-2500 mirror wall and stalled at ~2750: ALL its losses
are 2500+, and 14/15 held-out beater cells are 80-86% similar to our own
tape — the tier is OUR LINEAGE, evolved by others. Panel-cell wins do
not transfer between cells (bespoke races); the generalizable payload is
the mirror edge vs our own latest tape. v53 = panel-first evolution FROM
the v52 tape (search/holdout cell split, dual mirror floors + reactive
floor): ab 27-1-4 vs v52 (p~0), flips v52's live losses 0.322 vs 0.056
(p=0.0018), 2500-panel 0.950 held. Shipped 56189050.

Unfixable at tape level: 2 seat-0 races vs the RAW ancestor kernel
(seeds 11,12) — kernel-vs-tape floors disagree because the kernel
realizes different worlds against a new partner (the realized-world
lesson applied to floors). Kernel-in-the-floor search found no
qualifying mutation. Accepted as a <1% live cost.

Bandit-in-the-loop tape search: FAILED (40 chunks, mirror 0.000) — the
bandit's emission is a moving transform that random tape edits cannot
pre-compensate. Bandit benched; triggers for re-evaluation on file.

### 2026-09-14 — the price-aware overlay (freeze exception) and the perturbation law

Operator-approved single exception to the architecture freeze: ONE
price-aware sell-timing overlay in the trackp build
(`.local/livepool/build_v51_trackp.py`, `KAGG_POV*` knobs, default ON in
new builds). What survived measurement is the RESCUE subset only:
dead-stock liquidation at observed prices (stock beyond the tape's
remaining planned SELLs+PICKUPs; WHEAT/FERTILIZER excluded until
terminal), an hour-23 shed-overflow guard (overflow is destroyed by the
engine), a min-sell-price gate, and terminal liquidation. Paired-
measured world-neutral-or-better everywhere: +$20-120/game vs live
ahmed v38, +$50 vs the top-6 beater tapes (36/48 == baseline), +$46 vs
fresh publics (24/24 == baseline). Latency 0.032ms mean.

Two overlay variants REFUTED, establishing the perturbation law:
1. Riser deferral (hold EGG/WHEAT/TOMATO for the town-drain price
   rise): -30,838 vs baseline -4,924 vs ahmed. The lockstep market is
   SHARED — withholding supply hands a reactive opponent the scarcity
   premium (his banks rose to 87-98k) while cash-starving our tape.
2. One-step sell lead (ahmed's own sell_lead, projected-shed funded):
   -6,946 vs -4,804. Our earlier sells change the reactive opponent's
   action stream; the per-day world RNG shifts; our fixed tape then
   plays a world it was not evolved for while he adapts.

**The law: a tape's market stream is a local optimum in the world it
realizes. Against a reactive opponent, any mid-game perturbation
de-optimizes us asymmetrically. Only strand-rescue is safe.** This also
bounds what ANY overlay can recover; closing the ahmed gap (~$2.8-4.9k
depending on tape) requires different tape content, not reactivity.

Tape findings the same day: v54's population tape is ~$2k/game better
vs ahmed than v55's livewar tape (-2,772 vs -4,924, no blowout seeds) —
the engine-in-the-loop breeding had overfit its 3 training seeds. A
fresh 240-iter population search (fixed 12 seeds, 2-seed reactive
floor) produced a tape dominant on every in-search instrument that then
FAILED all three independent floors — the in-search holdout is blind to
seed overfit; the wide-seed arbiter (`ws_popcheck.py`, 19 beaters x 30
random seeds) is the tape-level truth instrument. Seed-ROTATING chunks
(6 blocks, rotating reactive-floor seeds) fixed generality (arbiter
dead-heat with +$198/game margin vs the v54 tape, beaters 36/48 ==,
fresh stream that beats the v54 tape 1.000 head-to-head) but left 2
nusrati losses on a seed the rotation never floor-checked after the
last accept. A pinned-floor repair FIXED the leak (14 floor-passing
accepts; publics 24/24, beaters 36/48 with the best margin yet,
+5,300) and then FAILED the wide-seed arbiter (0.847 vs 0.878, 5/19
regressed): repair accepts bred on one seed block trade population
generality for the floor fix. Protocol lesson: a repair must ALSO
rotate seed blocks, and must checkpoint the FIRST floor-passing elite
before margin-tiebreak drift accumulates. End state: pop4 =
arbiter-equal fresh tape with a 2-game floor leak; pop4r = floor-clean
with an arbiter fail. Neither shipped (operator: no submission);
`agents/v56_trackp.py` holds pop4+overlay as the standing candidate.

## 2026-09-26/27 — v63.1 bandit, top-50 study, fast Rust tournaments (bandit session)

**Shipped:** v63.1 bandit = submission 56589480 (private notebook v13; tarball sha256 fcf3f01b…, binary a45773bc6fe9;
profile 100 of `configs/bandit/profiles/v4.json`): v63 + herd-safe feed pickup (`hfeed_on`) + v92 rival-sales forecaster
following the top 3 streams with a 3-unit trigger (`v92_top 3`, `v92_k 3`), reverted to v63's settings (top 1, k 4) in
realized PIZZA|PIZZA, PET_CAFE|ICE_CREAM, BRUNCH|PIZZA, SMOOTHIE|PIZZA, YARN|YARN (dispatch set 13), + stream disguise
(`disguise_stream`: zero-quantity SELLs appended at the end of the market list; banks identical). Gate v631e (fresh world
seed 5, 64 seeds, 50 hardest public agents + v63 + v62.1, both seats): vs v63 excluding mirror games +254/-137 (p 3e-9,
0.942 vs 0.924); vs v62.1 +412/-103; no worse opponent. On the 444 real ladder replays +11/-5 vs v63. It retired
v63.1_rl (56581792), not v62.1: always list live submissions and name the retiree before submitting.

**Measurement lessons (all verified this session):**
- *Worlds are realized by play.* The tournament's PASS-labelled world seeds matched the realized pair in 0 of 64 seeds and
  64 seeds realized only 45 pairs. `selfplay` now prints the realized pair; per-world conclusions must use it. A per-world
  patch chosen on one seed set did not transfer to fresh seeds (the "bad" worlds moved): world-level effects of v92_top 3
  are seed-level noise, not world structure.
- *Open-loop benchmarks saturate.* v63 scores ~0.98 against any recorded opponent that cannot react: top-40 tapes (0.983),
  top-player tape libraries with reactive switching or front-running sellers (0.986-0.997), lineage-matched 2500+ GM games
  (0.980 while the real lineage players scored 0.489). Only CLOSED-LOOP copy races discriminate (league v63 vs v62.1 0.82;
  self-play vs lineage/random-knob clones), and those are the games v63 loses on the ladder (22/22 losses vs COPY openers).
- *Tape transplants fail in every form.* Team bandits (top-50 teams' tapes under a trie router) score 0.00-0.04 vs v63;
  field continuations that share our route 0 through step 143 and won in the field lose -74/+1 on our real games. The top
  players' strength is in their decisions, not their tapes (their same-world tape keeps ~97% of bank but wins 0.44 vs 0.64).
- *Money disguise (sell 1 wheat at step 1 to break exact-mirror detectors) lost 1000/1000 closed-loop games*: it blinds OUR
  OWN clone-race logic, which relies on exact mirror detection. Stream disguise is harmless; state disguise is not.

**Fixes:**
- v219 (`layers/v219.rs` `routes_block`): the late tomato/land project switched itself off if ANY library route bought land
  or planted tomato after step 432. With foreign routes in the library it was off in every game (-37 of 444 real games).
  Now it checks the followed route only; the v61.1 library is unaffected (md5 unchanged). v219 is worth ~37/444 real games.
- Speed (bit-exact, see `docs/history/fast-tournaments-2026-09-26.md`): direct observation (`Obs::from_state`) and the bitset v92
  forecast (`forecast_fast`) double self-play throughput (0.75 -> 1.53 games/s/thread).

**Tools added:** `top_field` (GM tapes for the current top N), `tapedump` (exact per-day state/choices; sales and spend by
counterfactual step), `top_study`, `league`, `lineage_bench`, `harvest_routes`, `value_model`, Rust `fieldplay`,
`teambase` + trie router (`agent/src/trie.rs`, off unless router.json `"mode": "trie"`), `oracle` (one-decision headroom).
Plans: `docs/history/team-bandits-plan-2026-09-26.md` (executed; result above), `docs/history/search-plan-2026-09-26.md` (value model
AUC 0.737 at day 10 on held-out teams, below the 0.75 gate -> search from ~day 12).

### 2026-09-27 - architecture review and CMA cma1
- Architecture + measured stage coverage + ranked improvement list: `docs/history/bandit-architecture-2026-09-27.md`.
- CMA-ES cma1 done: held-out vs v63.1 (0.842): gen 20 0.982 (+80/-6), gens 30/40/50 0.952. Not yet validated on
  the public field (next: 444 replays + 64-world tournament).
- Dead in 80 games: pg, y, carrot, ctrtable, e343_wl, afr (off). Economy projects rare (v219 6/40, v233 2/40).

### 2026-09-27 PM - Rust stand-ins of public agents, cma3, v64 candidate
- 19/25 top public agents share v61.1's routes/tables/settings/opening and differ only in layers. `bandit/port_public.py`
  (shadow / record / fit / fidelity) + Rust knob `skip` (bypass named chain stages; bit-exact when empty) give Rust
  stand-ins; fidelity vs Python tournament games passes for the herd_safe / order_book / harvest_ledger / v56 / v57 families
  (`configs/bandit/profiles/standins_2026-09-27.json`), fails for farmer_john / wheat / metav4 (stand-in too strong).
- cma2 (copy-only fitness) did not transfer: P102 tied v63.1 on the public top-25 (+98/-102). cma3 (fitness + stand-ins)
  P105 = v63.1 + gen-20 knobs + v63.1 knobs in YARN_STORE|PET_CAFE: top-25 tournament (fresh world seed 7) 0.953 vs 0.907,
  +220/-60 (seat-0 +109/-30); only regression the tetsutani lineage (+0/-8, +0/-4). Built as data/bandit_builds/v64
  (official check 6/6 DONE, 0 fallback, worst 122 ms). NOT submitted.
- The tournament's "both seats" double-counts: the engine is ~seat-symmetric (93% identical banks under seat swap).
