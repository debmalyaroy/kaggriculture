# Track P — Phase A + Phase B joined, and the first REAL-opponent number (2026-09-03)

Phase A (`docs/history/trackp-compiled-2026-09-03.md`) built the TRANSPORT: a static
musl binary inside `submission.tar.gz`, a stdio bridge in `main.py`, a per-turn
watchdog, an action validator, and a pure-Python fallback that is byte-identical
to the compiled policy. Phase B (`docs/history/trackp-search-design-2026-09-03.md`)
built the SEARCH: an amortised day-plan searcher on the bit-exact Rust engine
with the objective `sigmoid((A_me - A_opp) / s_t)`.

They had never run together, and neither half had ever met a real opponent on
the official interpreter. Phase B's own report says so in as many words: "Phase B
cannot drive `pub_v16rc5.py` … that is Phase A's first integration test and the
number that will actually matter."

This is that test.

---

## 1. The join

| piece | change |
|---|---|
| `rustengine/src/obsstate.rs` | **new** — observation JSON → engine `State`. The missing link: `kagg play` receives the interpreter's per-seat observation, `search::decide` wants a `State`. |
| `rustengine/src/main.rs` | declares `plan`/`search`/`value`/`obsstate`; `kagg play [--budget-ms N]`, `TRACKP_BUDGET_MS` override |
| `rustengine/src/policy.rs` | `play_with(budget)`; `budget == 0` is the Phase-A skeleton path unchanged, `> 0` routes the turn through `search::decide` |
| `src/kaggriculture/trackp/compiled/bridge.py` | passes `--budget-ms`, sizes the watchdog from it, records `budget_ms` in the stderr beacon |
| `src/kaggriculture/trackp/compiled/harness.py` | `--budget-ms`, `--workers`, `--worst-gate`, `--ref` (play a plain Python agent in our seat so two runs pair cell-for-cell), per-opponent W-D-L block, kills the previous cell's `kagg` instead of leaking it, tolerates a one-argument `agent(obs)` opponent |
| `src/kaggriculture/trackp/compiled/verdict.py` | **new** — paired McNemar over two harness runs via `src/kaggriculture/measure/win_metric.py` |
| `tests/test_compiled_agent.py` | `--budget-ms`; identity gate at 0, transport gate above it |
| `rustengine/src/bin/search_eval.rs` | `--cells-tsv` per-cell dump, so a value-function A/B can be sign-tested across two runs |
| `rustengine/src/value.rs` | joint-liquidation model behind `TRACKP_JOINT_LIQ` (§5), **default OFF** |
| `scripts/run_trackp_linux_harness.ps1` | `-BudgetMs`, `-Workers`, `-WorstGate`, `-Cpus` |

### What the reconstructed state can and cannot know

`obsstate::from_obs` is exact for everything the interpreter publishes — both
boards, both moneys, the shared market inventory and prices, the unlocked shop
list, our own shed/seeds/inventories. Two things it cannot be:

* **`private[opponent]` is a BELIEF.** The interpreter hides it. The bridge
  leaves the block empty and calls `search::seed_opponent_belief`, which credits
  their shed from what is standing on their public board.
* **`seed` is a placeholder.** The episode seed is not in the observation, so
  weed spawns and the 3-daily shop draw are not reproducible; the search
  substitutes its own seeds and only the *ranking* of candidates has to be
  unbiased by that.

An unknown tile shape is read as `LOCKED`, not `Empty`: the conservative
reading makes the planner ignore the tile, where the permissive one would
invite it to plant on something it does not understand.

### Every Phase-A safety property is preserved

Degradation order inside `kagg play`, cheapest correct answer first:

1. the searcher;
2. a panic inside `decide` → the searcher is dropped (its committed plan may be
   half-built) and the turn is answered by the SKELETON planner, which reads the
   raw observation and needs no reconstruction;
3. an observation that will not reconstruct → the skeleton planner again;
4. a panic in that too → a legal all-PASS turn sized to the hand count.

The Python side is unchanged underneath all four: watchdog, validator, spawn
failure detection, permanent bridge retirement on a timeout, and the
`TRACKP start` / `TRACKP end` stderr beacons (now carrying `budget_ms`, so a
ladder replay says which budget actually ran).

### The equivalence guarantee, kept

`tests/test_compiled_agent.py` compares the compiled policy to the inlined
Python fallback action-for-action. With search ON they legitimately differ —
that is the entire point of the search — so the test now has two modes:

```
$ python tests/test_compiled_agent.py --seeds 3,4,5           # IDENTITY gate
seed 3 seat 0  turns 720  bank 55008  bridge 720 fallback 0 repairs 0  differs 0  OK
... (6/6)
mode: budget 0: IDENTITY gate (compiled policy == Python fallback)
PASS

$ python tests/test_compiled_agent.py --seeds 3 --budget-ms 100   # SEARCH smoke
seed 3 seat 0  turns 720  ... fallback 0 repairs 0  worst 108.6ms  differs 720  OK
seed 3 seat 1  turns 720  ... fallback 0 repairs 0  worst 107.9ms  differs 720  OK
PASS
```

The equivalence is still tested, at budget 0, which is a real shipped path
(`kagg play` bare). `search_selftest` independently proves inside Rust that a
zero-budget searcher is action-for-action the skeleton, so "the search is
strictly additive" remains a checked property rather than a claim.

`python tests/test_rust_engine.py`: **6/6 bit-identical** after all of it
(3 scripted-chaos episodes + 3 real ladder replays). No engine file was touched.

---

## 2. Latency, measured inside Linux on the unpacked tarball

Official interpreter, `python:3.11-slim`, the real `submission.tar.gz` untarred
into `/tmp` and run from there. 719 agent turns per cell, 2 cells per budget.

| search budget | mean | p95 | **worst** | overhead over budget | margin on the 1,000 ms actTimeout |
|---|---|---|---|---|---|
| 25 ms | 26.50 | 28.7 | 35.1 | +10.1 | 28.5x |
| 50 ms | 50.97 | 54.3 | 56.9 | +6.9 | 17.6x |
| 100 ms | 99.37 | 104.9 | 108.0 | +8.0 | 9.3x |
| **150 ms** | 147.98 | 154.8 | **159.0** | +9.0 | **6.3x** |

Under the 6-worker gauntlet load below, the 150 ms configuration measured mean
147.6 / p95 153.8 / worst **191.9** ms across 40 cells — i.e. contention costs
p99 tail, not the budget.

**The overhead over the budget is a flat 7–10 ms, not a multiple of it.** That
is the important shape: `budget_ms` is WALL CLOCK, checked after every rollout,
so a slower core buys fewer rollouts rather than more milliseconds. Only the
fixed work (json.dumps, the pipe, the rollout in flight when the deadline
passes) scales with core speed.

**Shipping budget: 150 ms.** Doubling only the overhead for Kaggle's 1.6 vCPU
puts the worst turn near 170 ms — about **6x** margin, comfortably past the 4x
bar. The cost is stated plainly: on a 2x slower core, 150 ms of wall clock
completes roughly the rollouts this box completes in 75 ms, so **Kaggle runs a
weaker search than any dev-150 number reports.** Buying dev-150 quality on
Kaggle would need a 300 ms budget, whose worst turn (~320 ms) is only 3.1x
margin, below the bar.

`bridge.py` sizes its watchdog as `budget + 150 ms`, floored at 250 ms and
capped at 600 ms, so the timeout path keeps at least 40% of `actTimeout` in
hand at any configured budget.

---

## 3. THE NUMBER: 0 wins in 40 cells

The assembled artefact — the real `submission.tar.gz`, untarred inside Linux,
150 ms search — against five real opponents on the **official vendored
interpreter**, 4 seeds x both seats each.

| opponent | cells | W-D-L | score | our median bank | their median bank |
|---|---|---|---|---|---|
| `data/gauntlet/kaito_v48.py` | 8 | 0-0-8 | 0.000 | 67,436 | 142,799 |
| `data/gauntlet/pub_rayk_c94.py` | 8 | 0-0-8 | 0.000 | 54,835 | 108,377 |
| `data/gauntlet/pub_v16rc5.py` | 8 | 0-0-8 | 0.000 | 56,812 | 127,697 |
| `agents/v42.1_trackp.py` (the seat it would replace) | 8 | 0-0-8 | 0.000 | 61,851 | 128,218 |
| `agents/v43.0_bandit.py` (our live slot-2) | 8 | 0-0-8 | 0.000 | 52,942 | 131,897 |
| **ALL** | **40** | **0-0-40** | **0.000** | **60,468** | **126,016** |

Transport, over the same 40 cells (28,760 timed turns):

| condition | measured | |
|---|---|---|
| full legal episode, both seats | 40/40 DONE, 719 agent turns each | **PASS** |
| dead-opponent cells | 0 | **PASS** |
| fallback fires | **0** | **PASS** |
| validator repairs | **0** | **PASS** |
| worst turn (6 concurrent workers) | 191.9 ms — 5.2x margin | **PASS** |
| beats `v42.1_trackp` head-to-head, sign-tested | 0-8 | **FAIL** |
| not worse than the seat it replaces vs the gauntlet | see §3.2 | **FAIL** |

**The transport is finished work. The agent is not competitive.**

### 3.1 Did the search help at all? Yes — and it does not matter

The same 40 cells replayed with `--budget-ms 0` (the Phase-A skeleton, byte-
identical to the Python fallback), paired cell for cell:

| | median own bank | median opponent bank | W-D-L |
|---|---|---|---|
| skeleton, budget 0 | 44,205 | 117,902 | 0-0-40 |
| **search @ 150 ms** | **60,468** | 126,016 | 0-0-40 |

`win_metric.paired_test`: **0 discordant pairs, p = 1.0000.** The search adds
**+16,263 of median bank (+37%)** against real opponents — a large, real,
reproducible economic gain — and it converts to **zero wins**, because the
opponents are banking twice what we bank either way. This is exactly the
`flips()` lesson from `docs/history/plan-2800-2026-09-03.md` read in reverse: dollars
only buy wins where the margin is thin, and none of these margins are thin.

### 3.2 What the gap actually is: market share, not farm output

The most informative table in this document. Same worlds, same seeds, same
seats, same opponents — only OUR seat changes.

**What the opponent banks (median of 8 cells):**

| opponent | vs our searcher | vs our skeleton | vs `v42.1_trackp` |
|---|---|---|---|
| `kaito_v48` | 137,130 | 129,814 | **76,034** |
| `pub_rayk_c94` | 107,195 | 110,399 | **63,162** |
| `pub_v16rc5` | 125,827 | 111,465 | **65,802** |

**What we bank in those same worlds:**

| opponent | searcher | skeleton | `v42.1_trackp` |
|---|---|---|---|
| `kaito_v48` | 67,158 | 56,918 | **92,576** |
| `pub_rayk_c94` | 53,072 | 46,115 | **78,924** |
| `pub_v16rc5` | 50,518 | 42,998 | **80,378** |

Pooled over the 32 cells the three seats share, as a share of all money banked
in the game:

| our seat | our median | opponent median | our share of total banked |
|---|---|---|---|
| compiled searcher @150 ms | 57,760 | 124,986 | **32.2%** |
| compiled skeleton | 43,905 | 117,219 | 30.1% |
| `v42.1_trackp` (a mined open-loop route) | 68,526 | 79,086 | **43.0%** |

Read it plainly: **the searcher does not merely bank less than the live seat,
it lets the opponent bank ~60,000 MORE in the same world.** `v42.1_trackp`
holds `pub_v16rc5` to 66k; the searcher lets it reach 126k. Bank is a
shared-world quantity (corr 0.73–0.80, `plan-2800` §1) and the searcher is
conceding the shared half of it. The searcher's own gain over the skeleton is
real, but it is *pie growth*, not share: both banks go up.

That is the honest shape of the gap. It is not "the search needs more compute",
and it is not the value function's tie-breaker. **The economy the day plan
expands — `plan.rs`'s field skeleton plus 26 searched knobs — produces roughly
half the output of a mined elite route and competes for the market far less.**
The Phase-B report's own next-work item 2 ("a stronger baseline") is not item 2;
on this evidence it is the entire problem.

### 3.3 Paired verdict against the seat it would replace

`v42.1_trackp` on the identical 32 cells (it cannot be its own opponent):

| opponent | pairs | searcher | v42.1 | diff | v42.1 better on | exact p | verdict |
|---|---|---|---|---|---|---|---|
| `kaito_v48` | 8 | 0.000 | 0.750 | -0.750 | 6 | 0.0312 | **candidate WORSE** |
| `pub_rayk_c94` | 8 | 0.000 | 1.000 | -1.000 | 8 | 0.0078 | **candidate WORSE** |
| `pub_v16rc5` | 8 | 0.000 | 1.000 | -1.000 | 8 | 0.0078 | **candidate WORSE** |
| `v43.0_bandit` | 8 | 0.000 | 0.000 | 0.000 | 0 | 1.0000 | not resolved |
| **ALL** | **32** | **0.000** | **0.688** | **-0.688** | **22** | **0.0000** | **candidate WORSE** |

Gates (iii) and (iv) both fail, significantly, not marginally.

### 3.4 A separate finding the operator should see

In that reference run `v42.1_trackp` — **our current live seat** — banked
**exactly $0** against `v43.0_bandit` on seeds 3, 4 and 5, and $35,157 on
seed 6. Reproduced outside the harness on the Rust engine
(`python -m kaggriculture.engine.serve_match agents/v42.1_trackp.py agents/v43.0_bandit.py --seed 3`
→ `138301 / 0`) and on the official engine, so it is a property of the agent,
not a harness artefact: its mean turn time drops to 0.07 ms in those games,
i.e. it is running its schedule while earning nothing, spending its purse to
zero. A rigid tape whose sells are worthless in a market a heavy dumper has
already drained will do exactly this. It is not a ladder pairing that occurs
(our two seats do not play each other there), but it is a measured fragility of
the live seat against exactly the market-glutting opponent
`plan-2800` says decides low-price worlds.

### 3.5 Standing gates NOT run, and why

`src/kaggriculture/measure/band_panel.py` and `src/kaggriculture/engine/serve_gate.py` were **not** run. The mission gates
them on "wins a material share of those cells"; the candidate won 0 of 40. Two
further reasons to record rather than paper over:

* `band_panel.py` drives **Python agent files**. Scoring a `tar.gz` artefact
  needs a runner shim that spawns the binary per cell. That shim is buildable,
  but building it to score an agent that lost 40/40 would be measuring a
  foregone conclusion at real cost.
* `serve_gate.py` asks whether the tournament may run on `kagg serve`. It is
  orthogonal to this candidate and its verdict is unchanged by it.

Stated plainly rather than substituted with a different number.

---

## 4. Verdict against the four gates

| gate | required | measured | |
|---|---|---|---|
| (i) worst-turn latency leaves >= 4x margin on Kaggle's slower core | worst < 250 ms | 159.0 ms clean / 191.9 ms under 6-way load; Kaggle-pessimistic ~170 ms | **PASS** (~6x) |
| (ii) fallback fires 0 | 0 | **0** of 28,760 turns; validator repairs also 0 | **PASS** |
| (iii) beats `agents/v42.1_trackp.py` head-to-head, sign-tested | score > 0.5, p < 0.05 | **0-0-8**, score 0.000; 8 discordant pairs all against us, exact p = 0.0078 | **FAIL — significantly worse** |
| (iv) not worse than the seat it replaces vs the reactive gauntlet | score >= v42.1's | 0.000 vs **0.688** over 32 paired cells, exact p < 0.0001 | **FAIL — significantly worse** |

Two of four. The two that pass are engineering; the two that fail are the agent.

---

## 5. The glutter weakness: hypothesis tested, NOT supported at this power

Phase B's weakest transfer cell was a wheat monoculture with every sell cap
removed — the market-glutting opponent — where its score fell 0.988 → 0.875 and
its edge collapsed +34.0k → +13.8k. The design doc's own hypothesis
(§11.9 item 1): *"the horizon value function prices shed stock at a liquidation
walk-down that assumes WE are the only seller."*

The hypothesis is factually true of the code. `value.rs::asset_value` carried
the comment "Both seats are valued against the same standing inventory; neither
is debited for the other's liquidation."

### The fix, and why this particular arithmetic

Under a FAIR INTERLEAVE — neither seat gets to sell first — seat *s*'s `u_s`
units sit at merged positions `k · U / u_s` in the combined walk-down. For a
locally linear price curve that is **exactly** equal to walking *s*'s own units
down starting `u_other / 2` deep into the curve. So the whole model is one
number per product per seat: half the other seat's realisable units, handed to
the valuation as a starting offset.

Two properties make it honest rather than a fudge: it **conserves revenue**
(the two liquidations sum to what a single seller would get for the combined
stack, instead of both sides double-counting the top of the curve), and it is
**symmetric**, so it cannot flatter our own seat. It is not a "don't dump" rule
— the search still discovers what each product tolerates from the engine's own
price curve, it just stops being told the curve is all ours.

Implemented in `value.rs` behind `TRACKP_JOINT_LIQ`, **default OFF**, and
`asset_value_off(st, seat, pre = 0)` reproduces the old valuation unit for unit
so the two readings are a clean A/B on one binary.

### Measured

vs `--real-style wheat`, 16 seeds x both seats = 32 cells, at a **reproducible**
32 rollouts/turn (`--rollouts`, not the wall clock — `src/kaggriculture/measure/determinism.py`'s rule
is that a paired A/B over a non-reproducible agent is invalid, not merely noisy):

| | score | W-D-L | own median | opp median | mean edge |
|---|---|---|---|---|---|
| sole-seller (today) | 0.875 | 28-0-4 | 70,421 | 53,674 | +15,679 |
| **joint liquidation** | **0.906** | 29-0-3 | 72,518 | **47,521** | **+22,648** |

**Paired, in the currency that pays: 1 discordant pair, exact p = 1.0000 —
NOT RESOLVED.** One flipped cell is not evidence, and the house rule is that no
evidence means no.

### Widened to 96 cells — still not resolved, AND it corrects the 32-cell reading

48 seeds x both seats, same reproducible 32 rollouts/turn:

| | score | own median | opp median |
|---|---|---|---|
| sole-seller | 0.875 | 66,915 | 49,067 |
| joint liquidation | 0.906 | 73,469 | 49,562 |

**Win test: +0.031, 5 discordant pairs to joint vs 2 to solo, exact
p = 0.4531 — NOT RESOLVED on 3x the cells.** The score difference is identical
to the 32-cell run (+0.031), which is the tell: the effect is small and stable,
not large and unlucky. It is not going to be rescued by more seeds at this
size.

**And the mechanism story from the 32-cell window does not survive.** There the
joint model looked like it was taking revenue OFF the glutter (opponent bank
lower on 22 of 32 cells, mean −5,635). At 96 cells the opponent's bank changes
by **+294 on average and is lower on 47 of 96 — a coin flip.** The entire gain
is our own bank (+4,335 mean, higher on 55 of 96); the per-cell edge is better
on 56 of 96, sign-test p = 0.1253. So the honest description is: the joint model
makes us hold less and realise more, and it does **not** contest the market. On
a 16-seed window it looked like it did. That is the "seeds 4000-4015 are a KIND
window" trap from the Phase-B memory file, re-paid: **never read a mechanism
off 32 cells.**

**Verdict: hypothesis supported mechanically (we stop over-valuing our own
hoard), NOT supported on wins at 96 cells, and NOT supported as a fix for the
thing §3.2 says actually beats us. The flag stays OFF.** It is not shipped and
it touched no number in §3.

**The `--real-style field` regression pair was NOT run** — the sequence was
stopped after the two wheat configs. Stated rather than quietly omitted, and it
does not change the verdict: the regression check exists to catch a change that
fixes the glut case at the modelled case's expense, and this change does not
fix the glut case, so there is nothing to trade off. Anyone reviving the flag
must run it first:

```
TRACKP_JOINT_LIQ=0 rustengine/target/release/search_eval --seeds 4000:4048     --real-style field --rollouts 32 --budget-ms 100000 --threads 12     --cells-tsv .local/trackp_search/w48_field_solo.tsv
TRACKP_JOINT_LIQ=1 ...  --cells-tsv .local/trackp_search/w48_field_joint.tsv
python .local/trackp_search/joint_liq_ab.py w48_field_joint.tsv w48_field_solo.tsv
```

The second candidate cause the design doc names — *"the opponent model never
dumps"*, addressable with the already-implemented `--opp-ensemble` — is
untested and is the better next experiment, because §3.2 says our failure
against REAL opponents is precisely that we do not contest the market.

---

## 6. What this decides, and what to do next

**Be blunt.** Four closed-loop attempts have now failed against real opposition:
planner_v0 0-64, the econ planner 0-6 vs seats, the Phase-A skeleton 0-16, and
this — the full compiled searcher — **0-40**. The 2800 plan's hypothesis was
that the missing ingredient was the self-imposed compute budget. That budget is
now genuinely gone: the searcher runs 150 ms of real rollout search per turn
inside a shipping tarball, uses it (the offline curve is monotone in budget) and
still loses every cell to three public agents and to both of our own seats.

**What is proven and should be kept regardless of the lane's fate:**

* the compiled tar.gz transport, end to end, with a fallback that costs speed
  and nothing else — `submission.tar.gz` is 0.48 MiB and plays a legal 719-turn
  official episode at 0.16% of `actTimeout` per turn on the skeleton path and
  16% on the searching one;
* the measurement rig: `harness.py --ref` + `verdict.py` gives a paired,
  sign-tested, official-engine comparison of an ARTEFACT against a Python agent,
  which nothing else in the repo could do;
* `obsstate.rs`, which makes the bit-exact engine usable as a forward model from
  a live observation.

**What is not proven, and what the evidence now says instead.** The lane's
problem was never the substrate and it is not per-turn compute. §3.2 says it is
market share: 32.2% against `v42.1_trackp`'s 43.0% in the same worlds. The
searched day plan grows our own bank well (+37% over the skeleton) and leaves
the opponent's untouched. Against a shared-market opponent that is the losing
half of the trade.

**In priority order, if the lane continues:**

1. **Replace the base economy, not the search.** `DayKnobs::skeleton` is the
   *median field* economy. The obvious candidate prior is a mined ELITE route's
   economy expressed as day knobs — the same thing `v42.1_trackp` executes
   open-loop, but re-seated on the live board each dawn. Bar before any further
   search work: the skeleton path alone (budget 0) must reach the mined route's
   median bank against the gauntlet, i.e. ~80k, not 44k.
2. **Contest the market.** Test `--opp-ensemble` (already implemented, never
   measured) and the joint liquidation of §5 together, and judge them on the
   OPPONENT's bank as well as ours — that is the coordinate we are losing.
3. Only then re-run this gauntlet. Do not tune against it before then; it has
   now been measured once, clean, and that is what makes it a valid instrument.

**For the Sep 24 go/no-go**, this is a strong negative. It does not end the lane
by itself, but it removes the reason to expect trackp to clear the band panels
on its current economy, and item 1 above is a bigger piece of work than the
search was. The plan's own contingency — two bandit-lane agents from different
base families — should be treated as the leading option unless item 1 lands and
measures.

**Nothing here was submitted.** `.local/candidates/trackp_compiled/submission.tar.gz`
exists and must not be shipped: it loses 40/40, and only the latest 2
submissions stay active, so uploading it would evict a live agent.

---

## Appendix — reproducing every number

```powershell
# build the artefact (Docker, x86_64-unknown-linux-musl)
.\scripts\build_trackp_linux.ps1

# latency sweep inside Linux on the unpacked tarball
foreach ($b in 25,50,100,150) {
  .\scripts\run_trackp_linux_harness.ps1 -Seeds "3" -Seats "0,1" `
    -Vs @("data/gauntlet/pub_v16rc5.py") -BudgetMs $b -Workers 1 -WorstGate 1000
}

# THE gauntlet: the artefact vs 5 real opponents, official engine
.\scripts\run_trackp_linux_harness.ps1 -Seeds "3,4,5,6" -Seats "0,1" -Workers 6 -WorstGate 1000 `
  -Vs @("data/gauntlet/pub_v16rc5.py","data/gauntlet/kaito_v48.py","data/gauntlet/pub_rayk_c94.py",
        "agents/v43.0_bandit.py","agents/v42.1_trackp.py") `
  -JsonOut ".local/candidates/trackp_compiled/gauntlet_cand.json"

# the same cells with the search OFF, and with the live seat in our chair
... -BudgetMs 0 -JsonOut ".local/candidates/trackp_compiled/gauntlet_skel.json"
... -Ref "agents/v42.1_trackp.py" -JsonOut ".local/candidates/trackp_compiled/gauntlet_ref.json"
```

```bash
python src/kaggriculture/trackp/compiled/verdict.py .local/candidates/trackp_compiled/gauntlet_cand.json \
    --ref .local/candidates/trackp_compiled/gauntlet_ref.json \
    --label "trackp compiled search@150ms" --ref-label "v42.1_trackp (the live seat)"

python tests/test_compiled_agent.py --seeds 3,4,5             # IDENTITY gate (budget 0)
python tests/test_compiled_agent.py --seeds 3 --budget-ms 100 # search transport smoke
python tests/test_rust_engine.py                              # 6/6 bit-identical
rustengine/target/release/search_selftest.exe                 # zero budget == skeleton

# the glutter A/B (reproducible rollout budget, per-cell dump)
TRACKP_JOINT_LIQ=0 rustengine/target/release/search_eval --seeds 4000:4016 \
    --real-style wheat --rollouts 32 --budget-ms 100000 --threads 8 \
    --cells-tsv .local/trackp_search/wheat_solo.tsv
TRACKP_JOINT_LIQ=1 ... --cells-tsv .local/trackp_search/wheat_joint.tsv
```
