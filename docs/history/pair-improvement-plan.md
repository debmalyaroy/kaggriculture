# Route + bandit pair improvement plan — the road to 3000+

*Updated 2026-08-21. **Strategy superseded by `docs/history/winning-plan.md`** (researched
2026-08-21 from the discussions, public notebooks, and our full submission
record — includes the corrected decline diagnosis: churn + the Aug-13
Valmorlee→kss base regression chosen by the anti-predictive offline crown).
This doc remains the implementation ledger. Companion doc (parked):
`docs/trackp-improvement-plan.md`.*

Goal restated by operator: **a 3000+ leaderboard rating.** This doc is the
plan of record for the pair; it gets updated as phases complete.

---

## 0. The Rust engine question, answered

**The full 1.32.7 engine is already ported to Rust and verified.** This is
not a future work item:

| Tier | What it runs | Speed | Fidelity | Status |
|---|---|---|---|---|
| `rust_prerank` | open-loop tape vs tape | ~90 s for the whole 5,200-route index | schedule-level | live (funnel widener) |
| `kagg batch` | tape pairs → final banks, threaded | 143 ep/s/core | full engine, open-loop agents | live (Track P) |
| `kagg serve` + `serve_match.py` | **real Python agents, all closed-loop layers** | 0.63 s/game | **exact-bank identical to official** (12/12 audit, engine 1.32.7) | live — crowned v28 in a 6-min tournament |
| official `kaggle-environments` | ground truth | 10.0 s/game | definition of truth | spot-check + self-revoking gate |

RNG is bit-exact (MT19937 port), per-step state was verified bit-identical on
50/50 episodes including real ladder replays, and every serve tournament runs
a post-halving official spot-check that closes the gate on any mismatch.

**What the port has NOT yet been pointed at — the actual gap:**

* **R1. Mass schedule search (the route factory).** `kagg batch` can play
  ~500k episodes/hour on 4 cores. Nobody is searching route-space with it
  yet — we only rank *mined* routes. CMA-ES/beam/annealing over the tape
  vocabulary (land buys, hires, plant/water/harvest schedule, sell days)
  against a fixed field-mix of elite opponent tapes = routes the field
  cannot mine back from the ladder, optimized natively for 1.32.7 hinge
  economics. This is the paper's "MILP" goal done with the right tool.
* **R2. Branched-tape evaluation.** The demand-gated route (branch on the
  public shop draw at day 3/6) is a *conditional* tape. Batch mode evaluates
  open-loop tapes; extending the search representation to a branch table is
  a data-format change, not an engine change — the serve tier can already
  run the real branched agent for confirmation.
* **R3. Scarcity-window simulation.** The hinge (drained CARROT/TOMATO/EGG
  quoting 10–50× base) is fully implemented in the Rust market
  (`market.rs`). Sweep sell-timing/withholding policies across thousands of
  seeds to map exactly when withholding into a drain beats dumping — offline
  answers, shipped as constants in the tape.

Porting anything *more* to Rust (e.g. the Python agent runtime) buys nothing:
serve at 0.63 s/game already makes the tournament 6 minutes, and search
(R1–R3) uses batch mode where agents are tapes anyway.

---

## 1. The 3000 math — what the target actually requires

Reference points (all measured, ours):
v18 converged at **2342.9** (75.7% wins vs a median-2275 field);
the copied v21.1 reached **2459.2** (36.8% wins vs a median-2557 field —
matchmaking raises the field as you climb). v27/v28 read 1900–1950 at
12–18h *because they keep being retired mid-climb*, not because they are
weaker than v18.

A 3000 rating means winning consistently against the 2500–2800 pool while
the submission ages undisturbed. Decomposition of the gap from our ~2350
converged baseline (estimates, each with a measurable gate):

| Lever | Est. rating value | Evidence gate |
|---|---|---|
| L1. Convergence discipline (stop retiring climbers; 2-day holds; endgame freeze) | +150–250 | free — v27 was retired at 18h still climbing; A/A noise floor quantifies it |
| L2. Slot-2 stops donating (bandit-clone or runner-up instead of −500 diversity route) | pair-level, protects the final-two scoring | rating curves, equal windows |
| L3. Hinge-native routes (mined FROM 1.32.7 play; field still tuned for 1.32.6) | +100–250 while the window lasts | v28+ tournaments already pool them; crown decides |
| L4. Demand-gated route (adaptive topology on the shop draw) | +50–150 | shop-draw study must show a sign-testable conditional gap FIRST |
| L5. Search-generated routes (R1, unminable) | +50–200 | must beat the crowned route paired on serve, then official spot-check |
| L6. Scarcity sell policy (R3: ordering + withholding into drains) | +0–100 | bounded by the multi-SELL audit; paired ab_test decides |
| L7. Track P planner (parked; graduates only through its 5 gates) | option value | see parked doc |

Sum of mid-points lands ~2900–3100. **Honest statement: 3000 is reachable
only if L1+L3 land fully and at least two of L4/L5/L6 are real.** The field
also improves; the hinge window (L3) decays as others adapt — which is why
the sequencing below front-loads it. Anything that fails its gate is
dropped, not argued with.

---

## 2. Phase 1 — tonight, before the 04:30 run (NEEDS OPERATOR GO — not yet coded)

1. **Adaptive-OFF** in `stage_second` route2 builds (5 measurements, 0 wins).
2. **Operator force marker** `models/second_slot_force.json`
   (`{"force": "bandit", "until": "2026-08-18"}`) → ships the bandit into
   slot 2, logged as `why: "operator forced"`; dated, self-expiring;
   SECOND-SLOT RULE stays the default. The bandit is a verified clone of the
   route (18 paired cells, 0 discordant), so slot 2 rates ~flagship instead
   of −500, and the pair becomes a free **A/A test** whose converged gap is
   the ladder's noise floor.
3. **Release hold** `models/release_hold.json` (until 2026-08-20) —
   `daily_release` skips ONLY the upload stage; fetch/train/tournament/crown
   keep running so candidates stay warm while the live pair converges (L1).

Fallback if no GO by ~04:00: the run ships route + diversity route2 with
adaptive ON — the recipe that lost twice.

## 3. Phase 2 — Aug 18–20, during the hold (build L4/L5/L6)

Priority order, all offline, all on infrastructure that exists:

1. **Shop-draw conditioning study** (analysis, ~half a day). Condition final
   banks in the route index on the day-3/day-6 shop draw. Gate: a
   sign-testable conditional best-route change, or L4 is dropped.
2. **Multi-SELL audit** (hours). Count multi-SELL turns in our live tapes and
   estimate reorder value against the sequential price-impact curve. Gate:
   material dollars-at-stake converted via `flips()`, or L6 is dropped.
3. **Route factory bring-up (R1)** — the long pole for L5, start now:
   define the tape vocabulary + mutation operators, fix the opponent
   field-mix (elite tapes stratified by family), and start CMA-ES/beam on
   `kagg batch`. Overnight budget: millions of episodes. Candidates that
   beat the crowned route paired-on-serve enter the normal tournament pool.
   Compute discipline: batch runs are ONE Rust process — respects the
   machine-wide 3-wide cap; never overlap the 04:30 tournament window.
4. **Demand-gated route build** (only if #1 passes): 2–4 elite routes
   sharing an opening through step ~72, branch table on the observed draw;
   single-file, math-only. Paired on serve, then the normal tournament.
5. **Scarcity sweep (R3)** (only if #2 shows stakes): map withhold-vs-dump
   policy across seeds; ship winners as tape constants.

## 4. Phase 3 — Aug 20, decision point

* Record the **A/A noise floor** from the clone pair → becomes the minimum
  meaningful pair difference for every future read (memory + this doc).
* Demand-gated route sign-passes vs flagship → it **becomes** the flagship.
  Passes vs slot-2 only → it takes slot 2.
* First factory-searched route enters the tournament; crown decides.
* Lift the hold; resume daily crown cadence. Slot-2 default becomes
  **tournament runner-up, adaptive OFF**.

## 5. Phases 4–6 — the calendar to the deadline (23 Sep)

| Window | Focus | Exit criterion |
|---|---|---|
| Aug 20–27 | L4/L5/L6 candidates through daily crowns; factory iterates nightly; 2-day converge windows between ships (L1) | at least one lever sign-passed and live |
| Aug 27–Sep 8 | exploit + widen: retrain identifier/famdumps on hinge-era data; factory targets the *current* field mix weekly; Track P unparked if pair work stabilizes | flagship converged ≥2500 on an aged window |
| Sep 8–15 | **candidate freeze selection**: best two agents by paired evidence + holdout; no new ideas, only measurement | final pair chosen |
| Sep 15/16 → 23 | **FREEZE.** Ship the final pair, then nothing. Every remaining day is convergence — the freeze is itself a scoring move (the leaderboard is computed from the last two submissions) | pair aging undisturbed |

## 6. Standing rules (unchanged)

SCORE not margin; paired sign tests decide; fixed panels, never ladder win
rate; `engine_check` before believing any number; serve gate self-revokes on
any official mismatch; 3-wide engine cap is machine-wide; levers that fail
their gate are dropped without appeal; operator forces are dated and logged.

---

## Changelog

* **2026-08-21 (late) — DIAGNOSIS AMENDED + v32 ROOT CAUSE CLOSED + SHELL
  FIXED.** Two corrections to the morning's diagnosis. (1) The submission-id →
  agent mapping shows the decline had a REAL component: the Aug-13 crown moved
  the base from Valmorlee 91464294_s1 (v22_2_bandit converged **2776.4 at
  71.8% over 103 games** — our true peak, not v24.1's 2339) to kss 92513718_s1
  (family converges 1650–2140). Churn hid a genuine base regression chosen by
  the anti-predictive offline crown. (2) v32's crater was MECHANICAL:
  `_safe_market` clamped SELLs to a projection that only credited `DROP`
  deposits, but the engine has a second path (`PLACE item n` beside the shed)
  and Izzoudine's route deposits that way (48 PLACE / 9 DROP) — the clamp
  cancelled virtually all its sales ($2,146 vs $58,292 same seed/opponent,
  layer-bisected). The engine's market partial-fills per unit, so the clamp's
  premise was false anyway. Shell fixes + new instruments (`src/kaggriculture/measure/gauntlet.py`
  reactive public-agent round-robin, `src/kaggriculture/measure/strict_future.py` chronological
  veto gate) in BUILD_JOURNAL 2026-08-21-evening; strategy in
  `docs/history/winning-plan.md`. Fast suite 16/16 green.

* **2026-08-21 — DECLINE DIAGNOSIS: it was never a regression, it was churn +
  variance.** *(amended above: churn AND a base regression)* Every route v24.1→v29 was built on the SAME base route
  (92513718_s1, kss) -- the crown never changed it. v25.0_route and
  v26.0_route are BYTE-IDENTICAL except the header label, yet scored 2286 vs
  2002 (284 apart) -- pure ladder variance (Ryo: byte-identical copies end
  300-1400 apart). The apparent 2339->1649 "steady decline" is dominated by
  (a) re-submission variance -- each resubmit restarts at 600 and rolls a new
  luck path -- and (b) age: v24/v25 were retired YOUNG (~1 day, <100 games)
  where ratings sit 100-200 ABOVE the converged value on a lucky draw, while
  v29 is converged (150 games) at 1649 = near the agent's TRUE level. So the
  base-92513718 route is genuinely a ~1650-1900 agent; 2339 was a young lucky
  peak that never converged. NO agent was "broken" in v24->v29. THE ROOT
  MISTAKE: re-submitting the same agent repeatedly (the daily auto-crown did
  this every day: HOLD->rebuild-same-base->resubmit), which resets
  convergence, rolls the dice, and discards the aged copy -- the exact
  anti-pattern Ryo warns against. (v32 IS a real regression -- frozen copies
  of others' routes that misfire, 386 -- separate from this.) FIX (no revert):
  stop churning; keep ONE agent aging; auto-crown stays OFF; to exceed ~1900
  build a genuinely DIFFERENT better agent validated on the LADDER (100+
  games), never re-submit the same thing. Live rating is cosmetic until the
  Sep-23 deadline (final BT is post-deadline from the last 2 subs), so there
  is time and no reason to churn.

* **2026-08-20 22:00 — SUBMITTING v32.0: source-rating-selected pair (the
  fix, operator-directed).** Stopped all auto-submission (KaggricultureRefresh
  Cycle DISABLED + release_hold to 08-24). Then selected by the ONLY validated
  predictor -- source-team LADDER RATING + open-loop consistency (a frozen
  copy preserves an open-loop team's strength):
    - Flagship v32.0_route = Izzoudine KANTA 94711806_s1 (ladder **2961**,
      **92% open-loop** -- highest-fidelity strong copy target).
    - Hedge v32.0_route2 = StackKnight 94774341_s1 (ladder **2834**, 80%
      open-loop, DIFFERENT day-3 family).
  Built as v32.0 (y=0) to bypass the offline intraday gate cleanly -- which
  MUST be bypassed because offline is anti-predictive (these routes LOSE to
  v29 offline yet their teams rate ~2900 vs our 1640). Both pass validation
  (self-play DONE, latency <2ms). Composition truth: the 2900-rated teams run
  the SAME 6C/2S/2land as us and sell 1872-2933 fertiliser (so "small herd"
  and "don't sell fertiliser" were both red herrings; the gap is the detailed
  tape). DELIBERATE BET with bounded downside (worst ~1600 = current) and
  large upside (~2900); the ladder validates over ~5h. Revert = re-submit v29
  (tapes retained). Submitting via daily_release --resume-publish.

* **2026-08-20 21:30 — THE decisive finding: offline win-rate does NOT
  predict ladder rating.** Built the new crown measure (`crown_eval.py`:
  per-strong-opponent win rate, dozens of seeds, both seats, sign-tested --
  the measure Ryo #1 describes) + the per-turn win-prob control variate
  (`crown_winprob.py`) + a winning-composition filter (`compose_filter.py`).
  Ran it: 3 mined routes beat our incumbent by **+12pp, p=0.0000**, robust
  across two seed sets, from different opening families -- looked like a
  clean diverse pair to ship. THEN checked the SOURCE TEAMS' actual ladder
  ratings: Daryl Brach **1855**, Dr Chandrasen Pandey 1960, Akira Imae 1957
  -- vs kss (our incumbent's source) **1911**. The +12pp flagship's team
  rates BELOW our incumbent's source. So offline tape-vs-tape win rate is
  NOT a ladder-rating predictor; beating frozen panel tapes != rating well
  against the live field (the "frozen reconstructions mislead" warning, and
  the champion-screen paradox, both confirmed). DID NOT SUBMIT -- shipping
  would gamble v29's converged 1640/1718 on a contradicted signal. Re-held
  releases (models/release_hold.json, until 08-24) so no cycle auto-ships a
  mirage crown. TWO REAL LEADS: (1) our kss-copy rates 1640 while kss rates
  1911 -- a ~270pt build/deploy gap worth investigating (maybe free points);
  (2) the ladder is the only reliable oracle -- offline can rank, only the
  ladder validates. Composition truth: winners bank 173-184k with a SMALL
  herd (5C/1S) vs our 6C/2S/109k -- action-efficiency, not fertiliser, is
  the lever; but even that only shows up offline, which we now distrust.

* **2026-08-20 20:15 — CHAMPION SCREEN: our open-loop route is NOT the
  problem.** Built the top-8 leaderboard teams' best recent winning routes
  and scored them + our incumbent against a panel of those champions
  (`champion_screen.py`, models/factory/champion_screen.json). Result: our
  incumbent scores **0.6823 — 2nd of 9**, beating カワシギ (0.649), tetsuya,
  Crop Dusta, Subramanya, Xiaowenhao, Izzoudine; only Ryo Hasegawa (0.762,
  p=0.20 NOT significant) edges it. NO champion sign-beats our incumbent on
  the field panel. Yet those teams rate 2900-3157 on the ladder and we rate
  1640. HYPOTHESIS (strong): the ladder's top agents are CLOSED-LOOP
  (adaptive); the index "routes" are frozen single-game tape snapshots of
  them. Our open-loop route beats their frozen snapshots but loses to their
  LIVE adaptive agents on the ladder. This also explains why our own
  adaptive layers measured dead: our offline panels are frozen tapes, so
  reactivity has nothing to react to. IMPLICATION: (1) re-mining/re-crowning
  open-loop routes cannot help -- ours is already near-best among them;
  (2) the gap to 3000 is ADAPTIVITY, which our tape-based measurement cannot
  see -- it must be tested against ADAPTIVE opponents (self-play leagues /
  the ladder itself), not frozen tapes. NEED: read the top players'
  discussions (WebFetch can't render Kaggle SPA) to confirm the architecture.

* **2026-08-20 19:45 — root cause of the confusion: a HARDCODED engine tag.**
  `sameday._ingest` stamped every route `"engine": "1.32.6"` regardless of the
  replay's real `module_version` (line 451). So the entire index engine field
  was fiction — that is what made the ladder look 1.32.6 and nearly caused a
  wrong revert. FIXED to read the real version. Net honest picture after a
  night of digging: (1) engine is CORRECT (1.32.7), no revert; (2) the index
  engine tag was a lie, now fixed; (3) a real ingest backlog existed — 4,546
  recent 1.32.7 ladder replays un-ingested, now drained (+9,060 routes, pool
  47,493) — so the crown pool is more complete, though it was never empty
  (recent routes were present, just mis-tagged). CONCLUSION: the systematic
  bugs were real but MODEST. v29 underperforms mainly because a mined
  1.32.6-era-crowned tape is not good enough vs the current field — reaching
  the top (3157) needs a genuinely better AGENT, not just pipeline fixes.
  Action: lifted the release hold so tomorrow's 04:30 cycle crowns AND
  publishes on the now-complete + correctly-tagged pool; if it finds a base
  that clears the bar it ships, else HOLD keeps v29 (safe). Note: the 9,060
  routes ingested pre-fix are still tagged 1.32.6 (cosmetic; crown selects by
  date, not engine) — a re-tag backfill is low-priority cleanup.

* **2026-08-20 19:00 — CORRECTION: engine is RIGHT (1.32.7); the real bug is
  an un-ingested 1.32.7 backlog.** I nearly reverted the engine to 1.32.6 on
  index evidence (index showed 08-18..20 as 1.32.6). The operator made me
  verify externally first — correctly. GROUND TRUTH: every raw replay fetched
  08-19/08-20 is module_version **1.32.7** (1,500 scanned, 0 on 1.32.6), and
  GitHub PR #1399 (hinge, = 1.32.7) is on master with no revert. The ladder
  IS on 1.32.7 and so are we — NO revert was run (engine_version.json still
  1.32.7, flags untouched). The index looked 1.32.6 because **~15.5k raw
  1.32.7 replays were downloaded but never ingested** (interrupted scrapes /
  throttle / BSODs left them in _stage) — so the crown pool + identifier +
  hinge screen have been training on a stale 1.32.6-dominated index while
  the ladder plays 1.32.7. THAT is a real driver of underperformance: we
  crown agents good against last week's field. Fix: draining the backlog
  (`backfill_ingest.py`, 4,546 in sameday/_stage), then re-crown on the real
  current meta. Kept (not executed): a two-way `engine_swap_1327` with an
  `unswap()` path + majority-based detection, so a genuine future rollback
  auto-reverts instead of sticking. LESSON: the index `engine`/`date` tags
  are unreliable; raw replay module_version is the only engine ground truth.

* **2026-08-20 18:35 — PUBLISH-NOW: HELD, nothing shipped (correct call).**
  Ran the cycle with the hinge quota live. Findings: (1) the hinge routes
  that scored +42pp vs my elite panel were SCREENED OUT round 1 against the
  loss-tape opponents who actually beat us — the hinge edge is
  panel-specific, not a ladder win. (2) The tournament winner (91664668,
  non-hinge, +6pp on the loss-tape panel) LOSES 0–64 head-to-head to the
  live incumbent (−692/game). Same pattern as the 08-20 morning A/B.
  Nothing beat the live v29 pair, so publishing would evict a converged
  rating for no gain. Kept v29 live; tomorrow's auto-publish stays held.
  Crash fix: the arm stage hit a cp1252 UnicodeEncodeError on a non-ASCII
  opponent name (my manual launcher lacked `-X utf8`; the real
  daily_release.bat already has it). Fixed the one-off wrappers +
  hardened train_arms.py's entry. Bandit remains measured-dead: the arm
  guard reported a commit as NEGATIVE-VALUE (E[gain|correct]=−$52,940).
  **LEADERBOARD REALITY: current #1 = 3157.7 (Ryo Hasegawa). "3200+" means
  WINNING. Our best-ever converged was 2342 (v18); we have never had a
  3000 agent.** The mine-and-tweak paradigm is exhausted (every lever this
  week failed its gate). The gap to the top is STRUCTURAL — see the new
  "Road to the top" section.

* **2026-08-20 evening — ROOT CAUSE FOUND AND FIXED IN SELECTION.** The
  flagship sells zero CARROT/TOMATO/EGG (the hinge trio) — own-games
  forensics show opponents' egg volume tracks their wins and our deficit
  compounds from day ~17. The pool already held the answer: 786 fresh
  hinge-selling routes, five of the top ten sign-tested better than the
  incumbent (best +42pp, p≈0, `models/factory/hinge_screen.json`) — the
  medoid/rank funnel just never ticketed them. Fix: `HINGE_SLOTS=4`
  reserved tournament tickets (same referees, same crown bar). First
  exercised by the Aug 21 04:30 post-hold cycle, which also publishes: if
  a hinge-seller clears the bar, v32 ships on a hinge-native base.

* **2026-08-20 07:15 — crown HOLD VINDICATED; no gate change.** The direct
  paired A/B (64 games, serve) went 0–64 AGAINST the held candidate — the
  incumbent beats it every seed head-to-head (mean margin −555: near-mirror
  dynamics, the better-timed schedule edges every game). Method lesson
  recorded: `ab_test` head-to-head and the crown panel answer DIFFERENT
  questions (direct play vs performance-against-the-field); a candidate
  must convince on the field panel, and +6pp under the bar plus a 0–64
  head-to-head is a keep. The flagship's ~1700 ladder level is therefore
  genuine field improvement — the fix is stronger candidates from
  hinge-era data (the pool is rich: 23,693 same-day fit routes today),
  not a looser gate. Tomorrow 04:30 publishes v32 (hold expires tonight).

* **2026-08-20 07:00 — DECISION POINT: three verdicts.**
  **(1) Ladder noise floor ≈ ±50 points.** The v29 A/A pair (two
  near-identical agents, equal windows) sits 36–56 points apart at 44h
  (route 1686/106 games, bandit 1742/114). Rating differences under ~100
  points between submissions are NOT evidence. Recorded as the calibration
  for every future pair read.
  **(2) Factory sell-schedule search PARKED after two honest fails.**
  Run 2 with proper power (288 cells, discordant 8–5, diff +0.87pp,
  p=0.58): the elite base's sell schedule has no transferable slack under
  local mutations. The harness is proven (searches, checkpoints, gates
  honestly) — the value must come from a richer search space (farm plan),
  not more re-rolls of this one.
  **(3) THE REAL FINDING: the flagship's true hinge-era rating is ~1700
  and falling — the field adapted to 1.32.7 faster than our base.** The
  crown has HELD on 92513718_s1 for 5 days (best candidate +6pp vs the
  +10pp bar) while the ladder marks the base down ~300 points from its
  1.32.6-era level. L1 (convergence) reveals truth, it does not add
  strength; the critical path is L3 — a hinge-era base. A high-n paired
  A/B (serve) of the held +6pp candidate vs the incumbent rebuild is
  running; if it sign-passes, the crown's +10pp margin threshold is
  blocking a real improvement and the gate gets a sign-test path (house
  doctrine: paired tests decide, not thresholds).

* **2026-08-19 01:00 — factory run 1: HOLDOUT FAIL (the gate worked).**
  Search gained ~+6pp on selection seeds; on 144 fresh cells candidate
  0.4653 vs base 0.4479, discordant 4–1, p=0.375 — positive direction, not
  significant, no agent built (seed overfit caught by design). Run 2
  scheduled 07:00 (post-cycle, cap-safe) with less overfit and more power:
  16 selection seeds, 48 holdout seeds, 300 gens, patience 60, --resume
  from run 1's checkpoint (`scripts/sell_search_run.bat`, task
  KaggSellSearch). Also fixed: cp1252 console crash on non-ASCII team
  names; checkpoint/resume added after a session-owned run was killed
  mid-search — long searches now run as scheduled tasks only.

* **2026-08-18 08:00 — PHASE 2 EXECUTED, two levers resolved, factory live.**
  v29 A/A pair shipped 06:31 (route 55589044 + bandit-clone 55589040; the
  rogue 55578702 retired from the active pair).
  **L4 DROPPED on evidence**: paired winner-vs-loser over 280 decided
  episodes with a clean day-3 draw — winners matched the drawn shop's
  products in 116 vs 128, pooled p=0.48, flat across all 8 draws
  (`models/shop_study.json`). Caveat recorded: evidence is 1.32.6-era;
  re-test when hinge-era volume accumulates.
  **L6 RESOLVED, mostly already shipped**: engine verification
  (`_process_market`) shows per-unit lockstep marginal pricing — within-turn
  reorder-by-revenue is mathematically void; what matters is QUEUE-INDEX
  priority on collisions (8,554 contested dumps/73 games, 4,214 exact ties,
  net race value +2,218/game per contested_dumps) — and `tape_runtime`
  already ships `_sell_first` at line ~1009. Residual (contested-first among
  sells) judged second-order.
  **L5 LAUNCHED**: `src/kaggriculture/pipeline/sell_search.py` — sell-schedule search on `kagg
  batch` (farm plan fixed = legality-safe; market channel mutated: move/
  split/merge/resize), elite panel, common seeds, elitist (1+16), FINAL GATE
  = fresh-seed paired sign test vs base. Smoke-tested; full run in progress.
  Pipeline integration: a PASSED factory agent enters the tournament as
  `factory::<path>` (stage_tourney) and `stage_build` copies a factory
  winner instead of rendering — the crown still decides. 16/16 suites green.

* **2026-08-18 00:30 — PHASE 1 IMPLEMENTED (operator GO), live for the
  04:30 run.** Adaptive flip removed from `stage_second` (retired, 5
  measurements 0 wins); `second_slot_force()` honors the dated marker →
  tomorrow ships **route + bandit (A/A pair)**; `release_held()` in
  `daily_release` skips the publish chain 19th–20th (cycle keeps
  training/crowning); bonus fix: 429-storm circuit breaker in `sameday.py`
  (trip at 40 consecutive failures, 55-min cooldown, 25-attempt probe,
  release never capped) after Kaggle throttled episode endpoints 30+h.
  Tests: `test_second_slot` 7 checks, `test_quota_guard` 7 checks, **16/16
  fast suites green**. Found during pre-flight: rogue submission 55578702
  (not from this pipeline, score 561.6) retired v28.0_route2 — the v29 pair
  retires it in turn at 04:30. NEXT: Phase 2 studies after the release
  lands (shop-draw conditioning, multi-SELL audit, factory bring-up).
* **2026-08-18 00:20** — added §0 (Rust engine reality + R1–R3), §1 (3000
  decomposition L1–L7), §5 calendar. Phase 1 still awaiting operator GO.
* **2026-08-18 00:05** — initial plan (phases 1–3, measurement rules).
