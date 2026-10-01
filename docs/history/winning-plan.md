# Winning plan — researched 2026-08-21

> **Execution log, 2026-08-21 late evening (Phases 0–2 executed).**
> * Gauntlet built (`data/gauntlet/`: V16-RC5, Kaito v27, Rayk C94/C95,
>   Adaptive-R1 + our v19_2/v22_2/v24_1/v25) and cross-measured
>   (`src/kaggriculture/measure/gauntlet.py`, serve substrate, exact-bank verified). Elite-panel
>   ranking: V16-RC5 0.767 > our v25/kss 0.667 > Adaptive-R1 0.633 >
>   C95 0.533 >> Kaito v27 = our v22_2 0.200.
> * Fresh top-team mining: top-5 teams are ADAPTIVE (field-sim 0.05–0.30,
>   uncopyable — the deep reason v32 died); stable teams (SaiKushal 0.992,
>   Izzoudine 0.926, peikopon 0.926) reconstructed by majority vote — ALL
>   failed the gauntlet veto (0.04–0.25). Fresh-tape copying is a dead end
>   in the current meta; both plan candidates from this lane were rejected.
> * v32 root cause closed (safe_market clamp + DROP-only projection vs the
>   engine's PLACE deposit path; see BUILD_JOURNAL). Shell fixed; fixed
>   builds sweep their old-template selves (val 12-0; kss 2-0 smoke).
> * `src/kaggriculture/measure/strict_future.py` chronological veto gate built; both finalists
>   PASS on 25 post-selection tapes from ≥2400-rated teams (val 33-17
>   p=0.033; kss 34-16 p=0.015).
> * Feed-pin measured for the shipped routes: kss already opens wheat@slot0
>   (inert); for val it cuts feed-denial loss margins from −21k/−22k to
>   −4.3k/−4.6k at equal W/L. Sell-first left ON (family-validated).
> * Second-slot A/B: val-bandit vs val-route = no sign-tested difference →
>   bandit loses the seat (house rule) → route2.
> * **SHIPPED: v33.0_route (Valmorlee 91464294_s1, fixed shell) +
>   v33.0_route2 (kss 92513718_s1, fixed shell)** — a deliberate
>   two-lineage ladder experiment. Judge at 100+ games (~48h); the
>   converged levels are the Phase-2 measurement for both lineages.
> * Auto-publish remains OFF (KaggricultureRefreshCycle disabled);
>   release-hold + expired force markers removed; submissions are manual
>   designed experiments only.

*Every claim below carries its source: a Kaggle discussion/notebook, an engine
measurement, or our own submission record. Nothing is assumed. Supersedes the
strategy sections of `pair-improvement-plan.md` (which stays as the
implementation ledger); this doc is WHY + WHAT + the calendar.*

---

## Part 1 — Why the scores declined (the full causal chain, evidenced)

### 1.1 The rating is path-dependent; we churned through it

* **Byte-identical agents diverge by 300–1400 points.** Rayk Kretzschmar
  (23rd): two identical submissions 2h apart → 1700 vs >3000
  ([discussion 734000](https://www.kaggle.com/competitions/kaggriculture/discussion/734000)).
  evnchn measured a 300-pt identical-copy gap. Rayk's notebook: same SHA-256
  c14 read 2182 vs 1210 in two instances.
* **A fresh copy has <5% chance of overtaking a converged copy before ~150
  games** (Ryo Hasegawa, 1st place,
  [discussion 736219](https://www.kaggle.com/competitions/kaggriculture/discussion/736219)).
  Convergence: ~90% by 60 games (~5h), then +50–70 per 100 games, ±25–50 noise.
* **What we did:** the daily release re-submitted every day. We retired
  `v19_2_route` at a **91.1% live win rate** (90 games, still climbing) and
  `v19_1_route` at **90.8%** (76 games) to make room for "improvements".
  Every reset returned us to 600 on a fresh luck path. The "decline"
  v24→v29 (2339→1649) is this churn plus young-rating noise — v25.0_route
  and v26.0_route are byte-identical (verified by hash) and scored 2286 vs
  2002.

### 1.2 We swapped a strong base for a weaker one, on a bad instrument

Our own per-submission ladder record (`src/kaggriculture/data/ourgames.py --by-submission`,
2026-08-21) — converged entries only:

| Sub | Agent | Base | Games | Win% | Converged score |
|---|---|---|---:|---:|---:|
| 55412820 | v22_2_bandit | **Valmorlee 91464294_s1** | 103 | **71.8%** | **2776.4** |
| 55397322 | v22_1_bandit | Valmorlee 91468035_s1 | 94 | 59.6% | 2768.0 |
| 55314513 | v18_route | THUNDER 90521287_s0 | 114 | 62.3% | 2315.6 |
| 55412832 | v19_2_route | Valmorlee 91464294_s1 | 90 | **91.1%** | 2262 (climbing, retired) |
| 55477866 | v24.1_route | **kss 92513718_s1** | 118 | 71.2% | 2141.7 |
| 55523976→55589044 | v25–v29 (all kss) | kss 92513718_s1 | 60–120 | 45–62% | 1649–2287 |

**The Aug-13 crown moved us from the Valmorlee base (2768–2776 converged) to
the kss base (1650–2140 converged).** The offline tape-vs-tape crown made
that call, and the offline crown is proven anti-predictive
(`.local/memory/offline-measure-doesnt-predict-ladder.md`: +12pp "winners"
sourced from 1855–1960-rated teams). The decline was partly real — a base
regression — chosen by a measure that cannot see it.

### 1.3 The v32 frozen-copy bet (411.8 / 320.2)

A 720-turn frozen tape of a 2900-rated team misfires against arbitrary live
opponents. Syed Asad Ali (253rd, in Ryo's thread): "reconstructions of
adaptive agents are frozen and can mislead." The public 2913-scoring
V16-RC5 is *not* a bare tape — it is a majority-vote route reconstruction
**plus a reactive shell** (hands-count alignment, weed repair, conservation
sell-lead). The bare tape without the shell is what we shipped as v32.

### 1.4 The population moved under our feet

* Engine 1.32.7 landed (~Aug 14 announced,
  [735311](https://www.kaggle.com/competitions/kaggriculture/discussion/735311));
  stale 1.32.6-era agents "are dropping out as we speak" (evnchn), so a base
  mined Aug 9–10 decays even at fixed code.
* The meta converged hard: **26/30 top teams share one 1-COW/4-SHEEP
  HIRE4/5 opening** (Kaito v27 notebook, frozen top-30 audit). A single
  field hash appeared in 144/200 episodes across 40 teams (Rayk §4.7).
  Differentiation moved entirely into **market timing**.

### 1.5 We optimized the wrong number entirely

The live rating is not the final ranking. Staff (María Cruz,
[731587](https://www.kaggle.com/competitions/kaggriculture/discussion/731587)):
after the Sep 23 deadline, submissions keep playing ~2 weeks, then **one
Bradley-Terry tournament over those games produces the final leaderboard**.
Bovard ([736187](https://www.kaggle.com/competitions/kaggriculture/discussion/736187)):
no midterm eval, but "the scores your submissions converge towards are a
useful signal" — converged rating ≈ final BT. So:

> **The only number that matters is the converged W/L strength of our final
> two agents against the post-deadline population. The live score is our
> measurement instrument, not our objective.**

---

## Part 2 — What wins (research findings, each with source)

1. **Win conversion vs strong peers is everything.** Rayk's C70: 83–5 with
   +14,196 mean margin — stalled below 3000, because the 5 losses were
   exactly against the peers that matter. W/L only; margin buys nothing.
2. **Opening feed denial** is a documented top loss mechanism: opponent buys
   14–19 wheat before your slot-8 five-wheat buy → you afford 4 → a sheep
   starves day 2 → avg **-13,606** (4 losses, Rayk §15.1). Fix that fixed
   every sampled case: **move the existing 5-wheat buy to market slot 0 on
   turn 0** (buying 6 regressed; buying 14–19 + resell was brittle).
   ⚠ Our `tape_runtime._sell_first` moves SELLs ahead of ALL buys every
   turn, turn 0 included — we are *more* exposed than the meta baseline.
3. **One-turn sell preemption with conservation** decides near-mirror games:
   11 of C92's losses came from opponents selling fertilizer/wheat one turn
   before its batch (~5,300–5,700 swing). The winning bounded fix (C94):
   *when the route will sell fertilizer next turn, sell ≤10 units now and
   subtract the same amount from the next sale.* Held-out 174–6 (96.7%),
   and beat the wheat+fertilizer variant 6-0 in the final (Rayk §16).
   V16-RC5 uses the same conservation lead for MELON/MILK/STRAWBERRY/WOOL.
4. **Weed-blocked-route repair** (C92): repair only a weed sitting under the
   route's own PLANT/BUILD/PLACE, delay that actor one turn, resync at next
   PASS. Turned a -7,288 loss into +3,455. "Clear every weed" was never the
   policy; broad cleanup measured neutral (C90/C91 17-17-66).
5. **The fourth quadrant is negative, broadly.** Every 4-quadrant variant
   lost 10-0 to C92 (450 games); even productive Seb 4-quad routes lost 10-0
   (Rayk §13). Do not spend on SE.
6. **Terminal window:** step 718 executes, action index 719 does not; move
   terminal takeover from 712→717 was a pure win (c27). Final shed must be
   liquidated (unsold inventory scores nothing).
7. **1.32.7 hinge is a tail effect — do not overfit it.** destbreso
   (independent replication in 735311): fires in tomato 55% / carrot 28% /
   egg 26% of games, but the *median* game pays 1.00× the old revenue;
   switching builds changed **0 of 224 winners** on fixed tapes. Carrot
   `below_target` also moved 0.20→1.00 (undocumented). Bounded endgame
   scarcity-selling in genuinely drained games is the only justified play.
8. **The game is non-transitive.** dzjiann's 960-game round-robin
   ([736439](https://www.kaggle.com/competitions/kaggriculture/discussion/736439)):
   B85 > Andrews (30-2) > Kaito v35 (21-11) > B85 (24-8). Opponent×opening
   interaction predicts outcomes (67.5%) far better than either alone
   (56.5%). Consequence: the 2nd slot should be a **deliberately different
   family** (Ryo's hedge advice), not a twin.
9. **The post-deadline population is largely knowable.** It will contain the
   top teams' originals plus many copies of the public notebooks
   (V16-RC5 2913, Strict-Future v25 2905 / v27, Clone-Preemption V14 2833,
   Adaptive-R1, Multi-Route 2462, Rayk's C94/C95 lineage 2837–2945). All of
   these embed executable `main.py` sources — **we pulled them; they run as
   real reactive agents locally**. Public "gets 3k" claims are over-claimed
   (evnchn), but as *population members* they are exact.
10. **Validation that works:** (a) paired seeds, both seats, per-opponent
    records, veto opponents (Rayk §8); (b) **strict-future chronology** —
    freeze the candidate, then gate on episodes recorded *after* the freeze
    (Kaito v27: 25/27 on post-freeze episodes while the incumbent read
    14/27); (c) the ladder itself at 100+ games as the only final oracle.

---

## Part 3 — The plan

**Budget reality:** entry deadline Sep 23 (33 days). 2 active slots; a
ladder experiment needs ~48h to converge (100+ games). Keeping one anchor
slot aging, we can run **~10–12 clean ladder experiments** — that is the
whole budget for the rest of the competition. Current live pair: v32
(411.8/320.2) — both slots are effectively free experiment slots right now.
Position: rank 4630. Bands (2026-08-21 LB): top-100 = 2558, top-25 = 2733,
top-10 = 2898, top-5 ≈ 3005+.

### Phase 0 — instrumentation before any submission (~1 day)

* **P0.1 Public gauntlet.** Extract the embedded agents from the pulled
  notebooks (already in scratchpad; land under `data/gauntlet/`):
  V16-RC5, Kaito v27, Rayk C94 + C95, Adaptive-R1, Multi-Route, plus our
  own v22_2_bandit / v19_2_route / v25.0_route. Verify each runs on
  `serve_match.py` (1.32.7, exact-bank). This replaces the frozen-tape
  panel as the primary offline instrument — these are *reactive* agents.
* **P0.2 Strict-future gate harness.** A freeze-timestamped gate: candidate
  is frozen, then evaluated only against opponent tapes whose episodes were
  recorded after the freeze (we ingest hourly; the data exists). Add to
  `refresh_cycle` as the crown's *veto* stage.
* **P0.3 Fresh top-10 mining on 1.32.7.** Re-mine the current top-10 teams,
  **score-matching submission only** (Rayk's downloader lesson — a team's
  newest submission is often not its scoring one), ≥3 replays per team,
  majority-vote reconstruction (boatlee's method). Source-team rating check
  stays mandatory (memory rule).
* **P0.4 No-submission policy stays** until Phase 2's designed experiment.

### Phase 1 — build the flagship (~2 days)

* **P1.1 Base:** majority-vote route from a current top-5 team (P0.3),
  chosen by source-team CURRENT ladder rating + gauntlet cross-play, never
  by tape-vs-tape crown alone.
* **P1.2 Shell fixes on `tape_runtime` (each one bounded, each with a
  paired A/B before it ships):**
  1. `_sell_first` exempts turn 0; pin the feed WHEAT BUY to slot 0
     (finding #2). Audit any other feed-critical buy turns.
  2. Verify/align our `_adaptive_sell`/`_adaptive_repay` with the C94
     conservation rule (fertilizer, cap 10, subtract from next sale); add
     the V16-RC5-style premium one-turn lead *only if paired-positive*.
  3. Verify `_weed_repair` matches C92 semantics (blocked-productive-action
     repair + resync; no broad cleanup).
  4. Confirm terminal takeover at 717 and full shed liquidation by 718.
  5. Bounded hinge endgame: only in detected deep-scarcity games (shop
     drained vs knee: tomato 200 / carrot 450 / egg 332), late-sell held
     hinge items; inert in the median game (finding #7).
* **P1.3 Gate:** full gauntlet, paired seeds, both seats, per-opponent
  records; veto = a losing record vs any gauntlet member our incumbent
  beats. Then the strict-future gate (P0.2). Then latency + self-play
  validation (existing `submit.py` checks).

### Phase 2 — one deliberate ladder experiment (48h each)

* Submit the flagship into a v32 slot. **Judge only at 100+ games / 48h**
  (Ryo's table; never before 60 games). Instrument: hourly
  `ourgames`/rating-track already runs.
* Read per-opponent W/L, not the score. Success band: converged ≥2550
  (top-100 zone). Whatever the result, download our actual losses and run
  the C94-style loss audit (mechanism, not margin) before designing the
  next change.
* **Never resubmit unchanged. Never fix a 30-point wiggle.** (<50 pts =
  noise, per Ryo.)

### Phase 3 — iterate + hedge (weeks 2–4, ≤2 experiments/week)

* **Anchor discipline:** best agent ages untouched in slot 1. All
  experiments go to slot 2.
* **Iteration source:** our own live losses (the only in-distribution
  signal). Each iteration = ONE bounded mechanism (the Rayk discipline),
  paired-gated, strict-future-gated, then one ladder experiment.
* **The hedge (2nd final slot):** a deliberately different family for the
  non-transitive field (finding #8) — candidate lanes, in order:
  (a) our bandit/identifier chassis on the new base with arms retrained
  against the *public gauntlet* classes (it out-scored its route sibling
  2776 vs 2262 on the same base once already);
  (b) a wheat-heavy Seb-family counter route;
  (c) Track P's planner if it graduates its 5-condition gate.
  The hedge must specifically beat the agents that beat the flagship in the
  gauntlet round-robin.
* **Weekly meta refresh** (Ryo's advice): re-mine top-10, refresh the
  gauntlet, re-run the round-robin, check for opening drift.

### Phase 4 — the freeze (Sep 16–23)

* By **Sep 20**: strongest agent + strongest hedge in the two slots, both
  ladder-validated ≥100 games, error-free.
* Sep 21–23: **no new submissions** (an erroring agent plays nothing and a
  600-start eats days of the post-deadline window's matchmaking). Verify
  both run clean; then hands off.
* Post-deadline: agents keep playing 2 weeks; the BT fit decides. Nothing
  we can do after Sep 23 — the work is all before it.

### Standing rules (the never-again list)

1. Never re-submit an unchanged agent; never retire a climbing one for a
   cosmetic reset.
2. Never judge below 60 games; decisions at 100+.
3. Offline measures RANK, only the ladder VALIDATES; every mined base gets
   a source-team current-rating check.
4. Every shipped change is one bounded mechanism with paired both-seat
   evidence + a strict-future pass.
5. No frozen tape ships without the reactive shell.
6. Daily auto-publish stays OFF; submissions are manual, designed
   experiments only.

### What "ensure the top" honestly means

Our proven converged capability is 2776 (Aug 10 population). Rank bands
today: 2558 = top-100, 2733 = top-25, 2898 = top-10. Phases 1–2 target
2550+ on the first experiment; the market-timing fixes + loss-audit loop
are the evidenced route to 2700–2900 (they are exactly what carried Rayk
1650→2945 and boatlee to 2913). 3000+ is top-5 and requires winning the
close games against the top-20 — achievable only through the Phase-3 loop,
not through any single trick. Every experiment's converged result is a real
measurement of final-BT strength (Bovard), so progress is verifiable
week by week.
