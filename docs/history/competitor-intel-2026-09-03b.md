# Competitor intel — same-day follow-up sweep, 2026-09-03 (afternoon/evening)

Read-only research. No agent, model, gate or engine file was touched; nothing
submitted; no kernel pushed. Raw pulls: `.local/mined/2026-09-03-b/`.
Leaderboard: `.local/freshness/lb/kaggriculture-publicleaderboard-2026-09-03T17_09_30.csv`
(newest snapshot, 17:09 UTC).

Read first, per the assignment: `docs/competitor-intel-2026-09-03.md` (full
7-kernel sweep, already covered #1 yhay81 Six-Day Fieldbook and #5 yhay81
Three-Day Shop Router, and boatlee's *other* kernel V29-R1 Adaptive Market
Hysteresis), `docs/closed-loop-intel-2026-09-03.md` (15-team production-fixed
audit, Crop Dusta the one exception), `docs/history/cropdusta-adaptive-study-2026-09-03.md`
(Crop Dusta's runtime rule fully reverse-engineered), `docs/history/gate-calibration-2026-09-03.md`
(panels anti-predict; held-out split is the only valid decision number).

**This sweep found no reason to revise anything in those four documents.**
What follows is new: two notebooks reviewed for the first time (#2
lynnsakurai, #4 boatlee's V14), one flag resolved with primary evidence (#3,
the "is this ours" kernel), and light re-confirmation of #1/#5.

---

## 0. The one finding that matters most

**#3 is resolved, with kernel-ID evidence, not narrative.** Kaggle kernel
`id_no` is a monotonically increasing, site-wide counter assigned at
creation. Pulling metadata for all three copies of the identically-titled
notebook gives an unambiguous creation order:

| owner | `id_no` | `is_private` | `lastRunTime` (UTC) | cells |
|---|---|---|---|---|
| **kaitofukami** | **129819535** | **false (public)** | 2026-08-06 04:28:18 | 14, full notebook |
| navazshfathi | 129886361 | false (public) | 2026-08-06 11:12:41 | 1, partial fork (code cell only) |
| **debmalya84 (us)** | **129893160** | **true (private)** | 2026-08-06 13:26:07 | 14, full notebook, byte-identical cell sources to kaitofukami's |

kaitofukami's kernel is the oldest of the three by both `id_no` and
run-time, and it is public. Ours is the newest, is private, and its 14 cell
sources are byte-identical to kaitofukami's (`diff` on parsed JSON, not
approximate). **kaitofukami's notebook is not a copy of ours. Ours is very
likely the fork** — the pattern (public original → our private study copy,
saved under the same auto-carried title, same day, ~9 hours later) is
exactly what "Copy & Edit Kernel" produces, and it is exactly the
`src/kaggriculture/data/routes.py --mine` / "mine the ladder for ideas" workflow CLAUDE.md
already documents as legitimate. This also resolves a standing loose end:
CLAUDE.md's house rule #6 already calls v21.1 **"the copied v21.1"** — we now
know precisely what it was copied from (this notebook or its immediate
public lineage), which matches perfectly.

**Exposure answer for the operator:** nothing of *ours* is exposed by this
chain. The only content flowing outward is kaitofukami's own v21.1 strategy,
which was already public before we ever touched it, and which is 20+ agent
versions and about a month stale (current lineup is v40–v44; see
`agents/`). Our own copy is private (`is_private: true`, confirmed by a
fresh pull), and our *current* production kernels (`kaggriculture-adaptive-bandit-private`,
confirmed private by a fresh pull; `kaggriculture-route-refresh-private`,
`kaggriculture-track-p-private`) are unrelated to this notebook chain
entirely. If there is a real exposure question in this competition, it is
the reverse direction and already flagged in the main sweep: **our own
public EDA/dossier kernels** (`kaggriculture-episode-data-a-gentle-eda`,
6 votes; `kaggriculture-agent-dossier-private` is itself private) were not
re-audited here — out of scope for this task, flagged for a future pass if
the operator wants it.

One more thing worth knowing about that notebook, found while resolving the
flag: `data/kernels/_src/177-180-fresh-top-30-v21-1-conditional-memory.py`
(our own prior decode) contains an assertion —
`for forbidden in ("Konstantin03", "Dennis Gioche", "Kaito Fukami",
"CanonicalTeamNames", "CanonicalTeamIds", "SubmissionIds"): assert forbidden
not in source` — i.e. the tool that *built* v21.1's 30-route memory bank
scrubs player-identifying strings (including "Kaito Fukami") before
embedding it. This confirms the mechanism description below: the memory
bank is mined from public replays (plural players, not just kaitofukami's
own), with authorship deliberately stripped — the same discipline our own
mining tools should already have, and worth a one-line check that they do.

---

## 1. Classification table

PRODUCTION: fixed embedded schedule vs. decided at runtime from the
observation. MARKET: fixed vs. price/opponent-reactive. Calibration
(`competitor-intel-2026-09-03.md` and `closed-loop-intel-2026-09-03.md`): 14
of 15 audited top teams are fixed-production; Crop Dusta (rank 2) is the one
confirmed runtime-production agent, branching on town consumption at fixed
clock times.

| # | notebook | author | rank / rating today | last run | PRODUCTION | MARKET | learned/selector component |
|---|---|---|---|---|---|---|---|
| 1 | Six-Day Public-State Fieldbook | yhay81 | **15 / 2875.8** (was 118/2638.2 this AM — freshness churn, see §4) | 2026-09-03 06:27 | fixed (8 tapes) | reactive (funding guard) | 4 depth-2 trees, 16 tests, select among 8 tapes — **SELECTOR**, offline-fit lookup table, not online learning |
| 2 | Farming Score — A Mathematical Approach | lynnsakurai | 245 / 2492.6 | 2026-09-03 14:28 | fixed (single `_ACTIONS` tape) | reactive | **not new**: `_adaptive_market` is boatlee's V29-R1 controller byte-for-byte (identical `_AM_CONFIG`, all 20 constants), re-credited to yhay81 ("Inspired by..."), plus one new 2-state regime gate (protect/convert/balanced) re-evaluated every 72 turns from day 18 — see §2 |
| 3 | 177/180 Fresh Top-30 \| v21.1 Conditional Memory | kaitofukami | 376 / 2348.1 | 2026-09-03 12:41 | **fixed** (single 719-turn tape) | **reactive, and it is a genuine SELECTOR**: k-NN match against 30 stored route "prototypes" (`signatures`+`sales`), current SELL set reordered from the nearest match, never creates a new SELL | mined-from-public-replays, 30-entry memory bank, anonymised at build time (see §0) — this is the shape the operator is hunting for, already tried by us as "the copied v21.1" with ambiguous field-adjusted results |
| 4 | 84/84 Base+Public Holdout \| V14 Clone Preemption | boatlee | 472 / 2233.1 | 2026-09-02 12:45 | fixed (single `_ACTIONS` tape, replay-reconstructed) | reactive, **clone-gated preemption, ENABLED** (contrast with boatlee's other kernel, V29-R1, where the equivalent `_front_run` is disabled) | none learned; near-clone gate (`_clone_distance <= 6`) + own-future-tape lookahead — see §3 |
| 5 | Three-Day Shop Router | yhay81 | 15 / 2875.8 (same team as #1) | 2026-09-03 06:27 | fixed (2 tapes) | funding guard only | the two numeric guard conditions are dead splits on our data (already established in the main sweep, §7 items 5–6); the whole router reduces to `first_shop ∈ {BAKERY, PET_CAFE}` |

---

## 2. #2 lynnsakurai — "Farming Score, A Mathematical Approach": not a math result, a relabelled heuristic, and it is testable — tested

The LaTeX in the notebook (`Q`, `ΔC`, `ΔK`, the epoch formula) is **notation
for an existing rule-based controller, not a derivation.** There is no
closed-form optimum, no proof, no equation solved for anything — it is
`boatlee`'s V29-R1 `_adaptive_market` liquidation controller
(`competitor-intel-2026-09-03.md` §3b: reserve ladders, price gates,
pressure hysteresis, mirror-guard) reproduced with an **identical
`_AM_CONFIG` dict, all 20 constants matching exactly** (`start_step: 456`,
`reserve_early/mid/late/tail: 12/8/6/3`, `shed_soft/hard: 72/88`,
`price_gate: 0.66`, `pressure_trigger: 2.0`, `tranche: 4`,
`mirror_latch_step: 289`, `mirror_composition_distance: 2`,
`mirror_money_distance: 250`, `max_extra_per_item: 18`, …), decoded from the
embedded payload in `.local/mined/2026-09-03-b/lynnsakurai_farming-score/decoded_main.py`.
The credit line in the notebook ("Inspired by Yusuke Hayashi") does not
mention boatlee at all, despite the core controller being boatlee's code.

**What is actually new** is a second layer on top: a `late_mode` regime
switch (`balanced` / `protect` / `convert`), recomputed only at 72-turn
epoch boundaries starting day 18 (steps 432/504/576/648), verified against
the code (`_am_update_late_mode`, lines ~330–370 of the decoded source):

```
protect: cash_gap >= 1000  AND  price_index <= 0.78  AND  total_shed < 72  AND  capacity_gap <= 0
         -> reserve +2, sell-gate +0.04 (hoard; sell less)   [only while step < 648]
convert: cash_gap <= -1000 AND  price_index >= 0.82  AND  capacity_gap >= 0
         -> reserve -2, sell-gate -0.04, tranche +2 (liquidate faster)
otherwise: balanced (no shift)
```

`cash_gap` = own money − opponent money (public). `price_index` = mean of
price/base over STRAWBERRY/MILK/WOOL (public). `capacity_gap` = summed
opponent-minus-own tile/animal count for the STRAWBERRY/COW/SHEEP
producers (public, but **not present in our 49-field `turn_features.py`
trace** — see limitation below).

**Tested against our own data** (no new fetching; 3,000 randomly sampled
episodes from `data/turntrace/`, both seats, script in
`.local/mined/2026-09-03-b/check_lynn_regime.py`): the two-condition
`(cash_gap, price_index)` part of the rule is **not degenerate** —
unlike several of yhay81's tree splits in the main sweep, which turned out
to be always/never-true. Marginal fire rates on real games (ignoring the
`capacity_gap` and shed/near-mirror gates, which our trace can't compute):

| step (day) | protect fires | convert fires | cash_gap p50 | price_index p50 |
|---|---|---|---|---|
| 432 (18) | 8.7% | 24.8% | 0 | 0.95 |
| 504 (21) | 19.9% | 16.9% | 0 | 0.74 |
| 576 (24) | 30.0% | 8.8% | 0 | 0.49 |
| 648 (27) | 31.1% | 8.9% | 0 | 0.51 |

Both conditions bind real games at every checkpoint, at rates that shift
sensibly with the season (protect gets easier late as prices sag; convert
gets rarer for the same reason) — a plausible, non-degenerate regime
detector, though the check is partial (missing the `capacity_gap` gate and
the shed/near-mirror qualifiers, which need per-tile replay parsing out of
scope for this pass).

**Verdict:** no math claim to check — there wasn't one. The underlying
controller is already a `TEST`-tier item in the main sweep
(`competitor-intel-2026-09-03.md` item 8). This adds one testable refinement
(the protect/convert cash-gap+price-regime gear shift) that is *not*
degenerate on our data and could be folded into the same experiment.

---

## 3. #4 boatlee V14 — "clone preemption": mechanism, provenance, and how it differs from our relay

### 3a. What it does, exactly (verified against the decoded `main.py`, `.local/mined/2026-09-03-b/boatlee_v14-clone-preemption/`)

`_preempt_shift` fires only while `120 <= step < 680` **and** the
public-state clone-distance gate passes: `_clone_distance(obs) <= 6`, where
clone distance is an L1 distance over `(hand count, 3×quadrant count,
per-species tile/structure counts)` between the two farms. When it fires, it
looks at **its own next-turn scheduled tape** (`_future_sells(step)`, read
straight from the embedded `_ACTIONS[step+1]`, not from any model of the
opponent) for STRAWBERRY/MELON/MILK/WOOL, and pulls forward
`min(remaining shed stock, next-turn quantity, 30, next-turn quantity × 2)`
units into *this* turn's SELL, then deducts the same amount from next
turn's SELL via `_repay_shift` so the two-turn total is conserved
(`quantity conserved; timing changed`, their own words).

**This is not opponent modelling.** It is: *"if the opponent's farm looks
enough like mine (clone distance ≤ 6), assume their tape sells the same
premium product on the same turn mine does, so beat it there by one turn."*
It never reads the opponent's shed, sell history, or price-impact
footprint — the only opponent signal used is the coarse clone-distance
gate.

### 3b. Provenance — undisclosed overlap with yamakawanin's Moon

boatlee's own provenance section (cell 9, quoted in full) credits: replay
reconstruction of Kakuteki's route, weed-repair/price-ranking reuse from
**Kaito V23**, and clone-distance/shift-repay inheritance from **boatlee's
own V13-R3**. It does not mention yamakawanin at all. But the parameter set
is a near-exact match to Moon's `_preempt_shift`
(`competitor-intel-2026-09-03.md` §4b): function names `_preempt_shift` /
`_repay_shift` identical; `_PREMIUM = (STRAWBERRY, MELON, MILK, WOOL)`
identical; `_PREEMPT_START = 120` identical; `_PREEMPT_STOP = 680`
identical; `_PREEMPT_MAX_CLONE_DISTANCE = 6` identical;
`_PREEMPT_MIN_FUTURE_QUANTITY = 4` identical. Only `_PREEMPT_MAX_BATCH`
differs (30 here vs. Moon's 12). Given boatlee's own timeline places this
mechanism at V13-R3 (an earlier kernel than V14), the most likely reading is
that this specific preemption shape — not the funding guard, not the
adaptive-market controller, a third distinct thing — is itself a converged
public technique with at least two independent lineages (boatlee's
V13-R3→V14, yamakawanin's Moon), neither crediting the other. Not
resolvable further from outside; flagged, not asserted.

### 3c. Different from our relay, and would it beat it?

`src/kaggriculture/data/contested_dumps.py`'s docstring states its own bet plainly: "the relay
layer exists on one bet — that when both farms are about to dump the same
product in the same window" — i.e. **our relay is a real collision
detector**, measured at 798/1175 collisions won (+1,914/game). boatlee's V14
mechanism is a **blind, clone-gated, one-turn pull-forward of its own
tape** — it never checks whether a collision is actually imminent; it fires
whenever the *farms look alike* and its *own* tape has a premium sale next
turn. These are different mechanisms answering different questions: ours
detects "is the opponent about to sell the same thing," theirs assumes
"if we're clones, they probably will."

Whether it would beat our relay is not answerable from outside without
running it — but two things bound the plausible size of any effect. First,
it is narrow: `_PREEMPT_MAX_CLONE_DISTANCE = 6` only fires against
near-mirror openings, which is a minority of games against a diverse field
(our own mirror-guard work and boatlee's V29-R1 near-mirror latch both
treat "opponent looks like me" as a rare, specific condition, not a
default). Second, **boatlee's own later kernel disables the sibling
mechanism**: V29-R1's `_front_run` (the same "pull forward a scheduled
premium sale by one turn" idea, applied without the clone gate) ships with
`_FR_ITEMS = ()` — off. Read together, boatlee's own trajectory across two
kernels is: keep preemption narrowly scoped to near-clones (V14, enabled),
retire the general form (V29-R1, disabled). That is a data point *against*
generalizing this mechanism beyond the near-mirror case, not for it.

**The confirming measurement, if the operator wants one:** replay
`src/kaggriculture/data/contested_dumps.py --losses` restricted to games where the opponent's
clone-distance-at-step-120 (same statistic boatlee uses) is ≤ 6, and check
whether our relay's win rate on that subset differs from the general
798/1175. If it's already at or above boatlee's implied edge there, their
mechanism adds nothing we don't already have; if it's measurably worse on
near-clone games specifically, that is the one subset worth testing a
clone-gated pull-forward against. Not done here (would need our own agent
source, out of scope for a read-only pass).

### 3d. The holdout protocol — this is the useful part

Boatlee's evaluation design is the cleanest public example of holdout
discipline found in either sweep, and it **fails honestly in public**,
which is the tell that it's real:

- A 6-agent "historical public strong agents" panel (kaito_v22/v23,
  andrew_v22_terminal, v13_r3, rayk_v21_impact, indar_verified) is used to
  fit/select the production route and to sweep 5 market-layer variants —
  all 5 score a perfect 12-0 on this panel, so the reported selection
  criterion is **margin among ties**, not win rate (worth flagging: this is
  the same shape as our own `dual_bar_miner` overlap-with-selection problem
  in `gate-calibration-2026-09-03.md` — a panel everyone clears cannot
  discriminate, and picking on margin among ties can still overfit).
- Four genuinely separate panels are then run and **reported including
  losses**: "untouched Ueddy replay proxies" (48-0 win), "later-sampled
  *current* CemBas replay proxies" (48-0 win, explicitly drawn from replays
  dated after V14 was built), "Seb *alternate-route* replay proxies"
  (**4-44, a loss-heavy record, published as-is**), "wangtf96 *alternate
  strategy*" (**4-8, also loss-heavy, published as-is**).

That is a genuine held-out test — opponents and routes not used to fit
V14 — and the notebook does not launder the losses out of its headline
"84/84" framing; a reader who opens the notebook sees the 4-44 and 4-8 rows
in the same table. This directly answers the operator's question: **yes,
this is a real holdout protocol**, and its discipline (partition the
panel into fit-set vs. genuinely-new opponents, report every panel's W-L
including the losing ones, hash-lock the artifact before evaluating) is
worth explicitly matching in our own release checklist, which
`gate-calibration-2026-09-03.md`'s STANDING RULES already move toward
(rule 2: "the held-out half is the decision number") but does not yet state
as bluntly as "publish the losses in the same table as the wins."

---

## 4. #1 / #5 yhay81 — confirmed, one correction

Both notebooks were already fully decoded and analysed in
`competitor-intel-2026-09-03.md` §2 (delivery mechanism, routing trees,
funding guard, the 0.9948/0.6322 split) — re-pulling would be redundant, and
nothing in a light re-check contradicts that write-up.

**One correction, not to the mechanism but to the standing rank number**:
the main sweep recorded yhay81 at rank 118 (2638.2) this morning. The
17:09 UTC leaderboard snapshot puts the same team at **rank 15 (2875.8)**.
This is not a new agent or a changed strategy — it is the same freshness
churn the main sweep's §1 (destbreso's discussion) and §7 check #9 already
documented (97% of the top 200 submit within 24h; median top-100
last-submission age ~6h). It is worth recording as a second, same-day data
point for that finding, not as new content about the agent itself.

---

## 5. Prioritised: adopt / test / ignore

### ADOPT

1. **Publish losses in the same table as wins, always** (boatlee's V14
   holdout discipline, §3d). Cheap, no code change — a reporting norm for
   every candidate write-up, reinforcing `gate-calibration-2026-09-03.md`
   rule 2.
2. **Verify our own replay-mining tools scrub player identity the way
   kaitofukami's build tool does** (§0's `assert forbidden not in source`
   check). One-line audit of `src/kaggriculture/data/routes.py` / `src/kaggriculture/data/backfill_ingest.py`
   output for embedded team names or submission IDs. Not urgent (no
   evidence we currently leak this), but cheap to confirm.

### TEST

3. **The `protect`/`convert` cash-gap + price-regime gear shift** (§2) —
   confirmed non-degenerate on 3,000 of our own episodes at all four
   72-turn checkpoints. This is an incremental addition to the
   already-`TEST`-tier adaptive-liquidation controller
   (`competitor-intel-2026-09-03.md` item 8), not a new standalone
   experiment — bundle it with that item if/when it's run.
4. **Clone-distance-conditioned subset check on our own relay** (§3c) —
   `src/kaggriculture/data/contested_dumps.py --losses` restricted to clone-distance-≤6 games,
   to see whether our existing collision detector already dominates
   boatlee's blind near-clone pull-forward on the one subset where their
   mechanism could plausibly matter. Cheap (existing tool, existing data,
   no new fetching), and it directly answers "would it beat our relay"
   instead of leaving it as inference.

### IGNORE

5. **The "mathematical approach" framing itself.** There is no closed-form
   result to adopt or refute — the LaTeX describes an existing heuristic,
   most of which (the base `_adaptive_market` controller) is already a
   known, unshipped `TEST`-tier item under a different author's name.
6. **Generalizing boatlee's V14 clone-preemption beyond near-mirror
   openings.** boatlee's own later kernel (V29-R1) disables the
   un-gated version of the same idea (§3c) — their own trajectory argues
   against it, not for it.
7. **kaitofukami's v21.1 "Top-30 memory" selector as new material.** It is
   a real selector (k-NN over 30 mined route prototypes, market-layer
   only) and structurally the shape the operator is hunting for — but it
   is not new to us; it is what CLAUDE.md already calls "the copied v21.1,"
   already measured, already superseded by four version generations. Re-run
   only if the operator wants a field-matched (same opponent-rating band as
   v18's 75.7%/2342.9 measurement) re-test, which was not done here.

---

## 6. What is unknowable from outside, stated plainly

- Whether boatlee's V14 clone-preemption genuinely originates from V13-R3
  independent of Moon, or one lineage quietly reused the other's code
  (§3b). Both fit the evidence equally; the parameter match is too exact to
  be coincidence, but direction of borrowing is not recoverable without
  private history from either author.
- Whether V14's clone-preemption would net-beat our relay. Bounded by
  argument in §3c, not measured; the confirming measurement is specified
  and cheap but requires our own agent's collision-detection code, which
  this read-only task did not touch.
- Whether kaitofukami's v21.1 route-memory bank (§0, §1) was mined
  partly from his *own* replays vs. purely from third parties — the
  anonymization assertion strips his name from the *output*, which says
  nothing about the *input* corpus.

## 7. Contradicts nothing already on record

Every claim in `competitor-intel-2026-09-03.md`,
`closed-loop-intel-2026-09-03.md`, `cropdusta-adaptive-study-2026-09-03.md`
and `gate-calibration-2026-09-03.md` that this sweep touched (yhay81's
mechanism, boatlee's V29-R1 controller and its disabled front-run, the
production-fixed/market-reactive split, the panel-anti-prediction lesson)
is corroborated, not contradicted, by this pass. The one genuinely new
factual correction is the yhay81 rank number (§4), which is churn, not a
retraction.

---

## Appendix — artefact inventory

```
.local/mined/2026-09-03-b/
  ours_177-180/kernel-metadata.json         # id_no 129893160, is_private=true
  kaito_177-180/kernel-metadata.json        # id_no 129819535, is_private=false
  navaz_177-180/kernel-metadata.json        # id_no 129886361, is_private=false
  check_priv/kernel-metadata.json           # kaggriculture-adaptive-bandit-private, is_private=true (confirmed)
  lynnsakurai_farming-score/
    farming-score-a-mathematical-approach.ipynb / .extracted.txt
    decoded_main.py                         # "V31 three-day expiring late router with capacity confirmation"
  boatlee_v14-clone-preemption/
    84-84-base-public-holdout-v14-clone-preemption.ipynb / .extracted.txt
  check_lynn_regime.py                      # protect/convert non-degeneracy check, 3,000-episode sample
```

Pre-existing local artefacts used, not re-fetched:
`data/kernels/mine_v21/`, `data/kernels/kaitofukami_177-180-fresh-top-30-v21-1-conditional-memory/`,
`data/kernels/navazshfathi_177-180-fresh-top-30-v21-1-conditional-memory/`,
`data/kernels/_src/177-180-fresh-top-30-v21-1-conditional-memory.py`,
`data/kernels/_agents/177-180-fresh-top-30-v21-1-conditional-memory___AGENT_B85_PARTS.py`,
`data/notebooks.csv`.
