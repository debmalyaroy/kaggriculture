# Public field analysis + top-10 plan (2026-09-16)

Scope: decode the current public Kaggle field, place our two lanes (bandit,
trackp) against it, and lay out what each needs to reach top-10, including how
to use the external datasets/models the operator supplied. **Every architecture
claim below was read off the actual submission code (notebooks decoded to
standalone `agent(obs)` files), not from memory.** Extracted agents live in
`.local/scratch/pub_notebooks/*/agent_extracted.py`; the distinct ones are
staged in `data/gauntlet/pub_*.py` and registered in `band_panel.REACTIVE_OPPS`.

---

## 1. Headline

**The entire public frontier is ONE architecture — the same one our bandit
already uses:** a set of pre-computed 719-step action *tapes*, dispatched at the
day-6 shop boundary on the observed first-two `unlocked_shops`, executed
open-loop, wrapped in a thin reactive shell (weed repair, sell-timing, endgame
liquidation). There is no per-turn searcher and only two exploratory learned
nets in the whole field, neither with a claimed rating.

So the race is **not** about a smarter runtime algorithm. It is about **(a) the
strength of the baked economy and (b) how well the tape is matched to the
world.** That is exactly where our bandit's measured gap is (the D9–D24
mid-game ramp), and exactly what the external episode datasets are fuel for.

**Critical caveat that governs everything:** the strongest public *claim*
(Tschinkel "93.8%") is an **offline-frozen** number — replay vs 44,096
*non-reactive recorded* opponents. That structurally inflates win-rate and is
the precise failure mode our own CLAUDE.md warns about. No public agent has a
*verified live* rating above our live bandit (2336). Our v57 pair already sweeps
this field 1.000 offline. **Therefore the improvements here must be validated on
reactive opponents and (ultimately) the live ladder — not on a frozen panel.**

---

## 2. The field: two originals, everyone else a fork

| Agent (notebook) | What it really is | Dispatch | Economy | Sell logic | Learned? |
|---|---|---|---|---|---|
| **thomastschinkel** 93.8% router | **5 SPECIALIST tapes + learned CART trees** | YARN_STORE & MILK-demand @D6; CARROT price @D24 | per-world: cow/milk (9c/5s) vs sheep/wool (11s/0g) | brute oversized "sell-all", partial-fill | **CART trees** (dispatch only) |
| **yhay81** Shop Router 0909 (= prvsiyan repackage) | 13 GENERALIST tapes | rules table on first-two shops @D6; D27 endgame swap | one wheat/carrot generalist, herd 6c/10s | aurax7 front-run + projected-shed + liquidation | no |
| flexon V45 (== reyhanksatria) | "frankenstein": assembled fork of both + **bounded terminal (1+λ) search** @step712 | 64-key shop-pair table @D6 + D27 | full 9-product farm, ~8c/6s/3g, 2 land, fertilizer trade | `_r37` marginal-price scarcity reorder + sell_lead | no |
| tetsutani market-smart | V45 + **mirror_reorder** hill-climb | same | same | + mirror-symmetric sell A/B | no (name is a misnomer) |
| nathanjacob pipe-7 | V45 + **EXP283 anti-clone race** + symmetric-wheat open | same | same | + anti-clone reservation escalation 8→24 | no |
| evgendvorkin | ≈ V45 base | same | same | same | no |
| pilkwang | Ahmed **V36** tape + **sheep-admission stress-gate** (multi-file bundle) | same | wool-centric | `_r37` scarcity reorder | no (UNVERIFIED, `submission_allowed:false`) |
| **hesoponyo** BC+PPO | **entity-token Transformer** (~340K, fp16, pure-python fwd) | emergent (no table) | emergent | learned; SELL-before-other ordering | **yes** — BC + partial PPO |
| djamila graph-RL | GNN Double-DQN **residual** on a fixed V16 route | fixed route | fixed | ≤25% one-turn sell-DELAY residual | yes (exploratory) |

Micro-edges catalogue (the only real innovations layered on the two originals):

- **Symmetric wheat round-trip open** (`BUY_PRODUCT WHEAT n; SELL WHEAT n`) — a
  cash-neutral, un-front-runnable step-0 order (defends against a rival's
  same-turn round-trip). [V45/nathanjacob]
- **`_r37` marginal-price scarcity reorder** — reconstructs the engine price
  curve and ranks each SELL block by *revenue-now minus revenue-after-a-rival-
  batch* (batch = clamp(rival visible ripe yield, 8..24)). Public info only.
- **`sell_lead`** — sell next step's lots one step early on `step%4≠0` (no town
  consumption between the pair), suppress the duplicate next step.
- **`front_run`** — pull our sells ahead of the opponent's expected sell of a
  fragile premium. (Present but usually *disabled* — no `opponent_plan`.)
- **EXP283 anti-clone race** — detect a rival replaying *our* public tape (tile
  layout ≥0.90 match), pre-empt its sells, escalate the reservation horizon.
- **E182 terminal physical-closure planner** — a bounded last-7-turn search on
  an inlined exact-engine copy that routes every carried unit into the shed
  before final liquidation (SELL draws from shed only).
- **Sheep-admission stress-gate** — a deterministic 2-scenario projection at D12
  that declines the 6-sheep investment if worst-case surplus < 0. [pilkwang]

---

## 3. Where our two lanes actually stand (verified from disk)

**Bandit** — the live/RC path is a **config-driven multi-checkpoint shop
dispatcher over a mutable 719-tape**: dispatch at **D6 (mandatory) + D12/D15/D21/
D27**, keyed on the cumulative unlocked-shop set, splicing nested evolved
branches; a 6-rail guardrail registry (`weed_repair, cash_guard, escalation,
endgame, anti_dump, front_run`). Implemented identically in Python
(`build_bandit_harness.py`) and Rust (`mbandit.rs`); branches from
`multi_checkpoint_evolve.py`. Newest file `agents/v52_bandit.py`; next-ship
compiled pair `.local/candidates/v57_release/` (all gates PASS, **held for
operator approval**). **Live sub 56260608 rated 2336; the open problem is the
D9–D24 mid-game economy ramp**, not the opening or endgame.

**Track P** — per-turn neural planner is **dead** (measured 0.000). The live
seat is an **economy**, in two equivalent forms: the tape `agents/v57_trackp.py`
(EGG/YARN fork + reactive overlay) and the **compiled genome day-planner**
(`.local/candidates/trackp_compiled/`, `policy.rs`, **search OFF** — Phase-B
search measured −14,755 bank, p=0.0064). It ties the v56y tape and beats
pub_v16rc5 on the Sep-15 Linux gauntlet.

**Key comparison to the field:** our bandit's dispatch (5 checkpoints, cumulative
shop keys, nested branches) is **already more sophisticated than any public
agent** (they branch once at D6 + a fixed D27 swap). We are not behind on
architecture. We are behind on **economy**.

**Measured this session (fast Rust `serve_match`, our newest `agents/v52_bandit.py`,
seeds 3-5; results in `.local/scratch/pubmatch/results.tsv`):**

| opponent | s3 | s4 | s5 | verdict |
|---|---|---|---|---|
| Tschinkel CART router | W 90,222/82,570 | W 69,643/62,636 | L 78,232/81,964 | **2-1** |
| flexon V45 (frankenstein) | L 96,283/106,611 | L 109,933/118,980 | L 96,989/108,102 | **0-3** |
| prvsiyan (yhay81 13-tape) | — crashes under serve (tape[step] IndexError, no bounds guard vs the serve harness's extra turn) |

**Follow-up (decisive), our packed economies vs the two originals, seeds 3-5:**

| our economy | vs flexon V45 | vs Tschinkel |
|---|---|---|
| v52 (older bandit) | 0-3 | 2-1 |
| **v56y** (v56 bandit seat) | **0-3** (70k/97k/86k vs 88k/119k/102k) | **0-2-1** (L/draw/L) |
| **v57** (v56 trackp seat) | **0-3 — COLLAPSES to 23k/29k/27k** vs 106k/135k/142k | **0-3** |

**Two findings that reframe the v57 build:**
1. **None of our economies beat the live field** — all 0-3 vs flexon V45; v56y
   also loses to Tschinkel. The gap is real and current, not a memory artefact.
2. **The v57 trackp economy is FRAGILE under contention** — it collapses to
   ~23-29k vs flexon (bankruptcy-grade), far worse than v56y — yet the offline
   gate rated it "strength 100,817 / sweeps public 0.987". **The gate is not
   merely stale, it is actively misleading**: its public set (v16rc5, kaito
   v43/48, rayk c94/c95, moon, munib…) predates the Sep-16 field and never
   included flexon/Tschinkel. This is the "premium collapse under contention"
   failure, measured. **`v57_release`'s "sweeps all public 1.000" is against
   stale opponents and must not be trusted; the v57 trackp seat as currently
   embedded is a liability vs the real frontier.**

Caveat: these are small-N single-seat serve reads; confirm with `measure.evaluate`
(both seats, ≥8 seeds) + `win_metric.paired_test` before acting. But the
direction (0-3, and a 23k collapse) is a signal, not noise.

## 3c. ALL-WORLDS GATE (world generator, 512 games, 0 errors)

Ran the world generator (`world_seeds.py` → `data/worlds/seed_bank.json`, all 64
shop-worlds, 1,500 seeds) and gated v56y (`baseline_v56y.py`) vs the live
frontier across EVERY world, both seats, K=2 seeds/world, live reactive play,
labelled by the REALIZED world (t146). Full table:
`docs/history/v56y-allworlds-gate-2026-09-16.txt`.

| opponent | overall | win-rate | mean margin | worlds lost |
|---|---|---|---|---|
| flexon V45 | **0-0-256** | **0.000** | **-26,883/game** | 54 of 54 |
| Tschinkel | 20-2-234 | 0.082 | -10,329/game | 51 of 53 |

- **v56y loses EVERY game to flexon in EVERY world** (256/256), mean -27k. This
  is not a per-world tuning problem — it is a base-economy deficit everywhere.
  Least-bad worlds still -10k (BAKERY|PET_CAFE, BAKERY|PIZZA); worst -54k..-64k
  (SMOOTHIE|PIZZA, YARN|BAKERY, YARN|PET_CAFE).
- **YARN_STORE worlds are the single worst cluster vs BOTH opponents** (the
  wool/herd-routing gap) → the #1 specialist branch to fix.
- vs Tschinkel the near-even worlds are PET_CAFE/ICE_CREAM/SMOOTHIE/BRUNCH combos
  (a few draws / 0.5) — a stronger base + a YARN fix flips Tschinkel fast; flexon
  needs the full economy rebuild.

This harness (`.local/scratch/pubmatch/world_sweep.py`) is now the v57 acceptance
gate: a candidate must clear the frontier across ALL worlds, not a stale panel.

## 3d. PER-DAY LOSS DIAGNOSIS (v56y vs flexon, worst worlds)

Instrumented the two worst worlds per day (YARN|BAKERY seed 10, SMOOTHIE|PIZZA
seed 43). Root cause is concrete:

- **v56y is HERD-capped at ~12 animals and LAND-capped at 3 quadrants** for the
  entire back half. flexon buys a **4th quadrant** and ramps its **herd to
  ~17-23** at D12-D15. The bank gap opens exactly there (D12 near-even → D30
  -55k). Cash is NOT the blocker (banks even at D12) — v56y's tape stops
  expanding. Economy ceiling, not a resource constraint.
- In PIZZA-shop worlds flexon plants **10 TOMATO** (world demand) + reaches 68
  plants; v56y plants ZERO tomato and caps at 58. World-specific crop capture is
  missing.

**v57 base-economy spec (concrete):** (1) buy the 4th quadrant ~D12-15; (2)
~double the herd to 20-23, placed D9-D18; (3) world-conditional crops (TOMATO in
pizza worlds, etc.) — the per-world specialization; (4) plant count ~65. Build
route: record flexon / a real 2600+ winner on our engine → self-consistent
richer base tape → `build_bandit_harness --base-tape` → re-gate on world_sweep.

## 3b. What this means for v57

- The current pair (v56y bandit + v57 trackp) is the right ARCHITECTURE but a
  losing ECONOMY vs the live field. v57 must be driven by head-to-head vs the
  CURRENT field, not the saturated gate.
- The v57 trackp economy (balanced carrot/strawberry) must be **replaced**, not
  tuned — it is fragile under market contention. v56y is the more robust base.
- Urgent prerequisite: refresh the gate (add flexon V45 as a first-class
  opponent; build the georgymamarin elite panel). Optimising against the stale
  gate is how the fragile v57 economy got blessed in the first place.

---

## 4. The two genuinely-learned agents + the operator's two models

- **hesoponyo Transformer** — the only end-to-end learned *policy* seat in the
  field. BC on a scripted expert + 40 real replays, then *partial* PPO. It
  PASSes 78% of unit-steps (papered over with sticky-target heuristics) and
  claims **no rating**. Its author's own diagnosis: the bottleneck is **BC data
  breadth** — which a large offline corpus fixes directly.
- **djamila GNN** — a ≤25%/one-turn sell-*delay* residual on a fixed route,
  trained on 18 online games, no rating. Overlaps our **already-retired**
  sell-timing lever. Only its 204-node/24-dim graph featurizer is reusable.
- **`vijaikm/kaggriculture-spatial-bc-policy`** (operator-linked) — a BC MLP
  `1706→256→128→35`, pure-NumPy. Same idea as hesoponyo, one head. **Ships
  without its featurizer/action-decoder**, so not runnable as-is.
- **`jek1wantaufik/buddy`** (operator-linked) — a single-farmer greedy heuristic
  (no ML despite the tag). Sub-2000 filler; skip.

**Conclusion on learned policies as a SEAT: no.** Our own hard-won result
(planner/per-turn policy 0.091 vs a tape 0.922; the labour-logistics wall — obs
hides hand positions, multi-unit carry/drop can't be expressed per-turn) is
confirmed by the field: the one learned policy that exists is weak and unrated.
**A learned model earns its place only as a low-dimensional DISPATCHER** (like
Tschinkel's CART trees, or our `multi_checkpoint` selector) — choosing *which
economy* to play, not each turn's action.

---

## 5. The gap to top-10 (diagnosis)

Top-10 cut is ~2900–2970; we are ~2336 live. The field and our own recon agree
on the binding constraint:

1. **Mid-game production ramp (D9–D24).** Live winners ramp cow 8→12 and
   strawberry 16→42 at D9–12; our tape's economy is flatter and stalls in the
   middle third. This is where ~half our live games are lost.
2. **Per-world economy specialization.** Tschinkel fields a *cow/milk* economy
   when milk demand exists and a *sheep/wool* economy when YARN_STORE is up. Our
   branches are closer to one generalist. Fielding the herd the world rewards is
   the single clearest lever the field demonstrates.
3. **Price realization.** Our own forensic note: we dump high volume at ~$1/unit
   while winners restrict supply and capture scarcity at ~$4.4/unit. NB this is
   in tension with Tschinkel's brute-dump-everything approach *offline* — the
   resolution (scarcity-capture matters on the *reactive* ladder, volume+world-
   match matters more than the sell mechanism) must be settled by measurement,
   not assumed.

Everything else (opening, endgame, dispatch machinery, guardrails) is at or
ahead of the field.

---

## 6. Improvements — BANDIT → top-10

B1. **Per-world specialist branch economies (highest value).** Today's D6/D12…
   branches are evolved from one base. Replace the branch *targets* with genuine
   per-world specialists mined from real 2600+ winner replays: a cow/milk
   economy for milk-demand worlds, a sheep/wool economy for YARN worlds, an
   egg/crop economy for no-YARN worlds (the field's no-YARN money is EGG/STRAW,
   not wool). Feed these into `multi_checkpoint_evolve.py` as seed tapes.
B2. **Mid-game ramp.** Rebuild the D9–D24 segment of the base tape to hit the
   winners' cow 8→12 / strawberry 16→42 ramp; the constraint is LAND+cash
   timing, so the ramp must be co-scheduled with the land buys. Validate own-
   bank *and* opponent-bank (share), banded by `op_rating_pre`.
B3. **Terminal physical-closure planner (E182-style).** We liquidate at endgame
   but do not optimize the last-7-turn *carry/drop* routing. Port a bounded,
   physically-dominant-only terminal planner (accept only if it strictly
   improves delivered shed value) — it is a free, risk-gated tail gain.
B4. **Anti-clone race.** The field clones openings (our biggest live loss source
   is our own route mirror). Our `front_run`/`escalation` rails are the right
   place; add EXP283-style mirror detection to arm them against a detected clone
   rather than leaving `front_run` inert.
B5. **Resolve the sell-mechanism question.** A/B `anti_dump` (scarcity-hold) vs
   Tschinkel-style oversized dump on the *reactive* panel + live, banded by
   opponent rating. Decide with a paired sign test, not a dollar threshold.

## 7. Improvements — TRACK P → top-10 (or a second seat)

Track P's compiled genome-planner is a legal, in-budget, self-consistent
*economy executor* that ties our best tape but does not exceed it — because the
executor is land/labour-bound (the reactive genome ceiling ~89k vs ~150k tapes).
Two viable roles:

TP1. **Ship it as a tape-economy seat, not a per-turn planner.** The compiled
   transport is proven (0 fallbacks, worst turn well under budget). Load the
   *specialist* economies from B1 into the genome/tape and use the compiled
   shell purely as a fast, robust executor + reactive sell rails. This is the
   `{bandit, trackp}` compiled pair the v57 release already embodies.
TP2. **Learned DISPATCHER (the safe use of the datasets).** Train a small CART/
   tree (à la Tschinkel) or reuse our `dispatch_oracle` on real winner replays:
   input = public state at D6/D12/D24 (unlocked shops, shop-derived demand,
   market prices/scarcity, opponent census), output = which specialist economy.
   This is low-dimensional, robust, and the one place a learned model beats
   rules — and it is directly trainable from the episode corpora.

Keep search OFF (it is measured net-negative). Do **not** revive the per-turn
policy seat.

## 8. How to use the datasets & models (operator's earlier question)

Ranked by value to the top-10 program:

D1. **`georgymamarin/kaggriculture-episodes` → the winner-tape + opponent-rating
   tap (do this first).** It is a daily, pre-featurized, `episode_id`-filterable
   parquet mirror. Use it to (a) **mine per-world specialist economies** for B1/
   TP2: filter `episode_features.csv` for high-`final_bank` seats with a given
   `plants_*`/herd signature, join `agents⋈teams` for `rating_after ≥ 2600` on
   `engine_version==1.32.7`, pull *only* those `replay_json` rows, and record
   them on our env (recording avoids tape desync); and (b) **build the
   LB-current reactive panel** `elite_panel.py` consumes, so gates predict the
   ladder. This fixes the "offline panel doesn't predict live" problem the whole
   program is stuck behind. Cheaper than the raw 21 GB daily datasets.
D2. **`stream_hashes.csv`** — field-wide opening census (turn 24/48 cuts) to
   quantify opening-clone rate and detect mirror opponents → arms B4.
D3. **`KiroSamurai/kaggriculture-il`** — appears to be *our own* IL store
   mirrored to HF; treat as a portable Colab BC corpus, not new signal.
D4. **Learned models as a DISPATCHER only (TP2), never a seat.** The hesoponyo
   Transformer and the `spatial-bc-policy` MLP both confirm BC-policy-as-seat is
   weak; use the corpora instead to train the economy *selector*. If we ever
   want a learned policy experiment, hesoponyo's embedded `spec.py`
   (featurizer + action space + legality masks) is fully reusable and its stated
   blocker (BC breadth) is exactly what D1's corpus supplies — but this is R&D,
   not the top-10 path.
D5. **`buddy`** — skip. **`spatial-bc-policy`** — study only (missing
   featurizer; per-turn BC not a seat).

## 9. Sequenced plan

Phase 0 (this session, done): downloaded 24 public agents; decoded the 5 named +
14 more; proved the field is one architecture with two originals; staged 7
distinct publics into `data/gauntlet/` + registered in `REACTIVE_OPPS`;
confirmed they play (serve_match).

Phase 1 — panel that predicts the ladder: build the georgymamarin ingest adapter
(D1b) → refresh `elite_panel` from real 2600+/1.32.7 tapes; re-gate v57 pair on
the reactive+elite panels banded by opponent rating.

Phase 2 — economy (the actual lever): mine per-world specialist economies (D1a) →
seed `multi_checkpoint_evolve` (B1) + rebuild the D9–D24 ramp (B2); gate on
Phase-1 panels, own-bank AND share, paired sign test.

Phase 3 — tails: terminal physical-closure planner (B3) + anti-clone arming
(B4); resolve the sell-mechanism A/B (B5).

Phase 4 — dispatcher: train the learned economy selector on real replays (TP2);
compare to our hand-written multi_ckpt keys.

Phase 5 — ship the compiled `{bandit, trackp}` pair (TP1) once Phase-1 panels
show a real, banded, live-correlated gain. Never auto-submit; operator go only.

Gate discipline throughout: score not margin; `win_metric.paired_test`; band by
`op_rating_pre`; verify engine (1.32.7) first; reactive/live over frozen panels.

## 10. This session's concrete changes

- `data/gauntlet/pub_{tschinkel_cart_router, prvsiyan_yhay81_13tape, flexon_v45,
  tetsutani_mirror, nathanjacob_anticlone, hesoponyo_transformer,
  djamila_v16_route}.py` — 7 distinct public agents, decoded & load-tested.
- `src/kaggriculture/measure/band_panel.py` — `REACTIVE_OPPS` extended with the 7.
- Extraction/analysis artefacts under `.local/scratch/pub_notebooks/` (24 agents)
  and `.local/scratch/external_datasets/` (the two operator models).
