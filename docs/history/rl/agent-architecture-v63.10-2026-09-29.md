# Agent architecture: v63.10_rl (live as submission 56661813), 29 Sep 2026

This document describes the agent exactly as it ships. It covers every layer: what it does, when it runs, how
it is kept in sync with the chassis, and what it is good and bad at. The weaknesses are measured, not guessed:
most come from the 29 Sep per-layer activity logs (`KRL_LAYER_LOG`, section 11) and the per-turn RCA of losses.

v63.9_rl (submission 56661730) is the same agent with one change: a knob overlay applied against DIFFERENT /
PARTIAL rivals and no preempt layer. Section 10 covers the differences.

---

## 1. The game in one paragraph (what the agent is solving)

- Two players, 720 steps (30 days × 24 hours), 10×10 farm in four quadrants (NW free, then $1,000 / $2,000 / $4,000).
- Crops: wheat, carrot, tomato, strawberry, melon. Animals: goose → egg, cow → milk, sheep → wool.
- A shared market holds 10,000 units per product. Price is a function of the market stock: a scarcity curve and a
  glut curve, plus a hinge for carrot, tomato and egg.
- Both players sell into the same market in lockstep, unit by unit, in the order of each player's order list.
  So whoever sells first takes the higher price.
- The town drains the market every 4 steps: each shop takes 1 unit per stocked product (2 for single-product shops).
- Score = final bank. Only wins count on the ladder.
- Actions per step:
  - the farmer and each hired hand get one unit action each (move, plant, water, harvest, collect, drop, place,
    feed, care, dig);
  - up to 10 market orders (SELL, BUY_PRODUCT, BUY_SEED, BUY_ANIMAL, BUY_LAND, HIRE).
- The shed holds up to 100 units. SELL draws from the shed only.

The top of the ladder is dominated by **tape agents**: an opening/farm plan recorded from a strong game and
replayed step by step, with reactive layers on top. Our agent is one of them. Our tapes are public (replays),
so rivals copy them. Many ladder games are therefore **races against near-copies of ourselves**, won or lost
on sale timing by one step.

---

## 2. The big picture

```
 Kaggle kernel (Python)                       agent-stdio (Rust, one process for the whole game)
 ┌───────────────────────┐   obs JSON line   ┌─────────────────────────────────────────────────────────────┐
 │ main.py  agent(obs)   │ ────────────────▶ │ Base::act(obs)                                              │
 │  - spawns the binary  │                   │  0. View (parse, projected shed, farm, rival)                │
 │  - 0.25 s/turn budget │ ◀──────────────── │  1. Group controller   (step 25: DIFFERENT / PARTIAL / COPY) │
 │  - shape check        │   action JSON     │  2. PPO policy         (hour 1 of each day: profile 0..35)   │
 │  - fallback.py on any │                   │  3. Chain PRE phases   (outer → inner)                       │
 │    failure            │                   │  4. Endgame model      (step 649: patch endgame knobs)       │
 └───────────────────────┘                   │  5. CHASSIS = routed tape + repairs  (+ terminal planner)    │
                                             │  6. Chain POST phases  (~60 rule stages, inner → outer)      │
                                             │  7. Reactive shell v2  (learned sale quantity + order)       │
                                             │  8. Sales shell big1   (learned hold / part / all override)  │
                                             │  9. Preempt            (lineage fingerprint → sell first)    │
                                             │ 10. Disguise           (zero-qty orders: hide our stream)    │
                                             └─────────────────────────────────────────────────────────────┘
```

**One sentence.** The chassis plays a recorded tape so the farm stays physically consistent. Everything above
it changes **when and how much we sell** (market side), plus a few bounded economy projects. A learned policy
picks, each day, which of 36 market "profiles" (knob sets) the rule layers use.

### Package and runtime

| Item | Value |
|---|---|
| Submission | `submission.tar.gz` = `main.py` + `agent-stdio` (static Linux musl binary) + config files |
| Build | `v63.10_rl bin=d8b044e1b319`, tar sha d70b141b |
| Files in the tar | `base/` (routes + router), `profiles.json` (rl3, 36 profiles), `policy.bin` (PPO i910), `shield.json`, `shell.json` (big1), `rshell/{rshell,net,lineage}.json`, `knobs.json`, `preempt.json` + `lineages.json`, `endg.json`, `fallback.py` |
| Flags | `--chain-off r127,sm,r95 --disguise` |
| Latency | Kaggle validation worst turn 325 ms (first turn includes loading). Typical turn 5–40 ms. Budget 1 s; the bridge allows 0.25 s per turn and 5 s for the first. |
| Failure mode | Binary missing, crashed, timed out or malformed reply → the bridge kills it and plays `fallback.py` (pure Python v9/3 tape agent) for the rest of the game. Validation: 0 fallbacks. |

---

## 3. The bridge: `main.py`

`kaggle/submission/main_template.py`, filled in by `scripts/build_submission.ps1`.

1. The first call finds `agent-stdio` next to `main.py` (or `/kaggle_simulations/agent`) and spawns it once
   with the flags above. Each flag is added only if its file exists in the tarball.
2. Each turn it writes the observation as one JSON line and waits up to `TURN_BUDGET` (0.25 s; 5 s on the
   first turn) for one reply line.
3. `_shape()` is a structural check only: the farmer is a list, hands align positionally with
   `farms[me].hands`, at most 10 market orders. Market entries (including `[]` holes and zero quantities)
   pass through untouched: holes matter for the positional race, and zero quantities are the disguise.
4. Any error → `_kill(reason)` → `fallback.py` for the rest of the game. `STATS` counts turns, fallbacks,
   timeouts and the worst latency, and prints `RUSTV61 start/end` lines (seen in validation logs).

**Strength:** the game never forfeits on an infrastructure error.
**Weakness:** the fallback is a much weaker agent (v9/3). A single timeout late in the game costs the rest of
it, so the Rust turn must stay far below 0.25 s.

---

## 4. Layer 0: View and observation

`crates/agent/src/obs.rs`, `view.rs`. The obs JSON is parsed into `Obs`:
- step, day, hour;
- both farms: money, farmer and hands positions, quadrants, tiles with crop/animal, yield, watering and feeding state;
- our shed, seeds and unit inventories;
- market inventory and prices;
- the town's unlocked shops.

`View` adds derived values: our farm, the rival's farm, price per item, and our shed per item.

**Visibility (important for everything below):**
- We see the rival's tiles (what grows, and its `yield_units`) and its money.
- We do **not** see the rival's shed.
- We do not see the rival's orders. Its sales are recovered each step as `Δ market inventory + town drain − our
  own sales`, which is exact for products we did not also sell that step.

---

## 5. Layer 1: the CHASSIS (the tape player)

`crates/agent/src/chassis.rs` is a line-for-line port of the v61.1 Python chassis (agent lines 404–936).

### 5.1 Routes and the router

- **41 route tapes** (`base/routes.json`). Each tape holds a full 720-step action stream recorded from a
  strong game: every unit action and every market order.
- **Router** (`router.rs`, `base/router.json`):
  - steps 0–143 play **route 0**;
  - at step 144 (day 6, when the first two shops are known) the route is chosen from those two shops.
    YARN_STORE worlds use the old V39 table, other worlds the new EXP240 table, and the V92 table overrides both;
  - from step 648 (day 27) it plays the **endgame route 2**.
- The chassis keeps per-player state: the current route, pending queues and debts. A route switch keeps the
  farm consistent because every route shares the same opening.

### 5.2 What the chassis does each step

`Chassis::act` takes the routed tape's action for this step, then runs its own repair layers:

| Repair | What it does |
|---|---|
| hand alignment | Maps tape hands onto our actual hands (hiring may differ). |
| weed repair | Clears weeds the tape did not expect (unwatered plants turn into weeds). |
| `sell_lead` / `lead_core` | Pulls planned SELLs slightly forward when stock is already there. |
| `front_run` | Sells ahead of the rival where the tape would sell into the rival's batch. |
| R36 debts | Settles sale reservations booked by the R36 layer on their due step. |
| `budget_guard` | Drops purchases we cannot afford (the tape assumed the recorded game's cash). |
| `room_guard` | Prevents shed overflow (cap 100). |
| clamp sells | Never sells more than the projected shed. |
| `dead_stock` | Sells stock the tape will never use. |
| `terminal_liquidation` | Final 9-SELL liquidation from step 712. |

The **projected shed** (`projected_shed`) is the key quantity for every layer after it: the shed after this
step's unit actions (drops, places, pickups).

### 5.3 The terminal closure planner (step 712, `terminal.rs`)

At step 712 the chassis's remaining actions (712–718) are replayed on a **shadow** copy of the state and
simulated with the engine's exact unit semantics. A bounded search over worker harvest, collect and deliver
routes proposes a plan. The plan is used only if it **physically dominates** the baseline. Steps 713–718
follow the plan while the observed farm matches the expected one; otherwise it is abandoned and the chassis resumes.

### 5.4 Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Farm plans recorded from top games: this is where most of our money comes from. | **Open loop.** The tape does not know the rival exists, so its sale steps are fixed and public. |
| Physically consistent by construction (every repair keeps the tape in sync). | **The farm plan is locked early.** Our loss RCA on 465 top-tape losses: 403 are "plan-fixable" (the rival simply produced more milk, tomato, strawberry or wool). The cow→goose decision is made at steps 144–191, before milk shops unlock in 61% of games. The chassis can't change this without breaking tape sync. |
| Deterministic and fast. | Copyable: rivals replay our tapes and race us on the exact same sale steps. |
| The terminal planner adds a little in the last hours (it changed actions in 344 of 465 logged loss games). | The terminal planner only ever starts at 712. Any other start fails the shadow check, so it is off (a documented quirk). |

---

## 6. Layer 2: the RULE CHAIN (~60 stages after the chassis)

`crates/agent/src/layers/` is a line-for-line port of the v61.1 Python wrapper stack. Each Python wrapper is
`pre(obs) → parent(obs) → post(obs, action)`:

- The chain runs the **pre** phases outermost first, then the chassis, then the **post** phases innermost first.
  That is the exact order Python ran them.
- State that an outer layer writes before an inner one reads it (e.g. the R37 horizons read by R36) lives in a
  shared `Ctx`.
- `CUTS` (65 names, `layers/mod.rs`) is the canonical order. Stage *k* post runs only if it is not switched off
  (`knobs.off | game_off`). `--chain-off r127,sm,r95` switches three stages off for the whole game.

### 6.1 The stages, grouped by purpose

The last column is how often each stage changed the action (mean steps per game) over the 465 v63.10 top-tape
losses, from `KRL_LAYER_LOG` on 29 Sep.

| Group | Stages (CUTS order) | What they do | Steps changed / game |
|---|---|---|---|
| **Economy projects** (change the farm, bounded) | v219 (late tomato investment), v231 (sheep→cow swap in milk worlds), v233 (six-sheep SE expansion), r51_input / r51_warehouse (fertilizer tour + warehouse close), r85 / r95 / r97 (feed skip, fertilizer sale, wheat replenishment and supply guard; r95 OFF), courier, carrot, fert, opening, ca (carrot instead of wheat when the carrot book pays), or2 + ch (sales ordered by the rival's stock; harvest animals that would overflow), sr / hd2 / cs (shed room; herd choice at goose purchase; cow→goose swap), y (shop-aware herd), pipe, e402 (late seed cap), e410 (fertilizer guard) | Bounded deviations from the tape when the tape's own plan leaves money on the table. Each keeps the tape in sync by booking its own labour and settling it. | v219 33.9, r51_input 59.9, r85 32.1, v233 25.9, ca 34.5, or2 24.0, cs 2.8, hd2 2.4 |
| **Sale timing / race** | order (APPLY sales-first), r36 (reserve: pull planned SELLs of stock already in the shed forward, booked as debts; RACEGATE leaves near-base prices to the tape), r37 (reservation horizon 2–4 + R44 cash probe + revenue-at-risk ordering), race (clone-gated horizon), v9 race, racepx / racegate, ctrtable, overflow, v44y (lockstep best-response SELL ordering), pg, e335 (sale-slot compaction), adv (ready-stock sale advance), rsa (our route-sale advance, glut guard 0.5× base), afr (anti-front-run, off by default), v92 (library-forecast pre-emption, off in v63.10), cxd (exact best-response ordering), fx / ev / dp / mp / mpx / bd / wb3 (quote-history lead-sells, buy-the-dip, wheat buy-first), tsell (final-day sale advance) | Decide which planned sales go now, how much, and in which slot order, reacting to the rival's recovered sales. | order 91.2, e335 47.8, r36 34.6, r37 22.1, v44y 12.0, cxd 9.6, rsa 5.5, adv 5.4, afr 0.1, tsell 0.1 |
| **Hygiene** | room, v28/v31 guards, mg (merge same-item SELLs), ig (close queue holes), t62a (terminal clear), wl (weed lag) | Keep the action legal and compact. | ig 15.6, mg 6.9 |

### 6.2 Knobs and profiles (how the chain is steered)

- Every tunable constant is a **knob** (`layers/knobs.rs`): race horizons, lead windows, margins, on/off
  switches, the endgame `term_*` and `tsell_*` knobs.
- A **profile** is a set of knob overrides. `profiles.json` (rl3) holds **36 profiles**; profile 0 = v61.1 exactly.
- **Profiles only touch market-side and timing knobs** plus a few labour-safe switches, never planting,
  hiring, land or animals. That is what keeps any profile switch in sync with the tape.
- The profile changes only at a day boundary (hour 1, step 24d+1) or at the hour-13 mid-day decision, and
  holds until the next one.
- `knobs.json` (`--knob-over`) is applied on top of **every** profile: the CMA-ES-tuned constants that were
  tuned together with the reactive shell (lead_frac, lead_from, r36_from, r37_late_from, r37_sim, race_from).

### 6.3 Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Years of ladder-tested rules (the v61.1 lineage reached the top of the ladder). | Rules written against specific past rivals. Many are hard-coded windows and horizons, i.e. predictable. |
| Bit-exact port: every stage verified against the Python agent. | 65 stages interact; a change in one (e.g. v92) shifts others. Ablations are the only safe way to change them. |
| The economy projects add money the tape leaves behind. | The farm-plan projects cannot fix a plan that is wrong for the realised world: cows vs geese, and too few tomatoes (see 5.4). |

---

## 7. Layer 3: the DECISION layers (who picks the profile)

### 7.1 Group controller (`layers/group.rs`) + shield (`shield.json`)

- **Classification** at step 25 (day-1 boundary), from public day-0 signals, **fixed for the game**:
  - **0 DIFFERENT:** our units stood on the rival's squares on < 10% of day-0 turns (a different farm plan);
  - **1 PARTIAL:** same squares but different money at step 1 (a copy of our farm with a different opening trade);
  - **2 COPY:** same squares and the same money at step 1 (a near copy: the pure market race).
- **Shield** (`shield.json`): masks two profiles (9 leads_off, 11 no_reorder) on every day; the league audit
  found both significantly worse. It also has the **D24 mirror guard**: if the rival mirrored us on day 23,
  the day-1..23 choice is pinned for day 24.
- **Jitter:** per-game hash offsets on `rsa_look` and the race horizons, so replays show a distribution, not our next draw.
- Group distribution over the 465 top-tape losses: DIFFERENT 60, PARTIAL 285, COPY 120. Top players are
  mostly classified as PARTIAL or COPY, because the top of the ladder runs copies of the same chassis family.

**Strength:** cheap, early and reliable (81% identification by day 1.5).
**Weakness:** three coarse buckets. A COPY that deviates in the endgame (the 29 Sep strawberry case) is still
"COPY", and nothing in the group tells us *when* it will dump.

### 7.2 PPO policy (`policy.bin`, i910)

- **Input:** the dayobs vector (`crates/dayobs`, the same code as the corpus adapter). Market, farm, money gap,
  shops, the rival's recovered behaviour, group one-hot and sell timing, about 90+ features.
- **Network:** Linear → GRU(64) → heads over the 36 profiles, with a shield mask (profiles the shield forbids get −∞).
- **When:** once per day at hour 1 (step 24d+1), plus hour-13 decisions on the "mid" days. The pick sets
  `chain.requested`, and the chain switches profile at its boundary.
- **Trained:** BC warm start on top-player days, then PPO in a league of Rust opponents (lineage clones,
  random-knob clones, our releases).

**Strength:** adapts the rule knobs to the day's situation, the only per-day learned decision in the stack.
**Weakness:** it chooses among 36 fixed knob sets; it cannot express "sell strawberry one step earlier today".
In the 465 top-tape losses it held profile 35 for most of the game (the logs show profile ids by day).

### 7.3 Endgame model (`endg.rs`, `endg.json`)

- **Decides once, at step 649:** picks one of 24 **proposals** (patches over endgame knobs: terminal planner
  off, final-day sale advance windows, race horizons, AFR, RSA look, lead fraction, lead-sells off, and so on)
  and applies it from step 673 on every later step, on top of whatever profile PPO picks.
- **Model:** two MLPs over 140 inputs (the reactive shell's 35 global + 90 per-item inputs + 6 knobs).
  Utility = P(up) − P(down) vs proposal 0. `rerank` mode keeps an alternative only if it beats base by `gate` = 0.2.
- **Status, 29 Sep (found with the activity log):**
  - **Dead in v63.8–v63.10.** Trained only on copy games, so 62 inputs were constant in training. Against
    any other rival they arrive thousands of σ out, both sigmoids saturate, and every utility = 0.000.
    It picked "base" in 465 of 465 logged games.
  - **Fixed in code:** constant-in-training inputs now enter as 0 and all inputs are clamped to ±8σ. With
    that fix it acts in 62 of 465 games but flips 0 losses.
  - **Low headroom:** relabelling on 5,107 close top-tape games shows the terminal-planner and tsell options
    fix 0 losses (they change results in up to 4,126 games, never a loss into a win). The oracle over all 24
    options fixes at most ~20 of 465.
  - **Verdict:** the endgame is not where the losses are. See §12.

---

## 8. Layer 4: the SALE controllers after the chain

These run in `Base::act` after the chain, in this order. Each may change **only market orders**; they never
change unit actions, so they cannot break tape sync.

### 8.1 Reactive shell v2 (`rshell.rs`, `rshell/rshell.json` + `net.json` + `lineage.json`)

- One learned, config-driven sale controller: per product and turn it decides **how much of the projected shed
  to sell now** (fraction 0, ¼, ½, ¾, all) and **in which slot order**.
- **Inputs:** 35 global (time, money gap, cash flows, opponent identity: group, cluster, clone gate, position
  streaks, layout similarity, step-1 mirror, lineage match against our own route library; race state; today's
  profile; shed room) + 52 per item (stock, market, town demand, the rival's recovered sales and visible
  production, three rival-sale forecasts, our plan, the chain's order and debts, and the **price calculator**).
- **Price calculator:** the exact engine price curve with the inventory simulated `horizon` steps ahead (town
  drain + a mix of rival forecasts). It gives the revenue of each fraction sold now plus the rest at the best
  later step.
- **Score per class:** `model_w·net + bias + stay·[chain's class] + beta_price·price_advantage + urge·(2·frac − 1)`,
  where `urge = Σ urge_w · signals + item_urge[item]`. It acts only if the best class ≠ the chain's class and
  p ≥ τ + group_τ, the direction is allowed, the glut floor holds and the margin gate passes.
- **Hard limits:** SELL quantities and order only; clamp to the projected shed; fertilizer never below committed
  need; the 10-order cap; active only in the window `from..to` (71..692 in v63.10).
- **Tuned:** CMA-ES over 54 dimensions against the v63.8-era agent ("mean_rshell"). v63.10 ships the v63.8 values.

| Strengths | Weaknesses (measured 29 Sep) |
|---|---|
| The only layer with a real price model: it knows glut vs premium and waits out a glut. | **Rarely acts:** it changes the action on ~3 steps per game (2.8 mean over 465 logged games; dead in 116 of them). |
| Hard limits make it safe. | **Mistuned for copies:** v63.10's per-item urges (milk +0.74, tomato +1.27, carrot +1.64, wheat −2.93) make it hold strawberry and wheat too long against chassis copies. Swapping in the older urge set fixes 84 bad-seed losses and breaks 0; the per-setting ablation isolates the urges and the decision weights. |
| | The same older urges are slightly worse against top players (+4/−9 at 5,016 top tapes), so the fix must be per group. A `group_urge` override was built on 29 Sep; gating is in progress. |

### 8.2 Sales shell big1 (`shell.rs`, `shell.json`)

- A small learned classifier per product and turn: hold / sell half / sell all. It overrides the chain's SELL
  quantity for an item when it is at least `tau` confident and disagrees.
- Trained on `branch` labels: exact counterfactual sell choices.
- **Strength:** cheap and targeted.
- **Weakness:** changes the action on ~3.4 steps per game and is dead in 345 of 465 logged top-tape losses.
  Almost inert against top players.

### 8.3 Preempt: lineage fingerprint (`preempt.rs`, `preempt.json` + `lineages.json`)

- **Idea (operator, 28 Sep):** rivals fingerprint our lineage and sell a small batch just before our known
  sale. We fingerprint them the same way and sell first.
- **Offline:** 421,539 player-games from the GM corpus are clustered into **150 lineages** by an exact hash of
  their day 0–5 action stream (consistency ≥ 0.5, ≥ 15 games) plus a background class, covering 67%.
  Each lineage stores, per product and step (0–719), the probability that it sells (`p_step`) and its typical
  lot (`lot`).
- **Online:**
  - From step 120 (`fp_from`) the rival's recovered sales update a log-likelihood per lineage.
  - `matched(step)` = step ≥ 144 and lineage weight ≥ 0.8 (`gate_w`).
  - When matched and the rival is forecast to sell item X within `look` = 1 step with p ≥ `p_min` = 0.3, and
    we hold X with a sale of X planned by our tape within `horizon` steps, we sell that planned amount **now,
    at the front of the queue**, above a price floor.
- Items: strawberry, milk, wool, melon, carrot, tomato, egg (not wheat: wheat never gluts).
- Runs last before the disguise, so no later layer undoes it.

| Strengths | Weaknesses (measured 29 Sep) |
|---|---|
| The correct idea: know when the rival dumps and sell one step earlier. | **Off in the endgame:** window `to` defaults to 647, and `preempt.json` does not override it. Days 27–29, where the late races happen, are never covered. |
| Matches often: 406 of 465 logged top-tape losses had a lineage match. | **The forecast is clock-based:** "probability lineage L sells X at step t", averaged over thousands of games. Endgame dumps depend on the rival's **state** (it holds stock and the price is high), not the clock. Traced case: forecast 8% for a 17-unit strawberry dump that happened the next step. |
| | **Barely fires:** 0.7 steps per game on average; dead in 284 of 465. The dominant skip reasons (`KRL_PRE_DBG`): price floor, nothing left, below p_min. |
| | **Fix in design:** a state-based forecast that rebuilds the rival's stock from its visible harvests minus its recovered sales, and opens the window to 711. |

### 8.4 Disguise (`disguise.rs`)

- Appends **zero-quantity SELL orders** of products we hold none of, at the end of the market list, on a
  per-game pseudo-random pattern (secret salt + opening state).
- Zero units sell nothing: the state and banks are unchanged (verified identical). But our action stream (and
  any hash, prefix or exact-match lineage built on it) differs from game to game.
- Active on ~227 steps per game.

**Strength:** defeats exact-stream fingerprinting of us at zero cost.
**Weakness:** a rival that fingerprints on the *effective* sales (market inventory deltas), as our own preempt
does, is not fooled. The disguise hides the stream, not the behaviour.

---

## 9. How the agent keeps the chassis and the layers in sync

This is the central design constraint: the tape assumes a specific farm state at every step, and a single
missed WATER or wrong PLACE turns into weeds, dead animals or a desynced route.

1. **Ownership split.**
   - Unit actions (farmer and hands) belong to the chassis and the economy projects.
   - Market orders belong to the market layers.
   - Every layer after the chain (rshell, shell, preempt, disguise) may touch **only market orders**, and only
     SELL quantity and order.
   - A profile can only move market and timing knobs.
2. **Projects that change the farm book their own labour and state.** V219, V231, V233, R51, CA and CS each
   carry an explicit "committed" state. The chassis's `budget_guard` / `room_guard` and the R36 debt ledger
   settle the consequences on the due steps. The terminal planner runs on a shadow copy and abandons its plan
   the moment the observed farm deviates.
3. **Projected shed.** Every sale layer sizes its SELLs against the shed after this step's unit actions, so a
   sale never outruns the drops and pickups the tape is doing.
4. **Hard caps everywhere:** 10 orders (extras dropped), clamp to the projected shed, fertilizer need floor,
   `_shape()` in the bridge.
5. **Decision boundaries.**
   - The group is fixed at step 25.
   - The profile changes at hour 1 (and hour 13 on mid days).
   - The endgame proposal is chosen once at 649.
   - Between those points everything is deterministic given the observation.
6. **Route switches** happen only at step 144 (route chosen from the first two shops) and step 648 (endgame
   route). All routes share the opening, so the switch is safe.
7. **Verified by construction.** With every learned layer off, the agent is byte-identical to the v61.1 port.
   Each layer was added with an "off = identical" test. The 29 Sep activity log (§11) is also verified to not
   change a single game (465/465 identical banks).

---

## 10. v63.10 vs v63.9 (the two live submissions)

| | v63.9_rl (56661730) | v63.10_rl (56661813) |
|---|---|---|
| Base | v63.8 stack (chassis, chain, PPO i910, rshell v2, big1, endgame v1, disguise) | same |
| Extra | **P105 group knob overlay** against DIFFERENT and PARTIAL rivals (v92 on, carrot margin, lead windows) | **Preempt (lineage fingerprint)** |
| Top tapes (25,415) | 511 losses | **465 losses** |
| 79-agent gate | — | 0.942 (553 L of 10,112) |
| Known issue | v92 in the overlay costs top-tape games | The dead layers in §8 |

---

## 11. Diagnostics (29 Sep): how to see what every layer does

All are off by default; set the env variable on any runner (tapeplay, selfplay, agent-stdio). Built into the
`target-lay` binaries.

| Env | Output |
|---|---|
| `KRL_LAYER_LOG=<file>` | One JSON line per game: installed layers, `dead` (installed, never changed an action), `hits{layer: [steps, first, last]}` including each chain stage, group, lineage {matched, weight, fired}, endgame {pick, name, u_best_alt, u_exact_zero, gate}, terminal {planned, skip_shadow, skip_baseline, no_plan, changed_steps}, group-overlay hits, profile per day. |
| `KRL_LAYER_TRACE=<file>` | Per step: which layers changed the action, with the market diff (before → after). |
| `KRL_PRE_DBG=<file>` | Preempt per (step, item): forecast p and the gate that stopped it (`below_p_min`, `price_floor`, `no_route`, `nothing_left`, `tape_plans_none`) or `FIRE`. |
| `KRL_ENDG_LOG=<file>` | Endgame model inputs, utilities, pick. |
| `KRL_TERM_LOG=<file>` | Terminal planner plans and skips. |

**Rule from 29 Sep:** a layer whose on/off comparison is identical must be checked with these logs before it is
called "neutral". The endgame model was called neutral for three releases while it was dead.

---

## 12. Where the agent loses (29 Sep) and what that says about the architecture

| Where | Evidence | Layer responsible | Fixable in the architecture? |
|---|---|---|---|
| **Late top-tape losses** (465 of 25,415 = 1.8%) | 403 plan-fixable (volume: milk, tomato, strawberry, wool); 37 sale-fixable; 25 out of reach | Chassis farm plan (cow/goose at 144–191, tomato count) | Hard: the farm plan is the tape. Every tested change breaks ~as many wins as it fixes. |
| **Public agents on bad seeds** (smoothie+pizza, pizza+bakery, yarn+yarn) | 193 of 280 lost; the urge swap fixes 84 / breaks 0 | Reactive shell urges (§8.1) | Yes: per-group urges (built; being gated). |
| **Endgame races vs copies** (days 26–28) | Traced: the rival dumps 17 strawberries at step 683, and the winner sells at 682 | Preempt (window + clock forecast), rshell | Yes: a state-based rival stock forecast plus the window to 711 (designed). |
| **v63.7 beats v63.10 88–40 head to head** | Same shell-urge mechanism in copy races | rshell | Same fix as above. |
| **Endgame knobs** | Oracle ≤ 20 of 465; the tested options fix 0 | Endgame model | Little to gain. The model is now alive but should not be expected to move results. |

### The architecture in one line of judgement

It is strong where a tape is strong (a proven farm plan, physical consistency, speed and safety) and weak
where a tape is weak: it is open loop against a rival who copies it. The fixes that work are the ones that give
the market layers a better model of **the rival's state**, not more rules on the clock.
