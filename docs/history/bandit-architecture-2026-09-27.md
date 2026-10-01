# Bandit architecture: chassis and reactive shell (2026-09-27)

A complete description of the Rust bandit that ships as v63 / v63.1 (`rustengine/v62/crates/agent`), detailed
enough to re-implement, followed by a measured review of where it can improve. Every claim about what fires, how
often and what it is worth comes from a measurement named in the text; hypotheses are marked as such.

Code map: `base.rs` (one turn), `router.rs` + `trie.rs` (tape choice), `chassis.rs` (tape replay and repairs),
`terminal.rs` (end-game planner), `layers/mod.rs` (the chain), `layers/*.rs` (the reactive layers),
`layers/knobs.rs` (knobs and profiles), `layers/group.rs` + `layers/dispatch.rs` (controllers), `disguise.rs`.
Config: `configs/bandit/bases/v61.1/{routes.json, router.json}` (41 routes), `configs/bandit/profiles/v4.json`
(101 profiles, 14 dispatch rule sets).

---

## 1. The idea

The agent is two things:

1. **A tape library** - recorded 720-step action lists (farmer, hands, market orders) from strong games. The tape
   carries the STRATEGY: what to plant, when to hire, buy land, buy animals.
2. **A reactive shell** - about 65 ordered layers, each `Action -> Action`, applied every turn. The shell carries
   the TACTICS: repairing the tape against the real world (weeds, missing stock, a different rival) and winning
   the sell race.

The rule that shapes everything: **the shell may bend the tape but never break sync**. Later tape steps assume
earlier ones happened (a pickup expects stock in the shed, a water expects a plant). Most layers therefore touch
only the market list (when and how much we sell); the few that change unit actions are bounded projects that hand
the units back in a state the tape expects.

## 2. One turn

```
observation --> View (positions, tiles, shed, prices, market inventory, shops, rival farm)
   |
   +--> Chain.pre      outermost -> innermost: trackers, day-boundary profile switch, horizons into Ctx
   |
   +--> Terminal planner (active from step 712) wrapping
   |       Chassis.act: router -> tape[step] -> chassis repair layers
   |       + step >= 718 shop rescue
   |
   +--> Chain.post     innermost -> outermost: ~65 stages, each Action -> Action
   |
   +--> Disguise (stream; money) -> market truncated to 10 orders
```

Implementation rules:

- Each layer = a pure function of (action, view, chassis) plus per-player state keyed by `obs.player`. A new game
  is detected by the step going backwards and resets that state.
- Pre phases run outermost first, post phases innermost first (the Python decorator order the chain was ported
  from). Values an outer pre phase hands to an inner layer live in `Ctx`: `r37_horizon`, `v9_item_hz`,
  `race_horizon`.
- If the chassis layers raise, the raw tape action is played (`layer_fallbacks` counts it).
- The engine silently ignores illegal commands and silently drops market orders beyond the 10th. A broken layer
  looks like laziness, not a crash; order in the market list is priority.
- `router.json` `"cut"` runs the chain only through a named stage (`"chassis"` = tape only); `CUTS` in
  `layers/mod.rs` names all 66 stages.

## 3. Tape choice (router)

**Table router (v61.1):**

| step | route |
|---|---|
| 0-143 | route 0, the shared opening. Every route has the same farm actions on steps 1-143, which is what makes a later switch legal |
| 144 (day 6) | first two unlocked shops `A|B` -> route id. Worlds with a YARN_STORE use the old (V39) table, the rest the new (EXP240) table; the `shop_routes_v92` table and a D6 dispatch route (section 8) override |
| 648 (day 27) | route 2, the shared end-game tape |

The step-0 market of every route is replaced by `opening_step0_market`. At step 2 the router stores a rival
fingerprint `(round(rival money, 3), market WHEAT)` for later layers.

**Trie router (team bases, `"mode": "trie"`):** for libraries without a shared opening. Tapes are grouped by an
FNV hash of the full action history played so far (units and market). At a branch, candidates are scored by the
distance between the current situation (16 features) and the situation recorded on the tape, plus a world
penalty; ties are broken by support (`tie_eps`, `base_support`). Only tapes whose whole history equals ours can
be switched to, so a switch never desyncs.

The world (first two shops) is realized by play, not by seed: route on OBSERVED shops only.

## 4. Chassis repair layers (inside `Chassis.act`, in order)

| layer | what it does |
|---|---|
| `hand_align` | keeps the tape's hand commands aligned with the hands that exist (a hire can fail) |
| `weed_repair` | a PLANT/BUILD aimed at a tile that is now a WEED becomes DIG and the original is queued for that unit; later, when the unit's tape command would be a no-op on its tile, the queued command is replayed instead. A queued PLANT is dropped if the next tape command is a move (the WATER could never follow; the seed is kept). An idle unit on a weed DIGs |
| suppression + R36 debts | a SELL pulled forward earlier is removed on its original step; the debt ledger `{due_step: {item: qty}}` is settled |
| `sell_lead` | sells next step's planned SELLs now when the stock is in the projected shed, and books the suppression. RACEPX: while any product is glutted (quote <= base + `racepx_margin`), lead-sell only non-glutted products. R36 window: native lead only outside `r36_lead_from..to` |
| `front_run` | same, against a known opponent plan (unused on the ladder) |
| `budget_guard` | drops buys the cash cannot cover |
| `room_guard` | keeps the shed under capacity |
| `clamp_sells` | trims SELLs to the projected shed (never cancels one: SELL fills partially) |
| `dead_stock` | sells stock the tape will never use |
| `terminal_liquidation` | sells everything at the end |

`projected_shed(action)` - the shed after this turn's unit actions (DROP, PLACE and PICKUP next to the shed),
before the market - is the stock number every sell decision uses.

## 5. The market model the shell reasons with

- Price falls as market inventory rises: the first seller of a product gets the better price.
- Town demand drains inventory on a fixed schedule: every 4 steps each unlocked shop takes 1 of each of its items
  (2 for a single-item shop); every day start takes 1 of every item.
- Hence the rival's exact sales are recoverable each turn:
  `rival_sold[X] = delta_market_inventory[X] + town_draw[X] - our_sold[X]`
  (RACE, AFR and v92 all use this, from a per-turn snapshot of inventory, prices, shops and shed).
- Premium products (MILK, WOOL, STRAWBERRY, EGG, ...) are where races happen. A product quoted at or below base
  (+ margin) is glutted and not worth racing: RACEPX and RACEGATE skip it.

## 6. Opponent sensing

| signal | definition | read by |
|---|---|---|
| layout similarity | share of occupied tiles with the same (crop, animal) on both farms | R37, RACE, clone gates |
| clone gate | >= 4 of the last 6 turns with our farmer and hands on exactly the rival's squares, and similarity >= 0.95 | RACE, clone-profile controller |
| strict clone day | >= 20 of the last 24 steps on the rival's squares and similarity >= 0.95 | `--strict` controller |
| step-1 cash mirror | |rival money - ours| < 0.5 at step 1 (same opening trade) | RACE (level -> `race_mirror`) |
| group (fixed at step 25) | DIFFERENT: on the rival's squares on < 10% of day-0 turns; PARTIAL: same squares, different step-1 money; COPY: same squares and money | group controller, dispatch |
| cluster (step 144) | group + M (still on our squares on >= 90% of day 5's turns) or S (split by day 5) | D6 dispatch |
| race lost | the rival sold a race product last turn well ahead of our tape's own sale of it | RACE escalation |
| pre-emption | the rival sold X while we held X with a route sale of X still ahead | AFR |
| R44 cash probe | the rival's cash responds like ours | R37 horizon |

Identification speed (324 real games): 70% correct at turn 1, 77% at turn 24, 81% by day 1.5. Copies are the
threat: every one of v63's analysed losses was to a copy of our public replays, where the game is a pure race to
sell first.

## 7. The post chain, by job

Stage numbers are indices into `CUTS`. "Fires" = mean steps per game on which the stage changed the action, and the
number of games (of 40) where it fired at least once, for v63.1 (P100) vs v63 (a copy race) - see section 10.1.

### A. Safety and hygiene

| stage | what | fires |
|---|---|---|
| 2 room | hour 23: if shed + carried cargo > 99, sell unplanned stock, highest price first | 0.7 / 20 |
| 6, 9 sales_first | from step 144 (and inside R36 from 288): drop empty orders; move each SELL ahead of earlier non-SELL orders, stopping at a SELL or a BUY of the same item (cash before spend) | 94.0 / 40 (`order`) |
| 20 r97 | the wheat the next tape step picks up must be in the shed (buy it) | 2.1 / 37 |
| 38 r127 | drop last-hour plantings that would end unwatered (a weed in two days); fund urgent grain in slot 0 | 0.8 / 32 |
| 39 pg | hours 21-22 pre-guard | never |
| 42 e335 | sale-slot compaction | 50.9 / 40 |
| 47 t62a | terminal clear | 0.4 / 16 |
| 56 sm | shield milk | 0.1 / 5 |
| 58 e410 | never sell fertilizer a live fertilizer-tour worker needs | 3.0 / 34 |
| 59 e402 | late seed cap | 10.3 / 28 |
| 60 mg / 61 ig | merge same-item orders; close queue holes | 8.1 / 40, 18.6 / 40 |

### B. Economy projects (the only layers that change unit actions)

Each is a bounded project: eligibility test, worker plan, rescue path, hand-back.

| stage | what | fires |
|---|---|---|
| 4 v219 | late tomato investment: when the followed route has no late land buy or tomato planting, request extra tomato planting with fertilizer; labour by permutation search (`r53_labor`); fertilizer only if worthwhile (`r79`); skips days 19/21/23 | 26.1 / 6 |
| 8 v231 | swap a scheduled 1-2 SHEEP buy for COW in milk worlds; carry the cows to the sheep's slots; add the extra milk to an existing MILK sale | 0.5 / 4 |
| 12 v233 | six-sheep south-east expansion with a compact service path (shortest permutation over six tiles), a one-hand service day, a day-29 wind-down, a wheat rescue | 19.4 / 2 |
| 14 r51_input | finite fertilizer tour: forecast harvests, beam-searched delivery walk (`r68` joint plans) | 72.6 / 40 |
| 15 r51_warehouse | hour 23, days 12-28: project the last hour on the engine, sell what would not fit at midnight | 0.4 / 17 |
| 18 r85 | skip a FEED whose care bonus is worth less than the wheat; sell fertilizer beyond every scheduled and committed need | 42.4 / 40 |
| 19 r95 | days 10-11: trim wheat buys to a two-day reserve | 0.7 / 29 |
| 21 courier / 22 carrot / 24 fert / 25 opening | v9 courier routes, carrot handling (never fired), fertilizer, opening tweaks | 6.0 / 13.8 fire; carrot never |
| 31 ca | plant carrot instead of wheat when the carrot pays: simulate the tape's future visits to the tile, yield paths of both crops (decay, water, fertilizer), swap if carrot wins by `ca_margin`; rescue swapped carrots; trim the wheat seed | 45.7 / 29 |
| 33 ch | harvest animals whose product would overflow tonight | 8.1 / 30 |
| 34 sr / 35 hd2 / 36 cs | shed room; at the tape's goose purchase pick the herd by expected value (cost, first yield, interval, yield, price over a 30-day schedule); cow -> goose swap at the first cow buy | 1.9 / 22, 2.1 / 4, 2.3 / 5 |
| 41 y | shop-aware herd, days 8-11 | never |
| 48 pipe | install the EarlyCycle opening into route 0 | 1.0 / 40 |
| 65 hfeed | steps 192-695: when the farmer's tape picks up wheat next to the shed, count unfed animals it will FEED today and pick up the shortfall (<= 2, keep 2 in the shed) | 0.8 / 34 |

### C. Sell timing - the race

| stage | what | fires |
|---|---|---|
| 9 r36 | steps 192-695: pull the tape's upcoming SELLs of stock already in the shed forward to now, up to horizon H, booked as debts; RACEGATE leaves glutted products to the tape | 34.4 / 40 |
| 10 r37 | sets H: 2; 3 after a 6-step similarity streak (336-647); 4 when the cash probe matched or in 288-695. Reorders each run of distinct SELLs by quote priority (revenue a small rival batch would take off the item) | 20.8 / 40 |
| 26 v9 race + pre | per-item horizon clamped by `v9_race_default/max/margin` | via r36 |
| 37 race (pre) | clone gate on (216-695): H = `race_clone` (9); `race_escalated` (24) after a lost race; `race_mirror` (24) after a step-1 cash mirror | via r36 |
| 32 or2 | order sales by the rival's estimated dumpable stock (its visible tiles: kind, birth, yield -> `exposure`); swap when better by `or2_slot_margin` | 25.8 / 40 |
| 40 v44y | from step 216: for each item simulate a lockstep race (both queues sell one unit at a time against one inventory and price curve), try permutations of our SELL slots, keep the best margin | 11.2 / 40 |
| 57 cxd | exact best-response ordering against rival models: `cxd_model` 0 = the rival copies our orders, 1 = v92 forecast, 2 = worst case of both | 8.8 / 40 |
| 25 v92 | forecast the rival's premium sales (section 7.1) and sell our lots just before | inside `opening` |
| 46 adv | steps 216-717, not hour 23: look `adv_look` steps ahead in the tape (net of R36 debts), skip the item the tape sells first, no BUY_PRODUCT queued -> sell planned premium stock from the projected shed now, highest price first | 4.1 / 34 |
| 62 rsa | the same over `rsa_look` steps, with a glut guard (no advance below `rsa_min_frac` x base) | 5.5 / 39 |
| 51 fx / 52 dp / 53 mp / 55 mpx | quote history; in hours 15-20 (EV), 0-2 (DP), 10-13 (MP), lead-sell up to 3/4 of the next `h` turns' planned sells of an item quoted at or above its trailing average; MPX a model-based lead | 1.2 / 7.3 / 4.2 / 2.9 |
| 54 bd / 50 wb3 | buy the dip on inputs; buy wheat first | 6.0 / 40, 1.7 / 2 |
| 30 overflow | hour 23: sell exactly what midnight cargo would push out of the shed | 0.5 / 9 |
| 63 afr | anti-front-run on detected pre-emption (off in every shipped profile) | never |
| 64 tsell | from `tsell_from` (672): shift remaining sales within `tsell_window`, keep the best | 0.1 / 2 |

#### 7.1 v92 PREDICT
- Library: recorded top-player sale streams per shop pair.
- Each turn: recover the rival's executed sales (section 5), mark them in per-product bitsets.
- Every `v92_every` turns from step 150: score each stream of the pair on the last 240 steps (matches within +-1
  step, misses, false alarms); stable sort; keep the top `v92_top`.
- If the best stream sells >= `v92_k` units of MILK/WOOL/STRAWBERRY in the next 2 turns (or the next W turns
  with the extension, `v92_ext_window`, when the stream matched recently), sell our planned lots of it within
  `v92_h` steps now.
- Measured worth: removing it from ca25 moved v62.1's score vs ca25 from 0.521 to 0.729 (paired +10/-0).
- Fast path: per stream, sorted keys plus per-product bitsets of steps within 1 of a recorded sale; bit-exact with
  the slow path (`KAGG_V92_SLOW=1`).

### D. End game

- **Terminal planner** (`terminal.rs`, from `term_start` 712): save players' state, replay the chassis's own
  remaining actions on a shadow (baseline), search worker harvest/collect/deliver routes (`term_props` proposals
  per actor, `term_sims` simulations, `term_passes` passes); accept only a plan that physically dominates the
  baseline (at least as much of every product); follow it on 713-718 while the observed farm matches the expected
  one; abstain when a V219/V233 project is committed. v63 runs 512 sims, 2 passes, 8 proposals.
- **Shop rescue** (step >= 718): units next to the shed with stock DROP, the rest PASS, the market sells the whole
  projected shed, largest value first. Anything still carried at the end scores zero.

## 8. Knobs, profiles, controllers

`Knobs::default()` is v61.1. A profile is a JSON object of overrides; unknown names are load errors.

- A game starts on profile 0; profiles change only at hour 1 of a day (step 24d+1) and hold for the day.
- At a boundary, the first that answers wins: explicit request -> AFR escalation (>= `afr_trigger` pre-emptions
  yesterday) -> clone-day profile -> group controller -> fixed schedule -> dispatch.
- Profiles change market and timing knobs and a few labour-safe switches only - never planting, hiring, land or
  animals - so a switch cannot desync the tape.
- **Group controller**: group fixed at step 25; per-group profile; optional secret per-day jitter (+-J on
  `rsa_look` and the race horizons) drawn from a hash of a build salt, the game's opening state and the day;
  optional day-24 change of course with a mirror guard.
- **D6 dispatch**: at step 144, (shop pair, cluster) -> first matching rule: base route (only if its farm
  actions on 1-143 equal route 0's), end-game tape (installed as route 2 for this game), knob overrides from
  `from_day`. Rules are config; v63.1 uses set 13 (`v92_top 1 / v92_k 4` in five realized pairs).

Shipped profiles:

| agent | profile | knobs over v61.1 |
|---|---|---|
| v63 | 71 | ev/dp/mp_h 24, v9_race 60/72, rsa_look 12, v92 on (ext 4), terminal 512/2/8, tsell on (model 2, min 50) |
| v63.1 | 100 | v63 + hfeed, v92_top 3, v92_k 3, dispatch set 13, disguise_stream |

## 9. Disguise (last)

- **Stream**: zero-quantity SELL orders of products we hold none of, appended at the end of the market list
  (never moving a real order, never past 10), per-game pseudo-random. Banks unchanged; byte-level lineage
  matching of our replays breaks.
- **Money** (`disguise_money`): sell N wheat at step 1. Measured harmful (lost 1000/1000 against its own base): it
  blinds our step-1 cash-mirror detection. Keep off.

---

## 10. Where it can improve

### 10.1 Measurements taken for this review (2026-09-27)

- **Stage coverage**: `selfplay` now writes, with `KAGG_FIRED_DUMP=<file>`, one line per game and seat with the
  number of steps on which each stage changed the action. 40 seeds, v63.1 (P100) vs v63 (P71): 31-8-1; v63.1 vs
  the DECEM team bandit: 40-0. The two runs fire almost identically (table in section 7).
- **CMA-ES `cma1`** (50 generations, lambda 12, 6 seeds, fitness vs a closed-loop copy population; held-out 330
  games on unseen seeds vs P100 at 0.842):

| gen | held-out | better/worse | main moves |
|---|---|---|---|
| 10 | 0.870 | +52/-35 | |
| 20 | **0.982** | +80/-6 | v92_k 2, ext 6, v92_h 47, rsa_look 13, rsa_min_frac 0.37, ev/dp/mp_h 25/28/41, v9_race 55/93 margin 4, racepx -4, ca_margin -30, or2 21, adv_look 4, tsell_window 16, tsell_min 33 |
| 30 | 0.952 | +74/-12 | |
| 40 | 0.952 | +70/-12 | |
| 50 | 0.952 | +70/-14 | |

  Consistent across all checks: `v92_k` 3 -> 2, `adv_look` 3 -> 4, `v9_race_margin` 12 -> 1-4, `v9_race_max`
  up, `or2_slot_margin` 12 -> 15-22, `tsell_window` 24 -> 12-16, `mp_h` up. Not yet validated on the public field.

### 10.2 Opportunities, ranked by expected value

1. **Knob tuning is the largest measured lever - finish the validation.** The gen-20 vector wins +80/-6 against
   v63.1 on unseen copy races. The risk is that aggressive early selling costs price against non-copies
   (ca_margin -25 hurt earlier on the public field). Next: 444 real replays + the 64-world tournament vs the
   public field. If it regresses there, tune with a MIXED fitness (copies + team bandits + public tapes) instead
   of copies only.

2. **Tune per cluster / per world, not one vector.** The chassis already switches knobs per (shop pair, cluster)
   through dispatch. One global vector compromises between copies (want early, aggressive selling) and
   different-plan rivals (want price). A CMA run per group (COPY vs DIFFERENT) with the dispatch layer applying
   the result is the natural next step; it needs per-group seeds and held-out checks.

3. **Dead or near-dead layers.** In 80 games these never changed an action: `pg`, `y`, `carrot`, `ctrtable`
   (PIPE turns it off), `e343_wl`, `afr` (off), plus the no-op wrappers (`v28`, `v31`, `release`, `r46`, `r53`,
   `r70`, `herd`, `racepx`, `racegate`, `v11`, `v13v`, `ma`, `race2` - their logic lives elsewhere). `wb3`, `sm`,
   `tsell` fire in <= 5 games. Action: confirm on a world-stratified run (the Y herd layer is gated to days 8-11
   in yarn worlds, so 40 random seeds may just have missed it), then remove true dead code for latency and
   clarity, or fix a gate that never opens.

4. **The economy projects rarely engage.** v219 fires in 6/40 games, v233 2/40, v231 4/40, hd2 4/40, cs 5/40.
   The strategy is effectively the 41-route library. Evidence that the economy matters: crop-heavy top players
   win 0.75 vs 0.60 for the meta; in 113764426 we out-sold Lam Dang and lost by out-spending them by $12.7k.
   Hypotheses to test: a spend/ROI guard on late buys; loosening v219/v233 eligibility; per-world route
   refresh. Tape transplants and team bandits did NOT beat v63 (league 0.988 for v63), so new routes must be
   tested closed-loop, not open-loop.

5. **Race decisions are rule-based, while copy losses are coin flips.** Median loss margin vs copies is $24-29;
   a single "hold" decision would have flipped most losses in hindsight, but every fixed hold rule was
   catastrophic (0.75 -> 0.008). The race needs state-conditioned choices; the value model is not ready for search
   before day 15 (held-out-team AUC 0.64 day 3, 0.737 day 10, 0.847 day 15, 0.905 day 20). Option: restrict search
   to days 15+, or learn per-item sell/hold from closed-loop self-play instead of from the ladder.

6. **Rival model breadth.** `cxd_model` is 0 (copy assumption) in every shipped profile; CMA did not tune it. v92
   uses a stream library per shop pair built from older games; CMA drove `v92_k` down and `ext` up, i.e. it wants
   the forecaster to act more often. Refresh the stream library with this week's top-50 tapes and test
   `cxd_model` 1 and 2.

7. **Race coverage before step 216.** RACE, V44Y and ADV start at 216 (day 9); R36 at 192. Early premium sales
   (days 6-8) are left to the tape and to the lead-sell windows. Hypothesis only: check in copy races whether
   early-day sales are lost to the rival.

8. **Identification.** Group is fixed at step 25 from day-0 signals (81% correct by day 1.5); the cluster is fixed
   at 144. A misclassified rival keeps the wrong profile for the whole game. A day-boundary re-check (e.g. at
   days 6 and 12) with the strict clone gate would bound the damage.

9. **Latency and speed.** Worst turn 122 ms for v63.1 (house cap ~150 ms). The terminal planner ran 12,624
   simulations over 64 plans in 40 games (~12 ms each plan). Heaviest shell stages: or2, r51_input, ca, cxd, ch,
   fx, e410, r36, fert, v44y. Removing dead stages (item 3) and caching `projected_shed` per turn are cheap wins.

10. **Evaluation gap.** Open-loop benchmarks saturate (every candidate ~0.98 vs recorded tapes); only closed-loop
    copy races discriminate, and the public field is still Python (slow). Converting the public roster into Rust
    tape libraries (team-bandit style) would let every tuning run use a mixed, fast, closed-loop fitness.

### 10.3 On running CMA-ES longer or larger

The held-out score plateaued from generation 20 on (0.98, then 0.95 three times), and sigma stayed ~0.07: more
generations of the same run would not help. The per-generation fitness is noisy (best candidate 0.91-1.09 between
consecutive generations with only 6 seeds), so a larger run should spend compute on MORE SEEDS per candidate and a
larger lambda, not more generations - and above all on a broader opponent population (item 1), because the open
question is overfitting to copies, not under-optimization.
