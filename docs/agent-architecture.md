# Agent architecture (v63.17)

The submitted agent is the Rust crate `crates/agent`, compiled to the static Linux binary `agent-stdio` and driven by a
Python bridge (`main.py`). This document describes it at the v63.17 state, module by module. File references are
relative to `crates/agent/src/` unless stated otherwise.

```
 Kaggle runner ──obs──▶ main.py (bridge) ──JSON line──▶ agent-stdio ──JSON line──▶ main.py ──action──▶ engine
                         │ fallback.py on any failure        │
                         ▼                                   ▼
                   (Python, optional)            cli::build(--config agent.json) -> Base
                                                             │
                     ┌───────────────────────── Base::act (base.rs) ─────────────────────────┐
                     │ 1 parse obs (obs.rs) -> View (view.rs)                                  │
                     │ 2 rival tracker update (gt.rs) -> race signal (lead_price / lead_signal) │
                     │ 3 dispatcher at D6 (dispatch.rs): swap market-side config packages       │
                     │ 4 controller: day profile at hour 1 (policy / group / schedule / ...)    │
                     │ 5 chain PRE phases, outermost first (layers/mod.rs)                     │
                     │ 6 learned endgame controller patches endgame knobs (endg.rs)             │
                     │ 7 CHASSIS: router -> route tape action -> repair guards (chassis.rs)     │
                     │      (terminal closure planner replaces it on steps 712-718)             │
                     │ 8 chain POST phases, innermost first: 64 stages                          │
                     │ 9 sale stack: rshell -> shell -> sale manager -> cash floor              │
                     │10 rival stack: tpp_mkt -> preempt -> gt standoff -> premium/forced buys  │
                     │11 final check -> stream disguise                                         │
                     └─────────────────────────────────────────────────────────────────────────┘
```

## 1. Process boundary

### `main.py` bridge (`kaggle/submission/main_config.py`)
- Pure stdlib. Finds `agent-stdio` next to itself (Kaggle `exec`s the file without `__file__`, so the directory is
  resolved from `__file__`, `configuration["__raw_path__"]`, `/kaggle_simulations/agent`, then the working dir).
- Spawns the binary once per episode: `agent-stdio --config agent.json`. One observation JSON per line in, one action
  JSON per line out, read by a reader thread with a watchdog (`FIRST_BUDGET` 5 s for the first turn: spawn + route
  table load; `TURN_BUDGET` 0.25 s afterwards; Kaggle's `actTimeout` is 1 s).
- The market list is passed through **unchanged**: empty `[]` orders and zero-quantity SELLs hold positional slots in
  the engine's lockstep market race.
- Any failure (spawn, crash, timeout, malformed reply) retires the binary for the episode; `fallback.py` (if shipped)
  plays on, else a legal PASS action. `agent()` never raises. `STATS` (turns, bridge, fallback, timeouts, worst_ms,
  build) is printed as a `RUSTV61 start/end` beacon on stderr — the private kernel checks it.
- `BUILD = "<name> bin=<sha12>"` is stamped by `python/release/pack_submission.py`.

### `agent-stdio` (`bin/agent-stdio.rs`, `cli.rs`)
- `cli::build(args)` turns either `--config agent.json` (→ `managers::expand`, which produces the equivalent flag list
  plus the settings that have no flag, applied by `managers::finish`) or explicit flags into a ready `Base`. The same
  builder is used by `chassis-vs-tapes --agent`, so tests set the agent up exactly as the submission does.
- Flags: `--base --profiles --profile --policy --shield --group --jitter --jitter-end --sched --clone-profile
  --clone-strict --disguise --chain-off --cut --shell --rshell --knob-over --preempt --group-knobs --group-knobs-for
  --group-knobs-lineage --endg --gt --dispatch --endgame --mirror-tol --route-table --timing --dump-knobs`.
- A stdin line `{"cmd":"swap_base","dir":...}` hot-swaps the base between turns. `--timing` prints per-turn latency.

### Observation and actions
- `obs.rs`: zero-allocation parser of the official observation into compact `Obs` (farms, tiles, market inventory and
  prices, shops, private shed/seeds/inventories). Every string is interned to `&'static str`; dict order is preserved
  wherever the Python reference iterates it.
- `view.rs`: `View`, the per-step snapshot every layer reads (step, own/rival farm, prices, shops, helpers such as
  `shed_adjacent`, `price`, `rival()`).
- `act.rs`: actions as token lists exactly as the engine reads them (`["SELL","MILK",12]`, `["HIRE"]`, `[]`); length
  and empty entries are significant.
- `market.rs`: the engine's price curve as the layers compute it (hinge gain 8, floor 1, banker's rounding).
- `sim.rs`: bridge from `Obs` to the bit-exact engine (`crates/engine`) for layers that simulate unit actions.

## 2. The chassis (`chassis.rs`, `router.rs`, `cluster.rs`)

The chassis **replays a recorded route** and keeps it legal. It owns the farm: everything structural (moves, plants,
builds, animals, hires, land) comes from the route tape; later layers change market orders only (the **sync rule**:
nothing after the chassis may desynchronise the tape from the farm).

### Base folder (`configs/bases/<name>/` or a candidate's `base/`)
- `routes.json`: route id → tape (one recorded top-player action per step, 720 steps).
- `router.json`: route tables + chassis `settings` (repair guards on/off and their parameters).

### Router (`router.rs`)
- Steps 0..143: the opening route (`opening_route`, default 0 = the base's opening). At step 2 it records the rival
  fingerprint `(round(rival money,3), market WHEAT)`.
- At `select_step` (144, day 6, both first shops known) it reads the first two unlocked shops (`"SHOP1|SHOP2"`):
  YARN worlds use the old table (`shop_routes_old`, if `yarn_uses_old`), the others the new table
  (`shop_routes_new`); `shop_routes_v92` overrides both; unknown worlds use `default_new/old`.
- At `cluster_step` (145) a per-(world, rival cluster) override may apply: `cluster_routes["SHOP1|SHOP2#<hex>"]`. The
  cluster key (`cluster.rs`) is an FNV hash of the rival's bucketed farm (tile counts per crop/animal, hands,
  quadrants), the same function used offline by `chassis-screen`.
- From `endgame_step` (648) the endgame route may take over (`endgame_route`).

### Turn order inside `Chassis::act`
1. Route tape action for (route, step).
2. **hand_align**: per-hand actions re-aligned to the hands actually hired (positional alignment with
   `farms[me]["hands"]`).
3. **weed_repair**: actions blocked by weeds are queued and replayed after a DIG (the `pending` replay queue).
4. **land_repair**: a PLANT / BUILD on a LOCKED tile is queued for replay and the next quadrant is bought this step
   when affordable (fixes the missed-BUY_LAND cascade).
5. Market guard hook A (owned by the market_guard manager, all OFF in a bare chassis): suppression of last step's
   early sales + R36 debt settlement, `sell_lead` (one-step-early sale of the route's next planned sells, with the
   RACEPX glut gate and the R36 window), `front_run` (pre-sell what the opponent's plan sells next step).
6. **budget_guard**: drops buys the bank cannot cover.
7. **animal_guard** (off): no new purchase of an animal kind that sits unplaced in the shed.
8. **room_guard**: keeps `room_margin` shed slots free.
9. **clamp_sells**: SELL quantities clamped to the projected shed (never cancels a route sale; an oversized SELL is
   harmless, an undersized one loses money).
10. Market guard hook B (off by default): `dead_stock` (sell surplus never used later), `terminal_liquidation`.

`projected_shed(action, view)` is the shed after this turn's unit actions; most market layers use it.
Diagnostics (`guard:count@first_step`) are printed per game by the tape tools.

## 3. The layer chain (`layers/`)

The v61.1 Python agent was a stack of ~64 wrappers; the Rust port keeps them in **Python call order** as 65 stages
(`layers::CUTS`). A wrapper is `pre(obs) → parent(obs) → post(obs, action)`: pre phases run outermost first, then the
chassis, then post phases innermost first. Cross-layer state written by an outer pre and read by an inner layer lives
in `Ctx` (R37 horizon, V9 item horizons, RACE horizon). `--cut NAME` truncates the chain after a stage (`chassis` = bare
chassis). A stage is skipped when its bit is set in `knobs.off` (per profile) or `game_off` (whole game, from a
manager's `stages_off`); a game-off stage also skips its pre phase.

| # | Stage | Manager | What it does |
|---|---|---|---|
| 0 | chassis | — | route replay + repair guards (section 2) |
| 1 | terminal | endgame | 7-turn terminal closure planner (wraps the chassis, steps 712-718) |
| 2 | room | market_guard | shed-room guard on market orders |
| 3 | v28 | market_guard | entry guard (no-op in Rust) |
| 4 | v219 | economy | V219 finite late tomato investment (PIZZA / FARMERS_MARKET worlds) |
| 5 | experiment | sale | no-op (APPLY_TIMING = False) |
| 6 | order | sale | sales first from step 144 |
| 7 | v31 | market_guard | entry guard (no-op) |
| 8 | v231 | economy | V231 sheep → cow substitution in milk worlds (pre + post) |
| 9 | r36 | sale | R36 reservation: pull upcoming tape SELLs of stock already held forward (192..695), book debts; RACEGATE leaves glutted items to the tape |
| 10 | r37 | sale | R37 horizons (2/3/4) + R44 cash probe; reorder SELL runs by rival-batch revenue loss (pre + post) |
| 11 | release | sale | no-op post |
| 12 | v233 | economy | V233 six-sheep SE expansion |
| 13 | r46 | endgame | no-op post (terminal planner interplay) |
| 14 | r51_input | economy | R51 finite-harvest fertilizer tour (pre + post) |
| 15 | r51_warehouse | economy | R51 warehouse close |
| 16-17 | r53, r70 | economy | helpers used by V219 (no-op posts) |
| 18 | r85 | economy | R85 economic overlay: feed skip, surplus fertilizer sale, hour-23 warehouse pass |
| 19 | r95 | economy | R95 wheat-replenishment trim (off in v63.x) |
| 20 | r97 | economy | R97 wheat-supply guard |
| 21 | courier | economy | V9 courier (pre + post) |
| 22 | carrot | economy | V9 carrot (pre + post) |
| 23 | herd | economy | no-op |
| 24 | fert | economy | V9 fertilizer |
| 25 | opening | economy | V9 opening + v92 PREDICT (rival premium-sale forecast, off unless `v92_on`) |
| 26 | v9_race | sale | V9 RACE item horizons (pre) + snapshot (post) |
| 27-28 | racepx, racegate | sale | wrappers (logic lives in chassis RACEPX / R36 RACEGATE) |
| 29 | ctrtable | sale | counter table: scripted wheat buy/sell vs two known opponents |
| 30 | overflow | market_guard | V9 overflow |
| 31 | ca | economy | CA carrot instead of wheat when the carrot book pays |
| 32 | or2 | sale | OR2 sales ordered by the rival's estimated sellable stock |
| 33 | ch | economy | CH harvest animals that would overflow tonight |
| 34 | sr | economy | SR shed room |
| 35 | hd2 | economy | HD2 herd choice at the tape's goose purchase (off in v63.x) |
| 36 | cs | economy | CS cow → goose swap (off in v63.x) |
| 37 | race | sale | RACE clone-gated reservation horizon (pre) + rival snapshot (post) |
| 38 | r127 | sale | R127 drop last-hour plants that end unwatered (off in v63.x) |
| 39 | preguard | sale | PG pre-guard (hours 21-22) |
| 40 | v44y | sale | V44Y lockstep best-response SELL ordering |
| 41 | y | economy | Y shop-aware herd (off in v63.x) |
| 42 | e335 | sale | E334 / E335 sale-slot compaction |
| 43-44 | v11, v13v | sale | no-op posts (V13V wraps V219's request) |
| 45 | wl | economy | E343 weed-lag replay |
| 46 | adv | sale | ADV ready-stock sale advance |
| 47 | t62a | market_guard | T62A terminal clear |
| 48 | pipe | economy | PIPE EarlyCycle opening (rewrites route 0: off in v63.x) |
| 49 | ma | economy | MA mirror classifier (pre): rival drained >= 15 wheat on turn 1 |
| 50 | wb3 | sale | WB3 wheat buys first in BRUNCH worlds; `wb3_mode` 2 = only on the MA signal and only if cash covers the other buys |
| 51 | fx | sale | FX/EV quote-history lead-sells (EV window 15-20h) |
| 52 | dp | sale | DP lead-sell window (0-2h) |
| 53 | mp | sale | MP lead-sell window (10-13h) |
| 54 | bd | sale | BD buy-the-dip |
| 55 | mpx | sale | MPX model lead |
| 56 | sm | sale | SM shield-milk (off in v63.x) |
| 57 | cxd | sale | CXD exact best-response SELL ordering vs a rival model (copy / v92 / worst case) |
| 58 | e410 | economy | E410 fertilizer guard |
| 59 | e402 | economy | E402 late seed cap |
| 60 | mg | market_guard | MG same-item order merge |
| 61 | ig | market_guard | IG queue hole-closure |
| 62 | rsa | sale | RSA route sale advance (glut guard 0.5 × base) |
| 63 | afr | rival | AFR anti-front-run (off unless `afr_on`) |
| 64 | tsell | endgame | end-game sale-timing search (off unless `tsell_on`) |

Stages with a pre phase: 8 v231, 10 r37, 14 r51, 21 courier, 22 carrot, 26 v9_race, 37 race, 45 wl, 48 pipe, 49 ma,
51 fx, 54 bd, 55 mpx.

## 4. Knobs and the controller (`layers/knobs.rs`, `layers/group.rs`, `layers/mod.rs`)

### Knobs
`Knobs` holds every tunable of the chain (80 fields; `Knobs::default()` = v61.1 exactly). A **profile** is a full
`Knobs` value; profiles change only market-side timing / quantities and a few labour-safe switches, never the tape's
planting, hiring, land or animals. Owners (`managers::KNOBS`):

| Group | Knobs (default) | Owner |
|---|---|---|
| Route sale advance | `rsa_on` (on), `rsa_look` 5, `rsa_min_frac` 0.5 | sale |
| Lead-sell windows | `ev_on/ev_h` (on, 8), `dp_on/dp_h`, `mp_on/mp_h`, `mpx_on`, `lead_frac` 0.75, `lead_from` 96, `ev_hours` 15-20, `dp_hours` 0-2, `mp_hours` 10-13 | sale |
| Order books | `cxd_on`, `v44y_on`, `cxd_model` 0 (copy) | sale / rival |
| Ready-stock advance | `adv_on`, `adv_look` 3 | sale |
| Race horizons | `race_clone` 9, `race_escalated` 24, `race_mirror` 24, `race_from` 216 | clone |
| V9 race clamp | `v9_race_default` 44, `v9_race_max` 48, `v9_race_margin` 12 | sale |
| Glut gates | `racepx_margin` 0 (v63.x: −10), `racegate_margin` 0 | market_guard / sale |
| R36 / R37 | `r36_from` 192, `r37_base` 2, `r37_streak` 3, `r37_late` 4, `r37_late_from` 288, `r37_sim` 0.90 | sale / clone |
| Economy switches | `r85_feed_on`, `e410_on`, `r51_input_on`, `v9_fert_on`, `courier_on`, `carrot_on`, `e402_on`, `v233_on`, `v231_on`, `hd2_on`, `cs_on`, `y_on`, `ca_margin` −20, `v219_min_shops` 3 | economy |
| Slot swap | `or2_slot_margin` 12 | sale |
| AFR | `afr_on` (off), `afr_extra` 1, `afr_jitter` 2, `afr_hold` 48, `afr_look_max` 24, `afr_min_frac` 0.5 | rival |
| v92 predictor | `v92_on` (off), `v92_h` 48, `v92_k` 4, `v92_every` 2, `v92_top` 1, `v92_ext_window` 0 | rival |
| Terminal planner | `term_on` (on), `term_start` 712, `term_sims` 64, `term_passes` 1, `term_props` 4 | endgame |
| Market pressure | `press_on` (off), `press_from` 456, `press_trigger` 2.0, `press_h` 24, `press_tranche` 4, `press_max` 18 | sale |
| End-game sale timing | `tsell_on` (off), `tsell_from` 672, `tsell_window` 24, `tsell_model` 0, `tsell_min` 1.0 | endgame |
| WB3 | `wb3_mode` 0 (v63.16+: 2) | sale |
| Stage mask | `off` (bit = chain stage) | sale |

Knob sources: every profile table entry is loaded with `knob_over` (chain-constant overrides) and then the manager
`knobs` from `agent.json` applied on top (a knob may only be set by its owning manager). On each profile switch the
chosen profile's knobs then get the per-game timing jitter of the rival's group and, last, the group overlay
(`group_knobs` for rival groups in `group_knobs_for`, optionally only when the lineage fingerprint matched).

### Controller: which profile plays today
At each day boundary (hour 1) the chain picks the day's profile from these sources, first match wins (default
precedence `request, afr, clone, group, schedule`; configurable as `controller.precedence`):
- **request**: the learned macro policy (`policy.bin`, a GRU over the per-day observation, `crates/policy`) picks a
  profile greedily; an optional **shield** masks profiles per rival group.
- **afr**: escalate to a profile after `afr_trigger` pre-emptions in the past day.
- **clone**: a clone-gated profile on days a clone is detected (strict gate: ≥ 20 of 24 steps on the rival's squares
  and layout similarity ≥ 0.95).
- **group**: the opponent-group controller's profile for the rival's group.
- **schedule**: a fixed per-day profile list.

### Opponent groups (`layers/group.rs`)
Fixed at the day-1 boundary (step 25) from public day-0 signals, measured on 324 real ladder games:
`0 DIFFERENT` (our units on the rival's squares < 10% of day 0 — a different plan), `1 PARTIAL` (same squares,
different money at step 1 — our plan, another opening trade), `2 COPY` (same squares and identical money — a near copy,
the pure market race). A group only selects a profile and jitters market-side timing knobs (`rsa_look`, race horizons)
by a per-game, per-day offset drawn from a secret salt and the opening state.

## 5. Managers (`managers/`)

The layers are grouped into six plug-and-play managers. `agent.json` turns a manager off whole (`"on": false`, which
also forces its out-of-chain layers off: endgame → `term_on/tsell_on` false, rival → `afr_on/v92_on` false, sale →
`press_on` false), switches its stages off (`stages_off`) and sets the knobs it owns.

| Manager | Owns |
|---|---|
| market_guard | chassis hooks A/B (sell_lead + R36 window + RACEPX, front_run, dead_stock, terminal_liquidation), the guard settings (`land_repair`, `animal_guard`, `verify_fills`, `lead_signal*`, `lead_price*`), hygiene stages (room, v28, v31, overflow, t62a, mg, ig), the final check |
| sale | sale-timing stages, the reactive shell, the sales shell, the sale manager (batch / front), the cash floor |
| endgame | terminal planner knobs, R46, TSELL, the learned endgame model, the game-theory layer |
| rival | rival identification and front-running: preempt (lineage fingerprint), AFR, V92, the group knob overlay |
| clone | clone / lineage handling: race + R37 horizons, clone profile, per-game jitter, the disguise |
| economy | economy projects (V219, V231, V233, R51, R85/R95/R97, courier, carrot, CA, CH, herd, ...) |

`managers::expand` validates ownership (a knob or stage set by a manager that does not own it is an error) and maps
the file to flags; `agent-stdio --dump-knobs` prints the full table.

## 6. Opponent model

| Component | File | Signal |
|---|---|---|
| Group | `layers/group.rs` | DIFFERENT / PARTIAL / COPY at step 25 |
| Cluster | `cluster.rs` | rival farm-economy key at step 145 (route overrides, preempt clusters) |
| Rival tracker | `gt.rs` (`RivalTracker`) | rival unsold stock per item every turn (harvests − market sales, from public state), per-hour sale table → `sale_forecast` |
| Race signal | `base.rs` | items actually in a race now (`lead_signal`: the rival sold it at the coming hours on ≥ p of days and holds stock; `lead_price`: selling now pays vs the price at the planned step) |
| Lineage fingerprint | `preempt.rs` | rival matches a route of our own library (COPY with identical cash ≥ `lin_cash` of steps 1..144 → LINEAGE, else CLONE) |
| v92 predictor | `layers/v92.rs` | rival premium-sale forecast from a library of recorded top-player sale streams (`data/v92_lib.bin`, compile-time asset) |
| Dispatcher | `dispatch.rs` | at D6: class (DIFF / PARTIAL / LINEAGE / CLONE) × world → one package of market-side configs (gt, policy, rshell, preempt, endg) |

## 7. Sale stack (after the chain)

1. **Reactive shell v2** (`rshell.rs`, config `rshell/rshell.json` + `net.json` + `lineage.json`): one learned,
   config-driven sale controller over 9 items. Inputs: 35 global features (time, money gap, cash, group, cluster,
   clone gate, position streaks, layout similarity, lineage match, race horizons, AFR count, shed room, profile, ...)
   and 52 per-item features (stock, market inventory, price vs base and mean, glut, town drain, rival recovered
   sales, rival visible production, pre-emptions, three rival-sale forecasts — copy / v92 / lineage —, our plan,
   the chain's order and debts, and the **price calculator**: revenue of each sale fraction now and the rest at the
   best later step on the exact price curve). Decision per item: sale fraction of the projected shed
   (0, 1/4, 1/2, 3/4, all) and slot priority; score = model logits + bias + stay bonus + price term + urgency signals.
   Hard limits: only SELL quantities and order change, quantities ≤ projected shed, FERTILIZER never below committed
   need, 10-order cap, active only between `from` and `to`.
2. **Learned sales shell** (`shell.rs`, `shell.json`): per product (8, not fertilizer) and turn, hold / sell part
   (half) / sell all from 20 features (time, stock, market inventory, price, price deltas, money gap, shop demand,
   stock share). Overrides the chain only when ≥ `tau` confident and in disagreement. Training features are written by
   `crates/runner` `branch` and fitted by `python/top50/train_shell.py`.
3. **Sale manager** (`managers/sale.rs`, `batch`): an item is glutted when its quote ≤ `glut_ratio` × base. Not
   glutted: sell big down to the glut line. Glutted: only the demand batch (town drain over `window` steps × `demand_mult`
   + `min_batch`), carry the rest. Front-run: when the rival is forecast to sell X within `front_look` steps (hour
   table p ≥ `front_p` and stock ≥ `front_min_stock`, or a state-driven dump), sell a batch of X now. Market orders
   only; flush from `flush_from`.
4. **Cash floor** (`base.rs`, `cash_floor {money, until}`): while our money < `money` and step < `until`, every route
   SELL the three layers above cut is restored to the route's own quantity (v63.17: $1,000 until step 288).

## 8. Rival stack and endgame

- **tpp_mkt** (`tpp.rs`, off): the top-player cloned policy sets selected market ops in a step window.
- **Pre-emption** (`preempt.rs`, `preempt.json` + optional `lineages.json` clusters): bins the rival's sales by hour of
  day; when the rival is forecast (p ≥ `p_min` after `min_days`) to sell X within `look` (+ jitter) steps and we hold X
  with our own sale planned within `horizon` steps, sell that planned amount now above a price floor. Runs late so no
  later layer undoes it; leaves the final liquidation (from `to`) alone.
- **Game-theory layer** (`gt.rs`, `GtLayer`, `gt.json`): mid-game and end-game standoffs against chosen rival groups —
  per (item, phase, price level) a mixed strategy (`mix`, from `python/v6312/g2_msne.py`) between hold, tranche and
  dump; tranche size, minimum stock, end step and jitter are knobs.
- **Learned endgame controller** (`endg.rs`, `endg.json`): at one decision step (default 673) two small MLPs score K
  endgame proposals (patches over `term_*` / `tsell_*`) from the reactive shell's inputs; modes observe / force /
  rerank / replace; the chosen patch holds to the end. Labels: `crates/runner` `endg-label`; training:
  `python/endg/train.py`.
- **Terminal closure planner** (`terminal.rs`): at step 712 the chassis's remaining actions are replayed on a shadow
  state with the engine's unit semantics, and a bounded search over worker harvest / collect / deliver routes proposes
  a plan that must physically dominate the baseline; steps 713-718 follow it while the farm matches the expectation.
- **Premium sell / forced buys** (`base.rs`, experiment hooks set only by the runner tools, `crates/runner`
  `--prem-sell FILE` / `--force-buy STEP:Q`; never in a release): sell down to a price floor after the town's drain
  ticks; append wheat buys at fixed steps.

## 9. Final check and disguise
- **Final check** (`managers/market_guard.rs`, `FinalCheck`): legality of every order, the 10-order cap, optional
  clamps and minimum price.
- **Stream disguise** (`disguise.rs`): our replays are public and can be matched byte for byte, so zero-quantity SELL
  orders of products we hold none of are appended at the end of the market list (never moving a real order's slot,
  never past the cap) on a per-game pattern from a secret salt and the opening state. The game is unchanged.

## 10. Learned components and where they are trained

| Component | Agent side | Labels / data | Training |
|---|---|---|---|
| Macro policy (day profile) | `crates/policy`, `--policy` | `league-rand`, `ppo-rollout` | `python/learn/` (BC, PPO, oracle) |
| Reactive shell v2 | `rshell.rs` | `branch2`, `buylabel` (exact-engine labels) | `python/rshell/train.py`, `cmaes.py`, `dagger.py` |
| Sales shell | `shell.rs` | `branch` | `python/top50/train_shell.py`, `eval_shell.py` |
| Endgame controller | `endg.rs` | `endg-label` | `python/endg/train.py` |
| Game-theory mix | `gt.rs` | `gt-label` | `python/v6312/g2_msne.py` |
| Top-player policy | `tpp.rs` | `bcdump` | `python/tpp/train.py`, checked by `tppcheck` |
| Opponent clusters | `preempt.rs`, `cluster.rs` | corpus | `python/opp_cluster2.py` |

## 11. Configuration files of a release (v63.17)

```
agent.json                    base, controller, managers (docs/rust-agent.md shows the full v63.17 file)
base/routes.json              route tapes (f898 chassis, 64 worlds routed)
base/router.json              router tables + chassis settings (land_repair on)
profiles.json, knobs.json     profile table, chain-constant overrides
policy.bin, shield.json       macro policy + group shield
shell_open.json               learned sales shell
rshell/rshell.json (+ net, lineage)   reactive shell v2
preempt.json, lineages.json   pre-emption + clusters
group_knobs_rpx0.json         COPY-group knob overlay (racepx_margin 0 vs copies)
endg.json, gt.json            endgame controller, game-theory layer
dispatch/dispatch.json        specialist dispatcher packages
main.py, agent-stdio          bridge + binary
fallback.py                   Python fallback agent (used only if the binary fails)
```

## 12. Invariants
- **Sync rule**: after the chassis only market orders change; unit actions come from the tape (plus repair guards), so
  every profile, group or package switch stays in sync with the farm.
- **Route sales are sacred**: a layer may move a route sale earlier, split it or reorder it, but must not delete it
  without a signal (both late regressions — `wb3` and the sales shell cutting early wheat sales — broke this).
- **Determinism**: the agent is deterministic given the observation stream (jitter is seeded from the opening state),
  so live games replay exactly on the Rust engine.
- **Latency**: the worst turn reported by Kaggle validation (~550 ms for v63.16 / v63.17) is the FIRST turn: binary
  spawn + route-table load, under the bridge's 5 s first-turn budget. Every later turn must answer within the 0.25 s
  watchdog (else the binary is retired to the fallback); the validation runs had zero fallbacks.
