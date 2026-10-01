# Kaggriculture: route tapes, a repair chassis and reactive layers — what two months of building taught us

## TL;DR
- Our final agent is a **Rust** program (static Linux binary behind a tiny Python bridge) that replays a recorded
  top-player **route** for the world it is in, keeps that route legal with a **chassis of repair guards**, and adjusts
  only the **market side** through ~65 reactive layers grouped into six managers (market guard, sale, endgame,
  rival, clone, economy), configured by one `agent.json`.
- We got there the hard way: a heuristic agent, route mining, a bandit over economies, a from-scratch closed-loop
  planner, an OR/MILP tape generator and two rounds of reinforcement learning. Most of those taught us *what not to
  do*; the architecture that survived is the one the strongest public agents converged on too.
- What actually moved the needle: a **bit-exact Rust engine** (fast, trustworthy measurement), **paired** tests in
  **wins**, replaying **our own ladder games** as tests, and **finding the exact layer** behind every loss.
- After the submission deadline we kept going and built v63.17: replayed on all 356 of our recent ladder games it
  wins 343 (the submitted v63.16: 328). Two root causes: a sales shell that deleted early sales while we were broke,
  and a handful of worlds routed to the wrong tape.

---

## 1. The game, and what it rewards
Two farmers share one market for 30 days (720 turns). Each turn you move a farmer and the hands you hired that day,
plant, water, harvest, keep animals, and send up to 10 market orders. The larger final bank wins; the ladder is a
skill rating over wins, so **a $1 win and a $50k win count the same**.

Four properties shaped everything we did:
1. **The market is shared and lockstepped.** Both players' sells hit one price curve, unit by unit, in order. Who sells
   first, and how much, decides a lot of close games.
2. **The world is realized by play.** Shops unlock during the game and change demand. The unlock draw shares an RNG
   with weed spawns, so *both players' actions* change which shops you get. "Same seed" does not mean "same world".
3. **Labour resets every night.** Hands are re-hired daily and spawn at the shed; most of a naive planner's hand-turns
   are spent walking. Logistics, not strategy, caps a from-scratch planner.
4. **Illegal actions are silent no-ops.** A broken plan does not crash; it just quietly does nothing, and looks like
   a lazy agent.

## 2. The journey

| Period | What we built | What we learned |
|---|---|---|
| Early Aug | Heuristic agent that prices every job in dollars; parameter tuning | Fine in isolation, far from the frontier. Margins misled us — the ladder pays wins |
| Aug | Mined the strongest players' games into **routes** (full-game action tapes) and replayed them | Route freshness mattered more than anything we invented; the frontier copies itself fast |
| Aug–Sep | **Bandit** over a few economies, opponent identification, relay/mirror handling | Opponent modelling was mostly noise; the economy underneath decides |
| Aug–Sep | **Track P**: a closed-loop planner (Python, then compiled Rust with search) | Labour logistics wall: ~$20–30k banks where tapes made $100k+. Search on top of a weak executor regresses |
| Mid Sep | OR/MILP tape generator; macro-plan BC → RL with a hand-written executor | The executor capped everything at $0–20k; RL cannot fix an executor it does not control |
| Sep 15 | **Decoded the public frontier**: every strong agent = per-world offline tape, routed by the first two shops at day 6, wrapped in an ordered stack of repair/sell guards | That is our bandit's shape. Stop re-deriving per turn what they precomputed offline |
| Sep 16–24 | Bit-exact **Rust engine** port, faithful serve harness, public-agent panels, ablations of the best public layer stacks | Exact, fast measurement; found the layers that matter (route sale advance, rival-sale predictor) |
| Sep 24–26 | Ported the best layer stack (v61.1, ~64 Python wrappers) to Rust line by line, bank-exact | One binary, < 20 ms/turn, every layer switchable |
| Sep 26–28 | **RL track**: learned day-profile policy (BC → PPO), learned sales shell, reactive shell, learned endgame controller | Some real gains, some dead models — see §5 |
| Sep 29–30 | New **chassis** (f898): per-world routes screened on all top-player tapes, repair guards, managers | The big jumps came from routing and repairs, not from more layers |
| Oct 1 (after close) | Loss forensics on every live game → v63.17 | See §7 |

Peak live ratings were around 2,410 (bandit v63.1 and RL v63.5); the last two submissions (v63.14, v63.16) were still
converging at the close (v63.16 at 2,179, rank 388 of 10,246 when last checked). A Bradley-Terry fit of v63.16's own
games projects roughly rank 300.

## 3. How we arrived at the architecture
We tried both ends of the design space before settling:

- **Per-turn judgment (planner, end-to-end policies)** has to solve labour choreography every turn: who walks where,
  who waters what before it dies, who carries animals from the shed. Every from-scratch attempt hit the same wall
  (measured: the majority of hand-turns spent travelling, banks a fraction of a recorded route's).
- **Fixed tapes** have the opposite problem: superb economy, zero adaptation, and they break silently when the game
  drifts (a missed purchase, a weed, a different world).

The frontier's answer, and ours: **keep the recorded farm plan, repair it, and make the market side reactive.**
- The tape owns everything structural (moves, plants, builds, animals, hires, land), so the farm stays in sync.
- A **chassis of repair guards** fixes the drift: re-align hands, clear weeds and replay blocked actions, buy missing
  land and retry builds, drop unaffordable buys, keep shed room, clamp sells to stock.
- **Reactive layers** may only touch market orders: when to sell, how much, in what order, ahead of whom.
- A **router** picks the tape for the observed world at day 6 (both first shops known), with optional overrides by
  the rival's farm cluster.

The sync rule ("after the chassis, only market orders change") is what lets you stack dozens of layers, swap configs
mid-game per opponent, and still replay games exactly.

## 4. The final agent (v63.16 / v63.17)
```
 obs ─▶ rival tracker ─▶ dispatcher (D6) ─▶ day profile (policy / group / schedule)
     ─▶ CHASSIS: router → route tape → repair guards (hand_align, weed_repair, land_repair, budget, room, clamp)
     ─▶ 64 chain stages (economy projects, sale timing, races, glut gates, order books)      [managers]
     ─▶ reactive shell → learned sales shell → sale manager → cash floor
     ─▶ pre-emption → game-theory standoff → final check → stream disguise ─▶ action
```
- **Router**: one tape per world (`"SHOP1|SHOP2"`), selected at step 144; per-(world, rival cluster) overrides.
- **Managers**: `market_guard`, `sale`, `endgame`, `rival`, `clone`, `economy` — each owns its chain stages and knobs;
  one `agent.json` switches whole managers, single stages or single knobs.
- **Opponent model**: rival group from day-0 signals (DIFFERENT / PARTIAL / COPY), a rival-stock tracker rebuilt from
  public state every turn, an hour-of-day sales forecast, a lineage fingerprint against our own route library.
- **Bridge**: `main.py` spawns the binary once, passes JSON lines, keeps a watchdog (5 s first turn, 0.25 s after) and
  falls back to a Python agent on any failure; it never raises.

## 5. Reinforcement learning — what worked, what did not
We ran RL three ways. The short version: **RL paid off only where it chose among options that were already
competent, and only when trained and validated in exactly the configuration we shipped.**

### 5.1 End-to-end and macro-plan RL (did not work)
- *Macro plan + hand-written executor*: a policy chose a seasonal plan, an executor turned it into unit actions. BC
  from top players' macro plans, then PPO. The gate read 0.000 in every band. Root cause: the **executor**, not the
  policy — it realized $0–20k from plans that earned the original players $190k, and more crew made it worse. RL
  cannot learn its way out of an executor it does not control.
- *Per-turn transformer policy* (pointer decoder over farmer, hands and orders): designed and partly built, but the
  data and compute needed to beat recorded routes on labour logistics were out of reach in the time left.

### 5.2 Learned day-profile controller (worked, modestly)
- **Action**: once a day (hour 1) pick one of ~36–60 **profiles** — complete sets of market-side knobs (sale windows,
  race horizons, glut gates, ...). Profiles never touch the farm plan, so every action is safe by construction.
- **Model**: about 32k parameters — Linear(→64) → GRU(64) → heads, over a 90-feature per-day observation built by
  the *same Rust code* for training data and at play time, plus rival-group one-hot. A rule-based **shield** masks
  profiles per rival group.
- **Training**: BC on profiles that won in league games (advantage-weighted), then recurrent PPO over whole 30-day
  episodes (gamma 1, GAE 0.95, clip 0.2, reward win 1 / draw 0.5 / loss 0 plus an annealed margin term), rollouts on
  the Rust engine against a league of our own variants and ported public agents, KL to the BC policy.
- **Result**: v63.2_rl (PPO i280) and later v63.5/v63.6_rl (i790/i910 + learned shell) reached about the same live
  rating as our best hand-tuned bandit (about 2,410), won a round robin against it 201–181, and each PPO generation
  beat the previous one in paired public-panel tests (e.g. 0.955 vs 0.945 over 3,200 games). Useful, not decisive.
- **Traps we hit**: (1) expanding the action table with rows copied from a base row put half the probability on
  tied copies — new rows need a negative bias at init; (2) a policy trained **without** the shipped sales shell
  looked better in its own validation and lost when played as shipped (p = 0.003) — train and validate in the shipped
  config, always.

### 5.3 Learned market components
- **Sales shell** (hold / sell part / sell all per product): trained on exact-engine counterfactual labels
  (branch the game, price both choices). It helps on average; its one hard failure — zeroing route sales while we had
  $10 — cost our three worst live games and is what v63.17's cash floor fixes.
- **Reactive shell v2**: a config-driven sale controller (35 global + 52 per-item features, an exact price
  calculator, urgency signals), tuned with CMA-ES and DAgger. Hard limits (quantities only, never units or buys) kept
  it safe.
- **Learned endgame controller**: two small MLPs choose one of 24 endgame knob patches at day 28. Version 1 was
  **dead** for days: trained only on games against copies of ourselves, 62 of 140 inputs were constant, every other
  opponent pushed them thousands of sigma out and every output saturated to exactly 0 — "neutral" results meant it
  never fired. Lesson: log a learned gate's decisions on the target field before calling it neutral.

### 5.4 RL takeaways
1. Give RL a **safe action space** (choices among competent behaviours), not raw control of a game with silent
   no-ops and labour logistics.
2. **Train = serve**: one feature function in Rust for both; train and validate in the shipped configuration.
3. A fast exact simulator matters more than model size: our policies were tiny; the Rust engine ran ~140 full
   episodes per second per core.
4. Standardisation guards on every learned component (constant features enter as 0, clamp the rest), and decision
   logs on the real field.

## 6. Measurement and infrastructure
- **Bit-exact Rust engine** (state, step, market, CPython's Mersenne Twister, worlds), verified against real replays.
  Everything else stands on it.
- **Tape tournaments**: our agent in one seat of every recorded top-player game, the other seat's actions replayed
  on the same seed. The full agent plays ~31,000 such games in about 1.5 hours on a 16-thread laptop; the bare
  chassis with a forced route is several times faster.
- **Field harness**: public agents (Python) against ours on the Rust serve engine — both sides react.
- **Live replays**: our own ladder games converted to tapes; with the submitted config they reproduce the ladder
  banks exactly, so they became our strongest regression set.
- **Decisions**: paired comparisons (same games, seats, seeds) in wins, exact sign test; per-world keep/revert on
  held-out tapes.

## 7. What we fixed after the submission closed (v63.17)
We downloaded every game of our live submissions and replayed all of them.

1. **Cash starvation (our worst losses, −$88k to −$131k).** Against opponents running our own route family the farms
   were identical at day 4. A per-step layer trace showed our learned sales shell cutting the route's early
   `SELL WHEAT 7–9` to zero while we had ~$10; the next day's hires failed, crops went unwatered and died, and the farm
   never recovered. Fix: a **cash floor** — while money is below $1,000 in the first 12 days, any route sale a shell
   cut is restored.
2. **Routing.** Two route screens (all 76 routes forced on half of each target world's tapes) and validation on the
   other half plus every live game: 10 worlds re-routed (held-out +87 / −10 in the first round), 2 candidate switches
   rejected because they lost held-out games even though they won a live one.
3. **Earlier in the last week**: `land_repair` (a missed land purchase made every later build target locked land and
   fail silently), and `wb3` (a wheat-buy reorder that fired unconditionally) restricted to a real signal.

Result, replaying all 356 live games of our recent submissions: **343 wins vs 328** for v63.16 (+10 / −0 where the
replay is valid); in the field, the round-1 build won 128/128 against v63.13 and 0.904 against the top 20 public
agents (2,104 games).

One measurement lesson came out of this the hard way: **a fix that changes your early actions can change which shops
unlock**, and then the opponent's recorded moves no longer fit the game. Seven of our first nine "wins" from the cash
floor were in such changed worlds. Always split replay A/B results into same-world and changed-world games.

## 8. Notes for the next simulation competition
1. **Port the engine first, bit-exact, in a fast language.** Every later decision depends on trusting the numbers.
2. **Decode the public frontier early.** Read the best public agents' structure before inventing your own; in this
   competition they had converged on the right architecture weeks before we did.
3. **Measure in the currency the leaderboard pays** (wins), **paired**, with a significance test. Never mean margins.
4. **Your own ladder games are the best test set.** Pull them continuously, replay them exactly, classify every loss
   by mechanism (starvation / collapse / late race / economy).
5. **Find the layer, not the symptom.** Ablate managers → stages → per-step layer trace on one game. Reading code
   first misled us twice.
6. **Silent no-ops need traces.** Log per-day farm state (empty tiles, stranded animals, cash); our worst bugs looked
   like "playing badly".
7. **Reactive layers must act on a signal and never delete actions they did not create.** Both late regressions were
   unconditional edits to the route's own orders.
8. **Validate on held-out data and real games; confirm in the field** when a change can alter the game's trajectory.
9. **Know the final evaluation.** Here the final ranking was a Bradley-Terry fit on post-deadline games of the last
   two submissions; the live rating was only seeding. Plan the final pair days before the deadline.
10. **Finish early.** Our best version was ready one day after the deadline; the same collapse pattern was already in our
    live games before the deadline, we just had not traced it yet.

## Acknowledgements
Thanks to the Kaggle team for a deep, well-run simulation, and to the public notebook authors whose agents and
write-ups taught us the shape of the problem — the route-tape + reactive-guard lineage in particular.
