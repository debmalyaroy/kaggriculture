# Search on top of the tape router: MCTS online, genetic search offline (plan, 2026-09-26)

Status: **IN PROGRESS.** B0 value model built (`kaggriculture.bandit.value_model`; held-out-team AUC d3 .64, d10 .737 -
gate .75 FAILED - d15 .847, d20 .905): search runs from ~day 12. Sell-race headroom (`oracle`, perfect information, one
decision): in copy races one "hold sales" decision flips 7/8 (vs v63) and 3/4 (vs v62.1) of v63.1's losses (margins
$1-800, mostly at hour-0 steps); a fixed hold rule (no sales at hour 0 from day 12) then scored 0.750 -> 0.008 vs v63 and 0.896 -> 0.025 vs v62.1
(~-$2k/game): the oracle flips are multiple-comparison noise on coin-flip copy races (median loss margin $24-29). The
sell-race search (B2) is NOT justified; recommended replacement: CMA-ES tuning of shell knobs with closed-loop copy-race
fitness, confirmed on the 64-world tournament. Builds on `docs/history/team-bandits-plan-2026-09-26.md` (team bandits, trie router,
league). Nothing here is built yet.

## 0. Measured constraints (this box, Rust v62 engine)

| quantity | value | consequence |
|---|---|---|
| engine alone | ~48,000 steps/s, one thread (incl. tape parsing) | a 3-day (72-step) rollout following tapes costs ~1.5 ms |
| engine + v63 agent in one seat | 0.44 ms / step | a 3-day rollout with our agent in the loop costs ~32 ms: rollouts must follow tapes, not run the agent |
| Kaggle turn limit | 1 s on slower hardware; our house target worst turn <= ~150 ms (v63.1 worst 122 ms) | ~100 ms of search per decision point, only at decision points |
| cash lead -> win (top-40 games, 3,882 decided) | AUC 0.51 day 3, 0.57 day 6, 0.59 day 10, 0.64 day 15, 0.79 day 20, 0.96 day 29 | **cash is not a value before day 15**: early money is converted into animals, crops, land. The search needs an asset-aware value function (section 2.5) |

## 1. What to search over

Not single moves: one turn is farmer + every hand + up to 10 market orders, so raw-action trees never converge in a
turn. The decisions that matter are few and discrete:

| decision | when | options |
|---|---|---|
| R. route continuation | at a trie branch point of the agent's tape library (section 3 of the team plan); at most ~10-30 per game | continue, or switch to one of the sync-valid continuations (same recorded moves so far, same unit squares) |
| D. day plan | day boundaries (step % 24 == 0) where the library disagrees on buys | which tape's day: animal buy or not, crop, land now/later, hire count (these come bundled in continuations) |
| S. sell race | a turn where we hold ripe stock of a product the rival also produces | sell now / hold k steps (k in 1..6) per product |

R and D are both "which continuation", so one search handles them. S is a small separate search.

## 2. MCTS design (determinised, macro-action, anytime)

### 2.1 Root and actions
- Root = the live observation at a decision point. Actions = up to 8 sync-valid continuations (ranked by the trie
  router's similarity score; the rest pruned), plus "continue".
- Depth: the root choice plus the next branch point (depth 2 in macro decisions). Beyond that the rollout policy
  follows the chosen tape.

### 2.2 Determinisation (hidden information)
Each simulation samples one full engine state consistent with what we observe:
1. **Public state** from the observation (our farm, rival farm, market, town) via `obsstate` (it already builds
   an engine state from an observation; the rival's private part is not observed).
2. **Rival private state** (shed, seeds, carried items) and **rival continuation**: sample a tape from the global
   top-50 library whose recorded rival-side PUBLIC trajectory so far is closest to what the rival actually did
   (unit squares, board, market sales inferred from price moves); weight by similarity. Take that tape's recorded
   shed/seeds at this step, clipped to be consistent with the rival's visible board and past sales.
3. **World randomness** (future weeds, shop unlocks, market noise): a fresh RNG seed per simulation. The realized
   shops so far are fixed; future unlocks vary.

### 2.3 Rollout
Both sides follow tapes: we follow the candidate continuation, the rival follows its sampled tape, for H steps
(H = 72 by default, i.e. 3 days; 120 near the endgame). A light market rule runs on our side in the rollout (sell
ripe stock the tape sells, with the S-policy default) so rollouts see the race. Cost ~1.5 ms per rollout.

### 2.4 Selection and budget
- UCT (UCB1, c tuned in the league) at the root; progressive widening beyond 8 actions is not needed.
- Budget 100 ms per decision point -> ~50-60 rollouts; anytime: stop at the budget, pick the most-visited action.
- Fallback when out of time or when a switch is not sync-valid: the trie router's similarity choice (today's plan).
- Latency guard: per-turn cap 150 ms; if a decision point would exceed it, use the fallback.

### 2.5 Value at the horizon (gate B0, build first)
Cash lead is useless early (section 0). Fit a value model on the 237k player-days from `tapedump`
(`data/field/top40/days.jsonl`, later the week-50 set): features = money, rival money, shed contents at current
prices, seeds, plants by crop and ripeness, animals by kind, structures, quadrants, crew, day, shops; target =
final win (and final bank lead). Model: gradient-boosted trees (offline), exported as a compact table/tree
ensemble to Rust. Gate: AUC >= 0.75 at day 10 on held-out TEAMS (not held-out games), calibrated (Brier). The
search evaluates `P(win)` = model(state at horizon). If the gate fails, the horizon is extended (full-game
rollout to day 29 from day >= 15, ~10 ms each) and MCTS is restricted to the second half of the game.

### 2.6 Sell-race search (S)
At a turn with ripe stock of product X also held or produced by the rival: sample 16 rival continuations (as 2.2),
simulate "sell now" vs "hold k" for k in {1,2,4,6} over 24 steps, pick the best expected revenue minus the rival's.
~0.6 ms per rollout of 24 steps -> ~100 rollouts in 60 ms. Replaces the fixed sell_lead/racepx rules only if it
beats them in the league.

### 2.7 Where it lives
Rust, agent crate, new layer `layers/search.rs`, knob-driven (`search_on`, `search_ms`, `search_h`, `search_k`,
`search_c`), default off = today's behaviour bit for bit (md5 check of existing screens, as for dispatch). The
rival library ships inside the tarball: compact tapes (~2-5 MB zstd for 500 tapes), loaded once.

### 2.8 Validation
1. B0 value gate (2.5).
2. Offline A/B in the league: team bandit + shell vs team bandit + shell + MCTS, paired, realized worlds, both
   seats, against every other league agent.
3. Same on v63's own base (v63 + MCTS vs v63).
4. Latency: official-engine tarball check, worst turn <= 150 ms, 0 fallback turns.
5. Shipping only through the realized-world tournament + operator go.

## 3. Genetic search: where it makes sense

**Online (inside a turn): no.** Populations need thousands of evaluations; the turn budget allows ~60 rollouts.

**Offline: yes, in three places.**

### 3.1 GA over spliced routes (new chassis) - the main use
- Genome = a route built from segments of top-50 tapes, cut only at SYNC POINTS (steps where the two tapes'
  recorded unit squares and moves-so-far allow a switch with no desync), plus per-segment sell-timing genes
  (shift each recorded SELL by -3..+3 steps).
- Crossover = swap tails at a common sync point (desync-safe by construction). Mutation = replace a segment by
  another sync-valid continuation, or shift a sell gene.
- Fitness = mean win probability against a PANEL of rival tapes from the library (top-50, grouped by realized
  world), tape vs tape, no agent: ~15 ms per full game on one thread -> ~240k games/hour/core; 16 cores at low
  priority ~ 3.8M games/hour.
- Per realized world (64): evolve a route for each world, or a route family with world-keyed branches.
- Overfit guard: the panel is split by TEAM (train teams / held-out teams); a route is kept only if it wins on the
  held-out teams; final check with the full agent + shell in the league, then the realized-world tournament.
- Output: new routes for the trie library (a GA-found chassis), e.g. an early opening from one team spliced into
  a crop-heavy mid-game from another.

### 3.2 Knob tuning of the shell
Continuous knobs (margins, lookaheads, v92 settings) are better tuned with CMA-ES than a GA; we already have that
machinery. Use per-world or per-cluster knob sets only through the dispatch layer, validated on realized worlds.

### 3.3 Search hyper-parameters
`search_c`, `search_h`, `search_k`, value-model thresholds: small GA or grid in the league.

### 3.4 Honest risks
- Open-loop panel opponents do not react; a route that exploits a fixed tape can fail against a reactive player.
  Mitigation: later generations use the team bandits (reactive shell on) as fitness opponents (slower: ~0.9 ms/step).
- Track P's (1+lambda) evolution of schedules (2026-08-29) measured 0.000 against the reactive frontier, but on the
  old mis-ranking substrate with a weak panel; this plan differs in splicing only real top-player segments,
  desync-safe cuts, realized-world grouping and held-out teams.

## 4. Order of work and ETA

| phase | what | ETA |
|---|---|---|
| A | team bandits + trie router + gates + league (team plan) | ~5 h |
| B0 | value model from player-days, gate AUC >= 0.75 at day 10 on held-out teams | ~1.5 h |
| B1 | Rust `layers/search.rs`: determinisation, UCT over continuations, rollout, value, budget, fallback; bit-exact when off | ~4 h |
| B2 | sell-race search (S) | ~2 h |
| B3 | league A/B: +MCTS vs without, on team bandits and on v63 | ~1 h |
| C1 | sync-point index over the library; splicer; tape-vs-tape fitness runner (Rust) | ~3 h |
| C2 | GA per realized world with held-out teams; best routes into the trie library | ~3-6 h compute |
| D | combine: GA routes + trie router + MCTS + shell; league, realized-world tournament, tarball check | ~3 h |

Phases B and C are independent after A; B0 can start at once (it only needs the player-days already dumped).
