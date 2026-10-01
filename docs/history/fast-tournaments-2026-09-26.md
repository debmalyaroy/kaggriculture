# Fast tournaments on the Rust engine (2026-09-26)

How we now run thousands of full games in minutes instead of hours, what was slow, what was fixed,
how it was verified, and exactly how to run it.

## 1. Why the old tournament was slow

`python -m kaggriculture.bandit.tournament` plays our candidates against the public roster (93 Python
agents from Kaggle notebooks). Each game runs:
- our agent as a Rust subprocess (`agent-stdio`) behind a Python shim,
- the opponent as a PYTHON agent, turn by turn,
- the game loop through the Python bridge.

Measured on 2026-09-26: ~6.4 s per game on average, up to 27 s for the slowest opponents; 26,800 games
= ~4 h at 12 workers. The bottleneck is the Python opponents, not our agent.

## 2. The fast path: everything in Rust

When both players are Rust agents, the whole game runs in one Rust process on the bit-exact engine port
(`rustengine/v62`, same per-step state as the official interpreter; see CLAUDE.md "Build/verify the Rust
engine"). This covers:

| player | how it runs in Rust |
|---|---|
| our bandits (v63, v62.1, v63.1 / P92, any profile in `configs/bandit/profiles/v4.json`) | `agent::base::Base` with a profile |
| team bandits (top-50 players' tapes under the trie router) | `Base` loaded from `data/field/teams/bases/<team>/{S0,S1}` (`router.json` `"mode": "trie"`) |
| recorded opponents (ladder replays, open loop) | `tapeplay` (one side replays the recorded actions) |

Runners (all in `rustengine/v62/crates/runner/src/bin/`):

| binary | what it does |
|---|---|
| `selfplay` | base A vs base B over a seed list, N threads; one line per game: `seed, bank_a, bank_b, worst_us_a, worst_us_b, realized_world` |
| `tapeplay` | a Rust agent vs a recorded tape (`--other` puts the agent in the recorded player's seat; `--verify` replays both recorded sides and must reproduce the ladder banks) |
| `tapedump` | replays recorded games exactly and dumps per player per day: state, choices, EXACT sales and spend (counterfactual step per seat and product / buy key) |
| `teambase` | builds a team bandit (routes + trie router) from a team's GM tapes |
| `fieldplay` | a Rust agent vs a top player's tape library (reactive tape switching / seller); kept as a tool, not a benchmark |

The league driver is `python -m kaggriculture.bandit.league` (src/kaggriculture/bandit/league.py): every
pairing of the agent set, the same seeds from both seats, one `selfplay` process per ordered pairing,
results in `.local/league/<name>.jsonl` (resumable) and a report in `.local/league/<name>.json`.

## 3. Where the time went (profiled, not guessed)

Profiling is built in: `selfplay ... --stage-prof` prints per-game milliseconds for building the
observation, parsing it, the agents' act (with every chain stage and the pre/core/post phases) and the
engine step. `--slow-us N` lists every turn slower than N us with its top-5 stages; `--lat-dump FILE`
writes per-step agent times.

v63 vs v63, 6 games, one thread, BEFORE the fixes (1.34 s per game, 0.75 games/s):

| cost | ms / game | share |
|---|---|---|
| building the observation JSON (engine state -> ~40 KB string, twice a step) | 238 | 18% |
| parsing it back in the agent | 129 | 10% |
| agents' act, both seats | 803 | 60% |
| - of which stage 25 (reported as `opening`; it is the v92 rival-sales forecaster) | 268 (134 per agent) | 20% |
| - end-game planner simulations | ~40 | 3% |
| engine step | 16 | 1% |

The end-game planner was NOT the bottleneck.

## 4. The two fixes (both bit-exact)

### 4.1 Direct observation (`Obs::from_state`, `rustengine/v62/crates/agent/src/obs.rs`)
The harness used to serialise the full state to the official JSON observation and have the agent parse
it back, 1,440 times a game. `Obs::from_state(&State, seat)` builds the identical `Obs` struct straight
from the engine state (same field order, same interned names, same numbers). `runner::play_prof` uses it
by default.
- `KAGG_OBS_JSON=1` restores the JSON path.
- `KAGG_OBS_CHECK=1` builds both every step and panics on any difference (run over 16 games, both
  seats, every step: no difference).
- Kaggle submissions are unaffected: a real match hands the agent JSON (`agent-stdio`).

### 4.2 Fast v92 forecast (`forecast_fast`, `rustengine/v62/crates/agent/src/layers/v92.rs`)
From step 150 the forecaster re-scores every recorded sale stream of the shop pair every turn: for each
stream, each recorded sale in the last 240 steps is checked against the rival sales we observed (three
hash lookups), and each observed sale against the stream (three more). `forecast_fast` precomputes, per
stream, its sorted keys and a per-product bitset of "steps within 1 of a recorded sale", and keeps the
observed sales as bitsets too; every check is a bit test. Same integer counts, same score formula, same
stable sort, so the same streams in the same order. `KAGG_V92_SLOW=1` restores the old path.

### 4.3 Verification
- Final banks identical to the pre-fix binary: v63 (profile 71) vs v62.1, 24 seeds, md5 `d1a53e2e7f54`;
  v63.1 (profile 92, v92_top 3) vs v62.1, md5 `6018c96bdad4` (both JSON and direct paths, old and new).
- `KAGG_OBS_CHECK` clean on 16 games (v63.1 vs a team bandit, v63 vs v62.1).

### 4.4 Result

| per game, v63 vs v63 | before | after |
|---|---|---|
| observation build + parse | 367 ms | 13 ms |
| forecaster stage (both agents) | 268 ms | 38 ms |
| agents' act total | 803 ms | 482 ms |
| games per second per thread | 0.75 | 1.53 (2.0x) |

A team bandit with the shell off (S0, tapes only) costs ~9 ms per game for its own turns.

## 5. How to run

Build (use a separate target dir if a running job holds the binary; Windows locks running .exe files):

```
cd rustengine/v62
CARGO_TARGET_DIR=target-x cargo build --release -j 4 --bin selfplay --bin tapeplay --bin tapedump --bin teambase
```
(the fast build currently lives in `target-prof`; copy or rebuild into `target-x` once the running league
releases it - `league.py` calls `target-x/release/selfplay.exe`.)

One pairing, both seats, with profiling:
```
rustengine/v62/target-x/release/selfplay.exe --a configs/bandit/bases/v61.1 --b data/field/teams/bases/decem/S1 \
  --profiles configs/bandit/profiles/v4.json --pa 71 --pb 71 --seeds 700000,700001 --threads 1 --stage-prof
```

The league (all pairings, both seats, realized worlds recorded):
```
python -m kaggriculture.bandit.league --seeds 48 --workers 16 --name league1 [--only v63,v62.1,decem:S1]
```

Operator limits on this laptop: at most ~16-20 cores, launched at BelowNormal priority (children inherit
it):
```
$p = Start-Process python -ArgumentList '-m','kaggriculture.bandit.league','--seeds','48','--workers','16','--name','league1' -WindowStyle Hidden -PassThru
$p.PriorityClass = 'BelowNormal'
```

Throughput now: ~1.5 v63-vs-v63 games per second per thread, ~20 per second on 16 threads; a 22-agent
round robin with 48 seeds x 2 seats (22,176 games) is an estimated ~20 minutes (measured ~100 minutes on the
pre-fix binary; the estimate is from the per-thread rate and will be confirmed on the next league run).

## 6. Worlds: realized, never labelled

A world (the first two shops) is decided by play, not by the seed: the shop draw shares the per-day RNG
with weeds, so both players' actions change it. The old tournament labelled seeds by a PASS-only
simulation, and those labels matched the realized world in 0 of 64 seeds (64 seeds realized only 45
worlds). `selfplay` now prints the REALIZED world of every game (shops after step 145); the league groups
by it. To cover all 64 worlds, pick seeds by the world they realize in the pairing being tested.

## 7. What is still slow, and options

- The agents' shell (~60 stages, none above ~12 ms per agent per game; top: or2, r51_input, ca, cxd, ch,
  fx, e410, r36, fert, v44y). Another ~2x needs work across the top ten stages, each with an md5 check.
- The public roster is still Python (the slow tournament). Options: (a) keep it only as the final
  confirmation gate; (b) record each public agent's games and turn them into team-bandit style tape
  libraries (like the top-50 teams), which then play in Rust; (c) port the few strongest to Rust.
- `tapeplay`/`fieldplay` still build observations through JSON; switching them to `Obs::from_state` is the
  same one-line change as in `runner::play_prof`.

## 8. Stage coverage (2026-09-27)

`KAGG_FIRED_DUMP=<file> selfplay ...` appends, per game and seat, `seed, seat, bank, rival_bank, fired` where
`fired` is the comma list (one entry per `CUTS` stage) of steps on which that stage changed the action. Used for
the coverage table in `docs/history/bandit-architecture-2026-09-27.md`.
