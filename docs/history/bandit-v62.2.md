# Bandit track (v62.x): separate from the RL track

The bandit is the Rust port of v61.1 plus configurable knobs (lever profiles, `configs/profiles/v2.json`).
Every release is found by knob tournaments and shipped through the same verified path as v62 and v62.1.
Nothing here uses the RL queue (`ops/`), the RL trainers (`python/learn/`), the RL league or Kaggle data
jobs. The only shared code is the Rust agent crate; each new layer there is off by default and
sync-checked, so it cannot change RL or earlier bandit builds.

## v62.2 = the v62 / v62.1 process + one new layer

The new layer is the opponent-group end game with secret jitter (`crates/agent/src/layers/group.rs`).

**1. Identify the opponent's group.** Groups are defined beforehand from clone forensics on 324 real
games plus corpus clustering, and identified at runtime from public signals by the day-1 boundary.

| Group | Runtime signal |
|---|---|
| DIFFERENT | Our units stood on the rival's squares on less than 10% of day-0 turns (a different farm plan). |
| PARTIAL | Same squares, but different money at step 1 (our farm plan, a different opening trade). |
| COPY | Same squares and identical money at step 1 (a near copy: the pure market race, and the group that plans against us from our public replays). |

**2. Change course in the last days.** From day 24 the group's END-GAME profile replaces the base
profile for days 24–29. It is market-side only, so it stays in sync with the base.

**3. Jitter.** A secret per-game offset on the sale look-ahead and the race horizons, drawn from a build
salt and the game's opening state. Replays reveal only its distribution. It is set per group, and also
per group for the end game. Measured cost in 1,380 paired self-play games: ±1 free, ±3 significant
(p = .004).

**Knobs**, all agent-stdio flags:
- `--profile 19`: base, days 0–23 (v62.1)
- `--group 19,19,19`: per-group profile for days 1–23
- `--endgame X,Y,Z`: per-group profile for days 24–29
- `--jitter a,b,c`: per-group jitter, days 1–23
- `--jitter-end a,b,c`: per-group jitter, days 24–29

**Sync check:** `--group 19,19,19 --endgame 19,19,19 --jitter 0,0,0` must equal plain p19, bank for bank.

## Process (same as v62 / v62.1)

1. **Screen** the knob grid:
   - on the 324 real ladder games (exact replays, `tapeplay`), tuning on v61/v61.1 and judging on v62/v62.1;
   - with closed-loop self-play against copy and front-runner variants (`selfplay`).
2. **Tournament** for the top candidates: ALL public field agents plus v61, v61.1, v62 and v62.1, 24
   worlds, both seats, paired against v62.1 (p19) on identical games (`python/panel_gate.py --all`).
3. **Release candidate**, which must pass all of these:
   - significantly better than v62.1;
   - no opponent significantly worse;
   - at least even against v62 and v62.1.
4. **Build and ship:**
   - `scripts/build_submission.ps1` with the chosen flags;
   - the official-engine tarball check (719/719 turns through the bridge, 0 fallback);
   - the private payload dataset, and a new version of the private notebook `kaggriculture-adaptive-bandit-private`;
   - the self-check;
   - **operator approval**, then submit.
