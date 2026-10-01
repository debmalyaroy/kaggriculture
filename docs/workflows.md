# Workflows

All commands run from the repo root. Rust binaries land in `target/release/` (workspace) and
`rustengine/target/release/` (`kagg`). Everything they write goes to `data/` or `.local/` (gitignored).

## Build and test
```bash
cargo build --release                         # agent-stdio, chassis-vs-tapes, tapeplay, trace, ... (crates/*)
(cd rustengine && cargo build --release)      # kagg (serve / batch / prerank) for the Python package
pip install -e . && pip install -r requirements.txt
python -m kaggriculture.engine.engine_check   # the official engine is the one the ladder runs
python tests/run_all.py --fast                # --list shows the plan
```
A fresh clone has no data and no binaries, so suites that replay tapes, read generated models or call `kagg` report
missing files until those are rebuilt/regenerated ([data.md](data.md)).

## Tape tournament (fast, reproducible)
`chassis-vs-tapes` puts our agent in one seat of every recorded game and replays the other seat's stream (guarded) on
the same seed.
```bash
target/release/chassis-vs-tapes --agent CANDIDATE/agent.json --tapes data/tapes/top_all,data/ladder \
    --out .local/runs/cand.tsv --threads 16
target/release/chassis-vs-tapes --chassis CANDIDATE/base --route 1542 --tapes DIR --out r1542.tsv  # bare chassis, forced route
```
Row: `id seat band world our_bank their_bank gap route rec_us rec_them`, money us/them at days 6 / 12 / 24, guard
diagnostics (`guard:count@first_step`).

Environment hooks:
- `KRL_TRACE_DIR=DIR` — one `DIR/<id>.trace.tsv` per tape: `D` rows (per day: money, quadrants, hands, empty /
  locked / weed tiles, plants, dry plants, animals, empty structures, shed), `M` rows (every market order list of
  both sides), `E` rows (final money and unsold stock).
- `KRL_LAYER_TRACE=FILE` — one line per step on which any layer changed the action: `step  layer: before -> after`.
  This is how a deleted route sale is pinned to the exact layer (e.g. `shell: SELL WHEAT 7 -> 0`).
- `KRL_LAYER_LOG=FILE` — one JSON line per game: every installed layer, how often and from when it fired.

**Same-world rule.** A tape result is only valid when the realized world (column 4) is the same as the baseline's:
an earlier sale or buy can change which shops unlock, and then the opponent's recorded moves no longer fit the game.
Always split A/B results into same-world and changed-world games and decide on the former.

## Paired comparisons
Compare two candidates on the same tapes / seeds / seats: count discordant games (A wins & B loses vs the reverse) and
apply the exact sign test (McNemar). `src/kaggriculture/measure/win_metric.py` (`paired_test`, `flips`) implements it.
Never decide on mean margin.

## Ablations
To find which layer causes a behaviour, copy the candidate config and switch one thing off at a time:
- whole manager: `managers.<name>.on = false`
- one chain stage: append it to `managers.<owner>.stages_off`
- sale parts: drop `rshell`, set `shell` to another file, `batch.on = false`
Run each variant on the same tape set and rank by wins recovered. Then confirm with `KRL_LAYER_TRACE` on one game.

## Route screens
1. Pick target worlds (loss rate per world from a full tape run).
2. Split each world's tapes 50/50 by a hash of the tape id: screening half and held-out half.
3. Run every route forced (`--route R`, bare chassis) on the screening half.
4. Switch a world only when the best route beats the current one by a margin (wins, then mean gap).
5. Validate the full agent with the new routes against the old on the held-out half **and** all live replays;
   keep a world only if held-out is net positive and no live game is lost.

## Field runs (both sides react)
```bash
python python/rshell/public25.py --cand CANDIDATE/main.py --per-world 1 --workers 8 --out data/rshell/public25
python python/v6312/field_chassis.py make NAME CHASSIS_DIR          # candidate dir .local/fieldc/NAME
python python/v6312/field_chassis.py run NAME --workers 8 --per-opp 16 --vs BASELINE   # fast-track paired stop
python python/v6312/field_chassis.py cmp NAME_A NAME_B               # paired better/worse, sign test, per world
```
Python opponents run on the Rust serve engine (exact vs official). Worlds come from the 64-world bank, both seats.
A candidate folder is a submission stage (`main.py` + `agent-stdio` + `agent.json` and the files it names): build one
with `scripts/build_submission.ps1` or copy a release stage from `data/builds/<name>/stage`.
On Windows avoid `ProcessPoolExecutor(max_tasks_per_child=...)` — worker recycling deadlocks; reap finished
`agent-stdio` processes instead.

## Live games
```bash
python python/ladder_pull.py --subs 56720196,56718979 --sleep 2     # every finished game -> data/ladder/<sub>/tapes
target/release/chassis-vs-tapes --agent CANDIDATE/agent.json --tapes data/ladder/56720196/tapes --out live.tsv
python python/release/live_loss_classify.py live.tsv TRACE_DIR      # cash_starved / collapse / late / midgame
python python/release/bt_live.py --opp-subs 30                      # Bradley-Terry rank projection from live games
```
With the submitted config the replays reproduce the live banks exactly (reactive opponents excepted), so live tapes
are the strongest regression set.
