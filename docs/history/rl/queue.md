# RL task queue and Kaggle offload

`ops/queue.py` runs the RL track unattended. Tasks are specified in `ops/tasks.json`, and their state is kept in `ops/state.json`, written only by the tool.

```
python ops/queue.py status        # the real state: live pids, exit codes, log tails, free RAM
python ops/queue.py run           # the runner (start it outside Claude Code's memory reaper if possible)
python ops/queue.py mark ID done|running|failed|blocked [--note ..]   # dev (code) tasks
python ops/queue.py retry ID      # put a failed / lost job back to pending
```

## Rules

- **Two kinds of task.**
  - A `job` is launched by the runner when its dependencies are done, its lane has a free slot, and free RAM is at least `min_free_gb`.
  - A `dev` task is code work, marked by hand.
- **Jobs outlive the runner.** Each job runs detached through Git bash (called by its full path) and writes its exit code to `data/ops/logs/<id>.exit`.
  - A job whose process disappears without an exit code is reported as `lost`, never as done.
  - `retry` resumes it; every job skips work that is already on disk.
- **Recurring work.**
  - `every_h` re-arms a finished task that many hours after its last start. Q01, the delta, runs every 12 h.
  - `follow: [X]` re-runs a task once for each new round of X. This builds the delta chain below.
- **Gates run as jobs**, so a broken build never reaches training:
  - Q03c: the live dayobs vectors equal the corpus vectors on 180 real games.
  - Q10c: the Rust policy forward pass equals torch.
  - Q15c: a PPO smoke run.

## What runs where (box layout, 2026-09-25)

Everything runs on the AWS box (`aws/README.md`) except the delta extraction, which stays in the
private `krl-delta-*` notebooks. Setup jobs gate training: Q00p pulls the corpus and Q00v runs
`aws/verify.sh`. Training also waits on the B00 hold until the build items are done. `queue.py
dryrun [--release-holds]` simulates the runner (every job succeeds, lanes respected) and checks that
every command's script or binary exists. The previous Kaggle-offload spec is in
`.local/tasks.kaggle-offload.json`.

| Work | Job |
|---|---|
| Corpus pull (setup) | Q00p `python/delta.py pull-corpus` |
| Features, day_obs, caches | Q02, Q09, Q09c |
| Setup gate | Q00v `aws/verify.sh --no-start` |
| Band tapes (gate + PPO training sets) | Q20 `python/band_tapes.py` |
| League 57.5k games | Q07 `league-rand` → Q07c cache → Q05 audit |
| BC, fine-tune per delta | Q11, Q12 |
| PPO smoke, BC gate, PPO | Q15c, Q14 `python/learn/gate.py gate`, Q16 |
| Tournament every 8 h, candidate step | Q17 `gate.py tournament`, Q18 `rc.py cycle` (panel + band gate → `data/candidates/`) |
| Delta every 12 h | Q01 `python/delta.py run` |

Lanes: io, cpu, ppo, gate, rc, one job each. `{NCPU}` = cores - 2. A gate or rc job launched while PPO
runs gets `{NCPU_SHARE}` = 16 threads (`KRL_THREADS`). PPO gives the same 16 back while
`data/ops/lanes.json` shows a gate or rc job running.

## The daily cycle

1. **Every 12 h: delta.** Q01 → Q02 features → Q09 day_obs → Q09c cache → Q12 BC fine-tune. The new teacher goes into `weights/bc/LATEST` and `weights/registry.json`. The PPO run (Q16) picks it up at its next iteration (teacher hot-swap).
2. **Continuously: PPO.** Checkpoints and validation run every 10 iterations. `best.bin` is kept, and two drops in a row roll back to it.
3. **Every 8 h: tournament.** Q17 plays the BC teacher, PPO best and the last 3 snapshots against v61.1 on the box and writes `data/gates/tournament_latest.json`. Q18 then runs the public-panel gate and the band gate on the top PASS candidates.
4. **Submitting is never automatic.** A candidate that passes is handed to the laptop (`data/candidates/`, `aws/pull_rc.sh`), which builds the x86 package, runs the full tournament and asks the operator.

## Limits

- **Game speed.** About 5 games/s on the laptop (12 threads); the box's measured rate is in `data/ops/verify/report.txt`.
- **Band tapes do not react.** Rating-band results use real players' recorded streams on their own seeds, played through the guards that keep a stream legal (hand_align, weed_repair, clamp_sells: 60/60 exact when nothing is off). A stream cannot react to a different us, so this measures "against what that player actually did". The public-panel gate and the laptop tournament cover reactive opponents.
