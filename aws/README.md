# RL autopilot on AWS

The box runs the whole RL pipeline as `ops/queue.py` under systemd (`krl-queue`). Kaggle does only
two things: extract new ladder days in the private `krl-delta-*` notebooks, and host the submission
notebook, which the operator alone uses. The laptop builds the x86 package and runs the full
tournament for each candidate the box hands off. Architecture: `docs/ARCHITECTURE.md`.

## Operator: once

1. **Launch** a Graviton box (c8g.16xlarge, or c8g.8xlarge): Amazon Linux 2023 **ARM64**, 100 GB gp3,
   your key.
   - Security group: SSH (TCP 22) only, from your IP. scp, rsync and tunnels all use 22.
   - Spot: set the interruption behaviour to **Stop**.
   - vCPU quota: 64 for a 16xlarge or for two 8xlarge.
2. **Kaggle token:** copy `~/.kaggle/access_token` to the box's `~/.kaggle/` and chmod 600. The legacy
   `kaggle.json` key cannot read private notebook outputs.
3. **From the laptop** (`KRL_SSH_OPTS="-i ~/.ssh/KEY.pem"` for every `aws/*.sh`):
   - `bash aws/sync_up.sh ec2-user@HOST code`: code, configs and `harness/` (about 60 MB).
   - Send the corpus metadata once, as a tar: `data/slim/s1/{index,ledger,runs,corpus_manifest.tsv,source=league}`
     and `data/kaggle_out/delta_state.json`.
   - `ssh HOST bash ~/krl/aws/bootstrap.sh`: Python 3.11 venv with torch CPU and Kaggle CLI 2.2.4, a
     native ARM build of all binaries and of `harness/rustengine` (`kagg`), the `krl-queue` unit
     (`KillMode=process`: jobs outlive a runner restart), and auto-shutdown at 29 Sep 22:00Z.
   - `ssh HOST sudo systemctl enable --now krl-queue`. From here it is hands-off.

## What the queue does by itself

| Job | What | Time on a 16xlarge |
|---|---|---|
| Q00p | Pull the 12.6 GB corpus from our notebooks' outputs by `corpus_manifest.tsv`, exact sizes. Kaggle 429s back off automatically. | 30-60 min (rate limits) |
| Q02 → Q09 → Q09c | features, per-day observations (dayobs v2, 93 floats), training caches | ~10 min |
| Q00v | **`aws/verify.sh --no-start`**, the setup gate (list below); every training job waits on it | 5-10 min |
| Q20 | band tapes: the gate set (18 Sep on) and the PPO training set (10-17 Sep) | ~5 min |
| B00 | hold: training is released only after the build items are marked done | (manual) |
| Q07 → Q07c → Q05 | league 57.5k games on the box, cache, profile audit | ~50 min |
| Q11 → Q15c → Q14 | BC (with the family head) → PPO smoke → BC gate | ~40 min |
| Q16 | PPO nonstop. 30% of games vs real players' tapes; shield on; KL leash; Φ reward | to the lock |
| Q17 / Q18 | tournament every 8 h, then the candidate step: public panel + **band gate** → `data/candidates/` | per round |
| Q01 → Q02 … Q12, Q20 | delta every 12 h → features → BC fine-tune → new teacher; new band tapes | per delta |

`verify.sh` checks, all exact:
1. engine parity on real ladder games;
2. 180 exact replays, and live dayobs equal to corpus dayobs;
3. the Rust forward pass equal to torch;
4. v61.1 self-play banks equal to the laptop's, and profile 0 bank-identical to v61.1;
5. PPO smoke (when a teacher exists), and a games/s measurement.

## Candidates: box → laptop → operator

- The box writes `data/candidates/<box>-<name>/{policy.bin, report.json}` plus an index
  (`RC_PENDING.json`). No x86 build happens on ARM.
- The laptop runs `KRL_SSH_OPTS=... bash aws/pull_rc.sh ec2-user@HOST [HOST2]`, then:
  1. Rust == torch on x86;
  2. the x86 package (`scripts/build_submission.ps1`);
  3. the full tournament through `harness/` (all 88 public agents + v61, v61.1, v62, v62.1, 24 worlds,
     both seats) and `python/band_gate.py`;
  4. the official-engine spot check;
  5. the operator. **Nothing is ever submitted automatically.**

## Watching and stopping

- `ssh HOST 'cd ~/krl && source aws/env.sh && python ops/queue.py status'`: the real state.
- `python ops/queue.py dryrun [--release-holds]` checks the task graph without running anything.
- Stop the queue: `sudo systemctl stop krl-queue` (jobs keep running; `pkill` them to stop them too).
- Move the shutdown time: `bash ~/krl/aws/shutdown_at.sh "2026-09-29 18:00"`.
- A second box: launch it from an AMI of the first after the BC gate, set `KRL_BOX=B` in
  `aws/env.sh`, and give it one variant. Only box A pushes to Kaggle.
