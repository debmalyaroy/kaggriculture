#!/usr/bin/env bash
# ONE COMMAND from the laptop: an empty AWS box -> running RL autopilot.
#   bash aws/up.sh ubuntu@HOST            (or ec2-user@HOST on Amazon Linux)
# Steps (each re-runnable):
#   1. sync code + main-repo harness + training data (+ the slim corpus in the background at the end)
#   2. bootstrap on the box (toolchains, venv, native build, systemd service, auto-shutdown before the lock)
#   3. verify on the box (engine parity on real games, exact replays + obs equivalence, export
#      equivalence, closed-loop banks == laptop, PPO smoke, speed) -> starts the queue ONLY if all pass
# Prerequisite (operator): scp ~/.kaggle/kaggle.json HOST:~/.kaggle/kaggle.json
set -euo pipefail
HOST=${1:?usage: up.sh user@host}
cd "$(dirname "$0")/.."
ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'test -f ~/.kaggle/kaggle.json' \
  || { echo "copy your Kaggle token first: scp ~/.kaggle/kaggle.json $HOST:~/.kaggle/"; exit 1; }
bash aws/sync_up.sh "$HOST" code
bash aws/sync_up.sh "$HOST" repo
bash aws/sync_up.sh "$HOST" train
ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'bash ~/kaggriculture/aws/bootstrap.sh' 2>&1 | tail -25
# the slim corpus only feeds the delta index / re-rank; send it before verify (verify's engine parity reads it)
bash aws/sync_up.sh "$HOST" slim
ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'bash -lc "bash ~/krl/aws/verify.sh"' 2>&1 | tail -25
bash aws/status.sh "$HOST"
