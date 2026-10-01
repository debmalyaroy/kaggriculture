#!/usr/bin/env bash
# From the laptop: the REAL state of the box's queue, PPO progress and the latest verdicts.
#   bash aws/status.sh ubuntu@HOST
set -euo pipefail
HOST=${1:?usage: status.sh user@host}
ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'bash -lc "
cd ~/krl && source aws/env.sh
echo === box: \$(uname -m) \$(nproc) cpus, load \$(cut -d\" \" -f1-3 /proc/loadavg), free \$(free -g | awk \"/Mem/{print \\\$7}\") GB, disk \$(df -h ~ | awk \"NR==2{print \\\$4}\") free
systemctl is-active krl-queue >/dev/null && echo queue service: active || echo queue service: NOT ACTIVE
\$KRL_PY ops/queue.py status
r=\$(ls -d weights/ppo/ppo-* 2>/dev/null | tail -1); if [ -n \"\$r\" ]; then echo === PPO \$(basename \$r); tail -3 \$r/metrics.jsonl | cut -c1-240; fi
[ -f data/gates/tournament_latest.json ] && echo === tournament && cat data/gates/tournament_latest.json | head -40
systemctl list-timers krl-shutdown.timer --no-pager | sed -n 2p
"'
