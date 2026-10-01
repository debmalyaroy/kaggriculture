#!/bin/bash
# PPO3 projects + daily-options oracle on the LAPTOP (operator 2026-09-27: single box, use the local as much as needed).
# Pulls the box's newest weights/ppo3 current.bin every 5 min, labels here, pushes each finished batch
# (o_*.oracle/.tsv/.traj) to the box's data/oracle3, where PPO3 reads it.
#   bash python/learn/local_oracle.sh [THREADS]     (from the repo root, Git bash)
cd "$(dirname "$0")/../.." || exit 1
BOX=ec2-user@<box-ip>
SSH="ssh -i $HOME/.ssh/aicm-key-openssh.pem -o ConnectTimeout=20 -o LogLevel=ERROR"
SCP="scp -q -i $HOME/.ssh/aicm-key-openssh.pem -o ConnectTimeout=20 -o LogLevel=ERROR"
OUT=data/oracle3_local
mkdir -p "$OUT" .local/oracle_local
T=${1:-12}

pull() {
  while true; do
    run=$($SSH $BOX 'cd ~/krl && ls -td weights/ppo3/ppo-*p60*/ | head -1' | tr -d '\r' | sed 's:/$::')
    if [ -n "$run" ]; then
      mkdir -p "$run"
      $SCP "$BOX:krl/$run/current.bin" "$run/current.bin.part" && mv -f "$run/current.bin.part" "$run/current.bin"
    fi
    sleep 300
  done
}

push() {
  while true; do
    for f in "$OUT"/o_*.oracle; do
      [ -e "$f" ] || continue
      b=${f%.oracle}; n=$(basename "$b")
      [ -e ".local/oracle_local/$n.pushed" ] && continue
      # a batch is finished once its .oracle has not changed for a minute
      [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 60 ] && continue
      $SCP "$b.oracle" "$b.tsv" "$b.traj" "$BOX:krl/data/oracle3/" 2>/dev/null && touch ".local/oracle_local/$n.pushed" && echo "[local-oracle] pushed $n"
    done
    sleep 60
  done
}

pull & PULL=$!
push & PUSH=$!
trap 'kill $PULL $PUSH 2>/dev/null' EXIT
sleep 20
KRL_BIN=target/release KRL_PROFILES=configs/profiles/rl5.json python python/learn/oracle.py --root weights/ppo3 --out "$OUT" \
  --games 96 --threads "$T" --keep 200 --days 8,9,10,11,12,14,16,17,18 \
  --cands 35,34,31,19,36,37,38,39,40,41,42,43,44,45,46,47,53,54,55,56,57,58,59 \
  --tape-frac 0.2 --mix mirror:30,v63:40,rand:30 --seed-bank data/worlds/w64_bank.json --seed-base 47200000
