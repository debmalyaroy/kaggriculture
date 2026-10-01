#!/bin/bash
# (5) CHECKPOINT ARCHIVER -- runs detached on the box.
#
# Uploads the FULL checkpoint to gdrive every 30M RL steps under a UNIQUE name
# (step + UTC timestamp) so archives are NEVER overwritten. ALSO uploads a tagged
# archive at each round milestone the 30M grid misses (100M/200M/250M/350M/400M/
# 500M/700M/750M/1B) -- once each. Also uploads the full BC weights+checkpoint ONCE
# when BC completes. This is the paid gdrive egress (~70 MB per archive); the
# per-150s local checkpoints stay on the box for resume.
#
#   nohup setsid bash scripts/trackp/ckpt_upload_loop.sh >/root/kaggriculture/ckpts/archiver.log 2>&1 </dev/null &
#
# Stop:  pkill -f ckpt_upload_loop.sh
set -u
cd /root/kaggriculture
export PATH=/venv/main/bin:$HOME/.cargo/bin:$PATH
export PYTHONPATH=src:vendor
PY=${PY:-/venv/main/bin/python}
CKPTS=/root/kaggriculture/ckpts
GDRIVE=${GDRIVE:-gdrive:kaggriculture}
ARCH="$GDRIVE/ckpt_archive"
EVERY=${EVERY:-30000000}                 # 30M steps between RL archives
# Extra ROUND milestones that the 30M grid (30/60/90/120...) misses. Each is
# archived exactly ONCE, tagged milestone<N>M, in addition to the 30M cadence.
# NOTE: exact multiples of EVERY (e.g. 150/300/450/600/750/900M) are ALREADY
# covered by the 30M cadence and are skipped below to avoid duplicate uploads.
MILESTONES=${MILESTONES:-"100000000 200000000 250000000 350000000 400000000 500000000 700000000 1000000000"}
POLL_MIN=${POLL_MIN:-10}                  # how often to check (minutes)
STATE="$CKPTS/.archiver_state"           # last archived RL step (30M cadence)
MSTATE="$CKPTS/.archiver_milestones"     # milestones already archived (one per line)
BC_MARK="$CKPTS/.archiver_bc_done"

gstep() { $PY - "$1" <<'PY' 2>/dev/null
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("global_step", 0)))
except Exception:
    print(0)
PY
}
bc_done() { $PY - "$1" <<'PY' 2>/dev/null
import sys, torch
try:
    print("1" if torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("done") else "0")
except Exception:
    print("0")
PY
}

[ -f "$STATE" ] || echo 0 > "$STATE"
touch "$MSTATE"
# Seed: any milestone ALREADY passed at startup is marked done, so a restart
# never back-archives a passed milestone using a much later checkpoint.
if [ -f "$CKPTS/policy_rl.pt" ]; then
  GS0=$(gstep "$CKPTS/policy_rl.pt")
  for M in $MILESTONES; do
    [ "$EVERY" -gt 0 ] && [ "$((M % EVERY))" -eq 0 ] && continue   # on the 30M grid -> skip (no dup)
    if [ "${GS0:-0}" -ge "$M" ] && ! grep -qx "$M" "$MSTATE"; then echo "$M" >> "$MSTATE"; fi
  done
fi
echo "=== $(date -u) archiver start (every ${EVERY} steps + milestones [${MILESTONES}], poll ${POLL_MIN}m) ==="
while true; do
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)

  # --- BC: upload full weights+ckpt ONCE on completion ---
  if [ ! -f "$BC_MARK" ] && [ -f "$CKPTS/policy_bc.pt" ] && [ "$(bc_done "$CKPTS/policy_bc.pt")" = "1" ]; then
    echo "[$STAMP] BC complete -> archiving full BC ckpt + milestones + metrics"
    rclone copyto "$CKPTS/policy_bc.pt" "$ARCH/policy_bc_final_${STAMP}.pt" && \
    rclone copy "$CKPTS" "$ARCH/bc_milestones_${STAMP}" --include "policy_bc_ep*_q*.pt" ; \
    rclone copyto "$CKPTS/metrics.jsonl" "$ARCH/metrics_bc_${STAMP}.jsonl" 2>/dev/null
    touch "$BC_MARK"
  fi

  # --- RL: archive full ckpt every 30M steps, unique name ---
  if [ -f "$CKPTS/policy_rl.pt" ]; then
    GS=$(gstep "$CKPTS/policy_rl.pt"); LAST=$(cat "$STATE")
    if [ "${GS:-0}" -ge "$((LAST + EVERY))" ]; then
      NAME="policy_rl_step${GS}_${STAMP}.pt"
      echo "[$STAMP] RL step=$GS (last=$LAST) -> archive $NAME"
      if rclone copyto "$CKPTS/policy_rl.pt" "$ARCH/$NAME"; then
        rclone copyto "$CKPTS/metrics.jsonl" "$ARCH/metrics_rl_step${GS}_${STAMP}.jsonl" 2>/dev/null
        echo "$GS" > "$STATE"
      else
        echo "[$STAMP] archive upload FAILED -- will retry next poll"
      fi
    fi
    # --- RL: extra ROUND milestones (100M/200M/250M/... off the 30M grid) ---
    for M in $MILESTONES; do
      [ "$EVERY" -gt 0 ] && [ "$((M % EVERY))" -eq 0 ] && continue   # on the 30M grid -> skip (no dup)
      if [ "${GS:-0}" -ge "$M" ] && ! grep -qx "$M" "$MSTATE"; then
        LB="$((M / 1000000))M"
        NAME="policy_rl_milestone${LB}_step${GS}_${STAMP}.pt"
        echo "[$STAMP] RL milestone ${LB} reached (step=$GS) -> archive $NAME"
        if rclone copyto "$CKPTS/policy_rl.pt" "$ARCH/$NAME"; then
          rclone copyto "$CKPTS/metrics.jsonl" "$ARCH/metrics_rl_milestone${LB}_${STAMP}.jsonl" 2>/dev/null
          echo "$M" >> "$MSTATE"
        else
          echo "[$STAMP] milestone ${LB} upload FAILED -- will retry next poll"
        fi
      fi
    done
  fi

  sleep "$((POLL_MIN * 60))"
done
