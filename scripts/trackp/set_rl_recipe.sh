#!/bin/bash
# (10) SET RL RECIPE -- runs on the box. Persists an RL sample-efficiency recipe
# and cleanly restarts the pipeline so it takes effect. Durable: it writes
# /root/rl_recipe.env, which run_pipeline.sh sources (this script wires that in
# once), so the recipe survives supervisor restarts and box reboots.
#
# Pass the recipe as env vars (any subset; unset ones keep their default):
#   PPO_EPOCHS  (1|2|3..)   KL_C (e.g. 1.0)   TEACHER_MODE (bc|ema|refresh)
#   TEACHER_EMA_DECAY (0.999)   TEACHER_REFRESH_EVERY (200)
#   TEACHER_CKPT (/root/kaggriculture/ckpts/policy_bc_v2.pt)   ENVS   RL_STEPS
#
# Examples:
#   # fast 1-epoch with a firmer BC anchor:
#   PPO_EPOCHS=1 KL_C=1.0 bash scripts/trackp/set_rl_recipe.sh
#   # 1-epoch + EMA (target-net) teacher for consolidation:
#   PPO_EPOCHS=1 KL_C=1.0 TEACHER_MODE=ema bash scripts/trackp/set_rl_recipe.sh
#   # anchor to an improved local BC you uploaded:
#   TEACHER_CKPT=/root/kaggriculture/ckpts/policy_bc_v2.pt bash scripts/trackp/set_rl_recipe.sh
set -u
RP=/root/run_pipeline.sh
ENVFILE=/root/rl_recipe.env

# 1) wire run_pipeline.sh to source the recipe file (once, right after its export line)
if ! grep -q "rl_recipe.env" "$RP"; then
  sed -i '/^export GDRIVE=/a [ -f /root/rl_recipe.env ] \&\& source /root/rl_recipe.env  # RL recipe overrides' "$RP"
  echo "wired: run_pipeline.sh now sources $ENVFILE"
fi

# 2) write the recipe (only the vars you set; others are simply omitted -> defaults)
: > "$ENVFILE"
for V in PPO_EPOCHS KL_C TEACHER_MODE TEACHER_EMA_DECAY TEACHER_REFRESH_EVERY TEACHER_CKPT ENVS RL_STEPS; do
  if [ -n "${!V:-}" ]; then echo "export $V=${!V}" >> "$ENVFILE"; fi
done
echo "=== $ENVFILE ==="; cat "$ENVFILE"; [ -s "$ENVFILE" ] || echo "(empty -> all defaults)"

# 3) clean restart so the recipe applies (BC/RL resume from their checkpoints)
echo "=== restarting pipeline ==="
pkill -9 -f run_pipeline.sh; sleep 1
pkill -9 -f cloud_train; pkill -9 -f bc_train; pkill -9 -f rl_selfplay; sleep 3
setsid bash "$RP" >/dev/null 2>&1 </dev/null &
disown; sleep 2
echo "restarted. supervisors: run=$(pgrep -c -f run_pipeline.sh) ct=$(pgrep -c -f cloud_train)"
echo "verify the RL recipe once RL starts:  grep '\[rl\] SELF-PLAY' /root/kaggriculture/ckpts/pipeline.log"
