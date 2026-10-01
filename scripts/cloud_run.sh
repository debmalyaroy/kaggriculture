#!/usr/bin/env bash
# SINGLE entrypoint on the rented VM: setup + download + supervised training.
# Restarts cloud_train on any crash (resumes from the last ~150s checkpoint), so
# a failure costs at most ~2-3 min. Runs until the pipeline reports COMPLETE.
#
#   GDRIVE=gdrive:trackp WORKERS=16 bash scripts/cloud_run.sh
#
# Prereqs on the VM: rclone configured with a Google Drive remote (rclone config),
# this repo present (git clone or gdrive:code), a GPU with CUDA + torch.
set -uo pipefail
export GDRIVE="${GDRIVE:-gdrive:trackp}"
export ROOT="${ROOT:-$(pwd)}"
export WORKERS="${WORKERS:-16}"
export CKPTS="${CKPTS:-$ROOT/ckpts}"
export PYTHONPATH="src:vendor"
cd "$ROOT"

bash scripts/cloud_bootstrap.sh || { echo "bootstrap failed"; exit 1; }

echo "[cloud_run] starting supervised training loop (GDRIVE=$GDRIVE WORKERS=$WORKERS)"
tries=0
while true; do
  python -m kaggriculture.pipeline.cloud_train
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[cloud_run] cloud_train COMPLETE."
    break
  fi
  tries=$((tries+1))
  echo "[cloud_run] cloud_train exited rc=$rc (crash/partial). Resuming in 5s (restart #$tries)..."
  sleep 5
done
echo "[cloud_run] DONE. Final checkpoints synced to $GDRIVE/ckpt."
