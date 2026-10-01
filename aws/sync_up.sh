#!/usr/bin/env bash
# Laptop -> AWS box: stream the RL workspace over SSH (tar; no rsync on the laptop).
#   KRL_SSH_OPTS="-i ~/.ssh/KEY.pem" (optional, all aws/*.sh): the key for HOST
#   bash aws/sync_up.sh ubuntu@HOST code     # code + configs + ops spec (seconds; repeat after code changes)
#   bash aws/sync_up.sh ubuntu@HOST train    # what training needs: obs, caches, leagues, weights, index, ledgers (~2.5 GB)
#   bash aws/sync_up.sh ubuntu@HOST slim     # the slim corpus (~12 GB; needed for the delta index / re-rank)
#   bash aws/sync_up.sh ubuntu@HOST repo     # main-repo harness for the public-panel gate: src/, field agents, v61/v61.1, gate records, worlds, rustengine source
#   bash aws/sync_up.sh ubuntu@HOST all
# Remote layout mirrors the laptop: ~/kaggriculture/{src,agents,data/winplan,...} and ~/kaggriculture/kaggriculture-rl
# (~/krl is a link to it). Existing remote files are overwritten by the laptop copy,
# so run `train`/`slim` once at setup, and `code` whenever code changes. Build outputs are never sent.
set -euo pipefail
HOST=${1:?usage: sync_up.sh user@host code|train|slim|all}
WHAT=${2:-code}
cd "$(dirname "$0")/.."
send() {
  echo "[sync] $* -> $HOST:~/kaggriculture"
  # unpack into a staging folder, then rsync swaps each file in atomically (temp file + rename): a job
  # running on the box never reads a half-written file (a BC gate once read a half-synced routes.json)
  tar cf - "$@" | ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'mkdir -p ~/kaggriculture && ln -sfn ~/kaggriculture ~/krl && rm -rf ~/.krl-stage && mkdir -p ~/.krl-stage && tar xf - -C ~/.krl-stage && rsync -a ~/.krl-stage/ ~/kaggriculture/kaggriculture-rl/ && rm -rf ~/.krl-stage'
}
send_repo() {
  echo "[sync] main-repo harness -> $HOST:~/kaggriculture"
  (cd .. && tar cf - --exclude=__pycache__ src agents/v61_bandit.py agents/v61.1_bandit.py data/winplan/field data/worlds       $(ls data/winplan/gates/*v611*.jsonl 2>/dev/null) data/winplan/ladder_config.json rustengine/Cargo.toml rustengine/Cargo.lock rustengine/src pyproject.toml)     | ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'mkdir -p ~/kaggriculture && tar xf - -C ~/kaggriculture'
}
code=(Cargo.toml Cargo.lock crates configs ops/queue.py ops/tasks.json ops/checks ops/dash python kaggle aws docs harness)
# ops/state.json is NEVER sent: it is the box queue's live state. 27 Sep 17:03 IST a train sync reset it and the runner
# re-ran finished jobs (tape tree rebuilt, CMA-ES crashed at gen 12, a superseded PPO3 job re-armed).
train=(data/builds/rc_base data/obs data/train data/leagues weights data/slim/s1/index data/slim/s1/ledger data/slim/s1/runs data/kaggle_out/delta_state.json data/kaggle_out/league_state.json data/tapes)
case "$WHAT" in
  code)  send "${code[@]}" ;;
  train) send $(ls -d "${train[@]}" 2>/dev/null) ;;
  repo)  send_repo ;;
  slim)  send data/slim/s1/source=official data/slim/s1/source=gm data/slim/s1/source=league ;;
  all)   send "${code[@]}"; send_repo; send $(ls -d "${train[@]}" 2>/dev/null); send data/slim/s1/source=official data/slim/s1/source=gm data/slim/s1/source=league ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
echo "[sync] done"
