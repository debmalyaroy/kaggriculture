#!/usr/bin/env bash
# One-time setup ON THE AWS BOX (Ubuntu 22.04/24.04 or Amazon Linux 2023; ARM64 or x86_64).
#   bash ~/krl/aws/bootstrap.sh
# Installs build tools + Rust + a Python venv (numpy, pandas, pyarrow, psutil, kaggle, torch CPU),
# builds every binary natively (release), checks the Kaggle token, installs the queue as a systemd
# service (not started: aws/verify.sh starts it only after the exactness suite passes), and schedules
# the automatic shutdown before the lock.
set -euo pipefail
ln -sfn ~/kaggriculture ~/krl
cd ~/krl
if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential pkg-config git curl rsync python3 python3-venv python3-pip tmux htop
elif command -v dnf >/dev/null; then
  sudo dnf install -y gcc gcc-c++ make pkgconfig git tar gzip rsync python3.11 python3.11-pip tmux htop
  sudo alternatives --set python3 /usr/bin/python3.11 2>/dev/null || true
fi
PYBIN=$(command -v python3.11 || command -v python3)
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
"$PYBIN" -m venv ~/krl/.venv
~/krl/.venv/bin/pip install -q --upgrade pip
~/krl/.venv/bin/pip install -q numpy pandas pyarrow psutil kaggle==2.2.4   # same CLI as the laptop
~/krl/.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
# environment for every shell, the queue and its jobs
cat > ~/krl/aws/env.sh <<EOF
export KRL_PY=$HOME/krl/.venv/bin/python
export KRL_BIN=$HOME/krl/target-dev/release
export CARGO_TARGET_DIR=$HOME/krl/target-dev
export RUSTFLAGS="-C target-cpu=native"   # every cargo build (queue checks too) matches the verified native build
export KRL_PPO_GAMES=${KRL_PPO_GAMES:-8192}   # per PPO iteration (measured 2026-09-26: the update is ~9 s per 4k games; games dominate)
export KRL_RETRY_S=${KRL_RETRY_S:-120}   # a failed/lost queue job is retried (resuming from disk) this long after it ended
export KRL_BOX=${KRL_BOX:-A}   # box name: prefixes this box's candidates (data/candidates/<box>-<name>)
export KAGG_BIN=$HOME/krl/harness/rustengine/target/release/kagg
export PATH=$HOME/krl/.venv/bin:$HOME/.cargo/bin:\$PATH
EOF
grep -q krl/aws/env.sh ~/.bashrc || echo "source ~/krl/aws/env.sh" >> ~/.bashrc
source ~/krl/aws/env.sh
# native release build of everything the pipeline runs
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  -p corpus -p runner -p policy -p tools -p agent 2>&1 | tail -3
# the serve engine for the public-panel gate (Python agents on the Rust engine), from the harness
# copied into kaggriculture-rl/harness (the RL track never reads the root repo)
if [ -d ~/krl/harness/rustengine/src ]; then
  (cd ~/krl/harness/rustengine && CARGO_TARGET_DIR=$HOME/krl/harness/rustengine/target RUSTFLAGS="-C target-cpu=native" cargo build --release --bin kagg 2>&1 | tail -2)
  ls -l "$KAGG_BIN"
  ~/krl/.venv/bin/pip install -q -e ~/krl/harness --no-deps 2>&1 | tail -1 || true
else
  echo "[bootstrap] WARNING: harness/ not synced (bash aws/sync_up.sh HOST code); the panel gate needs it"
fi
ls -l "$KRL_BIN" | grep -E "corpus-extract|league-rand|ppo-rollout|tapeplay|selfplay|policy-check|slim-check|agent-stdio" || true
# Kaggle token (copied by the operator): ~/.kaggle/access_token (reads private notebook outputs;
# the legacy kaggle.json key cannot) or kaggle.json
if [ -f ~/.kaggle/access_token ] || [ -f ~/.kaggle/kaggle.json ]; then
  chmod 600 ~/.kaggle/* 2>/dev/null || true
  ~/krl/.venv/bin/kaggle kernels list --mine --page-size 1 >/dev/null && echo "[bootstrap] kaggle CLI ok"
else
  echo "[bootstrap] WARNING: no ~/.kaggle/access_token (delta and corpus pull need it)"
fi
# the queue as a service: restarts on failure, survives reboots / spot stop-start
sudo tee /etc/systemd/system/krl-queue.service >/dev/null <<EOF
[Unit]
Description=kaggriculture-rl task queue runner
After=network-online.target
[Service]
User=$USER
WorkingDirectory=$HOME/krl
ExecStart=/bin/bash -lc 'source $HOME/krl/aws/env.sh && exec \$KRL_PY ops/queue.py run >> data/ops/runner.log 2>&1'
Restart=always
RestartSec=30
# jobs are detached and must outlive a runner restart (the queue reconciles them from their exit files)
KillMode=process
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
mkdir -p ~/krl/data/ops
bash ~/krl/aws/shutdown_at.sh "2026-09-29 22:00"
echo "[bootstrap] done. Next: bash ~/krl/aws/verify.sh  (starts the queue only if every check passes)"
