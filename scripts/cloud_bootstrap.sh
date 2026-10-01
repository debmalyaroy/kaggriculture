#!/usr/bin/env bash
# vast.ai VM bootstrap for trackp BC+RL: deps + Linux engine + gdrive pull.
# Run once (cloud_run.sh calls it). Idempotent.
#   GDRIVE=gdrive:trackp bash scripts/cloud_bootstrap.sh
set -euo pipefail
GDRIVE="${GDRIVE:-gdrive:trackp}"
ROOT="${ROOT:-$(pwd)}"
cd "$ROOT"

echo "[bootstrap] python deps..."
pip install -q pyarrow numpy orjson 2>/dev/null || true   # torch preinstalled on GPU image

echo "[bootstrap] rclone..."
command -v rclone >/dev/null 2>&1 || { curl -fsSL https://rclone.org/install.sh | sudo bash; }
rclone listremotes | grep -q "^${GDRIVE%%:*}:" || {
  echo "!! rclone remote '${GDRIVE%%:*}' not configured. Run: rclone config  (add a Google Drive remote)"; exit 1; }

echo "[bootstrap] pull code (if staged in gdrive:code) ..."
rclone copy "$GDRIVE/code" "$ROOT" --exclude "data/**" --exclude "ckpts/**" 2>/dev/null || \
  echo "   (skip: code assumed already present via git clone)"

echo "[bootstrap] pull corpus -> data/trackp_corpus_v2 ..."
mkdir -p "$ROOT/data/trackp_corpus_v2"
rclone copy "$GDRIVE/corpus" "$ROOT/data/trackp_corpus_v2" --transfers 8 --progress

echo "[bootstrap] pull latest checkpoints (resume) ..."
mkdir -p "$ROOT/ckpts"
rclone copy "$GDRIVE/ckpt" "$ROOT/ckpts" 2>/dev/null || true

echo "[bootstrap] build Linux engine (kagg) for RL self-play ..."
if [ ! -x "$ROOT/rustengine/target/release/kagg" ] && [ ! -x "$ROOT/rustengine/kagg" ]; then
  command -v cargo >/dev/null 2>&1 || { curl -fsSL https://sh.rustup.rs | sh -s -- -y; source "$HOME/.cargo/env"; }
  ( cd "$ROOT/rustengine" && cargo build --release --bin kagg )
fi
echo "[bootstrap] done. corpus rows:"
PYTHONPATH="src:vendor" python -c "import glob,pyarrow.parquet as pq;fs=glob.glob('data/trackp_corpus_v2/shard_*.parquet');print(sum(pq.read_metadata(f).num_rows for f in fs[:50]),'... (sampled)')" 2>/dev/null || true
