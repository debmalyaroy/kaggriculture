#!/usr/bin/env bash
# Gate: the live agent's dayobs vectors == the corpus day_obs vectors on real games.
# Makes tapes from 60 official + 60 GM slim episodes, replays them exactly (tapeplay --verify),
# dumps both seats' vectors through the agent's dayview path, extracts the same episodes with
# corpus-extract --mode obs, and requires 0 mismatched cells. Exit 1 on any failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=data/ops/obs_equiv
rm -rf "$W"; mkdir -p "$W/tapes"
off=$(ls -d data/slim/s1/source=official/date=* | tail -1)
gm=$(ls -d data/slim/s1/source=gm/date=* | sed -n '25p')
"${KRL_PY:-python}" python/slim_to_tape.py "$off" --out "$W/tapes" --n 60
"${KRL_PY:-python}" python/slim_to_tape.py "$gm" --out "$W/tapes" --n 120
"${KRL_BIN:-target-dev/release}/tapeplay${KRL_EXE-.exe}" --tapes "$W/tapes" --verify --obs-dump "$W/live.tsv" > /dev/null 2> "$W/tapeplay.err"
tail -1 "$W/tapeplay.err"
grep -q "exact-bank reproductions \([0-9]*\)/\1" "$W/tapeplay.err" || { echo "FAIL: replays not exact"; exit 1; }
ids=$(cut -f1 "$W/live.tsv" | sort -u | paste -sd,)
"${KRL_BIN:-target-dev/release}/corpus-extract${KRL_EXE-.exe}" --mode obs --slim "$off" --slim "$gm" --out "$W/corpus" --threads 4 --only "$ids" 2>&1 | grep -E "done:|obs-check"
"${KRL_PY:-python}" - "$W" <<'EOF'
import glob, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
w = sys.argv[1]
live = pd.read_csv(f"{w}/live.tsv", sep="\t", header=None)
NV = live.shape[1] - 3  # the dayobs vector length (crates/dayobs N)
live.columns = ["episode_id", "seat", "day"] + [f"v{i}" for i in range(NV)]
t = pd.concat([pq.ParquetFile(f).read().to_pandas() for f in glob.glob(f"{w}/corpus/day_obs/**/*.parquet", recursive=True)])
oc = [c for c in t.columns if c.startswith("o_")]
m = live.merge(t[["episode_id", "seat", "day"] + oc], on=["episode_id", "seat", "day"])
L, C = m[[f"v{i}" for i in range(NV)]].to_numpy(), m[oc].to_numpy()
bad = int((np.abs(L - C) > 1e-4 * (1 + np.abs(C))).sum())
print(f"obs_equiv: live rows {len(live)}, corpus rows {len(t)}, matched {len(m)}, cells {L.size}, mismatched {bad}")
sys.exit(0 if bad == 0 and len(m) == len(live) == len(t) and len(m) > 0 else 1)
EOF
