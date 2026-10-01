#!/bin/bash
# rl5 gate: (1) the new binary with every option at its default plays v63.6_rl bank-exact vs the old binary
# (target-rc636), (2) each rl5 option row plays full games (vs v63.1) and differs from its base row somewhere.
set -e
cd /home/ec2-user/krl
X=/home/ec2-user/krl/.local/ext
NEW=/home/ec2-user/krl/target-ppo3/release/agent-stdio
OLD=/home/ec2-user/krl/target-rc636/release/agent-stdio
A="--base base --profiles profiles.json --policy policy.bin --shield shield.json --disguise --chain-off r127,sm,r95 --shell shell.json"
V631="v63.1=$X/v631b_stage::/home/ec2-user/krl/target-v62/release/agent-stdio --base base --profiles profiles.json --profile 100"
OUT=data/rl5_verify; mkdir -p $OUT
M=target-ppo3/release/stdio-match
$M --agent "new=$X/cand_v636::$NEW $A" --agent "old=$X/cand_v636::$OLD $A" --agent "$V631" --pairs new:v63.1,old:v63.1 \
   --bank data/worlds/w64_bank.json --split heldout --per-world 1 --threads 12 --out $OUT/eq.tsv
python3 - <<'PY'
rows = {}
for l in open("data/rl5_verify/eq.tsv"):
    p = l.split("\t"); a = p[0].split(":")[0]; rows.setdefault(a, {})[(p[1], p[3])] = (p[4], p[5])
n, o = rows["new"], rows["old"]
diff = [k for k in n if n[k] != o.get(k)]
print(f"[rl5] equivalence: {len(n)} games, {len(diff)} bank differences" + (" -> FAIL" if diff else " -> PASS"))
PY
P=$(pwd)/configs/profiles/rl5.json
for r in 35 34 53 54 55 56 57 58 59; do
  $M --agent "r$r=$X/cand_v636::$NEW --base base --profiles $P --profile $r --shield shield.json" --agent "$V631" --pairs r$r:v63.1 \
     --bank data/worlds/w64_bank.json --split heldout --per-world 1 --threads 12 --out $OUT/row$r.tsv 2>&1 | tail -1
done
python3 - <<'PY'
import glob
def load(r):
    d = {}
    for l in open(f"data/rl5_verify/row{r}.tsv"):
        p = l.split("\t"); d[(p[1], p[3])] = (float(p[4]), float(p[5]))
    return d
base = {35: load(35), 34: load(34)}
for r, b in [(53, 35), (54, 34), (55, 35), (56, 35), (57, 35), (58, 35), (59, 35)]:
    d = load(r)
    ch = sum(d[k] != base[b].get(k) for k in d)
    w = sum((v[0] > v[1]) if k[1] == "0" else (v[1] > v[0]) for k, v in d.items())
    print(f"[rl5] row {r} (base {b}): {len(d)} games, {ch} differ from base")
PY
