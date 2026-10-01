#!/bin/bash
# Gate: target-v3 (shell v3 code, no buy net) must play v63.7_rl bank-exact vs target-rc.
cd /home/ec2-user/krl
X=/home/ec2-user/krl/.local/ext
A="--base base --profiles profiles.json --policy policy.bin --shield shield.json --disguise --chain-off r127,sm,r95 --shell shell.json --rshell rshell/rshell.json --knob-over knobs.json --endg endg.json"
target-v3/release/stdio-match --agent "new=$X/cand_v637::/home/ec2-user/krl/target-v3/release/agent-stdio $A" --agent "old=$X/cand_v637::/home/ec2-user/krl/target-rc/release/agent-stdio $A" \
  --agent "v636=$X/cand_v636::$X/cand_v636/agent-stdio --base base --profiles profiles.json --policy policy.bin --shield shield.json --disguise --chain-off r127,sm,r95 --shell shell.json" \
  --pairs new:v636,old:v636 --bank data/worlds/w64_bank.json --split heldout --per-world 1 --threads 16 --out data/rshell/v3_eq.tsv 2>/dev/null
python3 - <<'PY'
rows = {}
for l in open("data/rshell/v3_eq.tsv"):
    p = l.split("\t"); rows.setdefault(p[0].split(":")[0], {})[(p[1], p[3])] = (p[4], p[5])
n, o = rows["new"], rows["old"]
d = [k for k in n if n[k] != o.get(k)]
print(f"[v3-eq] {len(n)} games, {len(d)} bank differences -> {'FAIL' if d else 'PASS'}")
PY
