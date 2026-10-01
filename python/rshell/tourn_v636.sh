#!/bin/bash
# v63.6_rl (i910) vs v63.5_rl vs live v63.1 (+ v63.6 binary with i790 = equivalence check), round robin incl. own
# clones, 64 worlds x 3 held-out seeds x both seats. Each tarball plays its OWN shipped agent-stdio (musl build).
cd /home/ec2-user/krl
X=/home/ec2-user/krl/.local/ext
A="--base base --profiles profiles.json --policy policy.bin --shield shield.json --disguise --chain-off r127,sm,r95 --shell shell.json"
exec nice target-v2/release/stdio-match \
  --agent "v63.6_rl=$X/cand_v636::$X/cand_v636/agent-stdio $A" \
  --agent "v63.5_rl=$X/cand_v635::$X/cand_v635/agent-stdio $A" \
  --agent "v63.6eq=$X/cand_v636eq::$X/cand_v636eq/agent-stdio $A" \
  --agent "v63.1=$X/v631b_stage::/home/ec2-user/krl/target-v62/release/agent-stdio --base base --profiles profiles.json --profile 100" \
  --bank data/worlds/w64_bank.json --split ${SPLIT:-heldout} --per-world ${PER:-3} --threads ${T:-16} --out ${OUT:-data/rshell/tourn_v636.tsv}
