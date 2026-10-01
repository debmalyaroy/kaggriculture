#!/bin/bash
# v63.7_rl vs the v64 bandit flagship vs v63.6_rl, round robin incl. own clones, 64 worlds x 3 held-out seeds x both seats.
# Native box builds of each tarball's source (the box is aarch64): target-rc (ours), target-v64 (main-repo agent, profile 105).
cd /home/ec2-user/krl
X=/home/ec2-user/krl/.local/ext
A="--base base --profiles profiles.json --policy policy.bin --shield shield.json --disguise --chain-off r127,sm,r95 --shell shell.json"
exec nice target-rc/release/stdio-match \
  --agent "v63.7_rl=$X/cand_v637::$X/cand_v637/agent-stdio $A --rshell rshell/rshell.json --knob-over knobs.json --endg endg.json" \
  --agent "v64=$X/v64_stage::$X/v64_stage/agent-stdio --base base --profiles profiles.json --profile 105" \
  --agent "v63.6_rl=$X/cand_v636::$X/cand_v636/agent-stdio $A" \
  --bank data/worlds/w64_bank.json --split ${SPLIT:-heldout} --per-world ${PER:-3} --threads ${T:-20} --out ${OUT:-data/rshell/tourn_v637.tsv}
