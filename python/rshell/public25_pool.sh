#!/bin/bash
# Public-25 tapes in PPO's training pool (run on the box after any train/ rebuild): every v63.5_rl loss x3 + the 600 wins.
cd "$(dirname "$0")/../.." || exit 1
D=data/tapes/train/public25
mkdir -p $D
for f in data/rshell/public25/rca/tapes/*.json; do b=$(basename $f .json); for k in 0 1 2; do ln -sf $(readlink -f $f) $D/${b}__p$k.json; done; done
for f in data/rshell/public25/wins/tapes/*.json; do ln -sf $(readlink -f $f) $D/$(basename $f); done
echo "[public25-pool] $(ls $D | wc -l) links in $D"
