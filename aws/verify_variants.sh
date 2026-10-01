#!/bin/bash
# One-off verification of the fixed v63 variants (26 Sep): 600-game h2h vs v63, the 8-opponent panel paired
# against v63 (same seeds), then the band gate on real players' tapes. Runs as queue job Q21 in the gate lane.
#   bash aws/verify_variants.sh THREADS
cd "$(dirname "$0")/.."
TH=${1:-16}
T=configs/profiles/cand_v63var.json; O=data/verify; R=target-dev/release/ppo-rollout
PY=${PY:-.venv/bin/python}
mkdir -p $O
for K in 49; do
  echo "h2h k$K $($R --learner-fixed $K --opp fixed:35 --profiles $T --games 600 --seed0 96000000 --threads $TH --greedy --out $O/h2h_$K 2>&1 | tail -1)"
done
for OP in 0 2 12 13 19 22 30 31; do for K in 35 51 47 46 49; do
  echo "panel op$OP k$K $($R --learner-fixed $K --opp fixed:$OP --profiles $T --games 150 --seed0 97000000 --threads $TH --greedy --out $O/p${OP}_$K 2>&1 | tail -1)"
done; done
for K in 51 47 46; do
  echo "band k$K"; $PY python/band_gate.py --cand-profile $K --name fixed-$K --profiles $T --ref-profile 35 --threads $TH 2>&1 | tail -4
done
echo VERIFY_DONE
