#!/usr/bin/env bash
# Exactness suite ON THE BOX. Training starts only if every check passes on this CPU / OS:
#   1. engine parity vs real ladder games (slim-check: both farms' money after EVERY step, exact)
#   2. exact replay of 180 real games + live-agent dayobs == corpus dayobs (ops/checks/obs_equiv.sh)
#   3. Rust policy forward == torch (python/learn/export_check.py)
#   4. v61.1 closed loop: Rust agent self-play banks on fixed seeds == the laptop's recorded banks
#   5. PPO smoke (2 short iterations in a scratch root), only if a BC teacher exists
# Then: systemctl enable --now krl-queue.  bash ~/krl/aws/verify.sh [--no-start]
set -euo pipefail
cd ~/krl
source aws/env.sh
R=data/ops/verify; mkdir -p "$R"
pass() { echo "PASS $*" | tee -a "$R/report.txt"; }
fail() { echo "FAIL $*" | tee -a "$R/report.txt"; exit 1; }
: > "$R/report.txt"
echo "[verify] $(uname -m) $(nproc) cpus $(date -u +%FT%TZ)" | tee -a "$R/report.txt"

# 1. engine parity on real games (one official day + one GM day)
for d in $(ls -d data/slim/s1/source=official/date=* | tail -1) $(ls -d data/slim/s1/source=gm/date=* | sed -n '20p'); do
  "$KRL_BIN/slim-check" "$d" --max 300 > "$R/slim-check.$(basename "$d").txt" 2>&1 || true
  sum=$(grep "^slim-check:" "$R/slim-check.$(basename "$d").txt" | tail -1)
  echo "$sum" | grep -q " 0 mismatched$" && ! grep -q "^MISMATCH" "$R/slim-check.$(basename "$d").txt" \
    || fail "engine parity $d: ${sum:-no summary} (see $R/slim-check.$(basename "$d").txt)"
  pass "engine parity $d: $sum"
done

# 2. exact replays + observation equivalence
bash ops/checks/obs_equiv.sh > "$R/obs_equiv.txt" 2>&1 || fail "obs_equiv (see $R/obs_equiv.txt)"
pass "$(grep obs_equiv: "$R/obs_equiv.txt")"

# 3. export equivalence
"$KRL_PY" python/learn/export_check.py > "$R/export.txt" 2>&1 || fail "export_check (see $R/export.txt)"
pass "$(grep policy-check "$R/export.txt")"

# 4. closed-loop banks vs the laptop reference (aws/ref_selfplay.tsv, written on the laptop)
if [ -f aws/ref_selfplay.tsv ]; then
  seeds=$(cut -f1 aws/ref_selfplay.tsv | paste -sd,)
  "$KRL_BIN/selfplay" --seeds "$seeds" --threads 8 2>/dev/null | sort -n > "$R/selfplay.tsv"
  diff <(cut -f1-3 aws/ref_selfplay.tsv | sort -n) <(cut -f1-3 "$R/selfplay.tsv") > "$R/selfplay.diff" \
    || fail "closed-loop banks differ from the laptop (see $R/selfplay.diff)"
  pass "closed-loop v61.1 self-play banks identical to the laptop on $(wc -l < aws/ref_selfplay.tsv) seeds"
  # 4b. profile 0 through the profile table must BE v61.1 (the neutral action every paired test and the
  #     policy's anchor assume; a mis-indexed "neutral" once faked a public PPO win)
  "$KRL_BIN/selfplay" --seeds "$seeds" --threads 8 --profiles configs/profiles/rl3.json --pa 0 --pb 0 2>/dev/null | sort -n > "$R/profile0.tsv"
  diff <(cut -f1-3 aws/ref_selfplay.tsv | sort -n) <(cut -f1-3 "$R/profile0.tsv") > "$R/profile0.diff" \
    || fail "profile 0 is not bank-identical to v61.1 (see $R/profile0.diff)"
  pass "profile 0 bank-identical to v61.1 on $(wc -l < aws/ref_selfplay.tsv) seeds"
fi

# 5. PPO smoke when a teacher exists
if [ -f weights/bc/LATEST ]; then
  rm -rf data/ops/ppo_smoke
  "$KRL_PY" python/learn/ppo.py --root data/ops/ppo_smoke --games 48 --iters 2 --val-every 2 --val-n 24 --threads "$(nproc)" --new > "$R/ppo_smoke.txt" 2>&1 \
    || fail "PPO smoke (see $R/ppo_smoke.txt)"
  pass "PPO smoke: $(grep 'iter 2' "$R/ppo_smoke.txt" | tail -1)"
fi

# speed: games per second on this box (v61.1 vs v61.1, all cores)
"$KRL_BIN/selfplay" --n $(( $(nproc) * 10 )) --seed0 900000 --threads "$(nproc)" > /dev/null 2> "$R/speed.txt"  # 10 games per core: one game per core measures the slowest game, not throughput
pass "speed: $(tail -1 "$R/speed.txt")"

echo "[verify] ALL PASS" | tee -a "$R/report.txt"
if [ "${1:-}" != "--no-start" ]; then
  sudo systemctl enable --now krl-queue
  echo "[verify] queue started: systemctl status krl-queue; python ops/queue.py status"
fi
