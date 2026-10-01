#!/usr/bin/env bash
# Laptop <- AWS box: fetch the candidates the box handed off (python/learn/rc.py hand_off) into
# data/candidates/ here, then mark them pulled on the box. Kilobytes: policy.bin + report.json each.
#   KRL_SSH_OPTS="-i ~/.ssh/KEY.pem" bash aws/pull_rc.sh ec2-user@HOST [ec2-user@HOST2 ...]
# Next, per new candidate (laptop): Rust==torch on x86, the x86 tarball (scripts/build_submission.ps1),
# the full band-scored tournament and the official-engine spot check. Nothing is submitted.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: pull_rc.sh user@host [user@host ...]"; exit 2; }
mkdir -p data/candidates
for HOST in "$@"; do
  echo "[pull_rc] $HOST"
  if ! ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'test -f ~/krl/data/candidates/RC_PENDING.json'; then
    echo "[pull_rc] $HOST: no candidates yet"
    continue
  fi
  ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'cd ~/krl/data/candidates && tar cf - --exclude=RC_PENDING.json .' | tar xf - -C data/candidates
  ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'cat ~/krl/data/candidates/RC_PENDING.json' > "data/candidates/.pending.$$.json"
  # merge the box's index into the laptop's (one entry per candidate name), then mark them pulled there
  "${KRL_PY:-python}" - "data/candidates/.pending.$$.json" <<'EOF'
import json, os, sys
new = json.load(open(sys.argv[1]))
idx = "data/candidates/RC_PENDING.json"
have = {c["name"]: c for c in (json.load(open(idx)) if os.path.exists(idx) else [])}
fresh = [c["name"] for c in new if c["name"] not in have]
for c in new:
    have.setdefault(c["name"], dict(c, pulled=True, laptop_status="pulled"))
json.dump(sorted(have.values(), key=lambda c: c["t"]), open(idx, "w"), indent=1)
os.remove(sys.argv[1])
print(f"[pull_rc] {len(new)} on the box, {len(fresh)} new: {fresh}")
EOF
  ssh ${KRL_SSH_OPTS:-} -o ServerAliveInterval=30 "$HOST" 'cd ~/krl && python3.11 -c "
import json; p=\"data/candidates/RC_PENDING.json\"; c=json.load(open(p))
[x.update(pulled=True) for x in c]; json.dump(c, open(p, \"w\"), indent=1)"'
done
