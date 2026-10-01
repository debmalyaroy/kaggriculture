"""Laptop leg for candidates the box handed off (docs/ARCHITECTURE.md s5.3). Never submits.

    KRL_SSH_OPTS="-i ~/.ssh/KEY.pem" bash aws/pull_rc.sh ec2-user@HOST     # fetch data/candidates/
    python python/rc_laptop.py [--name A-rc_...] [--workers 12]           # test every untested candidate

Per candidate (data/candidates/<name>/policy.bin):
  1. Rust == torch on x86: crates/policy `policy-check` on the candidate's weights (the same exactness
     the box proved on ARM; the deployed agent is the pure-Rust forward pass).
  2. The band gate (python/band_gate.py): real ladder players by rating band, one world per match, paired
     vs v62.1.
  3. The full public panel (python/panel_gate.py --k 99): every field agent plus v61, v61.1, v62 and
     v62.1, 24 worlds, both seats, on the faithful harness copied into harness/.
Writes data/candidates/<name>/laptop.json and updates the index entry (laptop_status, verdict).
Then the operator runs the x86 build: `.\\scripts\\build_submission.ps1 -Policy data\\candidates\\<name>\\policy.bin`
and the official-engine spot check, and decides. Nothing here uploads or submits.
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDS = os.path.join(RL, "data", "candidates")
GATES = os.path.join(RL, "data", "gates")
PY = sys.executable


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(GATES, pattern)))
    return json.load(open(fs[-1])) if fs else None


def run(cmd):
    print("[rc_laptop] $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=RL)
    return r.returncode


def test(name, workers):
    d = os.path.join(CANDS, name)
    w = os.path.join(d, "policy.bin")
    out = {"name": name, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    rc = run([PY, os.path.join(RL, "python", "learn", "export_check.py"), "--weights", d]) if os.path.exists(os.path.join(d, "ckpt.pt")) else None
    out["x86_forward_equal"] = None if rc is None else rc == 0
    run([PY, os.path.join(RL, "python", "band_gate.py"), "--cand", w, "--name", name, "--threads", str(workers)])
    band = latest(f"band__{name}__*.json")
    run([PY, os.path.join(RL, "python", "panel_gate.py"), "--cand", w, "--name", f"{name}-full", "--k", "99", "--worlds", "24",
         "--per-world", "1", "--workers", str(workers)])
    panel = latest(f"panel__{name}-full__*.json")
    out["band"] = {k: band.get(k) for k in ("losses_below_2500", "win_rate_2500plus", "ref_losses_below_2500", "ref_win_rate_2500plus",
                                            "paired", "pass", "no_worse_than_ref")} if band else None
    out["panel_checks"] = panel.get("checks") if panel else None
    ok_band = bool(band and (band["pass"] or band["no_worse_than_ref"]))
    ok_panel = bool(panel and all(panel.get("checks", {}).values()))
    out["verdict"] = "READY FOR OPERATOR" if ok_band and ok_panel and out["x86_forward_equal"] is not False else "NOT READY"
    json.dump(out, open(os.path.join(d, "laptop.json"), "w"), indent=1)
    idx = os.path.join(CANDS, "RC_PENDING.json")
    if os.path.exists(idx):
        pend = json.load(open(idx))
        for c in pend:
            if c["name"] == name:
                c.update(laptop_status="tested", verdict=out["verdict"])
        json.dump(pend, open(idx, "w"), indent=1)
    print(f"[rc_laptop] {name}: {out['verdict']}; band {out['band']}; panel {out['panel_checks']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    idx = os.path.join(CANDS, "RC_PENDING.json")
    names = [a.name] if a.name else [c["name"] for c in (json.load(open(idx)) if os.path.exists(idx) else []) if c.get("laptop_status") != "tested"]
    if not names:
        print("[rc_laptop] no untested candidates (run aws/pull_rc.sh first)")
        return
    for n in names:
        test(n, a.workers)


if __name__ == "__main__":
    main()
