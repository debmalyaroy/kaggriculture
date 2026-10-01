"""Daily midday rehab: full-budget arm retrain + re-judge, policy re-judge.

Operator decision 2026-08-14: arms and the policy head stay guard-blocked on
negative evidence, but the evidence must be allowed to CHANGE. This job runs
every midday and produces fresh, generation-tagged evidence; the gates
(train_gates.arms_allowed / policy_allowed, newest-generation scoped) decide
by themselves. If a generation measures positive, the next daily build ships
it automatically -- no human in the loop beyond the standing submit rule.

ARMS: retrain the top loss families at FULL budget (the guard's fast 2x4
profile is exactly the circularity that kept arms bad) against the newest
field tapes, build a --lab-force-arms pair, judge >= 24 paired games via
commit_judge, retrain gates.

POLICY: only when models/lab/policy_bc.json CHANGED since the last judgment
(re-proving the same negative daily would waste an hour); build a
--lab-force-policy twin and judge via judge_policy.

Never submits. Everything is logged; failures degrade to "no new evidence".

    python src/experiments/daily_rehab.py
"""
from kaggriculture.paths import ROOT
import datetime as dt
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

PY = sys.executable
JUDGE_DIR = os.path.join(ROOT, ".local", "rehab")
POLICY_HASH = os.path.join(ROOT, "models", "lab", "policy_judged.hash")
ARM_GENS, ARM_POP = "8", "12"          # the full budget the guard denies


def run(cmd, timeout=7200):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, timeout=timeout, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    tail = ((p.stdout or "") + (p.stderr or ""))[-1500:]
    print(tail, flush=True)
    return p.returncode, tail


def newest_tapes_dir():
    days = sorted(glob.glob(os.path.join(ROOT, "data", "refresh", "*",
                                         "loss_tapes")))
    return days[-1] if days else None


def tape_families(tape_dir):
    """{family: [tape paths]} from the tapes' own docstring labels; HOLDOUT
    and unknown excluded (the reserve never trains anything)."""
    fams = {}
    for p in sorted(glob.glob(os.path.join(tape_dir, "*.py"))):
        head = open(p, encoding="utf-8", errors="replace").read(300)
        m = re.search(r'(\S[^\n]*?)\s+(?:beat us|lost to us)', head)
        fam = re.sub(r"^.*?opponent:\s*", "", m.group(1)).strip() if m else "unknown"
        if fam in ("HOLDOUT", "unknown", "FIELD"):
            continue
        fams.setdefault(fam, []).append(p)
    return fams


def incumbent_base():
    import kaggriculture.pipeline.refresh_cycle as RC
    newest = max(glob.glob(os.path.join(ROOT, "agents", "v*_route.py")),
                 key=os.path.getmtime)
    return RC.incumbent_base_id(newest), newest


def active_ref():
    p = subprocess.run(["kaggle", "competitions", "submissions",
                        "kaggriculture"], capture_output=True, text=True,
                       timeout=300)
    for line in (p.stdout or "").splitlines():
        m = re.match(r"\s*(\d{6,})\s", line)
        if m and "COMPLETE" in line:
            return m.group(1)
    return None


def rehab_arms(base_id, tag):
    tape_dir = newest_tapes_dir()
    if not tape_dir:
        print("no tapes -- arms rehab skipped")
        return False
    fams = tape_families(tape_dir)
    targets = sorted(fams.items(), key=lambda kv: -len(kv[1]))[:2]
    if not targets:
        print("no non-holdout families -- arms rehab skipped")
        return False
    guard = os.path.join(ROOT, "data", "panel", "opp_r004_90558188_s0.py")
    arms = {}
    for i, (fam, tapes) in enumerate(targets):
        slug = re.sub(r"[^a-z0-9]+", "", fam.lower())[:12] or f"f{i}"
        arm = f"rehab{i}_{slug}"
        code, _ = run([PY, "src/kaggriculture/train/train_arms.py", "--arm", arm,
                       "--base", base_id, "--targets", *tapes[:3],
                       "--guards", guard, "--gens", ARM_GENS,
                       "--pop", ARM_POP, "--seeds", "2", "--sigma", "5"],
                      timeout=7200)
        if code == 0 and os.path.exists(os.path.join(
                ROOT, "models", "v22", "arms", arm, "best_route.json")):
            arms[arm] = fam
    if not arms:
        print("no arms trained -- rehab evidence unchanged")
        return False

    os.makedirs(JUDGE_DIR, exist_ok=True)
    on = os.path.join(JUDGE_DIR, f"arms_on_{tag}.py")
    off = os.path.join(JUDGE_DIR, f"arms_off_{tag}.py")
    import kaggriculture.agentbuild.v22_agent as v22_agent
    for out, force in ((on, True), (off, False)):
        v22_agent.TEAM2ARM = {fam: arm for arm, fam in arms.items()}
        argv = ["v22_agent", "--base", base_id,
                "--arms", *arms, "--out",
                os.path.relpath(out, ROOT), "--gru-auto", "--duel-policy"]
        if force:
            argv.append("--lab-force-arms")
        old, sys.argv = sys.argv, argv
        try:
            v22_agent.main()
        finally:
            sys.argv = old

    ref = active_ref()
    if not ref:
        print("no active ref for opponent sampling -- arms judging skipped")
        return False
    code, _ = run([PY, "src/kaggriculture/experiments/commit_judge.py",
                   "--bandit", on, "--control", off,
                   "--submission", ref, "--games", "24", "--seeds", "1",
                   "--source-label", f"arms-rehab-{tag}"], timeout=10800)
    run([PY, "src/kaggriculture/train/train_gates.py"])
    return code == 0


def rehab_policy(base_id, tag):
    pp = os.path.join(ROOT, "models", "lab", "policy_bc.json")
    if not os.path.exists(pp):
        print("no policy export -- policy rehab skipped")
        return False
    h = hashlib.sha256(open(pp, "rb").read()).hexdigest()
    prev = (open(POLICY_HASH).read().strip()
            if os.path.exists(POLICY_HASH) else "")
    if h == prev:
        print("policy export unchanged since last judgment -- skipped "
              "(retrain it to earn a new trial)")
        return False
    os.makedirs(JUDGE_DIR, exist_ok=True)
    on = os.path.join(JUDGE_DIR, f"policy_on_{tag}.py")
    off = os.path.join(JUDGE_DIR, f"policy_off_{tag}.py")
    import kaggriculture.agentbuild.v22_agent as v22_agent
    for out, force in ((on, True), (off, False)):
        argv = ["v22_agent", "--base", base_id, "--arms",
                "--out", os.path.relpath(out, ROOT),
                "--gru-auto", "--duel-policy"]
        if force:
            argv += ["--policy", "--lab-force-policy"]
        old, sys.argv = sys.argv, argv
        try:
            v22_agent.main()
        finally:
            sys.argv = old
    code, _ = run([PY, "src/kaggriculture/experiments/judge_policy.py", "--on", on,
                   "--off", off, "--label", f"policy-rehab-{tag}"],
                  timeout=10800)
    if code == 0:
        open(POLICY_HASH, "w").write(h)
    return code == 0


def main():
    tag = dt.date.today().isoformat()
    base_id, newest_route = incumbent_base()
    if not base_id:
        raise SystemExit("could not recover the incumbent base id")
    print(f"rehab {tag}: base {base_id} (from {os.path.basename(newest_route)})")

    rehab_arms(base_id, tag)
    rehab_policy(base_id, tag)

    import kaggriculture.train.train_gates as TG
    a_ok, a_why = TG.arms_allowed()
    p_ok, p_why = TG.policy_allowed()
    print(f"\nVERDICTS after rehab:")
    print(f"  arms_allowed:   {a_ok} -- {a_why}")
    print(f"  policy_allowed: {p_ok} -- {p_why}")
    print("(a True here ships in the NEXT daily build automatically; "
          "this job never submits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
