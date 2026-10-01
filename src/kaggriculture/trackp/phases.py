"""The Track P go-live roadmap as an executable phase pipeline.

Five phases, each with MEASURED exit gates (no phase is 'done' by opinion):

  0 economy     rebuild L0's first-week output.
                gate: takeover-regret@day4 >= -30k AND bank ratio vs the
                route incumbent >= 0.75
  1 search      ExIt + reverse curriculum training beats the hand rules.
                gate: PPO anchor score > 0 on the frozen anchor set
                build items: kagg LOADSTATE, exit.py (flagged until built)
  2 league      PSRO meta-Nash + per-family exploiters harden the policy.
                gate: newest-generation guard evidence >= 25 with a
                positive, sign-tested edge (guard.planner_allowed)
                build item: psro.py (flagged until built)
  3 hinge       exploit the 1.32.7 rebalance window.
                gate: engine_version == 1.32.7 (the armed watcher flips it)
  4 graduation  the 5-condition gate; operator go is manual, always.
                gate: graduation_report conditions 1-4 true

Usage:
  python src/trackp/phases.py --status          every phase, every gate
  python src/trackp/phases.py --run 0           run one phase's automation
  python src/trackp/phases.py --run all         run 0..4, stop at unmet gate
Never submits anything, ever.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys

from kaggriculture.trackp import common  # noqa: E402

PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(common.MODELS, "phase_report.json")

GATE0_REGRET = -30000.0
GATE0_RATIO = 0.75


def _run(cmd, timeout=7200):
    r = subprocess.run(cmd, cwd=common.ROOT, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, ((r.stdout or "") + (r.stderr or ""))


def _route_incumbent():
    from kaggriculture.trackp.graduation import route_incumbent
    return route_incumbent()


def _newest(path_glob):
    hits = glob.glob(path_glob)
    return max(hits, key=os.path.getmtime) if hits else None


# ------------------------------------------------------------------ gates --

def gate0() -> dict:
    """Economy: takeover regret + bank ratio, from the newest artifacts."""
    out = {"phase": 0, "name": "economy"}
    reg = _newest(os.path.join(common.MODELS, "regret", "*_s*.json"))
    if reg:
        d = json.load(open(reg, encoding="utf-8"))
        rows = {r["takeover_day"]: r["regret"]
                for r in d.get("takeover", {}).get("rows", [])}
        out["takeover_day4_regret"] = rows.get(4)
    p0 = os.path.join(common.MODELS, "phase0_arena.json")
    if os.path.exists(p0):
        d = json.load(open(p0, encoding="utf-8"))
        out["bank_ratio_vs_incumbent"] = d.get("bank_ratio")
        out["arena_score"] = d.get("score")
    ok_r = (out.get("takeover_day4_regret") is not None
            and out["takeover_day4_regret"] >= GATE0_REGRET)
    ok_b = (out.get("bank_ratio_vs_incumbent") is not None
            and out["bank_ratio_vs_incumbent"] >= GATE0_RATIO)
    out["gate"] = {"regret_ok": ok_r, "ratio_ok": ok_b,
                   "passed": bool(ok_r and ok_b),
                   "needs": f"regret@d4 >= {GATE0_REGRET:,.0f} and "
                            f"bank ratio >= {GATE0_RATIO}"}
    return out


def gate1() -> dict:
    out = {"phase": 1, "name": "search-training"}
    built = {"loadstate": "LOADSTATE" in open(
        os.path.join(common.ROOT, "rustengine", "src", "service.rs"),
        encoding="utf-8").read(),
        "exit_py": os.path.exists(os.path.join(SRC, "exit.py"))}
    out["build_items"] = built
    lp = os.path.join(common.MODELS, "ppo_log.jsonl")
    best = 0.0
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            if line.strip():
                best = max(best, json.loads(line).get("anchor_score", 0.0))
    out["best_anchor_score"] = best
    out["gate"] = {"passed": best > 0.0 and all(built.values()),
                   "needs": "anchor_score > 0 with LOADSTATE + ExIt built"}
    return out


def gate2() -> dict:
    from kaggriculture.trackp import guard
    out = {"phase": 2, "name": "league"}
    out["build_items"] = {"psro_py": os.path.exists(
        os.path.join(SRC, "psro.py"))}
    v = guard.planner_allowed(verbose=False)
    out["guard"] = {k: v.get(k) for k in ("allowed", "n", "score",
                                          "sign_p", "reason")}
    out["gate"] = {"passed": bool(v.get("allowed")),
                   "needs": ">=25 judged games, positive, sign-tested"}
    return out


def gate3() -> dict:
    out = {"phase": 3, "name": "hinge-window",
           "engine": common.engine_version()}
    out["gate"] = {"passed": common._ver_tuple(
        common.engine_version()) >= (1, 32, 7),
        "needs": "ladder flip (watcher armed, automatic)"}
    return out


def gate4() -> dict:
    out = {"phase": 4, "name": "graduation"}
    gp = os.path.join(common.MODELS, "graduation_report.json")
    if os.path.exists(gp):
        d = json.load(open(gp, encoding="utf-8"))
        out["conditions_1_to_4"] = d.get("graduated_1_to_4")
        out["age_hours"] = round((dt.datetime.now().timestamp()
                                  - os.path.getmtime(gp)) / 3600, 1)
    out["gate"] = {"passed": bool(out.get("conditions_1_to_4")),
                   "needs": "crown +10pp vs route, holdout, guard, "
                            "sim-to-real; condition 5 = operator, manual"}
    return out


GATES = [gate0, gate1, gate2, gate3, gate4]


# ------------------------------------------------------------------- runs --

def run0() -> dict:
    """Phase 0 automation: search -> rebuild -> arena vs incumbent ->
    takeover-regret probe -> gate. The STRUCTURAL executor work is
    engineering (sessions); this loop measures it and squeezes PARAMS."""
    log = []
    code, _ = _run([PY, os.path.join(SRC, "search.py"), "--gens", "40",
                    "--panel", "8", "--jobs", "8"])
    log.append(("search", code))
    args = [PY, os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "build_agent.py"), "--out",
            os.path.join(common.ROOT, "agents", "planner_v0.py")]
    pb = os.path.join(common.MODELS, "params_best.json")
    if os.path.exists(pb):
        args += ["--params", pb]
    code, _ = _run(args, timeout=600)
    log.append(("build", code))

    # arena vs the route incumbent: bank ratio is the phase-0 currency
    from kaggriculture.trackp import arena
    inc = _route_incumbent()
    res = arena.paired(os.path.join(common.ROOT, "agents", "planner_v0.py"),
                       inc, n=2, seed0=52011)
    ratio = (sum(r["bank_a"] for r in res["rows"])
             / max(1.0, sum(r["bank_b"] for r in res["rows"])))
    with open(os.path.join(common.MODELS, "phase0_arena.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"bank_ratio": round(ratio, 3), "score": res["score_a"],
                   "incumbent": os.path.basename(inc),
                   "when": dt.datetime.now().isoformat(
                       timespec="seconds")}, fh, indent=1)
    log.append(("arena", 0))

    # takeover-regret probe on the newest strong staged replay
    cands = sorted(glob.glob(os.path.join(
        common.ROOT, "data", "sameday", "_stage", "*", "*.json")),
        key=os.path.getmtime, reverse=True)
    pick = next((c for c in cands if os.path.getsize(c) > 3_000_000), None)
    if pick:
        code, _ = _run([PY, os.path.join(SRC, "regret.py"),
                        "--replay", pick, "--seat", "0"], timeout=3600)
        log.append(("regret", code))
    return {"ran": log, "gate": gate0()}


def run1() -> dict:
    g = gate1()
    if not all(g["build_items"].values()):
        return {"blocked": "build items missing (LOADSTATE / exit.py) -- "
                           "engineering work, not runnable", "gate": g}
    llm = r"C:/ProgramData/anaconda3/envs/llm/python.exe"
    code, _ = _run([llm, os.path.join(SRC, "exit.py")], timeout=14400)
    return {"ran": [("exit", code)], "gate": gate1()}


def run2() -> dict:
    llm = r"C:/ProgramData/anaconda3/envs/llm/python.exe"
    log = []
    code, _ = _run([llm, os.path.join(SRC, "ppo.py"), "--exploiter",
                    "--iters", "4", "--episodes", "32", "--jobs", "8"],
                   timeout=7200)
    log.append(("exploiter", code))
    if os.path.exists(os.path.join(SRC, "psro.py")):
        code, _ = _run([llm, os.path.join(SRC, "psro.py")], timeout=14400)
        log.append(("psro", code))
    # bank fresh guard evidence vs BOTH incumbents
    code, _ = _run([PY, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "pipeline.py"),
                    "--stages", "judge"], timeout=3600)
    log.append(("judge", code))
    return {"ran": log, "gate": gate2()}


def run3() -> dict:
    code, out = _run([PY, os.path.join(common.ROOT, "scripts",
                                       "engine_swap_1327.py"), "--status"],
                     timeout=300)
    return {"watcher": out.strip().splitlines()[-1:], "gate": gate3()}


def run4() -> dict:
    code, _ = _run([PY, os.path.join(SRC, "graduation.py"),
                    "--panel-n", "8", "--holdout-n", "4"], timeout=14400)
    return {"ran": [("graduation", code)], "gate": gate4()}


RUNS = [run0, run1, run2, run3, run4]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    if a.status or not a.run:
        rows = [g() for g in GATES]
        print(json.dumps(rows, indent=1))
        with open(REPORT, "w", encoding="utf-8") as fh:
            json.dump({"when": dt.datetime.now().isoformat(
                timespec="seconds"), "phases": rows}, fh, indent=1)
        for r in rows:
            mark = "PASS" if r["gate"]["passed"] else "open"
            print(f"phase {r['phase']} {r['name']:<16} [{mark}] "
                  f"{r['gate']['needs']}")
        return 0
    targets = range(5) if a.run == "all" else [int(a.run)]
    for i in targets:
        print(f"===== PHASE {i} =====", flush=True)
        res = RUNS[i]()
        print(json.dumps(res, indent=1), flush=True)
        gd = res.get("gate") or {}
        passed = bool((gd.get("gate") or {}).get("passed"))
        if a.run == "all" and not passed:
            print(f"phase {i} gate OPEN -- stopping the chain here "
                  f"(needs: {(gd.get('gate') or {}).get('needs')})")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
