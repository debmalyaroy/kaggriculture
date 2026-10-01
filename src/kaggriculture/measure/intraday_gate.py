"""Intraday-fix gate: a v{x}.{y>0} build must EARN the right to retire a
climbing submission.

v25.1 shipped with "no win conversion yet" in its own journal entry and cut
down v25.0_bandit four hours into its rating climb. Ratings converge with
games played, so replacing a live submission is a COST -- roughly a day of
climb -- and an untested fix pays it for nothing.

    python src/intraday_gate.py agents/v27.1_bandit.py
    python src/intraday_gate.py agents/v27.1_bandit.py --vs agents/v27.0_bandit.py

Runs a paired evaluation (same roster, same seeds -> the cells pair up),
McNemar-sign-tests the difference, and writes the verdict to
models/intraday_gate/<candidate>.json. submit.py refuses a y>0 agent without
a fresh PASSING verdict. Never submits anything itself.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

import kaggriculture.measure.win_metric as WM  # noqa: E402

OUT_DIR = os.path.join(ROOT, "models", "intraday_gate")
SEEDS = (60000, 150000)
VPAT = re.compile(r"v(\d+)(?:\.(\d+))?_(.+)\.py$")


def parse_version(path):
    m = VPAT.search(os.path.basename(path))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0), m.group(3))


def default_baseline(cand):
    """The newest same-kind agent strictly below the candidate's version --
    the submission the fix would retire."""
    v = parse_version(cand)
    if not v:
        return None
    x, y, kind = v
    best, bestv = None, (-1, -1)
    for p in glob.glob(os.path.join(ROOT, "agents", f"v*_{kind}.py")):
        pv = parse_version(p)
        if not pv:
            continue
        px, py, _ = pv
        if (px, py) < (x, y) and (px, py) > bestv:
            best, bestv = p, (px, py)
    return best


def roster():
    """Opponents for the paired gate — REACTIVE agents first.

    2026-09-04: this gate was BLIND. Its roster was 4 loss tapes + 4 panel
    tapes, which gave 16 cells and returned score_a 1.000 vs score_b 1.000
    with ZERO discordant cells (p = 1.0) for a candidate that a 691-world
    battery separated at p = 0.0386 — so `submit.py` refused a real
    improvement on no evidence, and the refusal had to be overridden.

    Two causes, both fixed here:
      * too few opponents (8, capped at 4 each) to resolve anything;
      * they were all TAPES, and tape worlds are a different economy --
        median shared bank 56-63k with 93-99% of worlds sub-90k, against a
        ladder that runs ~85k and 58% (docs/history/instrument-repair-2026-09-03.md).
        Reactive opponents produce ladder-like worlds (88.7k, 53%).

    Reactive agents lead the roster; tapes are kept as a minority so a
    candidate that only beats live agents still has to hold up against
    scripted ones. `.local/band_panel/reactive/` holds frozen copies so a
    concurrent rebuild cannot change the roster mid-gate.
    """
    reactive = sorted(glob.glob(os.path.join(
        ROOT, ".local", "band_panel", "reactive", "*.py")))
    if not reactive:
        reactive = [p for p in sorted(glob.glob(os.path.join(
            ROOT, "data", "gauntlet", "*.py"))) if os.path.exists(p)]
    tapes = sorted(glob.glob(os.path.join(
        ROOT, "data", "refresh", "*", "loss_tapes", "tape_*.py")),
        key=os.path.getmtime, reverse=True)[:3]
    panel = sorted(glob.glob(os.path.join(
        ROOT, "data", "panel", "opp_r0*.py")))[:3]
    out = reactive[:10] + tapes + panel
    if not out:                      # never silently gate on nothing
        return []
    print(f"  roster: {len(reactive[:10])} reactive + {len(tapes)} loss tapes "
          f"+ {len(panel)} panel tapes = {len(out)} opponents")
    return out


def run_eval(agent, opps, seed0, n):
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "measure", "evaluate.py"), agent,
         "--vs", *opps, "-n", str(n), "--seed0", str(seed0), "--no-record"],
        cwd=ROOT, capture_output=True, text=True, timeout=7200)
    return WM.parse_eval(r.stdout + r.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate")
    ap.add_argument("--vs", default=None,
                    help="the live agent the fix would retire "
                         "(default: newest same-kind lower version)")
    ap.add_argument("-n", type=int, default=4)
    a = ap.parse_args()

    cand = os.path.abspath(os.path.join(ROOT, a.candidate))
    if not os.path.exists(cand):
        sys.exit(f"no such agent: {cand}")
    v = parse_version(cand)
    base = os.path.abspath(os.path.join(ROOT, a.vs)) if a.vs \
        else default_baseline(cand)
    if not base or not os.path.exists(base):
        sys.exit("no baseline found; pass --vs explicitly")
    opps = roster()
    if not opps:
        sys.exit("no roster (need data/panel and/or loss tapes)")
    print(f"candidate {os.path.relpath(cand, ROOT)} vs baseline "
          f"{os.path.relpath(base, ROOT)}; {len(opps)} opponents, "
          f"seeds {SEEDS}, n={a.n}")

    cells = {}
    for name, agent in (("cand", cand), ("base", base)):
        for seed0 in SEEDS:
            for r in run_eval(agent, opps, seed0, a.n):
                cells.setdefault((seed0, r["opponent"]), {})[name] = r["score"]
                print(f"  {name} seed0={seed0} vs "
                      f"{os.path.basename(r['opponent'])}: "
                      f"{r['score']:.2f} ({r['wins']}/{r['games']})",
                      flush=True)
    xs = [v_ for v_ in cells.values() if "cand" in v_ and "base" in v_]
    t = WM.paired_test([c["cand"] for c in xs], [c["base"] for c in xs])
    passed = bool(xs) and t["score_diff"] > 0 and t["significant"]
    verdict = {
        "candidate": os.path.relpath(cand, ROOT),
        "baseline": os.path.relpath(base, ROOT),
        "version": f"{v[0]}.{v[1]}" if v else None,
        "cells": len(xs), **t, "passed": passed,
        "when": dt.datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, os.path.basename(cand) + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps({k: verdict[k] for k in
                      ("cells", "score_diff", "p_value", "passed")}, indent=1))
    print(("PASS -- the fix has a sign-tested edge; it may retire the live "
           "submission") if passed else
          ("FAIL -- no sign-tested edge; an untested fix does not retire a "
           "climbing submission"))
    print(f"verdict -> {os.path.relpath(out, ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
