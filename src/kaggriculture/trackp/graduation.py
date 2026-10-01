"""P5 -- the graduation gate. REPORT-ONLY: it never ships anything.

The five conditions, verbatim from the plan:
  1. beats the ROUTE incumbent by the crown margin (+10pp on wins) on a
     paired official-engine panel
  2. the held-out panel agrees (positive edge on games that never touched
     selection)
  3. PLANNER GUARD satisfied (guard.planner_allowed)
  4. sim-to-real re-validation on the official engine passed
  5. operator go -- ALWAYS manual; this tool only reports 1-4

Usage: python src/trackp/graduation.py [--panel-n 16]
Writes models/trackp/graduation_report.json.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os

try:
    from . import arena, common, guard
except ImportError:
    import sys
    from kaggriculture.trackp import arena, common, guard

CROWN_MARGIN_PP = 10.0
PLANNER = os.path.join(common.ROOT, "agents", "planner_v0.py")


def route_incumbent() -> str:
    """Newest v*_route agent file -- located by name, no registry import."""
    cands = glob.glob(os.path.join(common.ROOT, "agents", "v*_route.py"))
    if not cands:
        raise SystemExit("no route incumbent found in agents/")

    def key(p):
        base = os.path.basename(p)[1:].split("_")[0]
        parts = base.split(".")
        try:
            return tuple(int(x) for x in parts) + (0,) * (2 - len(parts))
        except ValueError:
            return (0, 0)
    return max(cands, key=key)


def run(panel_n: int = 16, holdout_n: int = 6, planner: str = "") -> dict:
    planner = planner or PLANNER
    incumbent = route_incumbent()
    print(f"graduation check: {os.path.basename(planner)} vs "
          f"{os.path.basename(incumbent)}", flush=True)

    # condition 1: paired panel (selection seeds)
    res = arena.paired(planner, incumbent, n=panel_n, seed0=11)
    edge_pp = 100.0 * (res["score_a"] - 0.5)
    c1 = edge_pp >= CROWN_MARGIN_PP

    # condition 2: held-out seeds (NEVER used by selection/search: 9xxxxx)
    hres = arena.paired(planner, incumbent, n=holdout_n, seed0=900011)
    c2 = hres["score_a"] > 0.5

    # condition 3: guard
    g = guard.planner_allowed(verbose=False)
    c3 = g["allowed"]

    # condition 4: sim-to-real record
    s2r_path = os.path.join(common.MODELS, "sim2real.json")
    if os.path.exists(s2r_path):
        with open(s2r_path, encoding="utf-8") as fh:
            s2r = json.load(fh)
        c4 = bool(s2r.get("ok"))
    else:
        s2r = {"ok": False, "reason": "no sim2real record"}
        c4 = False

    report = {
        "planner": planner, "incumbent": incumbent,
        "c1_crown_margin": {"pass": c1, "edge_pp": round(edge_pp, 1),
                            "needed_pp": CROWN_MARGIN_PP,
                            "score": res["score_a"], "games": res["games"]},
        "c2_holdout": {"pass": c2, "score": hres["score_a"],
                       "games": hres["games"]},
        "c3_guard": {"pass": c3, **{k: g[k] for k in g if k != "allowed"}},
        "c4_sim_to_real": {"pass": c4, **s2r},
        "c5_operator": "MANUAL -- never automated",
        "graduated_1_to_4": bool(c1 and c2 and c3 and c4),
    }
    out = os.path.join(common.MODELS, "graduation_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "planner"}, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-n", type=int, default=16)
    ap.add_argument("--holdout-n", type=int, default=6)
    ap.add_argument("--planner", default="")
    a = ap.parse_args()
    run(a.panel_n, a.holdout_n, a.planner)
