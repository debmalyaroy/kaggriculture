"""Sim-to-real: the planner's score on the SAME anchors, serve vs official.

The Rust engine is bit-exact, so any drift here comes from the harness
(obs shaping, timing), not the rules -- but the check runs anyway, because
"bit-exact" is an empirical property that revokes, not an assumption.
The recorded verdict gates graduation condition 4.

Usage: python src/trackp/sim2real.py [--n 6]
Writes models/trackp/sim2real.json.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os

try:
    from . import arena, common, guard, league, rollouts
except ImportError:
    import sys
    from kaggriculture.trackp import arena, common, guard, league, rollouts

PLANNER = os.path.join(common.ROOT, "agents", "planner_v0.py")


def _tape_agent(actions_tape: str):
    """An official-engine agent that replays a single-seat tape file."""
    acts = []
    with open(actions_tape, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        farmer = parts[0].split(" ") if parts and parts[0] else ["PASS"]
        hands = [h.split(" ") for h in parts[1].split(";")
                 if h] if len(parts) > 1 else []
        market = [o.split(" ") for o in parts[2].split(";")
                  if o] if len(parts) > 2 else []
        acts.append({"farmer": farmer, "hands": hands, "market": market})

    def agent(obs):
        t = obs.get("step", 0)
        if 0 <= t < len(acts):
            return acts[t]
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def run(n: int = 6, jobs: int = 6) -> dict:
    anchors = league.load_anchors()[:n]
    if not anchors:
        raise SystemExit("no anchors")

    # serve side
    runner = rollouts.Runner(jobs)
    try:
        sjobs = [{"seed": a["seed"], "me": {},
                  "opp": {"kind": "tape", "tape": a["tape"]}}
                 for a in anchors]
        res = runner.run(sjobs)
    finally:
        runner.close()
    s_scores = []
    for r in res:
        if "banks" in r:
            s_scores.append(1.0 if r["banks"][0] > r["banks"][1] else
                            (0.5 if r["banks"][0] == r["banks"][1] else 0.0))
    serve_score = sum(s_scores) / max(1, len(s_scores))

    # official side: same anchors as tape-replay agents, same seeds
    planner = arena.load_agent(PLANNER)
    o_scores = []
    for a in anchors:
        opp = _tape_agent(a["tape"])
        ba, bb = arena.play(planner, opp, a["seed"])
        o_scores.append(1.0 if ba > bb else (0.5 if ba == bb else 0.0))
    official_score = sum(o_scores) / max(1, len(o_scores))

    rep = guard.sim_to_real(serve_score, official_score)
    rep["n"] = len(anchors)
    with open(os.path.join(common.MODELS, "sim2real.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    run(a.n, a.jobs)
