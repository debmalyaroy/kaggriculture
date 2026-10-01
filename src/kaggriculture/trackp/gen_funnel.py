"""P2.2-P2.4 -- the generation funnel.

  sample K macro plans (generator.py)
    -> execute each with the planner (forced macro, argmax L0) against a
       strong reference anchor on `kagg serve`, on a seed whose shop draw
       matches the plan's assumed regime path; record the action tape
       (legal BY CONSTRUCTION -- the executor emits only legal ops)
    -> pre-rank by closed-loop gap vs the reference
    -> P2.4 JOINT novelty: opening hash NOT in the field corpus AND
       sell-curve distance from every known family above a floor
       (either alone is gameable)
    -> official confirm: top plans replayed as tape agents vs the route
       incumbent on the OFFICIAL engine
    -> survivors written to models/trackp/generated/ as crown-candidate
       tapes with a manifest (source="generated")

Usage: python src/trackp/gen_funnel.py [--top 4] [--confirm 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

try:
    from . import arena, common, families, league, macro, panel, trace_v2
    from .serve_env import ServeEnv, seat_view
except ImportError:
    import sys
    from kaggriculture.trackp import (arena, common, families, league, macro, panel,
                        trace_v2)
    from kaggriculture.trackp.serve_env import ServeEnv, seat_view

GENERATED = os.path.join(common.MODELS, "generated")
os.makedirs(GENERATED, exist_ok=True)
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "planner_template.py")


def _forced_ns(plan: dict) -> dict:
    """Planner namespace whose _macro returns the plan's bins per day."""
    with open(TEMPLATE, encoding="utf-8") as fh:
        code = compile(fh.read(), "planner_forced", "exec")
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    bins = plan["bins"]

    def _macro_forced(obs):
        day = min(obs["day"], len(bins) - 1)
        row = bins[day]
        return {name: row[i] for i, (name, _) in enumerate(macro.HEADS)}
    ns["_macro"] = _macro_forced
    return ns


def _seed_for_regime(regime: str, cache: dict, seed0: int = 71000) -> int:
    if regime in cache:
        return cache[regime]
    s = seed0
    while s < seed0 + 200000:
        if panel.stratum_of(s) == regime:
            cache[regime] = s
            return s
        s += 13
    cache[regime] = seed0
    return seed0


def _tape_hash(actions: list) -> str:
    rows = []
    for a in actions[:72]:
        rows.append(np.asarray(np.round(trace_v2._act_block(a)),
                               dtype=np.int32))
    return hashlib.sha1(np.stack(rows).tobytes()).hexdigest()[:16]


def _sell_curve(actions: list) -> np.ndarray:
    cur = np.zeros((30, len(common.PRODUCTS)))
    for t, a in enumerate(actions):
        day = min(29, t // 24)
        for o in (a.get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" \
                    and o[1] in common.PRODUCTS:
                try:
                    cur[day, common.PRODUCTS.index(o[1])] += int(o[2])
                except (TypeError, ValueError):
                    pass
    return cur


def _family_curves() -> list:
    if not os.path.exists(families.FAMILIES):
        return []
    with open(families.FAMILIES, encoding="utf-8") as fh:
        d = json.load(fh)
    out = []
    for fam in d["families"].values():
        cur = np.zeros((30, len(common.PRODUCTS)))
        for p, days in fam.get("dump_per_day", {}).items():
            cur[:, common.PRODUCTS.index(p)] = days[:30]
        out.append(cur)
    return out


def _emit_tape_agent(actions: list, out_path: str):
    """A single-file agent that replays the recorded actions (imports math
    only, per the contract) -- the official-confirm vehicle."""
    body = json.dumps(actions)
    src = (
        "# TRACKP GENERATED ROUTE -- tape replay agent\n"
        "import math  # noqa: F401 -- contract allows only math\n"
        f"_ACTS = {body}\n"
        "def agent(obs):\n"
        "    t = obs.get('step', 0)\n"
        "    if 0 <= t < len(_ACTS):\n"
        "        return _ACTS[t]\n"
        "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)


def run(plans_path: str = "", top: int = 4, confirm: int = 2,
        known_hashes: set = None) -> dict:
    plans_path = plans_path or os.path.join(common.MODELS,
                                            "generated_plans.json")
    with open(plans_path, encoding="utf-8") as fh:
        plans = json.load(fh)["plans"]
    anchors = league.load_anchors()
    if not anchors:
        raise SystemExit("no anchors")
    ref = anchors[0]  # the strongest reference
    cache: dict = {}
    env = ServeEnv()
    ranked = []
    for pi, plan in enumerate(plans):
        regime = next((r for r in plan["regime_path"] if r != "none"),
                      "neutral")
        seed = _seed_for_regime(regime, cache)
        ns = _forced_ns(plan)
        acts = []
        try:
            obs = env.reset(seed, opp_tape=ref["tape"], opp_seat=1)
            while not obs.get("done"):
                a = ns["agent"](seat_view(obs, 0))
                acts.append(a)
                obs = env.step(a)
        except Exception as e:  # noqa: BLE001
            env.close()
            print(f"plan {pi}: ERR {e}")
            continue
        gap = float(obs["farms"][0]["money"]) - float(
            obs["farms"][1]["money"])
        ranked.append({"plan": pi, "regime": regime, "seed": seed,
                       "gap": gap, "bank": float(obs["farms"][0]["money"]),
                       "acts": acts})
    env.close()
    ranked.sort(key=lambda r: -r["gap"])

    # joint novelty on the survivors
    if known_hashes is None:
        known_hashes = set()
        if os.path.exists(families.FAMILIES):
            with open(families.FAMILIES, encoding="utf-8") as fh:
                known_hashes = set(json.load(fh)["families"].keys())
    fam_curves = _family_curves()
    survivors = []
    for r in ranked[: top * 3]:
        h = _tape_hash(r["acts"])
        novel_hash = h not in known_hashes
        cur = _sell_curve(r["acts"])
        if fam_curves:
            dmin = min(float(np.abs(cur - fc).sum()) for fc in fam_curves)
        else:
            dmin = 1e9
        novel_curve = dmin > 30.0
        r["novelty"] = {"hash": h, "novel_hash": novel_hash,
                        "curve_dist": round(dmin, 1),
                        "novel_curve": novel_curve,
                        "joint": novel_hash and novel_curve}
        if r["novelty"]["joint"]:
            survivors.append(r)
        if len(survivors) >= top:
            break

    # official confirm: top `confirm` survivors vs the route incumbent
    try:
        from . import graduation
    except ImportError:
        from kaggriculture.trackp import graduation
    incumbent = graduation.route_incumbent()
    manifest = []
    for si, r in enumerate(survivors[:confirm]):
        agent_path = os.path.join(GENERATED, f"gen_{si}.py")
        _emit_tape_agent(r["acts"], agent_path)
        res = arena.paired(agent_path, incumbent, n=1, seed0=r["seed"])
        manifest.append({"file": agent_path, "plan": r["plan"],
                         "regime": r["regime"], "seed": r["seed"],
                         "prerank_gap": round(r["gap"], 0),
                         "novelty": r["novelty"],
                         "official_score_vs_incumbent": res["score_a"],
                         "source": "generated"})
    rep = {"sampled": len(plans), "played": len(ranked),
           "novel_joint": len(survivors),
           "best_prerank_gap": round(ranked[0]["gap"], 0) if ranked else None,
           "confirmed": manifest}
    with open(os.path.join(GENERATED, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "confirmed"},
                     indent=1))
    for m in manifest:
        print(json.dumps(m))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--confirm", type=int, default=2)
    a = ap.parse_args()
    run(a.plans, a.top, a.confirm)
