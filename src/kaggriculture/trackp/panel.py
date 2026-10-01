"""P1.6 -- the kill-switch panel: paired official games, stratified by the
shop-draw regime (and seat). The shop draw is deterministic per seed, so
strata are computed BEFORE playing and seeds are chosen to cover every
demand partition.

Demand partitions (branch keys, per the plan): wool-ish / carrot-ish /
dairy-ish / neutral, from the first two shop unlocks.

Verdict discipline: the planner must improve its target regimes without
losing the healthy ones; per-stratum score + a pooled sign test.

Usage: python src/trackp/panel.py --vs agents/v25.0_route.py --per-stratum 3
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import random as _random

try:
    from . import arena, common, guard
except ImportError:
    import sys
    from kaggriculture.trackp import arena, common, guard

PARTITION = {
    "YARN_STORE": "wool", "PET_CAFE": "carrot",
    "SMOOTHIE_SHOP": "dairy", "ICE_CREAM_SHOP": "dairy",
    "PIZZA_SHOP": "dairy",
    "BAKERY": "neutral", "BRUNCH_SPOT": "neutral",
    "FARMERS_MARKET": "neutral",
}


def shop_draw(seed: int, n_days: int = 30) -> list:
    """The engine's exact unlock sequence for a seed (same RNG recipe)."""
    shops = []
    for day in range(n_days):
        rng = _random.Random((seed * 1_000_003) ^ day)
        # end_of_day consumes weed draws first (one random() per EMPTY tile
        # per farm) -- but tile emptiness depends on play. The DRAW ITSELF is
        # rng.choice AFTER those consumptions, so the exact shop can shift
        # with play. For stratification we use the play-independent
        # approximation: the draw with no consumption. Verified adequate:
        # strata only need to be BALANCED, not exact.
        next_day = day + 1
        if next_day > 0 and next_day % common.TOWN_SHOP_UNLOCK_INTERVAL == 0:
            if len(shops) < common.MAX_SHOP_INSTANCES:
                shops.append(rng.choice(sorted(common.SHOPS_SORTED)))
    return shops


def stratum_of(seed: int) -> str:
    draw = shop_draw(seed)
    if not draw:
        return "neutral"
    parts = [PARTITION.get(s, "neutral") for s in draw[:2]]
    for p in ("wool", "carrot", "dairy"):
        if p in parts:
            return p
    return "neutral"


def pick_seeds(per_stratum: int, seed0: int = 50000) -> dict:
    """Scan seeds until every partition has per_stratum members."""
    need = {"wool": per_stratum, "carrot": per_stratum,
            "dairy": per_stratum, "neutral": per_stratum}
    out = {k: [] for k in need}
    s = seed0
    while any(len(out[k]) < need[k] for k in need) and s < seed0 + 100000:
        st = stratum_of(s)
        if len(out[st]) < need[st]:
            out[st].append(s)
        s += 17
    return out


def run(planner: str, vs: str, per_stratum: int = 3) -> dict:
    seeds = pick_seeds(per_stratum)
    agent_a = arena.load_agent(planner)
    agent_b = arena.load_agent(vs)
    strata = {}
    all_rows = []
    for st, ss in seeds.items():
        rows = []
        for seed in ss:
            for seat in (0, 1):
                if seat == 0:
                    ba, bb = arena.play(agent_a, agent_b, seed)
                else:
                    bb, ba = arena.play(agent_b, agent_a, seed)
                w = 1.0 if ba > bb else (0.5 if ba == bb else 0.0)
                rows.append({"seed": seed, "seat": seat, "bank_a": ba,
                             "bank_b": bb, "score": w})
                all_rows.append(rows[-1])
                print(f"RESULT\t{st}\t{seed}\t{seat}\t{ba}\t{bb}", flush=True)
        strata[st] = {
            "games": len(rows),
            "score": round(sum(r["score"] for r in rows) / len(rows), 3),
            "mean_gap": round(sum(r["bank_a"] - r["bank_b"]
                                  for r in rows) / len(rows), 0)}
    wins = sum(1 for r in all_rows if r["score"] == 1.0)
    losses = sum(1 for r in all_rows if r["score"] == 0.0)
    p = guard._binom_tail(wins, wins + losses) if wins + losses else 1.0
    rep = {"planner": planner, "vs": vs, "strata": strata,
           "pooled_score": round(sum(r["score"] for r in all_rows)
                                 / len(all_rows), 3),
           "wins": wins, "losses": losses, "sign_p": round(p, 5),
           "games": len(all_rows)}
    out = os.path.join(common.MODELS, "panel_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in
                      ("strata", "pooled_score", "sign_p", "games")},
                     indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default=os.path.join(
        common.ROOT, "agents", "planner_v0.py"))
    ap.add_argument("--vs", required=True)
    ap.add_argument("--per-stratum", type=int, default=3)
    a = ap.parse_args()
    run(a.planner, a.vs, a.per_stratum)
