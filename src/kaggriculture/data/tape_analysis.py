"""Summarise a recorded action tape (the open-loop plans top agents ship).

The top of this leaderboard does not ship a policy -- it ships a 720-step
recording of a near-optimal episode plus slip recovery, because the environment
is deterministic apart from weed spawns and shop-unlock order. Reading those
tapes tells us the target trajectory in exact numbers: what is bought on which
turn, how many hands are hired per day, when each crop goes in the ground and
when the produce is sold.

Usage:
    python -m kaggriculture.data.tape_analysis data/kernels/_agents/*__ACTIONS.json
    python -m kaggriculture.data.tape_analysis <tape.json> --day 0        # turn-by-turn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

TPD = 24


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def unit_ops(entry):
    ops = []
    f = entry.get("farmer")
    if isinstance(f, list) and f:
        ops.append(f)
    for h in entry.get("hands") or []:
        if isinstance(h, list) and h:
            ops.append(h)
    return ops


def summarise(path, day_detail=None):
    tape = load(path)
    print("=" * 78)
    print(f"{os.path.basename(path)}  {len(tape)} steps")

    ops = Counter()
    market = Counter()
    spend = Counter()
    per_day = {}
    seed_costs = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
    animal_costs = {"GOOSE": 300, "COW": 400, "SHEEP": 500}

    for t, entry in enumerate(tape):
        day = t // TPD
        d = per_day.setdefault(day, {"hire": 0, "hands": 0, "sell": Counter(),
                                     "buy": Counter(), "plant": Counter(),
                                     "ops": Counter(), "land": 0})
        us = unit_ops(entry)
        d["hands"] = max(d["hands"], len(entry.get("hands") or []))
        for a in us:
            ops[a[0]] += 1
            d["ops"][a[0]] += 1
            if a[0] == "PLANT" and len(a) > 1:
                d["plant"][a[1]] += 1
        for o in entry.get("market") or []:
            if not isinstance(o, list) or not o:
                continue
            op = o[0]
            market[op] += 1
            if op == "HIRE":
                d["hire"] += 1
            elif op == "BUY_LAND":
                d["land"] += 1
            elif len(o) >= 3:
                item, n = o[1], int(o[2])
                if op == "SELL":
                    d["sell"][item] += n
                else:
                    d["buy"][f"{op[4:]}:{item}"] += n
                    if op == "BUY_SEED":
                        spend["seed"] += seed_costs.get(item, 0) * n
                    elif op == "BUY_ANIMAL":
                        spend["animal"] += animal_costs.get(item, 0) * n
                    elif op == "BUY_PRODUCT":
                        spend[f"buy_{item}"] += n

    total = sum(ops.values())
    print(f"  unit-turns {total}   " + "  ".join(
        f"{k}={v / total * 100:.1f}%" for k, v in ops.most_common()))
    print(f"  market orders: " + "  ".join(f"{k}={v}" for k, v in market.most_common()))
    print(f"  spend: " + "  ".join(f"{k}={v:,}" for k, v in spend.most_common()))

    print(f"  {'day':>3} {'hands':>5} {'hire':>4} {'land':>4} | plants | buys | sells")
    for day in sorted(per_day):
        d = per_day[day]
        pl = " ".join(f"{k}x{v}" for k, v in d["plant"].most_common())
        bu = " ".join(f"{k}x{v}" for k, v in d["buy"].most_common())
        se = " ".join(f"{k}x{v}" for k, v in d["sell"].most_common())
        print(f"  {day:>3} {d['hands']:>5} {d['hire']:>4} {d['land']:>4} | {pl:<34} | {bu:<40} | {se}")

    if day_detail is not None:
        print(f"  --- turn detail, day {day_detail} ---")
        for t in range(day_detail * TPD, min(len(tape), (day_detail + 1) * TPD)):
            e = tape[t]
            hands = "; ".join(" ".join(map(str, h)) for h in (e.get("hands") or []))
            mk = "; ".join(" ".join(map(str, o)) for o in (e.get("market") or []))
            print(f"   t{t:>3} h{t % TPD:>2}  farmer={' '.join(map(str, e.get('farmer') or []))}"
                  f"  | hands: {hands}  | market: {mk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--day", type=int, default=None)
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p))
    for f in files:
        summarise(f, day_detail=args.day)


if __name__ == "__main__":
    main()
