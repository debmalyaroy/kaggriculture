"""Fit the planner genome to the ELITE cluster's measured macro-economy.

Walks each elite tape with position simulation (units start at shed tiles;
N/S/E/W move one tile) and aggregates, across the cluster: per-tile
species mode (PLANT sp / animal PLACEd after its build), cumulative hires
by day, land-purchase days, first-sell day and max daily sell per product,
feed/fert purchase levels, last plant day. Emits a genome JSON in
build_econ_agent's GENOME shape — the elite MACRO PLAN with closed-loop
execution left to the planner.

    python src/trackp/elite_fit.py --ids .local/audit/elite_ids_300.txt \
        --out models/trackp/genome_elite_fit.json
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

BOARD = 10
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
CROPS = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
ANIMALS = {"GOOSE", "SHEEP", "COW"}
SELLABLE = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
            "MILK", "WOOL", "FERTILIZER")


def quad(t):
    x, y = t
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def walk(route):
    """One tape -> its macro facts."""
    pos = {0: list(SHED[0])}          # unit index -> [x, y]; 0 = farmer
    n_hands = 0
    tile_sp = {}                       # (x,y) -> last species planted/placed
    hires_by_day = Counter()
    land_days = []
    first_sell = {}
    day_sold = defaultdict(lambda: defaultdict(int))
    wheat_buys = Counter()
    fert_buys = Counter()
    last_plant_day = 0
    used = {"CARE": 0, "FERTILIZE": 0}
    for t, a in enumerate(route):
        d = t // 24
        if not isinstance(a, dict):
            continue
        for o in (a.get("market") or []):
            if not (isinstance(o, list) and o):
                continue
            op = o[0]
            if op == "HIRE":
                hires_by_day[d] += 1
                n_hands += 1
                pos[n_hands] = list(SHED[n_hands % 4])
            elif op == "BUY_LAND":
                land_days.append(d)
            elif op == "SELL" and len(o) >= 3:
                it, q = o[1], int(o[2] or 0)
                if q > 0:
                    first_sell.setdefault(it, d)
                    day_sold[it][d] += q
            elif op == "BUY_PRODUCT" and len(o) >= 3:
                if o[1] == "WHEAT":
                    wheat_buys[d] += int(o[2] or 0)
                elif o[1] == "FERTILIZER":
                    fert_buys[d] += int(o[2] or 0)
        units = [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])
        for i, op in enumerate(units):
            if not (isinstance(op, list) and op):
                continue
            if i not in pos:
                pos[i] = list(SHED[i % 4])
            p = pos[i]
            v = op[0]
            if v in MOVES:
                dx, dy = MOVES[v]
                p[0] = max(0, min(BOARD - 1, p[0] + dx))
                p[1] = max(0, min(BOARD - 1, p[1] + dy))
            elif v == "PLANT" and len(op) >= 2 and op[1] in CROPS:
                tile_sp[tuple(p)] = op[1]
                last_plant_day = max(last_plant_day, d)
            elif v == "PLACE" and len(op) >= 2 and op[1] in ANIMALS:
                tile_sp[tuple(p)] = op[1]
            elif v in used:
                used[v] += 1
    return {"tiles": tile_sp, "hires": hires_by_day, "land": land_days,
            "first_sell": first_sell, "day_sold": day_sold,
            "wheat": wheat_buys, "fert": fert_buys,
            "last_plant": last_plant_day, "used": used}


def median(xs, default=0):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else default


def fit(ids):
    from kaggriculture.trackp import routes_io as R
    facts = []
    for rid in ids:
        try:
            facts.append(walk(R.load_route(rid)))
        except Exception:                                      # noqa: BLE001
            continue
    print(f"walked {len(facts)} tapes")
    # per-tile species mode -> per-quadrant alloc
    tile_votes = defaultdict(Counter)
    for f in facts:
        for t, sp in f["tiles"].items():
            tile_votes[t][sp] += 1
    alloc = {"NW": Counter(), "NE": Counter(), "SW": Counter(),
             "SE": Counter()}
    for t, votes in tile_votes.items():
        sp, n = votes.most_common(1)[0]
        if n >= len(facts) * 0.25:      # tile used by a quarter of elites
            alloc[quad(t)][sp] += 1
    # hires: median cumulative curve, converted to per-day plan
    hires = []
    for d in range(30):
        cum = [sum(f["hires"][dd] for dd in range(d + 1)) for f in facts]
        hires.append(median(cum))
    per_day = [max(0, hires[d] - (hires[d - 1] if d else 0)) for d in range(30)]
    # land purchase days (1st..3rd)
    land = {}
    for i, q in enumerate(("NE", "SW", "SE")):
        days = [f["land"][i] for f in facts if len(f["land"]) > i]
        if len(days) >= len(facts) * 0.3:
            land[q] = median(days)
        else:
            land[q] = -1
    holds = {}
    caps = {}
    for it in SELLABLE:
        fs = [f["first_sell"][it] for f in facts if it in f["first_sell"]]
        if fs:
            holds[it] = median(fs)
        daymax = [max(f["day_sold"][it].values())
                  for f in facts if f["day_sold"][it]]
        if daymax:
            caps[it] = median(daymax)
    genome = {
        "land": land,
        "hires": [int(x) for x in per_day],   # hired per day (crew is replaced as contracts expire)
        "alloc": {q: dict(c) for q, c in alloc.items()},
        "start": {"NW": 0, "NE": max(0, land.get("NE", 3)),
                  "SW": max(0, land.get("SW", 6)),
                  "SE": max(0, land.get("SE", 9))},
        "replant": {"WHEAT": True, "CARROT": True, "MELON": True},
        "feed_buy": int(median([median(list(f["wheat"].values()), 0)
                                for f in facts], 4)),
        "sell_every": 1,
        "fertilize": True,
        "care": True,
        "last_plant_day": int(median([f["last_plant"] for f in facts], 24)),
        "sell_cap": {k: int(v) for k, v in caps.items()},
        "fert_crops": True,
        "fert_buy": int(median([sum(f["fert"].values()) / 30 for f in facts], 0)),
        "fert_wheat": False,
        "hold_until": {k: int(v) for k, v in holds.items()},
        "backfill_wheat": True,
        "shop_mods": {},
    }
    return genome, per_day


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "genome_elite_fit.json"))
    args = ap.parse_args()
    ids = open(args.ids).read().split()
    genome, per_day = fit(ids)
    json.dump(genome, open(args.out, "w"), indent=1)
    print("alloc:", {q: c for q, c in genome["alloc"].items() if c})
    print("hires cumulative:", genome["hires"])
    print("land:", genome["land"], "| holds:", genome["hold_until"])
    print("caps:", genome["sell_cap"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
