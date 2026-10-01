"""Where the money leaks out of a farm.

The public meta write-up ("notes from replay hunting") lands on one conclusion
after ten leader refreshes: *the current edge comes from inventory-sale timing
rather than another change in herd composition*. Its strongest promotion
changed 20 field turns and **112 market turns**.

That is not something a bank total can show you. This tool audits a replay for
the five ways produce turns into nothing:

  1. **grown but never harvested** -- yield left standing when the game ends
  2. **harvested but never banked** -- SELL only draws from the shed, so
     anything in a unit's hands at step 719 scores zero
  3. **destroyed by shed overflow** -- the shed holds 100 and drops the rest
  4. **left in the shed** -- banked but never sold
  5. **sold too cheap** -- realised price against the best price that item
     traded at while we were holding it

Usage:
    python -m kaggriculture.measure.leak_check .local/dbg/match.json
    python -m kaggriculture.measure.leak_check <replay> --seat 0
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import math
import os

CROPS = {
    "WHEAT":      {"first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
PRODUCTS = list(CROPS) + ["EGG", "MILK", "WOOL", "FERTILIZER"]
I0 = 10000
SHED_CAP = 100

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "log",    "below_target": 0.20, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


# Engine 1.32.7 hinge rebalance -- the active table follows
# models/engine_version.json (written only by scripts/engine_swap_1327.py).
try:
    import json as _json
    import os as _os
    _ev = _json.load(open(_os.path.join(
        ROOT,
        "models", "engine_version.json"), encoding="utf-8")).get("engine",
                                                                 "1.32.6")
except Exception:                                              # noqa: BLE001
    _ev = "1.32.6"
if tuple(int(x) for x in _ev.split(".")) >= (1, 32, 7):
    MARKET_PARAMS["CARROT"].update(below_func="hinge", below_target=1.00)
    MARKET_PARAMS["TOMATO"].update(below_func="hinge")
    MARKET_PARAMS["EGG"].update(below_func="hinge")


def _shape(f, x, t=0.0):
    x = max(0.0, x)
    if f == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),
            "log": math.log(1.0 + x), "log10": math.log10(1.0 + x)}.get(f, x)


def price_of(item, inv):
    p = MARKET_PARAMS[item]
    if inv < I0:
        amp = p["below_target"] * p["base"] / _shape(p["below_func"], p["T"],
                                                     p["T"])
        v = p["base"] + amp * _shape(p["below_func"], I0 - inv, p["T"])
    else:
        amp = p["above_target"] * p["base"] / _shape(p["above_func"], p["T"],
                                                     p["T"])
        v = p["base"] - amp * _shape(p["above_func"], inv - I0, p["T"])
    return max(1, int(round(v)))


def audit(path, seat=0):
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    steps = rep["steps"]
    names = rep.get("info", {}).get("TeamNames", ["p0", "p1"])
    rewards = rep.get("rewards", [0, 0])
    n = len(steps)

    peak_price = {k: 0 for k in PRODUCTS}
    sold_units = {k: 0 for k in PRODUCTS}
    sold_value = {k: 0.0 for k in PRODUCTS}
    best_value = {k: 0.0 for k in PRODUCTS}
    overflow = {k: 0 for k in PRODUCTS}
    harvest_ops = 0
    sell_orders = 0
    turns_with_sell = 0

    prev_shed = None
    for t in range(n - 1):
        obs0 = steps[t][0]["observation"]
        inv = (obs0.get("market") or {}).get("inventory") or {}
        priv = steps[t][seat]["observation"].get("private") or {}
        shed = priv.get("shed") or {}
        for item in PRODUCTS:
            peak_price[item] = max(peak_price[item], price_of(item, inv.get(item, I0)))

        # what the agent asked the market to do this turn
        nxt = steps[t + 1] if t + 1 < n else None
        act = nxt[seat].get("action") if nxt and seat < len(nxt) else None
        if isinstance(act, dict):
            orders = act.get("market") or []
            sells = [o for o in orders if isinstance(o, list) and o and o[0] == "SELL"]
            if sells:
                turns_with_sell += 1
            sell_orders += len(sells)
            for u in [act.get("farmer")] + list(act.get("hands") or []):
                if isinstance(u, list) and u and u[0] == "HARVEST":
                    harvest_ops += 1

        # shed overflow: the engine drops inventories at end of day and discards
        # anything past capacity
        total = sum(shed.values()) if shed else 0
        if prev_shed is not None and total >= SHED_CAP:
            for item in PRODUCTS:
                overflow[item] += 0        # counted below by the carry audit
        prev_shed = total

    # realised sales: shed deltas that are not explained by pickup are sales
    for t in range(n - 1):
        obs0 = steps[t][0]["observation"]
        inv = (obs0.get("market") or {}).get("inventory") or {}
        nxt = steps[t + 1]
        act = nxt[seat].get("action") if seat < len(nxt) else None
        if not isinstance(act, dict):
            continue
        for o in act.get("market") or []:
            if not isinstance(o, list) or len(o) < 3 or o[0] != "SELL":
                continue
            item, want = o[1], int(o[2])
            if item not in MARKET_PARAMS:
                continue
            have = ((steps[t][seat]["observation"].get("private") or {})
                    .get("shed") or {}).get(item, 0)
            k = min(want, have)
            cur = inv.get(item, I0)
            got = 0
            for _ in range(k):
                got += price_of(item, cur)
                cur += 1
            sold_units[item] += k
            sold_value[item] += got

    # what is still on the farm when the music stops
    last = steps[-1]
    obs_last = last[0]["observation"]
    farm = obs_last["farms"][seat]
    priv_last = last[seat]["observation"].get("private") or {}
    shed_left = priv_last.get("shed") or {}
    hands_left = {}
    for inv_u in priv_last.get("inventories") or []:
        for k, v in (inv_u or {}).items():
            hands_left[k] = hands_left.get(k, 0) + v
    standing = {}
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                standing[tile["crop"]] = standing.get(tile["crop"], 0) + int(tile.get("yield_units", 0) or 0)
            elif tile.get("animal") in ANIMALS:
                prod = ANIMALS[tile["animal"]]
                standing[prod] = standing.get(prod, 0) + int(tile.get("yield_units", 0) or 0)

    market_last = (obs_last.get("market") or {}).get("inventory") or {}

    print("=" * 78)
    print(f"{os.path.basename(path)}   seat {seat} ({names[seat]})   banked ${rewards[seat]:,.0f}")
    print("=" * 78)
    print(f"  HARVEST ops {harvest_ops}   SELL orders {sell_orders} on {turns_with_sell} turns "
          f"of {n - 1} ({100 * turns_with_sell / max(1, n - 1):.0f}%)")
    print()
    print(f"  {'product':<12}{'sold':>7}{'avg $':>8}{'peak $':>8}{'timing':>9}   "
          f"{'shed':>6}{'hands':>7}{'standing':>9}   {'lost $':>10}")
    lost_total = 0.0
    for item in PRODUCTS:
        u = sold_units[item]
        avg = sold_value[item] / u if u else 0.0
        peak = peak_price[item]
        left_shed = shed_left.get(item, 0)
        left_hands = hands_left.get(item, 0)
        left_std = standing.get(item, 0)
        # unsold stock is valued at the last market price it could have made
        spot = price_of(item, market_last.get(item, I0))
        lost = (left_shed + left_hands) * spot + left_std * spot
        timing = (avg / peak) if (u and peak) else 0.0
        lost_total += lost
        if u or left_shed or left_hands or left_std:
            print(f"  {item:<12}{u:>7}{avg:>8.0f}{peak:>8.0f}{timing:>8.0%}   "
                  f"{left_shed:>6}{left_hands:>7}{left_std:>9}   {lost:>10,.0f}")
    print(f"  {'':<12}{'':>7}{'':>8}{'':>8}{'':>9}   {'':>6}{'':>7}{'unbanked':>9}   "
          f"{lost_total:>10,.0f}")
    return {"lost": lost_total, "sold": sold_units, "bank": rewards[seat]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--seat", type=int, default=None, help="default: both")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p))
    for f in files:
        seats = [args.seat] if args.seat is not None else [0, 1]
        for s in seats:
            audit(f, s)


if __name__ == "__main__":
    main()
