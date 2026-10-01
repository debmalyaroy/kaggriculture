"""Profile a replay: where the money comes from, what the crew does, when.

Reads a kaggle-environments replay JSON and prints, per player, a
turn-level accounting that answers "what is this agent's economic engine":

  * revenue by product (units sold x realised price, reconstructed from the
    market inventory trail), spend by category
  * the action histogram per unit-turn
  * the day-by-day timeline of hands, herd, tiles by crop, cash
  * fertilizer produced / collected / applied / sold

Usage:
    python -m kaggriculture.data.replay_profile data/top/90041552.json
    python -m kaggriculture.data.replay_profile data/top/*.json --compact
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import math
import os
import sys

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
PRODUCTS = CROPS + ["EGG", "MILK", "WOOL", "FERTILIZER"]
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_PRICES = [1000, 2000, 4000]

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
I0 = 10000



# Engine 1.32.7 hinge rebalance -- follow models/engine_version.json (written
# only by scripts/engine_swap_1327.py at the ladder flip).
try:
    import json as _json
    _ev = _json.load(open(os.path.join(
        ROOT,
        "models", "engine_version.json"), encoding="utf-8")).get(
        "engine", "1.32.6")
except Exception:                                              # noqa: BLE001
    _ev = "1.32.6"
_IS_1327 = tuple(int(x) for x in _ev.split(".")) >= (1, 32, 7)
if _IS_1327:
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
        amp = p["below_target"] * p["base"] / _shape(p["below_func"], p["T"], p["T"])
        v = p["base"] + amp * _shape(p["below_func"], I0 - inv, p["T"])
    else:
        amp = p["above_target"] * p["base"] / _shape(p["above_func"], p["T"], p["T"])
        v = p["base"] - amp * _shape(p["above_func"], inv - I0, p["T"])
    return max(1, int(round(v)))


def unit_actions(action):
    if not isinstance(action, dict):
        return []
    out = []
    f = action.get("farmer")
    if isinstance(f, list) and f:
        out.append(f)
    for h in action.get("hands", []) or []:
        if isinstance(h, list) and h:
            out.append(h)
    return out


def profile(path, compact=False):
    with open(path, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    steps = rep["steps"]
    names = rep.get("info", {}).get("TeamNames", ["p0", "p1"])
    rewards = rep.get("rewards", [0, 0])
    nplayers = len(steps[0])

    stats = []
    for p in range(nplayers):
        stats.append({
            "ops": {}, "sold": {k: 0 for k in PRODUCTS},
            "revenue": {k: 0.0 for k in PRODUCTS},
            "bought": {}, "spend": {"seed": 0.0, "animal": 0.0, "land": 0.0,
                                    "hire": 0.0, "wheat": 0.0, "fert": 0.0},
            "timeline": [], "unit_turns": 0, "hires": 0,
        })

    for t, step in enumerate(steps[:-1]):
        obs0 = step[0]["observation"]
        market_inv = dict(obs0.get("market", {}).get("inventory", {}))
        for p in range(nplayers):
            # steps[i]["action"] produced state i, so the action taken at turn
            # t is stored at t+1.
            nxt = steps[t + 1] if t + 1 < len(steps) else None
            act = (nxt[p].get("action") if nxt and p < len(nxt) else None)
            st = stats[p]
            for a in unit_actions(act):
                st["ops"][a[0]] = st["ops"].get(a[0], 0) + 1
                st["unit_turns"] += 1
            if not isinstance(act, dict):
                continue
            for order in act.get("market", []) or []:
                if not isinstance(order, list) or not order:
                    continue
                op = order[0]
                if op == "HIRE":
                    st["hires"] += 1
                elif op == "BUY_LAND":
                    st["spend"]["land"] += 1
                elif op == "SELL" and len(order) >= 3 and order[1] in MARKET_PARAMS:
                    item, n = order[1], int(order[2])
                    inv = market_inv.get(item, I0)
                    got = 0
                    for _ in range(n):
                        pr = price_of(item, inv)
                        got += pr
                        inv += 1
                    st["sold"][item] = st["sold"].get(item, 0) + n
                    st["revenue"][item] = st["revenue"].get(item, 0.0) + got
                elif op == "BUY_SEED" and len(order) >= 3:
                    st["spend"]["seed"] += SEED_COST.get(order[1], 0) * int(order[2])
                elif op == "BUY_ANIMAL" and len(order) >= 3:
                    st["spend"]["animal"] += ANIMAL_COST.get(order[1], 0) * int(order[2])
                elif op == "BUY_PRODUCT" and len(order) >= 3:
                    item, n = order[1], int(order[2])
                    key = "wheat" if item == "WHEAT" else "fert"
                    inv = market_inv.get(item, I0)
                    cost = 0
                    for _ in range(n):
                        cost += price_of(item, inv - 1)
                        inv -= 1
                    st["spend"][key] += cost
                    st["bought"][item] = st["bought"].get(item, 0) + n

    # ---- exact market flow, reconstructed from the shared inventory trail ----
    # inv(t+1) - inv(t) = (player sales) - (player buys) - (town consumption)
    # Town consumption is deterministic, so the residual is exactly what the two
    # players did between them.
    SHOPS = {
        "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
        "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
        "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
        "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    }
    flow = {k: 0 for k in PRODUCTS}          # net units players put INTO market
    gross = {k: 0.0 for k in PRODUCTS}       # cash the market paid out, net
    for t in range(len(steps) - 1):
        o_now = steps[t][0]["observation"]
        o_nxt = steps[t + 1][0]["observation"]
        inv_now = (o_now.get("market") or {}).get("inventory")
        inv_nxt = (o_nxt.get("market") or {}).get("inventory")
        if not inv_now or not inv_nxt:
            continue
        shops = (o_now.get("town") or {}).get("unlocked_shops", []) or []
        day = o_now.get("day", t // 24)
        town = {k: 0 for k in PRODUCTS}
        if t % 4 == 0:
            for s in shops:
                mult = 2 if len(SHOPS[s]) == 1 else 1
                for item in SHOPS[s]:
                    town[item] += mult
        if t % 12 == 0:
            cm = 4 if day >= 20 else (2 if day >= 10 else 1)
            for item in PRODUCTS:
                if item != "FERTILIZER":
                    town[item] += cm
        for item in PRODUCTS:
            delta = inv_nxt.get(item, I0) - inv_now.get(item, I0)
            net = delta + town[item]          # >0 sold to market, <0 bought
            if net == 0:
                continue
            flow[item] += net
            inv = inv_now.get(item, I0)
            cash = 0.0
            if net > 0:
                for _ in range(net):
                    cash += price_of(item, inv)
                    inv += 1
            else:
                for _ in range(-net):
                    cash -= price_of(item, inv - 1)
                    inv -= 1
            gross[item] += cash
    print_flow = (flow, gross)

    # daily timeline snapshots at mid-day (hour 12), where hands are alive
    tpd = 24
    for t in range(12, len(steps), tpd):
        step = steps[t]
        obs0 = step[0]["observation"]
        day = obs0.get("day", t // tpd)
        farms = obs0.get("farms", [])
        for p in range(nplayers):
            if p >= len(farms):
                continue
            farm = farms[p]
            counts = {k: 0 for k in CROPS + list(ANIMALS) + ["COOP", "PASTURE", "WEED", "EMPTY"]}
            for row in farm["tiles"]:
                for tile in row:
                    if tile is None:
                        counts["EMPTY"] += 1
                    elif isinstance(tile, dict):
                        if tile.get("kind") == "PLANT":
                            counts[tile["crop"]] += 1
                        elif "animal" in tile:
                            counts[tile["animal"]] += 1
                        elif tile.get("kind") in counts:
                            counts[tile["kind"]] += 1
            priv = step[p]["observation"].get("private", {}) if p < len(step) else {}
            shed = priv.get("shed", {}) if isinstance(priv, dict) else {}
            stats[p]["timeline"].append({
                "day": day, "money": farm["money"], "hands": len(farm["hands"]),
                "quads": len(farm["unlocked_quadrants"]), "counts": counts,
                "shed": sum(shed.values()) if shed else 0,
            })

    print("=" * 78)
    print(f"{os.path.basename(path)}   {names}   rewards={rewards}")
    for p in range(nplayers):
        st = stats[p]
        print("-" * 78)
        rev_total = sum(st["revenue"].values())
        print(f"[{p}] {names[p]}   final ${rewards[p]:,.0f}   revenue ${rev_total:,.0f}")
        rev = sorted(st["revenue"].items(), key=lambda kv: -kv[1])
        print("  revenue: " + "  ".join(
            f"{k}={v:,.0f}({st['sold'][k]}u@{v/max(1,st['sold'][k]):.0f})"
            for k, v in rev if v > 0))
        print("  spend:   " + "  ".join(f"{k}={v:,.0f}" for k, v in st["spend"].items() if v)
              + f"   hires={st['hires']}")
        if st["bought"]:
            print("  bought:  " + "  ".join(f"{k}={v}" for k, v in st["bought"].items()))
        fl, gr = print_flow
        if p == 0:
            print("  MARKET FLOW (both players combined, exact):")
            print("    " + "  ".join(f"{k}={fl[k]:+d}u/${gr[k]:,.0f}" for k in PRODUCTS if fl[k]))
        tot = max(1, st["unit_turns"])
        ops = sorted(st["ops"].items(), key=lambda kv: -kv[1])
        print(f"  unit-turns {tot}: " + "  ".join(f"{k}={v/tot*100:.1f}%" for k, v in ops))
        if not compact:
            print("  day  cash    hands quads shed | " + " ".join(f"{c[:4]:>4}" for c in CROPS + list(ANIMALS)))
            for row in st["timeline"]:
                cnt = row["counts"]
                print(f"  {row['day']:>3} {row['money']:>8,.0f} {row['hands']:>5} "
                      f"{row['quads']:>5} {row['shed']:>4} | "
                      + " ".join(f"{cnt[c]:>4}" for c in CROPS + list(ANIMALS)))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--compact", action="store_true")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        files += sorted(glob.glob(p))
    for f in files:
        profile(f, compact=args.compact)


if __name__ == "__main__":
    main()
