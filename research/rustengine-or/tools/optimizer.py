#!/usr/bin/env python
"""Shared economic-optimizer helpers (path-item a). Keep GREEDY routing; the
value here is the SELL / SIZE / FEED policy on the already-exact market model.

DON'T-DUMP: sell only units whose marginal engine price >= a threshold (withhold
the near-$1 tail instead of dumping). RIGHT-SIZE: pick herd/tiles to a product's
absorbable demand, not max tiles. GROW-FEED: feed animals from grown wheat.
"""
import math
from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS, PRICE_FLOOR


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "hinge":
        return x if T <= 0 else (x / T + 8.0 * max(0.0, x / T - 1.0) ** 2)
    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),
            "log": math.log(1.0 + x), "log10": math.log10(1.0 + x)}[f]


def market_price(item, inv):
    p = MARKET_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], T, T)
        price = base + amp * _shape(p["below_func"], I0 - inv, T)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], T, T)
        price = base - amp * _shape(p["above_func"], inv - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def exact_rev(item, q, v0):
    inv, tot = v0, 0
    for _ in range(q):
        pr = market_price(item, inv)
        tot += pr
        if pr > 1:
            inv += 1
    return tot


def dont_dump_qty(item, inv, have, thresh):
    """Units to SELL now: walk the staircase from the current market `inv`, count
    units whose marginal price >= thresh, capped by what we `have`. Withholds the
    cheap tail for a future scarce day (town re-drains inventory). This is the
    intertemporal don't-dump rule in myopic form."""
    n = 0
    cur = inv
    while n < have:
        pr = market_price(item, cur)
        if pr < thresh:
            break
        n += 1
        if pr > 1:
            cur += 1
    return n


def default_thresh(item, frac=0.5):
    return max(2, int(MARKET_PARAMS[item]["base"] * frac))
