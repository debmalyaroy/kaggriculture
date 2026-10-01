"""L2 -- price-at-harvest projector.

price_hat(product, days_ahead) = quote(product, projected_inventory) where

  projected_inventory = inv_now
                        - town_drain(observed shop draw) * days_ahead
                        + opp_sell_rate_blend * days_ahead
                        + own_planned_units

The opponent's selling is NOT directly observable, but its per-turn total is
EXACTLY recoverable from public state:

  opp_sells_p(t) = max(0, inv_p(t) - inv_p(t-1) + drain_p(t) - own_sells_p(t))

(town drain is deterministic given the draw; own sells are known; BUY_PRODUCT
volumes are tiny and clamped at 0). The projector blends that observed
cumulative dump curve with an optional family prior (family_dumps data,
passed in as DATA -- Track P imports no bandit/route code).

Validation (T8 gate): on real 1.32.6 traces, MAE at horizon vs realized
price, against the "current price" naive. Must beat the naive to be wired in.
"""
from __future__ import annotations

import glob
import json
import os

try:
    from . import common
except ImportError:  # script mode
    import sys
    from kaggriculture.trackp import common


class Projector:
    """Feed one observation per turn; ask for projected prices."""

    def __init__(self, family_prior=None):
        # family_prior: list of (step, product, qty) the opponent family is
        # expected to dump (already support-weighted by the caller).
        self.prior = family_prior or []
        self._last_inv = None
        self._opp_cum = {p: 0.0 for p in common.PRODUCTS}
        self._turns_seen = 0
        self._shops = []

    def observe(self, step: int, market_inv: dict, shops: list,
                own_sold_units: dict):
        """Call once per turn with the CURRENT observation."""
        self._shops = list(shops)
        inv = {p: float(market_inv.get(p, 0)) for p in common.PRODUCTS}
        if self._last_inv is not None:
            drain = self._turn_drain(step)
            for p in common.PRODUCTS:
                delta = inv[p] - self._last_inv[p]
                opp = delta + drain.get(p, 0.0) - float(
                    own_sold_units.get(p, 0))
                if opp > 0:
                    self._opp_cum[p] += opp
        self._last_inv = inv
        self._turns_seen += 1

    def _turn_drain(self, step: int) -> dict:
        d = {p: 0.0 for p in common.PRODUCTS}
        if step % common.TOWN_SHOP_SELL_INTERVAL == 0:
            for shop in self._shops:
                prods = common.SHOP_PRODUCTS.get(shop, [])
                mult = 2 if len(prods) == 1 else 1
                for p in prods:
                    d[p] += mult
        if step % common.TOWN_CENTER_SELL_INTERVAL == 0:
            for p in common.PRODUCTS:
                if p != "FERTILIZER":
                    d[p] += 1
        return d

    def opp_rate_per_day(self, product: str) -> float:
        """Observed opponent dump rate, units/day, over the episode so far."""
        days = max(1.0, self._turns_seen / common.TURNS_PER_DAY)
        return self._opp_cum[product] / days

    def project_inventory(self, product: str, days_ahead: float,
                          own_planned_units: float = 0.0,
                          step_now: int = 0) -> float:
        if self._last_inv is None:
            return common.MARKET_PARAMS[product][1]
        inv = self._last_inv[product]
        drain = common.town_drain_per_day(self._shops).get(product, 0)
        inv -= drain * days_ahead
        inv += self.opp_rate_per_day(product) * days_ahead
        # Family prior: expected dumps inside the window, minus what the
        # observed rate already accounts for (avoid double counting: prior
        # only ADDS the burst structure beyond the smooth rate).
        if self.prior:
            lo = step_now
            hi = step_now + days_ahead * common.TURNS_PER_DAY
            burst = sum(q for (s, p, q) in self.prior
                        if p == product and lo <= s < hi)
            smooth = self.opp_rate_per_day(product) * days_ahead
            inv += max(0.0, burst - smooth)
        inv += own_planned_units
        return inv

    def price_at(self, product: str, days_ahead: float,
                 own_planned_units: float = 0.0, step_now: int = 0) -> int:
        return common.quote(product, self.project_inventory(
            product, days_ahead, own_planned_units, step_now))


# ------------------------------------------------------------- validation --

def validate(horizon_days=(2, 4, 6), engine=None, limit=0) -> dict:
    """MAE of projected vs realized price on real traces, against the naive.

    Uses trace v2 files: global block carries market inv/prices, shops via
    shopn_*, and per-seat sell_* action fields give own sells.
    """
    if engine is None:
        engine = common.engine_version()  # follow the ladder
    import numpy as np
    try:
        from . import trace_v2
    except ImportError:
        from kaggriculture.trackp import trace_v2

    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    ix = {f: trace_v2.field_index(f) for f in trace_v2.FIELDS}
    inv_ix = [ix[f"minv_{p}"] for p in common.PRODUCTS]
    px_ix = [ix[f"mpx_{p}"] for p in common.PRODUCTS]
    shopn_ix = [ix[f"shopn_{s}"] for s in common.SHOPS_SORTED]
    sell_ix = [ix[f"am_sell_{p}"] for p in common.PRODUCTS]
    step_ix = ix["step"]

    err_proj = {p: [] for p in common.PRODUCTS}
    err_naive = {p: [] for p in common.PRODUCTS}
    n_ep = 0
    for f in files:
        X, meta = trace_v2.load(f)
        if engine and meta.get("engine") != engine:
            continue
        n_ep += 1
        T = X.shape[0]
        pr = Projector()
        for t in range(T):
            shops = []
            for si, s in enumerate(common.SHOPS_SORTED):
                shops += [s] * int(X[t, shopn_ix[si]])
            inv_now = {p: X[t, inv_ix[pi]]
                       for pi, p in enumerate(common.PRODUCTS)}
            own = {p: X[t, sell_ix[pi]]
                   for pi, p in enumerate(common.PRODUCTS)}
            pr.observe(int(X[t, step_ix]), inv_now, shops, own)
            if t % common.TURNS_PER_DAY != 0:
                continue
            for h in horizon_days:
                th = t + h * common.TURNS_PER_DAY
                if th >= T:
                    continue
                for pi, p in enumerate(common.PRODUCTS):
                    realized = X[th, px_ix[pi]]
                    proj = pr.price_at(p, float(h), 0.0, int(X[t, step_ix]))
                    naive = X[t, px_ix[pi]]
                    err_proj[p].append(abs(proj - realized))
                    err_naive[p].append(abs(naive - realized))
    rep = {"episodes": n_ep, "horizons": list(horizon_days), "per_product": {}}
    import statistics
    wins = 0
    for p in common.PRODUCTS:
        if not err_proj[p]:
            continue
        mp = statistics.fmean(err_proj[p])
        mn = statistics.fmean(err_naive[p])
        rep["per_product"][p] = {"mae_proj": round(mp, 2),
                                 "mae_naive": round(mn, 2),
                                 "n": len(err_proj[p]),
                                 "beats_naive": mp < mn}
        wins += mp < mn
    rep["products_beating_naive"] = wins
    rep["verdict"] = "PASS" if wins >= 6 else "FAIL"
    out = os.path.join(common.MODELS, "projector_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", default=None,
                    help="trace filter; default = the ACTIVE engine version")
    a = ap.parse_args()
    r = validate(engine=a.engine, limit=a.limit)
    print(json.dumps(r, indent=1))
