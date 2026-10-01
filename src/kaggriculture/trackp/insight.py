"""P0.2 -- the weekly insight run: four correlation passes over trace v2.

  pass 1  marginal     r(feature, win) and r(feature, bank) + elite-vs-mid
                       effect sizes (elite = winning seats in the top bank
                       decile; mid = the 30th-70th percentile band)
  pass 2  differential f(mine) - f(theirs) for every play feature, incl.
                       the demand-match score -- the only form shared
                       conditions can pay through
  pass 3  interaction  win-rate lift P(win | plan, regime) vs P(win | plan);
                       regimes = shop-draw partition; binomial significance
                       with Benjamini-Hochberg FDR over the grid
  pass 4  structure    opening families (families.py), copy-graph summary

Dated output: models/trackp/insight/<YYYY-MM-DD>.json -- a series, so drift
in what pays is itself measurable week over week.

Usage: python src/trackp/insight.py [--limit N]
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os

import numpy as np

try:
    from . import common, families, trace_v2
except ImportError:
    import sys
    from kaggriculture.trackp import common, families, trace_v2

_ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
OUT_DIR = os.path.join(common.MODELS, "insight")
os.makedirs(OUT_DIR, exist_ok=True)

PARTITION = {"YARN_STORE": "wool", "PET_CAFE": "carrot",
             "SMOOTHIE_SHOP": "dairy", "ICE_CREAM_SHOP": "dairy",
             "PIZZA_SHOP": "dairy", "BAKERY": "neutral",
             "BRUNCH_SPOT": "neutral", "FARMERS_MARKET": "neutral"}


def play_features(X: np.ndarray) -> dict:
    """Aggregate one seat's episode (seat-viewed trace) into play features."""
    T = X.shape[0]
    days = T // 24
    g = lambda name: X[:, _ix[name]]  # noqa: E731
    money = g("m_money")
    gap = money - g("t_money")
    out = {}
    over10k = np.nonzero(money >= 10000)[0]
    out["first_10k_day"] = float(over10k[0] / 24) if len(over10k) else 30.0
    # intraday max hands per day, then episode max (the day-boundary fix)
    hands = g("m_hands")
    out["max_hands"] = float(max(hands[d * 24:(d + 1) * 24].max()
                                 for d in range(days)))
    plants = sum(g(f"m_crop_{c}") for c in common.CROP_NAMES)
    at_risk = g("m_at_risk_unwatered")
    with np.errstate(divide="ignore", invalid="ignore"):
        dry = np.where(plants > 0, at_risk / np.maximum(plants, 1), 0.0)
    out["dry_share"] = float(dry[plants > 0].mean()) if (
        plants > 0).any() else 0.0
    out["cash_day5"] = float(money[min(T - 1, 5 * 24)])
    out["gap_day10"] = float(gap[min(T - 1, 10 * 24)])
    out["gap_day16"] = float(gap[min(T - 1, 16 * 24)])
    out["gap_day23"] = float(gap[min(T - 1, 23 * 24)])
    out["lead_share"] = float((gap > 0).mean())
    out["unsold_shed_end"] = float(X[-1, _ix["pm_shed_total"]])
    out["demand_match_mean"] = float(g("m_demand_match")[72:].mean())
    out["animals_mean"] = float(sum(g(f"m_anim_{a}")
                                    for a in common.ANIMAL_NAMES).mean())
    out["weeds_mean"] = float(g("m_weeds").mean())
    out["quads_final"] = float(X[-1, _ix["m_quads"]])
    out["hires_total"] = float(g("am_hire").sum())
    out["sell_units_total"] = float(sum(
        g(f"am_sell_{p}") for p in common.PRODUCTS).sum())
    # mean sell day (pacing)
    sells = sum(g(f"am_sell_{p}") for p in common.PRODUCTS)
    tot = sells.sum()
    out["mean_sell_day"] = float((sells * np.arange(T)).sum()
                                 / (24 * max(1.0, tot)))
    out["fert_collected"] = float(g("am_collect_fert").sum())
    for c in common.CROP_NAMES:
        out[f"share_{c}"] = float(g(f"m_crop_{c}")[120:].mean())

    # ---- second-generation aggregates (2026-08-16) ----
    # execution alpha: units-weighted price captured vs the best price in a
    # +/-1-day window around each sale -- "grew right" vs "sold badly".
    alpha_num = alpha_den = 0.0
    for p in common.PRODUCTS:
        units = g(f"am_sell_{p}")
        if units.sum() <= 0:
            continue
        px = g(f"mpx_{p}")
        for t in np.nonzero(units > 0)[0]:
            lo, hi = max(0, t - 24), min(T, t + 24)
            best = float(px[lo:hi].max())
            alpha_num += float(units[t]) * (float(px[t]) - best)
            alpha_den += float(units[t]) * max(1.0, best)
    out["exec_alpha"] = alpha_num / alpha_den if alpha_den else 0.0

    # demand-capture share: our market share of sales in DEMANDED products,
    # weighted by town drain (demand-match upgraded to market share).
    cap_num = cap_den = 0.0
    for p in common.PRODUCTS:
        drain = float(g(f"drain_{p}")[-1])
        if drain <= 0:
            continue
        mine_u = float(g(f"am_sell_{p}").sum())
        theirs_u = float(g(f"at_sell_{p}").sum())
        if mine_u + theirs_u > 0:
            cap_num += drain * mine_u / (mine_u + theirs_u)
            cap_den += drain
    out["demand_capture"] = cap_num / cap_den if cap_den else 0.5

    # per-product income gap (mine - theirs), price-at-turn weighted
    for p in common.PRODUCTS:
        px = g(f"mpx_{p}")
        mine_inc = float((g(f"am_sell_{p}") * px).sum())
        their_inc = float((g(f"at_sell_{p}") * px).sum())
        out[f"incgap_{p}"] = mine_inc - their_inc

    # weed-luck proxy + idle share + replant latency + sell chunk size
    out["weed_diff"] = float((g("t_weeds") - g("m_weeds")).mean())
    units_avail = 1.0 + g("m_hands")
    ops = sum(g(f"am_{k}") for k in ("water", "harvest", "feed", "care",
                                     "collect_fert", "fertilize", "dig",
                                     "build_coop", "build_pasture")) \
        + sum(g(f"am_plant_{c}") for c in common.CROP_NAMES)
    out["idle_frac"] = float((ops < units_avail).mean())
    harv = np.nonzero(g("am_harvest") > 0)[0]
    plants_t = np.nonzero(sum(g(f"am_plant_{c}")
                              for c in common.CROP_NAMES) > 0)[0]
    lat = []
    for h in harv[:200]:
        nxt = plants_t[plants_t > h]
        if len(nxt):
            lat.append(float(nxt[0] - h))
    out["replant_latency"] = float(np.mean(lat)) if lat else 48.0
    sells_all = sum(g(f"am_sell_{p}") for p in common.PRODUCTS)
    chunk = sells_all[sells_all > 0]
    out["sell_chunk_mean"] = float(chunk.mean()) if len(chunk) else 0.0
    return out


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s = math.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def _bh_fdr(pvals: list, q: float = 0.10) -> list:
    """Benjamini-Hochberg: returns the significant indices."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    keep = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            keep = rank
    return order[:keep] if keep > 0 else []


def _binom_two_sided(k: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    mean = n * p0
    if n > 300:
        # normal approximation with continuity correction (log-space exact
        # overflows nothing, but the approximation is plenty at this n)
        sd = math.sqrt(n * p0 * (1 - p0))
        if sd == 0:
            return 1.0
        z = (abs(k - mean) - 0.5) / sd
        return min(1.0, 2 * 0.5 * math.erfc(max(0.0, z) / math.sqrt(2)))
    # exact two-sided via doubling the smaller tail, in log space
    def _logpmf(i):
        return (math.lgamma(n + 1) - math.lgamma(i + 1)
                - math.lgamma(n - i + 1)
                + i * math.log(p0) + (n - i) * math.log(1 - p0))
    if k >= mean:
        tail = sum(math.exp(_logpmf(i)) for i in range(k, n + 1))
    else:
        tail = sum(math.exp(_logpmf(i)) for i in range(0, k + 1))
    return min(1.0, 2 * tail)


def run(engine: str = None, limit: int = 0) -> dict:
    if engine is None:
        engine = common.engine_version()  # follow the ladder
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    rows = []          # per seat: features, diff-features, win, bank, regime
    for f in files:
        try:
            X0, meta = trace_v2.load(f)
        except Exception:
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        # regime from the observed draw (first two unlocks)
        draw = []
        for s in common.SHOPS_SORTED:
            d0 = X0[-1, _ix[f"shopday_{s}"]]
            if d0 >= 0:
                draw.append((d0, s))
        draw.sort()
        parts = [PARTITION.get(s, "neutral") for _, s in draw[:2]]
        regime = next((p for p in ("wool", "carrot", "dairy")
                       if p in parts), "neutral")
        feats = [play_features(trace_v2.seat_view(X0, s)) for s in (0, 1)]
        for seat in (0, 1):
            me, them = feats[seat], feats[1 - seat]
            win = 1.0 if banks[seat] > banks[1 - seat] else (
                0.5 if banks[seat] == banks[1 - seat] else 0.0)
            rows.append({"f": me,
                         "d": {k: me[k] - them[k] for k in me},
                         "win": win, "bank": float(banks[seat]),
                         "regime": regime})
    n = len(rows)
    if n < 50:
        raise SystemExit(f"too few plays: {n}")
    wins = np.array([r["win"] for r in rows])
    banks = np.array([r["bank"] for r in rows])
    names = list(rows[0]["f"].keys())

    # pass 1: marginal + elite contrast
    p90 = np.quantile(banks, 0.90)
    p30, p70 = np.quantile(banks, 0.30), np.quantile(banks, 0.70)
    elite = [i for i in range(n) if banks[i] >= p90 and wins[i] == 1.0]
    mid = [i for i in range(n) if p30 <= banks[i] <= p70]
    pass1 = {}
    for k in names:
        v = np.array([r["f"][k] for r in rows])
        pass1[k] = {"r_win": round(_pearson(v, wins), 3),
                    "r_bank": round(_pearson(v, banks), 3),
                    "elite_mid_d": round(_cohens_d(v[elite], v[mid]), 3),
                    "elite_mean": round(float(v[elite].mean()), 2)
                    if elite else None,
                    "mid_mean": round(float(v[mid].mean()), 2)
                    if mid else None}

    # pass 2: differential
    pass2 = {}
    for k in names:
        v = np.array([r["d"][k] for r in rows])
        pass2[k] = {"r_win": round(_pearson(v, wins), 3)}

    # pass 3: interaction/lift with BH FDR
    # plan = dominant crop share; regime = shop partition
    plans = []
    for r in rows:
        shares = {c: r["f"][f"share_{c}"] for c in common.CROP_NAMES}
        plans.append(max(shares, key=shares.get))
    cells = {}
    for i in range(n):
        cells.setdefault((plans[i], rows[i]["regime"]), []).append(wins[i])
    base = {}
    for i in range(n):
        base.setdefault(plans[i], []).append(wins[i])
    grid, pvals = [], []
    for (plan, regime), ws in cells.items():
        if len(ws) < 20:
            continue
        p_cond = float(np.mean(ws))
        p_base = float(np.mean(base[plan]))
        k = int(sum(1 for w in ws if w == 1.0))
        nn = len([w for w in ws if w != 0.5])
        pv = _binom_two_sided(k, nn, max(0.01, min(0.99, p_base)))
        grid.append({"plan": plan, "regime": regime, "n": len(ws),
                     "p_win_cond": round(p_cond, 3),
                     "p_win_base": round(p_base, 3),
                     "lift": round(p_cond - p_base, 3), "p": round(pv, 5)})
        pvals.append(pv)
    sig = _bh_fdr(pvals) if pvals else []
    for i, gcell in enumerate(grid):
        gcell["significant_fdr10"] = i in sig

    # pass 4: structure
    fam = families.build(engine=engine) if not os.path.exists(
        families.FAMILIES) else json.load(
        open(families.FAMILIES, encoding="utf-8"))["summary"]

    report = {"date": dt.date.today().isoformat(), "plays": n,
              "engine": engine,
              "pass1_marginal": pass1, "pass2_differential": pass2,
              "pass3_interaction": sorted(grid, key=lambda c: -abs(c["lift"])),
              "pass4_structure": fam}
    out = os.path.join(OUT_DIR, f"{dt.date.today().isoformat()}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    top_d = sorted(names, key=lambda k: -abs(pass1[k]["elite_mid_d"]))[:8]
    print(json.dumps({"plays": n, "out": out,
                      "top_elite_markers": {k: pass1[k]["elite_mid_d"]
                                            for k in top_d},
                      "top_differential": sorted(
                          names, key=lambda k: -abs(pass2[k]["r_win"]))[:6],
                      "sig_interactions": sum(
                          1 for c in grid if c["significant_fdr10"])},
                     indent=1))
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(limit=a.limit)
