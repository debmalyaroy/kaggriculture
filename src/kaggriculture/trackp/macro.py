"""P3.2 -- the macro action space and the decision dataset builder.

The macro policy decides ~30-38 times per game (each day boundary, plus a
re-plan when a shop unlocks). One decision = discrete bins:

  mix_<8 products>  0-4   production-mix weight
  labour            0-6   hands target bin -> (0,2,3,5,7,9,12)
  invest            0-3   investment depth
  pace              0-3   sell pacing (0 hold-biased .. 3 dump-now)

The RUNTIME feature vector is planner_template._l1_features (imported here --
trackp's own code, so no drift is possible). For TRAINING, the same 32 dims
are reconstructed from trace v2 rows; test_trackp pins the two paths equal
on real replays.

Dataset: (features, bins, return) per decision, return gap-shaped
(win + lambda * clipped gap), split BY EPISODE. Elite filter for IQL.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

try:
    from . import common, trace_v2
    from . import planner_template as tpl
except ImportError:  # script mode
    import sys
    from kaggriculture.trackp import common, trace_v2
    from kaggriculture.trackp import planner_template as tpl

HEADS = [("mix_WHEAT", 5), ("mix_CARROT", 5), ("mix_TOMATO", 5),
         ("mix_STRAWBERRY", 5), ("mix_MELON", 5), ("mix_EGG", 5),
         ("mix_MILK", 5), ("mix_WOOL", 5),
         ("labour", 7), ("invest", 4), ("pace", 4)]
N_LOGITS = sum(k for _, k in HEADS)
FEAT_DIM = 41  # 32 base + 9 opponent-embedding dims (dump rate/product)
LAB_BINS = (0, 2, 3, 5, 7, 9, 12)
GAP_LAMBDA = 0.3
GAP_CLIP = 10000.0

_ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}


def l1_features_obs(obs: dict) -> list:
    """Runtime path -- exactly what the shipped agent computes."""
    return tpl._l1_features(obs)


def l1_features_trace(row: np.ndarray) -> list:
    """Training path -- the same 32 dims from a (seat-viewed) trace row."""
    g = lambda name: float(row[_ix[name]])  # noqa: E731
    day = g("day")
    money = g("m_money")
    gap = money - g("t_money")
    shops = sum(g(f"shopn_{s}") for s in common.SHOPS_SORTED)
    f = [day / 29.0,
         min(1.0, money / 20000.0),
         max(-1.0, min(1.0, gap / 10000.0)),
         shops / 8.0,
         g("m_hands") / 12.0,
         g("m_quads") / 4.0,
         g("t_quads") / 4.0]
    for p in common.PRODUCTS:
        # trace drain_* is per DAY; the runtime feature uses shop-only drain
        # (6 fires/day), i.e. drain minus the daily town-center unit.
        d = g(f"drain_{p}") - (1.0 if p != "FERTILIZER" else 0.0)
        f.append(min(1.0, d / 15.0))
    for p in common.PRODUCTS:
        base = common.MARKET_PARAMS[p][0]
        cur = g(f"mpx_{p}")
        f.append(max(-1.0, min(1.0, (cur - base) / max(1.0, base))))
    m_plants = sum(g(f"m_crop_{c}") for c in common.CROP_NAMES)
    t_plants = sum(g(f"t_crop_{c}") for c in common.CROP_NAMES)
    m_anim = sum(g(f"m_anim_{a}") for a in common.ANIMAL_NAMES)
    t_anim = sum(g(f"t_anim_{a}") for a in common.ANIMAL_NAMES)
    f.append(min(1.0, m_plants / 40.0))
    f.append(min(1.0, t_plants / 40.0))
    f.append(min(1.0, m_anim / 10.0))
    f.append(min(1.0, t_anim / 10.0))
    f.append(min(1.0, g("m_weeds") / 20.0))
    f.append(min(1.0, g("step") / 719.0))
    f.append(1.0)
    # Opponent embedding: their cumulative sells/day so far (trace ground
    # truth; the runtime path uses the market-delta residual, which recovers
    # the same quantity up to clamping noise).
    return f  # caller appends opponent dims via l1_features_trace_full


def l1_features_trace_full(X: np.ndarray, t: int) -> list:
    """The full 41-dim vector at row t: base 32 + opponent dump rates."""
    f = l1_features_trace(X[t])
    days_seen = max(1.0, float(X[t, _ix["step"]]) / 24.0)
    for p in common.PRODUCTS:
        cum = float(X[: t + 1, _ix[f"at_sell_{p}"]].sum())
        f.append(min(1.0, cum / days_seen / 15.0))
    return f


def bins_to_macro(bins: dict) -> dict:
    """Bin indices -> the macro dict the executor consumes (identity today)."""
    return dict(bins)


def infer_day_bins(X: np.ndarray, day: int) -> dict:
    """What macro decision did this (seat-viewed) trace's player take on
    `day`? Inferred from the day's aggregated actions."""
    lo, hi = day * common.TURNS_PER_DAY, (day + 1) * common.TURNS_PER_DAY
    seg = X[lo:min(hi, X.shape[0])]
    if seg.shape[0] == 0:
        return {}
    out = {}
    # mix: plants planted this day per crop; animals bought
    for c in common.CROP_NAMES:
        n = float(seg[:, _ix[f"am_plant_{c}"]].sum())
        out[f"mix_{c}"] = int(min(4, round(n / 2.0)))
    for a, prod in (("GOOSE", "EGG"), ("COW", "MILK"), ("SHEEP", "WOOL")):
        n = float(seg[:, _ix[f"am_buyanim_{a}"]].sum())
        out[f"mix_{prod}"] = int(min(4, n * 2))
    # labour: max hands seen during the day (the intraday-max fix)
    hands = float(seg[:, _ix["m_hands"]].max())
    lb = 0
    for i, v in enumerate(LAB_BINS):
        if hands >= v:
            lb = i
    out["labour"] = lb
    # invest: land + builds + animal buys this day
    spend = (float(seg[:, _ix["am_buy_land"]].sum()) * 2
             + float(seg[:, _ix["am_build_coop"]].sum())
             + float(seg[:, _ix["am_build_pasture"]].sum())
             + sum(float(seg[:, _ix[f"am_buyanim_{a}"]].sum())
                   for a in common.ANIMAL_NAMES))
    out["invest"] = int(min(3, spend))
    # pace: units sold this day relative to shed stock at day start
    sold = sum(float(seg[:, _ix[f"am_sell_{p}"]].sum())
               for p in common.PRODUCTS)
    stock = max(1.0, float(X[lo, _ix["pm_shed_total"]]) + sold)
    frac = sold / stock
    out["pace"] = int(min(3, round(frac * 3.0)))
    return out


def decision_days(X: np.ndarray) -> list:
    """Day starts, plus the day after each shop unlock (re-plan points)."""
    days = list(range(int(X[-1, _ix["day"]]) + 1))
    return days


def build_dataset(engine=None, elite_gap=0.0, limit=0,
                  out_name="macro_dataset.npz") -> dict:
    """(features, bins, return, episode, seat, day) over all traces.

    elite_gap: keep only seats that WON by at least this final gap
    (0 = keep every seat; IQL uses a positive filter downstream).
    """
    if engine is None:
        engine = common.engine_version()  # follow the ladder
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    F, B, R, EP, SEAT, DAY = [], [], [], [], [], []
    n_ep = 0
    for f in files:
        try:
            X0, meta = trace_v2.load(f)
        except Exception:
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        n_ep += 1
        for seat in (0, 1):
            gap = banks[seat] - banks[1 - seat]
            if elite_gap > 0 and gap < elite_gap:
                continue
            win = 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
            ret = win + GAP_LAMBDA * max(-1.0, min(1.0, gap / GAP_CLIP))
            X = trace_v2.seat_view(X0, seat)
            for day in decision_days(X):
                t = day * common.TURNS_PER_DAY
                if t >= X.shape[0]:
                    break
                bins = infer_day_bins(X, day)
                if not bins:
                    continue
                F.append(l1_features_trace_full(X, t))
                B.append([bins[name] for name, _ in HEADS])
                R.append(ret)
                EP.append(int(meta.get("episode") or 0))
                SEAT.append(seat)
                DAY.append(day)
    out = os.path.join(common.MODELS, out_name)
    np.savez_compressed(out,
                        F=np.asarray(F, dtype=np.float32),
                        B=np.asarray(B, dtype=np.int16),
                        R=np.asarray(R, dtype=np.float32),
                        EP=np.asarray(EP, dtype=np.int64),
                        SEAT=np.asarray(SEAT, dtype=np.int8),
                        DAY=np.asarray(DAY, dtype=np.int16),
                        heads=json.dumps(HEADS))
    rep = {"episodes": n_ep, "decisions": len(F), "out": out,
           "elite_gap": elite_gap}
    print(json.dumps(rep))
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--elite-gap", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="macro_dataset.npz")
    a = ap.parse_args()
    build_dataset(elite_gap=a.elite_gap, limit=a.limit, out_name=a.out)
