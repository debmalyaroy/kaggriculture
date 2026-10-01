"""C2: does the MARKET REGIME improve opponent identification? Measured.

The identifier reads 275 dims of prefix ACTION features and nothing about the
market -- yet corr(late-game mean price, our bank) is +0.88, the strongest
signal in the dataset, and it is not a feature anywhere. This was data-gated
until the 2026-08-14 historical backfill: market paths live in the per-turn
traces, which now cover ~2,950 episodes instead of 22.

Protocol (paired, so the comparison is the only thing that moves):

* Subset to routes whose episode has a trace -- both arms train on EXACTLY the
  same routes, labels, split and dropout masks.
* Baseline arm: the production feature set (prefix vector + t/720).
* Augmented arm: + 27 market dims at the checkpoint -- price level per product
  (/base), price trend over the trailing 4 game-days, and market inventory
  (/equilibrium). All three are PUBLIC in the live observation, so there is no
  train/serve skew: the agent can reconstruct every added dim at runtime.
* Dropout masks apply to the ACTION prefix only. Dropout mimics floored-sale
  invisibility, and prices are never invisible.
* Same held-out-DAY split as production: the newest archive day is the test
  set, because tomorrow is the only day the model is paid on.

Adopt into train_identifier.py ONLY on a held-out-day improvement.

    python src/experiments/identifier_market_features.py
"""
from kaggriculture.paths import ROOT
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT
import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402
import kaggriculture.data.turn_features as TF  # noqa: E402

TREND_STEPS = 96            # 4 game-days at 24 steps/day
N_MARKET = 27               # 9 price + 9 trend + 9 inventory


def market_at(trace_arr, t):
    """The 27 public market dims at checkpoint t, from either seat's trace
    (prices and inventory are shared state, so seat choice cannot matter)."""
    row = trace_arr[min(t, len(trace_arr) - 1)]
    prev = trace_arr[max(0, min(t, len(trace_arr) - 1) - TREND_STEPS)]
    price = row[3:12]
    trend = row[3:12] - prev[3:12]
    inv = row[12:21]
    return np.concatenate([price, trend, inv]).tolist()


def build_paired():
    idx = R.load_index()
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    import kaggriculture.data.feature_cache as FC
    cache_route, cache_prefix = FC.build([r["id"] for r in rows],
                                         TI.PREFIX_TURNS)

    # Traces, loaded once per EPISODE (both seats share the market path).
    traces, recs, feats = {}, [], {}
    for rec in rows:
        rid = rec["id"]
        v = cache_route.get(rid)
        if v is None or cache_prefix.get(rid) is None:
            continue
        ep = rid.rsplit("_s", 1)[0]
        if ep not in traces:
            tr = TF.load(ep)
            traces[ep] = None if not tr else tr.get(0, tr.get(1))
        if traces[ep] is None:
            continue
        feats[rid] = v
        recs.append(rec)
    print(f"{len(recs)} routes with BOTH features and a market trace "
          f"(of {len(rows)} in the index)")

    # Production clustering, restricted to the subset (same LINK/MIN_FAMILY).
    reps, labels = [], {}
    for rec in recs:
        f = feats[rec["id"]]
        for fi, rep in enumerate(reps):
            if features.distance(f, rep) < TI.LINK:
                labels[rec["id"]] = fi
                break
        else:
            reps.append(f)
            labels[rec["id"]] = len(reps) - 1
    sizes = {}
    for v in labels.values():
        sizes[v] = sizes.get(v, 0) + 1
    keep = {fi for fi, n in sizes.items() if n >= TI.MIN_FAMILY}
    remap = {fi: i for i, fi in enumerate(sorted(keep))}
    novel = len(remap)
    print(f"{len(reps)} raw clusters -> {len(remap)} families + NOVEL")

    rng = random.Random(11)
    Xb, Xa, y, split = [], [], [], []
    newest = max(r.get("date", "") for r in recs)
    for rec in recs:
        cls = remap.get(labels[rec["id"]], novel)
        pvecs = cache_prefix[rec["id"]]
        ep = rec["id"].rsplit("_s", 1)[0]
        arr = traces[ep]
        for ti, t in enumerate(TI.PREFIX_TURNS):
            base_vec = pvecs[ti].tolist()
            mkt = market_at(arr, t)
            for p in TI.DROPOUT:
                # ONE mask, applied to the action prefix of BOTH arms -- the
                # market dims are public and never dropped.
                masked = [v * (0 if rng.random() < p else 1)
                          for v in base_vec]
                Xb.append(masked + [t / 720.0])
                Xa.append(masked + mkt + [t / 720.0])
                y.append(cls)
                split.append("test" if rec.get("date") == newest else "train")
    Xb = np.array(Xb, dtype=np.float64)
    Xa = np.array(Xa, dtype=np.float64)
    y = np.array(y)
    tr = np.array(split) == "train"
    print(f"{len(Xb)} rows ({tr.sum()} train / {(~tr).sum()} held-out day); "
          f"baseline {Xb.shape[1]} dims, augmented {Xa.shape[1]} dims, "
          f"{novel + 1} classes")
    return Xb, Xa, y, tr, novel + 1


def main():
    Xb, Xa, y, tr, k = build_paired()
    if (~tr).sum() < 200:
        print("REFUSING: held-out day has <200 rows -- rerun after a mine")
        return 0
    out = {}
    for name, X in (("baseline", Xb), ("+market", Xa)):
        W, b = TI.train(X[tr], y[tr], k)
        out[name] = (TI.accuracy(X[tr], y[tr], W, b),
                     TI.accuracy(X[~tr], y[~tr], W, b))
        print(f"{name:<10} train {out[name][0]:.4f}   "
              f"held-out-day {out[name][1]:.4f}")
    d = out["+market"][1] - out["baseline"][1]
    n_te = int((~tr).sum())
    # McNemar-style scale check: is the delta bigger than binomial noise on
    # the held-out rows? (rows are correlated within a route -- 18 per route
    # -- so divide n by 18 for an honest effective sample size.)
    import math
    se = math.sqrt(out["baseline"][1] * (1 - out["baseline"][1])
                   / max(1, n_te / 18))
    print(f"\nheld-out-day delta {d:+.4f} (route-level SE ~{se:.4f})")
    if d > 2 * se:
        print("VERDICT: market regime IMPROVES identification -- adopt into "
              "train_identifier.py (extend feature_cache + the agent's "
              "runtime prefix builder with the same 27 public dims)")
    elif d < -2 * se:
        print("VERDICT: market regime HURTS -- do not adopt; record and close")
    else:
        print("VERDICT: within noise -- do not adopt (the identifier is not "
              "the consumer that monetises the +0.88 price signal; the "
              "policy/surrogate path is)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
