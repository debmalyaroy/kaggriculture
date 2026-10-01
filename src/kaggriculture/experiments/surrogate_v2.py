"""Surrogate v2: LightGBM quantile regression on REAL features.

The production surrogate (train_surrogate.py) fits ridge on a 32-bucket hash
of the agents' FILENAMES -- a matchup lookup that generalizes to nothing
unseen, which defeats its purpose (pre-ranking candidates never played).
This version featurizes each sink row with the actual ROUTE FEATURES of both
sides where the paths are route-built agents (c_<route>.py candidates,
tape_episode-<id>-replay_s<seat>.py referees, v{N}_route.py releases), plus
seat, and fits LightGBM at the 0.5 and 0.8 quantiles: rank candidates by the
UPPER quantile so screening stays optimistic about uncertain ones.

Holdout = newest 10% of rows. Reports rank correlation (Spearman) of the
median prediction vs the realized margin -- the metric that matters for a
pre-ranker -- next to the production featurizer's ridge as baseline.

    python src/experiments/surrogate_v2.py
"""
from kaggriculture.paths import ROOT
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.sink as sink  # noqa: E402
import kaggriculture.train.train_surrogate as TS  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "surrogate_v2.json")

_PATTERNS = (re.compile(r"c_(\d{6,}_s[01])"),
             re.compile(r"tape_episode-(\d{6,})-replay_s([01])"),
             re.compile(r"(\d{6,}_s[01])"))


def route_id_of(path):
    base = os.path.basename(str(path or ""))
    for pat in _PATTERNS:
        m = pat.search(base)
        if m:
            if len(m.groups()) == 2:
                return f"{m.group(1)}_s{m.group(2)}"
            return m.group(1)
    return None


_CACHE = {}


def route_vec(rid):
    if rid in _CACHE:
        return _CACHE[rid]
    try:
        v, _ = features.route_features(R.load_route(rid))
    except Exception:                                              # noqa: BLE001
        v = None
    _CACHE[rid] = v
    return v


def main():
    fp = sink._fingerprint()
    rows = [r for r in sink.rows() if r.get("engine") == fp
            and r.get("bank") is not None and r.get("opp_bank") is not None]
    X, Xh, y = [], [], []
    skipped = 0
    for r in rows:
        ra, ro = route_id_of(r.get("agent")), route_id_of(r.get("opponent"))
        va = route_vec(ra) if ra else None
        vo = route_vec(ro) if ro else None
        if va is None or vo is None:
            skipped += 1
            continue
        m = min(len(va), len(vo))
        X.append(list(va[:m]) + list(vo[:m])
                 + [float(r.get("seat") or 0)])
        Xh.append(TS.featurize(r))
        y.append(float(r["bank"]) - float(r["opp_bank"]))
    if not X:
        print("no rows with route-resolvable agents; nothing to fit")
        return 1
    m = min(len(x) for x in X)
    X = np.array([x[:m] for x in X])
    Xh = np.array(Xh)
    y = np.array(y)
    print(f"{len(y)} usable rows ({skipped} skipped: not route-built), "
          f"{X.shape[1]} real dims vs {Xh.shape[1]} hash dims")

    cut = int(0.9 * len(y))
    from scipy.stats import spearmanr

    # baseline: production ridge on hash features
    A = Xh[:cut].T @ Xh[:cut] + 1.0 * np.eye(Xh.shape[1])
    w = np.linalg.solve(A, Xh[:cut].T @ y[:cut])
    rho_h = spearmanr(Xh[cut:] @ w, y[cut:]).statistic

    import lightgbm as lgb
    preds = {}
    for q in (0.5, 0.8):
        gbm = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                n_estimators=300, learning_rate=0.06,
                                max_depth=5, num_leaves=31, n_jobs=4,
                                verbosity=-1)
        gbm.fit(X[:cut], y[:cut])
        preds[q] = gbm.predict(X[cut:])
    rho = spearmanr(preds[0.5], y[cut:]).statistic
    cover = float((y[cut:] <= preds[0.8]).mean())
    print(f"holdout Spearman: real-features LGBM {rho:.3f}  "
          f"hash-ridge baseline {rho_h:.3f}")
    print(f"q80 empirical coverage {cover:.3f} (target ~0.80)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"rows": len(y), "skipped": skipped,
               "spearman_lgbm": float(rho),
               "spearman_hash_ridge": float(rho_h),
               "q80_coverage": cover},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
