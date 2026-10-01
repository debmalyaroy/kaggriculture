"""Train the search surrogate: schedule features + opponent -> margin.

Ridge regression (pure numpy, no external deps) over the episode sink.
Never ships in the agent -- it pre-ranks candidate genomes inside
train_arms so real episodes are spent only on the promising ones.
Refuses to train below `--min-rows` sink rows on the current engine;
built now, activates itself as the sink fills.

    python -m kaggriculture.train.train_surrogate
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

import numpy as np

import kaggriculture.data.sink as sink  # noqa: E402

OUT = os.path.join(ROOT, "models", "v22", "surrogate.json")


def featurize(row):
    """Legacy filename-hash features. Retired 2026-08-13: measured R^2 0.04
    / Spearman 0.27 -- a matchup lookup that generalizes to nothing unseen.
    Kept only as the fallback for rows without route identity."""
    agent = os.path.basename(str(row.get("agent") or ""))
    opp = os.path.basename(str(row.get("opponent") or ""))
    h = [0.0] * 32
    for token, weight in ((agent, 1.0), (opp, -1.0)):
        for i, ch in enumerate(token[:32]):
            h[(i * 31 + ord(ch)) % 32] += weight
    return h + [float(row.get("seat") or 0)]


_ROUTE_VECS = {}


def route_features_of(rid):
    if rid in _ROUTE_VECS:
        return _ROUTE_VECS[rid]
    try:
        import kaggriculture.data.features as features
        import kaggriculture.data.routes as R
        v, _ = features.route_features(R.load_route(rid))
    except Exception:                                              # noqa: BLE001
        v = None
    _ROUTE_VECS[rid] = v
    return v


def featurize_real(row):
    """Real features: both sides' route signatures + seat. Uses the identity
    fields sink.record() persists since 2026-08-13, falling back to sink's
    own filename/docstring resolver for older rows. Measured Spearman 0.745
    vs 0.272 for the hash features (models/lab/surrogate_v2.json)."""
    ra = row.get("agent_route") or sink._route_identity(row.get("agent"))
    ro = (row.get("opponent_route")
          or sink._route_identity(row.get("opponent")))
    va = route_features_of(ra) if ra else None
    vo = route_features_of(ro) if ro else None
    if va is None or vo is None:
        return None
    m = min(len(va), len(vo))
    return list(va[:m]) + list(vo[:m]) + [float(row.get("seat") or 0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-rows", type=int, default=2000)
    args = ap.parse_args()

    fp = sink._fingerprint()
    rows = [r for r in sink.rows() if r.get("engine") == fp
            and r.get("bank") is not None and r.get("opp_bank") is not None]
    # Prefer identity-carrying rows with REAL route features (sink.record
    # persists agent_route/opponent_route since 2026-08-13); fall back to
    # the retired hash features only if too few exist yet.
    feats, kept = [], []
    for r in rows:
        v = featurize_real(r)
        if v is not None:
            feats.append(v)
            kept.append(r)
    real = len(feats) >= max(300, args.min_rows // 4)
    if not real:
        if len(rows) < args.min_rows:
            print(f"waiting for data: {len(rows)} current-engine rows "
                  f"({len(feats)} with route identity). The sink fills with "
                  f"every evaluate/tournament/arm run; no action required.")
            return 0
        feats = [featurize(r) for r in rows]
        kept = rows
    d = min(len(v) for v in feats)
    X = np.array([v[:d] for v in feats])
    # TARGET IS THE SCORE, NOT THE MARGIN (2026-08-13). The ladder pays
    # win/draw/loss, so regressing dollars optimises a currency the
    # competition does not use: a model can win the margin R^2 by predicting
    # blowouts accurately while being useless about the close games that
    # actually decide the ladder -- and 27.6% of our 1,748 real games are
    # decided by under $3,000. Predicting P(win) puts the model's capacity
    # where the reward step is.
    import kaggriculture.measure.win_metric as WM
    y = np.array([WM.score(r["bank"], r["opp_bank"]) for r in kept])
    y_margin = np.array([float(r["bank"]) - float(r["opp_bank"])
                         for r in kept])
    print(f"features: {'REAL route signatures' if real else 'legacy hash'} "
          f"({len(kept)} rows, {X.shape[1]} dims)")
    n = len(X)
    cut = int(0.9 * n)
    reg = 1.0
    A = X[:cut].T @ X[:cut] + reg * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[:cut].T @ y[:cut])
    pred = X[cut:] @ w
    ss_res = float(((y[cut:] - pred) ** 2).sum())
    ss_tot = float(((y[cut:] - y[cut:].mean()) ** 2).sum()) or 1.0
    r2 = 1 - ss_res / ss_tot

    # The metrics that matter for a win-denominated target: can it RANK a
    # winner above a loser (AUC), and does thresholding it call the game right
    # (accuracy)? R^2 on a 0/0.5/1 target is reported but is not the criterion.
    yt = y[cut:]
    pos = pred[yt == 1.0]
    neg = pred[yt == 0.0]
    if len(pos) and len(neg):
        auc = float((pos[:, None] > neg[None, :]).mean()
                    + 0.5 * (pos[:, None] == neg[None, :]).mean())
    else:
        auc = float("nan")
    acc = float((((pred >= 0.5).astype(float)) == (yt >= 0.5)).mean())
    # margin R^2 kept as a DIAGNOSTIC so the change of target is visible in
    # the logs next to the historical 0.801 figure -- not as a gate.
    wm = np.linalg.solve(A, X[:cut].T @ y_margin[:cut])
    pm = X[cut:] @ wm
    mr2 = 1 - (float(((y_margin[cut:] - pm) ** 2).sum())
               / (float(((y_margin[cut:] - y_margin[cut:].mean()) ** 2).sum())
                  or 1.0))
    json.dump({"w": [round(float(v), 6) for v in w], "r2_holdout": r2,
               "target": "score", "auc_holdout": auc, "acc_holdout": acc,
               "margin_r2_diagnostic": mr2,
               "n_rows": n, "engine": fp},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"surrogate trained on {n} rows (target = SCORE): holdout "
          f"AUC {auc:.3f}, accuracy {acc:.3f}, R^2 {r2:.3f} -> {OUT}")
    print(f"  (margin R^2 {mr2:.3f}, diagnostic only -- the ladder pays wins)")
    if not (auc == auc) or auc < 0.6:
        print("WARNING: AUC below 0.6 -- cannot rank candidates by win "
              "probability; train_arms should ignore the surrogate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
