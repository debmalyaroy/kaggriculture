"""Trajectory rating scorer v0: route features -> predicted ladder rating.

Plan item 11 (docs/history/model-improvement-plan.md addendum). v0 is deliberately
light -- ridge on the existing 283-dim route signatures, single-core, no
parallel load (the box BSODs under sustained multi-core; see
docs/history/issues-and-improvements.md 2026-08-13) -- because the immediate value
is the COLLAPSE ALARM: score our own pair's freshly scraped games hourly and
flag a predicted-rating drop within hours instead of waiting a day for the
ladder rating to develop.

Labels: the leaderboard rating of the TEAM that played each route, joined at
mine time (route records carry team names; ratings from the live board).
Guardrail: never a gate -- ranks and warns only.

    python src/experiments/rating_scorer.py            # train + validate
    python src/experiments/rating_scorer.py --score X  # score route id(s)
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.episodes as E  # noqa: E402
import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "rating_scorer.json")


def board_ratings():
    rows = E.leaderboard_rows(verbose=False) or []
    return {r["team"]: float(r["score"]) for r in rows
            if r.get("team") and r.get("score")}


def dataset():
    ratings = board_ratings()
    idx = R.load_index()
    X, y, dates, ids = [], [], [], []
    for rec in idx["routes"].values():
        team = rec.get("team")
        if not team or team == "?" or team not in ratings:
            continue
        try:
            v, _ = features.route_features(R.load_route(rec["id"]))
        except Exception:                                          # noqa: BLE001
            continue
        X.append(v)
        y.append(ratings[team])
        dates.append(rec.get("date", ""))
        ids.append(rec["id"])
    m = min(len(v) for v in X)
    X = np.array([v[:m] for v in X], dtype=np.float64)
    return X, np.array(y), np.array(dates), ids, m


def fit(X, y, reg=30.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    A = Xn.T @ Xn + reg * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xn.T @ (y - y.mean()))
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "y0": float(y.mean())}


def predict(model, X):
    Xn = (np.asarray(X) - np.array(model["mu"])) / np.array(model["sd"])
    return Xn @ np.array(model["w"]) + model["y0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", nargs="*", default=None,
                    help="route id(s) to score with the saved model")
    args = ap.parse_args()

    if args.score:
        model = json.load(open(OUT, encoding="utf-8"))["model"]
        for rid in args.score:
            v, _ = features.route_features(R.load_route(rid))
            p = predict(model, [v[:len(model['mu'])]])[0]
            print(f"  {rid:<24} predicted rating {p:,.0f}")
        return 0

    X, y, dates, ids, dims = dataset()
    days = sorted(set(d for d in dates.tolist() if d))
    # test on the newest day with a SUBSTANTIVE, rating-diverse sample --
    # 2026-08-13's 51 routes spanned 62 rating points (the hourly harvest
    # samples near our own band) and standardization exploded on unseen
    # feature scales: Spearman 0.008 / MAE 2,771 said nothing about the
    # model, only about the fold.
    test_day = None
    for d in reversed(days):
        m = dates == d
        if m.sum() >= 150 and float(np.ptp(y[m])) >= 300:
            test_day = d
            break
    test_day = test_day or days[-1]
    te = dates == test_day
    tr = ~te
    print(f"{len(y)} labeled routes, {dims} dims, "
          f"{int(tr.sum())} train / {int(te.sum())} test ({test_day})")

    model = fit(X[tr], y[tr])
    from scipy.stats import spearmanr
    pred = predict(model, X[te])
    rho = spearmanr(pred, y[te]).statistic
    mae = float(np.abs(pred - y[te]).mean())
    print(f"held-out day: Spearman {rho:.3f}, MAE {mae:,.0f} rating points")

    # Known-answer probe: top-field routes must outscore mid-field ones.
    hi = y[te] >= np.percentile(y[te], 80)
    lo = y[te] <= np.percentile(y[te], 40)
    gap = float(pred[hi].mean() - pred[lo].mean())
    print(f"top-vs-mid predicted gap: {gap:+,.0f} "
          f"(actual {float(y[te][hi].mean() - y[te][lo].mean()):+,.0f})")

    model_full = fit(X, y)
    json.dump({"model": model_full, "spearman_heldout": float(rho),
               "mae_heldout": mae, "n": len(y), "dims": dims,
               "test_day": str(test_day)},
              open(OUT, "w", encoding="utf-8"))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
