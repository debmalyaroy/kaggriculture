"""Relationship miner #1 -- gradient-boosted verdict model over play features.

HistGradientBoosting on (play features -> win), with:
  * held-out AUC/accuracy (episode-level split -- both seats of one episode
    stay on one side)
  * permutation importance (model-agnostic, on the held-out half)
  * top pairwise interactions via 2-D partial dependence variance
  * per-game LOCAL attribution for LOSSES: leave-one-feature-out re-scoring
    against the training median (a cheap, honest SHAP stand-in -- the shap
    package is deliberately not a dependency)

Output: models/trackp/verdict_model.json (+ per-loss attributions for the
newest N losses). Runs CPU-only; scheduled weekly in the trackp pipeline.

Usage: python src/trackp/verdict_model.py [--limit N] [--explain-losses 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

try:
    from . import common, insight, trace_v2
except ImportError:
    import sys
    from kaggriculture.trackp import common, insight, trace_v2


def collect(engine=None, limit=0):
    if engine is None:
        engine = common.engine_version()
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    feats, wins, eps, teams = [], [], [], []
    for f in files:
        try:
            X0, meta = trace_v2.load(f)
        except Exception:                                          # noqa: BLE001
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        tms = meta.get("teams") or ["?", "?"]
        for seat in (0, 1):
            if banks[seat] == banks[1 - seat]:
                continue                       # draws out of the classifier
            feats.append(insight.play_features(trace_v2.seat_view(X0, seat)))
            wins.append(1 if banks[seat] > banks[1 - seat] else 0)
            eps.append(int(meta.get("episode") or 0))
            teams.append(tms[seat])
    names = sorted(feats[0].keys())
    X = np.array([[f[k] for k in names] for f in feats], dtype=np.float64)
    y = np.array(wins)
    return X, y, names, np.array(eps), teams


def run(limit=0, explain_losses=40, seed=7):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance, partial_dependence
    from sklearn.metrics import roc_auc_score

    X, y, names, eps, teams = collect(limit=limit)
    n = len(y)
    rng = np.random.default_rng(seed)
    uniq = np.unique(eps)
    rng.shuffle(uniq)
    hold_eps = set(uniq[: len(uniq) // 5].tolist())
    te = np.array([e in hold_eps for e in eps])
    tr = ~te

    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                         learning_rate=0.08,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y[te], p))
    acc = float(((p > 0.5) == y[te]).mean())

    perm = permutation_importance(clf, X[te], y[te], n_repeats=8,
                                  random_state=seed, scoring="roc_auc")
    imp = sorted(zip(names, perm.importances_mean, perm.importances_std),
                 key=lambda t: -t[1])

    # top pairwise interactions: variance of the 2-D partial dependence
    # surface beyond the additive 1-D parts
    top = [names.index(nm) for nm, _, _ in imp[:6]]
    inter = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            pd2 = partial_dependence(clf, X[tr], [(a, b)],
                                     grid_resolution=8)
            z = pd2["average"][0]
            add = z.mean(axis=1, keepdims=True) + z.mean(
                axis=0, keepdims=True) - z.mean()
            inter.append((names[a], names[b],
                          float(np.abs(z - add).mean())))
    inter.sort(key=lambda t: -t[2])

    # local attribution for our newest losses: leave-one-out toward the
    # training median -- how much does restoring feature k to "typical"
    # move this game's win probability?
    med = np.median(X[tr], axis=0)
    loss_rows = [i for i in range(n)
                 if y[i] == 0 and teams[i] == "Debmalya"][-explain_losses:]
    explains = []
    for i in loss_rows:
        base = float(clf.predict_proba(X[i:i + 1])[0, 1])
        deltas = []
        for k in range(len(names)):
            xv = X[i].copy()
            xv[k] = med[k]
            pv = float(clf.predict_proba(xv[None])[0, 1])
            deltas.append((names[k], round(pv - base, 4)))
        deltas.sort(key=lambda t: -t[1])
        explains.append({"episode": int(eps[i]), "p_win": round(base, 3),
                         "top_fixes": deltas[:5]})

    rep = {"plays": int(n), "train": int(tr.sum()), "test": int(te.sum()),
           "auc": round(auc, 4), "acc": round(acc, 4),
           "importance": [{"feature": nm, "mean": round(m, 5),
                           "std": round(s, 5)} for nm, m, s in imp[:20]],
           "interactions": [{"a": a, "b": b, "strength": round(s, 5)}
                            for a, b, s in inter[:8]],
           "our_loss_explanations": explains}
    out = os.path.join(common.MODELS, "verdict_model.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("plays", "auc", "acc")}))
    print("top importance:", [(r["feature"], r["mean"])
                              for r in rep["importance"][:8]])
    print("top interactions:", [(r["a"], r["b"], r["strength"])
                                for r in rep["interactions"][:3]])
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--explain-losses", type=int, default=40)
    a = ap.parse_args()
    run(limit=a.limit, explain_losses=a.explain_losses)
