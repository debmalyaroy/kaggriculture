"""Relationship miner #2 -- per-turn win-probability model + decisive moments.

Train V(state_t) -> P(win) on per-turn trace rows (episode-level split), then
for each game compute the win-prob CURVE and extract its decisive moments:
the turns with the largest |dV| over a 24-turn window. Aggregated, this
answers "WHEN are games decided" per regime and per opponent family;
per-game, it points at the exact day a loss became a loss.

Output: models/trackp/winprob.json (model report + decided-when histogram +
per-loss decisive days for our games). The fitted model doubles as an
external critic sanity-check for PPO's value head.

Usage: python src/trackp/winprob.py [--limit N] [--stride 12]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

try:
    from . import common, macro, trace_v2
except ImportError:
    import sys
    from kaggriculture.trackp import common, macro, trace_v2

_ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}


def _turn_features(X: np.ndarray, t: int) -> list:
    """Reuses the macro feature path (41 dims incl. opponent embedding) --
    one definition of 'state' across the whole track."""
    return macro.l1_features_trace_full(X, t)


def collect(engine=None, limit=0, stride=12):
    if engine is None:
        engine = common.engine_version()
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    rows, wins, eps, ts = [], [], [], []
    metas = []
    for f in files:
        try:
            X0, meta = trace_v2.load(f)
        except Exception:                                          # noqa: BLE001
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        if banks[0] == banks[1]:
            continue
        metas.append((f, meta))
        for seat in (0, 1):
            X = trace_v2.seat_view(X0, seat)
            w = 1 if banks[seat] > banks[1 - seat] else 0
            for t in range(stride, X.shape[0], stride):
                rows.append(_turn_features(X, t))
                wins.append(w)
                eps.append(int(meta.get("episode") or 0))
                ts.append(t)
    return (np.asarray(rows, dtype=np.float64), np.asarray(wins),
            np.asarray(eps), np.asarray(ts), metas)


def run(limit=0, stride=12, seed=7, curves_for=200):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    X, y, eps, ts, metas = collect(limit=limit, stride=stride)
    uniq = np.unique(eps)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    hold = set(uniq[: len(uniq) // 5].tolist())
    te = np.array([e in hold for e in eps])
    tr = ~te
    clf = HistGradientBoostingClassifier(max_iter=250, max_depth=5,
                                         learning_rate=0.08,
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    auc_all = float(roc_auc_score(y[te], p))
    # calibration by phase: how early is the verdict visible?
    aucs_by_day = {}
    for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30)):
        m = te & (ts >= lo * 24) & (ts < hi * 24)
        if m.sum() > 200 and len(np.unique(y[m])) == 2:
            aucs_by_day[f"day{lo}-{hi}"] = round(float(
                roc_auc_score(y[m], clf.predict_proba(X[m])[:, 1])), 4)

    # decisive moments on held-out episodes: largest |dV| per game
    decided_hist = np.zeros(30)
    our_losses = []
    n_curves = 0
    for f, meta in metas:
        if int(meta.get("episode") or 0) not in hold:
            continue
        if n_curves >= curves_for:
            break
        try:
            X0, _ = trace_v2.load(f)
        except Exception:                                          # noqa: BLE001
            continue
        banks = meta.get("banks")
        teams = meta.get("teams") or ["?", "?"]
        for seat in (0, 1):
            Xv = trace_v2.seat_view(X0, seat)
            tt = list(range(stride, Xv.shape[0], stride))
            feats = np.asarray([_turn_features(Xv, t) for t in tt])
            curve = clf.predict_proba(feats)[:, 1]
            dv = np.abs(np.diff(curve))
            if not len(dv):
                continue
            k = int(np.argmax(dv))
            day = min(29, tt[k] // 24)
            decided_hist[day] += 1
            won = banks[seat] > banks[1 - seat]
            if teams[seat] == "Debmalya" and not won:
                our_losses.append({"episode": meta.get("episode"),
                                   "decisive_day": int(day),
                                   "dV": round(float(dv[k]), 3),
                                   "p_end": round(float(curve[-1]), 3)})
            n_curves += 1

    rep = {"rows": int(len(y)), "episodes": int(len(uniq)),
           "auc_heldout": round(auc_all, 4),
           "auc_by_phase": aucs_by_day,
           "decided_day_histogram": {str(d): int(v) for d, v
                                     in enumerate(decided_hist) if v},
           "our_loss_decisive_days": our_losses[:40]}
    out = os.path.join(common.MODELS, "winprob.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in
                      ("rows", "auc_heldout", "auc_by_phase")}, indent=1))
    top_days = sorted(((int(v), d) for d, v in
                       rep["decided_day_histogram"].items()), reverse=True)
    print("decided-when top days:", top_days[:6])
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=12)
    a = ap.parse_args()
    run(limit=a.limit, stride=a.stride)
