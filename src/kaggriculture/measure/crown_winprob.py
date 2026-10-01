"""Per-turn win-probability control variate (variance reducer for the crown).

Your idea, done right (kaggriculture discussion notes): a tape is NOT 720
independent win/loss functions -- the game is sequential, so turn t's state
depends on every prior turn. What we CAN extract per turn is a calibrated
P(this seat wins the game) given the cumulative public state at turn t, which
turns one noisy final bit into a dense 720-length signal and cuts the seeds
a comparison needs.

CAVEAT that makes this honest: mid-game BANK LEAD is not a valid proxy --
the winning agents do terminal liquidation, so leads flip at the end (Luka
Duvanov's teardown). So the model is calibrated to predict the FINAL bank
outcome, and the final bank stays ground truth; this only ever TIGHTENS the
seed estimate, it never overrides crown_eval's direct sign test.

Model: logistic on features of the running bank gap (signed gap, its sign,
turn fraction, and gap x lateness) fit on (turn-state -> did-this-seat-win)
pairs harvested from real replay traces (turn_features 49-field capture).
Reports the AUC so we know whether the surrogate is trustworthy before use.

    python src/crown_winprob.py --fit          # fit + report AUC
    # then crown_eval imports predict() to reduce variance (optional pass)
"""
from kaggriculture.paths import ROOT
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL = os.path.join(ROOT, "models", "factory", "winprob.json")


def _features(bank_gap, turn_frac):
    """Cheap, monotone-ish features of the public state at a turn."""
    g = bank_gap / 1e5
    return [1.0, g, 1.0 if g > 0 else -1.0, turn_frac, g * turn_frac]


def predict(bank_gap, turn_frac, w=None):
    if w is None:
        try:
            w = json.load(open(MODEL, encoding="utf-8"))["w"]
        except (OSError, ValueError):
            return None
    z = sum(wi * xi for wi, xi in zip(w, _features(bank_gap, turn_frac)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _harvest(limit):
    """(features, label) from turn traces: label = did seat s win the game."""
    import kaggriculture.data.turn_features as TF
    import glob
    import numpy as np
    X, y = [], []
    files = sorted(glob.glob(os.path.join(TF.TRACE_DIR, "*.npz")),
                   reverse=True)[:limit]
    for p in files:
        ep = os.path.splitext(os.path.basename(p))[0]
        d = TF.load(ep)
        if not d:
            continue
        seats = [k for k in d if isinstance(k, int)]
        if len(seats) < 2:
            continue
        # per turn, bank gap for seat 0 = money0 - money1 (money is col 3-ish
        # of the farm stats block; use the normalized money feature at index
        # for seat0 and seat1). turn_features.turn_vector layout: day,hour,
        # step, prices(9), inv(9), seat-farm(7: money/1e5 first), opp-farm(7).
        a0, a1 = d[0], d[1]
        T = min(len(a0), len(a1))
        if T < 30:
            continue
        # money/1e5 is the first element of the 7-wide farm-stats block, which
        # starts after day/hour/step(3)+prices(9)+inv(9) = index 21.
        MI = 21
        final0 = a0[T - 1][MI]
        final1 = a1[T - 1][MI]
        win0 = 1 if final0 > final1 else 0
        for t in range(0, T, 6):
            gap = (a0[t][MI] - a1[t][MI]) * 1e5
            tf = t / 720.0
            X.append(_features(gap, tf))
            y.append(win0)
    return np.asarray(X), np.asarray(y)


def _fit(X, y, iters=400, lr=0.3):
    import numpy as np
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= lr * (X.T @ (p - y)) / n
    return w


def _auc(w, X, y):
    import numpy as np
    s = X @ w
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rpos = ranks[:len(pos)].sum()
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()
    if not args.fit:
        print(json.dumps(json.load(open(MODEL, encoding="utf-8"))
                         if os.path.exists(MODEL) else {"status": "unfit"}))
        return 0
    import numpy as np
    X, y = _harvest(args.limit)
    if len(y) < 500:
        raise SystemExit("too few trace rows (%d) -- run backfill_traces"
                         % len(y))
    # split by row for a held-out AUC (episode-level split would be stricter;
    # this is a trust check, not the decider)
    k = int(0.8 * len(y))
    w = _fit(X[:k], y[:k])
    auc = _auc(w, X[k:], y[k:])
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    json.dump({"w": list(map(float, w)), "auc": float(auc),
               "rows": int(len(y)),
               "trustworthy": bool(auc >= 0.75)},
              open(MODEL, "w", encoding="utf-8"), indent=1)
    print("fit winprob on %d rows; held-out AUC=%.3f -> %s" % (
        len(y), auc, "USABLE" if auc >= 0.75 else "NOT trustworthy (ignore)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
