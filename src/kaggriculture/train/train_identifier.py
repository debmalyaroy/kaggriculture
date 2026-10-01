"""Train the opponent-program identifier as a model; export plain weights.

Multinomial logistic regression over behavioral-cluster labels, trained on
prefix features manufactured from every mined route (what an in-game
observer could have accumulated by turn t), with dropout augmentation to
mimic floored-sale invisibility. Pure-numpy training; weights exported as
JSON constants for hand-rolled stdlib inference inside the agent.

    python -m kaggriculture.train.train_identifier            # train + export + report
    python -m kaggriculture.train.train_identifier --check    # export-equivalence test
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402

OUT = os.path.join(ROOT, "models", "v22", "identifier")
PREFIX_TURNS = (144, 192, 240, 288, 360, 480)
LINK = 0.055
MIN_FAMILY = 4          # families smaller than this train the NOVEL class
DROPOUT = (0.0, 0.1, 0.2)
# RECENCY WINDOW (2026-08-20): the index grew 5,252 -> 26,929 routes in ten
# days when the fetch throttle cleared, and the full-pool retrain blew past
# its 1800s cycle budget (the feature cache only spares ALREADY-cached
# routes; each day's thousands of new ones are still derived, and the
# clustering loop is ~O(routes x families)). Train on the most recent
# TRAIN_WINDOW routes instead of all of them. This is not only faster: the
# newest-first rationale above says stale routes chain live styles into dead
# blobs, so a recency cap tracks the live meta better. Env-overridable; 0
# disables the cap (full pool). Held-out-day accuracy is the guard -- if a
# window ever drops it, widen the window, do not raise the timeout.
TRAIN_WINDOW = int(os.environ.get("KAGG_ID_WINDOW", "9000"))


def build_dataset():
    idx = R.load_index()
    # NEWEST-FIRST clustering, rebuilt each run. A registry-based
    # ascending-order variant (src/family_registry.py) was integrated
    # 2026-08-13 and REVERTED the same morning: its first production retrain
    # cratered held-out-day accuracy 0.893 -> 0.571 (train acc 0.589 -- the
    # labels stopped being fittable). Root cause hypothesis: append-only
    # representatives seeded by the OLDEST routes chain distinct current
    # styles into stale blobs, where newest-first reps track the live meta.
    # Label stability across retrains remains an open problem -- the RELAY
    # GUARD + relay_config-after-identifier ordering stays load-bearing.
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    if TRAIN_WINDOW and len(rows) > TRAIN_WINDOW:
        print(f"recency window: {TRAIN_WINDOW} of {len(rows)} routes "
              f"(newest first)")
        rows = rows[:TRAIN_WINDOW]
    # Cached feature vectors (src/feature_cache.py). This loop used to call
    # R.load_route + route_features for every route on every retrain, and the
    # X-building loop below loaded each route a SECOND time for its prefix
    # vectors -- ~6.5 min of an 8.1 min retrain re-deriving vectors for
    # immutable routes. The cache invalidates on a hash of features.py and
    # routes.py, so a code change still forces a full rebuild.
    import kaggriculture.data.feature_cache as FC
    cache_route, cache_prefix = FC.build([r["id"] for r in rows], PREFIX_TURNS)
    feats, recs = {}, []
    for rec in rows:
        v = cache_route.get(rec["id"])
        if v is None:
            continue                    # route unreadable; skipped as before
        feats[rec["id"]] = v
        recs.append(rec)
    # Cluster on full-route signatures (the label space).
    reps, labels = [], {}
    for rec in recs:
        f = feats[rec["id"]]
        for fi, rep in enumerate(reps):
            if features.distance(f, rep) < LINK:
                labels[rec["id"]] = fi
                break
        else:
            reps.append(f)
            labels[rec["id"]] = len(reps) - 1
    sizes = {}
    for v in labels.values():
        sizes[v] = sizes.get(v, 0) + 1
    keep = {fi for fi, n in sizes.items() if n >= MIN_FAMILY}
    remap = {fi: i for i, fi in enumerate(sorted(keep))}
    novel = len(remap)                      # last class = NOVEL
    print(f"{len(recs)} routes, {len(reps)} raw clusters, "
          f"{len(remap)} trained families + NOVEL")

    # NOTE: the SHIPPED identifier must use only features the agent can
    # reconstruct live -- sale curves (from shared inventory) and farm-event
    # diffs (from the public opponent farm). Observation-features
    # (reactivity, shop-mix, microstructure) read the opponent's ACTION dict,
    # which the runtime agent never sees, so they belong to the OFFLINE
    # clustering/census (src/kaggriculture/data/families.py), NOT here. Adding them here was a
    # train/serve skew bug caught in the 2026-08-11 audit and reverted.
    rng = random.Random(11)
    X, y, split = [], [], []
    newest = max(r.get("date", "") for r in recs)
    for rec in recs:
        cls = remap.get(labels[rec["id"]], novel)
        pvecs = cache_prefix.get(rec["id"])
        if pvecs is None:
            continue                    # cache miss implies unreadable route
        for ti, t in enumerate(PREFIX_TURNS):
            base_vec = pvecs[ti].tolist()
            for p in DROPOUT:
                vec = [v * (0 if rng.random() < p else 1) for v in base_vec]
                X.append(vec + [t / 720.0])
                y.append(cls)
                split.append("test" if rec.get("date") == newest else "train")
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    tr = np.array(split) == "train"
    print(f"{len(X)} rows ({tr.sum()} train / {(~tr).sum()} held-out day), "
          f"{X.shape[1]} dims, {novel + 1} classes")
    return X, y, tr, novel + 1, remap


def train(X, y, k, epochs=300, lr=0.5, reg=1e-4):
    n, d = X.shape
    W = np.zeros((d, k))
    b = np.zeros(k)
    Y = np.eye(k)[y]
    for e in range(epochs):
        z = X @ W + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (X.T @ g + reg * W)
        b -= lr * g.sum(axis=0)
    return W, b


def accuracy(X, y, W, b):
    return float((np.argmax(X @ W + b, axis=1) == y).mean())


def _fit_temperature(X, y, W, b, grid=60):
    """Scalar T minimizing NLL on (X, y); argmax (accuracy) is unaffected."""
    z = X @ W + b
    z -= z.max(axis=1, keepdims=True)
    best_t, best_nll = 1.0, None
    for t in np.geomspace(0.25, 4.0, grid):
        p = np.exp(z / t)
        p /= p.sum(axis=1, keepdims=True)
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y],
                                    1e-12, None)).mean())
        if best_nll is None or nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def export(W, b, path):
    payload = {"W": [[round(float(v), 6) for v in row] for row in W],
               "b": [round(float(v), 6) for v in b]}
    json.dump(payload, open(path, "w", encoding="utf-8"))
    return payload


def stdlib_predict(payload, x):
    """The exact inference the agent will run: pure python, no numpy."""
    W, b = payload["W"], payload["b"]
    k = len(b)
    z = [b[j] + sum(x[i] * W[i][j] for i in range(len(x))) for j in range(k)]
    m = max(z)
    e = [2.718281828459045 ** (v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    X, y, tr, k, remap = build_dataset()
    W, b = train(X[tr], y[tr], k)
    acc_tr = accuracy(X[tr], y[tr], W, b)
    acc_te = accuracy(X[~tr], y[~tr], W, b)
    print(f"train acc {acc_tr:.3f}   held-out-day acc {acc_te:.3f}")

    # Temperature calibration, folded INTO the exported weights (z/T ==
    # (W/T, b/T)) so the agent's embedded inference is calibrated with zero
    # runtime change. Measured 2026-08-12: the raw model is UNDER-confident
    # (T* 0.74-0.89), which starves the 0.85 commit gate. Fit on the
    # held-out day -- the only slice that behaves like tomorrow (fitting on
    # the train tail made test log-loss WORSE in the lab).
    T = _fit_temperature(X[~tr], y[~tr], W, b) if (~tr).sum() > 200 else 1.0
    print(f"temperature T* {T:.2f} folded into the export")
    W, b = W / T, b / T

    payload = export(W, b, os.path.join(OUT, "weights.json"))
    json.dump({"classes": k, "families": {str(v): int(kk) for kk, v in
                                          remap.items()},
               "train_acc": acc_tr, "heldout_acc": acc_te,
               "dims": int(X.shape[1])},
              open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"),
              indent=1)

    # Export-equivalence: stdlib inference must match numpy to 1e-6.
    worst = 0.0
    for i in range(0, len(X), max(1, len(X) // 50)):
        z = X[i] @ W + b
        z -= z.max()
        p_np = np.exp(z)
        p_np /= p_np.sum()
        p_py = stdlib_predict(payload, list(X[i]))
        worst = max(worst, float(np.abs(p_np - np.array(p_py)).max()))
    print(f"export equivalence: worst |diff| {worst:.2e} "
          f"({'OK' if worst < 1e-6 else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
