"""Export the identifier dataset as SEQUENCES for the GRU challenger.

One row per route: the 6 cumulative prefix_features snapshots (PREFIX_TURNS)
stacked into a (6, D) sequence, labeled with the same clustering-derived
family as train_identifier. Per-step supervision lets the trainer report
accuracy at every checkpoint (early identification is worth more than late).

Saved as models/lab/seq_dataset.npz: X (n, 6, D) float32, y (n,), dates (n,).
Torch is not needed here; this runs in the base env with features/routes.

    python src/experiments/seq_dataset.py
"""
from kaggriculture.paths import ROOT
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "seq_dataset.npz")


def main():
    idx = R.load_index()
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    # Same recency window as the identifier (src/train_identifier.py): this
    # builds the GRU's view of the SAME label space, so it must see the same
    # routes, and it loads features UNCACHED -- the full pool is what pushed
    # it past its 2700s budget once the index passed ~25k routes.
    if TI.TRAIN_WINDOW and len(rows) > TI.TRAIN_WINDOW:
        print(f"recency window: {TI.TRAIN_WINDOW} of {len(rows)} routes")
        rows = rows[:TI.TRAIN_WINDOW]
    feats, recs = {}, []
    for rec in rows:
        try:
            feats[rec["id"]], _ = features.route_features(R.load_route(rec["id"]))
            recs.append(rec)
        except Exception:                                          # noqa: BLE001
            continue
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

    X, y, dates = [], [], []
    for rec in recs:
        cls = remap.get(labels[rec["id"]], novel)
        acts = R.load_route(rec["id"])
        seq = [features.prefix_features(acts, t) + [t / 720.0]
               for t in TI.PREFIX_TURNS]
        X.append(seq)
        y.append(cls)
        dates.append(rec.get("date", ""))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    dates = np.array(dates)
    ids = np.array([rec["id"] for rec in recs])
    teams = np.array([str(rec.get("team") or "?") for rec in recs])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, dates=dates, ids=ids, teams=teams,
                        classes=np.int64(novel + 1))
    print(f"{X.shape} sequences, {novel + 1} classes -> {OUT}")


if __name__ == "__main__":
    main()
