"""M1 of the relay stack: train the near-mirror (schedule-collision) detector.

The hand-tuned relay locks on a structural distance threshold -- a guess. This
trains the guess away. The runtime question is not "do we look similar now"
but "WILL their sell schedule collide with mine later, so the race is worth
running". That is a prediction, and the corpus can supervise it:

  pair (A, B) of mined routes
    features  |prefix_features(A, t) - prefix_features(B, t)|   (observable)
    label     do A and B dump the same relay product within +/-3 turns,
              >= _MIN_COLLISIONS times, in turns 264..640?      (the future)

Held-out split is BY ROUTE (a route appears in only one side), so the score is
generalization to unseen programs, not pair memorization. Export is the same
stdlib dot-product format as the identifier.

    python src/train_mirror.py                 # train + export
    python src/train_mirror.py --pairs 8000
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.data.features as F  # noqa: E402

OUT = os.path.join(ROOT, "models", "relay")
PROBES = (216, 240, 264, 288)
RELAY_PRODUCTS = ("FERTILIZER", "MELON", "WOOL", "STRAWBERRY", "MILK")
DUMP_QTY = 8
SUFFIX = (264, 640)
COLLIDE_WITHIN = 3
_MIN_COLLISIONS = 3


def dumps_of(actions):
    """[(step, product)] of every relay-sized dump in the suffix window."""
    out = []
    lo, hi = SUFFIX
    for t in range(lo, min(hi, len(actions))):
        turn = actions[t]
        if not isinstance(turn, dict):
            continue
        for o in (turn.get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] in RELAY_PRODUCTS):
                try:
                    if int(o[2]) >= DUMP_QTY:
                        out.append((t, o[1]))
                except (TypeError, ValueError):
                    pass
    return out


def collisions(da, db):
    n = 0
    for ta, pa in da:
        if any(pb == pa and abs(tb - ta) <= COLLIDE_WITHIN for tb, pb in db):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--max-routes", type=int, default=1600,
                    help="cap on routes to featurize (newest first)")
    args = ap.parse_args()
    rng = random.Random(7)

    idx = R.load_index()
    recs = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]),
                  reverse=True)[:args.max_routes]
    feats, dump_cache = {}, {}
    for rec in recs:
        try:
            acts = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        if len(acts) < SUFFIX[1]:
            continue
        feats[rec["id"]] = {t: np.array(F.prefix_features(acts, t))
                            for t in PROBES}
        dump_cache[rec["id"]] = dumps_of(acts)
    ids = sorted(feats)
    print(f"{len(ids)} routes featurized at probes {PROBES}")

    # Route-level split BEFORE pairing: held-out routes never train.
    rng.shuffle(ids)
    cut = int(len(ids) * 0.75)
    train_ids, test_ids = ids[:cut], ids[cut:]

    def make_pairs(pool, n):
        rows, labels = [], []
        for _ in range(n):
            a, b = rng.sample(pool, 2)
            lab = 1 if collisions(dump_cache[a], dump_cache[b]) >= _MIN_COLLISIONS else 0
            t = rng.choice(PROBES)
            rows.append(np.abs(feats[a][t] - feats[b][t]))
            labels.append(lab)
        return np.array(rows), np.array(labels)

    Xtr, ytr = make_pairs(train_ids, args.pairs)
    Xte, yte = make_pairs(test_ids, max(600, args.pairs // 5))
    print(f"train {len(ytr)} pairs ({ytr.mean():.2f} positive), "
          f"test {len(yte)} pairs ({yte.mean():.2f} positive)")

    # Logistic regression, plain gradient descent (no sklearn dependency).
    w = np.zeros(Xtr.shape[1])
    b = 0.0
    lr, reg = 0.5, 1e-4
    for epoch in range(400):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - ytr
        w -= lr * (Xtr.T @ g / len(ytr) + reg * w)
        b -= lr * g.mean()

    def acc(X, y):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        return float(((p >= 0.5) == y).mean())

    print(f"train acc {acc(Xtr, ytr):.3f}   heldout-route acc {acc(Xte, yte):.3f}")
    base = max(yte.mean(), 1 - yte.mean())
    print(f"(majority-class baseline on test: {base:.3f})")

    os.makedirs(OUT, exist_ok=True)
    payload = {"w": [round(float(v), 6) for v in w], "b": round(float(b), 6),
               "probes": list(PROBES), "dims": int(Xtr.shape[1]),
               "heldout_acc": round(acc(Xte, yte), 4),
               "baseline": round(float(base), 4)}
    path = os.path.join(OUT, "mirror.json")
    json.dump(payload, open(path, "w", encoding="utf-8"))
    print(f"wrote {os.path.relpath(path, ROOT)}")

    # Export equivalence: stdlib dot product must match numpy.
    x = np.abs(feats[test_ids[0]][240] - feats[test_ids[1]][240])
    z_np = float(x @ w + b)
    z_py = payload["b"] + sum(float(x[i]) * payload["w"][i] for i in range(len(x)))
    print(f"export equivalence |diff| = {abs(z_np - z_py):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
