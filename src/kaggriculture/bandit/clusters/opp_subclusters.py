"""Second level: sub-cluster the LINEAGE opponents (same farm as ours) by TRADING behaviour, then
measure how early each trading style is identifiable at runtime.

    python -m kaggriculture.bandit.clusters.opp_subclusters  [--n 60000] [--kmax 8]

Lineage rows = the observing seat's rival stood on the same squares on day 1 (pos_equal_frac >= .5).
Trading signature (public, from our side of the dayobs vector): rival net sales per item (inferred from
market inventory) summed over days 1-3, 4-7, 8-14, 15-21; rival money at days 3,7,14,21 relative to ours;
step-1 cash mirror; layout similarity at days 7,14,21 (drift = a partial copy).
Identification: gradient-boosted classifier on everything visible by day d (d = 1,2,3,5,7,10,14),
episode-grouped 5-fold CV. Writes data/train/opp_sub.pkl and data/gates/opp_subclusters.json.
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kaggriculture.paths import ROOT

# inputs: the corpus caches (read only); outputs: models/bandit/clusters (bandit track)
CACHE = os.path.join(ROOT, "data", "train")
OUT = os.path.join(ROOT, "models", "bandit", "clusters")
TR = CACHE
ITEMS = ["carrot", "egg", "fertilizer", "melon", "milk", "strawberry", "tomato", "wheat", "wool"]
WINDOWS = [(1, 4), (4, 8), (8, 15), (15, 22)]
ID_DAYS = [1, 2, 3, 5, 7, 10, 14]


def unslog(x):
    return np.sign(x) * np.expm1(np.abs(x))


def signature(X, ix, upto=29):
    """Trading signature from days 0..upto (later windows are zero when not reached yet)."""
    n = X.shape[0]
    sold = unslog(X[:, :, [ix[f"rival_sold_{i}"] for i in ITEMS]])        # [n,30,9] per-day window sums
    parts = []
    for a, b in WINDOWS:
        b2 = min(b, upto + 1)
        parts.append(sold[:, a:b2, :].sum(1) if b2 > a else np.zeros((n, len(ITEMS))))
    rel = []
    for d in (3, 7, 14, 21):
        rel.append((X[:, d, ix["rival_money"]] - X[:, d, ix["own_money"]]) if d <= upto else np.zeros(n))
    lay = [X[:, d, ix["layout_sim"]] if d <= upto else np.zeros(n) for d in (7, 14, 21)]
    return np.concatenate([np.log1p(np.abs(np.concatenate(parts, 1))) * np.sign(np.concatenate(parts, 1)),
                           np.stack(rel, 1), np.stack(lay, 1), X[:, 1:2, ix["cash_equal_step1"]]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--kmax", type=int, default=8)
    a = ap.parse_args()
    names = json.load(open(os.path.join(TR, "obs_names.json")))
    ix = {n: i for i, n in enumerate(names)}
    obs = np.load(os.path.join(TR, "corpus_obs.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(TR, "corpus_meta.npz")))
    ok = np.where((meta["days"] >= 22) & np.isfinite(meta["score"]))[0]
    rng = np.random.default_rng(1)
    cand = np.sort(rng.choice(ok, size=min(a.n * 2, len(ok)), replace=False))
    X = np.asarray(obs[cand], dtype=np.float32)
    lin = X[:, 1, ix["pos_equal_frac"]] >= .5
    X, rows = X[lin][: a.n], cand[lin][: a.n]
    print(f"[sub] lineage rows {len(rows)}", flush=True)
    S = signature(X, ix)
    sc = StandardScaler().fit(S)
    Z = sc.transform(S)
    best = None
    sub = rng.choice(len(Z), size=min(8000, len(Z)), replace=False)
    for k in range(2, a.kmax + 1):
        km = KMeans(k, n_init=4, random_state=0).fit(Z)
        s = silhouette_score(Z[sub], km.labels_[sub])
        print(f"[sub] K={k}: silhouette {s:.3f}", flush=True)
        if best is None or s > best[0]:
            best = (s, k, km)
    s, K, km = best
    lab = km.labels_
    prof = []
    for c in range(K):
        m = lab == c
        sold = unslog(X[m][:, :, [ix[f"rival_sold_{i}"] for i in ITEMS]])
        early = sold[:, 1:8, :].sum(1).mean(0)
        late = sold[:, 15:30, :].sum(1).mean(0)
        prof.append({"sub": c, "n": int(m.sum()), "share": float(m.mean()), "observer_score": float(meta["score"][rows[m]].mean()),
                     "cash_mirror": float(X[m, 1, ix["cash_equal_step1"]].mean()),
                     "layout_d21": float(np.median(X[m, 21, ix["layout_sim"]])),
                     "rival_minus_own_money_d14": float(np.median(X[m, 14, ix["rival_money"]] - X[m, 14, ix["own_money"]]) * 1e5),
                     "rival_sales_d1_7": {i: round(float(v), 1) for i, v in zip(ITEMS, early) if abs(v) >= 1},
                     "rival_sales_d15_29": {i: round(float(v), 1) for i, v in zip(ITEMS, late) if abs(v) >= 1}})
    groups = meta["episode_id"][rows]
    acc, models = {}, {}
    for d in ID_DAYS:
        F = np.concatenate([signature(X, ix, upto=d), X[:, : d + 1, :].reshape(len(rows), -1)], 1)
        sc_ = []
        for tr, te in GroupKFold(5).split(F, lab, groups):
            clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(F[tr], lab[tr])
            sc_.append((clf.predict(F[te]) == lab[te]).mean())
        acc[d] = float(np.mean(sc_))
        models[d] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(F, lab)
        print(f"[sub-ident] by day {d}: {acc[d]:.3f} (base {np.bincount(lab).max() / len(lab):.3f})", flush=True)
    pickle.dump({"scaler": sc, "kmeans": km, "models": models, "K": K}, open(os.path.join(OUT, "opp_sub.pkl"), "wb"))
    json.dump({"K": K, "silhouette": s, "n": len(rows), "subclusters": prof, "ident_accuracy_by_day": acc},
              open(os.path.join(OUT, "opp_subclusters.json"), "w"), indent=1)
    for p in prof:
        print(json.dumps(p))


if __name__ == "__main__":
    main()
