"""Opponent clusters: learned OFFLINE from the ladder corpus, identified at RUNTIME from what we can see.

    python -m kaggriculture.bandit.clusters.opp_clusters  [--n 80000] [--kmax 12]

1. Signature of an opponent = the RIVAL half of our dayobs vector (31 public features: rival money,
   hands, land, hires, farm tiles by crop / animals, rival net sales per item inferred from the public
   market, layout similarity, same-squares share, step-1 cash mirror) at days 1,3,5,7,10,14.
   One row per (episode, observing seat) from data/train/corpus_obs.npy (games of >= 15 days).
2. Standardize -> PCA(20) -> k-means for K = 3..kmax; K chosen by silhouette (sampled).
3. Runtime identification: for each decision day d, a gradient-boosted classifier on the rival
   features of days 0..d predicts the cluster; accuracy by day, split by EPISODE (5-fold group CV).
4. Writes models: data/train/opp_clusters.npz (scaler, PCA, centroids) + data/train/opp_ident_d<d>.pkl,
   and a report data/gates/opp_clusters.json (cluster profiles, sizes, win rate of the observing seat,
   rival rating-band mix, identification accuracy by day).
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from kaggriculture.paths import ROOT

# inputs: the corpus caches (read only); outputs: models/bandit/clusters (bandit track)
CACHE = os.path.join(ROOT, "data", "train")
OUT = os.path.join(ROOT, "models", "bandit", "clusters")
TR = CACHE
SIG_DAYS = [1, 3, 5, 7, 10, 14]
ID_DAYS = [0, 1, 2, 3, 5, 7, 10]
BANDS = ["<2100", "2100-2300", "2300-2500", "2500-2700", "2700+"]


def rival_idx(names):
    return [i for i, n in enumerate(names) if n.startswith("rival") or n in ("layout_sim", "pos_equal_frac", "cash_equal_step1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--kmax", type=int, default=12)
    a = ap.parse_args()
    names = json.load(open(os.path.join(TR, "obs_names.json")))
    R = rival_idx(names)
    obs = np.load(os.path.join(TR, "corpus_obs.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(TR, "corpus_meta.npz")))
    ok = np.where((meta["days"] >= 15) & np.isfinite(meta["score"]))[0]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(ok, size=min(a.n, len(ok)), replace=False))
    X3 = np.asarray(obs[rows][:, :, R], dtype=np.float32)          # [n, 30, 31]
    sig = X3[:, SIG_DAYS, :].reshape(len(rows), -1)
    sc = StandardScaler().fit(sig)
    Z = PCA(20, random_state=0).fit(sc.transform(sig))
    P = Z.transform(sc.transform(sig))
    best = None
    sub = rng.choice(len(P), size=min(8000, len(P)), replace=False)
    for k in range(3, a.kmax + 1):
        km = KMeans(k, n_init=4, random_state=0).fit(P)
        s = silhouette_score(P[sub], km.labels_[sub])
        print(f"[clusters] K={k}: silhouette {s:.3f}", flush=True)
        if best is None or s > best[0]:
            best = (s, k, km)
    s, K, km = best
    lab = km.labels_
    print(f"[clusters] chose K={K} (silhouette {s:.3f})", flush=True)
    rn = [names[i] for i in R]
    col = {n: j for j, n in enumerate(rn)}
    prof = []
    for c in range(K):
        m = lab == c
        d14 = X3[m, 14, :]
        top = lambda keys: {k: float(np.median(d14[:, col[k]])) for k in keys}  # noqa: E731
        bands = meta["opp_band"][rows[m]]
        prof.append({
            "cluster": c, "n": int(m.sum()), "share": float(m.mean()),
            "observer_score": float(meta["score"][rows[m]].mean()),
            "same_squares_d1": float(np.median(X3[m, 1, col["pos_equal_frac"]])),
            "cash_mirror": float(np.mean(X3[m, 1, col["cash_equal_step1"]])),
            "layout_sim_d7": float(np.median(X3[m, 7, col["layout_sim"]])),
            "rival_band_mix": {BANDS[b]: float(np.mean(bands == b)) for b in range(5)},
            "day14": top([k for k in rn if "tiles" in k or "animals" in k or k in ("rival_hands", "rival_quads", "rival_money")]),
        })
    # ---- runtime identification by day (episode-grouped CV)
    groups = meta["episode_id"][rows]
    acc = {}
    models = {}
    for d in ID_DAYS:
        F = X3[:, : d + 1, :].reshape(len(rows), -1)
        scores = []
        for tr, te in GroupKFold(5).split(F, lab, groups):
            clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(F[tr], lab[tr])
            scores.append((clf.predict(F[te]) == lab[te]).mean())
        acc[d] = float(np.mean(scores))
        models[d] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(F, lab)
        print(f"[ident] by day {d}: accuracy {acc[d]:.3f} (base rate {np.bincount(lab).max() / len(lab):.3f})", flush=True)
    np.savez(os.path.join(OUT, "opp_clusters.npz"), mean=sc.mean_, scale=sc.scale_, pca=Z.components_, pca_mean=Z.mean_,
             centroids=km.cluster_centers_, sig_days=np.array(SIG_DAYS), rival_idx=np.array(R))
    pickle.dump(models, open(os.path.join(OUT, "opp_ident.pkl"), "wb"))
    os.makedirs(OUT, exist_ok=True)
    json.dump({"K": K, "silhouette": s, "n": len(rows), "clusters": prof, "ident_accuracy_by_day": acc},
              open(os.path.join(OUT, "opp_clusters.json"), "w"), indent=1)
    for p in prof:
        print(f"  c{p['cluster']}: {p['share']:.1%}  same-squares d1 {p['same_squares_d1']:.2f}  cash-mirror {p['cash_mirror']:.2f}  "
              f"layout d7 {p['layout_sim_d7']:.2f}  observer score {p['observer_score']:.2f}  "
              f"bands {', '.join(f'{k} {v:.0%}' for k, v in p['rival_band_mix'].items() if v > .05)}")


if __name__ == "__main__":
    main()
