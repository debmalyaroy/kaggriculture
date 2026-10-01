"""Cluster knowledge for the D24 decision: what each opponent cluster does in the end game.

    python -m kaggriculture.bandit.clusters.cluster_endgame  -> data/train/cluster_endgame.json + data/train/corpus_cluster.npy

1. Assigns EVERY corpus (episode, seat) row an opponent cluster: level-1 (opp_clusters.npz: k-means on
   the rival signature of days 1..14) and, for lineage rivals, the trading sub-cluster (opp_sub.pkl).
   Label = "D<c>" (different plan) or "L<s>" (lineage trading family).
2. Per cluster: the rival's net sales per item on each of days 24..29 (inferred from the public market:
   dayobs rival_sold at day d covers day d-1's steps), mean and quartiles; the observing seat's
   score; the D24 money-gap distribution; and the rival's end-game sales CONDITIONED on its day-24
   field (ripe tiles, animals) -- the "what will this kind of opponent do with what it has" prior.
"""
import json
import os
import pickle
import sys

import numpy as np

from kaggriculture.paths import ROOT

# inputs: the corpus caches (read only); outputs: models/bandit/clusters (bandit track)
CACHE = os.path.join(ROOT, "data", "train")
OUT = os.path.join(ROOT, "models", "bandit", "clusters")
TR = CACHE
from kaggriculture.bandit.clusters.opp_subclusters import signature  # noqa: E402

ITEMS = ["carrot", "egg", "fertilizer", "melon", "milk", "strawberry", "tomato", "wheat", "wool"]


def unslog(x):
    return np.sign(x) * np.expm1(np.abs(x))


def main():
    names = json.load(open(os.path.join(TR, "obs_names.json")))
    ix = {n: i for i, n in enumerate(names)}
    obs = np.load(os.path.join(TR, "corpus_obs.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(TR, "corpus_meta.npz")))
    C = np.load(os.path.join(OUT, "opp_clusters.npz"))
    sub = pickle.load(open(os.path.join(OUT, "opp_sub.pkl"), "rb"))
    R = C["rival_idx"]
    days = C["sig_days"]
    E = len(meta["episode_id"])
    lab = np.full(E, "", dtype=object)
    for a in range(0, E, 20000):
        X = np.asarray(obs[a:a + 20000], dtype=np.float32)
        sig = X[:, days][:, :, R].reshape(len(X), -1)
        z = ((sig - C["mean"]) / C["scale"] - C["pca_mean"]) @ C["pca"].T
        c1 = np.argmin(((z[:, None, :] - C["centroids"][None]) ** 2).sum(-1), 1)
        lin = X[:, 1, ix["pos_equal_frac"]] >= .5
        s = sub["kmeans"].predict(sub["scaler"].transform(signature(X, ix)))
        lab[a:a + len(X)] = np.where(lin, np.char.add("L", s.astype(str)), np.char.add("D", c1.astype(str)))
        print(f"[endgame] labelled {min(a + 20000, E)}/{E}", flush=True)
    np.save(os.path.join(OUT, "corpus_cluster.npy"), lab.astype(str))
    ok = (meta["days"] >= 30) & np.isfinite(meta["score"])
    out = {}
    for cl in sorted(set(lab[ok])):
        idx = np.where(ok & (lab == cl))[0]
        if len(idx) < 200:
            continue
        idx = np.sort(np.random.default_rng(0).choice(idx, size=min(20000, len(idx)), replace=False))
        X = np.asarray(obs[idx], dtype=np.float32)
        sold = unslog(X[:, 25:30][:, :, [ix[f"rival_sold_{i}"] for i in ITEMS]])   # days 24..28 sales (+ day 29 via step 719 not in obs)
        gap24 = X[:, 24, ix["gap"]] * 1e5
        ripe = X[:, 24, ix["rival_tiles_ripe"]] * 64
        per_item = {i: {"mean_per_day": [round(float(v), 1) for v in sold[:, :, k].mean(0)],
                        "total_q50": round(float(np.median(sold[:, :, k].sum(1))), 1),
                        "total_q90": round(float(np.quantile(sold[:, :, k].sum(1), .9)), 1)} for k, i in enumerate(ITEMS)}
        hi = ripe >= np.median(ripe)
        cond = {"rival_ripe_high": {i: round(float(sold[hi][:, :, k].sum(1).mean()), 1) for k, i in enumerate(ITEMS)},
                "rival_ripe_low": {i: round(float(sold[~hi][:, :, k].sum(1).mean()), 1) for k, i in enumerate(ITEMS)}}
        out[cl] = {"n": int((ok & (lab == cl)).sum()), "observer_score": float(meta["score"][idx].mean()),
                   "gap24_q": [round(float(q), 0) for q in np.quantile(gap24, [.1, .5, .9])],
                   "observer_win_when_behind_at_d24": float(meta["score"][idx][gap24 < 0].mean()) if (gap24 < 0).any() else None,
                   "rival_endgame_sales": per_item, "rival_endgame_sales_by_ripe": cond}
        print(f"[endgame] {cl}: n {out[cl]['n']}, observer score {out[cl]['observer_score']:.3f}, "
              f"win when behind at D24 {out[cl]['observer_win_when_behind_at_d24']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "cluster_endgame.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
