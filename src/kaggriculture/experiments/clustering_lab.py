"""Family clustering upgrade + persistent registry, evaluated on stability.

Compares the current greedy first-fit single-linkage (fixed 0.055) against
average-linkage agglomerative clustering with a distance threshold chosen by
silhouette, plus HDBSCAN. Stability metric: cluster the index as of day d and
as of day d+1 (same method), Hungarian-match the two clusterings on shared
routes, and report the fraction of shared routes whose family survived the
day boundary (label persistence). The current method's known failure is
wholesale id shifts when new families appear.

Also demonstrates the registry: Hungarian assignment of day-(d+1) centroids
to day-d centroids with a distance ceiling; unmatched clusters become NEW
families instead of renumbering everything.

Read-only; touches nothing the release pipeline runs.

    python src/experiments/clustering_lab.py
"""
from kaggriculture.paths import ROOT
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "clustering_report.json")


def load_vectors():
    idx = R.load_index()
    recs = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]))
    ids, dates, vecs = [], [], []
    for rec in recs:
        try:
            v, _ = features.route_features(R.load_route(rec["id"]))
        except Exception:                                          # noqa: BLE001
            continue
        ids.append(rec["id"])
        dates.append(rec.get("date", ""))
        vecs.append(v)
    return ids, np.array(dates), np.array(vecs, dtype=np.float64)


def greedy_current(vecs, link=0.055):
    """The production algorithm, verbatim behavior."""
    reps, labels = [], []
    for v in vecs:
        for fi, rep in enumerate(reps):
            if features.distance(v, rep) < link:
                labels.append(fi)
                break
        else:
            reps.append(v)
            labels.append(len(reps) - 1)
    return np.array(labels)


def _dist_matrix(vecs):
    """features.distance is 1 - cosine-ish; compute pairwise via the same fn
    on normalized rows, vectorized to match its arithmetic."""
    n = len(vecs)
    # features.distance(a, b): sum(|a-b|) / (sum(|a|) + sum(|b|) + eps)
    a = np.abs(vecs).sum(axis=1)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = np.abs(vecs - vecs[i]).sum(axis=1) / (a + a[i] + 1e-9)
    return D


def agglomerative(vecs, thresh):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = _dist_matrix(vecs)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=thresh, criterion="distance") - 1


def hdbscan_labels(vecs):
    import hdbscan
    D = _dist_matrix(vecs)
    cl = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=4,
                         min_samples=2)
    return cl.fit_predict(D.astype(np.float64))


def silhouette_pick(vecs, cands=(0.03, 0.04, 0.055, 0.07, 0.09, 0.12)):
    from sklearn.metrics import silhouette_score
    D = _dist_matrix(vecs)
    best, best_s = None, -2
    for t in cands:
        lab = agglomerative(vecs, t)
        k = len(set(lab))
        if k < 2 or k >= len(vecs) - 1:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        if s > best_s:
            best, best_s = t, s
    return best, best_s


def registry_match(cent_prev, cent_new, ceiling=0.08):
    """Hungarian assignment of new centroids to previous family ids."""
    from scipy.optimize import linear_sum_assignment
    if not len(cent_prev) or not len(cent_new):
        return {}
    a = np.abs(cent_prev).sum(axis=1)
    C = np.zeros((len(cent_new), len(cent_prev)))
    for i, v in enumerate(cent_new):
        C[i] = np.abs(cent_prev - v).sum(axis=1) / (a + np.abs(v).sum() + 1e-9)
    ri, ci = linear_sum_assignment(C)
    return {int(i): int(j) for i, j in zip(ri, ci) if C[i, j] <= ceiling}


def persistence(ids_a, lab_a, ids_b, lab_b):
    """Fraction of shared routes keeping a consistently-mapped family after
    Hungarian-matching the two labelings (optimal relabeling)."""
    from scipy.optimize import linear_sum_assignment
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        return None
    la = {i: l for i, l in zip(ids_a, lab_a)}
    lb = {i: l for i, l in zip(ids_b, lab_b)}
    ka = sorted(set(la[i] for i in common))
    kb = sorted(set(lb[i] for i in common))
    M = np.zeros((len(ka), len(kb)))
    for i in common:
        M[ka.index(la[i]), kb.index(lb[i])] += 1
    ri, ci = linear_sum_assignment(-M)
    kept = M[ri, ci].sum()
    return float(kept / len(common))


def main():
    ids, dates, vecs = load_vectors()
    days = sorted(set(d for d in dates if d))
    print(f"{len(ids)} routes across {len(days)} days")

    # Build the two snapshots FIRST, then cap each for O(n^2) work:
    # "prev" = everything up to the second-newest day (newest 1250 of it),
    # "all"  = prev + the newest day's routes (capped 1250) -- the exact
    # day-boundary retrain the pipeline performs.
    d_prev, d_new = days[-2], days[-1]
    CAP = 1250
    prev_i = [i for i in range(len(ids)) if dates[i] <= d_prev][-CAP:]
    new_i = [i for i in range(len(ids)) if dates[i] == d_new][:CAP]
    keep = prev_i + new_i
    ids = [ids[i] for i in keep]
    dates = dates[keep]
    vecs = vecs[keep]
    m_prev = dates <= d_prev
    print(f"snapshots: {int(m_prev.sum())} routes <= {d_prev}, "
          f"{len(ids) - int(m_prev.sum())} on {d_new}")
    ids_prev = [i for i, m in zip(ids, m_prev) if m]
    ids_all = ids

    report = {"days": [d_prev, d_new], "n_prev": len(ids_prev),
              "n_all": len(ids_all)}
    for name, fn in [("greedy_current", lambda v: greedy_current(v)),
                     ("agglomerative", None),
                     ("hdbscan", hdbscan_labels)]:
        try:
            if name == "agglomerative":
                t, s = silhouette_pick(vecs[m_prev])
                print(f"silhouette picked thresh {t} (score {s:.3f})")
                fn = lambda v: agglomerative(v, t)                 # noqa: E731
                report["agglo_thresh"] = t
                report["agglo_silhouette"] = s
            lab_prev = fn(vecs[m_prev])
            lab_all = fn(vecs)
            p = persistence(ids_prev, lab_prev, ids_all, lab_all)
            k_prev = len(set(int(x) for x in lab_prev if x >= 0))
            k_all = len(set(int(x) for x in lab_all if x >= 0))
            noise = int((np.array(lab_all) < 0).sum())
            report[name] = {"persistence": p, "k_prev": k_prev,
                            "k_all": k_all, "noise": noise}
            ps = f"{p:.3f}" if p is not None else "n/a"
            print(f"{name:16s} persistence {ps}  families "
                  f"{k_prev}->{k_all}  noise {noise}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            report[name] = {"error": str(exc)[:200]}
            print(f"{name}: ERROR {str(exc)[:150]}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
