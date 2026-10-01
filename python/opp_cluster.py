"""Offline opponent fingerprinting (operator 28 Sep): cluster opponents by WHEN and HOW they sell, so the agent can
recognise the type in-game and sell ahead of it (crates/agent/src/preempt.rs).

    python python/opp_cluster.py [--dirs .local/allreal,data/tapes/top_all,data/tapes/leaders,data/tapes/band] [--k 16]
                                 [--out configs/opp/clusters_v1.json] [--workers 6]

Per opponent game (the recorded seat opposite ours / the top player) and product:
  hour profile   share of the day's sale EVENTS at each hour (24)             -> when they sell
  day profile    share of units in days 0-9 / 10-19 / 20-26 / 27-29 (4)        -> early seller or hoarder
  lot            median order size (log), share of "sell all" orders (>= 999) -> drip or dump
  cadence        sale events per active day (log)
k-means (numpy, k-means++ init, 5 restarts) on the standardized vector. Export per cluster: size, and
p[item][hour] = P(they sell that item at that hour on a given day) -- the forecast table the agent reads -- plus
lot / day summaries for reading. The runtime classifier is the likelihood of the rival's observed (item, hour)
sale days under each cluster's table.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = ["WHEAT", "STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"]
DAYB = [(0, 10), (10, 20), (20, 27), (27, 30)]


def sig(path):
    try:
        t = json.load(open(path, encoding="utf-8"))
        opp = 1 - int(t["seat"])
        acts = t["actions"]
    except Exception:  # noqa: BLE001
        return None
    days_hr = np.zeros((len(ITEMS), 30, 24), dtype=np.int8)   # sold item at (day, hour)
    units = np.zeros((len(ITEMS), 30))
    lots = [[] for _ in ITEMS]
    for step, pair in enumerate(acts):
        a = pair[opp] if isinstance(pair, list) and len(pair) > opp else None
        if not isinstance(a, dict):
            continue
        for o in a.get("market") or []:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in ITEMS:
                try:
                    q = int(o[2])
                except Exception:  # noqa: BLE001
                    continue
                if q <= 0:
                    continue
                i = ITEMS.index(o[1])
                d, h = min(step // 24, 29), step % 24
                days_hr[i, d, h] = 1
                units[i, d] += min(q, 200)
                lots[i].append(q)
    if units.sum() == 0:
        return None
    return os.path.basename(path), days_hr, units, [np.array(x) for x in lots]


def featurize(days_hr, units, lots):
    f = []
    for i in range(len(ITEMS)):
        act_days = max(1, int((days_hr[i].sum(1) > 0).sum()))
        hp = days_hr[i].sum(0) / act_days                      # P(sale at hour h | active day)
        f += list(hp)
        tot = units[i].sum()
        f += [units[i, a:b].sum() / tot if tot else 0.0 for a, b in DAYB]
        l = lots[i]
        f += [np.log1p(np.median(l)) / 5 if len(l) else 0.0, (l >= 999).mean() if len(l) else 0.0,
              np.log1p(len(l) / act_days) / 2 if len(l) else 0.0]
    return np.array(f, dtype=np.float32)


def kmeans(X, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(2), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), 1)
        Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    inertia = ((X - C[lab]) ** 2).sum()
    return lab, C, inertia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default=".local/allreal,data/tapes/top_all,data/tapes/leaders,data/tapes/band")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(RL, "configs", "opp", "clusters_v1.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max", type=int, default=30000)
    a = ap.parse_args()
    files = []
    for d in a.dirs.split(","):
        files += glob.glob(os.path.join(RL, d, "**", "*.json"), recursive=True)
    files = sorted(set(files))[: a.max]
    print(f"[opp] {len(files)} tapes", flush=True)
    sigs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(sig, files, chunksize=64):
            if s is not None:
                sigs.append(s)
    print(f"[opp] {len(sigs)} opponent games with sales", flush=True)
    X = np.stack([featurize(s[1], s[2], s[3]) for s in sigs])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    best = None
    for r in range(5):
        lab, C, inert = kmeans(Z, a.k, 100 + r)
        if best is None or inert < best[2]:
            best = (lab, C, inert)
    lab = best[0]
    clusters = []
    for j in range(a.k):
        idx = np.where(lab == j)[0]
        if len(idx) == 0:
            continue
        # forecast table: P(sale of item at hour | day), over the cluster's games and all their days 1..26
        dh = np.stack([sigs[i][1] for i in idx]).astype(np.float32)   # n x items x 30 x 24
        p = dh[:, :, 1:27, :].mean((0, 2))                           # items x 24
        lots = {it: float(np.median(np.concatenate([sigs[i][3][k] for i in idx]) if any(len(sigs[i][3][k]) for i in idx) else [0])) for k, it in enumerate(ITEMS)}
        allq = {it: float(np.mean(np.concatenate([sigs[i][3][k] for i in idx]) >= 999) if any(len(sigs[i][3][k]) for i in idx) else 0.0) for k, it in enumerate(ITEMS)}
        clusters.append({"id": j, "n": int(len(idx)), "p": {it: [round(float(x), 4) for x in p[k]] for k, it in enumerate(ITEMS)},
                         "median_lot": lots, "share_sell_all": allq,
                         "examples": [sigs[i][0] for i in idx[:5]],
                         "src": {"ours": int(sum("__" in sigs[i][0] for i in idx))}})
    clusters.sort(key=lambda c: -c["n"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"items": ITEMS, "k": a.k, "games": len(sigs), "clusters": clusters}, open(a.out, "w"), indent=1)
    print(f"[opp] {len(clusters)} clusters -> {a.out}", flush=True)
    for c in clusters:
        peak = {it: int(np.argmax(c["p"][it])) for it in ITEMS if max(c["p"][it]) > 0.3}
        print(f"  cluster {c['id']:2d}: {c['n']:6d} games (ours {c['src']['ours']:4d}) | peak sale hour (P>0.3): {peak} | "
              f"median lot {{{', '.join(f'{k}:{v:.0f}' for k, v in c['median_lot'].items() if v)}}}", flush=True)


if __name__ == "__main__":
    main()
