"""Cluster every corpus player-game by its SALE POLICY and export what the agent needs to out-play each type
(operator 28 Sep): WHAT they sell, AT WHICH STEP, in what lots, and whether they front-run.

    python python/opp_cluster2.py [--sig data/opp/sig_v1.npz] [--k 24] [--out configs/opp/clusters_v2.json]

Input: python/opp_corpus_sig.py. Feature vector per player-game (standardized, k-means++ x 4 restarts):
  when   per item, P(sale at hour h | active day) (24)            -- intra-day timing
  when   per item, share of units in days 0-9 / 10-19 / 20-26 / 27-29 (4)
  what   share of total units per item (8) + log total units      -- the product mix / inventory they sell
  how    per item: log median lot, share of sell-all orders, log sale events per active day
  race   per item: front-run rate (share of the other seat's sale runs they pre-empted by 1-2 steps)
Export per cluster (the forecast the agent reads, crates/agent/src/preempt.rs):
  p_step[item][720]  P(a sale of item at that step)          -- which step to front-run
  units_day[item][30] mean units of item sold per day        -- how much is coming
  lot, sell_all, front[item], n, mean rating, top teams, win rate vs the rest
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYB = [(0, 10), (10, 20), (20, 27), (27, 30)]


def kmeans(X, k, seed, iters=50, sample=60000):
    rng = np.random.default_rng(seed)
    S = X[rng.choice(len(X), min(sample, len(X)), replace=False)]
    C = [S[rng.integers(len(S))]]
    for _ in range(1, k):
        d = np.min(((S[:, None, :] - np.array(C)[None]) ** 2).sum(2), 1)
        C.append(S[rng.choice(len(S), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.concatenate([np.argmin(((X[i:i + 20000, None, :] - C[None]) ** 2).sum(2), 1) for i in range(0, len(X), 20000)])
        Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C, atol=1e-4):
            break
        C = Cn
    inert = sum(((X[i:i + 20000] - C[lab[i:i + 20000]]) ** 2).sum() for i in range(0, len(X), 20000))
    return lab, C, inert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default=os.path.join(RL, "data", "opp", "sig_v1.npz"))
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(RL, "configs", "opp", "clusters_v2.json"))
    a = ap.parse_args()
    z = np.load(a.sig, allow_pickle=False)
    items = [str(x) for x in z["items"]]
    NI = len(items)
    n = len(z["eid"])
    ev = np.unpackbits(z["ev"], axis=1)[:, : NI * 30 * 24].reshape(n, NI, 30, 24)
    units = z["units"]
    act_days = np.maximum(1, (ev.sum(3) > 0).sum(2))                       # n x NI
    hour = ev.sum(2) / act_days[:, :, None]                                  # n x NI x 24
    tot = units.sum(2)                                                       # n x NI
    dayb = np.stack([units[:, :, a_:b_].sum(2) / np.maximum(tot, 1) for a_, b_ in DAYB], 2)
    mix = tot / np.maximum(tot.sum(1, keepdims=True), 1)
    evday = ev.sum((2, 3)) / act_days
    F = np.concatenate([hour.reshape(n, -1), dayb.reshape(n, -1), mix, np.log1p(tot.sum(1, keepdims=True)) / 8,
                        np.log1p(z["lot_med"]) / 5, z["lot_all"], np.log1p(evday) / 2, z["front"] * 2], 1).astype(np.float32)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    Z = (F - mu) / sd
    best = None
    for r in range(4):
        lab, C, inert = kmeans(Z, a.k, 7 + r)
        print(f"[cluster] restart {r}: inertia {inert:.0f}", flush=True)
        if best is None or inert < best[2]:
            best = (lab, C, inert)
    lab = best[0]
    win = (z["bank"] > z["obank"]).astype(np.float32)
    out = []
    for j in range(a.k):
        idx = np.where(lab == j)[0]
        if len(idx) == 0:
            continue
        p_step = ev[idx].reshape(len(idx), NI, 720).mean(0)
        teams = Counter(z["team"][idx].tolist()).most_common(8)
        out.append({
            "id": j, "n": int(len(idx)), "mean_rating": float(np.mean(z["rating"][idx][z["rating"][idx] > 0])) if (z["rating"][idx] > 0).any() else 0.0,
            "win_rate": float(win[idx].mean()),
            "p_hour": {it: [round(float(x), 4) for x in hour[idx, k].mean(0)] for k, it in enumerate(items)},
            "p_step": {it: [round(float(x), 4) for x in p_step[k]] for k, it in enumerate(items)},
            "units_day": {it: [round(float(x), 1) for x in units[idx, k].mean(0)] for k, it in enumerate(items)},
            "mix": {it: round(float(mix[idx, k].mean()), 3) for k, it in enumerate(items)},
            "lot": {it: round(float(np.median(z["lot_med"][idx, k][z["lot_med"][idx, k] > 0])) if (z["lot_med"][idx, k] > 0).any() else 0.0, 1) for k, it in enumerate(items)},
            "sell_all": {it: round(float(z["lot_all"][idx, k].mean()), 3) for k, it in enumerate(items)},
            "front": {it: round(float(z["front"][idx, k].mean()), 3) for k, it in enumerate(items)},
            "teams": teams,
        })
    out.sort(key=lambda c: -c["n"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"items": items, "k": a.k, "player_games": int(n), "clusters": out}, open(a.out, "w"))
    np.save(a.out.replace(".json", "_labels.npy"), lab)
    print(f"[cluster] {n} player-games, {len(out)} clusters -> {a.out}", flush=True)
    for c in out:
        peak = {it: int(np.argmax(c["p_hour"][it])) for it in items if max(c["p_hour"][it]) > 0.3}
        fr = {it: v for it, v in c["front"].items() if v > 0.15}
        print(f"  c{c['id']:2d} n {c['n']:6d} rating {c['mean_rating']:6.0f} win {c['win_rate']:.2f} | mix {{{', '.join(f'{k[:3]}:{v:.2f}' for k, v in c['mix'].items() if v > 0.05)}}} "
              f"| peak hr {peak} | front-run {fr} | teams {[t for t, _ in c['teams'][:3]]}", flush=True)


if __name__ == "__main__":
    main()
