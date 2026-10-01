"""Thorough head-to-head: sale (v1) vs sale_v2 (shed-masked), on a COMMON basis.

Both nets are scored on the SAME held-out chunk. The deployment-matched metric is
per-product AUC over rows where the product is actually HELD (shed>0) -- that is the
only situation in which the sell rail consults the head. We also report all-rows AUC
and the volume-weighted mean, then declare a winner by conditional volume-weighted
AUC and per-product win count. The winner is the sale variant that goes into the
joint (opponent+sale) multiclass NN.

Run: python -m kaggriculture.bandit.nn.verify_sale
"""
import glob, json, os
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.bandit.nn.extract as EX

OUT = os.path.join(ROOT, ".local", "nn")
CH = os.path.join(OUT, "chunks")
SHED_IDX = [j * 3 + 2 for j in range(len(EX.PRODUCTS))]
# a row's WORLD = which PREMIUM products are in play (held). Premium drives the
# economy, so its availability pattern is the cleanest per-row world signature we
# can read without game ids. (Faithful realized-world tags would need a re-extract;
# this proxy answers "does v2 win across the distinct premium-availability worlds".)
PREM_IDX = [EX.PRODUCTS.index(p) for p in EX.PREMIUM]           # product indices
PREM_SHED = [j * 3 + 2 for j in PREM_IDX]                        # their shed features


def fwd(nn, X):
    z = (X - np.array(nn["mu"], np.float32)) / np.array(nn["sd"], np.float32)
    L = nn["layers"]
    for i, l in enumerate(L):
        z = z @ np.array(l["w"], np.float32).T + np.array(l["b"], np.float32)
        if i < len(L) - 1:
            z = np.maximum(z, 0)
    return 1 / (1 + np.exp(-z))


def auc(y, p):
    o = np.argsort(p); ys = y[o]; pos = ys.sum(); neg = len(ys) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((np.cumsum(1 - ys) * ys).sum() / (pos * neg))


def vw(aucs, weights):
    num = sum(a * w for a, w in zip(aucs, weights) if a == a)
    den = sum(w for a, w in zip(aucs, weights) if a == a)
    return num / den if den else float("nan")


def main():
    v1p = os.path.join(OUT, "sale_nn.json"); v2p = os.path.join(OUT, "sale_v2_nn.json")
    assert os.path.exists(v1p) and os.path.exists(v2p), "need sale_nn.json AND sale_v2_nn.json"
    v1 = json.load(open(v1p)); v2 = json.load(open(v2p))
    xs = sorted(glob.glob(os.path.join(CH, "sale_X_*.npy")))
    ys = sorted(glob.glob(os.path.join(CH, "sale_Y_*.npy")))
    vi = len(xs) // 2                       # common held-out chunk (val for both trainings)
    X = np.load(xs[vi]).astype(np.float32); Y = np.load(ys[vi]).astype(np.float32)
    held = X[:, SHED_IDX] > 0               # per-product held mask
    P1 = fwd(v1, X); P2 = fwd(v2, X)
    print(f"held-out chunk {os.path.basename(xs[vi])}  n={len(X)}\n", flush=True)
    hdr = f"{'product':12s} {'held%':>6s} | {'v1_all':>6s} {'v2_all':>6s} | {'v1_held':>7s} {'v2_held':>7s}  winner"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    a1_all = []; a2_all = []; a1_h = []; a2_h = []; w_all = []; w_h = []
    wins = {"v1": 0, "v2": 0, "tie": 0}
    for j, pr in enumerate(EX.PRODUCTS):
        m = held[:, j]
        a1a = auc(Y[:, j], P1[:, j]); a2a = auc(Y[:, j], P2[:, j])
        a1h = auc(Y[m, j], P1[m, j]) if m.sum() else float("nan")
        a2h = auc(Y[m, j], P2[m, j]) if m.sum() else float("nan")
        a1_all.append(a1a); a2_all.append(a2a); a1_h.append(a1h); a2_h.append(a2h)
        w_all.append(float(Y[:, j].mean())); w_h.append(float(m.mean()))
        if a1h == a1h and a2h == a2h:
            d = a2h - a1h
            win = "v2" if d > 0.003 else ("v1" if d < -0.003 else "tie")
        else:
            win = "-"
        wins[win] = wins.get(win, 0) + 1
        print(f"{pr:12s} {m.mean()*100:5.1f}% | {a1a:6.3f} {a2a:6.3f} | {a1h:7.3f} {a2h:7.3f}  {win}", flush=True)
    print("-" * len(hdr), flush=True)
    print(f"\nmean AUC (all rows)      v1={np.nanmean(a1_all):.3f}  v2={np.nanmean(a2_all):.3f}", flush=True)
    print(f"mean AUC (held rows)     v1={np.nanmean(a1_h):.3f}  v2={np.nanmean(a2_h):.3f}", flush=True)
    print(f"vol-wtd AUC (all rows)   v1={vw(a1_all,w_all):.3f}  v2={vw(a2_all,w_all):.3f}", flush=True)
    print(f"vol-wtd AUC (held rows)  v1={vw(a1_h,w_h):.3f}  v2={vw(a2_h,w_h):.3f}   <-- deployment metric", flush=True)
    print(f"\nper-product held wins: {wins}", flush=True)

    # ---- PER-WORLD verification (premium-availability worlds) ----------------
    wkey = (X[:, PREM_SHED] > 0).astype(int)                     # n x 4 (premium held?)
    keys = [tuple(r) for r in wkey]
    uniq = {}
    for idx, k in enumerate(keys):
        uniq.setdefault(k, []).append(idx)
    worlds = sorted(uniq.items(), key=lambda kv: -len(kv[1]))
    print(f"\nPER-WORLD (premium held = {[p[:3] for p in EX.PREMIUM]}):", flush=True)
    print(f"{'world':>18s} {'rows':>7s} {'v1_vwAUC':>9s} {'v2_vwAUC':>9s}  win", flush=True)
    world_v2_wins = 0; world_n = 0; per_world = []
    for k, rows in worlds:
        if len(rows) < 2000:                                    # skip tiny/degenerate worlds
            continue
        rows = np.array(rows)
        Xw = X[rows]; Yw = Y[rows]; heldw = Xw[:, SHED_IDX] > 0
        P1w = P1[rows]; P2w = P2[rows]
        a1w = []; a2w = []; ww = []
        for j in range(len(EX.PRODUCTS)):
            m = heldw[:, j]
            if m.sum() < 50:
                continue
            a1w.append(auc(Yw[m, j], P1w[m, j])); a2w.append(auc(Yw[m, j], P2w[m, j]))
            ww.append(float(m.mean()))
        if not ww:
            continue
        v1w, v2w = vw(a1w, ww), vw(a2w, ww)
        win = "v2" if v2w > v1w + 0.003 else ("v1" if v1w > v2w + 0.003 else "tie")
        world_n += 1; world_v2_wins += (win != "v1")           # v2 not-worse counts
        per_world.append({"world": "".join(map(str, k)), "rows": len(rows),
                          "v1": v1w, "v2": v2w, "win": win})
        label = "".join(map(str, k))
        print(f"{label:>18s} {len(rows):7d} {v1w:9.3f} {v2w:9.3f}  {win}", flush=True)
    print(f"\nworlds where v2 >= v1: {world_v2_wins}/{world_n}", flush=True)

    # world-aware winner: v2 must win pooled AND not be worse in a majority of worlds
    win_v2 = (vw(a2_h, w_h) > vw(a1_h, w_h) and wins.get("v2", 0) >= wins.get("v1", 0)
              and world_v2_wins * 2 >= world_n)
    winner = "sale_v2" if win_v2 else "sale (v1)"
    print(f"\n>>> WINNER (into joint head): {winner}", flush=True)
    json.dump({"winner": winner,
               "vw_held": {"v1": vw(a1_h, w_h), "v2": vw(a2_h, w_h)},
               "wins": wins, "worlds_v2_ge_v1": [world_v2_wins, world_n],
               "per_world": per_world},
              open(os.path.join(OUT, "verify_sale_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
