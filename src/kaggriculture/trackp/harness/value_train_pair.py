"""Pairwise sibling-rank value net: train on the deployed question.

Reads data/selfplay/siblings_*.jsonl (sibling_gen.py: groups of plan
variants from one decision point, leaf features at +48, terminal margin).
Loss = margin regression (keeps the output in DOLLARS — the searcher's
_SEARCH_MIN_EDGE=1500 gate needs calibrated units) + pairwise logistic on
within-group score differences (the ranking the searcher actually uses).
Validation = held-out PAIR-ORDER ACCURACY on whole groups (split by gid),
not sign-acc on random states — the metric that failed to transfer four
times (v2/v3/v4/v5).

Export format identical to value_train.py: build_searcher --valuenet
swaps the blob with no code change.

    C:/ProgramData/anaconda3/envs/llm/python.exe \
        src/trackp/harness/value_train_pair.py \
        --out models/trackp/valuenet/value_net_p1.json
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

MIN_GAP = 500.0          # ignore pairs closer than this (label noise)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "valuenet", "value_net_p1.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, nargs=2, default=[128, 64])
    ap.add_argument("--dims", type=int, default=39)
    ap.add_argument("--glob", default="siblings_*.jsonl")
    ap.add_argument("--rank-weight", type=float, default=2.0)
    a = ap.parse_args()
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "selfplay",
                                           a.glob))):
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            if len(r["f"]) != a.dims:
                continue
            groups[r["gid"]].append((r["f"], float(r["y_margin"])))
    gids = sorted(groups)
    print(f"{len(gids):,} groups, "
          f"{sum(len(v) for v in groups.values()):,} rows")
    # split by GROUP so no sibling set straddles train/val
    n_val = max(200, len(gids) // 8)
    val_g, tr_g = gids[:n_val], gids[n_val:]

    def flatten(gs):
        X, Y, pi, pj = [], [], [], []
        for g in gs:
            rows = groups[g]
            base = len(X)
            for f, m in rows:
                X.append(f)
                Y.append(m)
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if abs(rows[i][1] - rows[j][1]) < MIN_GAP:
                        continue
                    if rows[i][1] > rows[j][1]:
                        pi.append(base + i)
                        pj.append(base + j)
                    else:
                        pi.append(base + j)
                        pj.append(base + i)
        return (torch.tensor(np.asarray(X, dtype=np.float32)),
                torch.tensor(np.asarray(Y, dtype=np.float32)) / 1e4,
                torch.tensor(pi, dtype=torch.long),
                torch.tensor(pj, dtype=torch.long))

    Xt, Yt, ti, tj = flatten(tr_g)
    Xv, Yv, vi, vj = flatten(val_g)
    print(f"train {len(Xt):,} rows / {len(ti):,} pairs;"
          f" val {len(Xv):,} rows / {len(vi):,} pairs")
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xt_, Xv_ = (Xt - mu) / sd, (Xv - mu) / sd
    h1, h2 = a.hidden
    net = nn.Sequential(nn.Linear(a.dims, h1), nn.Tanh(),
                        nn.Linear(h1, h2), nn.Tanh(),
                        nn.Linear(h2, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best = None
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(ti))
        for i in range(0, len(ti), 8192):
            pidx = perm[i:i + 8192]
            ii, jj = ti[pidx], tj[pidx]
            opt.zero_grad()
            si = net(Xt_[ii]).squeeze(-1)
            sj = net(Xt_[jj]).squeeze(-1)
            rank = nn.functional.softplus(-(si - sj)).mean()
            reg = (nn.functional.smooth_l1_loss(si, Yt[ii])
                   + nn.functional.smooth_l1_loss(sj, Yt[jj]))
            (a.rank_weight * rank + reg).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sv = net(Xv_).squeeze(-1)
            pacc = float((sv[vi] > sv[vj]).float().mean())
            mae = float(nn.functional.l1_loss(sv, Yv)) * 1e4
        if ep % 5 == 4 or ep == 0:
            print(f"epoch {ep+1:2d} val pair-acc {pacc:.3f} "
                  f"MAE ${mae:,.0f}", flush=True)
        if best is None or pacc > best[0]:
            best = (pacc, mae, [p.detach().clone()
                                for p in net.parameters()])
    pacc, mae, params = best
    W = [p.numpy().tolist() for p in params]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"x_mu": mu.tolist(), "x_sd": sd.tolist(),
               "W": [W[0], W[2], W[4]], "b": [W[1], W[3], W[5]],
               "rows": len(Xt) + len(Xv), "val_pair_acc": pacc,
               "val_mae": mae, "y_scale": 1e4},
              open(a.out, "w"))
    print(f"BEST pair-acc {pacc:.3f} MAE ${mae:,.0f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
