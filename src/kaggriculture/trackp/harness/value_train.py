"""Train the day-boundary value net locally (v2+: bigger data, wider net).

Reads every data/selfplay/states_*.jsonl (local generator writes y_margin,
the Colab notebook writes y — both accepted), trains margin regression,
reports held-out MAE + sign-accuracy, exports pure-JSON weights.

    C:/ProgramData/anaconda3/envs/llm/python.exe \
        src/trackp/harness/value_train.py --out models/trackp/valuenet/value_net_v2.json
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "valuenet", "value_net_v2.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, nargs=2, default=[128, 64])
    ap.add_argument("--dims", type=int, default=39)
    ap.add_argument("--glob", default="states_*.jsonl")
    a = ap.parse_args()
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    X, Y = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "selfplay",
                                           a.glob))):
        for ln in open(p, encoding="utf-8"):
            r = json.loads(ln)
            y = r.get("y_margin", r.get("y"))
            if y is None or len(r["f"]) != a.dims:
                continue
            X.append(r["f"])
            Y.append(y)
    print(f"{len(X):,} rows from {len(glob.glob(os.path.join(ROOT, 'data', 'selfplay', 'states_*.jsonl')))} shards")
    X = torch.tensor(np.asarray(X, dtype=np.float32))
    Y = torch.tensor(np.asarray(Y, dtype=np.float32)) / 1e4
    # split by position (rows from the same game are adjacent; use a
    # stride split to keep games together-ish)
    n_val = max(2000, len(X) // 8)
    Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
    mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    h1, h2 = a.hidden
    net = nn.Sequential(nn.Linear(a.dims, h1), nn.Tanh(),
                        nn.Linear(h1, h2), nn.Tanh(),
                        nn.Linear(h2, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt_, Yt_ = (Xt - mu) / sd, Yt
    Xv_, Yv_ = (Xv - mu) / sd, Yv
    best = None
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt_))
        net.train()
        for i in range(0, len(Xt_), 8192):
            idx = perm[i:i + 8192]
            opt.zero_grad()
            loss = nn.functional.smooth_l1_loss(
                net(Xt_[idx]).squeeze(-1), Yt_[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vp = net(Xv_).squeeze(-1)
            vl = float(nn.functional.l1_loss(vp, Yv_)) * 1e4
            mask = Yv_ != 0
            sign = float(((vp[mask] > 0) == (Yv_[mask] > 0)).float().mean())
        if ep % 5 == 4 or ep == 0:
            print(f"epoch {ep+1:2d} val MAE ${vl:,.0f} sign-acc {sign:.3f}",
                  flush=True)
        if best is None or sign > best[0]:
            best = (sign, vl, [p.detach().clone()
                               for p in net.parameters()])
    sign, vl, params = best
    W = [p.numpy().tolist() for p in params]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"x_mu": mu.tolist(), "x_sd": sd.tolist(),
               "W": [W[0], W[2], W[4]], "b": [W[1], W[3], W[5]],
               "rows": len(X), "val_mae": vl, "sign_acc": sign,
               "y_scale": 1e4}, open(a.out, "w"))
    print(f"BEST sign-acc {sign:.3f} MAE ${vl:,.0f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
