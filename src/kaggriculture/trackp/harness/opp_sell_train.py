"""Train the opponent sell-timing predictor (GRU) and judge its skill.

Consumes data/opp_sell/{train,val}.jsonl from opp_sell_dataset.py.
Per-item binary heads: "opponent sells >= BURST units of X within HORIZON
steps". Split is BY EPISODE upstream; here rows are treated as i.i.d.
feature vectors first (MLP baseline), because if a memoryless model already
carries the signal, the embedded agent layer can be a pure-python MLP --
cheaper and safer than a recurrent state. A GRU over the last K steps is
trained only if the MLP shows skill (AUC lift) to see whether history adds.

VERDICT RULE (dispatch-oracle lesson, 2026-09-05: +16-19pp oracle value was
unkeyable from observables): report held-out AUC per item vs base rate.
An action layer may be built ONLY for items with val AUC >= 0.70; below
that the situation is not identifiable from what the agent can see, and
grafting it would repeat the measured -20-25pp failure mode.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/trackp/harness/opp_sell_train.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import json
import os
import sys

DATA = os.path.join(ROOT, "data", "opp_sell")
OUTD = os.path.join(ROOT, "models", "trackp", "opp_sell")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass


def load(split):
    X, Y = [], []
    with open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            X.append(r["f"])
            Y.append(r["y"])
    return X, Y


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, 1):
        if y:
            rank_sum += i
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    items = [k for k, r in meta["base_rates"].items()
             if k != "ANY" and 0.005 < r < 0.95]
    Xtr, Ytr = load("train")
    Xva, Yva = load("val")
    print(f"train {len(Xtr)} rows, val {len(Xva)} rows, heads: {items}")
    xt = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    xv = torch.tensor(np.asarray(Xva, dtype=np.float32))
    yt = torch.tensor(np.asarray([[y[i] for i in items] for y in Ytr],
                                 dtype=np.float32))
    yv_np = np.asarray([[y[i] for i in items] for y in Yva], dtype=np.float32)
    mu, sd = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt = (xt - mu) / sd
    xv = (xv - mu) / sd

    net = nn.Sequential(nn.Linear(xt.shape[1], 64), nn.Tanh(),
                        nn.Linear(64, 32), nn.Tanh(),
                        nn.Linear(32, len(items)))
    # class-imbalance weights from the train base rates
    rates = yt.mean(0).clamp(1e-4, 1 - 1e-4)
    posw = ((1 - rates) / rates).clamp(max=30.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(xt)
    best = None
    for ep in range(12):
        perm = torch.randperm(n)
        net.train()
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            sv = torch.sigmoid(net(xv)).numpy()
        aucs = {it: auc(list(sv[:, j]), list(yv_np[:, j]))
                for j, it in enumerate(items)}
        mean_auc = float(np.nanmean(list(aucs.values())))
        print(f"epoch {ep+1:2d} loss {tot/n:.4f} val AUC "
              + " ".join(f"{it}:{a:.3f}" for it, a in aucs.items()))
        if best is None or mean_auc > best[0]:
            best = (mean_auc, {k: float(v) for k, v in aucs.items()},
                    [p.detach().clone() for p in net.parameters()])
    os.makedirs(OUTD, exist_ok=True)
    mean_auc, aucs, params = best
    ship = sorted(it for it, a in aucs.items() if a >= 0.70)
    W = [params[i].numpy().tolist() for i in (0, 2, 4)]
    B = [params[i].numpy().tolist() for i in (1, 3, 5)]
    json.dump({"items": items, "aucs": aucs, "mean_auc": mean_auc,
               "shippable_heads": ship,
               "base_rates": {i: meta["base_rates"][i] for i in items},
               "x_mu": mu.numpy().tolist(), "x_sd": sd.numpy().tolist(),
               "W": W, "b": B},
              open(os.path.join(OUTD, "mlp.json"), "w"))
    print(f"\nVERDICT mean val AUC {mean_auc:.3f}")
    print(f"heads clearing the 0.70 action bar: {ship or 'NONE'}")
    print(f"-> {os.path.join(OUTD, 'mlp.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
