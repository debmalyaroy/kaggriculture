"""Train the OPPONENT-REACTIVE head: features(32) -> P(opponent dumps a premium
item within OPP_HORIZON steps) in [0,1], from PUBLIC/observable state only. Tiny
MLP (32->H->1, ReLU, sigmoid). Exports .local/nn/opp_nn.json. The bandit uses
this signal to gate a defensive front-run (sell ahead when a dump is likely) --
NEVER a bet on unobserved future, only a learned prior over observable state.
Trains on GPU if available.
"""
import argparse, json, os
import numpy as np
from kaggriculture.paths import ROOT
OUT = os.path.join(ROOT, ".local", "nn")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(os.path.join(OUT, "opp_X.npy")); y = np.load(os.path.join(OUT, "opp_y.npy")).reshape(-1, 1)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = (X - mu) / sd
    n = len(Xn); idx = np.random.RandomState(0).permutation(n); cut = int(n * 0.9)
    tr, va = idx[:cut], idx[cut:]
    Xt = torch.tensor(Xn[tr], device=dev); yt = torch.tensor(y[tr], device=dev)
    Xv = torch.tensor(Xn[va], device=dev); yv = torch.tensor(y[va], device=dev)
    F = X.shape[1]
    net = nn.Sequential(nn.Linear(F, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    base = float(yt.mean().item())
    print(f"OPP train n={n} F={F} dev={dev} pos_rate={base:.3f}", flush=True)
    for ep in range(a.epochs):
        net.train(); p = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), a.batch):
            b = p[i:i + a.batch]; opt.zero_grad()
            loss = lossf(net(Xt[b]), yt[b]); loss.backward(); opt.step()
        if ep % 8 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xv))
                acc = ((pv > 0.5).float() == yv).float().mean().item()
                # AUC (rank) quick
                order = torch.argsort(pv.squeeze())
                yv_s = yv.squeeze()[order]
                pos = yv_s.sum().item(); neg = len(yv_s) - pos
                auc = ((torch.cumsum(1 - yv_s, 0) * yv_s).sum().item() / (pos * neg)) if pos and neg else 0.5
            print(f"  ep{ep:3d} val_acc={acc:.3f} auc={auc:.3f} (base_acc {max(base,1-base):.3f})", flush=True)
    W1 = net[0].weight.detach().cpu().numpy(); b1 = net[0].bias.detach().cpu().numpy()
    W2 = net[2].weight.detach().cpu().numpy(); b2 = net[2].bias.detach().cpu().numpy()
    out = {"kind": "opp", "hidden": a.hidden, "in": F, "out": 1,
           "mu": mu.tolist(), "sd": sd.tolist(),
           "w1": W1.tolist(), "b1": b1.tolist(), "w2": W2.tolist(), "b2": b2.tolist()}
    json.dump(out, open(os.path.join(OUT, "opp_nn.json"), "w"))
    print(f"wrote {OUT}/opp_nn.json (params {W1.size+b1.size+W2.size+b2.size})", flush=True)

if __name__ == "__main__":
    main()
