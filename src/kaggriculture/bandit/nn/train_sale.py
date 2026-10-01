"""Train the SALE sell-head: features(32) -> per-product sell-fraction(9) in [0,1].
BC on strong GM agents' actual sells. Tiny MLP (32->H->9, ReLU, sigmoid) so
inference is a trivial Rust matmul. Exports flat weights + input standardisation
to .local/nn/sale_nn.json (embed into config.mlp_sell.weights to ship).
Trains on GPU if available.
"""
import argparse, json, os
import numpy as np
from kaggriculture.paths import ROOT
OUT = os.path.join(ROOT, ".local", "nn")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(os.path.join(OUT, "sale_X.npy")); Y = np.load(os.path.join(OUT, "sale_Y.npy"))
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    n = len(Xn); idx = np.random.RandomState(0).permutation(n); cut = int(n * 0.9)
    tr, va = idx[:cut], idx[cut:]
    Xt = torch.tensor(Xn[tr], device=dev); Yt = torch.tensor(Y[tr], device=dev)
    Xv = torch.tensor(Xn[va], device=dev); Yv = torch.tensor(Y[va], device=dev)
    F, O = X.shape[1], Y.shape[1]
    net = nn.Sequential(nn.Linear(F, a.hidden), nn.ReLU(), nn.Linear(a.hidden, O), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    print(f"SALE train n={n} F={F} O={O} dev={dev}", flush=True)
    for ep in range(a.epochs):
        net.train(); p = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), a.batch):
            b = p[i:i + a.batch]; opt.zero_grad()
            loss = lossf(net(Xt[b]), Yt[b]); loss.backward(); opt.step()
        if ep % 8 == 0 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xv), Yv).item()
                # baseline MSE = predict training mean
                base = ((Yv - Yt.mean(0)) ** 2).mean().item()
            print(f"  ep{ep:3d} val_mse={vl:.5f} (mean-baseline {base:.5f})", flush=True)
    W1 = net[0].weight.detach().cpu().numpy(); b1 = net[0].bias.detach().cpu().numpy()
    W2 = net[2].weight.detach().cpu().numpy(); b2 = net[2].bias.detach().cpu().numpy()
    out = {"kind": "sale", "hidden": a.hidden, "in": F, "out": O,
           "mu": mu.tolist(), "sd": sd.tolist(),
           "w1": W1.tolist(), "b1": b1.tolist(), "w2": W2.tolist(), "b2": b2.tolist()}
    json.dump(out, open(os.path.join(OUT, "sale_nn.json"), "w"))
    print(f"wrote {OUT}/sale_nn.json  (params {W1.size+b1.size+W2.size+b2.size})", flush=True)

if __name__ == "__main__":
    main()
