"""Permutation feature importance for a trained head (data-driven feature work).

Loads {kind}_nn.json (mu/sd/layers) + one validation chunk, runs a numpy forward
pass, then for each of the 107 features shuffles that column and measures the AUC
drop. Big drop = the head relies on it; ~0 = dead weight (candidate to drop or a
signal that needs a better feature). Groups per-product features for readability.

Run: python -m kaggriculture.bandit.nn.importance --kind opp
"""
import argparse, glob, json, os
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.bandit.nn.extract as EX

OUT = os.path.join(ROOT, ".local", "nn")

def feature_names():
    P = EX.PRODUCTS
    n = []
    for p in P:
        n += [f"{p}.inv", f"{p}.price", f"{p}.shed"]
    n += ["day", "step", "me", "opp", "gap"]
    n += [f"opp_dmoney_{k}" for k in (1, 2, 4, 8, 16)]
    n += ["my_dmoney_4", "town4", "town24", "opp_cadence8", "opp_cadence16",
          "n_hands", "shed_total"]
    for k in (4, 8, 16):
        n += [f"{p}.inv_d{k}" for p in P]
    for k in (4, 8, 16):
        n += [f"{p}.price_d{k}" for p in P]
    n += [f"{p}.scarc" for p in P]
    return n

def forward(nn, X):
    z = (X - np.array(nn["mu"])) / np.array(nn["sd"])
    layers = nn["layers"]
    for li, L in enumerate(layers):
        W = np.array(L["w"]); b = np.array(L["b"])
        z = z @ W.T + b
        if li < len(layers) - 1:
            z = np.maximum(z, 0.0)
    return 1.0 / (1.0 + np.exp(-z))  # sigmoid

def auc(y, p):
    order = np.argsort(p); ys = y[order]
    pos = ys.sum(); neg = len(ys) - pos
    if pos == 0 or neg == 0:
        return 0.5
    return float((np.cumsum(1 - ys) * ys).sum() / (pos * neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sale", "opp", "clone"], required=True)
    ap.add_argument("--chunks-dir", default=os.path.join(OUT, "chunks"))
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    nn = json.load(open(os.path.join(OUT, f"{a.kind}_nn.json")))
    xpat, ypat = {"sale": ("sale_X", "sale_Y"), "opp": ("opp_X", "opp_y"),
                  "clone": ("clone_X", "clone_y")}[a.kind]
    xs = sorted(glob.glob(os.path.join(a.chunks_dir, f"{xpat}_*.npy")))
    ys = sorted(glob.glob(os.path.join(a.chunks_dir, f"{ypat}_*.npy")))
    vi = len(xs) // 2 if len(xs) > 2 else len(xs) - 1
    X = np.load(xs[vi]).astype(np.float64); Y = np.load(ys[vi]).reshape(len(X), -1).astype(np.float64)
    names = feature_names()
    assert X.shape[1] == len(names), f"feat count {X.shape[1]} != names {len(names)}"
    p0 = forward(nn, X)
    base = np.mean([auc(Y[:, j], p0[:, j]) for j in range(Y.shape[1])])
    print(f"{a.kind}: base mean_auc={base:.4f}  (n={len(X)}, feats={X.shape[1]})", flush=True)
    rng = np.random.default_rng(0)
    drops = []
    for fi in range(X.shape[1]):
        Xp = X.copy(); Xp[:, fi] = rng.permutation(Xp[:, fi])
        pp = forward(nn, Xp)
        a2 = np.mean([auc(Y[:, j], pp[:, j]) for j in range(Y.shape[1])])
        drops.append((names[fi], base - a2))
    drops.sort(key=lambda t: -t[1])
    print(f"\nTOP {a.top} features by AUC drop when shuffled:", flush=True)
    for nm, d in drops[:a.top]:
        print(f"  {nm:20s} {d:+.4f}", flush=True)
    dead = [nm for nm, d in drops if abs(d) < 1e-4]
    print(f"\n{len(dead)} near-dead features (|drop|<1e-4): {dead[:20]}", flush=True)

if __name__ == "__main__":
    main()
