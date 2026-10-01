"""Export the GRU identifier to pure-python constants + verify equivalence.

Trains on ALL data (the shipped model always uses everything), exports
weights as JSON, and checks a stdlib-only inference (no numpy, no torch)
against torch to 1e-5 -- the same export-equivalence discipline
train_identifier.py applies to the logistic. The pure-python step cost is
~3*H*(D+H) mults per turn (H=32, D=275 -> ~30k) and the identifier runs
every 12th turn, so the 20 ms budget is untouched.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/experiments/gru_export.py
"""
from kaggriculture.paths import ROOT
import json
import math
import os

import numpy as np
import torch

from gru_lab import HIDDEN, SeqID, augment  # noqa: F401

DATA = os.path.join(ROOT, "models", "lab", "seq_dataset.npz")
OUT = os.path.join(ROOT, "models", "lab", "gru_weights.json")


def train_full():
    z = np.load(DATA, allow_pickle=False)
    X, y, k = z["X"], z["y"], int(z["classes"])
    Xa, ya = augment(X, y)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu = Xa.reshape(-1, Xa.shape[-1]).mean(0)
    sd = Xa.reshape(-1, Xa.shape[-1]).std(0) + 1e-6
    xt = torch.tensor((Xa - mu) / sd, device=dev)
    yt = torch.tensor(ya, device=dev)
    counts = np.bincount(ya, minlength=k).astype(np.float64)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) ** 0.5,
                     dtype=torch.float32, device=dev)
    model = SeqID(Xa.shape[-1], k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    n = len(xt)
    for ep in range(25):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            logits = model(xt[b])
            loss = lossf(logits.reshape(-1, k),
                         yt[b].unsqueeze(1).expand(-1, logits.shape[1])
                         .reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu().eval(), mu, sd, k, X


def heldout_day_acc():
    """Held-out-DAY accuracy, measured the way train_identifier measures the
    logistic: train on every earlier day, test on the newest. This number
    rides in the export payload so the pipeline's --gru-auto can compare the
    challenger and the champion on the same yardstick -- without it the auto
    switch would have nothing but vibes to decide on. The SHIPPED weights
    still train on ALL data (train_full); this trains a twin for measurement.
    """
    z = np.load(DATA, allow_pickle=False)
    X, y, k = z["X"], z["y"], int(z["classes"])
    dates = z["dates"]
    # numpy 2.x has no maximum ufunc loop for unicode arrays -- max() the list.
    newest = max(dates.tolist())
    tr = dates != newest
    if tr.all() or not tr.any():
        return None
    Xtr, ytr = augment(X[tr], y[tr])
    Xte, yte = augment(X[~tr], y[~tr])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, device=dev)
    yt = torch.tensor(ytr, device=dev)
    counts = np.bincount(ytr, minlength=k).astype(np.float64)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) ** 0.5,
                     dtype=torch.float32, device=dev)
    model = SeqID(Xtr.shape[-1], k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    n = len(xt)
    for ep in range(25):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            logits = model(xt[b])
            loss = lossf(logits.reshape(-1, k),
                         yt[b].unsqueeze(1).expand(-1, logits.shape[1])
                         .reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xe = torch.tensor((Xte - mu) / sd, device=dev)
        pred = model(xe)[:, -1].argmax(-1).cpu().numpy()
    return float((pred == yte).mean())


def export(model, mu, sd, k):
    sdict = {n: p.detach().numpy() for n, p in model.state_dict().items()}
    payload = {
        "mu": mu.tolist(), "sd": sd.tolist(), "k": k, "hidden": HIDDEN,
        "ln_g": sdict["norm.weight"].tolist(),
        "ln_b": sdict["norm.bias"].tolist(),
        # torch GRU packs gates as [r, z, n] along dim 0
        "w_ih": sdict["gru.weight_ih_l0"].tolist(),
        "w_hh": sdict["gru.weight_hh_l0"].tolist(),
        "b_ih": sdict["gru.bias_ih_l0"].tolist(),
        "b_hh": sdict["gru.bias_hh_l0"].tolist(),
        "w_out": sdict["head.weight"].tolist(),
        "b_out": sdict["head.bias"].tolist(),
    }
    json.dump(payload, open(OUT, "w", encoding="utf-8"))
    return payload


def stdlib_forward(P, seq):
    """Pure-python (math only) GRU forward -- the exact agent-side code."""
    H = P["hidden"]
    h = [0.0] * H
    out = None
    for x in seq:
        # layernorm
        xn = [(v - m) / s for v, m, s in zip(x, P["mu"], P["sd"])]
        mean = sum(xn) / len(xn)
        var = sum((v - mean) ** 2 for v in xn) / len(xn)
        inv = 1.0 / math.sqrt(var + 1e-5)
        xl = [(v - mean) * inv * g + b
              for v, g, b in zip(xn, P["ln_g"], P["ln_b"])]
        gi = [sum(w * v for w, v in zip(P["w_ih"][j], xl)) + P["b_ih"][j]
              for j in range(3 * H)]
        gh = [sum(w * v for w, v in zip(P["w_hh"][j], h)) + P["b_hh"][j]
              for j in range(3 * H)]
        nh = []
        for j in range(H):
            r = 1.0 / (1.0 + math.exp(-(gi[j] + gh[j])))
            z = 1.0 / (1.0 + math.exp(-(gi[H + j] + gh[H + j])))
            n = math.tanh(gi[2 * H + j] + r * gh[2 * H + j])
            nh.append((1.0 - z) * n + z * h[j])
        h = nh
        out = h
    z = [sum(w * v for w, v in zip(P["w_out"][j], out)) + P["b_out"][j]
         for j in range(P["k"])]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def main():
    acc = heldout_day_acc()
    print(f"held-out-day accuracy (measurement twin): "
          f"{acc if acc is None else f'{acc:.4f}'}")
    model, mu, sd, k, X = train_full()
    P = export(model, mu, sd, k)
    if acc is not None:
        P["heldout_acc"] = acc
        json.dump(P, open(OUT, "w", encoding="utf-8"))
    # equivalence on 50 raw (un-normalized) sequences
    import time
    worst = 0.0
    t_tot = 0.0
    idxs = range(0, len(X), max(1, len(X) // 50))
    with torch.no_grad():
        for i in idxs:
            seq = X[i].astype(np.float64)
            xt = torch.tensor(((seq - mu) / sd)[None].astype(np.float32))
            p_t = torch.softmax(model(xt)[0, -1], dim=-1).numpy()
            t0 = time.perf_counter()
            p_py = stdlib_forward(P, [list(r) for r in seq])
            t_tot += time.perf_counter() - t0
            worst = max(worst, float(np.abs(p_t - np.array(p_py)).max()))
    n = len(list(idxs))
    print(f"export equivalence: worst |diff| {worst:.2e} "
          f"({'OK' if worst < 1e-5 else 'FAIL'}); "
          f"stdlib forward {1000 * t_tot / n:.1f} ms per full 6-step call")
    print(f"-> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
