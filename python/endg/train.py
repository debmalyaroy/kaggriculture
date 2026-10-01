"""Train the learned endgame controller (crates/agent/src/endg.rs) on exact counterfactual labels
(crates/runner/src/bin/endg-label.rs rows: id seat seed opp world x[NF] gap_0..gap_{K-1}).

    python python/endg/train.py --labels data/endg/labels/tr_*.tsv --proposals configs/endg/proposals.json \
        --out weights/endg/v1 [--epochs 300] [--hidden 64]

Two nets (inputs standardised, exported with mean/std; "util": "updown"):
  objective  x -> 2K: per proposal k, the logit that k turns the result UP vs proposal 0 (win/draw/loss 1/0.5/0) and
             the logit that it turns it DOWN (BCE; k = 0 masked); utility u_k = sigmoid(up_k) - sigmoid(down_k)
  proposal   x -> K logits: soft target softmax((W_k - W_0) / 0.1) (CE)
Validation = 20% of the SEEDS (grouped). The decision rule (rerank: top_m by prior + proposal 0, argmax utility,
kept when it beats proposal 0 by `gate`) is tuned on validation over (gate, top_m), scored as paired
(better - 3 x worse) vs proposal 0 with the FIT-only (no mirror / v63.5) part not negative, and the chosen setting is written with the nets to OUT/endg.json.
"""
import argparse
import glob
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(paths, k):
    ids, opps, X, G = [], [], [], []
    for p in paths:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if len(f) < 5 + k:
                continue
            x = [float(v) for v in f[5:-k]]
            g = [float(v) for v in f[-k:]]
            ids.append(f"{f[2]}_{f[1]}_{f[3]}")
            opps.append(f[3])
            X.append(x)
            G.append(g)
    return ids, opps, np.array(X, np.float32), np.array(G, np.float32)


STD_CONST, ZMAX = 2e-3, 8.0   # must match crates/agent/src/endg.rs


def standardise(X, mean, std):
    """Constant-in-training features (std at the 1e-3 floor) -> 0; others clamped to +-ZMAX (endg.rs Mlp::run)."""
    z = np.clip((X - mean) / std, -ZMAX, ZMAX)
    z[:, std <= STD_CONST] = 0.0
    return z.astype(np.float32)


def wins(g):
    return (g > 0).astype(np.float32) + 0.5 * (g == 0).astype(np.float32)


class Net(nn.Module):
    def __init__(self, nx, nout, h):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nx, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, nout))

    def forward(self, x):
        return self.f(x)


def export(net, mean, std):
    ls = [m for m in net.f if isinstance(m, nn.Linear)]
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "layers": [{"w": m.weight.detach().cpu().numpy().round(6).tolist(), "b": m.bias.detach().cpu().numpy().round(6).tolist()} for m in ls]}


def rule(obj_out, prop_out, k, gate, top_m):
    sg = lambda v: 1 / (1 + np.exp(-np.clip(v, -30, 30)))  # noqa: E731
    u = sg(obj_out[:, :k]) - sg(obj_out[:, k:2 * k])
    u[:, 0] = 0.0
    picks = []
    for i in range(len(u)):
        cands = list(np.argsort(-prop_out[i])[:top_m]) if top_m < k else list(range(k))
        if 0 not in cands:
            cands.append(0)
        best = max(cands, key=lambda c: (u[i, c], -c))
        picks.append(best if u[i, best] > gate else 0)
    return np.array(picks)


def paired(picks, W):
    ch = W[np.arange(len(picks)), picks]
    b = int(((ch > W[:, 0])).sum())
    w = int(((ch < W[:, 0])).sum())
    n = b + w
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n)
    return b, w, p, float(ch.mean()), float(W[:, 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="glob(s), comma separated")
    ap.add_argument("--proposals", default=os.path.join(RL, "configs", "endg", "proposals.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = json.load(open(a.proposals))
    k = len(cfg["proposals"])
    paths = sorted(p for g in a.labels.split(",") for p in glob.glob(g))
    ids, opps, X, G = load(paths, k)
    n, nx = X.shape
    print(f"[endg-train] {n} games x {k} proposals, {nx} inputs from {len(paths)} files", flush=True)
    W = wins(G)
    D = np.clip((G - G[:, :1]) / 2000.0, -3, 3)
    val = np.array([int(hashlib.md5(i.split("_")[0].encode()).hexdigest(), 16) % 5 == 0 for i in ids])
    oracle = W.max(1)
    print(f"[endg-train] headroom: base win {W[:, 0].mean():.3f}, oracle {oracle.mean():.3f}; games where some proposal flips the result: "
          f"{int((oracle > W[:, 0]).sum())} better / {int((W.min(1) < W[:, 0]).sum())} can-worsen", flush=True)
    for j, p in enumerate(cfg["proposals"]):
        b, w = int((W[:, j] > W[:, 0]).sum()), int((W[:, j] < W[:, 0]).sum())
        print(f"[endg-train]   {j:2d} {p['name']:22s} alone vs base +{b}/-{w}  mean dm ${D[:, j].mean() * 2000:+.0f}", flush=True)
    mean = X[~val].mean(0)
    std = np.maximum(X[~val].std(0), 1e-3)
    Xt = torch.tensor(standardise(X, mean, std))
    UP = torch.tensor((W > W[:, :1]).astype(np.float32))
    DN = torch.tensor((W < W[:, :1]).astype(np.float32))
    mask = torch.ones(k); mask[0] = 0
    soft = torch.softmax(torch.tensor(W - W[:, :1]) / 0.1, 1)
    tr, va = torch.tensor(~val), torch.tensor(val)
    obj, prop = Net(nx, 2 * k, a.hidden), Net(nx, k, a.hidden)
    opt = torch.optim.AdamW(list(obj.parameters()) + list(prop.parameters()), lr=1e-3, weight_decay=a.wd)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def oloss(o, up, dn):
        return ((bce(o[:, :k], up) + bce(o[:, k:], dn)) * mask).sum(1).mean() / mask.sum()

    best, best_state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        obj.train(); prop.train()
        idx = torch.nonzero(tr).flatten()
        idx = idx[torch.randperm(len(idx))]
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            xb = Xt[b] + 0.05 * torch.randn_like(Xt[b])
            loss = oloss(obj(xb), UP[b], DN[b]) - (soft[b] * torch.log_softmax(prop(xb), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        obj.eval(); prop.eval()
        with torch.no_grad():
            vl = float(oloss(obj(Xt[va]), UP[va], DN[va]))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = ({kk: v.clone() for kk, v in obj.state_dict().items()}, {kk: v.clone() for kk, v in prop.state_dict().items()}, ep)
        else:
            bad += 1
        if ep % 10 == 0:
            print(f"[endg-train] epoch {ep}: val up/down loss {vl:.4f} (best {best:.4f} @ {best_state[2]})", flush=True)
        if bad >= 30:
            break
    obj.load_state_dict(best_state[0]); prop.load_state_dict(best_state[1])
    with torch.no_grad():
        oo, pp = obj(Xt).numpy(), prop(Xt).numpy()
    fit = np.array([o not in ("mirror", "v63.5") for o in opps])
    grid = []
    for gate in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        for top_m in (3, 6, k):
            pk = rule(oo[val], pp[val], k, gate, top_m)
            b, w, p, m, m0 = paired(pk, W[val])
            fb, fw, _, _, _ = paired(pk[fit[val]], W[val][fit[val]])
            grid.append(((fb - fw) >= 0, b - 3 * w, -p, gate, top_m, b, w, p, m, m0, float((pk != 0).mean()), fb, fw))
    grid.sort(reverse=True)
    g = grid[0]
    gate, top_m = g[3], g[4]
    lam = 0.0
    pk_tr = rule(oo[~val], pp[~val], k, gate, top_m)
    btr = paired(pk_tr, W[~val])
    print(f"[endg-train] chosen gate {gate} top_m {top_m}: VAL +{g[5]}/-{g[6]} p {g[7]:.4f} win {g[9]:.3f} -> {g[8]:.3f} "
          f"(switch rate {g[10]:.2f}; FIT-only +{g[11]}/-{g[12]}); TRAIN +{btr[0]}/-{btr[1]}", flush=True)
    for x in grid[:8]:
        print(f"[endg-train]   gate {x[3]} top_m {x[4]}: val +{x[5]}/-{x[6]} fit +{x[11]}/-{x[12]} switch {x[10]:.2f}", flush=True)
    pk_all = rule(oo, pp, k, gate, top_m)
    hist = {cfg["proposals"][j]["name"]: int((pk_all == j).sum()) for j in range(k) if (pk_all == j).any()}
    print(f"[endg-train] picks: {hist}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    out = dict(cfg)
    out.update({"mode": "rerank", "util": "updown", "lam": lam, "gate": gate, "top_m": top_m, "objective": export(obj, mean, std), "proposal": export(prop, mean, std),
                "note": f"learned endgame controller, trained {n} games ({len(paths)} files), best epoch {best_state[2]}"})
    json.dump(out, open(os.path.join(a.out, "endg.json"), "w"), separators=(",", ":"))
    rep = {"games": n, "k": k, "val": {"better": g[5], "worse": g[6], "p": g[7], "win": g[8], "base_win": g[9], "switch": g[10], "fit_better": g[11], "fit_worse": g[12]},
           "train": {"better": btr[0], "worse": btr[1]}, "lam": lam, "gate": gate, "top_m": top_m, "picks": hist,
           "grid_top": [[float(v) if not isinstance(v, (bool, np.bool_)) else bool(v) for v in x] for x in grid[:10]]}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"[endg-train] -> {a.out}/endg.json", flush=True)


if __name__ == "__main__":
    main()
