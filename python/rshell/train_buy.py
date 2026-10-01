"""Train the shell v3 BUY head (BUY_PRODUCT WHEAT) on exact counterfactual labels (crates/runner/src/bin/buylabel.rs).

    python python/rshell/train_buy.py --labels data/rshell/buy/labels_r1.tsv --base-shell data/rshell/cma_v2b_final \
        --out weights/rshell/v3b1 [--epochs 60]

Row: id seat step opp x[0..NX] m_real m5 m10 m20 (m_q = our final margin when q wheat are bought at that step and the
game is played out exactly, both sides reacting). Target per q: the score gain over not buying, score = win 1 /
draw 0.5 / loss 0 + margin / 2e4, so result flips dominate and margin breaks ties. Net: NX -> 128 -> 64 -> 3 (ReLU),
Huber on the gains. Split by game (hash of id), 20% held out. The gate (minimum predicted gain to buy) is picked on the
held-out split: the gate whose decisions realize the most score. Writes OUT/buy_net.json, OUT/rshell.json (the base
shell's config + buy_net_file + buy_gate; net.json and lineage.json copied beside it) and OUT/report.json.
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np
import torch

QS = [5, 10, 20]


def load(path):
    ids, steps, xs, ms = [], [], [], []
    for ln in open(path):
        a = ln.rstrip("\n").split("\t")
        if len(a) < 10:
            continue
        ids.append(a[0] + ":" + a[1])
        steps.append(int(a[2]))
        xs.append([float(v) for v in a[4:-4]])
        ms.append([float(v) for v in a[-4:]])
    return ids, np.array(steps), np.array(xs, dtype=np.float32), np.array(ms, dtype=np.float64)


def score(m):
    return (m > 0) * 1.0 + (m == 0) * 0.5 + m / 2e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base-shell", required=True, help="folder with mean_rshell.json (or rshell.json), mean_knobs.json, net.json, lineage.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    ids, steps, x, m = load(a.labels)
    s = score(m)
    gain = s[:, 1:] - s[:, :1]                       # per q vs not buying
    held = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 == 0 for i in ids])
    print(f"[buy] {len(ids)} decisions from {len(set(ids))} games; held-out {held.sum()}", flush=True)
    print(f"[buy] label stats: best-q gain > 0 in {(gain.max(1) > 0).mean():.1%} of decisions; flips to a win "
          f"{((m[:, 1:] > 0) & (m[:, :1] <= 0)).any(1).mean():.1%}, flips to a loss {((m[:, 1:] <= 0) & (m[:, :1] > 0)).any(1).mean():.1%}; "
          f"mean margin gain per q ${np.mean(m[:, 1:] - m[:, :1], 0).round(0).tolist()}", flush=True)
    mean, std = x[~held].mean(0), x[~held].std(0) + 1e-6
    T = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    xn = T((x - mean) / std)
    y = T(gain * 10.0)                               # ~unit scale: a result flip = 10
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, len(QS)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr = np.where(~held)[0]
    torch.manual_seed(7)
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep).permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), 512):
            idx = perm[b:b + 512]
            loss = torch.nn.functional.huber_loss(net(xn[idx]), y[idx], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 9:
            with torch.no_grad():
                hl = torch.nn.functional.huber_loss(net(xn[held]), y[held], delta=1.0).item()
            print(f"[buy] epoch {ep + 1}: train {tot / len(tr):.4f} held {hl:.4f}", flush=True)
    with torch.no_grad():
        pred = net(xn).numpy() / 10.0
    # gate on held-out: buy argmax-q when its predicted gain > gate; realized = the true gain of that choice
    hp, hg = pred[held], gain[held]
    oracle = np.maximum(hg.max(1), 0).sum()
    rep = {"decisions": int(len(ids)), "held": int(held.sum()), "oracle_gain": float(oracle), "gates": {}}
    best_gate, best_real = None, 0.0
    for gate in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        j = hp.argmax(1)
        buy = hp[np.arange(len(hp)), j] > gate
        real = hg[np.arange(len(hg)), j][buy].sum()
        flips_w = ((m[held][buy, 1:][np.arange(buy.sum()), j[buy]] > 0) & (m[held][buy, 0] <= 0)).sum()
        flips_l = ((m[held][buy, 1:][np.arange(buy.sum()), j[buy]] <= 0) & (m[held][buy, 0] > 0)).sum()
        rep["gates"][str(gate)] = {"buys": int(buy.sum()), "realized": float(real), "wins_gained": int(flips_w), "wins_lost": int(flips_l)}
        print(f"[buy] held-out gate {gate}: buys {buy.sum()} / {len(buy)}, realized score {real:+.2f} (oracle {oracle:.2f}), "
              f"results +{flips_w}/-{flips_l}", flush=True)
        if real > best_real:
            best_gate, best_real = gate, real
    rep["gate"] = best_gate
    os.makedirs(a.out, exist_ok=True)
    layers = [{"w": l.weight.detach().numpy().tolist(), "b": l.bias.detach().numpy().tolist()} for l in net if isinstance(l, torch.nn.Linear)]
    json.dump({"qs": QS, "mean": mean.tolist(), "std": std.tolist(), "layers": layers, "score_units": "win=1, margin/2e4"}, open(os.path.join(a.out, "buy_net.json"), "w"))
    src = os.path.join(a.base_shell, "mean_rshell.json") if os.path.exists(os.path.join(a.base_shell, "mean_rshell.json")) else os.path.join(a.base_shell, "rshell.json")
    cfg = json.load(open(src))
    for f in ("net.json", "lineage.json"):
        shutil.copy(os.path.join(a.base_shell, f), os.path.join(a.out, f))
    if os.path.exists(os.path.join(a.base_shell, "mean_knobs.json")):
        shutil.copy(os.path.join(a.base_shell, "mean_knobs.json"), os.path.join(a.out, "knobs.json"))
    if best_gate is None:
        print("[buy] no gate realizes a held-out gain: the buy head stays OFF (no buy_net_file written into the config)", flush=True)
    else:
        cfg["buy_net_file"] = "buy_net.json"
        cfg["buy_gate"] = best_gate * 10.0  # the net outputs gain x 10 (training scale)
    json.dump(cfg, open(os.path.join(a.out, "rshell.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"[buy] gate {best_gate} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
