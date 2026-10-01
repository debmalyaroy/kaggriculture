"""Train the reactive shell v2 network on exact-engine labels (crates/runner/src/bin/branch2.rs).

    python python/rshell/train.py --labels data/rshell/labels/*.tsv --out weights/rshell/NAME [--epochs 40]

Row (branch2): id seat step item cur_cls stock kind x[0..NX] m0..m4 m_real, m_c = our final margin when this
item's sale at this step is forced to fraction c (0, 1/4, 1/2, 3/4, all) and the game played out exactly.
Score of an outcome = win 1 / draw 0.5 / loss 0 + margin / 1e6 (the ladder pays wins, margin only breaks ties).
  class head    soft target softmax((score_c - best) / T), weight = best - worst score (+ a small floor for
                decisions where only margin moves), i.e. decisions that can flip a result dominate
  margin head   (m_c - m_cur) / 1000, Huber (a confidence gate at play time: margin_gate)
  priority head how much selling this item THIS turn is worth: best score over c>0 minus the score of c=0,
                squashed; items whose sale matters more take the earlier slots (reorder)
Architecture (crates/agent/src/rshell.rs Net): per-item encoder NX -> E -> E (ReLU), mean/max pooled over the
items decided in the same turn, head [e, mean, max] (3E) -> H -> 11 (5 logits, priority, 5 margins).
Split by game (hash of id), 20% held out. Writes NAME/net.json, NAME/rshell.json (config, net_file, lineage),
NAME/report.json (held-out agreement with the best class, score gain when applied at each tau).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil

import numpy as np
import torch

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NC = 5
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def load(paths):
    ids, meta, xs, ms = [], [], [], []
    for p in paths:
        with open(p) as f:
            for ln in f:
                a = ln.rstrip("\n").split("\t")
                if len(a) < 14:
                    continue
                ids.append(a[0] + ":" + a[1])
                meta.append((int(a[2]), a[3], int(a[4]), int(a[5]), a[6]))
                xs.append([float(v) for v in a[7:-6]])
                ms.append([float(v) for v in a[-6:-1]])
    x = np.array(xs, dtype=np.float32)
    m = np.array(ms, dtype=np.float64)
    return ids, meta, x, m


def score(m):
    return (m > 0) + 0.5 * (m == 0) + m / 1e6


class Net(torch.nn.Module):
    def __init__(self, nx, e, h):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(nx, e), torch.nn.ReLU(), torch.nn.Linear(e, e), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(3 * e, h), torch.nn.ReLU(), torch.nn.Linear(h, 2 * NC + 1))

    def forward(self, x, group):
        """x: rows; group: turn index per row (rows of one turn share the pooled context)."""
        e = self.enc(x)
        g = int(group.max()) + 1
        mean = torch.zeros(g, e.shape[1]).index_add_(0, group, e) / torch.bincount(group, minlength=g).clamp(min=1)[:, None]
        mx = torch.full((g, e.shape[1]), -1e9).scatter_reduce(0, group[:, None].expand_as(e), e, reduce="amax")
        return self.head(torch.cat([e, mean[group], mx[group]], 1))


def export(net, mean, std):
    lin = lambda l: {"w": l.weight.detach().numpy().round(6).tolist(), "b": l.bias.detach().numpy().round(6).tolist()}  # noqa: E731
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist(),
            "enc": [lin(net.enc[0]), lin(net.enc[2])], "head": [lin(net.head[0]), lin(net.head[2])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--e", type=int, default=128)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.1, help="soft-target temperature in score units")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lead-labels", nargs="*", default=[], help="branch2 --tapes --other-seat rows on top players' tapes: cur_cls = what the LEADER sold")
    ap.add_argument("--imitate", type=float, default=0.5, help="on leader rows, blend the target toward the leader's own class by this much")
    ap.add_argument("--init-net", default=None, help="fine-tune from this net.json (keeps its input normalisation and output scale, so the tuned shell settings stay valid)")
    ap.add_argument("--lineage", default=os.path.join(RL, "configs", "lineage", "v61.1.json"))
    a = ap.parse_args()
    paths = sorted({p for g in a.labels for p in glob.glob(g)})
    ids, meta, x, m = load(paths)
    lpaths = sorted({p for g in a.lead_labels for p in glob.glob(g)})
    lead = np.zeros(len(ids), dtype=bool)
    if lpaths:
        li, lm, lx, lmm = load(lpaths)
        ids, meta = ids + li, meta + lm
        x, m = np.concatenate([x, lx]), np.concatenate([m, lmm])
        lead = np.concatenate([lead, np.ones(len(li), dtype=bool)])
        print(f"[rshell-train] + {len(li)} leader decisions (imitation weight {a.imitate})", flush=True)
    n = len(ids)
    print(f"[rshell-train] {n} decisions from {len(paths)} files, {len(set(ids))} games, inputs {x.shape[1]}", flush=True)
    s = score(m)
    best, worst = s.max(1), s.min(1)
    tgt = np.exp((s - best[:, None]) / a.temp)
    tgt /= tgt.sum(1, keepdims=True)
    w = (best - worst) + 1e-3 * (np.abs(m.max(1) - m.min(1)) > 0)
    cur = np.array([mm[2] for mm in meta])
    if lead.any():
        # imitation: what the top player actually did in this exact situation (their recorded sale class)
        oh = np.zeros_like(tgt)
        oh[np.arange(n), cur] = 1.0
        tgt[lead] = (1 - a.imitate) * tgt[lead] + a.imitate * oh[lead]
        w[lead] = np.maximum(w[lead], 0.05)
    mcur = m[np.arange(n), cur]
    marg = np.clip((m - mcur[:, None]) / 1000.0, -20, 20)
    prio = np.tanh((s[:, 1:].max(1) - s[:, 0]) * 2.0)
    # turn groups (rows of one game and step)
    keys = {}
    grp = np.array([keys.setdefault((ids[i], meta[i][0]), len(keys)) for i in range(n)])
    held = np.array([int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 == 0 for i in ids])
    if held.all() or not held.any():  # tiny smoke sets: train on everything, report on everything
        held = np.zeros(n, dtype=bool)
        eval_rows = np.arange(n)
    else:
        eval_rows = np.nonzero(held)[0]
    mean, std = x[~held].mean(0), x[~held].std(0) + 1e-6
    init = json.load(open(a.init_net)) if a.init_net else None
    if init:
        mean, std = np.array(init["mean"], dtype=np.float32), np.array(init["std"], dtype=np.float32)
    xn = (x - mean) / std
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt)  # noqa: E731
    torch.manual_seed(7)
    net = Net(x.shape[1], a.e, a.h)
    if init:
        def put(layer, d):
            layer.weight.data = T(d["w"])
            layer.bias.data = T(d["b"])
        put(net.enc[0], init["enc"][0]); put(net.enc[2], init["enc"][1]); put(net.head[0], init["head"][0]); put(net.head[2], init["head"][1])
        print(f"[rshell-train] fine-tuning from {a.init_net}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-5)
    tr_groups = np.unique(grp[~held])
    for ep in range(a.epochs):
        net.train()
        np.random.default_rng(ep).shuffle(tr_groups)
        tot = 0.0
        for chunk in np.array_split(tr_groups, max(1, len(tr_groups) // 512)):
            rows = np.nonzero(np.isin(grp, chunk))[0]
            g = torch.tensor(np.unique(grp[rows], return_inverse=True)[1], dtype=torch.long)
            o = net(T(xn[rows]), g)
            lp = torch.log_softmax(o[:, :NC], 1)
            ce = -(T(tgt[rows]) * lp).sum(1)
            lm = torch.nn.functional.huber_loss(o[:, NC + 1:], T(marg[rows]), reduction="none").mean(1)
            lpr = (o[:, NC] - T(prio[rows])) ** 2
            loss = ((ce + 0.5 * lm + 0.2 * lpr) * T(w[rows] + 0.05)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"[rshell-train] epoch {ep + 1}: loss {tot:.3f}", flush=True)
    # held-out: agreement with the best class and the score gain over the chain's own class at each tau
    net.eval()
    rows = eval_rows
    with torch.no_grad():
        g = torch.tensor(np.unique(grp[rows], return_inverse=True)[1], dtype=torch.long)
        o = net(T(xn[rows]), g).numpy()
    p = np.exp(o[:, :NC] - o[:, :NC].max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    pick, conf = p.argmax(1), p.max(1)
    sb = s[rows]
    agree = float((pick == sb.argmax(1)).mean())
    rep = {"decisions": n, "held_out": int(len(eval_rows)), "agree_best": agree, "tau": {}}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        use = (conf >= tau) & (pick != cur[rows])
        gain = sb[np.arange(len(rows)), np.where(use, pick, cur[rows])] - sb[np.arange(len(rows)), cur[rows]]
        rep["tau"][str(tau)] = {"overrides": int(use.sum()), "better": int((gain > 1e-9).sum()), "worse": int((gain < -1e-9).sum()), "score_gain": float(gain.sum())}
    rep["kinds"] = {k: int(sum(1 for mm in meta if mm[4] == k)) for k in sorted({mm[4] for mm in meta})}
    os.makedirs(a.out, exist_ok=True)
    json.dump(export(net, mean, std), open(os.path.join(a.out, "net.json"), "w"))
    shutil.copy(a.lineage, os.path.join(a.out, "lineage.json"))
    best_tau = max(rep["tau"], key=lambda t: rep["tau"][t]["better"] - rep["tau"][t]["worse"])
    cfg = {"name": os.path.basename(a.out.rstrip("/")), "note": "reactive shell v2 (python/rshell/train.py); knobs tuned by python/rshell/cmaes.py",
           "net_file": "net.json", "lineage_file": "lineage.json", "tau": float(best_tau), "trained": {"labels": len(paths), "decisions": n}}
    json.dump(cfg, open(os.path.join(a.out, "rshell.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"[rshell-train] held-out agree {agree:.3f}; tau sweep " + " | ".join(f"{t}: +{v['better']}/-{v['worse']}" for t, v in rep["tau"].items()), flush=True)
    print(f"[rshell-train] -> {a.out}/rshell.json", flush=True)


if __name__ == "__main__":
    main()
