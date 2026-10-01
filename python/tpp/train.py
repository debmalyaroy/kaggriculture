"""Train the top-player policy (TPP) by behaviour cloning on bcdump records (operator 28 Sep: learn the top
players' turns, economy, decisions and selling -- every action).

    python python/tpp/train.py --data data/tpp/d1 --out weights/tpp/bc1 [--epochs 3] [--batch 1024]

Net (the Rust agent runs the same forward, crates/agent/src/tpp.rs):
  x   = board (C x 10 x 10, u8/255) ++ broadcast(relu(Wg . glob))            -> CIN = C + GE channels
  trunk: conv3x3 d1 -> conv3x3 d1 -> conv3x3 d2 -> conv3x3 d4 -> conv3x3 d1, ReLU each (48 ch; residual from layer 2)
  pooled = mean ++ max over the board (2 x 48)
  unit head (per unit): trunk at the unit's tile and its N/S/E/W neighbours (5 x 48) ++ pooled ++ genc ++ unit row
                        -> 128 -> {44 classes, 8 qty buckets}
  market head: pooled ++ genc ++ glob -> 256 -> {34 issue logits, 34 x 11 qty buckets, 11 hire counts}
Split by game (5% held out). Losses: unit CE (+ qty CE on PICKUP/PLACE with a count), market BCE + qty CE + hire CE.
Winning games weigh 1.0, draws 0.75, losses 0.5. Writes OUT/net.json (+ OUT/report.json) and checkpoints each epoch.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

CH, GE = 48, 32
DIL = [1, 1, 2, 4, 1]
# neighbour offsets for the unit head: self, N, S, E, W (y grows downward)
NB = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.genc = nn.Linear(D.G, GE)
        cin = D.C + GE
        self.convs = nn.ModuleList([nn.Conv2d(cin if i == 0 else CH, CH, 3, padding=d, dilation=d) for i, d in enumerate(DIL)])
        uin = 5 * CH + 2 * CH + GE + D.UF
        self.u1 = nn.Linear(uin, 128)
        self.ucls = nn.Linear(128, D.NUNIT)
        self.uq = nn.Linear(128, D.NUQ)
        min_ = 2 * CH + GE + D.G
        self.m1 = nn.Linear(min_, 256)
        self.missue = nn.Linear(256, D.NMKT)
        self.mq = nn.Linear(256, D.NMKT * D.NMQ)
        self.mhire = nn.Linear(256, 11)

    def forward(self, board, glob, units):
        n = board.shape[0]
        g = F.relu(self.genc(glob))
        x = torch.cat([board, g[:, :, None, None].expand(-1, -1, D.B, D.B)], 1)
        h1 = None
        for i, c in enumerate(self.convs):
            y = F.relu(c(x))
            if i == 1:
                h1 = y
            x = y + h1 if i == len(self.convs) - 1 else y
        pooled = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        # unit features: gather trunk at each unit's tile and its neighbours (zero off-board)
        ux = (units[:, :, 0] * 9 + 0.5).long().clamp(0, 9)
        uy = (units[:, :, 1] * 9 + 0.5).long().clamp(0, 9)
        xp = F.pad(x, (1, 1, 1, 1))  # off-board neighbours read zeros
        feats = []
        bi = torch.arange(n, device=x.device)[:, None].expand(-1, D.MAXU)
        for dx, dy in NB:
            feats.append(xp[bi, :, uy + 1 + dy, ux + 1 + dx])  # (n, MAXU, CH)
        uin = torch.cat(feats + [pooled[:, None, :].expand(-1, D.MAXU, -1), g[:, None, :].expand(-1, D.MAXU, -1), units], 2)
        uh = F.relu(self.u1(uin))
        mh = F.relu(self.m1(torch.cat([pooled, g, glob], 1)))
        return self.ucls(uh), self.uq(uh), self.missue(mh), self.mq(mh).view(n, D.NMKT, D.NMQ), self.mhire(mh)


def batch_tensors(r, dev):
    board = torch.from_numpy(r["board"].astype(np.float32) / 255.0).to(dev)
    glob = torch.from_numpy(r["glob"].astype(np.float32)).to(dev)
    units = torch.from_numpy(r["units"].astype(np.float32) / 255.0).to(dev)
    ul = torch.from_numpy(r["ulab"].astype(np.int64)).to(dev)
    ml = r["mlab"]
    n = len(r)
    issue = np.zeros((n, D.NMKT), np.float32)
    mq = np.full((n, D.NMKT), -1, np.int64)
    hire = np.zeros(n, np.int64)
    for i in range(D.MAXM):
        ids, qb = ml[:, i, 0].astype(np.int64), ml[:, i, 1].astype(np.int64)
        ok = ids != 255
        rows = np.where(ok)[0]
        issue[rows, ids[ok]] = 1.0
        hire += (ids == 0)
        sel = ok & (ids != 0) & (ids != 1)
        mq[np.where(sel)[0], ids[sel]] = np.maximum(mq[np.where(sel)[0], ids[sel]], qb[sel])
    w = np.where(r["res"] > 0.75, 1.0, np.where(r["res"] > 0.25, 0.75, 0.5)).astype(np.float32)
    return (board, glob, units, ul, torch.from_numpy(issue).to(dev), torch.from_numpy(mq).to(dev),
            torch.from_numpy(np.minimum(hire, 10)).to(dev), torch.from_numpy(w).to(dev))


def losses(net, bt):
    board, glob, units, ul, issue, mq, hire, w = bt
    uc, uqo, mi, mqo, mh = net(board, glob, units)
    cls, qb = ul[:, :, 0], ul[:, :, 1]
    valid = cls != 255
    ce = F.cross_entropy(uc.reshape(-1, D.NUNIT), cls.clamp(max=D.NUNIT - 1).reshape(-1), reduction="none").view(cls.shape)
    wu = w[:, None].expand_as(cls) * valid
    lu = (ce * wu).sum() / wu.sum().clamp(min=1)
    qv = valid & (cls >= 20) & (qb > 0)
    lq = (F.cross_entropy(uqo.reshape(-1, D.NUQ), qb.reshape(-1), reduction="none").view(cls.shape) * qv).sum() / qv.sum().clamp(min=1)
    lm = (F.binary_cross_entropy_with_logits(mi, issue, reduction="none").sum(1) * w).mean()
    mv = mq >= 0
    lmq = (F.cross_entropy(mqo.reshape(-1, D.NMQ), mq.clamp(min=0).reshape(-1), reduction="none").view(mq.shape) * mv).sum() / mv.sum().clamp(min=1)
    lh = (F.cross_entropy(mh, hire, reduction="none") * w).mean()
    with torch.no_grad():
        acc = ((uc.argmax(2) == cls) & valid).sum().float() / valid.sum().clamp(min=1)
        macc = (((mi > 0).float() == issue).all(1)).float().mean()
        nonpass = valid & (cls != 0)
        acc_np = ((uc.argmax(2) == cls) & nonpass).sum().float() / nonpass.sum().clamp(min=1)
    return lu + 0.5 * lq + lm + 0.5 * lmq + 0.5 * lh, dict(unit=lu.item(), uq=lq.item(), mkt=lm.item(), mq=lmq.item(), hire=lh.item(),
                                                           unit_acc=acc.item(), unit_acc_nonpass=acc_np.item(), mkt_exact=macc.item())


def export(net, path):
    sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    out = {"version": 1, "C": D.C, "G": D.G, "UF": D.UF, "CH": CH, "GE": GE, "dil": DIL, "nb": NB,
           "nunit": D.NUNIT, "nuq": D.NUQ, "nmkt": D.NMKT, "nmq": D.NMQ}
    for k, v in sd.items():
        out[k] = v.reshape(-1).round(6).tolist()
        out[k + ".shape"] = list(v.shape)
    json.dump(out, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts = D.open_parts(a.data)
    # index = (part, row); split by game id (held-out = game % 20 == 0)
    idx = []
    for pi, p in enumerate(parts):
        g = np.asarray(p["game"])
        idx.append(np.stack([np.full(len(g), pi), np.arange(len(g)), g % 20 == 0], 1))
    idx = np.concatenate(idx)
    tr, va = idx[idx[:, 2] == 0][:, :2], idx[idx[:, 2] == 1][:, :2]
    print(f"[tpp] {len(idx)} records ({len(tr)} train / {len(va)} held-out games' turns) on {dev}", flush=True)
    torch.manual_seed(7)
    net = Net().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, steps), pct_start=0.05)
    os.makedirs(a.out, exist_ok=True)

    def fetch(ix):
        ix = ix[np.lexsort((ix[:, 1], ix[:, 0]))]  # sorted reads from each memmap
        return np.concatenate([np.asarray(parts[pi][ix[ix[:, 0] == pi][:, 1]]) for pi in np.unique(ix[:, 0])])

    vsel = va[np.random.default_rng(0).permutation(len(va))[:40000]]

    def evaluate():
        net.eval()
        agg = {}
        with torch.no_grad():
            for b in range(0, len(vsel), 2048):
                _, m = losses(net, batch_tensors(fetch(vsel[b:b + 2048]), dev))
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
        net.train()
        return {k: float(np.mean(v)) for k, v in agg.items()}

    rep = {"records": int(len(idx)), "epochs": []}
    t0 = time.time()
    k = 0
    for ep in range(a.epochs):
        perm = np.random.default_rng(100 + ep).permutation(len(tr))
        run = {}
        for b in range(0, len(perm) - a.batch + 1, a.batch):
            loss, m = losses(net, batch_tensors(fetch(tr[perm[b:b + a.batch]]), dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            k += 1
            for kk, v in m.items():
                run.setdefault(kk, []).append(v)
            if k % 500 == 0:
                print(f"[tpp] ep {ep} step {k}/{steps} {time.time() - t0:.0f}s " + " ".join(f"{kk} {np.mean(v[-500:]):.3f}" for kk, v in run.items()), flush=True)
        ev = evaluate()
        print(f"[tpp] epoch {ep} HELD-OUT: " + " ".join(f"{kk} {v:.4f}" for kk, v in ev.items()), flush=True)
        rep["epochs"].append(ev)
        torch.save(net.state_dict(), os.path.join(a.out, f"ep{ep}.pt"))
        export(net, os.path.join(a.out, "net.json"))
        json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"[tpp] done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
