"""P3.3b -- IQL warm start on the elite macro-decision corpus.

Implicit Q-Learning (Kostrikov et al.) over one-step decision data:
  V  <- expectile regression of Q toward tau=0.7
  Q  <- reward + gamma * V(s')   (episodic day chain within a seat)
  pi <- advantage-weighted regression: log pi(a|s) * exp(beta * A)

Learns from suboptimal data properly where plain BC cannot (the 0/52
per-turn BC lesson). NEVER ships alone -- the checkpoint initializes the
PPO actor (P3.3). Held-out metrics: decision accuracy per head + advantage
calibration. Episode-level split.

Run with the GPU env:
  C:/ProgramData/anaconda3/envs/llm/python.exe src/trackp/iql.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from kaggriculture.trackp import common, macro  # noqa: E402

GAMMA = 0.98
TAU = 0.7
BETA = 3.0
HID = 256


def _mlp(sizes, out_dim):
    import torch.nn as nn
    layers = []
    d = sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def load_dataset(name="macro_dataset_elite.npz"):
    z = np.load(os.path.join(common.MODELS, name), allow_pickle=False)
    return z["F"], z["B"], z["R"], z["EP"], z["SEAT"], z["DAY"]


def train(epochs=30, dataset_name="macro_dataset_elite.npz", lr=3e-4,
          batch=1024, device=None, seed=7):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    F, B, R, EP, SEAT, DAY = load_dataset(dataset_name)
    n = len(F)
    if n < 500:
        raise SystemExit(f"dataset too small: {n}")

    # episode-level split (the discipline rule)
    eps = np.unique(EP)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    hold = set(eps[: max(1, len(eps) // 10)].tolist())
    test = np.array([e in hold for e in EP])
    tr = ~test

    # next-state chain: same (episode, seat), day+1
    key = {}
    for i in range(n):
        key[(int(EP[i]), int(SEAT[i]), int(DAY[i]))] = i
    nxt = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        j = key.get((int(EP[i]), int(SEAT[i]), int(DAY[i]) + 1), -1)
        nxt[i] = j
    # terminal reward at the last decision of the chain
    rew = np.where(nxt < 0, R, 0.0).astype(np.float32)

    Ft = torch.tensor(F, dtype=torch.float32, device=dev)
    Bt = torch.tensor(B, dtype=torch.long, device=dev)
    RWt = torch.tensor(rew, device=dev)
    NXt = torch.tensor(nxt, device=dev)

    actor = _mlp([macro.FEAT_DIM, HID, HID], macro.N_LOGITS).to(dev)
    qnet = _mlp([macro.FEAT_DIM + len(macro.HEADS), HID, HID], 1).to(dev)
    vnet = _mlp([macro.FEAT_DIM, HID, HID], 1).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=lr)

    tr_idx = np.where(tr)[0]
    b_norm = Bt.float() / torch.tensor(
        [k - 1 for _, k in macro.HEADS], dtype=torch.float32,
        device=dev)

    def q_in(idx):
        return torch.cat([Ft[idx], b_norm[idx]], dim=1)

    for ep in range(epochs):
        rng.shuffle(tr_idx)
        for s in range(0, len(tr_idx), batch):
            idx = torch.tensor(tr_idx[s:s + batch], device=dev)
            # V: expectile regression toward Q
            with torch.no_grad():
                q = qnet(q_in(idx)).squeeze(-1)
            v = vnet(Ft[idx]).squeeze(-1)
            diff = q - v
            w = torch.where(diff > 0, TAU, 1 - TAU)
            loss_v = (w * diff ** 2).mean()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()
            # Q: TD target with V(s')
            with torch.no_grad():
                nx = NXt[idx]
                vnext = torch.zeros(len(idx), device=dev)
                m = nx >= 0
                if m.any():
                    vnext[m] = vnet(Ft[nx[m]]).squeeze(-1)
                target = RWt[idx] + GAMMA * vnext
            q = qnet(q_in(idx)).squeeze(-1)
            loss_q = ((q - target) ** 2).mean()
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            # actor: advantage-weighted per-head cross-entropy
            with torch.no_grad():
                adv = (qnet(q_in(idx)).squeeze(-1)
                       - vnet(Ft[idx]).squeeze(-1))
                wa = torch.exp(BETA * adv).clamp(max=100.0)
            logits = actor(Ft[idx])
            loss_a = 0.0
            off = 0
            for hi, (_, k) in enumerate(macro.HEADS):
                lg = logits[:, off:off + k]
                ce = torch.nn.functional.cross_entropy(
                    lg, Bt[idx, hi], reduction="none")
                loss_a = loss_a + (wa * ce).mean()
                off += k
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

    # held-out decision accuracy per head
    te_idx = torch.tensor(np.where(test)[0], device=dev)
    accs = {}
    with torch.no_grad():
        logits = actor(Ft[te_idx])
        off = 0
        for hi, (name, k) in enumerate(macro.HEADS):
            pred = logits[:, off:off + k].argmax(dim=1)
            accs[name] = float((pred == Bt[te_idx, hi]).float().mean())
            off += k
        # advantage calibration: corr(A, realized return) on held-out
        adv = (qnet(q_in(te_idx)).squeeze(-1)
               - vnet(Ft[te_idx]).squeeze(-1)).cpu().numpy()
    ret_te = R[np.where(test)[0]]
    calib = float(np.corrcoef(adv, ret_te)[0, 1]) if len(adv) > 2 else 0.0

    out = {"heads_acc": {k: round(v, 4) for k, v in accs.items()},
           "mean_acc": round(float(np.mean(list(accs.values()))), 4),
           "adv_return_corr": round(calib, 4),
           "n_train": int(tr.sum()), "n_test": int(test.sum()),
           "dataset": dataset_name, "epochs": epochs}
    torch.save({"actor": actor.state_dict(), "meta": out},
               os.path.join(common.MODELS, "iql_actor.pt"))
    with open(os.path.join(common.MODELS, "iql_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset", default="macro_dataset_elite.npz")
    a = ap.parse_args()
    train(epochs=a.epochs, dataset_name=a.dataset)
