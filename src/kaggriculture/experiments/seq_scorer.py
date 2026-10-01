"""Sequence rating scorer: GRU encoder + rating head (plan item 11, v2).

The flat scorer ceilinged at Spearman 0.288 (route signatures cannot rank
elite play). This one reads the same 6-checkpoint sequences the GRU
identifier uses and regresses the LADDER RATING of the team that played the
route. Walk-forward validated against the flat baseline. Never a gate --
ranks candidates and powers model-based collapse detection.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/experiments/seq_scorer.py
"""
from kaggriculture.paths import ROOT
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

DATA = os.path.join(ROOT, "models", "lab", "seq_dataset.npz")
OUT = os.path.join(ROOT, "models", "lab", "seq_scorer.json")
HIDDEN = 32


class SeqRater(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.gru = nn.GRU(d_in, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h, _ = self.gru(self.norm(x))
        return self.head(h[:, -1]).squeeze(-1)


def board_ratings():
    import kaggriculture.data.episodes as E
    rows = E.leaderboard_rows(verbose=False) or []
    return {r["team"]: float(r["score"]) for r in rows
            if r.get("team") and r.get("score")}


def main():
    z = np.load(DATA, allow_pickle=False)
    X, dates, teams = z["X"], z["dates"], z["teams"]
    ratings = board_ratings()
    keep = np.array([t in ratings and t != "?" for t in teams.tolist()])
    X, dates = X[keep], dates[keep]
    y = np.array([ratings[t] for t in teams[keep].tolist()],
                 dtype=np.float32)
    print(f"{len(y)} rated sequences "
          f"(of {len(teams)}; ratings {y.min():.0f}..{y.max():.0f})")

    days = sorted(set(d for d in dates.tolist() if d))
    test_day = None
    for d in reversed(days):
        m = dates == d
        if m.sum() >= 150 and float(np.ptp(y[m])) >= 300:
            test_day = d
            break
    test_day = test_day or days[-1]
    te = dates == test_day
    tr = ~te
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"test {test_day}: {int(tr.sum())} train / {int(te.sum())} test, "
          f"dev={dev}")

    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    xtr = torch.tensor((X[tr] - mu) / sd, device=dev)
    ytr = torch.tensor((y[tr] - y[tr].mean()) / y[tr].std(), device=dev)
    xte = torch.tensor((X[te] - mu) / sd, device=dev)

    model = SeqRater(X.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(xtr)
    for ep in range(40):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = nn.functional.smooth_l1_loss(model(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = (model(xte).cpu().numpy() * y[tr].std()) + y[tr].mean()

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        return float((ra * rb).sum()
                     / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

    rho = _spearman(pred, y[te])
    mae = float(np.abs(pred - y[te]).mean())
    hi = y[te] >= np.percentile(y[te], 80)
    lo = y[te] <= np.percentile(y[te], 40)
    gap = float(pred[hi].mean() - pred[lo].mean())
    agap = float(y[te][hi].mean() - y[te][lo].mean())
    print(f"held-out day: Spearman {rho:.3f} (flat baseline 0.288), "
          f"MAE {mae:,.0f}")
    print(f"top-vs-mid predicted gap {gap:+,.0f} (actual {agap:+,.0f})")
    json.dump({"spearman": rho, "mae": mae, "gap_pred": gap,
               "gap_actual": agap, "n": int(len(y)),
               "test_day": str(test_day), "flat_baseline": 0.288},
              open(OUT, "w", encoding="utf-8"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
