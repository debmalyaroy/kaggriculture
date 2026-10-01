"""GRU challenger for the identifier, under the EXACT bake-off protocol.

Input: models/lab/seq_dataset.npz -- per route the 6 cumulative
prefix_features snapshots (identical to what the shipped identifier sees).
Protocol parity with identifier_lab.py (2026-08-12 bake-off):
  - 3 walk-forward folds (train days < d, test day == d);
  - dropout augmentation x3 (p = 0, .1, .2) applied to BOTH train and test
    rows, mimicking floored-sale invisibility -- without it the GRU's
    numbers are not comparable to the flat models'.
Metrics per fold at the final step + accuracy at every checkpoint.

Runs in the llm conda env (torch + CUDA); single process, brief GPU bursts
(the box BSODs under sustained multi-core CPU load -- see
docs/history/issues-and-improvements.md 2026-08-13).

    C:/ProgramData/anaconda3/envs/llm/python.exe src/experiments/gru_lab.py
"""
from kaggriculture.paths import ROOT
import json
import os

import numpy as np
import torch
import torch.nn as nn

DATA = os.path.join(ROOT, "models", "lab", "seq_dataset.npz")
OUT = os.path.join(ROOT, "models", "lab", "gru_report.json")

HIDDEN = 32
EPOCHS = 25
BATCH = 256
STEPS = (144, 192, 240, 288, 360, 480)


class SeqID(nn.Module):
    def __init__(self, d_in, k):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.gru = nn.GRU(d_in, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, k)

    def forward(self, x):
        h, _ = self.gru(self.norm(x))
        return self.head(h)                    # (B, T, k) per-step logits


def macro_f1(y_true, y_pred, k):
    f1s = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def augment(X, y, rates=(0.0, 0.1, 0.2), seed=11):
    rng = np.random.default_rng(seed)
    outs, ys = [], []
    for p in rates:
        outs.append(X if p == 0.0 else X * (rng.random(X.shape) >= p))
        ys.append(y)
    return np.concatenate(outs).astype(np.float32), np.concatenate(ys)


def run_fold(Xtr, ytr, Xte, yte, k, dev, tag):
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    xtr = torch.tensor((Xtr - mu) / sd, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    xte = torch.tensor((Xte - mu) / sd, device=dev)

    counts = np.bincount(ytr, minlength=k).astype(np.float64)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1.0)) ** 0.5,
                     dtype=torch.float32, device=dev)
    model = SeqID(Xtr.shape[-1], k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    n = len(xtr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            logits = model(xtr[b])
            loss = lossf(
                logits.reshape(-1, k),
                ytr_t[b].unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    out = {"per_step": {}}
    with torch.no_grad():
        proba = torch.softmax(model(xte), dim=-1).cpu().numpy()
    for j, t in enumerate(STEPS):
        p = proba[:, j]
        pred = p.argmax(1)
        acc = float((pred == yte).mean())
        f1 = macro_f1(yte, pred, k)
        ll = float(-np.log(np.clip(p[np.arange(len(yte)), yte],
                                   1e-12, None)).mean())
        out["per_step"][str(t)] = {"acc": round(acc, 4),
                                   "macro_f1": round(f1, 4),
                                   "log_loss": round(ll, 4)}
    last = out["per_step"][str(STEPS[-1])]
    print(f"  {tag}  t480: acc {last['acc']:.3f}  "
          f"macroF1 {last['macro_f1']:.3f}  ll {last['log_loss']:.3f}",
          flush=True)
    return out


def main():
    z = np.load(DATA, allow_pickle=False)
    X, y, dates, k = z["X"], z["y"], z["dates"], int(z["classes"])
    days = sorted(set(d for d in dates.tolist() if d))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{X.shape} k={k} folds={days[-3:]} dev={dev}")
    report = {"hidden": HIDDEN, "protocol": "bake-off parity: 3 folds, "
              "dropout aug x3 train+test", "folds": {}}
    for test_day in days[-3:]:
        te = dates == test_day
        tr = (dates < test_day) & (dates != "")
        if tr.sum() < 200 or te.sum() < 50:
            print(f"  skip {test_day}: train {int(tr.sum())} "
                  f"test {int(te.sum())}")
            continue
        Xtr, ytr = augment(X[tr], y[tr])
        Xte, yte = augment(X[te], y[te])
        report["folds"][str(test_day)] = run_fold(
            Xtr, ytr, Xte, yte, k, dev, str(test_day))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
