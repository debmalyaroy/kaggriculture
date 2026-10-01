"""P2 -- the route generator: return- + shop-conditioned macro decoder.

Decision-Transformer family, macro-tokenized: a small causal GRU decoder
predicts each day's macro bins given (target return bin, the shop-draw
partition observed SO FAR, day index, previous days' bins). Trained on the
full macro corpus, fine-tuned on elite wins, so sampling conditioned on
"how the elite win" is a one-flag choice.

The decoder emits MACRO PLANS, not raw actions: the planner executes a
sampled plan (deterministic argmax L0) into a legal tape -- generation can
never produce an illegal route, which replaces the legality/repair stage of
the original spec by construction.

Run with the GPU env:
  .../envs/llm/python.exe src/trackp/generator.py --epochs 12
  .../envs/llm/python.exe src/trackp/generator.py --sample 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from kaggriculture.trackp import common, macro  # noqa: E402

N_DAYS = 30
RET_BINS = 8       # return conditioning bins
REGIMES = ["wool", "carrot", "dairy", "neutral", "none"]
D_MODEL = 128
CKPT = os.path.join(common.MODELS, "generator.pt")


def _regime_series(F: np.ndarray) -> np.ndarray:
    """Per-decision regime id from the macro feature drain block.

    Features 7..15 are drain/15 per product (WHEAT..FERTILIZER). The regime
    is the dominant non-neutral drain signal, 'none' before the first shop.
    """
    drains = F[:, 7:16]
    wool = drains[:, 7]
    carrot = drains[:, 1]
    dairy = drains[:, 6] + drains[:, 3]
    out = np.full(len(F), REGIMES.index("none"), dtype=np.int64)
    any_shop = drains.sum(axis=1) > 0.01
    dom = np.stack([wool, carrot, dairy], axis=1)
    which = dom.argmax(axis=1)
    strong = dom.max(axis=1) > 0.05
    out[any_shop & strong & (which == 0)] = REGIMES.index("wool")
    out[any_shop & strong & (which == 1)] = REGIMES.index("carrot")
    out[any_shop & strong & (which == 2)] = REGIMES.index("dairy")
    out[any_shop & ~strong] = REGIMES.index("neutral")
    return out


def _episode_sequences(name: str):
    z = np.load(os.path.join(common.MODELS, name), allow_pickle=False)
    F, B, R, EP, SEAT, DAY = (z["F"], z["B"], z["R"], z["EP"], z["SEAT"],
                              z["DAY"])
    reg = _regime_series(F)
    seqs = {}
    for i in range(len(F)):
        seqs.setdefault((int(EP[i]), int(SEAT[i])), []).append(
            (int(DAY[i]), B[i], int(reg[i]), float(R[i])))
    out = []
    for key, rows in seqs.items():
        rows.sort()
        if len(rows) < 10:
            continue
        days = np.array([r[0] for r in rows])
        bins = np.stack([r[1] for r in rows])
        regs = np.array([r[2] for r in rows])
        ret = rows[0][3]
        out.append((days, bins, regs, ret))
    return out


class _Model:
    def build(self):
        import torch.nn as nn

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.day_emb = nn.Embedding(N_DAYS + 1, D_MODEL)
                self.reg_emb = nn.Embedding(len(REGIMES), D_MODEL)
                self.ret_emb = nn.Embedding(RET_BINS, D_MODEL)
                self.bin_in = nn.Linear(len(macro.HEADS), D_MODEL)
                self.gru = nn.GRU(D_MODEL, D_MODEL, num_layers=2,
                                  batch_first=True)
                self.head = nn.Linear(D_MODEL, macro.N_LOGITS)

            def forward(self, days, prev_bins, regs, ret_bin):
                x = (self.day_emb(days) + self.reg_emb(regs)
                     + self.ret_emb(ret_bin)[:, None, :]
                     + self.bin_in(prev_bins))
                h, _ = self.gru(x)
                return self.head(h)
        return Decoder()


def train(epochs=12, elite_epochs=6, lr=1e-3, seed=7):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _Model().build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run_phase(name, n_epochs, ret_lo):
        seqs = _episode_sequences(name)
        rets = np.array([s[3] for s in seqs])
        qs = np.quantile(rets, np.linspace(0, 1, RET_BINS + 1)[1:-1])
        losses = []
        rng = np.random.default_rng(seed)
        idx = np.arange(len(seqs))
        for ep in range(n_epochs):
            rng.shuffle(idx)
            tot = cnt = 0.0
            for s0 in range(0, len(idx), 64):
                batch = [seqs[i] for i in idx[s0:s0 + 64]]
                L = max(len(b[0]) for b in batch)
                days = torch.zeros(len(batch), L, dtype=torch.long)
                bins = torch.zeros(len(batch), L, len(macro.HEADS),
                                   dtype=torch.long)
                regs = torch.zeros(len(batch), L, dtype=torch.long)
                mask = torch.zeros(len(batch), L, dtype=torch.bool)
                retb = torch.zeros(len(batch), dtype=torch.long)
                for bi, (d, b, r, ret) in enumerate(batch):
                    n = len(d)
                    days[bi, :n] = torch.tensor(d)
                    bins[bi, :n] = torch.tensor(b, dtype=torch.long)
                    regs[bi, :n] = torch.tensor(r)
                    mask[bi, :n] = True
                    retb[bi] = int(np.searchsorted(qs, ret))
                days, bins, regs = days.to(dev), bins.to(dev), regs.to(dev)
                mask, retb = mask.to(dev), retb.to(dev)
                prev = torch.zeros_like(bins)
                prev[:, 1:] = bins[:, :-1]
                norm = torch.tensor([k - 1 for _, k in macro.HEADS],
                                    dtype=torch.float32, device=dev)
                logits = model(days, prev.float() / norm, regs, retb)
                loss = 0.0
                off = 0
                for hi, (_, k) in enumerate(macro.HEADS):
                    lg = logits[:, :, off:off + k][mask]
                    tg = bins[:, :, hi][mask]
                    loss = loss + torch.nn.functional.cross_entropy(lg, tg)
                    off += k
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                cnt += 1
            losses.append(tot / max(1, cnt))
            print(f"{name} epoch {ep}: loss {losses[-1]:.3f}", flush=True)
        return losses[-1] if losses else None, qs

    base_loss, qs = run_phase("macro_dataset.npz", epochs, None)
    elite_loss, _ = run_phase("macro_dataset_elite.npz", elite_epochs, None)
    import torch as _t
    _t.save({"model": model.state_dict(), "ret_quantiles": qs.tolist(),
             "meta": {"base_loss": base_loss, "elite_loss": elite_loss}},
            CKPT)
    rep = {"base_loss": round(base_loss, 3),
           "elite_loss": round(elite_loss, 3), "ckpt": CKPT}
    with open(os.path.join(common.MODELS, "generator_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep))
    return rep


def sample(n=64, ret_bin=RET_BINS - 1, temp=0.9, seed=11,
           out_name="generated_plans.json"):
    """Sample n macro plans conditioned on the TOP return bin (elite mode).

    The regime series is sampled from the empirical unlock calendar (shops
    at days 3,6,9...); each plan records the regime path it assumed, so the
    funnel can match plans to seeds with that draw.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(CKPT, map_location=dev, weights_only=False)
    model = _Model().build().to(dev)
    model.load_state_dict(blob["model"])
    model.eval()
    rng = np.random.default_rng(seed)
    plans = []
    with torch.no_grad():
        for pi in range(n):
            # a regime path: none until day 3, then a sampled partition that
            # can refine at day 6 (the P1.4 branch points)
            main = rng.choice(["wool", "carrot", "dairy", "neutral"])
            second = rng.choice(["wool", "carrot", "dairy", "neutral"])
            reg_path = []
            for d in range(N_DAYS):
                reg_path.append("none" if d < 3 else
                                (main if d < 6 else second
                                 if rng.random() < 0.3 else main))
            bins_seq = []
            prev = torch.zeros(1, 1, len(macro.HEADS), device=dev)
            norm = torch.tensor([k - 1 for _, k in macro.HEADS],
                                dtype=torch.float32, device=dev)
            for d in range(N_DAYS):
                days = torch.tensor([[d]], device=dev)
                regs = torch.tensor([[REGIMES.index(reg_path[d])]],
                                    device=dev)
                retb = torch.tensor([ret_bin], device=dev)
                logits = model(days, prev / norm, regs, retb)[0, 0]
                row = []
                off = 0
                for _, k in macro.HEADS:
                    p = torch.softmax(logits[off:off + k] / temp, dim=0)
                    b = int(torch.multinomial(p, 1))
                    row.append(b)
                    off += k
                bins_seq.append(row)
                prev = torch.tensor([[row]], dtype=torch.float32,
                                    device=dev)
            plans.append({"regime_path": reg_path, "bins": bins_seq,
                          "ret_bin": int(ret_bin)})
    out = os.path.join(common.MODELS, out_name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"heads": [[n_, k] for n_, k in macro.HEADS],
                   "plans": plans}, fh)
    print(json.dumps({"sampled": len(plans), "out": out}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    if a.epochs:
        train(epochs=a.epochs)
    if a.sample:
        sample(n=a.sample)
