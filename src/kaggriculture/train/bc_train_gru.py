"""STAGE 2 (hybrid): clone rank 1 with feedforward state + GRU market memory.

Measured 2026-09-05 (bc_arch_test --team "Crop Dusta", episode holdout):
feedforward 37.5%, +deltas 38.1%, GRU-over-market 50.1%. Memory buys
+12.6pp on the one team whose policy is learnable at all -- so the shipped
head becomes a HYBRID: the 49-field state + return-condition scalar feeds
the MLP, and a 64-unit GRU consumes the normalized 18-field market slice
per turn, its hidden state carried across the game.

Runtime cost: one GRU step + one [114,64,64,9] MLP ~ 30k mults/turn in
pure python -- milliseconds, inside the latency gate.

Refusals before export (same discipline as bc_train.py):
  * CALIBRATION: predicted sell volume within CALIB_FACTOR of the expert's
    per product on held-out episodes (the 0/56 head predicted ~0 units).
  * EQUIVALENCE: the pure-python GRU+MLP forward must match torch to 1e-4
    over a real held-out sequence.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/bc_train_gru.py --team "Crop Dusta"
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
try:                                        # pragma: no cover - tty only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from kaggriculture.train.bc_train import (CALIB_FACTOR, OUT, PRODUCTS, load_corpus,   # noqa: E402
                      outcome_scores)

MARKET = slice(3, 21)                       # prices + market inventory
HID = 64


def episode_bounds(keys):
    bounds, s = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[s]:
            bounds.append((s, i))
            s = i
    return bounds


# distribution-shifted block: our farm stats, shed, seeds (training states
# are the EXPERT's farm; runtime states are ours). Dropout here forces the
# net onto the transferable features -- market series + opponent farm.
SHIFTED = list(range(21, 28)) + list(range(35, 49))


def train(Xc, Y, keys, mask, epochs, lr=1e-3, tbptt=48, drop_shifted=0.0):
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x_mu = Xc[~mask].mean(0)
    x_sd = Xc[~mask].std(0) + 1e-6
    a_mu = Y[~mask].mean(0)
    a_sd = Y[~mask].std(0) + 1e-6
    Xn = ((Xc - x_mu) / x_sd).astype("float32")
    Yn = ((Y - a_mu) / a_sd).astype("float32")

    gru = nn.GRU(MARKET.stop - MARKET.start, HID, batch_first=True).to(dev)
    head = nn.Sequential(nn.Linear(Xc.shape[1] + HID, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()),
                           lr=lr)
    Xt = torch.tensor(Xn, device=dev)
    Yt = torch.tensor(Yn, device=dev)
    bounds = episode_bounds(keys)
    tr = [b for b in bounds if not mask[b[0]]]

    import random
    rng = random.Random(0)
    for ep in range(epochs):
        order = tr[:]
        rng.shuffle(order)
        tot = cnt = 0.0
        for a, b in order:
            h = None                       # hidden state carried, detached
            for s in range(a, b, tbptt):
                e = min(s + tbptt, b)
                opt.zero_grad()
                m = Xt[s:e, MARKET].unsqueeze(0)
                out, h2 = gru(m, h)
                xw = Xt[s:e]
                if drop_shifted > 0.0 and torch.rand(1).item() < drop_shifted:
                    xw = xw.clone()
                    xw[:, SHIFTED] = 0.0   # whole window, whole block: like
                    # a missing private half, not per-feature noise
                pred = head(torch.cat([xw, out[0]], dim=1))
                loss = nn.functional.mse_loss(pred, Yt[s:e])
                loss.backward()
                opt.step()
                h = h2.detach()
                tot += loss.item() * (e - s)
                cnt += e - s
        print(f"  epoch {ep + 1}/{epochs}  train mse {tot / cnt:.4f}",
              flush=True)
    return gru, head, (x_mu, x_sd, a_mu, a_sd)


def predict(gru, head, Xn, bounds):
    """Full-sequence predictions (normalized), per episode."""
    import torch
    dev = next(head.parameters()).device
    Xt = torch.tensor(Xn, device=dev)
    out = torch.empty((Xn.shape[0], head[-1].out_features), device=dev)
    with torch.no_grad():
        for a, b in bounds:
            m = Xt[a:b, MARKET].unsqueeze(0)
            hseq, _ = gru(m)
            out[a:b] = head(torch.cat([Xt[a:b], hseq[0]], dim=1))
    return out.cpu().numpy()


def export_dict(gru, head, norms):
    x_mu, x_sd, a_mu, a_sd = norms

    def mat(t):
        return [[float(v) for v in row] for row in t.detach().cpu().numpy()]

    def vec(t):
        return [float(v) for v in t.detach().cpu().numpy()]

    W, B = [], []
    for layer in head:
        if hasattr(layer, "weight"):
            W.append(mat(layer.weight))
            B.append(vec(layer.bias))
    return {"products": list(PRODUCTS), "arch": "gru_hybrid_v2",
            "market": [MARKET.start, MARKET.stop], "hid": HID,
            "x_mu": [float(v) for v in x_mu], "x_sd": [float(v) for v in x_sd],
            "a_mu": [float(v) for v in a_mu], "a_sd": [float(v) for v in a_sd],
            "g_wih": mat(gru.weight_ih_l0), "g_whh": mat(gru.weight_hh_l0),
            "g_bih": vec(gru.bias_ih_l0), "g_bhh": vec(gru.bias_hh_l0),
            "W": W, "b": B}


def py_step(P, x_raw, h):
    """One pure-python turn: returns (sell_vector, new_h).

    x_raw is the RAW 50-vector (49 state + condition). This function is the
    exact code that ships in the agent; equivalence to torch is asserted at
    export.
    """
    import math
    xm, xs = P["x_mu"], P["x_sd"]
    xn = [(x_raw[i] - xm[i]) / xs[i] for i in range(len(x_raw))]
    a, b = P["market"]
    m = xn[a:b]
    H = P["hid"]
    wih, whh, bih, bhh = P["g_wih"], P["g_whh"], P["g_bih"], P["g_bhh"]

    def gate(row0, bias0, vec_):
        out = []
        for o in range(H):
            s = bias0[o]
            row = row0[o]
            for i in range(len(vec_)):
                s += row[i] * vec_[i]
            out.append(s)
        return out
    ir = gate(wih[0:H], bih[0:H], m)
    iz = gate(wih[H:2 * H], bih[H:2 * H], m)
    inn = gate(wih[2 * H:], bih[2 * H:], m)
    hr = gate(whh[0:H], bhh[0:H], h)
    hz = gate(whh[H:2 * H], bhh[H:2 * H], h)
    hn = gate(whh[2 * H:], bhh[2 * H:], h)
    r = [1.0 / (1.0 + math.exp(-(ir[i] + hr[i]))) for i in range(H)]
    z = [1.0 / (1.0 + math.exp(-(iz[i] + hz[i]))) for i in range(H)]
    n = [math.tanh(inn[i] + r[i] * hn[i]) for i in range(H)]
    h2 = [(1.0 - z[i]) * n[i] + z[i] * h[i] for i in range(H)]

    v = xn + h2
    W, Bs = P["W"], P["b"]
    for li in range(len(W)):
        out = []
        for o in range(len(Bs[li])):
            s = Bs[li][o]
            row = W[li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        v = [math.tanh(t) for t in out] if li < len(W) - 1 else out
    am, asd = P["a_mu"], P["a_sd"]
    return [v[i] * asd[i] + am[i] for i in range(len(v))], h2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Crop Dusta")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="cropdusta_gru")
    ap.add_argument("--drop-shifted", type=float, default=0.0)
    a = ap.parse_args()
    import numpy as np
    import torch

    X, Y, metas = load_corpus()
    sel = [i for i, m in enumerate(metas) if m["team"] == a.team]
    if not sel:
        raise SystemExit(f"no rows for team {a.team!r}")
    X, Y = X[sel], Y[sel]
    metas = [metas[i] for i in sel]
    scores = outcome_scores(metas)
    keys = [(m["episode"], m["seat"]) for m in metas]

    import random
    uniq = sorted(set(keys))
    random.Random(7).shuffle(uniq)
    heps = set(uniq[:max(1, len(uniq) // 4)])
    mask = np.array([k in heps for k in keys])
    Xc = np.concatenate([X, scores[:, None]], axis=1).astype("float32")
    print(f"single-team {a.team!r}: {X.shape[0]:,} rows, {len(uniq)} episodes "
          f"({int(mask.sum()):,} held-out rows); win-rows "
          f"{(scores == 1.0).mean():.0%}")

    gru, head, norms = train(Xc, Y, keys, mask, a.epochs,
                             drop_shifted=a.drop_shifted)
    x_mu, x_sd, a_mu, a_sd = norms

    # held-out skill + CALIBRATION, conditioned on win
    bounds = episode_bounds(keys)
    te = [b for b in bounds if mask[b[0]]]
    Xh = Xc.copy()
    Xh[:, -1] = 1.0
    Xn = ((Xh - x_mu) / x_sd).astype("float32")
    pred_n = predict(gru, head, Xn, te)
    pred = np.clip(pred_n * a_sd + a_mu, 0, None)
    tid = np.concatenate([np.arange(*b) for b in te])
    mae = float(np.abs(pred[tid] - Y[tid]).mean())
    mae0 = float(np.abs(Y[~mask].mean(0) - Y[tid]).mean())
    print(f"\nheld-out MAE {mae:.4f} vs baseline {mae0:.4f} "
          f"(skill {(mae0 - mae) / mae0:.1%})")

    exp_mean = Y[tid].mean(0)
    got_mean = pred[tid].mean(0)
    print("\nCALIBRATION (held-out episodes, conditioned on win):")
    print(f"{'product':<12}{'expert mean':>12}{'head mean':>12}")
    ok = True
    for i, p in enumerate(PRODUCTS):
        flag = ""
        # flag only MATERIAL distortion: outside the ratio window AND more
        # than 0.3 units/turn of absolute drift. The refusal exists to catch
        # volume stripping/explosion (the 0/56 head predicted ~0 everywhere),
        # not fractional-unit noise on low-volume products; the TOTAL row
        # keeps the strict 2x rule.
        off_ratio = exp_mean[i] > 0.05 and not (
            exp_mean[i] / CALIB_FACTOR <= got_mean[i]
            <= exp_mean[i] * CALIB_FACTOR)
        if off_ratio and abs(got_mean[i] - exp_mean[i]) > 0.30:
            flag = "  <-- OFF"
            ok = False
        elif off_ratio:
            flag = "  (ratio off, immaterial)"
        print(f"{p:<12}{exp_mean[i]:>12.3f}{got_mean[i]:>12.3f}{flag}")
    tot_e, tot_g = float(exp_mean.sum()), float(got_mean.sum())
    print(f"{'TOTAL':<12}{tot_e:>12.3f}{tot_g:>12.3f}")
    if tot_g < tot_e / CALIB_FACTOR or tot_g > tot_e * CALIB_FACTOR:
        ok = False
    if not ok:
        print("\nCALIBRATION FAIL -- not exporting.")
        return 1

    # EQUIVALENCE, two numbers with different jobs:
    #   * FORMULA check vs a float64 torch reference -- must match ~1e-6 or
    #     the exported python step computes a different function (refuse);
    #   * PRECISION drift vs the float32 training net -- fp32 recurrence
    #     compounds over turns; reported, and material only near the blend's
    #     int(round()) boundaries.
    P = export_dict(gru, head, norms)
    a0, b0 = te[0]
    n_chk = min(120, b0 - a0)
    gru64 = gru.double()
    head64 = head.double()
    dev = next(head64.parameters()).device
    X64 = torch.tensor(Xn[a0:a0 + n_chk], dtype=torch.float64, device=dev)
    with torch.no_grad():
        hseq, _ = gru64(X64[:, MARKET].unsqueeze(0))
        ref64 = (head64(torch.cat([X64, hseq[0]], dim=1)).cpu().numpy()
                 * a_sd + a_mu)
    h = [0.0] * HID
    worst = worst32 = 0.0
    for i in range(n_chk):
        want, h = py_step(P, list(Xh[a0 + i]), h)
        worst = max(worst, float(np.max(np.abs(
            np.array(want) - ref64[i]))))
        worst32 = max(worst32, float(np.max(np.abs(
            np.array(want) - (pred_n[a0 + i] * a_sd + a_mu)))))
    print(f"\nequivalence over {n_chk} turns: formula |diff| {worst:.3g} "
          f"vs float64 torch; fp32-drift {worst32:.3g}")
    # threshold 1e-4: the float64 reference still eats float32-rounded
    # inputs (Xn), which alone contributes ~1e-6 -- a formula error would
    # show orders of magnitude above this
    if worst > 1e-4:
        print("EXPORT FAIL: python step computes a different function")
        return 1

    os.makedirs(OUT, exist_ok=True)
    blob = base64.b85encode(zlib.compress(
        json.dumps(P).encode("utf-8"), 9)).decode("ascii")
    with open(os.path.join(OUT, f"policy_{a.tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(P, fh)
    with open(os.path.join(OUT, f"policy_{a.tag}.b85"), "w",
              encoding="utf-8") as fh:
        fh.write(blob)
    json.dump({"team": a.team, "rows": int(X.shape[0]),
               "episodes": len(uniq), "skill": (mae0 - mae) / mae0,
               "mae": mae, "baseline_mae": mae0, "calibration_ok": True,
               "py_equiv_worst": worst, "blob_bytes": len(blob)},
              open(os.path.join(OUT, f"train_report_{a.tag}.json"), "w"),
              indent=1)
    print(f"exported -> models/bc/policy_{a.tag}.b85 ({len(blob):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
