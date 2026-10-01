"""STAGE 2: retrain the policy head on the top-100 corpus and export it.

Plan: `docs/history/plan-2026-09-05.md` section 3. Runs ONLY after Stage 1
(`bc_arch_test.py`) has shown the top-100 action is learnable from state --
the kill-switch lives there, not here.

The socket is the one already shipped in v44 (`_policy_head`):

  * input  = the 49-field `turn_features.turn_vector` + 1 return-condition
             scalar (1.0 at runtime = "sell like a game you WIN")
  * net    = tanh MLP, [50, 64, 64, 9]
  * output = 9-product sell vector, denormalised by (a_mu, a_sd)
  * export = {"products", "x_mu", "x_sd", "W", "b", "a_mu", "a_sd"} ->
             json -> zlib -> base85, pasted into `_POLICY`
  * blend  = delta vs the schedule, clamped +/-_POLICY_DELTA_CAP -- the
             agent side is unchanged

What broke the previous head was CALIBRATION, not mechanics: it predicted ~0
units everywhere, so the bounded blend stripped 4 units/product/turn from the
schedule (0/56 on the hard band, -58,894 median). So this trainer refuses to
export a head whose predicted sell volume is badly off the expert's, and
prints the calibration table first.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/bc_train.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import glob
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:                                        # pragma: no cover - tty only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CORPUS = os.path.join(ROOT, "data", "bc_corpus")
OUT = os.path.join(ROOT, "models", "bc")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")
# calibration bar: mean predicted units per turn must be within this factor
# of the expert's held-out mean, in BOTH directions. The broken head was off
# by ~1000x (predicted ~0); a factor-2 window is generous to a working head.
CALIB_FACTOR = 2.0


def load_corpus():
    import numpy as np
    Xs, Ys, metas = [], [], []
    for p in sorted(glob.glob(os.path.join(CORPUS, "shard_*.npz"))):
        z = np.load(p, allow_pickle=True)
        Xs.append(z["X"])
        Ys.append(z["Y"])
        metas.extend(json.loads(str(z["meta"])))
    if not Xs:
        raise SystemExit(f"no shards in {CORPUS} -- run bc_corpus.py --build")
    return np.concatenate(Xs), np.concatenate(Ys), metas


def outcome_scores(metas):
    """Per row: 1.0 win / 0.5 tie / 0.0 loss for the (episode, seat) played.

    Joined from the route index at train time -- the shards deliberately
    carry no outcome so the corpus never needs rebuilding for label changes.
    """
    import numpy as np
    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    by = {}
    for v in idx["routes"].values():
        key = (str(v.get("episode")), int(v.get("seat", 0)))
        won = v.get("won")
        bank, opp = v.get("bank"), v.get("opp_bank")
        if won is None and bank is not None and opp is not None:
            won = bank > opp
        if bank is not None and opp is not None and bank == opp:
            by[key] = 0.5
        else:
            by[key] = 1.0 if won else 0.0
    s = np.array([by.get((str(m["episode"]), int(m["seat"])), 0.5)
                  for m in metas], dtype=np.float32)
    return s


def team_holdout(metas, frac=0.25, seed=17):
    import numpy as np
    import random
    teams = [m["team"] for m in metas]
    uniq = sorted(set(teams))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    hold = set(uniq[:max(1, int(len(uniq) * frac))])
    return np.isin(np.array(teams), list(hold)), hold


# feature indices whose TRAINING distribution (expert farms) differs from
# the RUNTIME one (our tape's farm): our farm stats, shed, seeds. The market
# and the opponent-farm block transfer cleanly (both are real ladder play).
# Critique 2026-09-05 #6: train-time dropout here forces the net to lean on
# the transferable half; whether that wins is decided on the hard band.
SHIFTED = list(range(21, 28)) + list(range(35, 49))


def train(X, Y, mask, epochs, hidden=(64, 64), lr=1e-3, bs=4096,
          drop_shifted=0.0):
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    x_mu = X[~mask].mean(0)
    x_sd = X[~mask].std(0) + 1e-6
    a_mu = Y[~mask].mean(0)
    a_sd = Y[~mask].std(0) + 1e-6

    Xn = (X - x_mu) / x_sd
    Yn = (Y - a_mu) / a_sd
    layers, prev = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, Y.shape[1])]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.tensor(Xn[~mask], device=dev)
    yt = torch.tensor(Yn[~mask], device=dev)
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            xb = xt[idx]
            if drop_shifted > 0.0:
                keep = (torch.rand(len(idx), 1, device=dev)
                        >= drop_shifted).float()
                xb = xb.clone()
                xb[:, SHIFTED] *= keep     # whole block per row, like a
                # missing private half -- not per-feature noise
            loss = nn.functional.mse_loss(net(xb), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {ep + 1}/{epochs}  train mse {tot / n:.4f}",
              flush=True)
    return net, (x_mu, x_sd, a_mu, a_sd)


def export_dict(net, norms):
    x_mu, x_sd, a_mu, a_sd = norms
    W, B = [], []
    for layer in net:
        if hasattr(layer, "weight"):
            W.append([[float(w) for w in row]
                      for row in layer.weight.detach().cpu().numpy()])
            B.append([float(b) for b in layer.bias.detach().cpu().numpy()])
    return {"products": list(PRODUCTS),
            "x_mu": [float(v) for v in x_mu], "x_sd": [float(v) for v in x_sd],
            "a_mu": [float(v) for v in a_mu], "a_sd": [float(v) for v in a_sd],
            "W": W, "b": B}


def py_forward(P, x):
    """The agent's pure-python forward, verbatim (equivalence is asserted)."""
    import math
    v = [(x[i] - P["x_mu"][i]) / P["x_sd"][i] for i in range(len(x))]
    W, B = P["W"], P["b"]
    for li in range(len(W)):
        out = []
        for o in range(len(B[li])):
            s = B[li][o]
            row = W[li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        v = [math.tanh(z) for z in out] if li < len(W) - 1 else out
    return [v[i] * P["a_sd"][i] + P["a_mu"][i] for i in range(len(v))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--drop-shifted", type=float, default=0.0,
                    help="train-time dropout on the distribution-shifted "
                         "private block (our farm/shed/seeds)")
    ap.add_argument("--no-condition", action="store_true",
                    help="fix the return-condition input to 1.0 in training "
                         "too (tests whether conditioning earns anything)")
    ap.add_argument("--peak-only", action="store_true",
                    help="train ONLY on rows where some price >= 1.5x base "
                         "-- the slice the retimer coupling consults; "
                         "cross-team skill there measured +5.7%% vs -31%% "
                         "on the full policy (2026-09-05)")
    ap.add_argument("--team", default=None,
                    help="train on ONE team's rows (single-team cloning; "
                         "cross-team transfer measured -31%%)")
    ap.add_argument("--tag", default="top100",
                    help="output name suffix: policy_<tag>.b85")
    a = ap.parse_args()
    import numpy as np
    import torch

    X, Y, metas = load_corpus()
    if a.peak_only:
        pk = (X[:, 3:12] >= 1.5).any(axis=1)
        X, Y = X[pk], Y[pk]
        metas = [m for m, k in zip(metas, pk) if k]
        print(f"peak slice: {X.shape[0]:,} rows")
    if a.team:
        import numpy as np_
        sel = [i for i, m in enumerate(metas) if m["team"] == a.team]
        if not sel:
            raise SystemExit(f"no rows for team {a.team!r}")
        X, Y = X[sel], Y[sel]
        metas = [metas[i] for i in sel]
        print(f"single-team {a.team!r}: {X.shape[0]:,} rows")
    if a.rows:
        X, Y, metas = X[:a.rows], Y[:a.rows], metas[:a.rows]
    scores = outcome_scores(metas)
    if a.no_condition:
        scores = scores * 0.0 + 1.0
    if a.team:
        # ONE team: hold out EPISODES, not teams
        import random
        keys = [(m["episode"], m["seat"]) for m in metas]
        uniq = sorted(set(keys))
        random.Random(7).shuffle(uniq)
        heps = set(uniq[:max(1, len(uniq) // 4)])
        mask = np.isin(np.arange(len(keys)),
                       [i for i, k in enumerate(keys) if k in heps])
        hold = heps
    else:
        mask, hold = team_holdout(metas)
    Xc = np.concatenate([X, scores[:, None]], axis=1)   # 49 + 1 = 50 inputs
    print(f"corpus {X.shape[0]:,} rows, {len(set(m['team'] for m in metas))} "
          f"teams ({len(hold)} held out, {mask.sum():,} rows); "
          f"win-rows {(scores == 1.0).mean():.0%}")

    net, norms = train(Xc, Y, mask, a.epochs, drop_shifted=a.drop_shifted)

    # ---- CALIBRATION FIRST (the check that would have caught the 0/56 head)
    dev = next(net.parameters()).device
    x_mu, x_sd, a_mu, a_sd = norms
    Xh = Xc[mask].copy()
    Xh[:, -1] = 1.0                       # runtime conditioning: "win"
    with torch.no_grad():
        pred = net(torch.tensor((Xh - x_mu) / x_sd, dtype=torch.float32,
                                device=dev)).cpu().numpy() * a_sd + a_mu
    pred = np.clip(pred, 0, None)
    exp_mean = Y[mask].mean(0)
    got_mean = pred.mean(0)
    print("\nCALIBRATION (held-out teams, conditioned on win):")
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

    # held-out skill, same definition as Stage 1
    mae = float(np.abs(pred - Y[mask]).mean())
    mae0 = float(np.abs(Y[~mask].mean(0) - Y[mask]).mean())
    print(f"\nheld-out MAE {mae:.4f} vs baseline {mae0:.4f} "
          f"(skill {(mae0 - mae) / mae0:.1%})")

    if not ok:
        print("\nCALIBRATION FAIL: the head's predicted sell volume is off "
              "the expert's by more than the allowed factor. NOT exporting -- "
              "this is exactly how the 0/56 head would have been caught.")
        return 1

    # ---- export + pure-python equivalence
    P = export_dict(net, norms)
    worst = 0.0
    rng = np.random.default_rng(0)
    for i in rng.choice(len(Xh), size=64, replace=False):
        with torch.no_grad():
            t = net(torch.tensor((Xh[i] - x_mu) / x_sd, dtype=torch.float32,
                                 device=dev)[None]).cpu().numpy()[0]
        t = t * a_sd + a_mu
        pyv = py_forward(P, list(Xh[i]))
        worst = max(worst, float(np.max(np.abs(np.array(pyv) - t))))
    print(f"pure-python forward equivalence: worst |diff| {worst:.3g}")
    if worst > 1e-4:
        print("EXPORT FAIL: python forward diverges from torch")
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
    json.dump({"rows": int(X.shape[0]), "skill": (mae0 - mae) / mae0,
               "mae": mae, "baseline_mae": mae0, "calibration_ok": True,
               "blob_bytes": len(blob)},
              open(os.path.join(OUT, f"train_report_{a.tag}.json"), "w"), indent=1)
    print(f"exported -> models/bc/policy_{a.tag}.b85 ({len(blob):,} bytes)")
    print("NEXT: embed into a candidate, then gates in order: "
          "policy_allowed() -> hard band CREDITED > 26/56 -> reactive "
          "no-regression -> latency. NEVER submits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
