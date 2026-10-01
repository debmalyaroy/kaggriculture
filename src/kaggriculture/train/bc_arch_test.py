"""STAGE 1: does the top-100 policy predict from state, and does it need memory?

Plan: `docs/history/plan-2026-09-05.md` section 2. This runs BEFORE any training
compute is committed and carries the project's kill-switch.

It answers two questions with one experiment:

  1. ARE THE TOP 100 LEARNABLE FROM STATE AT ALL? If no model beats the
     trivial baseline, their actions are not a function of the observation --
     they are partly open-loop, or keyed on something we cannot see -- and
     behavioural cloning is dead. We stop and spend the slots on economy
     search instead. This is the cheap kill-switch.

  2. DOES THE POLICY NEED MEMORY (RNN) OR IS THE BOARD ENOUGH?

     (a) FEEDFORWARD on the current state only
     (b) feedforward + k-step MARKET DELTAS
     (c) SEQUENCE model (GRU) over the episode

     The board carries much of its own history: a tile exposes planted_day,
     placed_day, max_lifespan_step, yield_units, pending_care_bonus,
     consecutive_unwatered/unfed, fertilized_until_day -- so the consequences
     of a day-7 decision are visible on day 12 AS STATE, and the game is
     near-Markovian for FIELD decisions.

     Three things are genuinely NOT observable and are why (b)/(c) may win on
     MARKET decisions: price/inventory TRENDS (the market gives levels, not
     rates), the opponent's SHED (private and per-seat -- we see what they
     grow, never what they hold), and our own ledger commitments. The shipped
     agent already hand-codes the first of these via `state["prev_inv"]`,
     which is evidence the need is real.

  DECISION: if (a) ~= (c), ship feedforward and drop the RNN. If (b) or (c)
  wins materially, adopt the hybrid -- feedforward over the rich state plus a
  small recurrent channel over the MARKET SERIES ONLY.

SPLIT BY TEAM, never only by episode: a model that has seen a team's other
games is not being tested on generalisation.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/bc_arch_test.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:                                        # pragma: no cover - tty only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CORPUS = os.path.join(ROOT, "data", "bc_corpus")
OUT = os.path.join(ROOT, "models", "bc")
# feature-vector layout from src/turn_features.py: 3 clock, 9 price,
# 9 market inventory, 7 us, 7 them, 9 shed, 5 seeds
MARKET_SLICE = slice(3, 21)                 # prices + market inventory


def load_shards(limit_rows=0, sample=1.0, team=None):
    """Load shards; subsample (episode, seat) GROUPS while loading so the
    full corpus never sits in RAM (15.7GB box, harness reaps on low mem)."""
    import numpy as np
    import random
    rng = random.Random(41)
    Xs, Ys, teams, eps = [], [], [], []
    for p in sorted(glob.glob(os.path.join(CORPUS, "shard_*.npz"))):
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        X_, Y_ = z["X"], z["Y"]
        # group rows by (episode, SEAT): both seats of one episode sit
        # adjacent in the shard, and a delta/GRU sequence must not run across
        # the seat junction
        if team is not None:
            sel = [i for i, m in enumerate(meta) if m["team"] == team]
            if not sel:
                continue
            X_, Y_ = X_[sel], Y_[sel]
            meta = [meta[i] for i in sel]
        keys = [(m["episode"], m["seat"]) for m in meta]
        groups, s = [], 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[s]:
                groups.append((s, i))
                s = i
        for a, b in groups:
            if sample < 1.0 and rng.random() >= sample:
                continue
            Xs.append(X_[a:b])
            Ys.append(Y_[a:b])
            teams.extend(m["team"] for m in meta[a:b])
            eps.extend(keys[a:b])
        del X_, Y_
        if limit_rows and sum(len(x) for x in Xs) >= limit_rows:
            break
    if not Xs:
        raise SystemExit(f"no shards in {CORPUS} -- run bc_corpus.py --build")
    X = np.concatenate(Xs)
    Xs.clear()
    Y = np.concatenate(Ys)
    Ys.clear()
    return X, Y, teams, eps


def team_split(teams, holdout_frac=0.25, seed=17):
    """Held-out TEAMS, not held-out rows."""
    import random
    uniq = sorted(set(teams))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = max(1, int(len(uniq) * holdout_frac))
    hold = set(uniq[:n])
    return hold


def add_deltas(X, eps, k):
    """Append k-step MARKET deltas, computed within each episode."""
    import numpy as np
    D = np.zeros((X.shape[0], MARKET_SLICE.stop - MARKET_SLICE.start),
                 dtype=np.float32)
    start = 0
    for i in range(1, len(eps) + 1):
        if i == len(eps) or eps[i] != eps[start]:
            seg = X[start:i, MARKET_SLICE]
            d = np.zeros_like(seg)
            if len(seg) > k:
                d[k:] = seg[k:] - seg[:-k]
            D[start:i] = d
            start = i
    return np.concatenate([X, D], axis=1)


def evaluate(name, Xtr, Ytr, Xte, Yte, epochs, hidden=(64, 64), lr=1e-3):
    """Train a small MLP and report held-out MAE against the trivial baseline."""
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # STANDARDIZE x and y with TRAIN stats and train in normalized space --
    # the same regime as bc_train.py. The first version trained raw-unit MSE
    # (dominated by rare big sells) and judged raw MAE, a mismatch that read
    # -9% "skill" on data the normalized trainer handles fine.
    mu_x = Xtr.mean(0); sd_x = Xtr.std(0) + 1e-6
    mu_y = Ytr.mean(0); sd_y = Ytr.std(0) + 1e-6
    layers, prev = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, Ytr.shape[1])]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.tensor((Xtr - mu_x) / sd_x, device=dev)
    yt = torch.tensor((Ytr - mu_y) / sd_y, device=dev)
    xv = torch.tensor((Xte - mu_x) / sd_x, device=dev)
    yv = torch.tensor(Yte, device=dev)                  # raw for judging
    tmu = torch.tensor(mu_y, device=dev)
    tsd = torch.tensor(sd_y, device=dev)
    feats = int(Xtr.shape[1])
    del Xtr, Ytr, Xte, Yte             # 15.7GB box: the GPU copy suffices
    n, bs = len(xt), 4096
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(xv) * tsd + tmu                      # back to raw units
        mae = (pred - yv).abs().mean().item()
        # trivial baseline: always predict the TRAINING mean sell vector
        base = tmu.unsqueeze(0).expand_as(yv)
        mae0 = (base - yv).abs().mean().item()
    return {"model": name, "features": feats, "mae": mae,
            "baseline_mae": mae0,
            "skill": (mae0 - mae) / mae0 if mae0 else 0.0}


def evaluate_gru(name, X, Y, eps, mask, epochs, hid=64, lr=1e-3, tbptt=48):
    """(c) SEQUENCE model: GRU over the MARKET series + the static state.

    The GRU consumes ONLY the market slice per turn (the one channel that is
    genuinely non-Markovian); its hidden state is concatenated with the full
    current-state vector into the same MLP head as (a). Sequences are whole
    episodes, trained with truncated BPTT windows of `tbptt` turns.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    msl = MARKET_SLICE

    # episode boundaries (rows are stored in episode order)
    bounds, start = [], 0
    for i in range(1, len(eps) + 1):
        if i == len(eps) or eps[i] != eps[start]:
            bounds.append((start, i))
            start = i
    tr = [b for b in bounds if not mask[b[0]]]
    te = [b for b in bounds if mask[b[0]]]

    nm = msl.stop - msl.start
    gru = nn.GRU(nm, hid, batch_first=True).to(dev)
    head = nn.Sequential(nn.Linear(X.shape[1] + hid, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(),
                         nn.Linear(64, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()),
                           lr=lr)
    Xt = torch.tensor(X, device=dev)
    Yt = torch.tensor(Y, device=dev)

    def run_ep(a, b, train):
        m = Xt[a:b, msl].unsqueeze(0)                 # 1 x T x market
        h, _ = gru(m)                                 # 1 x T x hid
        pred = head(torch.cat([Xt[a:b], h[0]], dim=1))
        return nn.functional.mse_loss(pred, Yt[a:b]) if train else pred

    import random
    rng = random.Random(0)
    for ep in range(epochs):
        order = tr[:]
        rng.shuffle(order)
        for a, b in order:
            # carry the hidden state across truncated-BPTT windows (detached)
            # -- a window that starts with h=0 forgets everything before it,
            # which handicaps exactly the long-range memory this arm exists
            # to measure (critique 2026-09-05 #8)
            h = None
            for s in range(a, b, tbptt):
                e = min(s + tbptt, b)
                opt.zero_grad()
                m = Xt[s:e, msl].unsqueeze(0)
                out, h2 = gru(m, h)
                pred = head(torch.cat([Xt[s:e], out[0]], dim=1))
                loss = nn.functional.mse_loss(pred, Yt[s:e])
                loss.backward()
                opt.step()
                h = h2.detach()
    with torch.no_grad():
        errs, base_errs = [], []
        mean = Yt[[i for a, b in tr for i in range(a, b)]].mean(0)
        for a, b in te:
            pred = run_ep(a, b, False)
            errs.append((pred - Yt[a:b]).abs().mean().item() * (b - a))
            base_errs.append((mean - Yt[a:b]).abs().mean().item() * (b - a))
        n = sum(b - a for a, b in te)
        mae, mae0 = sum(errs) / n, sum(base_errs) / n
    return {"model": name, "features": int(X.shape[1] + hid), "mae": mae,
            "baseline_mae": mae0,
            "skill": (mae0 - mae) / mae0 if mae0 else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=0, help="cap rows for a quick pass")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--gru", action="store_true",
                    help="include arm (c): GRU over the market series")
    ap.add_argument("--team", default=None)
    ap.add_argument("--sample", type=float, default=1.0,
                    help="fraction of (episode, seat) groups to use -- whole "
                         "groups, deterministic; this box has 15.7GB and the "
                         "harness kills everything when free RAM runs low, "
                         "so the arch test runs on a sample and Stage 2 "
                         "trains on the full corpus in a streamed loop")
    a = ap.parse_args()

    import numpy as np
    X, Y, teams, eps = load_shards(a.rows, a.sample, team=a.team)
    if a.sample < 1.0:
        print(f"sampled {a.sample:.0%} of groups -> {X.shape[0]:,} rows")
    if a.team:
        # ONE team: hold out EPISODES, not teams
        import random
        uniq = sorted(set(eps))
        random.Random(7).shuffle(uniq)
        heps = set(uniq[:max(1, len(uniq) // 4)])
        mask = np.array([e in heps for e in eps])
        print(f"single-team {a.team!r}: {X.shape[0]:,} rows, "
              f"{len(uniq)} episodes ({int(mask.sum()):,} held-out rows)\n")
    else:
        hold = team_split(teams)
        teams_arr = np.array(teams)
        mask = np.isin(teams_arr, list(hold))
        print(f"corpus {X.shape[0]:,} rows x {X.shape[1]} features, "
              f"{len(set(teams))} teams")
        print(f"held-out TEAMS: {len(hold)} ({mask.sum():,} rows)\n")

    def note(r):
        print(f"  done: {r['model']}  skill {r['skill']:.1%}", flush=True)
        return r

    results = []
    results.append(note(evaluate("(a) feedforward, state only",
                                 X[~mask], Y[~mask], X[mask], Y[mask],
                                 a.epochs)))
    import gc
    for k in (1, 4, 12):
        Xd = add_deltas(X, eps, k)
        results.append(note(evaluate(f"(b) + market deltas k={k}",
                                     Xd[~mask], Y[~mask], Xd[mask], Y[mask],
                                     a.epochs)))
        del Xd
        gc.collect()
    if a.gru:
        results.append(note(evaluate_gru("(c) GRU over market series",
                                         X, Y, eps, mask, a.epochs)))

    print(f"{'model':<34}{'feats':>7}{'held-out MAE':>14}"
          f"{'baseline':>11}{'skill':>9}")
    for r in results:
        print(f"{r['model']:<34}{r['features']:>7}{r['mae']:>14.4f}"
              f"{r['baseline_mae']:>11.4f}{r['skill']:>8.1%}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "arch_test.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    best = max(results, key=lambda r: r["skill"])
    print(f"\nbest: {best['model']} (skill {best['skill']:.1%})")
    if best["skill"] < 0.05:
        print("KILL-SWITCH: no model beats the trivial baseline by 5%. The "
              "top-100 action is not a function of the observation we capture "
              "-- behavioural cloning is dead; spend the slots on economy "
              "search instead.")
    else:
        ff = results[0]["skill"]
        if best["skill"] - ff < 0.02:
            print("VERDICT: memory buys nothing. Ship FEEDFORWARD; drop the "
                  "RNN.")
        else:
            print("VERDICT: market history helps. Adopt the hybrid -- "
                  "feedforward over state + a recurrent channel over the "
                  "MARKET SERIES only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
