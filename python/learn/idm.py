"""Inverse model: which of our profiles does a top player's day look like? (option A, queue Q23)

    python python/learn/idm.py [--league data/leagues/rl3] [--threads 16] [--epochs 6]

Our random-profile league (league-rand on the rl3 table) records the profile the learner played each
day. From the learner's own dayobs on day d and d+1 (what it held, sold, earned; only its own view,
the same builder as the corpus) an MLP learns P(profile on day d | behaviour). Applied to the corpus
seats rated 2700+ it gives each top player's day a distribution over our 36 profiles (soft labels:
profiles that act identically on that day share the mass), which a BC teacher can imitate.

Honesty checks in the report (data/idm/report.json):
  - held-out league accuracy (top-1 / top-3) and mean max-prob: how identifiable a profile is at all;
  - the same confidence on top players and on < 2300 players: if top players' days fit no profile
    (confidence far below the league's), their play is outside our action space and imitation is closed;
  - out-of-distribution: mean |z| of the top players' inputs against the league's.
Outputs data/train/top_idm.npz: corpus row, days, probs [n, D, 36] float16.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(RL, "data", "train")
OUT = os.path.join(RL, "data", "idm")
DAYS = [6, 10] + list(range(12, 29))  # policy days whose next day exists
N_ACT = 36


def feats(obs, days):
    """[G, 30, F] -> [G, len(days), 2F]: day d and the change to d+1."""
    a = obs[:, days].astype(np.float32)
    b = obs[:, [d + 1 for d in days]].astype(np.float32)
    return np.concatenate([a, b - a], axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=os.path.join(RL, "data", "leagues", "rl3"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--top-band", type=int, default=4, help="corpus own_band: 4 = 2700+")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    cache.build_league(a.league, prefix="league_rl3")
    lo = np.load(os.path.join(TRAIN, "league_rl3_obs.npy"), mmap_mode="r")
    lm = dict(np.load(os.path.join(TRAIN, "league_rl3_meta.npz")))
    X = feats(lo, DAYS)                                   # [G, D, 2F]
    Y = lm["sched"][:, DAYS].astype(np.int64)             # [G, D]
    live = np.abs(lo[:, [d + 1 for d in DAYS]].astype(np.float32)).sum(2) > 0  # game reached d+1
    hold = lm["seed"] % 10 == 0
    mu = X[~hold][live[~hold]].mean(0)
    sd = X[~hold][live[~hold]].std(0) + 1e-3
    dayoh = np.eye(len(DAYS), dtype=np.float32)

    def inp(x):  # [n, D, 2F] -> [n*D, 2F + D]
        n = x.shape[0]
        z = ((x - mu) / sd).clip(-8, 8)
        return np.concatenate([z, np.broadcast_to(dayoh, (n, len(DAYS), len(DAYS)))], 2).reshape(n * len(DAYS), -1)

    xtr, ytr = inp(X[~hold])[live[~hold].reshape(-1)], Y[~hold].reshape(-1)[live[~hold].reshape(-1)]
    xte, yte = inp(X[hold])[live[hold].reshape(-1)], Y[hold].reshape(-1)[live[hold].reshape(-1)]
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, N_ACT))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.tensor(xtr), torch.tensor(ytr)
    for ep in range(a.epochs):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 4096):
            j = perm[i:i + 4096]
            loss = torch.nn.functional.cross_entropy(net(xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        print(f"[idm] epoch {ep}: train ce {tot / len(xt):.3f}", flush=True)

    def probs(x):
        with torch.no_grad():
            return torch.cat([torch.softmax(net(torch.tensor(x[i:i + 65536])), 1) for i in range(0, len(x), 65536)]).numpy()

    pte = probs(xte)
    top1 = float((pte.argmax(1) == yte).mean())
    top3 = float((np.argsort(-pte, 1)[:, :3] == yte[:, None]).any(1).mean())
    # the true profile's probability vs chance: how much the behaviour identifies it
    p_true = float(pte[np.arange(len(yte)), yte].mean())
    rep = {"league_games": int(len(lo)), "samples_train": int(len(xtr)), "samples_heldout": int(len(xte)),
           "heldout_top1": round(top1, 3), "heldout_top3": round(top3, 3), "heldout_p_true": round(p_true, 3),
           "chance": round(1 / N_ACT, 3), "heldout_maxprob": round(float(pte.max(1).mean()), 3),
           "per_day_top1": {int(d): round(float((pte.argmax(1) == yte)[(np.tile(np.arange(len(DAYS)), len(X[hold]))[live[hold].reshape(-1)]) == k].mean()), 3)
                            for k, d in enumerate(DAYS)}}
    print(f"[idm] held-out league: top-1 {top1:.3f}, top-3 {top3:.3f}, p(true) {p_true:.3f} (chance {1 / N_ACT:.3f})", flush=True)

    co = np.load(os.path.join(TRAIN, "corpus_obs.npy"), mmap_mode="r")
    cm = dict(np.load(os.path.join(TRAIN, "corpus_meta.npz")))
    groups = {"top": np.where((cm["own_band"] == a.top_band) & (cm["days"] >= 30))[0],
              "low": np.where((cm["own_band"] >= 0) & (cm["own_band"] <= 1) & (cm["days"] >= 30))[0]}
    for g, rows in groups.items():
        Xc = feats(co[rows], DAYS) if len(rows) else np.zeros((0, len(DAYS), X.shape[2]), np.float32)
        pc = probs(inp(Xc)) if len(rows) else np.zeros((0, N_ACT), np.float32)
        z = np.abs((Xc - mu) / sd).mean() if len(rows) else float("nan")
        mass = pc.mean(0) if len(pc) else np.zeros(N_ACT)
        rep[g] = {"seats": int(len(rows)), "maxprob": round(float(pc.max(1).mean()), 3) if len(pc) else None,
                  "mean_abs_z": round(float(z), 3), "z_league_heldout": round(float(np.abs((X[hold] - mu) / sd).mean()), 3),
                  "top_profiles": [[int(k), round(float(mass[k]), 3)] for k in np.argsort(-mass)[:8]]}
        if g == "top":
            np.savez_compressed(os.path.join(TRAIN, "top_idm.npz"), row=rows, days=np.array(DAYS),
                                probs=pc.reshape(len(rows), len(DAYS), N_ACT).astype(np.float16), score=cm["score"][rows])
        print(f"[idm] {g}: {len(rows)} seats, max-prob {rep[g]['maxprob']}, mean |z| {rep[g]['mean_abs_z']} "
              f"(league {rep[g]['z_league_heldout']}), top profiles {rep[g]['top_profiles'][:5]}", flush=True)
    rep["sec"] = round(time.time() - t0)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "days": DAYS}, os.path.join(OUT, "idm.pt"))
    print(f"[idm] done in {rep['sec']}s -> {OUT}/report.json, {TRAIN}/top_idm.npz")


if __name__ == "__main__":
    main()
