"""CLONE detection via SYMMETRY DISTANCE in the GM embedding space.

Uses the unsupervised encoder from embed_gm.py. The idea you asked for: instead
of a supervised "is-clone" label learned from self-play (which risks memorising
agent fingerprints), we cluster/embed GM behaviour and define clone as a RELATION:

    encode the board from MY seat  -> z0
    encode the board from OPP seat -> z1
    clone score(t) = -||z0 - z1||          (a true mirror keeps z0 ~= z1)

No clone labels are needed to TRAIN (the encoder is unsupervised on GM). We only
use mirror/cross games to VALIDATE and report an AUC, and to compare against the
supervised baseline clone_nn.json when it exists.

Also reports which STYLE CLUSTER (embed_kmeans.json) each roster agent lands in.

Run: python -m kaggriculture.bandit.nn.clone_cluster --seeds 4
"""
import argparse, glob, itertools, json, os
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.measure.eval_harness as EH
import kaggriculture.bandit.nn.extract as EX
from kaggriculture.bandit.nn.selfplay import ROSTER

OUT = os.path.join(ROOT, ".local", "nn")


def enc_forward(nn, X):
    z = (X - np.array(nn["mu"])) / np.array(nn["sd"])
    L = nn["layers"]
    for i, l in enumerate(L):
        z = z @ np.array(l["w"]).T + np.array(l["b"])
        if i < len(L) - 1:
            z = np.maximum(z, 0.0)          # ReLU on hidden, linear bottleneck
    return z


def auc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    o = np.argsort(p); ys = y[o]; pos = ys.sum(); neg = len(ys) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((np.cumsum(1 - ys) * ys).sum() / (pos * neg))


def _steps(a0, a1, seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([a0, a1])
    return env.steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--warmup", type=int, default=120, help="skip first N steps (board not yet divergent)")
    a = ap.parse_args()
    enc = json.load(open(os.path.join(OUT, "embed_encoder.json")))
    km = None
    kmp = os.path.join(OUT, "embed_kmeans.json")
    if os.path.exists(kmp):
        km = json.load(open(kmp)); C = np.array(km["centroids"])
    roster = [(n, p) for n, p in ROSTER if os.path.exists(p)]
    agents = {n: EH._as_agent(p) for n, p in roster}
    names = [n for n, _ in roster]
    print(f"clone_cluster roster: {names}", flush=True)

    game_scores = []; game_labels = []; step_scores = []; step_labels = []
    clusters = {n: [] for n in names}   # style cluster votes per agent (from seat-0 embeddings)
    for si in range(a.seeds):
        seed = a.seed0 + si
        for i, j in itertools.combinations_with_replacement(names, 2):
            clone = 1.0 if i == j else 0.0
            try:
                steps = _steps(agents[i], agents[j], seed)
            except Exception as e:
                print(f"  skip {i} vs {j} seed{seed}: {e}", flush=True); continue
            T = len(steps); dists = []
            for t in range(a.warmup, T):
                obs0 = (steps[t][0] or {}).get("observation") or {}
                if not obs0.get("market"):
                    continue
                f0, _ = EX._feat(steps, t, 0)
                f1, _ = EX._feat(steps, t, 1)
                z0 = enc_forward(enc, np.asarray(f0, float)[None, :])[0]
                z1 = enc_forward(enc, np.asarray(f1, float)[None, :])[0]
                d = float(np.linalg.norm(z0 - z1))
                dists.append(d)
                step_scores.append(-d); step_labels.append(clone)
                if km is not None:
                    clusters[i].append(int(((z0 - C) ** 2).sum(1).argmin()))
            if dists:
                game_scores.append(-float(np.mean(dists))); game_labels.append(clone)
        print(f"  seed {seed} done: games={len(game_labels)}", flush=True)

    g_auc = auc(game_labels, game_scores)
    s_auc = auc(step_labels, step_scores)
    npos = int(sum(game_labels))
    print(f"\nCLUSTER-SYMMETRY clone detector:", flush=True)
    print(f"  game-level AUC = {g_auc:.3f}  (n={len(game_labels)} games, {npos} mirror positives)", flush=True)
    print(f"  step-level AUC = {s_auc:.3f}  (n={len(step_scores)} rows)", flush=True)
    if km is not None:
        print(f"\n  dominant style cluster per agent (of K={km['k']}):", flush=True)
        for n in names:
            if clusters[n]:
                vals, cnts = np.unique(clusters[n], return_counts=True)
                print(f"    {n:14s} -> cluster {int(vals[cnts.argmax()])}  "
                      f"dist={np.round(cnts/cnts.sum(),2).tolist()}", flush=True)
    # compare to supervised baseline if present
    base = os.path.join(OUT, "clone_nn.json")
    if os.path.exists(base):
        print(f"\n  (supervised baseline clone_nn.json exists -- compare its held-out AUC "
              f"against {g_auc:.3f} to choose the clone head)", flush=True)
    json.dump({"game_auc": g_auc, "step_auc": s_auc, "n_games": len(game_labels),
               "n_mirror": npos, "seeds": a.seeds},
              open(os.path.join(OUT, "clone_cluster_result.json"), "w"), indent=1)
    print(f"\nwrote clone_cluster_result.json", flush=True)


if __name__ == "__main__":
    main()
