"""RL bot — Monte-Carlo control over the allocation policy.

This is reinforcement learning scoped to what this compute budget can actually
carry. Full deep RL on the raw board needs ~10^6 episodes; at ~5 s an episode on
two cores that is decades (see docs/history/ml-roadmap.md). So we shrink the problem
until RL fits it:

* **Policy**   ASSET_BIAS: a multiplier per (asset, day-bucket, cash-bucket)
               applied to that asset's planning score. 8 assets x 4 x 4 = 128
               states, each with 5 discrete actions.
* **Episode**  one 720-turn match against a frozen opponent pool.
* **Reward**   1 win / 0.5 tie / 0 loss — the ladder scores exactly this.
* **Update**   Monte-Carlo control: every (state, action) visited in the episode
               is credited with the episode return; act epsilon-greedy on the
               running means, with optimistic initialisation.

Why MC and not TD: the reward is genuinely terminal, and intermediate rewards
would need shaping — and the shaping function *is* the hand-written value model
we are trying to improve on. MC keeps the objective honest.

Resumable: state is checkpointed after every episode.

    python -m kaggriculture.train.train_rl --episodes 200
    python -m kaggriculture.train.train_rl --episodes 200 --resume
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402

WORK = os.path.join(ROOT, "models", "rl")
ASSETS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP"]
ACTIONS = [0.5, 0.75, 1.0, 1.5, 2.0]
OPTIMISTIC = 0.60          # above a realistic win rate -> encourages exploration


def all_keys():
    return [f"{a}|{d}|{c}" for a in ASSETS for d in range(4) for c in range(4)]


def greedy_table(Q):
    out = {}
    for k, arms in Q.items():
        best = max(range(len(ACTIONS)),
                   key=lambda i: arms[i]["mean"] if arms[i]["n"] else OPTIMISTIC)
        if ACTIONS[best] != 1.0:
            out[k] = ACTIONS[best]
    return out


def sample_table(Q, eps, rng):
    tbl, chosen = {}, {}
    for k, arms in Q.items():
        if rng.random() < eps:
            i = rng.randrange(len(ACTIONS))
        else:
            i = max(range(len(ACTIONS)),
                    key=lambda j: arms[j]["mean"] if arms[j]["n"] else OPTIMISTIC)
        chosen[k] = i
        if ACTIONS[i] != 1.0:
            tbl[k] = ACTIONS[i]
    return tbl, chosen


def play(cand, opp, seed):
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([cand, opp])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "ml_rl.py"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.30)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--seed0", type=int, default=31000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    qpath = os.path.join(WORK, "q.json")

    base_agent = args.base
    if not base_agent:
        import glob
        c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        base_agent = c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")
    P = paramio.load(base_agent)
    src = os.path.join(ROOT, "agents", "v1_heuristic.py")

    pool = [p for p in (base_agent,
                        os.path.join(ROOT, "agents", "v2_tuned.py"),
                        os.path.join(ROOT, "agents", "v1_heuristic.py"))
            if os.path.exists(p)]

    if args.resume and os.path.exists(qpath):
        st = json.load(open(qpath))
        Q, ep0 = st["Q"], st["episodes"]
        print(f"resumed after {ep0} episodes")
    else:
        Q = {k: [{"n": 0, "mean": 0.0} for _ in ACTIONS] for k in all_keys()}
        ep0 = 0

    rng = random.Random(4242 + ep0)
    cand = os.path.join(WORK, "cand.py")
    wins = 0.0
    t0 = time.time()

    for e in range(ep0, ep0 + args.episodes):
        frac = min(1.0, (e - ep0) / max(1, args.episodes - 1))
        eps = args.eps + (args.eps_final - args.eps) * frac
        tbl, chosen = sample_table(Q, eps, rng)
        paramio.write(src, cand, P, header=f"RL episode {e}",
                      module_doc="RL training candidate\n", asset_bias=tbl)
        opp = pool[e % len(pool)]
        seed = args.seed0 + (e % 40)
        mine, theirs = play(cand, opp, seed)
        r = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
        wins += r
        for k, i in chosen.items():
            arm = Q[k][i]
            arm["n"] += 1
            arm["mean"] += (r - arm["mean"]) / arm["n"]
        json.dump({"Q": Q, "episodes": e + 1}, open(qpath, "w"))
        if (e - ep0 + 1) % 5 == 0 or e == ep0:
            print(f"  ep {e+1:>4}  eps {eps:.2f}  running win rate "
                  f"{100*wins/(e-ep0+1):5.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    tbl = greedy_table(Q)
    visited = sum(1 for k in Q if any(a["n"] for a in Q[k]))
    paramio.write(src, args.out, P,
                  header=f"RL (Monte-Carlo control) after {ep0+args.episodes} "
                         f"episodes -- do not hand-edit.",
                  module_doc="Kaggriculture agent — RL (Monte-Carlo control).\n\n"
                             "GENERATED. The heuristic planner is unchanged; an\n"
                             "ASSET_BIAS table learned by MC control reweights the\n"
                             "allocation decision only. Retrain with src/kaggriculture/train/train_rl.py.\n",
                  asset_bias=tbl)
    print(f"\n{visited}/{len(Q)} states visited; {len(tbl)} non-neutral biases learned")
    print(f"wrote {args.out}")
    print("VALIDATE over three seed sets before believing it:")
    print(f"  python -m kaggriculture.measure.evaluate agents/ml_rl.py --vs "
          f"{os.path.relpath(base_agent, ROOT)} -n 3 --seed0 91000")


if __name__ == "__main__":
    main()
