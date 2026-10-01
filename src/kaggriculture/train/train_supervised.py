"""Supervised bot — ridge regression on decision-level outcomes.

A deliberately different family from src/kaggriculture/train/train_rl.py, on the same hook.

RL credits a whole episode's return to every choice it made, which is unbiased
but very high variance — 128 states share one scalar per episode. Supervised
learning instead builds a row **per allocation decision**:

    features(day bucket, cash bucket, labour slack, herd, crop count, price
             ratio for that asset)  x  one-hot(asset)      ->   episode return

and fits ridge regression. Every decision becomes a training row, so a single
episode yields dozens rather than one, and the model can *interpolate* to states
never visited — which the tabular RL cannot.

Trade-off, stated plainly: regression on episode return confounds "this choice
was good" with "this agent was good". Ridge sees correlation, MC control sees
(noisy) credit. They fail differently, which is exactly why it is worth having
both and letting self-play referee.

Pure numpy — no sklearn.

    python -m kaggriculture.train.train_supervised --episodes 40
    python -m kaggriculture.train.train_supervised --episodes 40 --resume
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402

WORK = os.path.join(ROOT, "models", "sup")
ASSETS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP"]
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "GOOSE": 50, "COW": 160, "SHEEP": 200}
PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
JITTER = [0.5, 0.75, 1.0, 1.5, 2.0]


def feats(day, money, herd, crops, hands, price_ratio):
    return [day / 30.0, min(money, 40000) / 40000.0, herd / 30.0,
            crops / 60.0, hands / 16.0, min(price_ratio, 3.0) / 3.0, 1.0]


NF = 7


def collect(cand, opp, seed, bias, rows):
    """Play one match, logging every allocation the agent committed to."""
    from kaggle_environments import make
    import importlib.util
    spec = importlib.util.spec_from_file_location("c", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log = []
    two = mod.agent.__code__.co_argcount > 1
    prev = {}

    def wrapped(obs, cfg=None):
        act = mod.agent(obs, cfg) if two else mod.agent(obs)
        p = obs["player"]
        farm = obs["farms"][p]
        op = act.get("farmer") or ["PASS"]
        if op[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE"):
            asset = op[1] if op[0] == "PLANT" else None
            if asset is None:
                # which animal the pen is for is decided later; attribute to the
                # cheapest matching animal we currently target
                asset = "GOOSE" if op[0] == "BUILD_COOP" else "COW"
            c, _e = mod.census(farm["tiles"], len(farm["tiles"]))
            herd = c["GOOSE"] + c["COW"] + c["SHEEP"]
            crops = sum(c[k] for k in mod.CROPS)
            prod = PRODUCT.get(asset, asset)
            pr = obs["market"]["prices"].get(prod, BASE_PRICE[asset]) / BASE_PRICE[asset]
            log.append((asset, feats(int(obs["day"]), float(farm["money"]),
                                     herd, crops, len(farm["hands"]), pr)))
        return act

    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([wrapped, opp])
    f = env.steps[-1]
    mine, theirs = float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)
    y = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    for asset, x in log:
        rows.append({"asset": asset, "x": x, "y": y})
    return mine, theirs, len(log)


def fit(rows, l2=1.0):
    """One ridge model per asset: features -> P(win). Returns coefficients."""
    import numpy as np
    models = {}
    for a in ASSETS:
        sub = [r for r in rows if r["asset"] == a]
        if len(sub) < NF + 4:
            continue
        X = np.array([r["x"] for r in sub], float)
        y = np.array([r["y"] for r in sub], float)
        A = X.T @ X + l2 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        models[a] = w.tolist()
    return models


def to_bias(models, rows):
    """Turn predicted win probability into a multiplier per (asset, day, cash)."""
    import numpy as np
    if not models:
        return {}
    grid_day = [4, 12, 19, 26]
    grid_cash = [700, 3000, 12000, 30000]
    med = {}
    for a in ASSETS:
        sub = [r for r in rows if r["asset"] == a]
        med[a] = (np.median([r["x"][2] for r in sub]) if sub else 0.3,
                  np.median([r["x"][3] for r in sub]) if sub else 0.5,
                  np.median([r["x"][4] for r in sub]) if sub else 0.5,
                  np.median([r["x"][5] for r in sub]) if sub else 0.4)
    preds = {}
    for a, w in models.items():
        for di, d in enumerate(grid_day):
            for ci, cash in enumerate(grid_cash):
                h, cr, hd, pr = med[a]
                x = np.array(feats(d, cash, h * 30, cr * 60, hd * 16, pr * 3.0))
                preds[(a, di, ci)] = float(np.dot(w, x))
    vals = list(preds.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    bias = {}
    for (a, di, ci), v in preds.items():
        # map predicted win-prob into [0.6, 1.6]; neutral omitted
        m = 0.6 + (v - lo) / span
        m = round(min(1.6, max(0.6, m)), 3)
        if abs(m - 1.0) > 0.05:
            bias[f"{a}|{di}|{ci}"] = m
    return bias


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "ml_ridge.py"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=52000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    rpath = os.path.join(WORK, "rows.json")

    base_agent = args.base
    if not base_agent:
        c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        base_agent = c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")
    P = paramio.load(base_agent)
    src = os.path.join(ROOT, "agents", "v1_heuristic.py")
    pool = [p for p in (base_agent,
                        os.path.join(ROOT, "agents", "v2_tuned.py"),
                        os.path.join(ROOT, "agents", "v1_heuristic.py"))
            if os.path.exists(p)]

    rows = []
    ep0 = 0
    if args.resume and os.path.exists(rpath):
        st = json.load(open(rpath))
        rows, ep0 = st["rows"], st["episodes"]
        print(f"resumed with {len(rows):,} rows from {ep0} episodes")

    rng = random.Random(909 + ep0)
    cand = os.path.join(WORK, "cand.py")
    t0 = time.time()
    for e in range(ep0, ep0 + args.episodes):
        # jitter the bias so the logged decisions vary; without exploration the
        # regression only ever sees the incumbent policy's choices
        bias = {f"{a}|{d}|{c}": rng.choice(JITTER)
                for a in ASSETS for d in range(4) for c in range(4)}
        paramio.write(src, cand, P, header=f"sup episode {e}",
                      module_doc="supervised training candidate\n", asset_bias=bias)
        mine, theirs, n = collect(cand, pool[e % len(pool)],
                                  args.seed0 + (e % 40), bias, rows)
        json.dump({"rows": rows, "episodes": e + 1}, open(rpath, "w"))
        if (e - ep0 + 1) % 3 == 0 or e == ep0:
            print(f"  ep {e+1:>3}  {n:>3} decisions  ${mine:>8,.0f} vs ${theirs:>8,.0f}  "
                  f"({len(rows):,} rows, {time.time()-t0:.0f}s)", flush=True)

    models = fit(rows)
    bias = to_bias(models, rows)
    paramio.write(src, args.out, P,
                  header=f"Ridge regression on {len(rows):,} decisions from "
                         f"{ep0+args.episodes} episodes -- do not hand-edit.",
                  module_doc="Kaggriculture agent — supervised (ridge regression).\n\n"
                             "GENERATED. Planner unchanged; the ASSET_BIAS table is\n"
                             "fitted per asset from decision-level outcomes.\n"
                             "Retrain with src/kaggriculture/train/train_supervised.py.\n",
                  asset_bias=bias)
    print(f"\nfitted {len(models)}/{len(ASSETS)} asset models from {len(rows):,} rows")
    print(f"{len(bias)} non-neutral biases; wrote {args.out}")
    print("VALIDATE over three seed sets:")
    print(f"  python -m kaggriculture.measure.evaluate agents/ml_ridge.py --vs "
          f"{os.path.relpath(base_agent, ROOT)} -n 3 --seed0 91000")


if __name__ == "__main__":
    main()
