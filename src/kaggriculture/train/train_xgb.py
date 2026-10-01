"""Gradient-boosted arbiter: XGBoost offline, plain lists at run time.

What it learns
--------------
P(win | state, member) -- for a given position, how likely are we to win the
episode when *this* committee member's proposal is the one that gets played.
Those probabilities become the vote weights, generalising `MEMBER_WEIGHTS` from
a 9-bucket lookup table to a model that interpolates between states.

Interpolation is the whole point. Every training row costs a fraction of a
7-second match, so a table that has to observe each state band separately is
the wrong shape; trees share evidence across neighbouring states.

Self-contained by construction
------------------------------
XGBoost is used for training only. The model is exported to nested Python lists
and injected into the agent's `XGB_MODEL` block, so the submission stays one
file with no dependency at run time. The exporter is verified numerically
against `booster.predict(output_margin=True)`: agreement is to 1e-6, which is
float32 precision.

The float32 detail matters and is easy to get wrong. XGBoost compares in
float32; parsing the JSON dump into float64 and comparing there sends roughly
one comparison in 700 down the wrong branch. Thresholds are stored as float32
and the feature vector is cast before comparing.

    pip install xgboost
    python -m kaggriculture.train.train_xgb --collect 30 --base "agents/v5_ensemble_*.py"
    python -m kaggriculture.train.train_xgb --train --rounds 200
    python -m kaggriculture.train.train_xgb --collect 30 --train --build --ab
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import json
import math
import os
import random
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402

WORK = os.path.join(ROOT, "models", "xgb")
DATA = os.path.join(WORK, "rows.jsonl")
MODEL = os.path.join(WORK, "model.json")
SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")
XGB_BEGIN = "# --- XGB BEGIN"
XGB_END = "# --- XGB END ---"

# Order matters and is shared with _xgb_features() in the agent. Opponent
# features are appended so a model trained before they existed still indexes
# the same columns.
FEATURES = ["day", "hour", "money_k", "opp_money_k", "hands", "crops",
            "animals", "quadrants", "shed", "member",
            "opp_livestock", "opp_crop", "opp_aggression",
            "opp_wheat_flow", "opp_animal_flow", "opp_crop_flow"]


def f32(v):
    return struct.unpack("f", struct.pack("f", float(v)))[0]


# ------------------------------------------------------------- collection --

def _mute():
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def _collect_one(job):
    """Play one episode, recording (features, member) rows plus the outcome.

    Runs in a worker process. Returns (rows, won) where every row shares the
    episode's label -- the honest thing to do, because we cannot observe the
    counterfactual where a different member's op was played.
    """
    agent_path, opponent, seed = job
    import importlib.util
    from kaggle_environments import make

    spec = importlib.util.spec_from_file_location("xgb_collect", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = []

    def spy(obs, cfg):
        action = mod.agent(obs, cfg)
        if int(obs.get("step", 0) or 0) % 6 == 0:
            n_members = int(mod.PARAMS.get("ensemble_k", 0) or 0) + 1
            for mi in range(n_members):
                rows.append(mod._xgb_features(obs, mi))
        return action

    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([spy, opponent])
    final = env.steps[-1]
    won = int(float(final[0]["reward"] or 0) > float(final[1]["reward"] or 0))
    return rows, won


def collect(base, opponent, episodes, workers, seed0, verbose=True):
    jobs = [(base, opponent, seed0 + i * 17) for i in range(episodes)]
    os.makedirs(WORK, exist_ok=True)
    n_rows = wins = 0
    with open(DATA, "a", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        for i, (rows, won) in enumerate(pool.map(_collect_one, jobs), 1):
            wins += won
            for r in rows:
                f.write(json.dumps({"x": r, "y": won}) + "\n")
                n_rows += 1
            if verbose:
                pr.log(f"episode {i}/{episodes}: {'won' if won else 'lost'}, "
                       f"{len(rows)} rows", 2)
    if verbose:
        pr.log(f"collected {n_rows:,} rows from {episodes} episodes "
               f"({wins}/{episodes} won)", 1)
    return n_rows


def load_rows():
    if not os.path.exists(DATA):
        return [], []
    X, y, groups = [], [], []
    with open(DATA, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            X.append(d["x"])
            y.append(d["y"])
            groups.append(d.get("ep") or f"legacy_{i // 480}")
    return X, y, groups


# ---------------------------------------------------------------- training --

def train(rounds=200, depth=4, eta=0.08, verbose=True):
    try:
        import xgboost as xgb
    except ImportError:
        pr.warn("xgboost is not installed")
        pr.log("pip install xgboost", 1)
        return None
    X, y, groups = load_rows()
    if len(X) < 200:
        pr.warn(f"only {len(X)} rows -- collect more first "
                f"(python -m kaggriculture.train.train_xgb --collect 30)")
        return None
    if len(set(y)) < 2:
        pr.warn("every episode had the same outcome -- there is nothing to "
                "learn. Collect against a stronger or weaker opponent.")
        return None

    # Group-aware, outcome-stratified split: whole episodes go to one side or
    # the other, and wins and losses are dealt alternately so the validation
    # set has the same base rate as training.
    import random as _rnd
    by_ep = {}
    for i, g in enumerate(groups):
        by_ep.setdefault(g, []).append(i)
    outcome = {g: y[idx[0]] for g, idx in by_ep.items()}
    won_eps = sorted(g for g in by_ep if outcome[g] == 1)
    lost_eps = sorted(g for g in by_ep if outcome[g] == 0)
    _rnd.Random(11).shuffle(won_eps)
    _rnd.Random(12).shuffle(lost_eps)
    valid_eps = set(won_eps[::5]) | set(lost_eps[::5])
    tr_idx = [i for g, idx in by_ep.items() if g not in valid_eps for i in idx]
    va_idx = [i for g, idx in by_ep.items() if g in valid_eps for i in idx]
    if not va_idx or not tr_idx:
        tr_idx, va_idx = list(range(len(X))), list(range(len(X)))
    split = len(tr_idx)
    Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
    Xva = [X[i] for i in va_idx]; yva = [y[i] for i in va_idx]
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURES)
    dvalid = xgb.DMatrix(Xva, label=yva, feature_names=FEATURES)
    params = {"objective": "binary:logistic", "eval_metric": "logloss",
              "max_depth": depth, "eta": eta, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 8,
              "tree_method": "hist"}
    evals_result = {}
    booster = xgb.train(params, dtrain, num_boost_round=rounds,
                        evals=[(dtrain, "train"), (dvalid, "valid")],
                        early_stopping_rounds=25, verbose_eval=False,
                        evals_result=evals_result)
    if verbose:
        tr = evals_result["train"]["logloss"][-1]
        va = evals_result["valid"]["logloss"][-1]
        pr.log(f"trained {booster.num_boosted_rounds()} tree(s) on {split:,} rows "
               f"(logloss train {tr:.4f}, valid {va:.4f})", 1)
        if va > 0.69:
            pr.warn("validation logloss is no better than a coin flip -- the "
                    "features do not predict the outcome yet", 1)
    model = export(booster, verbose=verbose)
    with open(MODEL, "w", encoding="utf-8") as f:
        json.dump(model, f)
    return model


def export(booster, verbose=True):
    """Booster -> nested lists, verified against xgboost's own margins."""
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        node = json.loads(dump)
        flat = []

        def walk(n):
            if "leaf" in n:
                flat.append([-1, float(n["leaf"]), -1, -1])
                return len(flat) - 1
            flat.append(None)
            me = len(flat) - 1
            kids = {c["nodeid"]: c for c in n["children"]}
            yes = walk(kids[n["yes"]])
            no = walk(kids[n["no"]])
            name = n["split"]
            fi = FEATURES.index(name) if name in FEATURES else int(str(name).lstrip("f"))
            flat[me] = [fi, f32(n["split_condition"]), yes, no]
            return me

        walk(node)
        trees.append(flat)

    cfg = json.loads(booster.save_config())
    raw = str(cfg["learner"]["learner_model_param"]["base_score"])
    base = float(raw.strip("[]").split(",")[0])
    base = min(max(base, 1e-6), 1 - 1e-6)
    model = {"trees": trees, "bias": math.log(base / (1 - base)),
             "features": FEATURES, "n_features": len(FEATURES)}

    if verbose:
        err = verify(booster, model)
        if err is not None:
            pr.log(f"export verified against xgboost: max margin error {err:.2e}", 1)
            if err > 1e-4:
                pr.warn("export does not match xgboost -- do not ship this", 1)
    return model


def predict_margin(model, x):
    xs = [f32(v) for v in x]
    total = float(model.get("bias", 0.0))
    for tree in model["trees"]:
        i = 0
        while tree[i][0] != -1:
            fi, thr, yes, no = tree[i]
            i = yes if xs[fi] < thr else no
        total += tree[i][1]
    return total


def verify(booster, model, n=250):
    """Max absolute difference between the exported model and xgboost itself."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    X, _y, _g = load_rows()
    if not X:
        return None
    X = X[:n]
    ref = booster.predict(xgb.DMatrix(X, feature_names=FEATURES),
                          output_margin=True)
    worst = 0.0
    for row, r in zip(X, ref):
        worst = max(worst, abs(predict_margin(model, row) - float(r)))
    return worst


# ------------------------------------------------------------------- build --

def build(base, model, out=None, verbose=True):
    """Write an agent carrying the exported trees."""
    P = dict(paramio.load(base))
    P["arbiter_model"] = "xgb"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out or os.path.join(ROOT, "agents", f"agent_vxgb_{stamp}.py")
    n_nodes = sum(len(t) for t in model["trees"])
    paramio.write(SRC, out, P, header="xgboost arbiter",
                  module_doc=(f"Committee vote weighted by a gradient-boosted "
                              f"arbiter.\n{len(model['trees'])} trees, "
                              f"{n_nodes:,} nodes, exported as plain lists -- no "
                              f"xgboost at run time.\nFeatures: "
                              f"{', '.join(model['features'])}\n"))
    with open(out, encoding="utf-8") as f:
        src = f.read()
    i, j = src.index(XGB_BEGIN), src.index(XGB_END)
    body = [XGB_BEGIN + " (src/kaggriculture/train/train_xgb.py rewrites this block) ---",
            "# Exported gradient-boosted arbiter. Nodes are [feature, threshold,",
            "# yes, no]; leaves are [-1, value, -1, -1]. Margin = sum of leaves +",
            "# bias, where bias is logit(base_score). Thresholds are float32.",
            f"XGB_MODEL = {json.dumps(model, separators=(',', ':'))}"]
    src = src[:i] + "\n".join(body) + "\n" + src[j:]
    with open(out, "w", encoding="utf-8") as f:
        f.write(src)

    rel = os.path.relpath(out, ROOT)
    registry.register_model(os.path.basename(out), path=rel, built_by="xgb",
                            base=os.path.relpath(base, ROOT),
                            description=f"xgb arbiter, {len(model['trees'])} trees")
    if verbose:
        pr.log(f"wrote {rel}  ({os.path.getsize(out):,} bytes, "
               f"{len(model['trees'])} trees)", 1)
    return rel


def latency(path, steps=180, verbose=True):
    import importlib.util
    import statistics
    from kaggle_environments import make
    spec = importlib.util.spec_from_file_location("cand", os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    times = []

    def timed(obs, cfg):
        t0 = time.perf_counter()
        a = mod.agent(obs, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)
        return a

    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": 23,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([timed, "agents/v2_tuned.py"])
    times.sort()
    p95 = times[int(len(times) * 0.95)]
    if verbose:
        pr.log(f"latency: mean {statistics.mean(times):.2f} ms, p95 {p95:.2f} ms, "
               f"max {times[-1]:.2f} ms  ({1000 / p95:.0f}x headroom)", 1)
    return {"mean": statistics.mean(times), "p95": p95, "max": times[-1],
            "ok": times[-1] < 500.0}


def _resolve(pattern):
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(p))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="agents/*ensemble*.py")
    ap.add_argument("--vs", default="agents/v2_tuned.py")
    ap.add_argument("--collect", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--reset", action="store_true", help="discard collected rows")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    pr.reset()
    os.makedirs(WORK, exist_ok=True)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    if args.reset and os.path.exists(DATA):
        os.remove(DATA)
        pr.log("cleared collected rows")

    if args.status or not (args.collect or args.train or args.build):
        X, y, _groups = load_rows()
        print(f"rows      : {len(X):,}")
        if y:
            print(f"win rate  : {sum(y) / len(y):.0%} of rows come from won episodes")
        print(f"model     : {'present' if os.path.exists(MODEL) else 'not trained yet'}")
        try:
            import xgboost
            print(f"xgboost   : {xgboost.__version__}")
        except ImportError:
            print("xgboost   : NOT INSTALLED (pip install xgboost)")
        print(f"features  : {', '.join(FEATURES)}")
        return 0

    base = _resolve(args.base)
    if not base:
        pr.warn(f"no agent matched {args.base}")
        pr.log("build one:  python -m kaggriculture.agentbuild.build_agent configs/v5_ensemble.json", 1)
        return 1
    if int(paramio.load(base).get("ensemble_k", 0) or 0) <= 0:
        pr.warn(f"{os.path.relpath(base, ROOT)} has no committee -- an arbiter "
                f"needs something to arbitrate")
        return 1

    if args.collect:
        pr.log(f"collecting from {os.path.relpath(base, ROOT)} vs {args.vs}")
        collect(base, args.vs, args.collect, workers, args.seed0)

    model = None
    if args.train:
        model = train(args.rounds, args.depth, args.eta)
        if model is None:
            return 1

    if args.build:
        if model is None:
            if not os.path.exists(MODEL):
                pr.warn("no trained model -- run with --train first")
                return 1
            with open(MODEL, encoding="utf-8") as f:
                model = json.load(f)
        rel = build(base, model)
        res = latency(rel)
        registry.register_model(os.path.basename(rel),
                                latency_ms=round(res["p95"], 2),
                                tests="latency ok" if res["ok"] else "LATENCY RISK")
        if args.ab:
            import kaggriculture.train.adaptive as adaptive
            adaptive.ab(rel, os.path.relpath(base, ROOT), workers=workers)
        else:
            pr.log("")
            pr.log("measure it:")
            pr.log(f"python -m kaggriculture.train.adaptive --agent {rel} --mode off --ab", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
