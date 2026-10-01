"""How wrong is the forecaster? Measure it against the official interpreter.

`src/kaggriculture/engine/fastsim.py` estimates what a position is worth. Any such estimate is
useless until you know its error, and dangerous if you assume it. This plays
real episodes in the vendored `kaggle-environments` interpreter -- the ground
truth -- records a forecast at checkpoints, then compares each forecast to what
that farm was actually worth `horizon` turns later.

    python -m kaggriculture.engine.parity --matches 4 --horizon 72
    python -m kaggriculture.engine.parity --matches 8 --json          # for the dashboard

Read the ratio column, not the dollar error. A forecaster that is 20% low
everywhere is perfectly usable for *ranking* candidate plans, which is all
search needs; one that is unbiased on average but swings either way is not.
Bias you can correct; variance you cannot.

Exit code is 1 when the error exceeds --max-error, so this can gate a build.
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.engine.fastsim as fastsim  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402


def run_match(agent, opponent, seed, horizon, steps, checkpoints):
    """Play one episode, recording (forecast, later actual) pairs for seat 0."""
    from kaggle_environments import make
    snaps = {}

    import importlib.util
    spec = importlib.util.spec_from_file_location("parity_agent",
                                                  os.path.join(ROOT, agent))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def spy(obs, cfg):
        step = int(obs.get("step", 0) or 0)
        if step in checkpoints or step % max(1, horizon // 2) == 0:
            snaps[step] = {"forecast": fastsim.forecast(obs, cfg, horizon),
                           "actual_now": fastsim.farm_value(obs)}
        return mod.agent(obs, cfg)

    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([spy, opponent])

    pairs = []
    for step, rec in sorted(snaps.items()):
        later = step + horizon
        if later in snaps:
            pairs.append({
                "step": step,
                "predicted": rec["forecast"]["forecast"],
                "actual": snaps[later]["actual_now"],
                "labour_bound": rec["forecast"]["labour_bound"],
            })
    final = float(env.steps[-1][0]["reward"] or 0)
    return pairs, final


def summarise(pairs):
    if not pairs:
        return {"n": 0, "error": "no comparable checkpoints -- raise --steps "
                                 "or lower --horizon"}
    errs = [p["predicted"] - p["actual"] for p in pairs]
    ratios = [(p["predicted"] / p["actual"]) if p["actual"] else 0.0 for p in pairs]
    abs_pct = [abs(e) / max(abs(p["actual"]), 1.0) for e, p in zip(errs, pairs)]

    # Does it rank positions correctly? That is the property search relies on.
    ordered = 0
    comparisons = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            if a["actual"] == b["actual"]:
                continue
            comparisons += 1
            ordered += int((a["predicted"] > b["predicted"]) ==
                           (a["actual"] > b["actual"]))
    return {
        "n": len(pairs),
        "mean_error": statistics.mean(errs),
        "median_abs_pct": statistics.median(abs_pct),
        "mean_ratio": statistics.mean(ratios),
        "ratio_spread": (statistics.pstdev(ratios) if len(ratios) > 1 else 0.0),
        "rank_accuracy": (ordered / comparisons) if comparisons else None,
        "labour_bound_share": sum(p["labour_bound"] for p in pairs) / len(pairs),
    }


def verdict(s, max_error):
    if s.get("n", 0) == 0:
        return "no data", False
    pct = s["median_abs_pct"]
    rank = s.get("rank_accuracy") or 0.0
    if pct <= max_error and rank >= 0.75:
        return (f"usable: median error {pct:.0%}, ranks positions correctly "
                f"{rank:.0%} of the time"), True
    if rank >= 0.75:
        return (f"biased but usable for ranking: median error {pct:.0%} is high, "
                f"but it still orders positions correctly {rank:.0%} of the time. "
                f"Use it to compare plans, never to predict a bank."), True
    return (f"NOT usable: median error {pct:.0%} and it only ranks positions "
            f"correctly {rank:.0%} of the time. Do not train against it."), False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="agents/agent_v4_optimal_20260805_014340.py")
    ap.add_argument("--vs", default="agents/v2_tuned.py")
    ap.add_argument("--matches", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=72)
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--seed0", type=int, default=1200)
    ap.add_argument("--max-error", type=float, default=0.35)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pr.reset()
    import glob
    hits = sorted(glob.glob(os.path.join(ROOT, args.agent)))
    agent = os.path.relpath(hits[-1], ROOT) if hits else args.agent

    checkpoints = set(range(args.horizon, args.steps - args.horizon,
                            max(1, args.horizon // 2)))
    all_pairs = []
    for i in range(args.matches):
        seed = args.seed0 + i * 37
        pairs, final = run_match(agent, args.vs, seed, args.horizon,
                                 args.steps, checkpoints)
        all_pairs.extend(pairs)
        if not args.json:
            pr.log(f"seed {seed}: {len(pairs)} comparable checkpoint(s), "
                   f"final ${final:,.0f}", 1)

    s = summarise(all_pairs)
    msg, ok = verdict(s, args.max_error)
    s["verdict"] = msg
    s["ok"] = ok
    s["agent"] = agent
    s["horizon"] = args.horizon

    if args.json:
        print(json.dumps(s, indent=1))
        return 0 if ok else 1

    print()
    if s.get("n", 0) == 0:
        pr.warn(s.get("error", "no data"))
        return 1
    print(f"forecast horizon      : {args.horizon} turns")
    print(f"comparisons           : {s['n']}")
    print(f"median absolute error : {s['median_abs_pct']:.0%}")
    print(f"mean predicted/actual : {s['mean_ratio']:.2f}  "
          f"(spread {s['ratio_spread']:.2f})")
    print(f"rank accuracy         : {(s['rank_accuracy'] or 0):.0%}")
    print(f"labour-bound positions: {s['labour_bound_share']:.0%}")
    print(f"\n{msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
