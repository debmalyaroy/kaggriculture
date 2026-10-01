"""Run a single Kaggriculture match and print a summary.

Usage:
    python -m kaggriculture.engine.run_match agents/v1_heuristic.py random
    python -m kaggriculture.engine.run_match agents/v1_heuristic.py agents/v0_baseline.py --seed 7 --replay .local/replay.json
"""
import argparse
import json
import os
import sys
import time

import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

from kaggle_environments import make  # noqa: E402

BUILTINS = {"pass", "random", "starter"}


def resolve(spec):
    if spec in BUILTINS:
        return spec
    return os.path.abspath(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--render", action="store_true", help="print the final ASCII board")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    config = {"episodeSteps": args.steps, "actTimeout": 60, "runTimeout": 100000}
    if args.seed is not None:
        config["seed"] = args.seed

    env = make("kaggriculture", configuration=config, debug=args.debug)
    t0 = time.time()
    env.run([resolve(args.left), resolve(args.right)])
    elapsed = time.time() - t0

    final = env.steps[-1]
    rewards = [s["reward"] for s in final]
    statuses = [s["status"] for s in final]
    print(f"{args.left:>34}  ${rewards[0]:>10,.0f}   {statuses[0]}")
    print(f"{args.right:>34}  ${rewards[1]:>10,.0f}   {statuses[1]}")
    print(f"elapsed {elapsed:.1f}s ({elapsed / max(1, len(env.steps)) * 1000:.1f} ms/step)")

    if args.render:
        print(env.render(mode="ansi"))

    if args.replay:
        os.makedirs(os.path.dirname(os.path.abspath(args.replay)), exist_ok=True)
        with open(args.replay, "w") as f:
            json.dump(env.toJSON(), f)
        print(f"replay -> {args.replay}")


if __name__ == "__main__":
    main()
