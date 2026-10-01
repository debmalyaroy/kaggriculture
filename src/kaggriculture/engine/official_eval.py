"""Evaluate an agent exactly the way Kaggle does.

`src/kaggriculture/measure/evaluate.py` is the fast development harness: it relaxes `actTimeout` so
many matches can be batched. This script does the opposite — it runs the agent
under the **stock competition configuration**, with no overrides at all, which
is the same code path Kaggle's servers use to score a submission:

    env = make("kaggriculture", debug=True)     # episodeSteps 720, actTimeout 1
    env.run([your_agent, opponent])

That means a slow turn really does forfeit here, exactly as it would on the
leaderboard. Run this before every submission.

It also mirrors the official "Kaggriculture: Getting Started" notebook flow:
play against the three built-in agents, print rewards and statuses, and dump a
replay JSON plus a standalone HTML replay you can open in a browser.

Usage:
    python -m kaggriculture.engine.official_eval --agent agents/v2_tuned.py
    python -m kaggriculture.engine.official_eval --agent agents/v2_tuned.py --vs starter --replay
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys
import time

import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

BUILTINS = ["pass", "random", "starter"]


def resolve(spec):
    return spec if spec in BUILTINS else os.path.abspath(os.path.join(ROOT, spec))


def run_one(agent_path, opponent, seed, want_replay, out_dir):
    from kaggle_environments import make

    # No configuration overrides: stock 720 steps, stock 1-second actTimeout,
    # stock market curves. This is what the competition runs.
    config = {} if seed is None else {"seed": seed}
    env = make("kaggriculture", configuration=config, debug=True)

    t0 = time.time()
    env.run([agent_path, opponent])
    wall = time.time() - t0

    final = env.steps[-1]
    rewards = [float(s["reward"] or 0) for s in final]
    statuses = [s["status"] for s in final]

    print(f"  vs {opponent:<10} "
          f"you ${rewards[0]:>10,.0f} [{statuses[0]}]   "
          f"opp ${rewards[1]:>9,.0f} [{statuses[1]}]   "
          f"{'WIN ' if rewards[0] > rewards[1] else 'LOSS'}  {wall:5.1f}s")

    if statuses[0] not in ("DONE", "ACTIVE"):
        print(f"    !! your agent finished with status {statuses[0]} -- on Kaggle "
              f"this forfeits the episode (usually a turn over actTimeout=1s, "
              f"or an exception)")

    if want_replay:
        os.makedirs(out_dir, exist_ok=True)
        tag = opponent.replace("/", "_").replace(".py", "")
        with open(os.path.join(out_dir, f"replay_{tag}.json"), "w") as f:
            json.dump(env.toJSON(), f)
        try:
            html = env.render(mode="html", width=1100, height=800)
            with open(os.path.join(out_dir, f"replay_{tag}.html"), "w",
                      encoding="utf-8") as f:
                f.write(html)
        except Exception as exc:                                   # noqa: BLE001
            print(f"    (html render unavailable: {exc})")
    return rewards, statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agents/v2_tuned.py")
    ap.add_argument("--vs", nargs="*", default=BUILTINS,
                    help="opponents; default is the three built-ins")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a random episode, as Kaggle does")
    ap.add_argument("--replay", action="store_true", help="write replay JSON + HTML")
    ap.add_argument("--out", default=os.path.join(ROOT, ".local", "replays"))
    args = ap.parse_args()

    agent_path = resolve(args.agent)
    if not os.path.exists(agent_path):
        sys.exit(f"no such agent: {agent_path}")

    print(f"official-configuration evaluation of {args.agent}")
    print("  configuration: stock (episodeSteps 720, actTimeout 1s, default market)\n")

    wins = 0
    for opp in args.vs:
        rewards, statuses = run_one(agent_path, resolve(opp), args.seed,
                                    args.replay, args.out)
        wins += rewards[0] > rewards[1]

    print(f"\n{wins}/{len(args.vs)} wins under stock configuration")
    if args.replay:
        print(f"replays in {args.out}  (open the .html in a browser)")
    print("\nNote: the Kaggle leaderboard is a *skill rating* from head-to-head")
    print("episodes against other people's submissions, not a bank total. Local")
    print("results against the built-ins only prove legality and a floor.")


if __name__ == "__main__":
    main()
