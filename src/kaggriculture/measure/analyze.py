"""Play one match and print a day-by-day diagnostic of an agent's farm.

Usage:
    python -m kaggriculture.measure.analyze agents/v1_heuristic.py --vs pass --seed 3 --every 4
"""
import argparse
import os
import sys

import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

BUILTINS = {"pass", "random", "starter"}
SYM = {"WHEAT": "w", "CARROT": "c", "TOMATO": "t", "STRAWBERRY": "s", "MELON": "m"}


def resolve(spec):
    return spec if spec in BUILTINS else os.path.abspath(spec)


def sym(t):
    if t is None:
        return "."
    if t == "LOCKED":
        return "#"
    kind = t.get("kind")
    if kind == "PLANT":
        return SYM.get(t["crop"], "?")
    if kind == "WEED":
        return "x"
    if t.get("animal") == "GOOSE":
        return "G"
    if t.get("animal") == "COW":
        return "C"
    if t.get("animal") == "SHEEP":
        return "S"
    if kind == "COOP":
        return "o"
    if kind == "PASTURE":
        return "p"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--vs", default="pass")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--hour", type=int, default=8)
    ap.add_argument("--board", action="store_true")
    args = ap.parse_args()

    from kaggle_environments import make

    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": args.seed,
                              "actTimeout": 60, "runTimeout": 100000},
               debug=True)
    env.run([resolve(args.agent), resolve(args.vs)])

    prev = 3000.0
    print(f"{'day':>4}{'bank':>9}{'delta':>9}{'hands':>6}{'quads':>6}"
          f"{'crops':>6}{'live':>5}{'pens':>5}{'weed':>5}  shed / prices")
    for s in env.steps:
        obs = s[0]["observation"]
        if obs["hour"] != args.hour:
            continue
        d = obs["day"]
        if d % args.every and d != 29:
            continue
        f = obs["farms"][0]
        priv = s[0]["observation"]["private"]
        crops = live = pens = weed = 0
        for row in f["tiles"]:
            for t in row:
                if isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        crops += 1
                    elif k == "WEED":
                        weed += 1
                    elif "animal" in t:
                        live += 1
                    elif k in ("COOP", "PASTURE"):
                        pens += 1
        shed = {k: v for k, v in priv["shed"].items() if v}
        print(f"{d:>4}{f['money']:>9,.0f}{f['money'] - prev:>9,.0f}{len(f['hands']):>6}"
              f"{len(f['unlocked_quadrants']):>6}{crops:>6}{live:>5}{pens:>5}{weed:>5}  {shed}")
        prices = {k: v for k, v in obs["market"]["prices"].items() if k != "FERTILIZER"}
        print(f"      prices {prices}")
        if args.board:
            for row in f["tiles"]:
                print("      " + " ".join(sym(t) for t in row))
        prev = f["money"]

    final = env.steps[-1]
    print(f"\nFINAL  {args.agent}: ${final[0]['reward']:,.0f}   "
          f"{args.vs}: ${final[1]['reward']:,.0f}")
    inv = env.steps[-1][0]["observation"]["market"]["inventory"]
    print("market inventory delta vs I0:", {k: v - 10000 for k, v in inv.items()})


if __name__ == "__main__":
    main()
