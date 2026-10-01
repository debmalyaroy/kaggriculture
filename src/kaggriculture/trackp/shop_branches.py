"""Shop-conditioned route branches (the Kaito v48 pattern, 2026-08-30).

The strongest public agent branches full routes on town shop-unlock events
(public state, fixed once drawn). A single majority vote over a stable
team's tapes averages across shop-worlds; voting PER FIRST-SHOP PARTITION
recovers the team's world-conditioned play, and because the shop draw is
world-decided (not player-decided), the branches share their pre-unlock
prefix by construction -- dispatch happens once, at the first unlock, with
no mid-game switching risk.

Shop sequences aren't in the tapes; they are re-derived by exact
re-simulation (both seats' tapes + the recorded seed replay the world
bit-for-bit -- the ladder-parity property).

    python src/trackp/shop_branches.py --team Kaileh57
"""
from kaggriculture.paths import ROOT
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(ROOT, "models", "trackp", "shop_branches.json")


def episode_shops(job):
    """(route_id, first_shop, full_sequence) via exact re-simulation."""
    rid, other_id, seed, steps_cap = job
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggriculture.trackp import routes_io as R
    from kaggle_environments import make
    mine = R.load_route(rid)
    theirs = R.load_route(other_id)
    seat = int(rid.rsplit("_s", 1)[1])
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": int(seed)},
               info={"seed": int(seed)})
    env.reset(2)
    seq = []
    for step in range(min(steps_cap, len(mine), len(theirs))):
        if env.done:
            break
        a = mine[step] if isinstance(mine[step], dict) else {}
        b = theirs[step] if isinstance(theirs[step], dict) else {}
        pair = [a, b] if seat == 0 else [b, a]
        env.step(pair)
        town = env.state[0].observation.get("town") or {}
        shops = list(town.get("unlocked_shops") or [])
        if len(shops) > len(seq):
            seq = shops
        if len(seq) >= 3 and step > 300:
            break
    return rid, (seq[0] if seq else None), seq


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="Kaileh57")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--steps-cap", type=int, default=400)
    args = ap.parse_args()

    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    recs = [r for r in idx["routes"].values()
            if r.get("team") == args.team and r.get("engine") == "1.32.7"
            and r.get("source") not in ("majority-vote", "trackp-factory")]
    jobs = []
    for r in recs:
        other = f"{r.get('episode')}_s{1 - r.get('seat', 0)}"
        if other in idx["routes"] and r.get("seed") is not None:
            jobs.append((r["id"], other, r["seed"], args.steps_cap))
    print(f"{args.team}: {len(jobs)} re-simulatable episodes", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for rid, first, seq in ex.map(episode_shops, jobs):
            results[rid] = {"first": first, "seq": seq}
            if len(results) % 20 == 0:
                print(f"  {len(results)}/{len(jobs)}", flush=True)

    parts = collections.Counter(v["first"] for v in results.values())
    print("first-shop partition sizes:", dict(parts))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"team": args.team, "routes": results},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
