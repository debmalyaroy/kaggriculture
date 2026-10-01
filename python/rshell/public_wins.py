"""Replay a seeded sample of v63.5_rl's public-25 WINS into tapes (the losses are replayed by loss_rca.py), so the
shell's open-loop public set is not made of losses only.

    PYTHONPATH=~/kaggriculture/src KAGG_BIN=... python python/rshell/public_wins.py --cand DIR/main.py [--n 600] [--workers 12]

Tapes: data/rshell/public25/wins/tapes/<opp>__<seed>_<seat>.json (`seat` = OUR seat), stratified over opponents.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_rca as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--field", default=os.path.join(R.RL, ".local", "ext", "field25"))
    ap.add_argument("--out", default=os.path.join(R.RL, "data", "rshell", "public25", "wins"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tapes"), exist_ok=True)
    games = [json.loads(l) for l in open(os.path.join(R.RL, "data", "rshell", "public25", "games.jsonl"))]
    by = defaultdict(list)
    for g in games:
        if g.get("gap", 0) > 0:
            by[g["opp"]].append(g)
    rng = random.Random(27)
    per = max(1, a.n // max(1, len(by)))
    pick = [g for o in sorted(by) for g in rng.sample(by[o], min(per, len(by[o])))]
    tasks = [(os.path.abspath(a.cand), g["opp"], os.path.join(a.field, g["opp"] + ".py"), g["seed"], g["seat"], a.out) for g in pick
             if not os.path.exists(os.path.join(a.out, "tapes", f"{g['opp']}__{g['seed']}_{g['seat']}.json"))]
    print(f"[wins] {len(pick)} wins sampled over {len(by)} opponents; {len(tasks)} to replay", flush=True)
    ok = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(R.replay, t) for t in tasks]), 1):
            ok += "error" not in f.result()
            if i % 100 == 0:
                print(f"[wins] {i}/{len(tasks)}", flush=True)
    print(f"[wins] {ok} tapes written -> {a.out}/tapes", flush=True)


if __name__ == "__main__":
    main()
