"""Generate episode data offline, without Kaggle.

Real ladder replays need the Kaggle API and a networked machine. This produces
the same feature rows from locally played matches instead, so the EDA and the
learning pipeline can be exercised (and can find real things) before any
download happens.

Diversity matters more than volume here: a dataset where every player uses the
same PARAMS has no variance to explain. So each match pairs randomly perturbed
configurations, which gives the analysis something to separate.

    python -m kaggriculture.data.gen_local_dataset --matches 8
    python -m kaggriculture.data.gen_local_dataset --matches 8 --append      # keep going
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.data.build_dataset as bd  # noqa: E402

# Knobs worth varying, with plausible ranges. These are the axes the EDA can
# then attribute outcomes to.
JITTER = {
    "travel_weight": (1.0, 8.0), "max_herd": (6, 24), "hands_max": (6, 16),
    "target_cow": (2, 20), "target_sheep": (2, 18), "target_goose": (0, 16),
    "target_melon": (0, 20), "target_strawberry": (4, 32), "target_wheat": (2, 20),
    "fert_weight": (0.0, 4.0), "capacity_util": (0.7, 1.0),
    "cost_per_crop_day": (1.6, 3.0), "cost_per_animal_day": (3.0, 6.0),
    "feed_runway_days": (4.0, 14.0), "land_min_used": (0.3, 0.9),
}


def perturb(base, rng):
    p = dict(base)
    for k, (lo, hi) in JITTER.items():
        if k not in p:
            continue
        p[k] = rng.randint(int(lo), int(hi)) if isinstance(p[k], int) \
            else round(rng.uniform(lo, hi), 3)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--base", default=None, help="default: newest agents/agent_v*.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "episodes_local.csv"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--work", default=os.path.join(ROOT, ".local", "gen"))
    args = ap.parse_args()

    base_agent = args.base
    if not base_agent:
        import glob
        c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        base_agent = c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")
    base = paramio.load(base_agent)
    src = os.path.join(ROOT, "agents", "v1_heuristic.py")
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    seed0 = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = random.Random(seed0)

    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=bd.feature_names())
    if new:
        w.writeheader()

    from kaggle_environments import make
    n = 0
    for m in range(args.matches):
        a = os.path.join(args.work, f"a_{seed0}_{m}.py")
        b = os.path.join(args.work, f"b_{seed0}_{m}.py")
        paramio.write(src, a, perturb(base, rng), header="jittered A",
                      module_doc="local dataset agent A\n")
        paramio.write(src, b, perturb(base, rng), header="jittered B",
                      module_doc="local dataset agent B\n")
        env = make("kaggriculture",
                   configuration={"episodeSteps": 720, "seed": seed0 * 7 + m,
                                  "actTimeout": 60, "runTimeout": 100000})
        env.run([a, b])
        j = env.toJSON()
        j.setdefault("info", {})["EpisodeId"] = seed0 * 1000 + m
        j["info"]["TeamNames"] = ["jitterA", "jitterB"]
        path = os.path.join(tempfile.gettempdir(), f"gen_{seed0}_{m}.json")
        with open(path, "w") as f:
            json.dump(j, f)
        try:
            for row in bd.rows_from_replay(path, "local"):
                w.writerow(row)
                n += 1
            fh.flush()
        finally:
            for pth in (path, a, b):
                try:
                    os.remove(pth)
                except OSError:
                    pass
        print(f"  match {m+1}/{args.matches}  rewards {j.get('rewards')}", flush=True)
    fh.close()
    print(f"wrote {n} rows to {args.out}")
    print("next:  python -m kaggriculture.pipeline.eda_report --data " + os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
