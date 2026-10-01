"""Derive and optimise the agent's SELL_SCHEDULE.

The top of this ladder does not out-farm the field, it out-*sells* it. Their
market plans are recorded schedules refined by screening a dozen traces of the
current leader; a closed-loop agent cannot replay a schedule, but it can carry
one as a bias.

`SELL_SCHEDULE` maps `"<PRODUCT>|<day bucket>"` to a multiplier applied to two
things at once: how much of that product we release in a turn, and how hard it
competes for one of the two SELL slots the engine actually processes. 1.0
everywhere is exactly the unscheduled agent, which is what makes this safe to
search -- the identity element is the incumbent.

Three ways to fill it:

    python -m kaggriculture.pipeline.schedule --derive          # from the mined field schedule
    python -m kaggriculture.pipeline.schedule --optimise --generations 10
    python -m kaggriculture.pipeline.schedule --show

`--derive` reads `data/market_schedule.csv` (produced by src/kaggriculture/data/mine_top.py) and
turns the field's units-sold-per-day into multipliers around 1.0. That is a
prior, not an answer: the field's schedule suits the field's farm. `--optimise`
then runs CEM on the table against live opponents, which is the part that
actually has to earn its keep.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import json
import os
import pprint
import random
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402

BEGIN = "# --- SCHEDULE BEGIN"
END = "# --- SCHEDULE END ---"
MINED = os.path.join(ROOT, "data", "market_schedule.csv")
WORK = os.path.join(ROOT, ".local", "sched")
STATE = os.path.join(WORK, "cem.json")

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER"]
DAY_EDGES = (5, 10, 15, 20, 25)
N_BUCKETS = len(DAY_EDGES) + 1
LO, HI = 0.25, 3.0


def bucket(day):
    for i, edge in enumerate(DAY_EDGES):
        if day < edge:
            return i
    return N_BUCKETS - 1


# ------------------------------------------------------------------- file io --

def read_schedule(path):
    src = open(path, encoding="utf-8").read()
    m = re.search(re.escape(BEGIN) + r".*?\nSELL_SCHEDULE = (\{.*?\})\n" + re.escape(END),
                  src, re.S)
    if not m:
        return {}
    import ast
    return ast.literal_eval(m.group(1))


def write_schedule(base_path, out_path, table, note=""):
    src = open(base_path, encoding="utf-8").read()
    i = src.index(BEGIN)
    j = src.index(END, i) + len(END)
    head = src[:i]
    tail = src[j:]
    body = pprint.pformat({k: round(float(v), 4) for k, v in sorted(table.items())},
                          width=78, sort_dicts=True)
    block = (f"{BEGIN} (src/kaggriculture/pipeline/schedule.py rewrites this block)\n"
             + (f"# {note}\n" if note else "")
             + f"SELL_SCHEDULE = {body}\n{END}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(head + block + tail)
    return out_path


# ------------------------------------------------------------------- derive --

def derive(verbose=True):
    """Turn the field's measured schedule into multipliers around 1.0."""
    if not os.path.exists(MINED):
        print(f"no mined schedule at {os.path.relpath(MINED, ROOT)} -- "
              f"run src/kaggriculture/data/mine_top.py first")
        return {}
    per = {}
    for row in csv.DictReader(open(MINED, encoding="utf-8")):
        item = row["product"]
        if item not in PRODUCTS:
            continue
        b = bucket(int(row["day"]))
        per[(item, b)] = per.get((item, b), 0.0) + float(row["units_per_episode"])

    table = {}
    for item in PRODUCTS:
        vals = [per.get((item, b), 0.0) for b in range(N_BUCKETS)]
        total = sum(vals)
        if total <= 0:
            continue
        mean = total / N_BUCKETS
        for b, v in enumerate(vals):
            # relative to this product's own average day, so a product the
            # field barely trades does not get scaled to nothing
            mult = max(LO, min(HI, v / mean if mean else 1.0))
            table[f"{item}|{b}"] = round(mult, 3)
    if verbose:
        show(table)
    return table


def show(table=None, path=None):
    table = table if table is not None else read_schedule(
        path or os.path.join(ROOT, "agents", "v1_heuristic.py"))
    if not table:
        print("schedule is empty (agent behaves exactly as unscheduled)")
        return
    edges = ["d0-4", "d5-9", "d10-14", "d15-19", "d20-24", "d25-29"]
    print(f"  {'product':<12}" + "".join(f"{e:>9}" for e in edges))
    for item in PRODUCTS:
        row = [table.get(f"{item}|{b}") for b in range(N_BUCKETS)]
        if not any(v is not None for v in row):
            continue
        print(f"  {item:<12}" + "".join(
            f"{(v if v is not None else 1.0):>9.2f}" for v in row))


# ----------------------------------------------------------------- optimise --

def _play(job):
    left, right, seed, steps = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def evaluate(paths, opponents, seeds, seed0, pool, steps=720):
    jobs, meta = [], []
    for ci, p in enumerate(paths):
        for opp in opponents:
            o = os.path.abspath(opp)
            for s in range(seeds):
                jobs.append((p, o, seed0 + s, steps))
                meta.append((ci, 0))
                jobs.append((o, p, seed0 + s, steps))
                meta.append((ci, 1))
    out = list(pool.map(_play, jobs))
    rows = {i: [] for i in range(len(paths))}
    for (ci, seat), (r0, r1) in zip(meta, out):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        rows[ci].append((mine, theirs))
    scored = []
    for ci, data in rows.items():
        wins = sum(1 for m, t in data if m > t)
        scored.append({
            "i": ci, "win": wins / max(1, len(data)),
            "margin": statistics.mean(m - t for m, t in data),
            "bank": statistics.mean(m for m, _ in data),
        })
    return scored


def optimise(base, out, opponents, generations, popsize, seeds, seed0, workers,
             sigma, elite, resume, rng_seed=5):
    os.makedirs(WORK, exist_ok=True)
    keys = [f"{item}|{b}" for item in PRODUCTS for b in range(N_BUCKETS)]
    state = None
    if resume and os.path.exists(STATE):
        state = json.load(open(STATE, encoding="utf-8"))
    if state is None:
        seed_table = read_schedule(base) or derive(verbose=False)
        mean = {k: float(seed_table.get(k, 1.0)) for k in keys}
        state = {"mean": mean, "sigma": {k: sigma for k in keys}, "gen": 0,
                 "best": None, "best_score": None, "history": []}

    rng = random.Random(rng_seed)
    base_params = paramio.load(base)
    print(f"CEM over {len(keys)} schedule cells, pop {popsize}, "
          f"{seeds} seeds x 2 seats x {len(opponents)} opponent(s) "
          f"= {popsize * seeds * 2 * len(opponents)} matches/generation")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in range(generations):
            cands = [dict(state["mean"])]
            for _i in range(popsize - 1):
                cands.append({k: max(LO, min(HI, rng.gauss(state["mean"][k],
                                                           state["sigma"][k])))
                              for k in keys})
            paths = []
            for i, table in enumerate(cands):
                p = os.path.join(WORK, f"cand_{state['gen']}_{i}.py")
                write_schedule(base, p, table, note="CEM candidate")
                paths.append(p)

            scored = evaluate(paths, opponents, seeds, seed0, pool)
            scored.sort(key=lambda r: -(r["win"] * 1e6 + r["margin"]))
            n_elite = max(2, int(round(elite * len(cands))))
            top_tables = [cands[r["i"]] for r in scored[:n_elite]]
            for k in keys:
                vals = [t[k] for t in top_tables]
                state["mean"][k] = statistics.mean(vals)
                spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                state["sigma"][k] = max(0.05, spread)

            top = scored[0]
            if state["best_score"] is None or \
                    (top["win"] * 1e6 + top["margin"]) > state["best_score"]:
                state["best_score"] = top["win"] * 1e6 + top["margin"]
                state["best"] = cands[top["i"]]
            state["history"].append({"gen": state["gen"], "win": top["win"],
                                     "margin": top["margin"], "bank": top["bank"]})
            state["gen"] += 1
            json.dump(state, open(STATE, "w", encoding="utf-8"), indent=1)
            print(f"gen {state['gen']:>3}  best win {top['win'] * 100:>5.1f}%  "
                  f"margin {top['margin']:>+10,.0f}  bank {top['bank']:>10,.0f}  "
                  f"({(time.time() - t0) / 60:.1f} min)")

    if state["best"]:
        write_schedule(base, out, state["best"],
                       note=f"optimised against {', '.join(os.path.basename(o) for o in opponents)}")
        print(f"\nbest schedule -> {os.path.relpath(out, ROOT)}")
        show(state["best"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "agents", "v9_cem.py"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "v11_schedule.py"))
    ap.add_argument("--vs", nargs="+", default=None)
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--optimise", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=6100)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--elite", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.show:
        show(path=args.base)
        return 0
    if args.derive:
        table = derive()
        if table:
            write_schedule(args.base, args.out, table,
                           note="derived from the field's measured schedule "
                                "(data/market_schedule.csv)")
            print(f"\nwrote {os.path.relpath(args.out, ROOT)}")
        return 0
    if args.optimise:
        if os.path.abspath(args.out) == os.path.abspath(args.base):
            raise SystemExit("--out must not be --base")
        opponents = args.vs or [os.path.join(ROOT, "agents", "v9_cem.py")]
        return optimise(args.base, args.out, opponents, args.generations,
                        args.popsize, args.seeds, args.seed0, args.workers,
                        args.sigma, args.elite, args.resume)
    show(path=args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
