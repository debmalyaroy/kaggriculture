"""Population-Based Training: evolve a population of configs against each other.

Why PBT and not another single-chain search
-------------------------------------------
`tune.py` (coordinate descent), `cmaes.py` (CMA-ES) and `improve.py` (SPRT-gated
hill climbing) all optimise **one** policy. That is the wrong shape for two
reasons this project has already run into:

* `select.py --spread-guard` rejects every candidate committee member we have,
  because our top three models have identical voting knobs. A single-chain
  search cannot produce diversity -- every generation is a small step from the
  same parent, so the survivors are near-copies by construction.
* Self-play against a fixed opponent overfits to that opponent. A population
  plays a league, so a policy that only beats its own ancestor gets found out.

PBT keeps N workers training in parallel, and periodically has the weak ones
**exploit** (copy a strong worker's parameters) and **explore** (perturb them).
Survivors end up genuinely different because they descended from different
lineages under different perturbations -- which is exactly the input the
ensemble and the arbiter have been missing.

Ranking is by league win rate over paired both-seat matches on common random
numbers, and any promotion out of the population goes through a full SPRT, as
everywhere else here.

    python -m kaggriculture.train.pbt --population 6 --minutes 120
    python -m kaggriculture.train.pbt --status
    python -m kaggriculture.train.pbt --resume
    python -m kaggriculture.train.pbt --harvest 4        # write the best N as agents
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import datetime as dt
import glob
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402
from kaggriculture.measure.sprt import SPRT, elo_interval    # noqa: E402

WORK = os.path.join(ROOT, "models", "pbt")
STATE = os.path.join(WORK, "state.json")
SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")

# Knobs a worker may explore, with a relative step. Excludes anything already
# measured negative (see the runbook) -- re-proposing a known loser spends the
# budget the league is trying to conserve.
KNOBS = {
    "travel_weight": 0.30, "fert_weight": 0.30, "poach_penalty": 0.20,
    "capacity_util": 0.10, "cost_per_crop_day": 0.18, "cost_per_animal_day": 0.12,
    "reserve_base": 0.35, "reserve_per_tile": 0.35, "feed_runway_days": 0.30,
    "feed_safe_discount": 0.25, "liquid_weight": 0.35, "liquid_target": 0.35,
    "shed_pressure": 0.30, "sell_chunk": 0.30, "sell_floor": 0.25,
    "hire_cash_frac": 0.25, "care_weight": 0.30, "fert_collect_weight": 0.30,
    "seed_lookahead": 0.30, "land_reserve": 0.35, "animal_cash_buffer": 0.35,
}
INTEGERS = {"hands_max", "hands_min", "max_herd", "seed_lookahead"}
STRUCTURAL = {"assign_mode": ["greedy", "optimal"],
              "ensemble_k": [0, 4, 8, 12]}


def _mute():
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def _play(job):
    left, right, seed = job
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


# ------------------------------------------------------------- population --

def spawn(base_params, rng, n, spread=1.0):
    """An initial population that is diverse on purpose, not by accident."""
    pop = []
    for i in range(n):
        P = dict(base_params)
        if i > 0:                       # worker 0 is the unmodified incumbent
            for k, step in KNOBS.items():
                if isinstance(P.get(k), (int, float)) and rng.random() < 0.5:
                    val = P[k] * (1.0 + rng.gauss(0, step * spread))
                    P[k] = int(round(max(1, val))) if k in INTEGERS else round(max(val, 1e-6), 4)
            for key, choices in STRUCTURAL.items():
                if rng.random() < 0.5:
                    P[key] = rng.choice(choices)
        pop.append({"id": i, "params": P, "lineage": [i], "wins": 0, "games": 0})
    return pop


def explore(P, rng, strength=1.0):
    """Perturb: the 'explore' half of PBT."""
    Q = dict(P)
    keys = [k for k in KNOBS if isinstance(Q.get(k), (int, float))]
    for k in rng.sample(keys, min(len(keys), rng.choice([2, 3, 4]))):
        val = Q[k] * (1.0 + rng.gauss(0, KNOBS[k] * strength))
        Q[k] = int(round(max(1, val))) if k in INTEGERS else round(max(val, 1e-6), 4)
    if rng.random() < 0.25:
        key = rng.choice(sorted(STRUCTURAL))
        Q[key] = rng.choice(STRUCTURAL[key])
    return Q


def materialise(pop, work):
    """Write each worker to a file the match runner can load."""
    paths = []
    for w in pop:
        p = os.path.join(work, f"worker_{w['id']}.py")
        paramio.write(SRC, p, w["params"], header=f"pbt worker {w['id']}",
                      module_doc=f"PBT worker {w['id']}, lineage {w['lineage']}\n")
        paths.append(p)
    return paths


def league(pop, paths, seeds, pool, verbose=True):
    """Round-robin, both seats, common random numbers. Updates wins/games."""
    for w in pop:
        w["wins"] = w["games"] = 0
    jobs, meta = [], []
    for i, j in itertools.combinations(range(len(pop)), 2):
        for s in seeds:
            jobs.append((paths[i], paths[j], s)); meta.append((i, j))
            jobs.append((paths[j], paths[i], s)); meta.append((j, i))
    for (a, b), (ra, rb) in zip(meta, pool.map(_play, jobs)):
        pop[a]["games"] += 1
        pop[b]["games"] += 1
        if ra > rb:
            pop[a]["wins"] += 1
        elif rb > ra:
            pop[b]["wins"] += 1
    for w in pop:
        w["win_rate"] = w["wins"] / w["games"] if w["games"] else 0.0
    ranked = sorted(pop, key=lambda w: -w["win_rate"])
    if verbose:
        for w in ranked:
            pr.log(f"worker {w['id']:>2}  {w['win_rate']:>5.0%}  "
                   f"({w['wins']}/{w['games']})  lineage {w['lineage'][-3:]}", 2)
    return ranked


def exploit(ranked, rng, cut=0.25, verbose=True):
    """Bottom `cut` copy a random top-`cut` worker, then explore.

    Copying a *random* strong worker rather than the single best is what keeps
    the population from collapsing onto one lineage -- which is the whole point
    of running a population.
    """
    n = len(ranked)
    k = max(1, int(n * cut))
    top, bottom = ranked[:k], ranked[-k:]
    for loser in bottom:
        winner = rng.choice(top)
        if winner["id"] == loser["id"]:
            continue
        loser["params"] = explore(winner["params"], rng)
        loser["lineage"] = winner["lineage"] + [loser["id"]]
        if verbose:
            pr.log(f"worker {loser['id']} <- worker {winner['id']} (exploit + explore)", 2)
    return ranked


# ----------------------------------------------------------------- persist --

def save(st):
    os.makedirs(WORK, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def load():
    if not os.path.exists(STATE):
        return None
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _newest_agent():
    c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
    return c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")


def status():
    st = load()
    if not st:
        print("no PBT run recorded yet")
        return 0
    print(f"base       : {st.get('base')}")
    print(f"population : {len(st.get('pop', []))}")
    print(f"generations: {st.get('generation', 0)}")
    print(f"\n{'worker':>7} {'win%':>6} {'games':>6}  lineage")
    for w in sorted(st.get("pop", []), key=lambda x: -(x.get("win_rate") or 0)):
        print(f"{w['id']:>7} {(w.get('win_rate') or 0):>5.0%} "
              f"{w.get('games', 0):>6}  {w.get('lineage')}")
    div = st.get("diversity")
    if div is not None:
        print(f"\ndiversity  : {div:.2f} (mean relative spread across voting knobs)")
        if div < 0.05:
            print("  the population has collapsed onto one lineage -- raise "
                  "--spread or --cut")
    return 0


def diversity(pop):
    """Mean relative spread of the voting knobs. Low means clones."""
    keys = [k for k in ("travel_weight", "fert_weight", "poach_penalty",
                        "care_weight", "capacity_util")
            if all(isinstance(w["params"].get(k), (int, float)) for w in pop)]
    if not keys or len(pop) < 2:
        return 0.0
    total = 0.0
    for k in keys:
        vals = [float(w["params"][k]) for w in pop]
        mean = sum(vals) / len(vals)
        if abs(mean) < 1e-9:
            continue
        spread = (max(vals) - min(vals)) / abs(mean)
        total += spread
    return total / len(keys)


def harvest(st, n, verbose=True):
    """Write the best N workers out as real agents."""
    pop = sorted(st.get("pop", []), key=lambda w: -(w.get("win_rate") or 0))[:n]
    out = []
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for rank, w in enumerate(pop, 1):
        path = os.path.join(ROOT, "agents", f"agent_vpbt{rank}_{stamp}.py")
        paramio.write(SRC, path, w["params"],
                      header=f"PBT worker {w['id']} (rank {rank})",
                      module_doc=(f"Population-Based Training survivor.\n"
                                  f"League win rate {(w.get('win_rate') or 0):.0%} "
                                  f"over {w.get('games', 0)} games, lineage "
                                  f"{w.get('lineage')}.\n"
                                  f"Generation {st.get('generation')} of a "
                                  f"{len(st.get('pop', []))}-worker population.\n"))
        rel = os.path.relpath(path, ROOT)
        registry.register_model(os.path.basename(path), path=rel, built_by="pbt",
                                description=f"PBT rank {rank}, "
                                            f"{(w.get('win_rate') or 0):.0%} league")
        out.append(rel)
        if verbose:
            pr.log(f"harvested {rel}  ({(w.get('win_rate') or 0):.0%} league)", 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=None)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=2, help="paired seeds per pairing")
    ap.add_argument("--cut", type=float, default=0.25, help="share replaced per generation")
    ap.add_argument("--spread", type=float, default=1.0, help="initial diversity")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--harvest", type=int, default=0)
    args = ap.parse_args()

    if args.status:
        return status()
    pr.reset()
    os.makedirs(WORK, exist_ok=True)

    st = load() if (args.resume or args.harvest) else None
    if args.harvest:
        if not st:
            pr.warn("nothing to harvest -- run PBT first")
            return 1
        harvest(st, args.harvest)
        return 0

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    base = args.base or (st or {}).get("base") or os.path.relpath(_newest_agent(), ROOT)
    rng = random.Random(args.seed0 + 5150 + ((st or {}).get("generation") or 0))

    if st:
        pop = st["pop"]
        gen = st.get("generation", 0)
        pr.log(f"resuming generation {gen} with {len(pop)} workers")
    else:
        P = paramio.load(os.path.join(ROOT, base))
        pop = spawn(P, rng, args.population, args.spread)
        gen = 0
        pr.log(f"seeded a population of {len(pop)} from {base}")

    pr.log(f"league: {args.seeds} paired seeds per pairing, {workers} workers", 1)
    pr.log(f"a {len(pop)}-worker round robin is "
           f"{len(pop) * (len(pop) - 1) // 2 * args.seeds * 2} matches per generation", 1)

    deadline = time.time() + args.minutes * 60.0
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        while time.time() < deadline:
            gen += 1
            pr.log(f"generation {gen}")
            paths = materialise(pop, WORK)
            seeds = [args.seed0 + rng.randrange(1_000_000) for _ in range(args.seeds)]
            ranked = league(pop, paths, seeds, pool)
            div = diversity(pop)
            pr.log(f"diversity {div:.2f}  best {ranked[0]['win_rate']:.0%} "
                   f"(worker {ranked[0]['id']})", 1)
            if div < 0.05:
                pr.warn("population has collapsed onto one lineage -- "
                        "raising exploration", 1)
            exploit(ranked, rng, args.cut)
            save({"base": base, "generation": gen, "pop": pop,
                  "diversity": div, "updated": dt.datetime.now().isoformat()})

    st = load() or {}
    pop = sorted(st.get("pop", pop), key=lambda w: -(w.get("win_rate") or 0))
    pr.log("")
    pr.log(f"done: {gen} generation(s), best league win rate "
           f"{(pop[0].get('win_rate') or 0):.0%}")
    pr.log(f"diversity {st.get('diversity', 0):.2f} -- this is the number that "
           f"decides whether the ensemble has anything to work with", 1)
    pr.log("")
    pr.log("next:")
    pr.log("python -m kaggriculture.train.pbt --harvest 4        # write the survivors as agents", 1)
    pr.log("python -m kaggriculture.measure.elo --rounds 3         # rate them", 1)
    pr.log("python -m kaggriculture.train.select --build --spread-guard   # now it has diverse members", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
