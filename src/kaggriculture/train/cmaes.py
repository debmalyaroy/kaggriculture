"""CMA-ES parameter search — the right optimiser for this problem.

Why this and not the coordinate descent in tune.py:

* The agent is a **parameterised policy** with ~45 continuous knobs. That is a
  textbook low-dimensional, expensive, noisy black-box optimisation.
* Coordinate descent moves one knob at a time, so it cannot find optima that
  require two knobs to move **together** — and this project has already hit that
  wall: `travel_weight` only pays off alongside the capacity model, and the
  capacity knobs measured flat when moved alone *after* travel_weight changed.
* CMA-ES adapts a full covariance matrix, so it learns those correlations.

Variance control matters more than the optimiser here. Every candidate in a
generation is evaluated on the **same seed set** (common random numbers) and in
**both seats**, so comparisons are paired and seed noise largely cancels.

Runs are **checkpointed after every generation** to `models/cmaes/state.json`,
so it survives being killed, and resumes with `--resume`. That also makes it
safe to drive from a scheduler.

    python -m kaggriculture.train.cmaes --generations 20 --popsize 8 --seeds 4
    python -m kaggriculture.train.cmaes --resume
    python -m kaggriculture.train.cmaes --resume --generations 5     # a few more
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.pipeline.params as paramio  # noqa: E402

WORK = os.path.join(ROOT, "models", "cmaes")

# knob -> (low, high, is_int). Only knobs that are safe to vary continuously.
SPACE = {
    "travel_weight":       (1.0, 9.0, False),
    "poach_penalty":       (0.2, 1.0, False),
    "capacity_util":       (0.65, 1.05, False),
    "cost_per_crop_day":   (1.4, 3.2, False),
    "cost_per_animal_day": (3.0, 6.5, False),
    "hands_max":           (6, 16, True),
    "hire_cash_frac":      (0.10, 0.55, False),
    "reserve_base":        (120, 800, True),
    "reserve_per_tile":    (0.0, 18.0, False),
    "feed_runway_days":    (3.0, 15.0, False),
    "max_herd":            (6, 30, True),
    "animal_cash_buffer":  (100, 1500, True),
    "target_cow":          (0, 24, True),
    "target_sheep":        (0, 22, True),
    "target_goose":        (0, 20, True),
    "target_melon":        (0, 22, True),
    "target_strawberry":   (0, 34, True),
    "target_wheat":        (0, 22, True),
    "feed_self_frac":      (0.2, 1.4, False),
    "liquid_weight":       (0.0, 0.20, False),
    "feed_safe_discount":  (0.05, 0.50, False),
    "care_weight":         (0.2, 1.8, False),
    "fert_weight":         (0.0, 4.5, False),
    "sell_chunk":          (3, 18, True),
    "shed_pressure":       (55, 95, True),
    "feed_buffer_days":    (1.5, 5.5, False),
    "land_min_used":       (0.25, 0.95, False),
    "seed_lookahead":      (2, 18, True),
}
KEYS = sorted(SPACE)


def to_vec(P):
    v = []
    for k in KEYS:
        lo, hi, _ = SPACE[k]
        x = float(P.get(k, (lo + hi) / 2))
        v.append((min(max(x, lo), hi) - lo) / (hi - lo))     # normalise to [0,1]
    return v


def to_params(base, v):
    P = dict(base)
    for k, x in zip(KEYS, v):
        lo, hi, is_int = SPACE[k]
        # float() matters: x may be a numpy scalar, and pprint renders those as
        # "np.float64(0.345)" -- which is a NameError inside the generated agent.
        val = float(lo) + min(max(float(x), 0.0), 1.0) * (float(hi) - float(lo))
        P[k] = int(round(val)) if is_int else float(round(val, 4))
    return P


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


def fitness(cand_paths, opponents, seeds, workers):
    """Win rate against the opponent pool, both seats, common random numbers."""
    jobs, meta = [], []
    for ci, c in enumerate(cand_paths):
        for opp in opponents:
            for s in seeds:
                jobs.append((c, opp, s)); meta.append((ci, 0))
                jobs.append((opp, c, s)); meta.append((ci, 1))
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as ex:
        res = list(ex.map(_play, jobs))
    wins = [0.0] * len(cand_paths)
    games = [0] * len(cand_paths)
    banks = [[] for _ in cand_paths]
    for (ci, seat), (r0, r1) in zip(meta, res):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        games[ci] += 1
        banks[ci].append(mine)
        wins[ci] += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return [(wins[i] / max(1, games[i]), statistics.mean(banks[i]) if banks[i] else 0.0)
            for i in range(len(cand_paths))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=None, help="default: newest agents/agent_v*.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "agent_cma.py"))
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=77000)
    ap.add_argument("--sigma", type=float, default=0.22)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import numpy as np
    os.makedirs(WORK, exist_ok=True)
    state_path = os.path.join(WORK, "state.json")

    base_agent = args.base
    if not base_agent:
        import glob
        c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        base_agent = c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")
    base = paramio.load(base_agent)
    src = os.path.join(ROOT, "agents", "v1_heuristic.py")

    # Opponent pool: a small league, not one frozen copy. Tuning against a
    # single opponent over-fits to it -- the ladder is varied.
    pool = [p for p in (base_agent,
                        os.path.join(ROOT, "agents", "v2_tuned.py"),
                        os.path.join(ROOT, "agents", "v1_heuristic.py"))
            if os.path.exists(p)]
    frozen = []
    for i, p in enumerate(pool):
        f = os.path.join(WORK, f"pool{i}.py")
        if not os.path.exists(f):
            import shutil
            shutil.copy(p, f)
        frozen.append(f)

    n = len(KEYS)
    if args.resume and os.path.exists(state_path):
        st = json.load(open(state_path))
        mean = np.array(st["mean"]); sigma = st["sigma"]
        C = np.array(st["C"]); gen0 = st["gen"]
        best = (st["best_fit"], np.array(st["best_x"]))
        print(f"resumed at generation {gen0}, best win rate {best[0]*100:.1f}%")
    else:
        mean = np.array(to_vec(base)); sigma = args.sigma
        C = np.eye(n); gen0 = 0; best = (-1.0, mean.copy())

    lam = args.popsize
    mu = max(1, lam // 2)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    seeds = [args.seed0 + i for i in range(args.seeds)]
    rng = np.random.default_rng(12345 + gen0)

    for g in range(gen0, gen0 + args.generations):
        t0 = time.time()
        try:
            L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            C = np.eye(n); L = np.eye(n)
        X = [np.clip(mean + sigma * (L @ rng.standard_normal(n)), 0, 1)
             for _ in range(lam)]
        paths = []
        for i, x in enumerate(X):
            p = os.path.join(WORK, f"gen{g}_c{i}.py")
            paramio.write(src, p, to_params(base, x), header=f"cmaes gen{g} cand{i}",
                          module_doc=f"CMA-ES candidate g{g} c{i}\n")
            paths.append(p)
        scored = fitness(paths, frozen, seeds, args.workers)
        order = sorted(range(lam), key=lambda i: (-scored[i][0], -scored[i][1]))
        mean = sum(w[k] * X[order[k]] for k in range(mu))
        # rank-mu covariance update (simplified CMA)
        Y = np.array([X[order[k]] - mean for k in range(mu)])
        C = 0.75 * C + 0.25 * (Y.T @ (w[:mu, None] * Y)) / (sigma ** 2 + 1e-12)
        sigma *= 0.93 if scored[order[0]][0] <= best[0] else 1.05
        sigma = float(np.clip(sigma, 0.03, 0.5))
        if scored[order[0]][0] > best[0]:
            best = (scored[order[0]][0], X[order[0]].copy())
        print(f"gen {g:>3}  best {scored[order[0]][0]*100:>5.1f}%  "
              f"bank ${scored[order[0]][1]:>9,.0f}  sigma {sigma:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump({"mean": mean.tolist(), "sigma": sigma, "C": C.tolist(),
                   "gen": g + 1, "best_fit": best[0], "best_x": best[1].tolist(),
                   "keys": KEYS}, open(state_path, "w"))
        paramio.write(src, args.out, to_params(base, best[1]),
                      header=f"CMA-ES best after generation {g} "
                             f"({best[0]*100:.1f}% vs the pool) -- do not hand-edit.",
                      module_doc="Kaggriculture agent — CMA-ES tuned.\n\n"
                                 "GENERATED. Regenerate with src/kaggriculture/train/cmaes.py.\n")
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"\nbest win rate vs pool: {best[0]*100:.1f}%\nwrote {args.out}")
    print("VALIDATE before believing it:")
    print(f"  python -m kaggriculture.measure.evaluate {os.path.relpath(args.out, ROOT)} "
          f"--vs {os.path.relpath(base_agent, ROOT)} -n 3 --seed0 91000")
    print("  ...then repeat with two more --seed0 values (three sets, always).")


if __name__ == "__main__":
    main()
