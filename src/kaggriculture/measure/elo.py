"""Local Elo ladder for our own agents.

The competition ranks by skill rating from head-to-head results, and the public
meta write-up flags the matching failure mode outright: *"Optimize only mean
bank vs starter -> high local bank, mediocre Elo."* Mean bank is the wrong
scoreboard. This keeps a real one.

Every agent in `agents/` gets a rating. Matches are round-robin, **both seats**
per seed (shared-market games are not symmetric — a one-seat test can reverse
the apparent winner), and ratings update with standard Elo. Results accumulate
in `models/elo/ladder.json`, so each iteration's agent is rated against every
previous one and the ladder is the project's running scoreboard.

    python -m kaggriculture.measure.elo --rounds 2                 # play and rate everything
    python -m kaggriculture.measure.elo --show                     # current table, no matches
    python -m kaggriculture.measure.elo --only agents/ml_rl.py     # rate one newcomer
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

WORK = os.path.join(ROOT, "models", "elo")
LADDER = os.path.join(WORK, "ladder.json")
START = 600.0          # Kaggle initialises submissions at 600
K = 24.0
BUILTINS = ["starter", "random"]


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


def load():
    if os.path.exists(LADDER):
        return json.load(open(LADDER))
    return {"rating": {}, "games": {}, "wins": {}, "history": []}


def save(st):
    os.makedirs(WORK, exist_ok=True)
    tmp = LADDER + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, LADDER)


def expected(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def show(st):
    rows = sorted(st["rating"].items(), key=lambda kv: -kv[1])
    print(f"\n{'agent':<38}{'Elo':>8}{'games':>8}{'win%':>8}")
    print("-" * 62)
    for name, r in rows:
        g = st["games"].get(name, 0)
        w = st["wins"].get(name, 0.0)
        print(f"{name:<38}{r:>8.0f}{g:>8}{(100.0*w/g if g else 0):>7.0f}%")
    print("-" * 62)
    if st["history"]:
        h = st["history"][-1]
        print(f"last run: {h['at']} — {h['matches']} matches")


def main():
    # The ladder's engine, or nothing this prints means anything.
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=1, help="seeds per pairing")
    ap.add_argument("--seed0", type=int, default=61000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--only", default=None,
                    help="rate just this agent against the existing field")
    ap.add_argument("--include-builtins", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    st = load()
    if args.show:
        show(st)
        return

    field = sorted(glob.glob(os.path.join(ROOT, "agents", "*.py")))
    field = [f for f in field if "__" not in os.path.basename(f)]
    names = {f: os.path.basename(f) for f in field}
    if args.include_builtins:
        for b in BUILTINS:
            names[b] = b
    for n in names.values():
        st["rating"].setdefault(n, START)
        st["games"].setdefault(n, 0)
        st["wins"].setdefault(n, 0.0)

    entrants = list(names)
    if args.only:
        target = os.path.abspath(os.path.join(ROOT, args.only))
        if target not in names:
            names[target] = os.path.basename(target)
            st["rating"].setdefault(names[target], START)
            st["games"].setdefault(names[target], 0)
            st["wins"].setdefault(names[target], 0.0)
        pairs = [(target, o) for o in entrants if o != target]
    else:
        pairs = list(itertools.combinations(entrants, 2))

    jobs, meta = [], []
    for a, b in pairs:
        for r in range(args.rounds):
            s = args.seed0 + r
            jobs.append((a, b, s)); meta.append((a, b))
            jobs.append((b, a, s)); meta.append((b, a))
    if not jobs:
        print("nothing to play")
        show(st)
        return

    print(f"{len(jobs)} matches across {len(pairs)} pairings "
          f"({args.rounds} seed(s) x both seats)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_mute) as ex:
        res = list(ex.map(_play, jobs))

    for (a, b), (ra, rb) in zip(meta, res):
        na, nb = names[a], names[b]
        sa = 1.0 if ra > rb else (0.5 if ra == rb else 0.0)
        ea = expected(st["rating"][na], st["rating"][nb])
        st["rating"][na] += K * (sa - ea)
        st["rating"][nb] += K * ((1 - sa) - (1 - ea))
        st["games"][na] += 1; st["games"][nb] += 1
        st["wins"][na] += sa; st["wins"][nb] += 1 - sa

    st["history"].append({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "matches": len(jobs),
                          "table": {k: round(v, 1) for k, v in st["rating"].items()}})
    save(st)
    print(f"played in {time.time()-t0:.0f}s")
    show(st)
    print("\nElo is relative to this field only — it is not the Kaggle rating.")
    print("Its job is to stop 'high bank vs starter, mediocre ladder' regressions.")


if __name__ == "__main__":
    main()
