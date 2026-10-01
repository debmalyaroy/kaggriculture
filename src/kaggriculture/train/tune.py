"""Self-play parameter search for the Kaggriculture agent (produces v2).

Method: coordinate descent (a.k.a. one-knob-at-a-time hill climbing) over the
agent's PARAMS dict, scored by self-play against a *frozen* copy of the current
best agent plus the v0 baseline.

Why coordinate descent rather than CMA-ES / Bayesian optimisation: each match
costs seconds, the knobs are largely separable (labour, cash, portfolio,
market), and the search must stay reproducible and interruptible. Every
evaluation uses the same fixed seed set and plays both seats, so the score is a
paired comparison and seat/seed noise mostly cancels.

Usage:
    python -m kaggriculture.train.tune --base agents/v1_heuristic.py --out agents/v2_tuned.py \
        --seeds 4 --passes 2
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

import kaggriculture.pipeline.params as paramio  # noqa: E402

V2_DOC = """Kaggriculture agent v2 -- self-play tuned.

GENERATED FILE -- do not hand-edit. This is agents/v1_heuristic.py with a
PARAMS block found by src/kaggriculture/train/tune.py (coordinate descent over 28 knobs, scored
by paired self-play against a frozen v1 and the v0 baseline, both seats, fixed
seeds). To change behaviour, edit agents/v1_heuristic.py and re-run:

    python -m kaggriculture.train.tune --seeds 4 --passes 2

Self-contained: submittable directly as main.py. See docs/history/agent-v2.md for the
tuning table and what the search found.
"""

# (name, [candidate values]) -- explored in this order.
SEARCH_SPACE = [
    ("hands_max",            [10, 13, 16]),
    ("hire_cash_frac",       [0.15, 0.30, 0.50]),
    ("capacity_util",        [0.70, 0.82, 0.95]),
    ("cost_per_crop_day",    [1.6, 2.2, 3.0]),
    ("cost_per_animal_day",  [3.5, 4.5, 6.0]),
    ("reserve_base",         [150, 260, 600]),
    ("reserve_per_tile",     [0.0, 9.0, 20.0]),
    ("feed_runway_days",     [3.0, 7.0, 12.0]),
    ("max_herd",             [12, 22, 34]),
    ("animal_cash_buffer",   [100, 400, 1200]),
    ("target_cow",           [8, 16, 26]),
    ("target_sheep",         [6, 12, 18]),
    ("target_goose",         [0, 6, 12]),
    ("target_melon",         [4, 12, 20]),
    ("target_strawberry",    [8, 16, 24]),
    ("target_tomato",        [0, 6]),
    ("filler_crop",          ["WHEAT", "CARROT"]),
    ("feed_self_frac",       [0.4, 0.85, 1.3]),
    ("liquid_weight",        [0.0, 0.06, 0.14]),
    ("liquid_target",        [1500, 2500, 5000]),
    ("feed_safe_discount",   [0.10, 0.20, 0.40]),
    ("care_weight",          [0.4, 0.85, 1.5]),
    ("sell_chunk",           [4, 8, 16]),
    ("dump_day",             [26, 28, 29]),
    ("shed_pressure",        [60, 76, 90]),
    ("feed_buffer_days",     [2.0, 3.0, 5.0]),
    ("land_min_used",        [0.35, 0.60, 0.85]),
    ("seed_lookahead",       [4, 8, 16]),
]


def _mute_stdin():
    """Keep pool workers off the parent's stdin (see src/kaggriculture/pipeline/pipeline.py)."""
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
    except OSError:
        pass


def play(job):
    left, right, seed = job
    from kaggle_environments import make

    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def score(candidate_path, opponents, seeds, workers):
    jobs, meta = [], []
    for opp in opponents:
        for seed in seeds:
            jobs.append((candidate_path, opp, seed))
            meta.append(0)
            jobs.append((opp, candidate_path, seed))
            meta.append(1)
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute_stdin) as ex:
        out = list(ex.map(play, jobs))
    margins, banks = [], []
    for seat, (r0, r1) in zip(meta, out):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        margins.append(mine - theirs)
        banks.append(mine)
    # The ladder scores wins only: "the actual coin difference in a match does
    # not affect the rating change - only the win, loss, or tie outcome matters"
    # (competition Overview). So win rate dominates; margin is a tie-break for
    # resolution, since win rate over a small seed set is very granular.
    wins = sum(1.0 if m > 0 else (0.5 if m == 0 else 0.0) for m in margins)
    win_rate = wins / max(1, len(margins))
    return 1000.0 * win_rate + 0.001 * statistics.mean(margins), statistics.mean(banks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "agents", "v1_heuristic.py"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "v2_tuned.py"))
    ap.add_argument("--work", default=os.path.join(ROOT, ".local", "tune"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=4000)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these knobs")
    ap.add_argument("--budget-min", type=float, default=None, help="stop after N minutes")
    args = ap.parse_args()

    if os.path.abspath(args.base) == os.path.abspath(args.out):
        sys.exit("--out must differ from --base: the tuner re-reads --base on "
                 "every trial, so writing over it corrupts the search")
    os.makedirs(args.work, exist_ok=True)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    best = paramio.load(args.base)
    frozen = os.path.join(args.work, "frozen.py")
    shutil.copy(args.base, frozen)
    opponents = [frozen, os.path.abspath(os.path.join(ROOT, "agents", "v0_baseline.py"))]

    cand_path = os.path.join(args.work, "cand.py")
    paramio.write(args.base, cand_path, best)
    t0 = time.time()
    best_score, best_bank = score(cand_path, opponents, seeds, args.workers)
    print(f"baseline score {best_score:,.0f}  bank ${best_bank:,.0f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    history = [{"knob": "<baseline>", "value": None, "score": best_score, "bank": best_bank}]
    space = SEARCH_SPACE if not args.only else [kv for kv in SEARCH_SPACE if kv[0] in args.only]

    for p in range(args.passes):
        print(f"\n=== pass {p + 1}/{args.passes} ===", flush=True)
        for knob, values in space:
            if args.budget_min and (time.time() - t0) / 60.0 > args.budget_min:
                print("budget exhausted", flush=True)
                p = args.passes
                break
            current = best.get(knob)
            for v in values:
                if v == current:
                    continue
                trial = dict(best)
                trial[knob] = v
                paramio.write(args.base, cand_path, trial)
                s, bank = score(cand_path, opponents, seeds, args.workers)
                flag = ""
                if s > best_score:
                    best, best_score, best_bank = trial, s, bank
                    flag = "  <-- accept"
                    current = v
                history.append({"knob": knob, "value": v, "score": s, "bank": bank})
                print(f"  {knob:<22} {str(v):>8}  score {s:>10,.0f}  "
                      f"bank ${bank:>9,.0f}{flag}", flush=True)
            paramio.write(args.base, args.out, best,
                          header="Auto-tuned by src/kaggriculture/train/tune.py -- do not hand-edit.",
                          module_doc=V2_DOC)
        with open(os.path.join(args.work, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    paramio.write(args.base, args.out, best,
                  header=("Kaggriculture agent v2 -- v1's planner with a self-play tuned\n"
                          "PARAMS block. Generated by src/kaggriculture/train/tune.py; see docs/history/agent-v2.md."),
                  module_doc=V2_DOC)
    print(f"\nbest score {best_score:,.0f}  bank ${best_bank:,.0f}\nwrote {args.out}")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
