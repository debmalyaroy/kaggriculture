"""Evaluate an agent against a set of opponents over many seeded matches.

Plays every (agent, opponent, seed) pair in BOTH seat orders so seat bias
cancels out. Runs matches in a process pool.

Usage:
    python -m kaggriculture.measure.evaluate agents/v1_heuristic.py --vs random starter agents/v0_baseline.py -n 6
"""
import argparse
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)

BUILTINS = {"pass", "random", "starter"}


def resolve(spec):
    return spec if spec in BUILTINS else os.path.abspath(spec)


def play(job):
    """job = (left, right, seed, steps) -> (left_money, right_money)"""
    left, right, seed, steps = job
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000},
    )
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def main():
    # The ladder's engine, or nothing this prints means anything.
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--vs", nargs="+", default=["random", "starter"])
    ap.add_argument("-n", "--matches", type=int, default=4, help="seeds per opponent (x2 seats)")
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--no-record", action="store_true",
                    help="do not write the result into the model registry")
    args = ap.parse_args()

    me = resolve(args.agent)
    jobs, meta = [], []
    for opp in args.vs:
        o = resolve(opp)
        for k in range(args.matches):
            seed = args.seed0 + k
            jobs.append((me, o, seed, args.steps))
            meta.append((opp, seed, 0))
            jobs.append((o, me, seed, args.steps))
            meta.append((opp, seed, 1))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(play, jobs))

    import kaggriculture.data.sink as sink
    by_opp = {}
    for (opp, seed, my_seat), (r0, r1) in zip(meta, results):
        mine, theirs = (r0, r1) if my_seat == 0 else (r1, r0)
        by_opp.setdefault(opp, []).append((mine, theirs))
        sink.record(source="evaluate", agent=args.agent, opponent=opp,
                    seed=seed, seat=my_seat, bank=mine, opp_bank=theirs)

    import kaggriculture.measure.win_metric as WM
    print(f"agent: {args.agent}   ({args.matches} seeds x 2 seats per opponent)")
    # SCORE is the headline: the ladder pays win/draw/loss, so $1 and $10,000
    # of margin are the same result. Margin stays on the row as a diagnostic
    # (it is lower-variance and useful for spotting *why*), never as the
    # criterion. See src/win_metric.py.
    print(f"{'opponent':<28}{'score':>7}{'W-D-L':>10}{'my mean':>12}"
          f"{'opp mean':>12}{'margin~':>12}")
    all_mine = []
    all_scores = []
    measured = []
    for opp, rows in by_opp.items():
        w, d, l, sc = WM.summarise(rows)
        mine = [m for m, _ in rows]
        theirs = [t for _, t in rows]
        all_mine += mine
        all_scores += [WM.score(m, t) for m, t in rows]
        print(
            f"{opp:<28}{sc:>7.2f}{f'{w}-{d}-{l}':>10}"
            f"{statistics.mean(mine):>12,.0f}{statistics.mean(theirs):>12,.0f}"
            f"{statistics.mean(mine) - statistics.mean(theirs):>12,.0f}"
        )
        measured.append({
            "opponent": opp,
            "win_rate": sc,          # now expected SCORE (draws = 0.5)
            "margin": statistics.mean(mine) - statistics.mean(theirs),
            "bank": statistics.mean(mine),
        })
        # STABLE MACHINE-READABLE LINE. Parsers must read this, never the
        # human table above: three call sites in refresh_cycle scraped the
        # old `win%` column, so changing a display format silently made every
        # tournament cell return 0/0 and would have wrecked the crown gate.
        # Fields: RESULT opp score wins draws losses games bank opp_bank margin
        print(f"RESULT\t{opp}\t{sc:.4f}\t{w}\t{d}\t{l}\t{len(rows)}\t"
              f"{statistics.mean(mine):.1f}\t{statistics.mean(theirs):.1f}\t"
              f"{statistics.mean(mine) - statistics.mean(theirs):.1f}")
    print(f"\nOVERALL EXPECTED SCORE: {WM.expected_score(all_scores):.3f}"
          f"   (mean bank ${statistics.mean(all_mine):,.0f}, diagnostic)")

    # Persist what we just measured. --no-record has advertised this since the
    # flag was added, but nothing ever wrote the row, which is why only 6 of 32
    # models carried any per-opponent result at all -- and those 6 were typed
    # in by hand. A model card that says "registered" and shows no measurement
    # is how three contaminated A/B tests got through.
    if not args.no_record:
        try:
            import kaggriculture.data.registry as registry
            name = os.path.basename(me)
            for row in measured:
                registry.record_eval(
                    name, row["opponent"], row["win_rate"], row["margin"],
                    row["bank"], seeds=args.matches,
                    matches=args.matches * 2,
                    note=f"evaluate.py seed0={args.seed0}")
            print(f"recorded {len(measured)} result(s) against {name} "
                  f"in the model registry")
        except Exception as exc:                               # noqa: BLE001
            # Never let bookkeeping lose a measurement that just cost minutes.
            print(f"(registry not updated: {exc})")


if __name__ == "__main__":
    main()
