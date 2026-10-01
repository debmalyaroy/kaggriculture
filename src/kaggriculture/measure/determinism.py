"""Determinism pre-check for A/B drivers -- the last unguarded bug class.

The bug this exists to prevent, and I committed it myself on 2026-08-13: an A/B
of a harness change was run with the built-in `random` agents, which are NOT
seeded. Two runs of the same seed give different banks, so the "difference" the
test reported was noise about noise. The result looked like a real divergence
and was meaningless until re-run with deterministic agents.

Any paired comparison must therefore establish, BEFORE spending compute, that
each side is reproducible. A non-deterministic agent does not make an A/B
harder; it makes it invalid, because the paired-seed variance reduction the
whole design relies on evaporates.

    import kaggriculture.measure.determinism as determinism
    determinism.require(["agents/v24.1_bandit.py", "starter"], seeds=(11,))

    python src/determinism.py agents/v24.1_bandit.py starter
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The built-in agents known to draw from an unseeded RNG. `starter` and `pass`
# are deterministic; `random` is not (measured: two runs of seed 21 gave
# (1820, 1920) then (1750, 1400)).
KNOWN_NONDETERMINISTIC = {"random"}


def _play(left, right, seed, steps=240):
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed,
        "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    f = env.steps[-1]
    return (round(float(f[0]["reward"] or 0), 6),
            round(float(f[1]["reward"] or 0), 6))


def check(agents, seeds=(11,), steps=240, opponent="starter", verbose=True):
    """[(agent, deterministic, detail)] -- each agent played twice per seed."""
    out = []
    for a in agents:
        name = os.path.basename(str(a))
        if str(a) in KNOWN_NONDETERMINISTIC:
            out.append((a, False, "known unseeded built-in"))
            if verbose:
                print(f"  {name:<30} NON-DETERMINISTIC (known)")
            continue
        detail, ok = "", True
        for s in seeds:
            left = a if os.path.exists(str(a)) or "/" not in str(a) else a
            r1 = _play(left, opponent, s, steps)
            r2 = _play(left, opponent, s, steps)
            if r1 != r2:
                ok = False
                detail = f"seed {s}: {r1} then {r2}"
                break
        out.append((a, ok, detail or "reproducible"))
        if verbose:
            print(f"  {name:<30} "
                  f"{'deterministic' if ok else 'NON-DETERMINISTIC ' + detail}")
    return out


def require(agents, seeds=(11,), steps=240, opponent="starter"):
    """Raise unless every agent is reproducible. Call this before any A/B.

    Deliberately fatal rather than a warning: a paired test on a
    non-deterministic side produces a number that reads exactly like a result.
    """
    rows = check(agents, seeds=seeds, steps=steps, opponent=opponent,
                 verbose=False)
    bad = [(a, d) for a, ok, d in rows if not ok]
    if bad:
        lines = "\n".join(f"  {a}: {d}" for a, d in bad)
        raise SystemExit(
            "DETERMINISM CHECK FAILED -- these agents do not reproduce on a "
            f"fixed seed, so a paired A/B over them is invalid:\n{lines}\n"
            "Use deterministic agents (route/bandit builds, tapes, 'starter', "
            "'pass'); never the unseeded 'random' built-in.")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11])
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--opponent", default="starter")
    args = ap.parse_args()
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    print(f"determinism check ({len(args.seeds)} seed(s), "
          f"{args.steps} steps, vs {args.opponent}):")
    rows = check(args.agents, seeds=tuple(args.seeds), steps=args.steps,
                 opponent=args.opponent)
    return 0 if all(ok for _, ok, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
