"""How long does a complete pipeline run take, and where does the time go?

Built from MEASURED stage costs rather than guesses, and it re-measures the two
things that actually move (engine throughput and the feature cache) so the
answer tracks reality instead of a stale comment.

Everything engine-bound is priced from one measured episode time and the
concurrency cap, because that product is the whole story: at 3 workers the
tournament is the pipeline, and at 12 it is a rounding error next to the fetch.

    python src/pipeline_budget.py
    python src/pipeline_budget.py --jobs 12        # what a driver fix would buy
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Measured 2026-08-13 on this box (i7-14650HX), full 720 steps, UNCONTENDED:
#   random vs random        2.40 s   (few actions -- NOT representative)
#   real agent vs starter   4.70 s   <- what a tournament game actually costs
# An earlier 8.24 s figure was taken while other jobs were running and is
# therefore wrong; anything quoting it overstates every engine-bound stage.
EPISODE_SEC = 4.70
# Measured: 1.30 MB wire, ~2.7-3.1 s per replay, latency-bound.
REPLAY_SEC = 2.9


def measure_episode():
    """Re-time one FULL episode with a REPRESENTATIVE matchup.

    Two traps, both hit while building this:
      * the episode must be the full 720 steps -- per-step cost GROWS with farm
        complexity (more hands means more unit ops), so scaling a 120-step run
        by 6 under-measured by ~2x;
      * the AGENTS matter as much as the length. random-vs-random takes 2.40 s
        because random farms do little, while a real agent against a tape takes
        4.70 s. The tournament plays the latter, so measuring the former makes
        every engine stage look half price.
    """
    try:
        import kaggriculture.engine._vendor as _vendor  # noqa: F401
        from kaggle_environments import make
        agent = None
        for pat in ("agents/v2[4-9]*_bandit.py", "agents/v2_tuned.py"):
            hits = sorted(glob.glob(os.path.join(ROOT, pat)))
            if hits:
                agent = hits[-1]
                break
        if not agent:
            return EPISODE_SEC
        t0 = time.time()
        env = make("kaggriculture", configuration={
            "episodeSteps": 720, "seed": 3, "actTimeout": 60,
            "runTimeout": 100000})
        env.run([agent, "starter"])
        return time.time() - t0
    except Exception:                                           # noqa: BLE001
        return EPISODE_SEC


def measure_cache():
    """Is the feature cache warm? A cold cache costs ~6 min more."""
    try:
        import kaggriculture.data.feature_cache as FC
        import kaggriculture.data.routes as R
        import kaggriculture.train.train_identifier as TI
        route, prefix, _ = FC.load(TI.PREFIX_TURNS)
        total = len(R.load_index()["routes"])
        return len(route), total
    except Exception:                                           # noqa: BLE001
        return 0, 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=None,
                    help="engine concurrency (default: the cycle's TOURNEY_JOBS)")
    ap.add_argument("--fetch-jobs", type=int, default=None)
    ap.add_argument("--no-measure", action="store_true")
    args = ap.parse_args()

    import kaggriculture.pipeline.refresh_cycle as RC
    jobs = args.jobs or RC.TOURNEY_JOBS
    fetch_jobs = args.fetch_jobs or 8
    ep = EPISODE_SEC if args.no_measure else measure_episode()
    cached, total_routes = measure_cache()
    warm = total_routes and cached >= 0.9 * total_routes

    # --- engine-bound stages, priced from (games x episode) / jobs ----------
    n_ref = RC.MAX_LOSS_TAPES + RC.FIELD_STRATA + 1        # + crowd guard
    n_cand = RC.MAX_CANDIDATES
    seeds = len(RC.SEED_SETS)

    # Successive halving: screen -> full roster -> final (see run_halving).
    screen_ref = max(1, RC.FIELD_STRATA)
    keep1 = max(RC.PLAYOFF_FINALISTS + 1, (n_cand + 1) // 2)
    r1 = (n_cand + 1) * 1 * screen_ref * 1 * 2
    r2 = (keep1 + 1) * 1 * n_ref * 1 * 2
    r3 = (RC.PLAYOFF_FINALISTS + 1) * seeds * n_ref * 2 * 2
    tourney_games = r1 + r2 + r3
    old_games = (n_cand + 1) * seeds * n_ref * 2 * 2

    playoff_games = (RC.PLAYOFF_FINALISTS + 1) * RC.PLAYOFF_FINALISTS * 4
    gate_games = 40
    arms_games = 0 if True else 1920        # ARM GUARD holds -> fast profile

    def eng(games):
        return games * ep / max(1, jobs)

    # --- fetch stages, priced from replay latency and concurrency ----------
    replays_day = 450          # ~900 routes/day at 2 seats per episode
    tapes = RC.MAX_LOSS_TAPES + RC.FIELD_STRATA

    rows = [
        ("detect (Kaggle API)", 5, "network"),
        ("games: ourgames delta", 60 * 2, "network"),
        ("mine: leaderboard + archive",
         replays_day * REPLAY_SEC / fetch_jobs, "network"),
        ("tapes: fetch + build", tapes * (REPLAY_SEC + 3), "network"),
        ("rust parity (3 episodes, both engines)", 60, "cpu1"),
        ("train: identifier" + (" (cache warm)" if warm else " (COLD CACHE)"),
         70 if warm else 70 + 6 * 60, "cpu1"),
        ("train: relay + gates + surrogate", 90, "cpu1"),
        ("tourney (successive halving)", eng(tourney_games), "engine"),
        ("playoff", eng(playoff_games), "engine"),
        ("arms (fast profile under ARM GUARD)", eng(arms_games) + 60, "engine"),
        ("build + model graphs", 150, "cpu1"),
        ("gate: paired validation + latency", eng(gate_games), "engine"),
        ("notebooks: push x2 (parallel-capable)", 240, "network"),
        ("upload + sha poll", 300, "network"),
    ]

    print(f"machine: episode {ep:.2f}s, engine jobs {jobs}, fetch jobs "
          f"{fetch_jobs}")
    print(f"feature cache: {cached:,}/{total_routes:,} routes "
          f"({'WARM' if warm else 'COLD -- add ~6 min'})")
    print(f"tournament: {tourney_games:,} games via halving "
          f"vs {old_games:,} fixed ({old_games / max(1, tourney_games):.2f}x)")
    print()
    print(f"{'stage':<42}{'minutes':>9}  bound by")
    tot = 0.0
    by_kind = {}
    for name, sec, kind in rows:
        tot += sec
        by_kind[kind] = by_kind.get(kind, 0.0) + sec
        print(f"{name:<42}{sec / 60:>9.1f}  {kind}")
    print(f"{'TOTAL (serial)':<42}{tot / 60:>9.1f}")
    print()
    for k in sorted(by_kind, key=lambda k: -by_kind[k]):
        print(f"  {k:<8} {by_kind[k] / 60:>6.1f} min "
              f"({100 * by_kind[k] / tot:.0f}%)")

    # Two-track split: fetch moves to the hourly job, so the morning path is
    # everything that is not network-fetch.
    fetch_names = ("detect", "games:", "mine:", "tapes:")
    morning = sum(s for n, s, _k in rows
                  if not n.startswith(fetch_names))
    print(f"\ntwo-track: continuous (hourly) fetch + a morning release of "
          f"{morning / 60:.1f} min")
    print(f"  04:30 start -> pair ready ~"
          f"{(4 * 60 + 30 + morning / 60) // 60:.0f}:"
          f"{(4 * 60 + 30 + morning / 60) % 60:02.0f}")
    print("\nnote: uploads are currently OFF (--no-submit in the launcher), so "
          "the last two rows run but publish nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
