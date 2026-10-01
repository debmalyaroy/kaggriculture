"""Turn adaptive play on or off, and measure whether it helped.

Two adaptations live in the agent; this switches them and A/Bs them.

  bandit  Exp3 over the committee, reweighted from an in-game wealth signal, so
          the vote leans on whichever member is working in *this* game. Exp3
          rather than UCB because the reward is non-stationary by construction:
          what pays on day 3 is wrong on day 27, and UCB's confidence bounds
          assume a fixed arm quality. The weight floor keeps every member
          exploring -- a bandit that fully commits cannot notice the game moved.

  risk    Score-aware play. Behind late, press: release reserves, plant longer
          and higher, accept variance -- on a skill ladder a narrow loss and a
          wide loss score identically, so protecting a losing position is
          strictly worse than gambling out of it. Ahead late, protect.

    python -m kaggriculture.train.adaptive --list
    python -m kaggriculture.train.adaptive --agent "agents/agent_v4_*.py" --mode both
    python -m kaggriculture.train.adaptive --agent "agents/agent_v4_*.py" --mode risk --ab
    python -m kaggriculture.train.adaptive --off "agents/agent_vadapt_*.py"

`--ab` runs a full SPRT of the adaptive build against the static one it came
from, because "adaptive" is a claim like any other and this project does not
accept claims on 10 matches.
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
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

SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")
MODES = ("off", "bandit", "risk", "both")
NEEDS_COMMITTEE = ("bandit", "both")


def _resolve(pattern):
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(p))
    return hits[-1] if hits else None


def build(base, mode, ensemble_k=None, eta=None, floor=None, risk_gain=None,
          arbiter=None, out=None, verbose=True):
    """Write a variant of `base` with adaptive play set to `mode`."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    P = dict(paramio.load(base))
    P["adaptive_mode"] = mode
    if ensemble_k is not None:
        P["ensemble_k"] = int(ensemble_k)
    if eta is not None:
        P["bandit_eta"] = float(eta)
    if floor is not None:
        P["bandit_floor"] = float(floor)
    if risk_gain is not None:
        P["risk_gain"] = float(risk_gain)
    if arbiter is not None:
        P["arbiter_model"] = str(arbiter)

    k = int(P.get("ensemble_k", 0) or 0)
    if mode in NEEDS_COMMITTEE and k <= 0:
        # A bandit over one arm is not a bandit.
        P["ensemble_k"] = 8
        if verbose:
            pr.warn(f"{mode} needs a committee; setting ensemble_k=8", 1)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out or os.path.join(ROOT, "agents", f"agent_vadapt_{mode}_{stamp}.py")
    doc = [f"Adaptive play: {mode}.", ""]
    if mode in NEEDS_COMMITTEE:
        doc.append(f"Exp3 over {P['ensemble_k']} committee members, eta="
                   f"{P.get('bandit_eta')}, floor={P.get('bandit_floor')}.")
        doc.append("Members are reweighted once per in-game day from the change")
        doc.append("in sellable wealth, credited to whoever proposed the op that")
        doc.append("was actually played.")
    if mode in ("risk", "both"):
        doc.append(f"Score-aware from day {P.get('risk_late_day')}, gain "
                   f"{P.get('risk_gain')}: presses when behind, protects when ahead.")
    doc.append("")
    doc.append(f"Base: {os.path.relpath(base, ROOT)}")

    paramio.write(SRC, out, P, header=f"adaptive {mode}",
                  module_doc="\n".join(doc) + "\n")
    rel = os.path.relpath(out, ROOT)
    registry.register_model(os.path.basename(out), path=rel, built_by="adaptive",
                            base=os.path.relpath(base, ROOT), adaptive_mode=mode,
                            description=f"adaptive: {mode}")
    if verbose:
        pr.log(f"wrote {rel}  (adaptive_mode={mode}, ensemble_k={P.get('ensemble_k')})")
    return rel


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


def ab(cand, base, elo1=25.0, max_games=200, workers=0, seed0=0, verbose=True):
    """Full SPRT of the adaptive build against the static one."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    test = SPRT(elo0=0.0, elo1=elo1, max_games=max_games)
    rng = random.Random(seed0 + 31337)
    cpath = os.path.join(ROOT, cand)
    bpath = os.path.join(ROOT, base) if not os.path.isabs(base) else base
    if verbose:
        pr.log(f"SPRT: {os.path.basename(cand)} vs {os.path.basename(base)} "
               f"(elo1={elo1:.0f}, {workers} workers)")
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        while not test.decided():
            seeds = [seed0 + rng.randrange(1_000_000)
                     for _ in range(max(1, workers // 2))]
            jobs = []
            for s in seeds:
                jobs.append((cpath, bpath, s))
                jobs.append((bpath, cpath, s))
            for (a, _b, _s), (ra, rb) in zip(jobs, pool.map(_play, jobs)):
                mine, theirs = (ra, rb) if a == cpath else (rb, ra)
                if mine == theirs:
                    test.record(draw=True)
                else:
                    test.record(win=mine > theirs)
                if test.decided():
                    break
            if verbose:
                pr.log(f"{test.wins}W-{test.losses}L  llr {test.llr:+.2f}", 2)
    if verbose:
        pr.log(test.summary(), 1)
    lo, mid, hi = elo_interval(test.wins, max(1, test.wins + test.losses))
    registry.register_model(os.path.basename(cand),
                            ab_verdict=test.verdict(),
                            ab_record=f"{test.wins}W-{test.losses}L",
                            ab_elo=round(mid, 1))
    return test


def show():
    rows = []
    for m in registry.ranked():
        if not m.get("exists"):
            continue
        path = os.path.join(ROOT, m.get("path") or f"agents/{m['name']}")
        try:
            P = paramio.load(path)
        except Exception:                                          # noqa: BLE001
            continue
        rows.append((m["name"], str(P.get("adaptive_mode", "off")),
                     int(P.get("ensemble_k", 0) or 0), m.get("elo", 0),
                     m.get("games", 0), m.get("ab_verdict", "")))
    print(f"{'model':<44} {'adaptive':<8} {'k':>3} {'elo':>6} {'games':>6}  ab")
    print("-" * 88)
    for name, mode, k, elo, games, verdict in rows:
        print(f"{name:<44} {mode:<8} {k:>3} {elo:>6.0f} {games:>6}  {verdict}")
    print("\nbandit needs a committee (ensemble_k > 0); risk works with or without one.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", help="base agent (path or glob)")
    ap.add_argument("--mode", choices=MODES, default="both")
    ap.add_argument("--off", metavar="AGENT",
                    help="write a static copy of this agent (adaptive_mode=off)")
    ap.add_argument("--ensemble-k", type=int, default=None)
    ap.add_argument("--eta", type=float, default=None, help="Exp3 learning rate")
    ap.add_argument("--floor", type=float, default=None, help="Exp3 weight floor")
    ap.add_argument("--risk-gain", type=float, default=None)
    ap.add_argument("--arbiter", choices=("table", "xgb"), default=None,
                    help="which arbiter weights the committee vote")
    ap.add_argument("--ab", action="store_true", help="SPRT it against the base")
    ap.add_argument("--elo1", type=float, default=25.0)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    pr.reset()
    if args.list or not (args.agent or args.off):
        return show()

    if args.off:
        base = _resolve(args.off)
        if not base:
            pr.warn(f"no agent matched {args.off}")
            return 1
        build(base, "off", verbose=True)
        return 0

    base = _resolve(args.agent)
    if not base:
        pr.warn(f"no agent matched {args.agent}")
        return 1
    rel = build(base, args.mode, args.ensemble_k, args.eta, args.floor,
                args.risk_gain, args.arbiter)

    if args.ab:
        test = ab(rel, os.path.relpath(base, ROOT), args.elo1, args.max_games,
                  args.workers)
        if test.verdict() != "accept":
            pr.log("adaptive play is not a significant improvement here. That is "
                   "a result: it says the static policy already handles these "
                   "positions, or the committee is too uniform for the bandit "
                   "to have anything to choose between.", 1)
    else:
        pr.log("")
        pr.log("measure it before believing it:")
        pr.log(f"python -m kaggriculture.train.adaptive --agent {os.path.relpath(base, ROOT)} "
               f"--mode {args.mode} --ab", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
