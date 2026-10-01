"""SPRT-gated improvement loop: propose, test properly, keep only what is real.

The problem this solves
-----------------------
Tuning stopped moving the Elo. Not because the search was bad, but because the
acceptance test was: `tune.py` and `cmaes.py` compare candidates on a fixed
handful of seeds and keep whichever scored higher. At 8-16 matches the standard
error on a win rate is around 12 points of Elo per match played, so a run of
"improvements" is mostly a random walk that ratchets on noise. The project has
the receipt: `cost_per_animal_day` 4.5 -> 3.5 measured 80% over 10 matches and
56% over 16, and several other "wins" reversed the same way.

This loop changes the accept rule, not the search:

    propose a candidate
      -> play paired games against the incumbent on common random numbers
      -> feed each result to a sequential probability ratio test
      -> accept only on a statistically significant gain, reject early otherwise
      -> the accepted candidate becomes the new incumbent

Because SPRT stops as soon as the evidence is decisive, a clearly bad candidate
costs ~30 games instead of a fixed budget, and the savings go into testing more
candidates. See `python -m kaggriculture.measure.sprt` for the measured games-per-decision.

Three proposal families, chosen per round
-----------------------------------------
  perturb    Gaussian jitter on a random subset of numeric knobs. Subset size is
             drawn small: changing everything at once makes an accepted result
             uninterpretable and usually just adds variance.
  structural Flip a discrete design choice -- currently assign_mode, greedy vs
             optimal task matching. These are the changes that actually move a
             plateaued agent; knob tuning cannot invent a better algorithm.
  recombine  Blend the incumbent with an earlier accepted agent. Cheap way to
             escape a local optimum without a full restart.

Everything is checkpointed to .local/improve/state.json after every game, so an
interrupted run resumes exactly where it stopped.

    python -m kaggriculture.pipeline.improve --minutes 60
    python -m kaggriculture.pipeline.improve --minutes 240 --elo1 15 --workers 8
    python -m kaggriculture.pipeline.improve --resume
    python -m kaggriculture.pipeline.improve --status
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import datetime as dt
import glob
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
from kaggriculture.measure.sprt import SPRT, elo_interval   # noqa: E402

WORK = os.path.join(ROOT, ".local", "improve")
STATE = os.path.join(WORK, "state.json")
SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")

# Knobs worth perturbing, with a sane relative step. Deliberately excludes the
# ones already measured negative -- see docs/history/runbook.md section 5 -- because
# re-proposing a known loser wastes the budget the test is trying to conserve.
KNOBS = {
    "travel_weight": 0.25, "fert_weight": 0.25, "poach_penalty": 0.15,
    "capacity_util": 0.10, "cost_per_crop_day": 0.15, "cost_per_animal_day": 0.12,
    "reserve_base": 0.30, "reserve_per_tile": 0.30, "feed_runway_days": 0.25,
    "feed_safe_discount": 0.20, "feed_buffer_days": 0.25, "liquid_weight": 0.30,
    "liquid_target": 0.30, "shed_pressure": 0.25, "sell_chunk": 0.25,
    "sell_floor": 0.20, "hire_cash_frac": 0.20, "hands_max": 0.15,
    "hands_min": 0.25, "max_herd": 0.15, "seed_lookahead": 0.25,
    "land_reserve": 0.30, "animal_cash_buffer": 0.30, "care_weight": 0.25,
    "fert_collect_weight": 0.25, "wheat_tile_yield": 0.15,
    "ensemble_spread": 0.35, "ensemble_consensus": 0.15,
}
INTEGER_KNOBS = {"hands_max", "hands_min", "max_herd", "seed_lookahead",
                 "land_min_day", "land_last_day", "dump_day", "hands_late_day",
                 "hands_max_late", "target_cow", "target_sheep", "target_goose",
                 "target_melon", "target_strawberry", "target_carrot"}
# Discrete design choices. These are the changes that move a plateaued agent:
# optimal assignment was worth +163 Elo where months of knob tuning was worth
# nothing measurable. ensemble_k trades per-turn latency for decision quality --
# at k=12 the agent uses 5.7 ms of a 1000 ms actTimeout, so the ceiling here is
# decision quality, never the clock.
STRUCTURAL = {
    "assign_mode": ["greedy", "optimal"],
    "ensemble_k": [0, 4, 8, 12, 16, 24],
}


def _mute():
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def _play(job):
    """One 720-turn match. Runs in a worker process."""
    left, right, seed = job
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


def newest_agent():
    c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
    return c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")


# ------------------------------------------------------------------ proposals

def propose(P, rng, history):
    """Return (candidate_params, description). One of three families."""
    roll = rng.random()

    if roll < 0.20:
        # structural: the only family that can change the algorithm itself
        key = rng.choice(sorted(STRUCTURAL))
        cur = P.get(key, STRUCTURAL[key][0])
        alt = [v for v in STRUCTURAL[key] if v != cur]
        if alt:
            Q = dict(P)
            Q[key] = rng.choice(alt)
            return Q, f"structural {key}: {cur} -> {Q[key]}"

    if roll < 0.32 and len(history) >= 2:
        # recombine with an earlier accepted point
        other = rng.choice(history[:-1])
        Q = dict(P)
        w = rng.uniform(0.25, 0.75)
        changed = 0
        for k, v in other.items():
            if k in KNOBS and isinstance(v, (int, float)) and \
                    isinstance(P.get(k), (int, float)):
                blend = P[k] * (1 - w) + v * w
                Q[k] = int(round(blend)) if k in INTEGER_KNOBS else round(blend, 4)
                changed += 1
        if changed:
            return Q, f"recombine with an earlier point (w={w:.2f}, {changed} knobs)"

    # perturb: a small subset, so an acceptance is interpretable
    keys = [k for k in KNOBS if isinstance(P.get(k), (int, float))]
    if not keys:
        return dict(P), "no numeric knobs to perturb"
    n = min(len(keys), rng.choice([1, 1, 2, 2, 3, 4]))
    picked = rng.sample(keys, n)
    Q = dict(P)
    for k in picked:
        step = KNOBS[k]
        val = P[k] * (1.0 + rng.gauss(0, step))
        if k in INTEGER_KNOBS:
            val = int(round(val))
            if val == P[k]:
                val += rng.choice([-1, 1])
            val = max(1, val)
        else:
            val = round(max(val, 1e-6), 4)
        Q[k] = val
    detail = ", ".join(f"{k} {P[k]}->{Q[k]}" for k in picked)
    return Q, f"perturb {n} knob(s): {detail}"


# ----------------------------------------------------------------- the tester

def run_sprt(cand_path, base_path, test, rng, workers, seed0, deadline,
             batch=None, verbose=True):
    """Play paired games until the SPRT decides, the clock runs out, or the cap.

    Paired means every seed is played in both seats: the shared market makes
    the game asymmetric, so a one-seat sample can reverse the apparent winner.
    Both seats of a seed are always played together, so the test never sees a
    half-pair.
    """
    batch = batch or max(2, workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        while not test.decided():
            if time.time() > deadline:
                return "timeout"
            seeds = [seed0 + rng.randrange(1_000_000) for _ in range(max(1, batch // 2))]
            jobs = []
            for s in seeds:
                jobs.append((cand_path, base_path, s))     # candidate as player 0
                jobs.append((base_path, cand_path, s))     # and as player 1
            for (mine, theirs), job in zip(pool.map(_play, jobs), jobs):
                cand_first = job[0] == cand_path
                cand_score = mine if cand_first else theirs
                base_score = theirs if cand_first else mine
                if cand_score == base_score:
                    test.record(draw=True)
                else:
                    test.record(win=cand_score > base_score)
                if test.decided():
                    break
            if verbose:
                pr.log(f"{test.wins}W-{test.losses}L  llr {test.llr:+.2f} "
                       f"of [{test.lower:.2f}, {test.upper:.2f}]  "
                       f"({test.games} games)", 2)
    return test.verdict()


# ---------------------------------------------------------------- persistence

def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(st):
    os.makedirs(WORK, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def show_status():
    st = load_state()
    if not st:
        print("no improve run recorded yet")
        return 0
    print(f"incumbent : {st['incumbent']}")
    print(f"rounds    : {st['rounds']}  accepted {st['accepted']}  "
          f"rejected {st['rejected']}  inconclusive {st['inconclusive']}")
    print(f"games     : {st['games']:,} played")
    print(f"started   : {st.get('started')}")
    print("\nlast 12 rounds:")
    for r in st.get("log", [])[-12:]:
        print(f"  {r['verdict']:<13} {r['games']:>4}g  {r['record']:<10} "
              f"{r['elo']:>+7.0f} Elo   {r['proposal'][:64]}")
    return 0


# ------------------------------------------------------------------- the loop

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="starting agent; default is the newest")
    ap.add_argument("--minutes", type=float, default=60.0,
                    help="wall-clock budget for the whole run")
    ap.add_argument("--elo1", type=float, default=25.0,
                    help="smallest gain worth adopting; smaller costs far more "
                         "games (see python -m kaggriculture.measure.sprt)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-games", type=int, default=200,
                    help="cap per candidate before calling it inconclusive")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        return show_status()

    os.makedirs(WORK, exist_ok=True)
    pr.reset()
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    st = load_state() if args.resume else None
    if st:
        incumbent = st["incumbent"]
        pr.log(f"resuming from {incumbent} "
               f"({st['rounds']} rounds, {st['games']:,} games so far)")
    else:
        incumbent = args.base or newest_agent()
        incumbent = os.path.relpath(incumbent, ROOT) if os.path.isabs(incumbent) else incumbent
        st = {"incumbent": incumbent, "rounds": 0, "accepted": 0, "rejected": 0,
              "inconclusive": 0, "games": 0, "log": [], "history": [],
              "started": dt.datetime.now().replace(microsecond=0).isoformat()}

    P = paramio.load(os.path.join(ROOT, st["incumbent"]))
    if not st["history"]:
        st["history"] = [copy.deepcopy(P)]

    rng = random.Random(args.seed0 + 4242 + st["rounds"])
    deadline = time.time() + args.minutes * 60.0
    cand_path = os.path.join(WORK, "candidate.py")

    pr.log(f"incumbent {st['incumbent']}")
    pr.log(f"budget {args.minutes:.0f} min, {workers} workers, "
           f"SPRT elo1={args.elo1:.0f} alpha={args.alpha} beta={args.beta}", 1)
    pr.log("only statistically significant gains are kept -- expect most "
           "candidates to be rejected", 1)

    while time.time() < deadline:
        st["rounds"] += 1
        Q, desc = propose(P, rng, st["history"])
        if Q == P:
            continue
        pr.log(f"round {st['rounds']}: {desc}")

        paramio.write(SRC, cand_path, Q, header=f"improve round {st['rounds']}",
                      module_doc=f"candidate: {desc}\n")

        test = SPRT(elo0=0.0, elo1=args.elo1, alpha=args.alpha,
                    beta=args.beta, max_games=args.max_games)
        verdict = run_sprt(cand_path, os.path.join(ROOT, st["incumbent"]),
                           test, rng, workers, args.seed0, deadline)
        st["games"] += test.games
        decided = max(1, test.wins + test.losses)
        lo, mid, hi = elo_interval(test.wins, decided)
        st["log"].append({
            "round": st["rounds"], "proposal": desc, "verdict": verdict,
            "games": test.games, "record": f"{test.wins}W-{test.losses}L",
            "elo": mid, "elo_lo": lo, "elo_hi": hi,
        })

        if verdict == "accept":
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join("agents", f"agent_vsprt_{stamp}.py")
            paramio.write(SRC, os.path.join(ROOT, out), Q,
                          header=f"SPRT-accepted: {desc}",
                          module_doc=(f"Accepted by src/kaggriculture/pipeline/improve.py round "
                                      f"{st['rounds']}: {desc}\n"
                                      f"{test.wins}W-{test.losses}L over "
                                      f"{test.games} paired games, "
                                      f"{mid:+.0f} Elo [{lo:+.0f}, {hi:+.0f}]\n"))
            st["incumbent"] = out
            st["accepted"] += 1
            st["history"].append(copy.deepcopy(Q))
            P = Q
            pr.log(f"ACCEPTED -> {out}", 1)
            pr.log(test.summary(), 1)
        elif verdict == "timeout":
            pr.log("budget reached mid-test; candidate discarded "
                   "(an undecided test is not evidence)", 1)
            st["log"][-1]["verdict"] = "timeout"
            save_state(st)
            break
        else:
            st["rejected" if verdict == "reject" else "inconclusive"] += 1
            pr.log(f"{verdict}: {test.wins}W-{test.losses}L in {test.games} games "
                   f"({mid:+.0f} Elo)", 1)
        save_state(st)

    pr.log("")
    pr.log(f"done: {st['rounds']} rounds, {st['games']:,} games, "
           f"{st['accepted']} accepted / {st['rejected']} rejected / "
           f"{st['inconclusive']} inconclusive")
    pr.log(f"incumbent is now {st['incumbent']}", 1)
    if st["accepted"]:
        pr.log("rate it:  python -m kaggriculture.measure.elo --only " + st["incumbent"], 1)
        pr.log("then:     python -m kaggriculture.pipeline.dashboard", 1)
    else:
        pr.log("nothing beat the incumbent. That is a result, not a failure -- "
               "it says the knobs are exhausted and the next gain has to come "
               "from a structural change.", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
