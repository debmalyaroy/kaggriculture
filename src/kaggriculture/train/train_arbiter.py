"""Learn the ensemble arbiter: which committee member to trust, and when.

The RL problem this project actually has
----------------------------------------
`ml_rl.py` runs Monte-Carlo control over the allocation policy, and it is stuck
for a structural reason rather than an algorithmic one: it credits one scalar
episode return to 128 state buckets. Swapping in PPO or DQN does not fix that --
the signal is thin at the source. You fix it by giving the learner a smaller,
better-shaped decision.

The ensemble created exactly that decision. Every turn the committee disagrees
about some units, and the vote resolves it by counting heads. Counting heads
assumes every member is equally trustworthy in every state, which is plainly
false: a policy that over-invests early is right on day 2 and wrong on day 27.
So the thing to learn is a **state-conditioned weight per member** -- a small
table, exercised 720 times an episode instead of once.

Method: cross-entropy method (CEM)
----------------------------------
CEM over the weight table, not a gradient method, for three reasons that matter
here rather than in general:

* The objective is win rate over paired matches -- a noisy, non-differentiable
  Monte-Carlo estimate. There is no gradient to follow.
* The table is small (9 buckets x k members). CEM is at its best in exactly
  this regime and needs no learning-rate tuning to be stable.
* Every generation's elite set is a distribution we can inspect, so a run that
  is going wrong is visible while it happens rather than afterwards.

Each candidate weight table is scored by paired both-seat matches against the
base agent on common random numbers. The winner is then put through a full SPRT
before it is allowed to become an agent, so a lucky generation cannot promote
itself -- the same accept rule as everything else in this project.

    python -m kaggriculture.train.train_arbiter --base agents/agent_v5_ensemble_*.py --minutes 60
    python -m kaggriculture.train.train_arbiter --resume
    python -m kaggriculture.train.train_arbiter --status
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import json
import math
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

WORK = os.path.join(ROOT, "models", "arbiter")
STATE = os.path.join(WORK, "state.json")
SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")
WEIGHTS_BEGIN = "# --- WEIGHTS BEGIN"
WEIGHTS_END = "# --- WEIGHTS END ---"

# Must match _weight_bucket() in the agent: 3 day bands x 3 cash bands.
BUCKETS = [f"{d}|{c}" for d in range(3) for c in range(3)]


# ------------------------------------------------------------------- tables --

def flat_table(n_members, value=1.0):
    return {b: [value] * n_members for b in BUCKETS}


def table_to_vector(table, n_members):
    return [table[b][i] for b in BUCKETS for i in range(n_members)]


def vector_to_table(vec, n_members):
    out = {}
    for bi, b in enumerate(BUCKETS):
        out[b] = [max(0.0, round(vec[bi * n_members + i], 4)) for i in range(n_members)]
    return out


def write_agent(base, table, out_path, note=""):
    """Materialise an agent carrying this weight table."""
    P = paramio.load(base)
    paramio.write(SRC, out_path, P, header="arbiter weights",
                  module_doc=("Ensemble with learned arbiter weights.\n" + note + "\n"))
    with open(out_path, encoding="utf-8") as f:
        src = f.read()
    i, j = src.index(WEIGHTS_BEGIN), src.index(WEIGHTS_END)
    body = [WEIGHTS_BEGIN + " (src/kaggriculture/train/train_arbiter.py rewrites this block) ---",
            "# Learned by cross-entropy method over paired self-play.",
            "# Key is \"<day band>|<cash band>\"; one weight per committee member,",
            "# index 0 being the incumbent. A zero mutes that member in that state.",
            "MEMBER_WEIGHTS = {"]
    for b in BUCKETS:
        body.append(f"    {json.dumps(b)}: {json.dumps(table[b])},")
    body.append("}")
    src = src[:i] + "\n".join(body) + "\n" + src[j:]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src)
    return out_path


# ------------------------------------------------------------------ scoring --

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


def score(cand_path, base_path, seeds, pool):
    """Win rate over paired both-seat matches on common random numbers.

    Paired because the shared market makes seats asymmetric; common random
    numbers because comparing two policies on different seeds measures the
    seeds.
    """
    jobs = []
    for s in seeds:
        jobs.append((cand_path, base_path, s))
        jobs.append((base_path, cand_path, s))
    wins = games = 0
    for (a, _b, _s), (ra, rb) in zip(jobs, pool.map(_play, jobs)):
        mine, theirs = (ra, rb) if a == cand_path else (rb, ra)
        games += 1
        wins += int(mine > theirs)
    return (wins / games) if games else 0.0, wins, games


# ---------------------------------------------------------------------- CEM --

def cem(base, n_members, minutes, popsize, elite_frac, sigma, seeds_per_cand,
        workers, seed0, state=None, verbose=True):
    """Cross-entropy method over the weight table. Returns the best table found."""
    dim = len(BUCKETS) * n_members
    mu = state["mu"] if state else [1.0] * dim
    sd = state["sd"] if state else [sigma] * dim
    history = state["history"] if state else []
    best = state.get("best") if state else None

    rng = random.Random(seed0 + 77 + len(history))
    deadline = time.time() + minutes * 60.0
    n_elite = max(2, int(popsize * elite_frac))
    cand_path = os.path.join(WORK, "cand.py")
    os.makedirs(WORK, exist_ok=True)

    gen = len(history)
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        while time.time() < deadline:
            gen += 1
            seeds = [seed0 + rng.randrange(1_000_000) for _ in range(seeds_per_cand)]
            scored = []
            for ci in range(popsize):
                if time.time() > deadline:
                    break
                vec = [max(0.0, rng.gauss(mu[i], sd[i])) for i in range(dim)]
                table = vector_to_table(vec, n_members)
                write_agent(base, table, cand_path,
                            note=f"CEM generation {gen} candidate {ci + 1}")
                wr, w, g = score(cand_path, base, seeds, pool)
                scored.append((wr, vec, w, g))
                if verbose:
                    pr.log(f"gen {gen} cand {ci + 1}/{popsize}: {w}/{g} = {wr:.0%}", 2)
            if not scored:
                break

            scored.sort(key=lambda t: -t[0])
            elite = [v for _wr, v, _w, _g in scored[:n_elite]]
            mu = [sum(e[i] for e in elite) / len(elite) for i in range(dim)]
            sd = [max(0.05,
                      math.sqrt(sum((e[i] - mu[i]) ** 2 for e in elite) / len(elite)))
                  for i in range(dim)]
            top = scored[0]
            history.append({"gen": gen, "best_wr": top[0],
                            "mean_wr": sum(s[0] for s in scored) / len(scored),
                            "elite": n_elite, "seeds": len(seeds)})
            if best is None or top[0] > best["wr"]:
                best = {"wr": top[0], "vec": top[1], "gen": gen}
            if verbose:
                pr.log(f"gen {gen}: best {top[0]:.0%}, mean "
                       f"{history[-1]['mean_wr']:.0%}, sigma "
                       f"{sum(sd) / len(sd):.2f}", 1)
            save_state({"mu": mu, "sd": sd, "history": history, "best": best,
                        "base": base, "n_members": n_members})
    return {"mu": mu, "sd": sd, "history": history, "best": best,
            "base": base, "n_members": n_members}


# ----------------------------------------------------------------- plumbing --

def save_state(st):
    os.makedirs(WORK, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def load_state():
    if not os.path.exists(STATE):
        return None
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _resolve(pattern):
    p = pattern if os.path.isabs(pattern) else os.path.join(ROOT, pattern)
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(p))
    return hits[-1] if hits else None


def n_members_of(path):
    """How many committee members the base agent actually runs."""
    P = paramio.load(path)
    k = int(P.get("ensemble_k", 0) or 0)
    return k + 1        # + the incumbent


def show_status():
    st = load_state()
    if not st:
        print("no arbiter run recorded yet")
        return 0
    print(f"base      : {st.get('base')}")
    print(f"members   : {st.get('n_members')}")
    print(f"generations: {len(st.get('history', []))}")
    b = st.get("best") or {}
    print(f"best      : {b.get('wr', 0):.0%} win rate (generation {b.get('gen')})")
    print("\ngeneration  best   mean")
    for h in st.get("history", [])[-15:]:
        print(f"   {h['gen']:>7}  {h['best_wr']:.0%}   {h['mean_wr']:.0%}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="agents/*ensemble*.py",
                    help="an agent with ensemble_k > 0")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--elite-frac", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--seeds", type=int, default=3,
                    help="paired seeds per candidate (x2 seats)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--elo1", type=float, default=25.0)
    ap.add_argument("--max-games", type=int, default=160)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.status:
        return show_status()

    pr.reset()
    os.makedirs(WORK, exist_ok=True)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    st = load_state() if args.resume else None
    base = _resolve(st["base"] if st else args.base)
    if not base:
        pr.warn(f"no base agent matched {args.base}")
        pr.log("build one:  python -m kaggriculture.agentbuild.build_agent configs/v5_ensemble.json", 1)
        return 1
    n_members = (st or {}).get("n_members") or n_members_of(base)
    if n_members < 2:
        pr.warn(f"{os.path.relpath(base, ROOT)} has ensemble_k=0 -- there is no "
                f"committee to arbitrate")
        pr.log("use an ensemble config, e.g. configs/v5_ensemble.json", 1)
        return 1

    pr.log(f"base {os.path.relpath(base, ROOT)}  ({n_members} members incl. incumbent)")
    pr.log(f"CEM: pop {args.popsize}, elite {args.elite_frac:.0%}, "
           f"{args.seeds} paired seeds/candidate, {workers} workers", 1)
    pr.log(f"table: {len(BUCKETS)} state buckets x {n_members} members = "
           f"{len(BUCKETS) * n_members} weights", 1)

    st = cem(base, n_members, args.minutes, args.popsize, args.elite_frac,
             args.sigma, args.seeds, workers, args.seed0, st)

    best = st.get("best")
    if not best:
        pr.warn("no generation completed inside the budget")
        return 1
    table = vector_to_table(best["vec"], n_members)
    pr.log(f"best candidate: {best['wr']:.0%} win rate in generation {best['gen']}", 1)

    # A CEM winner is the best of a noisy sample, which is exactly the thing
    # that looks better than it is. Nothing is promoted without a full SPRT.
    cand = os.path.join(WORK, "best.py")
    write_agent(base, table, cand, note=f"CEM best, generation {best['gen']}")
    if args.no_verify:
        pr.log("skipping SPRT verification (--no-verify)", 1)
        verdict, test = "unverified", None
    else:
        pr.log("verifying the winner with a full SPRT before promoting it", 1)
        test = SPRT(elo0=0.0, elo1=args.elo1, max_games=args.max_games)
        rng = random.Random(args.seed0 + 991)
        with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
            while not test.decided():
                seeds = [args.seed0 + rng.randrange(1_000_000)
                         for _ in range(max(1, workers // 2))]
                jobs = []
                for s in seeds:
                    jobs.append((cand, base, s))
                    jobs.append((base, cand, s))
                for (a, _b, _s), (ra, rb) in zip(jobs, pool.map(_play, jobs)):
                    mine, theirs = (ra, rb) if a == cand else (rb, ra)
                    test.record(win=mine > theirs) if mine != theirs else test.record(draw=True)
                    if test.decided():
                        break
                pr.log(f"{test.wins}W-{test.losses}L  llr {test.llr:+.2f}", 2)
        verdict = test.verdict()
        pr.log(test.summary(), 1)

    if verdict == "accept":
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(ROOT, "agents", f"agent_varb_{stamp}.py")
        lo, mid, hi = elo_interval(test.wins, max(1, test.wins + test.losses))
        write_agent(base, table, out,
                    note=(f"Arbiter weights learned by CEM over {len(st['history'])} "
                          f"generations.\nSPRT-accepted vs {os.path.basename(base)}: "
                          f"{test.wins}W-{test.losses}L, {mid:+.0f} Elo "
                          f"[{lo:+.0f}, {hi:+.0f}]."))
        registry.register_model(os.path.basename(out),
                                path=os.path.relpath(out, ROOT), built_by="arbiter",
                                base=os.path.relpath(base, ROOT),
                                description=f"CEM arbiter, {mid:+.0f} Elo vs base")
        pr.log(f"ACCEPTED -> {os.path.relpath(out, ROOT)}")
        pr.log(f"rate it:  python -m kaggriculture.measure.elo --only {os.path.relpath(out, ROOT)}", 1)
    else:
        pr.log(f"{verdict}: the learned arbiter is not a significant improvement "
               f"on a flat vote. Keeping the base.", 1)
        pr.log("That is a result. It says the committee members are close enough "
               "in quality that weighting them does not pay -- make them more "
               "diverse (src/kaggriculture/train/select.py --spread-guard) before trying again.", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
