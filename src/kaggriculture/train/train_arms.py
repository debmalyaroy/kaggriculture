"""Train per-family counter-schedule arms by evolutionary search.

Stage 2 of the v22 build. Each arm retimes the base route's OWN sells --
nothing is created, resized, or borrowed from another farm's plan -- against
tapes of the specific opponent family it must beat, scored only by playing
full episodes. The genome is 28 integers: a turn-shift per (product, window),
windows starting at turn 192 (day 8) so every arm shares the identification
prefix and the live agent can switch between them without splicing.

    python -m kaggriculture.train.train_arms --arm tt --targets <tapes...> --guards <tapes...>
"""
from kaggriculture.paths import ROOT
import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
# Opponent team names include non-ASCII handles; a cp1252 console (a
# scheduled task without -X utf8) crashed the whole arm stage on a print
# (2026-08-20). Harden the entry point so a name can never abort training.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.engine.tape_runtime as tape_runtime  # noqa: E402

WORK = os.path.join(ROOT, "models", "v22", "arms")
PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT")
WINDOWS = ((192, 320), (320, 448), (448, 576), (576, 708))
SHIFT_MAX = 36
PREFIX_END = 192          # arms are identical before this turn, by construction


def apply_genome(route, genome):
    """Retime the base schedule's sells. Returns a new route list.

    Slot-aware: a turn's market queue holds 10 orders, and the runtime sorts
    sells first, so overflowing a turn silently drops the BUY/HIRE orders at
    the tail -- the mistake that cost -$1.4M/candidate in the first training
    attempt. A move that cannot find a turn with room within +/-6 turns of
    its target simply stays where it was.
    """
    out = [copy.deepcopy(t) for t in route]
    moves = []                                 # (target_turn, origin, item, qty)
    for wi, (lo, hi) in enumerate(WINDOWS):
        for pi, item in enumerate(PRODUCTS):
            shift = int(round(genome[wi * len(PRODUCTS) + pi]))
            if shift == 0:
                continue
            for t in range(lo, min(hi, len(out))):
                market = out[t].get("market") or []
                kept = []
                for o in market:
                    if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                            and o[1] == item):
                        nt = max(PREFIX_END, min(707, t + shift))
                        moves.append((nt, t, item, int(o[2])))
                    else:
                        kept.append(o)
                out[t]["market"] = kept

    def place(turn, item, qty):
        market = out[turn].setdefault("market", [])
        for o in market:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] == item:
                o[2] = int(o[2]) + qty
                return True
        if len(market) < 9:                    # leave headroom under the cap
            market.append(["SELL", item, qty])
            return True
        return False

    for nt, origin, item, qty in moves:
        placed = False
        for d in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            turn = nt + d
            if PREFIX_END <= turn <= 707 and place(turn, item, qty):
                placed = True
                break
        if not placed:
            place(origin, item, qty) or out[origin].setdefault(
                "market", []).append(["SELL", item, qty])
    return out


def render(route, path, label):
    import base64
    import zlib
    payload = base64.b85encode(zlib.compress(
        json.dumps(route, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    src = tape_runtime.TEMPLATE.format(
        label=label, route_id="arm", team="v22 arm", episode="-", seat=0,
        bank=0.0, opp_bank=0.0, built="2026-08-09", payload=payload,
        weed_catchup=8, impact_slots=True, mirror_tiebreak=False,
        floor_guard=False, endgame_pull=False, guard_ratio=0.0,
        swap_advance=False, sell_first=True, feed_pin=True,
        premium_lead=False, deposit_advance=False, floor_seller=False,
        branchpack="")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    return path


def _play(job):
    left, right, seed = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": 720, "seed": seed, "actTimeout": 60,
        "runTimeout": 100000})
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def evaluate(cands, targets, guards, seeds, workers):
    """Fitness per candidate: SCORE on targets, score-regression penalty on
    guards, margin only as an epsilon tie-break. Same seeds for every
    candidate (paired).

    Was `margin + 2500 if margin > 0`, which is margin-dominated: real margins
    run to +/-20,000+, so the win bonus was rounding error and the search
    optimised dollars. The ladder pays win/draw/loss (see src/win_metric.py:
    $1 and $10,000 are the same result), so score is the objective. Margin is
    kept at 1e-6 weight purely to order genomes that are TIED on score, where
    it carries a little robustness signal and cannot outvote a single win.
    """
    jobs, meta = [], []
    for ci, cand in enumerate(cands):
        for kind, opps in (("t", targets), ("g", guards)):
            for opp in opps:
                for seed in seeds:
                    jobs.append((cand, opp, seed)); meta.append((ci, kind, 0))
                    jobs.append((opp, cand, seed)); meta.append((ci, kind, 1))
    import kaggriculture.measure.win_metric as WM
    fit = [0.0] * len(cands)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (ci, kind, seat), (a, b) in zip(meta, ex.map(_play, jobs, chunksize=1)):
            mine, theirs = (a, b) if seat == 0 else (b, a)
            margin = mine - theirs
            sc = WM.score(mine, theirs)
            if kind == "t":
                fit[ci] += sc + 1e-6 * margin
            else:
                # guards: a LOST game is the regression that matters, weighted
                # 1.5x so a counter-schedule cannot buy target wins by
                # collapsing against the crowd.
                fit[ci] += 1.5 * (sc - 1.0) + 1e-6 * min(0.0, margin)
    return fit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--base", default="90525850_s0")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--guards", nargs="*", default=[])
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=310000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=10.0)
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    if os.path.exists(args.base):
        base = json.load(open(args.base, encoding="utf-8"))
    else:
        base = R.load_route(args.base)
    work = os.path.join(WORK, args.arm)
    os.makedirs(work, exist_ok=True)
    targets = [os.path.abspath(p) for p in args.targets]
    guards = [os.path.abspath(p) for p in args.guards]

    dim = len(PRODUCTS) * len(WINDOWS)
    rng = random.Random(17)
    state_path = os.path.join(work, "state.json")
    if os.path.exists(state_path):
        st = json.load(open(state_path, encoding="utf-8"))
        center, sigma, gen0 = st["center"], st["sigma"], st["gen"]
        best_ever = st.get("best_ever")
    else:
        center, sigma, gen0, best_ever = [0.0] * dim, args.sigma, 0, None

    # (1+lambda) hill-climb: the landscape is hostile far from the base, so
    # the centre only ever moves to a candidate that beat it on the same
    # seeds, and the step size shrinks when a generation finds nothing.
    for gen in range(gen0, args.gens):
        t0 = time.time()
        # Mutate a random subset of dimensions, not all 28 at once.
        genomes = [list(center)]
        for _ in range(args.pop - 1):
            g = list(center)
            for d in rng.sample(range(dim), k=rng.randint(1, 4)):
                g[d] = max(-SHIFT_MAX, min(SHIFT_MAX, g[d] + rng.gauss(0, sigma)))
            genomes.append(g)
        cands = []
        for i, g in enumerate(genomes):
            path = os.path.join(work, f"cand_{i}.py")
            render(apply_genome(base, g), path, f"arm {args.arm} g{gen} c{i}")
            cands.append(path)
        seeds = [args.seed0 + gen * 100 + k for k in range(args.seeds)]
        fit = evaluate(cands, targets, guards, seeds, args.workers)
        best_i = max(range(len(genomes)), key=lambda i: fit[i])
        if best_i != 0 and fit[best_i] > fit[0]:
            center = genomes[best_i]
        else:
            sigma = max(2.0, sigma * 0.85)
        row = {"gen": gen, "fit": fit[best_i], "genome": genomes[best_i]}
        if best_ever is None or row["fit"] > best_ever["fit"]:
            best_ever = row
        json.dump({"center": center, "sigma": sigma, "gen": gen + 1,
                   "best_ever": best_ever},
                  open(state_path, "w", encoding="utf-8"))
        # score-denominated fitness is O(n_games), not O(1e5) -- print
        # decimals or every generation logs as "0"
        print(f"[{args.arm}] gen {gen}: centre {fit[0]:>10.3f}  "
              f"best {fit[best_i]:>10.3f}  sigma {sigma:.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Render the winner (vs the zero genome as control) and save its route.
    final_route = apply_genome(base, best_ever["genome"])
    json.dump(final_route, open(os.path.join(work, "best_route.json"), "w",
                                encoding="utf-8"), separators=(",", ":"))
    render(final_route, os.path.join(work, "best.py"), f"arm {args.arm} best")
    render(base, os.path.join(work, "control.py"), "base control")
    print(f"[{args.arm}] done: best fitness {best_ever['fit']:.3f} (score units)  "
          f"genome {[round(x) for x in best_ever['genome']]}")


if __name__ == "__main__":
    main()
