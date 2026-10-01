"""Cross-entropy-method policy search against a chosen opponent set.

Why this and not `tune.py`: coordinate descent optimises one knob at a time
against a frozen copy of *ourselves*, so it can only ever find the best version
of the strategy it started with, and it grades that strategy against an
opponent with the same blind spots. The knobs in this agent are not separable
either -- the feed runway, the herd size and the cash crop are one decision
about cash flow wearing three names.

CEM fixes both. It samples whole parameter vectors from a Gaussian, keeps the
elite fraction, refits the Gaussian to them, and repeats: a derivative-free
policy search that handles coupled knobs, and it is scored against whatever
opponents you point it at -- including the recorded top-ladder tapes, which is
the only local signal that correlates with the leaderboard.

    python -m kaggriculture.train.optimize --vs opponents/tape_90036815_s1.py --generations 8
    python -m kaggriculture.train.optimize --resume --budget-min 45
    python -m kaggriculture.train.optimize --objective margin --popsize 16 --seeds 4

State lives in .local/cem/<run>.json and every run is resumable. The best
vector is written to --out as a normal agent file (PARAMS block replaced), so
it can be evaluated, registered and submitted like any other agent.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

import kaggriculture.pipeline.params as paramio  # noqa: E402

# (low, high, kind). "int" rounds; "float" does not. Bounds are the search box,
# not a claim about the optimum -- CEM will happily sit against an edge, which
# is itself information that the box was drawn too small.
SPACE = {
    # cash flow -- the coupled group that decides the opening
    "feed_runway_days":   (1.0, 6.0, "float"),
    "income_gap_cap":     (2.0, 12.0, "float"),
    "income_gap_cash":    (100.0, 2000.0, "float"),
    "reserve_base":       (0.0, 800.0, "float"),
    "reserve_per_tile":   (0.0, 12.0, "float"),
    "cash_crop_gap":      (1.0, 8.0, "float"),
    "cash_crop_tiles":    (0, 18, "int"),
    "cash_crop_cash":     (200.0, 6000.0, "float"),
    "liquid_target":      (500.0, 5000.0, "float"),
    "liquid_weight":      (0.0, 0.6, "float"),

    # herd
    "max_herd":           (8, 20, "int"),
    "animals_per_turn":   (1, 6, "int"),
    "animal_pipeline":    (0, 8, "int"),
    "animal_cash_buffer": (0.0, 1500.0, "float"),
    "target_cow":         (0, 14, "int"),
    "target_sheep":       (0, 14, "int"),
    "target_goose":       (0, 8, "int"),

    # portfolio
    "target_strawberry":  (0, 50, "int"),
    "target_melon":       (0, 24, "int"),
    "target_tomato":      (0, 20, "int"),
    "target_carrot":      (0, 20, "int"),
    "target_wheat":       (0, 20, "int"),
    "feed_self_frac":     (0.0, 1.2, "float"),
    "filler_max":         (0, 50, "int"),
    "min_tile_rate":      (0.0, 30.0, "float"),

    # market model
    "outlook_weight":     (0.0, 2.0, "float"),
    "outlook_horizon":    (0.1, 1.0, "float"),
    "mirror_weight":      (0.0, 2.0, "float"),
    "demand_horizon":     (2.0, 30.0, "float"),
    "fert_income_weight": (0.0, 1.2, "float"),
    "fert_coverage":      (0.0, 1.0, "float"),
    "fert_opportunity":   (0.0, 2.0, "float"),
    "fert_stock":         (0, 20, "int"),
    "sell_chunk":         (2, 30, "int"),
    "sell_orders":        (1, 8, "int"),
    "shed_pressure":      (40, 95, "int"),
    "dump_day":           (24, 29, "int"),
    "feed_buffer_days":   (1.0, 6.0, "float"),

    # labour
    "hands_max":          (6, 18, "int"),
    "hands_max_late":     (6, 20, "int"),
    "hands_min":          (0, 8, "int"),
    "hands_slack":        (0, 6, "int"),
    "hire_cash_frac":     (0.05, 0.9, "float"),
    "capacity_util":      (0.5, 1.2, "float"),
    "cost_per_crop_day":  (1.0, 4.0, "float"),
    "cost_per_animal_day": (2.0, 8.0, "float"),

    # task weights
    "travel_weight":      (0.5, 9.0, "float"),
    "care_weight":        (0.0, 2.5, "float"),
    "fert_collect_weight": (0.0, 2.5, "float"),
    "fert_weight":        (0.5, 5.0, "float"),
    "feed_safe_discount": (0.0, 1.0, "float"),
    "poach_penalty":      (0.3, 1.0, "float"),
    "land_min_used":      (0.2, 0.95, "float"),
    "land_max":           (0, 3, "int"),
}

DOC_TEMPLATE = """Kaggriculture agent -- CEM-optimised.

GENERATED FILE -- do not hand-edit. This is agents/v1_heuristic.py with a
PARAMS block found by cross-entropy-method policy search (src/kaggriculture/train/optimize.py),
scored on paired both-seat matches against {opponents}.

To change behaviour, edit agents/v1_heuristic.py and re-run the search.
"""


BUILTINS = {"pass", "random", "starter"}


def clip(value, lo, hi, kind):
    value = max(lo, min(hi, value))
    return int(round(value)) if kind == "int" else float(value)


def play(job):
    """(left, right, seed) -> (left_bank, right_bank). Runs in a worker."""
    left, right, seed, steps = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def _resolve(spec):
    return spec if spec in BUILTINS else os.path.abspath(spec)


class Run:
    def __init__(self, args):
        self.args = args
        self.base_path = os.path.abspath(args.base)
        self.base_params = paramio.load(self.base_path)
        self.knobs = [k for k in (args.knobs or list(SPACE)) if k in SPACE]
        self.work = os.path.join(ROOT, ".local", "cem")
        os.makedirs(self.work, exist_ok=True)
        self.state_path = os.path.join(self.work, f"{args.run}.json")
        self.state = self._load() if args.resume else None
        if self.state is None:
            mean, sigma = {}, {}
            for k in self.knobs:
                lo, hi, kind = SPACE[k]
                cur = self.base_params.get(k, (lo + hi) / 2.0)
                try:
                    cur = float(cur)
                except (TypeError, ValueError):
                    cur = (lo + hi) / 2.0
                mean[k] = cur
                sigma[k] = args.sigma * (hi - lo)
            self.state = {"mean": mean, "sigma": sigma, "gen": 0,
                          "best": None, "best_score": None, "history": [],
                          "knobs": self.knobs, "opponents": args.vs}

    def _load(self):
        if not os.path.exists(self.state_path):
            return None
        with open(self.state_path, encoding="utf-8") as fh:
            st = json.load(fh)
        self.knobs = st.get("knobs", self.knobs)
        return st

    def save(self):
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=1)

    def sample(self, rng):
        out = {}
        for k in self.knobs:
            lo, hi, kind = SPACE[k]
            v = rng.gauss(self.state["mean"][k], max(1e-9, self.state["sigma"][k]))
            out[k] = clip(v, lo, hi, kind)
        return out

    def materialise(self, overrides, tag):
        params = dict(self.base_params)
        params.update(overrides)
        path = os.path.join(self.work, f"cand_{tag}.py")
        paramio.write(self.base_path, path, params,
                      header=f"# CEM candidate {tag} -- generated by src/kaggriculture/train/optimize.py")
        return path

    def score(self, candidates, pool):
        """Paired evaluation: every candidate meets every opponent on every seed
        in both seats, so seat and seed noise cancel inside a generation."""
        jobs, meta = [], []
        for ci, path in enumerate(candidates):
            for opp in self.args.vs:
                o = _resolve(opp)
                for s in range(self.args.seeds):
                    seed = self.args.seed0 + s
                    jobs.append((path, o, seed, self.args.steps))
                    meta.append((ci, 0))
                    jobs.append((o, path, seed, self.args.steps))
                    meta.append((ci, 1))
        results = list(pool.map(play, jobs))
        rows = {ci: [] for ci in range(len(candidates))}
        for (ci, seat), (r0, r1) in zip(meta, results):
            mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
            rows[ci].append((mine, theirs))
        scores = []
        for ci in range(len(candidates)):
            data = rows[ci]
            wins = sum(1 for m, t in data if m > t)
            margin = statistics.mean(m - t for m, t in data)
            bank = statistics.mean(m for m, _t in data)
            win_rate = wins / max(1, len(data))
            if self.args.objective == "margin":
                score = margin
            elif self.args.objective == "bank":
                score = bank
            else:                       # win rate, margin as the tie-break
                score = win_rate * 1e6 + margin
            scores.append({"i": ci, "score": score, "win": win_rate,
                           "margin": margin, "bank": bank})
        return scores

    def generation(self, pool, rng):
        gen = self.state["gen"]
        cands, paths = [], []
        # The incumbent mean is always in the population: CEM can otherwise
        # drift off a good vector on one unlucky generation.
        cands.append({k: clip(self.state["mean"][k], *SPACE[k]) for k in self.knobs})
        for i in range(self.args.popsize - 1):
            cands.append(self.sample(rng))
        for i, c in enumerate(cands):
            paths.append(self.materialise(c, f"{self.args.run}_{gen}_{i}"))

        scores = self.score(paths, pool)
        scores.sort(key=lambda r: -r["score"])
        n_elite = max(2, int(round(self.args.elite * len(cands))))
        elite = [cands[r["i"]] for r in scores[:n_elite]]

        for k in self.knobs:
            lo, hi, kind = SPACE[k]
            vals = [float(e[k]) for e in elite]
            self.state["mean"][k] = statistics.mean(vals)
            spread = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            floor = self.args.sigma_floor * (hi - lo)
            self.state["sigma"][k] = max(floor, spread)

        top = scores[0]
        if self.state["best_score"] is None or top["score"] > self.state["best_score"]:
            self.state["best_score"] = top["score"]
            self.state["best"] = cands[top["i"]]
        self.state["history"].append({
            "gen": gen, "best_win": top["win"], "best_margin": top["margin"],
            "best_bank": top["bank"],
            "mean_score": statistics.mean(r["score"] for r in scores)})
        self.state["gen"] = gen + 1
        self.save()
        return top


def main():
    # The ladder's engine, or nothing this prints means anything.
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "agents", "v1_heuristic.py"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "v9_cem.py"))
    ap.add_argument("--vs", nargs="+", default=None,
                    help="opponents; default = every tape in opponents/")
    ap.add_argument("--knobs", nargs="*", default=None)
    ap.add_argument("--run", default="main")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--elite", type=float, default=0.3)
    ap.add_argument("--sigma", type=float, default=0.20, help="initial sigma, as a fraction of the box")
    ap.add_argument("--sigma-floor", type=float, default=0.03)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=8100)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--objective", choices=("win", "margin", "bank"), default="win")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rng", type=int, default=7)
    args = ap.parse_args()

    if not args.vs:
        import glob
        tapes = sorted(glob.glob(os.path.join(ROOT, "opponents", "tape_*_s*.py")))
        args.vs = tapes[:2] or ["starter"]

    if os.path.abspath(args.out) == os.path.abspath(args.base):
        raise SystemExit("--out must not be --base (the search re-reads the base each trial)")

    run = Run(args)
    rng = random.Random(args.rng)
    print(f"CEM over {len(run.knobs)} knobs, pop {args.popsize}, "
          f"{args.seeds} seeds x 2 seats x {len(args.vs)} opponent(s) "
          f"= {args.popsize * args.seeds * 2 * len(args.vs)} matches/generation")
    for o in args.vs:
        print(f"  vs {o}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(args.generations):
            if args.budget_min and (time.time() - t0) / 60.0 > args.budget_min:
                print("budget reached")
                break
            top = run.generation(pool, rng)
            gen = run.state["gen"]
            print(f"gen {gen:>3}  best win {top['win'] * 100:>5.1f}%  "
                  f"margin {top['margin']:>+10,.0f}  bank {top['bank']:>10,.0f}  "
                  f"({(time.time() - t0) / 60:.1f} min)")

    if run.state["best"]:
        params = dict(run.base_params)
        params.update(run.state["best"])
        paramio.write(run.base_path, args.out, params,
                      header="# best CEM vector -- src/kaggriculture/train/optimize.py",
                      module_doc=DOC_TEMPLATE.format(
                          opponents=", ".join(os.path.basename(o) for o in args.vs)))
        print(f"\nbest vector -> {args.out}")
        for k in sorted(run.state["best"]):
            before = run.base_params.get(k)
            after = run.state["best"][k]
            if before != after:
                print(f"  {k:<22} {before} -> {after}")


if __name__ == "__main__":
    main()
