"""P4.2 -- CMA-ES over the planner's PARAMS, fitness on the Rust league.

Own compact CMA-ES (rank-1 + rank-mu updates, no dependencies beyond numpy).
Fitness of a candidate = mean gap-shaped reward over a FIXED panel of anchor
games (same anchors, same seeds every generation -- candidates stay
comparable; the panel rotates only between runs). Elite priors seed x0.

Winners re-validate on the OFFICIAL engine (sim-to-real) before anything
downstream trusts them. Output: models/trackp/params_searched.json.

Usage: python src/trackp/search.py [--gens 40] [--panel 8] [--jobs 8]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

try:
    from . import common, league, rollouts
except ImportError:
    import sys
    from kaggriculture.trackp import common, league, rollouts

# name, lo, hi, integer?
SPACE = [
    ("labour_max", 4, 13, True),
    ("labour_ramp_day", 1, 6, True),
    ("labour_base", 2, 8, True),
    ("labour_per_tiles", 4.0, 20.0, False),
    ("plant_value_margin", 1.0, 1.6, False),
    ("plants_per_unit", 3.0, 12.0, False),
    ("plant_budget_turn", 4, 14, True),
    ("idle_cash_target", 100.0, 1500.0, False),
    ("invest_land_day_ne", 1, 6, True),
    ("invest_land_day_sw", 3, 12, True),
    ("invest_land_day_se", 6, 18, True),
    ("max_coops", 1, 4, True),
    ("max_pastures", 1, 6, True),
    ("animal_start_day", 1, 6, True),
    ("animal_last_day", 12, 24, True),
    ("sell_horizon_days", 1.0, 6.0, False),
    ("sell_hold_gain", 1.0, 1.20, False),
    ("sell_chunk", 5, 30, True),
    ("shed_pressure", 50, 95, True),
    ("terminal_day", 25, 28, True),
    ("wheat_floor", 0, 8, True),
    ("fert_sell_min", 0, 12, True),
    ("water_growth_bonus", 1.0, 2.0, False),
    ("care_value", 0.2, 1.2, False),
]
DIM = len(SPACE)


def decode(z: np.ndarray) -> dict:
    """z in R^DIM (unbounded) -> PARAMS overrides via sigmoid squash."""
    out = {}
    for i, (name, lo, hi, is_int) in enumerate(SPACE):
        v = lo + (hi - lo) / (1.0 + math.exp(-float(z[i])))
        out[name] = int(round(v)) if is_int else round(v, 3)
    return out


def encode_defaults() -> np.ndarray:
    """Template defaults -> z (inverse sigmoid), the elite-prior start."""
    from kaggriculture.trackp import planner_template as tpl
    z = np.zeros(DIM)
    for i, (name, lo, hi, _) in enumerate(SPACE):
        v = float(tpl.PARAMS[name])
        frac = min(0.98, max(0.02, (v - lo) / (hi - lo)))
        z[i] = math.log(frac / (1.0 - frac))
    return z


def make_panel(n_games: int, seed: int = 20260815) -> list:
    anchors = league.load_anchors()
    if not anchors:
        raise SystemExit("no anchors -- run league.py --build-anchors first")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(anchors))[:n_games]
    return [anchors[int(i)] for i in idx]


def fitness_jobs(cands: list, panel: list) -> list:
    jobs = []
    for ci, params in enumerate(cands):
        for a in panel:
            jobs.append({"seed": a["seed"], "me": {"params": params},
                         "opp": {"kind": "tape", "tape": a["tape"]},
                         "_ci": ci})
    return jobs


def evaluate(runner: rollouts.Runner, cands: list, panel: list) -> np.ndarray:
    jobs = fitness_jobs(cands, panel)
    metas = [(j["_ci"]) for j in jobs]
    for j in jobs:
        j.pop("_ci")
    res = runner.run(jobs)
    fit = np.zeros(len(cands))
    cnt = np.zeros(len(cands))
    for ci, r in zip(metas, res):
        if "banks" in r:
            # search fitness: the standard gap reward PLUS a wide gap term so
            # the surface has gradient even while every game is a >$10k loss.
            # Still GAP-shaped -- a bank term would just ride the tide.
            gap = r["banks"][0] - r["banks"][1]
            wide = max(-1.0, min(1.0, gap / 80000.0))
            fit[ci] += r["reward"] + 0.5 * wide
            cnt[ci] += 1
    cnt[cnt == 0] = 1
    return fit / cnt


def cma_es(gens: int, panel_n: int, jobs: int, pop: int = 0,
           sigma0: float = 0.6) -> dict:
    pop = pop or (4 + int(3 * math.log(DIM)))
    mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cc = (4 + mueff / DIM) / (DIM + 4 + 2 * mueff / DIM)
    cs = (mueff + 2) / (DIM + mueff + 5)
    c1 = 2 / ((DIM + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((DIM + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (DIM + 1)) - 1) + cs
    chiN = math.sqrt(DIM) * (1 - 1 / (4 * DIM) + 1 / (21 * DIM ** 2))

    m = encode_defaults()
    sigma = sigma0
    C = np.eye(DIM)
    pc = np.zeros(DIM)
    ps = np.zeros(DIM)
    rng = np.random.default_rng(7)

    panel = make_panel(panel_n)
    runner = rollouts.Runner(jobs)
    history = []
    best = {"fitness": -1e9, "params": decode(m)}
    t0 = time.time()
    try:
        for g in range(gens):
            A = np.linalg.cholesky(C + 1e-10 * np.eye(DIM))
            Z = rng.standard_normal((pop, DIM))
            X = m + sigma * Z @ A.T
            cands = [decode(x) for x in X]
            fit = evaluate(runner, cands, panel)
            order = np.argsort(-fit)
            if fit[order[0]] > best["fitness"]:
                best = {"fitness": float(fit[order[0]]),
                        "params": cands[order[0]], "gen": g}
            xw = (w[:, None] * X[order[:mu]]).sum(axis=0)
            zw = (w[:, None] * Z[order[:mu]]).sum(axis=0)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * zw
            hsig = (np.linalg.norm(ps)
                    / math.sqrt(1 - (1 - cs) ** (2 * (g + 1)))
                    / chiN) < (1.4 + 2 / (DIM + 1))
            pc = (1 - cc) * pc + (math.sqrt(cc * (2 - cc) * mueff)
                                  * (xw - m) / sigma if hsig else 0)
            artmp = (X[order[:mu]] - m) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc)
                         + (0 if hsig else 1) * cc * (2 - cc) * C)
                 + cmu * (artmp.T * w) @ artmp)
            m = xw
            sigma *= math.exp((cs / damps)
                              * (np.linalg.norm(ps) / chiN - 1))
            history.append({"gen": g, "best": float(fit[order[0]]),
                            "mean": float(fit.mean()),
                            "sigma": float(sigma)})
            print(f"gen {g}: best {fit[order[0]]:.3f} "
                  f"mean {fit.mean():.3f} sigma {sigma:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        runner.close()
    out = {"best": best, "history": history, "pop": pop, "gens": gens,
           "panel": [{k: a[k] for k in ("episode", "seat", "bank", "team")}
                     for a in panel],
           "space": [s[0] for s in SPACE]}
    path = os.path.join(common.MODELS, "params_searched.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    pp = os.path.join(common.MODELS, "params_best.json")
    with open(pp, "w", encoding="utf-8") as fh:
        json.dump(best["params"], fh, indent=1)
    print("saved", path)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--panel", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--pop", type=int, default=0)
    a = ap.parse_args()
    cma_es(a.gens, a.panel, a.jobs, a.pop)
