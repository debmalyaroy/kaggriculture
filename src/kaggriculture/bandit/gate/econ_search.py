"""(1+lambda) BASE-TAPE ECONOMY search, scored FAITHFULLY vs the reactive killer
cluster on the Rust serve engine (no tape proxy, no kaggle_environments).

The 2026-09-20 verification showed the bandit beats most of the public field but
loses every world to tschinkel/tetsutani/nathanjacob/k0006 by $7-13k, and that
NO reactive config lever flips a game -- the gap is the base-tape ECONOMY. This
searches the economy directly: mutate the base tape (labour + purchases +
plantings + sells via sell_search.mutate field_ops), build each candidate as a
PURE tape (checkpoints=[], only the mandatory endgame rail, so dispatch/rails
never confound the economy signal), and score margin vs the killers over
world-diverse seeds x both seats -- the FAITHFUL reactive matchup. Margin (not
own bank) is the objective because improving our economy also feeds the shared
market and lifts the opponent's bank (the wall prior economy work hit); margin
nets that out.

Parallel across workers (box is memory-constrained -- default 5). Best tape ->
.local/econ_search/best.tape; NEVER overwrites models/bandit/base.tape (operator
installs a winner by hand after a faithful gate).

Run: NN_WORKERS=5 python -m kaggriculture.bandit.gate.econ_search --gens 5 --lam 6 --seeds 4
"""
import os, copy, json, random, argparse
import multiprocessing as mp
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
import kaggriculture.measure.eval_harness as EH
import kaggriculture.pipeline.sell_search as SS
from kaggriculture.bandit.build.build_rust_bandit import parse_tape

WORK = os.path.join(ROOT, ".local", "econ_search")
os.makedirs(WORK, exist_ok=True)
BASE_TAPE = os.path.join(ROOT, "models", "bandit", "base.tape")

# the cluster we LOSE to (verify_levers 2026-09-20). Beating THIS family is the goal.
KILLERS = [
    ("tschinkel", os.path.join(ROOT, "agents", "pub_tschinkel_2945.py")),
    ("tetsutani", os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_tetsutani_mirror.py")),
    ("nathanjacob", os.path.join(ROOT, ".local", "crown_panel", "refs", "2500-2700", "pub_nathanjacob_anticlone.py")),
    ("k0006", os.path.join(ROOT, ".local", "crown_panel", "refs", "2300-2500", "ref_k0006_2494.py")),
]


def pure_cfg(tape_path):
    """Config that plays ONLY the given tape + the mandatory endgame dump -- no
    dispatch, no optional rails -- so the eval measures the ECONOMY alone."""
    c = H.base_config()
    c["checkpoints"] = []
    c["d6_optional"] = True
    c["base_tape"] = tape_path
    for g in c["guardrails"]:
        if g.get("name") not in ("endgame",):
            g["on"] = False
    return c


def _cell(task):
    """(cand_id, stage, killer_name, killer_path, seeds) -> (cand_id, win, margin)."""
    cid, stage, kn, kp, ws = task
    try:
        res = H.eval_vs(EH.bandit_binary_agent(stage), kp, ws)
        return (cid, kn, res["win"], res["margin"], None)
    except Exception as e:
        return (cid, kn, None, None, f"{type(e).__name__}: {str(e)[:50]}")


def evaluate(cands, ws, killers, workers):
    """cands: list of (cid, rows). Build each as a pure tape, score vs killers.
    Returns {cid: {'win':.., 'margin':.., 'per':{kn:(win,margin)}}}."""
    stage_of = {}
    for cid, rows in cands:
        tp = os.path.join(WORK, f"{cid}.tape")
        SS.write_tape(rows, tp)
        H.build_agent(f"es_{cid}", pure_cfg(tp))
        stage_of[cid] = os.path.join(H.SCRATCH, f"es_{cid}_wingate")
    tasks = [(cid, stage_of[cid], kn, kp, ws) for cid, _ in cands for kn, kp in killers]
    agg = {cid: {"per": {}} for cid, _ in cands}
    with mp.Pool(workers, maxtasksperchild=4) as pool:
        for cid, kn, win, margin, err in pool.imap_unordered(_cell, tasks):
            if err:
                print(f"    ! {cid} vs {kn}: {err}", flush=True)
                continue
            agg[cid]["per"][kn] = (win, margin)
    for cid in agg:
        per = agg[cid]["per"]
        agg[cid]["win"] = float(np.mean([w for w, _ in per.values()])) if per else 0.0
        agg[cid]["margin"] = float(np.mean([m for _, m in per.values()])) if per else -1e9
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--lam", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--n-ops", type=int, default=6, help="mutations per candidate (bigger = more structural reach)")
    a = ap.parse_args()
    workers = int(os.environ.get("NN_WORKERS", "5"))
    rng = random.Random(1729)
    ws = H.world_seeds(a.seeds)
    killers = [(n, p) for n, p in KILLERS if os.path.exists(p)]
    print(f"ECON SEARCH  gens={a.gens} lam={a.lam}  killers={[n for n,_ in killers]}  "
          f"worlds={len(ws)} x2 seats  workers={workers}", flush=True)

    base = parse_tape(BASE_TAPE)[:719]
    agg = evaluate([("base", base)], ws, killers, workers)["base"]
    best, best_m, best_w = base, agg["margin"], agg["win"]
    print(f"gen0 BASE  win={best_w:.2f} margin={best_m:+.0f}  per={ {k:round(v[1]) for k,v in agg['per'].items()} }", flush=True)

    for g in range(1, a.gens + 1):
        cands = []
        for i in range(a.lam):
            m = SS.mutate(copy.deepcopy(best), rng, window=a.window, n_ops=a.n_ops, field_ops=True)
            cands.append((f"g{g}m{i}", m[:719]))
        res = evaluate(cands, ws, killers, workers)
        gi = max(cands, key=lambda c: res[c[0]]["margin"])[0]
        gm, gw = res[gi]["margin"], res[gi]["win"]
        tag = ""
        if gm > best_m + 1e-9:
            best, best_m, best_w = dict_rows(cands, gi), gm, gw
            SS.write_tape(best, os.path.join(WORK, "best.tape"))
            tag = "  <== NEW BEST (saved)"
        print(f"gen{g}  best_mutant={gi} win={gw:.2f} margin={gm:+.0f}   "
              f"(incumbent margin={best_m:+.0f}){tag}", flush=True)

    print(f"\nDONE. best win={best_w:.2f} margin={best_m:+.0f} vs base "
          f"-> {os.path.join(WORK, 'best.tape')}", flush=True)
    print("NOT installed. Gate faithfully vs the full public panel before adopting.", flush=True)


def dict_rows(cands, cid):
    for c, rows in cands:
        if c == cid:
            return rows
    return None


if __name__ == "__main__":
    main()
