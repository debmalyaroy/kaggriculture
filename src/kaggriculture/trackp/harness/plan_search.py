"""P1: build-time PLAN SEARCH -- edit the tape's production, score vs elite.

The critique (2026-09-05): every reactive layer modulates sells, but every
measured loss mechanism is PRODUCTION (inventory into peaks); and the Rust
engine idles as an evaluator when a deterministic public engine begs for
search. This is the planner: local edits to the base tape's PRODUCTION,
fitness = win rate against a fixed sample of >=2500 teams' recorded games
(the elite panel), hill-climbing generation by generation.

Differences from the GA that measured 0.000: edit operators are MINIMAL and
LOCAL (move/scale one purchase, not rewrite 92-131 actions); fitness is the
certified elite panel (not unguarded self-play); and the P2 oracle says the
prize is real (+19.3pp between best-single 0.767 and oracle 0.960).

Edit operators, biased to days 4-12 (the measured decision window):
  MOVE   shift one BUY_* market order to an adjacent day (earlier/later)
  SCALE  +-1 quantity on one BUY_SEED / BUY_ANIMAL / BUY_PRODUCT
  CLONE  duplicate one purchase order one day later (grow harder)
Illegal results are silent no-ops in the engine -- the fitness function
plays the REAL engine, so a broken edit scores as what it is.

    python src/trackp/harness/plan_search.py --gens 8 --pop 20 --cells 80
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from kaggriculture.trackp import routes_io as R                              # noqa: E402
import kaggriculture.pipeline.sell_search as SS  # noqa: E402

CELLS = os.path.join(ROOT, ".local", "hardband", "elite_cells.json")
WORK = os.path.join(ROOT, ".local", "hardband", "plansearch")
OUT = os.path.join(ROOT, "models", "trackp", "plan_search.json")
KAGG = os.path.join(ROOT, "rustengine", "kagg.exe")
BASE = "105370586_s0"
EDIT_LO, EDIT_HI = 4 * 24, 28 * 24      # widened to d28 (2026-09-05: the d20-29 bleed is where games are decided)
BUYS = ("BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT")


def mutate(acts, rng, bias_item=None):
    acts = copy.deepcopy(acts)
    def ok(o):
        if not (isinstance(o, list) and o):
            return False
        if o[0] in BUYS:
            return True
        # under an item bias, that item's SELLs are editable too (a sale
        # you can scale is a production target the engine will verify)
        return (bias_item is not None and o[0] == "SELL"
                and len(o) >= 2 and o[1] == bias_item)
    spots = [(i, j) for i in range(EDIT_LO, min(EDIT_HI, len(acts)))
             for j, o in enumerate(acts[i].get("market") or []) if ok(o)]
    if bias_item and rng.random() < 0.7:
        b = [(i, j) for (i, j) in spots
             if bias_item in str(acts[i]["market"][j])]
        if b:
            spots = b
    if not spots:
        return acts, "noop"
    op = rng.choice(("move", "scale", "clone"))
    i, j = rng.choice(spots)
    order = acts[i]["market"][j]
    if op == "move":
        d = rng.choice((-24, -12, 12, 24))
        k = max(EDIT_LO, min(len(acts) - 1, i + d))
        if len(acts[k].setdefault("market", [])) < 10:
            acts[k]["market"].append(copy.deepcopy(order))
            del acts[i]["market"][j]
    elif op == "scale" and len(order) >= 3:
        try:
            q = int(order[2])
            order[2] = max(1, q + rng.choice((-1, 1)))
        except (TypeError, ValueError):
            pass
    elif op == "clone":
        k = min(len(acts) - 1, i + rng.choice((12, 24)))
        if len(acts[k].setdefault("market", [])) < 10:
            acts[k]["market"].append(copy.deepcopy(order))
    return acts, op


def fitness(cand_tapes, panel, threads):
    """Win rate of each candidate tape over the elite panel (kagg batch)."""
    jobs, meta = [], []
    for ti, tp in enumerate(cand_tapes):
        for pi, (opp_tape, seed, es) in enumerate(panel):
            if es == 0:
                jobs.append(f"{seed}\t{opp_tape}\t{tp}")
            else:
                jobs.append(f"{seed}\t{tp}\t{opp_tape}")
            meta.append((ti, es))
    jp = os.path.join(WORK, "jobs.tsv")
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jobs))
    out = subprocess.run([KAGG, "batch", jp, str(threads)],
                         capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise SystemExit(f"kagg batch failed: {out.stderr[:300]}")
    score = defaultdict(float)
    cnt = defaultdict(int)
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        ti, es = meta[int(parts[0])]
        b0, b1 = float(parts[2]), float(parts[3])
        mine, theirs = (b1, b0) if es == 0 else (b0, b1)
        score[ti] += 1.0 if mine > theirs else (0.5 if mine == theirs else 0)
        cnt[ti] += 1
    return [score[t] / max(cnt[t], 1) for t in range(len(cand_tapes))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--cells", type=int, default=80)
    ap.add_argument("--confirm-cells", type=int, default=80,
                    help="fresh paired panel per accepted step; a mutant is "
                         "accepted only if it beats the incumbent on cells "
                         "NEITHER has selected on (the 2026-09-05 first run "
                         "climbed +3.3pp on its panel and 0.0 on holdout)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--bias-item", default=None,
                    help="bias 70%% of mutations to orders involving this "
                         "product (2026-09-05: WHEAT shortfall ~140 units "
                         "= the narrow-loss margin)")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    rng = random.Random(7)

    cells = json.load(open(CELLS, encoding="utf-8"))
    by = defaultdict(list)
    for c in cells:
        by[c["team"]].append(c)
    picks = [rng.choice(cs) for cs in by.values()]
    rng.shuffle(picks)
    picks = picks[:a.cells]
    panel = []
    for c in picks:
        ep, es = str(c["episode"]), int(c["elite_seat"])
        try:
            t = SS.write_tape(R.load_route(f"{ep}_s{es}"),
                              os.path.join(WORK, f"opp_{ep}.tape"))
        except Exception:                                      # noqa: BLE001
            continue
        panel.append((t, c["seed"], es))
    print(f"panel: {len(panel)} elite cells; base {a.base}", flush=True)

    best_acts = R.load_route(a.base)
    bt = SS.write_tape(best_acts, os.path.join(WORK, "best.tape"))
    best_fit = fitness([bt], panel, a.threads)[0]
    print(f"gen 0: base fitness {best_fit:.3f}", flush=True)

    history = [{"gen": 0, "fit": best_fit, "op": "base"}]
    for g in range(1, a.gens + 1):
        muts, ops = [], []
        for m in range(a.pop):
            acts, op = mutate(best_acts, rng, a.bias_item)
            muts.append(acts)
            ops.append(op)
        tapes = [SS.write_tape(x, os.path.join(WORK, f"m{m}.tape"))
                 for m, x in enumerate(muts)]
        fits = fitness(tapes, panel, a.threads)
        mi = max(range(len(fits)), key=lambda k: fits[k])
        print(f"gen {g}: best mutant {fits[mi]:.3f} ({ops[mi]}) "
              f"vs incumbent {best_fit:.3f}", flush=True)
        if fits[mi] > best_fit:
            # CONFIRM on a fresh paired panel before accepting
            crng = random.Random(1000 + g)
            cp = [crng.choice(cs) for cs in by.values()]
            crng.shuffle(cp)
            confirm = []
            for c in cp[:a.confirm_cells]:
                ep, es = str(c["episode"]), int(c["elite_seat"])
                try:
                    t = SS.write_tape(R.load_route(f"{ep}_s{es}"),
                                      os.path.join(WORK, f"c_{ep}.tape"))
                except Exception:                              # noqa: BLE001
                    continue
                confirm.append((t, c["seed"], es))
            cand_t = SS.write_tape(muts[mi],
                                   os.path.join(WORK, "confirm_cand.tape"))
            inc_t = SS.write_tape(best_acts,
                                  os.path.join(WORK, "confirm_inc.tape"))
            cf, incf = fitness([cand_t, inc_t], confirm, a.threads)
            print(f"       confirm panel ({len(confirm)}): mutant {cf:.3f} "
                  f"vs incumbent {incf:.3f} -> "
                  f"{'ACCEPT' if cf > incf else 'reject'}", flush=True)
            if cf > incf:
                best_fit = fits[mi]
                best_acts = muts[mi]
                history.append({"gen": g, "fit": best_fit, "op": ops[mi],
                                "confirm": [cf, incf]})
    out_route = os.path.join(WORK, "best_route.json")
    json.dump(best_acts, open(out_route, "w", encoding="utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"base": a.base, "final_fitness": best_fit,
               "history": history, "route": out_route,
               "panel_cells": len(panel)},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nfinal fitness {best_fit:.3f} -> {os.path.relpath(OUT, ROOT)}")
    print("NEXT: embed best_route.json as the chassis base (route swap), "
          "then the full gate chain. NEVER submits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
