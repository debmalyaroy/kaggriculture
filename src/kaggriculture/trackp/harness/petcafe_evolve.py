"""Targeted PET_CAFE-world economy evolution (2026-09-07).

Live diagnosis: v47.1's base economy banks ~43k in PET_CAFE (animal)
worlds where the field's winners bank 65-94k -- PET_CAFE is in 4 of the 5
biggest live losses. Action diff vs a winner: we issue 423 SELL orders to
their 234 (we over-sell into marginal pricing) and build 1 coop to their 4
(a smaller animal operation). The day-6 branch set does not cover
PET_CAFE|PET_CAFE.

This evolves our base continuation from t=74 (day 3, so coop/herd
commitments are in range) against the REAL PET_CAFE winners extracted
from the loss replays, over cells whose world REALIZES with PET_CAFE as
the first shop under our prefix vs each winner (realized.py -- the world
is made by play, not the seed). field_ops lets mutation add BUY/BUILD and
re-time sells; prefix [0:74] restored every mutation so the splice is
exact and the branch keys on the observed first shop.

Output: models/trackp/petcafe_branch.json
  {"PET_CAFE": {"score","base_score","cells","cont":[...645...]}}
merged into day3_branches.json by the caller (day-3 fork owns the game).

    python src/trackp/harness/petcafe_evolve.py --gens 30 --lam 10
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.pipeline.sell_search as SS  # noqa: E402
import realized as RW                                          # noqa: E402
from day3_evolve import our_route                              # noqa: E402

WORK = os.path.join(ROOT, ".local", "petcafe_evolve")
TGT = os.path.join(ROOT, ".local", "petcafe_targets")
SPLIT = 74


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--lam", type=int, default=10)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "petcafe_branch.json"))
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    route = our_route()[:719]
    targets = json.load(open(os.path.join(TGT, "targets.json"),
                             encoding="utf-8"))
    panel = [t["tape"] for t in targets]
    seeds = json.load(open(os.path.join(TGT, "pet_seeds.json"),
                           encoding="utf-8"))
    print(f"panel: {len(panel)} PET_CAFE winners; {len(seeds)} seeds",
          flush=True)
    # realized map vs the WINNERS (world depends on the opponent)
    rmap = RW.build_map(route, panel, seeds, progress=False)
    cells = [(panel[oi], s, seat) for (oi, s, seat), w in rmap.items()
             if w.split("|")[0] == "PET_CAFE"]
    rng = random.Random(23)
    if len(cells) > a.cells:
        cells = rng.sample(cells, a.cells)
    print(f"PET_CAFE-first cells vs winners: {len(cells)}", flush=True)
    if len(cells) < 8:
        print("too few PET_CAFE cells; abort")
        return 1

    pt = SS.write_tape(route, os.path.join(WORK, "parent.tape"))
    base_score = SS.batch_eval_cells([pt], cells, 6)[0]
    print(f"base vs PET_CAFE winners: score {base_score['score']:.3f} "
          f"margin {base_score['margin']:,.0f}", flush=True)
    best, best_score, best_margin = route, base_score["score"], \
        base_score["margin"]
    for g in range(a.gens):
        muts = []
        for _ in range(a.lam):
            m = SS.mutate(best, rng, window=120, field_ops=True)
            m[:SPLIT] = copy.deepcopy(route[:SPLIT])
            muts.append(m)
        paths = [SS.write_tape(m, os.path.join(WORK, f"m{i}.tape"))
                 for i, m in enumerate(muts)]
        res = SS.batch_eval_cells(paths, cells, 6)
        gi = max(range(len(muts)),
                 key=lambda i: (res[i]["score"], res[i]["margin"]))
        if (res[gi]["score"], res[gi]["margin"]) > (best_score, best_margin):
            best, best_score, best_margin = \
                muts[gi], res[gi]["score"], res[gi]["margin"]
            print(f"gen {g:>2} score {best_score:.3f} margin "
                  f"{best_margin:,.0f}  <== improve", flush=True)
        elif g % 5 == 0:
            print(f"gen {g:>2} (best {best_score:.3f} / "
                  f"{best_margin:,.0f})", flush=True)
    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out, encoding="utf-8"))
    out["PET_CAFE"] = {"score": best_score, "base_score": base_score["score"],
                       "base_margin": base_score["margin"],
                       "margin": best_margin, "cells": len(cells),
                       "cont": best[SPLIT:719]}
    json.dump(out, open(a.out, "w", encoding="utf-8"))
    print(f"\nPET_CAFE branch: base {base_score['score']:.3f} -> "
          f"{best_score:.3f} (margin {base_score['margin']:,.0f} -> "
          f"{best_margin:,.0f}) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
