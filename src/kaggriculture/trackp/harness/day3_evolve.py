"""Day-3 fork: evolve a per-FIRST-SHOP economy continuation from t=74.

Field intel (2026-09-06/07): every top brancher forks at BOTH unlock
turns; our economy commits once, at day 6. The first shop is visible from
t=73, and days 3-6 hold the land/herd/expansion commitments -- the
biggest lever the router doesn't touch yet.

REALIZED-WORLD GROUPING (2026-09-07 rewrite): shop unlocks share the
per-day RNG with weeds, so the world depends on the seed AND both action
streams AND seat -- the PASS-drive seed bank labels worlds real play does
not visit. Classes here are therefore formed from cells (opp, seed, seat)
whose REALIZED first shop (under our base prefix, fixed through t=74 for
every mutant) matches; fitness is batch_eval_cells over exactly those
cells. The first unlock lands at t~73 < 74, so the class label is stable
across mutants by construction.

Per realized-first-shop class (8 classes):
  start   = our base route's own continuation [74:719];
  mutate  = sell_search ops with the window biased into the commitment
            span (74-240), prefix [0:74] restored after every mutation;
  fitness = mean W/L over the class's realized cells;
  accept  = strictly better score than the parent.

Output: models/trackp/day3_branches.json {SHOP: {"score", "base_score",
"cells", "cont": [...645 actions...]}} -- the builder embeds branches
only where score beats base by MIN_EDGE. Runtime rule: a firing day-3
branch OWNS the game (pair-tails are skipped -- they assume the clone
prefix through step 145, which a day-3 fork abandons).

    python src/trackp/harness/day3_evolve.py --gens 14 --lam 8
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import copy
import json
import os
import random
import re
import sys
import zlib
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.pipeline.sell_search as SS  # noqa: E402
from kaggriculture.trackp import routes_io as R                              # noqa: E402
from kaggriculture.trackp.harness import realized as RW        # noqa: E402

WORK = os.path.join(ROOT, ".local", "day3_evolve")
SPLIT = 74
MIN_EDGE = 0.08


def our_route():
    src = open(os.path.join(ROOT, "agents", "v45.0_bandit.py"),
               encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    return json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))


def build_panel():
    os.makedirs(WORK, exist_ok=True)
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    donors = sorted(hits, key=lambda h: -(h.get("bank") or 0))[:3]
    panel = []
    for j, h in enumerate(donors):
        try:
            panel.append(SS.write_tape(R.load_route(h["id"]),
                                       os.path.join(WORK, f"p{j}.tape")))
        except Exception:                                      # noqa: BLE001
            pass
    idx = R.load_index()
    added = 0
    for k, v in sorted(idx["routes"].items(),
                       key=lambda t: -(t[1].get("bank") or 0)):
        if (v.get("engine") == "1.32.7" and v.get("won")
                and str(v.get("date", "")) >= "2026-09-01"):
            try:
                panel.append(SS.write_tape(
                    R.load_route(k), os.path.join(WORK, f"q{added}.tape")))
                added += 1
            except Exception:                                  # noqa: BLE001
                continue
        if added >= 2:
            break
    import glob as _glob
    for _sp in sorted(_glob.glob(os.path.join(ROOT, '.local', 'candidates', 'strong_tapes', '*.tape'))):
        panel.append(_sp)   # high-scoring live-opponent sparring tapes
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=14)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--sample-seeds", type=int, default=300,
                    help="seeds sampled to build the realized map")
    ap.add_argument("--cells-per-class", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "day3_branches.json"))
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    route = our_route()
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    all_seeds = sorted({s for ss in bank.values() for s in ss})
    rng = random.Random(19)
    seeds = rng.sample(all_seeds, min(a.sample_seeds, len(all_seeds)))
    panel = build_panel()
    print(f"panel: {len(panel)} tapes; building realized map over "
          f"{len(seeds)} seeds x {len(panel)} opps x 2 seats", flush=True)
    rmap = RW.build_map(route[:719], panel, seeds)
    by_shop = defaultdict(list)
    for (oi, s, seat), world in rmap.items():
        by_shop[world.split("|")[0]].append((panel[oi], s, seat))
    print("realized first-shop class sizes: "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_shop.items())),
          flush=True)

    out = {}
    outp = a.out
    for shop, cells in sorted(by_shop.items(), key=lambda kv: kv[0]):
        if len(cells) < 10:
            print(f"{shop:<20s} SKIP ({len(cells)} cells)", flush=True)
            continue
        cells = rng.sample(cells, min(a.cells_per_class, len(cells)))
        parent = copy.deepcopy(route[:719])
        pt = SS.write_tape(parent, os.path.join(WORK, "parent.tape"))
        base_score = SS.batch_eval_cells([pt], cells, 6)[0]["score"]
        best, best_score = parent, base_score
        for g in range(a.gens):
            muts = []
            for _ in range(a.lam):
                m = SS.mutate(best, rng, window=80, field_ops=True)
                m[:SPLIT] = copy.deepcopy(route[:SPLIT])   # prefix guard
                muts.append(m)
            paths = [SS.write_tape(m, os.path.join(WORK, f"m{i}.tape"))
                     for i, m in enumerate(muts)]
            res = SS.batch_eval_cells(paths, cells, 6)
            gi = max(range(len(muts)), key=lambda i: res[i]["score"])
            if res[gi]["score"] > best_score:
                best, best_score = muts[gi], res[gi]["score"]
        tag = ""
        if best_score >= base_score + MIN_EDGE:
            out[shop] = {"score": best_score, "base_score": base_score,
                         "cells": len(cells),
                         "cont": best[SPLIT:719]}
            tag = "  <== BRANCH IN"
        print(f"{shop:<20s} base {base_score:.3f} -> best "
              f"{best_score:.3f}  ({len(cells)} cells){tag}", flush=True)
        json.dump(out, open(outp, "w", encoding="utf-8"))
    json.dump(out, open(outp, "w", encoding="utf-8"))
    print(f"\n{len(out)} day-3 branches -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
