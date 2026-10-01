"""Day-6 fork: evolve a per-WORLD economy continuation from t=146.

Second level of the day-3 fork (day3_evolve.py). Both shops are realized
by t~145, which pins the WORLD; the splice lands at t=146, so mutants
(which only touch steps >= 146) can never shift the world they are being
selected for.

REALIZED-WORLD GROUPING (2026-09-07 rewrite): the PASS-drive seed bank
does not label the worlds real play visits (shop unlocks share the
per-day RNG with weeds). Cells (opp, seed, seat) are grouped by the world
REALIZED under the parent prefix actually played there:

  parent prefix = route[:74] + day3_branch[s1].cont[:72]  (class has a
                  day-3 branch)  else  route[:146];
  the second shop must be re-measured under the DAY-3 PARENT (the branch
  rewrites steps 74..145, which shifts the RNG draws that pick shop 2);
  start   = parent's own continuation [146:719];
  mutate  = sell_search ops, prefix [0:146] restored after every
            mutation;
  fitness = mean W/L over the world's realized cells (batch_eval_cells);
  accept  = strictly better score than the parent.

Output: models/trackp/day6_branches.json
  {WORLD("shop1|shop2"): {"score", "base_score", "parent": "d3"|"base",
                          "cells", "cont": [...573 actions...]}}
The builder splices a day-6 branch ONLY when its recorded parent matches
the prefix actually played (d3 branch fired for shop1, or base route).
Runtime precedence: day-3 (74) > family counter (122) > day-6 (146) >
pair-tails.

    python src/trackp/harness/day6_evolve.py --gens 12 --lam 8
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.pipeline.sell_search as SS  # noqa: E402
import realized as RW                                          # noqa: E402

WORK = os.path.join(ROOT, ".local", "day6_evolve")
D3SPLIT = 74
SPLIT = 146
MIN_EDGE = 0.08

from day3_evolve import build_panel, our_route                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--sample-seeds", type=int, default=300)
    ap.add_argument("--cells-per-world", type=int, default=16)
    ap.add_argument("--min-cells", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "day6_branches.json"))
    ap.add_argument("--d3", default=os.path.join(
        ROOT, "models", "trackp", "day3_branches.json"))
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    route = our_route()
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    all_seeds = sorted({s for ss in bank.values() for s in ss})
    rng = random.Random(37)
    seeds = rng.sample(all_seeds, min(a.sample_seeds, len(all_seeds)))
    d3 = json.load(open(a.d3, encoding="utf-8"))         if os.path.exists(a.d3) else {}
    panel = build_panel()
    print(f"panel: {len(panel)} tapes; {len(d3)} day-3 branches; "
          f"base realized map over {len(seeds)} seeds", flush=True)
    base_map = RW.build_map(route[:719], panel, seeds)
    cells_by_s1 = defaultdict(list)
    for (oi, s, seat), world in base_map.items():
        cells_by_s1[world.split("|")[0]].append(((oi, s, seat), world))

    out = {}
    outp = a.out
    for s1, tagged in sorted(cells_by_s1.items(), key=lambda kv: kv[0]):
        br = d3.get(s1)
        if br:
            parent = (copy.deepcopy(route[:D3SPLIT])
                      + copy.deepcopy(br["cont"]))[:719]
            parent_tag = "d3"
            # shop 2 shifts under the branch prefix: re-measure it
            sub_seeds = sorted({s for (oi, s, seat), _ in tagged})
            pmap = RW.build_map(parent, panel, sub_seeds)
            worlds = {c: w for c, w in pmap.items()}
        else:
            parent = copy.deepcopy(route[:719])
            parent_tag = "base"
            worlds = {c: w for c, w in tagged}
        prefix = copy.deepcopy(parent[:SPLIT])
        by_world = defaultdict(list)
        for c, w in worlds.items():
            if w.split("|")[0] != s1:
                continue                # seat/opp cell realizing another s1
            oi, s, seat = c
            by_world[w].append((panel[oi], s, seat))
        for world, cells in sorted(by_world.items(), key=lambda kv: kv[0]):
            if len(cells) < a.min_cells:
                continue
            cells = rng.sample(cells, min(a.cells_per_world, len(cells)))
            pt = SS.write_tape(parent, os.path.join(WORK, "parent.tape"))
            base_score = SS.batch_eval_cells([pt], cells, 6)[0]["score"]
            best, best_score = parent, base_score
            for g in range(a.gens):
                muts = []
                for _ in range(a.lam):
                    m = SS.mutate(best, rng, window=80, field_ops=True)
                    m[:SPLIT] = copy.deepcopy(prefix)      # prefix guard
                    muts.append(m)
                paths = [SS.write_tape(m, os.path.join(WORK, f"m{i}.tape"))
                         for i, m in enumerate(muts)]
                res = SS.batch_eval_cells(paths, cells, 6)
                gi = max(range(len(muts)), key=lambda i: res[i]["score"])
                if res[gi]["score"] > best_score:
                    best, best_score = muts[gi], res[gi]["score"]
            tag = ""
            if best_score >= base_score + MIN_EDGE:
                out[world] = {"score": best_score,
                              "base_score": base_score,
                              "parent": parent_tag, "cells": len(cells),
                              "cont": best[SPLIT:719]}
                tag = "  <== BRANCH IN"
            print(f"{world:<28s} [{parent_tag:4s}] base {base_score:.3f}"
                  f" -> best {best_score:.3f}  ({len(cells)} cells){tag}",
                  flush=True)
            json.dump(out, open(outp, "w", encoding="utf-8"))
    json.dump(out, open(outp, "w", encoding="utf-8"))
    print(f"\n{len(out)} day-6 branches -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
