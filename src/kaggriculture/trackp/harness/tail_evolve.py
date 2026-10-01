"""Owned-tail evolution per world: (1+lambda) on manufactured seeds.

Operator direction (2026-09-06): don't stop at selecting the field's best
tail per world — SOLVE the world. For each shop-pair world:

  start   = the tournament-winning tail (or the base continuation);
  mutate  = sell_search ops (HIRE add/del, purchase shift/resize, sell
            retime/resize) restricted to the tail region — the opening
            [0:145] is restored verbatim after every mutation, so every
            candidate stays whole-prefix splice-safe by construction;
  fitness = W/L (ties 0.5) vs a panel of the world's strongest donor
            tapes + our shipped v46.1 tape, over the world's own seeds
            from data/worlds/seed_bank.json, both seats;
  accept  = mutant strictly beats the parent's score.

Open-loop pre-ranker caveat applies; band/gauntlet stay the ship judge.
Worlds are processed WORST-FIRST (our live-loss worlds, then uncovered).

    python src/trackp/harness/tail_evolve.py --gens 15 --lam 8
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.pipeline.sell_search as SS  # noqa: E402
from kaggriculture.trackp import routes_io as R                              # noqa: E402

WORK = os.path.join(ROOT, ".local", "tail_evolve")
SPLIT = 145
MIN_EDGE = 0.08


def our_route():
    src = open(os.path.join(ROOT, "agents", "v45.0_bandit.py"),
               encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    return json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--lam", type=int, default=8)
    ap.add_argument("--seeds-per-world", type=int, default=8)
    ap.add_argument("--worlds", default=None,
                    help="comma-separated world keys; default worst-first")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    route = our_route()
    tails_p = os.path.join(ROOT, ".local", "candidates", "world_tails.json")
    tails = json.load(open(tails_p, encoding="utf-8"))
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    by_world = {}
    for h in hits:
        if h.get("world"):
            by_world.setdefault(h["world"], []).append(h)

    if a.worlds:
        worlds = a.worlds.split(",")
    else:
        # worst-first: live-loss worlds, then bare worlds, then the rest
        loss_worlds = []
        try:
            rep = json.load(open(os.path.join(ROOT, ".local",
                                              "loss_autopsy.json"),
                                 encoding="utf-8"))
            loss_worlds = [r.get("world") for r in rep.get("reports", [])
                           if r.get("world")]
        except Exception:                                      # noqa: BLE001
            pass
        bare = [w for w in bank if w not in tails]
        rest = [w for w in bank if w in tails and w not in loss_worlds]
        seen = set()
        worlds = [w for w in loss_worlds + bare + rest
                  if w in bank and not (w in seen or seen.add(w))]

    rng = random.Random(11)
    improved = 0
    for w in worlds:
        seeds = (bank.get(w) or [])[:a.seeds_per_world]
        if len(seeds) < 4:
            continue
        base_tail = (tails[w]["tail"] if w in tails
                     else route[SPLIT:719])
        parent = route[:SPLIT] + copy.deepcopy(base_tail)
        panel = []
        for j, h in enumerate(sorted(by_world.get(w, []),
                                     key=lambda x: -x["bank"])[:2]):
            try:
                panel.append(SS.write_tape(R.load_route(h["id"]),
                                           os.path.join(WORK, f"p{j}.tape")))
            except Exception:                                  # noqa: BLE001
                pass
        panel.append(SS.write_tape(route[:719],
                                   os.path.join(WORK, "p_base.tape")))
        pt = SS.write_tape(parent, os.path.join(WORK, "parent.tape"))
        p_score = SS.batch_eval([pt], panel, seeds, 6)[0]["score"]
        best, best_score = parent, p_score
        for g in range(a.gens):
            muts = []
            for i in range(a.lam):
                m = SS.mutate(best, rng, field_ops=True)
                m[:SPLIT] = copy.deepcopy(route[:SPLIT])   # prefix guard
                muts.append(m)
            paths = [SS.write_tape(m, os.path.join(WORK, f"m{i}.tape"))
                     for i, m in enumerate(muts)]
            res = SS.batch_eval(paths, panel, seeds, 6)
            gi = max(range(len(muts)), key=lambda i: res[i]["score"])
            if res[gi]["score"] > best_score:
                best, best_score = muts[gi], res[gi]["score"]
        tag = ""
        if best_score >= p_score + MIN_EDGE:
            tails[w] = {"bank": 0, "opp_bank": 0, "episode": "evolved",
                        "seat": -1, "team": f"EVOLVED({w})",
                        "tail": best[SPLIT:]}
            improved += 1
            tag = "  <== EVOLVED IN"
        print(f"{w:36s} start {p_score:.3f} -> best {best_score:.3f}{tag}",
              flush=True)
        json.dump(tails, open(tails_p, "w", encoding="utf-8"))
    print(f"\n{improved} worlds improved by evolution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
