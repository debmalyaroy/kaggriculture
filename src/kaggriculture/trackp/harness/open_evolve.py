"""OWNED OPENING / FAMILY-KILLER: evolve a full tape (no prefix guard).

Operator direction (2026-09-07): escape the clone-family economy entirely.
32% of the field plays our public opening; mirrors trend to half-point
ties; our line is extractable and counter-tunable. This evolves a
genuinely different line:

  seed    = the strongest fresh NON-CLONE elite route (already a
            different family -- different tile geometry from turn 1);
  mutate  = sell_search ops over the WHOLE tape, opening included;
  fitness = W/L vs a panel of top clone donors (the family we must beat)
            + other-family elites, across random seeds spanning worlds;
  output  = models/trackp/owned_open.json (the tape + provenance) --
            packaged into an agent only after fitness earns it.

Long-running by design; safe to re-run (resumes from the saved best).

    python src/trackp/harness/open_evolve.py --gens 40 --lam 10
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
from kaggriculture.trackp import routes_io as R                              # noqa: E402

WORK = os.path.join(ROOT, ".local", "open_evolve")
OUT = os.path.join(ROOT, "models", "trackp", "owned_open.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--lam", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--family-killer", action="store_true",
                    help="fitness vs the CLONE FAMILY only; seed = our own "
                         "line (out = owned_counter.json)")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    idx = R.load_index()
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    clone_ids = {h["id"] for h in hits}
    clone_teams = {h.get("team") for h in hits}

    # seed parent: strongest fresh non-clone winner
    parent_id = None
    for k, v in sorted(idx["routes"].items(),
                       key=lambda t: -(t[1].get("bank") or 0)):
        if (v.get("engine") == "1.32.7" and v.get("won")
                and str(v.get("date", "")) >= "2026-09-01"
                and k not in clone_ids
                and v.get("team") not in clone_teams):
            parent_id = k
            break
    global OUT
    if a.family_killer:
        OUT = os.path.join(ROOT, "models", "trackp", "owned_counter.json")
    prior = None
    if os.path.exists(OUT):
        prior = json.load(open(OUT, encoding="utf-8"))
        print(f"resuming from saved best (score {prior['score']:.3f}, "
              f"gen {prior['gen']})")
    if a.family_killer and not prior:
        parent_id = max(hits, key=lambda h: h.get("bank") or 0)["id"]
    if prior:
        parent = prior["tape"]
        parent_id = prior["seed_id"]
        start_gen = prior["gen"]
        best_score_ever = prior["score"]
    else:
        parent = R.load_route(parent_id)[:719]
        start_gen = 0
        best_score_ever = None
    print(f"seed line: {parent_id} "
          f"(bank {idx['routes'][parent_id].get('bank'):,.0f}, "
          f"team {idx['routes'][parent_id].get('team')})")

    # panel: the clone family we must beat + other-family elites
    panel = []
    n_family = 5 if a.family_killer else 3
    for j, h in enumerate(sorted(hits, key=lambda x: -(x.get("bank") or 0))[:n_family]):
        try:
            panel.append(SS.write_tape(R.load_route(h["id"]),
                                       os.path.join(WORK, f"c{j}.tape")))
        except Exception:                                      # noqa: BLE001
            pass
    added = 2 if a.family_killer else 0   # family-killer: family-only panel
    for k, v in sorted(idx["routes"].items(),
                       key=lambda t: -(t[1].get("bank") or 0)):
        if (v.get("engine") == "1.32.7" and v.get("won")
                and k != parent_id and k not in clone_ids
                and str(v.get("date", "")) >= "2026-09-01"):
            try:
                panel.append(SS.write_tape(
                    R.load_route(k), os.path.join(WORK, f"e{added}.tape")))
                added += 1
            except Exception:                                  # noqa: BLE001
                continue
        if added >= 2:
            break
    rng = random.Random(23 + start_gen)
    eval_seeds = [rng.randrange(1, 10**6) for _ in range(a.seeds)]
    print(f"panel {len(panel)} tapes, {len(eval_seeds)} seeds")

    pt = SS.write_tape(parent, os.path.join(WORK, "parent.tape"))
    best_score = SS.batch_eval([pt], panel, eval_seeds, 6)[0]["score"]
    best = parent
    if best_score_ever is None:
        best_score_ever = best_score
    print(f"gen {start_gen}: score {best_score:.3f}")
    for g in range(start_gen + 1, start_gen + 1 + a.gens):
        muts = [SS.mutate(best, rng, window=90, n_ops=3, field_ops=True)
                for _ in range(a.lam)]
        paths = [SS.write_tape(m, os.path.join(WORK, f"m{i}.tape"))
                 for i, m in enumerate(muts)]
        res = SS.batch_eval(paths, panel, eval_seeds, 6)
        gi = max(range(len(muts)), key=lambda i: res[i]["score"])
        if res[gi]["score"] > best_score:
            best, best_score = muts[gi], res[gi]["score"]
        if g % 5 == 0 or res[gi]["score"] > best_score_ever:
            best_score_ever = max(best_score_ever, best_score)
            json.dump({"seed_id": parent_id, "gen": g,
                       "score": best_score, "tape": best},
                      open(OUT, "w", encoding="utf-8"))
            print(f"gen {g}: best {best_score:.3f} (saved)", flush=True)
    json.dump({"seed_id": parent_id, "gen": start_gen + a.gens,
               "score": best_score, "tape": best},
              open(OUT, "w", encoding="utf-8"))
    print(f"\nFINAL score {best_score:.3f} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
