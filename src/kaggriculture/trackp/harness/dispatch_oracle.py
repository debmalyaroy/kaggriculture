"""P2 ORACLE: is opponent-keyed dispatch worth building?

Old dispatch keyed on MARKET conditions and measured dead at every pool
size. Never tested: keying on the OPPONENT -- their economy family is
visible from their opening, and different families may lose to different
counter-tapes. This is the cheap decisive test, run BEFORE any dispatcher
is built (the kill-switch discipline).

Grid: N sampled elite cells (>=2500 opponents in their own worlds) x K of
our candidate base tapes, via `kagg batch` (pure Rust, tape vs tape).

  ORACLE HEADROOM = mean(best tape per cell) - best(mean tape overall).
  KEYABILITY      = does the opponent's 72-step family predict the argmax
                    tape better than always picking the global best?

If headroom ~0 (as market-keyed dispatch measured), P2 closes tonight and
the effort goes to P1. If real and keyable, a dispatcher is worth building.

    python src/trackp/harness/dispatch_oracle.py --cells 150
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import hashlib
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
OUT = os.path.join(ROOT, "models", "trackp", "dispatch_oracle.json")
KAGG = os.path.join(ROOT, "rustengine", "kagg.exe")
WORK = os.path.join(ROOT, ".local", "hardband", "oracle_work")
# top-3 only (2026-09-05 retry): the 7 weak economies added label noise
# without ever being the right pick -- margins for them were mostly
# dominated. Smaller K, denser signal.
BASES = ["105370586_s0", "105367892_s0", "105023582_s1"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", type=int, default=150)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    cells = json.load(open(CELLS, encoding="utf-8"))
    rng = random.Random(31)
    # stratify: one cell per elite team first, then fill
    by = defaultdict(list)
    for c in cells:
        by[c["team"]].append(c)
    picks = []
    for t, cs in by.items():
        rng.shuffle(cs)
        picks.extend(cs if a.cells >= 6000 else cs[:3])
    rng.shuffle(picks)
    picks = picks[:a.cells]
    print(f"{len(picks)} cells (one per elite team, capped)")

    # our candidate tapes
    ours = {}
    for b in BASES:
        ours[b] = SS.write_tape(R.load_route(b),
                                os.path.join(WORK, f"our_{b}.tape"))

    jobs, meta = [], []
    fam_of = {}
    for ci, c in enumerate(picks):
        ep, es = str(c["episode"]), int(c["elite_seat"])
        try:
            elite_acts = R.load_route(f"{ep}_s{es}")
        except Exception:                                      # noqa: BLE001
            continue
        opp_tape = SS.write_tape(elite_acts,
                                 os.path.join(WORK, f"opp_{ep}.tape"))
        fam_of[ci] = hashlib.sha256(
            json.dumps(elite_acts[:72], sort_keys=True)
            .encode()).hexdigest()[:12]
        for bi, b in enumerate(BASES):
            # we in seat (1 - elite_seat) of the recorded world
            if es == 0:
                jobs.append(f"{c['seed']}\t{opp_tape}\t{ours[b]}")
            else:
                jobs.append(f"{c['seed']}\t{ours[b]}\t{opp_tape}")
            meta.append((ci, bi, es))
    jp = os.path.join(WORK, "jobs.tsv")
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jobs))
    print(f"{len(jobs)} games -> kagg batch")
    out = subprocess.run([KAGG, "batch", jp, str(a.threads)],
                         capture_output=True, text=True, timeout=7200)
    if out.returncode != 0:
        raise SystemExit(f"kagg batch failed: {out.stderr[:300]}")

    score = defaultdict(dict)          # ci -> bi -> margin
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        ci, bi, es = meta[int(parts[0])]
        b0, b1 = float(parts[2]), float(parts[3])
        mine, theirs = (b1, b0) if es == 0 else (b0, b1)
        score[ci][bi] = mine - theirs

    cells_ok = [ci for ci in score if len(score[ci]) == len(BASES)]
    n = len(cells_ok)
    if not n:
        raise SystemExit("no scored cells")
    # win-rate currency (score), margins as tiebreak
    win = lambda m: 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
    per_base = [sum(win(score[ci][bi]) for ci in cells_ok) / n
                for bi in range(len(BASES))]
    best_single = max(per_base)
    oracle = sum(max(win(score[ci][bi]) for bi in range(len(BASES)))
                 for ci in cells_ok) / n
    print(f"\ncells scored: {n}")
    for bi, b in enumerate(BASES):
        print(f"  {b:<16} win {per_base[bi]:.3f}")
    print(f"  BEST SINGLE      {best_single:.3f}")
    print(f"  ORACLE (perfect opponent-keyed pick) {oracle:.3f}")
    print(f"  HEADROOM         {oracle - best_single:+.3f}")

    # KEYABILITY: family -> argmax base; leave-one-out family vote
    fam_votes = defaultdict(lambda: defaultdict(float))
    for ci in cells_ok:
        bi = max(range(len(BASES)), key=lambda b: score[ci][b])
        fam_votes[fam_of[ci]][bi] += 1
    keyed = 0.0
    gbi = per_base.index(best_single)
    for ci in cells_ok:
        votes = dict(fam_votes[fam_of[ci]])
        bi = max(range(len(BASES)), key=lambda b: score[ci][b])
        votes[bi] -= 1                  # leave this cell out
        pick = max(votes, key=votes.get) if any(votes.values()) else gbi
        keyed += win(score[ci][pick])
    keyed /= n
    print(f"  FAMILY-KEYED (leave-one-out) {keyed:.3f}  "
          f"vs best-single {best_single:.3f}")
    verdict = ("BUILD the dispatcher" if keyed - best_single > 0.02
               else "P2 CLOSED: no keyable headroom -- effort goes to P1")
    print(f"\nVERDICT: {verdict}")
    grid_out = os.path.join(ROOT, "models", "trackp",
                            "dispatch_grid.json")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"bases": BASES,
               "cells": [{"episode": str(picks[ci]["episode"]),
                          "elite_seat": int(picks[ci]["elite_seat"]),
                          "team": picks[ci]["team"],
                          "seed": picks[ci]["seed"],
                          "family": fam_of.get(ci),
                          "margins": [score[ci].get(b) for b in
                                      range(len(BASES))]}
                         for ci in cells_ok]},
              open(grid_out, "w", encoding="utf-8"), indent=1)
    print(f"grid -> {os.path.relpath(grid_out, ROOT)}")
    json.dump({"n": n, "per_base": dict(zip(BASES, per_base)),
               "best_single": best_single, "oracle": oracle,
               "family_keyed": keyed, "verdict": verdict},
              open(OUT, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
