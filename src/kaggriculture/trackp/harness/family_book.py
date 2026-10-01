"""FAMILY BOOK: the field's behavior, compressed into the submission.

Operator directive (2026-09-05): leverage the 100MB limit. The searcher's
measured weakness is rollout OPPONENT-MODEL bias (PASS/mirror vs reality).
This book turns 31k recorded routes into per-family behavior profiles:

  opening fingerprint (72-step hash) ->
      per-product MEDIAN sell schedule (qty per 24-step day-bucket),
      seats observed, elite share.

At runtime the agent hashes the opponent's observed opening at step 72,
looks up the family, and the searcher rolls out against the family's
MEDIAN schedule as the opponent -- a measured prior instead of a guess.
Mirror detection falls out free (our own families are flagged).

    python src/trackp/harness/family_book.py          # writes the book
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OUT = os.path.join(ROOT, "models", "trackp", "family_book.json")
PREFIX = 72
BUCKETS = 30                               # one per day
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")
MIN_SEATS = 5                              # families below this are noise


def main():
    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    fams = defaultdict(list)               # hash -> list of per-route grids
    ours = set()
    import importlib.util
    for p in ("agents/v44.1_bandit.py", "agents/v45.0_bandit.py",
              "agents/v45.1_bandit.py"):
        fp = os.path.join(ROOT, p)
        if not os.path.exists(fp):
            continue
        spec = importlib.util.spec_from_file_location("a", fp)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            h = hashlib.sha256(json.dumps(mod._ROUTE[:PREFIX],
                                          sort_keys=True)
                               .encode()).hexdigest()[:12]
            ours.add(h)
        except Exception:                                      # noqa: BLE001
            continue

    n = 0
    for k, v in idx["routes"].items():
        if v.get("engine") != "1.32.7":
            continue
        try:
            acts = R.load_route(k)
        except Exception:                                      # noqa: BLE001
            continue
        h = hashlib.sha256(json.dumps(acts[:PREFIX], sort_keys=True)
                           .encode()).hexdigest()[:12]
        grid = [[0] * len(PRODUCTS) for _ in range(BUCKETS)]
        for i, a in enumerate(acts[:BUCKETS * 24]):
            b = i // 24
            for o in (a.get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in PRODUCTS):
                    grid[b][PRODUCTS.index(o[1])] += int(o[2] or 0)
        fams[h].append(grid)
        n += 1
        if n % 5000 == 0:
            print(f"  {n} routes profiled...", flush=True)

    # OBSERVABLE KEY (v2): the opponent's ACTIONS are not in obs at
    # runtime -- only their FARM is. Each family gets a day-3 field
    # profile (crop counts after its first 72 steps, simulated vs PASS in
    # a fixed world); runtime matches the opponent's visible farm to the
    # nearest profile by L1. Approximation: planting patterns are nearly
    # world-independent over 3 days.
    import kaggriculture.engine.serve_match as SM
    reps = {}
    for k, v in idx["routes"].items():
        if v.get("engine") != "1.32.7":
            continue
        try:
            acts = R.load_route(k)
        except Exception:                                      # noqa: BLE001
            continue
        h = hashlib.sha256(json.dumps(acts[:PREFIX], sort_keys=True)
                           .encode()).hexdigest()[:12]
        if h in fams and len(fams[h]) >= MIN_SEATS and h not in reps:
            reps[h] = acts
    CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    profiles = {}
    srv = SM.Serve()
    try:
        for h, acts in reps.items():
            js = srv.cmd("RESET 424242")
            for i in range(PREFIX):
                la = SM.action_to_line(acts[i])
                lb = SM.action_to_line({})
                js = srv.cmd("STEP2 " + la + chr(30) + lb)
            farm = (js.get("farms") or [{}])[0]
            prof = {c: 0 for c in CROPS}
            animals = 0
            for row in (farm.get("tiles") or []):
                for t in (row if isinstance(row, list) else [row]):
                    if isinstance(t, dict):
                        if t.get("crop") in prof:
                            prof[t["crop"]] += 1
                        if t.get("animal"):
                            animals += 1
            profiles[h] = [prof[c] for c in CROPS] + [animals]
    finally:
        srv.close()

    book = {}
    for h, grids in fams.items():
        if len(grids) < MIN_SEATS or h not in profiles:
            continue
        med = []
        for b in range(BUCKETS):
            row = []
            for p in range(len(PRODUCTS)):
                vals = sorted(g[b][p] for g in grids)
                row.append(vals[len(vals) // 2])
            med.append(row)
        book[h] = {"n": len(grids), "ours": h in ours,
                   "profile": profiles[h], "sched": med}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {"products": list(PRODUCTS), "prefix": PREFIX,
               "buckets": BUCKETS, "families": book}
    json.dump(payload, open(OUT, "w", encoding="utf-8"),
              separators=(",", ":"))
    sz = os.path.getsize(OUT)
    print(f"{len(book)} families (>= {MIN_SEATS} seats) from {n:,} routes "
          f"-> {os.path.relpath(OUT, ROOT)} ({sz:,} bytes)")
    print(f"our flagged families: {sum(1 for b in book.values() if b['ours'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
