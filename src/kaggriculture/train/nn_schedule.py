"""Nearest-neighbour opponent SCHEDULE retrieval for the market relay.

WHY THIS EXISTS (measured 2026-09-03, `.local/predict_feasibility.py` and
`.local/nn_robustness.py`, 2,000 library routes / 600 held-out-day routes):

The relay's M2 layer races the opponent's dumps against a per-identifier-CLASS
consensus schedule (`models/relay/family_dumps.json`).  The shipped 20-class
identifier has COLLAPSED on the current field -- 1,624 / 279 / 97 members in
three live classes -- so "the class schedule" is one global schedule for 80%
of the ladder, and it predicts the step of the opponent's next >=8-unit dump
no better than the unconditional median:

    s=360, hit within +-6 turns    WHEAT  STRAW  MELON   MILK   WOOL  FERT
    global median (what ships)     0.287  0.870  0.103  0.389  0.396  0.325
    class-conditional               0.302  0.879  0.192  0.399  0.356  0.335
    NEAREST NEIGHBOUR (this)        0.808  0.947  0.745  0.860  0.853  0.617

Retrieval is what carries the information, not classification -- and it is
NOT retrieving the same team's other episode (same-team rate 1.6-3.3%).  It
survives the observation noise the live agent actually has: at feature
dropout 0.2 (the top of `train_identifier.DROPOUT`, which exists because
floored sales go unseen) a 250-route library still scores 0.771 / 0.936 /
0.707 / 0.781 / 0.784 / 0.539.  250 routes captures ~95% of a 2,000-route
library, which is what makes it embeddable in a single file.

WHAT THIS DOES NOT DO: it does not change production, purchases, or the
route.  It changes only WHICH scheduled SELL the relay races and WHEN --
the same thing `_FAMILY_DUMPS` already decides, from a better predictor.

    python src/nn_schedule.py --build            # -> models/relay/nn_lib.json
    python src/nn_schedule.py --build --routes 400
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(ROOT, "models", "relay", "nn_lib.json")

# Checkpoints must be steps the live agent can hit exactly, and the vector's
# last element is t/720, so the query and the library must share t.  These
# are inside the relay window (_RELAY_START 216 .. 640).
# 288 == day 13 and 360 == day 16: the loss study puts the decisive
# day at 13-16 in 5 of 8 recent losses, so retrieval must be live
# and refreshed across that window, not only before it.
CHECKPOINTS = (240, 288, 360, 480)
# Products the relay may race (v22 _RELAY_M2_PRODUCTS) plus WHEAT, which the
# measurement shows is the single most predictable dump (0.287 -> 0.808).
PRODUCTS = ("FERTILIZER", "MELON", "WOOL", "STRAWBERRY", "MILK", "WHEAT")
DUMP_QTY = 8
SCALE = 1000            # feature quantisation; 3 decimals is below the noise


def _dumps(actions, lo=216, hi=680):
    out = []
    for t in range(lo, min(hi, len(actions))):
        turn = actions[t]
        if not isinstance(turn, dict):
            continue
        for o in (turn.get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] in PRODUCTS):
                try:
                    q = int(o[2])
                except (TypeError, ValueError):
                    continue
                if q >= DUMP_QTY:
                    out.append([t, PRODUCTS.index(o[1]), q])
    return out


def _one(rid):
    """Worker: (rid, [quantised vector per checkpoint], dump schedule)."""
    import kaggriculture.data.features as F
    p = os.path.join(ROOT, "data", "routes", rid + ".json.gz")
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            acts = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(acts, list) or len(acts) < 700:
        return None
    sched = _dumps(acts)
    if not sched:
        return None                     # nothing to race; costs library slots
    vecs = []
    for c in CHECKPOINTS:
        v = F.prefix_features(acts, c) + [c / 720.0]
        vecs.append([int(round(x * SCALE)) for x in v])
    return (rid, vecs, sched)


def build(n_routes=250, days=5, workers=6, verbose=True):
    """Library from the freshest `days` days of the route index.

    Freshness matters the way it matters for the base: the top is a
    freshness board and schedules decay in days.  Rebuild with the daily
    identifier retrain, not once.
    """
    import kaggriculture.data.routes as R
    idx = R.load_index()["routes"]
    by_date = defaultdict(list)
    for rid, rec in idx.items():
        if rec.get("date"):
            by_date[rec["date"]].append(rid)
    dates = sorted(by_date)
    pool = [r for d in dates[-days:] for r in by_date[d]]
    random.Random(7).shuffle(pool)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_one, pool[:n_routes * 3], chunksize=8):
            if r is not None:
                rows.append(r)
            if len(rows) >= n_routes:
                break
    if verbose:
        print(f"{len(rows)} library routes from {dates[-days]}..{dates[-1]}")
    lib = {
        "checkpoints": list(CHECKPOINTS),
        "products": list(PRODUCTS),
        "scale": SCALE,
        "dim": len(rows[0][1][0]) if rows else 0,
        "built": dates[-1] if dates else None,
        "ids": [r[0] for r in rows],
        # V[checkpoint_index][route_index] = quantised feature vector
        "V": [[r[1][ci] for r in rows] for ci in range(len(CHECKPOINTS))],
        "S": [r[2] for r in rows],
    }
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--routes", type=int, default=250)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not args.build:
        ap.error("nothing to do; pass --build")
    lib = build(args.routes, args.days, args.workers)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(lib, open(args.out, "w", encoding="utf-8"),
              separators=(",", ":"))
    print(f"wrote {args.out} "
          f"({os.path.getsize(args.out):,} bytes raw, dim {lib['dim']}, "
          f"{len(lib['ids'])} routes x {len(lib['checkpoints'])} checkpoints)")


if __name__ == "__main__":
    main()
