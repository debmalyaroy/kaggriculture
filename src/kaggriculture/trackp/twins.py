"""Relationship miner #3 -- within-family twin studies.

Two seats that played the SAME opening (identical opening hash) under the
SAME shop-draw regime are natural twins: strategy identity is controlled
for, so mid/late-game differences between the winning and losing twin are
causal candidates in a way field-wide correlations never are.

For every (family, regime) cell with at least one winner and one loser,
pair them up and compute paired feature differences (winner - loser) over
the full play-feature set; aggregate with a sign test across pairs.

Output: models/trackp/twins.json -- features ranked by |sign consistency|,
with mean paired differences.

Usage: python src/trackp/twins.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

try:
    from . import common, families, insight, trace_v2
except ImportError:
    import sys
    from kaggriculture.trackp import common, families, insight, trace_v2

PARTITION = insight.PARTITION


def _regime(X0):
    draw = []
    ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
    for s in common.SHOPS_SORTED:
        d0 = X0[-1, ix[f"shopday_{s}"]]
        if d0 >= 0:
            draw.append((d0, s))
    draw.sort()
    parts = [PARTITION.get(s, "neutral") for _, s in draw[:2]]
    return next((p for p in ("wool", "carrot", "dairy") if p in parts),
                "neutral")


def _binom_tail(k, n):
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


def run(engine=None, limit=0, max_pairs_per_cell=6):
    if engine is None:
        engine = common.engine_version()
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))
    if limit:
        files = files[:limit]
    cells = {}          # (family, regime) -> {"win": [feat], "loss": [feat]}
    n_seats = 0
    for f in files:
        try:
            X0, meta = trace_v2.load(f)
        except Exception:                                          # noqa: BLE001
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        if banks[0] == banks[1]:
            continue
        regime = _regime(X0)
        for seat in (0, 1):
            fam = families.opening_hash(X0, seat)
            key = (fam, regime)
            side = "win" if banks[seat] > banks[1 - seat] else "loss"
            cell = cells.setdefault(key, {"win": [], "loss": []})
            if len(cell[side]) < max_pairs_per_cell:
                cell[side].append(insight.play_features(
                    trace_v2.seat_view(X0, seat)))
            n_seats += 1

    # pair winners with losers inside each cell
    diffs = []
    n_cells = 0
    for (fam, regime), cell in cells.items():
        if not cell["win"] or not cell["loss"]:
            continue
        n_cells += 1
        for w, l in zip(cell["win"], cell["loss"]):
            diffs.append({k: w[k] - l[k] for k in w})
    if not diffs:
        raise SystemExit("no twin pairs found")
    names = sorted(diffs[0].keys())
    report = []
    for k in names:
        v = np.array([d[k] for d in diffs], dtype=np.float64)
        pos = int((v > 0).sum())
        neg = int((v < 0).sum())
        nz = pos + neg
        p = _binom_tail(max(pos, neg), nz) * 2 if nz else 1.0
        report.append({"feature": k, "pairs": nz,
                       "winner_higher_frac": round(pos / nz, 3) if nz else .5,
                       "mean_diff": round(float(v.mean()), 3),
                       "sign_p": round(min(1.0, p), 5)})
    report.sort(key=lambda r: r["sign_p"])
    rep = {"twin_pairs": len(diffs), "cells": n_cells,
           "seats_scanned": n_seats, "features": report}
    out = os.path.join(common.MODELS, "twins.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({"twin_pairs": len(diffs), "cells": n_cells}))
    for r in report[:12]:
        print(f"  {r['feature']:<24} winner-higher {r['winner_higher_frac']:.2f}"
              f"  mean diff {r['mean_diff']:>10}  p {r['sign_p']}")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(limit=a.limit)
