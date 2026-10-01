"""P0.5 -- opening-hash-anchored family labels.

A family label is the sha1-16 of the seat's quantized per-turn action blocks
over days 0-2 (72 turns) -- purely observational, so labels are STABLE
across retrains by construction (the 0.893->0.571 clustering incident can't
recur). The gate the plan set (>=95% persistence across three retrains) is
measured anyway, on disjoint subsets, and recorded.

Each family also carries its DUMP SCHEDULE (mean sell units per 24-turn
bucket per product) -- a direct, better-grounded FAMILY_PRIOR for the
planner's L2 than the legacy family_dumps data.

Usage: python src/trackp/families.py [--rebuild]
Writes models/trackp/families.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

try:
    from . import common, trace_v2
except ImportError:
    import sys
    from kaggriculture.trackp import common, trace_v2

OPENING_TURNS = 72  # days 0-2, before the first shop unlock
FAMILIES = os.path.join(common.MODELS, "families.json")

_ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
_ACT_M = [i for f, i in _ix.items() if f.startswith("am_")]


def opening_hash(X: np.ndarray, seat: int = 0) -> str:
    """sha1-16 of the seat's action blocks over the opening turns."""
    V = trace_v2.seat_view(X, seat)[:OPENING_TURNS, _ACT_M]
    q = np.asarray(np.round(V), dtype=np.int32)
    return hashlib.sha1(q.tobytes()).hexdigest()[:16]


def build(engine: str = None) -> dict:
    if engine is None:
        engine = common.engine_version()  # follow the ladder
    fams: dict = {}
    n_seats = 0
    for f in glob.glob(os.path.join(common.TRACES, "*.npz")):
        try:
            X, meta = trace_v2.load(f)
        except Exception:
            continue
        if engine and meta.get("engine") != engine:
            continue
        banks = meta.get("banks") or [0, 0]
        teams = meta.get("teams") or ["?", "?"]
        for seat in (0, 1):
            h = opening_hash(X, seat)
            n_seats += 1
            fam = fams.setdefault(h, {"count": 0, "teams": {}, "banks": [],
                                      "dump": {}})
            fam["count"] += 1
            fam["teams"][teams[seat]] = fam["teams"].get(teams[seat], 0) + 1
            fam["banks"].append(float(banks[seat]))
            V = trace_v2.seat_view(X, seat)
            for p in common.PRODUCTS:
                col = V[:, _ix[f"am_sell_{p}"]]
                if col.sum() <= 0:
                    continue
                buckets = fam["dump"].setdefault(p, [0.0] * 30)
                for day in range(min(30, V.shape[0] // 24)):
                    buckets[day] += float(
                        col[day * 24:(day + 1) * 24].sum())
    out = {}
    for h, fam in fams.items():
        n = fam["count"]
        dump = {p: [round(v / n, 2) for v in b]
                for p, b in fam["dump"].items()}
        out[h] = {"count": n, "n_teams": len(fam["teams"]),
                  "top_team": max(fam["teams"], key=fam["teams"].get),
                  "mean_bank": round(sum(fam["banks"]) / n, 0),
                  "dump_per_day": dump}
    rep = {"engine": engine, "seats": n_seats, "families": len(out),
           "multi_member": sum(1 for f in out.values() if f["count"] >= 2),
           "coverage_5plus": round(sum(f["count"] for f in out.values()
                                       if f["count"] >= 5) / max(1, n_seats),
                                   3)}
    with open(FAMILIES, "w", encoding="utf-8") as fh:
        json.dump({"summary": rep, "families": out}, fh)
    print(json.dumps(rep, indent=1))
    return rep


def persistence_check(runs: int = 3, seed: int = 5) -> dict:
    """The plan's gate: labels persist across retrains. Anchored hashing is
    persistent BY CONSTRUCTION; this measures it anyway on disjoint subsets
    (same episode hashed in different 'retrains' must get the same label)."""
    files = sorted(glob.glob(os.path.join(common.TRACES, "*.npz")))[:300]
    rng = np.random.default_rng(seed)
    labels = {}
    stable = total = 0
    for r in range(runs):
        order = rng.permutation(len(files))
        for i in order:
            f = files[int(i)]
            try:
                X, meta = trace_v2.load(f)
            except Exception:
                continue
            h = opening_hash(X, 0)
            key = meta.get("episode")
            if key in labels:
                total += 1
                stable += labels[key] == h
            else:
                labels[key] = h
    rate = stable / max(1, total)
    rep = {"persistence": round(rate, 4), "checked": total,
           "gate_95": rate >= 0.95}
    print(json.dumps(rep))
    return rep


def family_prior_for(top_k: int = 1) -> list:
    """The most common families' dump schedules as [(step, product, qty)]
    -- the planner builder's PRIOR source (Track P's own, replacing the
    legacy family_dumps data when present)."""
    if not os.path.exists(FAMILIES):
        return []
    with open(FAMILIES, encoding="utf-8") as fh:
        d = json.load(fh)
    fams = sorted(d["families"].items(), key=lambda kv: -kv[1]["count"])
    prior = []
    for h, fam in fams[:top_k]:
        for p, days in fam["dump_per_day"].items():
            for day, q in enumerate(days):
                if q >= 1.0:
                    prior.append((day * 24 + 12, p, round(q, 1)))
    prior.sort()
    return prior


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--persistence", action="store_true")
    a = ap.parse_args()
    if a.rebuild or not os.path.exists(FAMILIES):
        build()
    if a.persistence:
        persistence_check()
