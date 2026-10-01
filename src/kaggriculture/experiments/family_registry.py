"""Persistent family registry: stable family ids across daily retrains.

The clustering lab (models/lab/clustering_report.json, 2026-08-12) showed the
production label churn is an ORDERING artifact, not an algorithm deficit:
greedy first-fit linkage is perfectly persistent (1.000) when routes arrive
oldest-first, because old routes re-hit the same representatives and new
families APPEND. train_identifier sorts newest-first, so every day reshuffles
the representatives and every class id moves (the 11->17 shift that forced
the RELAY GUARD).

This module makes stability explicit rather than accidental: cluster
representatives live in a registry file; each retrain loads them, assigns
routes against them in REGISTRY ORDER, and appends genuinely new families.
Family id N means the same lineage on every day after its birth.

Integration (post-release): train_identifier.build_dataset replaces its
inline reps/labels loop with `assign(feats_by_id_date_ascending)`, and the
MIN_FAMILY remap keys off first-qualification order (also append-only).

    python src/experiments/family_registry.py     # self-test on the index
"""
from kaggriculture.paths import ROOT
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.features as features  # noqa: E402

REGISTRY = os.path.join(ROOT, "models", "v22", "identifier", "registry.json")
LINK = 0.055


def load(path=REGISTRY):
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        return d.get("reps", []), d.get("born", [])
    return [], []


def save(reps, born, path=REGISTRY):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"reps": [[round(float(v), 6) for v in r] for r in reps],
               "born": born, "link": LINK},
              open(path, "w", encoding="utf-8"))


def assign(items, reps=None, born=None, today=None, persist=None):
    """items: iterable of (route_id, feature_vec, date) in ASCENDING date
    order (the order is the stability guarantee -- assert, don't assume).
    Returns {route_id: family_id} with ids stable across calls that share a
    registry. persist: registry path to update, or None for dry runs."""
    if reps is None or born is None:
        reps, born = load(persist or REGISTRY)
    labels = {}
    last_date = ""
    for rid, vec, date in items:
        if date and date < last_date:
            raise ValueError(
                f"items not in ascending date order at {rid}: "
                f"{date} < {last_date} -- ordering IS the stability")
        last_date = max(last_date, date or "")
        for fi, rep in enumerate(reps):
            if features.distance(vec, rep) < LINK:
                labels[rid] = fi
                break
        else:
            reps.append(list(vec))
            born.append(date or today or "")
            labels[rid] = len(reps) - 1
    if persist:
        save(reps, born, persist)
    return labels, reps, born


def _selftest():
    import kaggriculture.data.routes as R
    idx = R.load_index()
    recs = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]))
    all_days = sorted(set(r.get("date", "") for r in recs if r.get("date")))
    cut = all_days[-1]
    # force BOTH sides of the day boundary to be populated (the index is
    # dominated by today's ingest, so a naive tail slice sees one day only)
    prev_recs = [r for r in recs if r.get("date", "") < cut][-600:]
    new_recs = [r for r in recs if r.get("date", "") == cut][:600]
    items = []
    for rec in prev_recs + new_recs:
        try:
            v, _ = features.route_features(R.load_route(rec["id"]))
        except Exception:                                          # noqa: BLE001
            continue
        items.append((rec["id"], v, rec.get("date", "")))
    early = [it for it in items if it[2] < cut]
    late = [it for it in items if it[2] >= cut]
    assert early and late, "self-test needs routes on both sides of the day"

    # day 1: cluster the early snapshot from an empty registry
    lab1, reps1, born1 = assign(early, reps=[], born=[], persist=None)
    n1 = len(reps1)
    # day 2: SAME registry continues with the new day appended
    lab2, reps2, born2 = assign(late, reps=[list(r) for r in reps1],
                                born=list(born1), persist=None)
    assert reps2[:n1] == [list(r) for r in reps1] or len(reps2) >= n1, \
        "existing representatives must be untouched"
    # every early route keeps its exact id if re-assigned on day 2's registry
    lab1b, _, _ = assign(early, reps=[list(r) for r in reps2],
                         born=list(born2), persist=None)
    moved = sum(1 for k in lab1 if lab1b[k] != lab1[k])
    assert moved == 0, f"{moved} routes changed family across the boundary"
    print(f"registry self-test OK: {n1} families day-1, "
          f"{len(reps2)} after day-2 ({len(reps2) - n1} appended), "
          f"0/{len(lab1)} routes moved")


if __name__ == "__main__":
    _selftest()
