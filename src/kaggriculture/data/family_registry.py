"""Persistent family registry: stable family ids across daily retrains.

The 2026-08-12 clustering lab (models/lab/clustering_report.json) showed the
daily class-id churn was an ORDERING artifact, not an algorithm deficit:
greedy first-fit linkage is perfectly persistent when routes arrive
OLDEST-FIRST (old routes re-hit the same representatives; new families
append), and train_identifier's newest-first sort was what reshuffled every
label each day (the 11->17 shift that forced the RELAY GUARD).

Two persisted structures make the stability explicit:

  reps          cluster representatives, in birth order. Family id == index.
  trained_order family ids in the order they FIRST met MIN_FAMILY -- the
                exported class id is the position here, so a family that
                qualifies later never renumbers earlier classes.

Registry file: models/v22/identifier/registry.json. Delete it to hard-reset
the label space (the next retrain rebuilds it from scratch, ascending).
"""
from kaggriculture.paths import ROOT
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRY = os.path.join(ROOT, "models", "v22", "identifier", "registry.json")
LINK = 0.055


def load(path=REGISTRY):
    if os.path.exists(path):
        d = json.load(open(path, encoding="utf-8"))
        return (d.get("reps", []), d.get("born", []),
                d.get("trained_order", []))
    return [], [], []


def save(reps, born, trained_order, path=REGISTRY):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"reps": [[round(float(v), 6) for v in r] for r in reps],
               "born": born, "trained_order": trained_order, "link": LINK},
              open(path, "w", encoding="utf-8"))


def assign(items, reps, born, distance):
    """items: (route_id, feature_vec, date) in ASCENDING date order -- the
    ordering IS the stability guarantee, so it is asserted, not assumed.
    Mutates reps/born in place (append-only). Returns {route_id: family_id}."""
    labels = {}
    last_date = ""
    for rid, vec, date in items:
        if date and date < last_date:
            raise ValueError(f"items not date-ascending at {rid}: "
                             f"{date} < {last_date}")
        last_date = max(last_date, date or "")
        for fi, rep in enumerate(reps):
            if distance(vec, rep) < LINK:
                labels[rid] = fi
                break
        else:
            reps.append(list(vec))
            born.append(date or "")
            labels[rid] = len(reps) - 1
    return labels


def class_map(labels, trained_order, min_family):
    """Stable class ids: families meeting min_family join trained_order in
    qualification order and KEEP that position forever. Returns
    (remap {family_id: class_id}, novel_class_id). Mutates trained_order."""
    sizes = {}
    for fi in labels.values():
        sizes[fi] = sizes.get(fi, 0) + 1
    # family ids are already birth-ordered (registry index), so iterating
    # sorted ids appends qualifiers oldest-first; once in, never removed --
    # a family shrinking below min_family later must not renumber anyone.
    for fi in sorted(sizes):
        if sizes.get(fi, 0) >= min_family and fi not in trained_order:
            trained_order.append(fi)
    remap = {fi: i for i, fi in enumerate(trained_order)}
    return remap, len(trained_order)
