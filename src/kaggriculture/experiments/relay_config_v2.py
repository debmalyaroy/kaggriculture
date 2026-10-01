"""M2 v2: recency+bank weighted consensus dump schedules with confidence.

Same class assignment as src/relay_config.py (the shipped identifier labels
its own members), different consensus:

  - each member route is weighted by recency (half-life 3 days) x final bank
    (normalized within the class) -- a $155k route from today out-votes a
    $120k route from last week;
  - each event's time and size are the WEIGHTED median of members;
  - each event carries a confidence tuple: [support, time_iqr, qty_iqr] so
    the runtime can demand tight consensus before racing and fall back to
    live observation otherwise.

Writes models/lab/family_dumps_v2.json (schema: v1 events extended from
[t, product, qty] to [t, product, qty, support, t_iqr, q_iqr]) and prints a
v1-vs-v2 diff. Does NOT touch models/relay/family_dumps.json -- integration
into the build happens only after the current release ships.

    python src/experiments/relay_config_v2.py
"""
from kaggriculture.paths import ROOT
import datetime as dt
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.data.features as F  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402
import kaggriculture.agentbuild.relay_config as V1  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "family_dumps_v2.json")
HALF_LIFE_DAYS = 3.0


def _weight(rec, today, banks):
    age = 0.0
    try:
        d = dt.date.fromisoformat(str(rec.get("date", ""))[:10])
        age = max(0.0, (today - d).days)
    except ValueError:
        age = 7.0
    recency = 0.5 ** (age / HALF_LIFE_DAYS)
    bank = float(rec.get("bank") or 0.0)
    lo, hi = banks
    bank_n = 0.5 if hi <= lo else max(0.0, min(1.0, (bank - lo) / (hi - lo)))
    return recency * (0.25 + 0.75 * bank_n)     # bank scales, never zeroes


def _wmedian(pairs):
    """pairs: [(value, weight)] -> weighted median."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _iqr(vals):
    vals = sorted(vals)
    n = len(vals)
    if n < 2:
        return 0
    return vals[(3 * n) // 4] - vals[n // 4]


def consensus_v2(scheds, weights):
    by_product = defaultdict(list)
    for si, sched in enumerate(scheds):
        for t, p, q in sched:
            by_product[p].append((t, q, si))
    events = []
    total_w = sum(weights) or 1.0
    for p, rows in by_product.items():
        rows.sort()
        used = [False] * len(rows)
        for i, (t0, _, _) in enumerate(rows):
            if used[i]:
                continue
            ts, qs, ws, srcs = [], [], [], set()
            for j in range(i, len(rows)):
                tj, qj, sj = rows[j]
                if tj - t0 > V1.CLUSTER_TOL * 2:
                    break
                if not used[j]:
                    used[j] = True
                    ts.append(tj)
                    qs.append(qj)
                    ws.append(weights[sj])
                    srcs.add(sj)
            support = sum(weights[s] for s in srcs) / total_w
            if support >= V1.MIN_SUPPORT:
                events.append([
                    _wmedian(list(zip(ts, ws))), p,
                    _wmedian(list(zip(qs, ws))),
                    round(support, 3), _iqr(ts), _iqr(qs)])
    events.sort()
    return events


def main():
    idw = json.load(open(V1.IDW, encoding="utf-8"))
    n_classes = len(idw["b"])
    idx = R.load_index()
    recs = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]),
                  reverse=True)[:1600]
    today = dt.date.today()
    members, wrecs = defaultdict(list), defaultdict(list)
    for rec in recs:
        try:
            acts = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        if len(acts) < V1.WINDOW[1]:
            continue
        p = TI.stdlib_predict(idw, F.prefix_features(acts, 288))
        cls = max(range(len(p)), key=lambda i: p[i])
        if p[cls] >= 0.6 and cls != n_classes - 1:
            members[cls].append(V1.dumps_of(acts))
            wrecs[cls].append(rec)

    out = {"classes": {}, "members": {}, "schema":
           "[t, product, qty, support, t_iqr, q_iqr]"}
    for cls, scheds in sorted(members.items()):
        banks = [float(r.get("bank") or 0) for r in wrecs[cls]]
        rng = (min(banks), max(banks)) if banks else (0, 0)
        weights = [_weight(r, today, rng) for r in wrecs[cls]]
        ev = consensus_v2(scheds, weights)
        if ev:
            out["classes"][str(cls)] = ev
            out["members"][str(cls)] = len(scheds)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"))

    # ---- diff vs the shipped v1 file -------------------------------------
    v1_path = V1.OUT
    if os.path.exists(v1_path):
        v1 = json.load(open(v1_path, encoding="utf-8"))
        print("class | v1 events | v2 events | moved(>3t) | tight(iqr<=3)")
        for cls in sorted(out["classes"], key=int):
            e2 = out["classes"][cls]
            e1 = v1.get("classes", {}).get(cls, [])
            moved = 0
            for t2, p2, q2, *_ in e2:
                near = [abs(t2 - t1) for t1, p1, q1 in e1 if p1 == p2]
                if near and min(near) > 3:
                    moved += 1
            tight = sum(1 for e in e2 if e[4] <= 3)
            print(f"  {cls:>3} | {len(e1):>9} | {len(e2):>9} | "
                  f"{moved:>10} | {tight}/{len(e2)}")
    print(f"{len(out['classes'])} classes -> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
