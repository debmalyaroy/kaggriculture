"""M2 of the relay stack: per-identifier-class dump schedules for collision
targeting.

The heuristic relay assumes the opponent runs OUR schedule -- true only for
exact clones. This computes, for every class the shipped identifier can commit
to, the class's consensus dump schedule (which relay products it dumps, at
which steps, in what size), mined from the routes the identifier itself
assigns to that class. The runtime then races only real collisions, against
THEIR timing:

  - their dump collides with ours       -> pull ours ahead of the earlier one
  - no collision at this dump           -> leave the schedule alone (zero
                                           distortion where a clone-assumption
                                           relay would still have pulled)

M1 note (measured 2026-08-11, models/relay/mirror.json): a learned binary
"will we collide" detector is uninformative -- 99% of route pairs in the
current meta collide, so majority-class = 0.985 = model accuracy. Identity IS
the mirror model; this file is where that identity becomes a schedule.

    python src/relay_config.py            # -> models/relay/family_dumps.json
"""
from kaggriculture.paths import ROOT
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.data.features as F  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402

IDW = os.path.join(ROOT, "models", "v22", "identifier", "weights.json")
OUT = os.path.join(ROOT, "models", "relay", "family_dumps.json")
RELAY_PRODUCTS = ("FERTILIZER", "MELON", "WOOL", "STRAWBERRY", "MILK")
DUMP_QTY = 8
WINDOW = (216, 648)         # relay-relevant part of the season
CLUSTER_TOL = 3             # dump steps within +/-3 are the same event
MIN_SUPPORT = 0.4           # fraction of member routes that must share a dump


def dumps_of(actions):
    out = []
    lo, hi = WINDOW
    for t in range(lo, min(hi, len(actions))):
        turn = actions[t]
        if not isinstance(turn, dict):
            continue
        for o in (turn.get("market") or []):
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] in RELAY_PRODUCTS):
                try:
                    q = int(o[2])
                except (TypeError, ValueError):
                    continue
                if q >= DUMP_QTY:
                    out.append((t, o[1], q))
    return out


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
    return 0 if n < 2 else vals[(3 * n) // 4] - vals[n // 4]


MAX_T_IQR = 6      # events with wider member timing than this are too loose
                   # to race unless their weighted support is decisive


def consensus(schedules, weights=None):
    """Cluster member dumps into consensus events with enough support.

    v2 (2026-08-13, from the relay_config_v2 lab): members are WEIGHTED
    (recency x final bank -- a $155k route from today out-votes a stale
    $120k one), event time/size are the weighted median, and loose events
    (timing IQR > MAX_T_IQR without decisive support) are dropped rather
    than shipped as false precision. Output schema stays [t, product, qty]
    -- the runtime's parser contract; the confidence-aware runtime is a
    panel-gated agent change, not a trainer change."""
    if weights is None:
        weights = [1.0] * len(schedules)
    by_product = defaultdict(list)
    for si, sched in enumerate(schedules):
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
                if tj - t0 > CLUSTER_TOL * 2:
                    break
                if not used[j]:
                    used[j] = True
                    ts.append(tj)
                    qs.append(qj)
                    ws.append(weights[sj])
                    srcs.add(sj)
            support = sum(weights[s] for s in srcs) / total_w
            if support < MIN_SUPPORT:
                continue
            if _iqr(ts) > MAX_T_IQR and support < 2 * MIN_SUPPORT:
                continue
            # extended schema (v24.1 runtime, tolerant of both): the agent
            # races only tight events (t_iqr <= 3 or support >= 0.55) and
            # defers loose ones to live observation.
            events.append([_wmedian(list(zip(ts, ws))), p,
                           _wmedian(list(zip(qs, ws))),
                           round(support, 3), _iqr(ts), _iqr(qs)])
    events.sort()
    return events


def main():
    idw = json.load(open(IDW, encoding="utf-8"))
    n_classes = len(idw["b"])
    idx = R.load_index()
    recs = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]),
                  reverse=True)[:1600]
    import datetime as _dt
    today = _dt.date.today()

    def _weight(rec, lo, hi):
        try:
            age = max(0.0, (today - _dt.date.fromisoformat(
                str(rec.get("date", ""))[:10])).days)
        except ValueError:
            age = 7.0
        recency = 0.5 ** (age / 3.0)           # 3-day half-life
        bank = float(rec.get("bank") or 0.0)
        bank_n = 0.5 if hi <= lo else max(0.0, min(1.0, (bank - lo) / (hi - lo)))
        return recency * (0.25 + 0.75 * bank_n)  # bank scales, never zeroes

    members, mrecs = defaultdict(list), defaultdict(list)
    for rec in recs:
        try:
            acts = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        if len(acts) < WINDOW[1]:
            continue
        p = TI.stdlib_predict(idw, F.prefix_features(acts, 288))
        cls = max(range(len(p)), key=lambda i: p[i])
        if p[cls] >= 0.6 and cls != n_classes - 1:      # skip novel/unsure
            members[cls].append(dumps_of(acts))
            mrecs[cls].append(rec)

    out = {"classes": {}, "members": {}}
    for cls, scheds in sorted(members.items()):
        banks = [float(r.get("bank") or 0) for r in mrecs[cls]]
        lo, hi = (min(banks), max(banks)) if banks else (0, 0)
        ev = consensus(scheds, [_weight(r, lo, hi) for r in mrecs[cls]])
        if ev:
            out["classes"][str(cls)] = ev
            out["members"][str(cls)] = len(scheds)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"))
    print(f"{len(out['classes'])} classes with consensus dump schedules "
          f"({sum(out['members'].values())} member routes)")
    for cls, ev in sorted(out["classes"].items(), key=lambda kv: int(kv[0])):
        head = ", ".join(f"t{e[0]}:{e[1][:4]}x{e[2]}" for e in ev[:5])
        tight = sum(1 for e in ev if len(e) < 6 or e[4] <= 3 or e[3] >= 0.55)
        print(f"  class {cls:>2} ({out['members'][cls]:>4} routes): "
              f"{len(ev)} events ({tight} tight)  {head}")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
