"""Harvest sync-compatible route continuations from the field (lineage-matched GM games).

A lineage player whose unit moves equal our pristine route 0 on steps 1..143 is interchangeable with us up
to the step-144 route choice (the D6 dispatch sync rule), so its recorded continuation from step 144 is a
route our bandit can switch to at 144 without desync. Each comes with a real result against a strong
opponent. Per REALIZED world, continuations are grouped (identical unit moves from 144) and compared with
the continuation our own router plays in that world.

    python -m kaggriculture.bandit.harvest_routes [--tapes data/field/lineage2500/tapes] [--days .../days.jsonl]

Needs the tapes' tapedump (for the realized world): tapedump --tapes DIR > DIR/../days.jsonl
Writes data/field/lineage2500/continuations.csv (one row per game) and prints the per-world table.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import pandas as pd

from kaggriculture.paths import ROOT

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def units(a):
    if not isinstance(a, dict):
        return '["PASS"]|[]'
    return json.dumps(a.get("farmer") or ["PASS"]) + "|" + json.dumps(a.get("hands") or [])


def main():
    ap = argparse.ArgumentParser()
    base = os.path.join(ROOT, "data", "field", "lineage2500")
    ap.add_argument("--tapes", default=os.path.join(base, "tapes"))
    ap.add_argument("--days", default=os.path.join(base, "days.jsonl"))
    ap.add_argument("--routes", default=os.path.join(ROOT, "configs", "bandit", "bases", "v61.1", "routes.json"))
    a = ap.parse_args()
    R = json.load(open(a.routes))
    r0 = [units(x) for x in R["0"]]
    ours = {rid: "".join(units(x) for x in tape[144:]) for rid, tape in R.items()}
    ours_h = {hashlib.sha1(v.encode()).hexdigest()[:12]: rid for rid, v in ours.items()}
    world = {}
    for l in open(a.days, encoding="utf-8"):
        d = json.loads(l)
        if d["day"] == 7:
            world[(d["key"], d["seat"])] = "|".join(d["shops"].split("|")[:2])
    rows = []
    for f in sorted(os.listdir(a.tapes)):
        t = json.load(open(os.path.join(a.tapes, f)))
        s = t["seat"]
        key = f[:-5]
        seq = [units(p[s]) for p in t["actions"]]
        div = next((i for i in range(1, 719) if seq[i] != r0[i]), 719)
        cont = hashlib.sha1("".join(seq[144:]).encode()).hexdigest()[:12]
        rw = t["rewards"]
        rows.append({"key": key, "seat": s, "world": world.get((key, s), "?"), "sync144": div >= 144, "diverge": div,
                     "cont": cont, "our_route": ours_h.get(cont), "win": 1.0 if rw[s] > rw[1 - s] else 0.0 if rw[s] < rw[1 - s] else 0.5,
                     "bank": rw[s], "opp_bank": rw[1 - s]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base, "continuations.csv"), index=False)
    sy = df[df.sync144]
    print(f"{len(df)} lineage games; {len(sy)} sync-compatible at 144; {sy.cont.nunique()} distinct continuations; "
          f"{sy.our_route.notna().sum()} games play one of our 41 routes' continuations exactly")
    out = []
    for w, g in sy.groupby("world"):
        c = g.groupby("cont").agg(n=("win", "size"), win=("win", "mean"), ours=("our_route", "first")).sort_values("n", ascending=False)
        top = c.head(3)
        out.append((w, len(g), g.win.mean(), "; ".join(f"{k[:6]}{'(r'+str(r.ours)+')' if isinstance(r.ours, str) else ''} n{r.n} w{r.win:.2f}" for k, r in top.iterrows())))
    for w, n, win, s in sorted(out, key=lambda x: -x[1])[:30]:
        print(f"{w:32} n{n:4} win {win:.2f} | {s}")


if __name__ == "__main__":
    main()
