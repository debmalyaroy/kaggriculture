"""Replay-wave tracker: who is cloning whom, and are they cloning US?

Measured by sobameshi (discussion 739273, 2026-09-05): replay waves clone a
strong public episode within ~1 day (18 teams on one opening in the 09-02
dump); once an opening falls below 5%% of seats it never returns. Two
consequences this tool watches for:

  * OUR openings being cloned -- the moment our economy is the wave, half
    the field plays our mirror and our edge halves;
  * NEW wave sources -- the freshest strong economy in the field is exactly
    what the daily economy re-screen should ingest first.

Uses the same-day route index (already mined hourly): groups fresh routes
by opening prefix hash (first N steps), reports the biggest clusters, and
flags any cluster whose prefix matches OUR live bases.

    python src/trackp/harness/meta_tracker.py [--days 2] [--prefix 72]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OUT = os.path.join(ROOT, "models", "trackp", "meta_waves.json")
OURS = [os.path.join(ROOT, "agents", "v44.1_bandit.py"),
        os.path.join(ROOT, "agents", "v45.0_bandit.py")]


def prefix_hash(acts, n):
    return hashlib.sha256(json.dumps(acts[:n], sort_keys=True)
                          .encode()).hexdigest()[:12]


def our_prefixes(n):
    import importlib.util
    out = {}
    for p in OURS:
        if not os.path.exists(p):
            continue
        spec = importlib.util.spec_from_file_location("a", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            out[prefix_hash(mod._ROUTE, n)] = os.path.basename(p)
        except Exception:                                      # noqa: BLE001
            continue
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--prefix", type=int, default=72,
                    help="opening length in steps (72 = day 3)")
    a = ap.parse_args()
    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    cutoff = (dt.date.today() - dt.timedelta(days=a.days)).isoformat()
    fresh = [(k, v) for k, v in idx["routes"].items()
             if str(v.get("date", "")) >= cutoff
             and v.get("engine") == "1.32.7"]
    print(f"routes since {cutoff}: {len(fresh)}")

    clusters = Counter()
    teams_by = defaultdict(set)
    sample = {}
    for k, v in fresh:
        try:
            acts = R.load_route(k)
        except Exception:                                      # noqa: BLE001
            continue
        h = prefix_hash(acts, a.prefix)
        clusters[h] += 1
        teams_by[h].add(str(v.get("team")))
        sample.setdefault(h, k)

    mine = our_prefixes(a.prefix)
    total = sum(clusters.values()) or 1
    rows = clusters.most_common(10)
    print(f"\n{'share':>7} {'seats':>6} {'teams':>6}  opening (sample route)")
    alert = []
    for h, n in rows:
        us = f"  <== OUR {mine[h]}" if h in mine else ""
        if h in mine and n / total > 0.05:
            alert.append((mine[h], n / total))
        print(f"{n / total:>6.1%} {n:>6} {len(teams_by[h]):>6}  "
              f"{h} ({sample[h]}){us}")
    for name, share in alert:
        print(f"\nALERT: {name} opening is {share:.0%} of fresh seats -- "
              f"the field is cloning us; expect mirror matches.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"when": dt.datetime.now().isoformat(timespec="seconds"),
               "total": total,
               "clusters": [{"hash": h, "n": n,
                             "teams": sorted(teams_by[h]),
                             "sample": sample[h],
                             "ours": mine.get(h)}
                            for h, n in rows]},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
