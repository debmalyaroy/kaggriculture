"""Backfill team_score/rank onto index routes from the leaderboard CSV.

B4 unblock (2026-09-05): only 120 of 30,008 current-engine route rows carry
any rating -- the bulk ingest path (backfill_ingest) skips the leaderboard
join that routes.py --mine performs. Without it the >=2500 ELITE PANEL
cannot be assembled at all.

This joins on TEAM NAME against the freshest leaderboard csv on disk (the
same source bc_corpus uses). It fills `team_score` and `rank` where MISSING
and never touches the at-game-time `rating` field. Team score is today's
rating, not at-game rating -- exactly right for selecting elite
demonstrators, and it refreshes with every leaderboard download.

    python src/trackp/harness/backfill_team_scores.py [--engine 1.32.7]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def leaderboard():
    files = sorted(glob.glob(os.path.join(ROOT, ".local", "band_panel",
                                          "lb", "*.csv")))
    if not files:
        raise SystemExit("no leaderboard csv on disk")
    out = {}
    with open(files[-1], encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["TeamName"]] = (float(r["Score"]), int(r["Rank"]))
    return out, os.path.basename(files[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="1.32.7")
    a = ap.parse_args()
    from kaggriculture.trackp import routes_io as R
    lb, src_name = leaderboard()
    idx = R.load_index()
    filled = missing = had = 0
    for k, v in idx["routes"].items():
        if a.engine and v.get("engine") != a.engine:
            continue
        if isinstance(v.get("team_score"), (int, float)) and v["team_score"]:
            had += 1
            continue
        row = lb.get(str(v.get("team")))
        if row is None:
            missing += 1
            continue
        v["team_score"], v["rank"] = row
        filled += 1
    import json as _json
    import tempfile
    # atomic write of the index (routes_io has no save; mirror the miner's
    # write-then-replace so a crash cannot truncate 30k rows)
    d = os.path.dirname(R.INDEX)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        _json.dump(idx, fh)
    os.replace(tmp, R.INDEX)
    print(f"leaderboard {src_name}: filled {filled:,}, already had {had:,}, "
          f"team not on LB {missing:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
