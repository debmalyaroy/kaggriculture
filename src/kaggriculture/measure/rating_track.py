"""Append one rating snapshot per active submission to a time series.

The 'bandit is declining' scare of 2026-08-15 was partly an artifact of
comparing submissions retired at different ages -- ratings CONVERGE with
games played, and we had no curve to see it. Kaggle exposes only the current
publicScore, so history must be collected while it happens: this runs from
the hourly scrape (one cheap submissions call, the same one stage_detect
makes) and appends to data/lb/rating_track.jsonl:

    {"ts": ..., "ref": ..., "agent": ..., "score": ..., "games": ...}

`games` is the count in the ourgames index at snapshot time (may lag the
ladder by one mine). The dashboard renders these as rating-vs-time curves
with equal-window comparison. Never fetches replays; never touches quota.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import csv
import datetime as dt
import io
import json
import os
import re
import subprocess
import sys

OUT = os.path.join(ROOT, "data", "lb", "rating_track.jsonl")
OURGAMES = os.path.join(ROOT, "data", "ourgames", "index.json")


def _kaggle_cmd():
    try:
        import kaggriculture.data.episodes as E
        return E.kaggle_cmd()
    except Exception:                                              # noqa: BLE001
        return ["kaggle"]


def snapshot():
    r = subprocess.run(_kaggle_cmd() + ["competitions", "submissions",
                                        "kaggriculture", "-v"],
                       capture_output=True, text=True, timeout=300)
    rows = list(csv.DictReader(io.StringIO(r.stdout or "")))
    if not rows:
        print("no submissions parsed (throttled?); nothing appended")
        return 0
    games_by_sub = {}
    try:
        idx = json.load(open(OURGAMES, encoding="utf-8"))
        for g in (idx.get("games") or {}).values():
            s = str(g.get("submission") or "")
            games_by_sub[s] = games_by_sub.get(s, 0) + 1
    except (OSError, ValueError):
        pass
    now = dt.datetime.now().isoformat(timespec="seconds")
    out = []
    # Newest two submissions are the active pair; track only those (retired
    # ratings freeze, so appending them forever is noise).
    for row in rows[:2]:
        ref = str(row.get("ref") or "")
        m = re.search(r"(v[\w.]+_(?:bandit|route2?|planner)\S*?\.py)",
                      row.get("description") or "")
        try:
            score = float(row.get("publicScore") or "nan")
        except ValueError:
            continue
        out.append({"ts": now, "ref": ref,
                    "agent": m.group(1) if m else "?",
                    "score": score, "games": games_by_sub.get(ref, 0)})
    if out:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a", encoding="utf-8") as fh:
            for o in out:
                fh.write(json.dumps(o) + "\n")
        for o in out:
            print(f"tracked {o['agent']} ({o['ref']}): {o['score']} "
                  f"after {o['games']} indexed game(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(snapshot())
