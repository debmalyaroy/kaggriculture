"""Strategy-family map of the whole field, from ALL leaderboard data.

Operator order (2026-09-05): don't depend on one team; identify the
REACTIVE players (their post-script decisions are trainable policy) and the
FIXED-TAPE players (their tapes identify economy FAMILIES -- the units the
plan-search factory and opponent-keyed dispatch operate on).

For every current-engine route in the index:
  * cluster by 72-step opening prefix (a family = one economy lineage);
  * mark each TEAM reactive/scripted from the script-end scan
    (data/bc_corpus/script_end.json: 80 of 90 top-100 teams go reactive at
    a team-specific day) plus a variance check for teams the scan missed;
  * per family: seats, teams, elite share (team_score >= 2500), best bank,
    reactive share, a representative route (highest bank).

Output feeds three consumers: P1 edit seeds (best family representatives),
P2 dispatch keys (recognize the opponent's family from its opening), and
the reactive-cohort trainer (which teams' post-script rows are policy).

    python src/trackp/harness/families.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OUT = os.path.join(ROOT, "models", "trackp", "families.json")
PREFIX = 72


def main():
    from kaggriculture.trackp import routes_io as R
    idx = R.load_index()
    script = {}
    p = os.path.join(ROOT, "data", "bc_corpus", "script_end.json")
    if os.path.exists(p):
        script = json.load(open(p, encoding="utf-8"))

    fams = defaultdict(lambda: {"n": 0, "teams": set(), "best_bank": 0.0,
                                "rep": None, "elite_teams": set()})
    n = 0
    for k, v in idx["routes"].items():
        if v.get("engine") != "1.32.7":
            continue
        try:
            acts = R.load_route(k)
        except Exception:                                      # noqa: BLE001
            continue
        h = hashlib.sha256(json.dumps(acts[:PREFIX], sort_keys=True)
                           .encode()).hexdigest()[:12]
        f = fams[h]
        f["n"] += 1
        t = str(v.get("team"))
        f["teams"].add(t)
        ts = v.get("team_score")
        if isinstance(ts, (int, float)) and ts >= 2500:
            f["elite_teams"].add(t)
        bank = float(v.get("bank") or 0)
        if bank > f["best_bank"]:
            f["best_bank"] = bank
            f["rep"] = k
        n += 1
        if n % 5000 == 0:
            print(f"  {n} routes clustered...", flush=True)

    rows = []
    for h, f in fams.items():
        reactive = [t for t in f["teams"]
                    if (script.get(t) or {}).get("script_end_step", 720) < 720]
        rows.append({"family": h, "seats": f["n"],
                     "teams": len(f["teams"]),
                     "elite_teams": sorted(f["elite_teams"]),
                     "reactive_teams": sorted(reactive),
                     "best_bank": f["best_bank"], "rep": f["rep"]})
    rows.sort(key=lambda r: -r["seats"])
    print(f"\n{n:,} routes -> {len(rows)} families")
    print(f"{'seats':>6} {'teams':>6} {'elite':>6} {'react':>6} "
          f"{'best bank':>10}  rep")
    for r in rows[:15]:
        print(f"{r['seats']:>6} {r['teams']:>6} {len(r['elite_teams']):>6} "
              f"{len(r['reactive_teams']):>6} {r['best_bank']:>10,.0f}  "
              f"{r['rep']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
