"""World mining: shop-pair worlds, our per-world record, the cloner tail library.

Field intel 2026-09-06 (docs in .local/memory/field-intel-2026-09-06.md):
the rank-1 architecture is a WORLD ROUTER -- a trunk that forks on the
town's shop draw at the unlock turns (t~72 and t~144). The world is
DIRECTLY observable: obs["town"]["unlocked_shops"] (ordered). This module
builds the three data products the router needs:

  1. WORLD INDEX -- every on-disk replay tagged (world1, world2, banks,
     teams, seats).
  2. OUR RECORD PER WORLD -- where the live pair wins and where it bleeds,
     so tails are screened where they matter.
  3. CLONER TAIL LIBRARY -- 32% of fresh seats play OUR v45.0 opening
     verbatim; every such seat's recorded tail (actions from the day-6
     unlock on) shares our prefix, so it can be spliced under a prefix
     guard with zero tile desync. For each cloned seat: episode, seat,
     world, opening agreement vs our line, final banks.

    python src/trackp/harness/worlds.py            # build all three
    python src/trackp/harness/worlds.py --report   # print the per-world table
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

OUT = os.path.join(ROOT, "data", "worlds")
REPLAY_DIRS = (os.path.join(ROOT, ".local", "lossreplays"),
               os.path.join(ROOT, ".local", "hardband", "replays"))
OUR_AGENT = os.path.join(ROOT, "agents", "v45.0_bandit.py")
OPEN_END = 144          # opening window: before the second unlock
AGREE_T = 0.98          # plan-agreement threshold for kinship


def plan_key(act):
    """The plan channel only: farmer + hands, market excluded (siblings
    that changed only selling still share the prefix)."""
    a = act or {}
    return json.dumps([a.get("farmer"), a.get("hands")],
                      separators=(",", ":"))


def our_opening():
    """The v45.0 opening actions, replayed from its own route blob."""
    import base64
    import re
    import zlib
    src = open(OUR_AGENT, encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    route = json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))
    return [plan_key(a) for a in route[:OPEN_END]]


def world_of(steps):
    w = [None, None]
    for t, upto in ((80, 0), (155, 1)):
        if t < len(steps):
            shops = ((steps[t][0].get("observation") or {})
                     .get("town") or {}).get("unlocked_shops") or []
            if len(shops) > 0:
                w[0] = shops[0]
            if len(shops) > 1:
                w[1] = shops[1]
    return tuple(w)


def scan():
    ours = our_opening()
    rows = []
    seen = set()
    for d in REPLAY_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn in seen:
                continue
            seen.add(fn)
            try:
                rep = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            steps = rep.get("steps") or []
            if len(steps) < 700:
                continue
            w1, w2 = world_of(steps)
            info = rep.get("info") or {}
            names = (info.get("TeamNames") or [None, None])
            banks = []
            for s in (0, 1):
                farms = steps[-1][0]["observation"]["farms"]
                banks.append(float(farms[s].get("money") or 0))
            # per-seat opening agreement vs our line (action at t is at t+1)
            agree = []
            for s in (0, 1):
                same = tot = 0
                for t in range(min(OPEN_END, len(steps) - 1)):
                    k = plan_key(steps[t + 1][s].get("action"))
                    same += (k == ours[t])
                    tot += 1
                agree.append(same / tot if tot else 0.0)
            rows.append({"episode": int(fn[:-5]) if fn[:-5].isdigit() else fn,
                         "world": [w1, w2], "banks": banks,
                         "teams": names, "opening_agree": agree,
                         "src": os.path.basename(d)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    idxp = os.path.join(OUT, "index.json")
    if not a.report:
        rows = scan()
        json.dump(rows, open(idxp, "w", encoding="utf-8"))
        print(f"{len(rows)} replays indexed -> {idxp}")
    rows = json.load(open(idxp, encoding="utf-8"))
    # our record per world (seats where a team name is ours)
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    clones = []
    for r in rows:
        w = tuple(r["world"])
        for s in (0, 1):
            if (r["teams"][s] or "") == "Debmalya":
                won = r["banks"][s] > r["banks"][1 - s]
                per[w][0 if won else 1] += 1
            elif r["opening_agree"][s] >= AGREE_T:
                clones.append({"episode": r["episode"], "seat": s,
                               "world": r["world"],
                               "agree": round(r["opening_agree"][s], 3),
                               "bank": r["banks"][s],
                               "opp_bank": r["banks"][1 - s],
                               "team": r["teams"][s]})
    clones.sort(key=lambda c: -c["bank"])
    json.dump(clones, open(os.path.join(OUT, "clone_tails.json"), "w",
                           encoding="utf-8"), indent=1)
    print(f"\nOUR RECORD BY WORLD ({sum(v[0]+v[1] for v in per.values())} "
          f"seat-games):")
    for w, (win, loss) in sorted(per.items(),
                                 key=lambda kv: -(kv[1][0] + kv[1][1])):
        n = win + loss
        print(f"  {str(w):40s} {win:3d}-{loss:<3d} "
              f"({win/n:5.1%})" if n else "")
    print(f"\nCLONE TAIL LIBRARY: {len(clones)} seats share our opening "
          f"(>= {AGREE_T:.0%} plan agreement); top banks:")
    for c in clones[:8]:
        print(f"  ep{c['episode']} seat{c['seat']} {str(tuple(c['world'])):36s}"
              f" bank {c['bank']:>9,.0f} (opp {c['opp_bank']:>9,.0f}) "
              f"agree {c['agree']:.2f} {c['team']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
