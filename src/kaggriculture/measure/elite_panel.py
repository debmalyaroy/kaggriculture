"""Elite gate panel: the freshest winning tape of each CURRENT top-N team.

The reactive gauntlet saturates at ~2300 (our agents go 72-0 on it), so it
cannot see the 2300->2800 gap. This panel is the instrument that can: it is
rebuilt from the LIVE leaderboard on every invocation (the ladder is
dynamic — never a hardcoded team list), takes each top team's freshest
1.32.7 winning route from the index, renders them as tapes, and scores an
agent by paired cells against the whole panel.

    python src/elite_panel.py --build             # refresh panel from live LB
    python src/elite_panel.py --score A.py [B.py] # cells vs current panel
    python src/elite_panel.py --top 100 --build
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(ROOT, ".local", "elite_panel")


def live_top_teams(n=50):
    """Fresh leaderboard fetch, every call — the ladder is dynamic."""
    p = subprocess.run(["kaggle", "competitions", "leaderboard",
                        "kaggriculture", "-s", "--page-size", str(n)],
                       capture_output=True,
                       text=True, timeout=120, encoding="utf-8",
                       errors="replace")
    teams = []
    for ln in (p.stdout or "").splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[0].isdigit():
            score = parts[-1]
            date, tm = parts[-3], parts[-2]
            name = " ".join(parts[1:-3])
            try:
                teams.append((name, float(score)))
            except ValueError:
                continue
    return teams[:n]


def epnum(m):
    try:
        return int(str(m.get("episode") or "0").split("+")[0])
    except ValueError:
        return 0


def build(n=50):
    import kaggriculture.data.routes as R
    import kaggriculture.train.train_arms as TA
    os.makedirs(PANEL_DIR, exist_ok=True)
    teams = live_top_teams(n)
    if len(teams) < n // 2:
        raise SystemExit(f"leaderboard fetch too thin ({len(teams)}) — refuse")
    idx = R.load_index()["routes"]
    best = {}
    for rid, m in idx.items():
        if not (m.get("won") and m.get("engine") == "1.32.7"):
            continue
        t = str(m.get("team", "")).lower()
        for name, score in teams:
            if name.lower() in t or t in name.lower():
                keyv = (epnum(m), float(m.get("bank") or 0))
                if name not in best or keyv > best[name][0]:
                    best[name] = (keyv, rid)
                break
    made = []
    for name, score in teams:
        if name not in best:
            continue
        rid = best[name][1]
        out = os.path.join(PANEL_DIR, f"{rid}.py")
        if not os.path.exists(out):
            TA.render(R.load_route(rid), out, f"elite_{rid}")
        made.append({"team": name, "lb_score": score, "rid": rid,
                     "tape": out, "bank": best[name][0][1]})
    manifest = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "top_n": n, "covered": len(made), "teams": made}
    json.dump(manifest, open(os.path.join(PANEL_DIR, "manifest.json"), "w",
              encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"panel: {len(made)}/{len(teams)} top-{n} teams covered "
          f"(missing = no 1.32.7 win in index yet)")
    return manifest


def score(agent_path, seeds=(501,)):
    """Paired cells vs the panel, with MARGIN and the held-out split.

    Repaired 2026-09-03: this used to return win counts only, so a candidate
    that banked $30k more per game read identically to one that banked $1
    more, and the panel that SELECTED a candidate was the panel that scored
    it. Both are now reported (`band_panel.summarise_cells` /
    `band_panel.assign_splits`) -- the HELD-OUT line is the decision number.
    """
    import kaggriculture.engine.serve_match as SM
    import kaggriculture.measure.band_panel as BP
    man = json.load(open(os.path.join(PANEL_DIR, "manifest.json"),
                    encoding="utf-8"))
    BP.ensure_splits(man["teams"])
    srv = SM.Serve()
    w = l = 0
    rows, cells = [], []
    for t in man["teams"]:
        tw = 0
        for seed in seeds:
            for seat in (0, 1):
                a, b = SM.load_agent(agent_path), SM.load_agent(t["tape"])
                ab, tb = (SM.run_match(a, b, seed, srv) if seat == 0
                          else SM.run_match(b, a, seed, srv)[::-1])
                tw += ab > tb
                cells.append({"own": float(ab), "opp": float(tb),
                              "team": t["team"], "seed": seed, "seat": seat,
                              "split": t["split"]})
        w += tw
        l += 2 * len(seeds) - tw
        rows.append({"team": t["team"], "lb": t["lb_score"], "won": tw,
                     "split": t["split"]})
    swept = [r["team"] for r in rows if r["won"] == 0]
    overall = BP.summarise_cells(cells)
    per_split = {s: BP.summarise_cells([c for c in cells if c["split"] == s])
                 for s in BP.SPLITS}
    print(f"ELITE {os.path.basename(agent_path)}: {BP.fmt_summary(overall)} "
          f"| swept-by {len(swept)}")
    print(f"  SELECTION  {BP.fmt_summary(per_split['selection'])}")
    print(f"  HELD-OUT   {BP.fmt_summary(per_split['holdout'])}"
          "   <- DECISION NUMBER")
    print(f"  worlds     {BP.fmt_world(BP.world_stats(cells))}")
    return {"agent": agent_path, "cells": [w, l], "rows": rows,
            "summary": overall, "by_split": per_split,
            "worlds": BP.world_stats(cells), "panel_built": man["built"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--score", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    if args.build or not os.path.exists(
            os.path.join(PANEL_DIR, "manifest.json")):
        build(args.top)
    out = []
    for a in args.score:
        out.append(score(a, seeds=tuple(500 + i for i in range(1, args.seeds + 1))))
    if out:
        json.dump(out, open(os.path.join(PANEL_DIR, "last_scores.json"), "w",
                  encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
