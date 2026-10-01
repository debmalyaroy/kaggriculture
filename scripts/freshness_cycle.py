"""Freshness treadmill: keep the bandit base younger than the meta.

The decline mechanism (measured 2026-08-31): a copied base ages against a
moving meta -- v36 fell 2480 -> 2260 in ~30h. Confirmed 2026-09-03 from the
ladder: 97% of the top 200 submitted within 24h (median age 6.2h at top-100),
and teams harvested days earlier as "rank 3" now sit at 142. A base decays in
days, so a fresh gated candidate must always be ready.

REWRITTEN 2026-09-03. The previous version was actively harmful:

  * it mined a HARDCODED team (Kaileh57) whose strategy changed on 2026-08-31
    to a script that goes 2-94 against our own gauntlet -- the treadmill would
    have re-based us onto it every 6 hours;
  * it gated on `src/gauntlet.py`, which is SATURATED (96-0 for candidate and
    incumbent alike -- it discriminates nothing);
  * its panel opponents were four hardcoded paths that may not exist;
  * it picked the "live" bandit by string-sorting a glob, so v9.0 sorts after
    v43.0 and the wrong incumbent gets compared.

What it does now, each run:

  1. ingest staged replays into the route index
  2. pick candidate bases from the CURRENT top-30 teams on a freshly
     downloaded leaderboard, newest 1.32.7 win per team, and RECORD the
     source team's live rating (a 2026-08-27 lesson: +12pp offline
     candidates came from teams rated 1855-1960 -- always check the source)
  3. build the v22 chassis on each
  4. gate on `src/band_panel.py` -- the MID band is the bar that predicts
     convergence (it is what killed v42.0 at 0.775 while the gauntlet said
     96-0) -- reading the HELD-OUT split and the MARGIN, not raw cells
  5. paired `serve_gate` vs the true newest live bandit; REGRESSION is loud
  6. write a report with the verdict and the manual submit commands

NEVER submits, never pushes a kernel. Schedule deliberately:

  schtasks /Create /TN KaggricultureFreshness /SC HOURLY /MO 6 ^
      /TR "python D:\\codebase\\kaggriculture\\scripts\\freshness_cycle.py"
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

SRC = os.path.join(ROOT, "src")
LOG = os.path.join(ROOT, ".local", "freshness")
LB_DIR = os.path.join(ROOT, ".local", "freshness", "lb")
# How many current top teams to consider, and how many bases to actually build.
TOP_TEAMS = 30
N_BASES = 3


def run(cmd, timeout=3600):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout, encoding="utf-8",
                       errors="replace")
    tail = "\n".join(((p.stdout or "") + (p.stderr or "")).splitlines()[-8:])
    print(tail, flush=True)
    return p.returncode, tail


def newest_live_bandit():
    """The live bandit by VERSION, not by string sort.

    `sorted(glob(...))[-1]` put v9.0_bandit after v43.0_bandit and silently
    compared every candidate against the wrong incumbent.
    """
    best, bestv = None, (-1, -1)
    for p in glob.glob(os.path.join(ROOT, "agents", "v*_bandit.py")):
        m = re.search(r"v(\d+)(?:\.(\d+))?_", os.path.basename(p))
        if not m:
            continue
        v = (int(m.group(1)), int(m.group(2) or 0))
        if v > bestv:
            best, bestv = p, v
    return best, bestv


def live_top_teams(n=TOP_TEAMS):
    """(name, score) for the current top n, from a FRESH leaderboard CSV."""
    os.makedirs(LB_DIR, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "leaderboard", "kaggriculture",
                    "--download", "-p", LB_DIR], capture_output=True,
                   text=True, timeout=300)
    z = os.path.join(LB_DIR, "kaggriculture.zip")
    if os.path.exists(z):
        import zipfile
        with zipfile.ZipFile(z) as zf:
            for nm in zf.namelist():
                if nm.endswith(".csv"):
                    safe = os.path.join(LB_DIR,
                                        os.path.basename(nm).replace(":", "_"))
                    with zf.open(nm) as fi, open(safe, "wb") as fo:
                        fo.write(fi.read())
        os.remove(z)
    csvs = sorted(glob.glob(os.path.join(LB_DIR, "*.csv")),
                  key=os.path.getmtime)
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(ROOT, ".local", "lb*", "*.csv")),
                      key=os.path.getmtime)
    if not csvs:
        return []
    rows = []
    with open(csvs[-1], encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((r.get("TeamName", "").strip(), float(r["Score"])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda x: -x[1])
    return rows[:n]


def fresh_bases(teams, k=N_BASES):
    """Newest 1.32.7 win per current top team -> [(rid, team, lb_score)].

    Freshest by episode id, NOT biggest bank: bank is a shared-world quantity
    and a fraudulent tape signal under 1.32.7 scarcity pricing.
    """
    idx_path = os.path.join(ROOT, "data", "routes", "index.json")
    idx = json.load(open(idx_path, encoding="utf-8"))["routes"]
    want = {t.lower(): s for t, s in teams}
    best = {}
    for rid, m in idx.items():
        if not (m.get("won") and m.get("engine") == "1.32.7"):
            continue
        t = str(m.get("team", "")).strip().lower()
        if t not in want:
            continue
        try:
            ep = int(str(m.get("episode") or "0").split("+")[0])
        except ValueError:
            continue
        if t not in best or ep > best[t][0]:
            best[t] = (ep, rid, m.get("team"), want[t])
    ranked = sorted(best.values(), key=lambda x: -x[0])
    return [(rid, team, score) for _ep, rid, team, score in ranked[:k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bases", type=int, default=N_BASES)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    py = sys.executable
    report = {"when": stamp, "steps": {}, "candidates": []}

    if not args.skip_ingest:
        _, out = run([py, "-X", "utf8", os.path.join(ROOT, "src", "kaggriculture", "data", "backfill_ingest.py"),
                      "--jobs", "4"])
        report["steps"]["ingest"] = out

    teams = live_top_teams()
    bases = fresh_bases(teams, args.bases) if teams else []
    report["top_teams_sampled"] = len(teams)
    report["bases"] = [{"rid": r, "team": t, "lb_score": s}
                       for r, t, s in bases]
    print(f"bases from current top-{TOP_TEAMS}: "
          + ", ".join(f"{r} ({t} @{s:.0f})" for r, t, s in bases), flush=True)
    if not bases:
        report["steps"]["bases"] = "NO BASES -- leaderboard or index unusable"
        json.dump(report, open(os.path.join(LOG, f"report_{stamp}.json"), "w",
                               encoding="utf-8"), indent=1, ensure_ascii=False)
        print("no bases; nothing to gate")
        return 0

    live, lv = newest_live_bandit()
    print(f"incumbent: {live} v{lv}", flush=True)

    for rid, team, score in bases:
        cand = os.path.join(".local", "candidates", f"fresh_{stamp}_{rid}.py")
        code, out = run([py, os.path.join(ROOT, "src", "kaggriculture", "agentbuild", "v22_agent.py"), "--base", rid,
                         "--arms", "--premium-lead", "--out", cand])
        entry = {"rid": rid, "team": team, "lb_score": score,
                 "path": cand, "build": out}
        if code != 0:
            entry["verdict"] = "BUILD FAILED"
            report["candidates"].append(entry)
            continue
        # The MID band is the bar that predicts convergence; the reactive
        # gauntlet is saturated and is hygiene only. band_panel reports the
        # held-out split and the median margin -- read those, not raw cells.
        _, out = run([py, "-X", "utf8", os.path.join(ROOT, "src", "kaggriculture", "measure", "band_panel.py"),
                      "--score", cand, live, "--seeds", str(args.seeds),
                      "--workers", str(args.workers)], timeout=7200)
        entry["band_panel"] = out
        # CLUSTER-TRAP GATE: a dominant cluster can BE the source's new and
        # weaker strategy (Kaileh's 2026-08-31 script went 2-94). Paired
        # serve gate against the true incumbent; REGRESSION is loud.
        code, out = run([py, "-X", "utf8", os.path.join(ROOT, "src", "kaggriculture", "engine", "serve_gate.py"),
                         cand, live], timeout=3600)
        entry["live_gate"] = out
        entry["live_gate_pass"] = ("REGRESSION" not in out)
        entry["verdict"] = ("CANDIDATE" if entry["live_gate_pass"]
                            else "REGRESSION -- rejected")
        report["candidates"].append(entry)

    json.dump(report, open(os.path.join(LOG, f"report_{stamp}.json"), "w",
                           encoding="utf-8"), indent=1, ensure_ascii=False)
    passed = [c for c in report["candidates"]
              if c.get("verdict") == "CANDIDATE"]
    print(f"\nreport: .local/freshness/report_{stamp}.json")
    print(f"{len(passed)}/{len(report['candidates'])} candidates passed the "
          f"live gate. SUBMIT IS ALWAYS MANUAL:")
    for c in passed:
        print(f"  {c['path']}   (base {c['rid']} from {c['team']} "
              f"@{c['lb_score']:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
