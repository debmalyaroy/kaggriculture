"""World-conditioned plan choice: refit the router's per-world route table (queue Q54).

    python python/top50/route_screen.py [--tapes 1800] [--threads 24] [--min-games 6]

At day 6 (step 144) the router picks one of our mined routes from the first two shops (64 worlds). The world
is already drawn by then (the second shop at step 143), so switching the route does not change the world
and the comparison is clean. The top-100 analysis showed each top team plays a per-world plan table
(plan uncertainty 2.87 bits -> 0.57 given team and world), and the wool family dominates yarn worlds.
  1. SELECTION set: real opponents' recorded streams from PPO's training pool (bands + 2700+ players), never
     the band gate's tapes or our ladder games.
  2. v63.1_rl (profiles v64rl, profile 35, group 35,35,36) plays them as is (recorded, then traced for the world
     each game lands in) and once per candidate route with `--force-route R`.
  3. per world: the route with the best paired result vs the current table (better - worse >= 2, at least
     --min-games games in that world); other worlds keep their current route.
  4. TEST: v63.1_rl + the refitted table (`--route-table`) vs v63.1_rl on the held-out ladder half and the band gate.
Writes configs/route_tables/refit_v1.json (world -> route) and data/top50/route_screen/report.json.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_screen import RL, paired, play  # noqa: E402

BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
TRACE = os.path.join(BIN, "trace" + (".exe" if os.name == "nt" else ""))
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
BASE = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
OUT = os.path.join(RL, "data", "top50", "route_screen")


def worlds_of(files, threads):
    """world (first two shops) each game lands in under v63.1_rl: record the games, trace them."""
    rec = os.path.join(OUT, "rec")
    shutil.rmtree(rec, ignore_errors=True)
    os.makedirs(rec)
    n = max(1, threads)
    shards = [files[i::n] for i in range(n)]

    def one(sh):
        d = tempfile.mkdtemp(prefix="rscr-")
        try:
            for f in sh:
                os.symlink(f, os.path.join(d, os.path.basename(f)))
            subprocess.run([TAPEPLAY, "--tapes", d, *BASE, "--record", rec], cwd=RL, capture_output=True, text=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(one, shards))
    tr = os.path.join(OUT, "rec.tsv")
    subprocess.run([TRACE, "--tapes", rec, "--out", tr, "--seats", "tape"], capture_output=True, text=True)
    w = {}
    for ln in open(tr, encoding="utf-8"):
        if ln.startswith("W\t"):
            x = ln.rstrip("\n").split("\t")
            w[x[1][:-3] if x[1].endswith("__r") else x[1]] = x[5]
    shutil.rmtree(rec, ignore_errors=True)
    os.remove(tr)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", type=int, default=1800)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--min-games", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    routes = sorted(int(k) for k in json.load(open(os.path.join(RL, "configs", "bases", "v61.1", "routes.json"))))
    router = json.load(open(os.path.join(RL, "configs", "bases", "v61.1", "router.json")))
    cands = [r for r in routes if r not in (0, router.get("endgame_route", 2))]
    pool = []
    for d in ("lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus", "top"):
        pool += sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", d, "**", "*.json"), recursive=True))
    pool = sorted({os.path.realpath(f) for f in pool})
    random.Random(11).shuffle(pool)
    files = pool[:a.tapes]
    print(f"[routes] {len(files)} selection tapes, {len(cands)} candidate routes", flush=True)
    world = worlds_of(files, a.threads)
    print(f"[routes] worlds known for {len(world)} games, {len(set(world.values()))} distinct", flush=True)
    base = play(BASE, files, a.threads)
    res = {}
    for r in cands:
        res[r] = play(BASE + ["--force-route", str(r)], files, a.threads)
        print(f"[routes] route {r}: {sum(v == 1 for v in res[r].values())} wins vs table {sum(v == 1 for v in base.values())}", flush=True)
    by_world = {}
    for tid, w in world.items():
        by_world.setdefault(w, []).append(tid)
    table, rows = {}, []
    for w, ids in sorted(by_world.items()):
        best = None
        for r in cands:
            b = sum(res[r].get(i, 0) > base.get(i, 0) for i in ids)
            l = sum(res[r].get(i, 0) < base.get(i, 0) for i in ids)
            if best is None or b - l > best[1] - best[2]:
                best = (r, b, l)
        n = best[1] + best[2]
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(best[1], best[2]) + 1)) / 2 ** n) if n else 1.0
        take = len(ids) >= a.min_games and best[1] - best[2] >= 2 and p < 0.3
        rows.append({"world": w, "games": len(ids), "best_route": best[0], "better": best[1], "worse": best[2], "p": round(p, 3), "adopted": take})
        if take:
            table[w] = best[0]
    os.makedirs(os.path.join(RL, "configs", "route_tables"), exist_ok=True)
    tpath = os.path.join(RL, "configs", "route_tables", "refit_v1.json")
    json.dump(table, open(tpath, "w"), indent=1)
    print(f"[routes] refit table: {len(table)} of {len(by_world)} worlds changed -> {tpath}", flush=True)
    rep = {"selection_tapes": len(files), "routes": cands, "per_world": rows, "table": table,
           "route_totals": {r: int(sum(v == 1 for v in res[r].values())) for r in cands}, "table_wins": int(sum(v == 1 for v in base.values()))}
    # TEST on data the selection never saw
    va = os.path.join(RL, "data", "tapes", "ladder_val")
    grp = {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(va, "index.tsv"), encoding="utf-8"), delimiter="\t")}
    vf = sorted(glob.glob(os.path.join(va, "**", "*.json"), recursive=True))
    ref = play(BASE, vf, a.threads)
    cand = play(BASE + ["--route-table", tpath], vf, a.threads)
    rep["test_ladder"] = {"all": paired(cand, ref)}
    for G in ("COPY", "PARTIAL", "DIFFERENT"):
        rep["test_ladder"][G] = paired({k: v for k, v in cand.items() if grp.get(k) == G}, ref)
    x = rep["test_ladder"]["all"]
    print(f"[routes] held-out ladder: +{x['better']}/-{x['worse']} p {x['p']}", flush=True)
    r2 = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", "35", "--cand-group", "35,35,36",
                         "--cand-args", f"--route-table {tpath}", "--profiles", V64, "--ref-profile", "35", "--ref-group", "35,35,36",
                         "--name", "route-refit-v1", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
    rep["test_band"] = (r2.stdout.strip().splitlines() or [r2.stderr[-400:]])[-1]
    print(r2.stdout[-1500:], flush=True)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(f"[routes] -> {OUT}/report.json")


if __name__ == "__main__":
    main()
