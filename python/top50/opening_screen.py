"""Opening divergence: leave the shared lineage opening without losing the economy (queue Q56).

    python python/top50/opening_screen.py [--threads 24] [--top 4]

Stream hashes show our opening is shared, action for action, by thousands of teams (the public lineage
our routes were mined from), and all 22 of v63's ladder losses came from copy races against state-level
copies (same squares, same cash). Playing a different, equally strong opening makes us a PARTIAL /
DIFFERENT opponent to their copy detectors (v63 went 20-0 against those groups).
Each candidate plays route K's actions before day 6 (`--opening-route K`, the router then picks the day-6
route as usual), everything else = v63.1_rl.
  1. SELECTION: the training half of our ladder games + 600 real players' tapes from PPO's training bands,
     paired vs v63.1_rl (same tapes, same seeds)
  2. TEST (never seen by the selection): the held-out ladder half and the 991-tape band gate, for the --top
     candidates
Caveat: recorded opponents cannot react, so "copies stop racing us" is only visible on the ladder; these
tests show whether the economy survives the change.
Writes data/top50/opening_screen/report.json.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_screen import RL, paired, play  # noqa: E402

V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
BASE = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
OUT = os.path.join(RL, "data", "top50", "opening_screen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    routes = sorted(int(k) for k in json.load(open(os.path.join(RL, "configs", "bases", "v61.1", "routes.json"))))
    router = json.load(open(os.path.join(RL, "configs", "bases", "v61.1", "router.json")))
    cands = [r for r in routes if r not in (0, router.get("endgame_route", 2))]
    sel = sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", "ladder", "*.json")))
    band_tr = []
    for d in ("lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"):
        band_tr += sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", d, "*.json")))
    random.Random(7).shuffle(band_tr)
    sel += band_tr[:600]
    ref = play(BASE, sel, a.threads)
    rep = {"selection_games": len(sel), "selection": {}, "test": {}}
    for k in cands:
        r = paired(play(BASE + ["--opening-route", str(k)], sel, a.threads), ref)
        rep["selection"][k] = r
        print(f"[opening] route {k:4d}: +{r['better']}/-{r['worse']} p {r['p']} (wins {r['wins']} vs {r['ref_wins']})", flush=True)
    top = sorted(cands, key=lambda k: rep["selection"][k]["better"] - rep["selection"][k]["worse"], reverse=True)[:a.top]
    va = os.path.join(RL, "data", "tapes", "ladder_val")
    grp = {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(va, "index.tsv"), encoding="utf-8"), delimiter="\t")}
    vf = sorted(glob.glob(os.path.join(va, "**", "*.json"), recursive=True))
    ref_v = play(BASE, vf, a.threads)
    for k in top:
        c = play(BASE + ["--opening-route", str(k)], vf, a.threads)
        t = {"ladder": {"all": paired(c, ref_v)}}
        for G in ("COPY", "PARTIAL", "DIFFERENT"):
            t["ladder"][G] = paired({i: v for i, v in c.items() if grp.get(i) == G}, ref_v)
        r2 = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", "35", "--cand-group", "35,35,36",
                             "--cand-args", f"--opening-route {k}", "--profiles", V64, "--ref-profile", "35", "--ref-group", "35,35,36",
                             "--name", f"opening-{k}", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        t["band"] = (r2.stdout.strip().splitlines() or [r2.stderr[-300:]])[-1]
        rep["test"][k] = t
        x = t["ladder"]["all"]
        print(f"[opening] TEST route {k}: held-out ladder +{x['better']}/-{x['worse']} p {x['p']} | {t['band'][:200]}", flush=True)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    print(f"[opening] -> {OUT}/report.json")


if __name__ == "__main__":
    main()
