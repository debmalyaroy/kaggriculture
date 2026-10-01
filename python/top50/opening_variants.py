"""Opening divergence from the top teams' own openings (queue Q57; replaces the invalid Q56 screen).

    python python/top50/opening_variants.py [--families 12] [--threads 24] [--top 4]

Every route in our library starts with route 0's first 144 steps (they were spliced onto one opening), so
there is no alternative opening inside it. The top teams' GM games hold other lineages. For each of the
--families most-played top-team openings through turn 136 that differ from ours (stream hashes, GM method)
and win at least as often as ours, the best-scoring game's first 144 actions of the top seat replace route
0's first 144 steps in a copy of the base (data/top50/opening_bases/<hash>/); from day 6 on it is v63.1_rl.
Selection / test exactly as opening_screen.py (train-half ladder + 600 band-train tapes; held-out ladder +
band gate for the --top candidates).
Writes data/top50/opening_variants/report.json.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_screen import RL, paired, play  # noqa: E402

V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
BASE_DIR = os.path.join(RL, "configs", "bases", "v61.1")
OUT = os.path.join(RL, "data", "top50", "opening_variants")
OB = os.path.join(RL, "data", "top50", "opening_bases")


def h136(tape, seat):
    h = hashlib.sha256()
    for pair in tape["actions"][:136]:
        a = pair[seat] if len(pair) > seat and isinstance(pair[seat], dict) else {}
        h.update(json.dumps(a, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=12)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    for d in (OUT, OB):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    ours = set()
    for f in glob.glob(os.path.join(RL, "data", "ladder", "*", "tapes", "*.json")):
        t = json.load(open(f, encoding="utf-8"))
        ours.add(h136(t, t["seat"]))
    fam = defaultdict(list)
    for f in glob.glob(os.path.join(RL, "data", "top50", "gm_local", "tapes", "*", "*.json")) + glob.glob(os.path.join(RL, "data", "top50", "fetch", "tapes", "*", "*.json")):
        t = json.load(open(f, encoding="utf-8"))
        s = t["seat"]
        rw = t.get("rewards") or [0, 0]
        win = 1.0 if (rw[s] or 0) > (rw[1 - s] or 0) else 0.0 if (rw[s] or 0) < (rw[1 - s] or 0) else 0.5
        fam[h136(t, s)].append((win, (rw[s] or 0) - (rw[1 - s] or 0), f, t.get("team")))
    cand = [(k, v) for k, v in fam.items() if k not in ours and len(v) >= 10]
    cand.sort(key=lambda kv: (-len(kv[1]), -sum(x[0] for x in kv[1]) / len(kv[1])))
    cand = cand[:a.families]
    routes = json.load(open(os.path.join(BASE_DIR, "routes.json")))
    fams = []
    for k, v in cand:
        best = max(v, key=lambda x: (x[0], x[1]))
        t = json.load(open(best[2], encoding="utf-8"))
        s = t["seat"]
        opening = [p[s] if len(p) > s and isinstance(p[s], dict) else {"farmer": ["PASS"], "hands": [], "market": []} for p in t["actions"][:144]]
        r2 = dict(routes)
        r2["0"] = opening + routes["0"][144:]
        d = os.path.join(OB, k)
        os.makedirs(d)
        json.dump(r2, open(os.path.join(d, "routes.json"), "w"))
        shutil.copy(os.path.join(BASE_DIR, "router.json"), d)
        teams = sorted({x[3] for x in v if x[3]})
        fams.append({"hash": k, "games": len(v), "win": round(sum(x[0] for x in v) / len(v), 3), "teams": teams[:10], "base": d})
        print(f"[openvar] family {k}: {len(v)} top games, win {fams[-1]['win']}, teams {teams[:6]}", flush=True)
    sel = sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", "ladder", "*.json")))
    band_tr = []
    for dd in ("lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"):
        band_tr += sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", dd, "*.json")))
    random.Random(7).shuffle(band_tr)
    sel += band_tr[:600]
    base = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
    ref = play(base, sel, a.threads)
    for f in fams:
        f["selection"] = paired(play(["--base", f["base"]] + base, sel, a.threads), ref)
        x = f["selection"]
        print(f"[openvar] {f['hash']}: +{x['better']}/-{x['worse']} p {x['p']} (wins {x['wins']} vs {x['ref_wins']})", flush=True)
    top = sorted(fams, key=lambda f: f["selection"]["better"] - f["selection"]["worse"], reverse=True)[:a.top]
    va = os.path.join(RL, "data", "tapes", "ladder_val")
    grp = {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(va, "index.tsv"), encoding="utf-8"), delimiter="\t")}
    vf = sorted(glob.glob(os.path.join(va, "**", "*.json"), recursive=True))
    ref_v = play(base, vf, a.threads)
    for f in top:
        c = play(["--base", f["base"]] + base, vf, a.threads)
        f["test_ladder"] = {"all": paired(c, ref_v)}
        for G in ("COPY", "PARTIAL", "DIFFERENT"):
            f["test_ladder"][G] = paired({i: v for i, v in c.items() if grp.get(i) == G}, ref_v)
        r2 = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", "35", "--cand-group", "35,35,36",
                             "--cand-args", f"--base {f['base']}", "--profiles", V64, "--ref-profile", "35", "--ref-group", "35,35,36",
                             "--name", f"openvar-{f['hash']}", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        f["test_band"] = (r2.stdout.strip().splitlines() or [r2.stderr[-300:]])[-1]
        x = f["test_ladder"]["all"]
        print(f"[openvar] TEST {f['hash']}: held-out ladder +{x['better']}/-{x['worse']} p {x['p']} (COPY +{f['test_ladder']['COPY']['better']}/-{f['test_ladder']['COPY']['worse']}) | {f['test_band'][:180]}", flush=True)
    json.dump({"families": fams, "our_openings": len(ours)}, open(os.path.join(OUT, "report.json"), "w"), indent=1, ensure_ascii=False)
    print(f"[openvar] -> {OUT}/report.json")


if __name__ == "__main__":
    main()
