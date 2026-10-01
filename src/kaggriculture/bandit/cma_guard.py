"""Overfitting guard for a CMA-ES shell run: score checkpoints against opponents the fitness NEVER sees.

cmaes_shell's fitness and its held-out check both play the same copy population (v63, v62.1, v62, v61.1, the base
profile, random-knob clones); the held-out check only changes the seeds. A vector can therefore over-fit to that
population and still pass. This guard plays each checkpoint, paired against the base profile, on:

  REAL   our real ladder games (.local/tapes/{v61,v611,v62,v621}, 324 tapes): the recorded opponent replays its moves
         (open loop, the opponents that actually played us);
  TEAM   closed-loop games vs the 10 top-50 team bandits (data/field/teams/bases/<team>/S1, shell on), both seats,
         fixed seeds disjoint from the CMA banks.

    python -m kaggriculture.bandit.cma_guard --run cma2 [--every 5] [--threads 2] [--watch]

A checkpoint = the mean vector of every --every-th generation plus every held-out check vector. Results go to
.local/cmaes/<run>/guard.jsonl. Verdict per checkpoint: OVERFIT when REAL+TEAM together are paired-worse than the base
(more worse than better, sign-test p < 0.1); the run's best vector is the latest checkpoint that is not OVERFIT and
whose held-out check is not below an earlier one's by more than noise.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from kaggriculture.paths import ROOT

REL = os.path.join(ROOT, "rustengine", "v62", "target-x", "release")
BASE_DIR = os.path.join(ROOT, "configs", "bandit", "bases", "v61.1")
V4 = os.path.join(ROOT, "configs", "bandit", "profiles", "v4.json")
TAPES = [os.path.join(ROOT, ".local", "tapes", d) for d in ("v61", "v611", "v62", "v621")]
TEAMS = os.path.join(ROOT, "data", "field", "teams", "bases")
TEAM_SEEDS = list(range(770000, 770016))
TEAM_PROFILE = 71  # the team bandits' shell knobs (v63's), as in the stage-coverage runs


def sign_p(b, w):
    n = b + w
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0


def real_job(args):
    tapes, prof, pid = args
    out = subprocess.run([os.path.join(REL, "tapeplay.exe"), "--tapes", tapes, "--base", BASE_DIR, "--profiles", prof, "--pa", str(pid)],
                         capture_output=True, text=True).stdout
    res = {}
    for ln in out.splitlines():
        p = ln.split("\t")
        if len(p) >= 6:
            us, them = float(p[4]), float(p[5])
            res[("real", os.path.basename(tapes), p[0], p[1])] = 1.0 if us > them else 0.0 if us < them else 0.5
    return res


def team_job(args):
    team, seat, prof, pid = args
    tb = os.path.join(TEAMS, team, "S1")
    a, b, pa, pb = (BASE_DIR, tb, pid, TEAM_PROFILE) if seat == 0 else (tb, BASE_DIR, TEAM_PROFILE, pid)
    out = subprocess.run([os.path.join(REL, "selfplay.exe"), "--a", a, "--b", b, "--profiles", prof, "--pa", str(pa), "--pb", str(pb),
                          "--seeds", ",".join(map(str, TEAM_SEEDS)), "--threads", "1"], capture_output=True, text=True).stdout
    res = {}
    for ln in out.splitlines():
        p = ln.split("\t")
        if len(p) >= 3 and p[0].isdigit():
            us, them = (float(p[1]), float(p[2])) if seat == 0 else (float(p[2]), float(p[1]))
            res[("team", team, seat, p[0])] = 1.0 if us > them else 0.0 if us < them else 0.5
    return res


def score(prof, pid, threads):
    jobs = [(real_job, (t, prof, pid)) for t in TAPES]
    teams = sorted(d for d in os.listdir(TEAMS) if os.path.isdir(os.path.join(TEAMS, d, "S1")))
    jobs += [(team_job, (t, s, prof, pid)) for t in teams for s in (0, 1)]
    res = {}
    with ThreadPoolExecutor(threads) as ex:
        for r in ex.map(lambda j: j[0](j[1]), jobs):
            res.update(r)
    return res


def compare(c, b):
    out = {}
    for part in ("real", "team", "all"):
        ks = [k for k in set(c) & set(b) if part == "all" or k[0] == part]
        bw, ww = sum(c[k] > b[k] for k in ks), sum(c[k] < b[k] for k in ks)
        out[part] = {"n": len(ks), "score": sum(c[k] for k in ks) / max(1, len(ks)), "base": sum(b[k] for k in ks) / max(1, len(ks)),
                     "better": bw, "worse": ww, "p": sign_p(bw, ww)}
    return out


def checkpoints(run_dir, every):
    cps = []
    for ln in open(os.path.join(run_dir, "log.jsonl"), encoding="utf-8"):
        d = json.loads(ln)
        if d.get("check"):
            cps.append((f"check_g{d['gen']}", d["gen"], d["knobs"], d))
        elif (d["gen"] + 1) % every == 0:
            cps.append((f"mean_g{d['gen']}", d["gen"], d["mean_knobs"], d))
    return cps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cma2")
    ap.add_argument("--base", type=int, default=100)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--watch", action="store_true", help="keep scoring new checkpoints until the run's last check")
    ap.add_argument("--gens", type=int, default=30, help="the run's generation count (for --watch)")
    a = ap.parse_args()
    run_dir = os.path.join(ROOT, ".local", "cmaes", a.run)
    gpath = os.path.join(run_dir, "guard.jsonl")
    done = set()
    if os.path.exists(gpath):
        done = {json.loads(l)["name"] for l in open(gpath, encoding="utf-8")}
    v4 = json.load(open(V4, encoding="utf-8"))
    base = v4["profiles"][a.base]
    prof = os.path.join(run_dir, "guard_profiles.json")
    base_res = None
    while True:
        todo = [c for c in checkpoints(run_dir, a.every) if c[0] not in done]
        for name, gen, knobs, rec in todo:
            table = copy.deepcopy(v4)
            table["profiles"].append({**{k: v for k, v in base.items() if k != "name"}, **knobs, "name": f"guard_{name}"})
            json.dump(table, open(prof, "w", encoding="utf-8"))
            if base_res is None:
                t0 = time.time()
                base_res = score(prof, a.base, a.threads)
                print(f"[guard] base P{a.base}: {len(base_res)} games in {time.time() - t0:.0f}s", flush=True)
            t0 = time.time()
            cr = score(prof, len(table["profiles"]) - 1, a.threads)
            cmp = compare(cr, base_res)
            al = cmp["all"]
            overfit = al["worse"] > al["better"] and al["p"] < 0.1
            out = {"name": name, "gen": gen, "train_fit": rec.get("fit_of_mean"), "heldout": rec.get("score"), "cmp": cmp,
                   "verdict": "OVERFIT" if overfit else "ok", "knobs": knobs}
            open(gpath, "a", encoding="utf-8").write(json.dumps(out) + "\n")
            done.add(name)
            r, t = cmp["real"], cmp["team"]
            print(f"[guard] {name}: REAL {r['score']:.3f} vs {r['base']:.3f} +{r['better']}/-{r['worse']} | TEAM {t['score']:.3f} vs "
                  f"{t['base']:.3f} +{t['better']}/-{t['worse']} | all p {al['p']:.3g} -> {out['verdict']} ({time.time() - t0:.0f}s)", flush=True)
        if not a.watch or any(c[0] == f"check_g{a.gens - 1}" for c in checkpoints(run_dir, a.every) if c[0] in done):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
