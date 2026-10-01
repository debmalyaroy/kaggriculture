"""The objective gate: a candidate vs real ladder players by rating band (PLAN s18b / s20 G1).

    python python/band_gate.py --cand weights/ppo/<run>/best.bin [--name X] [--ref-profile 19] [--threads 16]
    python python/band_gate.py --cand-profile 13 --name v62          (a fixed-profile candidate, e.g. live v62)

Opponents: data/tapes/band (python/band_tapes.py): recent real players' recorded streams, each on its own
seed (one world per match, as on the ladder), played through the guards (tapeplay --guarded). The
candidate and the reference (default v62.1 = fixed profile 19) meet the SAME tapes, so every tape is a
paired cell. Pass rule (operator objective): 0 losses to opponents rated below 2500, and >= 80% wins at
2500+ (90% is the target); plus paired McNemar vs the reference not significantly worse. Draws are 0.5
for the rate and are not losses.
Writes data/gates/band__<name>__<UTC>.json and prints one line per band.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
TAPES = os.path.join(RL, "data", "tapes", "band")
PROFILES = os.path.join(RL, "configs", "profiles", "rl3.json")
SHIELD = os.path.join(RL, "configs", "shield", "v1.json")
OUT = os.path.join(RL, "data", "gates")
LOW = ("lt2100", "2100-2300", "2300-2500")
HIGH = ("2500-2700", "2700plus")


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def play(agent_args, files, threads, profiles=None):
    """tapeplay over `files` in `threads` shards; returns {tape id: (us, them)}."""
    shards = [files[i::threads] for i in range(threads) if files[i::threads]]

    def one(shard):
        d = tempfile.mkdtemp(prefix="bandgate-")
        try:
            for f in shard:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--profiles", profiles or PROFILES, "--guarded", *agent_args],
                               cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"tapeplay rc {r.returncode}: {r.stderr[-500:]}")
            return {ln.split("\t")[0]: (float(ln.split("\t")[4]), float(ln.split("\t")[5])) for ln in r.stdout.splitlines() if ln.count("\t") >= 5}
        finally:
            shutil.rmtree(d, ignore_errors=True)

    out = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for res in ex.map(one, shards):
            out.update(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", help="policy weights (.bin)")
    ap.add_argument("--cand-profile", type=int, help="or a fixed profile id")
    ap.add_argument("--cand-args", default="", help="extra tapeplay arguments for the candidate, e.g. \"--route-table F\"")
    ap.add_argument("--cand-shell", default=None, help="with --cand-profile: the learned sales shell (tapeplay --shell)")
    ap.add_argument("--cand-group", default=None, help="with --cand-profile: P_DIFF,P_PARTIAL,P_COPY, one profile per day-0 opponent group from step 25 (tapeplay --group)")
    ap.add_argument("--profiles", default=None, help="profile table (default configs/profiles/rl3.json); ref and candidate ids index it")
    ap.add_argument("--name", default=None)
    ap.add_argument("--ref-group", default=None, help="reference plays --ref-profile on day 0, then one profile per day-0 opponent group "
                    "(tapeplay --group). v63.1_rl = --ref-profiles configs/profiles/v64rl.json --ref-profile 35 --ref-group 35,35,36")
    ap.add_argument("--ref-profiles", default=None, help="profile table for the reference (default: --profiles)")
    ap.add_argument("--ref-profile", type=int, default=35, help="reference: fixed profile (rl3 ids: 35 = v63, 19 = v62.1, 13 = v62, 0 = v61.1)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("KRL_THREADS") or max(2, (os.cpu_count() or 4) - 2)))
    a = ap.parse_args()
    global PROFILES
    if a.profiles:
        PROFILES = os.path.abspath(a.profiles)
    if not a.cand and a.cand_profile is None:
        ap.error("--cand or --cand-profile")
    labels = {}
    for ln in open(os.path.join(TAPES, "index.tsv"), encoding="utf-8").read().splitlines()[1:]:
        tid, band, team, rating, date, seed = ln.split("\t")
        labels[tid] = (band, team, float(rating), seed)
    files = sorted(glob.glob(os.path.join(TAPES, "*", "*.json")))
    # (fixed 2026-09-27: the conditional used to swallow --cand-args whenever --cand was given, so every
    # "PPO + shell" band result before this date played PPO WITHOUT the shell)
    if a.cand:
        cargs = ["--policy", a.cand] + (["--shield", SHIELD] if os.path.exists(SHIELD) else [])
    else:
        cargs = ["--pa", str(a.cand_profile)] + (["--group", a.cand_group] if a.cand_group else []) + (["--shell", os.path.abspath(a.cand_shell)] if a.cand_shell else [])
    cargs += a.cand_args.split()
    name = a.name or (os.path.basename(os.path.dirname(a.cand)) if a.cand else f"p{a.cand_profile}" + (f"-g{a.cand_group}" if a.cand_group else ""))
    cand = play(cargs, files, a.threads)
    ref = play(["--pa", str(a.ref_profile)] + (["--group", a.ref_group] if a.ref_group else []), files, a.threads,
               os.path.abspath(a.ref_profiles) if a.ref_profiles else None)
    sc = lambda us, them: 1.0 if us > them else 0.0 if us < them else 0.5  # noqa: E731
    per, better, worse = {}, 0, 0
    for tid, (band, team, rating, seed) in labels.items():
        if tid not in cand or tid not in ref:
            continue
        c, r = sc(*cand[tid]), sc(*ref[tid])
        better += c > r
        worse += c < r
        p = per.setdefault(band, {"games": 0, "cand": 0.0, "ref": 0.0, "cand_losses": 0, "ref_losses": 0, "lost_to": []})
        p["games"] += 1
        p["cand"] += c
        p["ref"] += r
        p["cand_losses"] += c == 0.0
        p["ref_losses"] += r == 0.0
        if c == 0.0:
            p["lost_to"].append({"tape": tid, "team": team, "rating": rating, "seed": seed, "gap": cand[tid][0] - cand[tid][1]})
    for p in per.values():
        p["cand"] /= max(1, p["games"])
        p["ref"] /= max(1, p["games"])
    low_losses = sum(per.get(b, {}).get("cand_losses", 0) for b in LOW)
    ref_low = sum(per.get(b, {}).get("ref_losses", 0) for b in LOW)
    hi_g = sum(per.get(b, {}).get("games", 0) for b in HIGH)
    hi_rate = sum(per.get(b, {}).get("cand", 0) * per.get(b, {}).get("games", 0) for b in HIGH) / max(1, hi_g)
    ref_hi = sum(per.get(b, {}).get("ref", 0) * per.get(b, {}).get("games", 0) for b in HIGH) / max(1, hi_g)
    p_val = mcnemar(better, worse)
    checks = {"no_loss_below_2500": low_losses == 0, "win_2500plus_ge_80": hi_rate >= 0.80,
              "not_worse_than_ref": not (worse > better and p_val < 0.05)}
    rep = {"name": name, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cand": a.cand or f"profile {a.cand_profile}",
           "ref": f"profile {a.ref_profile}" + (f" group {a.ref_group} ({os.path.basename(a.ref_profiles or PROFILES)})" if a.ref_group else ""), "tapes": len(files), "paired": {"better": better, "worse": worse, "p": p_val},
           "losses_below_2500": low_losses, "win_rate_2500plus": hi_rate, "target_2500plus": 0.90, "checks": checks,
           "pass": all(checks.values()),
           # PLAN s18b: while nothing meets the objective, a candidate must be no worse than the reference
           # on it -- no more sub-2500 losses and at least its 2500+ rate -- and not worse paired
           "ref_losses_below_2500": ref_low, "ref_win_rate_2500plus": ref_hi,
           "no_worse_than_ref": low_losses <= ref_low and hi_rate >= ref_hi and checks["not_worse_than_ref"],
           "per_band": per}
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"band__{name}__{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%MZ}.json")
    json.dump(rep, open(fn, "w"), indent=1)
    for b in LOW + HIGH:
        if b in per:
            v = per[b]
            print(f"[band] {b:10s} games {v['games']:4d}  {name} {v['cand']:.3f} (losses {v['cand_losses']})  ref {v['ref']:.3f} (losses {v['ref_losses']})")
    print(f"[band] {name}: losses below 2500 = {low_losses}; win rate 2500+ = {hi_rate:.3f} (floor 0.80, target 0.90); "
          f"paired vs ref +{better}/-{worse} p {p_val:.3g}; ref: {ref_low} losses, {ref_hi:.3f}; "
          f"objective PASS = {rep['pass']}; no worse than ref = {rep['no_worse_than_ref']} -> {fn}")


if __name__ == "__main__":
    main()
