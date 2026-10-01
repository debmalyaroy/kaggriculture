"""Gate and checkpoint tournament for macro-policy weights (tasks P6.4, P7.6, P9.1; queue Q14/Q17).

    python python/learn/gate.py gate --cand weights/bc/LATEST            # one candidate vs the reference (v63; KRL_GATE_REF)
    python python/learn/gate.py tournament                                # BC + PPO best/last snapshots

Every candidate (greedy policy) and the reference (v63 = rl3 profile 35; KRL_GATE_REF) play the SAME validation
seeds (500000.., never used in training) against the same Rust panel, both seats (seat = seed & 1):
  v61.1, rsa8 (2), aggr (12), aggr_deep (13), ad_rsa12_l24 (19), rsa12 (22), aggr_deep_afr (30), p19_afr (31)
-- our own lineage plus its escalated and front-running variants.
Verdict per opponent and overall: paired McNemar on discordant seeds (candidate won where the
reference lost, and vice versa). PASS = overall significantly better (p < .05) and no opponent
significantly worse. Results: data/gates/<candidate>__<UTC>.json (+ tournament summary).
Limit, stated plainly: this panel is Rust-only (our lineage). Rating-band results against real
ladder opponents and a Bradley-Terry fit anchored to public ladder ratings need the Python
opponent host (P4.2), which is not built.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

# KRL_RL / KRL_ROLL let the Kaggle gate notebook run this file against its own layout + Linux binary
RL = os.environ.get("KRL_RL") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROLL = os.environ.get("KRL_ROLL") or os.path.join(os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release"), "ppo-rollout" + (".exe" if os.name == "nt" else ""))
PANEL = ["fixed:0", "fixed:2", "fixed:12", "fixed:13", "fixed:19", "fixed:22", "fixed:30", "fixed:31", "fixed:35"]  # 35 = v63
SEED0 = 500_000
# the reference every candidate must beat, paired on the same seeds: rl3 profile 35 = v63, the live agent
# (was v61.1 = profile 0 until 2026-09-26, which every candidate beat, so "PASS" meant nothing)
REF = int(os.environ.get("KRL_GATE_REF", "35"))
REF_NAME = {35: "v63", 19: "v62.1", 13: "v62", 0: "v61.1"}.get(REF, f"profile {REF}")
OUT = os.path.join(RL, "data", "gates")


def play(args, prefix, n, opp, threads):
    cmd = [ROLL, *args, "--out", prefix, "--games", str(n), "--seed0", str(SEED0), "--threads", str(threads),
           "--greedy", "--opp", opp]
    shield = os.path.join(RL, "configs", "shield", "v1.json")
    if os.path.exists(shield) and "--weights" in args:  # the candidate plays under the shield, as deployed
        cmd += ["--shield", shield]
    r = subprocess.run(cmd, cwd=RL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    rows = [l.split("\t") for l in open(prefix + ".tsv", encoding="utf-8").read().splitlines()]
    for f in (prefix + ".tsv", prefix + ".traj"):
        os.remove(f)
    return {int(r[0]): float(r[6]) for r in rows}


def mcnemar(b, c):
    """Two-sided exact binomial p on discordant pairs (b = cand better, c = cand worse)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def evaluate(name, weights, n, threads, ref_cache):
    os.makedirs(OUT, exist_ok=True)
    per, tb, tc, cs, rs = {}, 0, 0, [], []
    for opp in PANEL:
        if opp not in ref_cache:
            ref_cache[opp] = play(["--learner-fixed", str(REF)], os.path.join(OUT, "_ref"), n, opp, threads)
        cand = play(["--weights", weights], os.path.join(OUT, "_cand"), n, opp, threads)
        ref = ref_cache[opp]
        seeds = sorted(set(cand) & set(ref))
        b = sum(1 for s in seeds if cand[s] > ref[s])
        c = sum(1 for s in seeds if cand[s] < ref[s])
        per[opp] = {"cand_score": float(np.mean([cand[s] for s in seeds])), "ref_score": float(np.mean([ref[s] for s in seeds])),
                    "better": b, "worse": c, "p": mcnemar(b, c), "n": len(seeds)}
        tb, tc = tb + b, tc + c
        cs += [cand[s] for s in seeds]
        rs += [ref[s] for s in seeds]
    worse_family = [o for o, r in per.items() if r["worse"] > r["better"] and r["p"] < .05]
    p = mcnemar(tb, tc)
    res = {"candidate": name, "weights": weights, "ref": REF_NAME, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cand_score": float(np.mean(cs)), "ref_score": float(np.mean(rs)), "better": tb, "worse": tc, "p": p,
           "worse_families": worse_family, "per_opponent": per,
           "verdict": "PASS" if (tb > tc and p < .05 and not worse_family) else "FAIL"}
    json.dump(res, open(os.path.join(OUT, f"{name}__{res['t'].replace(':', '')}.json"), "w"), indent=1)
    print(f"[gate] {name}: {res['verdict']} score {res['cand_score']:.3f} vs {REF_NAME} {res['ref_score']:.3f}; "
          f"paired better {tb} / worse {tc}, p {p:.3g}; worse families {worse_family or 'none'}", flush=True)
    for o, r in per.items():
        print(f"   {o:<9} cand {r['cand_score']:.3f} ref {r['ref_score']:.3f}  +{r['better']}/-{r['worse']}  p {r['p']:.3g}")
    return res


def resolve(w):
    if w.endswith("LATEST"):
        w = os.path.join(os.path.dirname(w), open(w).read().strip())
    if os.path.isdir(w):
        return os.path.basename(w.rstrip("/\\")), os.path.join(w, "weights.bin")
    return os.path.splitext(os.path.basename(w))[0], w


def discover():
    """Tournament candidates: the newest BC teacher, the newest PPO run's best and last 3 snapshots."""
    cands = []
    try:
        cands.append(resolve(os.path.join(RL, "weights", "bc", "LATEST")))
    except OSError:
        pass
    # candidates must match the current profile table (rl3: 36 actions); a 32-action BC or run is skipped
    from model import profile_names
    n_act = len(profile_names())
    cands = [c for c in cands if os.path.getsize(c[1]) // 4 - 2 * 93 - (64 * 93 + 64) - 3 * 64 * 64 * 2 - 3 * 64 * 2 - 64 * n_act - n_act - 64 - 1 - 9 * 64 - 9 - 5 * 64 - 5 == 0]
    runs = sorted(d for d in glob.glob(os.path.join(RL, "weights", "ppo", "ppo-*")) if f"-p{n_act}-" in os.path.basename(d))
    if runs:
        r = runs[-1]
        if os.path.exists(os.path.join(r, "best.bin")):
            cands.append((os.path.basename(r) + "__best", os.path.join(r, "best.bin")))
        for s in sorted(glob.glob(os.path.join(r, "snapshots", "*.bin")))[-3:]:
            cands.append((os.path.basename(r) + "__" + os.path.basename(s)[:-4], s))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "tournament"])
    ap.add_argument("--cand", default=os.path.join(RL, "weights", "bc", "LATEST"))
    ap.add_argument("--n", type=int, default=100, help="seeds per opponent")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cands", nargs="*", help="tournament: explicit NAME=WEIGHTS.bin list (the Kaggle notebook passes these)")
    a = ap.parse_args()
    ref_cache = {}
    if a.mode == "gate":
        name, w = resolve(os.path.join(RL, a.cand) if not os.path.isabs(a.cand) else a.cand)
        evaluate(name, w, a.n, a.threads, ref_cache)
        return
    cands = [tuple(c.split("=", 1)) for c in a.cands] if a.cands else discover()
    if not cands:
        sys.exit("no candidates")
    results = [evaluate(n, w, a.n, a.threads, ref_cache) for n, w in cands]
    results.sort(key=lambda r: (r["verdict"] == "PASS", r["cand_score"]), reverse=True)
    summ = {"t": results[0]["t"], "ranking": [{k: r[k] for k in ("candidate", "weights", "verdict", "cand_score", "ref_score", "better", "worse", "p")} for r in results]}
    json.dump(summ, open(os.path.join(OUT, f"tournament__{summ['t'].replace(':', '')}.json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(OUT, "tournament_latest.json"), "w"), indent=1)
    print("[tournament] ranking: " + "; ".join(f"{r['candidate']} {r['verdict']} {r['cand_score']:.3f}" for r in summ["ranking"]), flush=True)


if __name__ == "__main__":
    main()
