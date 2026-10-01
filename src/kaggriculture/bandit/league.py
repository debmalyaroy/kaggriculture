"""Round-robin league on the Rust engine: team bandits (top-50 players' tapes under the trie router,
`data/field/teams/bases/<team>/{S0,S1}`) against each other, v63 and v62.1.

Every pairing plays the same seeds from both seats (`selfplay --a A --b B`, then B vs A). Results are
grouped by the REALIZED world each game played (selfplay prints it; seeds are never labelled by a
PASS simulation). Resumable: finished (pair, seat order) jobs are skipped.

    python -m kaggriculture.bandit.league [--seeds 48] [--seed0 700000] [--workers 16] [--name league1]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from kaggriculture.paths import ROOT

BIN = os.path.join(ROOT, "rustengine", "v62", "target-x", "release", "selfplay.exe")
PROFILES = os.path.join(ROOT, "configs", "bandit", "profiles", "v4.json")
V611 = os.path.join(ROOT, "configs", "bandit", "bases", "v61.1")
TEAMS = os.path.join(ROOT, "data", "field", "teams")
OUT = os.path.join(ROOT, ".local", "league")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def agents():
    """name -> (base dir, profile id). Team S0 = tapes only (profile ignored at cut "chassis"),
    team S1 = the same tapes under v63's shell (profile 71)."""
    a = {"v63": (V611, 71), "v62.1": (V611, 19)}
    for line in open(os.path.join(TEAMS, "teams.unix.tsv"), encoding="utf-8"):
        slug = line.split("\t")[0].strip()
        if slug:
            for s in ("S0", "S1"):
                d = os.path.join(TEAMS, "bases", slug, s)
                if os.path.isdir(d):
                    a[f"{slug}:{s}"] = (d, 71)
    return a


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def run(job):
    a, b, A, B, seeds = job
    args = [BIN, "--a", A[0], "--b", B[0], "--profiles", PROFILES, "--pa", str(A[1]), "--pb", str(B[1]),
            "--seeds", ",".join(map(str, seeds)), "--threads", "1"]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        p = ln.split("\t")
        if len(p) >= 6 and p[0].lstrip("-").isdigit():
            rows.append({"a": a, "b": b, "seed": int(p[0]), "bank_a": float(p[1]), "bank_b": float(p[2]),
                         "us_a": float(p[3]), "us_b": float(p[4]), "world": p[5]})
    return job, rows


def report(rows, names):
    sc = collections.defaultdict(list)
    per_world = collections.defaultdict(lambda: collections.defaultdict(list))
    mat = collections.defaultdict(list)
    for r in rows:
        x = 1.0 if r["bank_a"] > r["bank_b"] else 0.0 if r["bank_a"] < r["bank_b"] else 0.5
        for me, them, v in ((r["a"], r["b"], x), (r["b"], r["a"], 1 - x)):
            sc[me].append(v)
            mat[(me, them)].append(v)
            per_world[me][r["world"]].append(v)
    rank = sorted(names, key=lambda n: -sum(sc[n]) / max(1, len(sc[n])))
    print(f"{'agent':32} {'score':>6} {'games':>6}  vs v63  vs v62.1")
    for n in rank:
        s = sc[n]
        v63 = mat[(n, "v63")]
        v621 = mat[(n, "v62.1")]
        print(f"{n:32} {sum(s) / max(1, len(s)):6.3f} {len(s):6}  {sum(v63) / max(1, len(v63)):6.3f}  {sum(v621) / max(1, len(v621)):6.3f}")
    # does v63's shell help a team's tapes? S1 vs S0, paired on (opponent, seed, seat order)
    key = {}
    for r in rows:
        key[(r["a"], r["b"], r["seed"], "A")] = 1.0 if r["bank_a"] > r["bank_b"] else 0.0 if r["bank_a"] < r["bank_b"] else 0.5
        key[(r["b"], r["a"], r["seed"], "B")] = 1.0 if r["bank_b"] > r["bank_a"] else 0.0 if r["bank_b"] < r["bank_a"] else 0.5
    print("\nshell effect (S1 vs S0, same opponent/seed/seat):")
    for n in sorted({x.split(":")[0] for x in names if ":" in x}):
        s0, s1 = f"{n}:S0", f"{n}:S1"
        b = w = 0
        for (me, opp, seed, side), v in key.items():
            if me == s1 and opp not in (s0,):
                u = key.get((s0, opp, seed, side))
                if u is not None:
                    b += v > u
                    w += v < u
        print(f"  {n:28} +{b}/-{w}  p {mcnemar(b, w):.3g}")
    return {"rank": rank, "score": {n: sum(sc[n]) / max(1, len(sc[n])) for n in names},
            "per_world": {n: {w: sum(v) / len(v) for w, v in d.items()} for n, d in per_world.items()},
            "matrix": {f"{a}|{b}": sum(v) / len(v) for (a, b), v in mat.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="league1")
    ap.add_argument("--only", default=None, help="comma list of agent names to include")
    a = ap.parse_args()
    ag = agents()
    if a.only:
        ag = {k: v for k, v in ag.items() if k in a.only.split(",")}
    names = list(ag)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{a.name}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
    done = {(r["a"], r["b"]) for r in rows}
    jobs = [(x, y, ag[x], ag[y], seeds) for x, y in itertools.permutations(names, 2) if (x, y) not in done]
    print(f"[league] {len(names)} agents, {a.seeds} seeds, both seats: {len(jobs)} of {len(names) * (len(names) - 1)} ordered pairings to play", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(path, "a", encoding="utf-8") as fh:
        for i, (job, rs) in enumerate(ex.map(run, jobs), 1):
            for r in rs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows += rs
            if i % 40 == 0:
                print(f"[league] {i}/{len(jobs)} pairings", flush=True)
    rep = report(rows, names)
    json.dump(rep, open(os.path.join(OUT, f"{a.name}.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
