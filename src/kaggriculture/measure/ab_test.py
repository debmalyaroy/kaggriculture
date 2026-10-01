"""One-off paired A/B between two agents, with the verdict computed for you.

    python src/ab_test.py <A.py> <B.py> [-n 8] [--seeds 60000,150000]

Runs evaluate.py A --vs B for each seed set (both seats, fixed seeds -- the
games pair up), reads the stable RESULT lines through win_metric.parse_eval,
and prints the exact binomial sign test on wins. This is the same instrument
that validated the demand-model fix (22-10/32, p~0.03) and retired the
adaptive sell-timing layer (neutral twice); it exists as a tool so the next
question costs one command -- or one dashboard button -- instead of a script.

Writes models/ab/<A>__vs__<B>.json. Exit code 0 = A sign-tested better,
1 = no verdict (or B better). Never submits anything.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys

import kaggriculture.measure.win_metric as WM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-n", type=int, default=8,
                    help="games per seed set (both seats)")
    ap.add_argument("--seeds", default="60000,150000")
    args = ap.parse_args()

    a = os.path.join(ROOT, args.a) if not os.path.isabs(args.a) else args.a
    b = os.path.join(ROOT, args.b) if not os.path.isabs(args.b) else args.b
    for p in (a, b):
        if not os.path.exists(p):
            sys.exit(f"no such agent: {p}")

    wins = losses = draws = 0
    margin = 0.0
    per_set = []
    for seed0 in [int(s) for s in args.seeds.split(",") if s.strip()]:
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "measure", "evaluate.py"), a,
             "--vs", b, "-n", str(args.n), "--seed0", str(seed0),
             "--no-record"],
            cwd=ROOT, capture_output=True, text=True, timeout=7200)
        rows = WM.parse_eval(r.stdout + r.stderr)
        w = sum(x["wins"] for x in rows)
        d = sum(x["draws"] for x in rows)
        g = sum(x["games"] for x in rows)
        m = sum(x["margin"] for x in rows)
        wins += w
        draws += d
        losses += g - w - d
        margin += m
        per_set.append({"seed0": seed0, "w": w, "d": d, "g": g,
                        "margin": round(m, 1)})
        print(f"seed0={seed0}: {w}/{g} wins ({d} draws), "
              f"margin {m:+,.0f}", flush=True)

    games = wins + losses + draws
    disc = wins + losses
    if disc == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        tail = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2 ** disc)
        p = min(1.0, 2 * tail)
    score = (wins + 0.5 * draws) / games if games else float("nan")
    a_better = wins > losses and p < 0.05
    verdict = {
        "a": os.path.relpath(a, ROOT), "b": os.path.relpath(b, ROOT),
        "games": games, "a_wins": wins, "b_wins": losses, "draws": draws,
        "a_score": round(score, 4), "mean_margin": round(margin / max(1, games), 1),
        "p_value": round(p, 4), "per_set": per_set,
        "verdict": ("A sign-tested better" if a_better else
                    "B sign-tested better" if losses > wins and p < 0.05 else
                    "no sign-tested difference"),
        "when": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = os.path.join(ROOT, "models", "ab")
    os.makedirs(out_dir, exist_ok=True)
    stem = (os.path.basename(a).replace(".py", "") + "__vs__"
            + os.path.basename(b).replace(".py", "") + ".json")
    with open(os.path.join(out_dir, stem), "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"verdict -> models/ab/{stem}")
    return 0 if a_better else 1


if __name__ == "__main__":
    raise SystemExit(main())
