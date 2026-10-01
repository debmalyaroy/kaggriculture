"""Bradley-Terry tournament over local paired duels -- the FINAL metric.

Host confirmation (discussion 739410, staff, 2026-09-05): the final
leaderboard is a single Bradley-Terry fit over post-deadline episodes among
deadline-active agents, ties counted half. A harness that crowns by raw win
rate optimizes a different number than the one that decides placement; this
ranks the candidate pool the way Kaggle will.

Duels run on `kagg serve` (Rust engine, exact-bank verified), both seats per
seed, every pair of agents on the same seed set -- the paired grid that makes
cells comparable. The BT strengths come from the standard MM iteration;
ties add half a win to each side, per the host's answer.

    python src/trackp/harness/bt_rank.py agents/v44.1_bandit.py \
        agents/v45.0_bandit.py .local/candidates/v451_flush.py --seeds 8
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import kaggriculture.engine.serve_match as SM  # noqa: E402

OUT = os.path.join(ROOT, "models", "trackp", "bt_rank.json")


def duel(srv, pa, pb, seed):
    """One seed, both seats; returns (wins_a, wins_b) with ties = 0.5 each."""
    wa = wb = 0.0
    for flip in (False, True):
        A = SM.load_agent(pa)
        B = SM.load_agent(pb)
        if flip:
            b1, b0 = SM.run_match(B, A, seed, srv)
        else:
            b0, b1 = SM.run_match(A, B, seed, srv)
        if b0 > b1:
            wa += 1
        elif b1 > b0:
            wb += 1
        else:                              # tie = half a win each (host rule)
            wa += 0.5
            wb += 0.5
    return wa, wb


def bt_fit(names, wins, iters=200):
    """Standard Bradley-Terry MM iteration on a win matrix."""
    n = len(names)
    p = [1.0] * n
    for _ in range(iters):
        q = []
        for i in range(n):
            num = sum(wins[i][j] for j in range(n) if j != i)
            den = 0.0
            for j in range(n):
                if j == i:
                    continue
                g = wins[i][j] + wins[j][i]
                if g:
                    den += g / (p[i] + p[j])
            q.append(num / den if den else p[i])
        s = sum(q)
        p = [v * n / s for v in q]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agents", nargs="+", help="agent .py files to rank")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=70000)
    a = ap.parse_args()
    paths = [os.path.abspath(p) for p in a.agents]
    names = [os.path.basename(p) for p in paths]
    n = len(paths)
    wins = [[0.0] * n for _ in range(n)]

    srv = SM.Serve()
    try:
        for (i, pa), (j, pb) in itertools.combinations(enumerate(paths), 2):
            for s in range(a.seeds):
                wa, wb = duel(srv, pa, pb, a.seed0 + s)
                wins[i][j] += wa
                wins[j][i] += wb
            print(f"  {names[i]} vs {names[j]}: "
                  f"{wins[i][j]:.1f}-{wins[j][i]:.1f}", flush=True)
    finally:
        srv.close()

    p = bt_fit(names, wins)
    order = sorted(range(n), key=lambda k: -p[k])
    print(f"\nBRADLEY-TERRY ranking ({a.seeds} seeds x both seats, "
          f"ties=half):")
    for r, k in enumerate(order, 1):
        games = sum(wins[k]) + sum(wins[j][k] for j in range(n))
        print(f"  {r}. {names[k]:<28} strength {p[k]:.3f}  "
              f"score {sum(wins[k]):.1f}/{games:.0f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"names": names, "wins": wins, "strength": p,
               "seeds": a.seeds},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
