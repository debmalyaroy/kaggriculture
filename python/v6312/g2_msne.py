"""G2 MSNE solver (operator's game-theory notes, Module 2) over the G1 payoff tables.

    python python/v6312/g2_msne.py data/v6312/g1_payoffs.json --out data/v6312/g2_msne.json [--k 30]

Per (item, regime, price bin): the observed matrix is made a symmetric zero-sum game (A - A^T)/2 with the cells shrunk
toward 0 by n/(n+k) (small cells are noise), then the row player's maximin mix over {HOLD, TRANCHE, DUMP} is solved as
an LP. For the endgame the notes' "sell gate" distribution follows: P(gate = hold) / P(gate = tranche) / P(gate = dump).
"""
import argparse
import json

import numpy as np
from scipy.optimize import linprog

S = ["HOLD", "TRANCHE", "DUMP"]


def maximin(A):
    """max_x min_y x^T A y, x in the simplex. LP: max v s.t. A^T x >= v, sum x = 1."""
    n = A.shape[0]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_ub = np.hstack([-A.T, np.ones((A.shape[1], 1))])
    b_ub = np.zeros(A.shape[1])
    A_eq = np.array([[1.0] * n + [0.0]]); b_eq = [1.0]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n + [(None, None)], method="highs")
    return r.x[:n], r.x[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("payoffs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=30.0)
    a = ap.parse_args()
    P = json.load(open(a.payoffs))
    out = {"source": a.payoffs, "k": a.k, "strategies": S, "mix": {}}
    for key, t in P["tables"].items():
        A = np.array(t["A"], float); N = np.array(t["n"], float)
        # pooled antisymmetric estimate, shrunk toward 0: G_xy = (A_xy n_xy - A_yx n_yx) / (n_xy + n_yx + k)
        G = (A * N - (A * N).T) / (N + N.T + a.k)
        x, v = maximin(G)
        pure = [float(min(G[i])) for i in range(3)]
        out["mix"][key] = {"p": [round(float(q), 4) for q in x], "value": round(float(v), 2), "G": np.round(G, 1).tolist(),
                           "n": int(N.sum()), "pure_worst": [round(q, 1) for q in pure]}
        print(f"{key:24s} n={int(N.sum()):5d}  mix H/T/D = " + " ".join(f"{q:.2f}" for q in x) + f"   worst-case pure H/T/D = " + " ".join(f"{q:+6.0f}" for q in pure))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
