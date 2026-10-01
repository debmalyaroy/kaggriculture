"""G1 Replay Subgame Miner (operator's game-theory notes, Module 1).

    python python/v6312/g1_miner.py TRACE.tsv [TRACE2.tsv ...] --out data/v6312/g1_payoffs.json

Input: `trace` S rows (engine truth after every step: money, shed of both seats, market inventory and quotes).
A COLLISION on item c at step t: both players hold >= MIN units of c in the shed (c in STRAWBERRY/MELON/MILK/WOOL).
Regimes: mid = steps 144-647 (days 6-26), end = 648-711 (days 27-29). Onsets are taken at most once per H steps per
(game, item) so windows do not overlap.
Each side's move over the next H steps is classified by the share of its onset stock it released (shed decrease;
deposits during the window are netted, so released = start + deposited - end >= start - end):
    HOLD    < 25%       TRANCHE 25-75%       DUMP  > 75%
Payoff to the row player = change of (money + shed_c x quote_c) over the window, minus the same for the column player
(zero-sum relative bank: the ladder pays the winner). The quote at the window end values the stock still held, so a
hold that is later crashed by the rival pays its loss here.
Output: per item x regime x price bin: 3x3 mean payoff matrix A (row = us), counts, and the stderr of each cell.
"""
import argparse
import collections
import json
import math
import os

ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
GT = ["STRAWBERRY", "MELON", "MILK", "WOOL"]
S = ["HOLD", "TRANCHE", "DUMP"]


def price_bin(px, base):
    r = px / max(1.0, base)
    return "low" if r < 0.6 else ("mid" if r < 1.2 else "high")


BASE = {"STRAWBERRY": 40.0, "MELON": 131.0, "MILK": 60.0, "WOOL": 150.0}


def cls(start, end):
    rel = max(0, start - end) / max(1, start)
    return 0 if rel < 0.25 else (1 if rel <= 0.75 else 2)


def games(paths):
    """Yields (game id, {step: (money_me, money_rv, shed_me[9], shed_rv[9], px[9])}) per traced seat."""
    cur, rows = None, {}
    for p in paths:
        for line in open(p):
            if not line.startswith("S\t"):
                continue
            x = line.rstrip("\n").split("\t")
            gid = (x[1], x[2])
            if gid != cur:
                if rows:
                    yield cur, rows
                cur, rows = gid, {}
            v = list(map(float, x[4:]))
            rows[int(x[3])] = (v[0], v[1], v[2:11], v[11:20], v[29:38])
    if rows:
        yield cur, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", type=int, default=15)
    ap.add_argument("--h", type=int, default=24)
    a = ap.parse_args()
    cells = collections.defaultdict(list)  # (item, regime, bin, s_me, s_rv) -> payoffs
    n_games = 0
    for gid, R in games(a.traces):
        n_games += 1
        for c in GT:
            i = ITEMS.index(c)
            t = 144
            while t <= 711:
                r = R.get(t)
                if r and r[2][i] >= a.min and r[3][i] >= a.min:
                    t2 = min(t + a.h, 718)
                    e = R.get(t2)
                    if e is None:
                        break
                    reg = "mid" if t < 648 else "end"
                    b = price_bin(r[4][i], BASE[c])
                    sm, sr = cls(r[2][i], e[2][i]), cls(r[3][i], e[3][i])
                    val = lambda m, sh, px: m + sh * px
                    pay = (val(e[0], e[2][i], e[4][i]) - val(r[0], r[2][i], r[4][i])) - (val(e[1], e[3][i], e[4][i]) - val(r[1], r[3][i], r[4][i]))
                    cells[(c, reg, b, sm, sr)].append(pay)
                    t = t2
                else:
                    t += 1
    out = {"games": n_games, "min": a.min, "h": a.h, "strategies": S, "tables": {}}
    for c in GT:
        for reg in ("mid", "end"):
            for b in ("low", "mid", "high"):
                A, N, SE = [[0.0] * 3 for _ in range(3)], [[0] * 3 for _ in range(3)], [[0.0] * 3 for _ in range(3)]
                for x in range(3):
                    for y in range(3):
                        v = cells.get((c, reg, b, x, y), [])
                        N[x][y] = len(v)
                        if v:
                            m = sum(v) / len(v)
                            A[x][y] = m
                            SE[x][y] = math.sqrt(sum((q - m) ** 2 for q in v) / max(1, len(v) - 1) / len(v)) if len(v) > 1 else 0.0
                if sum(map(sum, N)):
                    out["tables"][f"{c}|{reg}|{b}"] = {"A": A, "n": N, "se": SE}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("games", n_games)
    for k, t in out["tables"].items():
        tot = sum(map(sum, t["n"]))
        print(f"{k:24s} n={tot:5d}  " + "  ".join(f"{S[x][0]}{S[y][0]}:{t['A'][x][y]:+6.0f}({t['n'][x][y]})" for x in range(3) for y in range(3) if t["n"][x][y]))


if __name__ == "__main__":
    main()
