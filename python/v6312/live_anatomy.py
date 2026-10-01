"""Anatomy of our live ladder losses from `trace` output (engine truth, both seats traced).

    python python/v6312/live_anatomy.py .local/live6312/loss_trace.tsv

Per loss: world, opponent farm vs ours (same composition each day = a copy of our plan), the margin at every day end,
the last day we led, and per product the units each side sold (shed decreases, ~) and the money each side took for them
(units x the step's quote).
"""
import collections as C
import sys

ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
S = C.defaultdict(dict)  # (id, seat) -> step -> row
F = C.defaultdict(dict)  # (id, seat) -> day -> composition
W = {}
for l in open(sys.argv[1]):
    x = l.rstrip("\n").split("\t")
    if x[0] == "S":
        v = list(map(float, x[4:]))
        S[(x[1], int(x[2]))][int(x[3])] = (v[0], v[1], v[2:11], v[11:20], v[20:29], v[29:38])
    elif x[0] == "F":
        F[(x[1], int(x[2]))][int(x[3])] = tuple(x[4:])
    elif x[0] == "W":
        W[x[1]] = x
for key in sorted(S):
    tid, seat = key
    if not tid.endswith("_" + str(seat)):
        continue  # the tape's own seat = our seat
    R = S[key]
    steps = sorted(R)
    days = {t // 24: R[t] for t in steps if t % 24 == 23 or t == steps[-1]}
    marg = [(d, R_[0] - R_[1]) for d, R_ in sorted(days.items())]
    last_lead = max([d for d, m in marg if m > 0], default=-1)
    fin = marg[-1][1]
    us_f, rv_f = F.get((tid, seat), {}), F.get((tid, 1 - seat), {})
    same = sum(1 for d in us_f if d in rv_f and us_f[d][1:] == rv_f[d][1:])
    # per product: units sold (shed decrease minus nothing; deposits make it a lower bound) and money at the quote
    sold = C.defaultdict(float); rev = C.defaultdict(float)
    prev = None
    for t in steps:
        r = R[t]
        if prev:
            for i, it in enumerate(ITEMS):
                for who, a, b in (("us", prev[2][i], r[2][i]), ("rv", prev[3][i], r[3][i])):
                    if a > b:
                        sold[(who, it)] += a - b
                        rev[(who, it)] += (a - b) * prev[5][i]
        prev = r
    gaps = sorted(((rev[("us", it)] - rev[("rv", it)]), it) for it in ITEMS)
    w = W.get(tid, [])
    print(f"== {tid} world {w[5] if len(w) > 5 else '?'}  final {fin:+.0f}  last led day {last_lead}  farm identical on {same}/{len(us_f)} days")
    print("   margin by day: " + " ".join(f"d{d}:{m:+.0f}" for d, m in marg if d >= 18 or d % 6 == 0))
    print("   product money us-rv (worst 3): " + ", ".join(f"{it} {g:+.0f} (us {sold[('us', it)]:.0f}u / rv {sold[('rv', it)]:.0f}u)" for g, it in gaps[:3]))
