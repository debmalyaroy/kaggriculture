"""Summarise G1b closed-loop labels: per (item, phase, price bin) which forced strategy beats TAPE, and by how much.

    python python/v6312/g1b_sum.py F.tsv [F2.tsv ...]"""
import sys, collections as C, statistics as S
BASE = {"STRAWBERRY": 40.0, "MELON": 131.0, "MILK": 60.0, "WOOL": 150.0}
N = ["HOLD", "TRANCHE", "DUMP"]
cell = C.defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if len(x) < 14: continue
        item, step, px = x[5], int(x[4]), float(x[6])
        g = list(map(float, x[10:14]))
        r = px / BASE[item]; b = "low" if r < 0.6 else ("mid" if r < 1.2 else "high")
        ph = "mid" if step < 648 else "end"
        cell[(item, ph, b)].append(g)
print(f"{'cell':24s} {'n':>4s}  mean(gap - tape): HOLD TRANCHE DUMP   win-flips vs tape (+/-) per strategy")
for k in sorted(cell):
    rows = cell[k]
    d = [[r[i] - r[3] for r in rows] for i in range(3)]
    fl = []
    for i in range(3):
        up = sum(1 for r in rows if r[i] > 0 >= r[3]); dn = sum(1 for r in rows if r[i] <= 0 < r[3])
        fl.append(f"{up}/{dn}")
    print(f"{'|'.join(k):24s} {len(rows):4d}  " + " ".join(f"{S.mean(x):+7.0f}" for x in d) + "   " + " ".join(f"{N[i][0]}:{fl[i]}" for i in range(3)))
