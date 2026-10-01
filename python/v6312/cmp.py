"""Paired comparison of two panel TSVs (same opp/seat/seed): outcome flips + exact sign test.

    python python/v6312/cmp.py BASE.tsv CAND.tsv"""
import sys, math, collections as C

def load(p):
    return {(r[0], r[1], r[2]): (r[3], float(r[6])) for r in (l.rstrip("\n").split("\t") for l in open(p))}

def sgn(g):
    return 1 if g > 0 else (-1 if g < 0 else 0)

def binom_p(k, n):
    if n == 0: return 1.0
    t = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * t)

b, c = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(b) & set(c))
up = dn = 0; per = C.defaultdict(lambda: [0, 0, 0, 0]); sb = sc = 0
for k in keys:
    x, y = sgn(b[k][1]), sgn(c[k][1]); sb += (x + 1) / 2; sc += (y + 1) / 2
    d = per[k[0]]; d[0] += 1; d[1] += (x + 1) / 2; d[2] += (y + 1) / 2
    if y > x: up += 1; d[3] += 1
    elif y < x: dn += 1; d[3] -= 1
print(f"games {len(keys)}  score {sb/len(keys):.4f} -> {sc/len(keys):.4f}  better {up} worse {dn}  p={binom_p(min(up,dn), up+dn):.3g}")
for o, d in sorted(per.items()):
    print(f"  {o:10s} n{d[0]:4d}  {d[1]/d[0]:.3f} -> {d[2]/d[0]:.3f}  net flips {d[3]:+d}")
