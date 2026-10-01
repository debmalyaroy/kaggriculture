"""V1 report for c4 (v63.11 + COPY-gated gtS4 + COPY-gated endK4) on the 80-agent gate.

c4 plays exactly like c2 against COPY rivals (256/256 panel games bank-identical) and exactly like v63.11 against every
other rival (80/80 PARTIAL field games bank-identical), so its gate result is: c2's rows on COPY games + v63.11's rows
elsewhere. Rows missing from the c2 gate are reported, never filled in.

    python python/v6312/v1_report.py [--c2 data/rshell/g80_c2.jsonl]
"""
import argparse, collections as C, json, math

ap = argparse.ArgumentParser()
ap.add_argument("--c2", default="data/rshell/g80_c2.jsonl")
a = ap.parse_args()
base = json.load(open(".local/v6312/a0_v6311.json"))
c2 = {}
for l in open(a.c2):
    r = json.loads(l)
    if "gap" in r:
        c2[(r["opp"], r["seed"], r["seat"])] = r
s = lambda g: 1 if g > 0 else (0.5 if g == 0 else 0)
tot = C.Counter(); miss = 0; up = dn = 0
cls = C.defaultdict(lambda: [0, 0, 0])       # n, losses v63.11, losses c4
opp = C.defaultdict(lambda: [0, 0.0, 0.0])
world = C.defaultdict(lambda: [0, 0, 0])
for r in base:
    k = (r["opp"], r["seed"], r["seat"]); g = (r.get("lay") or {}).get("group")
    if g == 2:
        x = c2.get(k)
        if x is None:
            miss += 1
            continue
        gc = x["gap"]
    else:
        gc = r["gap"]
    gb = r["gap"]
    name = {0: "DIFF", 1: "PARTIAL", 2: "COPY"}.get(g, "NONE")
    cls[name][0] += 1; cls[name][1] += gb < 0; cls[name][2] += gc < 0
    tot["n"] += 1; tot["sb"] += s(gb); tot["sc"] += s(gc); tot["lb"] += gb < 0; tot["lc"] += gc < 0
    up += s(gc) > s(gb); dn += s(gc) < s(gb)
    o = opp[r["opp"]]; o[0] += 1; o[1] += s(gb); o[2] += s(gc)
    w = world[r["world"]]; w[0] += 1; w[1] += gb < 0; w[2] += gc < 0
n = up + dn
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(up, dn) + 1)) / 2 ** n) if n else 1.0
print(f"games {tot['n']} (COPY rows missing from the c2 gate: {miss})")
print(f"score {tot['sb']/tot['n']:.4f} -> {tot['sc']/tot['n']:.4f}   losses {tot['lb']} -> {tot['lc']}   better {up} worse {dn}  p={p:.3g}")
for k, (m, lb, lc) in sorted(cls.items()):
    print(f"  {k:8s} n{m:5d}  losses {lb} -> {lc}")
print("head-to-head (score v63.11 -> c4):")
for o in ["ours_v63_6_rl", "ours_v63_7_rl", "ours_v63_8_rl", "ours_v64"]:
    m, b_, c_ = opp[o]
    if m:
        print(f"  {o:16s} n{m:4d}  {b_/m:.3f} -> {c_/m:.3f}")
print("weak worlds (losses v63.11 -> c4):")
for w in ["YARN_STORE|YARN_STORE", "SMOOTHIE_SHOP|PIZZA_SHOP", "PIZZA_SHOP|PIZZA_SHOP", "SMOOTHIE_SHOP|PET_CAFE", "PET_CAFE|BRUNCH_SPOT"]:
    m, lb, lc = world[w]
    print(f"  {w:26s} n{m:4d}  {lb} -> {lc}")
