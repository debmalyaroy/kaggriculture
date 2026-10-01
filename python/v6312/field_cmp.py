"""Paired field comparison vs the v63.11 80-agent baseline: python python/v6312/field_cmp.py OUT.jsonl"""
import json, math, sys, collections as C
base = {(r["opp"], r["seed"], r["seat"]): r for r in json.load(open(".local/v6312/a0_v6311.json"))}
up = dn = 0; lw = C.Counter()
for l in open(sys.argv[1]):
    r = json.loads(l); b = base.get((r["opp"], r["seed"], r["seat"]))
    if not b or "gap" not in r: continue
    s = lambda g: 1 if g > 0 else (0.5 if g == 0 else 0)
    x, y = s(b["gap"]), s(r["gap"])
    lw["loss" if b["gap"] < 0 else "win"] += 1
    if y > x: up += 1
    elif y < x: dn += 1
n = up + dn
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(up, dn) + 1)) / 2 ** n) if n else 1.0
print(f"games {sum(lw.values())} (base losses {lw['loss']}, wins {lw['win']})  fixed {up}  broken {dn}  p={p:.3g}")
