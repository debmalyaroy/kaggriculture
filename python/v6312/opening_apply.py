"""Apply the opening screen (fix A): replace a chassis' route 0 by the best alternative opening IF it beats the current
opening paired on the screen tapes (better > worse, exact sign test p < 0.05).

    python python/v6312/opening_apply.py CLASS [--chassis data/chassis/final/CLASS] [--p 0.05]

Reads data/chassis/open/CLASS/cells.tsv (chassis-grid --openings: cell "router" = current opening, "open<id>" = pool
route <id>'s opening) and extracts route <id> from weights/chassis_rep/CLASS/routes.json with mmap (the pool file is
~0.8 GB; nothing big is loaded). The old route 0 is kept as routes.json.bak_open. Prints the decision.
"""
import argparse, json, math, mmap, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("cls")
ap.add_argument("--chassis", default=None)
ap.add_argument("--p", type=float, default=0.05)
a = ap.parse_args()
ch = a.chassis or f"data/chassis/final/{a.cls}"
rows = [l.rstrip("\n").split("\t") for l in open(f"data/chassis/open/{a.cls}/cells.tsv")][1:]
base = next(r for r in rows if r[0] == "router")


def sign_p(b, w):
    m, k = b + w, min(b, w)
    return 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)


cands = []
for r in rows:
    if r[0].startswith("open"):
        b, w = int(r[6]), int(r[7])
        cands.append((float(r[2]), r[0][4:], b, w, sign_p(b, w)))
cands.sort(reverse=True)
print(f"current opening: score {float(base[2]):.4f} (n {base[1]})")
for s, i, b, w, p in cands[:5]:
    print(f"  opening {i}: score {s:.4f}  better {b} worse {w}  p {p:.2e}")
best = next((c for c in cands if c[2] > c[3] and c[4] < a.p), None)
if best is None:
    print("DECISION: keep the current opening (no alternative is significantly better)")
    raise SystemExit
rid = best[1]
with open(f"weights/chassis_rep/{a.cls}/routes.json", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    key = f'"{rid}":['.encode()
    i = mm.find(key)
    assert i >= 0, key
    j = i + len(key) - 1
    depth, k, instr = 0, j, False
    while True:
        c = mm[k]
        if instr:
            if c == 0x5C:
                k += 1
            elif c == 0x22:
                instr = False
        elif c == 0x22:
            instr = True
        elif c == 0x5B:
            depth += 1
        elif c == 0x5D:
            depth -= 1
            if depth == 0:
                break
        k += 1
    tape = json.loads(mm[j:k + 1].decode())
routes = json.load(open(f"{ch}/routes.json"))
if not os.path.exists(f"{ch}/routes.json.bak_open"):
    shutil.copy(f"{ch}/routes.json", f"{ch}/routes.json.bak_open")
routes["0"] = tape
json.dump(routes, open(f"{ch}/routes.json", "w"))
print(f"DECISION: route 0 <- opening {rid} (score {best[0]:.4f} vs {float(base[2]):.4f}, +{best[2]}/-{best[3]}, p {best[4]:.1e})")
