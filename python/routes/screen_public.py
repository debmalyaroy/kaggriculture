"""Route library v2, step A3 (public stage) and A4 (held-out): candidate routes vs the top-25 public agents.

    PYTHONPATH=~/kaggriculture/src KAGG_BIN=... python python/routes/screen_public.py --keys K1,K2 [--split train]
        [--skip 100] [--seeds 4] [--workers 12] [--out data/routes/screen_public.json]

For each candidate: a stage folder (.local/ext/routes/<key>/) = v63.5_rl's stage with the candidate base
(configs/bases/v61.1x or v61.1xl), a route table {its world: its route}, and main.py passing --route-table. The
reference is v63.5_rl's own stage (.local/ext/cand_v635). Both play the 25 public agents (Python, main-repo harness)
on the candidate world's bank seeds from both seats; paired better / worse per candidate. Each game ends its
agent-stdio child (python/rshell/public25.py play()).
"""
import argparse
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python", "rshell"))
import public25 as P25  # noqa: E402

REF_STAGE = os.path.join(RL, ".local", "ext", "cand_v635")
FIELD = os.path.join(RL, ".local", "ext", "field25")


def stage_for(c):
    d = os.path.join(RL, ".local", "ext", "routes", c["key"])
    if os.path.isdir(d):
        return d
    os.makedirs(d)
    for f in os.listdir(REF_STAGE):
        if f in ("base", "main.py"):
            continue
        src = os.path.join(REF_STAGE, f)
        os.symlink(os.path.realpath(src), os.path.join(d, f))
    os.symlink(os.path.join(RL, "configs", "bases", c["base"]), os.path.join(d, "base"))
    json.dump({c["world"]: c["route"]}, open(os.path.join(d, "route_table.json"), "w"))
    m = open(os.path.join(REF_STAGE, "main.py"), encoding="utf-8").read()
    a = 'cmd = [path, "--base", os.path.join(root, "base")]'
    assert a in m
    m = m.replace(a, a + '\n        cmd += ["--route-table", os.path.join(root, "route_table.json")]', 1)
    open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(m)
    return d


def signp(b, w):
    n = b + w
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--cands", default=os.path.join(RL, "data", "routes", "cands_stage2.json"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(RL, "data", "routes", "screen_public.json"))
    a = ap.parse_args()
    keys = set(a.keys.split(","))
    cands = [c for c in json.load(open(a.cands)) if c["key"] in keys]
    opps = json.load(open(os.path.join(RL, ".local", "ext", "top25_public.json"), encoding="utf-8"))
    bank = json.load(open(os.path.join(RL, "data", "worlds", "w64_bank.json")))[a.split]
    tasks = []
    for w in sorted({c["world"] for c in cands}):
        for s in bank[w][a.skip: a.skip + a.seeds]:
            for o in opps:
                for seat in (0, 1):
                    tasks.append(("REF", w, (os.path.join(REF_STAGE, "main.py"), o, os.path.join(FIELD, o + ".py"), w, s, seat)))
                    for c in cands:
                        if c["world"] == w:
                            tasks.append((c["key"], w, (os.path.join(stage_for(c), "main.py"), o, os.path.join(FIELD, o + ".py"), w, s, seat)))
    print(f"[routes-public] {len(cands)} candidates, {len(tasks)} games", flush=True)
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(P25.play, t[2]): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            who, w, t = futs[f]
            r = f.result()
            if "gap" in r:
                res[(who, t[1], t[4], t[5])] = 1.0 if r["gap"] > 0 else 0.0 if r["gap"] < 0 else 0.5
            if i % 500 == 0:
                print(f"[routes-public] {i}/{len(tasks)}", flush=True)
    rows = []
    for c in cands:
        ks = [k for k in res if k[0] == c["key"]]
        pairs = [(res[k], res.get(("REF",) + k[1:])) for k in ks if ("REF",) + k[1:] in res]
        b = sum(x > y for x, y in pairs)
        w = sum(x < y for x, y in pairs)
        rows.append({"key": c["key"], "world": c["world"], "route": c["route"], "base": c["base"], "games": len(pairs),
                     "score": sum(x for x, _ in pairs) / max(1, len(pairs)), "ref_score": sum(y for _, y in pairs) / max(1, len(pairs)),
                     "better": b, "worse": w, "net": b - w, "p": signp(b, w)})
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["world"], -r["net"])):
        print(f"[routes-public] {r['world']:34s} {r['key']:16s} +{r['better']}/-{r['worse']} score {r['score']:.3f} vs {r['ref_score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
