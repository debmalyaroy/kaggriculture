"""v63.6_rl release-candidate report: the round robin (stdio-match TSV) and the public-25 tournament paired against
v63.5_rl's run on the same cells (opp, world, seed, seat).

    python python/rshell/rc_report.py [--rr data/rshell/tourn_v636.tsv] [--pub data/rshell/public25_v636/games.jsonl]
        [--ref-pub data/rshell/public25/games.jsonl] [--out data/rshell/rc_v636.json]
"""
import argparse
import json
import math
import os
from collections import defaultdict

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mcnemar(b, w):
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wld(gaps):
    w = sum(g > 0 for g in gaps); l = sum(g < 0 for g in gaps); d = len(gaps) - w - l
    return {"W": w, "L": l, "D": d, "n": len(gaps), "score": round((w + 0.5 * d) / max(1, len(gaps)), 4),
            "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else None}


def round_robin(path):
    pairs = defaultdict(list)
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 6:
            continue
        a, b = p[0].split(":")
        bf, bs = float(p[4]), float(p[5])
        if a == b:
            pairs[(a, a)].append(bf - bs)
            continue
        pairs[(a, b)].append(bf - bs)
    out = {}
    for (a, b), g in sorted(pairs.items()):
        out[f"{a} vs {b}"] = wld(g)
    return out


def load_pub(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if "gap" in r:
            rows[(r["opp"], r["world"], r["seed"], r["seat"])] = r
    return rows


def public(cand, ref):
    per = defaultdict(lambda: {"c": [], "r": [], "better": 0, "worse": 0})
    worlds = defaultdict(lambda: {"c": [], "r": []})
    for k, c in cand.items():
        o = per[k[0]]
        o["c"].append(c["gap"])
        worlds[k[1]]["c"].append(c["gap"])
        r = ref.get(k)
        if r is None:
            continue
        o["r"].append(r["gap"])
        worlds[k[1]]["r"].append(r["gap"])
        cw, rw = c["gap"] > 0, r["gap"] > 0
        o["better"] += cw and not rw
        o["worse"] += rw and not cw
    tot = {"better": sum(v["better"] for v in per.values()), "worse": sum(v["worse"] for v in per.values())}
    tot["p"] = round(mcnemar(tot["better"], tot["worse"]), 4)
    allc = [g for v in per.values() for g in v["c"]]
    allr = [g for v in per.values() for g in v["r"]]
    return {"cand": wld(allc), "ref": wld(allr), "paired": tot,
            "per_opponent": {o: {"cand": wld(v["c"]), "ref": wld(v["r"]), "better": v["better"], "worse": v["worse"],
                                 "p": round(mcnemar(v["better"], v["worse"]), 4)} for o, v in sorted(per.items())},
            "per_world": {w: {"cand": wld(v["c"])["score"], "ref": wld(v["r"])["score"], "n": len(v["c"])}
                          for w, v in sorted(worlds.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", default=os.path.join(RL, "data", "rshell", "tourn_v636.tsv"))
    ap.add_argument("--pub", default=os.path.join(RL, "data", "rshell", "public25_v636", "games.jsonl"))
    ap.add_argument("--ref-pub", default=os.path.join(RL, "data", "rshell", "public25", "games.jsonl"))
    ap.add_argument("--out", default=os.path.join(RL, "data", "rshell", "rc_v636.json"))
    a = ap.parse_args()
    rep = {}
    if os.path.exists(a.rr):
        rep["round_robin"] = round_robin(a.rr)
    if os.path.exists(a.pub):
        rep["public25"] = public(load_pub(a.pub), load_pub(a.ref_pub))
    json.dump(rep, open(a.out, "w"), indent=1)
    for k, v in rep.get("round_robin", {}).items():
        print(f"[rr] {k:24s} W {v['W']:4d} L {v['L']:4d} D {v['D']:3d} score {v['score']:.3f} median gap {v['median_gap']}")
    p = rep.get("public25")
    if p:
        print(f"[pub] v63.6 {p['cand']}  v63.5 {p['ref']}  paired +{p['paired']['better']}/-{p['paired']['worse']} p {p['paired']['p']}")
        for o, v in p["per_opponent"].items():
            print(f"[pub] {o[:48]:48s} {v['cand']['score']:.3f} vs {v['ref']['score']:.3f}  +{v['better']}/-{v['worse']}")
        wb = [w for w, v in p["per_world"].items() if v["cand"] < v["ref"]]
        print(f"[pub] worlds worse than v63.5: {len(wb)}/{len(p['per_world'])} {wb}")


if __name__ == "__main__":
    main()
