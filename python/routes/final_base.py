"""Route library v2, step A5 helper: ONE base with only the chosen routes (v61.1 library + survivors, ids kept).

    python python/routes/final_base.py --keys K1,K2 [--name v61.1r]

Survivors come from different screening bases (v61.1x ids 1000+, v61.1y ids 2000+); their ids are kept, so a route
table written against those ids plays the same routes here. None of them has a late land buy or tomato planting,
so V219's whole-library check is unchanged. Writes configs/bases/<name>/ and data/routes/final_cands.json.
"""
import argparse
import json
import os
import shutil

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--name", default="v61.1r")
    a = ap.parse_args()
    import glob
    allc = [c for f in sorted(glob.glob(os.path.join(RL, "data", "routes", "cands*_stage2.json"))) for c in json.load(open(f))]
    base = os.path.join(RL, "configs", "bases", "v61.1")
    routes = json.load(open(os.path.join(base, "routes.json")))
    out = dict(routes)
    chosen = []
    cache = {}
    for k in a.keys.split(","):
        c = next(x for x in allc if x["key"] == k)
        assert not c.get("late"), k
        if c["base"] not in cache:
            cache[c["base"]] = json.load(open(os.path.join(RL, "configs", "bases", c["base"], "routes.json")))
        out[str(c["route"])] = cache[c["base"]][str(c["route"])]
        chosen.append({**c, "base": a.name})
    d = os.path.join(RL, "configs", "bases", a.name)
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "routes.json"), "w"), separators=(",", ":"))
    shutil.copy(os.path.join(base, "router.json"), os.path.join(d, "router.json"))
    json.dump(chosen, open(os.path.join(RL, "data", "routes", "final_cands.json"), "w"), indent=1)
    print(f"[final-base] {a.name}: {len(out)} routes ({len(chosen)} added)")


if __name__ == "__main__":
    main()
