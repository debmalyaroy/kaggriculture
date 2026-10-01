"""Route library v2: finalists of a lineage screen (top --per candidates per world with net >= --min-net).

    python python/routes/finalists.py data/routes/screen_lineage_c1.json [--per 2] [--min-net 4]

Prints the keys (comma list, for screen_public.py --keys) and writes <screen>.finalists.json.
"""
import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("screen")
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--min-net", type=int, default=4)
    a = ap.parse_args()
    rows = json.load(open(a.screen))
    by = collections.defaultdict(list)
    for r in rows:
        if r["net"] >= a.min_net and r["world_mismatch"] == 0:
            by[r["world"]].append(r)
    fin = [r for w in sorted(by) for r in sorted(by[w], key=lambda r: (-r["net"], r["p"]))[: a.per]]
    json.dump(fin, open(a.screen.replace(".json", ".finalists.json"), "w"), indent=1)
    med = sorted(r["net"] for r in rows)[len(rows) // 2] if rows else None
    print(f"# {len(rows)} screened, median net {med}; {len(fin)} finalists in {len(by)} worlds")
    for r in fin:
        print(f"# {r['world']:34s} {r['key']:14s} +{r['better']}/-{r['worse']} net {r['net']:+d} p {r['p']:.3f}")
    print(",".join(r["key"] for r in fin))


if __name__ == "__main__":
    main()
