"""Probe episode seeds for their shop draws on the RUST SERVE substrate.

The shop sequence is a function of the episode seed (shared RNG). The
router's per-shop selection needs seed pools per first-shop and per
shop-pair; this probes seeds by stepping a PASS-only game to step 145 and
reading the town. ~0.4s/seed on serve; results cached in
models/router/seed_shops.json (append-only).

    python src/seed_shops.py --start 200000 --count 400
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CACHE = os.path.join(ROOT, "models", "router", "seed_shops.json")


def probe(seeds):
    import kaggriculture.engine.serve_match as SM
    srv = SM.Serve()
    out = {}
    try:
        for sd in seeds:
            js = srv.cmd(f"RESET {sd}")
            shops = []
            step = 0
            while not js.get("done") and step <= 146:
                js = srv.cmd("STEP2 PASS\t\t\x1ePASS\t\t")
                step += 1
                cur = list((js.get("town") or {}).get("unlocked_shops") or [])
                if len(cur) > len(shops):
                    shops = cur
                if len(shops) >= 2:
                    break
            out[str(sd)] = shops[:2]
    finally:
        srv.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=200000)
    ap.add_argument("--count", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding="utf-8"))
    todo = [s for s in range(args.start, args.start + args.count)
            if str(s) not in cache]
    print(f"probing {len(todo)} seeds ({len(cache)} cached)")
    if todo:
        cache.update(probe(todo))
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    from collections import Counter
    firsts = Counter(v[0] for v in cache.values() if v)
    print("first-shop distribution:", dict(firsts))
    print(f"cached: {len(cache)} seeds -> {CACHE}")


if __name__ == "__main__":
    main()
