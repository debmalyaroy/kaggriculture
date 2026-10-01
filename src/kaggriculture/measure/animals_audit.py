"""F3.3 -- animals + fertilizer profitability audit over the replay corpus.

Field-wide weak spot (memory `frontier-gap-is-wool-yarn-routing`,
`trackp-marginal-unit`): do WINNERS actually invest in animals/fertilizer and
profit, or is livestock a cash sink? This tells the Slot-1 OR model (B1.2)
whether — and how much — to build the animal/fertilizer economy.

Counts, per episode per seat: animals bought (COW/SHEEP/GOOSE), coops/pastures
built, FERTILIZE / COLLECT_FERTILIZER ops, and animal-product SELL revenue
(MILK/WOOL/EGG at quoted price). Splits WINNER vs LOSER. Anchored on final
`rewards`. Buy-side animal/feed COST needs the engine (later); this is the
investment+revenue picture, which is enough to answer "do winners lean in?".

    python -m kaggriculture.measure.animals_audit --limit 800
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import collections
import glob
import json
import os

ANIMAL_PRODUCTS = ("MILK", "WOOL", "EGG")


def _iter(parts_dir, limit):
    import pyarrow.parquet as pq
    n = 0
    for f in sorted(glob.glob(os.path.join(parts_dir, "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=16,
                                                columns=["replay_json"]):
            for rj in b.column("replay_json").to_pylist():
                try:
                    yield json.loads(rj)
                except (ValueError, TypeError):
                    continue
                n += 1
                if limit and n >= limit:
                    return


def _seat_stats(steps, seat):
    s = collections.Counter()
    prod_rev = collections.defaultdict(float)
    for t in range(len(steps)):
        cell = steps[t][seat]
        act = cell.get("action") or {}
        prices = (cell.get("observation", {}) or {}).get("market", {}).get("prices", {})
        for unit in ([act.get("farmer")] + (act.get("hands") or [])):
            if not isinstance(unit, list) or not unit:
                continue
            op = unit[0]
            if op == "BUILD_COOP":
                s["coop"] += 1
            elif op == "BUILD_PASTURE":
                s["pasture"] += 1
            elif op == "FERTILIZE":
                s["fertilize"] += 1
            elif op == "COLLECT_FERTILIZER":
                s["collect_fert"] += 1
            elif op == "FEED":
                s["feed"] += 1
            elif op == "CARE":
                s["care"] += 1
        for order in (act.get("market") or []):
            if not isinstance(order, list) or len(order) < 3:
                continue
            if order[0] == "BUY_ANIMAL":
                s["animals_bought"] += int(order[2]) if str(order[2]).isdigit() else 1
            elif order[0] == "SELL" and order[1] in ANIMAL_PRODUCTS:
                try:
                    q = int(order[2])
                except (ValueError, TypeError):
                    continue
                prod_rev[order[1]] += q * (prices.get(order[1], 0) or 0)
    return s, prod_rev


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--parts", default=os.path.join(
        ROOT, ".local", "scratch", "gm", "top100_replays_parts"))
    ap.add_argument("--out", default=".local/animals_audit_2026-09-18.json")
    args = ap.parse_args()

    agg = {"win": collections.Counter(), "los": collections.Counter()}
    rev = {"win": collections.defaultdict(float), "los": collections.defaultdict(float)}
    n = 0
    for rj in _iter(args.parts, args.limit):
        rewards = rj.get("rewards") or []
        steps = rj.get("steps") or []
        if len(rewards) != 2 or not steps or rewards[0] == rewards[1]:
            continue
        w = 0 if rewards[0] > rewards[1] else 1
        n += 1
        for tag, seat in (("win", w), ("los", 1 - w)):
            s, pr = _seat_stats(steps, seat)
            agg[tag].update(s)
            for k, v in pr.items():
                rev[tag][k] += v

    if not n:
        print("no usable replays"); return 1

    print(f"=== F3.3 animals+fertilizer audit over {n} decided games ===\n")
    keys = ["animals_bought", "coop", "pasture", "feed", "care",
            "fertilize", "collect_fert"]
    print(f"{'metric':<16} {'winner/g':>10} {'loser/g':>10}  {'W-L':>8}")
    for k in keys:
        wv, lv = agg["win"][k] / n, agg["los"][k] / n
        print(f"{k:<16} {wv:>10.2f} {lv:>10.2f}  {wv - lv:>+8.2f}")
    print(f"\n{'anim product':<16} {'W rev/g':>10} {'L rev/g':>10}  {'W-L':>8}")
    for p in ANIMAL_PRODUCTS:
        wv, lv = rev["win"][p] / n, rev["los"][p] / n
        print(f"{p:<16} {wv:>10.0f} {lv:>10.0f}  {wv - lv:>+8.0f}")
    tot_w = sum(rev["win"].values()) / n
    tot_l = sum(rev["los"].values()) / n
    print(f"{'TOTAL anim rev':<16} {tot_w:>10.0f} {tot_l:>10.0f}  {tot_w - tot_l:>+8.0f}")

    out = {"games": n,
           "winner": {k: round(agg["win"][k] / n, 3) for k in keys},
           "loser": {k: round(agg["los"][k] / n, 3) for k in keys},
           "winner_anim_rev": {p: round(rev["win"][p] / n, 1) for p in ANIMAL_PRODUCTS},
           "loser_anim_rev": {p: round(rev["los"][p] / n, 1) for p in ANIMAL_PRODUCTS}}
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")
    print("READ: if winners buy MORE animals + earn MORE animal-product revenue, "
          "the OR model (B1.2) should build the animal economy, not skip it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
