"""F3.1 -- net-P&L / cost-leak accounting over the top-100 replay corpus.

The whole ladder-loss diagnosis (see memory `root-cause-price-realization`,
`wool-overproduction-root-cause`) points at ONE gap: we dump high VOLUME at a
crashed price while winners RESTRICT supply and capture scarcity. This
quantifies that at corpus scale and PER PRODUCT, so the Slot-1 OR economy (B)
gets concrete per-product price/volume targets instead of a slogan.

Method (sell-side, the proven gap): for each replay, `rewards` gives the final
banks → winner/loser seats. Walk each seat's steps; for every SELL <item> <n>
add `n` to that product's volume and `n * price[item]` (the pre-order quoted
price at that step) to its revenue. Aggregate WINNER vs LOSER: per-product
volume, revenue, and volume-weighted average realized price.

Caveat (honest): the quoted pre-order price OVERSTATES realized $/unit when
dumping (marginal price decays within the turn), so the winner/loser price gap
here is a CONSERVATIVE lower bound on the real dump penalty. Exact realized
revenue needs a market re-sim (a later refinement). Buy-side costs (dynamic buy
prices, Fibonacci HIRE, tiered LAND) also need the engine and are out of this
first cut; the outcome is anchored on the actual final `rewards`.

    python -m kaggriculture.measure.pnl_replay --limit 500
    python -m kaggriculture.measure.pnl_replay            # whole corpus
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import collections
import glob
import json
import os

PARTS = os.path.join(ROOT, ".local", "scratch", "gm", "top100_replays_parts")
PRODUCTS = ("CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
            "STRAWBERRY", "TOMATO", "WHEAT", "WOOL")


def _iter_replays(parts_dir, limit):
    import pyarrow.parquet as pq
    n = 0
    for f in sorted(glob.glob(os.path.join(parts_dir, "*.parquet"))):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=16, columns=["episode_id", "replay_json"]):
            for eid, rj in zip(b.column("episode_id").to_pylist(),
                               b.column("replay_json").to_pylist()):
                try:
                    yield eid, json.loads(rj)
                except (ValueError, TypeError):
                    continue
                n += 1
                if limit and n >= limit:
                    return


def _seat_sells(steps, seat):
    """{product: [volume, revenue]} from this seat's SELL actions."""
    acc = collections.defaultdict(lambda: [0, 0.0])
    for t in range(len(steps)):
        cell = steps[t][seat]
        act = cell.get("action") or {}
        market = act.get("market") or []
        if not market:
            continue
        prices = (cell.get("observation", {}) or {}).get("market", {}).get("prices", {})
        for order in market:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                item, qty = order[1], order[2]
                try:
                    qty = int(qty)
                except (ValueError, TypeError):
                    continue
                px = prices.get(item, 0) or 0
                acc[item][0] += qty
                acc[item][1] += qty * px
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=500,
                    help="max replays to scan (0 = all)")
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--out", default=".local/pnl_replay_2026-09-18.json")
    args = ap.parse_args()

    # per-product [volume, revenue] for winners and losers, and per-game banks
    win = collections.defaultdict(lambda: [0, 0.0])
    los = collections.defaultdict(lambda: [0, 0.0])
    n_games = 0
    win_bank = los_bank = 0.0

    for _eid, rj in _iter_replays(args.parts, args.limit):
        rewards = rj.get("rewards") or []
        steps = rj.get("steps") or []
        if len(rewards) != 2 or not steps or rewards[0] == rewards[1]:
            continue
        w = 0 if rewards[0] > rewards[1] else 1
        n_games += 1
        win_bank += rewards[w]; los_bank += rewards[1 - w]
        for prod, (v, r) in _seat_sells(steps, w).items():
            win[prod][0] += v; win[prod][1] += r
        for prod, (v, r) in _seat_sells(steps, 1 - w).items():
            los[prod][0] += v; los[prod][1] += r

    if not n_games:
        print("no usable replays found under", args.parts)
        return 1

    def avg_px(d, p):
        v, r = d[p]
        return (r / v) if v else 0.0

    print(f"=== F3.1 net-P&L (sell-side) over {n_games} decided games ===")
    print(f"winner avg bank {win_bank / n_games:,.0f}  vs  loser {los_bank / n_games:,.0f}\n")
    print(f"{'product':<11}  {'W vol/g':>8} {'W $/u':>7}  {'L vol/g':>8} {'L $/u':>7}"
          f"  {'price gap':>9}  {'vol gap':>8}")
    rows = []
    for p in PRODUCTS:
        wv = win[p][0] / n_games; lv = los[p][0] / n_games
        wpx = avg_px(win, p); lpx = avg_px(los, p)
        print(f"{p:<11}  {wv:>8.1f} {wpx:>7.1f}  {lv:>8.1f} {lpx:>7.1f}"
              f"  {wpx - lpx:>+9.1f}  {wv - lv:>+8.1f}")
        rows.append({"product": p, "win_vol_per_game": round(wv, 2),
                     "win_price": round(wpx, 2), "los_vol_per_game": round(lv, 2),
                     "los_price": round(lpx, 2), "price_gap": round(wpx - lpx, 2),
                     "vol_gap": round(wv - lv, 2)})

    out = {"when": "2026-09-18", "games": n_games,
           "win_bank": round(win_bank / n_games, 1),
           "los_bank": round(los_bank / n_games, 1),
           "note": "quoted-price approx overstates realized $/u when dumping "
                   "-> price gap is a conservative lower bound; buy-side costs "
                   "not included; outcome anchored on final rewards",
           "products": rows}
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}")
    print("READ: products where winners get a higher $/u AND sell less volume "
          "are the scarcity-capture targets for the Slot-1 OR economy (B1.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
