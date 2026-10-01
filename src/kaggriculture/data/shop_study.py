"""Shop-draw conditioning study + multi-SELL audit (levers L4/L6 gates).

docs/history/pair-improvement-plan.md phase 2: two analysis questions decide whether
two build levers live or die. No games are played -- everything reads the
route index and the obsfeat sidecars the mine already captured.

PART A -- demand gating (lever L4).
  The town draws one shop every 3 days, publicly (uniform over 8, with
  replacement); each unlocked shop consumes its products every 4 steps. The
  obsfeat sidecar's `shop_profile` records, per shop, the fraction of steps
  it was unlocked -- 0.9 means the day-3 draw, 0.8 day 6, and so on -- so
  the early demand lottery is recoverable for every indexed game.
  PAIRED TEST (confound-free): winner and loser of the SAME episode saw the
  same draws. If winners systematically put more of their premium sell mass
  on the drawn shop's products than the loser they beat, demand-matching
  pays and the gated route is worth building. Sign test per draw and pooled.

PART B -- multi-SELL audit (lever L6).
  Payload reordering inside a turn only matters on turns that sell 2+
  products. Count them in our live tape and the elite index; small counts
  kill the lever cheaply.

    python src/shop_study.py            # both parts, writes models/shop_study.json
    python src/shop_study.py --min-bank 20000   # elite games only
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
            "CARROT", "WHEAT", "FERTILIZER")
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP",
         "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
# Engine v1.32.7 SHOPS map (vendor/kaggle_environments/envs/kaggriculture).
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
OBSDIR = os.path.join(ROOT, "data", "obsfeat")
OUT = os.path.join(ROOT, "models", "shop_study.json")

# flat() layout in obs_features.py: reactivity[2] shop_mix[9] shop_profile[8]
# cash_shape[30] intraday[24] weed_slip[1]
MIX0, PROF0 = 2, 11


def _sidecar(rid):
    p = os.path.join(OBSDIR, rid + ".json")
    if not os.path.exists(p):
        return None
    try:
        v = json.load(open(p, encoding="utf-8"))
    except ValueError:
        return None
    return v if isinstance(v, list) and len(v) >= 74 else None


def day3_shop(profile):
    """The first-drawn shop, or None when duplicates make it ambiguous.

    A clean day-3 draw shows profile ~0.9 ((720-72)/720). A shop drawn twice
    sums its instances (e.g. 0.5+0.4 = 0.9 -- indistinguishable from a day-3
    single), so require the max to sit in the clean band AND no other shop
    to reach it.
    """
    mx = max(profile)
    if not (0.85 <= mx <= 1.05):
        return None
    tops = [s for s, v in zip(SHOPS, profile) if v >= 0.85]
    return tops[0] if len(tops) == 1 else None


def _sign_test(pos, neg):
    """Two-sided exact binomial sign test on discordant pairs."""
    import math
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def part_a(idx, min_bank):
    routes = idx["routes"]
    by_episode = defaultdict(dict)
    for rid, rec in routes.items():
        by_episode[rec.get("episode")][rec.get("seat")] = rec

    per_draw = defaultdict(lambda: [0, 0, []])   # draw -> [win+, lose+, deltas]
    pooled_pos = pooled_neg = 0
    used = 0
    for ep, seats in by_episode.items():
        if len(seats) != 2:
            continue
        a, b = seats.get(0), seats.get(1)
        if not a or not b or a.get("won") == b.get("won"):
            continue                      # need a decided game, both seats
        w, l = (a, b) if a["won"] else (b, a)
        if max(w.get("bank", 0), l.get("bank", 0)) < min_bank:
            continue
        sw, sl = _sidecar(w["id"]), _sidecar(l["id"])
        if not sw or not sl:
            continue
        draw = day3_shop(sw[PROF0:PROF0 + 8])
        if draw is None:
            continue
        prods = SHOP_PRODUCTS[draw]
        pick = [i for i, p in enumerate(PRODUCTS) if p in prods]
        wm = sum(sw[MIX0 + i] for i in pick)
        lm = sum(sl[MIX0 + i] for i in pick)
        d = wm - lm
        used += 1
        cell = per_draw[draw]
        if d > 0:
            cell[0] += 1
            pooled_pos += 1
        elif d < 0:
            cell[1] += 1
            pooled_neg += 1
        cell[2].append(d)

    rows = {}
    for draw, (pos, neg, deltas) in sorted(per_draw.items()):
        mean = sum(deltas) / len(deltas) if deltas else 0.0
        rows[draw] = {"episodes": len(deltas), "winner_matched_more": pos,
                      "loser_matched_more": neg,
                      "mean_delta_pp": round(100 * mean, 2),
                      "p": round(_sign_test(pos, neg), 5)}
    pooled_p = _sign_test(pooled_pos, pooled_neg)
    return {"episodes_used": used, "per_draw": rows,
            "pooled": {"winner_matched_more": pooled_pos,
                       "loser_matched_more": pooled_neg,
                       "p": round(pooled_p, 6)},
            "verdict": ("PASS -- winners demand-match; the gated route is "
                        "worth building"
                        if pooled_p < 0.05 and pooled_pos > pooled_neg else
                        "FAIL -- no demand-matching edge; drop lever L4")}


def _tape_sells(actions):
    """Per-turn SELL op counts from a route's action stream."""
    multi = total_sell_turns = 0
    for turn in actions:
        n = 0
        market = turn.get("market") if isinstance(turn, dict) else None
        for op in (market or []):
            if isinstance(op, (list, tuple)) and op and op[0] == "SELL":
                n += 1
        if n >= 1:
            total_sell_turns += 1
        if n >= 2:
            multi += 1
    return total_sell_turns, multi


def part_b(idx, top_n=40):
    import kaggriculture.data.routes as R
    ranked = sorted(idx["routes"].values(),
                    key=lambda r: -float(r.get("bank", 0)))[:top_n]
    tot = mul = loaded = 0
    for rec in ranked:
        try:
            rt = R.load_route(rec["id"])
        except Exception:                                          # noqa: BLE001
            continue
        actions = rt.get("actions") if isinstance(rt, dict) else rt
        if not actions:
            continue
        s, m = _tape_sells(actions)
        tot += s
        mul += m
        loaded += 1
    share = (100.0 * mul / tot) if tot else 0.0
    return {"elite_routes_scanned": loaded,
            "sell_turns": tot, "multi_sell_turns": mul,
            "multi_share_pct": round(share, 1),
            "verdict": ("worth auditing further -- reorder value bounded by "
                        f"{share:.0f}% of sell turns" if share >= 10 else
                        "FAIL -- multi-SELL turns are rare; drop lever L6")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-bank", type=float, default=0.0,
                    help="only episodes whose winner banked at least this")
    ap.add_argument("--top", type=int, default=40,
                    help="elite routes to scan in the multi-SELL audit")
    args = ap.parse_args()
    import kaggriculture.data.routes as R
    idx = R.load_index()
    print(f"index: {len(idx['routes'])} routes")

    a = part_a(idx, args.min_bank)
    print(f"\nPART A -- demand gating (paired winner-vs-loser, "
          f"{a['episodes_used']} decided episodes with clean day-3 draw)")
    for draw, r in a["per_draw"].items():
        print(f"  {draw:<15} n={r['episodes']:<5} winner-matched "
              f"{r['winner_matched_more']:>4} vs {r['loser_matched_more']:<4} "
              f"mean {r['mean_delta_pp']:+.2f}pp  p={r['p']}")
    print(f"  POOLED: {a['pooled']['winner_matched_more']} vs "
          f"{a['pooled']['loser_matched_more']}, p={a['pooled']['p']}")
    print(f"  VERDICT: {a['verdict']}")

    b = part_b(idx, args.top)
    print(f"\nPART B -- multi-SELL audit ({b['elite_routes_scanned']} elite "
          f"routes)")
    print(f"  sell turns {b['sell_turns']}, multi-SELL {b['multi_sell_turns']}"
          f" ({b['multi_share_pct']}%)")
    print(f"  VERDICT: {b['verdict']}")

    json.dump({"when": __import__("datetime").datetime.now()
               .isoformat(timespec="seconds"),
               "min_bank": args.min_bank, "part_a": a, "part_b": b},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
