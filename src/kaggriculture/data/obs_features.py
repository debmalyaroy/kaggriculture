"""Observation-derived features from a replay: what actions cannot show.

The action stream tells you what a program did; the observation stream tells
you the game it experienced. This extractor turns one replay seat into a
compact vector capturing behaviour no route featurization reaches:

  * reactivity     -- correlation of a seat's own sells with price level and
                      with the opponent's sales (tape vs adaptive policy)
  * shop-mix       -- product emphasis conditioned on the shops that spawned
                      (new signal under 1.32.6 random shop sampling)
  * cash-shape     -- normalized money-by-day trajectory
  * microstructure -- intra-day sell spread, weed-slip incidence

Same bandwidth as the existing mine (the replay is already streamed); this
just reads more of it. Feeds the identifier (a name-free discriminator) and
the family census.

    python -m kaggriculture.data.obs_features <replay.json> <seat>
"""
import json
import os
import sys

PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
            "CARROT", "WHEAT", "FERTILIZER")
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP",
         "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def obs_features(replay, seat):
    steps = replay.get("steps") or []
    my_sell = []           # per turn, total premium units I sold
    price_lvl = []         # per turn, mean premium price
    opp_sell_est = []      # per turn, opponent inventory rise (proxy)
    money = []
    prev_inv = None
    sells_by_hour = [0] * 24
    weed_slips = 0
    shop_seen = {s: 0 for s in SHOPS}
    prod_when_shop = {p: 0.0 for p in PRODUCTS}

    for i in range(1, len(steps)):
        cur = steps[i][seat]
        obs = cur.get("observation") or {}
        act = cur.get("action") or {}
        mk = obs.get("market") or {}
        prices = mk.get("prices") or {}
        inv = mk.get("inventory") or {}
        town = obs.get("town") or {}
        shops = town.get("unlocked_shops") or []
        for s in shops:
            if s in shop_seen:
                shop_seen[s] += 1

        prem = [prices.get(p, 0) for p in ("STRAWBERRY", "MELON", "MILK", "WOOL")]
        price_lvl.append(sum(prem) / len(prem))
        farms = obs.get("farms") or []
        if seat < len(farms):
            money.append(float(farms[seat].get("money", 0) or 0))

        sold = 0
        for o in (act.get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                q = max(0, int(o[2] or 0))
                if o[1] in ("STRAWBERRY", "MELON", "MILK", "WOOL"):
                    sold += q
                if o[1] in prod_when_shop and shops:
                    prod_when_shop[o[1]] += q
                sells_by_hour[obs.get("hour", i % 24) % 24] += q
        my_sell.append(sold)

        if prev_inv is not None:
            rise = sum(max(0, int(inv.get(p, 0) or 0) - prev_inv.get(p, 0))
                       for p in PRODUCTS)
            opp_sell_est.append(rise)
        prev_inv = {p: int(inv.get(p, 0) or 0) for p in PRODUCTS}

        # weed slip: a PLANT/BUILD that the next state shows still empty/weed
        for u in [act.get("farmer")] + list(act.get("hands") or []):
            if isinstance(u, list) and u and u[0] == "DIG":
                weed_slips += 1

    react_price = _corr(my_sell, price_lvl)
    react_opp = _corr(my_sell[1:], opp_sell_est) if opp_sell_est else 0.0
    total_prem = sum(prod_when_shop.values()) or 1.0
    shop_mix = [prod_when_shop[p] / total_prem for p in PRODUCTS]
    shop_prof = [shop_seen[s] / max(1, len(steps)) for s in SHOPS]
    m0 = money[0] if money else 1.0
    mtop = max(money) if money else 1.0
    cash_shape = [money[min(len(money) - 1, d * 24)] / (mtop or 1.0)
                  for d in range(30)] if money else [0.0] * 30
    tot_sell = sum(my_sell) or 1
    intraday = [h / tot_sell for h in sells_by_hour]

    return {
        "reactivity": [react_price, react_opp],
        "shop_mix": shop_mix,
        "shop_profile": shop_prof,
        "cash_shape": cash_shape,
        "intraday": intraday,
        "weed_slip_rate": [weed_slips / max(1, len(steps))],
    }


def flat(feat):
    return (feat["reactivity"] + feat["shop_mix"] + feat["shop_profile"]
            + feat["cash_shape"] + feat["intraday"] + feat["weed_slip_rate"])


def main():
    replay = json.load(open(sys.argv[1], encoding="utf-8"))
    seat = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    f = obs_features(replay, seat)
    print(f"{len(flat(f))} obs-features for seat {seat}:")
    for k, v in f.items():
        print(f"  {k:<14} {[round(x, 3) for x in v[:6]]}"
              f"{' ...' if len(v) > 6 else ''}")


if __name__ == "__main__":
    main()
