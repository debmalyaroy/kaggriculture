"""Dispatcher feature extractor (2026-09-07).

The observable state a learned tape-selector conditions on — mirrors the
93.8%-router meta: our vs rival money, market prices/inventory, shop counts
+ derived DEMAND (shop->product), and BOTH farms' board (crops, animals,
yields, weeds), plus our shed. Pure-python (math only) so the exact same
function can be inlined into the shipped agent and used offline for
training — no train/serve skew.

`features(obs)` -> fixed-length list (FEATN). Robust to missing fields.
"""
from __future__ import annotations

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG",
         "MILK", "WOOL", "FERTILIZER")
SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")
# shop -> product indices it drives demand for (coarse public mapping)
SHOP_DEMAND = {
    "BAKERY": (0, 3), "BRUNCH_SPOT": (5, 6), "FARMERS_MARKET": (0, 1, 2),
    "ICE_CREAM_SHOP": (6, 4), "PET_CAFE": (5, 6, 7), "PIZZA_SHOP": (2, 3),
    "SMOOTHIE_SHOP": (3, 4), "YARN_STORE": (7,),
}
ANIMAL_PROD = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
FEATN = 60


def _board(farm):
    """crops[9-ish], animals, yields, weeds for one farm."""
    crop = {k: 0 for k in ITEMS}
    yld = {k: 0.0 for k in ITEMS}
    weeds = animals = 0
    for row in (farm.get("tiles") or []):
        cells = row if isinstance(row, list) else [row]
        for t in cells:
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT":
                c = t.get("crop")
                if c in crop:
                    crop[c] += 1
                    yld[c] += float(t.get("yield_units") or 0)
            elif t.get("animal"):
                a = t.get("animal")
                animals += 1
                p = ANIMAL_PROD.get(a)
                if p:
                    yld[p] += float(t.get("yield_units") or 0)
            elif t.get("kind") == "WEED":
                weeds += 1
    return crop, yld, weeds, animals


def features(obs):
    try:
        p = int(obs.get("player", 0) or 0)
    except Exception:                                          # noqa: BLE001
        p = 0
    farms = obs.get("farms") or [{}, {}]
    own = farms[p] if p < len(farms) else {}
    riv = farms[1 - p] if (1 - p) < len(farms) else {}
    market = obs.get("market") or {}
    prices = market.get("prices") or {}
    inv = market.get("inventory") or {}
    x = [float(own.get("money") or 0) / 1e5,
         float(riv.get("money") or 0) / 1e5,
         (float(own.get("money") or 0) - float(riv.get("money") or 0)) / 1e5]
    x += [float(prices.get(i, 0) or 0) / 100.0 for i in ITEMS]
    x += [(float(inv.get(i, 10000) or 10000) - 10000) / 10000.0
          for i in ITEMS]
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    counts = [shops.count(s) for s in SHOPS]
    x += counts
    demand = [0] * 9
    for s, c in zip(SHOPS, counts):
        for i in SHOP_DEMAND.get(s, ()):
            demand[i] += c
    x += demand
    for farm in (own, riv):
        crop, yld, weeds, animals = _board(farm)
        x += [crop[i] for i in ITEMS[:5]]           # crops
        x += [animals, weeds]
        x += [yld[i] / 100.0 for i in ("MELON", "MILK", "WOOL", "EGG")]
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    x += [float(shed.get(i, 0) or 0) / 100.0 for i in ITEMS]
    step = int(obs.get("step", 0) or 0)
    x.append(step / 720.0)
    # pad/truncate to fixed length
    if len(x) < FEATN:
        x += [0.0] * (FEATN - len(x))
    return x[:FEATN]
