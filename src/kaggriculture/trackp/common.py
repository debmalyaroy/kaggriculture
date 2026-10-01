"""Track P foundation: paths, engine tables, price model, replay IO, tapes.

Freshly transcribed from the official interpreter
(vendor/kaggle_environments/envs/kaggriculture/kaggriculture.py) and cross-
checked against the bit-exact Rust port. No imports from any bandit/route
module -- that isolation is load-bearing and tested.
"""
from __future__ import annotations

import json
import math
import os
import sys

# ------------------------------------------------------------------- paths --

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data", "trackp")
MODELS = os.path.join(ROOT, "models", "trackp")
REPLAYS = os.path.join(DATA, "replays")        # Track P's own replay cache
TRACES = os.path.join(DATA, "traces_v2")       # ~115-dim per-turn traces
LOGS = os.path.join(ROOT, "data", "logs")
KAGG = os.path.join(ROOT, "rustengine", "kagg.exe")
VENDOR = os.path.join(ROOT, "vendor")

for _d in (DATA, MODELS, REPLAYS, TRACES):
    os.makedirs(_d, exist_ok=True)


def vendored_env():
    """Import the vendored official interpreter (the ladder's engine)."""
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import kaggle_environments  # noqa: WPS433 (deliberate lazy import)
    return kaggle_environments


# ----------------------------------------------------------- engine tables --

BOARD = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
SHED_CAP = 100
MAX_MARKET_ORDERS = 10
STARTING_MONEY = 3000
WEED_CHANCE = 0.005
TOWN_SHOP_SELL_INTERVAL = 4
TOWN_CENTER_SELL_INTERVAL = 24
TOWN_SHOP_UNLOCK_INTERVAL = 3
MAX_SHOP_INSTANCES = 8

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMAL_NAMES = ["GOOSE", "COW", "SHEEP"]

# name: (seed_cost, first_yield_day, max_yield_day, interval, max_yield, ongoing)
CROPS = {
    "WHEAT":      (10, 2, 4, 0, 6, False),
    "CARROT":     (20, 2, 3, 0, 4, False),
    "TOMATO":     (50, 8, 8, 1, 4, True),
    "STRAWBERRY": (100, 10, 10, 2, 4, True),
    "MELON":      (80, 10, 12, 0, 6, False),
}
# name: (cost, structure, first_yield_day, interval, max_held, product)
ANIMALS = {
    "GOOSE": (300, "COOP", 4, 1, 4, "EGG"),
    "COW":   (400, "PASTURE", 8, 2, 6, "MILK"),
    "SHEEP": (500, "PASTURE", 6, 3, 6, "WOOL"),
}

SHOPS_SORTED = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
                "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
SHOP_PRODUCTS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

# item: (base, I0, T, below_func, below_target, above_func, above_target)
MARKET_PARAMS = {
    "WHEAT":      (25.0, 10000.0, 400.0, "sqrt", 0.80, "log", 0.20),
    "CARROT":     (35.0, 10000.0, 450.0, "log", 0.20, "sqrt", 0.70),
    "TOMATO":     (60.0, 10000.0, 200.0, "linear", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120.0, 10000.0, 100.0, "sqrt", 0.70, "linear", 1.60),
    "MELON":      (250.0, 10000.0, 300.0, "log", 0.20, "sq", 3.60),
    "EGG":        (50.0, 10000.0, 332.0, "linear", 0.40, "log", 0.20),
    "MILK":       (160.0, 10000.0, 122.0, "sqrt", 0.60, "linear", 1.60),
    "WOOL":       (200.0, 10000.0, 105.0, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100.0, 10000.0, 200.0, "linear", 0.40, "linear", 0.40),
}
PRICE_FLOOR = 1
HINGE_GAIN = 8.0

# Engine 1.32.7 (PR #1399): CARROT/TOMATO/EGG scarcity side -> "hinge"
# (calm to capacity T, quadratic runaway past it). The active table is
# selected by models/engine_version.json, written ONLY by
# scripts/engine_swap_1327.py when the LADDER's replays flip.
_MARKET_PARAMS_1327 = dict(MARKET_PARAMS)
_MARKET_PARAMS_1327["CARROT"] = (35.0, 10000.0, 450.0, "hinge", 1.00,
                                 "sqrt", 0.70)
_MARKET_PARAMS_1327["TOMATO"] = (60.0, 10000.0, 200.0, "hinge", 0.40,
                                 "sqrt", 0.60)
_MARKET_PARAMS_1327["EGG"] = (50.0, 10000.0, 332.0, "hinge", 0.40,
                              "log", 0.20)

ENGINE_STATE = os.path.join(ROOT, "models", "engine_version.json")


def engine_version() -> str:
    try:
        with open(ENGINE_STATE, encoding="utf-8") as fh:
            return json.load(fh).get("engine", "1.32.6")
    except (OSError, ValueError):
        return "1.32.6"


def _ver_tuple(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def active_market_params() -> dict:
    return _MARKET_PARAMS_1327 if _ver_tuple(engine_version()) >= (1, 32, 7) \
        else MARKET_PARAMS


def _shape(f: str, x: float, t: float = 0.0) -> float:
    if x < 0.0:
        x = 0.0
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "log":
        return math.log(1.0 + x)
    if f == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(f)


def quote(item: str, inventory: float) -> int:
    """The engine's quoted price at a given market inventory.

    Mirrors `_market_price`: banker's rounding (CPython round()), floored
    at 1. Follows the ACTIVE engine version (models/engine_version.json).
    """
    base, i0, t, bf, bt, af, at = active_market_params()[item]
    if inventory < i0:
        amp = bt * base / _shape(bf, t, t)
        raw = base + amp * _shape(bf, i0 - inventory, t)
    else:
        amp = at * base / _shape(af, t, t)
        raw = base - amp * _shape(af, inventory - i0, t)
    r = _round_half_even(raw)
    return PRICE_FLOOR if r < PRICE_FLOOR else int(r)


def _round_half_even(x: float) -> float:
    f = math.floor(x)
    diff = x - f
    if diff > 0.5:
        return f + 1.0
    if diff < 0.5:
        return float(f)
    return float(f) if int(f) % 2 == 0 else f + 1.0


def fib_hire(n_already_today: int) -> int:
    """Cost of the (n+1)-th hire of the day: fib with fib(0)=1, fib(1)=1."""
    a, b = 1, 1
    for _ in range(n_already_today):
        a, b = b, a + b
    return a


def quadrant_of(x: int, y: int) -> str:
    half = BOARD // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles():
    half = BOARD // 2
    return [(half - 1, half - 1), (half, half - 1),
            (half - 1, half), (half, half)]


def town_drain_per_day(unlocked_shops) -> dict:
    """Exact town demand per product per DAY given the observed shop draw.

    Shops fire every TOWN_SHOP_SELL_INTERVAL steps (6x/day at 24 steps/day),
    draining 1 per listed product (2 if single-product). The town center
    fires once per day draining 1 of every non-FERTILIZER product.
    """
    per_day = TURNS_PER_DAY // TOWN_SHOP_SELL_INTERVAL
    drain = {p: 0 for p in PRODUCTS}
    for shop in unlocked_shops:
        prods = SHOP_PRODUCTS.get(shop, [])
        mult = 2 if len(prods) == 1 else 1
        for p in prods:
            drain[p] += mult * per_day
    for p in PRODUCTS:
        if p != "FERTILIZER":
            drain[p] += TURNS_PER_DAY // TOWN_CENTER_SELL_INTERVAL
    return drain


# -------------------------------------------------------------- replay IO --

def load_replay(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def replay_seed(rep: dict) -> int:
    return int(rep.get("info", {}).get("seed",
               rep.get("configuration", {}).get("seed", 0)))


def replay_teams(rep: dict):
    return rep.get("info", {}).get("TeamNames", ["?", "?"])


def replay_actions(rep: dict, seat: int):
    """Per-step action dicts for one seat ({} where the agent errored).

    ALIGNMENT: steps[t].action is the action that PRODUCED state t, so the
    action applied at engine step t lives in steps[t+1]. steps[0] is the
    initial state (PASS placeholders). Getting this wrong shifts every
    watering by one turn and the whole farm weeds over -- measured as final
    banks 0/304 vs the recorded 67836/83257 on the first conversion attempt.
    """
    out = []
    for st in rep["steps"][1:]:
        a = st[seat].get("action")
        out.append(a if isinstance(a, dict) else {})
    return out


def final_banks(rep: dict):
    last = rep["steps"][-1]
    return [float(last[i]["observation"]["farms"][i]["money"])
            for i in range(2)]


# ------------------------------------------------------------------- tapes --

def action_to_tape_lines(action: dict) -> str:
    """One seat's action dict -> one tape line (kagg episode/batch format)."""
    farmer = action.get("farmer") or ["PASS"]
    hands = action.get("hands") or []
    market = action.get("market") or []
    f = " ".join(str(t) for t in farmer)
    # POSITIONAL: an empty / non-list entry is kept as an empty segment. The
    # official interpreter applies hands[i] to hand i and settles the market
    # per order INDEX, so skipping a `[]` slot-holder (the clone field's
    # index-race trick) shifts every later hand action / order pairing.
    # `kagg` parses this with engine::positional (empty field = no entries).
    h = ";".join(" ".join(str(t) for t in a) if isinstance(a, list) else ""
                 for a in hands)
    m = ";".join(" ".join(str(t) for t in o) if isinstance(o, list) else ""
                 for o in market)
    return f"{f}\t{h}\t{m}"


def replay_to_tape(rep: dict, seat: int, out_path: str):
    """Write ONE seat's action stream as a single-seat tape.

    The pairing into a two-seat episode happens in the batch jobs file: a
    tape here is `SEAT` line-per-step; kagg batch zips two of them.
    """
    acts = replay_actions(rep, seat)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {replay_seed(rep)}\n")
        for a in acts:
            fh.write(action_to_tape_lines(a) + "\n")


def write_episode_tape(seed: int, actions0, actions1, out_path: str):
    """Full two-seat tape in the exact `kagg episode` format."""
    n = min(len(actions0), len(actions1))
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {seed}\n")
        for i in range(n):
            fh.write(action_to_tape_lines(actions0[i]) + "\n")
            fh.write(action_to_tape_lines(actions1[i]) + "\n")


def now_stamp() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
