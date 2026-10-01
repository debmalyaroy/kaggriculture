"""E1.3 -- the MICRO option space and a LEGALITY MASK, Python reference.

The micro policy chooses per-turn ops. An op is a ``(verb, target, qty)``
triple. Two disjoint op streams exist in the engine:

  * MARKET orders  -- BUY_LAND / HIRE / BUY_SEED / BUY_PRODUCT / BUY_ANIMAL /
    SELL. At most ``maxMarketOrdersPerTurn`` (10) per turn, priority-ordered;
    extras are dropped silently. These are the ones with a clean, cheap
    legality test (cash / shed / seeds), so this module masks them exactly.
  * FIELD ops (farmer + each hand) -- movement, WATER, PLANT, HARVEST, FEED,
    CARE, PICKUP, DROP, PLACE, DIG, COLLECT_FERTILIZER, FERTILIZE, BUILD_*.
    Their legality is BOARD-STATE dependent (tile occupancy, adjacency,
    labour). This module enumerates them and applies the cheap invariants
    (a seat has seed to PLANT, a tile budget) but full board legality needs
    the engine -- see the note on E1.1 below.

E1.1 (LATER): a byte-identical Rust encoder/mask. This file is the reference
oracle it must match action-for-action; ``rustengine/src/policy.rs`` is where
that hook lands (see ``tests/test_compiled_agent.py`` for the equivalence
pattern already used for the compiled agent). Nothing here imports the engine
so it stays a pure, testable spec.

    python -m kaggriculture.train.micro_options            # schema + self-test
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SEEDS = ("CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT")
PRODUCTS = ("CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
            "STRAWBERRY", "TOMATO", "WHEAT", "WOOL")
ANIMALS = ("COW", "GOOSE", "SHEEP")

MAX_MARKET_ORDERS = 10

# ---- verb catalogue --------------------------------------------------------
MOVE_VERBS = ("NORTH", "SOUTH", "EAST", "WEST")
# field ops with no target argument
FIELD_NOARG = MOVE_VERBS + (
    "WATER", "HARVEST", "FEED", "CARE", "PICKUP", "DROP", "DIG",
    "COLLECT_FERTILIZER", "FERTILIZE", "BUILD_PASTURE", "BUILD_COOP", "PASS")
# field ops that take a target (PLANT a seed, PLACE a product)
FIELD_TARGETED = {"PLANT": SEEDS, "PLACE": PRODUCTS}

# market verbs and their target vocabularies (None = no target)
MARKET_TARGETED = {
    "BUY_SEED": SEEDS,
    "BUY_PRODUCT": PRODUCTS,
    "BUY_ANIMAL": ANIMALS,
    "SELL": PRODUCTS,
}
MARKET_NOARG = ("BUY_LAND", "HIRE")

# ---- canonical option enumeration (verb, target|None) ----------------------
Option = Tuple[str, Optional[str]]


def _build_options() -> List[Option]:
    opts: List[Option] = []
    for v in FIELD_NOARG:
        opts.append((v, None))
    for v, targets in FIELD_TARGETED.items():
        for t in targets:
            opts.append((v, t))
    for v in MARKET_NOARG:
        opts.append((v, None))
    for v, targets in MARKET_TARGETED.items():
        for t in targets:
            opts.append((v, t))
    return opts


OPTIONS: List[Option] = _build_options()
OPTION_INDEX: Dict[Option, int] = {o: i for i, o in enumerate(OPTIONS)}
N_OPTIONS = len(OPTIONS)

MARKET_VERBS = set(MARKET_NOARG) | set(MARKET_TARGETED)


def encode(op) -> Optional[int]:
    """Map a raw engine op list (``["SELL","WOOL",40]``) to an option id.

    Quantity is NOT part of the option id (the id is verb+target); the qty is
    a separate scalar head. Returns None for an unrecognised op.
    """
    if not (isinstance(op, (list, tuple)) and op):
        return None
    verb = op[0]
    target = op[1] if len(op) >= 2 and isinstance(op[1], str) else None
    if (verb, target) in OPTION_INDEX:
        return OPTION_INDEX[(verb, target)]
    # verbs that carry a target vocab but this op omitted/!matched it
    if (verb, None) in OPTION_INDEX:
        return OPTION_INDEX[(verb, None)]
    return None


def decode(idx: int) -> Option:
    return OPTIONS[idx]


def _price(obs, item) -> float:
    return float(((obs.get("market") or {}).get("prices") or {}).get(item, 0) or 0)


def market_legal_mask(obs: dict, seat: int,
                      orders_already: int = 0) -> List[bool]:
    """Boolean legality over the MARKET options, in ``OPTIONS`` order.

    Checks the cheap, engine-independent invariants:
      * order budget: nothing is legal once ``maxMarketOrdersPerTurn`` reached
      * cash: every BUY_* needs money > 0 (a BUY of qty>=1 costs > 0)
      * shed: SELL <item> legal only if the shed holds >=1 of <item>
        (SELL draws from the shed, never from unit inventories -- gotcha)
      * BUY_SEED is always cash-gated only (seeds have no shed precondition)

    Field ops are marked True (this mask governs the market queue; field-op
    legality is board-state dependent -- see module docstring). Quantity
    feasibility (how MANY you can afford / hold) is a separate clamp, not a
    legality bit, because SELL partially fills per unit and an oversized BUY
    is clamped by the engine, not rejected.
    """
    farms = obs.get("farms") or []
    money = 0.0
    if seat < len(farms):
        money = float((farms[seat] or {}).get("money", 0) or 0)
    shed = ((obs.get("private") or {}).get("shed") or {})
    budget_left = orders_already < MAX_MARKET_ORDERS

    mask = [True] * N_OPTIONS
    for i, (verb, target) in enumerate(OPTIONS):
        if verb not in MARKET_VERBS:
            continue                                   # field op, not our job
        if not budget_left:
            mask[i] = False
            continue
        if verb == "SELL":
            mask[i] = int(shed.get(target, 0) or 0) >= 1
        elif verb in ("BUY_LAND", "HIRE", "BUY_SEED",
                      "BUY_PRODUCT", "BUY_ANIMAL"):
            mask[i] = money > 0.0
        else:
            mask[i] = True
    return mask


def field_op_prereqs(obs: dict, seat: int) -> Dict[str, bool]:
    """Cheap engine-independent preconditions for the TARGETED field ops.

    Full board legality (tile occupancy, adjacency, labour capacity) needs the
    engine; these are the invariants a mask can assert without it:
      * PLANT <seed> requires the seat to HOLD that seed (private.seeds > 0)
      * PLACE <product> requires the shed/inventory to hold that product
    Returns ``{f"{verb}:{target}": bool}`` for the targeted field ops.
    """
    priv = obs.get("private") or {}
    seeds = priv.get("seeds") or {}
    shed = priv.get("shed") or {}
    out: Dict[str, bool] = {}
    for s in SEEDS:
        out[f"PLANT:{s}"] = int(seeds.get(s, 0) or 0) >= 1
    for p in PRODUCTS:
        out[f"PLACE:{p}"] = int(shed.get(p, 0) or 0) >= 1
    return out


def main():
    print(f"MICRO option space: {N_OPTIONS} options "
          f"({len(FIELD_NOARG)} no-arg field + targeted field + market)")
    # self-test on the day-1 opener seen in the corpus
    obs = {
        "farms": [{"money": 403.0}, {"money": 3000.0}],
        "private": {"shed": {"WOOL": 12, "CARROT": 0},
                    "seeds": {"MELON": 7, "WHEAT": 0}},
        "market": {"prices": {"WOOL": 200, "CARROT": 35}},
    }
    assert encode(["SELL", "WOOL", 40]) == OPTION_INDEX[("SELL", "WOOL")]
    assert encode(["BUY_LAND"]) == OPTION_INDEX[("BUY_LAND", None)]
    assert encode(["NORTH"]) == OPTION_INDEX[("NORTH", None)]
    m = market_legal_mask(obs, 0)
    assert m[OPTION_INDEX[("SELL", "WOOL")]] is True    # shed has wool
    assert m[OPTION_INDEX[("SELL", "CARROT")]] is False  # shed empty
    assert m[OPTION_INDEX[("BUY_LAND", None)]] is True   # money > 0
    full = market_legal_mask(obs, 0, orders_already=10)
    assert full[OPTION_INDEX[("SELL", "WOOL")]] is False  # budget spent
    pre = field_op_prereqs(obs, 0)
    assert pre["PLANT:MELON"] is True and pre["PLANT:WHEAT"] is False
    n_market_legal = sum(1 for i, o in enumerate(OPTIONS)
                         if o[0] in MARKET_VERBS and m[i])
    print(f"self-test ok. day-1 opener: {n_market_legal} market options legal")
    print("note: E1.1 byte-identical Rust encoder is a LATER hook "
          "(rustengine/src/policy.rs); this is its reference oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
