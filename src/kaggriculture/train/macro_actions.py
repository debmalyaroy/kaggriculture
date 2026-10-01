"""E1.2 -- the MACRO (daily-plan) action space for the Slot-2 BC/RL policy.

A full game is 720 turns, but a *strategy* is only ~30 daily decisions: how
much land to buy, how many hands to hire, what seed mix to plant, which
animals to buy, what to build, and how much of each product to sell. The
runtime micro-ops (move/water/harvest) are executors of that plan, not the
plan itself. The macro policy learns the plan; a scripted/searched executor
carries it out.

This module is the schema for ONE day's plan:

  * ``MacroAction`` -- a structured dataclass (seed mix, buys, hires, builds,
    sell vector) with ``to_vector`` / ``from_vector`` for a learner, and
    ``primary_class`` -- the single categorical label used for class-balance
    reporting and for the E4.1 dispatcher target.
  * ``day_macro(day_cells)`` -- distil the 24 turn-cells of one game-day for
    one seat into a MacroAction by aggregating that day's market orders and
    field ops. This is what ``bc_corpus`` calls.

The VOCABULARY (seed/animal/build/product names, hire+sell quantisation
buckets) is FIXED by the game rules; the empirical frequencies of each class
are what the corpus supplies (see ``bc_corpus --limit 800`` report).

    python -m kaggriculture.train.macro_actions          # self-test + schema
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# --- fixed game vocabulary (verified against the top-100 replay corpus) ------
SEEDS = ("CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT")           # plantable
PRODUCTS = ("CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
            "STRAWBERRY", "TOMATO", "WHEAT", "WOOL")                    # sellable
ANIMALS = ("COW", "GOOSE", "SHEEP")
BUILDS = ("BUILD_PASTURE", "BUILD_COOP")
FIELD_BUILD_VERBS = set(BUILDS)

# The single categorical label per day, in PRIORITY order (a day that both
# buys land and hires is labelled by its most strategically committing move).
MACRO_CLASSES = (
    "BUY_LAND",     # bought a land tier this day
    "BUY_ANIMAL",   # bought livestock
    "BUILD",        # built a pasture / coop
    "PLANT_EXPAND", # bought seed and/or planted crops (crop-expansion day)
    "HIRE",         # hired labour (no bigger commitment above)
    "SELL",         # a selling day, no meaningful buy-side commitment
    "IDLE",         # no economic action
)
CLASS_INDEX = {c: i for i, c in enumerate(MACRO_CLASSES)}

# Quantisation buckets keep the vector small and the class-balance readable.
HIRE_BUCKETS = (0, 1, 2, 4, 8)        # >=  -> bucket index
SELL_BUCKETS = (0, 1, 10, 50, 200)    # per-product total units sold that day


def _bucket(x: float, edges) -> int:
    b = 0
    for i, e in enumerate(edges):
        if x >= e:
            b = i
    return b


@dataclass
class MacroAction:
    """One seat's economic plan for one game-day."""
    buy_seed: Dict[str, int] = field(default_factory=dict)   # seed -> units
    plant: Dict[str, int] = field(default_factory=dict)      # seed -> tiles planted
    buy_animal: Dict[str, int] = field(default_factory=dict)  # animal -> head
    buy_land: int = 0                                        # BUY_LAND count
    hire: int = 0                                            # HIRE count
    build: Dict[str, int] = field(default_factory=dict)     # BUILD_* -> count
    sell: Dict[str, int] = field(default_factory=dict)      # product -> units

    # ---- categorical label -------------------------------------------------
    def primary_class(self) -> str:
        if self.buy_land > 0:
            return "BUY_LAND"
        if sum(self.buy_animal.values()) > 0:
            return "BUY_ANIMAL"
        if sum(self.build.values()) > 0:
            return "BUILD"
        if sum(self.buy_seed.values()) > 0 or sum(self.plant.values()) > 0:
            return "PLANT_EXPAND"
        if self.hire > 0:
            return "HIRE"
        if sum(self.sell.values()) > 0:
            return "SELL"
        return "IDLE"

    def class_id(self) -> int:
        return CLASS_INDEX[self.primary_class()]

    # ---- numeric encoding for a learner ------------------------------------
    def to_vector(self) -> List[float]:
        """Fixed-length numeric encoding (see ``FEATURE_NAMES``)."""
        v: List[float] = []
        v += [float(self.buy_seed.get(s, 0)) for s in SEEDS]
        v += [float(self.plant.get(s, 0)) for s in SEEDS]
        v += [float(self.buy_animal.get(a, 0)) for a in ANIMALS]
        v.append(float(self.buy_land))
        v.append(float(_bucket(self.hire, HIRE_BUCKETS)))
        v += [float(self.build.get(b, 0)) for b in BUILDS]
        v += [float(_bucket(self.sell.get(p, 0), SELL_BUCKETS)) for p in PRODUCTS]
        return v

    @classmethod
    def from_vector(cls, v: List[float]) -> "MacroAction":
        it = iter(v)

        def take(n):
            return [next(it) for _ in range(n)]
        bs = take(len(SEEDS)); pl = take(len(SEEDS)); ba = take(len(ANIMALS))
        land = next(it); hire_b = next(it); bd = take(len(BUILDS))
        sl = take(len(PRODUCTS))
        return cls(
            buy_seed={s: int(bs[i]) for i, s in enumerate(SEEDS) if bs[i]},
            plant={s: int(pl[i]) for i, s in enumerate(SEEDS) if pl[i]},
            buy_animal={a: int(ba[i]) for i, a in enumerate(ANIMALS) if ba[i]},
            buy_land=int(land),
            hire=int(HIRE_BUCKETS[min(int(hire_b), len(HIRE_BUCKETS) - 1)]),
            build={b: int(bd[i]) for i, b in enumerate(BUILDS) if bd[i]},
            sell={p: int(SELL_BUCKETS[min(int(sl[i]), len(SELL_BUCKETS) - 1)])
                  for i, p in enumerate(PRODUCTS) if sl[i]},
        )


FEATURE_NAMES = (
    [f"buyseed_{s}" for s in SEEDS]
    + [f"plant_{s}" for s in SEEDS]
    + [f"buyanimal_{a}" for a in ANIMALS]
    + ["buy_land", "hire_bucket"]
    + [f"build_{b}" for b in BUILDS]
    + [f"sellb_{p}" for p in PRODUCTS]
)
VECTOR_LEN = len(FEATURE_NAMES)


def _iter_units(cell):
    """farmer op + each hand op for one turn-cell's action."""
    act = cell.get("action") or {}
    fm = act.get("farmer")
    if isinstance(fm, list) and fm:
        yield fm
    for h in (act.get("hands") or []):
        if isinstance(h, list) and h:
            yield h


def day_macro(day_cells) -> MacroAction:
    """Aggregate one seat's 24 turn-cells of a game-day into a MacroAction.

    ``day_cells`` is the list of that seat's per-turn cells (each with
    ``observation`` and ``action``) for one day. Robust to short/partial days
    and malformed orders.
    """
    ma = MacroAction()
    for cell in day_cells:
        act = cell.get("action") or {}
        for o in (act.get("market") or []):
            if not (isinstance(o, list) and o):
                continue
            verb = o[0]
            if verb == "BUY_LAND":
                ma.buy_land += 1
            elif verb == "HIRE":
                ma.hire += 1
            elif verb == "BUY_SEED" and len(o) >= 3 and o[1] in SEEDS:
                ma.buy_seed[o[1]] = ma.buy_seed.get(o[1], 0) + _int(o[2])
            elif verb == "BUY_ANIMAL" and len(o) >= 3 and o[1] in ANIMALS:
                ma.buy_animal[o[1]] = ma.buy_animal.get(o[1], 0) + _int(o[2])
            elif verb == "SELL" and len(o) >= 3 and o[1] in PRODUCTS:
                ma.sell[o[1]] = ma.sell.get(o[1], 0) + _int(o[2])
        for u in _iter_units(cell):
            verb = u[0]
            if verb == "PLANT" and len(u) >= 2 and u[1] in SEEDS:
                ma.plant[u[1]] = ma.plant.get(u[1], 0) + 1
            elif verb in FIELD_BUILD_VERBS:
                ma.build[verb] = ma.build.get(verb, 0) + 1
    return ma


def _int(x) -> int:
    try:
        return max(0, int(x))
    except (TypeError, ValueError):
        return 0


# --- P1: plan guardrails (coherent + bootstrap-viable) ----------------------
LAND_PRICES = (1000, 2000, 4000)             # engine LAND_PRICES
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
FAST_CROPS = ("CARROT", "WHEAT")             # early-income crops (first_yield day 2)


def sanitize_plan(plan, m0: float = 3000.0, reserve: float = 250.0):
    """Make a plan coherent + bootstrap-viable WITHOUT overriding intent where
    affordable. Fixes the degeneracies that make the untrained policy bank 0:

      * day-0 spend can't exceed the $3000 bootstrap (land is $1000/$2000/$4000
        — buying 3 tiers from $3000 is impossible; clamp to what cash affords);
      * a plan with only slow crops (MELON day-10) and no early-income crop
        starves the bootstrap — guarantee at least a little CARROT early;
      * animals are deferred off day 0 (they need pasture+feed first) and
        right-sized to the herd the early economy can bootstrap.

    Mutates and returns ``plan`` (list of MacroAction). Idempotent.
    """
    if not plan:
        return plan
    # 1. clamp cumulative land purchases to the cash schedule (pessimistic:
    #    ignore income, so we never authorise an unaffordable early tier).
    cash = m0
    quads = 0
    for day, ma in enumerate(plan):
        # small daily income once past the first fast-crop yield, so mid-game
        # land isn't over-clamped (rough; the engine is the real judge).
        if day >= 4:
            cash += 1500
        while ma.buy_land > 0 and quads < len(LAND_PRICES):
            price = LAND_PRICES[quads]
            if cash - price < reserve:
                ma.buy_land -= 1                 # can't afford this tier yet
            else:
                cash -= price
                quads += 1
                ma.buy_land -= 1
            if quads >= len(LAND_PRICES):
                ma.buy_land = 0
        ma.buy_land = 0 if quads >= len(LAND_PRICES) else ma.buy_land

    # 2. animals off day 0 (need pasture+feed); fold day-0 animal buys into day 2
    d0 = plan[0]
    if sum(d0.buy_animal.values()) > 0 and len(plan) > 2:
        for a, n in list(d0.buy_animal.items()):
            plan[2].buy_animal[a] = plan[2].buy_animal.get(a, 0) + n
        d0.buy_animal = {}
    # CO2: the ~100k economy is a WHEAT-FED HERD (milk $264). Cap the herd only at a
    # bootstrap-sane MAX (not the old tiny 6) — a big herd is the point — and GUARANTEE
    # wheat feed scales with it (each animal eats 1 WHEAT/day; grow ~2 wheat tiles/head).
    total_an = sum(sum(ma.buy_animal.values()) for ma in plan)
    HERD_MAX = 12
    if total_an > HERD_MAX:
        budget = HERD_MAX
        for ma in plan:
            for a in list(ma.buy_animal):
                take = min(ma.buy_animal[a], budget)
                ma.buy_animal[a] = take
                budget -= take
                if budget <= 0:
                    ma.buy_animal[a] = 0
        total_an = HERD_MAX
    # scale WHEAT feed to the herd (a wheat-light herd starves = net-negative)
    if total_an > 0:
        wheat_planted = sum(ma.plant.get("WHEAT", 0) for ma in plan)
        need = max(2 * total_an, total_an)         # ~2 wheat tiles/head
        if wheat_planted < need:
            add = need - wheat_planted
            plan[0].plant["WHEAT"] = plan[0].plant.get("WHEAT", 0) + add
            plan[0].buy_seed["WHEAT"] = plan[0].buy_seed.get("WHEAT", 0) + add

    # 3. guarantee an early-income crop in the opening (days 0-3)
    early_fast = any((ma.plant.get(c, 0) for ma in plan[:4] for c in FAST_CROPS))
    if not early_fast:
        d0.plant["CARROT"] = max(d0.plant.get("CARROT", 0), 6)
        d0.buy_seed["CARROT"] = max(d0.buy_seed.get("CARROT", 0),
                                    d0.plant["CARROT"])
    return plan


def main():
    print(f"MACRO action schema: {VECTOR_LEN}-dim vector, "
          f"{len(MACRO_CLASSES)} primary classes")
    print("classes:", ", ".join(MACRO_CLASSES))
    print("vector fields:", ", ".join(FEATURE_NAMES))
    # round-trip self-test
    demo_cells = [
        {"action": {"market": [["BUY_LAND"], ["HIRE"], ["HIRE"],
                               ["BUY_SEED", "MELON", 7], ["SELL", "WOOL", 40]],
                    "farmer": ["PLANT", "MELON"], "hands": [["PLANT", "MELON"]]}}
    ]
    ma = day_macro(demo_cells)
    v = ma.to_vector()
    rt = MacroAction.from_vector(v)
    assert ma.primary_class() == "BUY_LAND", ma.primary_class()
    assert len(v) == VECTOR_LEN
    assert rt.buy_land == 1 and rt.plant.get("MELON") == 2
    print("\ndemo day ->", ma.primary_class(), "| vector len", len(v))
    print("round-trip ok:", rt.buy_land, rt.plant, rt.sell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
