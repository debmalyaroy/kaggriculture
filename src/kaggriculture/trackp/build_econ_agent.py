"""Build the CLOSED-LOOP skeleton planner for the Track-P seat.

v3 template (2026-09-03, the skeleton build). The seat law is absolute:
every decision is derived from the live observation plus a parameter
genome -- no tape, no route prefix, no router, no embedded action
sequence, not even as a fallback.

What changed from v2: the planner no longer searches a free-form static
tile allocation. It EXECUTES the measured top-10 field skeleton (641
seat records + 300 traces, engine 1.32.7) expressed as day-indexed
targets, re-derived from the real board every dawn:

  * 2 BUY_LAND -- 2nd quadrant usable day 5, 3rd day 12
  * herd ramp gated on cash: cows 1@d0 -> 6 by d9 -> 9; sheep 1@d0 -> 5;
    zero geese. An animal bought today is PLACED (and fed, and cared
    for) tomorrow, because the purchase clears after the unit turn.
  * hire ramp to 10 hands, all hired in the hour-0 market batch so the
    spawn tiles are deterministic
  * per-crop TILE targets with planting windows: melon 12 (d0-2),
    strawberry 38 (from the 2nd quadrant, through d13), wheat everywhere
    else on its 4-day replant cycle, nothing planted past its maturity
    horizon
  * fertiliser -> STRAWBERRY only, on the two production-eve ages
  * seed bought only for the plantings LABOUR HAS ALREADY BEEN ASSIGNED
    this turn (held seed ~0)
  * milk / wool / fertiliser sold on production, wheat sold continuously
    above the feed reserve, melon capped, everything liquidated by 718

The genome parameterises the targets; the DEFAULTS are the skeleton.

    python src/trackp/build_econ_agent.py --genome skeleton \
        --out .local/candidates/trackp_skel.py
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT


# --------------------------------------------------------------- skeleton
GENOME_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genome.json")


def skeleton_genome():
    """The measured top-10 economy as day-indexed targets (the DEFAULT).

    G2 (2026-09-18): the genome is EXTERNALIZED to genome.json, the single
    source of truth shared with the Rust policy (rustengine/src/policy.rs loads
    the same file at runtime). This function loads that file; the inline dict
    below is the byte-identical fallback used only when the file is missing, so
    editing genome.json changes BOTH the Python planner and the compiled agent.
    """
    try:
        with open(GENOME_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    return _skeleton_genome_default()


def _skeleton_genome_default():
    return {
        # 2 BUY_LAND, AS EARLY AS CASH ALLOWS. The scheduled day is only a
        # lower bound -- the buy is cash-gated, so a "day 4" schedule with an
        # empty purse actually bought NE on day 9 and SW on day 14, leaving
        # the farm on NW's 21 usable tiles for a third of the season with ten
        # hands PASSing 150-180 unit-turns a day.
        #
        # Measured 2026-09-03 on the official engine, 5 gauntlet opponents x
        # both seats, 40 cells per row (dev seeds 11-14):
        #   NE 4  / SW 11  (the old schedule)          58,426 median own bank
        #   NE 0  / SW 8                               73,656
        #   NE 0  / SW 5   (this)                      77,998
        #   NE 0  / SW 6                               77,998  (cash-gated to
        #                                              the same real day)
        #   NE 0  / SW 10                              70,120
        #   NE 0  / SW 8 / SE 12 (a third buy)         61,294  -- SE costs
        #                                              4,000 and never repays
        # Pooled over 80 cells (seeds 3,4,5,6,11,12,13,14): 54,552 -> 70,208.
        # LAND IS THE BINDING CONSTRAINT, not labour and not compute.
        "land": {"NE": 0, "SW": 5, "SE": -1},
        # the crew. Every hire sits in the hour-0 market batch, which holds
        # 10 orders, so 10 is the hard ceiling: more would spawn at an
        # unpredictable shed tile and desynchronise the whole tour. Values
        # above 10 are clamped and are kept only so the vector reads as the
        # mined top-30 ramp it is.
        #
        # This is the CURRENT top-30 ramp, mined from 360 engine-1.32.7 wins
        # of the 30 teams rated 2806-2978 (docs/history/trackp-base-economy-2026-09-03.md).
        # It is LEANER than the old ramp on days 1-7 and that is the whole
        # point: an idle hand still costs its fibonacci fee, and the old ramp
        # hired ten hands into a one-quadrant farm. Worth +2.4k median own
        # bank on its own, and it is what makes the day-0 land buy affordable.
        "hires": [5, 4, 4, 5, 4, 5, 8, 8, 10, 11,
                  11, 11, 9, 10, 10, 12, 12, 12, 12, 12,
                  12, 12, 12, 12, 12, 11, 11, 11, 11, 11],
        # cumulative herd cap by day; every purchase is cash-gated. One day
        # earlier than the measured placement day to absorb the buy->place
        # lag.
        "herd": {"COW": [[0, 1], [2, 2], [3, 3], [4, 4], [5, 5], [7, 6],
                         [10, 7], [13, 8], [16, 9]],
                 "SHEEP": [[0, 1], [6, 2], [9, 3], [12, 4], [15, 5]],
                 "GOOSE": []},
        # standing-tile targets and the windows they may be planted in.
        "tiles": {"MELON": 12, "STRAWBERRY": 38, "CARROT": 0},
        "window": {"MELON": [4, 12], "STRAWBERRY": [5, 14],
                   "CARROT": [0, 24], "WHEAT": [0, 25]},
        # cash the planner refuses to spend past, per class of purchase.
        "floor_seed": 20,
        "floor_straw": 700,
        "floor_land": 500,
        # Days AHEAD of a scheduled BUY_LAND that its cost is withheld from
        # the herd (0 = from the scheduled day until it is bought; -1 = off).
        # Measured 2026-09-03 over 24 official-engine cells, 6 seeds x 2 seats
        # x {v43.0_bandit, pub_v16rc5}: off 48.4k median / 47.8k mean,
        # lead 0 55.0k / 52.1k, lead 1 48.5k / 50.3k, lead 2+ 43.4k mean.
        # Without it the herd ramp (a cow every day from d2) spent the land
        # money and the 2nd quadrant arrived on day 10-11 instead of day 5,
        # capping the farm at NW's 24 usable tiles for a third of the season
        # -- 18 standing crops against the field's 243 plants. Reserving too
        # far ahead is worse than not reserving: SW costs 2,000 and holding
        # that from day 5 starves the herd for six days.
        # This is not a new strategy; it is the documented skeleton executing.
        "land_reserve_lead": 0,
        "floor_animal": 300,
        # -1 = hire the `hires` ramp blindly (the shipped behaviour). >= 0
        # treats the ramp as a CEILING and hires only the hands the day's
        # paced task list can feed, plus this many spares.
        "hire_slack": -1,
        "pen_lookahead": 5,
        "feed_buffer": 6,
        "fert_crop": "STRAWBERRY",
        "fert_buy": 0,          # the herd already drops ~14 fertiliser/day
        "care": True,
        # daily sell caps for the price-impact-sensitive products; WHEAT
        # and EGG are log-priced (dump-proof) and FERTILIZER is gentle.
        "sell_cap": {"MELON": 24, "WOOL": 40, "STRAWBERRY": 60, "MILK": 40,
                     "EGG": 60, "CARROT": 60, "TOMATO": 60,
                     "FERTILIZER": 60},
        "fert_keep": 12,        # shed fertiliser held back for tomorrow
        "sell_floor": 0.0,      # fraction of base below which we hold
        "plant_pace": 0.95,     # share of crew ops new planting may claim
        # Shed pressure valve. The shed caps at 100 TOTAL and dusk overflow is
        # DISCARDED, so once the dawn total passes valve_hi the planner dumps
        # impact-cheap stock down to valve_lo.
        "valve_hi": 55,
        "valve_lo": 45,
        # ---- market contest (2026-09-04) --------------------------------
        # All three read the LIVE observation only.  All three default OFF,
        # so the shipped skeleton is byte-for-byte what it was.
        #
        # sell_order: 1 = put the biggest-dollar SELL in the earliest market
        # slot.  Slot index is a hard price ladder inside `_process_market`.
        "sell_order": 0,
        # sell_prio: the order used by sell_order mode 2.
        "sell_prio": ["MELON", "FERTILIZER", "STRAWBERRY", "MILK", "WOOL",
                      "CARROT", "TOMATO", "EGG", "WHEAT"],
        # hour0_sells: market slots reserved at hour 0 for SELL orders, paid
        # for by hiring that many fewer hands.
        "hour0_sells": 0,
        # drop_daily: 1 = a loaded unit walks back to the shed and DROPs at
        # the end of its tour, so today's harvest can be sold today.
        "drop_daily": 0,
        # hold_price: absolute $ per unit below which a product is NOT sold
        # (released under valve pressure and in the liquidation window).
        "hold_price": {},
        # demand_crops: tile target for a product sized from the town's REAL
        # consumption rate, read off `town.unlocked_shops`.  `per_tile` is
        # units produced per standing tile per day.
        "demand_crops": {},
        # ---- herd_gate (2026-09-04) -------------------------------------
        # THE MARGINAL-ANIMAL LEDGER. Measured over 24 official-engine cells
        # (12 seeds x both seats vs v43.0_bandit), 16 sheep against the
        # 5-sheep skeleton: +12.88 sheep bought, +28.1 WOOL units SOLD, and
        # the wool line went DOWN $2,238 -- $12,767 (77u @ $165.8) became
        # $10,510 (105u @ $100.0). The marginal wool unit is worth MINUS $80.
        #
        # The cause is an engine fact, not an executor one. WOOL's above-I0
        # branch is QUADRATIC (target 3.2 on T=105), so the price hits the $1
        # floor only 59 units above I0, and the town eats wool ONLY if the
        # shop draw gave a YARN_STORE (12/day) -- the town centre alone eats
        # 1/day. Measured over 16 seeds, the shared wool inventory pins at
        # I0+59 from day 20 in 6 of them and our realised price is $35-76;
        # in the rest it stays below I0 and we get $205-242. Our EXISTING
        # 5-sheep herd is already the whole market in the bad half.
        #
        # herd_gate caps a species at `floor` unless the town's live
        # consumption rate for its product (from `town.unlocked_shops`, which
        # is in the observation) and the live price both clear a bar:
        #   {"SHEEP": {"min_rate": 6, "min_price": 0.9, "floor": 2}}
        # DEFAULT OFF (empty) -- the shipped skeleton is unchanged.
        "herd_gate": {},
    }


TEMPLATE = '''"""Kaggriculture Track-P closed-loop skeleton planner -- {label}

Every action is derived from the live observation plus the parameter
genome below. No tape, no route prefix, no router, no embedded action
sequence. The genome states the measured field skeleton as day-indexed
targets; the planner re-derives the whole day from the real board each
dawn and refreshes market orders every hour.
Built {built} by src/trackp/build_econ_agent.py. Lane: trackp-economy."""
import math                                                      # noqa: F401

GENOME = {genome!r}

TPD = 24
BOARD = 10
LAST_DAY = 29
MAX_HIRE = 10               # the hour-0 market batch holds 10 orders
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
LAND_ORDER = ("NE", "SW", "SE")
LAND_COST = (1000, 2000, 4000)
BASE_PRICE = {{"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
              "FERTILIZER": 100}}
SELLABLE = ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG", "CARROT",
            "TOMATO", "FERTILIZER", "WHEAT")
CROPS = {{
    "WHEAT":      {{"seed": 10, "first": 2, "maxd": 4, "iv": 0, "maxy": 6,
                   "ong": 0}},
    "CARROT":     {{"seed": 20, "first": 2, "maxd": 3, "iv": 0, "maxy": 4,
                   "ong": 0}},
    "TOMATO":     {{"seed": 50, "first": 8, "maxd": 8, "iv": 1, "maxy": 4,
                   "ong": 1}},
    "STRAWBERRY": {{"seed": 100, "first": 10, "maxd": 10, "iv": 2,
                   "maxy": 4, "ong": 1}},
    "MELON":      {{"seed": 80, "first": 10, "maxd": 12, "iv": 0, "maxy": 6,
                   "ong": 0}},
}}
ANIMALS = {{
    "GOOSE": {{"cost": 300, "build": "BUILD_COOP", "pen": "COOP"}},
    "COW":   {{"cost": 400, "build": "BUILD_PASTURE", "pen": "PASTURE"}},
    "SHEEP": {{"cost": 500, "build": "BUILD_PASTURE", "pen": "PASTURE"}},
}}
ANIMAL_PRODUCT = {{"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}}
PLANT_ORDER = ("MELON", "STRAWBERRY", "CARROT", "TOMATO")
# What each shop instance consumes. The town centre eats 1 of every product
# except FERTILIZER every 24 steps; each unlocked shop instance eats 1 of
# each listed product every 4 steps (x2 for a single-product shop). Both are
# ENGINE CONSTANTS, and `town.unlocked_shops` is in the live observation, so
# the day's real demand rate per product is computable at dawn.
SHOP_ITEMS = {{
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}}


def _demand(town):
    """Units/day the town will consume of each product, from the live
    unlocked-shop list. Exact, not estimated."""
    rate = {{}}
    for s in (_get(town, "unlocked_shops", []) or []):
        lst = SHOP_ITEMS.get(s)
        if not lst:
            continue
        m = 12 if len(lst) == 1 else 6
        for it in lst:
            rate[it] = rate.get(it, 0) + m
    for it in BASE_PRICE:
        if it != "FERTILIZER":
            rate[it] = rate.get(it, 0) + 1
    return rate


def _get(o, k, d=None):
    try:
        v = o[k]
        return d if v is None else v
    except (KeyError, TypeError, IndexError):
        return getattr(o, k, d)


def _quad(t):
    return ("NE" if t[0] >= 5 else "NW") if t[1] < 5 else \\
           ("SE" if t[0] >= 5 else "SW")


def _shed_dist(t):
    return min(abs(t[0] - s[0]) + abs(t[1] - s[1]) for s in SHED)


def _tile_order():
    """Boustrophedon per quadrant, each starting at its shed corner: a
    contiguous slice of this order is a cheap walking tour."""
    out = []
    for xs, ys in (((4, 3, 2, 1, 0), (4, 3, 2, 1, 0)),
                   ((5, 6, 7, 8, 9), (4, 3, 2, 1, 0)),
                   ((4, 3, 2, 1, 0), (5, 6, 7, 8, 9)),
                   ((5, 6, 7, 8, 9), (5, 6, 7, 8, 9))):
        for i, y in enumerate(ys):
            row = list(xs) if i % 2 == 0 else list(reversed(xs))
            for x in row:
                if (x, y) not in SHED:
                    out.append((x, y))
    return out


TILE_ORDER = _tile_order()
TIDX = dict((t, i) for i, t in enumerate(TILE_ORDER))
_D = {{"day": -1, "plan": None}}


def _sched(pairs, d):
    """Cumulative day-indexed cap: the last [day, n] whose day <= d."""
    n = 0
    for pair in (pairs or []):
        try:
            if d >= int(pair[0]):
                n = int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
    return n


class _U:
    __slots__ = ("pos", "ops", "cap", "start", "loaded")

    def __init__(self, pos, start, cap):
        self.pos = tuple(pos)
        self.ops = []
        self.start = start
        self.cap = cap
        self.loaded = False

    def budget(self):
        return self.cap - len(self.ops)

    def walk(self, t):
        ax, ay = self.pos
        bx, by = t
        while ax < bx and self.budget() > 0:
            self.ops.append(["EAST"])
            ax += 1
        while ax > bx and self.budget() > 0:
            self.ops.append(["WEST"])
            ax -= 1
        while ay < by and self.budget() > 0:
            self.ops.append(["SOUTH"])
            ay += 1
        while ay > by and self.budget() > 0:
            self.ops.append(["NORTH"])
            ay -= 1
        self.pos = (ax, ay)

    def act(self, op):
        if self.budget() <= 0:
            return False
        self.ops.append(op)
        if op[0] in ("HARVEST", "COLLECT_FERTILIZER"):
            self.loaded = True
        return True


def _cost(pos, tile, ops):
    return abs(pos[0] - tile[0]) + abs(pos[1] - tile[1]) + len(ops)


ANIMAL_RATE = {{"COW": 3.0, "SHEEP": 2.0, "GOOSE": 4.0}}  # ~product units/day
DUMP_PROOF = ("EGG", "WHEAT")                              # log-priced, never glut


def _plan(d, farm, priv, mkt, town, opp=None):
    g = GENOME
    money = float(_get(farm, "money", 0) or 0)
    tiles = _get(farm, "tiles", []) or []
    shed = {{}}
    for k, v in (_get(priv, "shed", {{}}) or {{}}).items():
        shed[k] = int(v or 0)
    seeds = {{}}
    for k, v in (_get(priv, "seeds", {{}}) or {{}}).items():
        seeds[k] = int(v or 0)
    owned = set(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    prices = dict(_get(mkt, "prices", {{}}) or {{}})
    sell_all = d >= LAST_DAY - 1
    final = d >= LAST_DAY

    def tl_at(t):
        try:
            return tiles[t[1]][t[0]]
        except (IndexError, TypeError):
            return "LOCKED"

    # ---- census the REAL board -----------------------------------------
    animals, pens, plants, weeds, empties = [], [], [], [], []
    n_an = {{}}
    n_cr = {{}}
    for t in TILE_ORDER:
        if _quad(t) not in owned:
            continue
        tl = tl_at(t)
        if tl == "LOCKED":
            continue
        if tl is None:
            empties.append(t)
            continue
        if not isinstance(tl, dict):
            continue
        if "animal" in tl:
            animals.append((t, tl))
            n_an[tl["animal"]] = n_an.get(tl["animal"], 0) + 1
        elif tl.get("kind") == "PLANT":
            plants.append((t, tl))
            c = tl.get("crop", "WHEAT")
            n_cr[c] = n_cr.get(c, 0) + 1
        elif tl.get("kind") == "WEED":
            weeds.append(t)
        elif tl.get("kind") in ("PASTURE", "COOP"):
            pens.append(t)

    # animals bought on an earlier day are sitting in the shed: THEY are
    # today's placements. Animals bought today land in the shed after the
    # unit turn, so they are placed (and fed, and cared for) tomorrow.
    pen_need = {{}}
    for sp in ("COW", "SHEEP", "GOOSE"):
        pen_need[sp] = int(shed.get(sp, 0))

    late = []            # (rank, op): everything after the hires

    # ---- hires: the whole crew in the hour-0 batch ----------------------
    hires = g.get("hires") or []
    n_plan = int(hires[min(d, len(hires) - 1)]) if hires else 0
    # THE HOUR-0 SLOT. `farm["hands"]` is cleared every night, so the crew is
    # re-hired from scratch each dawn; from day 8 the ramp asks for ten hands
    # and ten HIRE orders fill all ten of hour 0's market slots -- so for two
    # thirds of the season this seat cannot sell on the first turn of a day.
    # `_process_market` walks slot 0 of BOTH seats, reprices, then slot 1,
    # and `_town_consume` only restocks on steps divisible by 4, so a seat
    # selling at hour 1 always sells into the inventory the hour-0 seller
    # just raised. v43.0_bandit puts 84% of its sells in slot 0.
    # `hour0_sells` buys back the earliest slots by hiring that many fewer
    # hands; the marginal hand is also the most expensive (fib(9) = 55/day).
    n0 = max(0, min(MAX_HIRE, int(g.get("hour0_sells", 0) or 0)))
    n_plan = max(0, min(MAX_HIRE - n0, n_plan))
    fib = [1, 1]
    while len(fib) < max(2, n_plan):
        fib.append(fib[-1] + fib[-2])
    n_h = 0
    spend = 0.0
    for i in range(n_plan):
        c = fib[min(i, len(fib) - 1)]
        if money - spend - c < 40:
            break
        spend += c
        n_h += 1
    money -= spend

    # ---- land: 2 buys, on the schedule, cash-gated ----------------------
    # land_res is the cost of the NEXT scheduled quadrant, withheld from the
    # herd until it is bought. Tiles, not animals, are the binding constraint
    # early: NW holds 24 usable tiles and the skeleton wants 243 plants.
    land_res = 0.0
    n_extra = len(owned) - 1
    if n_extra < 3 and not sell_all:
        nq = LAND_ORDER[n_extra]
        ld = (g.get("land") or {{}}).get(nq, -1)
        need_land = LAND_COST[n_extra] + float(g.get("floor_land", 300))
        if isinstance(ld, int) and ld >= 0 and d >= ld and money >= need_land:
            late.append((1, ["BUY_LAND"]))
            money -= LAND_COST[n_extra]
        elif isinstance(ld, int) and ld >= 0 and \\
                d >= ld - int(g.get("land_reserve_lead", 6)):
            land_res = need_land

    herd = g.get("herd") or {{}}

    # ---- what a newly available tile should become ----------------------
    want = dict(n_cr)
    tile_t = dict(g.get("tiles") or {{}})
    windows = g.get("window") or {{}}
    purse = [money]

    # ---- DEMAND CAPTURE -------------------------------------------------
    # CARROT and TOMATO are the two products nobody supplies: the town eats
    # them every day and neither elite routes nor this planner ever plant
    # them, so their inventory runs a season-long deficit and the price runs
    # away (both are HINGE-priced below I0). The tile target is sized from
    # the REAL consumption rate the unlocked-shop list implies, so a seed
    # that never draws PET_CAFE never grows a carrot.
    dem = g.get("demand_crops") or {{}}
    if dem:
        rate = _demand(town)
        for c in sorted(dem):
            spec = dem[c] or {{}}
            mx = int(spec.get("max", 0) or 0)
            if mx <= 0:
                continue
            r = float(rate.get(c, 0))
            if r < float(spec.get("min_rate", 0) or 0):
                continue
            p = float(prices.get(c, BASE_PRICE.get(c, 1)))
            if p < float(spec.get("min_price", 0) or 0) * BASE_PRICE.get(c, 1):
                continue
            per = float(spec.get("per_tile", 0.75) or 0.75)
            tile_t[c] = max(int(tile_t.get(c, 0)),
                            min(mx, int(r / per)))

    def _open(c):
        w = windows.get(c) or [0, 25]
        return int(w[0]) <= d <= int(w[1])

    def _pick():
        """Species for one newly available tile; None to leave it fallow."""
        for c in PLANT_ORDER:
            tgt = int(tile_t.get(c, 0))
            if tgt <= 0 or not _open(c) or want.get(c, 0) >= tgt:
                continue
            if d + CROPS[c]["first"] > LAST_DAY:
                continue
            floor = float(g.get("floor_straw", 250)) \\
                if CROPS[c]["seed"] >= 80 else float(g.get("floor_seed", 40))
            if purse[0] < CROPS[c]["seed"] + floor:
                continue
            purse[0] -= CROPS[c]["seed"]
            want[c] = want.get(c, 0) + 1
            return c
        if _open("WHEAT") and d + CROPS["WHEAT"]["first"] <= LAST_DAY and \\
                purse[0] >= CROPS["WHEAT"]["seed"] + float(
                    g.get("floor_seed", 40)):
            purse[0] -= CROPS["WHEAT"]["seed"]
            want["WHEAT"] = want.get("WHEAT", 0) + 1
            return "WHEAT"
        return None

    # ---- tasks from what STANDS on the board ----------------------------
    # (tile, ops, priority, needs) -- needs are shed withdrawals
    tasks = []
    fert_left = shed.get("FERTILIZER", 0)
    fert_crop = g.get("fert_crop", "STRAWBERRY")
    for t, tl in animals:
        ops = []
        need = {{}}
        if not tl.get("fed_today") and not final:
            ops.append(["FEED"])
            need["WHEAT"] = 1
        if int(_get(tl, "yield_units", 0) or 0) > 0:
            ops.append(["HARVEST"])
        if tl.get("fertilizer_available"):
            ops.append(["COLLECT_FERTILIZER"])
        if g.get("care") and not tl.get("cared_today") and not final:
            ops.append(["CARE"])
        if ops:
            tasks.append((t, ops, 0, need))
    for t, tl in plants:
        crop = tl.get("crop", "WHEAT")
        c = CROPS.get(crop) or CROPS["WHEAT"]
        age = d - int(tl.get("planted_day", d) or 0)
        yld = int(_get(tl, "yield_units", 0) or 0)
        ops = []
        need = {{}}
        if c["ong"]:
            last = c["first"] + c["iv"] * (c["maxy"] - 1)
            if yld > 0 and age >= c["first"]:
                ops.append(["HARVEST"])
            if age < last and not final:
                if crop == fert_crop and fert_left > 0 and \\
                        int(tl.get("fertilized_until_day", -1)) < d and \\
                        age >= c["first"] - 1 and \\
                        (age - (c["first"] - 1)) % max(1, 2 * c["iv"]) == 0:
                    ops.append(["FERTILIZE"])
                    need["FERTILIZER"] = 1
                    fert_left -= 1
                if not tl.get("watered_today"):
                    ops.append(["WATER"])
            elif age >= last and not ops and not sell_all:
                ops.append(["DIG"])
                nxt = _pick()
                if nxt:
                    ops.append(["PLANT", nxt])
                    ops.append(["WATER"])
                    need["SEED_" + nxt] = 1
        else:
            ripe = age >= c["maxd"] or (yld >= c["maxy"] and age >= c["first"])
            if final or sell_all:
                if yld > 0 and age >= c["first"]:
                    ops.append(["HARVEST"])
                elif not final and not tl.get("watered_today"):
                    ops.append(["WATER"])
            elif ripe and yld > 0:
                ops.append(["HARVEST"])
                nxt = _pick()
                if nxt:
                    ops.append(["PLANT", nxt])
                    ops.append(["WATER"])
                    need["SEED_" + nxt] = 1
            elif ripe:
                ops.append(["DIG"])
            elif not tl.get("watered_today"):
                ops.append(["WATER"])
        if ops:
            tasks.append((t, ops, 1, need))
    if not sell_all:
        for t in weeds:
            tasks.append((t, [["DIG"]], 3, {{}}))
        for t in pens:
            tlp = tl_at(t)
            kind = tlp.get("kind") if isinstance(tlp, dict) else None
            for s in ("COW", "SHEEP", "GOOSE"):
                if ANIMALS[s]["pen"] == kind and pen_need.get(s, 0) > 0:
                    pen_need[s] -= 1
                    ops = [["PLACE", s], ["FEED"]]
                    if g.get("care"):
                        ops.append(["CARE"])
                    tasks.append((t, ops, 2, {{"AN_" + s: 1, "WHEAT": 1}}))
                    break
        # Tiles the herd will need SOON are held back from the plough: a
        # day-0 board planted wall to wall left every bought animal
        # rotting in the shed until a crop cycle freed a tile.
        look = int(g.get("pen_lookahead", 5))
        soon = 0
        for s in ("COW", "SHEEP", "GOOSE"):
            soon += _sched(herd.get(s), d + look)
        held = max(0, soon - (len(animals) + len(pens)
                              + sum(pen_need.values())))
        free = sorted(empties, key=lambda t: (_shed_dist(t), TIDX.get(t, 0)))
        for t in free:
            placed = False
            for s in ("COW", "SHEEP", "GOOSE"):
                if pen_need.get(s, 0) > 0:
                    pen_need[s] -= 1
                    ops = [[ANIMALS[s]["build"]], ["PLACE", s], ["FEED"]]
                    if g.get("care"):
                        ops.append(["CARE"])
                    tasks.append((t, ops, 2, {{"AN_" + s: 1, "WHEAT": 1}}))
                    placed = True
                    break
            if placed:
                continue
            if held > 0:
                held -= 1
                continue
            nxt = _pick()
            if nxt:
                tasks.append((t, [["PLANT", nxt], ["WATER"]], 3,
                              {{"SEED_" + nxt: 1}}))

    # ---- herd ramp, from what the CROPS left behind ---------------------
    # Capital priority is crops > animals > feed: the day-ledger showed
    # dawn money pinned at $41 through day 11 when the herd was funded
    # first, and a farm with no seed money never recovers.
    money = purse[0]
    bought = {{}}
    # HERD GATE. An animal is only worth its $400-500 if the town will eat
    # what it makes. WOOL's above-I0 price branch is quadratic and floors at
    # I0+59, and the town eats 12 wool/day per YARN_STORE against 1/day from
    # the town centre -- so in a draw with no YARN_STORE our own 5-sheep herd
    # already pins the price at $1 and the marginal wool unit is worth MINUS
    # $80. Both inputs are in the live observation.
    gate = g.get("herd_gate") or {{}}
    rate_g = _demand(town) if gate else {{}}
    # OPPONENT-ADAPTIVE HERD MIX (2026-09-08). Keep the total herd-size schedule
    # (how many animals the cash/labour supports by day) but ALLOCATE the mix to
    # the animal whose product the town wants and the OPPONENT is NOT already
    # flooding. Producing into an uncrowded product is the whole contested
    # ceiling: a mirror-herd floods the shared market and halves the price,
    # while the same herd pointed at a product the opponent skipped keeps the
    # price up. demand is exact (shops); opponent supply is read from its board.
    # Adapt only from day 6 (shops + the opponent's early herd are revealed;
    # before that there is no signal and the tuned skeleton is best). Between
    # checkpoints the derived mix is stable because the signal is.
    if g.get("adaptive_herd", 0) and d >= 6 and not sell_all:
        demand = _demand(town)
        opp_an = {{}}
        for _row in (_get(opp or {{}}, "tiles", []) or []):
            for _t in (_row or []):
                if isinstance(_t, dict) and _t.get("animal"):
                    opp_an[_t["animal"]] = opp_an.get(_t["animal"], 0) + 1

        _planbuy = {{}}

        def _an_score(sp):
            # Realized $/animal: a herd of rate units/day sells the part the
            # town still wants (headroom) near base price, the oversupplied
            # rest at the floor -- $1 for glut products, ~half-base for the
            # log-priced dump-proof ones (EGG). This picks sheep in YARN
            # worlds, geese in PET_CAFE worlds, and stops at saturation.
            pr = ANIMAL_PRODUCT[sp]
            base_p = float(BASE_PRICE.get(pr, 1))
            rate = ANIMAL_RATE[sp]
            combined = (opp_an.get(sp, 0) + n_an.get(sp, 0) + shed.get(sp, 0)
                        + _planbuy.get(sp, 0)) * rate
            head = max(0.0, float(demand.get(pr, 1)) - combined)
            at_base = min(rate, head)
            floor_p = 0.5 * base_p if pr in DUMP_PROOF else 1.0
            return at_base * base_p + (rate - at_base) * floor_p

        total_cap = sum(_sched(herd.get(sp), d) for sp in ANIMALS)
        # herd gate: a floored product's animal is dropped from the menu
        allowed = set(ANIMALS)
        for sp in ANIMALS:
            spec = gate.get(sp)
            if spec:
                pr = ANIMAL_PRODUCT[sp]
                base_p = float(BASE_PRICE.get(pr, 1))
                px = float(prices.get(pr, base_p))
                if float(rate_g.get(pr, 0)) < float(spec.get("min_rate", 0)) \\
                        or px < float(spec.get("min_price", 0)) * base_p:
                    allowed.discard(sp)
        have_total = sum(n_an.get(sp, 0) + shed.get(sp, 0) for sp in ANIMALS)
        floor = float(g.get("floor_animal", 250)) + land_res
        guard = 0
        while have_total < total_cap and guard < 40:
            guard += 1
            cand = [sp for sp in allowed
                    if money >= ANIMALS[sp]["cost"] + floor
                    and _an_score(sp) > 0]
            if not cand:
                break
            sp = max(cand, key=_an_score)
            late.append((2, ["BUY_ANIMAL", sp, 1]))
            money -= ANIMALS[sp]["cost"]
            bought[sp] = bought.get(sp, 0) + 1
            _planbuy[sp] = _planbuy.get(sp, 0) + 1   # balances the mix
            have_total += 1
    elif not sell_all:
        for sp in ("COW", "SHEEP", "GOOSE"):
            cap = _sched(herd.get(sp), d)
            spec = gate.get(sp)
            if spec:
                pr = ANIMAL_PRODUCT[sp]
                base_p = float(BASE_PRICE.get(pr, 1))
                px = float(prices.get(pr, base_p))
                if float(rate_g.get(pr, 0)) < float(spec.get("min_rate", 0)) \\
                        or px < float(spec.get("min_price", 0)) * base_p:
                    cap = min(cap, int(spec.get("floor", 0)))
            have = n_an.get(sp, 0) + shed.get(sp, 0)
            cost = ANIMALS[sp]["cost"]
            floor = float(g.get("floor_animal", 250)) + land_res
            while have < cap and money >= cost + floor:
                late.append((2, ["BUY_ANIMAL", sp, 1]))
                money -= cost
                bought[sp] = bought.get(sp, 0) + 1
                have += 1
    fbuy = int(g.get("fert_buy", 0) or 0)
    if fbuy > 0 and 6 <= d <= 24 and money > 2000:
        n = min(fbuy, int((money - 1500) // 110))
        if n > 0:
            late.append((3, ["BUY_PRODUCT", "FERTILIZER", n]))
            money -= n * 110

    # ---- tomorrow's feed, bought today so the dawn pickup can find it ---
    n_beasts = len(animals)
    beasts_tomorrow = n_beasts + sum(shed.get(s, 0) for s in ANIMALS) \\
        + sum(bought.values())
    reserve_w = 0
    if beasts_tomorrow and not final:
        reserve_w = beasts_tomorrow + int(g.get("feed_buffer", 6))

    # ---- pace: expansion never starves the standing farm ----------------
    # Priced against the PLANNED crew, so the pace decision does not depend on
    # the hire trim below (which is decided from the paced task list).
    cap_ops = (1 + n_h) * (TPD - 1)
    tend = sum(len(o) + 2 for _t, o, p, _n in tasks if p <= 2)
    room = max(0, int(cap_ops * float(g.get("plant_pace", 0.95))) - tend)
    keep, grow = [], []
    for tk in tasks:
        if tk[2] >= 3 and any(k.startswith("SEED_") for k in tk[3]):
            grow.append(tk)
        else:
            keep.append(tk)
    used = 0
    for tk in grow:
        c = len(tk[1]) + 2
        if used + c > room:
            continue
        used += c
        keep.append(tk)
    tasks = keep
    tasks.sort(key=lambda tk: (TIDX.get(tk[0], 0),))

    # ---- hire to the WORK, not to a fixed ramp --------------------------
    # An idle hand still costs its fibonacci fee. Measured on the skeleton:
    # 2,387 PASS unit-turns a season against the top-30 field's 515, most of
    # them on days 1-9 when the farm is one quadrant and ten hands are hired.
    # `hire_slack` = -1 keeps the old behaviour (hire the ramp blindly);
    # >= 0 hires only the hands the paced task list can actually feed, plus
    # that many spares. The fee for the dropped hands is simply never spent,
    # so it shows up as higher money at tomorrow's dawn.
    slack = int(g.get("hire_slack", -1))
    if slack >= 0 and n_h > 0:
        need = sum(len(o) + 2 for _t, o, _p, _n in tasks)
        n_h = min(n_h, max(0, -(-need // (TPD - 1)) - 1) + slack)

    # ---- the crew (every hand hired at hour 0 -> deterministic spawn) ---
    units = [_U(SHED[0], 1, TPD - 1)]
    for k in range(n_h):
        units.append(_U(SHED[(k + 1) % 4], 1, TPD - 1))

    # ---- assign: contiguous tours, shed withdrawals batched up front ----
    shed_pool, field_pool = [], []
    for tk in tasks:
        if any(not k.startswith("SEED_") for k in tk[3]):
            shed_pool.append(tk)
        else:
            field_pool.append(tk)
    picked = {{}}

    # QUADRANT-MATCHED ASSIGNMENT. Units spawn at the four shed corners (one
    # per quadrant); TILE_ORDER is a per-quadrant boustrophedon from that
    # corner. Giving each unit ONLY its own quadrant's tasks turns every tour
    # into a tight local sweep from the spawn corner, instead of a hand walking
    # across the board to a chunk that happened to fall at its pool position.
    def _q(t):
        return (0 if t[1] < 5 else 2) + (1 if t[0] >= 5 else 0)
    uq = {{0: [], 1: [], 2: [], 3: []}}
    for u in units:
        uq[_q(u.pos)].append(u)

    def _take(pool, ul, pickup):
        ui = 0
        while pool and ui < len(ul):
            u = ul[ui]
            ui += 1
            if u.budget() <= 0:
                continue
            chunk = []
            items = set()
            pos = u.pos
            budget = u.budget()
            while pool:
                tile, ops, _p, nd = pool[0]
                new_items = items
                if pickup:
                    new_items = items | set(
                        k for k in nd if not k.startswith("SEED_"))
                c = _cost(pos, tile, ops) + (len(new_items) - len(items)
                                             if pickup else 0)
                if budget - c < 0:
                    break
                chunk.append(pool.pop(0))
                items = new_items
                budget -= c
                pos = tile
            if not chunk:
                continue
            if pickup and items:
                tot = {{}}
                for _t2, _o2, _p2, nd2 in chunk:
                    for k2, v2 in nd2.items():
                        if k2.startswith("SEED_"):
                            continue
                        it = k2[3:] if k2.startswith("AN_") else k2
                        tot[it] = tot.get(it, 0) + v2
                for it in sorted(tot):
                    avail = shed.get(it, 0) - picked.get(it, 0)
                    n = min(tot[it], max(0, avail))
                    if n > 0:
                        u.act(["PICKUP", it, n])
                        picked[it] = picked.get(it, 0) + n
            for tile, ops, _p, _nd in chunk:
                u.walk(tile)
                for o in ops:
                    u.act(o)
        return pool                      # tasks this quadrant could not fit

    for q in (0, 1, 2, 3):
        _take([tk for tk in shed_pool if _q(tk[0]) == q], uq[q], True)
    field_left = []
    for q in (0, 1, 2, 3):
        field_left += _take([tk for tk in field_pool if _q(tk[0]) == q],
                            uq[q], False)
    # global sweep: any unit with spare budget takes the nearest leftover
    for u in units:
        while field_left and u.budget() > 0:
            field_left.sort(key=lambda tk: _cost(u.pos, tk[0], tk[1]))
            tile, ops, _p, _nd = field_left[0]
            if u.budget() < _cost(u.pos, tile, ops):
                break
            field_left.pop(0)
            u.walk(tile)
            for o in ops:
                u.act(o)
    # SAME-DAY MARKETING. Harvested goods sit in the farmer's inventory
    # until `_drop_inventories_to_shed` runs at dusk, so this seat can only
    # ever sell yesterday's harvest -- and `_fresh_sells` reads the shed, so
    # the whole farm is one day late to market every day. v43.0_bandit is
    # not: its melons reached the market DURING day 10 (inventory +49 at the
    # last step of day 10), which is how it took the top of the melon curve
    # ($242/unit against our $169) on a pool that has no shop demand at all
    # and therefore never refills.
    if int(g.get("drop_daily", 0) or 0) and not sell_all:
        for u in units:
            if u.loaded:
                tgt = min(SHED, key=lambda s: abs(u.pos[0] - s[0])
                          + abs(u.pos[1] - s[1]))
                if u.budget() > abs(u.pos[0] - tgt[0]) + abs(u.pos[1] - tgt[1]):
                    u.walk(tgt)
                    u.act(["DROP"])
                    u.loaded = False
    if sell_all:                         # bank the day's take before dusk
        for u in units:
            if u.loaded and u.budget() > 1:
                tgt = min(SHED, key=lambda s: abs(u.pos[0] - s[0])
                          + abs(u.pos[1] - s[1]))
                u.walk(tgt)
                u.act(["DROP"])

    # ---- seed: exactly the plantings labour actually scheduled ----------
    planted = {{}}
    for u in units:
        for o in u.ops:
            if o[0] == "PLANT":
                planted[o[1]] = planted.get(o[1], 0) + 1
    seed_orders = []
    for sp in sorted(planted):
        n = planted[sp] - seeds.get(sp, 0)
        if n > 0:
            seed_orders.append(["BUY_SEED", sp, n])

    # ---- feed wheat for TOMORROW ----------------------------------------
    short = reserve_w - (shed.get("WHEAT", 0) - picked.get("WHEAT", 0))
    if short > 0:
        late.append((4, ["BUY_PRODUCT", "WHEAT", short]))

    # ---- sells: on production, above the reserves -----------------------
    caps = dict(g.get("sell_cap") or {{}})
    floor = float(g.get("sell_floor", 0.0) or 0.0)
    sold = {{}}
    for item in SELLABLE:
        have = shed.get(item, 0) - picked.get(item, 0)
        if item == "WHEAT" and not sell_all:
            have -= reserve_w
        elif item == "FERTILIZER" and not sell_all:
            have -= int(g.get("fert_keep", 12))
        if have <= 0:
            continue
        if sell_all:
            sold[item] = have
            continue
        p = float(prices.get(item, BASE_PRICE.get(item, 25)))
        if floor > 0 and p < floor * BASE_PRICE.get(item, 25):
            continue
        q = min(have, int(caps.get(item, 999)))
        if q > 0:
            sold[item] = q
    # pressure valve: the shed caps at 100 TOTAL, dusk overflow is DROPPED
    total = sum(shed.values()) - sum(picked.values()) - sum(sold.values())
    v_hi = int(g.get("valve_hi", 55))
    v_lo = int(g.get("valve_lo", 45))
    if not sell_all and total > v_hi:
        for item in ("WHEAT", "EGG", "FERTILIZER", "CARROT", "TOMATO",
                     "MILK", "STRAWBERRY", "WOOL", "MELON"):
            room2 = shed.get(item, 0) - picked.get(item, 0) - sold.get(item, 0)
            if item == "WHEAT":
                room2 -= reserve_w
            take = min(room2, total - v_lo)
            if take > 0:
                sold[item] = sold.get(item, 0) + take
                total -= take
            if total <= v_lo:
                break

    # ---- serialise the market queue (hires FIRST: hour-0 batch) ---------
    sell_q = _order_sells([["SELL", item, int(sold[item])]
                           for item in sorted(sold) if sold[item] > 0], prices)
    # The reserved sells go at slot 0, AHEAD of the hires and of the day's
    # purchases, on purpose: slot index is the price ladder, and a SELL that
    # commits in an earlier slot also credits money the later HIRE/BUY slots
    # can spend -- the dawn purse is $132-$2,600 through day 13.
    queue = list(sell_q[:n0])
    queue.extend([["HIRE"]] * n_h)
    queue.extend(seed_orders)
    for _r, op in sorted(late, key=lambda x: x[0]):
        queue.append(op)
    queue.extend(sell_q[n0:])

    plan = []
    for h in range(TPD):
        mk = queue[h * 10:(h + 1) * 10]
        row = {{"farmer": ["PASS"], "hands": [], "market": mk}}
        u0 = units[0]
        if h >= u0.start and h - u0.start < len(u0.ops):
            row["farmer"] = u0.ops[h - u0.start]
        hands = []
        for u in units[1:]:
            if h >= u.start and h - u.start < len(u.ops):
                hands.append(u.ops[h - u.start])
            else:
                hands.append(["PASS"])
        row["hands"] = hands
        plan.append(row)
    return plan


def _order_sells(out, prices):
    """Market-order SLOT is a hard price ladder: `_process_market` walks
    slot 0 of BOTH players to exhaustion, reprices, then slot 1. The seat
    whose order sits in the earlier slot takes the top of the curve and
    leaves the walked-down remainder to the other. Measured against
    v43.0_bandit: it puts 84% of its sells in slot 0 and we spread ours
    over slots 0-9, so it out-priced us on every product we both sell
    (STRAWBERRY $145 vs $99, MELON $242 vs $169, FERTILIZER $63 vs $48).
    Sorting our own queue by the dollars at stake puts the biggest order
    where it can still win the collision."""
    mode = int(GENOME.get("sell_order", 0) or 0)
    if mode == 1:                       # biggest dollar order first
        return sorted(out, key=lambda o: -(float(prices.get(o[1], 1) or 1)
                                           * int(o[2])))
    if mode == 2:                       # explicit race order
        # MELON and FERTILIZER are the only two products with (near) ZERO
        # town demand -- FERTILIZER is in no shop and not a town-centre
        # product at all, MELON is in no shop and the centre eats 1/day --
        # so every unit either seat ever sells stays in the inventory for
        # the rest of the season and the price only ever falls. They are
        # one-shot pools split by WHO SELLS FIRST, and this seat currently
        # loses both (MELON $169 vs $242, FERTILIZER $48 vs $63).
        prio = list(GENOME.get("sell_prio") or ())
        rank = dict((p, i) for i, p in enumerate(prio))
        return sorted(out, key=lambda o: (rank.get(o[1], len(prio)),
                                          -(float(prices.get(o[1], 1) or 1)
                                            * int(o[2]))))
    return out


def _fresh_sells(d, farm, priv, prices=None):
    """Market-only refresh: whatever the shed holds right now, at the live
    price. Oversized SELLs partial-fill, so an overlap with the dawn queue
    costs a queue slot at most."""
    g = GENOME
    prices = prices or {{}}
    shed = {{}}
    for k, v in (_get(priv, "shed", {{}}) or {{}}).items():
        shed[k] = int(v or 0)
    tiles = _get(farm, "tiles", []) or []
    n_beasts = sum(shed.get(s, 0) for s in ANIMALS)     # awaiting a pen
    for row in tiles:
        for tl in row:
            if isinstance(tl, dict) and "animal" in tl:
                n_beasts += 1
    sell_all = d >= LAST_DAY - 1
    reserve_w = 0 if not n_beasts or sell_all else \\
        n_beasts + int(g.get("feed_buffer", 6))
    caps = dict(g.get("sell_cap") or {{}})
    hold = dict(g.get("hold_price") or {{}})
    total = sum(shed.values())
    pressed = total > int(g.get("valve_hi", 55))
    out = []
    for item in SELLABLE:
        have = shed.get(item, 0)
        if item == "WHEAT" and not sell_all:
            have -= reserve_w
        elif item == "FERTILIZER" and not sell_all:
            have -= int(g.get("fert_keep", 12))
        if have <= 0:
            continue
        # HOLD-WHEN-WORTHLESS. Selling a unit at $1 banks $1 and, by the
        # engine's own rule ("sales at $1 do not increase market supply"),
        # does not even cost the other seat anything. The shed cap and the
        # dusk discard are the only reasons to sell into a floored market,
        # so the hold is released under valve pressure and in liquidation.
        if not sell_all and not pressed:
            hp = float(hold.get(item, 0) or 0)
            if hp > 0 and float(prices.get(item, BASE_PRICE.get(item, 1))) < hp:
                continue
        q = have if sell_all else min(have, int(caps.get(item, 999)))
        if q > 0:
            out.append(["SELL", item, q])
    return _order_sells(out, prices)


def agent(obs, cfg=None):
    try:
        d = _get(obs, "day", None)
        h = _get(obs, "hour", None)
        if d is None or h is None:
            step = int(_get(obs, "step", 0) or 0)
            d, h = step // TPD, step % TPD
        d, h = int(d), int(h)
        me = int(_get(obs, "player", 0) or 0)
        farms = _get(obs, "farms", []) or []
        farm = farms[me] if me < len(farms) else {{}}
        opp = farms[1 - me] if (1 - me) < len(farms) else {{}}
        priv = _get(obs, "private", {{}}) or {{}}
        mkt = _get(obs, "market", {{}}) or {{}}
        town = _get(obs, "town", {{}}) or {{}}
        if _D["day"] != d or _D["plan"] is None:
            _D["plan"] = _plan(d, farm, priv, mkt, town, opp)
            _D["day"] = d
        a = _D["plan"][h] if h < len(_D["plan"]) else None
        if a is None:
            a = {{"farmer": ["PASS"], "hands": [], "market": []}}
        market = list(a.get("market") or [])
        if h > 0:
            keep = [o for o in market if o and o[0] != "SELL"]
            market = keep + _fresh_sells(
                d, farm, priv, dict(_get(mkt, "prices", {{}}) or {{}}))
        real = _get(farm, "hands", []) or []
        hands = list(a.get("hands") or [])
        if len(hands) < len(real):
            hands += [["PASS"]] * (len(real) - len(hands))
        elif len(hands) > len(real):
            hands = hands[:len(real)]
        return {{"farmer": a.get("farmer") or ["PASS"], "hands": hands,
                "market": market[:10]}}
    except Exception:
        n = 0
        try:
            farms = _get(obs, "farms", []) or []
            me = int(_get(obs, "player", 0) or 0)
            if me < len(farms):
                n = len(_get(farms[me], "hands", []) or [])
        except Exception:
            n = 0
        return {{"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}}
'''


def load_genome(name):
    if os.path.exists(name):
        return json.load(open(name, encoding="utf-8"))
    if name in ("skeleton", "default"):
        return skeleton_genome()
    if name == "live":
        st = json.load(open(os.path.join(
            ROOT, "models", "trackp", "genome_ga", "state_live.json"),
            encoding="utf-8"))
        return st["center"]
    st = json.load(open(os.path.join(
        ROOT, "models", "trackp", "genome_ga", "state_adv.json"),
        encoding="utf-8"))
    return st["center"] if name == "center" else st["best_ever"]["genome"]


# G7.T2 (2026-09-18): knobs the PYTHON planner honours but the Rust skeleton
# (rustengine/src/policy.rs) does NOT yet implement, with the value policy.rs
# effectively assumes. A genome that deviates would make the compiled Rust agent
# and its Python fallback DIVERGE (Python acts on the knob, Rust ignores it). We
# refuse to render such a genome loudly at BUILD time rather than let
# test_compiled_agent catch it later (or, worse, ship a mismatch). Port the knob
# into policy.rs to lift the guard.
RUST_UNIMPLEMENTED_KNOBS = {
    "drop_daily": 0,       # policy.rs never walks a loaded unit back to DROP mid-day
    "hire_slack": -1,      # policy.rs hires the full ramp (no work-paced trimming)
}


def assert_rust_honored(genome):
    """Raise if a genome sets a knob the Rust compiled agent cannot honour."""
    g = dict(skeleton_genome())
    g.update(genome or {})
    bad = {k: g.get(k) for k, dflt in RUST_UNIMPLEMENTED_KNOBS.items()
           if int(g.get(k, dflt) or 0) != dflt}
    if bad:
        raise ValueError(
            "G7.T2: genome sets knob(s) the Rust policy.rs does NOT implement, "
            f"so the compiled agent would diverge from its Python fallback: {bad}. "
            f"Rust-honoured defaults are {RUST_UNIMPLEMENTED_KNOBS}. "
            "Port the knob into rustengine/src/policy.rs (then update "
            "RUST_UNIMPLEMENTED_KNOBS), or keep the genome at the default.")


def render(genome, path):
    import datetime as dt
    base = skeleton_genome()
    base.update(genome or {})
    assert_rust_honored(base)               # loud build-time parity guard (G7.T2)
    src = TEMPLATE.format(label=os.path.basename(path),
                          built=dt.date.today().isoformat(), genome=base)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--genome", default="skeleton",
                    help="'skeleton' (the measured field economy), "
                         "'live', 'center', 'best_ever', or a JSON path")
    ap.add_argument("--out", default=os.path.join(
        ROOT, ".local", "candidates", "trackp_skel.py"))
    args = ap.parse_args()
    render(load_genome(args.genome), args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
