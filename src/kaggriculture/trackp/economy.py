"""Track-P owned economy: genome -> compiled 720-step tape.

The synthesis of every measured verdict this campaign (2026-08-30):
per-turn judgment loses (planner_v0 0-64); mutating copies saturates
(sell/economy searches); copies of originals cap ~350 points below their
source (mv_Kaileh 2420 vs Kaileh57 ~2780). The remaining road to 2800+ is
an economy WE own, searched in plan-space.

The GENOME is a small, interpretable farm plan: when to buy land, the
daily crew curve, what each quadrant grows or raises and from when,
feed/sell/fertilize policy. The COMPILER expands it deterministically into
a full tape: tile layout, daily task lists (water/feed/harvest/plant/
build/collect), greedy travel-aware labor assignment over units that all
respawn at the shed each dawn, market orders (feed-pinned buys, seed and
animal purchases, cadenced sells, terminal liquidation). Search owns the
tradeoffs; the compiler owns correctness; the ENGINE owns legality --
fitness is only ever played episodes (kagg batch, rotating seeds,
holdout sign gate).

Engine facts baked in (all verified against the interpreter/docs):
units respawn at the shed daily and auto-drop inventory at dusk; watering
once/day (mandatory on planting day); FEED takes wheat from the unit's
inventory (PICKUP first); one-time crops gain +1/watered-day (+2 fert)
inside [ceil(max/2), max]; animals produce on fixed intervals with
max_held caps; HIRE cost fib resets daily; shed caps at 100 so harvests
must sell promptly; only WHEAT and FERTILIZER are buyable back.
"""
import copy
import random

TPD = 24
BOARD = 10
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
QUADS = ("NW", "NE", "SW", "SE")
LAND_ORDER = ("NE", "SW", "SE")
LAND_COST = (1000, 2000, 4000)

CROPS = {
    "WHEAT":      {"seed": 10, "first": 2, "max_day": 4, "ongoing": False},
    "CARROT":     {"seed": 20, "first": 2, "max_day": 3, "ongoing": False},
    "MELON":      {"seed": 80, "first": 10, "max_day": 10, "ongoing": False},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 16, "ongoing": True,
                   "prod_days": (10, 12, 14, 16)},
    "TOMATO":     {"seed": 50, "first": 8, "max_day": 11, "ongoing": True,
                   "prod_days": (8, 9, 10, 11)},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "build": "BUILD_COOP", "first": 4, "every": 1,
              "product": "EGG"},
    "COW":   {"cost": 400, "build": "BUILD_PASTURE", "first": 8, "every": 2,
              "product": "MILK"},
    "SHEEP": {"cost": 500, "build": "BUILD_PASTURE", "first": 6, "every": 3,
              "product": "WOOL"},
}
PLANTS = tuple(CROPS)
BEASTS = tuple(ANIMALS)


def quad_tiles(q):
    xs = range(5, 10) if q in ("NE", "SE") else range(0, 5)
    ys = range(5, 10) if q in ("SW", "SE") else range(0, 5)
    return [(x, y) for y in ys for x in xs]


def default_genome():
    """A sane hand seed: wheat+carrot engine in NW, melon/strawberry in NE,
    sheep+cows in SW; SE unbought (measured negative)."""
    return {
        "land": {"NE": 4, "SW": 8, "SE": -1},
        "hires": [4, 5, 5, 6, 7, 8, 9, 10, 10, 11, 11, 11, 11, 11, 11,
                  11, 11, 11, 11, 11, 11, 10, 10, 9, 9, 8, 7, 6, 5, 4],
        "alloc": {
            "NW": {"WHEAT": 14, "CARROT": 8},
            "NE": {"MELON": 12, "STRAWBERRY": 10},
            "SW": {"SHEEP": 6, "COW": 6, "GOOSE": 4, "WHEAT": 6},
            "SE": {},
        },
        "start": {"NW": 0, "NE": 5, "SW": 9, "SE": -1},
        "replant": {"WHEAT": True, "CARROT": True, "MELON": True},
        "feed_buy": 4,
        "sell_every": 1,
        "fertilize": True,
        "care": False,
        "last_plant_day": 24,
        # Per-product daily sell caps: the market is shared-inventory with
        # per-unit price impact. MELON's glut side is QUADRATIC and no shop
        # consumes melons (recovery 1/day) -- a dump is unrecoverable, and a
        # mirrored pair selling ~144 each floors the price. WOOL sq too;
        # STRAWBERRY/MILK linear with tiny T; WHEAT/EGG log = dump-proof.
        "sell_cap": {"MELON": 8, "WOOL": 8, "STRAWBERRY": 12, "MILK": 12,
                     "TOMATO": 25, "CARROT": 50},
        "fert_crops": True,     # free (collected) fertilizer -> melon window
        # Town shops DRAIN the shared market (each instance eats its listed
        # products every 4 steps), so scarcity prices climb all season --
        # strawberry was observed at 267 (base 120) on day 29. hold_until
        # delays selling an item until day N to ride that curve; the shed
        # pressure valve still overrides at cap.
        "hold_until": {},
        # Kaileh forensics (2026-08-30): the elite economy is a fertilized-
        # WHEAT replant factory (fert doubles the harvest to 6; one fert at
        # age 2 covers the whole [2,4] window) fed by cow fertilizer plus
        # BOUGHT fertilizer (they buy ~250). Both knobs GA-searchable.
        "fert_wheat": False,
        "fert_buy": 0,          # BUY_PRODUCT FERTILIZER per day (days 1..24)
    }


# ------------------------------------------------------------- compiler

class _Unit:
    __slots__ = ("pos", "ops", "carrying_wheat")

    def __init__(self, spawn):
        self.pos = spawn
        self.ops = []
        self.carrying_wheat = 0

    def budget(self):
        return TPD - 1 - len(self.ops)          # hour 0 is the spawn turn

    def walk_to(self, tile):
        (ax, ay), (bx, by) = self.pos, tile
        while ax < bx and self.budget() > 0:
            self.ops.append(["EAST"]); ax += 1
        while ax > bx and self.budget() > 0:
            self.ops.append(["WEST"]); ax -= 1
        while ay < by and self.budget() > 0:
            self.ops.append(["SOUTH"]); ay += 1
        while ay > by and self.budget() > 0:
            self.ops.append(["NORTH"]); ay -= 1
        self.pos = (ax, ay)
        return self.pos == tile

    def act(self, op):
        if self.budget() > 0:
            self.ops.append(op)
            return True
        return False


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def measure_money(tape, steps=720, seed=9000):
    """Engine-oracle pass: the ACTUAL per-step cash of a compiled tape in
    self-play. Projections lied three times (day-8 land, hires, sells);
    the engine never does."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": seed},
               info={"seed": seed})
    env.reset(2)
    money = []
    for a in tape:
        if env.done:
            break
        money.append(float(env.state[0].observation.farms[0]["money"]))
        env.step([a if isinstance(a, dict) else {}] * 2)
    return money


def compile_two_pass(g, steps=720):
    """compile -> simulate -> recompile with the measured cash curve."""
    first = compile_genome(g, steps)
    oracle = measure_money(first, steps)
    return compile_genome(g, steps, oracle=oracle)


def compile_genome(g, steps=720, oracle=None):
    """Deterministic plan -> tape. Internal world model mirrors the engine's
    growth/production arithmetic; anything it gets slightly wrong is a
    no-op the fitness sees, never corruption."""
    days = steps // TPD
    tape = [{"farmer": ["PASS"], "hands": [], "market": []}
            for _ in range(steps)]

    # ---- static layout ------------------------------------------------
    plants = {}          # tile -> {species, planted_day(None until planted)}
    animals = {}         # tile -> {species, built(bool), placed_day(None)}
    unlock_day = {"NW": 0}
    for i, q in enumerate(LAND_ORDER):
        d = g["land"].get(q, -1)
        if isinstance(d, int) and d >= 0:
            unlock_day[q] = d
    for q in QUADS:
        if q not in unlock_day and q != "NW":
            continue
        pool = quad_tiles(q)
        pi = 0
        for species, count in (g["alloc"].get(q) or {}).items():
            for _ in range(int(count)):
                if pi >= len(pool):
                    break
                tile = pool[pi]; pi += 1
                if species in CROPS:
                    plants[tile] = {"sp": species, "pd": None, "q": q}
                elif species in ANIMALS:
                    animals[tile] = {"sp": species, "built": False,
                                     "placed": None, "q": q}

    # ---- pending market ledgers ---------------------------------------
    day_market = [[] for _ in range(days)]
    shed_est = {}                 # rough shed content for sell cadence

    def order(d, op, pri=5):
        if 0 <= d < days:
            day_market[d].append((pri, len(day_market[d]), op))

    land_pending = {q: unlock_day[q] for q in LAND_ORDER
                    if q in unlock_day}

    # ---- per-day simulation + emission --------------------------------
    # FEASIBILITY PACING (the day-9 graveyard lesson: 25 weeds, 0 animals):
    # the compiler defers what the plan cannot yet afford or water. cash is
    # a conservative projection (sell revenue at 0.7x base); labor capacity
    # ~7 tasks/unit/day including travel.
    BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
                  "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
                  "FERTILIZER": 100}
    cash = 3000.0
    TASKS_PER_UNIT = 7
    harvested_ongoing = {}        # tile -> set of paid prod_days
    for d in range(days):
        # HIRES first (front so hour-0 pairing is stable), feed wheat pinned.
        for qi, qq in enumerate(LAND_ORDER):
            plan_day = land_pending.get(qq)
            if plan_day is not None and d >= plan_day:
                real = None
                if oracle:
                    s_probe = min(len(oracle) - 1, d * TPD + 2)
                    real = oracle[s_probe]
                have = real if real is not None else cash
                if have >= LAND_COST[qi] + 400:
                    order(d, ["BUY_LAND"], pri=2)
                    cash -= LAND_COST[qi]
                    unlock_day[qq] = d          # actual unlock day
                    land_pending.pop(qq)
                else:
                    unlock_day[qq] = d + 1      # slide the whole quadrant
        n_h = int(g["hires"][min(d, len(g["hires"]) - 1)])
        live_animals = [t for t, a in animals.items()
                        if a["placed"] is not None]
        feed_need = len(live_animals)
        wheat_buy = max(0, feed_need + int(g["feed_buy"]) - 0)
        if wheat_buy and d > 0:
            order(d, ["BUY_PRODUCT", "WHEAT", wheat_buy], pri=0)
            cash -= wheat_buy * 30
        fib = [1, 1]
        while len(fib) < max(2, n_h):
            fib.append(fib[-1] + fib[-2])
        afford = 0
        spend = 0.0
        for i in range(n_h):
            c = fib[min(i, len(fib) - 1)]
            if cash - spend - c < 60:      # never hire into insolvency
                break
            spend += c
            afford += 1
        n_h = afford
        cash -= spend
        for _ in range(n_h):
            order(d, ["HIRE"], pri=1)

        # Task list for today.
        seed_need = {}
        plant_wishes = []
        tasks = []                # (tile, [ops...], priority)
        for t, a in sorted(animals.items()):
            if unlock_day.get(a["q"], 99) > d:
                continue
            if not a["built"] and d >= unlock_day.get(a["q"], 99):
                cost = ANIMALS[a["sp"]]["cost"]
                if not a.get("bought"):
                    real = None
                    if oracle:
                        s_probe = min(len(oracle) - 1, d * TPD + 2)
                        real = oracle[s_probe]
                    have = real if real is not None else cash
                    if have >= cost + 300:
                        order(d, ["BUY_ANIMAL", a["sp"], 1], pri=3)
                        a["bought"] = True
                        cash -= cost
                    continue
                # bought yesterday or earlier: place today
                tasks.append((t, [["PICKUP", a["sp"], 1],
                                  [ANIMALS[a["sp"]]["build"]],
                                  ["PLACE", a["sp"]]], 1))
                a["built"] = True
                a["placed"] = d
            elif a["placed"] is not None:
                ops = [["FEED"]]
                age = d - a["placed"]
                sp = ANIMALS[a["sp"]]
                if age >= sp["first"] and (age - sp["first"]) % sp["every"] == 0:
                    ops.append(["HARVEST"])
                    shed_est[sp["product"]] = shed_est.get(sp["product"], 0) + 2
                if g.get("care"):
                    ops.append(["CARE"])
                if g.get("fertilize"):
                    ops.append(["COLLECT_FERTILIZER"])
                    shed_est["FERTILIZER"] = shed_est.get("FERTILIZER", 0) + 1
                tasks.append((t, ops, 2))
        for t, p in sorted(plants.items()):
            if unlock_day.get(p["q"], 99) > d:
                continue
            sp = CROPS[p["sp"]]
            if p["pd"] is None:
                if d >= max(g["start"].get(p["q"], 0), unlock_day.get(p["q"], 0)) \
                        and d <= g.get("last_plant_day", 24) - sp["first"]:
                    tasks.append((t, [["PLANT", p["sp"]], ["WATER"]], 1))
                    p["pd"] = d
                    seed_need[p["sp"]] = seed_need.get(p["sp"], 0) + 1
                continue
            age = d - p["pd"]
            ops = []
            if sp["ongoing"]:
                if age in sp["prod_days"]:
                    ops.append(["WATER"])
                    ops.append(["HARVEST"])
                    shed_est[p["sp"]] = shed_est.get(p["sp"], 0) + 1
                elif age < sp["prod_days"][-1]:
                    ops.append(["WATER"])
                else:
                    ops.append(["DIG"])
                    p["pd"] = None
                    continue
            else:
                if age < sp["max_day"]:
                    ops.append(["WATER"])
                    if (g.get("fertilize") and shed_est.get("FERTILIZER", 0) > 0
                            and age == (sp["max_day"] + 1) // 2):
                        ops.insert(0, ["FERTILIZE"])
                        shed_est["FERTILIZER"] -= 1
                elif age == sp["max_day"]:
                    ops.append(["WATER"])
                    ops.append(["HARVEST"])
                    shed_est[p["sp"]] = shed_est.get(p["sp"], 0) + 5
                    if g["replant"].get(p["sp"]) and \
                            d <= g.get("last_plant_day", 24) - sp["first"]:
                        ops.append(["PLANT", p["sp"]])
                        ops.append(["WATER"])
                        seed_need[p["sp"]] = seed_need.get(p["sp"], 0) + 1
                        p["pd"] = d
                    else:
                        p["pd"] = -999
                else:
                    continue
            if ops:
                tasks.append((t, ops, 2 if ["WATER"] in ops else 3))

        # Pace new plantings: only what today's crew can water ON TOP of
        # standing tasks, and only what projected cash can seed.
        capacity = (1 + n_h) * TASKS_PER_UNIT
        standing = len(tasks)
        room_tasks = max(0, capacity - standing)
        for t, p in plant_wishes[:room_tasks]:
            spc = CROPS[p["sp"]]
            if cash < spc["seed"] + 150:      # keep a feed cushion
                break
            tasks.append((t, [["PLANT", p["sp"]], ["WATER"]], 1))
            p["pd"] = d
            seed_need[p["sp"]] = seed_need.get(p["sp"], 0) + 1
            cash -= spc["seed"]

        # Feeding needs wheat in hand: prepend one PICKUP per feeder later.
        units = [_Unit(SHED_TILES[i % 4]) for i in range(1 + n_h)]
        tasks.sort(key=lambda x: (x[2], x[0]))
        # Greedy nearest-unit assignment with budgets.
        for tile, ops, _pri in tasks:
            needs_feed = any(o[0] == "FEED" for o in ops)
            cost = len(ops) + (2 if needs_feed else 0)
            best, bd = None, 10 ** 9
            for u in units:
                dd = _dist(u.pos, tile)
                if u.budget() >= dd + cost and dd < bd:
                    best, bd = u, dd
            if best is None:
                continue
            if needs_feed and best.carrying_wheat < 1:
                take = min(6, feed_need)
                best.act(["PICKUP", "WHEAT", take])
                best.carrying_wheat += take
            # Shed pickups (animals) must happen at spawn, BEFORE the walk.
            while ops and ops[0][0] == "PICKUP" and ops[0][1] != "WHEAT":
                best.act(ops[0])
                ops = ops[1:]
            best.walk_to(tile)
            for o in ops:
                best.act(o)
            if needs_feed:
                best.carrying_wheat -= 1

        # Batched seed purchases for today's plantings.
        for sp_name, n in sorted(seed_need.items()):
            order(d, ["BUY_SEED", sp_name, n], pri=4)
        # Serialize the day: hour 0 = market (priority-sorted); overflow
        # spills across the following turns, 10 per turn.
        s0 = d * TPD
        queue = [op for _pri, _i, op in sorted(day_market[d])]
        for k in range(0, len(queue), 10):
            if s0 + k // 10 >= len(tape):
                break
            tape[s0 + k // 10].setdefault("market", []).extend(
                queue[k:k + 10])
            tape[s0 + k // 10]["market"] = tape[s0 + k // 10]["market"][:10]
        for h in range(1, TPD):
            s = s0 + h
            farmer_op = units[0].ops[h - 1] if h - 1 < len(units[0].ops) \
                else ["PASS"]
            tape[s]["farmer"] = farmer_op
            hands_ops = []
            for u in units[1:]:
                hands_ops.append(u.ops[h - 1] if h - 1 < len(u.ops)
                                 else ["PASS"])
            tape[s]["hands"] = hands_ops

        # Sells: cadence policy, drain the estimated shed.
        if d > 0 and d % max(1, int(g.get("sell_every", 1))) == 0:
            s_sell = s0 + 2
            while s_sell < s0 + TPD - 1 and                     len(tape[s_sell].get("market") or []) >= 8:
                s_sell += 1
            mk = tape[s_sell].setdefault("market", [])
            for item in ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG",
                         "CARROT", "FERTILIZER", "TOMATO", "WHEAT"):
                have = int(shed_est.get(item, 0))
                if len(mk) < 10 and have > 0:
                    mk.append(["SELL", item, max(1, have)])
                    cash += have * BASE_PRICE.get(item, 25) * 0.5
                    shed_est[item] = 0

    # Terminal liquidation: big sells on the final days.
    for s in range(steps - 2 * TPD, steps - 1):
        mk = tape[s].setdefault("market", [])
        if len(mk) < 10 and (s % 3 == 0):
            for item in ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG",
                         "CARROT", "WHEAT", "FERTILIZER"):
                if len(mk) >= 10:
                    break
                mk.append(["SELL", item, 25])
    return tape


# ------------------------------------------------------------- mutation

def mutate_genome(g, rng):
    m = copy.deepcopy(g)
    kind = rng.choice(("hires", "alloc", "start", "land", "policy", "swap",
                       "sellcap", "hold", "market"))
    if kind == "market":
        f = rng.choice(("sell_frac", "ride_scarcity", "shopmod"))
        if f == "sell_frac":
            cur = float(m.get("sell_frac", 0.0))
            m["sell_frac"] = round(max(0.0, min(1.2,
                cur + rng.choice((-0.25, -0.1, 0.1, 0.25)))), 2)
        elif f == "ride_scarcity":
            m["ride_scarcity"] = not m.get("ride_scarcity")
        else:
            shops = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
                     "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP",
                     "FARMERS_MARKET")
            items = ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG",
                     "CARROT", "TOMATO", "WHEAT")
            sm = m.setdefault("shop_mods", {})
            shop = rng.choice(shops)
            e = sm.setdefault(shop, {})
            if rng.random() < 0.5:
                e.setdefault("sell_cap", {})[rng.choice(items)] =                     rng.choice((2, 5, 10, 20, 50, 200))
            else:
                e.setdefault("hold_until", {})[rng.choice(items)] =                     rng.choice((0, 6, 12, 18, 24))
        return m
    if kind == "hold":
        hu = m.setdefault("hold_until", {})
        item = rng.choice(("MELON", "STRAWBERRY", "WOOL", "MILK", "TOMATO",
                           "CARROT", "EGG", "WHEAT"))
        cur = int(hu.get(item, 0))
        hu[item] = max(0, min(27, cur + rng.choice((-8, -4, 4, 8, 12))))
        if hu[item] == 0:
            hu.pop(item, None)
        return m
    if kind == "sellcap":
        caps = m.setdefault("sell_cap", {})
        item = rng.choice(("MELON", "STRAWBERRY", "WOOL", "MILK", "TOMATO",
                           "CARROT", "EGG"))
        cur = int(caps.get(item, 20))
        caps[item] = max(1, min(200, cur + rng.choice((-6, -3, 3, 6, 12))))
        return m
    if kind == "hires":
        i = rng.randrange(len(m["hires"]))
        m["hires"][i] = max(0, min(13, m["hires"][i] + rng.choice((-2, -1, 1, 2))))
    elif kind == "alloc":
        q = rng.choice([q for q in QUADS if m["alloc"].get(q)] or ["NW"])
        al = m["alloc"].setdefault(q, {})
        sp = rng.choice(PLANTS + BEASTS)
        cur = int(al.get(sp, 0))
        al[sp] = max(0, cur + rng.choice((-2, -1, 1, 2)))
        if al[sp] == 0:
            al.pop(sp, None)
        total = sum(al.values())
        while total > 25:
            k = rng.choice(list(al))
            al[k] -= 1
            if al[k] <= 0:
                al.pop(k)
            total -= 1
    elif kind == "start":
        q = rng.choice(QUADS)
        m["start"][q] = max(0, min(20, int(m["start"].get(q, 0))
                                   + rng.choice((-2, -1, 1, 2))))
    elif kind == "land":
        q = rng.choice(LAND_ORDER)
        cur = m["land"].get(q, -1)
        m["land"][q] = rng.choice((-1, 1, 2, 3, 4, 6, 8)) \
            if cur == -1 or rng.random() < 0.5 else max(0, cur + rng.choice((-1, 1)))
    elif kind == "policy":
        f = rng.choice(("feed_buy", "sell_every", "fertilize", "care",
                        "last_plant_day", "fert_wheat", "fert_buy",
                        "backfill_wheat"))
        if f == "fert_buy":
            m[f] = max(0, min(20, int(m.get(f, 0)) + rng.choice((-4, -2, 2, 4))))
        elif f in ("fertilize", "care", "fert_wheat",
                   "backfill_wheat"):
            m[f] = not m.get(f)
        elif f == "sell_every":
            m[f] = rng.choice((1, 2, 3))
        elif f == "feed_buy":
            m[f] = max(0, int(m.get(f, 4)) + rng.choice((-2, -1, 1, 2)))
        else:
            m[f] = max(12, min(26, int(m.get(f, 24)) + rng.choice((-2, 2))))
    elif kind == "swap":
        q = rng.choice([q for q in QUADS if m["alloc"].get(q)] or ["NW"])
        al = m["alloc"].get(q) or {}
        if al:
            old = rng.choice(list(al))
            new = rng.choice(PLANTS + BEASTS)
            if new != old:
                al[new] = al.get(new, 0) + al.pop(old)
    return m


# ---------------------------------------------------- co-simulated compile

def compile_cosim(g, steps=720, seed=9000, return_bank=False, debug=False):
    """The definitive compiler: plan each day against the ENGINE'S real dawn
    state (exact money, exact shed/seeds, real tiles including weeds), emit
    the day's 24 turns, step the engine, repeat. Open-loop tape out,
    closed-loop planning in. Kills the model-drift class outright: the
    projection lied about cash three separate times, and any static oracle
    diverges from the world its own decisions create (two-pass lost NE)."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        CROPS as ECROPS)          # engine truth; local table drifted once
    days = steps // TPD
    tape = [{"farmer": ["PASS"], "hands": [], "market": []}
            for _ in range(steps)]
    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": seed},
               info={"seed": seed})
    env.reset(2)

    # Static intent from the genome: tile -> wanted species.
    want = {}
    unlock_plan = {}
    for q in LAND_ORDER:
        d = g["land"].get(q, -1)
        if isinstance(d, int) and d >= 0:
            unlock_plan[q] = d
    for q in QUADS:
        if q != "NW" and q not in unlock_plan:
            continue
        pool = [t for t in quad_tiles(q) if t not in SHED_TILES]
        pi = 0
        for species, count in (g["alloc"].get(q) or {}).items():
            for _ in range(int(count)):
                if pi >= len(pool):
                    break
                want[pool[pi]] = species
                pi += 1

    def quad_of(t):
        return ("NE" if t[0] >= 5 else "NW") if t[1] < 5 else \
               ("SE" if t[0] >= 5 else "SW")

    animal_ordered = {}      # species -> buy orders in flight (arrive same day)
    for d in range(days):
        obs = env.state[0].observation
        farm = obs.farms[0]
        money = float(farm["money"])
        shed = dict(obs.private["shed"])
        seeds = dict(obs.private["seeds"])
        tiles = farm["tiles"]
        owned = set(farm["unlocked_quadrants"])
        quads_owned = len(owned)

        def tile_at(t):
            return tiles[t[1]][t[0]]

        def is_animal(tl):
            return isinstance(tl, dict) and "animal" in tl

        market = []          # (priority, op); serialized 10/turn from hour 0

        # 1. Feed wheat, exact need from the REAL animal census.
        feed_need = sum(1 for row in tiles for tl in row if is_animal(tl))
        reserve = feed_need + int(g["feed_buy"])
        wheat_short = max(0, reserve - int(shed.get("WHEAT", 0)))
        if wheat_short and d > 0 and feed_need:
            market.append((0, ["BUY_PRODUCT", "WHEAT", wheat_short]))
            money -= wheat_short * 40
        fbuy = int(g.get("fert_buy", 0))
        if fbuy > 0 and 1 <= d <= 24 and money > 600:
            n = min(fbuy, int((money - 500) // 110))
            if n > 0:
                market.append((4, ["BUY_PRODUCT", "FERTILIZER", n]))
                money -= n * 110

        # 2. Land: try every day once past the planned day; REAL cash gate.
        if quads_owned <= 3:
            next_q = LAND_ORDER[quads_owned - 1]
            if next_q in unlock_plan and d >= unlock_plan[next_q]:
                cost = LAND_COST[quads_owned - 1]
                if money >= cost + 300:
                    market.append((2, ["BUY_LAND"]))
                    money -= cost
                    owned.add(next_q)

        # 3+4 moved BELOW the task builder (2026-08-31): capital priority
        # is CROPS > ANIMALS > HANDS. The debug ledger showed dawn money
        # pinned at $4-532 through day 12 -- hires and animal purchases
        # were eating the planting capital while 12 units idled 80-185
        # ops/day. Seeds now reserve their money first.

        # 5. Tasks from the REAL board: every owned tile is examined; what
        # STANDS on it drives tending (the actual crop/animal, never the
        # plan); the want-map only drives NEW planting. Empty tiles beyond
        # the plan grow BACKFILL WHEAT so spare labor becomes crops
        # (kaileh plants 246/season vs our 186 with the same op budget).
        # Priorities: feed 0, tend 1, place 1, dig 2, plan-plant 3,
        # backfill 4 -- standing plants are never starved by expansion.
        tasks = []
        seed_need = {}
        seed_avail = dict(seeds)
        fert_left = int(shed.get("FERTILIZER", 0))
        for y in range(BOARD):
            for x in range(BOARD):
                t = (x, y)
                if t in SHED_TILES or quad_of(t) not in owned:
                    continue
                tl = tile_at(t)
                sp = want.get(t)
                if is_animal(tl):
                    ops = [["FEED"]]
                    if tl.get("yield_units", 0) > 0:
                        ops.append(["HARVEST"])
                    if g.get("fertilize") and tl.get("fertilizer_available"):
                        ops.append(["COLLECT_FERTILIZER"])
                    if g.get("care"):
                        ops.append(["CARE"])
                    tasks.append((t, ops, 0))
                    continue
                if isinstance(tl, dict) and tl.get("kind") == "WEED":
                    tasks.append((t, [["DIG"]], 2))
                    continue
                if isinstance(tl, dict) and tl.get("kind") in ("COOP",
                                                               "PASTURE"):
                    if sp in ANIMALS and int(shed.get(sp, 0)) > 0:
                        tasks.append((t, [["PICKUP", sp, 1],
                                          ["PLACE", sp]], 1))
                        shed[sp] = int(shed[sp]) - 1
                    continue
                if isinstance(tl, dict) and tl.get("kind") == "PLANT":
                    crop = tl["crop"]
                    spc = ECROPS[crop]
                    age = d - tl.get("planted_day", d)
                    yld = tl.get("yield_units", 0)
                    ops = []
                    fert_ok = (g.get("fert_crops") and fert_left > 0
                               and tl.get("fertilized_until_day", -1) < d
                               and (crop in ("MELON", "STRAWBERRY", "TOMATO")
                                    or (crop == "WHEAT"
                                        and g.get("fert_wheat"))))
                    if spc["ongoing"]:
                        last = spc["first_yield_day"] \
                            + spc["interval"] * (spc["max_yield"] - 1)
                        if yld > 0 and age >= spc["first_yield_day"]:
                            ops.append(["HARVEST"])
                        if age < last:
                            if not tl.get("watered_today"):
                                ops.append(["WATER"])
                            if fert_ok and \
                                    spc["first_yield_day"] - 1 <= age:
                                ops.insert(0, ["FERTILIZE"])
                                fert_left -= 1
                        elif not ops:
                            ops.append(["DIG"])   # spent: free for replant
                    else:
                        ripe = (yld >= spc["max_yield"]
                                and age >= spc["first_yield_day"]) \
                            or age >= spc["max_yield_day"]
                        if ripe and yld > 0:
                            ops.append(["HARVEST"])
                        else:
                            if not tl.get("watered_today"):
                                ops.append(["WATER"])
                            if fert_ok and \
                                    age == (spc["max_yield_day"] + 1) // 2:
                                ops.insert(0, ["FERTILIZE"])
                                fert_left -= 1
                    if ops:
                        tasks.append((t, ops, 1))
                    continue
                # empty tile
                if sp in ANIMALS:
                    if int(shed.get(sp, 0)) > 0:
                        tasks.append((t, [["PICKUP", sp, 1],
                                          [ANIMALS[sp]["build"]],
                                          ["PLACE", sp]], 1))
                        shed[sp] = int(shed[sp]) - 1
                    else:
                        tasks.append((t, [[ANIMALS[sp]["build"]]], 3))
                    continue
                plant_sp, pri = sp, 3
                if plant_sp is None and g.get("backfill_wheat", True):
                    plant_sp, pri = "WHEAT", 4
                if plant_sp:
                    spc = ECROPS[plant_sp]
                    startd = max(g["start"].get(quad_of(t), 0), 0) \
                        if sp else 0
                    if startd <= d <= g.get("last_plant_day", 24) \
                            - spc["first_yield_day"]:
                        if seed_avail.get(plant_sp, 0) > 0:
                            seed_avail[plant_sp] -= 1
                            tasks.append((t, [["PLANT", plant_sp],
                                              ["WATER"]], pri))
                        elif money >= spc["seed"] + 30:
                            money -= spc["seed"]
                            seed_need[plant_sp] = \
                                seed_need.get(plant_sp, 0) + 1
                            tasks.append((t, [["PLANT", plant_sp],
                                              ["WATER"]], pri))
        for sp, n in sorted(seed_need.items()):
            market.append((4, ["BUY_SEED", sp, n]))

        # 4. Animals, from what remains after seeds.
        need_sp = {}
        for t, sp in want.items():
            if sp in ANIMALS and quad_of(t) in owned \
                    and not is_animal(tile_at(t)):
                need_sp[sp] = need_sp.get(sp, 0) + 1
        for sp in sorted(need_sp):
            spare = int(shed.get(sp, 0))
            missing = need_sp[sp] - spare
            while missing > 0 and money >= ANIMALS[sp]["cost"] + 250:
                market.append((3, ["BUY_ANIMAL", sp, 1]))
                money -= ANIMALS[sp]["cost"]
                animal_ordered[sp] = animal_ordered.get(sp, 0) + 1
                missing -= 1

        # 3. Hires last, exactly affordable (fib resets daily): hands are
        # worthless without funded work.
        n_plan = int(g["hires"][min(d, len(g["hires"]) - 1)])
        fib = [1, 1]
        while len(fib) < max(2, n_plan):
            fib.append(fib[-1] + fib[-2])
        n_h = 0
        for i in range(n_plan):
            c = fib[min(i, len(fib) - 1)]
            if money - c < 30:
                break
            market.append((1, ["HIRE"]))
            money -= c
            n_h += 1

        # 6. Sells: EXACT dawn shed (keep the feed reserve until the end).
        sell_all = d >= days - 2
        caps = g.get("sell_cap") or {}
        holds = g.get("hold_until") or {}
        sold = {}
        if d > 0 and (sell_all
                      or d % max(1, int(g.get("sell_every", 1))) == 0):
            for item in ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG",
                         "CARROT", "TOMATO", "FERTILIZER"):
                if not sell_all and d < int(holds.get(item, 0)):
                    continue
                have = int(shed.get(item, 0))
                if not sell_all:
                    have = min(have, int(caps.get(item, 999)))
                if have > 0:
                    sold[item] = have
            if sell_all or d >= int(holds.get("WHEAT", 0)):
                wsurplus = int(shed.get("WHEAT", 0)) \
                    - (0 if sell_all else reserve)
                if wsurplus > 0:
                    sold["WHEAT"] = sold.get("WHEAT", 0) + wsurplus
        # PRESSURE VALVE: the shed caps at 100 TOTAL and dusk overflow is
        # DISCARDED. If the post-sell total leaves no room for today's
        # harvest, sell past the caps, impact-cheap items first.
        total = sum(int(v) for v in shed.values()) - sum(sold.values())
        if d > 0 and not sell_all and total > 70:
            for item in ("WHEAT", "EGG", "CARROT", "TOMATO", "MILK",
                         "STRAWBERRY", "FERTILIZER", "WOOL", "MELON"):
                room = int(shed.get(item, 0)) - sold.get(item, 0)
                if item == "WHEAT":
                    room -= reserve
                take = min(room, total - 60)
                if take > 0:
                    sold[item] = sold.get(item, 0) + take
                    total -= take
                if total <= 60:
                    break
        for item, q in sorted(sold.items()):
            market.append((6, ["SELL", item, q]))

        # 7. Labor. A PICKUP executes at the unit's CURRENT tile, so any
        # shed withdrawal (fertilizer, animals, feed wheat) is legal only
        # as a unit's FIRST act, before it walks anywhere -- a mid-tour
        # PICKUP is a silent no-op (the bug that voided most FERTILIZEs).
        # Phase A therefore hands each shed job to a FRESH unit: fert
        # tiles are chained per unit behind ONE batched pickup; animal
        # jobs keep their own pickup. Phase B runs the rest greedy,
        # nearest unit first, positions tour-updated.
        units = [_Unit(SHED_TILES[i % 4]) for i in range(1 + n_h)]
        fert_tasks, shed_jobs, rest = [], [], []
        for tk in tasks:
            if any(o[0] == "FERTILIZE" for o in tk[1]):
                fert_tasks.append(tk)
            elif tk[1] and tk[1][0][0] == "PICKUP":
                shed_jobs.append(tk)
            else:
                rest.append(tk)
        if fert_tasks:
            k = max(1, min(len(units) - 1 or 1,
                           (len(fert_tasks) + 5) // 6))
            fert_tasks.sort(key=lambda tk: tk[0])
            for chain in (fert_tasks[i::k] for i in range(k)):
                shed_jobs.append((None, chain, 1))
        ui = 0
        for tile, payload, _pri in shed_jobs:
            if ui >= len(units):
                break
            u = units[ui]
            ui += 1
            if tile is None:                       # batched fert chain
                u.act(["PICKUP", "FERTILIZER", len(payload)])
                for ct, cops, _cp in payload:
                    u.walk_to(ct)
                    for o in cops:
                        u.act(o)
            else:                                  # animal pickup/place
                ops = list(payload)
                while ops and ops[0][0] == "PICKUP":
                    u.act(ops[0])
                    ops = ops[1:]
                u.walk_to(tile)
                for o in ops:
                    u.act(o)
        rest.sort(key=lambda x: (x[2], x[0]))
        dropped = []
        for tile, ops, _pri in rest:
            needs_feed = any(o[0] == "FEED" for o in ops)
            cost = len(ops) + (2 if needs_feed else 0)
            best, bd = None, 10 ** 9
            for u in units:
                dd = _dist(u.pos, tile)
                if u.budget() >= dd + cost and dd < bd:
                    # feed wheat is ALSO a shed withdrawal: only a unit
                    # carrying wheat or still standing on a shed tile may
                    # take a feed tour.
                    if needs_feed and u.carrying_wheat < 1 \
                            and tuple(u.pos) not in SHED_TILES:
                        continue
                    best, bd = u, dd
            if best is None:
                dropped.append((tile, ops, _pri))
                continue
            if needs_feed and best.carrying_wheat < 1:
                take = min(6, max(1, feed_need))
                best.act(["PICKUP", "WHEAT", take])
                best.carrying_wheat += take
            best.walk_to(tile)
            for o in ops:
                best.act(o)
            if needs_feed:
                best.carrying_wheat -= 1
        if debug:
            from collections import Counter as _C
            pris = _C(p for _, _, p in tasks)
            dpris = _C(p for _, _, p in dropped)
            idle = sum(u.budget() for u in units)
            print(f"d{d:>2} money {money:>7,.0f} tasks {dict(pris)} "
                  f"dropped {dict(dpris)} shed_jobs {len(shed_jobs)} "
                  f"units {len(units)} idle-ops {idle}")

        # 8. Serialize the day, then STEP THE ENGINE through it.
        s0 = d * TPD
        queue = [op for _pri, op in sorted(market, key=lambda x: x[0])]
        for k in range(0, len(queue), 10):
            si = s0 + k // 10
            if si < len(tape):
                tape[si]["market"] = queue[k:k + 10]
        for h in range(1, TPD):
            si = s0 + h
            tape[si]["farmer"] = units[0].ops[h - 1] \
                if h - 1 < len(units[0].ops) else ["PASS"]
            tape[si]["hands"] = [
                u.ops[h - 1] if h - 1 < len(u.ops) else ["PASS"]
                for u in units[1:]]
        if sell_all:                 # sweep late-arriving drops, oversized
            for h in range(4, TPD - 1, 4):     # SELL is harmless
                si = s0 + h
                if not tape[si]["market"]:
                    tape[si]["market"] = [
                        ["SELL", it, 30] for it in
                        ("MELON", "STRAWBERRY", "WOOL", "MILK", "EGG",
                         "CARROT", "WHEAT", "FERTILIZER", "TOMATO")][:10]
        for h in range(TPD):
            si = s0 + h
            if si < len(tape) and not env.done:
                a = tape[si]
                env.step([a, a])
    if return_bank:
        return tape, float(env.state[0].observation.farms[0]["money"])
    return tape
