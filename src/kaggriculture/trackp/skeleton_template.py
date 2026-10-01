"""Kaggriculture Track-P closed-loop skeleton planner -- __LABEL__
Built __BUILT__ by src/trackp/build_econ_agent.py (lane: trackp-economy).

Every action is derived, on the turn it is played, from the live observation
and the GENOME below. No tape, no route prefix, no router, no embedded action
sequence. The genome parameterises the FIELD SKELETON (the one farm economy
the whole ladder runs, top-10 included): land days, herd ramp on cashflow,
hire cap, crop tile targets, sell policy. The planner executes it as
CONSTRAINTS on the real board: tasks come from what STANDS on each tile,
seeds are bought only for tiles planted today, cargo (feed wheat,
fertiliser, animals) is picked up at the shed by the unit that carries it,
and every op is validated against the tile the unit is actually standing on
before it is emitted (a failed buy or a late hire degrades to a skipped op,
never to a drifting tour).
"""
import math                                                      # noqa: F401

GENOME = __GENOME__

TPD = 24
DAYS = 30
BOARD = 10
SHED = ((4, 4), (5, 4), (4, 5), (5, 5))
LAND_ORDER = ("NE", "SW", "SE")
LAND_COST = (1000, 2000, 4000)
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
        "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
CROPS = {   # engine 1.32.7 table (vendored source, not a local drift)
    "WHEAT":      {"seed": 10, "first": 2, "maxd": 4, "iv": 0, "maxy": 6, "ong": False},
    "CARROT":     {"seed": 20, "first": 2, "maxd": 3, "iv": 0, "maxy": 4, "ong": False},
    "TOMATO":     {"seed": 50, "first": 8, "maxd": 8, "iv": 1, "maxy": 4, "ong": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "maxd": 10, "iv": 2, "maxy": 4, "ong": True},
    "MELON":      {"seed": 80, "first": 10, "maxd": 12, "iv": 0, "maxy": 6, "ong": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "kind": "COOP", "build": "BUILD_COOP", "product": "EGG"},
    "COW":   {"cost": 400, "kind": "PASTURE", "build": "BUILD_PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "kind": "PASTURE", "build": "BUILD_PASTURE", "product": "WOOL"},
}
SELL_ORDER = ("MELON", "STRAWBERRY", "WOOL", "MILK", "FERTILIZER", "EGG",
              "TOMATO", "CARROT", "WHEAT")
# priorities: 0 feed/tend animals, 1 tend/harvest crops, 2 place animals,
# 3 dig weeds, 4 plant strawberry/melon, 5 plant wheat
_S = {}


def _get(o, k, d=None):
    try:
        v = o[k]
        return d if v is None else v
    except (KeyError, TypeError, IndexError):
        return getattr(o, k, d)


def _quad(t):
    return ("N" if t[1] < 5 else "S") + ("W" if t[0] < 5 else "E")


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shed_dist(p):
    return min(_dist(p, s) for s in SHED)


def _nearest_shed(p):
    return min(SHED, key=lambda s: _dist(p, s))


def _move(p, t):
    if p[0] < t[0]:
        return ["EAST"]
    if p[0] > t[0]:
        return ["WEST"]
    if p[1] < t[1]:
        return ["SOUTH"]
    return ["NORTH"]


def _target(sched, d):
    n = 0
    for dd, nn in sched:
        if d >= int(dd):
            n = int(nn)
    return n


def _fib_sum(n):
    a, b, s = 1, 1, 0
    for _ in range(n):
        s += a
        a, b = b, a + b
    return s


def _inv_total(inv):
    return sum(int(v) for v in inv.values()) if inv else 0


# ------------------------------------------------------------------ sells

def _sell_orders(g, S, d, shed, prices, keep, final):
    out = []
    for item in SELL_ORDER:
        have = int(shed.get(item, 0))
        if not final:
            have -= int(keep.get(item, 0))
        if have <= 0:
            continue
        if not final:
            if d < int(g["hold_until"].get(item, 0)):
                continue
            p = float(prices.get(item, BASE[item]))
            if p < float(g["sell_floor"].get(item, 0.0)) * BASE[item]:
                continue
            cap = g["sell_cap"].get(item)
            if cap is not None:
                have = min(have, int(cap) - S["sold_today"].get(item, 0))
            if item == "MELON":
                have = min(have, int(g["melon_cap"]) - S["melon_sold"])
        if have > 0:
            out.append(["SELL", item, int(have)])
    out.sort(key=lambda o: -o[2] * float(prices.get(o[1], BASE[o[1]])))
    return out


def _book_sells(S, orders):
    for o in orders:
        if o[0] == "SELL":
            S["sold_today"][o[1]] = S["sold_today"].get(o[1], 0) + o[2]
            if o[1] == "MELON":
                S["melon_sold"] += o[2]


# ------------------------------------------------------------------ dawn

def _dawn(g, S, d, farm, priv, mkt, town):
    """Plan the day: market queue (priority-ordered, 10/turn spill) and one
    ordered job list per unit, from the REAL dawn board."""
    money = float(_get(farm, "money", 0))
    tiles = _get(farm, "tiles", [])
    shed = {k: int(v) for k, v in (dict(_get(priv, "shed", {}) or {})).items()}
    seeds = {k: int(v) for k, v in (dict(_get(priv, "seeds", {}) or {})).items()}
    owned = set(_get(farm, "unlocked_quadrants", ["NW"]) or ["NW"])
    prices = {k: float(v) for k, v in (dict(_get(mkt, "prices", {}) or {})).items()}
    final = d >= DAYS - 1
    S["sold_today"] = {}
    S["waits"] = {}

    def tl_at(t):
        try:
            return tiles[t[1]][t[0]]
        except (IndexError, TypeError):
            return "LOCKED"

    # ---- census -------------------------------------------------------
    animals, plants, weeds, empty, structs = [], [], [], [], []
    for y in range(BOARD):
        for x in range(BOARD):
            t = (x, y)
            if _quad(t) not in owned:
                continue
            tl = tl_at(t)
            if tl == "LOCKED":
                continue
            if tl is None:
                empty.append(t)
            elif isinstance(tl, dict) and "animal" in tl:
                animals.append((t, tl))
            elif isinstance(tl, dict) and tl.get("kind") == "PLANT":
                plants.append((t, tl))
            elif isinstance(tl, dict) and tl.get("kind") == "WEED":
                weeds.append(t)
            elif isinstance(tl, dict):
                structs.append((t, tl))          # empty coop/pasture
    n_animals = len(animals)
    have_sp = {}
    for _t, tl in animals:
        have_sp[tl["animal"]] = have_sp.get(tl["animal"], 0) + 1
    for sp in ANIMALS:
        have_sp[sp] = have_sp.get(sp, 0) + shed.get(sp, 0)

    # ---- dawn sells + cash estimate ----------------------------------
    fert_need = 0
    fert_ages = tuple(int(a) for a in g["fert_straw_ages"])
    for _t, tl in plants:
        if tl.get("crop") == "STRAWBERRY" and g["fert_straw"]:
            age = d - int(tl.get("planted_day", d))
            if age in fert_ages and int(tl.get("fertilized_until_day", -1)) < d:
                fert_need += 1
    feed_days = int(g["feed_days"])
    feed_today = n_animals if (not final or g["feed_last_day"]) else 0
    reserve_w = feed_today * feed_days + int(g["feed_buffer"]) if feed_today else 0
    keep = {"WHEAT": reserve_w, "FERTILIZER": fert_need}
    for sp in ANIMALS:
        keep[sp] = 10 ** 6
    sells = _sell_orders(g, S, d, shed, prices, keep, final)
    dawn_sells = sells[: int(g["dawn_sell_slots"])]
    income = sum(o[2] * prices.get(o[1], BASE[o[1]]) * float(g["income_haircut"])
                 for o in dawn_sells)
    avail = money + income
    reserve_cash = float(g["cash_reserve"])

    q = []   # (priority, order)
    # feed wheat: exact shortfall, first in the queue when dawn cash covers it
    wheat_short = max(0, reserve_w - shed.get("WHEAT", 0))
    if wheat_short > 0 and d > 0:
        cost = wheat_short * (prices.get("WHEAT", 25) + 3)
        q.append((0 if money >= cost else 2, ["BUY_PRODUCT", "WHEAT", wheat_short]))
        avail -= cost
    for o in dawn_sells:
        q.append((1, o))
    _book_sells(S, dawn_sells)

    # ---- land ---------------------------------------------------------
    new_quads = set()
    if len(owned) <= 3:
        nq = LAND_ORDER[len(owned) - 1]
        ld = int(g["land"].get(nq, -1))
        if 0 <= ld <= d and not final:
            cost = LAND_COST[len(owned) - 1]
            if avail >= cost + float(g["land_reserve"]):
                q.append((4, ["BUY_LAND"]))
                avail -= cost
                owned.add(nq)
                new_quads.add(nq)
                for y in range(BOARD):
                    for x in range(BOARD):
                        if _quad((x, y)) == nq:
                            empty.append((x, y))

    # ---- animals on cashflow, by schedule ----------------------------
    empty.sort(key=lambda t: (_shed_dist(t), t[1], t[0]))
    buy_sp = {}
    if d <= int(g["animal_last_day"]):
        for sp, sched in (("COW", g["cows"]), ("SHEEP", g["sheep"]),
                          ("GOOSE", g["geese"])):
            tgt = _target(sched, d)
            while have_sp.get(sp, 0) + buy_sp.get(sp, 0) < tgt \
                    and avail >= ANIMALS[sp]["cost"] + reserve_cash \
                    and len(empty) + len(structs) > sum(buy_sp.values()):
                buy_sp[sp] = buy_sp.get(sp, 0) + 1
                avail -= ANIMALS[sp]["cost"]
    for sp, n in sorted(buy_sp.items()):
        for _ in range(n):
            q.append((5, ["BUY_ANIMAL", sp, 1]))
    # tiles reserved for the herd ramp of the next two days
    ahead = 0
    for sp, sched in (("COW", g["cows"]), ("SHEEP", g["sheep"]),
                      ("GOOSE", g["geese"])):
        ahead += max(0, _target(sched, d + 2) - have_sp.get(sp, 0) - buy_sp.get(sp, 0))

    # ---- tasks from what stands on every tile -------------------------
    tasks = []           # (pri, tile, ops, cargo dict)
    fert_left = shed.get("FERTILIZER", 0)
    fert_wheat_ok = (g["fert_wheat"] and prices.get("FERTILIZER", 100)
                     < float(g["fert_wheat_below"]))
    for t, tl in animals:
        ops, cargo = [], {}
        if feed_today and not tl.get("fed_today"):
            ops.append(["FEED"])
            cargo["WHEAT"] = 1
        if g["care"] and feed_today and not tl.get("cared_today"):
            ops.append(["CARE"])
        if int(tl.get("yield_units", 0)) >= int(g["harvest_at"]) or \
                (final and int(tl.get("yield_units", 0)) > 0):
            ops.append(["HARVEST"])
        if tl.get("fertilizer_available"):
            ops.append(["COLLECT_FERTILIZER"])
        if ops:
            tasks.append((0, t, ops, cargo))
    straw_fert_tiles = []
    wheat_fert_tiles = []
    n_straw_alive = 0
    for t, tl in plants:
        crop = tl.get("crop", "WHEAT")
        c = CROPS.get(crop, CROPS["WHEAT"])
        age = d - int(tl.get("planted_day", d))
        yld = int(tl.get("yield_units", 0))
        unw = int(tl.get("consecutive_unwatered", 0))
        watered = bool(tl.get("watered_today"))
        ops, cargo = [], {}
        replant = False
        if crop == "STRAWBERRY":
            n_straw_alive += 1
            eves = (c["first"] - 1 + c["iv"] * k for k in range(c["maxy"]))
            last_prod = c["first"] + c["iv"] * (c["maxy"] - 1)
            if yld > 0 and age >= c["first"]:
                ops.append(["HARVEST"])
            if age >= last_prod:
                if yld <= 0 or age > last_prod:
                    ops.append(["DIG"])
                    replant = True
            else:
                if g["fert_straw"] and age in fert_ages and fert_left > 0 \
                        and int(tl.get("fertilized_until_day", -1)) < d:
                    ops.insert(0, ["FERTILIZE"])
                    cargo["FERTILIZER"] = 1
                    fert_left -= 1
                    straw_fert_tiles.append(t)
                if not watered and (age in tuple(eves) or unw >= 1):
                    ops.append(["WATER"])
        elif crop == "TOMATO":
            last_prod = c["first"] + c["iv"] * (c["maxy"] - 1)
            if yld > 0 and age >= c["first"]:
                ops.append(["HARVEST"])
            if age > last_prod or (age >= last_prod and yld <= 0):
                ops.append(["DIG"])
                replant = True
            elif not watered:
                ops.append(["WATER"])
        else:   # one-shot: WHEAT / CARROT / MELON
            win0 = (c["maxd"] + 1) // 2
            ripe = (yld >= c["maxy"] and age >= c["first"]) or age >= c["maxd"]
            if not watered and not ripe and (win0 <= age <= c["maxd"] or unw >= 1):
                ops.append(["WATER"])
                if crop == "WHEAT" and fert_wheat_ok and age == win0 \
                        and fert_left > 0 and int(tl.get("fertilized_until_day", -1)) < d:
                    ops.insert(0, ["FERTILIZE"])
                    cargo["FERTILIZER"] = 1
                    fert_left -= 1
                    wheat_fert_tiles.append(t)
            if ripe or (final and yld > 0 and age >= c["first"]):
                if not watered and win0 <= age <= c["maxd"] and yld < c["maxy"] \
                        and ["WATER"] not in ops:
                    ops.append(["WATER"])
                ops.append(["HARVEST"])
                replant = True
        if replant and d <= int(g["wheat_last_plant"]) and not final:
            ops.append(["PLANT", "WHEAT"])
            ops.append(["WATER"])
            cargo["SEED_WHEAT"] = 1
        if ops:
            tasks.append((1, t, ops, cargo))
    for t in weeds:
        ops = [["DIG"]]
        cargo = {}
        if d <= int(g["wheat_last_plant"]) and not final:
            ops += [["PLANT", "WHEAT"], ["WATER"]]
            cargo["SEED_WHEAT"] = 1
        tasks.append((3, t, ops, cargo))

    # ---- placements + new plantings on empty tiles ---------------------
    place = []
    for sp, n in sorted(buy_sp.items()):
        place += [sp] * n
    for sp in ANIMALS:                                # animals already in shed
        place += [sp] * shed.get(sp, 0)
    used = set()
    for sp in place:
        spot = None
        for t, tl in structs:
            if t not in used and tl.get("kind") == ANIMALS[sp]["kind"]:
                spot = t
                break
        if spot is not None:
            used.add(spot)
            tasks.append((2, spot, [["PLACE", sp]], {sp: 1}))
            continue
        for t in empty:
            if t not in used:
                spot = t
                break
        if spot is None:
            break
        used.add(spot)
        tasks.append((2, spot, [[ANIMALS[sp]["build"]], ["PLACE", sp]], {sp: 1}))
    free = [t for t in empty if t not in used]
    free = free[ahead:] if ahead > 0 else free          # herd reservation
    plan_new = {"MELON": 0, "STRAWBERRY": 0, "WHEAT": 0, "CARROT": 0}
    if not final:
        melon_room = int(g["melon_tiles"]) - S["melon_planted"]
        if d <= int(g["melon_last_plant"]) and melon_room > 0:
            plan_new["MELON"] = min(melon_room, int(g["melon_per_day"]))
        straw_room = int(g["straw_tiles"]) - S["straw_planted"]
        if d <= int(g["straw_last_plant"]) and straw_room > 0:
            plan_new["STRAWBERRY"] = min(straw_room, int(g["straw_per_day"]))
        carrot_room = int(g["carrot_tiles"]) - S["carrot_planted"]
        if d <= int(g["carrot_last_plant"]) and carrot_room > 0:
            plan_new["CARROT"] = min(carrot_room, int(g["carrot_per_day"]))
        if d <= int(g["wheat_last_plant"]):
            plan_new["WHEAT"] = int(g["wheat_wave"])
    # cash-gate the expensive seeds (seeds in hand are free)
    seed_free = dict(seeds)
    for crop in ("STRAWBERRY", "MELON", "CARROT"):
        n = plan_new[crop]
        can = seed_free.get(crop, 0)
        while can < n and avail - CROPS[crop]["seed"] >= reserve_cash:
            avail -= CROPS[crop]["seed"]
            can += 1
        plan_new[crop] = min(n, can)
    fi = 0
    for crop in ("MELON", "STRAWBERRY", "CARROT", "WHEAT"):
        pri = 5 if crop == "WHEAT" else 4
        for _ in range(plan_new[crop]):
            if fi >= len(free):
                break
            tasks.append((pri, free[fi], [["PLANT", crop], ["WATER"]],
                          {"SEED_" + crop: 1}))
            fi += 1

    # ---- labour: sequential fill, one ordered tour per unit -----------
    tasks.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
    caps = g["hands_cap"]
    max_hands = int(caps[min(d, len(caps) - 1)])
    budget_h = int(g["hand_budget"]) if not final else int(g["final_budget"]) - 1
    budget_f = TPD - 1 if not final else int(g["final_budget"])
    units = [{"pos": (4, 4), "left": budget_f, "jobs": [], "cargo": {}, "n": 0}]
    for i in range(max_hands):
        units.append({"pos": SHED[(i + 1) % 4], "left": budget_h, "jobs": [],
                      "cargo": {}, "n": 0})
    pool = []
    ui = 0
    for pri, t, ops, cargo in tasks:
        placed = False
        while ui < len(units):
            u = units[ui]
            extra = sum(1 for k in cargo if k in ("WHEAT", "FERTILIZER") or k in ANIMALS
                        if k not in u["cargo"])
            cost = _dist(u["pos"], t) + len(ops) + extra
            if u["left"] >= cost:
                u["jobs"].append([t, [list(o) for o in ops], 0])
                for k, v in cargo.items():
                    u["cargo"][k] = u["cargo"].get(k, 0) + v
                u["pos"] = t
                u["left"] -= cost
                u["n"] += len(ops)
                placed = True
                break
            ui += 1
        if not placed:
            pool.append([t, [list(o) for o in ops], cargo])
    n_used = sum(1 for u in units[1:] if u["n"] > 0)
    # hires: exactly the opened hands, cash-checked (fib resets daily)
    n_hire = n_used
    while n_hire > 0 and _fib_sum(n_hire) > avail - reserve_cash:
        n_hire -= 1
    avail -= _fib_sum(n_hire)
    for _ in range(n_hire):
        q.append((3, ["HIRE"]))
    live_units = units[: 1 + n_hire]
    for u in units[1 + n_hire:]:
        for j in u["jobs"]:
            pool.append([j[0], j[1], {}])
    # seeds: only for plantings that a live unit will execute today
    seed_need = {}
    for u in live_units:
        for k, v in u["cargo"].items():
            if k.startswith("SEED_"):
                seed_need[k[5:]] = seed_need.get(k[5:], 0) + v
    for crop, n in sorted(seed_need.items()):
        short = n - seeds.get(crop, 0)
        if short > 0:
            q.append((6 if crop != "WHEAT" else 5, ["BUY_SEED", crop, short]))
    # cargo pickups become each unit's first ops (units spawn at the shed)
    for u in live_units:
        pre = []
        for k in ("WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE"):
            n = u["cargo"].get(k, 0)
            if n > 0:
                pre.append(["PICKUP", k, int(n)])
        if pre:
            u["jobs"].insert(0, ["SHED", pre, 0])
    fert_short = max(0, sum(u["cargo"].get("FERTILIZER", 0) for u in live_units)
                     - shed.get("FERTILIZER", 0))
    if fert_short > 0 and g["fert_buy"]:
        q.append((7, ["BUY_PRODUCT", "FERTILIZER", fert_short]))
    for u in live_units:
        for j in u["jobs"]:
            if j[0] != "SHED":
                for o in j[1]:
                    if o[0] == "PLANT":
                        S[o[1].lower() + "_planted"] = S.get(o[1].lower() + "_planted", 0) + 1

    q.sort(key=lambda x: x[0])
    orders = [o for _p, o in q]
    S["spill"] = [orders[i:i + 10] for i in range(0, len(orders), 10)]
    S["jobs"] = [u["jobs"] for u in live_units]
    S["pool"] = pool
    S["keep"] = keep
    S["day"] = d


# ------------------------------------------------------------------ exec

def _valid(op, tl, pos, inv, ctx, unit_jobs):
    """1 = emit, 0 = skip this op, -1 = wait one turn (max 2), -2 = requeued."""
    k = op[0]
    if k == "PICKUP":
        if pos not in SHED:
            return 0
        if ctx["shed"].get(op[1], 0) <= 0:
            return -1
        return 1
    if k == "DROP":
        return 1 if (pos in SHED and _inv_total(inv) > 0) else 0
    if tl == "LOCKED":
        return -1
    if k == "PLANT":
        if tl is not None:
            return 0
        if ctx["seeds"].get(op[1], 0) <= 0:
            return -1 if ctx["h"] <= 2 else 0
        return 1
    if k == "WATER":
        return 1 if (isinstance(tl, dict) and tl.get("kind") == "PLANT"
                     and not tl.get("watered_today")) else 0
    if k == "HARVEST":
        if not isinstance(tl, dict) or int(tl.get("yield_units", 0)) <= 0:
            return 0
        if tl.get("kind") == "PLANT":
            c = CROPS.get(tl.get("crop"))
            if c and ctx["d"] - int(tl.get("planted_day", 0)) < c["first"]:
                return 0
        return 1
    if k == "FEED":
        if not (isinstance(tl, dict) and "animal" in tl) or tl.get("fed_today"):
            return 0
        if inv.get("WHEAT", 0) > 0:
            return 1
        if ctx["shed"].get("WHEAT", 0) > 0:
            n = sum(1 for j in unit_jobs for o in j[1] if o[0] == "FEED")
            unit_jobs.insert(0, ["SHED", [["PICKUP", "WHEAT", max(1, n)]], 0])
            return -2
        return 0
    if k == "CARE":
        return 1 if (isinstance(tl, dict) and "animal" in tl
                     and not tl.get("cared_today")) else 0
    if k == "COLLECT_FERTILIZER":
        return 1 if (isinstance(tl, dict) and "animal" in tl
                     and tl.get("fertilizer_available")) else 0
    if k == "FERTILIZE":
        if not (isinstance(tl, dict) and tl.get("kind") == "PLANT") \
                or int(tl.get("fertilized_until_day", -1)) >= ctx["d"]:
            return 0
        return 1 if inv.get("FERTILIZER", 0) > 0 else 0
    if k == "DIG":
        if tl is None or (isinstance(tl, dict) and "animal" in tl):
            return 0
        return 1
    if k in ("BUILD_PASTURE", "BUILD_COOP"):
        if tl is None:
            return 1
        if isinstance(tl, dict) and tl.get("kind") in ("PASTURE", "COOP") \
                and "animal" not in tl:
            return 0
        return 0
    if k == "PLACE":
        sp = op[1]
        if not (isinstance(tl, dict) and tl.get("kind") == ANIMALS[sp]["kind"]
                and "animal" not in tl):
            return 0
        if inv.get(sp, 0) > 0:
            return 1
        if ctx["shed"].get(sp, 0) > 0:
            unit_jobs.insert(0, ["SHED", [["PICKUP", sp, 1]], 0])
            return -2
        return 0
    return 1


def _grab(S, g, pos, inv, hours_left, ctx):
    """Nearest feasible unassigned task for an idle unit."""
    best, bi, bd = None, -1, 10 ** 9
    for i, (t, ops, cargo) in enumerate(S["pool"]):
        need = [k for k in cargo if k in ("WHEAT", "FERTILIZER") or k in ANIMALS]
        if any(inv.get(k, 0) <= 0 for k in need):
            continue
        if any(ctx["seeds"].get(k[5:], 0) <= 0 for k in cargo if k.startswith("SEED_")):
            continue
        dd = _dist(pos, t)
        if dd + len(ops) > hours_left:
            continue
        if dd < bd:
            best, bi, bd = (t, ops), i, dd
    if best is None:
        return None
    S["pool"].pop(bi)
    return [best[0], best[1], 0]


def _step_unit(S, g, d, h, ui, pos, inv, ctx):
    jobs = S["jobs"][ui] if ui < len(S["jobs"]) else None
    if jobs is None:
        S["jobs"].append([])
        jobs = S["jobs"][ui]
    final = d >= DAYS - 1
    hours_left = (int(g["final_budget"]) if final else TPD) - h
    guard = 0
    while guard < 40:
        guard += 1
        if not jobs:
            job = _grab(S, g, pos, inv, hours_left, ctx)
            if job is None:
                if _inv_total(inv) > 0 and (final or hours_left <= _shed_dist(pos) + 1
                                            or _inv_total(inv) >= int(g["drop_at"])):
                    if pos in SHED:
                        return ["DROP"]
                    return _move(pos, _nearest_shed(pos))
                return ["PASS"]
            jobs.append(job)
        job = jobs[0]
        t = job[0]
        if t == "SHED":
            if pos not in SHED:
                return _move(pos, _nearest_shed(pos))
            tl = None
        else:
            if pos != t:
                # mid-tour drop: heavy cargo of produce, shed adjacent
                if _inv_total(inv) >= int(g["drop_at"]) and _shed_dist(pos) == 0 \
                        and not any(k in inv for k in ("WHEAT", "FERTILIZER", "COW", "SHEEP", "GOOSE")
                                    if any(o[0] in ("FEED", "FERTILIZE", "PLACE")
                                           for j in jobs for o in j[1])):
                    return ["DROP"]
                return _move(pos, t)
            try:
                tl = ctx["tiles"][t[1]][t[0]]
            except (IndexError, TypeError):
                tl = "LOCKED"
        ops = job[1]
        if not ops:
            jobs.pop(0)
            continue
        op = ops[0]
        v = _valid(op, tl, pos, inv, ctx, jobs)
        if v == -2:
            continue
        if v == -1:
            job[2] = job[2] + 1
            if job[2] <= 2:
                return ["PASS"]
            jobs.pop(0)
            continue
        if v == 0:
            ops.pop(0)
            continue
        ops.pop(0)
        if op[0] == "PLANT":
            ctx["seeds"][op[1]] = ctx["seeds"].get(op[1], 0) - 1
        elif op[0] == "PICKUP":
            n = min(int(op[2]), ctx["shed"].get(op[1], 0))
            ctx["shed"][op[1]] = ctx["shed"].get(op[1], 0) - n
            return ["PICKUP", op[1], int(n)]
        return op
    return ["PASS"]


def agent(obs, cfg=None):
    step = int(_get(obs, "step", 0))
    d, h = step // TPD, step % TPD
    me = int(_get(obs, "player", 0))
    farms = _get(obs, "farms", [])
    farm = farms[me] if me < len(farms) else {}
    priv = _get(obs, "private", {}) or {}
    mkt = _get(obs, "market", {}) or {}
    town = _get(obs, "town", {}) or {}
    S = _S.get(me)
    if S is None or step == 0:
        S = {"day": -1, "jobs": [], "pool": [], "spill": [], "keep": {},
             "sold_today": {}, "melon_sold": 0, "melon_planted": 0,
             "straw_planted": 0, "carrot_planted": 0, "wheat_planted": 0}
        _S[me] = S
    try:
        g = GENOME
        if S["day"] != d:
            _dawn(g, S, d, farm, priv, mkt, town)
        shed = {k: int(v) for k, v in (dict(_get(priv, "shed", {}) or {})).items()}
        seeds = {k: int(v) for k, v in (dict(_get(priv, "seeds", {}) or {})).items()}
        prices = {k: float(v) for k, v in (dict(_get(mkt, "prices", {}) or {})).items()}
        ctx = {"tiles": _get(farm, "tiles", []), "shed": shed, "seeds": seeds,
               "d": d, "h": h}
        # market: spilled dawn orders first, then fresh sells from the live shed
        market = list(S["spill"].pop(0)) if S["spill"] else []
        if h > 0 or not market:
            final = d >= DAYS - 1
            have = {o[1] for o in market if o[0] == "SELL"}
            for o in _sell_orders(g, S, d, shed, prices, S["keep"], final):
                if len(market) >= 10:
                    break
                if o[1] not in have:
                    market.append(o)
                    _book_sells(S, [o])
        market = market[:10]
        invs = _get(priv, "inventories", []) or []
        poss = [tuple(_get(farm, "farmer", (4, 4)))]
        for hp in (_get(farm, "hands", []) or []):
            poss.append(tuple(hp))
        acts = []
        for ui, pos in enumerate(poss):
            inv = invs[ui] if ui < len(invs) and invs[ui] else {}
            acts.append(_step_unit(S, g, d, h, ui, pos, dict(inv), ctx))
        return {"farmer": acts[0], "hands": acts[1:], "market": market}
    except Exception:
        n = len(_get(farm, "hands", []) or [])
        return {"farmer": ["PASS"], "hands": [["PASS"]] * n, "market": []}
