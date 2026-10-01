"""Track-P A2: production-growth packages (the operator class the

single-order searches proved missing, 2026-08-29: three searches, three
negatives -- reallocating an elite schedule's own orders cannot beat it;
only GROWING production can).

A package is a complete commitment (Zhang, disc 738079, inside our gated
harness): the purchase, the unit-op chain, and the sell-through, inserted
as a unit. The synthesis trick that makes it safe on an open-loop tape:
every package-day appends ONE DEDICATED HAND -- an extra HIRE whose whole
day we script from spawn to vanish. It touches no existing unit's recorded
ops, its harvest auto-drops to the shed at day end, and it disappears at
nightfall, so there is nothing to desync. A package whose path assumptions
miss (weed on the tile, different spawn corner) degrades to a small hire
cost and dies in selection -- never to corruption.

v1 operators: MELON and CARROT plant-chains (one-time crops; melon = value,
carrot = the post-1.32.7 repricing). Livestock lines and transplants are
v2 if plant-chains show gains.
"""
import copy
import random

CROP = {
    "MELON":  {"seed_cost": 80, "growth": 10, "water_days": 10, "sell_step_pad": 26},
    "CARROT": {"seed_cost": 20, "growth": 3, "water_days": 3, "sell_step_pad": 26},
}
TURNS_PER_DAY = 24
SHED_SPAWNS = ((5, 4), (4, 5), (5, 5), (4, 4))   # NWSE-preference candidates
BOARD = 10


def _path(a, b):
    """Orthogonal move list a->b (moves are 1 cell/turn, all tiles passable)."""
    (ax, ay), (bx, by) = a, b
    out = []
    while ax < bx:
        out.append(["EAST"]); ax += 1
    while ax > bx:
        out.append(["WEST"]); ax -= 1
    while ay < by:
        out.append(["SOUTH"]); ay += 1
    while ay > by:
        out.append(["NORTH"]); ay -= 1
    return out


def day_state(base_actions, seed=9000):
    """One self-play simulation of the base tape; per-day empty-tile sets
    and per-turn recorded-hands lengths (both seats play the same tape)."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": seed},
               info={"seed": seed})
    env.reset(2)
    empties = {}                     # day -> set of (x, y) empty at day start
    money = []                       # seat-0 cash BEFORE each step's actions
    for step, a in enumerate(base_actions):
        if env.done:
            break
        if step % TURNS_PER_DAY == 0:
            farm = env.state[0].observation.farms[0]
            tiles = farm["tiles"]
            empties[step // TURNS_PER_DAY] = {
                (x, y) for y in range(len(tiles))
                for x in range(len(tiles[y])) if tiles[y][x] is None}
        money.append(float(env.state[0].observation.farms[0]["money"]))
        act = a if isinstance(a, dict) else {}
        env.step([act, act])
    hands_len = [len((a or {}).get("hands") or [])
                 if isinstance(a, dict) else 0 for a in base_actions]
    quads = list(env.state[0].observation.farms[0]["unlocked_quadrants"])
    return {"empties": empties, "hands_len": hands_len, "quads": quads,
            "money": money}


def _append_hand_day(acts, day, program, hands_len):
    """Append one dedicated hand for `day`: HIRE at hour 0, scripted ops
    from hour 1. Pads each touched turn's hands list to its recorded
    length first so the appended op lands at the new hand's index."""
    day0 = day * TURNS_PER_DAY
    s0 = None
    for st in range(day0, min(day0 + 7, len(acts))):
        if isinstance(acts[st], dict) and                 len(acts[st].get("market") or []) < 10:
            s0 = st
            break
    if s0 is None:
        return False
    mk = acts[s0].setdefault("market", [])
    mk.append(["HIRE"])
    for i, op in enumerate(program):
        s = s0 + 1 + i
        if s >= len(acts) or s >= (day + 1) * TURNS_PER_DAY:
            break
        if not isinstance(acts[s], dict):
            acts[s] = {"farmer": ["PASS"], "hands": [], "market": []}
        hands = acts[s].setdefault("hands", [])
        while len(hands) < hands_len[s]:
            hands.append(["PASS"])
        hands.append(op)
    return True


def _quadrant_tiles(q):
    xs = range(5, 10) if q in ("NE", "SE") else range(0, 5)
    ys = range(5, 10) if q in ("SW", "SE") else range(0, 5)
    return [(x, y) for y in ys for x in xs]


LAND_ORDER = ("NE", "SW", "SE")


def gen_land_package(base_actions, state, rng, claimed=frozenset()):
    """BUY_LAND + several plant packages on the newly unlocked quadrant.

    The one space a land-saturated elite base leaves open. The field's
    published '4th quadrant negative' predates carrot repricing and stole
    crew labor; the dedicated-hand pattern adds its own. The paired panels
    judge, as always. Returns (actions, claims') or None."""
    owned = state.get("quads") or ["NW"]
    if len(owned) >= 4 or ("LAND",) in claimed:
        return None
    quadrant = LAND_ORDER[len(owned) - 1]
    cost = (1000, 2000, 4000)[len(owned) - 1]
    # Pick a buy step where the base world actually HOLDS the cash (the
    # first land attempt no-oped: elite economies run near zero at dawn).
    money = state.get("money") or []
    cands = [st for st in range(4 * TURNS_PER_DAY,
                                min(17 * TURNS_PER_DAY, len(money)))
             if money[st] >= cost + 1500
             and isinstance(base_actions[st], dict)
             and len(base_actions[st].get("market") or []) < 10]
    if not cands:
        return None
    s_buy = cands[rng.randrange(len(cands))]
    d_buy = s_buy // TURNS_PER_DAY + 1   # plant only after the unlock day
    if s_buy >= len(base_actions) or not isinstance(base_actions[s_buy], dict):
        return None
    acts = copy.deepcopy(base_actions)
    mk = acts[s_buy].setdefault("market", [])
    if len(mk) >= 10:
        return None
    mk.insert(0, ["BUY_LAND"])
    pool = [t for t in _quadrant_tiles(quadrant) if t not in claimed]
    rng.shuffle(pool)
    claims = set(claimed)
    claims.add(("LAND",))
    added = 0
    for tile in pool:
        if added >= 5:
            break
        d0 = rng.randint(d_buy, max(d_buy, min(22, d_buy + 5)))
        nxt = _plant_at(acts, state, rng, tile, d0)
        if nxt is not None:
            acts = nxt
            claims.add(tile)
            added += 1
    if added < 2:
        return None                      # a bare land buy is pure cost
    return acts, frozenset(claims)


def _plant_at(base_actions, state, rng, tile, d0):
    """One plant package on an EXPLICIT tile (assumed empty from d0)."""
    crop = rng.choice(tuple(CROP))
    spec = CROP[crop]
    last_day = (len(base_actions) // TURNS_PER_DAY) - 2
    if d0 > last_day - spec["growth"] - 1:
        crop, spec = "CARROT", CROP["CARROT"]
        if d0 > last_day - spec["growth"] - 1:
            return None
    spawn = SHED_SPAWNS[0]
    walk = _path(spawn, tile)
    if len(walk) > TURNS_PER_DAY - 4:
        return None
    acts = copy.deepcopy(base_actions)
    if not _append_hand_day(acts, d0, walk + [["PLANT", crop], ["WATER"]],
                            state["hands_len"]):
        return None
    for k in range(1, spec["water_days"]):
        _append_hand_day(acts, d0 + k, walk + [["WATER"]], state["hands_len"])
    _append_hand_day(acts, d0 + spec["growth"], walk + [["HARVEST"]],
                     state["hands_len"])
    s0 = d0 * TURNS_PER_DAY
    acts[s0].setdefault("market", [])
    if len(acts[s0]["market"]) < 10:
        acts[s0]["market"].insert(0, ["BUY_SEED", crop, 1])
    sell_step = min(len(acts) - 2,
                    (d0 + spec["growth"] + 1) * TURNS_PER_DAY + 2)
    if isinstance(acts[sell_step], dict):
        smk = acts[sell_step].setdefault("market", [])
        if len(smk) < 10:
            smk.append(["SELL", crop, 6])
    return acts


def gen_plant_package(base_actions, state, rng, claimed=frozenset()):
    """One crop package: seed buy + dedicated-hand plant/water/harvest days
    + the sell. `claimed` = tiles already taken by earlier packages (the
    base-world empty map cannot see them). Returns (actions, tile) or None."""
    crop = rng.choice(tuple(CROP))
    spec = CROP[crop]
    last_day = (len(base_actions) // TURNS_PER_DAY) - 2
    d0 = rng.randint(8, max(8, min(16, last_day - spec["growth"] - 1)))
    # A tile empty at d0 AND still empty at harvest day in the base world --
    # the base schedule never uses it across the window.
    window_ok = [t for t in state["empties"].get(d0, set())
                 if t not in claimed
                 and all(t in state["empties"].get(d, ())
                        for d in range(d0, min(d0 + spec["growth"] + 1,
                                               max(state["empties"]) + 1))
                        if d in state["empties"])]
    if not window_ok:
        return None
    tile = window_ok[rng.randrange(len(window_ok))]
    spawn = SHED_SPAWNS[0]
    acts = copy.deepcopy(base_actions)
    walk = _path(spawn, tile)
    if len(walk) > TURNS_PER_DAY - 4:
        return None

    # Day d0: walk, PLANT, WATER (planting day counts as unwatered=1).
    if not _append_hand_day(acts, d0, walk + [["PLANT", crop], ["WATER"]],
                            state["hands_len"]):
        return None
    # Water days: one dedicated hand each, daily through the growth window.
    for k in range(1, spec["water_days"]):
        _append_hand_day(acts, d0 + k, walk + [["WATER"]], state["hands_len"])
    # Harvest day: walk, HARVEST; the yield auto-drops to the shed at dusk.
    _append_hand_day(acts, d0 + spec["growth"], walk + [["HARVEST"]],
                     state["hands_len"])
    # The purchase (hour 0 of d0; market resolves after unit ops, and PLANT
    # happens the NEXT day-hour anyway) and the sell-through.
    s0 = d0 * TURNS_PER_DAY
    acts[s0].setdefault("market", [])
    if len(acts[s0]["market"]) < 10:
        acts[s0]["market"].insert(0, ["BUY_SEED", crop, 1])
    sell_step = min(len(acts) - 2,
                    (d0 + spec["growth"] + 1) * TURNS_PER_DAY + 2)
    if isinstance(acts[sell_step], dict):
        smk = acts[sell_step].setdefault("market", [])
        if len(smk) < 10:
            smk.append(["SELL", crop, 6])
    return acts, tile


def mutate_packages(base_actions, state, rng, max_packages=2, tries=12,
                    claimed=frozenset()):
    """Returns (actions, claimed') or None. Claims accumulate across the
    search so stacked packages never collide on a tile."""
    acts = base_actions
    claims = set(claimed)
    added = 0
    # First mutation of a run may open the next quadrant (land package);
    # afterwards, fill remaining quadrant tiles with ordinary packages.
    if ("LAND",) not in claims and rng.random() < 0.5:
        land = gen_land_package(acts, state, rng, frozenset(claims))
        if land is not None:
            return land
    want = rng.randint(1, max_packages)
    for _ in range(tries):
        if added >= want:
            break
        nxt = gen_plant_package(acts, state, rng, frozenset(claims))
        if nxt is not None:
            acts, tile = nxt
            claims.add(tile)
            added += 1
    # With land already bought, also try filling unclaimed quadrant tiles.
    if ("LAND",) in claims:
        owned = state.get("quads") or ["NW"]
        quadrant = LAND_ORDER[len(owned) - 1]
        pool = [t for t in _quadrant_tiles(quadrant) if t not in claims]
        rng.shuffle(pool)
        for tile in pool[:2]:
            d0 = rng.randint(10, 22)
            nxt = _plant_at(acts, state, rng, tile, d0)
            if nxt is not None:
                acts = nxt
                claims.add(tile)
                added += 1
    return (acts, frozenset(claims)) if added else None
