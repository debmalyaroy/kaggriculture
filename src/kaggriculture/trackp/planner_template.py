"""Track P planner -- L2 projections / L1 macro-policy / L0 executor.

FRESH CODE (Track P isolation rule): shares nothing with the bandit/route
agents. Single self-contained file, imports only math. The submission
contract: agent(obs) -> {"farmer": [...], "hands": [[...]], "market": [[...]]}.

Layers:
  L2  price-at-harvest projection: engine quote curve (exact transcription)
      + deterministic town drain from the OBSERVED shop draw + the opponent
      dump residual (recovered exactly from public market inventory deltas)
      + optional family prior spliced in by the builder.
  L1  macro decisions at day boundaries and shop-unlock turns: production
      mix over 8 products, labour target, investment depth, sell pacing.
      v0 = hand rules with elite priors; an MLP can replace the rules via
      the L1 sentinel block (P3.5 export) -- same features, same bins.
  L0  the executor: prices every candidate job in dollars using L2, assigns
      units greedily, keeps watering/feeding discipline as hard constraints,
      emits at most 10 priority-ordered market orders.

Fail-soft property: a wrong L1 choice only re-weights L0's job pricing; it
cannot emit an illegal action (L0 validates everything against the rules).
"""
import math

# --- PARAMS BEGIN ---
PARAMS = {
    # labour (top-10: aggressive crew -- winners run 11-13 hands by D6)
    "labour_base": 3,            # hands/day floor once economy is running
    "labour_max": 13,            # hard cap on hands per day (top teams: 11-13)
    "labour_ramp_day": 3,        # start ramping labour after this day
    "labour_per_tiles": 9.0,     # 1 extra hand per this many active tiles
    # ---- proven opening (from top-10 turn-by-turn analysis 2026-09-15) ----
    # Top-10 canonical D0 open (verbatim across teams): all-in, zero reserve --
    # BUILD_PASTURE 4 + BUY_COW 2 + BUY_SHEEP 2 + PLANT_MELON 12 + PLANT_WHEAT 7
    # + HIRE 5. Then FEED+CARE every animal every day, sell premium continuously,
    # herd wave at D6. The closed-loop planner replans per-turn AFTER open_days.
    "open_tape_days": 6,         # replay the OPENING_TAPE through this day (D0..D5)
    # re-plan / adapt checkpoints (shop-unlock & key decision days) -- the macro
    # economy (tilt, herd target, sell posture) refreshes at these days + endgame.
    "replan_days": (6, 12, 15, 21, 27),
    "open_days": 3,              # parametric-opening fallback when no OPENING_TAPE
    "open_cow": 2,              # cows bought during the D0 opening
    "open_sheep": 2,            # + sheep -> D6 wool payday funds the herd wave
    "open_melon": 8,            # melon plants in the opening (trim to free D0
                                # cash for animals; animals bootstrap the economy)
    "open_reserve": 30.0,       # opening cash floor (all-in, like the top teams)
    "open_hire": (5, 7, 9),     # hire target by opening day (D0,D1,D2)
    # planting
    "plants_per_unit": 6.0,      # watering capacity guard per unit per day
    "plant_budget_turn": 8,      # max PLANT jobs queued per turn
    "plant_value_margin": 1.10,  # projected value must beat seed cost x this
    "wheat_floor": 4,            # keep at least this many wheat plots (feed)
    "last_plant_margin_days": 1, # stop planting when harvest can't finish
    # investment (elite prior: idle cash is a bug)
    "idle_cash_target": 350.0,   # invest cash above this
    "invest_land_day_ne": 2,     # earliest day to buy NE
    "invest_land_day_sw": 6,
    "invest_land_day_se": 13,    # 4th quad (SE, $4000) -- top teams buy it (100
                                 # tiles for 18 animals + 60 crops); repays for a
                                 # strong economy (moon holds 18 animals on 4 quads)
    "max_coops": 3,
    "max_pastures": 12,   # offline search: smaller herd sells better (don't glut own markets)
    "animal_start_day": 0,       # top teams buy 2 cow + 2 sheep on D0
    "animal_last_day": 21,       # keep growing the herd; reinvest idle cash
    # selling
    "sell_horizon_days": 3.0,    # project prices this far for hold/sell
    "sell_hold_gain": 1.06,      # hold if projected/current exceeds this
    "sell_chunk": 12,            # max units per sell order (price impact)
    "shed_pressure": 78,
    "scarcity_floor": 1.02,      # sell crashable premium only above base*this (scarce)
    "sell_dp_floor": 0.5,        # dump-proof (wheat/egg) sell floor frac of cur
    "sell_prem_floor": 0.7,      # crashable premium sell floor frac (normal)
    "sell_forced_floor": 0.35,   # sell floor frac when terminal/shed-pressure
    "feed_pri": 12000.0,         # FEED survival priority (>urgent water 10000)
    "compact_layout": 1,         # grow farm near shed (offline search: 0 = off better)         # start forced selling above this shed load
    "terminal_day": 27,          # all-out sell mode from this day
    "fert_sell_min": 6,          # sell FERTILIZER above this held count
    # watering / care discipline (elite prior: dry share is a loss marker)
    "water_growth_bonus": 1.35,  # extra weight for growth-window waterings
    "care_value": 0.55,          # weight of CARE jobs vs base product value
    # L2 projection
    "proj_blend_prior": 0.5,     # weight of family prior vs observed rate
    # macro cadence
    "macro_shop_react": 1,       # re-plan on new shop unlock
}
# --- PARAMS END ---

# --- PRIOR BEGIN ---
FAMILY_PRIOR = []  # [(step, product, qty), ...] spliced by the builder
# --- PRIOR END ---

# --- L1 BEGIN ---
L1_WEIGHTS = None  # builder splices {"w1": [...], "b1": [...], ...} or None
# --- L1 END ---

# --- ENGINE BEGIN ---
ENGINE_1327 = False  # builder splices True once the ladder runs >=1.32.7
# --- ENGINE END ---

# --- OPENING BEGIN ---
# A small fixed OPENING TAPE (D0..D6) lifted from a proven top-10 premium team.
# The opening is the most world-INDEPENDENT part of the game (identical empty NW
# quadrant, $3000, same shop schedule), so a recorded opening replays cleanly and
# solves the D1-D8 herd bootstrap trap. The crew is cleared nightly, so at the
# hand-off day the crew is empty anyway -- the tape leaves a built board (pens,
# animals, crops) that the closed-loop planner continues from. Spliced by the
# builder; empty = no opening tape (fall back to the parametric opening).
OPENING_TAPE = []  # [ {"farmer":..,"hands":..,"market":..}, ... ] per step
# --- OPENING END ---

BOARD = 10
TURNS_PER_DAY = 24
EPISODE_STEPS = 720
SHED_CAP = 100
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER"]
CROP_NAMES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
# crop: (seed_cost, first_yield_day, max_yield_day, interval, max_yield, ongoing)
CROPS = {"WHEAT": (10, 2, 4, 0, 6, False), "CARROT": (20, 2, 3, 0, 4, False),
         "TOMATO": (50, 8, 8, 1, 4, True),
         "STRAWBERRY": (100, 10, 10, 2, 4, True),
         "MELON": (80, 10, 12, 0, 6, False)}
# animal: (cost, structure, first_yield_day, interval, max_held, product)
ANIMALS = {"GOOSE": (300, "COOP", 4, 1, 4, "EGG"),
           "COW": (400, "PASTURE", 8, 2, 6, "MILK"),
           "SHEEP": (500, "PASTURE", 6, 3, 6, "WOOL")}
SHOP_PRODUCTS = {"BAKERY": ["EGG", "WHEAT"],
                 "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
                 "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
                 "YARN_STORE": ["WOOL"], "PET_CAFE": ["CARROT"],
                 "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
                 "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
                 "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
MARKET_PARAMS = {
    "WHEAT": (25.0, 10000.0, 400.0, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35.0, 10000.0, 450.0, "log", 0.20, "sqrt", 0.70),
    "TOMATO": (60.0, 10000.0, 200.0, "linear", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120.0, 10000.0, 100.0, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250.0, 10000.0, 300.0, "log", 0.20, "sq", 3.60),
    "EGG": (50.0, 10000.0, 332.0, "linear", 0.40, "log", 0.20),
    "MILK": (160.0, 10000.0, 122.0, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200.0, 10000.0, 105.0, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100.0, 10000.0, 200.0, "linear", 0.40, "linear", 0.40)}
if ENGINE_1327:
    # 1.32.7 hinge rebalance: scarcity side spikes quadratically past T.
    MARKET_PARAMS["CARROT"] = (35.0, 10000.0, 450.0, "hinge", 1.00,
                               "sqrt", 0.70)
    MARKET_PARAMS["TOMATO"] = (60.0, 10000.0, 200.0, "hinge", 0.40,
                               "sqrt", 0.60)
    MARKET_PARAMS["EGG"] = (50.0, 10000.0, 332.0, "hinge", 0.40,
                            "log", 0.20)
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
# Crew self-tracking (the observation NEVER exposes hand positions, and the
# engine CLEARS the whole crew every night, re-hiring from scratch daily). We
# reconstruct the crew from the deterministic engine rules:
#   - shed-access spawn tiles in NWSE order; farmer spawns/resets to (4,4);
#   - a new hand spawns on the first free shed-access tile (min occupancy, NWSE);
#   - movement is pos +/- 1 clamped to the board, no other blocking;
#   - unit count = len(private["inventories"]).
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))   # rules.shed_access_tiles(10)
MOVE_DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def _spawn_hand_pos(farmer_pos, hand_positions):
    """Replicate engine spawn_hand: first free shed-access tile, ties by min
    occupancy then NWSE order."""
    occ = [0, 0, 0, 0]
    for pos in [tuple(farmer_pos)] + [tuple(p) for p in hand_positions]:
        if pos in SHED_ACCESS:
            occ[SHED_ACCESS.index(pos)] += 1
    order = sorted(range(4), key=lambda i: (occ[i], i))
    return list(SHED_ACCESS[order[0]])


def _track_crew(obs, n_hands):
    """Return the tracked (x,y) of each hand this turn, in hire order. The crew
    resets nightly; new hands (n_hands grew) spawn deterministically."""
    day = obs["day"]
    farmer = tuple(obs["farms"][obs["player"]]["farmer"])
    if S.get("crew_day") != day:
        S["crew_day"] = day
        S["hand_pos"] = []
        S["unit_jobs"] = {}   # stable per-unit target commitments (reset nightly)
    hp = S["hand_pos"]
    del hp[n_hands:]                      # crew never shrinks mid-day, but guard
    while len(hp) < n_hands:              # add newly hired hands at their spawn
        hp.append(_spawn_hand_pos(farmer, hp))
    return hp


def _apply_hand_moves(hand_actions):
    """Dead-reckon tracked hand positions from the moves we just issued so they
    are correct next turn (engine: clamp to board, no blocking)."""
    hp = S.get("hand_pos") or []
    for i, act in enumerate(hand_actions):
        if i >= len(hp) or not act:
            continue
        d = MOVE_DELTA.get(act[0])
        if d:
            hp[i][0] = min(BOARD - 1, max(0, hp[i][0] + d[0]))
            hp[i][1] = min(BOARD - 1, max(0, hp[i][1] + d[1]))


def _shape(f, x, t):
    if x < 0.0:
        x = 0.0
    if f == "linear":
        return x
    if f == "sq":
        return x * x
    if f == "sqrt":
        return math.sqrt(x)
    if f == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return math.log(1.0 + x)


def _quote(item, inventory):
    base, i0, t, bf, bt, af, at = MARKET_PARAMS[item]
    if inventory < i0:
        raw = base + bt * base / _shape(bf, t, t) * _shape(bf, i0 - inventory,
                                                           t)
    else:
        raw = base - at * base / _shape(af, t, t) * _shape(af, inventory - i0,
                                                           t)
    fl = math.floor(raw)
    d = raw - fl
    if d > 0.5:
        r = fl + 1.0
    elif d < 0.5:
        r = float(fl)
    else:
        r = float(fl) if int(fl) % 2 == 0 else fl + 1.0
    return 1 if r < 1.0 else int(r)


def _quadrant(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


# ------------------------------------------------------------ turn state --

S = {"ep_key": None}


def _reset_state(seedish):
    S.clear()
    S.update({
        "ep_key": seedish,
        "last_inv": None,
        "opp_cum": {p: 0.0 for p in PRODUCTS},
        "own_sold_last": {p: 0 for p in PRODUCTS},
        "macro": None,
        "macro_day": -1,
        "macro_shops": 0,
        "planned_seeds": {},
    })


def _l2_observe(obs):
    """Recover the opponent dump residual from public inventory deltas."""
    step = obs["step"]
    inv = {p: float(obs["market"]["inventory"].get(p, 0)) for p in PRODUCTS}
    shops = obs["town"]["unlocked_shops"]
    if S["last_inv"] is not None:
        for p in PRODUCTS:
            drain = 0.0
            if step % 4 == 0:
                for sh in shops:
                    prods = SHOP_PRODUCTS.get(sh, ())
                    if p in prods:
                        drain += 2 if len(prods) == 1 else 1
            if step % 24 == 0 and p != "FERTILIZER":
                drain += 1
            delta = inv[p] - S["last_inv"][p]
            opp = delta + drain - S["own_sold_last"].get(p, 0)
            if opp > 0:
                S["opp_cum"][p] += opp
    S["last_inv"] = inv
    S["own_sold_last"] = {p: 0 for p in PRODUCTS}


def _l2_price(obs, product, days_ahead, own_planned=0.0):
    """Projected quoted price of `product` days_ahead from now."""
    inv = float(obs["market"]["inventory"].get(product, 10000))
    shops = obs["town"]["unlocked_shops"]
    drain = 0.0
    for sh in shops:
        prods = SHOP_PRODUCTS.get(sh, ())
        if product in prods:
            drain += (2 if len(prods) == 1 else 1) * (TURNS_PER_DAY // 4)
    if product != "FERTILIZER":
        drain += 1.0
    days = max(1.0, obs["step"] / TURNS_PER_DAY)
    rate = S["opp_cum"][product] / days
    if FAMILY_PRIOR:
        lo = obs["step"]
        hi = lo + days_ahead * TURNS_PER_DAY
        burst = 0.0
        for (t, p, q) in FAMILY_PRIOR:
            if p == product and lo <= t < hi:
                burst += q
        extra = burst - rate * days_ahead
        if extra > 0:
            rate += PARAMS["proj_blend_prior"] * extra / max(0.5, days_ahead)
    proj_inv = inv - drain * days_ahead + rate * days_ahead + own_planned
    return _quote(product, proj_inv)


# -------------------------------------------------------------------- L1 --

def _l1_features(obs):
    """Fixed 32-dim macro feature vector (shared with trackp.macro)."""
    day = obs["day"]
    me = obs["player"]
    farms = obs["farms"]
    mine, theirs = farms[me], farms[1 - me]
    shops = obs["town"]["unlocked_shops"]
    drain = {p: 0.0 for p in PRODUCTS}
    for sh in shops:
        prods = SHOP_PRODUCTS.get(sh, ())
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            drain[p] += m * (TURNS_PER_DAY // 4)
    f = [day / 29.0,
         min(1.0, mine["money"] / 20000.0),
         max(-1.0, min(1.0, (mine["money"] - theirs["money"]) / 10000.0)),
         len(shops) / 8.0,
         len(mine.get("hands", ())) / 12.0,
         len(mine.get("unlocked_quadrants", ("NW",))) / 4.0,
         len(theirs.get("unlocked_quadrants", ("NW",))) / 4.0]
    for p in PRODUCTS:
        f.append(min(1.0, drain[p] / 15.0))
    for p in PRODUCTS:
        base = MARKET_PARAMS[p][0]
        cur = obs["market"]["prices"].get(p, base)
        f.append(max(-1.0, min(1.0, (cur - base) / max(1.0, base))))
    mine_counts = _tile_counts(mine, day)
    their_counts = _tile_counts(theirs, day)
    f.append(min(1.0, mine_counts["plants"] / 40.0))
    f.append(min(1.0, their_counts["plants"] / 40.0))
    f.append(min(1.0, mine_counts["animals"] / 10.0))
    f.append(min(1.0, their_counts["animals"] / 10.0))
    f.append(min(1.0, mine_counts["weeds"] / 20.0))
    f.append(min(1.0, obs["step"] / 719.0))
    f.append(1.0)
    # Opponent embedding (2026-08-16): the opponent's recovered dump rate
    # per product, units/day -- an explicit estimate of THEIR function y,
    # recovered exactly from public market-inventory deltas by _l2_observe.
    # Appended AFTER the bias so all earlier indices stay stable.
    days_seen = max(1.0, obs["step"] / 24.0)
    opp_cum = S.get("opp_cum") or {}
    for p in PRODUCTS:
        f.append(min(1.0, opp_cum.get(p, 0.0) / days_seen / 15.0))
    return f  # 41 dims


def _l1_forward(f):
    """2x256 MLP with tanh, discrete heads -- only if weights are spliced."""
    W = L1_WEIGHTS
    h = f
    for li in (1, 2):
        w, b = W["w%d" % li], W["b%d" % li]
        nh = []
        for j in range(len(b)):
            s = b[j]
            wj = w[j]
            for i in range(len(h)):
                s += wj[i] * h[i]
            nh.append(math.tanh(s))
        h = nh
    out = {}
    off = 0
    wo, bo = W["wo"], W["bo"]
    logits = []
    for j in range(len(bo)):
        s = bo[j]
        wj = wo[j]
        for i in range(len(h)):
            s += wj[i] * h[i]
        logits.append(s)
    for name, k in W["heads"]:
        seg = logits[off:off + k]
        best = 0
        for i in range(1, k):
            if seg[i] > seg[best]:
                best = i
        out[name] = best
        off += k
    return out


def _macro_rules(obs, feats):
    """v0 hand rules with the measured elite priors. Returns bin indices in
    the SAME space the MLP head uses (macro.HEADS)."""
    day = obs["day"]
    shops = obs["town"]["unlocked_shops"]
    drain = {p: 0.0 for p in PRODUCTS}
    for sh in shops:
        prods = SHOP_PRODUCTS.get(sh, ())
        m = 2 if len(prods) == 1 else 1
        for p in prods:
            drain[p] += m * 6.0
    # mix weights: projected price x demand support, per product bins 0-4
    mix = {}
    for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
              "EGG", "MILK", "WOOL"):
        px = _l2_price(obs, p, 4.0)
        base = MARKET_PARAMS[p][0]
        score = (px / base) + 0.08 * drain[p]
        mix[p] = max(0, min(4, int(round(score * 1.8 - 0.8))))
    # WHEAT floor: feed stock for animals
    mix["WHEAT"] = max(mix["WHEAT"], 2)
    # UNCONTESTED TOMATO commitment. Tomato is the ONE premium we produce zero
    # of, so in pizza/farmers-demand worlds it sells at scarcity price (~$640/u,
    # the 1.32.7 hinge) -- worth +$50k/world (the frontier's key edge). Commit
    # hard once pizza demand shows, planting through D20 so it matures (fyd=8,
    # ongoing) and sells D26-29. Deferred to D8+ so the world has resolved.
    pizza = sum(1 for s in shops if s in ("PIZZA_SHOP", "FARMERS_MARKET"))
    if pizza >= 1 and 8 <= day <= 20:
        mix["TOMATO"] = 4
    elif day > 20:
        mix["TOMATO"] = 0            # too late to mature -- stop planting tomato
    # REACT TO OPPONENT (produce the COMPLEMENT). Deep scarcity price needs us to
    # be the MAIN seller of a product; a market both of us flood stays glutted at
    # base. So back off the crops the opponent floods and lean into the ones they
    # under-produce (those stay scarce for us -> the hinge premium). opp_cum is
    # their cumulative sells per product, recovered from public inventory deltas.
    opp = S.get("opp_cum", {})
    contest_crops = ("CARROT", "TOMATO", "STRAWBERRY", "MELON")
    tot = sum(opp.get(p, 0) for p in contest_crops)
    if tot > 30:                     # enough opponent signal to react to
        for p in contest_crops:
            share = opp.get(p, 0) / tot
            if share > 0.40:         # they flood it -> contested, back off
                mix[p] = max(0, mix[p] - 2)
            elif share < 0.08:       # they ignore it -> stays scarce for us
                mix[p] = min(4, mix[p] + 1)
    # labour target bin 0-6 -> hands 0,2,3,4,6,8,10
    me = obs["player"]
    active = 0
    for row in obs["farms"][me]["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") not in ("WEED",):
                active += 1
    if day < 1:
        lab = 1
    elif day < PARAMS["labour_ramp_day"]:
        lab = 2
    else:
        lab = max(2, int(PARAMS["labour_base"] // 2)) \
            + int(active / PARAMS["labour_per_tiles"] / 2.0) \
            + (1 if day >= 10 else 0)
    lab = min(6, lab)
    # investment depth 0-3: how aggressively to convert cash to assets;
    # idle cash above target is a bug (elite prior), so cash pressure bumps it
    inv_depth = 3 if day <= 12 else (2 if day <= 18 else
                                     (1 if day <= 24 else 0))
    if (obs["farms"][me]["money"] > 3 * PARAMS["idle_cash_target"]
            and day <= 20):
        inv_depth = min(3, inv_depth + 1)
    # sell pacing 0-3: 0 hold-biased .. 3 dump-now
    pace = 1
    if day >= PARAMS["terminal_day"]:
        pace = 3
    return {"mix_WHEAT": mix["WHEAT"], "mix_CARROT": mix["CARROT"],
            "mix_TOMATO": mix["TOMATO"], "mix_STRAWBERRY": mix["STRAWBERRY"],
            "mix_MELON": mix["MELON"], "mix_EGG": mix["EGG"],
            "mix_MILK": mix["MILK"], "mix_WOOL": mix["WOOL"],
            "labour": lab, "invest": inv_depth, "pace": pace}


def _macro(obs):
    day = obs["day"]
    shops = len(obs["town"]["unlocked_shops"])
    if (S["macro"] is not None and S["macro_day"] == day
            and (not PARAMS["macro_shop_react"] or S["macro_shops"] == shops)):
        return S["macro"]
    feats = _l1_features(obs)
    if L1_WEIGHTS is not None:
        m = _l1_forward(feats)
    else:
        m = _macro_rules(obs, feats)
    S["macro"] = m
    S["macro_day"] = day
    S["macro_shops"] = shops
    return m


# -------------------------------------------------------------------- L0 --

def _tile_counts(farm, day):
    plants = animals = weeds = 0
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    plants += 1
                elif k == "WEED":
                    weeds += 1
                elif "animal" in t:
                    animals += 1
    return {"plants": plants, "animals": animals, "weeds": weeds}


def _scan(farm, day):
    """One pass over our tiles -> job source lists."""
    waters, harvests, feeds, cares, collects, ferts = [], [], [], [], [], []
    weeds, empties, empty_structs = [], [], []
    plants_alive = 0
    for y in range(BOARD):
        row = farm["tiles"][y]
        for x in range(BOARD):
            t = row[x]
            if t is None:
                if _quadrant(x, y) in farm["unlocked_quadrants"]:
                    empties.append((x, y))
                continue
            if t == "LOCKED" or not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "WEED":
                weeds.append((x, y))
            elif k == "PLANT":
                plants_alive += 1
                cd = CROPS[t["crop"]]
                age = day - t["planted_day"]
                if not t["watered_today"]:
                    urgent = t["consecutive_unwatered"] >= 1
                    win_lo = (cd[2] + 1) // 2
                    growth = (not cd[5]) and win_lo <= age <= cd[2]
                    waters.append((x, y, t["crop"], urgent, growth))
                if t["yield_units"] > 0 and age >= cd[1]:
                    harvests.append((x, y, t["crop"], t["yield_units"]))
                if (not cd[5]) and t.get("fertilized_until_day", -1) < day:
                    win_lo = (cd[2] + 1) // 2
                    if win_lo <= age <= cd[2]:
                        ferts.append((x, y, t["crop"]))
            elif "animal" in t:
                ad = ANIMALS[t["animal"]]
                if not t["fed_today"]:
                    # carry consecutive_unfed: >=1 means the animal ESCAPES if
                    # not fed today (must-do); ==0 means feeding is optional
                    # (production only) -- lets the crew harvest instead.
                    feeds.append((x, y, t["animal"],
                                  int(t.get("consecutive_unfed", 0))))
                if not t["cared_today"]:
                    cares.append((x, y, t["animal"]))
                if t.get("fertilizer_available"):
                    collects.append((x, y))
                if t["yield_units"] > 0:
                    harvests.append((x, y, ad[5], t["yield_units"]))
            elif k in ("COOP", "PASTURE"):
                empty_structs.append((x, y, k))
    # Grow the farm COMPACTLY around the central shed (param-gated so the offline
    # search can settle it): order empty tiles by distance to the shed so pens
    # (feed trips) and crops cluster near the shed -- cuts the ~54% travel.
    if PARAMS.get("compact_layout", 1):
        empties.sort(key=lambda p: (p[0] - 4.5) ** 2 + (p[1] - 4.5) ** 2)
    return {"waters": waters, "harvests": harvests, "feeds": feeds,
            "cares": cares, "collects": collects, "ferts": ferts,
            "weeds": weeds, "empties": empties,
            "empty_structs": empty_structs, "plants_alive": plants_alive}


def _step_toward(pos, target):
    px, py = pos
    tx, ty = target
    if px < tx:
        return ["EAST"]
    if px > tx:
        return ["WEST"]
    if py < ty:
        return ["SOUTH"]
    if py > ty:
        return ["NORTH"]
    return None


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_shed(pos):
    best = SHED_TILES[0]
    bd = _dist(pos, best)
    for t in SHED_TILES[1:]:
        d = _dist(pos, t)
        if d < bd:
            best, bd = t, d
    return best


def agent(obs):
    try:
        return _agent(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


def _agent(obs):
    obs = dict(obs)
    step = obs.get("step", 0)
    day = obs.get("day", step // TURNS_PER_DAY)
    hour = obs.get("hour", step % TURNS_PER_DAY)
    obs["step"], obs["day"], obs["hour"] = step, day, hour
    # OPENING TAPE: replay the proven D0..D6 opening verbatim, then hand off to
    # the closed-loop planner (crew is cleared nightly, so the hand-off is clean).
    # opening tape is HARD-CAPPED at D6 -- the adaptive planner must carry D6->D30
    tape_steps = min(6, int(PARAMS.get("open_tape_days", 0))) * TURNS_PER_DAY
    if OPENING_TAPE and step < min(tape_steps, len(OPENING_TAPE)):
        act = OPENING_TAPE[step]
        return {"farmer": act.get("farmer") or ["PASS"],
                "hands": list(act.get("hands") or []),
                "market": list(act.get("market") or [])[:10]}
    me = obs.get("player", 0)
    farm = obs["farms"][me]
    priv = obs["private"]
    shed = priv.get("shed", {})
    seeds = priv.get("seeds", {})
    invs = priv.get("inventories", [{}])
    money = float(farm["money"])

    if S.get("ep_key") is None or step == 0 or step < S.get("last_step", -1):
        _reset_state("run0")
    S["last_step"] = step
    _l2_observe(obs)
    macro = _macro(obs)
    scan = _scan(farm, day)
    days_left = 29 - day

    # Crew = farmer + hands. farm["hands"] is ALWAYS empty in the obs (hand
    # positions are hidden), so the true unit count is len(inventories) and the
    # hand positions must be self-tracked. This is the fix for the planner
    # running the whole farm on ONE worker.
    n_units = max(1, len(invs))
    hand_pos = _track_crew(obs, n_units - 1)
    units = [tuple(farm["farmer"])] + [tuple(p) for p in hand_pos]
    n_units = len(units)
    while len(invs) < n_units:
        invs.append({})

    # ---------------- job board (each job: (priority$, tile, op)) ----------
    jobs = []
    # watering: hard discipline. Urgent = dies tonight.
    for (x, y, crop, urgent, growth) in scan["waters"]:
        cd = CROPS[crop]
        px = _l2_price(obs, crop, max(1.0, cd[1] / 2.0))
        val = px * (1.4 if growth else 0.7) * (
            PARAMS["water_growth_bonus"] if growth else 1.0)
        if urgent:
            val += 10000.0
        jobs.append((val, (x, y), ["WATER"], None))
    # harvest: units x projected price now
    for (x, y, prod, n) in scan["harvests"]:
        px = _l2_price(obs, prod, 0.5)
        jobs.append((px * n, (x, y), ["HARVEST"], None))
    # feed: preserves the animal's future production stream
    minv_now = obs["market"]["inventory"]
    for (x, y, animal, unfed) in scan["feeds"]:
        ad = ANIMALS[animal]
        prod_px = _l2_price(obs, ad[5], 1.0)
        future = max(0, min(days_left, 10)) / max(1, ad[3])
        wheat_px = _quote("WHEAT", minv_now.get("WHEAT", 10000))
        val = prod_px * min(4.0, future) - wheat_px
        if future >= 1:
            if unfed >= 1:
                # SURVIVAL feed: escapes today if unfed. Out-ranks urgent WATER
                # (10000) -- a lost cow ($80/day forever) dwarfs a dying crop.
                pri = PARAMS["feed_pri"]
            else:
                # PRODUCTION feed (fed yesterday, safe today): worth the future
                # yield but competes normally with harvest so the crew isn't
                # starved of selling. This is the every-other-day efficiency.
                pri = max(60.0, val)
            jobs.append((pri, (x, y), ["FEED"], "WHEAT"))
    # care: +1 product on next fed production
    for (x, y, animal) in scan["cares"]:
        ad = ANIMALS[animal]
        if days_left >= ad[3]:
            px = _l2_price(obs, ad[5], float(ad[3]))
            jobs.append((px * PARAMS["care_value"], (x, y), ["CARE"], None))
    # collect fertilizer: item worth its sale price or +1 yield use
    for (x, y) in scan["collects"]:
        px = _l2_price(obs, "FERTILIZER", 1.0)
        jobs.append((px * 0.9, (x, y), ["COLLECT_FERTILIZER"], None))
    # fertilize growth-window crops if a unit carries FERTILIZER
    for (x, y, crop) in scan["ferts"]:
        px = _l2_price(obs, crop, 2.0)
        jobs.append((px * 0.8, (x, y), ["FERTILIZE"], "FERTILIZER"))
    # planting per macro mix
    plant_plan = _plant_plan(obs, macro, scan, seeds, days_left, n_units)
    for (crop, (x, y)) in plant_plan:
        cd = CROPS[crop]
        horizon = float(cd[1])
        px = _l2_price(obs, crop, horizon)
        est_units = cd[4] * 0.7
        val = px * est_units - cd[0]
        if val > cd[0] * (PARAMS["plant_value_margin"] - 1.0):
            jobs.append((val * 0.9, (x, y), ["PLANT", crop], None))
    # weeds: clear when there is slack
    for (x, y) in scan["weeds"][:6]:
        jobs.append((25.0, (x, y), ["DIG"], None))
    # place bought animals standing in shed
    carried_animals = {}
    for inv in invs:
        for a in ANIMALS:
            if inv.get(a, 0) > 0:
                carried_animals[a] = carried_animals.get(a, 0) + inv[a]
    for (x, y, kind) in scan["empty_structs"]:
        for a, (cost, st, fyd, ivl, mh, prod) in ANIMALS.items():
            if st == kind and (carried_animals.get(a, 0) > 0
                               or shed.get(a, 0) > 0):
                px = _l2_price(obs, prod, float(fyd))
                if day <= PARAMS["animal_last_day"]:
                    jobs.append((px * 3.0, (x, y), ["PLACE", a], a))
                break
    # build structures per investment plan
    builds = _build_plan(obs, macro, scan, day, money)
    for (x, y, op, val) in builds:
        jobs.append((val, (x, y), [op], None))

    jobs.sort(key=lambda j: -j[0])

    # ------------------------------- assignment ----------------------------
    unit_actions = [None] * n_units
    taken_tiles = set()
    # STABLE COMMITMENT: a unit keeps walking to the job it was assigned last
    # turn until it arrives, instead of being re-routed to a newly-cheaper job
    # every turn (that zigzag is the ~54% travel tax). We only re-plan a unit
    # when its target job is DONE/GONE. Validity = the (tile, op-family) is still
    # a live job this turn.
    job_at = {}
    for (val, tile, op, need) in jobs:
        job_at.setdefault(tile, (op, need))
    prior = S.get("unit_jobs") or {}
    for ui in range(n_units):
        tgt = prior.get(ui)
        if not tgt:
            continue
        tile, op, need = tgt
        if tile in taken_tiles or tile not in job_at or job_at[tile][0] != op:
            continue                         # target done/gone/taken -> re-plan
        if not _can_get(invs, ui, need, shed):
            continue
        if units[ui] == tile and _has_item(invs, ui, need, shed, units, ui):
            unit_actions[ui] = op            # arrived -> do it
        else:
            unit_actions[ui] = _fetch_or_move(units[ui], tile, need, invs,
                                              ui, shed)
        taken_tiles.add(tile)
    new_jobs = {}
    for ui in range(n_units):
        if unit_actions[ui] is not None:
            t = prior.get(ui)
            if t:
                new_jobs[ui] = t
    # 1. jobs a unit is already standing on (no travel cost)
    for ji, (val, tile, op, need) in enumerate(jobs):
        if tile in taken_tiles:
            continue
        for ui in range(n_units):
            if unit_actions[ui] is not None or units[ui] != tile:
                continue
            if not _has_item(invs, ui, need, shed, units, ui):
                continue
            unit_actions[ui] = op
            taken_tiles.add(tile)
            new_jobs[ui] = (tile, op, need)
            break
    # 2. remaining jobs -> nearest free unit walks toward them (and COMMITS)
    for (val, tile, op, need) in jobs:
        if tile in taken_tiles:
            continue
        best_ui, best_d = -1, 999
        for ui in range(n_units):
            if unit_actions[ui] is not None:
                continue
            if need and not _can_get(invs, ui, need, shed):
                continue
            d = _dist(units[ui], tile)
            if d < best_d:
                best_ui, best_d = ui, d
        if best_ui < 0:
            continue
        mv = _fetch_or_move(units[best_ui], tile, need, invs, best_ui, shed)
        unit_actions[best_ui] = mv
        taken_tiles.add(tile)
        new_jobs[best_ui] = (tile, op, need)
    S["unit_jobs"] = new_jobs
    # 3. idle units: drop carried goods at shed / walk home
    for ui in range(n_units):
        if unit_actions[ui] is not None:
            continue
        carried = sum(v for k, v in invs[ui].items())
        if carried > 0 and (hour >= 20 or carried >= 6
                            or day >= PARAMS["terminal_day"]):
            if units[ui] in SHED_TILES:
                unit_actions[ui] = ["DROP"]
            else:
                unit_actions[ui] = _step_toward(
                    units[ui], _nearest_shed(units[ui])) or ["PASS"]
        else:
            unit_actions[ui] = ["PASS"]

    # ------------------------------- market --------------------------------
    market = _market_orders(obs, macro, scan, shed, seeds, money, day,
                            days_left, n_units, plant_plan)

    farmer_action = unit_actions[0] or ["PASS"]
    hands_actions = [a or ["PASS"] for a in unit_actions[1:]]
    # dead-reckon the hands' new positions from the moves we just issued
    _apply_hand_moves(hands_actions)
    return {"farmer": farmer_action, "hands": hands_actions,
            "market": market[:10]}


def _has_item(invs, ui, need, shed, units, ui2):
    if not need:
        return True
    return invs[ui].get(need, 0) > 0


def _can_get(invs, ui, need, shed):
    if not need:
        return True
    return invs[ui].get(need, 0) > 0 or shed.get(need, 0) > 0


def _fetch_or_move(pos, tile, need, invs, ui, shed):
    """Walk to the job; detour via the shed first if an item is required."""
    if need and invs[ui].get(need, 0) <= 0:
        if shed.get(need, 0) > 0:
            if pos in SHED_TILES:
                # Carry a STACK of wheat so one shed trip feeds several animals
                # -- the crew is 63% travel-bound; amortising the shed round-trip
                # over many feeds is the biggest efficiency lever.
                grab = min(int(shed.get(need, 0)), 6) if need == "WHEAT" else 1
                return ["PICKUP", need, max(1, grab)]
            return _step_toward(pos, _nearest_shed(pos)) or ["PASS"]
        return ["PASS"]
    mv = _step_toward(pos, tile)
    return mv or ["PASS"]


def _plant_plan(obs, macro, scan, seeds, days_left, n_units):
    """Which crops on which empty tiles this turn, capacity-guarded."""
    day = obs["day"]
    out = []
    empties = list(scan["empties"])
    if not empties:
        return out
    cap = int(PARAMS["plants_per_unit"] * n_units)
    if scan["plants_alive"] >= cap:
        return out
    budget_tiles = min(len(empties), cap - scan["plants_alive"],
                       PARAMS["plant_budget_turn"])
    # deadline: a crop planted today must reach first yield before day 29
    prefs = []
    for c in CROP_NAMES:
        cd = CROPS[c]
        if day + cd[1] + PARAMS["last_plant_margin_days"] > 29:
            continue
        w = macro.get("mix_%s" % c, 2)
        if w <= 0:
            continue
        prefs.append((w, c))
    prefs.sort(reverse=True)
    have = {c: seeds.get(c, 0) for c in CROP_NAMES}
    ei = 0
    for (w, c) in prefs:
        n = min(w * 2, have.get(c, 0))
        for _ in range(n):
            if ei >= budget_tiles:
                return out
            # crops take the OUTER tiles (harvested less often than animals are
            # fed); pens get the inner ring near the shed (built from the front).
            out.append((c, empties[len(empties) - 1 - ei]))
            ei += 1
    return out


def _build_plan(obs, macro, scan, day, money):
    """COOP/PASTURE builds on empty tiles, per investment depth."""
    out = []
    inv_depth = macro.get("invest", 2)
    if inv_depth <= 0 or day > PARAMS["animal_last_day"]:
        return out
    n_coops = n_pastures = 0
    me = obs["player"]
    for row in obs["farms"][me]["tiles"]:
        for t in row:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "COOP" or (isinstance(t, dict) and t.get("animal")
                                   and ANIMALS.get(t.get("animal"),
                                                   (0, ""))[1] == "COOP"):
                    n_coops += 1
                elif k == "PASTURE" or (t.get("animal")
                                        and ANIMALS.get(t.get("animal"),
                                                        (0, ""))[1]
                                        == "PASTURE"):
                    n_pastures += 1
    empties = scan["empties"]
    ei = 0  # pens take the inner ring nearest the shed (frequent feed trips)
    # JUST-IN-TIME lockstep: the killer bug was building pens far faster than we
    # could BUY (and FEED) animals -- 12 empty pens by D1 drained cash so no $420
    # cow could be bought and the few animals starved. So never build a new pen
    # while empty pens are still waiting to be stocked. An empty pen (built but no
    # animal, incl. one whose animal is already bought and sitting in the shed) is
    # counted; we allow only a small in-flight buffer, then wait for BUY_ANIMAL to
    # fill them. Each pen must also leave cash for the animal that fills it.
    shed = (obs.get("private") or {}).get("shed") or {}
    empty_past = sum(1 for (_, _, k) in scan["empty_structs"] if k == "PASTURE")
    empty_coop = sum(1 for (_, _, k) in scan["empty_structs"] if k == "COOP")
    shed_pasture_animals = shed.get("COW", 0) + shed.get("SHEEP", 0)
    shed_coop_animals = shed.get("GOOSE", 0)
    coop_cost = 60.0
    past_cost = 60.0
    opening = day < PARAMS["open_days"]
    reserve = PARAMS["open_reserve"] if opening else 250.0
    # a pen is only worth building if we can also afford to stock it
    past_gate = past_cost + 460.0 + reserve   # ~= pasture + a cow/sheep + reserve
    coop_gate = coop_cost + 300.0 + reserve
    # max empty pens allowed in flight: grow the herd hard whenever cash is
    # flush (idle cash is a bug -- the top teams run ~18 animals at ~$0 cash),
    # stay lockstep-tight when broke so we never strand empty pens.
    buffer = 6 if money > 2500 else (4 if opening else 2)
    max_builds = 4 if (opening or money > 2500) else 3   # pens started per turn
    built = 0
    # PASTURES first (cow/sheep = milk+wool, the top-10 premium engine); the
    # canonical D0 open builds 4 pastures. Coops (geese) are secondary.
    in_flight_p = empty_past  # pens without an animal (shed animals will fill some)
    if day >= PARAMS["animal_start_day"]:
        while (n_pastures < PARAMS["max_pastures"] and ei < len(empties)
               and built < max_builds and money > past_gate
               and in_flight_p < buffer):
            x, y = empties[ei]
            out.append((x, y, "BUILD_PASTURE", 900.0))
            ei += 1; n_pastures += 1; built += 1; money -= past_cost
            in_flight_p += 1
    in_flight_c = empty_coop
    if day >= PARAMS["animal_start_day"] + 1:
        while (n_coops < PARAMS["max_coops"] and ei < len(empties)
               and built < max_builds and money > coop_gate
               and in_flight_c < buffer):
            x, y = empties[ei]
            out.append((x, y, "BUILD_COOP", 850.0))
            ei += 1; n_coops += 1; built += 1; money -= coop_cost
            in_flight_c += 1
    return out


def _market_orders(obs, macro, scan, shed, seeds, money, day, days_left,
                   n_units, plant_plan):
    orders = []
    inv = obs["market"]["inventory"]
    pace = macro.get("pace", 1)
    terminal = day >= PARAMS["terminal_day"]

    # 1. HIRE up to the labour target (hands are daily, cheap early)
    lab_bins = (0, 2, 3, 5, 7, 9, 12)
    target = min(lab_bins[min(macro.get("labour", 2), 6)],
                 PARAMS["labour_max"])
    if terminal:
        target = max(target, 6)
    # Proven opening: hire aggressively to the top-10 schedule (D0->5, D1->7,
    # D2->9), then keep ramping toward labour_max -- the herd needs hands to
    # feed/care/harvest or it produces nothing.
    if day < PARAMS["open_days"]:
        target = max(target, PARAMS["open_hire"][min(day, 2)])
    hires_needed = max(0, target - (n_units - 1))
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    hires_today = obs["farms"][obs["player"]].get("hires_today", 0)
    cash = money
    for i in range(min(hires_needed, 5)):
        c = fib[min(hires_today + i, 9)]
        if cash < c + 60:
            break
        orders.append(["HIRE"])
        cash -= c

    # 2. BUY_LAND per schedule when affordable
    quads = len(obs["farms"][obs["player"]].get("unlocked_quadrants", ["NW"]))
    if quads - 1 < len(LAND_ORDER):
        price = LAND_PRICES[quads - 1]
        gate_day = (PARAMS["invest_land_day_ne"], PARAMS["invest_land_day_sw"],
                    PARAMS["invest_land_day_se"])[quads - 1]
        if day >= gate_day and cash >= price + 150 and not terminal:
            orders.append(["BUY_LAND"])
            cash -= price

    # 3. BUY_SEED for the macro mix (keep a small planting pipeline).
    # OPENING: reserve the animal budget first -- seeds must not starve the herd
    # (the D1-D8 bootstrap trap). Animals are what compound; melon can wait a turn.
    seed_reserve = 0.0
    if day < PARAMS["open_days"]:
        me = obs["player"]
        placed = sum(1 for r in obs["farms"][me]["tiles"] for t in r
                     if isinstance(t, dict) and t.get("animal") in ("COW", "SHEEP"))
        have_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)
        still_need = max(0, (PARAMS["open_cow"] + PARAMS["open_sheep"])
                         - placed - have_shed)
        seed_reserve = still_need * 470.0
    if not terminal:
        for c in CROP_NAMES:
            cd = CROPS[c]
            if day + cd[1] + PARAMS["last_plant_margin_days"] > 29:
                continue
            want = macro.get("mix_%s" % c, 0) * 2
            have = seeds.get(c, 0)
            n = min(max(0, want - have), 6)
            cost = cd[0] * n
            if n > 0 and cash - cost >= 100 + seed_reserve:
                orders.append(["BUY_SEED", c, n])
                cash -= cost

    # 4. BUY_ANIMAL -- fill EVERY empty pen each turn (lockstep with builds so we
    # never sit on empty pens = the "17 pens, 5 animals, no milk/wool" bug), COW
    # and SHEEP allocated by macro demand (milk vs wool = the top-10 premium
    # engine). During the opening also secure the canonical 2 cow + 2 sheep quota
    # even before pens finish (they wait in the shed, placed as pastures come up).
    if not terminal and day <= PARAMS["animal_last_day"]:
        me = obs["player"]
        placed = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
        for row in obs["farms"][me]["tiles"]:
            for t in row:
                if isinstance(t, dict) and t.get("animal") in placed:
                    placed[t["animal"]] += 1
        empty_pastures = sum(1 for (_, _, k) in scan["empty_structs"]
                             if k == "PASTURE")
        empty_coops = sum(1 for (_, _, k) in scan["empty_structs"]
                          if k == "COOP")
        margin = 20.0 if day < PARAMS["open_days"] else 120.0
        # WORLD-CONDITIONED herd tilt. CRITICAL (operator): NEVER buy sheep in a
        # non-YARN world -- WOOL is consumed ONLY by YARN_STORE, so without it wool
        # crashes to ~$1 and sheep are dead weight. Likewise geese only pay in
        # egg-demand (BAKERY/BRUNCH) worlds. MILK is the most broadly demanded
        # premium (PIZZA/ICE_CREAM/SMOOTHIE), so COW is the safe default and the
        # right pick before the world is known (shops unlock D3/D6).
        shops = obs["town"]["unlocked_shops"]
        yarn = sum(1 for s in shops if s == "YARN_STORE")
        milk_sh = sum(1 for s in shops
                      if s in ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"))
        egg_sh = sum(1 for s in shops if s in ("BAKERY", "BRUNCH_SPOT"))
        wool_px = _quote("WOOL", inv.get("WOOL", 10000))
        milk_px = _quote("MILK", inv.get("MILK", 10000))
        wool_score = (2 * yarn + wool_px / 200.0) if yarn > 0 else 0.0
        dairy_score = milk_sh + milk_px / 160.0
        want_cow = want_sheep = 0
        for _ in range(empty_pastures):
            cs = placed["COW"] + want_cow
            ss = placed["SHEEP"] + want_sheep
            # sheep ONLY when YARN present and wool demand out-scores dairy
            if yarn > 0 and wool_score > dairy_score and ss <= cs:
                want_sheep += 1
            else:
                want_cow += 1
        # GEESE regardless of world (operator): EGG is DUMP-PROOF (log price
        # curve barely moves on a glut), so geese are a stable income floor no
        # opponent can crash -- a defensive anti-dump anchor in EVERY world. Fill
        # the coops always (egg also feeds BAKERY/BRUNCH when present).
        want_goose = empty_coops
        for a, cnt in (("COW", want_cow), ("SHEEP", want_sheep),
                       ("GOOSE", want_goose)):
            cost, st, fyd, ivl, mh, prod = ANIMALS[a]
            if days_left <= fyd + ivl:
                continue
            cnt -= shed.get(a, 0)  # already waiting in shed for placement
            for _ in range(max(0, cnt)):
                if cash < cost + margin:
                    break
                orders.append(["BUY_ANIMAL", a, 1])
                cash -= cost

    # 5. BUY_PRODUCT WHEAT for feed when short
    n_animals = sum(1 for r in obs["farms"][obs["player"]]["tiles"]
                    for t in r if isinstance(t, dict) and t.get("animal"))
    if n_animals and shed.get("WHEAT", 0) < n_animals and not terminal:
        px = _quote("WHEAT", inv.get("WHEAT", 10000) - 1)
        n = min(max(0, n_animals - shed.get("WHEAT", 0)), 8)  # keep herd fed
        if n > 0 and cash >= px * n + 80:
            orders.append(["BUY_PRODUCT", "WHEAT", n])
            cash -= px * n

    # 6. SELL: pacing via projection; terminal sweep; shed pressure
    shed_load = sum(shed.get(p, 0) for p in shed)
    sells = []
    for p in PRODUCTS:
        have = int(shed.get(p, 0))
        if p == "WHEAT" and not terminal:
            have -= max(PARAMS["wheat_floor"], n_animals)  # feed reserve
        if p == "FERTILIZER" and not terminal:
            have = have - PARAMS["fert_sell_min"] if \
                have > PARAMS["fert_sell_min"] else 0
        if have <= 0:
            continue
        minv = int(inv.get(p, 10000))
        cur = _quote(p, minv)
        base = MARKET_PARAMS[p][0]
        pressure = shed_load >= PARAMS["shed_pressure"]
        forced = terminal or pace >= 3 or pressure
        # THE SCARCITY RULE (1.32.7 hinge): a market is worth selling into ONLY
        # while it is SCARCE (inventory below I0=10000 -> price above base). Selling
        # PUSHES inventory up and the price DOWN, so dumping a crashable premium
        # gluts our own market and we sell it BELOW base (the measured MILK $107 <
        # $160 leak). So: sell crashable premium only above base, and meter so we
        # never push the market into glut. Dump-proof wheat/egg/fert (log curve)
        # barely move -> sell freely. terminal/shed-pressure forces liquidation.
        if p in ("WHEAT", "EGG"):
            floor = 0.0 if forced else PARAMS["sell_dp_floor"] * cur
            cap = have
            glut_stop = False
        else:
            if not forced and cur <= base * PARAMS["scarcity_floor"]:
                continue                      # glutted -> hold; shops re-scarce it
            floor = (PARAMS["sell_forced_floor"] * cur if forced
                     else max(base, PARAMS["sell_prem_floor"] * cur))
            cap = have if forced else min(have, 40)
            glut_stop = not forced
        n = 0
        for j in range(cap):
            price = _quote(p, minv + j)
            if not forced and price < floor:
                break
            if glut_stop and minv + j >= 10000:   # never push a premium into glut
                break
            n += 1
        if n > 0:
            sells.append((cur * n, ["SELL", p, int(n)]))
    sells.sort(key=lambda s: -s[0])
    room = 10 - len(orders)
    orders.extend(o for _, o in sells[:room])

    # keep total sold this turn for the L2 residual bookkeeping
    for o in orders:
        if o[0] == "SELL":
            S["own_sold_last"][o[1]] = S["own_sold_last"].get(o[1], 0) + o[2]
    return orders
