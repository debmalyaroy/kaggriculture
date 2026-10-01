"""Kaggriculture agent v1 -- heuristic planner.

Self-contained: this file can be submitted directly as `main.py`.

Pipeline, once per turn
-----------------------
    obs
     -> CENSUS        what does the farm hold right now?
     -> TILE PLAN     what should each *empty* tile become? (budget-aware)
     -> TASKS         every actionable job on the farm, priced in dollars
     -> ASSIGNMENT    greedy value-density matching of units to tasks
     -> MARKET        hire / sell / feed / seeds / land / livestock
     -> action dict

Every decision is expressed in dollars so unrelated jobs (watering a melon,
milking a cow, digging a weed) compete on one axis.  `PARAMS` holds every
tunable knob; v2 is this same code with a searched `PARAMS` block.

See docs/history/agent-v1.md for the full rationale.
"""

import math

# =============================================================================
# 1. ENVIRONMENT CONSTANTS (mirrored from kaggle_environments/envs/kaggriculture)
# =============================================================================

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
# FERTILIZER *is* sellable, and it is the most under-priced product in the game:
# every animal drops one free unit a day, nothing in the town consumes it, and
# the first ~250 units into the market still fetch $50-100. Excluding it (which
# every build up to v4 did) threw away roughly $20k a game.
SELLABLE = list(PRODUCTS)

MARKET_I0 = 10000
PRICE_FLOOR = 1
MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "log",    "below_target": 0.20, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

# Engine 1.32.7 hinge rebalance (PR #1399). Flipped by
# scripts/engine_swap_1327.py when the LADDER's replays show 1.32.7; v2 and
# every route build inherit on their next regeneration.
_ENGINE_1327 = True
if _ENGINE_1327:
    MARKET_PARAMS["CARROT"].update(below_func="hinge", below_target=1.00)
    MARKET_PARAMS["TOMATO"].update(below_func="hinge")
    MARKET_PARAMS["EGG"].update(below_func="hinge")

LAND_PRICES = [1000, 2000, 4000]
MAX_ORDERS = 10

# =============================================================================
# 2. TUNABLE PARAMETERS
# =============================================================================

# --- PARAMS BEGIN (tools/tune.py rewrites this block verbatim) ---
PARAMS = {
    # --- labour ----------------------------------------------------------
    "hands_max": 14,
    "hire_cash_floor": 40,       # never hire below this bank balance
    "hire_cash_frac": 0.55,      # share of the bank a day of hiring may cost
    "hands_min": 4,              # always try for at least this many hands
    "hands_slack": 2,            # hire this many past what the job list needs

    # --- labour capacity: how much land the crew can actually maintain ----
    "capacity_util": 0.82,       # fraction of unit-turns that reach a tile
    "cost_per_crop_day": 2.2,    # unit-turns/day to keep one crop tile alive
    "cost_per_animal_day": 4.5,  # feed + care + harvest + walking
    "herd_labour_share": 1.0,    # share of spare labour livestock may claim

    # --- cash management --------------------------------------------------
    "reserve_base": 120,         # always-on working capital
    "reserve_per_tile": 2.0,     # working capital per worked tile
    "feed_runway_days": 2.5,     # days of bought feed the bank must cover
    "income_gap_cap": 8.0,       # longest income gap the runway plans around
    "wheat_tile_yield": 0.6,     # wheat units/day actually delivered per wheat tile
    "max_herd": 15,              # ceiling on livestock; labour caps it too
    "animal_cash_buffer": 0,     # extra cash required before buying livestock
    "animals_per_turn": 4,       # pace livestock purchases
    "animal_pipeline": 4,        # heads allowed to sit in the shed awaiting a pen
    "pen_reserve_max": 0,        # tiles held back from crops for the planned herd
                                 # (measured -$2,145 at 8; off, but searchable)

    # --- land -------------------------------------------------------------
    "land_reserve": [200, 500, 3000],
    "land_last_day": [22, 20, 16],
    "land_min_day": [3, 6, 10],
    "land_min_used": 0.55,       # only expand once this much land is in use
    "land_max": 2,               # quadrants to buy; the third costs $4,000

    # --- portfolio: target tile counts across the whole farm --------------
    # Set from the measured ladder trajectory: ~40 strawberry, ~10 melon (the
    # town centre is melon's only buyer, ~140 units a season between two
    # farms), no geese at all -- eggs are the cheapest animal product and the
    # coop competes for the same feed and unit-turns as a sheep.
    "target_melon": 10,
    "target_strawberry": 40,
    "target_tomato": 0,
    "target_wheat": 4,           # floor; the real target tracks the herd size
    "target_carrot": 0,
    "target_goose": 0,
    "target_cow": 8,
    "target_sheep": 7,
    "filler_crop": "auto",       # "auto" = whatever the market pays most for
    "filler_max": 45,            # tiles the filler may take beyond its target
    "filler_ongoing_only": False,  # restrict the filler to tomato/strawberry
    "min_tile_rate": 6.0,        # $/tile-day below which a tile is left empty
    "cash_crop_gap": 3.0,        # income gap (days) that triggers a cash crop
    "cash_crop_tiles": 10,       # tiles given to it
    "cash_crop_cash": 400.0,     # ... or whenever the bank is under this
                                 # (1800 measured -$16,003: melon beats a fast
                                 #  crop badly enough to be worth being broke)
    "cash_crop_last_day": 24,    # after this a fast crop cannot pay back
    "cash_crop_span": 0.0,       # widen the "fast crop" definition past the gap
    "income_gap_cash": 400.0,    # sellable stock that counts as "earning"
    "feed_self_frac": 0.25,      # wheat tiles per head -- a hedge, not a field
    "animal_min_extra_days": 2,  # need first_yield_day + this days left to invest

    # --- market -----------------------------------------------------------
    "sell_floor": {              # sell while marginal price >= frac * base
        "WHEAT": 0.60, "CARROT": 0.55, "TOMATO": 0.55, "STRAWBERRY": 0.45,
        "MELON": 0.35, "EGG": 0.60, "MILK": 0.45, "WOOL": 0.35,
        "FERTILIZER": 0.30,
    },
    "sell_chunk": 10,            # max units of one product sold per turn
    "sell_orders": 4,            # sell orders per turn (of the 10 allowed)
    "use_schedule": True,        # apply SELL_SCHEDULE (empty table = no-op)
    "same_turn_sell": True,      # count this turn's DROPs as sellable stock
    "sell_first": True,          # SELL takes the low order indices
    "sell_hour_gain": 0.0,       # sell harder right after the town consumes
    "no_buy_last_turns": 0,      # final turns where only SELL is allowed
    "glut_priority": True,       # rank SELL slots by glut risk x their exposure
    "front_run_gain": 0.0,       # priority bump for a line they are about to dump
    "mirror_threshold": 0.75,    # farm similarity above which they mirror us
    "fert_stock": 4,             # fertilizer kept back for the fields
    "dump_day": 28,              # from this day on, ignore price floors
    "endgame_turns": 24,         # turns of pure haul-to-shed at the very end
                                 # (8 measured -$11,040 and 34 -$14,253: our crew
                                 #  needs the whole last day to walk the load in)
    "shed_pressure": 76,         # shed fuller than this -> sell at any price
    "feed_buffer_days": 2.0,     # wheat held in the shed per animal
    "feed_buy_price_cap": 6.0,   # buy feed wheat while price <= cap * base
    "seed_lookahead": 8,
    "liquid_target": 1200,       # bank level above which payback speed stops mattering
    "liquid_weight": 0.10,       # strength of the fast-payback bias when broke

    # --- market outlook: price against where the market will be ----------
    # 0 reproduces the old spot-price behaviour exactly.
    "outlook_weight": 1.0,       # how much projected supply moves the price
    "mirror_weight": 1.0,        # assume the opponent grows what we grow
    "demand_horizon": 10.0,      # days of town demand a harvest sells into
    "outlook_horizon": 0.5,      # fraction of that supply landed when we sell
    "fert_income_weight": 0.8,   # fertilizer units realised per head per day
    "fert_coverage": 0.35,       # share of production days we expect to fertilize
    "fert_opportunity": 1.0,     # sale price forgone by spreading a unit

    # --- task scoring ------------------------------------------------------
    # --- spatial efficiency -----------------------------------------------
    # Measured on v2: 50.6% of all unit-turns were movement, 8% PASS, leaving
    # only 41% doing productive work. Production is the binding constraint in
    # this game, so cutting walking is worth more than any portfolio change.
    "travel_weight": 5.0,        # >1 makes distance matter more in task choice
    "hands_late_day": 10,        # from this day the crew may grow
    "hands_max_late": 15,        # hand cap once the farm is at full size
    "fert_weight": 1.0,          # scales the value of applying fertilizer
    # "greedy" reproduces the original pair-picking exactly; "optimal" solves
    # the max-weight matching. A knob rather than a replacement, so the two can
    # be A/B'd on identical seeds instead of argued about.
    "assign_mode": "greedy",

    # --- ensemble: 0 disables it and costs nothing -----------------------
    "ensemble_k": 0,             # committee members beyond the incumbent
    "ensemble_spread": 0.12,     # relative jitter on each member's weights
    "ensemble_consensus": 0.55,  # share of votes needed to overrule the incumbent

    # --- adaptive play: "off" | "bandit" | "risk" | "both" ---------------
    # off    plays the same policy regardless of how the game is going
    # bandit Exp3 over committee members, reweighted from an in-game signal
    # risk   score-aware: press when behind late, protect a lead when ahead
    "adaptive_mode": "off",
    # "table" uses MEMBER_WEIGHTS; "xgb" uses the exported gradient-boosted
    # arbiter in XGB_MODEL, falling back to the table when none is embedded.
    "arbiter_model": "table",

    # --- opponent modelling: "off" | "observe" | "counter" ---------------
    # The farms are physically independent; the shared market is the only
    # channel between them, and it is a loud one. Measured on this build:
    # swapping the opponent moves the wheat price we see by up to 32%.
    # Market inventory is fully observable and we know our own orders, so the
    # opponent's net trade is not estimated -- it is derived exactly.
    "opponent_mode": "off",
    "opp_halflife": 6.0,         # turns; how fast the flow estimate forgets
    "opp_counter_gain": 0.6,     # how hard to trade against their flow

    "bandit_eta": 0.35,          # Exp3 learning rate; higher adapts faster, noisier
    "bandit_floor": 0.08,        # never fully mute a member -- keeps exploring
    "risk_gain": 0.45,           # how hard to press when behind
    "risk_late_day": 18,         # before this, the game is too young to gamble

    "poach_penalty": 1.0,        # <1 discourages taking a task nearer another unit

    # From the public meta write-up (raykkretzschmar, "Findings from Zero to
    # Top Meta"), bug table §7: "CARE ranked above melon WATER -> day 9 waters
    # only part of the field; yield ~70 not ~96". Watering inside a one-time
    # crop's bonus window is a hard deadline; CARE is a bank that can wait.
    # MEASURED OFF (1.0). At 3.0 this scored 0% win rate (-$6,181) against v3.
    # Their bug was architecture-specific: a priority *ladder* can rank CARE
    # above WATER. This agent prices both in dollars, so the trade-off is
    # already encoded and boosting WATER just makes units cross the farm for a
    # $250 job -- undoing the spatial-efficiency win.
    "water_window_priority": 1.0,   # multiplier on in-window WATER
    # Same table: "No DROP after HARVEST -> produce stuck in unit inventory".
    # SELL only draws from the shed, so a loaded farmer is unbanked revenue.
    # MEASURED OFF (0.0). At 900 this scored 25% (-$874). Interrupting a busy
    # unit to bank a load costs more than the delay, because inventories are
    # auto-dropped to the shed every night anyway; only the *final* day's load
    # is genuinely at risk, and the endgame haul already handles that.
    "drop_stack_value": 0.0,        # carry worth this much -> go bank it
    "care_weight": 1.0,
    "fert_collect_weight": 1.0,
    "feed_safe_discount": 0.20,  # FEED urgency when the animal can skip a day
}
# --- PARAMS END ---

# =============================================================================
# 2b. LEARNED POLICY HOOK
# =============================================================================
# ASSET_BIAS maps "<ASSET>|<day bucket>|<cash bucket>" to a multiplier on that
# asset's *planning* score -- i.e. how attractive it looks when deciding what an
# empty tile should become. An empty table means pure heuristic, which is what
# agents/v1..v3 ship with. tools/train_rl.py and tools/train_supervised.py fill
# it in.
#
# Deliberately narrow: the learned part only reweights an allocation choice the
# heuristic was already going to make legally. Feeding, watering, harvesting and
# the endgame haul stay hand-written, because a mislearned action there is
# unrecoverable (a missed feed kills an animal for the rest of the season) while
# a mis-weighted allocation just costs a tile.

# --- XGB BEGIN (tools/train_xgb.py rewrites this block) ---
# A gradient-boosted arbiter, exported as plain nested lists so the submission
# stays one self-contained file with no xgboost at run time.
#
# Each tree is a flat list of nodes: [feature, threshold, yes, no] for a split
# and [-1, leaf_value, -1, -1] for a leaf. Margin is the sum of leaf values plus
# `bias`, which is logit(base_score). Thresholds are stored as float32 and the
# feature vector is cast to float32 before comparing, because xgboost compares
# in float32 -- skip that and roughly one comparison in 700 lands on the wrong
# side of a split.
XGB_MODEL = {}
# --- XGB END ---


# --- WEIGHTS BEGIN (tools/train_arbiter.py rewrites this block) ---
# How much each committee member's vote counts, conditioned on the state.
# Keyed "<day bucket>|<cash bucket>" -> a weight per member (index 0 is the
# incumbent). Empty means every member counts 1.0, which is plain plurality.
#
# This is the learned part of the ensemble. Which member to trust is not
# constant: a policy that over-invests early is right on day 2 and wrong on
# day 27, and a flat vote cannot express that.
MEMBER_WEIGHTS = {}
# --- WEIGHTS END ---


# --- MEMBERS BEGIN (tools/select.py rewrites this block) ---
# An explicit committee: each entry is a partial parameter override applied on
# top of PARAMS. Written by tools/select.py from the top of the Elo ladder, so
# the committee is made of models that independently earned their place rather
# than of random jitter around one of them. Empty means fall back to jitter.
ENSEMBLE_MEMBERS = []
# --- MEMBERS END ---


# --- ASSET_BIAS BEGIN (trainers rewrite this block) ---
ASSET_BIAS = {}
# --- ASSET_BIAS END ---

DAY_EDGES = (8, 16, 23)
CASH_EDGES = (1500, 6000, 20000)


def _bucket(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def bias_key(asset, day, money):
    return f"{asset}|{_bucket(day, DAY_EDGES)}|{_bucket(money, CASH_EDGES)}"


def asset_bias(asset, day, money):
    if not ASSET_BIAS:
        return 1.0
    return ASSET_BIAS.get(bias_key(asset, day, money), 1.0)



# =============================================================================
# 2c. MARKET SCHEDULE
# =============================================================================
#
# Two independent top-ten write-ups reach the same conclusion about where the
# remaining edge at the top of this ladder lives:
#
#   "The farmer and hand tapes contribute almost nothing. The market tape
#    carries the frontier gain."                    -- prvsiyan, Frontier
#
#   "c18 changes only 20 pre-terminal field turns from c16 but 112 market
#    turns ... the current edge comes from inventory-sale timing rather than
#    another change in herd composition."           -- Findings from Zero to Top
#
# Their market plans are recorded schedules. Ours is a rule that sells whatever
# is in the shed. SELL_SCHEDULE is the middle ground a closed-loop agent can
# actually use: a multiplier per (day bucket, product) applied to both how much
# of that product we release and how hard it competes for one of the two SELL
# slots the engine grants us each turn.
#
# 1.0 everywhere reproduces the unscheduled behaviour exactly. The table is
# seeded from the field's measured schedule (tools/mine_top.py) and then
# optimised against live opponents (tools/schedule.py).

# --- SCHEDULE BEGIN (tools/schedule.py rewrites this block) ---
SELL_SCHEDULE = {}
# --- SCHEDULE END ---

SCHEDULE_DAYS = (5, 10, 15, 20, 25)


def schedule_bucket(day):
    for i, edge in enumerate(SCHEDULE_DAYS):
        if day < edge:
            return i
    return len(SCHEDULE_DAYS)


def sell_multiplier(item, day):
    """How aggressively to release `item` today. 1.0 = unscheduled."""
    if not SELL_SCHEDULE:
        return 1.0
    return float(SELL_SCHEDULE.get(f"{item}|{schedule_bucket(day)}", 1.0))

# =============================================================================
# 3. MARKET MODEL
# =============================================================================


def _shape(func, x, t=0.0):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        if not t or t <= 0:
            return x
        u = x / t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inv):
    """Exact replica of the environment's price curve."""
    p = MARKET_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inv, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inv - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def sellable_units(item, inv, have, floor_price, cap):
    """Units we can sell before the marginal price falls under `floor_price`."""
    n, cur = 0, inv
    limit = min(have, cap)
    while n < limit:
        price = market_price(item, cur)
        if price < floor_price:
            break
        n += 1
        if price > PRICE_FLOOR:
            cur += 1
    return n


# =============================================================================
# 3b. MARKET OUTLOOK -- what a unit will fetch when we sell it, not now
# =============================================================================
#
# Both farms are public and the town's appetite is a published schedule, so the
# price a harvest will *realise* is computable rather than guessable:
#
#     inventory_when_we_sell ~= inventory_now
#                               + (our standing supply + theirs) - town demand
#
# A planting decision is made ten days before the sale. MELON is the clearest
# case: the town centre is its only buyer (~140 units a season) and its glut
# curve is squared, so two farms each planting a dozen melon tiles collapse the
# price from $250 to under $100 before either of them harvests. Valuing a tile
# at the spot price is how a farm ends up growing something nobody will pay for.
#
# The same arithmetic run the other way is why WOOL is the best asset in the
# game: only sheep make it, the yarn store alone drains ~12 units a day, and a
# season of six sheep lands almost exactly on the town's appetite.

SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def town_demand_per_day(obs, item):
    """Units of `item` the town takes off the market each day, THIS game.

    Shops fire every 4 steps (6 a day) taking one unit of each product they
    list, or two when they list only one; `unlocked_shops` may hold the same
    shop several times (drawn WITH REPLACEMENT, one draw every 3 days, capped
    at 8 instances) and each instance consumes independently. The town centre
    fires once a day taking exactly 1 of everything -- flat, no decade
    scaling (verified against the 1.32.7 interpreter's _town_consume; the
    old "every 12 steps, 1/2/4 by decade" here was the 1.32.4 engine and
    overstated late-game drain by up to 8x). Fertilizer has no buyer at all.

    DEMAND CAPTURE (2026-08-16): the future-shop prior is the REPLACEMENT
    expectation, not "every never-seen shop at half weight". Each remaining
    draw lands on any given shop with probability 1/8, so as the game reveals
    its actual draw the estimate converges to THIS game's demand -- in the
    games where the tomato/carrot/egg shops really spawned, the outlook sees
    the drain (and under 1.32.7's hinge, the price spike) that most games
    never get. demand_match is the #1 elite marker; the hinge pays it.
    """
    if item == "FERTILIZER":
        return 0.0
    unlocked = list(((obs.get("town") or {}).get("unlocked_shops")) or [])
    rate = 0.0
    for shop in unlocked:
        products = SHOP_PRODUCTS.get(shop, ())
        if item in products:
            rate += 6.0 * (2.0 if len(products) == 1 else 1.0)
    # Remaining draws, each uniform over the 8 shops; half weight because
    # they arrive spread over the coming days rather than now.
    draws_left = max(0, 8 - len(unlocked))
    if draws_left:
        per_draw = sum(6.0 * (2.0 if len(p) == 1 else 1.0)
                       for p in SHOP_PRODUCTS.values() if item in p)
        rate += 0.5 * draws_left * per_draw / max(1, len(SHOP_PRODUCTS))
    rate += 1.0                        # town centre: 1/day of everything
    return rate


def _tile_future_units(tile, day, days_left, out):
    """Add the units this tile will still deliver to `out`."""
    if not isinstance(tile, dict):
        return
    kind = tile.get("kind")
    if kind == "PLANT":
        crop = tile.get("crop")
        cd = CROPS.get(crop)
        if cd is None:
            return
        age = day - int(tile.get("planted_day", day))
        held = float(tile.get("yield_units", 0) or 0)
        if cd["ongoing"]:
            done = 0 if age < cd["first_yield_day"] else \
                (age - cd["first_yield_day"]) // cd["interval"] + 1
            left = max(0, cd["max_yield"] - done)
            left = min(left, max(0, days_left) // max(1, cd["interval"]) + 1)
            out[crop] = out.get(crop, 0.0) + held + left
        else:
            togo = max(0, HARVEST_DAY[crop] - age)
            if togo > days_left:
                out[crop] = out.get(crop, 0.0) + held
            else:
                out[crop] = out.get(crop, 0.0) + max(held, crop_expected_units(crop))
        return
    animal = tile.get("animal")
    if animal and animal in ANIMALS:
        a = ANIMALS[animal]
        age = day - int(tile.get("placed_day", day))
        wait = max(0, a["first_yield_day"] - age)
        productive = max(0, days_left - wait)
        out[a["product"]] = (out.get(a["product"], 0.0)
                             + float(tile.get("yield_units", 0) or 0)
                             + animal_units_per_day(animal) * productive)
        out["FERTILIZER"] = out.get("FERTILIZER", 0.0) + max(0, days_left - 1)


def projected_supply(obs, days_left, mirror=1.0):
    """Units both farms will still put on the market, from public tiles.

    Standing tiles are only the supply that *already exists*. On day zero
    neither farm has planted anything, so a pure census says the market is
    empty and melon is worth $250 a unit -- right up to the moment both farms
    harvest sixty of them into a town that buys 140 a season.

    `mirror` closes that gap with the one prior this meta actually supports:
    the opponent is running a strategy very like ours. Whatever we are about to
    grow, assume they are growing it too unless their own tiles already say
    otherwise.
    """
    day = int(obs.get("day", 0) or 0)
    seat = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    mine, theirs = {}, {}
    for i, farm in enumerate(farms):
        into = mine if i == seat else theirs
        for row in (farm.get("tiles") or []):
            for tile in row:
                _tile_future_units(tile, day, days_left, into)
    out = {}
    for item in PRODUCTS:
        ours = mine.get(item, 0.0)
        them = theirs.get(item, 0.0)
        out[item] = ours + max(them, mirror * ours)
    return out


_OUTLOOK = {"key": None, "prices": {}, "supply": {}}


def market_outlook(obs, P, days_left):
    """Per-item price to plan against. Cached for the turn."""
    key = (int(obs.get("day", 0) or 0), int(obs.get("hour", 0) or 0),
           int(obs.get("player", 0) or 0), id(P))
    if _OUTLOOK["key"] == key:
        return _OUTLOOK["prices"]
    inv = (obs.get("market") or {}).get("inventory") or {}
    supply = projected_supply(obs, days_left, float(P.get("mirror_weight", 1.0)))
    weight = float(P.get("outlook_weight", 1.0))
    horizon = float(P.get("outlook_horizon", 0.5))
    # Demand accrues while we hold stock, but a harvest lands in a burst: the
    # town has not drained a season's worth by the afternoon we sell.
    window = min(float(days_left), float(P.get("demand_horizon", 10.0)))
    prices, effective = {}, {}
    for item in PRODUCTS:
        cur = float(inv.get(item, MARKET_I0))
        net = supply.get(item, 0.0) - town_demand_per_day(obs, item) * window
        eff = cur + weight * horizon * net
        effective[item] = eff
        prices[item] = float(market_price(item, eff))
    _OUTLOOK["key"] = key
    _OUTLOOK["prices"] = prices
    _OUTLOOK["supply"] = supply
    _OUTLOOK["effective"] = effective
    _OUTLOOK["horizon"] = horizon
    return prices


def marginal_price(obs, item, extra_units):
    """Price after `extra_units` more of our own supply reaches the market.

    This is what stops the planner filling a quadrant with the crop that looks
    best for the *first* tile. Melon is the trap: at $250 with a squared glut
    curve and a town centre that buys ~140 units a season, tile eleven is worth
    a third of tile one, and the spot price cannot see that.
    """
    eff = _OUTLOOK.get("effective", {}).get(item)
    if eff is None:
        return _out(obs, item)
    return float(market_price(item, eff + _OUTLOOK.get("horizon", 0.5) * extra_units))


def marginal_tile_value(obs, P, crop, extra_units):
    price = marginal_price(obs, crop, extra_units)
    units = crop_expected_units(crop, float(P.get("fert_coverage", 0.0)))
    return (units * price - CROPS[crop]["seed"]) / max(1, crop_cycle_days(crop))


def _out(obs, item):
    """Outlook price if one has been computed this turn, else the spot price."""
    price = _OUTLOOK["prices"].get(item)
    if price is None:
        return _price(obs, item)
    return price


# =============================================================================
# 4. VALUATION
# =============================================================================


def _price(obs, item):
    return obs["market"]["prices"].get(item, MARKET_PARAMS[item]["base"])


def _max_yield_day(crop):
    """First day a one-time crop's yield_units reaches max_yield (watered daily)."""
    cd = CROPS[crop]
    units = 1
    for d in range((cd["max_yield_day"] + 1) // 2, cd["max_yield_day"] + 1):
        units += 1
        if units >= cd["max_yield"] and d >= cd["first_yield_day"]:
            return d
    return cd["max_yield_day"]


HARVEST_DAY = {c: max(CROPS[c]["first_yield_day"], _max_yield_day(c)) for c in CROPS}


def crop_expected_units(crop, fert=0.0):
    """Units one tile delivers over its cycle.

    `fert` is the share of production days the tile is fertilized on. A watered
    day inside the yield window adds +1 unit, or **+2 when the tile is
    fertilized** -- so fertilizer does not trim a cost, it doubles the harvest.
    Ongoing crops (tomato, strawberry) get the doubling on every scheduled
    production, which is why strawberry is worth a whole quadrant.
    """
    cd = CROPS[crop]
    if cd["ongoing"]:
        return cd["max_yield"] * (1.0 + fert)
    start = (cd["max_yield_day"] + 1) // 2
    waterings = max(0, HARVEST_DAY[crop] - start + 1)
    return min(cd["max_yield"], 1.0 + waterings * (1.0 + fert))


def crop_cycle_days(crop):
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return max(1, HARVEST_DAY[crop])
    return cd["first_yield_day"] + (cd["max_yield"] - 1) * cd["interval"] + 1


def crop_value_per_tile_day(obs, crop, P=None):
    P = PARAMS if P is None else P
    price = _out(obs, crop)
    fert = float(P.get("fert_coverage", 0.0))
    units = crop_expected_units(crop, fert)
    return (units * price - CROPS[crop]["seed"]) / max(1, crop_cycle_days(crop))


def liquidity_discount(P, money, payback_days):
    """Slow crops are worth less when the bank is empty.

    Melons are the best $/tile-day in the game but pay nothing for ten days.
    Planting a whole field of them on day 0 leaves no cash for hands, seed or
    feed, and the farm stalls until day 10 -- so discount by payback time in
    proportion to how short of working capital we are.
    """
    pressure = max(0.0, 1.0 - money / float(P["liquid_target"]))
    return 1.0 / (1.0 + P["liquid_weight"] * pressure * payback_days)


def planning_score(obs, P, crop, money):
    return crop_value_per_tile_day(obs, crop, P) * liquidity_discount(P, money, HARVEST_DAY[crop])


def animal_planning_score(obs, P, animal, money, days_left):
    a = ANIMALS[animal]
    return (animal_lifetime_profit(obs, animal, days_left, P) / max(1, days_left)
            * liquidity_discount(P, money, a["first_yield_day"]))


def animal_units_per_day(animal):
    """Steady state with CARE every day: (1 + interval) units per interval days."""
    a = ANIMALS[animal]
    return (1.0 + a["interval"]) / a["interval"]


def animal_fert_per_day(P=None):
    """Fertilizer an animal drops per day, discounted by how much we realise.

    `fertilizer_available` is set on every daily refresh from the day after the
    animal is placed, so this is one free unit per head per day for the whole
    season -- worth more than a goose's eggs and completely free.
    """
    P = PARAMS if P is None else P
    return float(P.get("fert_income_weight", 0.8))


def animal_value_per_day(obs, animal, P=None):
    a = ANIMALS[animal]
    return (animal_units_per_day(animal) * _out(obs, a["product"])
            + animal_fert_per_day(P) * _out(obs, "FERTILIZER"))


def animal_lifetime_profit(obs, animal, days_left, P=None):
    """Every dollar a head still earns, net of feed and its purchase price.

    Two terms the earlier builds left out, and they are most of the answer:
    the daily fertilizer drop, and the fact that feed can be *bought* -- so the
    comparison is against the wheat price, not against a wheat field.
    """
    P = PARAMS if P is None else P
    a = ANIMALS[animal]
    productive = max(0, days_left - a["first_yield_day"])
    alive = max(0, days_left - 1)
    gross = animal_units_per_day(animal) * _out(obs, a["product"]) * productive
    gross += animal_fert_per_day(P) * _out(obs, "FERTILIZER") * alive
    feed = alive * _out(obs, "WHEAT")
    return gross - feed - a["cost"]


def remaining_plant_value(obs, tile, day):
    crop = tile["crop"]
    cd = CROPS[crop]
    price = _out(obs, crop)
    if cd["ongoing"]:
        age = day - tile["planted_day"]
        done = 0 if age < cd["first_yield_day"] else (age - cd["first_yield_day"]) // cd["interval"] + 1
        left = max(0, cd["max_yield"] - done)
        return (tile.get("yield_units", 0) + left) * price
    return max(tile.get("yield_units", 0), crop_expected_units(crop)) * price


def remaining_animal_value(obs, animal, held, days_left, age=0):
    """$ still to come from this animal.

    `age` matters: an animal that already passed its first_yield_day keeps
    producing for every remaining day. Ignoring age made mature livestock look
    worthless in the final week, so nothing fed it and the herd starved out
    two days before scoring.
    """
    a = ANIMALS[animal]
    price = _out(obs, a["product"])
    wait = max(0, a["first_yield_day"] - age)
    productive = max(0, days_left - wait)
    return (held * price + animal_units_per_day(animal) * price * productive
            + animal_fert_per_day() * _out(obs, "FERTILIZER") * max(0, days_left - 1))


# =============================================================================
# 5. GEOMETRY / CENSUS
# =============================================================================


def _shed_tiles(n):
    h = n // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def _is_shed_adjacent(pos, n):
    h = n // 2
    return (pos[0], pos[1]) in {(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)}


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_towards(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return None


def census(tiles, n):
    c = {k: 0 for k in list(CROPS) + list(ANIMALS)}
    c["COOP"] = c["PASTURE"] = c["WEED"] = 0
    empty = []
    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t is None:
                empty.append((x, y))
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind == "PLANT":
                    c[t["crop"]] += 1
                elif "animal" in t:
                    c[t["animal"]] += 1
                elif kind in ("COOP", "PASTURE"):
                    c[kind] += 1
                elif kind == "WEED":
                    c["WEED"] += 1
    return c, empty


def _filler_crop(P, ranked):
    """Which crop may fill unlimited spare tiles.

    "auto" means the best-valued crop that is still worth planting, which is
    the whole point: v4 pinned this to WHEAT, so every spare tile on every
    quadrant became the cheapest crop in the game regardless of what the market
    was paying for strawberries.
    """
    name = str(P.get("filler_crop", "auto"))
    if name in CROPS:
        return name
    # Only an ongoing crop may run past its target. One-shot crops have a hard
    # demand ceiling -- melon's single buyer is the town centre at ~140 units a
    # season -- and letting one fill spare land is how a farm grows 25 melons
    # and sells them at a third of the price it planned around.
    ongoing_only = bool(P.get("filler_ongoing_only", False))
    for crop in ranked:
        if crop not in CROPS:
            continue
        if ongoing_only and not CROPS[crop]["ongoing"]:
            continue
        return crop
    return "STRAWBERRY"


def crop_target(P, crop, c, shed=None):
    """Target tile count for a crop.

    WHEAT is feed, and feed is *bought*, not grown: a wheat tile returns about
    $40 a tile-day while a strawberry tile on the same land returns the same
    with four times the price ceiling, and the wheat we do not grow costs ~$40
    a head a day to replace. `feed_self_frac` is therefore a small hedge
    against the wheat price running away, not a herd-sized field. v4 grew up to
    72 wheat tiles on three quadrants of prime land; that alone was most of the
    gap to the top of the ladder.
    """
    base = P["target_" + crop.lower()]
    if crop == "WHEAT":
        herd = c["GOOSE"] + c["COW"] + c["SHEEP"]
        if shed:
            herd += sum(shed.get(a, 0) for a in ANIMALS)
        return max(base, int(math.ceil(P["feed_self_frac"] * herd)))
    return base


def best_filler_rate(obs, P, days_left):
    """Run-rate of the best crop we could put on a freed tile, in $/tile-day.

    Used to price DIG and "harvest to free the tile" jobs. It must track the
    crop the planner would actually plant there, otherwise clearing a weed is
    valued at the wheat rate while the tile is really worth a strawberry.
    """
    best = 1.0
    for crop in CROPS:
        if days_left < HARVEST_DAY[crop] + 1:
            continue
        best = max(best, crop_value_per_tile_day(obs, crop, P))
    return best


def herd_size(c, shed=None):
    n = c["GOOSE"] + c["COW"] + c["SHEEP"]
    if shed:
        n += sum(shed.get(a, 0) for a in ANIMALS)
    return n


def working_tiles(c):
    return sum(c[k] for k in CROPS) + herd_size(c) + c["COOP"] + c["PASTURE"]


def income_gap(obs, farm, shed, day, cap=8.0):
    """Days until this farm can put money in the bank.

    Zero if the shed already holds something sellable. Otherwise the shortest
    wait among the standing crops and animals, capped -- a longer horizon than
    `cap` is not a cash-flow plan, it is a bet.
    """
    # A single collected fertilizer is not an income stream. The gap only
    # closes once the shed holds enough to cover a few days of feed and hiring.
    banked = 0.0
    for item, qty in (shed or {}).items():
        if qty > 0 and item in SELLABLE and item != "WHEAT":
            banked += qty * _out(obs, item)
    if banked >= float(PARAMS.get("income_gap_cash", 400.0)):
        return 0.0
    best = cap
    for row in (farm.get("tiles") or []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                cd = CROPS.get(tile.get("crop"))
                if cd is None:
                    continue
                if tile.get("yield_units", 0) > 0 and                         day - int(tile.get("planted_day", day)) >= cd["first_yield_day"]:
                    return 0.0
                wait = cd["first_yield_day"] - (day - int(tile.get("planted_day", day)))
                best = min(best, max(0.0, wait))
            elif tile.get("animal") in ANIMALS:
                a = ANIMALS[tile["animal"]]
                if tile.get("yield_units", 0) > 0:
                    return 0.0
                wait = a["first_yield_day"] - (day - int(tile.get("placed_day", day)))
                best = min(best, max(0.0, wait))
    return max(0.0, min(cap, best))


def working_reserve(P, c, shed=None, wheat_price=25.0, gap=0.0):
    """Cash we refuse to spend on capital -- it is the herd's feed money.

    A starved animal is gone for good, so the bank must cover the feed we have
    not already bought. It must cover *days*, not weeks: at 14 head that is
    ~$600 a day against daily revenue in the thousands, and the earlier
    twelve-day runway locked up $6,700 through the exact window where livestock
    and land are the highest-return things cash can buy. The herd starving on
    day 18 was a purchase-side bug (buying heads the crew could not service),
    not a reason to hoard.
    """
    herd = herd_size(c, shed)
    stocked = float((shed or {}).get("WHEAT", 0))
    # Runway is measured in days *until the farm earns something*. A herd bought
    # on turn zero against a field of melons has no income for ten days, and two
    # missed feeds kill an animal for good.
    runway = max(float(P["feed_runway_days"]), float(gap))
    uncovered = max(0.0, herd * runway - stocked
                    - c["WHEAT"] * P["wheat_tile_yield"] * runway)
    return (P["reserve_base"] + P["reserve_per_tile"] * working_tiles(c)
            + uncovered * wheat_price)


# --- labour capacity ---------------------------------------------------------


def _fib_costs(k):
    out, a, b = [], 1, 1
    for _ in range(k):
        out.append(a)
        a, b = b, a + b
    return out


def affordable_hands(P, money):
    """Hands a day's hiring budget can pay for (cost = 1,1,2,3,5,8,13,...)."""
    budget = money * P["hire_cash_frac"]
    spend, k = 0, 0
    for cost in _fib_costs(P["hands_max"]):
        if spend + cost > budget:
            break
        spend += cost
        k += 1
    return k


def labour_used(P, crops, animals):
    return P["cost_per_crop_day"] * crops + P["cost_per_animal_day"] * animals


def labour_capacity(P, hands):
    return (1 + hands) * 24.0 * P["capacity_util"]


# =============================================================================
# 6. TILE PLAN -- what should each empty tile become?
# =============================================================================


def cash_crop_plan(obs, P, days_left, gap, money):
    """(crop, tiles) to plant purely to close an income gap, or (None, 0).

    Value per tile-day cannot express this. Melon is the best tile on the farm
    by a wide margin and pays nothing for ten days; a farm that fills its land
    with melon on turn zero cannot hire, cannot feed, and watches the herd it
    just bought starve on day three. The fast crop is not competing on rate --
    it is buying the days in between, which is why every top ladder tape puts
    ten wheat tiles in the ground on turn zero next to its seven melons.
    """
    # Trigger on the *bank*, not only on the gap heuristic. Measured failure:
    # `income_gap` returns 0 as soon as the shed holds a few hundred dollars of
    # produce, so from day 2 onward the rule never fired again -- the farm ran
    # on $71-$690 for ten days, could not buy the animals it had targets for,
    # and by day 9 every tile was full of crops that pay on day 10. A shed with
    # $400 in it is not an income stream.
    if int(obs.get("day", 0) or 0) > int(P.get("cash_crop_last_day", 24)):
        return None, 0
    if gap < float(P.get("cash_crop_gap", 3.0)):
        return None, 0
    # `cash_crop_broke_or` widens the trigger to "gap OR empty bank". Measured
    # -$16,003 at a $1,800 threshold: melon beats a fast crop by enough to be
    # worth being broke for ten days. Off by default, kept searchable.
    if money > float(P.get("cash_crop_cash", 1500.0)) and not P.get("cash_crop_broke_or", False):
        return None, 0
    best, best_key = None, None
    horizon = max(float(gap), float(P.get("cash_crop_span", 0.0)))
    for crop in CROPS:
        span = HARVEST_DAY[crop]
        if span > horizon or days_left < span + 1:
            continue
        rate = marginal_tile_value(obs, P, crop, 0.0)
        if rate <= 0:
            continue
        key = (span, -rate)
        if best_key is None or key < best_key:
            best, best_key = crop, key
    if best is None:
        return None, 0
    return best, int(P.get("cash_crop_tiles", 10))


def plan_tiles(obs, P, c, empty, days_left, money, shed, seeds, n):
    """Budget-aware assignment of empty tiles to crops / structures.

    Livestock is placed on the tiles nearest the shed (feeding trips start
    there); crops take the outer ring.  Structures are only committed when we
    can actually pay for the animal that will occupy them, which is the bug
    that sank the first draft of this agent.
    """
    plan = {}
    if not empty:
        return plan

    order = sorted(empty, key=lambda p: (min(_dist(p, s) for s in _shed_tiles(n)), p[1], p[0]))
    _farm = obs["farms"][obs["player"]]
    _gap = income_gap(obs, _farm, shed, int(obs.get("day", 0) or 0),
                      float(P.get("income_gap_cap", 8.0)))
    capital = (money - working_reserve(P, c, shed, _price(obs, "WHEAT"), _gap)
               - P["animal_cash_buffer"])
    idx = 0

    # ---- livestock -------------------------------------------------------
    ranked = []
    herd_room = P["max_herd"] - herd_size(c, shed)
    for animal, a in ANIMALS.items():
        if herd_room <= 0:
            break
        if days_left < a["first_yield_day"] + P["animal_min_extra_days"]:
            continue
        profit = animal_lifetime_profit(obs, animal, days_left, P)
        if profit <= 0:
            continue
        ranked.append((animal_planning_score(obs, P, animal, money, days_left)
                       / a["cost"] * asset_bias(animal, int(obs["day"]), money),
                       profit, animal))
    ranked.sort(reverse=True)

    free_struct = {"COOP": c["COOP"], "PASTURE": c["PASTURE"]}
    # NOTE: a "rescue pass" that builds pens for animals stranded in the shed
    # was tried and REVERTED -- it measured 62.5% vs 81% win rate against v2
    # over the same 16 matches. Building a pen the crew cannot staff or feed
    # costs more than the stranded animal does. Fix the purchase side instead.

    for _r, _p, animal in ranked:
        a = ANIMALS[animal]
        deficit = P["target_" + animal.lower()] - c[animal]
        if deficit <= 0:
            continue
        in_shed = shed.get(animal, 0)
        afford = int(max(0.0, capital) // a["cost"])
        # Structures we may commit to = animals owned or affordable, minus the
        # empty structures already standing.
        allowed = min(deficit, in_shed + afford) - free_struct[a["structure"]]
        for _ in range(max(0, allowed)):
            if idx >= len(order):
                break
            plan[order[idx]] = (a["structure"], animal)
            idx += 1
            free_struct[a["structure"]] += 1
            capital -= a["cost"]
            afford = int(max(0.0, capital) // a["cost"])

    # ---- crops -----------------------------------------------------------
    crop_ranked = []
    for crop in CROPS:
        if days_left < HARVEST_DAY[crop] + 1:
            continue
        crop_ranked.append((planning_score(obs, P, crop, money)
                            * asset_bias(crop, int(obs["day"]), money), crop))
    crop_ranked.sort(reverse=True)

    # Never plant more than the crew can water. An unwatered field becomes a
    # weed patch in two days, so over-planting is strictly negative value.
    hands = affordable_hands(P, money)
    capacity = labour_capacity(P, hands)
    animals_total = c["GOOSE"] + c["COW"] + c["SHEEP"] + sum(
        1 for v in plan.values() if v[0] in ("COOP", "PASTURE"))
    crop_tiles = sum(c[k] for k in CROPS)

    # Each tile is priced at the *margin*: after choosing a crop we add its
    # expected harvest to the supply the market has to absorb, so the eleventh
    # melon competes against the ten already planned rather than against an
    # empty market. This is what caps a crop at the size of its demand without
    # any hand-set tile count.
    planned = {crop: 0 for crop in CROPS}
    extra = {crop: 0.0 for crop in CROPS}
    fert = float(P.get("fert_coverage", 0.0))
    filler = _filler_crop(P, [cr for _s, cr in crop_ranked])
    floor = float(P.get("min_tile_rate", 0.0))

    # Land the herd has not claimed yet is not spare land. A pasture returns
    # ~$310 a day (milk plus the free fertilizer); a strawberry tile returns
    # ~$80. Measured failure: 44 strawberry and 10 wheat filled the farm by day
    # 18, so the last four cows had nowhere to stand and the agent finished with
    # 3 cows against the meta's 8 -- 96 milk sold where the ladder sells 181.
    reserved_pens = 0
    if days_left >= 3:
        for animal, a in ANIMALS.items():
            if days_left < a["first_yield_day"] + P["animal_min_extra_days"]:
                continue
            if animal_lifetime_profit(obs, animal, days_left, P) <= 0:
                continue
            want = P["target_" + animal.lower()] - c[animal] - shed.get(animal, 0)
            reserved_pens += max(0, want)
        reserved_pens = min(reserved_pens,
                            max(0, P["max_herd"] - herd_size(c, shed)),
                            int(P.get("pen_reserve_max", 0)))

    # Cash crop first: tiles bought to close the income gap, not to earn a rate.
    cash_crop, cash_tiles = cash_crop_plan(obs, P, days_left, _gap, money)
    if cash_crop is not None:
        want = max(0, cash_tiles - c[cash_crop])
        while want > 0 and idx < len(order) and seeds.get(cash_crop, 0) - planned[cash_crop] > 0:
            if labour_used(P, crop_tiles + 1, animals_total) > capacity:
                break
            plan[order[idx]] = ("CROP", cash_crop)
            planned[cash_crop] += 1
            extra[cash_crop] += crop_expected_units(cash_crop, fert)
            crop_tiles += 1
            idx += 1
            want -= 1
    while idx < len(order) - reserved_pens:
        if labour_used(P, crop_tiles + 1, animals_total) > capacity:
            break
        best, best_v = None, floor
        for _v0, crop in crop_ranked:
            target = crop_target(P, crop, c, shed)
            if crop == filler:
                target = max(target, int(P.get("filler_max", 10 ** 6)))
            if c[crop] + planned[crop] >= target:
                continue
            if seeds.get(crop, 0) - planned[crop] <= 0:
                continue
            v = (marginal_tile_value(obs, P, crop, extra[crop])
                 * asset_bias(crop, int(obs["day"]), money))
            if v > best_v:
                best, best_v = crop, v
        if best is None:
            break
        plan[order[idx]] = ("CROP", best)
        planned[best] += 1
        extra[best] += crop_expected_units(best, fert)
        crop_tiles += 1
        idx += 1
    return plan


def plantable_slots(P, c, empty_count, money, days_left):
    """Tiles we could sensibly plant soon -- bounds seed buying."""
    hands = affordable_hands(P, money)
    animals = c["GOOSE"] + c["COW"] + c["SHEEP"]
    crops = sum(c[k] for k in CROPS)
    headroom = (labour_capacity(P, hands) - labour_used(P, crops, animals)) / P["cost_per_crop_day"]
    return max(0, min(empty_count + P["seed_lookahead"], int(headroom)))


def seed_demand(obs, P, c, plantable, seeds, days_left, shed=None, gap=0.0, money=0.0):
    """How many seeds to *buy* per crop.

    `plantable` is the number of tiles we could realistically plant soon
    (empty tiles plus a small lookahead, already capped by labour). Seeds are
    allocated to that many slots in value order; we then subtract the stock we
    already hold. Without the slot cap the filler crop's effectively-infinite
    target drains the bank every turn -- that bug cost v1 its entire opening.
    """
    orders = {}
    slots = max(0, plantable)
    fert = float(P.get("fert_coverage", 0.0))
    floor = float(P.get("min_tile_rate", 0.0))
    extra = {crop: 0.0 for crop in CROPS}
    held = {crop: seeds.get(crop, 0) for crop in CROPS}
    room = {crop: max(0, crop_target(P, crop, c, shed) - c[crop]) for crop in CROPS}
    cash_crop, cash_tiles = cash_crop_plan(obs, P, days_left, gap, money)
    if cash_crop is not None:
        room[cash_crop] = max(room[cash_crop], min(slots, cash_tiles - c[cash_crop]))
        need = max(0, min(slots, cash_tiles - c[cash_crop]) - held[cash_crop])
        if need > 0:
            orders[cash_crop] = need
            held[cash_crop] += need
            slots -= need
    filler = _filler_crop(P, [cr for _s, cr in sorted(
        ((crop_value_per_tile_day(obs, cr, P), cr) for cr in CROPS), reverse=True)])
    room[filler] = max(room[filler], min(slots, int(P.get("filler_max", 10 ** 6))))
    while slots > 0:
        best, best_v = None, floor
        for crop in CROPS:
            if days_left < HARVEST_DAY[crop] + 1 or room[crop] <= 0:
                continue
            v = marginal_tile_value(obs, P, crop, extra[crop])
            if v > best_v:
                best, best_v = crop, v
        if best is None:
            break
        room[best] -= 1
        slots -= 1
        extra[best] += crop_expected_units(best, fert)
        if held[best] > 0:
            held[best] -= 1
        else:
            orders[best] = orders.get(best, 0) + 1
    return orders


# =============================================================================
# 7. TASKS
# =============================================================================


def build_tasks(obs, P, tiles, n, day, days_left, c, plan, shed, surplus):
    tasks = []

    def add(pos, op, value, need=None):
        tasks.append({"pos": pos, "op": op, "value": float(value), "need": need})

    filler_rate = max(1.0, best_filler_rate(obs, P, days_left))

    for y in range(n):
        for x in range(n):
            t = tiles[y][x]
            if t == "LOCKED":
                continue
            pos = (x, y)

            if t is None:
                job = plan.get(pos)
                if job is None:
                    continue
                kind, name = job
                if kind == "CROP":
                    add(pos, ["PLANT", name],
                        crop_value_per_tile_day(obs, name, P) * min(days_left, crop_cycle_days(name)))
                else:
                    add(pos, ["BUILD_" + kind],
                        0.8 * animal_lifetime_profit(obs, name, days_left, P))
                continue

            if not isinstance(t, dict):
                continue
            kind = t.get("kind")

            # -------- weed --------
            if kind == "WEED":
                if days_left >= 4:
                    add(pos, ["DIG"], 0.55 * filler_rate * days_left)
                continue

            # -------- plant --------
            if kind == "PLANT":
                crop = t["crop"]
                cd = CROPS[crop]
                age = day - t["planted_day"]
                price = _price(obs, crop)
                watered = bool(t.get("watered_today", False))
                units = int(t.get("yield_units", 0))
                wstart = (cd["max_yield_day"] + 1) // 2
                in_window = wstart <= age <= cd["max_yield_day"] and units < cd["max_yield"]

                if not watered:
                    val = 0.0
                    if int(t.get("consecutive_unwatered", 0)) >= 1:
                        val = remaining_plant_value(obs, t, day)
                    if not cd["ongoing"] and in_window:
                        bonus = 2 if int(t.get("fertilized_until_day", -1)) >= day else 1
                        # A missed in-window watering is a permanently smaller
                        # harvest; CARE only defers a bonus. Rank accordingly.
                        val = max(val, bonus * price * P.get("water_window_priority", 1.0))
                    elif cd["ongoing"] and int(t.get("fertilized_until_day", -1)) >= day:
                        val = max(val, 0.5 * price)
                    if val > 0:
                        add(pos, ["WATER"], val)

                if units > 0 and age >= cd["first_yield_day"]:
                    if cd["ongoing"]:
                        urgency = 1.0 if units >= cd["max_yield"] else 0.5
                        add(pos, ["HARVEST"], units * price * urgency)
                    elif age >= HARVEST_DAY[crop] and not (in_window and not watered):
                        # Freeing the tile is worth the next crop's run-rate.
                        add(pos, ["HARVEST"], units * price + 0.4 * filler_rate * days_left)

                if shed.get("FERTILIZER", 0) > 0 and int(t.get("fertilized_until_day", -1)) < day:
                    # Net of what the same unit would fetch on the market. A
                    # fertilizer spent here is a fertilizer not sold, and at
                    # $50-100 a unit that opportunity cost decides the call --
                    # which is why pushing fert_weight to 5.0 measured 2W-18L.
                    gain = _fertilizer_gain(crop, age, units, price,
                                            P.get("fert_weight", 1.0), days_left)
                    gain -= float(P.get("fert_opportunity", 1.0)) * _out(obs, "FERTILIZER")
                    if gain > 0:
                        add(pos, ["FERTILIZE"], gain, need="FERTILIZER")
                continue

            # -------- coop / pasture --------
            if kind in ("COOP", "PASTURE") and "animal" not in t:
                wanted = _animal_for_structure(P, c, kind, days_left, shed, obs)
                if wanted is not None and shed.get(wanted, 0) > 0:
                    add(pos, ["PLACE", wanted],
                        remaining_animal_value(obs, wanted, 0, days_left), need=wanted)
                elif surplus.get(kind, 0) > 0 and days_left >= 4:
                    # No animal is coming for this pen: reclaim the tile for crops.
                    surplus[kind] -= 1
                    add(pos, ["DIG"], 0.5 * filler_rate * days_left)
                elif days_left <= 3:
                    add(pos, ["DIG"], 0.2 * filler_rate * days_left)
                continue

            # -------- livestock --------
            if "animal" in t:
                animal = t["animal"]
                a = ANIMALS[animal]
                price = _out(obs, a["product"])
                units = int(t.get("yield_units", 0))

                if not t.get("fed_today", False) and days_left >= 1:
                    if int(t.get("consecutive_unfed", 0)) >= 1:
                        # Miss today and the animal is gone for good.
                        age = day - int(t.get("placed_day", day))
                        val = remaining_animal_value(obs, animal, units, days_left, age)
                    else:
                        # Safe for one more day, but keeping the herd alive is
                        # still the single most valuable job on the farm.
                        age = day - int(t.get("placed_day", day))
                        val = P["feed_safe_discount"] * remaining_animal_value(
                            obs, animal, units, days_left, age)
                    add(pos, ["FEED"], val, need="WHEAT")

                if units > 0:
                    urgency = 1.0 if units >= a["max_held"] else 0.5
                    add(pos, ["HARVEST"], units * price * urgency)

                if not t.get("cared_today", False) and days_left >= 2:
                    add(pos, ["CARE"], P["care_weight"] * price)

                if t.get("fertilizer_available", False) and days_left >= 1:
                    # A free unit worth $50-100 for one unit-turn. v4 priced
                    # this at 0.45x of a price it then never realised, because
                    # FERTILIZER was not in SELLABLE at all.
                    add(pos, ["COLLECT_FERTILIZER"],
                        P["fert_collect_weight"] * _out(obs, "FERTILIZER"))
                continue
    return tasks


def _fertilizer_gain(crop, age, units, price, weight=1.0, days_left=99):
    cd = CROPS[crop]
    if cd["ongoing"]:
        # Fertilizer doubles every scheduled production on an ongoing crop --
        # strawberry goes from 0.24 to 0.47 units/tile-day, which moves it
        # several places up the asset ranking. Cover is three days inclusive,
        # so the gain is one extra unit per production inside that window.
        covered = min(3, max(0, days_left))
        prods = max(0, covered // max(1, cd["interval"]))
        if prods <= 0:
            return 0.0
        return prods * price * weight
    wstart = (cd["max_yield_day"] + 1) // 2
    days = max(0, min(cd["max_yield_day"], age + 2) - max(age, wstart) + 1)
    headroom = max(0, cd["max_yield"] - units)
    return min(days, headroom) * price * 0.9 * weight


def _animal_for_structure(P, c, structure, days_left, shed, obs):
    best, best_val = None, -1e18
    for animal, a in ANIMALS.items():
        if a["structure"] != structure:
            continue
        if c[animal] >= P["target_" + animal.lower()]:
            continue
        if days_left < a["first_yield_day"] + 1:
            continue
        val = animal_value_per_day(obs, animal, P)
        if shed.get(animal, 0) > 0:
            val += 1e6
        if val > best_val:
            best, best_val = animal, val
    return best


# =============================================================================
# 8. ASSIGNMENT
# =============================================================================


_ASSIGN_INF = float("inf")


def _hungarian(cost, n_rows, n_cols):
    """Minimum-cost assignment of every row to a distinct column.

    Jonker-Volgenant shortest-augmenting-path Hungarian, O(n^2 m). `cost` is a
    flat row-major list of length n_rows * n_cols and n_rows <= n_cols.
    Returns the column chosen for each row.
    """
    if n_rows == 0 or n_cols == 0:
        return []
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [_ASSIGN_INF] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _ASSIGN_INF
            j1 = 0
            base = (i0 - 1) * n_cols
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[base + j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                break
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def _best_matching(score, n_rows, n_cols):
    """Max-weight matching of rows to columns. Pads when rows outnumber cols."""
    if n_rows == 0 or n_cols == 0:
        return []
    if n_rows > n_cols:
        pad = n_rows - n_cols
        wide = []
        for r in range(n_rows):
            wide.extend(score[r * n_cols:(r + 1) * n_cols])
            wide.extend([0.0] * pad)
        res = _hungarian([-x for x in wide], n_rows, n_rows)
        return [c if c < n_cols else -1 for c in res]
    return _hungarian([-x for x in score], n_rows, n_cols)


def endgame_drop_value(obs, invs, ui):
    """Value of the sellable goods unit `ui` is carrying."""
    return sum(_price(obs, k) * v for k, v in invs[ui].items()
               if k in SELLABLE and v > 0)


def assign(P, tasks, units, invs, n, shed, seeds, obs=None, endgame=False):
    ops = [["PASS"] for _ in units]
    claimed, busy = set(), set()
    plant_budget = dict(seeds)
    reserved = {}

    sheds = _shed_tiles(n)
    # Voronoi zoning: every task belongs to whichever unit is closest right
    # now. Units still *may* cross the farm for a big enough prize, but a task
    # someone else is standing next to is discounted, which keeps crews working
    # compact blocks instead of criss-crossing.
    owner = {}
    for ti, task in enumerate(tasks):
        best_u, best_d = 0, 10 ** 9
        for ui, pos in enumerate(units):
            d = _dist(pos, task["pos"])
            if d < best_d:
                best_d, best_u = d, ui
        owner[ti] = best_u

    if endgame and obs is not None:
        # Last day: anything not in the shed cannot be sold, so hauling beats
        # every other job on the farm.
        for ui in range(len(units)):
            val = endgame_drop_value(obs, invs, ui)
            if val > 0:
                tasks = tasks + [{"pos": min(sheds, key=lambda s: _dist(units[ui], s)),
                                  "op": ["DROP"], "value": val * 4.0, "need": None}]
                owner[len(tasks) - 1] = ui
    pairs = []
    for ti, task in enumerate(tasks):
        for ui, pos in enumerate(units):
            d = _dist(pos, task["pos"])
            need = task["need"]
            extra = 0
            if need is not None and invs[ui].get(need, 0) <= 0:
                extra = 1 + min(_dist(pos, s) for s in sheds)
            tw = P.get("travel_weight", 1.0)
            density = task["value"] / (1.0 + tw * (d + extra))
            if owner.get(ti, ui) != ui:
                density *= P.get("poach_penalty", 1.0)
            pairs.append((density, ti, ui))
    pairs.sort(key=lambda p: -p[0])

    # Greedy pair-picking is at best a 1/2-approximation of the best possible
    # allocation: it takes the single highest-scoring cell first, which can
    # strand a unit whose only good job was the one just taken. Solving the
    # actual max-weight matching costs well under a millisecond at farm scale
    # (10 hands x a few hundred tasks), so when assign_mode is "optimal" we
    # solve it and put those pairs at the front of the queue. Everything after
    # them is the old greedy order, which is what repairs a matched pair the
    # feasibility rules below reject (no seed budget, nothing in the shed).
    if P.get("assign_mode", "greedy") == "optimal" and tasks and units:
        nu, nt = len(units), len(tasks)
        dense = [0.0] * (nu * nt)
        for density, ti, ui in pairs:
            dense[ui * nt + ti] = density
        matched = _best_matching(dense, nu, nt)
        front = []
        for ui, ti in enumerate(matched):
            if ti >= 0 and dense[ui * nt + ti] > 0:
                front.append((dense[ui * nt + ti], ti, ui))
        front.sort(key=lambda q: -q[0])
        seen_pair = {(q[1], q[2]) for q in front}
        pairs = front + [q for q in pairs if (q[1], q[2]) not in seen_pair]

    for _density, ti, ui in pairs:
        if ti in claimed or ui in busy:
            continue
        task = tasks[ti]
        op, need = task["op"], task["need"]
        pos = units[ui]

        if op[0] == "PLANT" and plant_budget.get(op[1], 0) <= 0:
            continue

        if need is not None and invs[ui].get(need, 0) <= 0:
            avail = shed.get(need, 0) - reserved.get(need, 0)
            if avail <= 0:
                continue
            if _is_shed_adjacent(pos, n):
                take = min(avail, 10 if need == "WHEAT" else 1)
                ops[ui] = ["PICKUP", need, take]
                reserved[need] = reserved.get(need, 0) + take
            else:
                mv = _step_towards(pos, min(sheds, key=lambda s: _dist(pos, s)))
                if mv is None:
                    continue
                ops[ui] = mv
            busy.add(ui)
            continue

        if tuple(pos) == task["pos"]:
            ops[ui] = op
            if op[0] == "PLANT":
                plant_budget[op[1]] -= 1
        else:
            mv = _step_towards(pos, task["pos"])
            if mv is None:
                continue
            ops[ui] = mv
        claimed.add(ti)
        busy.add(ui)

    # Carrying valuable produce is unbanked revenue: SELL only draws from the
    # shed. Idle units always haul; busy units interrupt once the load is worth
    # more than the job they are on.
    drop_floor = P.get("drop_stack_value", 0.0)
    for ui, pos in enumerate(units):
        if ui in busy:
            if drop_floor <= 0 or obs is None:
                continue
            worth = sum(_price(obs, k) * v for k, v in invs[ui].items()
                        if k in SELLABLE and k != "WHEAT")
            if worth < drop_floor:
                continue
        carried = sum(v for k, v in invs[ui].items() if k in SELLABLE and k != "WHEAT")
        if carried <= 0:
            continue
        if _is_shed_adjacent(pos, n):
            ops[ui] = ["DROP"]
        else:
            mv = _step_towards(pos, min(sheds, key=lambda s: _dist(pos, s)))
            if mv:
                ops[ui] = mv
    return ops


# =============================================================================
# 9. MARKET ENGINE
# =============================================================================


def opponent_imminent_supply(obs, days_left):
    """Per product, units the opponent is about to put on the market.

    Their farm is public, so a harvest they are one turn from taking is
    observable: a one-shot crop at full yield, an ongoing crop holding units,
    an animal at or near its hold cap. Nothing here uses hidden state.

    This exists because of a published result. Two independent top-ten
    write-ups reached the same conclusion -- "the farmer and hand tapes
    contribute almost nothing; the market tape carries the frontier gain", and
    "the current edge comes from inventory-sale timing rather than another
    change in herd composition". The concrete mechanism both trace back to is a
    one-turn front-run: when the two farms mirror each other, whoever sells the
    premium line first takes the top of a convex glut curve and the other one
    eats the slide.
    """
    seat = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if len(farms) < 2:
        return {}
    day = int(obs.get("day", 0) or 0)
    out = {}
    for row in (farms[1 - seat].get("tiles") or []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            units = int(tile.get("yield_units", 0) or 0)
            if units <= 0:
                continue
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                cd = CROPS.get(crop)
                if cd is None:
                    continue
                age = day - int(tile.get("planted_day", day))
                if age >= cd["first_yield_day"]:
                    out[crop] = out.get(crop, 0.0) + units
            elif tile.get("animal") in ANIMALS:
                item = ANIMALS[tile["animal"]]["product"]
                out[item] = out.get(item, 0.0) + units
    return out


def mirror_score(obs):
    """How alike the two public farms are, 0..1. 1 means a true mirror match."""
    seat = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if len(farms) < 2:
        return 0.0
    mine = farms[seat].get("tiles") or []
    theirs = farms[1 - seat].get("tiles") or []
    same = total = 0
    for ry in range(min(len(mine), len(theirs))):
        for rx in range(min(len(mine[ry]), len(theirs[ry]))):
            a, b = mine[ry][rx], theirs[ry][rx]
            total += 1
            ka = a.get("crop") or a.get("animal") or a.get("kind") if isinstance(a, dict) else a
            kb = b.get("crop") or b.get("animal") or b.get("kind") if isinstance(b, dict) else b
            if ka == kb:
                same += 1
    return (same / total) if total else 0.0


def _hands_wanted(P, c, money, n_unlocked_tiles, day=0):
    """Hands are the cheapest asset in the game -- hire up to what the work
    (and the bank) justifies.

    The whole crew costs the Fibonacci sum: twelve hands are $376 for the day,
    fifteen are $1,596. Against a farm turning over thousands a day the hire
    price is not the constraint, so `hire_cash_frac` is a floor-protector, not
    a budget. Every top ladder tape runs 10-15 hands from the first week; v4
    ran 7 and did 4,972 unit-turns against their 7,321.
    """
    animals = c["GOOSE"] + c["COW"] + c["SHEEP"]
    crops = sum(c[k] for k in CROPS)
    jobs = labour_used(P, crops, animals) + 0.35 * n_unlocked_tiles
    by_work = int(math.ceil(jobs / (24.0 * P["capacity_util"]))) - 1
    by_work = max(P["hands_min"], by_work) + int(P.get("hands_slack", 0))
    cap = P["hands_max"]
    if day >= P.get("hands_late_day", 99):
        cap = max(cap, P.get("hands_max_late", cap))
    return max(0, min(cap, by_work, affordable_hands(P, money)))


# --- same-turn selling, and which line to spend a slot on -------------------
#
# Both mechanisms below are read from the decoded market channel of a top-10
# public agent, not guessed. The write-ups all say the market tape carries the
# frontier gain; this is what that tape is doing.
#
# 1. The interpreter applies every unit action *before* it processes the market
#    (kaggriculture.py: _apply_unit_action at line 904, _process_market at 910).
#    A DROP this turn is therefore sellable this turn. Our agent read the shed
#    as it stood at the start of the turn, so every harvest reached the market a
#    turn late -- measured as selling on 19% of turns against their 47%.
#
# 2. Which product gets one of the two SELL slots is scored, not just sorted by
#    value: the opponent's standing supply of that line, times how violently its
#    price collapses when glutted (that is the interpreter's own `above_target`
#    per product -- melon 3.6 and wool 3.2 fall off a cliff, wheat 0.2 barely
#    moves), times price, times log of our quantity so one big pile cannot
#    monopolise every slot.

GLUT_WEIGHT = {item: MARKET_PARAMS[item]["above_target"] for item in PRODUCTS}


def projected_shed(obs, shed, ops, invs, n):
    """The shed as the market will see it, counting this turn's DROP/PLACE."""
    out = {k: int(v) for k, v in shed.items() if v}
    if not ops:
        return out
    farm = obs["farms"][obs["player"]]
    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    for ui, op in enumerate(ops):
        if ui >= len(units) or ui >= len(invs):
            continue
        if not isinstance(op, (list, tuple)) or not op:
            continue
        if not _is_shed_adjacent(units[ui], n):
            continue
        inv = invs[ui] or {}
        if op[0] == "DROP":
            deposits = list(inv.items())
        elif op[0] == "PLACE" and len(op) >= 2 and op[1] not in ANIMALS:
            want = int(op[2]) if len(op) >= 3 else 1
            deposits = [(op[1], min(want, inv.get(op[1], 0)))]
        else:
            continue
        for item, qty in deposits:
            room = max(0, 100 - sum(out.values()))
            take = min(max(0, int(qty or 0)), room)
            if take:
                out[item] = out.get(item, 0) + take
    return out


def opponent_exposure(obs):
    """How much of each product the opponent's farm could put on the market.

    Every standing tile counts, not just the harvest-ready ones: the question
    is whether they can flood this line at all, not whether they will today.
    """
    seat = int(obs.get("player", 0) or 0)
    farms = obs.get("farms") or []
    if len(farms) < 2:
        return {}
    out = {}
    for row in (farms[1 - seat].get("tiles") or []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            if crop in CROPS:
                out[crop] = out.get(crop, 0.0) + max(1.0, float(tile.get("yield_units", 0) or 0))
            animal = tile.get("animal")
            if animal in ANIMALS:
                item = ANIMALS[animal]["product"]
                out[item] = out.get(item, 0.0) + 1.0 + max(0.0, float(tile.get("yield_units", 0) or 0))
            if tile.get("fertilizer_available", False):
                out["FERTILIZER"] = out.get("FERTILIZER", 0.0) + 1.0
    return out


def sell_priority(obs, item, qty, price, exposure):
    return ((1.0 + exposure.get(item, 0.0))
            * GLUT_WEIGHT.get(item, 1.0)
            * max(1.0, float(price))
            * math.log1p(max(0.0, float(qty))))


def hour_bias(P, hour):
    """Sell harder on the turns where the town has just taken stock out.

    `_town_consume` runs *after* `_process_market` in the same step, so the
    price we are quoted at hour h reflects the consumption at h-1. The town
    centre fires every 12 steps and the shops every 4, which makes hour 1 and
    hour 13 the two best-priced selling turns of the day; hour 0 is the other
    one, because the engine empties every unit inventory into the shed
    overnight and it is the first chance to move a full shed.

    Measured over 304 top-ladder episodes, the field puts 11.6% of its volume
    in hour 0 and 8.8% in hour 1 against a flat expectation of 4.2%. This is
    that schedule, not a guess.
    """
    gain = float(P.get("sell_hour_gain", 0.0))
    if gain <= 0:
        return 1.0
    if hour == 0 or hour % 12 == 1:
        return 1.0 + gain
    if hour % 4 == 1:
        return 1.0 + 0.5 * gain
    return max(0.1, 1.0 - 0.5 * gain)


def _herd_labour_cap(P, c, money, day):
    """How many head the crew can actually service.

    An animal is ~4.5 unit-turns a day (feed, care, collect, harvest, walking).
    Buying past this is how a herd starves: the cash was there, the hands were
    not.
    """
    hands = affordable_hands(P, money)
    if day >= P.get("hands_late_day", 99):
        hands = max(hands, min(P.get("hands_max_late", hands), hands))
    capacity = labour_capacity(P, min(hands, P["hands_max"]))
    crop_load = P["cost_per_crop_day"] * sum(c[k] for k in CROPS)
    room = (capacity - crop_load) * float(P.get("herd_labour_share", 1.0))
    return max(0, int(room // max(1e-6, P["cost_per_animal_day"])))


def market_orders(obs, P, farm, private, c, empty_count, day, hour, days_left, n,
                  unlocked_tiles, ops=None, invs=None):
    """Priority order matters: only 10 orders per turn are processed."""
    money = float(farm["money"])
    seeds = private["seeds"]
    # The shed the market will actually see: what is in it now plus what this
    # turn's DROPs put there, because unit actions resolve first.
    if ops is not None and invs is not None and P.get("same_turn_sell", True):
        shed = projected_shed(obs, private["shed"], ops, invs, n)
    else:
        shed = private["shed"]
    inv = obs["market"]["inventory"]
    gap = income_gap(obs, farm, shed, day, float(P.get("income_gap_cap", 8.0)))
    wheat_now = market_price("WHEAT", inv.get("WHEAT", MARKET_I0))
    reserve = working_reserve(P, c, shed, wheat_now, gap)
    orders = []

    animals_owned = c["GOOSE"] + c["COW"] + c["SHEEP"]
    feed_need = int(math.ceil(animals_owned * min(P["feed_buffer_days"], max(1, days_left))))

    # ---- 1. hire (cheap, highest ROI in the game) -----------------------
    if hour <= 2 and days_left >= 1:
        want = _hands_wanted(P, c, money, unlocked_tiles, day)
        already = int(farm.get("hires_today", 0))
        spend, hired = 0, 0
        for cost in _fib_costs(want)[already:]:
            if money - (spend + cost) < P["hire_cash_floor"]:
                break
            spend += cost
            hired += 1
        room = 7 if hour == 0 else 4
        for _ in range(min(hired, room)):
            orders.append(["HIRE"])
        money -= spend

    # ---- 2. sell (raises the cash everything else depends on) -----------
    shed_total = sum(shed.values())
    pressure = shed_total >= P["shed_pressure"] or day >= P["dump_day"]
    exposure = opponent_exposure(obs) if P.get("glut_priority", True) else {}
    sells = []
    for item in SELLABLE:
        have = shed.get(item, 0)
        if have <= 0:
            continue
        if item == "WHEAT":
            have = max(0, have - feed_need)
            if have <= 0:
                continue
        if item == "FERTILIZER":
            # Keep a few units back for the tiles worth doubling; sell the
            # rest, and sell it *early*. Nothing in the town consumes
            # fertilizer, so its price only ever falls -- the units we hold
            # while the opponent sells are worth strictly less tomorrow.
            have = max(0, have - int(P.get("fert_stock", 4)))
            if have <= 0:
                continue
        floor = 0 if pressure else P["sell_floor"].get(item, 0.5) * MARKET_PARAMS[item]["base"]
        if not pressure and str(P.get("opponent_mode", "off")) == "counter":
            # Hold out when they are dumping this item; sell into it when they
            # are buying it up. Shed pressure still overrides -- a full shed
            # loses harvests, which costs more than a bad price.
            floor *= _counter_bias(item, float(P.get("opp_counter_gain", 0.6)))
        mult = sell_multiplier(item, day) if P.get("use_schedule", True) else 1.0
        mult *= hour_bias(P, hour)
        cap = have if pressure else max(1, int(round(P["sell_chunk"] * mult)))
        k = sellable_units(item, inv.get(item, MARKET_I0), have, floor, cap)
        if k > 0:
            spot = market_price(item, inv.get(item, MARKET_I0))
            if P.get("glut_priority", True):
                value = sell_priority(obs, item, k, spot, exposure)
            else:
                value = spot * k
            sells.append((value * mult, item, k))
    # Front-run the mirror. Only a couple of SELL slots reach the engine each
    # turn, so which product uses them is the decision, not how much. When the
    # opponent's farm mirrors ours and they are holding a premium line ready to
    # harvest, that line's price is about to slide -- sell into it first.
    if float(P.get("front_run_gain", 0.0)) > 0 and not pressure:
        mirror = mirror_score(obs)
        if mirror >= float(P.get("mirror_threshold", 0.75)):
            incoming = opponent_imminent_supply(obs, days_left)
            gain = float(P.get("front_run_gain", 0.0))
            boosted = []
            for value, item, k in sells:
                threat = incoming.get(item, 0.0)
                if threat > 0:
                    # A convex glut curve means their dump costs us more the
                    # bigger it is; scale the bump by what is coming.
                    value *= 1.0 + gain * min(2.0, threat / 8.0)
                boosted.append((value, item, k))
            sells = boosted

    sells.sort(reverse=True)
    for _v, item, k in sells[:int(P.get("sell_orders", 4))]:
        orders.append(["SELL", item, k])
        money += _v

    # ---- 3. feed wheat (starving livestock is catastrophic) -------------
    # Feed is bought, not grown. A wheat tile earns about what a wheat unit
    # costs, and the tile could have been a strawberry instead -- so the farm
    # buys its feed every morning and keeps its land for the expensive crops.
    short = feed_need - shed.get("WHEAT", 0)
    if short > 0 and animals_owned > 0:
        wp = market_price("WHEAT", inv.get("WHEAT", MARKET_I0) - 1)
        feed_cap = P["feed_buy_price_cap"]
        if str(P.get("opponent_mode", "off")) == "counter":
            # Their wheat dump is our cheap feed: raise the cap exactly when
            # they are the ones pushing the price down.
            feed_cap *= _counter_bias("WHEAT", float(P.get("opp_counter_gain", 0.6)))
        if wp <= feed_cap * MARKET_PARAMS["WHEAT"]["base"]:
            keep = float(P.get("hire_cash_floor", 60))
            k = min(short, int(max(0.0, money - keep) // max(1, wp)), 40)
            if k > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", k])
                money -= k * wp

    # ---- 4. livestock ----------------------------------------------------
    # Moved ahead of seed and land, which is the single largest behavioural
    # change in this rewrite. A cow costs $400 and returns ~$240 of milk plus a
    # free fertilizer every day: it pays for itself inside three days and then
    # compounds, and every day it is not standing in a pasture is a day of that
    # return thrown away. Every top ladder tape buys 3 cows and a sheep on turn
    # zero with the starting $3,000; v4 bought its first animal on day 11
    # because seeds, land and a twelve-day feed runway were served first.
    free_struct = {"COOP": c["COOP"], "PASTURE": c["PASTURE"]}
    ranked = []
    for animal, a in ANIMALS.items():
        if days_left < a["first_yield_day"] + P["animal_min_extra_days"]:
            continue
        profit = animal_lifetime_profit(obs, animal, days_left, P)
        if profit > 0:
            ranked.append((profit / a["cost"] * asset_bias(animal, day, money), animal))
    ranked.sort(reverse=True)
    herd = animals_owned + sum(shed.get(a, 0) for a in ANIMALS)
    herd_cap = min(P["max_herd"], _herd_labour_cap(P, c, money, day))
    wheat_price = market_price("WHEAT", inv.get("WHEAT", MARKET_I0))
    waiting = sum(shed.get(a, 0) for a in ANIMALS)
    for _r, animal in ranked:
        if herd >= herd_cap:
            break
        a = ANIMALS[animal]
        pending = shed.get(animal, 0)
        # BUY_ANIMAL puts the head in the *shed*; a pen is only needed to PLACE
        # it, and pens are free to build. Gating the purchase on an empty pen
        # already standing is what stalled the opening: on turn zero there are
        # no pens, so no animal could be bought, so the $3,000 went into seed
        # and the first cow arrived on day 11.
        slots = free_struct[a["structure"]] - pending + max(
            0, int(P.get("animal_pipeline", 4)) - waiting)
        deficit = P["target_" + animal.lower()] - c[animal] - pending
        want = min(slots, deficit, P["animals_per_turn"], herd_cap - herd)
        k = 0
        while k < want:
            # Each extra head adds a few days of feed to the money that must
            # stay in the bank -- days, not the fortnight v4 insisted on.
            runway = max(float(P["feed_runway_days"]), gap)
            need_cash = (a["cost"] * (k + 1) + P["reserve_base"]
                         + runway * (herd + k + 1) * wheat_price
                         + P["animal_cash_buffer"])
            if money < need_cash:
                break
            k += 1
        if k > 0:
            orders.append(["BUY_ANIMAL", animal, k])
            money -= k * a["cost"]
            herd += k
            free_struct[a["structure"]] -= k

    # ---- 5. seeds --------------------------------------------------------
    # Recompute the reserve: the heads bought a few lines up are feed liability
    # the opening-turn figure could not know about, and seed used to eat it.
    bought_heads = {a: 0 for a in ANIMALS}
    for o in orders:
        if o[0] == "BUY_ANIMAL":
            bought_heads[o[1]] = bought_heads.get(o[1], 0) + int(o[2])
    if any(bought_heads.values()):
        shed_after = dict(shed)
        for a, k in bought_heads.items():
            shed_after[a] = shed_after.get(a, 0) + k
        reserve = working_reserve(P, c, shed_after, wheat_now, gap)
    slots = plantable_slots(P, c, empty_count, money, days_left)
    want_seeds = seed_demand(obs, P, c, slots, seeds, days_left, shed, gap, money)
    cash_crop, _cash_tiles = cash_crop_plan(obs, P, days_left, gap, money)
    # Cash crop first in the queue, then dearest seed: whichever order the
    # value model prefers, the money runs out partway down this list, so the
    # tile that pays on day four has to be bought before the one that pays on
    # day ten.
    for crop, k in sorted(want_seeds.items(),
                          key=lambda kv: (kv[0] != cash_crop, -CROPS[kv[0]]["seed"])):
        cost = CROPS[crop]["seed"]
        # The cash crop is exempt from the feed reserve: at $10-20 a tile it is
        # what *closes* the income gap the reserve exists to survive. Holding
        # $700 of feed money while refusing to spend $60 on the wheat that pays
        # for the next fortnight of feed is how the opening stalls.
        limit = reserve
        if crop == cash_crop:
            limit = min(reserve, float(P.get("hire_cash_floor", 40))
                        + cost * float(P.get("cash_crop_tiles", 10)) * 0.0)
        afford = int(max(0.0, money - limit) // cost)
        k = min(k, afford)
        if k > 0:
            orders.append(["BUY_SEED", crop, k])
            money -= k * cost

    # ---- 6. land ---------------------------------------------------------
    # The third purchase costs $4,000 for 25 tiles the crew has to staff as
    # well. Ladder winners take two quadrants and stop; v4 took all three and
    # filled the extra land with wheat.
    bought = len(farm["unlocked_quadrants"]) - 1
    if bought < len(LAND_PRICES) and bought < int(P.get("land_max", 3)):
        price = LAND_PRICES[bought]
        used_frac = 1.0 - empty_count / max(1.0, float(unlocked_tiles))
        if (P["land_min_day"][bought] <= day <= P["land_last_day"][bought]
                and used_frac >= P["land_min_used"]
                and money >= price + reserve + P["land_reserve"][bought]):
            orders.append(["BUY_LAND"])
            money -= price

    # Order *index* is a contested resource. _process_market pairs our i-th
    # order against their i-th order and quotes both against the same
    # pre-commit inventory; whoever holds the lower index sells into a market
    # the other one has not moved yet. Our queue led with HIRE -- 43% of every
    # slot we spend -- so our premium lines were routinely quoted after theirs.
    # The decoded top-10 market channel puts its sells at the front verbatim:
    #     action["market"] = (sells + remainder)[:10]
    # Selling first also banks the cash before the same turn's purchases are
    # checked against it.
    if P.get("sell_first", True):
        sells_q = [o for o in orders if o and o[0] == "SELL"]
        rest_q = [o for o in orders if not (o and o[0] == "SELL")]
        orders = sells_q + rest_q

    # Terminal turns: every dollar spent is a dollar off the score, and nothing
    # bought can pay itself back. A hired hand cannot walk anywhere useful, feed
    # keeps an animal alive past the last harvest that can reach the shed, and a
    # seed will never be planted. The public v19 write-up puts it as "final step
    # 718: replace useless BUY/HIRE with all SELLs".
    tail = int(P.get("no_buy_last_turns", 0))
    if tail > 0 and (day * 24 + hour) >= (30 * 24 - tail):
        orders = [o for o in orders if o and o[0] == "SELL"]
    return orders[:MAX_ORDERS]


# =============================================================================
# 10. ENTRY POINT
# =============================================================================


def _decide(obs, config, P):
    """One policy's decision for this turn: (unit ops, market orders).

    Split out of agent() so the same code can be run several times with
    different parameters in the same turn -- see _ensemble_decide.
    """
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    day = int(obs["day"])
    hour = int(obs["hour"])
    tiles = farm["tiles"]
    n = len(tiles)

    total_days = 30
    if config is not None:
        try:
            total_days = max(1, int(config["episodeSteps"]) // int(config["turnsPerDay"]))
        except Exception:
            total_days = 30
    days_left = max(0, total_days - day)

    shed = dict(private["shed"])
    seeds = dict(private["seeds"])
    c, empty = census(tiles, n)
    unlocked_tiles = 25 * len(farm["unlocked_quadrants"])

    # Price everything against where the market will be when we sell, not
    # where it is now. Every valuation below reads this through _out().
    market_outlook(obs, P, days_left)

    money = float(farm["money"])
    plan = plan_tiles(obs, P, c, empty, days_left, money, shed, seeds, n)

    # Empty pens with no animal on the way are dead land -- mark them for DIG.
    capital = money - working_reserve(P, c, shed, _price(obs, "WHEAT")) - P["animal_cash_buffer"]
    surplus = {}
    for structure in ("COOP", "PASTURE"):
        incoming = sum(shed.get(a, 0) for a in ANIMALS if ANIMALS[a]["structure"] == structure)
        cheapest = min(ANIMALS[a]["cost"] for a in ANIMALS if ANIMALS[a]["structure"] == structure)
        affordable = int(max(0.0, capital) // cheapest)
        planned = sum(1 for v in plan.values() if v[0] == structure)
        surplus[structure] = max(0, c[structure] - incoming - affordable - planned)
    tasks = build_tasks(obs, P, tiles, n, day, days_left, c, plan, shed, surplus)

    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    raw = private.get("inventories", [{}])
    invs = [dict(raw[i]) if i < len(raw) else {} for i in range(len(units))]

    # Terminal window. The public meta write-up isolated this as a promotion on
    # its own: their controller took over at step 712 and moving it to 717 --
    # five more turns of ordinary production -- was worth a promotion, because
    # "step 718 executes, action index 719 does not". Ours took over for the
    # whole of day 29, spending 24 turns hauling when the shed audit says $7k
    # still finished unbanked. Haul late and hard, not early and long.
    step_now = day * 24 + hour
    total_steps = total_days * 24
    endgame = step_now >= total_steps - int(P.get("endgame_turns", 24))
    ops = assign(P, tasks, units, invs, n, shed, seeds, obs, endgame=endgame)
    orders = market_orders(obs, P, farm, private, c, len(empty), day, hour,
                           days_left, n, unlocked_tiles, ops, invs)
    return ops, orders


# =============================================================================
# 11. ENSEMBLE
# =============================================================================

# Measured on this agent: 1.2 ms mean and 3.0 ms at p95 against a 1000 ms
# actTimeout -- roughly 330x of unused budget every turn. An ensemble is the
# cheapest way to convert that headroom into decision quality.
#
# What is ensembled, and what deliberately is not:
#
#   unit ops    -- voted. Each committee member proposes an op per unit; the
#                  op with the most votes wins. Members disagree exactly where
#                  the incumbent's choice was marginal, which is where a single
#                  parameterisation's idiosyncratic error lives.
#   market      -- NOT voted, taken from the incumbent alone. The market plan
#                  carries budget invariants (feed runway, working reserve,
#                  order priority); mixing two members' orders can spend the
#                  same cash twice. Averaging structured plans is how ensembles
#                  produce illegal states.
#
# Below `ensemble_consensus` support the incumbent's op stands. The committee
# overrules only where it actually agrees, so a diffuse split cannot outvote
# the tuned policy with an op that nothing much supports.

_COMMITTEE = None
_COMMITTEE_KEY = None

# Knobs the committee is allowed to vary. Spatial and valuation weights only:
# these change *which job a unit takes*, which is what the vote is over.
# Budget and market knobs are excluded -- varying them would make members
# disagree about spending while only their movement gets used.
_ENSEMBLE_KNOBS = ("travel_weight", "fert_weight", "poach_penalty",
                   "care_weight", "fert_collect_weight", "cost_per_crop_day",
                   "cost_per_animal_day", "capacity_util")


def _committee(P, k, spread):
    """`k` deterministic parameter variants around P, built once and cached.

    Deterministic on purpose: the same committee every turn and in both seats,
    so a result is reproducible and a paired test stays paired.
    """
    global _COMMITTEE, _COMMITTEE_KEY
    key = (k, spread, len(ENSEMBLE_MEMBERS),
           tuple((n, P.get(n)) for n in _ENSEMBLE_KNOBS))
    if _COMMITTEE_KEY == key and _COMMITTEE is not None:
        return _COMMITTEE

    # An explicit committee beats jitter: these members are distinct policies
    # that each earned a rating, so their disagreements carry information.
    # Jitter only ever explores a ball around one policy.
    if ENSEMBLE_MEMBERS:
        out = []
        for member in ENSEMBLE_MEMBERS[:k] if k else ENSEMBLE_MEMBERS:
            Q = dict(P)
            Q.update(member)
            out.append(Q)
        _COMMITTEE_KEY, _COMMITTEE = key, out
        return out

    import random as _random
    rng = _random.Random(0xC0FFEE)
    out = []
    for _ in range(k):
        Q = dict(P)
        for name in _ENSEMBLE_KNOBS:
            v = P.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                Q[name] = max(1e-6, v * (1.0 + rng.gauss(0.0, spread)))
        out.append(Q)
    _COMMITTEE_KEY, _COMMITTEE = key, out
    return out


# --- adaptive play -----------------------------------------------------------
#
# Everything above plays the same way whether it is winning by $40k or losing by
# $40k, and whether the opponent is farming livestock or flooding the market
# with wheat. Two adaptations, both cheap and both switchable:
#
#   bandit  Exp3 over the committee. Members are reweighted from an in-game
#           signal, so the vote leans on whichever policy is actually working
#           in *this* game. Exp3 rather than UCB because the reward here is
#           non-stationary by construction -- what works on day 3 is wrong on
#           day 27 -- and Exp3 is the standard choice for an adversarial or
#           drifting reward. The weight floor keeps every member exploring; a
#           bandit that fully commits cannot notice the game has changed.
#
#   risk    Score-aware play. Behind late, press: spend reserves, plant the
#           long high-value crops, accept variance because a narrow loss and a
#           wide loss score identically on a skill ladder. Ahead late, protect:
#           hold cash, avoid new commitments that mature after the horizon.
#
# State is per-episode and reset when the step counter goes backwards, which is
# how a fresh episode announces itself inside one process.

_BANDIT = None
_BANDIT_EP = None


def _step_index(obs):
    """Turn counter that works in *both* seats.

    The interpreter copies day and hour to every player but leaves `step` on
    seat 0's observation only, so anything keyed on obs["step"] silently reads
    0 forever in seat 1 -- which is how per-episode state failed to reset and
    the tape opponents replayed their opening move for 720 turns.
    """
    step = obs.get("step")
    if step is None:
        return int(obs.get("day", 0) or 0) * 24 + int(obs.get("hour", 0) or 0)
    return int(step or 0)


def _wealth(obs, seat=None):
    """Cheap in-game value signal: cash plus everything already sellable.

    Deliberately excludes standing crops and animals. Counting them would make
    the signal jump when we *plant* rather than when the plan pays off, and the
    bandit would learn to reward spending.
    """
    seat = obs["player"] if seat is None else seat
    try:
        total = float(obs["farms"][seat].get("money", 0) or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0
    private = obs.get("private") or {}
    # SELLABLE only: the shed also holds livestock waiting for a pen, and those
    # have no market price -- _price would raise on them.
    for item, qty in (private.get("shed") or {}).items():
        if item in SELLABLE:
            total += _price(obs, item) * float(qty or 0)
    for inv in (private.get("inventories") or []):
        for item, qty in (inv or {}).items():
            if item in SELLABLE:
                total += _price(obs, item) * float(qty or 0)
    return total


def _opponent_wealth(obs):
    seat = 1 - int(obs.get("player", 0) or 0)
    try:
        return float(obs["farms"][seat].get("money", 0) or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _bandit_state(obs, n_members, floor):
    """Per-episode Exp3 state, reset when a new episode starts."""
    global _BANDIT, _BANDIT_EP
    step = _step_index(obs)
    if _BANDIT is None or _BANDIT_EP is None or step < _BANDIT_EP or \
            len(_BANDIT.get("w", [])) != n_members:
        _BANDIT = {"w": [1.0] * n_members, "share": [0.0] * n_members,
                   "last_day": int(obs.get("day", 0) or 0),
                   "last_wealth": _wealth(obs)}
    _BANDIT_EP = step
    return _BANDIT


def _bandit_weights(obs, n_members, eta, floor):
    """Exp3 probabilities over members, updated once per in-game day.

    One update a day, not one a turn: a turn's wealth change is mostly noise
    from market drift, and updating on noise is how a bandit converges to
    whichever member happened to act during a good hour.
    """
    st = _bandit_state(obs, n_members, floor)
    day = int(obs.get("day", 0) or 0)
    if day > st["last_day"]:
        wealth = _wealth(obs)
        gained = wealth - st["last_wealth"]
        scale = max(abs(st["last_wealth"]), 500.0)
        reward = max(-1.0, min(1.0, gained / scale))       # bounded, as Exp3 needs
        total_share = sum(st["share"]) or 1.0
        for i in range(n_members):
            credit = st["share"][i] / total_share
            st["w"][i] *= _exp(eta * reward * credit)
        biggest = max(st["w"]) or 1.0
        st["w"] = [max(w / biggest, 1e-6) for w in st["w"]]   # rescale, keep ratios
        st["share"] = [0.0] * n_members
        st["last_day"] = day
        st["last_wealth"] = wealth

    total = sum(st["w"]) or 1.0
    probs = [w / total for w in st["w"]]
    # Mix with the uniform distribution so no member is ever silenced.
    return [floor + (1.0 - floor * n_members) * p for p in probs], st


def _exp(x):
    import math
    return math.exp(max(-8.0, min(8.0, x)))


def _adapt_params(P, obs, days_left):
    """Score-aware parameter shift. Returns P unchanged unless risk is on."""
    mode = str(P.get("adaptive_mode", "off"))
    if mode not in ("risk", "both"):
        return P
    day = int(obs.get("day", 0) or 0)
    if day < int(P.get("risk_late_day", 18)):
        return P

    mine, theirs = _wealth(obs), _opponent_wealth(obs)
    denom = max(abs(mine), abs(theirs), 1000.0)
    deficit = (theirs - mine) / denom                       # >0 means behind
    if abs(deficit) < 0.05:
        return P

    gain = float(P.get("risk_gain", 0.45))
    press = max(-1.0, min(1.0, deficit)) * gain
    Q = dict(P)
    # Behind: free up capital and take on more, because a narrow loss and a wide
    # loss score the same. Ahead: hold reserves and stop starting things that
    # mature after the horizon.
    Q["reserve_base"] = max(0.0, float(P.get("reserve_base", 260)) * (1.0 - press))
    Q["reserve_per_tile"] = max(0.0, float(P.get("reserve_per_tile", 9.0)) * (1.0 - press))
    Q["liquid_weight"] = max(0.0, float(P.get("liquid_weight", 1.0)) * (1.0 + press))
    Q["seed_lookahead"] = max(1, int(round(float(P.get("seed_lookahead", 3))
                                           * (1.0 + press * 0.5))))
    if press > 0:
        Q["animal_cash_buffer"] = max(0.0, float(P.get("animal_cash_buffer", 400))
                                      * (1.0 - press))
    return Q


# --- opponent modelling -------------------------------------------------------
#
# obs["market"]["inventory"] is shared and exact. Buying removes units from it,
# selling adds them, and price moves with the distance from I0 -- so a player who
# dumps wheat drives the wheat price down for both of us.
#
# Because we know our own submitted orders, the opponent's net trade follows by
# subtraction rather than inference:
#
#     opponent_flow[item] = inventory_delta[item] + our_buys - our_sells
#
# The one approximation: we assume our own orders executed. They can be rejected
# for cash or capacity, and that error lands in the opponent estimate. It is
# small and unbiased over a day, which is the window the counter-play uses.

_OPP = None


def _reset_opponent():
    global _OPP
    _OPP = {"inv": None, "flow": {}, "orders": [], "step": -1, "days": 0}
    return _OPP


def _our_market_delta(orders):
    """Signed effect of *our* own orders on market inventory."""
    out = {}
    for o in orders or []:
        if not isinstance(o, (list, tuple)) or len(o) < 2:
            continue
        op = o[0]
        item = o[1]
        qty = float(o[2]) if len(o) > 2 and isinstance(o[2], (int, float)) else 1.0
        if op == "SELL":
            out[item] = out.get(item, 0.0) + qty          # we add supply
        elif op in ("BUY_SEED", "BUY_PRODUCT"):
            out[item] = out.get(item, 0.0) - qty          # we remove supply
    return out


def _observe_opponent(obs, halflife):
    """Update the exponential estimate of the opponent's net trade per item.

    Positive flow means they are *selling* that item (adding supply, pushing the
    price down). Negative means they are buying it up (price rising).
    """
    global _OPP
    step = _step_index(obs)
    if _OPP is None or step < _OPP["step"]:
        _reset_opponent()
    inv = dict((obs.get("market") or {}).get("inventory") or {})
    if not inv:
        return {}
    prev = _OPP["inv"]
    if prev is not None:
        ours = _our_market_delta(_OPP["orders"])
        decay = 0.5 ** (1.0 / max(halflife, 1e-6))
        for item, now in inv.items():
            delta = float(now) - float(prev.get(item, now))
            theirs = delta - ours.get(item, 0.0)
            _OPP["flow"][item] = _OPP["flow"].get(item, 0.0) * decay + theirs
    _OPP["inv"] = inv
    _OPP["step"] = step
    return _OPP["flow"]


def _remember_orders(orders):
    if _OPP is not None:
        _OPP["orders"] = list(orders or [])


def _opponent_profile():
    """A cheap archetype read from the flow estimate.

    Not a classifier with a posterior -- three scores that say what they are
    doing right now. Livestock farms consume wheat and produce milk, wool and
    eggs; crop farms do the reverse. `aggression` is total traded volume, which
    is what decides whether countering them is worth anything at all.
    """
    if not _OPP or not _OPP["flow"]:
        return {"livestock": 0.0, "crop": 0.0, "aggression": 0.0}
    f = _OPP["flow"]
    animal_out = sum(max(0.0, f.get(k, 0.0)) for k in ("MILK", "WOOL", "EGG"))
    wheat_in = max(0.0, -f.get("WHEAT", 0.0))
    crop_out = sum(max(0.0, f.get(k, 0.0))
                   for k in ("CARROT", "TOMATO", "MELON", "STRAWBERRY"))
    volume = sum(abs(v) for v in f.values())
    total = animal_out + wheat_in + crop_out + 1e-9
    return {"livestock": (animal_out + wheat_in) / total,
            "crop": crop_out / total,
            "aggression": volume}


def _counter_bias(item, gain):
    """Multiplier on our willingness to sell `item` right now.

    Above 1 means hold out for a better price; below 1 means sell into their
    demand. They buy -> supply falls -> price rises -> we sell into it. They
    dump -> price falls -> we wait rather than sell into a glut we did not
    create.
    """
    if not _OPP or not _OPP["flow"]:
        return 1.0
    flow = _OPP["flow"].get(item, 0.0)
    scale = max(1.0, sum(abs(v) for v in _OPP["flow"].values()) / max(1, len(_OPP["flow"])))
    norm = max(-1.0, min(1.0, flow / (scale * 4.0)))
    return 1.0 + gain * norm          # they sell (flow>0) -> raise our floor


def _f32(v):
    """Round to float32, the precision xgboost splits on."""
    import struct
    try:
        return struct.unpack("f", struct.pack("f", float(v)))[0]
    except (OverflowError, ValueError):
        return 0.0


def _xgb_features(obs, member_index):
    """The state the arbiter sees. Must match tools/train_xgb.py exactly."""
    seat = int(obs.get("player", 0) or 0)
    try:
        farm = obs["farms"][seat]
        opp = obs["farms"][1 - seat]
    except (KeyError, IndexError, TypeError):
        return [0.0] * 10
    crops = animals = 0
    for row in farm.get("tiles") or []:
        cells = row if isinstance(row, (list, tuple)) else [row]
        for t in cells:
            if isinstance(t, dict):
                if t.get("kind") == "PLANT":
                    crops += 1
                elif t.get("animal"):
                    animals += 1
    shed = sum((obs.get("private") or {}).get("shed", {}).values() or [0])
    feats = [
        float(obs.get("day", 0) or 0),
        float(obs.get("hour", 0) or 0),
        float(farm.get("money", 0) or 0) / 1000.0,
        float(opp.get("money", 0) or 0) / 1000.0,
        float(len(farm.get("hands") or []) + 1),
        float(crops),
        float(animals),
        float(len(farm.get("unlocked_quadrants") or [])),
        float(shed),
        float(member_index),
    ]
    # Opponent features are appended, never inserted: a model trained on the
    # first ten still indexes the same columns.
    prof = _opponent_profile()
    flow = (_OPP or {}).get("flow") or {}
    feats += [
        float(prof["livestock"]),
        float(prof["crop"]),
        float(min(prof["aggression"], 500.0)),
        float(max(-99.0, min(99.0, flow.get("WHEAT", 0.0)))),
        float(max(-99.0, min(99.0, flow.get("MILK", 0.0) + flow.get("WOOL", 0.0)
                             + flow.get("EGG", 0.0)))),
        float(max(-99.0, min(99.0, sum(flow.get(k, 0.0) for k in
                             ("CARROT", "TOMATO", "MELON", "STRAWBERRY"))))),
    ]
    n = int(XGB_MODEL.get("n_features") or len(feats)) if XGB_MODEL else len(feats)
    return feats[:n] + [0.0] * max(0, n - len(feats))


def _xgb_margin(feats):
    trees = XGB_MODEL.get("trees") or []
    if not trees:
        return 0.0
    xs = [_f32(v) for v in feats]
    total = float(XGB_MODEL.get("bias", 0.0))
    for tree in trees:
        i = 0
        while tree[i][0] != -1:
            fi, thr, yes, no = tree[i]
            i = yes if xs[fi] < thr else no
        total += tree[i][1]
    return total


def _xgb_weights(obs, n_members):
    """P(win | state, member) per member, used as the vote weight.

    A learned generalisation of MEMBER_WEIGHTS: the bucketed table has to see
    every state band separately, while a tree ensemble interpolates -- which is
    the sample efficiency that matters when every training point costs a match.
    """
    if not XGB_MODEL.get("trees"):
        return None
    import math
    out = []
    for mi in range(n_members):
        m = _xgb_margin(_xgb_features(obs, mi))
        out.append(1.0 / (1.0 + math.exp(-max(-16.0, min(16.0, m)))))
    biggest = max(out) or 1.0
    return [max(0.02, w / biggest) for w in out]


def _weight_bucket(obs):
    """State key for the learned vote weights: coarse day and cash bands.

    Deliberately coarse. A fine grid would need far more self-play games than
    the budget allows to fill, and an unfilled bucket is a weight learned from
    one episode -- which is how a learned component becomes a random one.
    """
    try:
        day = int(obs.get("day", 0) or 0)
        money = float(obs["farms"][obs["player"]].get("money", 0) or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return "1|1"
    d = 0 if day <= 8 else (1 if day <= 16 else 2)
    c = 0 if money < 1500 else (1 if money < 6000 else 2)
    return f"{d}|{c}"


def _member_weights(obs, n_members):
    if not MEMBER_WEIGHTS:
        return [1.0] * n_members
    w = MEMBER_WEIGHTS.get(_weight_bucket(obs))
    if not w:
        return [1.0] * n_members
    out = [float(w[i]) if i < len(w) else 1.0 for i in range(n_members)]
    return [max(0.0, x) for x in out]


def _ensemble_decide(obs, config, P, k, spread, consensus):
    """Run the incumbent plus `k` variants; vote per unit on the op."""
    P = _adapt_params(P, obs, None)
    ops, orders = _decide(obs, config, P)
    committee = _committee(P, k, spread)
    weights = _member_weights(obs, len(committee) + 1)

    if str(P.get("arbiter_model", "table")) == "xgb":
        learned = _xgb_weights(obs, len(committee) + 1)
        if learned:
            weights = [w * l for w, l in zip(weights, learned)]

    bandit = None
    if str(P.get("adaptive_mode", "off")) in ("bandit", "both"):
        probs, bandit = _bandit_weights(obs, len(committee) + 1,
                                        float(P.get("bandit_eta", 0.35)),
                                        float(P.get("bandit_floor", 0.08)))
        # The learned table says who to trust in this *kind* of state; the
        # bandit says who is working in *this* game. Multiply: neither should
        # be able to silence the other on its own.
        weights = [w * p * len(probs) for w, p in zip(weights, probs)]

    votes = [{} for _ in ops]
    proposals = [ops] + [None] * len(committee)
    for i, op in enumerate(ops):
        votes[i][tuple(op)] = weights[0] + 1e-6   # incumbent breaks exact ties

    members = weights[0]
    for mi, Q in enumerate(committee, start=1):
        try:
            cand, _ = _decide(obs, config, Q)
        except Exception:                     # noqa: BLE001
            continue                          # a broken member never votes
        proposals[mi] = cand
        w = weights[mi] if mi < len(weights) else 1.0
        if w <= 0.0:
            continue                          # a member the arbiter learned to mute
        members += w
        for i, op in enumerate(cand):
            if i < len(votes):
                key = tuple(op)
                votes[i][key] = votes[i].get(key, 0.0) + w

    need = consensus * max(members, 1e-9)
    out = []
    for i, op in enumerate(ops):
        best, best_n = tuple(op), votes[i].get(tuple(op), 0.0)
        for cand_op, n in votes[i].items():
            if n > best_n:
                best, best_n = cand_op, n
        chosen = best if best_n >= need else tuple(op)
        out.append(list(chosen))
        if bandit is not None:
            # Credit every member whose proposal matched what actually got
            # played. Blaming the vote's winner alone would punish members for
            # being outvoted, which is not their decision to answer for.
            for mi, prop in enumerate(proposals):
                if i < len(prop) and tuple(prop[i]) == chosen:
                    bandit["share"][mi] += 1.0
    return out, orders


def agent(obs, config=None):
    P = PARAMS
    # Once per turn, before any policy runs -- the committee calls _decide many
    # times and the market must be observed exactly once.
    if str(P.get("opponent_mode", "off")) != "off":
        _observe_opponent(obs, float(P.get("opp_halflife", 6.0)))
    k = int(P.get("ensemble_k", 0) or 0)
    if k <= 0 and str(P.get("adaptive_mode", "off")) in ("risk", "both"):
        # Risk adaptation does not need a committee.
        ops, orders = _decide(obs, config, _adapt_params(P, obs, None))
        _remember_orders(orders)
        return {"farmer": list(ops[0]), "hands": [list(o) for o in ops[1:]],
                "market": orders}
    if k > 0:
        ops, orders = _ensemble_decide(
            obs, config, P, k,
            float(P.get("ensemble_spread", 0.12)),
            float(P.get("ensemble_consensus", 0.5)))
    else:
        ops, orders = _decide(obs, config, P)
    _remember_orders(orders)
    return {"farmer": list(ops[0]), "hands": [list(o) for o in ops[1:]],
            "market": orders}
