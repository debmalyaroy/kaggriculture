"""The runtime a mined route ships inside: replay, repair, clamp, liquidate.

`routes.py --build` renders `TEMPLATE` with a compressed 719-turn route and
writes a single self-contained `main.py`. Nothing here imports anything the
ladder does not already have.

Our previous tape template (`opponents.py:_TAPE_TEMPLATE`) had a position
repair layer and nothing else. That is fine for a sparring partner, which only
has to be *faithful*. It is not enough to score with, because it is missing the
three layers every agent at the top of this leaderboard carries -- and each of
them is an interpreter fact rather than a strategy:

**Projected shed.** `kaggriculture.py:904` applies unit actions; `:910` then
processes the market. Produce dropped this turn is sellable *this* turn. A tape
whose route sells what the shed held at the start of the turn banks every
harvest a turn late.

**No clamping.** `_commit_unit` sells per unit and returns False only for the
unit that finds the shed empty -- everything before it already sold and paid.
An oversized SELL therefore partially fills and is harmless (it costs a slot,
nothing more). The old clamp-to-projected-shed layer cancelled real income
whenever the projection undercounted (measured 2026-08-21: $2,146 vs $58,292
final bank on the same seed and opponent for a drop-and-sell-same-turn route);
only malformed-order filtering survives.

**Sell-first ordering.** `_process_market` walks orders by *index*, pairing our
i-th against their i-th and quoting both against the same pre-commit inventory.
The lower index sells into a market the other player has not moved yet. This is
worth reordering for and nothing else: the ablation that *created* new early
sells collapsed at ~28 interventions a game, while pure re-ordering scored
53/53. Change order, never inventory.

**Terminal liquidation.** Unsold inventory scores nothing, and step 718 is the
last executable turn. Day 29 alone carries 11.8% of the field's season volume.

**Weed repair.** A weed spawns on 0.5% of empty tiles nightly and silently
no-ops the PLANT or BUILD_PASTURE the route expected. DIG, retry next turn,
then replay one turn behind for a bounded catch-up window rather than
abandoning the rest of the route.
"""
from __future__ import annotations

TEMPLATE = '''"""Kaggriculture route agent -- {label}

Route: {route_id} ({team}), episode {episode} seat {seat}.
Recorded bank {bank:,.0f} against {opp_bank:,.0f}.
Mined from Kaggle's published episode archive by src/kaggriculture/data/routes.py on {built}.

An open-loop 719-turn route plus four repair layers. The route is replayed by
`obs["step"]` -- a `"shared": true` schema property the core copies into both
seats, verified 0..718 in each. Everything else exists because the interpreter
punishes a mistake silently rather than raising.
"""
import base64
import copy
import json
import math
import zlib

_ROUTE = json.loads(zlib.decompress(base64.b85decode("{payload}")).decode("utf-8"))

# Shop-conditioned branches (Kaito v48 pattern, 2026-08-30): the first town
# shop is world-decided and fixed once drawn; per-first-shop majority votes
# agree 0.71-0.998 within partitions vs 0.86 globally, so the branch IS the
# team's play for that world. All branches share the pre-unlock prefix by
# construction; dispatch happens once, when the first shop appears, and the
# town is shared so both seats agree. Empty dict = classic single-route.
_BRANCHES_RAW = "{branchpack}"
_BRANCHES = (json.loads(zlib.decompress(base64.b85decode(
    _BRANCHES_RAW)).decode("utf-8")) if _BRANCHES_RAW else {{}})

# The official price curve, transcribed from kaggriculture.py. Verified against
# the vendored 1.32.4 interpreter at 2,835 points across inventory 9,000-11,200:
# maximum difference zero. It is inlined rather than imported because the
# submission is one stdlib-only file.
#   (base, equilibrium, scale, below_func, below_target, above_func, above_target)
_MARKET_PARAMS = {{
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}}
_PRICE_FLOOR = 1
# Engine 1.32.7 hinge rebalance (PR #1399): flipped by
# scripts/engine_swap_1327.py when the LADDER's replays show 1.32.7.
_ENGINE_1327 = True
if _ENGINE_1327:
    _MARKET_PARAMS["CARROT"] = (35, 10000, 450, "hinge", 1.0, "sqrt", 0.7)
    _MARKET_PARAMS["TOMATO"] = (60, 10000, 200, "hinge", 0.4, "sqrt", 0.6)
    _MARKET_PARAMS["EGG"] = (50, 10000, 332, "hinge", 0.4, "log", 0.2)
_IMPACT_SLOTS = {impact_slots}      # rank SELLs by self-induced price damage

# Mirror tie-break. A fork of this agent replays the same 719 actions, so
# `_process_market` quotes both sides against the same pre-commit inventory and
# the episode ends in an exact tie -- measured at 91,075/91,075, 125,093/125,093
# and 118,735/118,735 on three seeds. A tie is not a loss, but it is not a win.
#
# The break: once the boards are provably identical, move a slice of the NEXT
# turn's premium SELL into this turn and delete it from next turn. We then sell
# into an unmoved market while the mirror sells into one we have already moved.
# Net inventory over the two turns is unchanged, which is the invariant every
# published ablation says matters.
#
# Gated hard on purpose. It fires only when the opponent's public farm and money
# match ours exactly for several consecutive turns -- i.e. only in games that
# would otherwise be a guaranteed draw, which bounds the downside to zero-value
# games. Against a merely *similar* opponent it never arms.
_MIRROR_TIEBREAK = {mirror_tiebreak}
# Relay v2 (2026-08-11). The v1 layer never fired against the population that
# matters: it demanded EXACT public equality (including money, which diverges
# within turns between near-clones) and looked only one turn ahead. The public
# meta has since weaponised exactly this mechanic -- boatlee V16-RC2 (3-turn
# fixed lead on the fertilizer dump, 22 forks) and shiv17 (5-turn fixed lead
# counter) -- so the race is live and a fixed lead is a constant an opponent
# can pre-empt. v2: NEAR-structural detection (money excluded), a scan window
# rather than a fixed lead, FERTILIZER included, per-step repayment ledger.
_MIRROR_STREAK = 12        # consecutive NEAR-matches before arming
_MIRROR_START = 216        # first checkpoint the public relays use
_MIRROR_FRACTION = 1.0     # pull the whole scheduled dump: win the race outright
_MIRROR_MAX = 40           # cap on units moved per order
_MIRROR_COOLDOWN = 2
_RELAY_TOL = 4             # structural L1 distance still counted as near-mirror
# Measured 2026-08-11: a 6-turn window over ALL premium products loses the
# mirror by -1,401/game -- it reshapes the whole mid-game curve into pre-peak
# prices. The winning shape (boatlee's) is NARROW: the recurring FERTILIZER
# dump only, minimal lead, big batches only. Our lock only engages clones of
# OUR schedule (lead-0), so 2 turns wins that race with minimal distortion.
_RELAY_LEAD = 2
_RELAY_MIN_QTY = 8         # only dumps, not routine sales
_RELAY_MIN_QUOTE = 3       # never race into an already-crashed market
_PREMIUM = ("FERTILIZER",)

# Premium one-turn sell lead (C94 / V16-RC5 class), NOT mirror-gated.
# Loss audit 2026-08-27 (v33.0 pair, 120 ladder games): 34 of 89 losses tip
# at exactly day 11 and 19 more at days 17-18 -- we hold level through the
# premium window, then lose the sell race. Two independent public
# implementations move part of the NEXT turn's premium SELL forward one turn
# so it quotes into an unmoved market; the quantity is debited from the due
# turn via the relay ledger, so across the pair of turns the route sells the
# same units -- only sooner. Bounded: cap 10/item (C94's measured cap),
# demand-parity gated (pull only on turns with no town tick step%24 and no
# shop tick step%4 -- quote ahead of the demand tick, not into it), healthy
# quotes only, and off before the endgame machinery owns timing.
_PLEAD = {premium_lead}
_PLEAD_ITEMS = ("WOOL", "MILK", "MELON", "STRAWBERRY")
_PLEAD_CAP = 10
_PLEAD_MIN_QUOTE = 3
_PLEAD_START = 216         # day 9: the premium window opens
_PLEAD_STOP = 640          # endgame pull owns the tail
# Deposit advance: the premium lead measured ZERO effect on mined routes
# (2026-08-27 paired A/B) because they deposit produce the same turn they
# sell it -- at the pull turn the shed is empty. This extension lets the
# lead fire by ALSO advancing the deposit: if a shed-adjacent unit is
# carrying the item and its scheduled op this turn is PASS, its op becomes
# DROP (everything it carries was bound for the shed anyway; the tape's own
# later DROP no-ops on an empty inventory). Bounded to ONE converted unit
# per turn, and only when it enables a pull that this turn would otherwise
# skip. Requires _PLEAD.
_DEPOSIT_ADVANCE = {deposit_advance}

# Floored premium seller (disc 737879 class, paired-validated 19-1 by its
# author; our price-curve data agrees: premium peaks days 9-14 and crashes
# barely recover). From _FSELL_DAY, sell shed stock of premium goods in
# bounded batches ONLY above per-item price floors, using ONLY free market
# slots, never touching existing orders. Honors the endgame_pull retirement
# note ("reimplement explicit subtraction first"): quantities borrowed from
# the schedule's own NEXT SELL of the item (scanned within _FSELL_HORIZON)
# are debited through the relay ledger; stock with NO upcoming scheduled
# SELL is schedule-surplus -- the unsold-value failure class -- and sells
# freely. Stops before the terminal window, which owns liquidation.
_FSELL = {floor_seller}
_FSELL_DAY = 14
_FSELL_FLOORS = {{"MILK": 120, "WOOL": 140, "STRAWBERRY": 90, "MELON": 60}}
_FSELL_BATCH = 15
_FSELL_HORIZON = 96        # turns ahead to find the schedule's own SELL
_FSELL_STOP = 700          # terminal takeover owns the rest

# Endgame accelerator. Located from 67 real ladder games: through day 18 the
# games we win and the games we lose are indistinguishable (mean cash gap
# -2,317 against -3,413). The separation is entirely in days 21-29 -- in wins
# the gap goes +9,781 -> +13,899 -> +13,842, in losses +1,451 -> +39 -> -2,909.
# The deficit opens at a median of day 27 and 8 of 12 losses break in days
# 25-29.
#
# That window is the field's liquidation stampede -- day 29 alone is 11.8% of
# the season's sell volume -- and the price goes to whoever sells first. Our
# route sells on a fixed schedule, so whether we are first is luck.
#
# So: from `_ENDGAME_DAY`, sell what is already in the shed rather than waiting
# for the turn the route planned. This is not new inventory. Every unit here
# would have been sold by `_terminal_market` at step 718 regardless; the only
# question is whether it goes into a market the field has already flooded.
# WARNING: this layer's self-repayment depended on the retired _safe_market
# clamp (later orders shrinking to the emptied shed). With the clamp gone,
# enabling it would sell pulled quantities TWICE. routes.py refuses to build
# with it on until explicit subtraction is reimplemented.
_ENDGAME_PULL = {endgame_pull}
_ENDGAME_DAY = 24          # step 576; the gap in losses is still ~0 here
_ENDGAME_PER_TURN = 3      # extra order slots to spend per turn
_ENDGAME_CHUNK = 18        # units per pulled order

# Floor guard. The one adaptive layer with ladder evidence behind it.
_FLOOR_GUARD = {floor_guard}
_FLOOR_MULT = 1.0          # "floored" means the quote is at PRICE_FLOOR
_FLOOR_MAX_HELD = 60       # bounded backlog per product
_FLOOR_RELEASE = 20        # units released per turn once the price recovers
_FLOOR_STOP = 700          # stop holding before terminal liquidation

# Glut trigger. The $1 floor is far too late -- strawberry quoting $20 off a
# $120 base is already a squeeze the floor guard never sees. With
# `_GUARD_RATIO` > 0 the guard holds the part of a SELL whose marginal unit
# quotes below ratio x base. 0.0 keeps the pure floor trigger byte-for-byte.
_GUARD_RATIO = {guard_ratio}

# Swap-advance. While the guard is holding a depressed product, pull forward an
# already-scheduled SELL of a *healthy* product so revenue-per-turn holds
# instead of pausing. WARNING: its self-repayment depended on the retired
# _safe_market clamp (the route's later SELL shrinking to the emptied shed);
# with the clamp gone, enabling it would sell pulled quantities twice.
# routes.py refuses to build with it on until explicit subtraction is
# reimplemented. What was different from the failed endgame accelerator: this
# fires only while an equal value is being *held back*, only for products
# quoting above `_SWAP_HEALTHY` x base, only from SELLs the route itself
# scheduled inside `_SWAP_HORIZON`, and it is capped by the running
# deferred-minus-advanced value balance -- the sell rate is hedged, never
# accelerated.
_SWAP_ADVANCE = {swap_advance}
_SWAP_HORIZON = 36         # turns ahead a scheduled SELL may be pulled from
_SWAP_HEALTHY = 0.65       # advance only products quoting above this x base
_SWAP_MAX = 30             # units advanced per turn, cap

# Adaptive sell-timing. Most of the field is open-loop routes, so their dumps
# repeat at fixed phases of the 24-step day. We reconstruct the opponent's
# sells from the PUBLIC inventory delta (delta minus our own filled sells is a
# LOWER bound on theirs -- town drain and their BUY_PRODUCTs only shrink it,
# never inflate it), and when a recurring dump phase collides with one of our
# own scheduled SELLs inside the horizon, we pull that SELL forward to quote
# BEFORE the crash instead of into it. Prediction-driven layers have collapsed
# in every published ablation; this one survives their failure mode because it
# is in the `_endgame_pull` safety class: it only moves sells the route
# already scheduled, earlier -- it changes *when*, never *whether* -- and
# `_adaptive_repay` settles every pull against the turn it came from, so the
# total volume sold is unchanged. Worst case: we quoted a healthy price a few
# turns early.
#
# MEASURED 2026-08-16 (paired, 24 cells / 192 games, ledger version): diff
# -1.3pp, discordant 1-3, p=0.625 -- NO edge on the open-loop panel, so the
# flagship ships with this OFF. The slot-2 diversity route ships with it ON
# (operator order: live-ladder evidence; panel validity vs live is WEAK,
# rho=0.347). stage_second flips the flag in the built file.
_ADAPT_SELL = False
_ADAPT_HORIZON = 8         # turns ahead a scheduled SELL may be pulled from
_ADAPT_MIN_UNITS = 4       # inventory jump (net of ours) that counts as a dump
_ADAPT_MIN_OBS = 2         # recurrences at one phase before it is a pattern
_ADAPT_MAX_PULL = 2        # pulled orders per turn, cap
_ADAPT_HEALTHY = 0.45      # pull only products quoting above this x base
_ADAPT_COOLDOWN = 24       # turns before the same product may be pulled again
_ADAPT_STOP = 672          # day 28+: the endgame machinery owns sell timing

_SELLABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
             "WHEAT", "FERTILIZER")
_ANIMALS = ("COW", "SHEEP", "GOOSE")
_MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
_UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
             "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
             "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER")
_MOVES = ("NORTH", "SOUTH", "EAST", "WEST")

_SHED_CAP = 100
_MAX_ORDERS = 10
_LAST_STEP = 718
_WEED_CATCHUP = {weed_catchup}      # turns the route may replay one behind

# Market-queue ordering flags (see _sell_first). FEED_PIN is the turn-0 feed
# denial defense; SELL_FIRST is the blanket sells-before-buys reorder, which
# helps some route families and hurts others -- the promotion gate decides it
# per route.
_FEED_PIN = {feed_pin}
_SELL_FIRST = {sell_first}

# Per-seat scratch. Kaggle starts a fresh container per episode, so this is a
# within-episode cache only -- never a learning channel. Reset when the step
# counter goes backwards, which is how a reused worker process presents.
_STATE = {{0: {{}}, 1: {{}}}}


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return default if value is None else value


def _tile_at(farm, position):
    tiles = _get(farm, "tiles", []) or []
    try:
        x, y = int(position[0]), int(position[1])
    except (TypeError, ValueError, IndexError):
        return None
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _shed_adjacent(position, size):
    """DROP only reaches the shed from the four centre tiles (AGENTS.md)."""
    half = size // 2
    try:
        spot = (int(position[0]), int(position[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return spot in ((half - 1, half - 1), (half, half - 1),
                    (half - 1, half), (half, half))


def _align_hands(action, obs):
    """`hands` must line up positionally with farms[me]["hands"]."""
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    want = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    while len(hands) < want:
        hands.append(["PASS"])
    action["hands"] = hands[:want]
    if not (isinstance(action.get("farmer"), list) and action["farmer"]
            and action["farmer"][0] in _UNIT_OPS):
        action["farmer"] = ["PASS"]
    action["hands"] = [h if (isinstance(h, list) and h and h[0] in _UNIT_OPS)
                       else ["PASS"] for h in action["hands"]]
    return action


def _projected_shed(obs, action):
    """The shed as it will stand when the market resolves, later this turn.

    Unit actions land first (kaggriculture.py:904), the market second (:910).
    A DROP from a shed-adjacent tile therefore adds to what we can sell now.
    """
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    private = _get(obs, "private", {{}}) or {{}}
    shed = dict(_get(private, "shed", {{}}) or {{}})
    inventories = _get(private, "inventories", []) or []
    size = len(_get(farm, "tiles", []) or []) or 10

    positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
    positions += [list(p) for p in (_get(farm, "hands", []) or [])]
    ops = [action.get("farmer")] + list(action.get("hands") or [])

    room = max(0, _SHED_CAP - sum(int(v or 0) for v in shed.values()))
    for i, op in enumerate(ops):
        if not (isinstance(op, list) and op):
            continue
        if i >= len(positions) or not _shed_adjacent(positions[i], size):
            continue
        carried = inventories[i] if i < len(inventories) else {{}}
        if op[0] == "DROP":
            # DROP dumps the unit's whole inventory at the shed.
            for item, count in (carried or {{}}).items():
                n = min(int(count or 0), room)
                if n <= 0:
                    continue
                shed[item] = int(shed.get(item, 0) or 0) + n
                room -= n
        elif op[0] == "PLACE" and len(op) >= 2 and op[1] not in _ANIMALS:
            # PLACE item [n] beside the shed is a per-item deposit -- the
            # engine's second drop path. Missing it made the projection
            # undercount every route that deposits this way (48 PLACEs vs 9
            # DROPs in the route the old clamp bankrupted). Animals are
            # skipped: PLACE on a structure tile is placement, not a drop,
            # and SELL only draws products anyway. PICKUP (shed withdrawal)
            # is deliberately ignored: it only makes the projection
            # overestimate, and every consumer treats the projection as a
            # lower-is-riskier estimate.
            item = op[1]
            try:
                want = int(op[2]) if len(op) >= 3 else 1
            except (TypeError, ValueError):
                want = 1
            have = int((carried or {{}}).get(item, 0) or 0)
            n = min(max(0, want), have, room)
            if n > 0:
                shed[item] = int(shed.get(item, 0) or 0) + n
                room -= n
    return shed


def _safe_market(obs, action):
    """Strip malformed orders only. NEVER clamp or cancel a SELL.

    The engine's market loop is per-unit lockstep: `_commit_unit` sells one
    unit at a time and simply stops when the shed runs dry, so an oversized
    SELL partially fills and is harmless. This layer once clamped SELLs to a
    projected shed and dropped "empty" ones -- but any projection error
    cancels income the engine would have paid. Measured 2026-08-21 on route
    94711806_s1 (same seed, same opponent): with the clamp $2,146 final
    bank, without it $58,292. The clamp is gone; only order-shape filtering
    remains.
    """
    out = []
    for raw in (action.get("market") or []):
        if not (isinstance(raw, list) and raw and raw[0] in _MARKET_OPS):
            continue
        out.append(list(raw))
    action["market"] = out[:_MAX_ORDERS]
    return action


def _sell_first(action, step=None):
    """Order, never inventory. Two independent, flag-gated reorderings.

    _process_market pairs our i-th order against their i-th against the same
    pre-commit inventory, so the lower index quotes into an unmoved market.
    Nothing is added, removed or resized here -- only moved.

    `_FEED_PIN` (turn 0, feed denial defense): an opponent buying 14-19
    wheat ahead of our feed order raises the shared price so our budget
    affords fewer units; one unfed animal disappears on day 2 (measured
    -13,606/game across the field's audited losses). Pinning the route's own
    WHEAT purchase to slot 0 pairs it against the opponent's slot-0 order at
    the same pre-commit inventory. Seed and animal prices are fixed, so the
    orders it jumps lose nothing.

    `_SELL_FIRST` (all turns): move SELLs ahead of everything else. Measured
    +53/53 on one route family and -15,872/game on another whose recorded
    order already encodes its pairing -- so this is a per-route build flag,
    decided by the promotion gate, never assumed.
    """
    orders = list(action.get("market") or [])
    if step == 0 and _FEED_PIN:
        feed = [o for o in orders
                if isinstance(o, list) and len(o) >= 2
                and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT"]
        rest0 = [o for o in orders if o not in feed]
        orders = feed + rest0
    if _SELL_FIRST and step != 0:
        sells = [o for o in orders if o and o[0] == "SELL"]
        rest = [o for o in orders if not (o and o[0] == "SELL")]
        orders = sells + rest
    action["market"] = orders[:_MAX_ORDERS]
    return action


def _shape(name, value, scale_t=0.0):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "hinge":
        if not scale_t or scale_t <= 0:
            return value
        u = value / scale_t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return value


def _quote(item, inventory):
    """What the market pays for one unit at this inventory."""
    spec = _MARKET_PARAMS.get(item)
    if not spec:
        return 0.0
    base, equilibrium, scale, below_f, below_t, above_f, above_t = spec
    if inventory < equilibrium:
        amplitude = below_t * base / _shape(below_f, scale, scale)
        price = base + amplitude * _shape(below_f, equilibrium - inventory,
                                          scale)
    else:
        amplitude = above_t * base / _shape(above_f, scale, scale)
        price = base - amplitude * _shape(above_f, inventory - equilibrium,
                                          scale)
    return max(_PRICE_FLOOR, int(round(price)))


def _impact_slots(obs, action):
    """Rank the SELLs we are already making by the price damage they do.

    Every unit sold pushes that product's inventory up and its quote down, and
    the curves are convex above equilibrium -- melon and wool are `sq`, milk
    and strawberry `linear`, wheat only `log`. So the order in which our own
    sells land changes what they fetch: a big melon order executed after a big
    wool order is quoted into a market our wool has already moved.

    Score = quantity x (quote now - quote after selling that quantity). Highest
    first. This creates no order, resizes none, and moves no BUY/HIRE slot --
    SELLs are permuted *within the slots they already occupy*, so every
    non-SELL keeps its index. That constraint is the whole reason this is safe:
    the ablation that created new early sells collapsed, while pure re-ordering
    is what kept winning.
    """
    market = list(action.get("market") or [])
    rows = []
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    for index, order in enumerate(market):
        if not (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in _MARKET_PARAMS):
            continue
        item = order[1]
        try:
            qty = max(0, int(order[2]))
        except (TypeError, ValueError):
            qty = 0
        have = int(_get(inventory, item, 10000) or 0)
        now = float(_get(prices, item, _quote(item, have)) or 0)
        after = float(_quote(item, have + qty))
        # -index keeps the original order among equal scores, so a tie never
        # silently reshuffles a route that was already in a deliberate order.
        rows.append((float(qty) * max(0.0, now - after), -index, list(order)))
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [
        next(ranked) if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                         and o[1] in _MARKET_PARAMS) else o
        for o in market
    ]
    return action


def _public_signature(farm):
    """Everything about a farm both players can see. Order-stable."""
    tiles = _get(farm, "tiles", []) or []
    counts = {{}}
    yields = 0
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict):
                key = tile.get("crop") or tile.get("animal") or tile.get("kind")
                counts[key] = counts.get(key, 0) + 1
                yields += int(tile.get("yield_units", 0) or 0)
    return (
        int(_get(farm, "money", 0) or 0),
        len(_get(farm, "hands", []) or []),
        len(_get(farm, "unlocked_quadrants", []) or []),
        yields,
        tuple(sorted((str(k), v) for k, v in counts.items())),
    )


def _structure_sig(farm):
    """Money-free structural signature: what the farm has BUILT, not what it
    has banked. Between near-clones the bank diverges within turns while the
    build stays aligned for hundreds -- so money must not be in the signature
    (the v1 exact-equality check died on exactly this)."""
    counts = {{}}
    tiles = _get(farm, "tiles", []) or []
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict):
                key = tile.get("crop") or tile.get("animal") or tile.get("kind")
                if key:
                    counts[str(key)] = counts.get(str(key), 0) + 1
    return (len(_get(farm, "hands", []) or []),
            len(_get(farm, "unlocked_quadrants", []) or []),
            counts)


def _is_near_mirror(obs):
    """Is the opponent structurally CLOSE to us (not necessarily equal)?"""
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    if len(farms) < 2:
        return False
    a, b = _structure_sig(farms[me]), _structure_sig(farms[1 - me])
    dist = 2 * abs(a[0] - b[0]) + 3 * abs(a[1] - b[1])
    for key in set(a[2]) | set(b[2]):
        dist += abs(a[2].get(key, 0) - b[2].get(key, 0))
    return dist <= _RELAY_TOL


def _future_premium(step):
    """Scheduled SELLs of relay products in the next _RELAY_LEAD turns.

    Returns [(due_step, item, qty)] in schedule order. A scan window, not a
    fixed lead: whatever lead an opposing fixed-constant relay runs inside our
    family, our pull fires at the top of the window the moment the lock holds,
    so a 3- or 5-turn constant cannot pre-empt it.
    """
    out = []
    for t in range(step + 1, min(step + 1 + _RELAY_LEAD, len(_ROUTE))):
        turn = _ROUTE[t]
        if not isinstance(turn, dict):
            continue
        for order in turn.get("market") or []:
            if (isinstance(order, list) and len(order) >= 3
                    and order[0] == "SELL" and order[1] in _PREMIUM):
                try:
                    qty = max(0, int(order[2]))
                except (TypeError, ValueError):
                    continue
                if qty >= _RELAY_MIN_QTY:
                    out.append((t, order[1], qty))
    return out


def _mirror_tiebreak(obs, action, step, state):
    """Break a guaranteed draw by borrowing from tomorrow, and repaying it.

    Two halves, and the second is what keeps this honest. `repay` removes
    exactly what was pulled forward last turn, so across the pair of turns the
    route sells the same units of the same products -- only sooner. Without the
    repayment this would be plain oversell, which is the intervention that
    collapsed in every published ablation.
    """
    # ---- repay first: whatever was borrowed against THIS step's schedule
    # comes off this step's orders. The ledger is per-step, so a multi-turn
    # scan window repays exactly where it borrowed.
    ledger = state.setdefault("relay_due", {{}})
    due = ledger.pop(step, None)
    if due:
        market = []
        for order in (action.get("market") or []):
            if (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"
                    and order[1] in due):
                owed = due[order[1]]
                try:
                    qty = max(0, int(order[2]))
                except (TypeError, ValueError):
                    qty = 0
                take = min(owed, qty)
                due[order[1]] = owed - take
                qty -= take
                if qty <= 0:
                    continue
                order = [order[0], order[1], qty]
            market.append(order)
        action["market"] = market

    if not _MIRROR_TIEBREAK or step < _MIRROR_START or step + 1 >= len(_ROUTE):
        return action
    if step >= 640:            # terminal liquidation owns the endgame
        return action

    # ---- arm on a sustained NEAR-match of the money-free structure.
    if _is_near_mirror(obs):
        state["mirror"] = int(state.get("mirror", 0)) + 1
    else:
        state["mirror"] = 0
    if state["mirror"] < _MIRROR_STREAK:
        return action
    if step - int(state.get("last_break", -10 ** 9)) < _MIRROR_COOLDOWN:
        return action

    future = _future_premium(step)
    if not future:
        return action
    shed = _projected_shed(obs, action)
    existing = list(action.get("market") or [])
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    room = _MAX_ORDERS - len(existing)
    new_orders = []
    # Highest self-induced damage first: that is the order whose price is worth
    # protecting, and the one the mirror will suffer most for selling second.
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}

    def _damage(item, qty):
        held = int(_get(inventory, item, 10000) or 0)
        return qty * max(0.0, _quote(item, held) - _quote(item, held + qty))

    for due_step, item, planned in sorted(
            future, key=lambda r: -_damage(r[1], r[2])):
        if room <= 0:
            break
        if item in already:
            continue
        held = int(_get(inventory, item, 0) or 0)
        if _quote(item, held) < _RELAY_MIN_QUOTE:
            continue           # market already crashed; the race pays nothing
        qty = min(int(planned * _MIRROR_FRACTION), _MIRROR_MAX,
                  max(0, int(shed.get(item, 0) or 0)))
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        slot = ledger.setdefault(due_step, {{}})
        slot[item] = slot.get(item, 0) + qty
        already.add(item)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
        room -= 1
    if not new_orders:
        return action
    # Ahead of our own queue: the point is to quote before the mirror does.
    action["market"] = (new_orders + existing)[:_MAX_ORDERS]
    state["last_break"] = step
    return action


def _premium_lead(obs, action, step, state):
    """One-turn conservation lead on premium goods, against ANY opponent.

    The mirror tiebreak above only fires after a sustained structural match;
    the day-11 loss cluster is against ordinary opponents it never arms for.
    This is the same borrow-and-repay move with a one-turn window: if the
    route sells a premium good NEXT step, quote up to _PLEAD_CAP of it now,
    and register the debt in the same relay ledger the repay pass already
    settles. It changes when, never whether.
    """
    if not _PLEAD or step < _PLEAD_START or step >= _PLEAD_STOP:
        return action
    if step + 1 >= len(_ROUTE):
        return action
    if step % 24 == 0 or step % 4 == 0:
        return action              # a demand tick lands now; nothing to outrun
    nxt = _ROUTE[step + 1]
    if not isinstance(nxt, dict):
        return action
    ledger = state.setdefault("relay_due", {{}})
    owed_next = ledger.get(step + 1, {{}})
    existing = list(action.get("market") or [])
    room = _MAX_ORDERS - len(existing)
    if room <= 0:
        return action
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    shed = _projected_shed(obs, action)
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    advanced_unit = False

    def _advance_deposit(item):
        """Convert ONE idle shed-adjacent carrier's op to DROP so `item`
        reaches the shed this turn. Returns units of `item` made sellable."""
        nonlocal advanced_unit
        if not _DEPOSIT_ADVANCE or advanced_unit:
            return 0
        me = int(_get(obs, "player", 0) or 0)
        farms = _get(obs, "farms", []) or []
        farm = farms[me] if me < len(farms) else {{}}
        size = len(_get(farm, "tiles", []) or []) or 10
        positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
        positions += [list(p) for p in (_get(farm, "hands", []) or [])]
        carried = _get(_get(obs, "private", {{}}) or {{}}, "inventories", []) or []
        ops = [action.get("farmer")] + list(action.get("hands") or [])
        for i, op in enumerate(ops):
            if not (isinstance(op, list) and op and op[0] == "PASS"):
                continue
            if i >= len(positions) or not _shed_adjacent(positions[i], size):
                continue
            inv = carried[i] if i < len(carried) else {{}}
            have = int(_get(inv or {{}}, item, 0) or 0)
            if have <= 0:
                continue
            if i == 0:
                action["farmer"] = ["DROP"]
            else:
                hands = list(action.get("hands") or [])
                hands[i - 1] = ["DROP"]
                action["hands"] = hands
            advanced_unit = True
            # The DROP dumps the whole inventory; credit it all to the
            # projection (it was bound for the shed at end of day anyway).
            for what, count in (inv or {{}}).items():
                shed[what] = int(shed.get(what, 0) or 0) + int(count or 0)
            return have
        return 0

    new_orders = []
    for order in nxt.get("market") or []:
        if room <= 0:
            break
        if not (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in _PLEAD_ITEMS):
            continue
        item = order[1]
        if item in already:
            continue
        try:
            planned = max(0, int(order[2])) - int(owed_next.get(item, 0) or 0)
        except (TypeError, ValueError):
            continue
        if planned <= 0:
            continue
        held = int(_get(inventory, item, 0) or 0)
        if _quote(item, held) < _PLEAD_MIN_QUOTE:
            continue               # crashed market; leading pays nothing
        if int(shed.get(item, 0) or 0) <= 0:
            _advance_deposit(item)
        qty = min(planned, _PLEAD_CAP, max(0, int(shed.get(item, 0) or 0)))
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        slot = ledger.setdefault(step + 1, {{}})
        slot[item] = slot.get(item, 0) + qty
        already.add(item)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
        room -= 1
    if new_orders:
        action["market"] = (new_orders + existing)[:_MAX_ORDERS]
    return action


def _floor_seller(obs, action, step, state):
    """Sell premium shed stock above per-item price floors, mid-game.

    Free slots only, existing orders untouched. Borrowed quantities are
    debited against the schedule's own next SELL of the item through the
    relay ledger (the repay pass settles them); stock with no scheduled
    SELL ahead is surplus that would otherwise die unsold at step 718.
    """
    if not _FSELL or step // 24 < _FSELL_DAY or step >= _FSELL_STOP:
        return action
    existing = list(action.get("market") or [])
    room = _MAX_ORDERS - len(existing)
    if room <= 0:
        return action
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    shed = _projected_shed(obs, action)
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    ledger = state.setdefault("relay_due", {{}})
    new_orders = []
    for item, floor in _FSELL_FLOORS.items():
        if room <= 0:
            break
        if item in already:
            continue
        have = max(0, int(shed.get(item, 0) or 0))
        if have <= 0:
            continue
        if float(prices.get(item, 0) or 0) < floor:
            continue
        # The schedule's own next SELL of this item, if any, bounds the
        # borrowable quantity; absence means surplus.
        due_step, due_qty = None, 0
        for t in range(step + 1, min(step + 1 + _FSELL_HORIZON, len(_ROUTE))):
            turn = _ROUTE[t]
            if not isinstance(turn, dict):
                continue
            for order in turn.get("market") or []:
                if (isinstance(order, list) and len(order) >= 3
                        and order[0] == "SELL" and order[1] == item):
                    try:
                        q = max(0, int(order[2]))
                    except (TypeError, ValueError):
                        continue
                    q -= int((ledger.get(t) or {{}}).get(item, 0) or 0)
                    if q > 0:
                        due_step, due_qty = t, q
                    break
            if due_step is not None:
                break
        qty = min(have, _FSELL_BATCH, due_qty if due_step is not None else have)
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        if due_step is not None:
            slot = ledger.setdefault(due_step, {{}})
            slot[item] = slot.get(item, 0) + qty
        already.add(item)
        shed[item] = have - qty
        room -= 1
    if new_orders:
        action["market"] = (existing + new_orders)[:_MAX_ORDERS]
    return action


def _endgame_pull(obs, action, step, state):
    """In the last days, sell what we hold instead of waiting our turn.

    Bounded three ways so this cannot become the "create new sells" mistake
    that has collapsed in every published ablation: it only runs inside the
    final window, it only sells what is *already in the shed*, and every unit
    it moves was going to be liquidated at step 718 anyway. It changes when,
    never whether.
    """
    if not _ENDGAME_PULL:
        return action
    day = step // 24
    if day < _ENDGAME_DAY or step >= _LAST_STEP:
        return action
    existing = list(action.get("market") or [])
    room = min(_ENDGAME_PER_TURN, _MAX_ORDERS - len(existing))
    if room <= 0:
        return action
    shed = _projected_shed(obs, action)
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}

    rows = []
    for item in _SELLABLE:
        if item in already:
            continue
        have = max(0, int(shed.get(item, 0) or 0))
        if have <= 0:
            continue
        qty = min(have, _ENDGAME_CHUNK)
        mkt = int(_get(inventory, item, 10000) or 0)
        price = _quote(item, mkt)
        if price <= _PRICE_FLOOR:
            continue                       # already floored; nothing to race for
        rows.append((qty * price, item, qty))
    if not rows:
        return action
    rows.sort(reverse=True)
    pulled = [["SELL", item, qty] for _, item, qty in rows[:room]]
    # Ahead of the scheduled queue: the whole point is to quote first.
    action["market"] = (pulled + existing)[:_MAX_ORDERS]
    state["pulled"] = int(state.get("pulled", 0)) + sum(r[2] for r in rows[:room])
    return action


def _floor_guard(obs, action, step, state):
    """Do not dump into a floored market. Hold it and sell it later.

    Measured on 67 real ladder games, not inferred. Our sales are effectively
    constant -- 1,275 units in wins, 1,261 in losses -- because an open-loop
    route sells the same basket whatever happens. What varies is the opponent:
    in the games we lose they sell STRAWBERRY 316 -> 410, MELON 144 -> 212,
    MILK 280 -> 320 and WOOL 192 -> 228, flooding precisely the premium markets
    our revenue depends on. The route keeps selling into the collapse because
    it cannot see it.

    So: before sending a SELL, price it. If the marginal unit would fetch at or
    near the floor, cut the order back to the part that still earns, and carry
    the remainder forward. Held units are not lost -- `_terminal_market` sells
    everything on the last turn regardless, so the worst case is selling the
    same units later at a price that cannot be lower than the floor we avoided.

    This reads only `market.inventory`, which is public and shared. It does not
    predict the opponent, and it does not act on a prediction -- both of which
    have collapsed in every published ablation and in ours. It reacts to a
    price that has already moved.
    """
    if not _FLOOR_GUARD or step >= _FLOOR_STOP:
        return action
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    held = state.setdefault("held", {{}})
    out = []
    for order in (action.get("market") or []):
        if not (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in _MARKET_PARAMS):
            out.append(order)
            continue
        item = order[1]
        try:
            qty = max(0, int(order[2]))
        except (TypeError, ValueError):
            out.append(order)
            continue
        have = int(_get(inventory, item, 10000) or 0)
        floor = max(_FLOOR_MULT * _PRICE_FLOOR,
                    _GUARD_RATIO * _MARKET_PARAMS[item][0])
        # Walk the order down until the marginal unit is worth keeping.
        keep = qty
        while keep > 0 and _quote(item, have + keep - 1) <= floor:
            keep -= 1
        if keep >= qty:
            out.append(order)
            continue
        deferred = qty - keep
        # Never hold more than a bounded backlog: an unbounded one just moves
        # the whole season's revenue into the terminal dump, which floors too.
        carried = int(held.get(item, 0) or 0)
        if carried + deferred > _FLOOR_MAX_HELD:
            deferred = max(0, _FLOOR_MAX_HELD - carried)
            keep = qty - deferred
        if deferred <= 0:
            out.append(order)
            continue
        held[item] = carried + deferred
        state["deferred_units"] = int(state.get("deferred_units", 0)) + deferred
        # The hedge budget: what the deferred units would have fetched now.
        state["def_value"] = (float(state.get("def_value", 0.0))
                              + deferred * float(_quote(item, have)))
        if keep > 0:
            out.append([order[0], item, keep])
    # Release what we held as soon as the price recovers, using spare slots.
    room = _MAX_ORDERS - len(out)
    if room > 0 and held:
        shed = _projected_shed(obs, action)
        for item in sorted(held, key=lambda k: -held[k]):
            if room <= 0:
                break
            n = int(held.get(item, 0) or 0)
            if n <= 0:
                continue
            have = int(_get(inventory, item, 10000) or 0)
            floor = max(_FLOOR_MULT * _PRICE_FLOOR,
                        _GUARD_RATIO * _MARKET_PARAMS[item][0])
            if _quote(item, have) <= floor:
                continue                      # still floored, keep holding
            give = min(n, max(0, int(shed.get(item, 0) or 0)), _FLOOR_RELEASE)
            if give <= 0:
                continue
            out.append(["SELL", item, give])
            held[item] = n - give
            room -= 1
    # Swap-advance: hedge the value being held back with an already-scheduled
    # healthy SELL pulled forward. Only while something is actually held, and
    # only up to the running deferred-minus-advanced value balance.
    room = _MAX_ORDERS - len(out)
    if (_SWAP_ADVANCE and room > 0 and step + 1 < len(_ROUTE)
            and any(int(v or 0) > 0 for v in held.values())):
        allowance = (float(state.get("def_value", 0.0))
                     - float(state.get("adv_value", 0.0)))
        if allowance > 0.0:
            shed = _projected_shed(obs, action)
            selling = {{o[1] for o in out
                       if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
            scheduled = {{}}
            for t in range(step + 1, min(len(_ROUTE), step + 1 + _SWAP_HORIZON)):
                for order in _ROUTE[t].get("market") or []:
                    if (isinstance(order, list) and len(order) >= 3
                            and order[0] == "SELL" and order[1] in _MARKET_PARAMS):
                        try:
                            scheduled[order[1]] = (scheduled.get(order[1], 0)
                                                   + max(0, int(order[2])))
                        except (TypeError, ValueError):
                            pass
            rows = []
            for item, qty in scheduled.items():
                if item in selling or int(held.get(item, 0) or 0) > 0:
                    continue
                have = int(_get(inventory, item, 10000) or 0)
                price = float(_quote(item, have))
                if price < _SWAP_HEALTHY * _MARKET_PARAMS[item][0]:
                    continue                  # not healthy; no hedge into it
                avail = min(qty, max(0, int(shed.get(item, 0) or 0)), _SWAP_MAX)
                if avail > 0:
                    rows.append((price, item, avail))
            rows.sort(reverse=True)
            for price, item, avail in rows:
                if room <= 0 or allowance <= 0.0:
                    break
                take = min(avail, max(1, int(allowance // max(1.0, price))))
                if take <= 0:
                    continue
                out.append(["SELL", item, int(take)])
                state["adv_value"] = (float(state.get("adv_value", 0.0))
                                      + take * price)
                allowance -= take * price
                room -= 1
    action["market"] = out[:_MAX_ORDERS]
    return action


def _weed_repair(obs, action, step, state):
    """A weed under a planned PLANT/BUILD is a silent no-op. DIG, then catch up.

    Empty tiles spawn a weed nightly at 0.5%. The route does not know, so the
    op does nothing and every later turn for that unit is off by one. DIG this
    turn, retry the intended op next turn, then replay one turn behind for a
    bounded window so the rest of the route still lines up.
    """
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
    positions += [list(p) for p in (_get(farm, "hands", []) or [])]
    ops = [action.get("farmer")] + list(action.get("hands") or [])

    active = state.setdefault("weed", {{}})
    for actor in list(active):
        job = active[actor]
        age = step - job["start"]
        idx = int(actor)
        if age == 1:
            if idx < len(ops):
                ops[idx] = list(job["intended"])
        elif 2 <= age <= 1 + _WEED_CATCHUP:
            prev = _ROUTE[step - 1] if 0 < step - 1 < len(_ROUTE) else None
            if prev and idx < len(ops):
                prev_ops = [prev.get("farmer")] + list(prev.get("hands") or [])
                if idx < len(prev_ops) and isinstance(prev_ops[idx], list):
                    ops[idx] = list(prev_ops[idx])
        else:
            active.pop(actor, None)

    for i, op in enumerate(ops):
        if not (isinstance(op, list) and op):
            continue
        blocked = op[0] in ("PLANT", "BUILD_PASTURE", "BUILD_COOP")
        # PLACE is weed-blockable only as ANIMAL placement (needs a structure
        # tile). PLACE-as-shed-deposit ignores the standing tile entirely, so
        # repairing it would delay a deposit for nothing.
        if (op[0] == "PLACE" and len(op) >= 2 and op[1] in _ANIMALS):
            blocked = True
        if not blocked:
            continue
        if i >= len(positions):
            continue
        tile = _tile_at(farm, positions[i])
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            active[str(i)] = {{"start": step, "intended": list(op)}}
            ops[i] = ["DIG"]

    action["farmer"] = ops[0] if ops else ["PASS"]
    action["hands"] = ops[1:]
    return action


def _observe_opp_sells(obs, step, state):
    """Fold last turn's public inventory delta into per-phase dump stats.

    Runs every turn before any transform. `state["prev_inv"]` and
    `state["my_sells"]` are recorded at the end of the previous `agent()`
    call, so `delta - ours` bounds the opponent's sells from below: the town
    drain and any of their BUY_PRODUCT orders only make the bound lower.
    """
    if not _ADAPT_SELL:
        return
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    prev = state.get("prev_inv")
    mine = state.get("my_sells") or {{}}
    if isinstance(prev, dict) and step > 0:
        phase = (step - 1) % 24
        dumps = state.setdefault("opp_dumps", {{}})
        for item in _SELLABLE:
            try:
                delta = (int(_get(inventory, item, 0) or 0)
                         - int(prev.get(item, 0) or 0))
            except (TypeError, ValueError):
                continue
            theirs = delta - int(mine.get(item, 0) or 0)
            if theirs >= _ADAPT_MIN_UNITS:
                slot = dumps.setdefault(item, {{}})
                slot[phase] = int(slot.get(phase, 0) or 0) + 1
    # Snapshot for the NEXT turn's delta; my_sells is filled at return time.
    state["prev_inv"] = {{k: int(_get(inventory, k, 0) or 0)
                        for k in _SELLABLE}}
    state["my_sells"] = {{}}


def _adaptive_sell(obs, action, step, state):
    """Pull a scheduled SELL forward when a recurring opponent dump would
    land on or before it. Quote before the crash, not into it."""
    if not _ADAPT_SELL or step >= _ADAPT_STOP:
        return action
    dumps = state.get("opp_dumps") or {{}}
    if not dumps:
        return action
    existing = list(action.get("market") or [])
    room = min(_ADAPT_MAX_PULL, _MAX_ORDERS - len(existing))
    if room <= 0:
        return action
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    cooldown = state.setdefault("adapt_cd", {{}})
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    shed = None
    pulled = []
    for t in range(step + 1, min(len(_ROUTE), step + 1 + _ADAPT_HORIZON)):
        if room <= 0:
            break
        for order in _ROUTE[t].get("market") or []:
            if room <= 0:
                break
            if not (isinstance(order, list) and len(order) >= 3
                    and order[0] == "SELL" and order[1] in _MARKET_PARAMS):
                continue
            item = order[1]
            if item in already or step < int(cooldown.get(item, 0) or 0):
                continue
            phases = {{p for p, n in (dumps.get(item) or {{}}).items()
                      if int(n or 0) >= _ADAPT_MIN_OBS}}
            if not phases:
                continue
            # A predicted dump between now and our scheduled turn?
            if not any(((p - step) % 24) <= (t - step) for p in phases):
                continue
            try:
                qty = max(0, int(order[2]))
            except (TypeError, ValueError):
                continue
            if shed is None:
                shed = _projected_shed(obs, action)
            qty = min(qty, max(0, int(shed.get(item, 0) or 0)))
            if qty <= 0:
                continue
            have = int(_get(inventory, item, 10000) or 0)
            if _quote(item, have) < _ADAPT_HEALTHY * _MARKET_PARAMS[item][0]:
                continue                  # already crashed; floor guard's job
            pulled.append(["SELL", item, qty])
            shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
            already.add(item)
            cooldown[item] = step + _ADAPT_COOLDOWN
            # LEDGER the pull against the turn it came from: `_safe_market`
            # clamps to the shed, but production refills the shed before the
            # original order lands, and then the route sells the pulled units
            # a SECOND time -- more volume into the same market, lower price.
            # Recording the debt keeps this a pure timing shift.
            slot = state.setdefault("adapt_ledger", {{}}).setdefault(t, {{}})
            slot[item] = int(slot.get(item, 0) or 0) + qty
            room -= 1
    if not pulled:
        return action
    # Ahead of the queue: the whole point is to quote before their dump.
    action["market"] = (pulled + existing)[:_MAX_ORDERS]
    state["adapt_pulled"] = (int(state.get("adapt_pulled", 0) or 0)
                             + sum(o[2] for o in pulled))
    return action


def _adaptive_repay(action, step, state):
    """Settle the adaptive ledger: shrink this turn's scheduled SELLs by
    whatever `_adaptive_sell` already sold from them on an earlier turn."""
    ledger = state.get("adapt_ledger") or {{}}
    debt = ledger.pop(step, None)
    if not debt:
        return action
    out = []
    for order in (action.get("market") or []):
        if (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and int(debt.get(order[1], 0) or 0) > 0):
            item = order[1]
            try:
                qty = max(0, int(order[2]))
            except (TypeError, ValueError):
                out.append(order)
                continue
            take = min(qty, int(debt.get(item, 0) or 0))
            debt[item] = int(debt.get(item, 0) or 0) - take
            if qty - take > 0:
                out.append(["SELL", item, qty - take])
            continue
        out.append(order)
    # Debt not settled this turn (order shrank some other way) rolls forward.
    left = {{k: v for k, v in debt.items() if int(v or 0) > 0}}
    if left:
        nxt = ledger.setdefault(step + 1, {{}})
        for k, v in left.items():
            nxt[k] = int(nxt.get(k, 0) or 0) + int(v)
    action["market"] = out
    return action


def _terminal_market(obs, action):
    """Step 718 is the last executable turn; unsold inventory scores nothing.

    Quantities are deliberately over-asked (projected shed + everything any
    unit still carries): the engine's SELL partially fills per unit and stops
    at an empty shed, so over-asking costs nothing while a projection
    undercount would leave paid-for goods unsold.
    """
    shed = _projected_shed(obs, action)
    private = _get(obs, "private", {{}}) or {{}}
    for carried in (_get(private, "inventories", []) or []):
        for item, count in (carried or {{}}).items():
            try:
                shed[item] = int(shed.get(item, 0) or 0) + max(0, int(count or 0))
            except (TypeError, ValueError):
                continue
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    rows = []
    for index, item in enumerate(_SELLABLE):
        qty = max(0, int(shed.get(item, 0) or 0))
        if qty <= 0:
            continue
        # Highest total value first -- only ten orders fit, and the last
        # season-end dump is 11.8% of the field's whole volume.
        rows.append((qty * max(1.0, float(prices.get(item, 1) or 1)), -index,
                     item, qty))
    rows.sort(reverse=True)
    action["market"] = [["SELL", item, qty] for _, _, item, qty in rows[:_MAX_ORDERS]]
    return action


def agent(obs, config=None):
    try:
        seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
        step = int(_get(obs, "step", 0) or 0)
        state = _STATE[seat]
        # A step that did not advance means a new episode in a reused process.
        if step == 0 or step < int(state.get("last", -1)):
            state = _STATE[seat] = {{"last": step}}
        state["last"] = step
        if _BRANCHES and "branch_done" not in state:
            _shops = (_get(obs, "town", {{}}) or {{}}).get("unlocked_shops") or []
            if _shops:
                state["branch_done"] = True
                _pick = _BRANCHES.get(str(_shops[0]))
                if _pick is not None:
                    global _ROUTE
                    _ROUTE = _pick
        _observe_opp_sells(obs, step, state)

        idx = min(max(0, step), len(_ROUTE) - 1)
        action = copy.deepcopy(_ROUTE[idx])
        action = _align_hands(action, obs)
        action = _weed_repair(obs, action, step, state)
        action = _align_hands(action, obs)
        if _ADAPT_SELL:
            action = _adaptive_repay(action, step, state)
        action = _safe_market(obs, action)
        action = _mirror_tiebreak(obs, action, step, state)
        action = _premium_lead(obs, action, step, state)
        action = _floor_seller(obs, action, step, state)
        action = _safe_market(obs, action)
        action = _adaptive_sell(obs, action, step, state)
        action = _endgame_pull(obs, action, step, state)
        action = _floor_guard(obs, action, step, state)
        action = _sell_first(action, step)
        if _IMPACT_SLOTS:
            action = _impact_slots(obs, action)
        if step >= _LAST_STEP:
            action = _terminal_market(obs, action)
        action = _align_hands(action, obs)
        if _ADAPT_SELL:
            mine = {{}}
            for o in action.get("market") or []:
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    try:
                        mine[o[1]] = mine.get(o[1], 0) + max(0, int(o[2]))
                    except (TypeError, ValueError):
                        pass
            state["my_sells"] = mine
        return action
    except Exception:                                          # noqa: BLE001
        # A raised agent forfeits the episode. A passing one merely loses it.
        me = 0
        try:
            me = int(_get(obs, "player", 0) or 0)
            farms = _get(obs, "farms", []) or []
            hands = _get(farms[me], "hands", []) or []
        except Exception:                                      # noqa: BLE001
            hands = []
        return {{"farmer": ["PASS"], "hands": [["PASS"] for _ in hands],
                "market": []}}
'''
