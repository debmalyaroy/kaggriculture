"""Build a counter-adaptive route agent: predict the opponent's next sells.

The 2026-08-08 request: on encountering an opponent, guess their next action
and counter it. The only channel two players share is the market, so the only
counterable action is a SELL -- and the counter to a predicted SELL is to
quote first: sell our own already-scheduled stock of that product one turn
before their dump lands, then repay the schedule so season inventory is
unchanged (the mirror tie-break's proven borrow/repay mechanics, with a
retrieval trigger instead of an exact-mirror trigger).

The predictor is retrieval over the mined route library: the opponent's
cumulative sales per product are estimated live from the shared market
inventory trail (their sells push inventory up; ours are known and
subtracted), and matched against every library route's cumulative curve.
When one route fits decisively better than the runner-up, the opponent is
assumed to be running that program and its next turns are read as the
forecast. Half the ladder ships one of 11 route families, so identification
is realistic; against a genuinely novel opponent the match never becomes
decisive and the layer stays silent, degrading to the base route.

Usage:
    python -m kaggriculture.agentbuild.counter_agent --base 90525850_s0 --out .local/cand/v21_counter.py
"""
from kaggriculture.paths import ROOT
import argparse
import base64
import datetime as dt
import json
import os
import sys
import zlib

import kaggriculture.data.routes as R  # noqa: E402

LIB_ROUTES = 90         # best-ranked fit-window routes kept in the library
TRACKED = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
           "WHEAT", "FERTILIZER")


def build_library(exclude_id):
    idx = R.load_index()
    fit = [r for r in idx["routes"].values()
           if r.get("window") == "fit" and r["id"] != exclude_id]
    fit.sort(key=lambda r: (r.get("rank") or 999, -(r.get("bank") or 0)))
    fit = fit[:LIB_ROUTES]
    lib, reps, teams = [], [], []
    for rec in fit:
        try:
            acts = R.load_route(rec["id"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {rec['id']}: {exc}")
            continue
        sched = {}
        cum = {item: 0 for item in TRACKED}
        curve = []
        for t in range(min(720, len(acts))):
            sells = []
            for o in (acts[t].get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in TRACKED):
                    try:
                        q = int(o[2])
                    except (TypeError, ValueError):
                        continue
                    if q > 0:
                        sells.append([o[1], q])
                        cum[o[1]] += q
            if sells:
                sched[str(t)] = sells
            if t % 24 == 0:
                curve.extend(cum[item] for item in TRACKED)
        # One representative per program: sibling recordings of the same
        # deterministic agent differ only by slip noise, and keeping them all
        # makes the matcher's best-vs-runner-up test veto its own answer --
        # the runner-up is just another recording of the same opponent.
        if any(sum(abs(a - b) for a, b in zip(curve, rep)) < 120
               for rep in reps):
            continue
        reps.append(curve)
        teams.append(str(rec.get("team") or "?"))
        lib.append(sched)
    print(f"library: {len(lib)} distinct programs "
          f"(from {len(fit)} routes, {len(set(teams))} teams)")
    return lib, teams


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="90525850_s0")
    ap.add_argument("--out", default=os.path.join(".local", "cand", "v21_counter.py"))
    args = ap.parse_args()

    idx = R.load_index()
    rec = idx["routes"].get(args.base)
    if rec is None:
        raise SystemExit(f"no route {args.base!r}")
    actions = R.load_route(args.base)
    lib, teams = build_library(args.base)

    def pack(obj):
        return base64.b85encode(zlib.compress(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")

    src = TEMPLATE.format(
        label=os.path.basename(args.out),
        route_id=rec["id"], team=rec.get("team", "?"),
        built=dt.date.today().isoformat(), n_lib=len(lib),
        payload=pack(actions), lib=pack(lib), libteam=pack(teams))
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out):,} bytes)")


TEMPLATE = '''"""Kaggriculture counter-adaptive route agent -- {label}

Base route {route_id} ({team}). Identifies the opponent's program by matching
their estimated sales against a library of {n_lib} top-100 routes, forecasts
their next sells, and quotes first -- borrowing only from our own schedule,
repaid in full. Built {built} by src/kaggriculture/agentbuild/counter_agent.py.
"""
import base64
import copy
import json
import math
import zlib

_ROUTE = json.loads(zlib.decompress(base64.b85decode("{payload}")).decode("utf-8"))
_LIB = json.loads(zlib.decompress(base64.b85decode("{lib}")).decode("utf-8"))
_LIBTEAM = json.loads(zlib.decompress(base64.b85decode("{libteam}")).decode("utf-8"))

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
# Engine 1.32.7 hinge rebalance (PR #1399). The flag default is flipped by
# scripts/engine_swap_1327.py when the LADDER's replays show 1.32.7, so
# agents built after the flip project scarcity prices correctly.
_ENGINE_1327 = True
if _ENGINE_1327:
    _MARKET_PARAMS["CARROT"] = (35, 10000, 450, "hinge", 1.0, "sqrt", 0.7)
    _MARKET_PARAMS["TOMATO"] = (60, 10000, 200, "hinge", 0.4, "sqrt", 0.6)
    _MARKET_PARAMS["EGG"] = (50, 10000, 332, "hinge", 0.4, "log", 0.2)
_TRACKED = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
            "WHEAT", "FERTILIZER")
# The counter only ever advances premium produce; feed and inputs stay put.
_COUNTERABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG")

# Town consumption, transcribed from kaggriculture.py so the opponent's sales
# can be reconstructed exactly from the shared inventory: every 4th step each
# unlocked shop consumes 1 of each product it stocks (2 if it stocks only
# one); every 12th step the town centre consumes 1/2/4 of everything but
# fertilizer, by day. Which shops are unlocked is public in obs["town"].
_SHOPS = {{
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}}
_CENTER_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL")

_MATCH_MIN_STEP = 96       # never act before day 4; everyone looks alike early
_MATCH_MIN_SOLD = 60       # opponent units seen before a match can be trusted
_MATCH_RATIO = 0.75        # best distance must be under this x the nearest
                           # OTHER team's program
_FORECAST = 3              # turns ahead read from the matched route
_BORROW_WINDOW = 8         # our own schedule may be advanced from this window
_COUNTER_CAP = 20          # units advanced per product per firing
_COOLDOWN = 4              # turns between firings per product

_SELLABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
             "WHEAT", "FERTILIZER")
_MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
_UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
             "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
             "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER")
_SHED_CAP = 100
_MAX_ORDERS = 10
_LAST_STEP = 718
_WEED_CATCHUP = 8
_STATE = {{0: {{}}, 1: {{}}}}

# Library cumulative-sales curves, built once at import.
_LIBCUM = []
for _sched in _LIB:
    _cum = {{item: [0] * 720 for item in _TRACKED}}
    _run = {{item: 0 for item in _TRACKED}}
    for _t in range(720):
        for _item, _q in _sched.get(str(_t), ()):
            _run[_item] += _q
        for _item in _TRACKED:
            _cum[_item][_t] = _run[_item]
    _LIBCUM.append(_cum)


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
    half = size // 2
    try:
        spot = (int(position[0]), int(position[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return spot in ((half - 1, half - 1), (half, half - 1),
                    (half - 1, half), (half, half))


def _align_hands(action, obs):
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
        if not (isinstance(op, list) and op) or op[0] != "DROP":
            continue
        if i >= len(positions) or not _shed_adjacent(positions[i], size):
            continue
        carried = inventories[i] if i < len(inventories) else {{}}
        for item, count in (carried or {{}}).items():
            n = min(int(count or 0), room)
            if n <= 0:
                continue
            shed[item] = int(shed.get(item, 0) or 0) + n
            room -= n
    return shed


def _safe_market(obs, action):
    remaining = _projected_shed(obs, action)
    out = []
    for raw in (action.get("market") or []):
        if not (isinstance(raw, list) and raw and raw[0] in _MARKET_OPS):
            continue
        order = list(raw)
        if order[0] == "SELL" and len(order) >= 3:
            item = order[1]
            try:
                want = max(0, int(order[2]))
            except (TypeError, ValueError):
                want = 0
            have = max(0, int(remaining.get(item, 0) or 0))
            qty = min(want, have)
            if qty <= 0:
                continue
            order[2] = qty
            remaining[item] = have - qty
        out.append(order)
    action["market"] = out[:_MAX_ORDERS]
    return action


def _sell_first(action):
    orders = action.get("market") or []
    sells = [o for o in orders if o and o[0] == "SELL"]
    rest = [o for o in orders if not (o and o[0] == "SELL")]
    action["market"] = (sells + rest)[:_MAX_ORDERS]
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


def _track_opponent(obs, step, state):
    """Estimate the opponent's cumulative sales from the shared inventory.

    Inventory rises when either player sells and falls when the town consumes.
    Our own sells last turn are known; the positive residual is theirs. The
    estimate is noisy on turns where consumption and sales overlap, which is
    why the matcher demands a decisive margin before it trusts anything.
    """
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    now = {{item: int(_get(inventory, item, 0) or 0) for item in _TRACKED}}
    prev = state.get("prev_inv")
    ours = state.get("our_last_sells") or {{}}
    opp = state.setdefault("opp_cum", {{item: 0 for item in _TRACKED}})
    if prev is not None:
        # Reconstruct what the town consumed on the step that produced this
        # delta, using the shop list as it stood then. Exact apart from sales
        # at the $1 floor, which never enter inventory at all (fact A4).
        consumed = {{item: 0 for item in _TRACKED}}
        prev_step = step - 1
        if prev_step % 4 == 0:
            for shop in state.get("prev_shops", ()):
                products = _SHOPS.get(shop, ())
                mult = 2 if len(products) == 1 else 1
                for item in products:
                    consumed[item] += mult
        # Town centre: once a day, exactly 1 of everything -- flat (verified
        # against the 1.32.7 interpreter). The old "% 12, 1/2/4 by decade"
        # was the 1.32.4 engine; it injected phantom consumption on 30 steps
        # a game and inflated every opponent-dump estimate downstream.
        if prev_step % 24 == 0:
            for item in _CENTER_PRODUCTS:
                consumed[item] += 1
        for item in _TRACKED:
            est = (now[item] - prev[item]) - int(ours.get(item, 0)) + consumed[item]
            if est > 0:
                opp[item] += est
    state["prev_inv"] = now
    town = _get(obs, "town", {{}}) or {{}}
    state["prev_shops"] = tuple(_get(town, "unlocked_shops", ()) or ())
    return opp


def _match_opponent(opp, step):
    """Which library route best explains what they have sold so far?"""
    total = sum(opp.values())
    if step < _MATCH_MIN_STEP or total < _MATCH_MIN_SOLD:
        return None
    t = min(step, 719)
    dists = []
    for i, cum in enumerate(_LIBCUM):
        d = 0
        for item in _TRACKED:
            d += abs(opp[item] - cum[item][t])
        dists.append((d, i))
    dists.sort()
    best, who = dists[0]
    # The runner-up that matters is the nearest program from a DIFFERENT
    # team: two recordings of the same opponent must not veto each other.
    rival = None
    for d, i in dists[1:]:
        if _LIBTEAM[i] != _LIBTEAM[who]:
            rival = d
            break
    if rival is None:
        return None
    # Decisive means clearly closer than any other team's program AND a
    # small error relative to the volume actually observed.
    if best > _MATCH_RATIO * rival or best > 0.35 * max(1, total):
        return None
    return who


def _counter(obs, action, step, state):
    """Sell before a forecast opponent dump, borrowed from our own schedule.

    Never creates inventory: every advanced unit is owed back and deducted
    from our own next scheduled SELL of that product (`due`), and the advance
    is capped by what our schedule actually plans to sell inside the borrow
    window. If the forecast is wrong the cost is bounded at selling our own
    goods a few turns early.
    """
    # ---- repay standing debt first, whatever the match says now.
    due = state.setdefault("due", {{}})
    if any(v > 0 for v in due.values()):
        market = []
        for order in (action.get("market") or []):
            if (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"
                    and due.get(order[1], 0) > 0):
                try:
                    qty = max(0, int(order[2]))
                except (TypeError, ValueError):
                    qty = 0
                take = min(due[order[1]], qty)
                due[order[1]] -= take
                qty -= take
                if qty <= 0:
                    continue
                order = [order[0], order[1], qty]
            market.append(order)
        action["market"] = market

    opp = _track_opponent(obs, step, state)
    who = _match_opponent(opp, step)
    state["matched"] = who
    if who is None or step + 1 >= _LAST_STEP:
        return action
    sched = _LIB[who]

    # Their forecast sells in the next few turns.
    threat = {{}}
    for dt_ in range(1, _FORECAST + 1):
        for item, q in sched.get(str(step + dt_), ()):
            if item in _COUNTERABLE:
                threat[item] = threat.get(item, 0) + q
    if not threat:
        return action

    shed = _projected_shed(obs, action)
    cooldown = state.setdefault("cool", {{}})
    existing = list(action.get("market") or [])
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    room = _MAX_ORDERS - len(existing)
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    new_orders = []
    ranked = sorted(threat.items(),
                    key=lambda kv: -kv[1] * _quote(kv[0], int(_get(inventory, kv[0], 10000) or 0)))
    for item, _q in ranked:
        if room <= 0:
            break
        if item in already:
            continue
        if step - int(cooldown.get(item, -10 ** 9)) < _COOLDOWN:
            continue
        # Only what our own schedule already plans to sell soon.
        planned = 0
        for dt_ in range(1, _BORROW_WINDOW + 1):
            idx = step + dt_
            if idx >= len(_ROUTE):
                break
            for o in (_ROUTE[idx].get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] == item):
                    try:
                        planned += max(0, int(o[2]))
                    except (TypeError, ValueError):
                        pass
        qty = min(planned - int(due.get(item, 0)), _COUNTER_CAP,
                  max(0, int(shed.get(item, 0) or 0)))
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        due[item] = int(due.get(item, 0)) + qty
        cooldown[item] = step
        room -= 1
    if new_orders:
        action["market"] = (new_orders + existing)[:_MAX_ORDERS]
        state["fired"] = int(state.get("fired", 0)) + len(new_orders)
    return action


def _threat_first(action, state, step):
    """Quote first into a forecast same-turn dump. Pure permutation.

    `_process_market` executes both players' i-th orders against the same
    pre-commit inventory, so when the matched program says the opponent sells
    product X on THIS turn, our own SELL X must sit at the lowest index we
    can give it. Nothing is created or resized -- this is the sell-first /
    impact-slots family of ordering edits, the one category with a measured
    positive record.
    """
    who = state.get("matched")
    if who is None:
        return action
    threatened = {{item for item, _q in _LIB[who].get(str(step), ())}}
    if not threatened:
        return action
    market = list(action.get("market") or [])
    hot = [o for o in market if isinstance(o, list) and len(o) >= 3
           and o[0] == "SELL" and o[1] in threatened]
    if not hot:
        return action
    rest = [o for o in market if o not in hot]
    state["reordered"] = int(state.get("reordered", 0)) + 1
    action["market"] = (hot + rest)[:_MAX_ORDERS]
    return action


def _impact_slots(obs, action):
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


def _weed_repair(obs, action, step, state):
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
        if op[0] not in ("PLANT", "BUILD_PASTURE", "BUILD_COOP"):
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


def _terminal_market(obs, action):
    shed = _projected_shed(obs, action)
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    rows = []
    for index, item in enumerate(_SELLABLE):
        qty = max(0, int(shed.get(item, 0) or 0))
        if qty <= 0:
            continue
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
        if step == 0 or step < int(state.get("last", -1)):
            state = _STATE[seat] = {{"last": step}}
        state["last"] = step
        idx = min(max(0, step), len(_ROUTE) - 1)
        action = copy.deepcopy(_ROUTE[idx])
        action = _align_hands(action, obs)
        action = _weed_repair(obs, action, step, state)
        action = _align_hands(action, obs)
        action = _counter(obs, action, step, state)
        action = _safe_market(obs, action)
        action = _sell_first(action)
        action = _impact_slots(obs, action)
        action = _threat_first(action, state, step)
        if step >= _LAST_STEP:
            action = _terminal_market(obs, action)
        # Remember what we sold this turn, for the opponent estimator.
        state["our_last_sells"] = {{
            o[1]: int(o[2]) for o in (action.get("market") or [])
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
        }}
        return _align_hands(action, obs)
    except Exception:                                          # noqa: BLE001
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


if __name__ == "__main__":
    main()
