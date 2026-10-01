"""Build a pool-adaptive route agent: one base route, 720-turn sell-move pool.

The experiment asked for on 2026-08-08: per-turn adaptive play whose search
space is every top-100 player's move at the same turn. Every previous form of
this collapsed by breaking farm-state coherence (consensus route $12,085;
channel splice $12,900), so this build removes that failure by construction:

* The FIELD channel and every BUY/HIRE order come from the base route,
  verbatim. The farm the actions assume is always the farm we actually have.
* Only the SELL channel is chosen per turn. At each of the 720 turns the
  candidate set is {the base route's own sells} + {each pool route's sells at
  this turn}. Each candidate is clamped to the projected shed (feasibility)
  and to the pool's 75th-percentile cumulative-sold envelope (no stampede),
  then valued against the LIVE market curve -- which embeds everything the
  opponent has sold so far, so the choice is opponent-adaptive by
  construction. Highest marginal value wins; the base route wins ties and is
  exempt from the envelope, so the worst case degrades to the base route.

Usage:
    python -m kaggriculture.agentbuild.pool_agent --base 90525850_s0 --out .local/cand/v20_pool.py
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

MAX_CANDIDATES = 24     # distinct sell-sets kept per turn, best-ranked teams first
ENV_PCTL = 0.75         # envelope: this percentile of pool cumulative-sold
SELLABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
            "WHEAT", "FERTILIZER")


# WHEAT is the herd's feed and FERTILIZER is applied to crops -- both sit in
# the shed as *inputs* the base route's field plan depends on. Borrowing
# another route's sale of them liquidates our own working capital (measured:
# $33k vs $171k on the first smoke). The base route's own sells still move
# their surplus; the pool may not touch them.
BORROWABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT")


def _sells(turn_action):
    out = []
    for o in (turn_action.get("market") or []):
        if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                and o[1] in BORROWABLE):
            try:
                q = int(o[2])
            except (TypeError, ValueError):
                continue
            if q > 0:
                out.append((o[1], q))
    return out


def build_pool(exclude_id):
    idx = R.load_index()
    fit = [r for r in idx["routes"].values()
           if r.get("window") == "fit" and r["id"] != exclude_id]
    fit.sort(key=lambda r: (r.get("rank") or 999, -(r.get("bank") or 0)))
    print(f"pool: {len(fit)} fit-window routes from "
          f"{len({r.get('team') for r in fit})} teams")

    pool = {}                 # turn -> list of sell-sets (list of [item, qty])
    seen = {}                 # turn -> set of dedupe keys
    curves = []               # per route: {item: [cumulative by turn]}
    for rec in fit:
        try:
            acts = R.load_route(rec["id"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {rec['id']}: {exc}")
            continue
        cum = {item: 0 for item in SELLABLE}
        curve = {item: [0] * 720 for item in SELLABLE}
        for t in range(min(720, len(acts))):
            sells = _sells(acts[t])
            for item, q in sells:
                cum[item] += q
            for item in SELLABLE:
                curve[item][t] = cum[item]
            if sells:
                key = tuple(sorted(sells))
                bucket = seen.setdefault(t, set())
                if key not in bucket and len(pool.get(t, ())) < MAX_CANDIDATES:
                    bucket.add(key)
                    pool.setdefault(t, []).append([[i, q] for i, q in sells])
        curves.append(curve)

    env = {}
    n = len(curves)
    k = min(n - 1, int(ENV_PCTL * n))
    for item in SELLABLE:
        env[item] = [sorted(c[item][t] for c in curves)[k] for t in range(720)]
    return pool, env


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="90525850_s0")
    ap.add_argument("--out", default=os.path.join(".local", "cand", "v20_pool.py"))
    args = ap.parse_args()

    idx = R.load_index()
    rec = idx["routes"].get(args.base)
    if rec is None:
        raise SystemExit(f"no route {args.base!r}")
    actions = R.load_route(args.base)
    pool, env = build_pool(args.base)
    n_cand = sum(len(v) for v in pool.values())
    print(f"pool candidates: {n_cand} across {len(pool)} turns; "
          f"envelope at {ENV_PCTL:.0%}")

    def pack(obj):
        return base64.b85encode(zlib.compress(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")

    src = TEMPLATE.format(
        label=os.path.basename(args.out),
        route_id=rec["id"], team=rec.get("team", "?"),
        built=dt.date.today().isoformat(),
        n_pool=n_cand,
        payload=pack(actions),
        pool=pack({str(t): v for t, v in pool.items()}),
        env=pack(env))
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out):,} bytes)")


TEMPLATE = '''"""Kaggriculture pool-adaptive route agent -- {label}

Base route {route_id} ({team}); sell channel chosen per turn from a pool of
{n_pool} recorded top-100 sell-moves, valued against the live market.
Built {built} by src/kaggriculture/agentbuild/pool_agent.py.
"""
import base64
import copy
import json
import math
import zlib

_ROUTE = json.loads(zlib.decompress(base64.b85decode("{payload}")).decode("utf-8"))
_POOL = json.loads(zlib.decompress(base64.b85decode("{pool}")).decode("utf-8"))
_ENV = json.loads(zlib.decompress(base64.b85decode("{env}")).decode("utf-8"))

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
_ENV_SLACK = 12            # units a pool candidate may run ahead of the envelope
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


def _sell_value(item, market_inv, qty):
    """Integrated marginal value of selling qty units at the live curve."""
    total = 0.0
    step = max(1, qty // 8)
    sold = 0
    while sold < qty:
        chunk = min(step, qty - sold)
        total += chunk * _quote(item, market_inv + sold)
        sold += chunk
    return total


def _pool_market(obs, action, step, state):
    """Choose this turn's sells from the top-100 pool, valued live.

    The base route's own sells are candidate zero: they win ties and are
    exempt from the envelope, so the floor of this agent is the base route.
    BUY/HIRE orders are never taken from the pool.
    """
    own = list(action.get("market") or [])
    own_sells, own_rest = [], []
    for o in own:
        if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                and o[1] in _MARKET_PARAMS):
            own_sells.append([o[0], o[1], o[2]])
        else:
            own_rest.append(o)
    shed = _projected_shed(obs, action)
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    cum = state.setdefault("cum", {{}})
    env_t = min(step, 719)

    def clamp(cand, enveloped):
        sells, value = [], 0.0
        rem = dict(shed)
        for o in cand:
            item = o[1] if len(o) >= 3 else None
            if item not in _MARKET_PARAMS:
                continue
            try:
                qty = max(0, int(o[2]))
            except (TypeError, ValueError):
                continue
            have = max(0, int(rem.get(item, 0) or 0))
            qty = min(qty, have)
            if enveloped:
                allowance = (int(_ENV.get(item, [0] * 720)[env_t]) + _ENV_SLACK
                             - int(cum.get(item, 0)))
                qty = min(qty, max(0, allowance))
            if qty <= 0:
                continue
            mkt = int(_get(inventory, item, 10000) or 0)
            value += _sell_value(item, mkt, qty)
            sells.append(["SELL", item, qty])
            rem[item] = have - qty
        return sells, value

    best_sells, best_value = clamp(own_sells, enveloped=False)
    for cand in _POOL.get(str(step), ()):
        sells, value = clamp([["SELL", i, q] for i, q in cand], enveloped=True)
        if value > best_value + 1e-6:
            best_sells, best_value = sells, value

    for o in best_sells:
        cum[o[1]] = int(cum.get(o[1], 0)) + int(o[2])
    action["market"] = (best_sells + own_rest)[:_MAX_ORDERS]
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
        action = _pool_market(obs, action, step, state)
        action = _safe_market(obs, action)
        action = _impact_slots(obs, action)
        if step >= _LAST_STEP:
            action = _terminal_market(obs, action)
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
