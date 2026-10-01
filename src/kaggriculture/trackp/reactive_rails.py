"""Config-driven reactive override layer ("rails/gates") for the trackp policy.

The learned policy proposes an action each turn; enabled rails then adjust it
(weed-repair, water-guard, sell-premium, endgame-liquidate, ... -- add your own),
and a MANDATORY ``sanitize`` runs LAST so the emitted action is always legal:
hands align positionally with farm.hands, at most 10 market orders, well-formed
ops. That legality guarantee is what makes overrides desync-proof on the engine.

Config shape mirrors the bandit guardrails (a list, order = apply order):
    [{"name": "weed_repair", "on": true},
     {"name": "sell_premium", "on": true, "min_price": 300},
     {"name": "endgame_liquidate", "on": true, "from_day": 29}]

Register a new gate with @rail("my_gate"); it receives (action, obs, seat, p)
and returns a (possibly) modified action dict. It need not be legality-perfect
-- sanitize fixes shape -- but it should only emit ops the state allows.
"""
from __future__ import annotations
import kaggriculture.data.trackp_corpus as TC

PRODUCTS = TC.PRODUCTS
MAX_MARKET = TC.MAX_MARKET

RULES = {}


def rail(name):
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


# --- obs helpers -------------------------------------------------------------
def _me(obs, seat):
    farms = obs.get("farms") or []
    return farms[seat] if seat < len(farms) else {}


def _tile_at(me, x, y):
    tiles = me.get("tiles") or []
    if 0 <= y < len(tiles):
        row = tiles[y]
        if isinstance(row, list) and 0 <= x < len(row) and isinstance(row[x], dict):
            return row[x]
    return None


def _hands_action(action, n):
    ha = list(action.get("hands") or [])[:n]
    while len(ha) < n:
        ha.append(["PASS"])
    return ha


# --- example rails (extend freely) -------------------------------------------
@rail("weed_repair")
def weed_repair(action, obs, seat, p):
    """Any mover standing ON a WEED tile -> DIG it (clears the weed)."""
    me = _me(obs, seat)
    fp = me.get("farmer") or None
    if fp:
        t = _tile_at(me, int(fp[0]), int(fp[1]))
        if t and t.get("kind") == "WEED":
            action["farmer"] = ["DIG"]
    hands = me.get("hands") or []
    ha = _hands_action(action, len(hands))
    for i, h in enumerate(hands):
        t = _tile_at(me, int(h[0]), int(h[1]))
        if t and t.get("kind") == "WEED":
            ha[i] = ["DIG"]
    action["hands"] = ha
    return action


@rail("water_guard")
def water_guard(action, obs, seat, p):
    """Mover on a PLANT tile unwatered today -> WATER (keeps it from weeding)."""
    me = _me(obs, seat)
    hands = me.get("hands") or []
    ha = _hands_action(action, len(hands))
    for movers, setter in (([me.get("farmer")], "farmer"), (hands, "hands")):
        for i, pos in enumerate(movers):
            if not pos:
                continue
            t = _tile_at(me, int(pos[0]), int(pos[1]))
            if t and t.get("kind") == "PLANT" and not t.get("watered_today"):
                op = ["WATER"]
                if setter == "farmer":
                    if (action.get("farmer") or ["PASS"])[0] == "PASS":
                        action["farmer"] = op
                elif (ha[i] or ["PASS"])[0] == "PASS":
                    ha[i] = op
    action["hands"] = ha
    return action


@rail("sell_premium")
def sell_premium(action, obs, seat, p):
    """Add SELL orders for held products whose price >= min_price (respects cap)."""
    thresh = float(p.get("min_price", 0) or 0)
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = (obs.get("private") or {}).get("shed") or {}
    mkt = list(action.get("market") or [])
    have = {o[1] for o in mkt if isinstance(o, list) and len(o) > 1 and o[0] == "SELL"}
    for prod in PRODUCTS:
        if len(mkt) >= MAX_MARKET:
            break
        held = int(shed.get(prod, 0) or 0)
        if held > 0 and prod not in have and float(prices.get(prod, 0) or 0) >= thresh:
            mkt.append(["SELL", prod, held])
    action["market"] = mkt
    return action


@rail("endgame_liquidate")
def endgame_liquidate(action, obs, seat, p):
    """From ``from_day`` onward, SELL the whole shed by price desc (cap 10)."""
    if int(obs.get("day", 0) or 0) < int(p.get("from_day", 29)):
        return action
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = (obs.get("private") or {}).get("shed") or {}
    held = [(prod, int(shed.get(prod, 0) or 0)) for prod in PRODUCTS
            if int(shed.get(prod, 0) or 0) > 0]
    held.sort(key=lambda kv: -float(prices.get(kv[0], 0) or 0))
    action["market"] = [["SELL", prod, q] for prod, q in held][:MAX_MARKET]
    return action


@rail("harvest_ready")
def harvest_ready(action, obs, seat, p):
    """Mover on a tile with yield_units>0 -> HARVEST (don't leave value on the vine)."""
    me = _me(obs, seat)
    hands = me.get("hands") or []
    ha = _hands_action(action, len(hands))
    fp = me.get("farmer")
    if fp:
        t = _tile_at(me, int(fp[0]), int(fp[1]))
        if t and int(t.get("yield_units", 0) or 0) > 0 and \
                (action.get("farmer") or ["PASS"])[0] == "PASS":
            action["farmer"] = ["HARVEST"]
    for i, h in enumerate(hands):
        t = _tile_at(me, int(h[0]), int(h[1]))
        if t and int(t.get("yield_units", 0) or 0) > 0 and (ha[i] or ["PASS"])[0] == "PASS":
            ha[i] = ["HARVEST"]
    action["hands"] = ha
    return action


@rail("feed_care")
def feed_care(action, obs, seat, p):
    """Mover on a COOP/PASTURE animal not fed/cared today -> FEED (then CARE)."""
    me = _me(obs, seat)
    hands = me.get("hands") or []
    ha = _hands_action(action, len(hands))
    fp = me.get("farmer")
    if fp:
        t = _tile_at(me, int(fp[0]), int(fp[1]))
        if t and t.get("kind") in ("COOP", "PASTURE") and \
                (action.get("farmer") or ["PASS"])[0] == "PASS":
            action["farmer"] = ["FEED"] if not t.get("fed_today") else \
                (["CARE"] if not t.get("cared_today") else action["farmer"])
    for i, h in enumerate(hands):
        t = _tile_at(me, int(h[0]), int(h[1]))
        if t and t.get("kind") in ("COOP", "PASTURE") and (ha[i] or ["PASS"])[0] == "PASS":
            if not t.get("fed_today"):
                ha[i] = ["FEED"]
            elif not t.get("cared_today"):
                ha[i] = ["CARE"]
    action["hands"] = ha
    return action


@rail("collect_fertilizer")
def collect_fertilizer(action, obs, seat, p):
    """Mover on a tile with fertilizer_available -> COLLECT_FERTILIZER (top agents
    lean heavily on fertilizer)."""
    me = _me(obs, seat)
    hands = me.get("hands") or []
    ha = _hands_action(action, len(hands))
    for i, h in enumerate(hands):
        t = _tile_at(me, int(h[0]), int(h[1]))
        if t and t.get("fertilizer_available") and (ha[i] or ["PASS"])[0] == "PASS":
            ha[i] = ["COLLECT_FERTILIZER"]
    action["hands"] = ha
    return action


@rail("scarcity_sell")
def scarcity_sell(action, obs, seat, p):
    """Engine >=1.32.7 hinge: sell held products whose market price >= mult x the
    typical base (drained markets quote 10-50x). ``mult`` and per-product ``base``
    are knobs; defaults use a coarse base table."""
    mult = float(p.get("mult", 3.0) or 3.0)
    base = p.get("base") or {"CARROT": 35, "EGG": 50, "FERTILIZER": 100, "MELON": 250,
                             "MILK": 160, "STRAWBERRY": 120, "TOMATO": 60, "WHEAT": 25,
                             "WOOL": 200}
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = (obs.get("private") or {}).get("shed") or {}
    mkt = list(action.get("market") or [])
    have = {o[1] for o in mkt if isinstance(o, list) and len(o) > 1 and o[0] == "SELL"}
    for prod in PRODUCTS:
        if len(mkt) >= MAX_MARKET:
            break
        held = int(shed.get(prod, 0) or 0)
        if held > 0 and prod not in have and \
                float(prices.get(prod, 0) or 0) >= mult * float(base.get(prod, 1e9)):
            mkt.append(["SELL", prod, held])
    action["market"] = mkt
    return action


@rail("budget_guard")
def budget_guard(action, obs, seat, p):
    """Drop BUY_* market orders once projected spend exceeds available money x
    ``spend_frac`` (prevents over-buying the policy can't afford)."""
    me = _me(obs, seat)
    money = float(me.get("money", 0) or 0) * float(p.get("spend_frac", 1.0) or 1.0)
    prices = (obs.get("market") or {}).get("prices") or {}
    out = []
    spent = 0.0
    for o in (action.get("market") or []):
        if isinstance(o, list) and len(o) >= 3 and str(o[0]).startswith("BUY"):
            cost = float(prices.get(o[1], 0) or 0) * float(o[2] or 0)
            if spent + cost > money:
                continue                  # skip unaffordable buy
            spent += cost
        out.append(o)
    action["market"] = out
    return action


# --- the desync guard (always runs last) -------------------------------------
def sanitize(action, obs, seat):
    """Guarantee a legal, well-formed action: farmer op present, hands aligned
    positionally with farm.hands (House Rule 5), <=10 market orders. This is what
    makes any override desync-proof on the engine."""
    me = _me(obs, seat)
    n = len(me.get("hands") or [])
    farmer = action.get("farmer")
    farmer = farmer if isinstance(farmer, list) and farmer else ["PASS"]
    hands = [h if (isinstance(h, list) and h) else ["PASS"]
             for h in (action.get("hands") or [])][:n]
    while len(hands) < n:
        hands.append(["PASS"])
    market = [o for o in (action.get("market") or [])
              if isinstance(o, list) and len(o) >= 1][:MAX_MARKET]
    return {"farmer": farmer, "hands": hands, "market": market}


def apply_rails(action, obs, seat, config):
    """Apply enabled rails in order, then sanitize (mandatory). ``config`` is a
    list of {name, on, ...params}. Unknown rule names are skipped safely."""
    for rule in (config or []):
        if not isinstance(rule, dict) or not rule.get("on", True):
            continue
        fn = RULES.get(rule.get("name"))
        if fn is not None:
            try:
                action = fn(action, obs, seat, rule)
            except Exception:
                pass                      # a bad rail never breaks the turn
    return sanitize(action, obs, seat)
