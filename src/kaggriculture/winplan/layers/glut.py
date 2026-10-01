

# ---------------------------------------------------------------------------
# GLUT HOLD (win-plan Stage 1, 2026-09-24). Against the demand-preserving
# family the late game is lost by dumping strawberry into a glutted market
# (quotes $1-9 on a $120 base) while the rival sells small lots and waits for
# town demand to lift the price. Before _GH_UNTIL, a SELL of an item quoting
# under _GH_FRAC x base is capped to _GH_CAP units this turn; the rest stays in
# the shed for later turns and the endgame liquidation. Nothing else changes.
_GH_PARENT = agent
_GH_FRAC = 0.15
_GH_CAP = 6
_GH_UNTIL = 648
_GH_BASE = {"STRAWBERRY": 120, "WOOL": 200, "EGG": 50, "MILK": 160, "MELON": 250, "CARROT": 35, "TOMATO": 60}


def _gh_hold(observation, action):
    step = int(observation.get("step", 0))
    if step >= _GH_UNTIL:
        return action
    prices = observation["market"]["prices"]
    market, changed, sold = [], False, {}
    for o in action.get("market") or []:
        o = list(o) if isinstance(o, list) else o
        if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in _GH_BASE \
                and int(prices.get(o[1], 0)) < _GH_FRAC * _GH_BASE[o[1]]:
            item = o[1]
            room = max(0, _GH_CAP - sold.get(item, 0))
            q = max(0, int(o[2]))
            if q > room:
                # an emptied order keeps its slot: market orders settle by index
                o = ["SELL", item, room] if room else []
                changed = True
            sold[item] = sold.get(item, 0) + min(q, room)
        market.append(o)
    return dict(action, market=market) if changed else action


def agent(observation, configuration=None):
    action = _GH_PARENT(observation, configuration)
    try:
        return _gh_hold(observation, action)
    except Exception:
        return action
