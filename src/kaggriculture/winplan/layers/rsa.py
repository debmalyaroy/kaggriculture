

# ---------------------------------------------------------------------------
# RSA: route sale advance (win-plan Stage 1, 2026-09-24). The clone field's
# decisive late-game edge is selling a premium product a few turns before the
# inherited route does (v15stack V12: 3 turns). RSA reads OUR route tape
# RSA_LOOK turns ahead and executes those premium SELLs now, so we sell before
# a clone that advances fewer turns. Preserves the first next-turn SELL that
# funds a buy, never touches BUY_PRODUCT turns, never sells what a worker is
# picking up this turn, and respects the 10-order cap. Knobs are baked by the
# win-plan builder.
_RSA_PARENT = agent
_RSA_LOOK = 3
_RSA_FROM = 144
_RSA_TO = 718
_RSA_ITEMS = ("STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO")
# Glut guard (2026-09-24): advance an item only while its quote is at least
# _RSA_MIN_FRAC x base. 0.0 keeps the original rule (quote >= 2). Losses to the
# demand-preserving family came from pulling strawberry forward into a glut.
_RSA_MIN_FRAC = 0.0
_RSA_BASE = {"STRAWBERRY": 120, "WOOL": 200, "EGG": 50, "MILK": 160, "MELON": 250, "CARROT": 35, "TOMATO": 60}


def _rsa_future_market(player, step):
    native = _IMPL.chassis.players[player]
    route = 2 if step >= 648 else native["route"]
    tape = _IMPL.chassis.routes.get(route) or []
    return (tape[step].get("market", []) or []) if step < len(tape) else []


def _rsa_advance(observation, action):
    step = int(observation.get("step", 0))
    if step % 24 == 23 or not _RSA_FROM <= step < _RSA_TO:
        return action
    player = int(observation["player"])
    plan, first = [], None
    for offset in range(1, _RSA_LOOK + 1):
        fs = step + offset
        if fs > 718:
            break
        for order in _rsa_future_market(player, fs):
            if not order or len(order) < 3:
                continue
            if first is None:
                first = order
            if order[0] == "SELL" and order[1] in _RSA_ITEMS and int(order[2]) > 0:
                plan.append((order[1], int(order[2])))
    protected = first[1] if first and first[0] == "SELL" else None
    plan = [(i, q) for i, q in plan if i != protected]
    if not plan:
        return action
    market = [list(o) for o in (action.get("market") or [])]
    if any(len(o) > 1 and o[0] == "BUY_PRODUCT" for o in market):
        return action
    stock = projected_shed(action, FarmView(observation))
    selling = {}
    for o in market:
        if len(o) >= 3 and o[0] == "SELL":
            selling[o[1]] = selling.get(o[1], 0) + max(0, int(o[2]))
    picked = {c[1] for c in [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
              if len(c) > 1 and c[0] == "PICKUP"}
    prices = observation["market"]["prices"]
    extra = []
    for item in sorted({i for i, _ in plan}, key=lambda n: -int(prices.get(n, 0))):
        if item in picked or int(prices.get(item, 0)) < max(2, _RSA_MIN_FRAC * _RSA_BASE.get(item, 0)):
            continue
        avail = int(stock.get(item, 0)) - selling.get(item, 0)
        want = min(avail, sum(q for i, q in plan if i == item))
        if want < 1:
            continue
        existing = next((o for o in market if len(o) >= 3 and o[0] == "SELL" and o[1] == item), None)
        if existing is not None:
            existing[2] = int(existing[2]) + want
        elif len(market) + len(extra) < 10:
            extra.append(["SELL", item, want])
    if not extra and market == [list(o) for o in (action.get("market") or [])]:
        return action
    return dict(action, market=extra + market)


def agent(observation, configuration=None):
    action = _RSA_PARENT(observation, configuration)
    try:
        return _rsa_advance(observation, action)
    except Exception:
        return action
