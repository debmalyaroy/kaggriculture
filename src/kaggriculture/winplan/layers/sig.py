

# ---------------------------------------------------------------------------
# SIG: opening signature shift (win-plan identity hiding H1, 2026-09-24). The
# field's mirror gate compares the two banks after turn 0 (EXP288: equal cash
# => "a copy of this agent" => 24-turn race horizon from day one). Our base
# opens BUY_PRODUCT WHEAT 8 / SELL WHEAT 3 like the whole herd-safe family.
# Growing both legs of the step-0 round trip by _SIG_K keeps the net wheat the
# same while the turn-0 cash no longer equals any public family's.
_SIG_PARENT = agent
_SIG_K = 1


def _sig_shift(observation, action):
    if int(observation.get("step", -1)) != 0 or not _SIG_K:
        return action
    market = [list(o) if isinstance(o, list) else o for o in (action.get("market") or [])]
    buy = next((o for o in market if isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT"), None)
    sell = next((o for o in market if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] == "WHEAT"), None)
    if buy is None or sell is None:
        return action
    buy[2] = int(buy[2]) + _SIG_K
    sell[2] = int(sell[2]) + _SIG_K
    return dict(action, market=market)


def agent(observation, configuration=None):
    action = _SIG_PARENT(observation, configuration)
    try:
        return _sig_shift(observation, action)
    except Exception:
        return action
