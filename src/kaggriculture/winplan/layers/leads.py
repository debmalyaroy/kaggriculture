

# ---------------------------------------------------------------------------
# LEADS: sell into strength ahead of the tape (win-plan v61.1, 2026-09-24).
# Ported from the public cha22 agent (abhinav0370/cha22-agent, Apache-2.0):
# the DAWN (DP) / MIDDAY (MP) chunked lead-sells, the model-based MILK/
# STRAWBERRY/WOOL lead seller (MPX) and the quote history they read (FX
# bookkeeping only; its flow trigger stays off as in cha22). Ablation vs v61
# (24 worlds): cha22 17-7; all four lead layers off 1-23; MPX off 9-15; the
# evening window (EV) HURT (19-5 without it) so it is not ported. A final
# pass re-runs the chassis's exact best-response ordering (_cxd_reorder) so
# the pulled-forward sells get optimal slots (cha22 runs CXD after its leads;
# without that pass it drops to 12-12).
_LD_PARENT = agent
_LD_ITEMS = ("CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
_LD_QUOTE_WIN = 12
_LD_DP_HOURS = (0, 1, 2)
_LD_MP_HOURS = (10, 11, 12, 13)
_LD_H = 8
_LD_MPX_ITEMS = ("MILK", "STRAWBERRY", "WOOL")
_LD_MPX_HOURS = tuple(range(12, 23))
_LD_CXD = True
_LD_DP_ON = True
_LD_MP_ON = True
_LD_MPX_ON = True
_LD_STATE = {}
_LD_REPORT = dict(dp_units=0, mp_units=0, mpx_units=0, cxd_turns=0, errors=0)


def _ld_window(observation, action, hours, key):
    """cha22 DP/MP: pull ~3/4 of the next _LD_H turns' planned sells of an item
    into now while its quote is at/above the trailing _LD_QUOTE_WIN average."""
    step = int(observation["step"])
    if step < 96 or step >= 700 or (step % 24) not in hours:
        return action
    market = [list(o) for o in (action.get("market") or [])]
    if len(market) >= MAX_ORDERS:
        return action
    already = {o[1] for o in market if len(o) > 1 and o[0] in ("SELL", "BUY_PRODUCT")}
    prices = observation["market"]["prices"]
    stock = projected_shed(action, FarmView(observation))
    native = _IMPL.chassis.players.get(int(observation["player"]))
    if not native or native.get("route") not in _IMPL.chassis.routes:
        return action
    tape = _IMPL.chassis.routes[native["route"]]
    hist = _LD_STATE.get(int(observation["player"]), {}).get("quotes", {})
    recent = [t for t in hist if step - _LD_QUOTE_WIN <= t < step]
    added = False
    for item in _LD_ITEMS:
        if item in already or len(market) >= MAX_ORDERS:
            continue
        q = int(prices.get(item, 0))
        if q <= 3:
            continue
        vals = [hist[t].get(item, 0) for t in recent if hist[t].get(item, 0) > 0]
        if vals and q < sum(vals) / len(vals):
            continue
        planned = 0
        for t in range(step + 1, min(len(tape), step + _LD_H + 1)):
            for o in (tape[t] or {}).get("market") or []:
                if len(o) >= 3 and o[0] == "SELL" and o[1] == item:
                    planned += max(0, int(o[2]))
        if planned <= 0:
            continue
        qty = min(int(stock.get(item, 0)), max(1, (3 * planned + 3) // 4))
        if qty <= 0:
            continue
        market.insert(0, ["SELL", item, qty])
        _LD_REPORT[key] += qty
        added = True
    return dict(action, market=market[:MAX_ORDERS]) if added else action


def _ld_draw_units(step):
    return (1 if step % 4 == 0 else 0) + (1 if step % 24 == 0 else 0)


def _ld_mpx(observation, action):
    """cha22 MPX: rival cadence from public inventory deltas; if the next quote
    is predicted to fall, sell half a planned lot now."""
    step = int(observation["step"])
    player = int(observation["player"])
    mk = observation["market"]
    inv_all = mk.get("inventory") or {}
    prices = mk.get("prices") or {}
    st = _LD_STATE.setdefault(player, {"quotes": {}})
    hist = st.setdefault("mpx", {})
    prev = hist.get("prev")
    inv_now = {i: int(inv_all.get(i, 0)) for i in _LD_MPX_ITEMS}
    if prev and prev["step"] == step - 1:
        for i in _LD_MPX_ITEMS:
            d = inv_now[i] - prev["inv"][i] + _ld_draw_units(step - 1) - prev["own"].get(i, 0)
            hist.setdefault(i, []).append(max(0, d))
            if len(hist[i]) > 12:
                del hist[i][:6]
    hist["prev"] = {"step": step, "inv": inv_now, "own": {}}
    orders = [list(o) for o in (action.get("market") or [])]
    for o in orders:
        if len(o) >= 3 and o[0] == "SELL" and o[1] in _LD_MPX_ITEMS:
            hist["prev"]["own"][o[1]] = hist["prev"]["own"].get(o[1], 0) + int(o[2])
    if step < 144 or step >= 696 or (step % 24) not in _LD_MPX_HOURS:
        return action
    already = {o[1] for o in orders if len(o) > 1 and o[0] == "SELL"}
    native = _IMPL.chassis.players.get(player)
    if not native or native.get("route") not in _IMPL.chassis.routes:
        return action
    stock = projected_shed(action, FarmView(observation))
    added = False
    for item in _LD_MPX_ITEMS:
        if item in already or len(orders) >= 10:
            continue
        avail = int(stock.get(item, 0))
        if avail <= 0 or int(prices.get(item, 0)) <= 1:
            continue
        rival = hist.get(item) or []
        rival_avg = (sum(rival[-4:]) / len(rival[-4:])) if rival else 0.0
        planned = 6
        inv = int(inv_all.get(item, 0))
        p_cur = float(_r37_market_price(item, inv))
        p_next = float(_r37_market_price(item, max(0, int(inv + rival_avg + planned - _ld_draw_units(step)))))
        if p_next < p_cur - 0.5:
            take = min(avail, max(1, planned // 2))
            orders.insert(0, ["SELL", item, take])
            _LD_REPORT["mpx_units"] += take
            hist["prev"]["own"][item] = hist["prev"]["own"].get(item, 0) + take
            added = True
    return dict(action, market=orders[:10]) if added else action


def agent(observation, configuration=None):
    step = int(observation.get("step", 0))
    player = int(observation.get("player", 0))
    if step == 0:
        _LD_STATE.pop(player, None)
        for k in _LD_REPORT:
            _LD_REPORT[k] = 0
    st = _LD_STATE.setdefault(player, {"quotes": {}})
    try:
        prices = observation["market"]["prices"]
        st["quotes"][step] = {i: int(prices.get(i, 0)) for i in _LD_ITEMS}
        if len(st["quotes"]) > 96:
            for k in sorted(st["quotes"])[:48]:
                st["quotes"].pop(k, None)
    except Exception:
        _LD_REPORT["errors"] += 1
    action = _LD_PARENT(observation, configuration)
    try:
        before = action
        if _LD_DP_ON:
            action = _ld_window(observation, action, _LD_DP_HOURS, "dp_units")
        if _LD_MP_ON:
            action = _ld_window(observation, action, _LD_MP_HOURS, "mp_units")
        if _LD_MPX_ON:
            action = _ld_mpx(observation, action)
        if _LD_CXD and action is not before:
            re = _cxd_reorder(observation, action)
            if re is not action:
                _LD_REPORT["cxd_turns"] += 1
            action = re
    except Exception:
        _LD_REPORT["errors"] += 1
    return action
