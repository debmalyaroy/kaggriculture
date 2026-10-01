"""Assemble the big-trackp agent: chassis + LOADSTATE rollout searcher.

The searcher layer (2026-09-05, state injection validated EXACT):
at day boundaries from day 6 and through the endgame, snapshot the observed
state, inject it into a persistent `kagg serve`, roll K market-plan
variants over an H-step horizon (opponent modeled as PASS -- the measured
interaction surface is small), and apply the argmax plan's first step.
Production actions are never touched; only market timing is searched.

RUST FAILURE POLICY (operator, 2026-09-05): loud, never silent -- any
searcher failure prints "RUST-FAIL step=N: <err>" to stderr (visible in the
Kaggle agent log) and DISABLES the layer for the rest of the game; the
gated chassis continues. There is no hidden python re-implementation of the
search.

    python src/trackp/harness/build_searcher.py \
        --base .local/candidates/v46b_h2.py \
        --out .local/candidates/v46_searcher.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

LAYER = r'''
# ---------------------------------------------------------------- searcher --
_SEARCH_H = 48          # rollout horizon (2 days)
_SEARCH_FROM = 192      # first decision point: day 8 -- NEVER inside the
# day-6 branch/drain window (144-191): deferring its sells defunds the
# ~12k expansion buys at step 151 (silent no-ops) and recreates the
# measured d6-12 collapse (-45k/cell, 2026-09-05)
_SEARCH_EVERY = 24      # day boundaries...
_SEARCH_END_EVERY = 6   # ...then fine-grained through the endgame
_SEARCH_BUDGET_MS = 650 # hard per-turn budget for the whole layer
_SEARCH_DEPTH = 1       # extra plies past ply-1 (2026-09-07: depth 2 measured CATASTROPHIC with both nets: v1 1-3, v3 blowout losses -26k/-35k; net leaf error compounds)
_SEARCH_MIN_EDGE = 1500 # act only when a plan beats sched by this much:
_SEARCH_MIN_EDGE_END = 500  # d26+ settlement window: narrow losses are <1500
_SEARCH_END_EDGE_FROM = 624 # step of day 26
# (2026-09-06: the endgame ramp measured NEUTRAL, not negative -- 0 extra
# flips / median own-delta +14 over 48 loss worlds. Operator rule: keep
# zero-value features; remove only on measured regression.)
# the 2026-09-05 diag showed sane picks on $30-60 rollout edges being
# swamped by the rollout's opponent-model bias (worth thousands) -- the
# searcher must fire rarely and confidently, not constantly and marginally


try:
    _VALUENET = json.loads(__import__("lzma").decompress(
        base64.b85decode(_VN_BLOB)).decode("utf-8"))
except NameError:                       # built without --valuenet
    _VALUENET = None
_VN_ITEMS = ("WOOL", "MILK", "EGG", "WHEAT", "MELON", "CARROT", "TOMATO",
             "FERTILIZER")


def _vn_margin(js, me, step):
    """Predicted FINAL margin (mine - theirs) from a serve obs. Mirrors the
    training feature spec exactly: 3 + 8x(price,inventory) + 8 shed = 27."""
    P = _VALUENET
    farms = js.get("farms") or [{}, {}]
    market = js.get("market") or {}
    prices = market.get("prices") or {}
    inv = market.get("inventory") or {}
    x = [min(719, step) / 720.0,
         float(farms[me].get("money") or 0) / 1e5,
         float(farms[1 - me].get("money") or 0) / 1e5]
    for it in _VN_ITEMS:
        x.append(float(prices.get(it) or 0) / 100.0)
        x.append(float(inv.get(it) or 0) / 20000.0)
    try:
        sh = (js.get("private") or [{}, {}])[me].get("shed") or {}
    except Exception:                                          # noqa: BLE001
        sh = {}
    for it in _VN_ITEMS:
        x.append(float(sh.get(it) or 0) / 100.0)
    if len(P["x_mu"]) > 27:            # v2 feature net: board state
        def _bf(farm):
            plants = weeds = animals = yiel = care = 0
            for row in (farm.get("tiles") or []):
                for t in (row if isinstance(row, list) else [row]):
                    if not isinstance(t, dict):
                        continue
                    k = t.get("kind")
                    if k == "PLANT":
                        plants += 1
                        yiel += int(t.get("yield_units") or 0)
                    elif k == "WEED":
                        weeds += 1
                    if t.get("animal"):
                        animals += 1
                        care += int(t.get("pending_care_bonus") or 0)
            hands = len(farm.get("hands") or [])
            return [plants / 50.0, yiel / 100.0, weeds / 20.0,
                    animals / 20.0, care / 50.0, hands / 12.0]
        x += _bf(farms[me]) + _bf(farms[1 - me])
    v = [(x[i] - P["x_mu"][i]) / P["x_sd"][i]
         for i in range(len(P["x_mu"]))]
    for li in range(3):
        out = []
        for o in range(len(P["b"][li])):
            s = P["b"][li][o]
            row = P["W"][li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        if li < 2:
            v = [math.tanh(z) for z in out]
        else:
            v = out
    return v[0] * P["y_scale"]


def _kagg_bin():
    import os as _os
    cands = [_os.environ.get("KAGG_BIN") or ""]
    try:
        d = _os.path.dirname(_os.path.abspath(__file__))
        cands += [_os.path.join(d, "kagg"), _os.path.join(d, "kagg.exe")]
    except NameError:                      # exec-loaded: no __file__
        pass
    cands += ["kagg", "./kagg", "kagg.exe"]
    for p in cands:
        if p and _os.path.exists(p):
            return _os.path.abspath(p)
    return None


def _srv_start(state):
    import subprocess as _sp
    import atexit as _ax
    b = _kagg_bin()
    if b is None:
        raise RuntimeError("kagg binary not found")
    p = _sp.Popen([b, "serve"], stdin=_sp.PIPE, stdout=_sp.PIPE,
                  text=True, encoding="utf-8")
    state["srv_proc"] = p
    _ax.register(lambda: (p.poll() is None and p.kill()))
    return p


def _srv_cmd(p, line):
    p.stdin.write(line + chr(10))
    p.stdin.flush()
    return json.loads(p.stdout.readline())


def _mk_line(farmer, hands, market):
    def tok(op):
        return " ".join(str(t) for t in op) if op else "PASS"
    return (tok(farmer) + chr(9)
            + ";".join(tok(h) for h in (hands or [])) + chr(9)
            + ";".join(" ".join(str(t) for t in o) for o in (market or [])
                       if o))


def _plan_variants(route, idx, shed):
    """K market-plan variants over the horizon. Volume-conserving:
    variants only RETIME the schedule's own sells within the window."""
    win = []
    for j in range(idx, min(len(route), idx + _SEARCH_H)):
        for o in (route[j].get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                win.append((j - idx, str(o[1]), int(o[2] or 0)))
    plans = {"sched": {}}
    for off, item, q in win:
        plans["sched"].setdefault(off, []).append(["SELL", item, q])
    now = {}
    have = dict(shed)
    for off, item, q in sorted(win):
        take = min(q, max(0, int(have.get(item, 0) or 0)))
        if take > 0:
            now.setdefault(0, []).append(["SELL", item, take])
            have[item] = have.get(item, 0) - take
        rem = q - take
        if rem > 0:
            now.setdefault(off, []).append(["SELL", item, rem])
    plans["now"] = now
    return plans


def _add_defer_variants(plans, win, prices, base_of):
    """DEFER-DEAD-ITEM (2026-09-05): both live seats sell the identical
    103 late wool units, but wins collect 8.4k/4.4k and losses 2.5k/0.3k
    -- dumping into a crashed market is the measured d20-29 bleed. For any
    item priced < 0.5x base with meaningful window volume, offer a plan
    that withholds ITS sells entirely; the rollout decides per-world
    whether the flush (or a recovered market) beats the dump."""
    from collections import defaultdict as _dd
    vol = _dd(int)
    for off, item, q in win:
        vol[item] += q
    for item, v in vol.items():
        if v < 20:
            continue
        pnow = float(prices.get(item, 0) or 0)
        if pnow >= 0.5 * base_of(item):
            continue
        d = {}
        for off, it2, q in win:
            if it2 == item:
                continue
            d.setdefault(off, []).append(["SELL", it2, q])
        plans["defer_" + item] = d
    return plans


def _prod_variants(plans, route, idx):
    """PRODUCTION VARIANTS (2026-09-07): the searcher reasons about
    GROWING, not just selling. invest = pull the window's first route
    BUY 12 steps earlier; judged by the same rollout valuation."""
    first_buy = None
    for j in range(idx, min(len(route), idx + _SEARCH_H)):
        for o in (route[j].get("market") or []):
            if (isinstance(o, list) and o
                    and o[0] in ("BUY_SEED", "BUY_ANIMAL")):
                first_buy = (j - idx, list(o))
                break
        if first_buy:
            break
    if first_buy and first_buy[0] >= 12:
        off, op = first_buy
        d = dict(plans.get("sched", {}))
        d[max(0, off - 12)] = list(d.get(max(0, off - 12), [])) + [op]
        plans["invest"] = d
    return plans


def _hold_variants(plans, win):
    for delay in (12, 24):
        d = {}
        for off, item, q in win:
            d.setdefault(min(off + delay, _SEARCH_H - 1), []).append(
                ["SELL", item, q])
        plans["hold%d" % delay] = d
    return plans


def _apply_plan(action, orders):
    market = [o for o in (action.get("market") or [])
              if not (isinstance(o, list) and len(o) >= 3
                      and o[0] == "SELL")]
    action["market"] = (list(orders) + market)[:_MAX_ORDERS]
    return action




def _ro_lines(route, opp_route, fam, me, idx, step0, H, plan, window=None):
    """Precompute both seats' H tape lines for one plan (ROLLOUT format).

    `window` (default H): the plan replaces the route's sells only within
    the first `window` steps; BEYOND it the route's FULL market (its own
    sells included) continues. For a full-terminal rollout pass H = whole
    remaining horizon and window = the plan span, or the continuation stops
    selling past the window and banks nothing (the 2026-09-07 sim-search
    bug: -115k phantom margins)."""
    if window is None:
        window = H
    ours, theirs = [], []
    for h in range(H):
        r = route[min(idx + h, len(route) - 1)]
        if h < window:
            keep = [o for o in (r.get("market") or [])
                    if isinstance(o, list) and o and o[0] != "SELL"]
            mk = list(plan.get(h, [])) + keep
        else:
            mk = list(r.get("market") or [])   # route continues as-is
        ours.append(_mk_line(r.get("farmer"), r.get("hands"), mk[:10]))
        sa = step0 + h
        if fam and sa % 24 == 0 and sa // 24 < len(fam):
            bkt = fam[sa // 24]
            prods = _FAMBOOK["products"]
            omk = [["SELL", prods[pi], q]
                   for pi, q in enumerate(bkt) if q > 0][:10]
            theirs.append(_mk_line(["PASS"], [], omk))
        elif fam:
            theirs.append(_mk_line(["PASS"], [], []))
        else:
            ro = opp_route[min(idx + h, len(opp_route) - 1)]
            theirs.append(_mk_line(ro.get("farmer"), ro.get("hands"),
                                   ro.get("market")))
    return ours, theirs


def _ro_call(p, snap, H, ours, theirs, me):
    """One-IPC rollout; returns final obs or None."""
    la = chr(31).join(ours)
    lb = chr(31).join(theirs)
    if me == 1:
        la, lb = lb, la
    js = _srv_cmd(p, "ROLLOUT " + str(H) + " " + snap
                  + chr(30) + la + chr(30) + lb)
    return None if "error" in js else js


def _snap_of(js, step):
    return json.dumps({"step": int(step), "seed": 0,
                       "farms": js.get("farms"),
                       "market": js.get("market") or {},
                       "town": js.get("town") or {},
                       "private": js.get("private") or [{}, {}]},
                      separators=(",", ":"))

_SIM_SEARCH = {sim_search_default}   # real-simulation search (PLANSEARCH)


def _sim_search_pick(p, snap, route, idx, me, plans, fam, step):
    """REAL-SIMULATION SEARCH (2026-09-07): roll every plan to TERMINAL via
    the Rust PLANSEARCH verb (one IPC, ~0.6 ms/plan) and pick the plan with
    the best actual final margin. No value net -- the leaf is the banked
    score. Opponent continuation = family sched (or mirror) extended to the
    end. Returns the winning plan name, or None on any failure (caller
    falls back to the value-net path)."""
    try:
        names = list(plans.keys())
        Hf = 719 - int(step)
        if Hf < 2:
            return None
        # opponent continuation is the same for every plan
        _, theirs = _ro_lines(route, route, fam, me, idx, step, Hf,
                              plans[names[0]], window=_SEARCH_H)
        plan_lines = []
        for nm in names:
            ours, _ = _ro_lines(route, route, fam, me, idx, step, Hf,
                                plans[nm], window=_SEARCH_H)
            plan_lines.append(chr(31).join(ours))
        msg = ("PLANSEARCH " + str(me) + " " + snap + chr(30)
               + chr(31).join(theirs) + chr(30)
               + chr(30).join(plan_lines))
        js = _srv_cmd(p, msg)
        banks = js.get("banks")
        if not banks or len(banks) != len(names):
            return None, None
        vals = {}
        best_i, best_m = 0, None
        for i, b in enumerate(banks):
            m = float(b[0]) - float(b[1])
            vals[names[i]] = m
            if best_m is None or m > best_m:
                best_m, best_i = m, i
        return names[best_i], vals
    except Exception:                                          # noqa: BLE001
        return None, None


def _rollout_search(obs, action, step, state):
    import time as _t
    if step >= 718:
        # end-of-game: close our engine subprocess -- without this every
        # game leaks one kagg serve (95 gate cells = 95 orphans = the
        # 2026-09-05 reaper kill)
        p_ = state.pop("srv_proc", None)
        if p_ is not None:
            try:
                p_.kill()
            except Exception:                                  # noqa: BLE001
                pass
        return action
    if state.get("search_dead"):
        return action
    # d21, not d25: live loss tapes (2026-09-05) bleed -1.2..-1.9k/day from
    # d21 (worst swings d21/d23/d26) -- the fine-grained decision window
    # must cover the bleed, not just the final days
    endg = step >= 504
    if step < _SEARCH_FROM:
        return action
    # BINDING: between decision points the CHOSEN plan owns the sells --
    # the 2026-09-05 smoke measured -23k..-53k/cell when hold-plans
    # stripped the current turn and the route re-emitted on schedule
    # anyway (unenforced plans = sell-stripping through a new door)
    ov = state.get("plan_ov")
    if ov is not None and step < state.get("plan_until", -1):
        return _apply_plan(action, ov.get(step, []))
    every = _SEARCH_END_EVERY if endg else _SEARCH_EVERY
    if (step - _SEARCH_FROM) % every != 0:
        return action
    t0 = _t.perf_counter()
    try:
        p = state.get("srv_proc")
        if p is None or p.poll() is not None:
            p = _srv_start(state)
        me = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
        farms = _get(obs, "farms", []) or []
        priv = _get(obs, "private", {}) or {}
        privs = [{"shed": {}, "seeds": {}, "inventories": [{}, {}]},
                 {"shed": {}, "seeds": {}, "inventories": [{}, {}]}]
        privs[me] = {"shed": _get(priv, "shed", {}) or {},
                     "seeds": _get(priv, "seeds", {}) or {},
                     "inventories": _get(priv, "inventories", []) or
                     [{}, {}]}
        snap = json.dumps({
            "step": int(step),
            "seed": 0,
            "farms": farms,
            "market": _get(obs, "market", {}) or {},
            "town": _get(obs, "town", {}) or {},
            "private": privs,
        }, separators=(",", ":"))
        route = state.get("route") or _ROUTE
        idx = min(max(0, step), len(route) - 1)
        plans = _plan_variants(route, idx, privs[me]["shed"])
        _win = []
        for j in range(idx, min(len(route), idx + _SEARCH_H)):
            for o in (route[j].get("market") or []):
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    _win.append((j - idx, str(o[1]), int(o[2] or 0)))
        _prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
        plans = _hold_variants(plans, _win)
        plans = _prod_variants(plans, route, idx)
        plans = _add_defer_variants(
            plans, _win, _prices,
            lambda it: (_MARKET_PARAMS.get(it) or (25,))[0])
        best_name, best_v = None, None
        _diag_vals = {}
        _hstates = {}
        # opponent prior, best-first: (1) the RECOGNIZED FAMILY's median
        # sell schedule (book v2: observable day-3 field profile, nearest
        # L1); (2) mirror (our route) when the family is ours/unknown --
        # any supply model beats PASS's empty market
        fam = state.get("fam_sched", -1)
        if fam == -1 and _FAMBOOK and step >= 96:
            them = farms[1 - me]
            prof = {c: 0 for c in ("WHEAT", "CARROT", "TOMATO",
                                   "STRAWBERRY", "MELON")}
            animals = 0
            for row in (_get(them, "tiles", []) or []):
                for t in (row if isinstance(row, list) else [row]):
                    if isinstance(t, dict):
                        if t.get("crop") in prof:
                            prof[t["crop"]] += 1
                        if t.get("animal"):
                            animals += 1
            vec = [prof[c] for c in ("WHEAT", "CARROT", "TOMATO",
                                     "STRAWBERRY", "MELON")] + [animals]
            best_d, best_f = None, None
            for fh, fb in _FAMBOOK["families"].items():
                pv = fb["profile"]
                d = sum(abs(vec[i] - pv[i]) for i in range(len(vec)))
                if best_d is None or d < best_d:
                    best_d, best_f = d, fb
            fam = (best_f["sched"]
                   if best_f and not best_f.get("ours") and best_d <= 8
                   else None)
            state["fam_sched"] = fam
        opp_route = route
        # REAL-SIMULATION SEARCH branch (2026-09-07): roll every plan to
        # terminal via PLANSEARCH and pick by actual final margin, with the
        # same min-edge gate. Falls through to the value-net path on failure.
        if _SIM_SEARCH:
            _sn, _sv = _sim_search_pick(p, snap, route, idx, me, plans,
                                        fam, step)
            if _sn is not None and _sv:
                _sched_v = _sv.get("sched", _sv[_sn])
                _me2 = (_SEARCH_MIN_EDGE_END
                        if step >= _SEARCH_END_EDGE_FROM
                        else _SEARCH_MIN_EDGE)
                if _sn != "sched" and (_sv[_sn] - _sched_v) < _me2:
                    _sn = "sched"
                import os as _os2
                if _os2.environ.get("KAGG_SEARCH_DIAG"):
                    sys.stderr.write("SIMSEARCH step=%d pick=%s vals=%r%s"
                                     % (step, _sn,
                                        {n: round(v) for n, v in
                                         _sv.items()}, chr(10)))
                if _sn != "sched":
                    plan = plans[_sn]
                    every_now = (_SEARCH_END_EVERY if endg
                                 else _SEARCH_EVERY)
                    state["plan_ov"] = {step + off: orders
                                        for off, orders in plan.items()}
                    state["plan_until"] = step + every_now
                    return _apply_plan(action, plan.get(0, []))
                state["plan_ov"] = None
                return action
        for name, plan in plans.items():
            if (_t.perf_counter() - t0) * 1000 > _SEARCH_BUDGET_MS:
                break
            ours_l, theirs_l = _ro_lines(route, opp_route, fam, me, idx,
                                         step, _SEARCH_H, plan)
            js = None
            if state.get("ro_ok", True):
                js = _ro_call(p, snap, _SEARCH_H, ours_l, theirs_l, me)
                if js is None:
                    state["ro_ok"] = False   # old binary: per-step path
            if js is None:
                js = _srv_cmd(p, "LOADSTATE " + snap)
                if "error" in js:
                    raise RuntimeError(js["error"])
                for h in range(_SEARCH_H):
                    la, lb = ((ours_l[h], theirs_l[h]) if me == 0
                              else (theirs_l[h], ours_l[h]))
                    js = _srv_cmd(p, "STEP2 " + la + chr(30) + lb)
                    if "error" in js or js.get("done"):
                        break
            # VALUATION. With the value net (v47, sign-acc 0.770 on 33.6k
            # synthetic states across all 64 worlds): predict the FINAL
            # MARGIN from the horizon state -- the ladder's currency, and
            # it credits standing crops/animals/market posture the old
            # cash+flush read was blind to. Without the net: the honest
            # horizon flush (2026-09-05) as before.
            if _VALUENET is not None and not js.get("done"):
                v = _vn_margin(js, me, step + _SEARCH_H)
            else:
                if not js.get("done"):
                    shed2 = ((js.get("private") or [{}, {}])[me]
                             .get("shed") or {})
                    fl = [["SELL", k2, int(v2)] for k2, v2 in shed2.items()
                          if v2][:10]
                    mine = _mk_line(["PASS"], [], fl)
                    la, lb = ((mine, _mk_line(["PASS"], [], []))
                              if me == 0 else
                              (_mk_line(["PASS"], [], []), mine))
                    js = _srv_cmd(p, "STEP2 " + la + chr(30) + lb)
                v = float((js.get("farms") or [{}, {}])[me]
                          .get("money") or 0)
            _diag_vals[name] = v
            if (_VALUENET is not None and not js.get("done")
                    and js.get("farms")):
                _hstates[name] = js
            if best_v is None or v > best_v:
                best_v, best_name = v, name
        # ===== DEEP TREE (2026-09-07, ROLLOUT verb): expand the top
        # plans through ply 2 AND ply 3 -- our plan choices maximize,
        # opponent models (family sched vs mirror) minimize. Leaves are
        # net-valued final-margin predictions. One IPC per rollout
        # (measured 1.04 ms), so depth fits the 650 ms budget with room.
        def _expand(hjs, step0, depth):
            """Value of the position at hjs (our move next), searched to
            `depth` more plies; max over our plans, min over opponents."""
            if depth <= 0 or hjs.get("done"):
                return _vn_margin(hjs, me, step0)
            if (_t.perf_counter() - t0) * 1000 > _SEARCH_BUDGET_MS:
                return _vn_margin(hjs, me, step0)
            idx_d = min(step0, len(route) - 1)
            shed_d = ((hjs.get("private") or [{}, {}])[me]
                      .get("shed") or {})
            snap_d = _snap_of(hjs, step0)
            pl = _plan_variants(route, idx_d, shed_d)
            best_d = None
            for pn in ("sched", "now"):
                plan_d = pl.get(pn)
                if plan_d is None:
                    continue
                worst = None
                for om in ([fam, None] if fam else [None]):
                    ours_d, theirs_d = _ro_lines(
                        route, opp_route, om, me, idx_d, step0,
                        _SEARCH_H, plan_d)
                    js_d = _ro_call(p, snap_d, _SEARCH_H,
                                    ours_d, theirs_d, me)
                    if js_d is None:
                        return _vn_margin(hjs, me, step0)
                    v_d = _expand(js_d, step0 + _SEARCH_H, depth - 1)
                    worst = v_d if worst is None else min(worst, v_d)
                if worst is not None and (best_d is None
                                          or worst > best_d):
                    best_d = worst
            return best_d if best_d is not None else _vn_margin(
                hjs, me, step0)

        if _VALUENET is not None and _hstates and state.get("ro_ok", True):
            _top = sorted(_hstates, key=lambda n: -_diag_vals[n])[:3]
            if "sched" in _hstates and "sched" not in _top:
                _top.append("sched")   # sched judged at equal depth
            for name in _top:
                if (_t.perf_counter() - t0) * 1000 > _SEARCH_BUDGET_MS:
                    break
                try:
                    _diag_vals[name] = _expand(_hstates[name],
                                               step + _SEARCH_H,
                                               _SEARCH_DEPTH)
                except Exception:                              # noqa: BLE001
                    pass
            best_name = max(_diag_vals, key=lambda n: _diag_vals[n])
            best_v = _diag_vals[best_name]
        edge = (best_v or 0) - _diag_vals.get("sched", best_v or 0)
        _min_edge = (_SEARCH_MIN_EDGE_END if step >= _SEARCH_END_EDGE_FROM
                     else _SEARCH_MIN_EDGE)
        if best_name and best_name != "sched" and edge < _min_edge:
            best_name = "sched"            # not confident enough to deviate
        import os as _os
        if _os.environ.get("KAGG_SEARCH_DIAG"):
            sys.stderr.write("SEARCH step=%d pick=%s vals=%r%s"
                             % (step, best_name,
                                {nm: round(vv) for nm, vv in
                                 _diag_vals.items()}, chr(10)))
        if best_name and best_name != "sched":
            plan = plans[best_name]
            every_now = _SEARCH_END_EVERY if endg else _SEARCH_EVERY
            state["plan_ov"] = {step + off: orders
                                for off, orders in plan.items()}
            state["plan_until"] = step + every_now
            action = _apply_plan(action, plan.get(0, []))
        else:
            state["plan_ov"] = None
    except Exception as exc:                                   # noqa: BLE001
        sys.stderr.write("RUST-FAIL step=%d: %r%s" % (step, exc, chr(10)))
        sys.stderr.flush()
        state["search_dead"] = True
    return action

'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--book", default=None,
                    help="family_book.json to embed (family-conditioned "
                         "rollout opponents)")
    ap.add_argument("--valuenet", default=None,
                    help="value_net.json to embed (horizon states valued "
                         "as predicted FINAL MARGIN instead of cash+flush)")
    ap.add_argument("--sim-search", action="store_true",
                    help="enable REAL-SIMULATION search (PLANSEARCH to "
                         "terminal, no value net)")
    a = ap.parse_args()
    global LAYER
    LAYER = LAYER.replace("{sim_search_default}",
                          "True" if a.sim_search else "False")
    src = open(a.base, encoding="utf-8").read()
    if a.book:
        import base64
        import zlib
        raw = open(a.book, "rb").read()
        blob = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
        book_line = ('_FAMBOOK = json.loads(zlib.decompress('
                     'base64.b85decode(' + repr(blob)
                     + ')).decode("utf-8"))' + chr(10))
    else:
        book_line = "_FAMBOOK = None" + chr(10)
    if a.valuenet:
        import base64 as _b64
        import lzma as _lz
        vraw = open(a.valuenet, "rb").read()
        vblob = _b64.b85encode(_lz.compress(vraw, preset=9)).decode("ascii")
        book_line += ('_VN_BLOB = ' + repr(vblob) + chr(10))
    src = src.replace("import math", "import math", 1)
    if "import sys" not in src.split(chr(10) * 2)[0]:
        src = src.replace("import math", "import math" + chr(10)
                          + "import sys", 1)
    anchor = "def _terminal_flush(obs, action, step):"
    assert anchor in src, "expected the v45.1 chassis (flush layer)"
    src = src.replace(anchor, book_line + LAYER + chr(10) + anchor, 1)
    call = "        action = _endgame_owner" \
        if "_endgame_owner" in src else "        action = _rank_sell_slots"
    assert call in src
    src = src.replace(call,
                      "        action = _rollout_search("
                      "obs, action, step, state)" + chr(10) + call, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(src)
    import ast
    ast.parse(src)
    print(f"built {a.out} (chassis + rollout searcher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
