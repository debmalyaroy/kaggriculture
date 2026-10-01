"""Assemble the FULL v46.1 chassis: world-tail router + opp-dump pre-empt.

Implements the 2026-09-06 field-intel action items on top of the already
gated base-swap + race-flush chassis (.local/candidates/v461_race.py):

  * WORLD-TAIL ROUTER -- at step 145/146, when both town shops are known,
    swap the route to a donor tail recorded by a clone of our own opening
    (whole-channel prefix agreement 1.000 over turns 0..144, winning game,
    bank >= 110k). The rank-1 field architecture, built from the 32% of
    the field that copied us. Tails from data/worlds via the extractor.
  * OPP-DUMP PRE-EMPT -- the observable-features MLP (held-out mean AUC
    0.807, models/trackp/opp_sell/mlp.json) predicts "opponent dumps item
    X within 12 steps"; when a shippable head (val AUC >= 0.70) fires
    above threshold, our own scheduled sells of that item within the next
    12 turns are advanced to NOW through the existing p2_adv retime
    ledger (volume-conserving by construction). Never before step 192
    (day-6 funding doctrine), never after 640 (endgame owned by the
    flush), one shot per item per day.

    python src/trackp/harness/build_v461_full.py \
        --chassis .local/candidates/v461_race.py \
        --out .local/candidates/v461_bandit_full.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import json
import os
import sys
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass


def pack(obj):
    return base64.b85encode(zlib.compress(
        json.dumps(obj, separators=(",", ":")).encode("utf-8"), 9)
    ).decode("ascii")


def pack_lzma(obj):
    import lzma
    return base64.b85encode(lzma.compress(
        json.dumps(obj, separators=(",", ":")).encode("utf-8"), preset=9)
    ).decode("ascii")


LAYER = r'''

# ===== v46.1 FULL: world-tail router + opp-dump pre-empt (2026-09-06) =====
_WORLDTAILS = json.loads(__import__("lzma").decompress(base64.b85decode(
    "{tails_blob}")).decode("utf-8"))
_DAY3 = json.loads(__import__("lzma").decompress(base64.b85decode(
    "{d3_blob}")).decode("utf-8"))
_WORLDTAILS2 = json.loads(__import__("lzma").decompress(base64.b85decode(
    "{tails2_blob}")).decode("utf-8"))
_DAY6 = json.loads(__import__("lzma").decompress(base64.b85decode(
    "{d6_blob}")).decode("utf-8"))


def _tail_select(obs, state, step, tails):
    """RUNTIME TAIL SELECTION (2026-09-07): roll each candidate tail 48
    steps from the CURRENT game state (one ROLLOUT IPC each); keep the
    higher horizon bank+shed value. Any failure -> primary tail."""
    import subprocess as _sp
    import os as _os
    p = state.get("ts_proc")
    if p is None or p.poll() is not None:
        b = None
        cand = [_os.environ.get("KAGG_BIN") or ""]
        try:
            d = _os.path.dirname(_os.path.abspath(__file__))
            cand += [_os.path.join(d, "kagg"),
                     _os.path.join(d, "kagg.exe")]
        except NameError:
            pass
        cand += ["kagg", "./kagg", "kagg.exe"]
        for c in cand:
            if c and _os.path.exists(c):
                b = c
                break
        if b is None:
            return tails[0]
        p = _sp.Popen([b, "serve"], stdin=_sp.PIPE, stdout=_sp.PIPE,
                      text=True, encoding="utf-8")
        state["ts_proc"] = p
    me = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    farms = _get(obs, "farms", []) or []
    priv = _get(obs, "private", {}) or {}
    privs = [{"shed": {}, "seeds": {}, "inventories": [{}, {}]},
             {"shed": {}, "seeds": {}, "inventories": [{}, {}]}]
    privs[me] = {"shed": _get(priv, "shed", {}) or {},
                 "seeds": _get(priv, "seeds", {}) or {},
                 "inventories": _get(priv, "inventories", []) or
                 [{}, {}]}
    snap = json.dumps({"step": int(step), "seed": 0, "farms": farms,
                       "market": _get(obs, "market", {}) or {},
                       "town": _get(obs, "town", {}) or {},
                       "private": privs}, separators=(",", ":"))
    def _lo(a):
        a = a or {}
        def tok(x):
            return " ".join(str(t) for t in x) if x else "PASS"
        return (tok(a.get("farmer")) + chr(9)
                + ";".join(tok(h) for h in (a.get("hands") or []))
                + chr(9)
                + ";".join(" ".join(str(t) for t in o)
                           for o in (a.get("market") or []) if o))
    # DISPATCHER (2026-09-07): with the `dispatcher` flag ON, roll each
    # candidate tail to TERMINAL via PLANSEARCH (one IPC, real final bank)
    # and pick the highest actual margin -- no 48-step-horizon proxy, no
    # shed heuristic. Opponent continuation = mirror (both play their tail).
    # Falls back to the ROLLOUT-48 proxy on any failure.
    if globals().get("_FLAGS", {}).get("dispatcher", True):
        try:
            hf = 719 - int(step)
            tl0 = tails[0]
            opp = chr(31).join(_lo(tl0[(step - 145) + h]
                               if 0 <= (step - 145) + h < len(tl0) else None)
                               for h in range(hf))
            plan_blk = []
            for tl in tails:
                plan_blk.append(chr(31).join(
                    _lo(tl[(step - 145) + h]
                        if 0 <= (step - 145) + h < len(tl) else None)
                    for h in range(hf)))
            p.stdin.write("PLANSEARCH " + str(me) + " " + snap + chr(30)
                          + opp + chr(30) + chr(30).join(plan_blk)
                          + chr(10))
            p.stdin.flush()
            js = json.loads(p.stdout.readline())
            banks = js.get("banks")
            if banks and len(banks) == len(tails):
                bi, bm = 0, None
                for i, b in enumerate(banks):
                    m = float(b[0]) - float(b[1])
                    if bm is None or m > bm:
                        bm, bi = m, i
                return tails[bi]
        except Exception:                                      # noqa: BLE001
            pass
    best, best_v = tails[0], None
    for tl in tails:
        blk = chr(31).join(_lo(tl[i] if i < len(tl) else None)
                           for i in range(48))
        p.stdin.write("ROLLOUT 48 " + snap + chr(30) + blk + chr(30)
                      + blk + chr(10))
        p.stdin.flush()
        js = json.loads(p.stdout.readline())
        if "error" in js:
            return tails[0]
        v = float((js.get("farms") or [{}, {}])[me].get("money") or 0)
        sh = ((js.get("private") or [{}, {}])[me].get("shed") or {})
        v += 0.5 * sum(int(q or 0) for q in sh.values())
        if best_v is None or v > best_v:
            best, best_v = tl, v
    return best
_PS = json.loads(zlib.decompress(base64.b85decode(
    "{mlp_blob}")).decode("utf-8"))
_PS_ITEMS = ("WOOL", "MILK", "EGG", "WHEAT", "MELON", "CARROT", "TOMATO",
             "FERTILIZER")
_PS_THRESH = 0.85
_PS_HEADS = [i for i, it in enumerate(_PS["items"])
             if _PS["aucs"].get(it, 0) >= 0.70]
_COUNTER = json.loads(__import__("lzma").decompress(base64.b85decode(
    "{counter_blob}")).decode("utf-8"))
_MIRROR = {"on": False}   # chassis layers read this (flush timing)


def _mirror_check(obs, state, step):
    """COUNTER-HARDENING (2026-09-06): detect an opponent running our own
    lineage. Probes at steps 73/97/121 compare the two farms' observable
    boards (crop counts + animals); mirrors match near-exactly. A detected
    mirror gets the aggressive settings: they know our public schedule, so
    we front-run harder and open the settlement race earlier."""
    if step in (73, 97, 121) and "mirror" not in state:
        try:
            farms = _get(obs, "farms", []) or [{}, {}]
            me = int(_get(obs, "player", 0) or 0)
            def prof(f):
                crops, animals = {}, 0
                for row in (_get(f, "tiles", []) or []):
                    for t in (row if isinstance(row, list) else [row]):
                        if isinstance(t, dict):
                            c = t.get("crop")
                            if c:
                                crops[c] = crops.get(c, 0) + 1
                            if t.get("animal"):
                                animals += 1
                return crops, animals
            mc, ma = prof(farms[me])
            oc, oa = prof(farms[1 - me])
            keys = set(mc) | set(oc)
            d = sum(abs(mc.get(k, 0) - oc.get(k, 0)) for k in keys)
            d += abs(ma - oa)
            hits = state.setdefault("mir_hits", 0)
            if d <= 2:
                state["mir_hits"] = hits + 1
            if step == 121:
                state["mirror"] = state.get("mir_hits", 0) >= 2
        except Exception:                                      # noqa: BLE001
            state["mirror"] = False
    _MIRROR["on"] = bool(state.get("mirror"))


def _ps_forward(x):
    P = _PS
    v = [(x[i] - P["x_mu"][i]) / P["x_sd"][i] for i in range(len(x))]
    for li in range(len(P["W"])):
        out = []
        for o in range(len(P["b"][li])):
            s = P["b"][li][o]
            row = P["W"][li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        if li < len(P["W"]) - 1:
            v = [math.tanh(z) for z in out]
        else:
            v = out
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) for z in v]


def _ps_features(obs, state, step):
    """MUST mirror opp_sell_dataset.episode_rows exactly, quirks included."""
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or [{}, {}]
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    inv = _get(market, "inventory", {}) or {}
    snaps = state.setdefault("ps_snap", [])
    solds = state.setdefault("ps_sold", [])
    # sells we emitted LAST step were recorded by the chassis
    last = state.get("our_last_sells") or {}
    solds.append({it: int(last.get(it, 0) or 0) for it in _PS_ITEMS})
    snaps.append((dict(inv), dict(prices)))
    if len(snaps) > 26:
        del snaps[0]
        del solds[0]
    f = [step / 720.0, (step % 24) / 24.0,
         float(_get(farms[me], "money", 0) or 0) / 1e5,
         float(_get(farms[1 - me], "money", 0) or 0) / 1e5]
    for it in _PS_ITEMS:
        f.append(float(_get(prices, it, 0) or 0) / 100.0)
    n = len(snaps)
    for w in (6, 24):
        j = max(0, n - 1 - w)
        inv0, pr0 = snaps[j]
        my_sold = sum(sum(d.values()) for d in solds[j:n - 1])
        d_inv = sum(float(inv.get(it) or 0) - float(inv0.get(it) or 0)
                    for it in _PS_ITEMS)
        f.append((d_inv - my_sold) / (10.0 * w))
        for it in ("WOOL", "MILK", "WHEAT", "MELON"):
            f.append((float(_get(prices, it, 0) or 0)
                      - float(pr0.get(it) or 0)) / 50.0)
    j6 = max(0, n - 1 - 6)
    inv6, _pr6 = snaps[j6]
    for it in _PS_ITEMS:
        d_it = (float(inv.get(it) or 0) - float(inv6.get(it) or 0)
                - solds[j6].get(it, 0))
        f.append(d_it / 60.0)
    return f


def _preempt_sells(obs, action, step, state):
    """Advance our own scheduled sells of an item the opponent is about to
    dump. Volume-conserving: every advanced unit is subtracted from its
    scheduled turn through the p2_adv ledger the retimer already honours."""
    try:
        _mirror_check(obs, state, step)
        feats = _ps_features(obs, state, step)   # buffers update every turn
        if step < 192 or step >= 640:
            return action
        probs = _ps_forward(feats)
        thresh = 0.80 if state.get("mirror") else _PS_THRESH
        shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        route = state.get("route") or _ROUTE
        cool = state.setdefault("ps_cool", {})
        adv = state.setdefault("p2_adv", {})
        market = list(action.get("market") or [])
        for hi in _PS_HEADS:
            it = _PS["items"][hi]
            if probs[hi] < thresh:
                continue
            if step - cool.get(it, -99) < 24:
                continue
            have = int(_get(shed, it, 0) or 0)
            if have <= 0 or len(market) >= _MAX_ORDERS:
                continue
            take = 0
            for off in range(1, 13):
                idx = step + off
                if idx >= len(route) or not (192 <= idx < 640):
                    continue
                for o in (route[idx].get("market") or []):
                    if (isinstance(o, list) and len(o) >= 3
                            and o[0] == "SELL" and o[1] == it):
                        q = int(o[2] or 0)
                        room = min(q, have - take)
                        if room > 0:
                            adv[(idx, it)] = adv.get((idx, it), 0) + room
                            take += room
                if take >= have:
                    break
            if take > 0:
                market.append(["SELL", it, int(take)])
                cool[it] = step
        action["market"] = market[:_MAX_ORDERS]
    except Exception:                                          # noqa: BLE001
        pass
    return action
# ===== end v46.1 FULL layers =====

'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chassis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flags", default=None,
                    help="feature_flags.json to embed (default: models one)")
    a = ap.parse_args()
    src = open(a.chassis, encoding="utf-8").read()

    tails_raw = json.load(open(os.path.join(
        ROOT, ".local", "candidates", "world_tails.json"), encoding="utf-8"))
    tails = {k: v["tail"] for k, v in tails_raw.items()
             if v["bank"] > v["opp_bank"]
             or str(v.get("team", "")).startswith("EVOLVED")}
    # winning donors, plus evolved tails (they carry bank 0/0 -- their win
    # is the tournament/evolution score, not a recorded game; the 2026-09-06
    # 16-tail near-miss was this filter silently dropping all 47 of them)
    prunep = os.path.join(ROOT, ".local", "candidates", "tail_prune.json")
    if os.path.exists(prunep):
        # tails the band gate measured as credited-win flips (2026-09-06)
        for k in json.load(open(prunep, encoding="utf-8")):
            tails.pop(k, None)
    print(f"embedding {len(tails)} winning world tails "
          f"(dropped {len(tails_raw) - len(tails)} losers/pruned)")
    mlp = json.load(open(os.path.join(ROOT, "models", "trackp", "opp_sell",
                                      "mlp.json"), encoding="utf-8"))
    assert len(mlp["x_mu"]) == 30, f"feature dim {len(mlp['x_mu'])} != 30"
    d3p = os.path.join(ROOT, "models", "trackp", "day3_branches.json")
    d3 = {}
    if os.path.exists(d3p):
        raw3 = json.load(open(d3p, encoding="utf-8"))
        d3 = {k: v["cont"] for k, v in raw3.items()}
    print(f"day-3 branches: {len(d3)}")
    t2p = os.path.join(ROOT, ".local", "candidates", "world_tails2.json")
    tails2 = {}
    if os.path.exists(t2p):
        raw2 = json.load(open(t2p, encoding="utf-8"))
        tails2 = {k: v["tail"] for k, v in raw2.items() if k in tails}
    print(f"secondary tails: {len(tails2)}")
    d6p = os.path.join(ROOT, "models", "trackp", "day6_branches.json")
    d6 = {}
    if os.path.exists(d6p):
        raw6 = json.load(open(d6p, encoding="utf-8"))
        # MIN_EDGE was already applied by day6_evolve; keep parent tag --
        # the runtime splice must only fire on a matching prefix
        d6 = {k: {"p": v["parent"], "c": v["cont"]} for k, v in raw6.items()}
    print(f"day-6 branches: {len(d6)}")
    cp = os.path.join(ROOT, "models", "trackp", "owned_counter.json")
    counter = []
    if os.path.exists(cp):
        cj = json.load(open(cp, encoding="utf-8"))
        if cj.get("score", 0) >= 0.9:
            counter = cj["tape"]
    print(f"family counter: {'ON' if counter else 'off'}")
    layer = LAYER.replace("{tails_blob}", pack_lzma(tails)) \
                 .replace("{mlp_blob}", pack(mlp)) \
                 .replace("{d3_blob}", pack_lzma(d3)) \
                 .replace("{tails2_blob}", pack_lzma(tails2)) \
                 .replace("{counter_blob}", pack_lzma(counter)) \
                 .replace("{d6_blob}", pack_lzma(d6))

    anchor_def = "def _terminal_flush(obs, action, step):"
    assert src.count(anchor_def) == 1
    src = src.replace(anchor_def, layer + anchor_def, 1)

    # world-tail route swap, just before the route is read
    anchor_route = '        _route = state.get("route") or _ROUTE'
    assert src.count(anchor_route) == 1
    swap = '        if (not state.get("d3_done")) and 74 <= step <= 76:\n            _t3 = _get(obs, "town", {}) or {}\n            _s3 = _get(_t3, "unlocked_shops", []) or []\n            if _s3:\n                state["d3_done"] = True\n                _c3 = _DAY3.get(str(_s3[0]))\n                if _c3 is not None:\n                    state["route"] = _ROUTE[:74] + _c3\n                    state["d3branch"] = True\n        if (state.get("mirror") and not state.get("d3branch")\n                and not state.get("counter_on") and 122 <= step <= 124\n                and _COUNTER):\n            # FAMILY-KILLER (2026-09-07): counter-line evolved vs the\n            # clone family (fitness 0.979); prefix byte-identical\n            # through t=122, so the splice is exact. Owns the game.\n            state["route"] = _ROUTE[:122] + _COUNTER[122:719]\n            state["counter_on"] = True\n            state["wtail_done"] = True\n        if (not state.get("d6_done")) and 145 <= step <= 146 and not state.get("counter_on"):\n            # DAY-6 FORK (2026-09-07): per-world continuation evolved on\n            # top of the day-3 branch (or base route). Fires ONLY when\n            # the recorded parent matches the prefix actually played.\n            _t6 = _get(obs, "town", {}) or {}\n            _s6 = _get(_t6, "unlocked_shops", []) or []\n            if len(_s6) >= 2:\n                state["d6_done"] = True\n                _b6 = _DAY6.get(str(_s6[0]) + "|" + str(_s6[1]))\n                if _b6 is not None and ((_b6["p"] == "d3") == bool(state.get("d3branch"))):\n                    state["route"] = (state.get("route") or _ROUTE)[:146] + _b6["c"]\n                    state["d6branch"] = True\n                    state["wtail_done"] = True\n        if (not state.get("wtail_done")) and step >= 145 and not state.get("d3branch"):\n            _town2 = _get(obs, "town", {}) or {}\n            _sh2 = _get(_town2, "unlocked_shops", []) or []\n            if len(_sh2) >= 2 and step <= 146:\n                state["wtail_done"] = True\n                _wk2 = str(_sh2[0]) + "|" + str(_sh2[1])\n                _wt = _WORLDTAILS.get(_wk2)\n                _wtb = _WORLDTAILS2.get(_wk2)\n                if _wt is not None and _wtb is not None:\n                    try:\n                        _wt = _tail_select(obs, state, step, [_wt, _wtb])\n                    except Exception:\n                        pass\n                if _wt is not None:\n                    state["route"] = _ROUTE[:145] + _wt\n                    state["wtail"] = True\n            elif step > 146:\n                state["wtail_done"] = True\n'
    src = src.replace(anchor_route, swap + anchor_route, 1)

    # arms must not override a committed world tail
    anchor_arm = ("        arm = _bandit(obs, step, state)\n"
                  "        if arm is not None:")
    assert src.count(anchor_arm) == 1
    src = src.replace(anchor_arm,
                      "        arm = _bandit(obs, step, state)\n"
                      "        if arm is not None and "
                      "globals().get(\"_FLAGS\", {}).get(\"arms\", True)"
                      " and not "
                      "(state.get(\"wtail\") or state.get(\"d6branch\")"
                      " or state.get(\"d3branch\")"
                      " or state.get(\"counter_on\")):", 1)

    # mirror-aware race flush: a detected mirror opens the settlement
    # race 4 steps earlier (700 vs 704) -- they know our public timing
    _fl_anchor = '"""' + chr(10) + "    if step < 704:"
    assert src.count(_fl_anchor) == 1, src.count(_fl_anchor)
    src = src.replace(_fl_anchor,
                      '"""' + chr(10)
                      + "    if step < (702 if _MIRROR[\"on\"] "
                      "else 704):", 1)

    # pre-empt runs every turn (buffers), before sell-slot ranking
    anchor_pre = "        action = _rank_sell_slots(obs, action)"
    assert src.count(anchor_pre) == 1
    src = src.replace(anchor_pre,
                      "        action = _preempt_sells(obs, action, step, "
                      "state) if globals().get(\"_FLAGS\", {}).get("
                      "\"opp_model\", True) else action\n" + anchor_pre, 1)

    # ROUTER-LAYER flag gates (full pluggability): each owned-route layer
    # fires only when its flag is ON. A gates form `_FLAGS.get(k) and (cond)`.
    _gf = 'globals().get("_FLAGS", {}).get'
    _router_gates = [
        ('if (not state.get("d3_done")) and 74 <= step <= 76:',
         'if ' + _gf + '("d3_fork", True) and (not state.get("d3_done")) '
         'and 74 <= step <= 76:'),
        ('if (state.get("mirror") and not state.get("d3branch")',
         'if ' + _gf + '("family_counter", True) and (state.get("mirror") '
         'and not state.get("d3branch")'),
        ('if (not state.get("d6_done")) and 145 <= step <= 146 and not '
         'state.get("counter_on"):',
         'if ' + _gf + '("d6_fork", True) and (not state.get("d6_done")) '
         'and 145 <= step <= 146 and not state.get("counter_on"):'),
        ('if (not state.get("wtail_done")) and step >= 145 and not '
         'state.get("d3branch"):',
         'if ' + _gf + '("world_tail", True) and (not '
         'state.get("wtail_done")) and step >= 145 and not '
         'state.get("d3branch"):'),
    ]
    for _old, _new in _router_gates:
        assert src.count(_old) == 1, ("router gate anchor not unique: "
                                      + _old[:40])
        src = src.replace(_old, _new, 1)

    # PLUGGABLE FEATURE FLAGS (2026-09-07): gate every pipeline layer so its
    # contribution is measurable/toggleable. Flags default ON = shipped
    # behavior; ablation flips one at a time. --flags points at a config.
    try:
        import feature_flags as FF
        flags = FF.load_flags(getattr(a, "flags", None) or FF.CONFIG)
        src, ng = FF.inject(src, flags)
        print(f"feature flags: {ng} pipeline gates injected "
              f"({sum(1 for v in flags.values() if v)}/{len(flags)} ON)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"WARN feature-flag injection skipped: {exc}")
    open(a.out, "w", encoding="utf-8").write(src)
    import ast
    ast.parse(src)
    print(f"built {a.out} ({len(src):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
