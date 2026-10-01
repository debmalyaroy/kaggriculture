"""Embed a trained policy blob into a candidate agent (v2 GRU socket).

Takes the incumbent (v44.1_bandit.py), swaps the empty `_POLICY` blob for
the trained one, and -- for arch "gru_hybrid_v2" -- replaces the forward
with the recurrent step: a 64-unit GRU over the normalized market slice
whose hidden state lives in the per-game `state` dict, plus the [114,64,64,9]
MLP. The BOUNDED BLEND (delta clamp +/-_POLICY_DELTA_CAP before
_safe_market) is untouched: a wrong head still degrades smoothly.

The built candidate is verified by importing it and stepping one synthetic
turn before the file is accepted. Then the gate chain decides, as always:
tests -> hard band CREDITED > 26/56 (--rust) -> reactive band -> latency.

    python src/bc_embed.py models/bc/policy_cropdusta_gru.b85 \
        --base agents/v44.1_bandit.py --out .local/candidates/v45_gru.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

STEP_FN = '''
def _policy_step(x_raw, h):
    """One turn of the v2 hybrid head: (sell_vector, new_hidden).

    Pure python, `math` only -- equivalence to the torch trainer asserted at
    export (bc_train_gru.py). x_raw is the 50-vector (49 state + condition).
    """
    P = _POLICY
    xm, xs = P["x_mu"], P["x_sd"]
    xn = [(x_raw[i] - xm[i]) / xs[i] for i in range(len(x_raw))]
    a, b = P["market"]
    m = xn[a:b]
    H = P["hid"]
    if h is None:
        h = [0.0] * H
    wih, whh, bih, bhh = P["g_wih"], P["g_whh"], P["g_bih"], P["g_bhh"]

    def _gate(rows, bias, vec):
        out = []
        for o in range(H):
            s = bias[o]
            row = rows[o]
            for i in range(len(vec)):
                s += row[i] * vec[i]
            out.append(s)
        return out
    ir = _gate(wih[0:H], bih[0:H], m)
    iz = _gate(wih[H:2 * H], bih[H:2 * H], m)
    inn = _gate(wih[2 * H:], bih[2 * H:], m)
    hr = _gate(whh[0:H], bhh[0:H], h)
    hz = _gate(whh[H:2 * H], bhh[H:2 * H], h)
    hn = _gate(whh[2 * H:], bhh[2 * H:], h)
    r = [1.0 / (1.0 + math.exp(-(ir[i] + hr[i]))) for i in range(H)]
    z = [1.0 / (1.0 + math.exp(-(iz[i] + hz[i]))) for i in range(H)]
    n = [math.tanh(inn[i] + r[i] * hn[i]) for i in range(H)]
    h2 = [(1.0 - z[i]) * n[i] + z[i] * h[i] for i in range(H)]

    v = xn + h2
    W, B = P["W"], P["b"]
    for li in range(len(W)):
        out = []
        for o in range(len(B[li])):
            s = B[li][o]
            row = W[li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        v = [math.tanh(t) for t in out] if li < len(W) - 1 else out
    am, asd = P["a_mu"], P["a_sd"]
    return [v[i] * asd[i] + am[i] for i in range(len(v))], h2

'''


RETIME_FN = '''
_RETIME_CAP = 4       # units/product/turn the head may pull FORWARD
_RETIME_WINDOW = 72   # how far ahead (turns) the head may reach
_RETIME_PRICE_MIN = 1.5   # advance ONLY into strength: price >= this x base.
# Measured 2026-09-05 (ep105627858): the whole d6-12 hole was the opponent
# dumping 48 melons into a $244 scarcity peak while the tape sold 23 on
# schedule. Advancing at ORDINARY prices is how the un-gated retimer read
# 4/56 -- it must fire only where the timing edge exists.

def _policy_head_retime(obs, action, step, state):
    """ADVANCE-ONLY coupling: the head is a TIMING signal, never a volume.

    The bounded BLEND failed 0/56 twice (2026-09-05): our chassis sells in
    bursts at price peaks, the cloned policy sells smoothly, and the blend
    shaved the peaks and smeared the volume (probe: 189 turns stripped, 292
    dribbled, bank 40k vs 97k). This coupling makes that failure impossible
    by construction: when the head wants product sold NOW, the sells are
    PULLED FORWARD from the route's own schedule inside a bounded window --
    total volume is conserved exactly, nothing is ever stripped, and a wrong
    head merely re-times a few units."""
    try:
        vec = _policy_vec(obs)
        if vec is None:
            return action
        want = __WANT__
        route = state.get("route") or _ROUTE
        idx = min(max(0, step), len(route) - 1)
        adv = state.setdefault("p2_adv", {})
        market = list(action.get("market") or [])
        sched = {}
        for o in market:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                sched[o[1]] = sched.get(o[1], 0) + int(o[2] or 0)
        prices = _get(_get(obs, "market", {}) or {}, "prices", {}) or {}
        shed = _get(_get(obs, "private", {}) or {}, "shed", {}) or {}
        for pi, item in enumerate(_POLICY["products"]):
            base = (_MARKET_PARAMS.get(item) or (25,))[0]
            pnow = float(_get(prices, item, 0) or 0)
            if pnow < _RETIME_PRICE_MIN * base:
                continue               # no peak, no retiming
            w = max(0, int(round(want[pi])))
            deficit = min(w - sched.get(item, 0), _RETIME_CAP)
            # ONLY advance what the shed can fill RIGHT NOW: a future sell
            # corresponds to a future harvest, and an advanced sell that
            # partial-fills still strips the full amount from the future
            # step -- destroyed volume wearing a timing-fix face
            cover = int(_get(shed, item, 0) or 0) - sched.get(item, 0)
            deficit = min(deficit, cover)
            if deficit <= 0:
                continue
            take = 0
            j = idx + 1
            end = min(len(route), idx + 1 + _RETIME_WINDOW)
            while j < end and take < deficit:
                for o in (route[j].get("market") or []):
                    if (isinstance(o, list) and len(o) >= 3
                            and o[0] == "SELL" and o[1] == item):
                        avail = int(o[2] or 0) - adv.get((j, item), 0)
                        if avail > 0:
                            t = min(avail, deficit - take)
                            adv[(j, item)] = adv.get((j, item), 0) + t
                            take += t
                    if take >= deficit:
                        break
                j += 1
            if take > 0:
                merged = False
                for o in market:
                    if (isinstance(o, list) and len(o) >= 3
                            and o[0] == "SELL" and o[1] == item):
                        o[2] = int(o[2]) + take
                        merged = True
                        break
                if not merged and len(market) < _MAX_ORDERS:
                    market.append(["SELL", item, take])
        action["market"] = market[:_MAX_ORDERS]
    except Exception:                                          # noqa: BLE001
        pass
    return action

'''

EMIT_PATCH = '''        action = copy.deepcopy(_route[idx])
        # RETIME: honour sells the head already pulled forward -- subtract
        # them from this step's scheduled SELLs so volume stays conserved
        _adv = state.get("p2_adv")
        if _adv:
            _m = action.get("market") or []
            _seen = {}
            for _o in _m:
                if (isinstance(_o, list) and len(_o) >= 3
                        and _o[0] == "SELL"):
                    _k = (idx, _o[1])
                    _left = _adv.get(_k, 0) - _seen.get(_k, 0)
                    if _left > 0:
                        _t = min(_left, int(_o[2] or 0))
                        _o[2] = int(_o[2] or 0) - _t
                        _seen[_k] = _seen.get(_k, 0) + _t
            for _k in _seen:
                _adv.pop(_k, None)
            action["market"] = [
                _o for _o in _m
                if not (isinstance(_o, list) and len(_o) >= 3
                        and _o[0] == "SELL" and int(_o[2] or 0) <= 0)]
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blob")
    ap.add_argument("--base", default=os.path.join(ROOT, "agents",
                                                   "v44.1_bandit.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--couple", choices=["blend", "retime"], default="blend",
                    help="how the head meets the schedule: bounded blend "
                         "(measured 0/56 twice) or advance-only retime")
    a = ap.parse_args()

    blob = open(a.blob, encoding="utf-8").read().strip()
    src = open(a.base, encoding="utf-8").read()
    import base64
    import json
    import zlib
    P0 = json.loads(zlib.decompress(base64.b85decode(blob)).decode("utf-8"))
    is_gru = P0.get("arch") == "gru_hybrid_v2"

    # 1. swap the blob
    old = '_POLICY = json.loads(zlib.decompress(base64.b85decode("c-qS=&B*}(1YZHX")).decode("utf-8"))'
    assert old in src, "empty-blob line not found in base agent"
    src = src.replace(old, '_POLICY = json.loads(zlib.decompress('
                           f'base64.b85decode({blob!r})).decode("utf-8"))')

    # 2. add the recurrent step next to the old forward (GRU blobs only --
    # an MLP blob uses the agent's existing _policy_forward)
    anchor = "def _policy_forward(x):"
    assert anchor in src
    if is_gru:
        src = src.replace(anchor, STEP_FN + "\n" + anchor)

    if a.couple == "blend":
        # 3a. route the head through the recurrent step, hidden per game
        assert is_gru, "blend coupling is wired for the GRU socket only"
        old = "        want = _policy_forward(vec + [1.0])"
        assert old in src
        src = src.replace(old,
                          '        want, state["p2_h"] = _policy_step('
                          'vec + [1.0], state.get("p2_h"))')
    else:
        # 3b. RETIME coupling: new head function, swapped call site, and the
        # route-emission decrement that conserves volume. Arch-aware want:
        # GRU steps the recurrent head; MLP calls _policy_forward directly.
        if is_gru:
            retime = RETIME_FN.replace(
                "want = __WANT__",
                'want, state["p2_h"] = _policy_step('
                'vec + [1.0], state.get("p2_h"))')
        else:
            retime = RETIME_FN.replace(
                "want = __WANT__", "want = _policy_forward(vec + [1.0])")
        src = src.replace(anchor, retime + "\n" + anchor)
        old = "            action = _policy_head(obs, action, step, state)"
        assert old in src, "head call site not found"
        src = src.replace(old, "            action = _policy_head_retime("
                               "obs, action, step, state)")
        old = "        action = copy.deepcopy(_route[idx])\n"
        assert old in src, "route emission not found"
        src = src.replace(old, EMIT_PATCH)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(src)

    # 4. smoke: import + one synthetic turn must not raise
    import importlib.util
    spec = importlib.util.spec_from_file_location("cand", a.out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    P = mod._POLICY
    assert P, "blob failed to decode in the built agent"
    if is_gru:
        want, h = mod._policy_step([0.0] * (len(P["x_mu"])), None)
        assert len(want) == 9 and len(h) == P["hid"]
        want2, h2 = mod._policy_step([0.0] * (len(P["x_mu"])), h)
        print(f"built {a.out}")
        print(f"  blob {len(blob):,} b85 chars; step OK, hidden carries "
              f"({sum(abs(x) for x in h):.3f} -> "
              f"{sum(abs(x) for x in h2):.3f})")
    else:
        want = mod._policy_forward([0.0] * (len(P["x_mu"])))
        assert len(want) == 9
        print(f"built {a.out}")
        print(f"  blob {len(blob):,} b85 chars (MLP head); forward OK")
    print("NEXT: tests/test_agents.py, then counterfactual --rust, "
          "band_panel reactive, latency. NEVER submits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
