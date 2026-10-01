"""Assemble the CONFIG-DRIVEN bandit: a FIXED tape-dispatch harness + a pluggable
guard-rail REGISTRY, all driven by configs/bandit_config.json.

NOTE (2026-09-18): this pure-Python economy is a DEV / MEASUREMENT tool, NOT the
shipped agent and NOT the ship fallback. The SHIPPED bandit economy lives ONLY
in the Rust `kagg mbandit` binary (rustengine/src/mbandit.rs); build_rust_bandit
no longer embeds this file as an "action-faithful" fallback (it was not -- bugs
C1/H3-H7), and now ships a LOUD legal-PASS safety net instead. This harness and
mbandit.rs are DIFFERENT agents; do not treat one as a parity reference for the
other. See docs/history/bug-findings-2026-09-18.md and tests/test_bandit_parity.py.

Architecture is fixed; everything is configurable:
  * checkpoints  -- dispatch points [step, day] (D6/D12/D15/D21/D27).
  * branches     -- generated tape file (swap to any search output; no legacy).
  * guardrails   -- an ORDERED list of reactive rules; each entry {name,on,...params}.
                    Toggle on/off, reorder, add or remove entries to change the
                    reactivity. Rules live in a named REGISTRY in the emitted agent
                    (weed_repair, cash_guard, escalation, endgame, anti_dump,
                    front_run); adding a new rule = add a _r_<name> fn + a config
                    entry.

Daily pipeline: search new tapes -> point config.branches at them -> tweak
guardrail knobs -> this build -> gate -> release.

Usage: python src/kaggriculture/bandit/build/build_bandit_harness.py \
           [--config configs/bandit_config.json] --out AGENT.py
"""
from kaggriculture.paths import ROOT
import argparse, base64, json, os, sys, zlib

# flexon _R37 marginal-price scarcity model (engine-1.32.7 curve), ported as a
# tunable sell rail. Faithful to pub_flexon_v45 _R37_MARKET_PARAMS.
_R37_MARKET_PARAMS = {
    'WHEAT': {'base': 25, 'I0': 10000, 'T': 400, 'below_func': 'sqrt', 'below_target': 0.8, 'above_func': 'log', 'above_target': 0.2},
    'CARROT': {'base': 35, 'I0': 10000, 'T': 450, 'below_func': 'hinge', 'below_target': 1.0, 'above_func': 'sqrt', 'above_target': 0.7},
    'TOMATO': {'base': 60, 'I0': 10000, 'T': 200, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'sqrt', 'above_target': 0.6},
    'STRAWBERRY': {'base': 120, 'I0': 10000, 'T': 100, 'below_func': 'sqrt', 'below_target': 0.7, 'above_func': 'linear', 'above_target': 1.6},
    'MELON': {'base': 250, 'I0': 10000, 'T': 300, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.6},
    'EGG': {'base': 50, 'I0': 10000, 'T': 332, 'below_func': 'hinge', 'below_target': 0.4, 'above_func': 'log', 'above_target': 0.2},
    'MILK': {'base': 160, 'I0': 10000, 'T': 122, 'below_func': 'sqrt', 'below_target': 0.6, 'above_func': 'linear', 'above_target': 1.6},
    'WOOL': {'base': 200, 'I0': 10000, 'T': 105, 'below_func': 'log', 'below_target': 0.2, 'above_func': 'sq', 'above_target': 3.2},
    'FERTILIZER': {'base': 100, 'I0': 10000, 'T': 200, 'below_func': 'linear', 'below_target': 0.4, 'above_func': 'linear', 'above_target': 0.4},
}


def pk(o):
    return base64.b85encode(zlib.compress(
        json.dumps(o, separators=(",", ":")).encode(), 9)).decode()


def _toks(s):
    out = []
    for t in s.split():
        try:
            out.append(int(t))
        except ValueError:
            out.append(t)
    return out


def parse_tape(path):
    """A raw .tape (farmer<TAB>hands<TAB>market, ';'-joined ops) -> route rows.
    Numeric args become ints, matching the generated-branch row schema."""
    rows = []
    for i, ln in enumerate(open(path, encoding="utf-8").read().splitlines()):
        if i == 0 and ln.startswith("SEED"):
            continue
        p = ln.split("\t")
        fs, hs, ms = (p + ["", "", ""])[:3]
        rows.append({
            "farmer": _toks(fs) or ["PASS"],
            "hands": [_toks(h) for h in hs.split(";") if h.strip()],
            "market": [_toks(o) for o in ms.split(";") if o.strip()],
        })
    return rows


TEMPLATE = r'''"""Bandit -- config-driven tape-dispatch harness + pluggable guard-rail registry."""
import base64, json, math, zlib

_BASE = json.loads(zlib.decompress(base64.b85decode("{base}")))
_BR = json.loads(zlib.decompress(base64.b85decode("{branches}")))
_CFG = json.loads(zlib.decompress(base64.b85decode("{cfg}")))
_LUX = ("BUY_ANIMAL", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT")
_S = {{}}


def _num(o, k, d=0.0):
    try:
        v = o.get(k)
        return float(v) if v is not None else d
    except Exception:
        return d


# ---- pluggable guard-rail REGISTRY (each mutates/returns F,H,M) -------------
def _r_weed_repair(F, H, M, c, p):
    tiles = c["tiles"]; fp = c["fpos"]
    if F and F[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE") and fp:
        try:
            t = tiles[fp[1]][fp[0]]
            if isinstance(t, dict) and t.get("kind") == "WEED":
                F = ["DIG"]
        except Exception:
            pass
    return F, H, M


def _r_cash_guard(F, H, M, c, p):
    if c["money"] < p.get("cash_floor", 180.0):
        M = [m for m in M if not (m and m[0] in _LUX)]
    return F, H, M


def _r_escalation(F, H, M, c, p):
    if c["step"] >= p.get("esc_from", 168) and c["step"] % 24 == 0:
        _S["press"] = (c["opp"] - c["money"]) > p.get("esc_gap", 800.0)
    return F, H, M


def _r_endgame(F, H, M, c, p):
    if c["step"] >= p.get("final_dump", 700):
        M = [m for m in M if not (m and m[0] == "SELL")]
        for prod, q in sorted(((k, int(v or 0)) for k, v in c["shed"].items()),
                              key=lambda kv: -kv[1]):
            if q > 0:
                M.insert(0, ["SELL", prod, q])
        _S["endgame"] = True
    return F, H, M


def _r_anti_dump(F, H, M, c, p):
    if _S.get("endgame") or _S.get("press"):
        return F, H, M
    prem = set(p.get("premium", []))
    i0 = p.get("scarcity_i0", 10000)
    M = [m for m in M if not (
        m and m[0] == "SELL" and m[1] in prem
        and int(c["minv"].get(m[1], i0)) >= i0)]
    return F, H, M


def _r_front_run(F, H, M, c, p):
    if _S.get("endgame"):
        return F, H, M
    look = p.get("fr_look_press", 2) if _S.get("press") else p.get("fr_look", 1)
    route = c["route"]; step = c["step"]; slots = 10 - len(M)
    for j in range(1, look + 1):
        nxt = route[step + j] if step + j < len(route) else None
        if not nxt or slots <= 0:
            break
        for m in (nxt.get("market") or []):
            if slots <= 0:
                break
            if m and m[0] == "SELL" and _S["pulled"].get(step + j) != m[1]:
                M.append(list(m)); _S["pulled"][step + j] = m[1]; slots -= 1
    return F, H, M


# ---- _r37 marginal-price scarcity sell rail (ported from pub_flexon_v45) -----
_R37P = json.loads('{r37p}')
_R37_HG = 8.0
_R37_FLOOR = 1
_R37_ANIMAL = {{"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}}


def _r37_shape(func, x, T=None):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq": return x * x
    if func == "sqrt": return math.sqrt(x)
    if func == "log": return math.log(1.0 + x)
    if func == "log10": return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0: return x
        u = x / T
        return u + _R37_HG * max(0.0, u - 1.0) ** 2
    return x


def _r37_price(item, inventory):
    pp = _R37P[item]
    base = pp["base"]; I0 = pp["I0"]; T = pp["T"]
    if inventory < I0:
        f = pp["below_func"]
        amp = pp["below_target"] * base / _r37_shape(f, T, T)
        price = base + amp * _r37_shape(f, I0 - inventory, T)
    else:
        f = pp["above_func"]
        amp = pp["above_target"] * base / _r37_shape(f, T, T)
        price = base - amp * _r37_shape(f, inventory - I0, T)
    return max(_R37_FLOOR, int(round(price)))


def _r37_prio(minv, rtiles, item, qty):
    if not qty or item not in _R37P: return 0.0
    inv = int(minv.get(item, _R37P[item]["I0"]))
    crop = item if item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON") else None
    animal = _R37_ANIMAL.get(item)
    standing = 0
    for row in rtiles:
        for t in row:
            if isinstance(t, dict) and ((crop and t.get("crop") == crop) or (animal and t.get("animal") == animal)):
                standing += max(0, int(t.get("yield_units", 0) or 0))
    batch = min(24, max(8, standing))
    now = sum(_r37_price(item, inv + j) for j in range(qty))
    later = sum(_r37_price(item, inv + batch + j) for j in range(qty))
    return now - later


def _r_scarcity_sell(F, H, M, c, p):
    """Marginal-price sell discipline: (1) HOLD units whose current marginal
    price < hold_frac*base (skip dumping into a floored market); (2) reorder
    contiguous distinct-item SELL blocks scarcest-value first. hold_frac=0 ==
    faithful flexon reorder-only."""
    if _S.get("endgame"): return F, H, M
    minv = c["minv"]; rtiles = c["rtiles"]; shed = c["shed"]
    hold = p.get("hold_frac", 0.0)
    if hold > 0:
        kept = []
        for m in M:
            if m and m[0] == "SELL" and m[1] in _R37P:
                inv = int(minv.get(m[1], _R37P[m[1]]["I0"]))
                if _r37_price(m[1], inv) < hold * _R37P[m[1]]["base"]:
                    continue
            kept.append(m)
        M = kept
    orders = [list(o) for o in M]; start = 0
    while start < len(orders):
        if orders[start][0] != "SELL": start += 1; continue
        end = start
        while end < len(orders) and orders[end][0] == "SELL": end += 1
        block = orders[start:end]
        if len(set(o[1] for o in block)) == len(block):
            orders[start:end] = sorted(block, key=lambda o: _r37_prio(
                minv, rtiles, o[1], min(int(o[2]), int(shed.get(o[1], 0) or 0))), reverse=True)
        start = end
    return F, H, orders


_RAILS = {{"weed_repair": _r_weed_repair, "cash_guard": _r_cash_guard,
          "escalation": _r_escalation, "endgame": _r_endgame,
          "anti_dump": _r_anti_dump, "front_run": _r_front_run,
          "scarcity_sell": _r_scarcity_sell}}


def agent(obs):
    try:
        return _run(obs)
    except Exception:
        return {{"farmer": ["PASS"], "hands": [], "market": []}}


def _run(obs):
    step = int((obs.get("step") or 0))
    me = int((obs.get("player") or 0))
    if step == 0 or _S.get("route") is None or step < _S.get("last", -1):
        _S.update({{"route": list(_BASE), "done": [], "pulled": {{}},
                   "press": False, "endgame": False}})
    _S["last"] = step
    shops = ((obs.get("town") or {{}}).get("unlocked_shops") or [])
    for (cstep, day) in _CFG["checkpoints"]:
        if cstep <= step <= cstep + 2 and day not in _S["done"] and len(shops) >= 2:
            _S["done"].append(day)
            br = _BR.get(str(day), {{}}).get("|".join(sorted(shops[:8])))
            if br and br.get("cont"):
                _S["route"] = _S["route"][:cstep] + br["cont"]
    route = _S["route"]
    act = route[step] if step < len(route) else {{"farmer": ["PASS"], "hands": [], "market": []}}
    F = list(act.get("farmer") or ["PASS"])
    H = [list(h) for h in (act.get("hands") or [])]
    M = [list(m) for m in (act.get("market") or []) if m]

    farms = obs.get("farms") or [{{}}, {{}}]
    fm = farms[me] if me < len(farms) else {{}}
    c = {{"step": step, "route": route, "tiles": fm.get("tiles") or [],
         "fpos": fm.get("farmer"), "shed": ((obs.get("private") or {{}}).get("shed") or {{}}),
         "minv": ((obs.get("market") or {{}}).get("inventory") or {{}}),
         "rtiles": ((farms[1 - me] if (1 - me) < len(farms) else {{}}).get("tiles") or []),
         "money": _num(fm, "money"),
         "opp": _num(farms[1 - me] if (1 - me) < len(farms) else {{}}, "money")}}
    for gr in _CFG["guardrails"]:
        if gr.get("on") and gr["name"] in _RAILS:
            F, H, M = _RAILS[gr["name"]](F, H, M, c, gr)
    return {{"farmer": F, "hands": H, "market": M[:10]}}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "bandit_config.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-tape", default=None,
                    help="hot-swap the base opening: a raw .tape used as _BASE "
                         "instead of the our_route scaffold (config-driven base).")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    # D6 dispatch is auto-added UNLESS cfg sets d6_optional (sweep can drop it).
    cps = [tuple(c) for c in cfg["checkpoints"]]
    if not cfg.get("d6_optional") and not any(d == 6 for _, d in cps):
        cps = [(146, 6)] + cps
    cfg["checkpoints"] = sorted({(s, d) for (s, d) in cps}, key=lambda c: c[0])
    cfg["checkpoints"] = [list(c) for c in cfg["checkpoints"]]
    # ENDGAME liquidation is MANDATORY; enforce it on.
    grs = cfg.get("guardrails", [])
    if not any(g.get("name") == "endgame" for g in grs):
        grs.append({"name": "endgame", "on": True, "final_dump": 700})
    for g in grs:
        if g.get("name") == "endgame":
            g["on"] = True
    cfg["guardrails"] = grs
    brp = cfg["branches"]
    brp = brp if os.path.isabs(brp) else os.path.join(ROOT, brp)
    br = json.load(open(brp, encoding="utf-8")) if os.path.exists(brp) else {}
    # base MUST come from --base-tape or config["base_tape"]; the our_route()/v45
    # fallback is removed (it silently shipped the wrong economy as v58).
    base_tape = a.base_tape or cfg.get("base_tape")
    if not base_tape:
        raise SystemExit("no base tape: set 'base_tape' in the config (or pass "
                         "--base-tape). The our_route()/v45 fallback is removed.")
    bt = base_tape if os.path.isabs(base_tape) else os.path.join(ROOT, base_tape)
    if not os.path.exists(bt):
        raise SystemExit(f"base_tape not found: {bt}")
    base = parse_tape(bt)[:719]
    n = sum(len(v) for v in br.values())
    embed = {"checkpoints": cfg["checkpoints"], "guardrails": cfg["guardrails"]}
    src = TEMPLATE.format(base=pk(base), branches=pk(br), cfg=pk(embed),
                          r37p=json.dumps(_R37_MARKET_PARAMS))
    open(a.out, "w", encoding="utf-8", newline="\n").write(src)
    compile(src, a.out, "exec")
    rails = [g["name"] for g in cfg["guardrails"] if g.get("on")]
    print(f"built {a.out} ({len(src):,} bytes): dispatch @ "
          f"{[c[1] for c in cfg['checkpoints']]}; rails ON = {rails}; {n} branches",
          flush=True)


if __name__ == "__main__":
    main()
