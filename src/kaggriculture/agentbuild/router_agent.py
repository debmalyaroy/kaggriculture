"""Build a ShopForge-style match-history ROUTER agent.

The architecture of the published #3 agent (yhay81, rating 2929): a base
history plays until the first shop reveal (step 72), a per-first-shop
specialist takes over, and a per-shop-PAIR specialist takes over at the
second reveal (step 144). Candidates are complete public histories from
ANY strong team; each specialist earns its slot by STAGED PLAYED GAMES
(src/router_select.py), never by vote popularity — that selection
discipline is the whole difference between 2929 and our 0.708 branches.

    python src/router_agent.py --config models/router/router_config.json --out .local/candidates/router_v1.py

Config: {"base": route_id, "first": {shop: route_id}, "pair": {"A|B": route_id}}
"""
import argparse
import base64
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = '''"""Kaggriculture match-history router -- {label}
Base + per-shop + per-shop-pair public histories, routed on the observed
draws (steps 72/144). Built {built} by src/router_agent.py.
Provenance: {prov}"""
import base64
import json
import zlib


def _unz(s):
    return json.loads(zlib.decompress(base64.b85decode(s)).decode("utf-8"))


_BASE = _unz("{base_z}")
_FIRST = _unz("{first_z}")
_PAIR = _unz("{pair_z}")


def _get(o, k, d=None):
    try:
        v = o[k]
        return d if v is None else v
    except (KeyError, TypeError, IndexError):
        return getattr(o, k, d)


def agent(obs, config=None):
    try:
        step = int(_get(obs, "step", 0) or 0)
        town = _get(obs, "town", {{}}) or {{}}
        shops = list(_get(town, "unlocked_shops", []) or [])
        actions = _BASE
        if len(shops) >= 2:
            actions = _PAIR.get(shops[0] + "|" + shops[1]) \\
                or _FIRST.get(shops[0]) or _BASE
        elif len(shops) >= 1:
            actions = _FIRST.get(shops[0]) or _BASE
        if step >= len(actions):
            return {{"farmer": ["PASS"], "hands": [], "market": []}}
        a = actions[max(0, step)]
        if not isinstance(a, dict):
            return {{"farmer": ["PASS"], "hands": [], "market": []}}
        me = int(_get(obs, "player", 0) or 0)
        farms = _get(obs, "farms", []) or []
        farm = farms[me] if me < len(farms) else {{}}
        real_hands = _get(farm, "hands", []) or []
        hands = list(a.get("hands") or [])
        if len(hands) < len(real_hands):
            hands = hands + [["PASS"]] * (len(real_hands) - len(hands))
        elif len(hands) > len(real_hands):
            hands = hands[:len(real_hands)]
        return {{"farmer": a.get("farmer") or ["PASS"], "hands": hands,
                 "market": a.get("market") or []}}
    except Exception:
        return {{"farmer": ["PASS"], "hands": [], "market": []}}
'''


def z(o):
    return base64.b85encode(zlib.compress(
        json.dumps(o, separators=(",", ":")).encode("utf-8"), 9)).decode()


def build(config, out):
    import kaggriculture.data.routes as R
    base = R.load_route(config["base"])
    first = {s: R.load_route(rid)
             for s, rid in (config.get("first") or {}).items()}
    pair = {p: R.load_route(rid)
            for p, rid in (config.get("pair") or {}).items()}
    for name, tape in [("base", base)] + list(first.items()) + list(pair.items()):
        if len(tape) != 720:
            raise SystemExit(f"tape {name}: {len(tape)} steps, need 720")
        worst = max(len(a.get("market") or []) for a in tape if isinstance(a, dict))
        if worst > 10:
            raise SystemExit(f"tape {name}: {worst} market orders in one turn "
                             f"(engine cap 10 — extras drop silently)")
    prov = {"base": config["base"], "first": config.get("first") or {},
            "pair": config.get("pair") or {}}
    src = TEMPLATE.format(label=os.path.basename(out),
                          built=__import__("datetime").date.today().isoformat(),
                          prov=json.dumps(prov)[:600],
                          base_z=z(base), first_z=z(first), pair_z=z(pair))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    print(f"wrote {out} ({os.path.getsize(out):,} bytes; "
          f"{len(first)} first-shop, {len(pair)} pair specialists)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(json.load(open(args.config, encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
