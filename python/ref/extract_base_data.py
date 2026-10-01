"""Extract the chassis data of a layered Python agent into plain config files for the Rust base.

    python python/ref/extract_base_data.py <agent.py> <base_id>

Writes configs/bases/<base_id>/: routes.json (route id -> list of per-step actions),
router.json (shop-pair tables, settings, opening override, route switch steps), and
SOURCE.md (agent path + sha). Works by executing the agent module once in a sandbox namespace
and reading the module-level tables the router uses (_ROUTES, _R108_SHOP_ROUTES,
_R110_OLD_SHOPS, _V92_TABLE, _SETTINGS, _R42_OPENING).
"""
import contextlib, hashlib, io, json, os, sys

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    path, base_id = sys.argv[1], sys.argv[2]
    src = open(path, encoding="utf-8").read()
    ns = {"__name__": "base_extract"}
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, "exec"), ns)
    out = os.path.join(RL, "configs", "bases", base_id)
    os.makedirs(out, exist_ok=True)
    routes = {str(k): v for k, v in ns["_IMPL"].chassis.routes.items()}
    json.dump(routes, open(os.path.join(out, "routes.json"), "w"), separators=(",", ":"))

    def keyed(d):
        return {"|".join(k): v for k, v in d.items()} if d else {}
    router = {
        "shop_routes_new": keyed(ns.get("_R108_SHOP_ROUTES")),
        "shop_routes_old": keyed(ns.get("_R110_OLD_SHOPS")),
        "shop_routes_v92": keyed(ns.get("_V92_TABLE")),
        "yarn_uses_old": True,
        "default_new": 100, "default_old": 0,
        "select_step": 144, "endgame_step": 648, "endgame_route": 2,
        "opening_step0_market": ns.get("_R42_OPENING"),
        "settings": ns.get("_SETTINGS"),
    }
    json.dump(router, open(os.path.join(out, "router.json"), "w"), indent=1)
    sha = hashlib.sha256(src.encode()).hexdigest()
    open(os.path.join(out, "SOURCE.md"), "w").write(
        f"# {base_id}\n\nExtracted from `{os.path.relpath(path, os.path.dirname(RL))}` (sha256 {sha}).\n"
        f"routes: {len(routes)} ids, steps per route: {sorted({len(v) for v in routes.values()})}\n")
    print(base_id, "routes", len(routes), "router keys", list(router), "->", out)


if __name__ == "__main__":
    main()
