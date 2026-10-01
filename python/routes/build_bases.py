"""Route library v2, step A2: candidate bases from mined route-0-compatible tapes.

    python python/routes/build_bases.py [--cands data/routes/cands_stage1.json]

Each candidate (python/routes/ mining: the player's farmer + hands equal route 0's on steps 1..143, opening purchases
equal, a win by a team rated >= 2500, engine 1.32.7) becomes a route: the player's own 719-step action list, id
1000 + i. Two bases (both = the v61.1 library + candidates, same router):
  configs/bases/v61.1x   candidates WITHOUT late land / tomato plantings (V219's routes_block scans the WHOLE library
                         for those after step 432; a single such route would switch V219 off in every game)
  configs/bases/v61.1xl  the candidates WITH late land / tomato (screened with that side effect included)
Sync check: every candidate's farmer + hands on steps 1..143 byte-equal to route 0 (re-asserted here).
Writes data/routes/cands_stage2.json (candidate -> base, route id, world).
"""
import argparse
import json
import os
import shutil

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
canon = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"))  # noqa: E731


def units(a):
    return canon([a.get("farmer") or ["PASS"], a.get("hands") or []])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default=os.path.join(RL, "data", "routes", "cands_stage1.json"))
    ap.add_argument("--id0", type=int, default=1000)
    ap.add_argument("--name", default="v61.1x", help="base for the candidates without late land / tomato (the late ones go to <name>l)")
    ap.add_argument("--stage-out", default=os.path.join(RL, "data", "routes", "cands_stage2.json"))
    a = ap.parse_args()
    base = os.path.join(RL, "configs", "bases", "v61.1")
    routes = json.load(open(os.path.join(base, "routes.json")))
    r0 = routes["0"]
    cands = json.load(open(a.cands))
    out = {a.name: dict(routes), a.name + "l": dict(routes)}
    rows = []
    for i, c in enumerate(cands):
        tp = json.load(open(os.path.join(RL, c["file"].replace("\\", "/"))))
        s = tp["seat"]
        acts = [(p[s] if len(p) > s else None) or {"farmer": ["PASS"], "hands": [], "market": []} for p in tp["actions"]]
        acts = [{"farmer": x.get("farmer") or ["PASS"], "hands": x.get("hands") or [], "market": [m for m in (x.get("market") or [])]} for x in acts]
        acts = (acts + [{"farmer": ["PASS"], "hands": [], "market": []}] * 719)[:719]
        assert all(units(acts[t]) == units(r0[t]) for t in range(1, 144)), c["key"]
        rid = a.id0 + i
        b = a.name + "l" if c.get("late") else a.name
        out[b][str(rid)] = acts
        rows.append({**c, "route": rid, "base": b})
    for name, rt in out.items():
        if len(rt) == len(routes):
            continue
        d = os.path.join(RL, "configs", "bases", name)
        os.makedirs(d, exist_ok=True)
        json.dump(rt, open(os.path.join(d, "routes.json"), "w"), separators=(",", ":"))
        shutil.copy(os.path.join(base, "router.json"), os.path.join(d, "router.json"))
        print(f"[bases] {name}: {len(rt)} routes ({len(rt) - len(routes)} candidates)")
    json.dump(rows, open(a.stage_out, "w"), indent=1)


if __name__ == "__main__":
    main()
