"""Build a per-world fork library from recorded strong reactive agents.

For each source agent x seed: play it on ServeEnv (vs a chosen opp), record its
action tape, note the realized D6 world-key (sorted unlocked shops), and store the
D6 continuation (rows 146..718) bucketed by world. Emits branches_<tag>.json in the
harness format {"6": {world_key: {"cont": rows, "src": name, "bank": b}}}, keeping
the highest-banking continuation per world.

Usage: python src/build_fork_library.py --seeds 1001-1030 --out models/trackp/baseline/branches_v58.json
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, json, os, sys

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
PANEL = os.path.join(ROOT, ".local", "panel_full")
CP = 146  # D6 splice step


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    except Exception:
        return None
    finally:
        sys.stdout = old


def toks(s):
    out = []
    for t in str(s).split():
        try: out.append(int(t))
        except ValueError: out.append(t)
    return out


def row(action):
    f = action.get("farmer") or ["PASS"]
    farmer = f[0] if (f and isinstance(f[0], list)) else f
    return {"farmer": [x if isinstance(x, (int,)) else (int(x) if str(x).lstrip('-').isdigit() else x) for x in (farmer or ["PASS"])],
            "hands": [[x for x in h] for h in (action.get("hands") or [])],
            "market": [[x for x in o] for o in (action.get("market") or [])]}


def play(ag, opp, seed):
    E = ServeEnv(); obs = E.reset(seed); rows = []; world_at_d6 = None
    while not obs.get("done"):
        cv = seat_view(obs, 0); ov = seat_view(obs, 1)
        step = int(obs.get("step") or 0)
        if step == CP and world_at_d6 is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2:
                world_at_d6 = "|".join(sorted(sh[:8]))
        act = ag(dict(cv)) or {"farmer": ["PASS"], "hands": [], "market": []}
        rows.append(row(act))
        obs = E.step_both(act, opp(dict(ov)))
    E.close()
    fm = obs["farms"]
    return rows, world_at_d6, fm[0]["money"], fm[1]["money"]


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1001-1030")
    ap.add_argument("--src", default="aurax7_kaggriculture_shop_router_reactive_v5")
    ap.add_argument("--opp", default="aurax7_kaggriculture_shop_router_reactive_v5")
    ap.add_argument("--out", default=os.path.join(ROOT, "models/trackp/baseline/branches_v58.json"))
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    srcs = a.src.split(",")
    opp_path = a.opp if a.opp.endswith(".py") else os.path.join(PANEL, a.opp + ".py")
    opp_path = opp_path if os.path.isabs(opp_path) else os.path.join(ROOT, opp_path)
    opp = load(opp_path)
    best = {}  # world -> {"cont", "src", "bank"}
    for name in srcs:
        ag = load(os.path.join(PANEL, name + ".py"))
        if not ag:
            print("skip (load fail):", name); continue
        for s in seeds:
            rows, world, us, th = play(ag, opp, s)
            if not world or len(rows) < 700:
                continue
            cont = rows[CP:719]
            cur = best.get(world)
            if cur is None or us > cur["bank"]:
                best[world] = {"cont": cont, "src": name, "bank": us, "seed": s}
    branches = {"6": {w: {"cont": v["cont"]} for w, v in best.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(branches, open(a.out, "w", encoding="utf-8"))
    print(f"\n{len(best)} worlds covered -> {a.out}")
    for w, v in sorted(best.items(), key=lambda kv: -kv[1]["bank"]):
        print(f"  {v['bank']:8.0f}  seed{v['seed']:<5} {v['src'][:22]:22}  {w}")


if __name__ == "__main__":
    main()
