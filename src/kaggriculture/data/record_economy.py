"""Record a (dense) agent's economy as a bandit-format tape.

The frontier winners are dense tape economies (buy all land -> +crops -> win the
endgame) that our v56y tape lacks. `sell_search`/`factory` mutators cannot build
density (they add HIRE + resize buys, never BUY_LAND or plantings). So capture a
winning agent's action stream directly: play it on the serve engine and serialise
each step to our tape format "farmer\thands\tmarket". The result can be installed
as a bandit economy (like v56y/v52) and dispatched in the crop-demand worlds where
our economy loses the endgame -- then gated to see if it transfers (foreign tapes
underperform open-loop, memory early-splice; the crop-world dispatch + our reactive
sell shell may recover it).

Usage: python src/record_economy.py AGENT.py --opp OPP.py --seed 1001 --out X.tape
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, os, sys

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("a" + os.path.basename(p), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    finally:
        sys.stdout = old


def op_str(op):
    return " ".join(str(t) for t in op)


def row(action):
    """Serialise an action dict to a bandit tape row 'farmer\\thands\\tmarket'."""
    f = action.get("farmer") or ["PASS"]
    farmer = op_str(f if isinstance(f, list) and f and isinstance(f[0], str) else (f[0] if f else ["PASS"]))
    # farmer can be ["PASS"] (single op) -- op_str handles a flat list of tokens
    if isinstance(f, list) and f and isinstance(f[0], list):
        farmer = op_str(f[0])
    hands = ";".join(op_str(h) for h in (action.get("hands") or []))
    market = ";".join(op_str(o) for o in (action.get("market") or []))
    return f"{farmer}\t{hands}\t{market}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--opp", default=os.path.join(ROOT, ".local/panel_full/aurax7_kaggriculture_shop_router_reactive_v5.py"))
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ag = load(a.agent); opp = load(a.opp)
    E = ServeEnv(); obs = E.reset(a.seed)
    rows = []; world = None
    while not obs.get("done"):
        cv = seat_view(obs, a.seat); ov = seat_view(obs, 1 - a.seat)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = f"{sh[0]}|{sh[1]}"
        act = ag(dict(cv)) or {"farmer": ["PASS"], "hands": [], "market": []}
        rows.append(row(act))
        oa = opp(dict(ov))
        obs = E.step_both(act if a.seat == 0 else oa, oa if a.seat == 0 else act)
    E.close()
    fm = obs["farms"]
    us, them = fm[a.seat]["money"], fm[1 - a.seat]["money"]
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"recorded {len(rows)} rows  world={world}  bank {us:.0f} vs {them:.0f} "
          f"({'WIN' if us>them else 'LOSS'}) -> {a.out}")


if __name__ == "__main__":
    main()
