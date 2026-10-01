"""Per-product realized SELL price, us vs opp -- reproduce the dump gap offline.

Plays A (us) vs B (opp) on ServeEnv. At every step, for each SELL order, records
the market price[prod] the seat saw and the units. Reports per-product
units-weighted average sell price per seat, plus the market inventory at sell time
(low inventory = scarcity = high price). Shows WHICH products we dump into low
prices vs where the opponent captures scarcity.

Usage: python src/analyze_sell_prices.py --a harness_base.py --opp OPP.py --seeds 2001-2008
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, os, sys
from collections import defaultdict

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
PANEL = os.path.join(ROOT, ".local", "panel_full")


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


def resolve(name):
    if name.endswith(".py"):
        return name if os.path.isabs(name) else os.path.join(ROOT, ".local/livepool", name) \
            if os.path.exists(os.path.join(ROOT, ".local/livepool", name)) else os.path.join(ROOT, name)
    return os.path.join(PANEL, name + ".py")


def seeds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def record_sells(cv, acc):
    """acc[prod] = [sum price*units, sum units, sum inv*units] from this seat's SELLs."""
    mk = cv.get("market") or {}
    prices = mk.get("prices") or {}; inv = mk.get("inventory") or {}
    shed = ((cv.get("private") or {}).get("shed") or {})
    for m in (cv.get("_pending_sells") or []):
        pass
    return prices, inv, shed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="harness_base.py")
    ap.add_argument("--opp", default="ahmedberatozer_kaggriculture_v41_review_candidate")
    ap.add_argument("--seeds", default="2001-2008")
    a = ap.parse_args()
    A = load(resolve(a.a)); B = load(resolve(a.opp))
    accs = [defaultdict(lambda: [0.0, 0.0, 0.0]) for _ in range(2)]  # per seat
    wins = 0; ng = 0
    for s in seeds_of(a.seeds):
        E = ServeEnv(); obs = E.reset(s)
        while not obs.get("done"):
            cvs = [seat_view(obs, 0), seat_view(obs, 1)]
            acts = [A(dict(cvs[0])) or {}, B(dict(cvs[1])) or {}]
            for seat in range(2):
                mk = cvs[seat].get("market") or {}
                prices = mk.get("prices") or {}; inv = mk.get("inventory") or {}
                shed = ((cvs[seat].get("private") or {}).get("shed") or {})
                for m in (acts[seat].get("market") or []):
                    if m and m[0] == "SELL":
                        prod = m[1]; want = int(m[2]) if len(m) > 2 else 0
                        units = min(want, int(shed.get(prod, 0) or 0)) if want else 0
                        if units <= 0:
                            units = min(want or 0, 1)  # count as attempt
                        pr = float(prices.get(prod, 0) or 0); iv = float(inv.get(prod, 0) or 0)
                        accs[seat][prod][0] += pr * max(units, 0)
                        accs[seat][prod][1] += max(units, 0)
                        accs[seat][prod][2] += iv * max(units, 0)
            obs = E.step_both(acts[0], acts[1])
        fm = obs["farms"]; wins += fm[0]["money"] > fm[1]["money"]; ng += 1
        E.close()
    print(f"A={a.a}  vs  {a.opp}   seeds={a.seeds}   A wins {wins}/{ng}\n")
    for seat, tag in [(0, "US (A)"), (1, "OPP")]:
        acc = accs[seat]
        tot_u = sum(v[1] for v in acc.values()); tot_pv = sum(v[0] for v in acc.values())
        print(f"{tag}: overall units-wtd sell price = {tot_pv/max(1,tot_u):.1f}   (units {tot_u:.0f})")
        print(f"  {'product':11} {'avgprice':>8} {'units':>8} {'avg_inv@sell':>12}")
        for p in sorted(acc, key=lambda k: -acc[k][1]):
            pv, u, iv = acc[p]
            if u <= 0: continue
            print(f"  {p:11} {pv/u:8.1f} {u:8.0f} {iv/u:12.0f}")
        print()


if __name__ == "__main__":
    main()
