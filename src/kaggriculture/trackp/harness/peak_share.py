"""HOLD-SHAPE screening: how much of a base's sell revenue lands on peaks?

Measured 2026-09-05 (ep105627858): the d6-12 collapse hole was the opponent
dumping 48 melons into a $244 scarcity peak while our tape sold 23 on
schedule -- identical spend, revenue 9.5k vs 22.6k. Guarded score alone
cannot see this trait; this screens each candidate base for PEAK-REVENUE
SHARE = (sell qty x price, where price >= 1.5x base) / total sell revenue,
measured by replaying the tape against real certified hard-band opponents
on the Rust engine and reading the serve market each turn.

    python .local/hardband/peak_share.py [--cells 12]
"""
from kaggriculture.paths import ROOT
import argparse, json, os, sys
HB = os.path.join(ROOT, ".local", "hardband")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import kaggriculture.engine.serve_match as SM
from kaggriculture.trackp import routes_io as R

BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
        "MELON": 80, "EGG": 40, "MILK": 90, "WOOL": 110, "FERTILIZER": 50}
PEAK = 1.5

def tape_agent(acts):
    def agent(obs):
        i = min(int(obs.get("step") or 0), len(acts) - 1)
        return acts[i]
    return agent

def peak_share(acts, opp_acts, seed, our_seat):
    """One cell: (peak_revenue, total_revenue, bank)."""
    srv = SM.Serve()
    try:
        js = srv.cmd(f"RESET {int(seed)}")
        peak = tot = 0.0
        while not js.get("done") and int(js.get("step") or 0) < 719:
            i = int(js.get("step") or 0)
            prices = (js.get("market") or {}).get("prices") or {}
            a = acts[min(i, len(acts) - 1)]
            for o in (a.get("market") or []):
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    p = float(prices.get(o[1], 0) or 0)
                    rev = p * float(o[2] or 0)
                    tot += rev
                    if p >= PEAK * BASE.get(o[1], 25):
                        peak += rev
            b_ = opp_acts[min(i, len(opp_acts) - 1)]
            mine = SM.action_to_line(a)
            theirs = SM.action_to_line(b_)
            la, lb = (mine, theirs) if our_seat == 0 else (theirs, mine)
            js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        bank = float(js["farms"][our_seat].get("money") or 0)
        return peak, tot, bank
    finally:
        srv.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=12)
    a = ap.parse_args()
    man = json.load(open(os.path.join(HB, "factory_panel", "manifest.json"),
                         encoding="utf-8"))
    opps = man["opponents"][:a.cells] if "opponents" in man else None
    if opps is None:
        # fall back: derive from files
        opps = []
        for f in sorted(os.listdir(os.path.join(HB, "factory_panel"))):
            if f.startswith("opp_") and f.endswith(".tape"):
                opps.append(f)
        opps = opps[:a.cells]
        print("manifest lacks 'opponents'; using file listing", file=sys.stderr)
    # candidate bases: this morning's re-screen set + incumbent
    cands = ["105370586_s0", "105367892_s0", "105021852_s0",
             "105017780_s1", "105023582_s1"]
    import glob as _g
    scr = os.path.join(HB, "econ_candidates.json")
    if os.path.exists(scr):
        cands += [c for c in json.load(open(scr, encoding="utf-8"))
                  if c not in cands]
    print(f"{'base':<18}{'peak-share':>11}{'peak-rev':>12}{'mean bank':>11}"
          f"  ({len(opps)} cells)")
    for rid in cands:
        try:
            acts = R.load_route(rid)
        except Exception as e:
            print(f"{rid:<18} load failed: {e}")
            continue
        idx = json.load(open(os.path.join(HB, "index.json"),
                             encoding="utf-8"))
        pk = tt = bk = 0.0
        n = 0
        for o in opps:
            ep = str(o["episode"])
            g = idx.get(ep) or {}
            our_seat = int(g.get("our_seat", 0))
            try:
                # opponent actions come from the RECORDED REPLAY (the tape
                # files were compiled from it; the route index has no row)
                rep = json.load(open(os.path.join(HB, "replays",
                                                  f"{ep}.json"),
                                     encoding="utf-8"))
                opp_acts = [((st[1 - our_seat] or {}).get("action") or {})
                            for st in rep.get("steps") or [] if st]
                p, t, b = peak_share(acts, opp_acts, o["seed"], our_seat)
            except Exception as e:
                print(f"  cell {ep}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            pk += p; tt += t; bk += b; n += 1
        if n:
            print(f"{rid:<18}{pk / max(tt, 1):>10.1%}{pk / n:>12,.0f}"
                  f"{bk / n:>11,.0f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
