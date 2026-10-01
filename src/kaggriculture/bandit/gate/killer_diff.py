"""Economy diff: OUR bandit vs a killer (tschinkel), full per-turn actions+banks.

Answers "where does the $N go" concretely: plays a real game, then breaks down
each agent's economy -- SELL revenue by product, production/purchase action counts,
and the bank trajectory (when the gap opens). This is the diagnostic that routes
the strategy decision (better route economy vs reactive policy).

Run: python -m kaggriculture.bandit.gate.killer_diff --seed 1175620406
"""
import argparse, os
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
import kaggriculture.measure.eval_harness as EH

BASE = {"WHEAT": 20, "CARROT": 35, "TOMATO": 50, "STRAWBERRY": 60, "MELON": 90,
        "EGG": 40, "MILK": 130, "WOOL": 200, "FERTILIZER": 25}  # ~base prices for revenue est


def _act(cell):
    return (cell or {}).get("action") or {}

def _money(pair, seat):
    o = ((pair[0] or {}).get("observation")) or {}
    farms = o.get("farms") or []
    return float((farms[seat] or {}).get("money") or 0.0) if len(farms) > seat else 0.0


def summarize(steps, seat, name):
    sells = defaultdict(int); buys = defaultdict(int); hands = defaultdict(int); plants = defaultdict(int)
    realized = defaultdict(float)   # price-at-sale x qty (actual cash, approx)
    for t in range(len(steps)):
        a = _act(steps[t][seat])
        obs = (steps[t][seat] or {}).get("observation") or {}
        prices = ((obs.get("market") or {}).get("prices")) or {}
        for o in (a.get("market") or []):
            if o and o[0] == "SELL" and len(o) >= 3:
                try:
                    q = int(o[2]); sells[o[1]] += q
                    realized[o[1]] += q * float(prices.get(o[1], BASE.get(o[1], 0)) or 0)
                except ValueError: pass
            elif o and o[0] == "BUY" and len(o) >= 2:
                buys[o[1] if len(o) > 1 else "?"] += 1
        for o in (a.get("hands") or []):
            if not o: continue
            op = o[0]; hands[op] += 1
            if op == "PLANT" and len(o) >= 2: plants[o[1]] += 1
    return sells, buys, hands, plants, dict(realized)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1175620406)
    a = ap.parse_args()
    ours = H.build_agent("diff_base", H.cfg_with([]))       # our shipped route economy
    tsch = EH._as_agent(os.path.join(ROOT, "agents", "pub_tschinkel_2945.py"))
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": a.seed}, debug=False)
    env.run([EH._as_agent(ours), tsch])
    steps = env.steps
    last = steps[-1]
    b0 = float(last[0]["reward"] or 0); b1 = float(last[1]["reward"] or 0)
    print(f"seed {a.seed}: OURS bank={b0:.0f}  tschinkel bank={b1:.0f}  gap={b0-b1:+.0f}\n", flush=True)

    fin = {0: b0, 1: b1}
    for seat, nm in ((0, "OURS"), (1, "tschinkel")):
        sells, buys, hands, plants, realized = summarize(steps, seat, nm)
        tot_units = sum(sells.values()); tot_real = sum(realized.values())
        ppu = tot_real / tot_units if tot_units else 0
        spend = 3000 + tot_real - fin[seat]     # bank = 3000 + revenue - spend
        print(f"=== {nm} ===  final_bank=${fin[seat]:.0f}", flush=True)
        print(f"  SELL units={tot_units}  REALIZED_REV=${tot_real:.0f}  price/unit=${ppu:.2f}  derived_spend=${spend:.0f}", flush=True)
        print("  realized by product: " + "  ".join(
            f"{p}={sells[p]}u/${realized.get(p,0):.0f}" for p in sorted(sells, key=lambda p: -realized.get(p, 0))), flush=True)
        print(f"  PLANT: " + " ".join(f"{p}={plants[p]}" for p in sorted(plants, key=lambda p: -plants[p]))
              + f"   COOP={hands.get('BUILD_COOP',0)} PASTURE={hands.get('BUILD_PASTURE',0)}", flush=True)
        print("", flush=True)

    # money trajectory: when does the gap open?
    print("=== money gap over time (ours - tschinkel) ===", flush=True)
    for t in range(0, len(steps), 96):
        m0 = _money(steps[t], 0); m1 = _money(steps[t], 1)
        print(f"  step {t:3d} (day {t//24:2d}): ours={m0:8.0f}  tsch={m1:8.0f}  gap={m0-m1:+8.0f}", flush=True)


if __name__ == "__main__":
    main()
