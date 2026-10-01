"""Audit the reactive sell TRIGGERS against our own inventory + schedule.

Operator observation (2026-09-20): the sell side does not understand our own
schedule before firing -- e.g. it holds MILK (glutted) and fires a SELL for WOOL,
so wool undersells and the held milk still dumps later. This captures a real
game and, for every step we emit a SELL, records: the product, OUR shed of it,
its price vs base, and OUR shed of the OTHER premium products at that moment --
so a "sell A below base while sitting on B" event is visible. It also measures,
per product, how much we HOLD to the endgame dump vs sell during the game, and
whether that held stock was ever going to de-glut (it can't if we keep making it).

Run: python -m kaggriculture.bandit.gate.sell_audit
"""
import os, argparse
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
from kaggriculture.bandit.gate import loss_forensics as LF
import kaggriculture.measure.eval_harness as EH

PREM = LF.PREM
BASE = LF.BASEPX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--killer", default="tschinkel")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    ws = H.world_seeds(2)
    seed = a.seed or ws[0][1]
    kp = dict((k, p) for k, p in LF.KILLERS)[a.killer]
    cfg, _ = VL.COMBOS["sweep15"]
    H.build_agent("sa_opt", cfg)
    LF.stage_g = os.path.join(H.SCRATCH, "sa_opt_wingate")
    ours = EH.bandit_binary_agent(LF.stage_g)
    recs, ob, us, them = LF.capture_game(ours, kp, seed, 0)
    print(f"AUDIT vs {a.killer} seed={seed}: us=${us:,.0f} vs ${them:,.0f} gap={us-them:+,.0f}\n", flush=True)

    # 1) wrong-product events: sell A below base while holding >=20 of premium B
    print("=== wrong-product events (sell A <base while sitting on premium B>=20) ===", flush=True)
    n_events = 0
    examples = []
    for r in recs:
        a2 = r["our_act"] or {}
        shed = r["our_shed"] or {}
        prices = r["prices"] or {}
        for o in (a2.get("market") or []):
            if not o or o[0] != "SELL" or len(o) < 3:
                continue
            p = o[1]; px = float(prices.get(p, 0) or 0)
            if p in PREM and px < BASE.get(p, 1):
                held = {q: int(shed.get(q, 0) or 0) for q in PREM if q != p and int(shed.get(q, 0) or 0) >= 20}
                if held:
                    n_events += 1
                    if len(examples) < 12:
                        examples.append((r["step"], p, px, BASE.get(p), held))
    for step, p, px, base, held in examples:
        print(f"  step {step:3d}: SELL {p} @${px:.0f} (base {base}) while holding {held}", flush=True)
    print(f"  TOTAL such events: {n_events}\n", flush=True)

    # 2) per-product: sold-during-game vs held-to-endgame(>=step648) dump, and price realized
    print("=== per premium product: during-game vs endgame dump (est-filled) ===", flush=True)
    e = LF.econ(recs, "our")
    print(f"  {'prod':10s} {'fill_all':>8s} {'fill_end':>8s} {'end%':>5s} {'below%':>6s} {'$/u':>6s}", flush=True)
    for p in PREM:
        fa = e["filled"].get(p, 0); fe = e["filled_end"].get(p, 0)
        bl = e["below"].get(p, 0); rv = e["realized"].get(p, 0)
        endp = 100 * fe / fa if fa else 0; blp = 100 * bl / fa if fa else 0
        print(f"  {p:10s} {fa:8.0f} {fe:8.0f} {endp:4.0f}% {blp:5.0f}% {rv/fa if fa else 0:6.1f}", flush=True)

    # 3) does the market ever de-glut? track market inventory of milk/wool over the game
    print("\n=== market inventory (glut proxy; I0=10000, above=glut) sampled by day ===", flush=True)
    line = defaultdict(list)
    for r in recs:
        if r["step"] % 72 == 0:
            for p in ("MILK", "WOOL", "STRAWBERRY"):
                line[p].append(int((r["inv"] or {}).get(p, 0) or 0))
    for p in ("MILK", "WOOL", "STRAWBERRY"):
        print(f"  {p:10s} " + " ".join(f"{v:6d}" for v in line[p]), flush=True)

    # cleanup
    try:
        import subprocess as _sp
        _sp.run(["powershell", "-NoProfile", "-Command",
                 "Get-Process kagg -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.Path -like '*sa_opt_wingate*' } | Stop-Process -Force"],
                capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
