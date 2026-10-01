"""Compatible vs incompatible ECONOMIES per world, empirically.

The engine's daily_demand table maps each SHOP to the products it consumes; a
product is COMPATIBLE with a world iff at least one of its demand shops is
unlocked there (and its realized value scales with how many). Our base tape is
NON-adaptive -- it produces the same dairy+wool mix in every world -- so in
worlds missing a product's demand shop we glut it to the $1 floor (WOOL with no
YARN = ~$6/u). This prints, per world: the demand-shop count for each product and
OUR realized $/u + below-base%, so the compatible/incompatible split is explicit.

Run: python -m kaggriculture.bandit.gate.compat --seeds 6
"""
import os, argparse
from collections import defaultdict
import numpy as np
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import verify_levers as VL
from kaggriculture.bandit.gate import loss_forensics as LF
import kaggriculture.measure.eval_harness as EH

# engine daily_demand: shop -> products it consumes (from mbandit.rs daily_demand)
SHOP_DEMAND = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]
ANIMAL_PRODUCT = {"MILK": "COW", "WOOL": "SHEEP", "EGG": "GOOSE"}


def demand_shops(product):
    return [s for s, prods in SHOP_DEMAND.items() if product in prods]


def world_demand_count(shops, product):
    return sum(1 for s in shops if product in SHOP_DEMAND.get(s, []))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=6); a = ap.parse_args()
    ws = H.world_seeds(a.seeds)
    kp = dict((k, p) for k, p in LF.KILLERS)["tschinkel"]
    cfg, _ = VL.COMBOS["sweep15"]
    H.build_agent("cp_opt", cfg)
    LF.stage_g = os.path.join(H.SCRATCH, "cp_opt_wingate")
    ours = EH.bandit_binary_agent(LF.stage_g)

    print("PRODUCT -> demand shops (fewer = more fragile):", flush=True)
    for p in PRODUCTS:
        print(f"  {p:10s} {demand_shops(p) or ['(none/base)']}", flush=True)
    print()

    agg = defaultdict(lambda: {"dc": [], "dpu": [], "below": [], "fill": []})
    for wn, seed in ws:
        recs, ob, us, them = LF.capture_game(ours, kp, seed, 0)
        shops = set()
        for o in ob:
            for s in (o.get("town") or {}).get("unlocked_shops") or []:
                shops.add(s)
        e = LF.econ(recs, "our")
        print(f"=== seed {seed}  gap={us-them:+.0f}  shops={sorted(shops)} ===", flush=True)
        print(f"  {'product':10s} {'demandShops':>11s} {'our$/u':>7s} {'below%':>6s} {'fill':>5s}  {'verdict'}", flush=True)
        for p in PRODUCTS:
            dc = world_demand_count(shops, p)
            f = e["filled"].get(p, 0); rv = e["realized"].get(p, 0); bl = e["below"].get(p, 0)
            dpu = rv / f if f else 0; blp = 100 * bl / f if f else 0
            verdict = ""
            if p in ("WOOL", "MILK", "EGG") and f > 20:  # our animal products
                if dc == 0:
                    verdict = "INCOMPATIBLE (glut -- no demand shop)"
                elif blp > 80:
                    verdict = "over-produced (glut)"
                else:
                    verdict = "ok"
            print(f"  {p:10s} {dc:>11d} {dpu:>7.1f} {blp:>5.0f}% {f:>5.0f}  {verdict}", flush=True)
            agg[p]["dc"].append(dc); agg[p]["dpu"].append(dpu); agg[p]["below"].append(blp); agg[p]["fill"].append(f)
        print()

    print("=== SUMMARY: our animal products vs demand (mean over worlds) ===", flush=True)
    for p in ("MILK", "WOOL", "EGG"):
        d = agg[p]
        print(f"  {p:6s} (animal {ANIMAL_PRODUCT[p]}): mean demandShops={np.mean(d['dc']):.1f}  "
              f"mean$/u={np.mean(d['dpu']):.1f}  mean below%={np.mean(d['below']):.0f}  "
              f"worlds_with_0_demand={sum(1 for x in d['dc'] if x==0)}/{len(d['dc'])}", flush=True)

    try:
        import subprocess as _sp
        _sp.run(["powershell", "-NoProfile", "-Command",
                 "Get-Process kagg -ErrorAction SilentlyContinue | "
                 "Where-Object { $_.Path -like '*cp_opt_wingate*' } | Stop-Process -Force"],
                capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
