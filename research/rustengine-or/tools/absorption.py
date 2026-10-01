#!/usr/bin/env python
"""B1.2-full: derive the T-absorption table A_c by SIMULATING one tile per crop
through the REAL vendored engine (no re-implementation of biology).

For each crop we run a real `kaggriculture` episode where player 0 keeps its
farmer on a single owned NW tile (the dawn spawn (4,4)), plants one seed, waters
it EVERY day (keeps it alive AND maximises yield), and harvests it (one-time: at
max_yield_day for full yield; ongoing: whenever yield accrues). Player 1 PASSes.
We never SELL, so the private shed accumulates exactly the units delivered; the
per-day increase of shed[product] is A_c for that day (units become sellable the
day AFTER they land in the shed, per the dusk-drop rule).

Run:
  PYTHONPATH=vendor python research/rustengine-or/tools/absorption.py
Emits research/rustengine-or/data/absorption.json and prints a verification table.
"""
import json
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, ANIMALS, pass_agent,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "absorption.json")

SPAWN = (4, 4)  # first NW shed-access tile == the dawn farmer spawn
RUN_DAYS = {  # how many days to simulate (cover full plant life)
    "WHEAT": 8, "CARROT": 8, "MELON": 16, "TOMATO": 22, "STRAWBERRY": 22,
}


def make_crop_agent(crop, log):
    """Closure agent: ONE tile at SPAWN, planted ONCE, watered daily, harvested at
    full yield. Records a per-(day,hour) snapshot of the product shed count."""
    cd = CROPS[crop]
    ongoing = cd["ongoing"]
    max_yield_day = cd["max_yield_day"]
    first_yield_day = cd["first_yield_day"]
    max_yield = cd["max_yield"]
    st = {"planted": False, "done": False}  # plant once; stop after final harvest

    def agent(obs):
        day = obs["day"]
        hour = obs["hour"]
        farm = obs["farms"][obs["player"]]
        priv = obs["private"]
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        log.append((day, hour, dict(shed)))

        fx, fy = farm["farmer"]
        tile = farm["tiles"][fy][fx]
        market = []
        farmer = ["PASS"]

        have_seed = seeds.get(crop, 0) > 0
        is_plant = isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop

        if st["done"]:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        if not st["planted"]:
            if tile is None and have_seed:
                st["planted"] = True
                return {"farmer": ["PLANT", crop], "hands": [], "market": []}
            if not have_seed:
                market.append(["BUY_SEED", crop, 1])
            return {"farmer": ["PASS"], "hands": [], "market": market}

        if is_plant:
            age = day - tile["planted_day"]
            yu = tile.get("yield_units", 0)
            # WATER FIRST every day (keeps alive + adds the day's yield for
            # one-time crops), THEN harvest.
            if not tile["watered_today"]:
                farmer = ["WATER"]
            elif not ongoing:
                # harvest once fully grown: capped, or past the yield window.
                if yu > 0 and (yu >= max_yield or age >= max_yield_day):
                    farmer = ["HARVEST"]
                    st["done"] = True
            else:
                # ongoing: harvest each morning's accrued yield after watering.
                if yu > 0 and age > first_yield_day:
                    farmer = ["HARVEST"]
        return {"farmer": farmer, "hands": [], "market": market}

    return agent


def derive_crop(crop):
    log = []
    env = make("kaggriculture", debug=False)
    agent = make_crop_agent(crop, log)
    env.run([agent, pass_agent])
    # steps to run: we let the full episode run but only read the plant's life.
    days = RUN_DAYS[crop]
    # For each day d, cumulative shed at dawn of day d+1 = delivered through end of d.
    cum = {}
    for (day, hour, shed) in log:
        if hour == 0:
            cum[day] = shed.get(crop, 0)
    # profile: units that LAND in shed at end of day d (sellable day d+1)
    profile = []
    prev = 0
    max_day = max([d for d in cum] + [0])
    for d in range(0, min(days, max_day) + 1):
        c = cum.get(d, prev)
        if c > prev:
            profile.append({"deliver_day": d - 1, "units": c - prev})
        prev = c
    total = prev
    # occupancy: days the tile is held (plant -> cleared). one-time cleared at
    # harvest; approximate by last delivery day + 1.
    occ = (profile[-1]["deliver_day"] + 2) if profile else 0
    return {
        "planted_day": 0,
        "ongoing": CROPS[crop]["ongoing"],
        "care": "daily_water_no_fert",
        "profile": profile,          # [{deliver_day, units}] units in shed at end of deliver_day
        "total_units": total,
        "occupancy_days": occ,
        "doc_T": None,
    }


def main():
    result = {}
    print(f"{'CROP':11} {'total':>6}  profile (deliver_day: units)")
    for crop in ["WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY"]:
        r = derive_crop(crop)
        result[crop] = r
        prof = ", ".join(f"d{p['deliver_day']}:{p['units']}" for p in r["profile"])
        print(f"{crop:11} {r['total_units']:>6}  {prof}")
    # expected sanity (no fertilizer, daily water)
    exp = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}
    print("\nverification vs expected total yield (no fert, daily water):")
    ok = True
    for c, e in exp.items():
        got = result[c]["total_units"]
        flag = "OK" if got == e else "MISMATCH"
        if got != e:
            ok = False
        print(f"  {c:11} expected {e}  got {got}  [{flag}]")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {os.path.normpath(OUT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
