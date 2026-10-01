#!/usr/bin/env python
"""Stage 4: STACK products with the optimizer policies (don't-dump + right-size +
grow-feed), greedy routing, one farm. Each product sized below its knee so
revenue STACKS: MELON (value) + CARROT (scarcity volume) + WOOL (animal) +
grown WHEAT (feed). Reports per-product sold/rev, total validated Cash720
(replay==record), routing efficiency, and where routing/shed becomes binding.

Run: PYTHONPATH=vendor python season_stack.py
"""
import json
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import market_price, exact_rev, dont_dump_qty, default_thresh

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LIQ = 27


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def stept(cur, tgt):
    if cur[0] != tgt[0]:
        return "EAST" if tgt[0] > cur[0] else "WEST"
    if cur[1] != tgt[1]:
        return "SOUTH" if tgt[1] > cur[1] else "NORTH"
    return None


def nearest(pos, tiles):
    return min(tiles, key=lambda t: manhattan(pos, t))


class StackController:
    def __init__(self, tile_crop, sheep_tiles, n_hands, land_days=()):
        self.tile_crop = tile_crop          # (x,y)->crop  (incl WHEAT feed tiles)
        self.crops = sorted(set(tile_crop.values()))
        self.sheep_tiles = sheep_tiles
        self.n_sheep = len(sheep_tiles)
        self.n_hands = n_hands
        self.land_days = set(land_days)
        self.actions = []
        self.sell_log = {p: [] for p in self.crops + ["WOOL"]}
        self.thresh = {p: default_thresh(p, 0.5) for p in self.crops + ["WOOL"]}
        self.feed_ok = 0
        self.feed_attempt = 0

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]
        priv = obs["private"]
        shed = priv["shed"]
        invs = priv["inventories"]
        seeds = priv["seeds"]
        minv = obs["market"]["inventory"]
        movers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]

        def mi(i):
            return invs[i] if i < len(invs) else {}

        # ---- sheep classification ----
        need_build, need_place, sheep_present = [], [], []
        for (x, y) in self.sheep_tiles:
            t = farm["tiles"][y][x]
            if t is None:
                need_build.append((x, y))
            elif isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                need_place.append((x, y))
            elif isinstance(t, dict) and "animal" in t:
                sheep_present.append(((x, y), t))
        unfed = [(p, t) for (p, t) in sheep_present if not t["fed_today"]]
        wool_ready = [(p, t) for (p, t) in sheep_present if t.get("yield_units", 0) > 0]

        # ---- crop tasks ----
        crop_water, crop_harv, crop_plant = [], [], []
        for (x, y), crop in self.tile_crop.items():
            t = farm["tiles"][y][x]
            if t == "LOCKED":
                continue
            cd = CROPS[crop]
            if t is None:
                if seeds.get(crop, 0) > 0 and hour <= 12:
                    crop_plant.append(((x, y), crop))
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop:
                age = day - t["planted_day"]; yu = t.get("yield_units", 0)
                capped = yu >= cd["max_yield"]
                if not t["watered_today"] and not capped:
                    crop_water.append((x, y))
                elif not cd["ongoing"] and (age >= cd["max_yield_day"] or capped) and yu > 0:
                    crop_harv.append((x, y))
                elif cd["ongoing"] and yu > 0 and age > cd["first_yield_day"]:
                    crop_harv.append((x, y))

        act = {i: ["PASS"] for i in range(len(movers))}
        free = set(range(len(movers)))

        def assign(tpos, action):
            if not free:
                return
            m = min(free, key=lambda i: manhattan(movers[i], tpos))
            free.discard(m)
            act[m] = action if movers[m] == tpos else ([stept(movers[m], tpos)] or ["PASS"])

        # P1 feed sheep (wheat carry)
        for (p, t) in unfed:
            if not free:
                break
            holders = [i for i in free if mi(i).get("WHEAT", 0) > 0]
            if holders:
                m = min(holders, key=lambda i: manhattan(movers[i], p)); free.discard(m)
                self.feed_attempt += 1
                if movers[m] == p:
                    act[m] = ["FEED"]; self.feed_ok += 1
                else:
                    act[m] = [stept(movers[m], p)] or ["PASS"]
            elif shed.get("WHEAT", 0) > 0:
                m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                st = nearest(movers[m], SHED_TILES)
                act[m] = ["PICKUP", "WHEAT", min(self.n_sheep, shed.get("WHEAT", 0))] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"])
        # P2 water crops (incl wheat feed tiles)
        for p in crop_water:
            assign(p, ["WATER"])
        # P3 harvest crops + wool
        for p in crop_harv:
            assign(p, ["HARVEST"])
        for (p, t) in wool_ready:
            assign(p, ["HARVEST"])
        # P4 care
        for (p, t) in sheep_present:
            if t["fed_today"] and not t["cared_today"]:
                assign(p, ["CARE"])
        # P5 plant crops
        for (p, crop) in crop_plant:
            assign(p, ["PLANT", crop])
        # P6 place/build sheep
        for p in need_place:
            holders = [i for i in free if mi(i).get("SHEEP", 0) > 0]
            if holders:
                m = min(holders, key=lambda i: manhattan(movers[i], p)); free.discard(m)
                act[m] = ["PLACE", "SHEEP"] if movers[m] == p else ([stept(movers[m], p)] or ["PASS"])
            elif shed.get("SHEEP", 0) > 0 and free:
                m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                st = nearest(movers[m], SHED_TILES)
                act[m] = ["PICKUP", "SHEEP", 1] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"])
        for p in need_build:
            assign(p, ["BUILD_PASTURE"])

        farmer = act[0]
        hands = [act[i] for i in range(1, len(movers))]

        # ---- market ----
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1:
            need = {}
            for (x, y), c in self.tile_crop.items():
                need[c] = need.get(c, 0) + 1
            for c in self.crops:
                deficit = need.get(c, 0) - seeds.get(c, 0)
                if deficit > 0 and len(market) < 10 and farm["money"] >= CROPS[c]["seed"]:
                    market.append(["BUY_SEED", c, min(deficit, 20)])
        # bootstrap wheat feed before grown wheat matures (days 0-5)
        if hour == 2 and self.n_sheep and day <= 5 and shed.get("WHEAT", 0) < self.n_sheep and len(market) < 10 and farm["money"] > 300:
            market.append(["BUY_PRODUCT", "WHEAT", self.n_sheep])
        if hour == 4 and day in self.land_days and len(market) < 10:
            market.append(["BUY_LAND"])
        if hour == 3 and self.n_sheep:
            have = shed.get("SHEEP", 0) + len(sheep_present)
            if have < self.n_sheep and farm["money"] >= ANIMALS["SHEEP"]["cost"] and len(market) < 10:
                market.append(["BUY_ANIMAL", "SHEEP", 1])
        # sells: don't-dump per product, one per hour slot (record v0)
        sellables = [c for c in self.crops if c != "WHEAT"] + (["WOOL"] if self.n_sheep else [])
        for k, p in enumerate(sellables):
            if hour == 5 + k:
                have = shed.get(p, 0)
                if have > 0:
                    if day < LIQ:
                        q = dont_dump_qty(p, minv[p], have, self.thresh[p])
                    else:
                        q = have
                    if q > 0:
                        market.append(["SELL", p, q])
                        self.sell_log[p].append((day, minv[p], q))
        return {"farmer": farmer, "hands": hands, "market": market}


def run(tile_crop, sheep_tiles, n_hands, label, land_days=()):
    ctrl = StackController(tile_crop, sheep_tiles, n_hands, land_days)
    env = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    env.reset(2)
    prev_seeds = {c: 0 for c in ctrl.crops}
    bought = {c: 0 for c in ctrl.crops}
    while not env.done:
        obs = env.state[0].observation
        s = obs["private"]["seeds"]
        for c in ctrl.crops:
            if s.get(c, 0) > prev_seeds[c]:
                bought[c] += s[c] - prev_seeds[c]
            prev_seeds[c] = s.get(c, 0)
        a0 = ctrl.decide(obs)
        ctrl.actions.append(a0)
        env.step([a0, PASS])

    fobs = env.state[0].observation
    cash = fobs["farms"][0]["money"]
    fshed = fobs["private"]["shed"]
    fseeds = fobs["private"]["seeds"]
    shops = fobs["town"]["unlocked_shops"]
    weeds = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and t.get("kind") == "WEED")
    sheep_alive = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
    FULL = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}

    print(f"=== STACK {label}: hands={n_hands} seed {SEED} ===")
    print(f"YARN_STORE x{shops.count('YARN_STORE')}  sheep_alive={sheep_alive}  weeds={weeds}")
    total_rev = 0
    for p in [c for c in ctrl.crops if c != "WHEAT"] + ["WOOL"]:
        log = ctrl.sell_log.get(p, [])
        sold = sum(q for (_, _, q) in log)
        rev = sum(exact_rev(p, q, v0) for (_, v0, q) in log)
        total_rev += rev
        eff = ""
        if p in bought:
            plant = bought[p] - fseeds.get(p, 0)
            deliv = sold + fshed.get(p, 0)
            ideal = plant * FULL[p]
            eff = f" eff={100*deliv/ideal:.0f}%" if ideal else ""
        per = rev / sold if sold else 0
        if sold:
            print(f"  {p:11} sold {sold:4}  rev ${rev:7} (${per:6.1f}/u){eff}")
    costs = 3000 + total_rev - cash

    # validate
    n_rows = env.configuration.episodeSteps - 1
    rows = [f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}'
            for a in (ctrl.actions[:n_rows] + [PASS] * max(0, n_rows - len(ctrl.actions)))]
    tape = os.path.join(HERE, "..", "data", f"stack_{label}.tape")
    with open(tape, "w") as f:
        f.write("\n".join(rows) + "\n")
    replay = [dict(farmer=json.loads(r.split("\t")[0]), hands=json.loads(r.split("\t")[1]), market=json.loads(r.split("\t")[2])) for r in rows]
    env2 = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    calls = {"i": 0}
    def tp(o, c=None):
        i = calls["i"]; calls["i"] += 1
        return replay[i] if i < len(replay) else PASS
    env2.run([tp, lambda o, c=None: PASS])
    rc = env2.steps[-1][0]["reward"]
    print(f"  total revenue ${total_rev}  costs ${costs:.0f}")
    print(f"  ACTUAL Cash720 ${cash:.0f}   feed {ctrl.feed_ok}/{ctrl.feed_attempt}   replay==record {abs(rc-cash)<1e-6} (${rc:.0f}) -> {os.path.basename(tape)}")
    return cash


def build():
    # NW 25 tiles: melon 10 + carrot 8 + wheat(feed) 4 + sheep 3, nearest-shed first
    cand = [(x, y) for y in range(5) for x in range(5)]
    cand.sort(key=lambda t: manhattan(t, (4, 4)))
    sheep = [t for t in cand if t not in SHED_TILES][:3]
    rest = [t for t in cand if t not in sheep and t not in SHED_TILES]
    tile_crop = {}
    mix = ["WHEAT"] * 4 + ["MELON"] * 10 + ["CARROT"] * 8
    for i, t in enumerate(rest[:len(mix)]):
        tile_crop[t] = mix[i]
    return tile_crop, sheep


def build_land():
    """NW as build(); add NE+SW quadrants of CARROT (routes well, town-demanded)."""
    tc, sheep = build()
    for q in ("NE", "SW"):
        xr = range(5, 10) if q == "NE" else range(0, 5)
        yr = range(0, 5) if q == "NE" else range(5, 10)
        for y in yr:
            for x in xr:
                tc[(x, y)] = "CARROT"
    return tc, sheep


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nw"
    if mode == "land":
        tc, sheep = build_land()
        run(tc, sheep, 10, "land_melon_carrot_wool", land_days=(9, 15))
    else:
        tc, sheep = build()
        run(tc, sheep, 10, "nw_melon_carrot_wool")
