#!/usr/bin/env python
"""Stage 6: DIVERSIFIED stack toward the competitive bar. Generalized husbandry
(COW->MILK, SHEEP->WOOL, GOOSE->EGG) + MELON anchor + CARROT, persistent-cluster
routing + don't-dump + grow-feed. Sized to the per-world scarcity-demand budget
(MILK ~$45k, STRAWBERRY ~$34k, WOOL ~$33k[YARN], MELON ~$27k anchor).

Reuses season_cbs's routing (persistent clusters, dedicated husbandry movers).
Reads full state (offline), emits a fixed .tape, validates replay==record.
Run: PYTHONPATH=vendor python season_full.py [seed]  |  season_full.py worlds
"""
import json
import os
import sys
import statistics

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import exact_rev, dont_dump_qty, default_thresh

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LIQ = 27
FULL = {"WHEAT": 4, "CARROT": 3, "MELON": 6}
STRUCT = {"COW": "PASTURE", "SHEEP": "PASTURE", "GOOSE": "COOP"}
PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
BUILD_OP = {"PASTURE": "BUILD_PASTURE", "COOP": "BUILD_COOP"}


def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def stept(cur, tgt):
    if cur[0] != tgt[0]:
        return "EAST" if tgt[0] > cur[0] else "WEST"
    if cur[1] != tgt[1]:
        return "SOUTH" if tgt[1] > cur[1] else "NORTH"
    return None


def nearest(pos, tiles):
    return min(tiles, key=lambda t: manh(pos, t))


def chunk(tiles, n):
    if n <= 0:
        return []
    ts = sorted(tiles, key=lambda t: (t[0], t[1] if t[0] % 2 == 0 else -t[1]))
    out, per = [], max(1, (len(ts) + n - 1) // n)
    for i in range(n):
        c = ts[i * per:(i + 1) * per]
        out.append(c)
    if sum(len(c) for c in out) < len(ts):
        out[-1].extend(ts[sum(len(c) for c in out):])
    return out


class FullController:
    def __init__(self, tile_crop, animals, wheat_tiles, n_hands, n_husb=3, land_days=(), reserve=400):
        self.tile_crop = tile_crop
        self.animals = animals                # [(x,y,type)]
        self.wheat_tiles = wheat_tiles
        self.n_hands = n_hands
        self.land_days = set(land_days)
        self.reserve = reserve                # cash cushion kept before buying animals/land
        self.crops = sorted(set(tile_crop.values()))
        self.tc_all = dict(tile_crop)
        for wt in wheat_tiles:
            self.tc_all[wt] = "WHEAT"
        self.products = ["MELON", "CARROT", "MILK", "WOOL", "EGG", "STRAWBERRY"]
        self.thresh = {p: default_thresh(p, 0.5) for p in self.products}
        self.sell_log = {p: [] for p in self.products}
        self.actions = []
        self.feed_ok = self.feed_attempt = 0
        self.n_animals = len(animals)
        n_animal_types = {a[2] for a in animals}
        # husbandry movers 0..n_husb-1 ; crop movers the rest
        n_crop = max(1, (n_hands + 1) - n_husb)
        self.husb = list(range(n_husb))
        cl = chunk(list(tile_crop.keys()), n_crop)
        self.mover_crop = {n_husb + j: set(c) for j, c in enumerate(cl)}
        # order animals: buy COWs (milk, top budget) first, then SHEEP, then GOOSE
        self.buy_order = sorted(animals, key=lambda a: {"COW": 0, "SHEEP": 1, "GOOSE": 2}[a[2]])

    def _crop_task(self, farm, seeds, day, hour, x, y):
        crop = self.tc_all[(x, y)]
        t = farm["tiles"][y][x]
        if t == "LOCKED":
            return None
        cd = CROPS[crop]
        if t is None:
            return ("PLANT", crop, 2) if seeds.get(crop, 0) > 0 and hour <= 14 else None
        if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop:
            age = day - t["planted_day"]; yu = t.get("yield_units", 0)
            capped = yu >= cd["max_yield"]
            if not t["watered_today"] and not capped:
                return ("WATER", None, 0)
            if not cd["ongoing"] and (age >= cd["max_yield_day"] or capped) and yu > 0:
                return ("HARVEST", None, 1)
            if cd["ongoing"] and yu > 0 and age > cd["first_yield_day"]:
                return ("HARVEST", None, 1)
        return None

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]; priv = obs["private"]
        shed = priv["shed"]; invs = priv["inventories"]; seeds = priv["seeds"]
        minv = obs["market"]["inventory"]
        movers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
        nm = len(movers)

        def mi(i):
            return invs[i] if i < len(invs) else {}

        act = {i: ["PASS"] for i in range(nm)}

        # ===== husbandry movers =====
        husb = [i for i in self.husb if i < nm]
        free = set(husb)
        # classify animal tiles
        tasks = []  # (pos, kind, action-or-None)
        n_present = 0
        for (x, y, atype) in self.animals:
            t = farm["tiles"][y][x]
            struct = STRUCT[atype]
            if t is None:
                tasks.append(((x, y), "build", [BUILD_OP[struct]]))
            elif isinstance(t, dict) and t.get("kind") == struct and "animal" not in t:
                tasks.append(((x, y), "place", atype))
            elif isinstance(t, dict) and "animal" in t:
                n_present += 1
                if not t["fed_today"]:
                    tasks.append(((x, y), "feed", None))
                elif t.get("yield_units", 0) > 0:
                    tasks.append(((x, y), "harvest", ["HARVEST"]))
                elif not t["cared_today"]:
                    tasks.append(((x, y), "care", ["CARE"]))
        # wheat feed tasks
        for (x, y) in self.wheat_tiles:
            tk = self._crop_task(farm, seeds, day, hour, x, y)
            if tk:
                op = ["PLANT", "WHEAT"] if tk[0] == "PLANT" else [tk[0]]
                tasks.append(((x, y), "wheat", op))
        # order: feed > harvest > wheat-water > care > wheat-harvest/plant > place > build
        prio = {"feed": 0, "harvest": 1, "care": 3, "place": 5, "build": 6, "wheat": 2}
        tasks.sort(key=lambda tk: prio[tk[1]])
        for (tpos, kind, action) in tasks:
            if not free:
                break
            if kind == "feed":
                holders = [i for i in free if mi(i).get("WHEAT", 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manh(movers[i], tpos)); free.discard(m)
                    self.feed_attempt += 1
                    if movers[m] == tpos:
                        act[m] = ["FEED"]; self.feed_ok += 1
                    else:
                        act[m] = [stept(movers[m], tpos)] or ["PASS"]
                elif shed.get("WHEAT", 0) > 0:
                    m = min(free, key=lambda i: manh(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                    st = nearest(movers[m], SHED_TILES)
                    act[m] = ["PICKUP", "WHEAT", min(self.n_animals, shed.get("WHEAT", 0))] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"])
            elif kind == "place":
                atype = action
                holders = [i for i in free if mi(i).get(atype, 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manh(movers[i], tpos)); free.discard(m)
                    act[m] = ["PLACE", atype] if movers[m] == tpos else ([stept(movers[m], tpos)] or ["PASS"])
                elif shed.get(atype, 0) > 0:
                    m = min(free, key=lambda i: manh(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                    st = nearest(movers[m], SHED_TILES)
                    act[m] = ["PICKUP", atype, 1] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"])
            else:
                m = min(free, key=lambda i: manh(movers[i], tpos)); free.discard(m)
                act[m] = action if movers[m] == tpos else ([stept(movers[m], tpos)] or ["PASS"])

        # ===== crop movers (persistent clusters) =====
        for m, tiles in self.mover_crop.items():
            if m >= nm:
                continue
            best = None; bestd = 1e9
            for (x, y) in tiles:
                tk = self._crop_task(farm, seeds, day, hour, x, y)
                if not tk:
                    continue
                d = manh(movers[m], (x, y)) + tk[2] * 0.01
                if d < bestd:
                    bestd, best = d, ((x, y), tk)
            if best:
                (tp, (op, arg, _)) = best
                act[m] = (["PLANT", arg] if op == "PLANT" else [op]) if movers[m] == tp else ([stept(movers[m], tp)] or ["PASS"])

        farmer = act[0]; hands = [act[i] for i in range(1, nm)]

        # ===== market =====
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1:
            need = {}
            for c in self.tc_all.values():
                need[c] = need.get(c, 0) + 1
            for c in sorted(need):
                deficit = need[c] - seeds.get(c, 0)
                if deficit > 0 and len(market) < 10 and farm["money"] >= CROPS[c]["seed"]:
                    market.append(["BUY_SEED", c, min(deficit, 20)])
        # bootstrap + ongoing feed wheat: keep shed wheat >= herd size (buy shortfall)
        if hour == 2 and self.n_animals:
            present = sum(1 for (x, y, a) in self.animals if isinstance(farm["tiles"][y][x], dict) and "animal" in farm["tiles"][y][x])
            if shed.get("WHEAT", 0) < present + 1 and farm["money"] > 400 and len(market) < 10:
                market.append(["BUY_PRODUCT", "WHEAT", max(1, present)])
        if hour in (3, 4) and day in self.land_days and farm["money"] >= 1000 + self.reserve and len(market) < 10:
            market.append(["BUY_LAND"])
        # buy 1 animal/day (cows first), staggered, keep cash cushion (reserve knob)
        if hour == 3 and self.n_animals:
            placed = sum(1 for (x, y, a) in self.animals if isinstance(farm["tiles"][y][x], dict) and ("animal" in farm["tiles"][y][x]))
            inshed = sum(shed.get(a, 0) for a in ("COW", "SHEEP", "GOOSE"))
            have = placed + inshed
            if have < self.n_animals and day >= 4:
                atype = self.buy_order[have][2]
                cost = ANIMALS[atype]["cost"]
                if farm["money"] >= cost + self.reserve and len(market) < 10:
                    market.append(["BUY_ANIMAL", atype, 1])
        # sells: don't-dump each product, one per hour slot
        for k, p in enumerate(self.products):
            if hour == 5 + k:
                have = shed.get(p, 0)
                if have > 0:
                    q = dont_dump_qty(p, minv[p], have, self.thresh[p]) if day < LIQ else have
                    if q > 0:
                        market.append(["SELL", p, q])
                        self.sell_log[p].append((day, minv[p], q))
        return {"farmer": farmer, "hands": hands, "market": market}


def build_farm(quads=("NW",), n_cow=5, n_sheep=0, n_goose=0, n_melon=10, n_straw=0):
    """MELON anchor carved from FAR-NW (available day 0); animals + wheat feed take
    the tiles NEAREST the central shed (feed carry works); a compact STRAWBERRY
    block near-center gets its own persistent cluster (dedicated waterer); carrot
    fills the rest + expansion quadrants."""
    def quad_tiles(q):
        xr = range(0, 5) if q in ("NW", "SW") else range(5, 10)
        yr = range(0, 5) if q in ("NW", "NE") else range(5, 10)
        return [(x, y) for y in yr for x in xr if (x, y) not in SHED_TILES]
    nw = sorted(quad_tiles("NW"), key=lambda t: manh(t, (4, 4)))
    melon_tiles = nw[-n_melon:]                       # far NW
    allt = []
    for q in quads:
        allt += quad_tiles(q)
    remaining = sorted([t for t in allt if t not in melon_tiles], key=lambda t: manh(t, (4, 4)))
    na = n_cow + n_sheep + n_goose
    animals = ([(x, y, "COW") for (x, y) in remaining[:n_cow]]
               + [(x, y, "SHEEP") for (x, y) in remaining[n_cow:n_cow + n_sheep]]
               + [(x, y, "GOOSE") for (x, y) in remaining[n_cow + n_sheep:na]])
    n_wheat = max(4, na)
    wheat = remaining[na:na + n_wheat]
    straw = remaining[na + n_wheat:na + n_wheat + n_straw]
    carrot = remaining[na + n_wheat + n_straw:]
    tile_crop = {t: "MELON" for t in melon_tiles}
    tile_crop.update({t: "STRAWBERRY" for t in straw})
    tile_crop.update({t: "CARROT" for t in carrot})
    return tile_crop, animals, wheat


FARM = dict(quads=("NW", "NE"), n_cow=6, n_sheep=0, n_goose=0, n_melon=10, n_straw=0)
LAND_DAYS = (13,)
RESERVE = 700


def run(seed, quiet=False, fixed_tape=None):
    tile_crop, animals, wheat = build_farm(**FARM)
    ctrl = FullController(tile_crop, animals, wheat, n_hands=10, n_husb=3, land_days=LAND_DAYS, reserve=RESERVE)
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.reset(2)
    if fixed_tape is not None:
        # replay a FIXED tape (open-loop) — ignore state
        calls = {"i": 0}
        def tp(o, c=None):
            i = calls["i"]; calls["i"] += 1
            return fixed_tape[i] if i < len(fixed_tape) else PASS
        env.run([tp, lambda o, c=None: PASS])
        cash = env.steps[-1][0]["reward"]
        shops = env.steps[-1][0]["observation"]["town"]["unlocked_shops"]
        if quiet:
            print(f"  seed {seed:3}: FIXED-tape Cash720 ${cash:7.0f} YARNx{shops.count('YARN_STORE')}")
        return cash, None

    while not env.done:
        obs = env.state[0].observation
        a0 = ctrl.decide(obs)
        ctrl.actions.append(a0)
        env.step([a0, PASS])

    fobs = env.state[0].observation
    cash = fobs["farms"][0]["money"]; fshed = fobs["private"]["shed"]
    shops = fobs["town"]["unlocked_shops"]
    weeds = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and t.get("kind") == "WEED")
    animals_alive = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and "animal" in t)

    contrib = {}
    total_rev = 0
    for p in ctrl.products:
        log = ctrl.sell_log[p]
        sold = sum(q for (_, _, q) in log)
        rev = sum(exact_rev(p, q, v0) for (_, v0, q) in log)
        contrib[p] = (sold, rev)
        total_rev += rev
    costs = 3000 + total_rev - cash

    n_rows = env.configuration.episodeSteps - 1
    rows = [f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}'
            for a in (ctrl.actions[:n_rows] + [PASS] * max(0, n_rows - len(ctrl.actions)))]
    tape = os.path.join(HERE, "..", "data", f"full_seed{seed}.tape")
    with open(tape, "w") as f:
        f.write("\n".join(rows) + "\n")
    replay = [dict(farmer=json.loads(r.split("\t")[0]), hands=json.loads(r.split("\t")[1]), market=json.loads(r.split("\t")[2])) for r in rows]
    env2 = make("kaggriculture", configuration={"seed": seed}, debug=False)
    calls = {"i": 0}
    def tp(o, c=None):
        i = calls["i"]; calls["i"] += 1
        return replay[i] if i < len(replay) else PASS
    env2.run([tp, lambda o, c=None: PASS])
    rc = env2.steps[-1][0]["reward"]

    if quiet:
        print(f"  seed {seed:3}: ${cash:7.0f} YARNx{shops.count('YARN_STORE')} " +
              " ".join(f"{p[:2]}{contrib[p][1]//1000}k" for p in ctrl.products if contrib[p][0]) +
              f" weeds{weeds} replay=={abs(rc-cash)<1e-6}")
        return cash, replay
    print(f"=== FULL seed {seed} === YARNx{shops.count('YARN_STORE')} animals_alive={animals_alive} weeds={weeds}")
    for p in ctrl.products:
        s, r = contrib[p]
        if s:
            print(f"  {p:8} sold {s:4} rev ${r:7} (${r/s:6.1f}/u)")
    print(f"  total rev ${total_rev} costs ${costs:.0f}  ACTUAL Cash720 ${cash:.0f}  replay==record {abs(rc-cash)<1e-6} (${rc:.0f})")
    return cash, replay


def set_cfg(n_cow, land_day, reserve, n_straw=0, quads=("NW", "NE")):
    global FARM, LAND_DAYS, RESERVE
    FARM = dict(quads=quads, n_cow=n_cow, n_sheep=0, n_goose=0, n_melon=10, n_straw=n_straw)
    LAND_DAYS = (land_day,)
    RESERVE = reserve


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        print("=== BOUNDED bootstrap grid (seed 42): cow-count x land-day x reserve ===")
        results = []
        for nc in (6, 7, 8):
            for ld in (11, 13, 15):
                for rs in (300, 700):
                    set_cfg(nc, ld, rs)
                    c, _ = run(42, quiet=True)
                    results.append((c, nc, ld, rs))
        results.sort(reverse=True)
        print("\nTOP 5 schedules:")
        for c, nc, ld, rs in results[:5]:
            print(f"  cow={nc} land_day={ld} reserve={rs} -> ${c:.0f}")
    elif len(sys.argv) > 1 and sys.argv[1] == "worlds":
        cs = []
        for sd in (1, 3, 7, 13, 21, 42, 77, 100):
            c, _ = run(sd, quiet=True)
            cs.append(c)
        print(f"\nADAPTIVE cross-world: mean ${statistics.mean(cs):.0f} min ${min(cs):.0f} max ${max(cs):.0f} cv {100*statistics.pstdev(cs)/statistics.mean(cs):.0f}%")
    elif len(sys.argv) > 1 and sys.argv[1] == "fixed":
        # build ONE fixed tape on a representative seed, replay across the panel
        base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
        _, tape = run(base_seed, quiet=True)
        print(f"[fixed tape recorded on seed {base_seed}] replayed across panel:")
        cs = []
        for sd in (1, 3, 7, 13, 21, 42, 77, 100):
            c, _ = run(sd, quiet=True, fixed_tape=tape)
            cs.append(c)
        print(f"\nFIXED-tape cross-world: mean ${statistics.mean(cs):.0f} min ${min(cs):.0f} max ${max(cs):.0f} cv {100*statistics.pstdev(cs)/statistics.mean(cs):.0f}%")
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
