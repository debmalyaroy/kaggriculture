#!/usr/bin/env python
"""Stage 5: PERSISTENT-CLUSTER routing (path-item b) to fix greedy's saturation
(75 tiles -> $14 because movers respawn at shed center, thrash on per-turn
nearest re-pick, and can't hold daily watering+feeding).

Fix:
  * PERSISTENT assignment: each mover owns a COMPACT cluster of tiles for the whole
    season (not a per-turn nearest re-pick). Crop tiles are partitioned into
    spatial bands (sort by x then y, chunk) so a mover walks a short local route.
  * DEDICATED husbandry movers (0=farmer + 1 hand) own the sheep+wheat-feed tiles
    near the shed: they PICKUP grown wheat at dawn (they spawn at the shed) and
    FEED, so husbandry never competes with distant crop watering.
  * Within each cluster, a Hungarian-lite greedy matches that cluster's movers to
    its pending tasks by walk distance (distinct tasks, 1 op/turn, no collision).
  * Economic optimizer kept (don't-dump / right-size / grow-feed / stack).

Reads full state (offline), emits a fixed .tape, validates replay==record.
Run: PYTHONPATH=vendor python season_cbs.py <ntiles:25|50|75|100> [n_hands]
"""
import json
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import exact_rev, dont_dump_qty, default_thresh

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LIQ = 27
FULL = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}


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


def chunk_compact(tiles, n):
    """Partition tiles into n compact clusters: sort by x (serpentine y), split
    into n contiguous equal-ish chunks (=> vertical bands, spatially compact)."""
    if n <= 0:
        return []
    ts = sorted(tiles, key=lambda t: (t[0], t[1] if t[0] % 2 == 0 else -t[1]))
    out, per = [], max(1, (len(ts) + n - 1) // n)
    for i in range(n):
        c = ts[i * per:(i + 1) * per]
        if c:
            out.append(c)
    # dump any remainder into the last cluster
    assigned = sum(len(c) for c in out)
    if assigned < len(ts):
        out[-1].extend(ts[assigned:])
    while len(out) < n:
        out.append([])
    return out


class CBSController:
    def __init__(self, tile_crop, sheep_tiles, wheat_tiles, n_hands):
        self.tile_crop = tile_crop
        self.sheep_tiles = sheep_tiles
        self.wheat_tiles = wheat_tiles
        self.tile_crop_all = dict(tile_crop)
        for wt in wheat_tiles:
            self.tile_crop_all[wt] = "WHEAT"
        self.n_sheep = len(sheep_tiles)
        self.n_hands = n_hands
        self.crops = sorted(set(tile_crop.values()))
        self.actions = []
        self.sell_log = {p: [] for p in self.crops + ["WOOL"]}
        self.thresh = {p: default_thresh(p, 0.5) for p in self.crops + ["WOOL"]}
        self.feed_ok = self.feed_attempt = 0
        # PERSISTENT clusters: movers 0,1 = husbandry (sheep+wheat); 2.. = crop bands
        n_husb = 2 if self.n_sheep else 0
        n_crop_movers = max(1, (n_hands + 1) - n_husb)
        crop_tiles = [t for t in tile_crop.keys()]
        self.crop_clusters = chunk_compact(crop_tiles, n_crop_movers)
        # mover_idx -> list of tiles (crops); husbandry handled separately
        self.mover_crop = {}
        for j, cl in enumerate(self.crop_clusters):
            self.mover_crop[n_husb + j] = set(cl)
        self.husb_movers = list(range(n_husb))

    # ---- crop task for one tile given current board ----
    def _crop_task(self, farm, seeds, day, hour, x, y):
        crop = self.tile_crop_all[(x, y)]
        t = farm["tiles"][y][x]
        if t == "LOCKED":
            return None
        cd = CROPS[crop]
        if t is None:
            if seeds.get(crop, 0) > 0 and hour <= 14:
                return ("PLANT", crop, 2)
            return None
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

        # ===== husbandry movers (0,1): sheep feed/care/harvest + wheat water/harvest =====
        if self.husb_movers:
            # sheep state
            need_build, need_place, present = [], [], []
            for (x, y) in self.sheep_tiles:
                t = farm["tiles"][y][x]
                if t is None:
                    need_build.append((x, y))
                elif isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                    need_place.append((x, y))
                elif isinstance(t, dict) and "animal" in t:
                    present.append(((x, y), t))
            unfed = [p for (p, t) in present if not t["fed_today"]]
            wool = [p for (p, t) in present if t.get("yield_units", 0) > 0]
            uncared = [p for (p, t) in present if t["fed_today"] and not t["cared_today"]]
            # wheat tasks
            wheat_tasks = []
            for (x, y) in self.wheat_tiles:
                tk = self._crop_task(farm, seeds, day, hour, x, y)
                if tk:
                    wheat_tasks.append(((x, y), tk[0], tk[1]))
            husb = [i for i in self.husb_movers if i < nm]
            free = set(husb)

            def do(m, action):
                act[m] = action

            # feed first
            for p in unfed:
                if not free:
                    break
                holders = [i for i in free if mi(i).get("WHEAT", 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manh(movers[i], p)); free.discard(m)
                    self.feed_attempt += 1
                    if movers[m] == p:
                        do(m, ["FEED"]); self.feed_ok += 1
                    else:
                        do(m, [stept(movers[m], p)] or ["PASS"])
                elif shed.get("WHEAT", 0) > 0:
                    m = min(free, key=lambda i: manh(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                    st = nearest(movers[m], SHED_TILES)
                    do(m, ["PICKUP", "WHEAT", min(self.n_sheep, shed.get("WHEAT", 0))] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"]))
            # then wheat water/harvest, wool harvest, care, place/build
            queue = [(pos, [op] if op != "PLANT" else ["PLANT", "WHEAT"]) for (pos, op, arg) in wheat_tasks if op == "WATER"]
            queue += [(pos, ["HARVEST"]) for (pos, op, arg) in wheat_tasks if op == "HARVEST"]
            queue += [(p, ["HARVEST"]) for p in wool]
            queue += [(p, ["CARE"]) for p in uncared]
            queue += [(pos, ["PLANT", "WHEAT"]) for (pos, op, arg) in wheat_tasks if op == "PLANT"]
            for (tpos, action) in queue:
                if not free:
                    break
                m = min(free, key=lambda i: manh(movers[i], tpos)); free.discard(m)
                do(m, action if movers[m] == tpos else ([stept(movers[m], tpos)] or ["PASS"]))
            # place sheep / build (use any leftover husbandry mover)
            for p in need_place:
                if not free:
                    break
                holders = [i for i in free if mi(i).get("SHEEP", 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manh(movers[i], p)); free.discard(m)
                    do(m, ["PLACE", "SHEEP"] if movers[m] == p else ([stept(movers[m], p)] or ["PASS"]))
                elif shed.get("SHEEP", 0) > 0:
                    m = min(free, key=lambda i: manh(movers[i], nearest(movers[i], SHED_TILES))); free.discard(m)
                    st = nearest(movers[m], SHED_TILES)
                    do(m, ["PICKUP", "SHEEP", 1] if movers[m] == st else ([stept(movers[m], st)] or ["PASS"]))
            for p in need_build:
                if not free:
                    break
                m = min(free, key=lambda i: manh(movers[i], p)); free.discard(m)
                do(m, ["BUILD_PASTURE"] if movers[m] == p else ([stept(movers[m], p)] or ["PASS"]))

        # ===== crop movers: each services ONLY its persistent cluster =====
        for m, tiles in self.mover_crop.items():
            if m >= nm:
                continue
            best = None; bestd = 1e9
            for (x, y) in tiles:
                tk = self._crop_task(farm, seeds, day, hour, x, y)
                if not tk:
                    continue
                d = manh(movers[m], (x, y)) + tk[2] * 0.01  # tie-break by priority
                if d < bestd:
                    bestd, best = d, ((x, y), tk)
            if best:
                (tp, (op, arg, _pri)) = best
                if movers[m] == tp:
                    act[m] = ["PLANT", arg] if op == "PLANT" else [op]
                else:
                    act[m] = [stept(movers[m], tp)] or ["PASS"]

        farmer = act[0]
        hands = [act[i] for i in range(1, nm)]

        # ===== market: hire, seeds, sheep, bootstrap feed, don't-dump sells =====
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1:
            need = {}
            for c in self.tile_crop.values():
                need[c] = need.get(c, 0) + 1
            for (x, y) in self.wheat_tiles:
                need["WHEAT"] = need.get("WHEAT", 0) + 1
            for c in sorted(need):
                deficit = need[c] - seeds.get(c, 0)
                if deficit > 0 and len(market) < 10 and farm["money"] >= CROPS[c]["seed"]:
                    market.append(["BUY_SEED", c, min(deficit, 20)])
        if hour == 2 and self.n_sheep and day <= 5 and shed.get("WHEAT", 0) < self.n_sheep and len(market) < 10 and farm["money"] > 300:
            market.append(["BUY_PRODUCT", "WHEAT", self.n_sheep])
        # delay sheep until early crop income arrives (avoid the bootstrap cash
        # drain that starves hiring), then buy 1/day; keep a cash cushion for hires.
        if hour == 3 and self.n_sheep and day >= 6:
            have = shed.get("SHEEP", 0) + sum(1 for (x, y) in self.sheep_tiles
                                              if isinstance(farm["tiles"][y][x], dict) and "animal" in farm["tiles"][y][x])
            if have < self.n_sheep and farm["money"] >= ANIMALS["SHEEP"]["cost"] + 500 and len(market) < 10:
                market.append(["BUY_ANIMAL", "SHEEP", 1])
        sellables = [c for c in self.crops if c != "WHEAT"] + (["WOOL"] if self.n_sheep else [])
        for k, p in enumerate(sellables):
            if hour == 5 + k:
                have = shed.get(p, 0)
                if have > 0:
                    q = dont_dump_qty(p, minv[p], have, self.thresh[p]) if day < LIQ else have
                    if q > 0:
                        market.append(["SELL", p, q])
                        self.sell_log[p].append((day, minv[p], q))
        return {"farmer": farmer, "hands": hands, "market": market}


def build_farm(ntiles):
    quads = {25: ["NW"], 50: ["NW", "NE"], 75: ["NW", "NE", "SW"], 100: ["NW", "NE", "SW", "SE"]}[ntiles]
    all_t = []
    for q in quads:
        xr = range(0, 5) if q in ("NW", "SW") else range(5, 10)
        yr = range(0, 5) if q in ("NW", "NE") else range(5, 10)
        all_t += [(x, y) for y in yr for x in xr]
    all_t = [t for t in all_t if t not in SHED_TILES]
    all_t.sort(key=lambda t: manh(t, (4, 4)))
    sheep = all_t[:3]
    wheat = all_t[3:7]
    rest = all_t[7:]  # sorted NEAR-center first
    # Put HIGH-op-frequency CARROT (3-day cycle: harvest+replant+water) NEAR the
    # shed (short dawn walk), and LOW-frequency MELON (10-day, one harvest) FAR.
    # Cap MELON at 10 tiles (market ~144u; more melon is wasted + cash drain).
    melon_count = min(10, len(rest))
    tile_crop = {}
    for i, t in enumerate(rest):
        tile_crop[t] = "MELON" if i >= len(rest) - melon_count else "CARROT"
    return tile_crop, sheep, wheat


def run(ntiles, n_hands=10, seed=SEED, quiet=False):
    tile_crop, sheep, wheat = build_farm(ntiles)
    ctrl = CBSController(tile_crop, sheep, wheat, n_hands)
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.reset(2)
    prev = {c: 0 for c in ctrl.crops}; bought = {c: 0 for c in ctrl.crops}
    while not env.done:
        obs = env.state[0].observation
        s = obs["private"]["seeds"]
        for c in ctrl.crops:
            if s.get(c, 0) > prev[c]:
                bought[c] += s[c] - prev[c]
            prev[c] = s.get(c, 0)
        a0 = ctrl.decide(obs)
        ctrl.actions.append(a0)
        env.step([a0, PASS])

    fobs = env.state[0].observation
    cash = fobs["farms"][0]["money"]; fshed = fobs["private"]["shed"]; fseeds = fobs["private"]["seeds"]
    shops = fobs["town"]["unlocked_shops"]
    weeds = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and t.get("kind") == "WEED")
    sheep_alive = sum(1 for row in fobs["farms"][0]["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")

    total_rev = 0; eff_num = eff_den = 0
    lines = []
    for p in [c for c in ctrl.crops if c != "WHEAT"] + ["WOOL"]:
        log = ctrl.sell_log.get(p, [])
        sold = sum(q for (_, _, q) in log)
        rev = sum(exact_rev(p, q, v0) for (_, v0, q) in log)
        total_rev += rev
        estr = ""
        if p in bought and p != "WOOL":
            plant = bought[p] - fseeds.get(p, 0)
            deliv = sold + fshed.get(p, 0); ideal = plant * FULL[p]
            if ideal:
                eff_num += deliv; eff_den += ideal
                estr = f" eff={100*deliv/ideal:.0f}%"
        if sold:
            lines.append(f"  {p:8} sold {sold:4} rev ${rev:7} (${rev/sold:6.1f}/u){estr}")
    costs = 3000 + total_rev - cash
    routing_eff = 100 * eff_num / eff_den if eff_den else 0

    n_rows = env.configuration.episodeSteps - 1
    rows = [f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}'
            for a in (ctrl.actions[:n_rows] + [PASS] * max(0, n_rows - len(ctrl.actions)))]
    tape = os.path.join(HERE, "..", "data", f"cbs_{ntiles}t.tape")
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
        top = sorted(set(shops))
        print(f"  seed {seed:3}: Cash720 ${cash:7.0f} eff {routing_eff:.0f}% YARNx{shops.count('YARN_STORE')} weeds {weeds} shops={top} replay=={abs(rc-cash)<1e-6}")
        return cash, routing_eff
    print(f"=== CBS {ntiles} tiles, {n_hands} hands (seed {seed}) === YARNx{shops.count('YARN_STORE')} sheep_alive={sheep_alive} weeds={weeds}")
    for ln in lines:
        print(ln)
    print(f"  routing eff (crops) {routing_eff:.0f}%  | rev ${total_rev} costs ${costs:.0f}")
    print(f"  ACTUAL Cash720 ${cash:.0f}  replay==record {abs(rc-cash)<1e-6} (${rc:.0f}) -> {os.path.basename(tape)}")
    return cash, routing_eff


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worlds":
        print("=== CROSS-WORLD validation: best config (25t) across realized worlds ===")
        cs = []
        for sd in (1, 3, 7, 13, 21, 42, 77, 100):
            c, e = run(25, 10, seed=sd, quiet=True)
            cs.append(c)
        import statistics
        print(f"\nCash720 across {len(cs)} worlds: mean ${statistics.mean(cs):.0f} "
              f"min ${min(cs):.0f} max ${max(cs):.0f} stdev ${statistics.pstdev(cs):.0f} "
              f"(cv {100*statistics.pstdev(cs)/statistics.mean(cs):.0f}%)")
    elif len(sys.argv) > 1:
        run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        for nt in (25, 50, 75, 100):
            run(nt, 10)
