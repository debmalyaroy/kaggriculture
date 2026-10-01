#!/usr/bin/env python
"""Stage 3-fixed: WOOL DONE RIGHT. Three optimizer fixes over the naive 6-sheep
breakeven ($3,448), each toggleable so we can ablate its $ impact:
  * DON'T-DUMP : sell wool only while marginal price >= threshold (withhold tail),
                 liquidate in the last days (unsold = $0).
  * RIGHT-SIZE : fewer sheep matched to absorbable wool demand (realized world has
                 NO YARN -> ~1/day town drain -> ~89-unit lifetime wool capacity).
  * GROW-FEED  : feed from wheat GROWN on a few tiles (~$2.50/u) not bought (~$25/u).

Greedy routing throughout (isolating optimizer value from routing). Reads full
state (offline), emits a fixed .tape, validates replay==record.

Run: PYTHONPATH=vendor python season_wool.py <n_sheep> <dump|hold> <buy|grow>
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
LIQUIDATE_DAY = 27
WOOL_THRESH = default_thresh("WOOL", 0.5)  # $100


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(cur, tgt):
    if cur[0] != tgt[0]:
        return "EAST" if tgt[0] > cur[0] else "WEST"
    if cur[1] != tgt[1]:
        return "SOUTH" if tgt[1] > cur[1] else "NORTH"
    return None


def nearest(pos, tiles):
    return min(tiles, key=lambda t: manhattan(pos, t))


class WoolController:
    def __init__(self, n_sheep, dont_dump, grow_feed, n_hands=8):
        self.n_sheep = n_sheep
        self.dont_dump = dont_dump
        self.grow_feed = grow_feed
        self.n_hands = n_hands
        cand = [(x, y) for y in range(5) for x in range(5) if (x, y) not in SHED_TILES]
        cand.sort(key=lambda t: manhattan(t, (4, 4)))
        self.sheep_tiles = cand[:n_sheep]
        self.wheat_tiles = cand[n_sheep:n_sheep + (max(2, n_sheep + 1) if grow_feed else 0)]
        self.actions = []
        self.sell_log = []
        self.day_v0 = {}
        self.wheat_bought = 0
        self.feed_ok = 0
        self.feed_attempt = 0

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]
        priv = obs["private"]
        shed = priv["shed"]
        invs = priv["inventories"]
        minv = obs["market"]["inventory"]
        seeds = priv["seeds"]
        movers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]

        def minv_of(i):
            return invs[i] if i < len(invs) else {}

        # classify sheep tiles
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

        # wheat tiles tasks
        wheat_tasks = []  # (pos, op)
        for (x, y) in self.wheat_tiles:
            t = farm["tiles"][y][x]
            if t is None:
                if seeds.get("WHEAT", 0) > 0 and hour <= 12:
                    wheat_tasks.append(((x, y), ["PLANT", "WHEAT"]))
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == "WHEAT":
                age = day - t["planted_day"]
                yu = t.get("yield_units", 0)
                if not t["watered_today"] and yu < CROPS["WHEAT"]["max_yield"]:
                    wheat_tasks.append(((x, y), ["WATER"]))
                elif age >= CROPS["WHEAT"]["max_yield_day"] and yu > 0:
                    wheat_tasks.append(((x, y), ["HARVEST"]))

        act = {i: ["PASS"] for i in range(len(movers))}
        free = set(range(len(movers)))

        def assign(tpos, action):
            if not free:
                return
            m = min(free, key=lambda i: manhattan(movers[i], tpos))
            free.discard(m)
            act[m] = action if movers[m] == tpos else ([step_toward(movers[m], tpos)] or ["PASS"])

        # P1: FEED (carry wheat; fetch from shed if none held)
        for (p, t) in unfed:
            if not free:
                break
            holders = [i for i in free if minv_of(i).get("WHEAT", 0) > 0]
            if holders:
                m = min(holders, key=lambda i: manhattan(movers[i], p))
                free.discard(m)
                self.feed_attempt += 1
                if movers[m] == p:
                    act[m] = ["FEED"]; self.feed_ok += 1
                else:
                    act[m] = [step_toward(movers[m], p)] or ["PASS"]
            elif shed.get("WHEAT", 0) > 0:
                m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES)))
                free.discard(m)
                st = nearest(movers[m], SHED_TILES)
                act[m] = ["PICKUP", "WHEAT", min(self.n_sheep, shed.get("WHEAT", 0))] if movers[m] == st else ([step_toward(movers[m], st)] or ["PASS"])
        # P2: water/harvest wheat (feed supply)
        for (pos, action) in wheat_tasks:
            if action[0] == "WATER":
                assign(pos, action)
        for (pos, action) in wheat_tasks:
            if action[0] == "HARVEST":
                assign(pos, action)
        # P3: harvest wool
        for (p, t) in wool_ready:
            assign(p, ["HARVEST"])
        # P4: care
        for (p, t) in sheep_present:
            if t["fed_today"] and not t["cared_today"]:
                assign(p, ["CARE"])
        # P5: plant wheat
        for (pos, action) in wheat_tasks:
            if action[0] == "PLANT":
                assign(pos, action)
        # P6: place sheep / build
        for p in need_place:
            holders = [i for i in free if minv_of(i).get("SHEEP", 0) > 0]
            if holders:
                m = min(holders, key=lambda i: manhattan(movers[i], p))
                free.discard(m)
                act[m] = ["PLACE", "SHEEP"] if movers[m] == p else ([step_toward(movers[m], p)] or ["PASS"])
            elif shed.get("SHEEP", 0) > 0 and free:
                m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES)))
                free.discard(m)
                st = nearest(movers[m], SHED_TILES)
                act[m] = ["PICKUP", "SHEEP", 1] if movers[m] == st else ([step_toward(movers[m], st)] or ["PASS"])
        for p in need_build:
            assign(p, ["BUILD_PASTURE"])

        farmer = act[0]
        hands = [act[i] for i in range(1, len(movers))]

        # ---- market ----
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1 and self.grow_feed:
            deficit = len(self.wheat_tiles) - seeds.get("WHEAT", 0)
            if deficit > 0 and len(market) < 10:
                market.append(["BUY_SEED", "WHEAT", min(deficit, 10)])
        # feed wheat: bootstrap (grow mode, days 0-5 before wheat matures) or buy (buy mode)
        if hour == 1:
            need_boot = (self.grow_feed and day <= 5) or (not self.grow_feed)
            if need_boot and shed.get("WHEAT", 0) < self.n_sheep and farm["money"] > 200 and len(market) < 10:
                market.append(["BUY_PRODUCT", "WHEAT", self.n_sheep])
                self.wheat_bought += self.n_sheep
        if hour == 2:
            have = shed.get("SHEEP", 0) + len(sheep_present)
            if have < self.n_sheep and farm["money"] >= ANIMALS["SHEEP"]["cost"] and len(market) < 10:
                market.append(["BUY_ANIMAL", "SHEEP", 1])
        if hour == 6:
            self.day_v0[day] = minv["WOOL"]
            havewool = shed.get("WOOL", 0)
            if havewool > 0:
                if self.dont_dump and day < LIQUIDATE_DAY:
                    q = dont_dump_qty("WOOL", minv["WOOL"], havewool, WOOL_THRESH)
                else:
                    q = havewool  # naive dump / end-game liquidation
                if q > 0:
                    market.append(["SELL", "WOOL", q])
                    self.sell_log.append((day, minv["WOOL"], q))

        return {"farmer": farmer, "hands": hands, "market": market}


def run(n_sheep, dont_dump, grow_feed, n_hands=8):
    ctrl = WoolController(n_sheep, dont_dump, grow_feed, n_hands)
    env = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    env.reset(2)
    while not env.done:
        obs = env.state[0].observation
        a0 = ctrl.decide(obs)
        ctrl.actions.append(a0)
        env.step([a0, PASS])

    fobs = env.state[0].observation
    actual_cash = fobs["farms"][0]["money"]
    fshed = fobs["private"]["shed"]
    sheep_alive = sum(1 for row in fobs["farms"][0]["tiles"]
                      for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
    wool_sold = sum(q for (_, _, q) in ctrl.sell_log)
    wool_rev = sum(exact_rev("WOOL", q, v0) for (_, v0, q) in ctrl.sell_log)

    n_rows = env.configuration.episodeSteps - 1
    rows = [f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}'
            for a in (ctrl.actions[:n_rows] + [PASS] * max(0, n_rows - len(ctrl.actions)))]
    label = f"{n_sheep}sheep_{'hold' if dont_dump else 'dump'}_{'grow' if grow_feed else 'buy'}"
    tape_path = os.path.join(HERE, "..", "data", f"wool_{label}.tape")
    with open(tape_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    replay = [dict(farmer=json.loads(r.split("\t")[0]), hands=json.loads(r.split("\t")[1]),
                   market=json.loads(r.split("\t")[2])) for r in rows]
    env2 = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    calls = {"i": 0}
    def tp(o, c=None):
        i = calls["i"]; calls["i"] += 1
        return replay[i] if i < len(replay) else PASS
    env2.run([tp, lambda o, c=None: PASS])
    replay_cash = env2.steps[-1][0]["reward"]

    per = wool_rev / wool_sold if wool_sold else 0
    print(f"[{label}] sheep_alive={sheep_alive} wool_sold={wool_sold} wool_rev=${wool_rev} "
          f"(${per:.0f}/u) feed={ctrl.feed_ok}/{ctrl.feed_attempt} boot_wheat={ctrl.wheat_bought} "
          f"| Cash720=${actual_cash:.0f} replay==rec={abs(replay_cash-actual_cash)<1e-6}")
    return actual_cash


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(int(sys.argv[1]), sys.argv[2] == "hold", sys.argv[3] == "grow")
    else:
        print("=== WOOL ablation (isolating each optimizer fix; seed 42) ===")
        run(6, False, False)   # naive baseline (dump, buy)  ~ $3448
        run(6, True, False)    # + DON'T-DUMP
        run(3, True, False)    # + RIGHT-SIZE (3 sheep)
        run(3, True, True)     # + GROW-FEED  = full optimizer