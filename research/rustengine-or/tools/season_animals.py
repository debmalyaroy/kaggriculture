#!/usr/bin/env python
"""Stage 3 : ANIMALS (SHEEP -> WOOL, the high-value scarcity product in a YARN
world). Tests whether the OFFLINE full-state harness can drive the husbandry
LOGISTICS the memory flagged as the wall: buy wheat -> PICKUP at shed -> carry ->
FEED each sheep daily (escape @2 unfed) -> CARE -> HARVEST wool -> DROP -> SELL,
with wheat feed competing against wheat sales.

We read each mover's inventory from full state (offline advantage) to route the
wheat carry. Emits a fixed .tape and validates open-loop replay. Reports:
sheep placed, feed success %, escapes, wool sold + revenue, Cash720, replay==record.

Run: PYTHONPATH=vendor python research/rustengine-or/tools/season_animals.py [n_sheep] [n_hands]
"""
import json
import math
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, MARKET_PARAMS, PRICE_FLOOR

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


def _shape(f, x, T):
    x = max(0.0, x)
    if f == "hinge":
        return x if T <= 0 else (x / T + 8.0 * max(0.0, x / T - 1.0) ** 2)
    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),
            "log": math.log(1.0 + x), "log10": math.log10(1.0 + x)}[f]


def market_price(item, inv):
    p = MARKET_PARAMS[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inv < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], T, T)
        price = base + amp * _shape(p["below_func"], I0 - inv, T)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], T, T)
        price = base - amp * _shape(p["above_func"], inv - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def exact_rev(item, q, v0):
    inv, tot = v0, 0
    for _ in range(q):
        pr = market_price(item, inv)
        tot += pr
        if pr > 1:
            inv += 1
    return tot


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


class SheepController:
    def __init__(self, sheep_tiles, n_hands):
        self.tiles = sheep_tiles          # pasture tiles (near shed)
        self.n_hands = n_hands
        self.actions = []
        self.sell_log = []                # (day, v0, qty) WOOL
        self.day_v0 = {}
        self.feed_ok = 0
        self.feed_attempt = 0

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]
        priv = obs["private"]
        shed = priv["shed"]
        invs = priv["inventories"]        # [farmer, *hands] inventories (full state)
        minv = obs["market"]["inventory"]
        movers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]

        def mover_inv(i):
            return invs[i] if i < len(invs) else {}

        # classify pasture tiles
        need_build, need_place, sheep_present = [], [], []
        for (x, y) in self.tiles:
            t = farm["tiles"][y][x]
            if t is None:
                need_build.append((x, y))
            elif isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t:
                need_place.append((x, y))
            elif isinstance(t, dict) and "animal" in t:
                sheep_present.append(((x, y), t))

        unfed = [(p, t) for (p, t) in sheep_present if not t["fed_today"]]
        wool_ready = [(p, t) for (p, t) in sheep_present if t.get("yield_units", 0) > 0]

        act = {i: ["PASS"] for i in range(len(movers))}
        free = set(range(len(movers)))

        def assign(tpos, opfn):
            # send nearest free mover to tpos; return True if a mover acted/moved
            if not free:
                return False
            m = min(free, key=lambda i: manhattan(movers[i], tpos))
            free.discard(m)
            if movers[m] == tpos:
                act[m] = opfn(m)
            else:
                mv = step_toward(movers[m], tpos)
                act[m] = [mv] if mv else ["PASS"]
            return True

        # priority 1: FEED unfed sheep (carry wheat). A mover with wheat feeds;
        # else it fetches wheat at the shed first.
        for (p, t) in unfed:
            if not free:
                break
            # prefer a free mover that already holds wheat
            holders = [i for i in free if mover_inv(i).get("WHEAT", 0) > 0]
            if holders:
                m = min(holders, key=lambda i: manhattan(movers[i], p))
                free.discard(m)
                self.feed_attempt += 1
                if movers[m] == p:
                    act[m] = ["FEED"]
                    self.feed_ok += 1
                else:
                    mv = step_toward(movers[m], p)
                    act[m] = [mv] if mv else ["PASS"]
            else:
                # nobody holds wheat: send nearest free mover to shed to pick up
                if shed.get("WHEAT", 0) > 0:
                    m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES)))
                    free.discard(m)
                    stile = nearest(movers[m], SHED_TILES)
                    if movers[m] == stile:
                        act[m] = ["PICKUP", "WHEAT", min(6, shed.get("WHEAT", 0))]
                    else:
                        mv = step_toward(movers[m], stile)
                        act[m] = [mv] if mv else ["PASS"]

        # priority 2: harvest wool -> then it auto-drops to shed at dusk
        for (p, t) in wool_ready:
            assign(p, lambda m: ["HARVEST"])
        # priority 3: CARE fed sheep (banks bonus)
        for (p, t) in sheep_present:
            if t["fed_today"] and not t["cared_today"]:
                assign(p, lambda m: ["CARE"])
        # priority 4: place a sheep if we carry one, else build pasture
        for p in need_place:
            def place_or_fetch(m):
                if mover_inv(m).get("SHEEP", 0) > 0:
                    return ["PLACE", "SHEEP"]
                return ["PASS"]
            # a mover holding a sheep places; otherwise a mover fetches from shed
            holders = [i for i in free if mover_inv(i).get("SHEEP", 0) > 0]
            if holders and not assign(p, place_or_fetch):
                pass
            elif shed.get("SHEEP", 0) > 0 and free:
                m = min(free, key=lambda i: manhattan(movers[i], nearest(movers[i], SHED_TILES)))
                free.discard(m)
                stile = nearest(movers[m], SHED_TILES)
                act[m] = ["PICKUP", "SHEEP", 1] if movers[m] == stile else ([step_toward(movers[m], stile)] or ["PASS"])
        for p in need_build:
            assign(p, lambda m: ["BUILD_PASTURE"])

        farmer = act[0]
        hands = [act[i] for i in range(1, len(movers))]

        # ---- market ----
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        # keep shed stocked with wheat for feed (buy up to a buffer)
        if hour == 1:
            wheat_have = shed.get("WHEAT", 0)
            want = len(self.tiles) * 2
            if wheat_have < want and farm["money"] > 200 and len(market) < 10:
                market.append(["BUY_PRODUCT", "WHEAT", min(want - wheat_have, 10)])
        # buy sheep for empty pastures / to reach target herd
        if hour == 2:
            have_sheep = shed.get("SHEEP", 0) + len(sheep_present)
            if have_sheep < len(self.tiles) and farm["money"] >= ANIMALS["SHEEP"]["cost"] and len(market) < 10:
                market.append(["BUY_ANIMAL", "SHEEP", 1])
        # sell wool daily
        if hour == 6:
            self.day_v0[day] = minv["WOOL"]
            q = shed.get("WOOL", 0)
            if q > 0:
                market.append(["SELL", "WOOL", q])
                self.sell_log.append((day, minv["WOOL"], q))

        return {"farmer": farmer, "hands": hands, "market": market}


def run(n_sheep, n_hands):
    # pasture tiles: NW tiles nearest the shed, excluding the 4 shed-access tiles
    cand = [(x, y) for y in range(5) for x in range(5) if (x, y) not in SHED_TILES]
    cand.sort(key=lambda t: manhattan(t, (4, 4)))
    tiles = cand[:n_sheep]
    ctrl = SheepController(tiles, n_hands)
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
    tiles_final = fobs["farms"][0]["tiles"]
    sheep_alive = sum(1 for row in tiles_final for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
    pastures = sum(1 for row in tiles_final for t in row if isinstance(t, dict) and t.get("kind") == "PASTURE" and "animal" not in t)

    wool_sold = sum(q for (_, _, q) in ctrl.sell_log)
    wool_rev = sum(exact_rev("WOOL", q, v0) for (_, v0, q) in ctrl.sell_log)
    feed_rate = 100.0 * ctrl.feed_ok / ctrl.feed_attempt if ctrl.feed_attempt else 0.0

    # tape + replay
    n_rows = env.configuration.episodeSteps - 1
    rows = []
    for i in range(n_rows):
        a = ctrl.actions[i] if i < len(ctrl.actions) else PASS
        rows.append(f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}')
    tape_path = os.path.join(HERE, "..", "data", f"animals_{n_sheep}sheep.tape")
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

    print(f"=== ANIMALS: {n_sheep} sheep, {n_hands} hands (seed {SEED}) ===")
    print(f"sheep alive (final)  : {sheep_alive}   empty pastures: {pastures}")
    print(f"FEED logistics       : {ctrl.feed_ok}/{ctrl.feed_attempt} ops on-tile ({feed_rate:.0f}%)  <-- husbandry-routing signal")
    print(f"WOOL sold            : {wool_sold} units, revenue ${wool_rev} (${wool_rev/wool_sold:.0f}/u)" if wool_sold else "WOOL sold: 0")
    print(f"final shed WOOL      : {fshed.get('WOOL',0)} (unsold scores $0)")
    print(f"ACTUAL Cash720       : ${actual_cash:.0f}")
    print(f"replay==record       : {abs(replay_cash-actual_cash)<1e-6} (${replay_cash:.0f}) -> {os.path.basename(tape_path)}")
    return actual_cash


if __name__ == "__main__":
    n_sheep = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_hands = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(n_sheep, n_hands)
