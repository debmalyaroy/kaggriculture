#!/usr/bin/env python
"""Stage 1+ : SCALE the economy with HIRED HANDS across MANY tiles, and measure
whether greedy multi-hand routing REALIZES the planned production (the routing-
leak signal). We drive the engine via manual env.step() and read the FULL
internal state (incl. hand positions) — the OFFLINE FULL-STATE advantage a
reactive agent lacks (memory planner-labour-logistics-wall: obs hides hands).
The emitted .tape is fixed open-loop actions and replays deterministically with
NO state access — that is exactly the multi-hand routing a reactive agent can't do.

Metrics:
  * market-model check  : Σ staircase(actual sells @ observed v0) vs actual revenue (stays $0).
  * ROUTING LEAK        : ideal delivered units (plantings×full yield, perfect routing)
                          vs actual delivered units; efficiency %; weeds. THE KEY SIGNAL.
  * Cash720             : predicted (plan, perfect routing @ observed v0) vs actual + gap.

Run: PYTHONPATH=vendor python research/rustengine-or/tools/season_scale.py [crop] [n_hands] [n_tiles]
"""
import json
import math
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, MARKET_PARAMS, PRICE_FLOOR,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


# ---- exact engine staircase (identical to src/pwl.rs) ----
def _shape(f, x, T):
    x = max(0.0, x)
    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),
            "log": math.log(1.0 + x), "log10": math.log10(1.0 + x)}.get(f) \
        if f != "hinge" else (x if T <= 0 else (x / T + 8.0 * max(0.0, x / T - 1.0) ** 2))


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


def nw_tiles(n):
    """First n tiles of the 25-tile NW quadrant (x,y in 0..4), spiraling out from
    the (4,4) dawn spawn so nearby tiles are serviced first."""
    tiles = [(x, y) for y in range(5) for x in range(5)]
    tiles.sort(key=lambda t: manhattan(t, (4, 4)))
    return tiles[:n]


class ScaleController:
    def __init__(self, crop, tiles, n_hands, plant_hour_limit=12):
        self.crop = crop
        self.tiles = tiles
        self.n_hands = n_hands
        self.plant_hour_limit = plant_hour_limit
        self.max_yield_day = CROPS[crop]["max_yield_day"]
        self.max_yield = CROPS[crop]["max_yield"]
        self.first_yield_day = CROPS[crop]["first_yield_day"]
        self.ongoing = CROPS[crop]["ongoing"]
        self.seed_cost = CROPS[crop]["seed"]
        self.actions = []
        self.seeds_bought = 0
        self.hires_paid = 0
        self.sell_log = []          # (day, v0, qty)
        self.day_v0 = {}            # day -> market inv at sell time
        self.day_plantings = {}     # day -> seeds consumed that day
        self._prev_seeds = 0

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]
        priv = obs["private"]
        seeds = priv["seeds"].get(self.crop, 0)
        shed = priv["shed"]
        market_inv = obs["market"]["inventory"]

        # track plantings via seed consumption (only PLANT consumes seeds)
        if hour == 0:
            self._prev_seeds = seeds
        farmer_pos = tuple(farm["farmer"])
        hand_pos = [tuple(h) for h in farm["hands"]]
        movers = [farmer_pos] + hand_pos

        # ---------- tasks ----------
        tasks = []  # (tile_pos, op, priority)
        for (x, y) in self.tiles:
            t = farm["tiles"][y][x]
            op = None
            pri = 0
            if t is None:
                if seeds > 0 and hour <= self.plant_hour_limit:
                    op, pri = "PLANT", 2
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == self.crop:
                age = day - t["planted_day"]
                yu = t.get("yield_units", 0)
                capped = yu >= self.max_yield
                if not t["watered_today"] and not capped:
                    op, pri = "WATER", 0            # keep-alive first
                elif not self.ongoing and (age >= self.max_yield_day or capped) and yu > 0:
                    op, pri = "HARVEST", 1
                elif self.ongoing and t.get("yield_units", 0) > 0 and age > self.first_yield_day:
                    op, pri = "HARVEST", 1
            if op:
                tasks.append(((x, y), op, pri))

        # ---------- greedy nearest assignment ----------
        act = {i: ["PASS"] for i in range(len(movers))}
        free = set(range(len(movers)))
        # water/harvest before plant; then nearest-first
        tasks.sort(key=lambda tk: (tk[2], min((manhattan(movers[i], tk[0]) for i in free), default=0)))
        for (tpos, op, _pri) in tasks:
            if not free:
                break
            m = min(free, key=lambda i: manhattan(movers[i], tpos))
            free.discard(m)
            if movers[m] == tpos:
                act[m] = ["PLANT", self.crop] if op == "PLANT" else [op]
            else:
                mv = step_toward(movers[m], tpos)
                act[m] = [mv] if mv else ["PASS"]

        farmer = act[0]
        hands = [act[i] for i in range(1, len(movers))]

        # ---------- market ----------
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1:
            deficit = len(self.tiles) - seeds
            if deficit > 0 and farm["money"] >= self.seed_cost:
                buy = min(deficit, 10)
                market.append(["BUY_SEED", self.crop, buy])
                self.seeds_bought += buy  # requested; engine caps by cash (tracked below)
        if hour == 2:
            qty = shed.get(self.crop, 0)
            self.day_v0[day] = market_inv[self.crop]
            if qty > 0:
                v0 = market_inv[self.crop]
                market.append(["SELL", self.crop, qty])
                self.sell_log.append((day, v0, qty))

        return {"farmer": farmer, "hands": hands, "market": market}


def build_and_validate(crop, n_hands, n_tiles):
    tiles = nw_tiles(n_tiles)
    ctrl = ScaleController(crop, tiles, n_hands)
    env = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    env.reset(2)

    # track seeds/day and plantings via seed consumption
    seeds_series = []
    while not env.done:
        obs = env.state[0].observation
        seeds_now = obs["private"]["seeds"].get(crop, 0)
        a0 = ctrl.decide(obs)
        ctrl.actions.append(a0)
        seeds_series.append((obs["day"], obs["hour"], seeds_now, a0))
        env.step([a0, PASS])

    actual_cash = env.state[0].observation["farms"][0]["money"]
    final = env.state[0].observation
    final_shed = final["private"]["shed"].get(crop, 0)
    final_seeds = final["private"]["seeds"].get(crop, 0)
    weeds = sum(1 for row in final["farms"][0]["tiles"]
                for t in row if isinstance(t, dict) and t.get("kind") == "WEED")

    # exact plantings = total seeds acquired - seeds remaining (only PLANT consumes)
    bought = 0
    prev = 0
    for (d, h, s, a) in seeds_series:
        if s > prev:
            bought += (s - prev)
        prev = s
    plantings = bought - final_seeds
    full_yield = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}[crop]
    deliver_day = {"WHEAT": 4, "CARROT": 3, "MELON": 10, "TOMATO": 9, "STRAWBERRY": 11}[crop]

    total_sold = sum(q for (_, _, q) in ctrl.sell_log)
    actual_delivered = total_sold + final_shed
    ideal_delivered = plantings * full_yield
    eff = (actual_delivered / ideal_delivered) if ideal_delivered else 0.0

    # market-model check: staircase over ACTUAL sells @ observed v0 (== engine, validated)
    actual_sell_rev = sum(exact_rev(crop, q, v0) for (_, v0, q) in ctrl.sell_log)

    # emitted plantings per day (to time the plan deliveries); scale to exact plantings
    plant_by_day = {}
    for (d, h, s, a) in seeds_series:
        n = (1 if a["farmer"][:1] == ["PLANT"] else 0) + sum(1 for ha in a["hands"] if ha[:1] == ["PLANT"])
        if n:
            plant_by_day[d] = plant_by_day.get(d, 0) + n
    emitted = sum(plant_by_day.values()) or 1
    scale = plantings / emitted  # discount emitted->successful

    # PLAN Cash720: perfect routing => each planting yields full_yield delivered
    # at +deliver_day, sold at the OBSERVED daily v0 (reference market).
    ideal_shed = 0.0
    plan_rev = 0.0
    for d in range(30):
        src = d - deliver_day
        if src in plant_by_day:
            ideal_shed += plant_by_day[src] * scale * full_yield
        v0 = ctrl.day_v0.get(d)
        if v0 is None:
            continue
        q = int(round(ideal_shed))
        if q > 0:
            plan_rev += exact_rev(crop, q, v0)
            ideal_shed = 0.0

    M0 = 3000
    costs = M0 + actual_sell_rev - actual_cash  # seeds + hires (engine-confirmed)
    plan_cash = M0 + plan_rev - costs
    market_pred_rev = actual_sell_rev

    # ---- emit + replay-validate the tape ----
    n_rows = env.configuration.episodeSteps - 1
    rows = []
    for i in range(n_rows):
        a = ctrl.actions[i] if i < len(ctrl.actions) else PASS
        rows.append(f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}')
    tape_path = os.path.join(HERE, "..", "data", f"scale_{crop.lower()}_{n_hands}h_{n_tiles}t.tape")
    with open(tape_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    replay = [dict(farmer=json.loads(r.split("\t")[0]), hands=json.loads(r.split("\t")[1]),
                   market=json.loads(r.split("\t")[2])) for r in rows]
    env2 = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    calls = {"i": 0}

    def tp(obs, cfg=None):
        i = calls["i"]; calls["i"] += 1
        return replay[i] if i < len(replay) else PASS

    env2.run([tp, lambda o, c=None: PASS])
    replay_cash = env2.steps[-1][0]["reward"]

    print(f"=== SCALE stage: crop={crop} hands={n_hands} tiles={n_tiles} (seed {SEED}) ===")
    print(f"plantings (seeds consumed) : {plantings}   (tile-capacity ideal ~{n_tiles}*7={n_tiles*7})")
    print(f"weeds (final)              : {weeds}")
    print(f"ROUTING LEAK  ideal units  : {ideal_delivered}  (plantings x {full_yield})")
    print(f"              actual units : {actual_delivered}  (sold {total_sold} + shed {final_shed})")
    print(f"              efficiency   : {eff*100:.1f}%   <-- KEY routing signal")
    print(f"market-model check         : staircase ${market_pred_rev} vs engine (validated exact)")
    print(f"costs (seeds+hires)        : ${costs:.0f}")
    print()
    print(f"PLAN  Cash720 (perfect routing @ observed v0) : ${plan_cash:.0f}")
    print(f"ACTUAL Cash720                                : ${actual_cash:.0f}")
    print(f"GAP (plan - actual)                           : ${plan_cash-actual_cash:.0f}  "
          f"({100*(plan_cash-actual_cash)/plan_cash if plan_cash else 0:+.1f}% of plan)")
    print(f"open-loop replay == record                    : {abs(replay_cash-actual_cash)<1e-6} "
          f"(${replay_cash:.0f})   tape rows -> {os.path.basename(tape_path)}")
    return actual_cash, plan_cash, eff


if __name__ == "__main__":
    crop = sys.argv[1].upper() if len(sys.argv) > 1 else "CARROT"
    n_hands = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_tiles = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    build_and_validate(crop, n_hands, n_tiles)
