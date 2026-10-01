#!/usr/bin/env python
"""B3: emit a FULL-SEASON .tape for one representative economy, PLAY it on the
real engine, and compare ACTUAL Cash720 vs the staircase-model's PREDICTED
Cash720. Closes B3.1/B3.2.

Economy (single representative YARN world, seed 42 — has YARN_STORE + PET_CAFE +
FARMERS_MARKET, so CARROT is town-demanded and stays scarce): a single-farmer
CARROT rotation on 6 owned NW tiles. No hire, no land expansion (keeps routing a
simple greedy the farmer can service in 24 turns/day). Opponent = PASS (a fixed
opponent, so the only thing moving market inventory off I0 is town drain + our
own sells — exactly the v0 recursion of spec §7.2, which lets us test the market
model to the dollar).

Method (record -> tape -> replay-validate, spec §9):
  1. Run a closed-loop controller vs PASS on seed 42; RECORD every turn's action.
  2. At each daily SELL, record the pre-sell market inventory v0 and qty sold;
     the model's predicted revenue = sum of exact staircase rev(qty | v0).
  3. Write the recorded actions as a 719-row .tape (farmer\thands\tmarket, ints).
  4. REPLAY the tape open-loop vs PASS on seed 42; confirm identical Cash720.
  5. Report predicted vs actual Cash720 and the gap (the key number).

Run: PYTHONPATH=vendor python research/rustengine-or/tools/season_tape.py
"""
import json
import math
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, MARKET_PARAMS, MARKET_I0, PRICE_FLOOR, pass_agent,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TAPE_OUT = os.path.join(HERE, "..", "data", "season_seed42.tape")
SEED = 42
CROP = "CARROT"
SEED_COST = CROPS[CROP]["seed"]
MAX_YIELD_DAY = CROPS[CROP]["max_yield_day"]  # 3
FIRST_YIELD_DAY = CROPS[CROP]["first_yield_day"]
# 6 owned NW tiles near the dawn spawn (4,4); all x<5,y<5 => NW, owned from start.
TILES = [(4, 4), (3, 4), (2, 4), (4, 3), (3, 3), (2, 3)]
PLANT_HOUR_LIMIT = 5  # only plant early in the day so there is time to water


# ---- exact engine staircase revenue (identical to research/rustengine-or/src/pwl.rs) ----
def _shape(f, x, T):
    x = max(0.0, x)
    if f == "linear": return x
    if f == "sq": return x * x
    if f == "sqrt": return math.sqrt(x)
    if f == "log": return math.log(1.0 + x)
    if f == "log10": return math.log10(1.0 + x)
    if f == "hinge":
        if T <= 0: return x
        u = x / T
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


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


# ---------------------------- controller ----------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(cur, tgt):
    cx, cy = cur
    tx, ty = tgt
    if cx != tx:
        return "EAST" if tx > cx else "WEST"
    if cy != ty:
        return "SOUTH" if ty > cy else "NORTH"
    return None


class Controller:
    """Closed-loop; records actions + sell (v0, qty) log."""

    def __init__(self):
        self.actions = []      # per-step action dict for player 0
        self.sell_log = []     # (day, v0, qty, price_credited_expected)
        self.seed_buys = 0
        self._sold_today = set()
        self._bought_today = set()

    def agent(self, obs, config=None):
        day = obs["day"]
        hour = obs["hour"]
        farm = obs["farms"][obs["player"]]
        priv = obs["private"]
        seeds = priv.get("seeds", {})
        shed = priv.get("shed", {})
        market_inv = obs["market"]["inventory"]
        pos = tuple(farm["farmer"])
        tiles = farm["tiles"]

        market = []
        # --- daily market ops (once per day) ---
        if day not in self._bought_today:
            self._bought_today.add(day)
            # keep a seed buffer >= number of managed tiles
            have = seeds.get(CROP, 0)
            deficit = len(TILES) - have
            if deficit > 0 and farm["money"] >= deficit * SEED_COST:
                market.append(["BUY_SEED", CROP, deficit])
                self.seed_buys += deficit
        if day not in self._sold_today:
            qty = shed.get(CROP, 0)
            if qty > 0:
                v0 = market_inv[CROP]
                market.append(["SELL", CROP, qty])
                self.sell_log.append((day, v0, qty))
                self._sold_today.add(day)

        # --- farmer op: greedy service of managed tiles ---
        farmer = ["PASS"]
        # find tiles needing an op
        need = []  # (dist, order_idx, op, tile)
        for i, (x, y) in enumerate(TILES):
            t = tiles[y][x]
            op = None
            if t is None:
                if seeds.get(CROP, 0) > 0 and hour <= PLANT_HOUR_LIMIT:
                    op = "PLANT"
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == CROP:
                age = day - t["planted_day"]
                if age >= MAX_YIELD_DAY and t.get("yield_units", 0) > 0 and t["watered_today"]:
                    op = "HARVEST"
                elif not t["watered_today"]:
                    op = "WATER"
            if op:
                need.append((manhattan(pos, (x, y)), i, op, (x, y)))
        if need:
            need.sort(key=lambda r: (r[0], r[1]))
            _, _, op, tgt = need[0]
            if pos == tgt:
                farmer = ["PLANT", CROP] if op == "PLANT" else [op]
            else:
                mv = step_toward(pos, tgt)
                farmer = [mv] if mv else ["PASS"]

        action = {"farmer": farmer, "hands": [], "market": market}
        self.actions.append(action)
        return action


# ---------------------------- run + validate ----------------------------
def tape_player(actions):
    calls = {"i": 0}

    def agent(obs, config=None):
        i = calls["i"]
        calls["i"] += 1
        if i < len(actions):
            return actions[i]
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent


def to_tape_rows(actions, n_rows):
    rows = []
    for i in range(n_rows):
        a = actions[i] if i < len(actions) else {"farmer": ["PASS"], "hands": [], "market": []}
        farmer = json.dumps(a.get("farmer", ["PASS"]))
        hands = json.dumps(a.get("hands", []))
        mkt = json.dumps(a.get("market", []))
        rows.append(f"{farmer}\t{hands}\t{mkt}")
    return rows


def main():
    # 1. closed-loop record
    ctrl = Controller()
    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.run([ctrl.agent, pass_agent])
    actual_cash = env.steps[-1][0]["reward"]
    n_action_rows = env.configuration.episodeSteps - 1  # 719

    # count weeds in final board (mis-timed watering => lost yield)
    final_tiles = env.steps[-1][0]["observation"]["farms"][0]["tiles"]
    weeds = sum(1 for row in final_tiles for t in row if isinstance(t, dict) and t.get("kind") == "WEED")

    # 2. model prediction from the sell log (staircase revenue)
    pred_rev = sum(exact_rev(CROP, q, v0) for (_, v0, q) in ctrl.sell_log)
    total_qty = sum(q for (_, _, q) in ctrl.sell_log)
    seed_cost_total = ctrl.seed_buys * SEED_COST
    M0 = 3000
    pred_cash = M0 + pred_rev - seed_cost_total

    # 3a. export the realized market scenario for the Rust multi-day model (B1.4)
    scen = {
        "seed": SEED, "crop": CROP, "m0": M0,
        "seed_cost_total": seed_cost_total,
        "actual_cash": actual_cash,
        "sells": [{"day": d, "v0": v0, "qty": q} for (d, v0, q) in ctrl.sell_log],
    }
    with open(os.path.join(HERE, "..", "data", "scenario_seed42.json"), "w") as f:
        json.dump(scen, f, indent=2)

    # 3. write tape
    rows = to_tape_rows(ctrl.actions, n_action_rows)
    with open(TAPE_OUT, "w") as f:
        f.write("\n".join(rows) + "\n")

    # 4. replay open-loop from the tape
    replay_actions = []
    for r in rows:
        farmer, hands, mkt = r.split("\t")
        replay_actions.append({
            "farmer": json.loads(farmer),
            "hands": json.loads(hands),
            "market": json.loads(mkt),
        })
    env2 = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env2.run([tape_player(replay_actions), pass_agent])
    replay_cash = env2.steps[-1][0]["reward"]

    # 5. report
    print("=== B3: full-season tape — predicted vs actual Cash720 ===")
    print(f"world seed         : {SEED} (YARN world; shops="
          f"{env.steps[-1][0]['observation']['town']['unlocked_shops']})")
    print(f"economy            : single-farmer CARROT x{len(TILES)} tiles, NW only, no hire/land")
    print(f"action rows in tape: {n_action_rows}  ->  {os.path.normpath(TAPE_OUT)}")
    print(f"carrot sells        : {len(ctrl.sell_log)} daily batches, {total_qty} units total")
    print(f"seeds bought        : {ctrl.seed_buys}  (cost ${seed_cost_total})")
    print(f"weed tiles (final)  : {weeds}")
    print()
    print(f"MODEL predicted revenue (staircase) : ${pred_rev}")
    print(f"MODEL predicted Cash720             : ${pred_cash}")
    print(f"ACTUAL Cash720 (closed-loop record) : ${actual_cash:.0f}")
    print(f"ACTUAL Cash720 (open-loop replay)   : ${replay_cash:.0f}")
    print()
    gap = actual_cash - pred_cash
    print(f"GAP (actual - predicted Cash720)    : ${gap:.0f}  "
          f"({100.0*gap/pred_cash:+.2f}%)")
    print(f"replay == record                    : {abs(replay_cash - actual_cash) < 1e-6}"
          f"  (open-loop tape reproduces the plan)")
    if abs(replay_cash - actual_cash) > 1e-6:
        print("  WARNING: tape replay diverged from the recorded run!")

    # dump a small per-day sell trace for the report
    print("\n  first 8 daily carrot sells (day, v0=mkt_inv, qty, staircase_rev, $/u):")
    for (d, v0, q) in ctrl.sell_log[:8]:
        r = exact_rev(CROP, q, v0)
        print(f"    day {d:2}  v0={v0:6}  qty={q:3}  rev=${r:5}  (${r/q:5.1f}/u)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
