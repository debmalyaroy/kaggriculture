#!/usr/bin/env python
"""Stage 1+2 : DIVERSIFIED multi-crop economy + LAND expansion. Each product is
sized to its market knee (single products cap for one seat — see season_scale
findings: melon ~144u/$30k regardless of tiles), so revenue STACKS across
products instead of saturating one. Land (BUY_LAND, order-forced NE->SW->SE)
unlocks more quadrants -> more tiles -> more product lines.

Same harness as season_scale: drive via manual env.step reading FULL state
(offline hand-position advantage); emit a fixed open-loop .tape and validate it
replays deterministically. Reports per-product delivered units + total Cash720
(plan vs actual) + routing efficiency.

Run: PYTHONPATH=vendor python research/rustengine-or/tools/season_diverse.py
"""
import json
import math
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, MARKET_PARAMS, PRICE_FLOOR

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
FULL_YIELD = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}
DELIVER_DAY = {"WHEAT": 4, "CARROT": 3, "MELON": 10, "TOMATO": 9, "STRAWBERRY": 11}


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


def quadrant_tiles(q):
    """5x5 tiles of a quadrant of the 10x10 board (half=5)."""
    xr = range(0, 5) if q in ("NW", "SW") else range(5, 10)
    yr = range(0, 5) if q in ("NW", "NE") else range(5, 10)
    return [(x, y) for y in yr for x in xr]


class DiverseController:
    def __init__(self, tile_crop, n_hands, land_plan):
        # tile_crop: dict (x,y)->crop ; land_plan: dict day->True (attempt BUY_LAND)
        self.tile_crop = tile_crop
        self.crops = sorted(set(tile_crop.values()))
        self.n_hands = n_hands
        self.land_plan = land_plan
        self.actions = []
        self.sell_log = {c: [] for c in self.crops}     # crop -> [(day,v0,qty)]
        self.day_v0 = {c: {} for c in self.crops}
        self.plant_ops = {c: 0 for c in self.crops}
        self._sold = set()

    def decide(self, obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][0]
        priv = obs["private"]
        seeds = priv["seeds"]
        shed = priv["shed"]
        minv = obs["market"]["inventory"]
        movers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]

        # ---- tasks over all managed tiles ----
        tasks = []
        for (x, y), crop in self.tile_crop.items():
            t = farm["tiles"][y][x]
            cd = CROPS[crop]
            op = pri = None
            if t == "LOCKED":
                continue
            if t is None:
                if seeds.get(crop, 0) > 0 and hour <= 12:
                    op, pri = ("PLANT", crop), 2
            elif isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop:
                age = day - t["planted_day"]
                yu = t.get("yield_units", 0)
                capped = yu >= cd["max_yield"]
                if not t["watered_today"] and not capped:
                    op, pri = ("WATER", None), 0
                elif not cd["ongoing"] and (age >= cd["max_yield_day"] or capped) and yu > 0:
                    op, pri = ("HARVEST", None), 1
                elif cd["ongoing"] and yu > 0 and age > cd["first_yield_day"]:
                    op, pri = ("HARVEST", None), 1
            if op:
                tasks.append(((x, y), op, pri))

        # ---- greedy nearest assignment ----
        act = {i: ["PASS"] for i in range(len(movers))}
        free = set(range(len(movers)))
        tasks.sort(key=lambda tk: (tk[2], min((manhattan(movers[i], tk[0]) for i in free), default=0)))
        for (tpos, (op, arg), _pri) in tasks:
            if not free:
                break
            m = min(free, key=lambda i: manhattan(movers[i], tpos))
            free.discard(m)
            if movers[m] == tpos:
                if op == "PLANT":
                    act[m] = ["PLANT", arg]
                    self.plant_ops[arg] += 1
                else:
                    act[m] = [op]
            else:
                mv = step_toward(movers[m], tpos)
                act[m] = [mv] if mv else ["PASS"]

        farmer = act[0]
        hands = [act[i] for i in range(1, len(movers))]

        # ---- market: hire, land, seeds, sells ----
        market = []
        if hour == 0 and self.n_hands > 0:
            market += [["HIRE"]] * min(self.n_hands, 10)
        if hour == 1:
            # seed buffer per crop = number of that crop's tiles (capped by 10 orders)
            need = {}
            for (x, y), c in self.tile_crop.items():
                need[c] = need.get(c, 0) + 1
            for c in self.crops:
                deficit = need.get(c, 0) - seeds.get(c, 0)
                if deficit > 0 and len(market) < 10 and farm["money"] >= CROPS[c]["seed"]:
                    market.append(["BUY_SEED", c, min(deficit, 20)])
        if hour == 3 and self.land_plan.get(day):
            market.append(["BUY_LAND"])
        # sells: one product per hour slot to keep <=10 orders and record v0
        for k, c in enumerate(self.crops):
            slot = 4 + k
            if hour == slot:
                self.day_v0[c][day] = minv[c]
                q = shed.get(c, 0)
                if q > 0:
                    market.append(["SELL", c, q])
                    self.sell_log[c].append((day, minv[c], q))

        return {"farmer": farmer, "hands": hands, "market": market}


def run(tile_crop, n_hands, land_plan, label):
    ctrl = DiverseController(tile_crop, n_hands, land_plan)
    env = make("kaggriculture", configuration={"seed": SEED}, debug=True)
    env.reset(2)
    seed_hist = {c: 0 for c in ctrl.crops}
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
    actual_cash = fobs["farms"][0]["money"]
    fshed = fobs["private"]["shed"]
    fseeds = fobs["private"]["seeds"]
    weeds = sum(1 for row in fobs["farms"][0]["tiles"]
                for t in row if isinstance(t, dict) and t.get("kind") == "WEED")
    unlocked = fobs["farms"][0]["unlocked_quadrants"]

    # per-crop accounting
    print(f"=== DIVERSE {label}: hands={n_hands} seed {SEED} ===")
    print(f"unlocked quadrants (final): {unlocked}   weeds={weeds}")
    total_actual_rev = 0
    total_plan_rev = 0
    print(f"{'crop':11} {'plant':>5} {'ideal':>6} {'deliv':>6} {'eff%':>5} {'sold':>5} {'rev$':>8} {'$/u':>6}")
    for c in ctrl.crops:
        plantings = bought[c] - fseeds.get(c, 0)
        sold = sum(q for (_, _, q) in ctrl.sell_log[c])
        deliv = sold + fshed.get(c, 0)
        ideal = plantings * FULL_YIELD[c]
        eff = 100.0 * deliv / ideal if ideal else 0.0
        rev = sum(exact_rev(c, q, v0) for (_, v0, q) in ctrl.sell_log[c])
        total_actual_rev += rev
        # plan: perfect routing, deliveries timed from plant days, sold at observed v0
        pbd = {}
        for (d, _h) in []:
            pass
        # reconstruct plant-by-day from actions
        for i, a in enumerate(ctrl.actions):
            dday = i // 24
            if a["farmer"][:2] == ["PLANT", c]:
                pbd[dday] = pbd.get(dday, 0) + 1
            for ha in a["hands"]:
                if ha[:2] == ["PLANT", c]:
                    pbd[dday] = pbd.get(dday, 0) + 1
        emit = sum(pbd.values()) or 1
        scale = plantings / emit
        ish = 0.0
        prev_rev = 0.0
        for d in range(30):
            src = d - DELIVER_DAY[c]
            if src in pbd:
                ish += pbd[src] * scale * FULL_YIELD[c]
            v0 = ctrl.day_v0[c].get(d)
            if v0 is None:
                continue
            q = int(round(ish))
            if q > 0:
                prev_rev += exact_rev(c, q, v0)
                ish = 0.0
        total_plan_rev += prev_rev
        per = rev / sold if sold else 0.0
        print(f"{c:11} {plantings:5} {ideal:6} {deliv:6} {eff:5.0f} {sold:5} {rev:8} {per:6.1f}")

    M0 = 3000
    costs = M0 + total_actual_rev - actual_cash
    plan_cash = M0 + total_plan_rev - costs

    # tape + replay validation
    n_rows = env.configuration.episodeSteps - 1
    rows = []
    for i in range(n_rows):
        a = ctrl.actions[i] if i < len(ctrl.actions) else PASS
        rows.append(f'{json.dumps(a["farmer"])}\t{json.dumps(a["hands"])}\t{json.dumps(a["market"])}')
    tape_path = os.path.join(HERE, "..", "data", f"diverse_{label}.tape")
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

    print(f"\ntotal revenue ${total_actual_rev}  costs ${costs:.0f}")
    print(f"PLAN  Cash720 : ${plan_cash:.0f}")
    print(f"ACTUAL Cash720: ${actual_cash:.0f}")
    print(f"GAP           : ${plan_cash-actual_cash:.0f}  ({100*(plan_cash-actual_cash)/plan_cash if plan_cash else 0:+.1f}%)")
    print(f"replay==record: {abs(replay_cash-actual_cash)<1e-6} (${replay_cash:.0f})  -> {os.path.basename(tape_path)}")
    return actual_cash


def build_plan(quads):
    """MELON is market-capped (~144u for one seat) so cap melon tiles to NW only
    (12); everything else CARROT (3-day cycle, cheap labor, town-demand-driven).
    Both are proven-routable (season_scale: carrot 86%, melon right-sized 67%);
    the failing ongoing crops (strawberry/tomato) are dropped."""
    tile_crop = {}
    for q in quads:
        tiles = quadrant_tiles(q)
        tiles.sort(key=lambda t: manhattan(t, (4, 4)))
        if q == "NW":
            mix = ["MELON"] * 12 + ["CARROT"] * 13
        else:
            mix = ["CARROT"] * 25
        for i, tp in enumerate(tiles):
            tile_crop[tp] = mix[i]
    return tile_crop


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "nw"
    if mode == "nw":
        tc = build_plan(["NW"])
        run(tc, 10, {}, "nw25")
    else:
        # land expansion: NW + buy NE, SW, SE over the season
        tc = build_plan(["NW", "NE", "SW", "SE"])
        land = {12: True, 18: True, 24: True}  # attempt BUY_LAND on these days
        run(tc, 10, land, "land100")
