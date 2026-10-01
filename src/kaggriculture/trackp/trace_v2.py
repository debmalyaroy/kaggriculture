"""P0.1 -- trace layout v2: the full per-turn state, both seats.

One float32 matrix per EPISODE, X of shape (T, D) with D = 206, built from
seat 0's perspective; `seat_view(X, 1)` swaps the mine/theirs blocks to give
seat 1's perspective. The layout is versioned and self-describing (FIELDS).

Design intent (Track P plan, P0.1): capture EVERYTHING the engine exposes --
the shop draw (the game's only structured randomness), both farms' full
public state, both privates (replays record both), and a complete per-seat
action summary so macro decisions are derivable without re-parsing replays.

Block layout (offsets in FIELDS):
  GLOBAL   46   step/day/hour, shop counts+first-unlock-day, market inv+prices,
                exact town drain/day per product
  MINE     28   public farm state, seat 0
  THEIRS   28   public farm state, seat 1
  PRIV_M   17   seat 0 private (shed, seeds, carried)
  PRIV_T   17   seat 1 private
  ACT_M    35   seat 0's action at this step (market + unit-op summary)
  ACT_T    35   seat 1's action

The action at row t is the action that PRODUCED state t (replay steps[t]
alignment) -- i.e. X[t] pairs "state after" with "action taken", and a
policy dataset uses (X[t-1] state blocks, X[t] action blocks).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import common

VERSION = 2

_FARM = ["money",
         "crop_WHEAT", "crop_CARROT", "crop_TOMATO", "crop_STRAWBERRY",
         "crop_MELON",
         "anim_GOOSE", "anim_COW", "anim_SHEEP",
         "empty_coop", "empty_pasture", "weeds", "empty_tiles", "locked_tiles",
         "age_0_1", "age_2_4", "age_5_9", "age_10p",
         "crop_pending", "anim_pending",
         "watered_today", "at_risk_unwatered", "fertilized_active",
         "hands", "hires_today", "farmer_shed_adj", "mean_dist_shed",
         "quads", "demand_match"]
assert len(_FARM) == 29

_PRIV = [f"shed_{p}" for p in common.PRODUCTS] + [
    "shed_total"] + [f"seeds_{c}" for c in common.CROP_NAMES] + [
    "carried_total", "carried_FERTILIZER"]

_ACT = ([f"sell_{p}" for p in common.PRODUCTS]
        + ["buyp_WHEAT", "buyp_FERTILIZER"]
        + [f"buyseed_{c}" for c in common.CROP_NAMES]
        + [f"buyanim_{a}" for a in common.ANIMAL_NAMES]
        + ["hire", "buy_land"]
        + [f"plant_{c}" for c in common.CROP_NAMES]
        + ["water", "harvest", "feed", "care", "collect_fert", "fertilize",
           "build_coop", "build_pasture", "dig"])

_GLOBAL = (["step", "day", "hour"]
           + [f"shopn_{s}" for s in common.SHOPS_SORTED]
           + [f"shopday_{s}" for s in common.SHOPS_SORTED]
           + [f"minv_{p}" for p in common.PRODUCTS]
           + [f"mpx_{p}" for p in common.PRODUCTS]
           + [f"drain_{p}" for p in common.PRODUCTS])

FIELDS = (_GLOBAL
          + [f"m_{f}" for f in _FARM]
          + [f"t_{f}" for f in _FARM]
          + [f"pm_{f}" for f in _PRIV]
          + [f"pt_{f}" for f in _PRIV]
          + [f"am_{f}" for f in _ACT]
          + [f"at_{f}" for f in _ACT])

N_GLOBAL, N_FARM, N_PRIV, N_ACT = len(_GLOBAL), len(_FARM), len(_PRIV), len(_ACT)
DIM = len(FIELDS)

_SHED_TILES = set(common.shed_access_tiles())


def _farm_block(farm: dict, drain: dict, day: int) -> list:
    crops = {c: 0 for c in common.CROP_NAMES}
    anims = {a: 0 for a in common.ANIMAL_NAMES}
    empty_coop = empty_pasture = weeds = empties = locked = 0
    age = [0, 0, 0, 0]
    crop_pending = anim_pending = watered = at_risk = fert = 0
    for y in range(common.BOARD):
        row = farm["tiles"][y]
        for x in range(common.BOARD):
            t = row[x]
            if t is None:
                empties += 1
            elif t == "LOCKED":
                locked += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "WEED":
                    weeds += 1
                elif k == "PLANT":
                    crops[t["crop"]] += 1
                    a = day - t["planted_day"]
                    age[0 if a <= 1 else 1 if a <= 4 else 2 if a <= 9 else 3] += 1
                    crop_pending += t["yield_units"]
                    watered += 1 if t["watered_today"] else 0
                    at_risk += 1 if t["consecutive_unwatered"] >= 1 else 0
                    fert += 1 if t.get("fertilized_until_day", -1) >= day else 0
                elif "animal" in t:
                    anims[t["animal"]] += 1
                    anim_pending += t["yield_units"]
                elif k == "COOP":
                    empty_coop += 1
                elif k == "PASTURE":
                    empty_pasture += 1
    # Production mix vs town drain: cosine over the 8 sellable products.
    prod = {p: 0.0 for p in common.PRODUCTS}
    for c, n in crops.items():
        prod[c] += n
    for a, n in anims.items():
        prod[common.ANIMALS[a][5]] += n
    num = sum(prod[p] * drain[p] for p in common.PRODUCTS)
    na = sum(v * v for v in prod.values()) ** 0.5
    nb = sum(v * v for v in drain.values()) ** 0.5
    dm = num / (na * nb) if na > 0 and nb > 0 else 0.0

    units = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    dists = [min(abs(ux - sx) + abs(uy - sy) for sx, sy in _SHED_TILES)
             for ux, uy in units]
    return ([float(farm["money"])]
            + [float(crops[c]) for c in common.CROP_NAMES]
            + [float(anims[a]) for a in common.ANIMAL_NAMES]
            + [float(empty_coop), float(empty_pasture), float(weeds),
               float(empties), float(locked)]
            + [float(v) for v in age]
            + [float(crop_pending), float(anim_pending),
               float(watered), float(at_risk), float(fert),
               float(len(farm["hands"])), float(farm.get("hires_today", 0)),
               1.0 if tuple(farm["farmer"]) in _SHED_TILES else 0.0,
               float(sum(dists)) / len(dists),
               float(len(farm.get("unlocked_quadrants", ["NW"]))), dm])


def _priv_block(private: dict) -> list:
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    invs = private.get("inventories", [])
    carried = sum(sum(i.values()) for i in invs)
    carried_f = sum(i.get("FERTILIZER", 0) for i in invs)
    return ([float(shed.get(p, 0)) for p in common.PRODUCTS]
            + [float(sum(shed.values()))]
            + [float(seeds.get(c, 0)) for c in common.CROP_NAMES]
            + [float(carried), float(carried_f)])


def _act_block(action: dict) -> list:
    sell = {p: 0 for p in common.PRODUCTS}
    buyp = {"WHEAT": 0, "FERTILIZER": 0}
    buyseed = {c: 0 for c in common.CROP_NAMES}
    buyanim = {a: 0 for a in common.ANIMAL_NAMES}
    hire = buy_land = 0
    for o in (action.get("market") or []):
        if not isinstance(o, list) or not o:
            continue
        op = o[0]
        if op == "HIRE":
            hire += 1
        elif op == "BUY_LAND":
            buy_land += 1
        elif len(o) >= 3:
            try:
                n = int(o[2])
            except (TypeError, ValueError):
                continue
            item = o[1]
            if op == "SELL" and item in sell:
                sell[item] += n
            elif op == "BUY_PRODUCT" and item in buyp:
                buyp[item] += n
            elif op == "BUY_SEED" and item in buyseed:
                buyseed[item] += n
            elif op == "BUY_ANIMAL" and item in buyanim:
                buyanim[item] += n
    ops = {k: 0 for k in ("water", "harvest", "feed", "care", "collect_fert",
                          "fertilize", "build_coop", "build_pasture", "dig")}
    plant = {c: 0 for c in common.CROP_NAMES}
    for u in [action.get("farmer") or ["PASS"]] + (action.get("hands") or []):
        if not isinstance(u, list) or not u:
            continue
        op = u[0]
        if op == "PLANT" and len(u) >= 2 and u[1] in plant:
            plant[u[1]] += 1
        elif op == "WATER":
            ops["water"] += 1
        elif op == "HARVEST":
            ops["harvest"] += 1
        elif op == "FEED":
            ops["feed"] += 1
        elif op == "CARE":
            ops["care"] += 1
        elif op == "COLLECT_FERTILIZER":
            ops["collect_fert"] += 1
        elif op == "FERTILIZE":
            ops["fertilize"] += 1
        elif op == "BUILD_COOP":
            ops["build_coop"] += 1
        elif op == "BUILD_PASTURE":
            ops["build_pasture"] += 1
        elif op == "DIG":
            ops["dig"] += 1
    return ([float(sell[p]) for p in common.PRODUCTS]
            + [float(buyp["WHEAT"]), float(buyp["FERTILIZER"])]
            + [float(buyseed[c]) for c in common.CROP_NAMES]
            + [float(buyanim[a]) for a in common.ANIMAL_NAMES]
            + [float(hire), float(buy_land)]
            + [float(plant[c]) for c in common.CROP_NAMES]
            + [float(ops[k]) for k in ("water", "harvest", "feed", "care",
                                       "collect_fert", "fertilize",
                                       "build_coop", "build_pasture", "dig")])


def extract(rep: dict) -> tuple:
    """Replay dict -> (X float32 (T, DIM), meta dict)."""
    steps = rep["steps"]
    T = len(steps)
    X = np.zeros((T, DIM), dtype=np.float32)
    shop_first = {s: -1.0 for s in common.SHOPS_SORTED}
    for t in range(T):
        obs0 = steps[t][0]["observation"]
        obs1 = steps[t][1]["observation"]
        day = int(obs0.get("day", t // common.TURNS_PER_DAY))
        shops = obs0["town"]["unlocked_shops"]
        for s in shops:
            if shop_first.get(s, -1.0) < 0:
                shop_first[s] = float(day)
        drain = common.town_drain_per_day(shops)
        minv = obs0["market"]["inventory"]
        mpx = obs0["market"]["prices"]
        g = ([float(obs0.get("step", t)), float(day),
              float(obs0.get("hour", t % common.TURNS_PER_DAY))]
             + [float(shops.count(s)) for s in common.SHOPS_SORTED]
             + [shop_first[s] for s in common.SHOPS_SORTED]
             + [float(minv.get(p, 0)) for p in common.PRODUCTS]
             + [float(mpx.get(p, 0)) for p in common.PRODUCTS]
             + [float(drain[p]) for p in common.PRODUCTS])
        a0 = steps[t][0].get("action")
        a1 = steps[t][1].get("action")
        row = (g
               + _farm_block(obs0["farms"][0], drain, day)
               + _farm_block(obs0["farms"][1], drain, day)
               + _priv_block(obs0.get("private", {}))
               + _priv_block(obs1.get("private", {}))
               + _act_block(a0 if isinstance(a0, dict) else {})
               + _act_block(a1 if isinstance(a1, dict) else {}))
        X[t] = row
    banks = common.final_banks(rep)
    meta = {
        "version": VERSION,
        "episode": rep.get("id"),
        "seed": common.replay_seed(rep),
        "teams": common.replay_teams(rep),
        "engine": rep.get("module_version"),
        "rewards": rep.get("rewards"),
        "banks": banks,
        "dim": DIM,
    }
    return X, meta


def seat_view(X: np.ndarray, seat: int) -> np.ndarray:
    """Seat 0's layout, or the mine/theirs-swapped copy for seat 1."""
    if seat == 0:
        return X
    Y = X.copy()
    o = N_GLOBAL
    f, p, a = N_FARM, N_PRIV, N_ACT
    Y[:, o:o + f], Y[:, o + f:o + 2 * f] = \
        X[:, o + f:o + 2 * f], X[:, o:o + f]
    o += 2 * f
    Y[:, o:o + p], Y[:, o + p:o + 2 * p] = \
        X[:, o + p:o + 2 * p], X[:, o:o + p]
    o += 2 * p
    Y[:, o:o + a], Y[:, o + a:o + 2 * a] = \
        X[:, o + a:o + 2 * a], X[:, o:o + a]
    return Y


def field_index(name: str) -> int:
    return FIELDS.index(name)


def save(path: str, X: np.ndarray, meta: dict):
    np.savez_compressed(path, X=X, meta=json.dumps(meta))


def load(path: str) -> tuple:
    z = np.load(path, allow_pickle=False)
    return z["X"], json.loads(str(z["meta"]))


def trace_path(episode_id) -> str:
    return os.path.join(common.TRACES, f"{episode_id}.npz")
