"""E3.1 (env) -- the MACRO decision environment: a daily plan -> engine reward.

The RL policy (``macro_rl``) chooses a DAILY plan (~30 ``MacroAction`` days, the
schema in ``train.macro_actions``). This module turns that plan into a *legal*
game and returns the terminal reward, so the policy is optimised against the
REAL engine, not a surrogate.

How the reward is produced (faithful, closed-loop):
  * OUR seat is driven by ``PlanController`` -- a plan-parameterised greedy
    realizer. Each day it emits that day's market orders (buy seed/hire/land/
    animal, sell) and greedily services owned crop tiles (PLANT/WATER/HARVEST)
    with the farmer + every hand, plus best-effort husbandry (FEED/CARE). It
    reads ``obs`` for the live board, so every op it emits is legal by
    construction (illegal ops are silent no-ops anyway -- the engine is the
    judge, never this file).
  * The OPPONENT seat is a REPLAY of a real high-rated match (destbreso
    ``opponent_actions`` or a top-100 replay), so we train against transcribed
    top-tier play with no opponent source needed (the operator's clean-room
    point).
  * The vendored ``kaggle_environments`` engine runs the episode; the terminal
    ``reward`` is each seat's final bank. Reward = win(1)/draw(0.5)/loss(0),
    with the bank margin available for optional shaping.

``PlanController`` is the DEFAULT executor. The production upgrade is the
PC-TAPF / plan-repair realizer (Workstream B2, ``research/rustengine-or/season_cbs``);
``MacroEnv`` takes an ``executor`` hook so that swaps in without touching the RL
loop. The Rust ``kagg batch`` open-loop path is the throughput upgrade (needs an
obs-free executor); this closed-loop Python path is the correctness reference.

    python -m kaggriculture.train.macro_env --smoke      # 2 rollouts, asserts banks
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import kaggriculture.train.macro_actions as MA

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# vendored engine lives under vendor/ (never pip-vendored into the pkg -- CLAUDE.md)
sys.path.insert(0, os.path.join(ROOT, "vendor"))

POOL_PATH = os.path.join(ROOT, "models", "opponent_pool_full.json")
DESTBRESO = os.path.join(ROOT, ".local", "scratch", "datasets", "destbreso",
                         "matchups_all.parquet")
SEEDS = MA.SEEDS
PRODUCTS = MA.PRODUCTS
ANIMALS = MA.ANIMALS
EPISODE_STEPS = 720
TURNS_PER_DAY = 24


# --------------------------------------------------------------------------- #
# obs helpers (obs is a Struct or a plain dict depending on engine internals)
# --------------------------------------------------------------------------- #
def _g(o, k, default=None):
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


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


def chunk_compact(tiles, n):
    """Partition tiles into n spatially-compact clusters (serpentine bands).

    Same partition validated at 86-87% routing efficiency in
    research/rustengine-or/season_cbs; used by PersistentClusterController (PC-TAPF).
    """
    if n <= 0:
        return []
    ts = sorted(tiles, key=lambda t: (t[0], t[1] if t[0] % 2 == 0 else -t[1]))
    out, per = [], max(1, (len(ts) + n - 1) // n)
    for i in range(n):
        c = ts[i * per:(i + 1) * per]
        if c:
            out.append(c)
    assigned = sum(len(c) for c in out)
    if assigned < len(ts):
        out[-1].extend(ts[assigned:])
    while len(out) < n:
        out.append([])
    return out


# --------------------------------------------------------------------------- #
# The plan-parameterised greedy controller (DEFAULT executor).
# --------------------------------------------------------------------------- #
class PlanController:
    """Realise a per-day ``MacroAction`` plan as legal engine actions.

    ``plan`` is a list indexed by game-day (0..29) of ``MacroAction``. A short
    plan is padded with IDLE days; a plan longer than the game is truncated.
    """

    def __init__(self, plan: List[MA.MacroAction]):
        self.plan = list(plan)
        self.actions: List[dict] = []       # recorded, for the tape
        self._day_done = set()              # days whose market orders fired
        self._sold_today = set()
        self._tile_crop: Dict[Tuple[int, int], str] = {}

    def _today(self, day: int) -> MA.MacroAction:
        if 0 <= day < len(self.plan):
            return self.plan[day]
        return MA.MacroAction()

    def _crop_for_tile(self, xy, day) -> Optional[str]:
        """Assign an owned empty tile a crop from today's (or the running) plan."""
        if xy in self._tile_crop:
            return self._tile_crop[xy]
        ma = self._today(day)
        # prefer a crop the plan planted today, else any seed we intend to buy
        want = ma.plant or ma.buy_seed
        if not want:
            # fall back to a cumulative preference: melon anchor, else carrot
            want = {"MELON": 1}
        crop = max(want.items(), key=lambda kv: kv[1])[0]
        self._tile_crop[xy] = crop
        return crop

    def agent(self, obs, config=None):
        day = int(_g(obs, "day", 0))
        hour = int(_g(obs, "hour", 0))
        seat = int(_g(obs, "player", 0))
        farms = _g(obs, "farms", []) or []
        farm = farms[seat] if seat < len(farms) else {}
        priv = _g(obs, "private", {}) or {}
        seeds = dict(_g(priv, "seeds", {}) or {})
        shed = dict(_g(priv, "shed", {}) or {})
        money = float(_g(farm, "money", 0) or 0)
        market = _g(obs, "market", {}) or {}
        minv = dict(_g(market, "inventory", {}) or {})
        tiles = _g(farm, "tiles", []) or []
        farmer_pos = tuple(_g(farm, "farmer", (0, 0)) or (0, 0))
        hands = [tuple(h) for h in (_g(farm, "hands", []) or [])]

        ma = self._today(day)
        orders: List[list] = []

        # ---- once-a-day market orders (buys/hires) -------------------------
        if day not in self._day_done:
            self._day_done.add(day)
            for _ in range(ma.buy_land):
                orders.append(["BUY_LAND"])
            for _ in range(ma.hire):
                orders.append(["HIRE"])
            for s, n in ma.buy_seed.items():
                if n > 0 and s in SEEDS:
                    orders.append(["BUY_SEED", s, int(n)])
            for a, n in ma.buy_animal.items():
                if n > 0 and a in ANIMALS:
                    orders.append(["BUY_ANIMAL", a, int(n)])

        # ---- sells: fire the plan's sells once the shed holds product ------
        if day not in self._sold_today:
            sold_any = False
            for p, n in ma.sell.items():
                have = int(shed.get(p, 0) or 0)
                if have > 0 and n > 0:
                    orders.append(["SELL", p, min(int(n), have)])
                    sold_any = True
            if sold_any:
                self._sold_today.add(day)

        orders = orders[:10]                # engine drops extras silently

        movers = [farmer_pos] + hands
        needs = self._collect_needs(tiles, seeds, day, hour)
        mover_ops = self._assign_ops(movers, needs, tiles)

        action = {"farmer": mover_ops[0],
                  "hands": mover_ops[1:],
                  "market": orders}
        self.actions.append(action)
        return action

    # ---- field-op helpers (overridable by routing strategies) --------------
    def _collect_needs(self, tiles, seeds, day, hour):
        """Owned tiles needing an op: [(x, y, (verb, target|None)), ...]."""
        needs = []
        for y in range(len(tiles)):
            row = tiles[y]
            for x in range(len(row)):
                t = row[x]
                if t == "LOCKED":
                    continue
                if t is None:
                    crop = self._crop_for_tile((x, y), day)
                    if seeds.get(crop, 0) > 0 and hour <= 16:
                        needs.append((x, y, ("PLANT", crop)))
                elif isinstance(t, dict) and t.get("kind") == "PLANT":
                    watered = t.get("watered_today", False)
                    yu = t.get("yield_units", 0)
                    age = day - t.get("planted_day", day)
                    if not watered:
                        needs.append((x, y, ("WATER", None)))
                    elif yu > 0 and age >= 2:
                        needs.append((x, y, ("HARVEST", None)))
        return needs

    @staticmethod
    def _emit(pos, x, y, op):
        if pos == (x, y):
            verb, tgt = op
            return [verb, tgt] if tgt else [verb]
        mv = step_toward(pos, (x, y))
        return [mv] if mv else ["PASS"]

    def _assign_ops(self, movers, needs, tiles=None):
        """GREEDY (B2.0): each mover takes the globally-nearest free need."""
        assigned = set()
        mover_ops = [["PASS"] for _ in movers]
        for mi, pos in enumerate(movers):
            best = None
            for (x, y, op) in needs:
                if (x, y) in assigned:
                    continue
                d = manhattan(pos, (x, y))
                if best is None or d < best[0]:
                    best = (d, x, y, op)
            if best is None:
                continue
            _, x, y, op = best
            assigned.add((x, y))
            mover_ops[mi] = self._emit(pos, x, y, op)
        return mover_ops


class PersistentClusterController(PlanController):
    """B2.2/2.3 PC-TAPF executor: PERSISTENT per-mover -> tile-cluster routing.

    The greedy executor re-picks the globally-nearest need every turn, so movers
    thrash between quadrants (measured ~57% efficiency on a spread farm). Here
    each mover OWNS a spatially-compact cluster of tiles (chunk_compact, the same
    partition validated at 86-87% in research/rustengine-or/season_cbs) and only services
    its own cluster -- movers stay local, walk-waste drops, more of the plan's
    economy is realised. Re-partitions when the owned-tile set grows (BUY_LAND).
    Drop-in: ``MacroEnv(executor=lambda p: PersistentClusterController(p))``.
    """

    def __init__(self, plan):
        super().__init__(plan)
        self._clusters = None               # mover_idx -> set((x,y))
        self._owned_key = None              # owned-tile signature (re-cluster on change)

    def _owned_tiles(self, tiles):
        out = []
        for y in range(len(tiles)):
            for x in range(len(tiles[y])):
                if tiles[y][x] != "LOCKED":
                    out.append((x, y))
        return out

    def _ensure_clusters(self, n_movers, tiles):
        owned = self._owned_tiles(tiles)
        key = (n_movers, len(owned))
        if key == self._owned_key and self._clusters is not None:
            return
        clusters = chunk_compact(owned, max(1, n_movers))
        self._clusters = {i: set(c) for i, c in enumerate(clusters)}
        self._owned_key = key

    def _assign_ops(self, movers, needs, tiles=None):
        if tiles is None:
            return super()._assign_ops(movers, needs, tiles)
        self._ensure_clusters(len(movers), tiles)
        by_tile = {(x, y): (x, y, op) for (x, y, op) in needs}
        mover_ops = [["PASS"] for _ in movers]
        taken = set()
        for mi, pos in enumerate(movers):
            cluster = self._clusters.get(mi, set())
            cand = [by_tile[t] for t in cluster if t in by_tile and t not in taken]
            if not cand:                    # cluster clear -> help nearest free need
                cand = [n for n in needs if (n[0], n[1]) not in taken]
                if not cand:
                    continue
            x, y, op = min(cand, key=lambda n: manhattan(pos, (n[0], n[1])))
            taken.add((x, y))
            mover_ops[mi] = self._emit(pos, x, y, op)
        return mover_ops


# --------------------------------------------------------------------------- #
# B3: the full season_cbs executor -- persistent clusters + dedicated husbandry
# movers + per-crop task timing + don't-dump sells. Ported from
# research/rustengine-or/tools/season_cbs.py (measured 86-87% routing efficiency) and
# made PLAN-PARAMETERISED (design derived from the MacroAction plan) and
# SEAT-GENERAL (reads obs["player"]).
# --------------------------------------------------------------------------- #
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
LIQ_DAY = 27                                 # liquidate (dump) from this day

try:
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        CROPS as _CROPS, ANIMALS as _ANIMALS, MARKET_PARAMS as _MKT,
        market_price as _market_price)
except Exception:                            # keep import-safe without the engine
    _CROPS = _ANIMALS = _MKT = None
    _market_price = None


def _default_thresh(item, frac=0.5):
    base = (_MKT[item]["base"] if _MKT and item in _MKT else 20)
    return max(2, int(base * frac))


def _dont_dump_qty(item, inv, have, thresh):
    """Sell only units whose marginal staircase price stays >= thresh."""
    if _market_price is None:
        return have
    n, cur = 0, inv
    while n < have:
        pr = _market_price(item, cur)
        if pr < thresh:
            break
        n += 1
        if pr > 1:
            cur += 1
    return n


def _nearest(pos, tiles):
    return min(tiles, key=lambda t: manhattan(pos, t))


class SeasonCbsController(PlanController):
    """PC-TAPF executor: persistent clusters + husbandry movers + don't-dump.

    Design (n_hands, crop mix, n_sheep, wheat feed tiles) is derived from the
    MacroAction plan; routing/market logic is the validated season_cbs. Drop-in:
    ``MacroEnv(executor=lambda p: SeasonCbsController(p))``.
    """

    def __init__(self, plan):
        super().__init__(plan)
        self._design = None                  # {tile_crop, sheep_tiles, wheat_tiles}
        self._owned_key = None
        # aggregate the plan into design parameters
        self._crop_units = {}
        animals = {"COW": 0, "GOOSE": 0, "SHEEP": 0}
        n_hands = 0
        total_land = 0
        for ma in plan:
            for c, q in (ma.plant or {}).items():
                self._crop_units[c] = self._crop_units.get(c, 0) + q
            for a, q in (ma.buy_animal or {}).items():
                if a in animals:
                    animals[a] += q
            n_hands = max(n_hands, ma.hire)
            total_land += ma.buy_land
        self.buy_land = int(total_land)             # X3: total land tiers to buy
        self._crop_units = {c: u for c, u in self._crop_units.items() if u > 0} \
            or {"MELON": 1}
        # X2: handle ALL animals (COW/GOOSE/SHEEP), not sheep only
        self.animals = {a: int(n) for a, n in animals.items() if n > 0}
        self.n_animals = sum(self.animals.values())
        self.n_sheep = self.animals.get("SHEEP", 0)     # compat
        self.n_hands = int(n_hands)
        self.thresh = {}

    # ---- derive the farm design from the CURRENT owned tiles ----------------
    def _ensure_design(self, tiles, n_movers):
        owned = [(x, y) for y in range(len(tiles))
                 for x in range(len(tiles[y])) if tiles[y][x] != "LOCKED"]
        key = (len(owned), n_movers)
        if key == self._owned_key and self._design is not None:
            return
        owned_set = set(owned)
        # X2: husbandry tiles near the shed, one per animal head (typed), plus
        # wheat feed tiles sized to feed the whole herd daily.
        animal_tiles, wheat_tiles = [], []
        if self.n_animals > 0:
            near = sorted(owned, key=lambda t: manhattan(t, (4, 4)))
            i = 0
            for a, n in self.animals.items():
                for _ in range(n):
                    if i < len(near):
                        animal_tiles.append((near[i], a))
                        i += 1
            wheat_tiles = near[i:i + max(1, self.n_animals)]
            i += len(wheat_tiles)
        sheep_tiles = [p for p, a in animal_tiles]      # compat alias
        reserved = set(sheep_tiles) | set(wheat_tiles)
        crop_tiles = [t for t in owned if t not in reserved]
        # assign crops to tiles by the plan's crop proportions
        crops = sorted(self._crop_units, key=lambda c: -self._crop_units[c])
        total = sum(self._crop_units.values()) or 1
        tile_crop = {}
        ci = 0
        for i, t in enumerate(crop_tiles):
            # round-robin weighted by proportion
            frac = (i + 1) / max(len(crop_tiles), 1)
            acc, pick = 0.0, crops[0]
            for c in crops:
                acc += self._crop_units[c] / total
                if frac <= acc:
                    pick = c
                    break
            tile_crop[t] = pick
        # husbandry movers 0,1 (farmer + 1 hand) when sheep exist
        n_husb = 2 if self.n_animals else 0
        n_crop_movers = max(1, n_movers - n_husb)
        clusters = chunk_compact(list(tile_crop.keys()), n_crop_movers)
        mover_crop = {}
        for j, cl in enumerate(clusters):
            mover_crop[n_husb + j] = set(cl)
        self._design = dict(tile_crop=tile_crop, animal_tiles=animal_tiles,
                            sheep_tiles=sheep_tiles, wheat_tiles=wheat_tiles,
                            mover_crop=mover_crop, husb=list(range(n_husb)))
        self._owned_key = key
        for c in list(tile_crop.values()) + ["WOOL"]:
            self.thresh.setdefault(c, _default_thresh(c, 0.5))

    def _crop_task(self, farm_tiles, seeds, day, hour, x, y, crop):
        t = farm_tiles[y][x]
        if t == "LOCKED" or _CROPS is None or crop not in _CROPS:
            return None
        cd = _CROPS[crop]
        if t is None:
            if seeds.get(crop, 0) > 0 and hour <= 14:
                return ("PLANT", crop, 2)
            return None
        if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == crop:
            age = day - t.get("planted_day", day)
            yu = t.get("yield_units", 0)
            capped = yu >= cd["max_yield"]
            if not t.get("watered_today", False) and not capped:
                return ("WATER", None, 0)
            if not cd["ongoing"] and (age >= cd["max_yield_day"] or capped) and yu > 0:
                return ("HARVEST", None, 1)
            if cd["ongoing"] and yu > 0 and age > cd["first_yield_day"]:
                return ("HARVEST", None, 1)
        return None

    def agent(self, obs, config=None):
        day = int(_g(obs, "day", 0))
        hour = int(_g(obs, "hour", 0))
        seat = int(_g(obs, "player", 0))
        farms = _g(obs, "farms", []) or []
        farm = farms[seat] if seat < len(farms) else {}
        priv = _g(obs, "private", {}) or {}
        seeds = dict(_g(priv, "seeds", {}) or {})
        shed = dict(_g(priv, "shed", {}) or {})
        invs = _g(priv, "inventories", []) or []
        tiles = _g(farm, "tiles", []) or []
        money = float(_g(farm, "money", 0) or 0)
        market = _g(obs, "market", {}) or {}
        minv = dict(_g(market, "inventory", {}) or {})
        movers = [tuple(_g(farm, "farmer", (4, 4)) or (4, 4))] + \
                 [tuple(h) for h in (_g(farm, "hands", []) or [])]
        nm = len(movers)
        self._ensure_design(tiles, nm)
        d = self._design
        act = {i: ["PASS"] for i in range(nm)}

        def mi(i):
            return invs[i] if i < len(invs) else {}

        # ---- husbandry movers: build/place/feed/care/harvest ALL animals ----
        free = set(i for i in d["husb"] if i < nm)
        if free:
            need_build, need_place, present = [], [], []
            for (x, y), atype in d["animal_tiles"]:
                t = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None
                struct = _ANIMALS[atype]["structure"] if _ANIMALS else "PASTURE"
                if t is None:
                    need_build.append(((x, y), struct))
                elif isinstance(t, dict) and t.get("kind") == struct \
                        and "animal" not in t:
                    need_place.append(((x, y), atype))
                elif isinstance(t, dict) and "animal" in t:
                    present.append(((x, y), t))
            unfed = [p for (p, t) in present if not t.get("fed_today", True)]
            ready = [p for (p, t) in present if t.get("yield_units", 0) > 0]
            uncared = [p for (p, t) in present
                       if t.get("fed_today") and not t.get("cared_today", True)]
            fert = [p for (p, t) in present if t.get("fertilizer_available")]  # X5
            # feed unfed animals first (all eat WHEAT)
            for p in unfed:
                if not free:
                    break
                holders = [i for i in free if mi(i).get("WHEAT", 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manhattan(movers[i], p))
                    free.discard(m)
                    act[m] = ["FEED"] if movers[m] == p else \
                        ([step_toward(movers[m], p)] or ["PASS"])
                elif shed.get("WHEAT", 0) > 0:
                    m = min(free, key=lambda i: manhattan(
                        movers[i], _nearest(movers[i], SHED_TILES)))
                    free.discard(m)
                    st = _nearest(movers[m], SHED_TILES)
                    act[m] = (["PICKUP", "WHEAT", min(self.n_animals,
                              shed.get("WHEAT", 0))] if movers[m] == st
                              else ([step_toward(movers[m], st)] or ["PASS"]))
            queue = [(p, ["HARVEST"]) for p in ready]
            queue += [(p, ["COLLECT_FERTILIZER"]) for p in fert]     # X5
            queue += [(p, ["CARE"]) for p in uncared]
            queue += [(pos, [struct]) for (pos, struct) in need_build]
            for (x, y) in d["wheat_tiles"]:
                tk = self._crop_task(tiles, seeds, day, hour, x, y, "WHEAT")
                if tk:
                    op = tk[0]
                    queue.append(((x, y),
                                  ["PLANT", "WHEAT"] if op == "PLANT" else [op]))
            for (tp, action) in queue:
                if not free:
                    break
                m = min(free, key=lambda i: manhattan(movers[i], tp))
                free.discard(m)
                act[m] = action if movers[m] == tp else \
                    ([step_toward(movers[m], tp)] or ["PASS"])
            # place animals we hold (need the animal in inventory/shed)
            for (p, atype) in need_place:
                if not free:
                    break
                holders = [i for i in free if mi(i).get(atype, 0) > 0]
                if holders:
                    m = min(holders, key=lambda i: manhattan(movers[i], p))
                    free.discard(m)
                    act[m] = ["PLACE", atype] if movers[m] == p else \
                        ([step_toward(movers[m], p)] or ["PASS"])
                elif shed.get(atype, 0) > 0:
                    m = min(free, key=lambda i: manhattan(
                        movers[i], _nearest(movers[i], SHED_TILES)))
                    free.discard(m)
                    st = _nearest(movers[m], SHED_TILES)
                    act[m] = (["PICKUP", atype, 1] if movers[m] == st
                              else ([step_toward(movers[m], st)] or ["PASS"]))

        # ---- crop movers: each services ONLY its persistent cluster ---------
        for m, cl in d["mover_crop"].items():
            if m >= nm:
                continue
            best, bestd = None, 1e9
            for (x, y) in cl:
                crop = d["tile_crop"].get((x, y))
                if crop is None:
                    continue
                tk = self._crop_task(tiles, seeds, day, hour, x, y, crop)
                if not tk:
                    continue
                dd = manhattan(movers[m], (x, y)) + tk[2] * 0.01
                if dd < bestd:
                    bestd, best = dd, ((x, y), tk)
            if best:
                (tp, (op, arg, _pri)) = best
                act[m] = (["PLANT", arg] if op == "PLANT" else [op]) \
                    if movers[m] == tp else ([step_toward(movers[m], tp)] or ["PASS"])

        # ---- market: hire h0, seeds h1, sheep, don't-dump sells h5+ ---------
        orders = []
        if hour == 0 and self.n_hands > 0:
            orders += [["HIRE"]] * min(self.n_hands, 10)
        # X3: buy land one tier/day (cash-gated) up to the plan's total
        if hour == 4 and self.buy_land > 0:
            n_extra = len(_g(farm, "unlocked_quadrants", ["NW"]) or ["NW"]) - 1
            if n_extra < 3:
                price = (1000, 2000, 4000)[n_extra]
                bought = n_extra  # tiers already owned
                if bought < self.buy_land and money >= price + 500 and len(orders) < 10:
                    orders.append(["BUY_LAND"])
        if hour == 1:
            need = {}
            for c in d["tile_crop"].values():
                need[c] = need.get(c, 0) + 1
            for _ in d["wheat_tiles"]:
                need["WHEAT"] = need.get("WHEAT", 0) + 1
            for c in sorted(need):
                deficit = need[c] - seeds.get(c, 0)
                cost = _CROPS[c]["seed"] if _CROPS and c in _CROPS else 20
                if deficit > 0 and len(orders) < 10 and money >= cost:
                    orders.append(["BUY_SEED", c, min(deficit, 20)])
        # X2: buy WHEAT feed for the whole herd early; buy animals 1/day (cash-gated)
        if hour == 2 and self.n_animals and day <= 8 and \
                shed.get("WHEAT", 0) < self.n_animals and len(orders) < 10 and money > 300:
            orders.append(["BUY_PRODUCT", "WHEAT", self.n_animals])
        if hour == 3 and self.n_animals and day >= 6:
            # count each animal type already owned; buy the next short one (cash-gated)
            for a in sorted(self.animals, key=lambda a: _ANIMALS[a]["cost"] if _ANIMALS else 0):
                have = int(shed.get(a, 0) or 0) + sum(
                    1 for (x, y), at in d["animal_tiles"]
                    if at == a and isinstance(tiles[y][x], dict)
                    and "animal" in tiles[y][x])
                cost = (_ANIMALS[a]["cost"] if _ANIMALS else 400)
                if have < self.animals[a] and money >= cost + 500 and len(orders) < 10:
                    orders.append(["BUY_ANIMAL", a, 1])
                    break
        sellables = [c for c in sorted(set(d["tile_crop"].values())) if c != "WHEAT"]
        for a in self.animals:                          # animal products
            prod = _ANIMALS[a]["product"] if _ANIMALS else None
            if prod and prod not in sellables:
                sellables.append(prod)
        if self.n_animals and int(shed.get("FERTILIZER", 0) or 0) > 0:  # X5
            sellables.append("FERTILIZER")
        for k, p in enumerate(sellables):
            if hour == 5 + k:
                have = int(shed.get(p, 0) or 0)
                if have > 0:
                    inv = int(minv.get(p, 0) or 0)
                    q = _dont_dump_qty(p, inv, have, self.thresh.get(p, 10)) \
                        if day < LIQ_DAY else have
                    if q > 0:
                        orders.append(["SELL", p, q])
        orders = orders[:10]

        action = {"farmer": act[0], "hands": [act[i] for i in range(1, nm)],
                  "market": orders}
        self.actions.append(action)
        return action


def replay_agent(cells: List[dict]):
    """An agent that replays a recorded action-cell list (the opponent)."""
    state = {"i": 0}

    def agent(obs, config=None):
        i = state["i"]
        state["i"] += 1
        if i < len(cells):
            c = cells[i]
            return {"farmer": c.get("farmer", ["PASS"]),
                    "hands": c.get("hands", []),
                    "market": c.get("market", [])}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return agent


def pass_agent(obs, config=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}


# --------------------------------------------------------------------------- #
# Opponent pool
# --------------------------------------------------------------------------- #
class OpponentPool:
    def __init__(self, path: str = POOL_PATH):
        self.entries: List[dict] = []
        self.weights: List[float] = []
        self._cells_cache: Dict[str, List[dict]] = {}
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
            for e in d.get("entries", []):
                self.entries.append(e)
                self.weights.append(float(e.get("weight", 0.0)) or 1e-6)
        if not self.entries:                # graceful: PASS-only pool
            self.entries = [{"kind": "pass", "team": "PASS", "rating": 0}]
            self.weights = [1.0]

    def sample(self, rng: random.Random) -> dict:
        return rng.choices(self.entries, weights=self.weights, k=1)[0]

    def prewarm_destbreso(self) -> int:
        """Materialise ALL destbreso opponents in ONE parquet pass (not per-lookup).

        Per-opponent `cells_for` otherwise rescans the 45k-row parquet each time
        -- fatal in an RL loop. Call once up front. Returns #cached.
        """
        try:
            import pyarrow.parquet as pq
        except Exception:
            return 0
        if not os.path.exists(DESTBRESO):
            return 0
        want = {}
        for e in self.entries:
            if e.get("kind") == "destbreso_tape":
                want[(str(e.get("episode_id")), int(e.get("seat", 0)))] = \
                    f"destbreso_tape:{e.get('episode_id')}:{e.get('seat')}"
        if not want:
            return 0
        n = 0
        pf = pq.ParquetFile(DESTBRESO)
        for b in pf.iter_batches(batch_size=8192,
                                 columns=["episode_id", "opponent_seat",
                                          "opponent_actions"]):
            d = b.to_pydict()
            for i, eid in enumerate(d["episode_id"]):
                k = (str(eid), int(d["opponent_seat"][i]))
                if k in want and want[k] not in self._cells_cache:
                    try:
                        self._cells_cache[want[k]] = json.loads(
                            d["opponent_actions"][i])
                        n += 1
                    except (ValueError, TypeError):
                        pass
        return n

    def cells_for(self, entry: dict) -> Optional[List[dict]]:
        """Materialise an opponent to an action-cell list, or None for PASS."""
        kind = entry.get("kind")
        if kind in (None, "pass", "frozen_self", "agent"):
            return None                     # caller substitutes PASS / self
        key = f"{kind}:{entry.get('episode_id')}:{entry.get('seat')}"
        if key in self._cells_cache:
            return self._cells_cache[key]
        cells = None
        if kind == "destbreso_tape":
            cells = self._destbreso_cells(entry)
        # replay_tape / ref_tape materialisation is a documented hook
        # (measure.opponents.extract_actions); destbreso covers 786/814.
        if cells is not None:
            self._cells_cache[key] = cells
        return cells

    def _destbreso_cells(self, entry) -> Optional[List[dict]]:
        try:
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
        except Exception:
            return None
        if not os.path.exists(DESTBRESO):
            return None
        eid = str(entry.get("episode_id"))
        seat = int(entry.get("seat", 0))
        pf = pq.ParquetFile(DESTBRESO)
        for b in pf.iter_batches(batch_size=4096,
                                 columns=["episode_id", "opponent_seat",
                                          "opponent_actions"]):
            d = b.to_pydict()
            for i, e in enumerate(d["episode_id"]):
                if str(e) == eid and int(d["opponent_seat"][i]) == seat:
                    try:
                        return json.loads(d["opponent_actions"][i])
                    except (ValueError, TypeError):
                        return None
        return None


# --------------------------------------------------------------------------- #
# The environment
# --------------------------------------------------------------------------- #
class MacroEnv:
    """Faithful macro-plan reward.

    ``backend`` picks the engine the rollout runs on -- both are bit-exact to the
    official interpreter (serve was fidelity-fixed 2026-09-18, S5/S3/S4):
      * ``"serve"`` (B4, DEFAULT, FAST): drives the Rust ``kagg serve`` stdio env
        via the fixed ``engine.serve_match.run_match`` over ONE persistent
        process -- ~0.63 s/game and reusable, ~2-3x the vendored Python engine
        and the path to a full RL budget. Reuse ONE ``MacroEnv`` per worker so
        the serve process is shared across thousands of rollouts.
      * ``"vendored"`` (reference): the Python ``kaggle_environments`` engine
        (also exposes the final-board weed count). Use to cross-check serve.
    """

    def __init__(self, pool: Optional[OpponentPool] = None,
                 executor: Optional[Callable[[List[MA.MacroAction]], object]] = None,
                 margin_shaping: float = 0.0, backend: str = "serve"):
        self.pool = pool or OpponentPool()
        # B3 default: the season_cbs PC-TAPF executor (persistent clusters +
        # husbandry + don't-dump) realises 2.6-8x more economy than the greedy
        # B2.0 realizer; pass executor=lambda p: PlanController(p) for greedy.
        self.executor = executor or (lambda plan: SeasonCbsController(plan))
        self.margin_shaping = margin_shaping
        self.backend = backend
        self._make = None
        self._srv = None
        if backend == "vendored":
            from kaggle_environments import make  # lazy (network dep)
            self._make = make

    def _serve(self):
        if self._srv is None:
            from kaggriculture.engine.serve_match import Serve
            self._srv = Serve()
        return self._srv

    def close(self):
        if self._srv is not None:
            try:
                self._srv.close()
            except Exception:
                pass
            self._srv = None

    def rollout(self, plan: List[MA.MacroAction], seed: int,
                opp_cells: Optional[List[dict]] = None,
                our_seat: int = 0) -> dict:
        """Play one game: our plan (seat ``our_seat``) vs an opponent.

        Returns dict(our_bank, opp_bank, win, reward, weeds, tape_cells).
        """
        ctrl = self.executor(plan)
        our = ctrl.agent
        opp = replay_agent(opp_cells) if opp_cells else pass_agent

        if self.backend == "serve":
            from kaggriculture.engine.serve_match import run_match
            a, b = (our, opp) if our_seat == 0 else (opp, our)
            b0, b1 = run_match(a, b, seed, srv=self._serve())
            our_bank = b0 if our_seat == 0 else b1
            opp_bank = b1 if our_seat == 0 else b0
            weeds = -1                       # serve returns banks only
        else:
            agents = [our, opp] if our_seat == 0 else [opp, our]
            env = self._make("kaggriculture", configuration={"seed": seed},
                             debug=False)
            env.run(agents)
            last = env.steps[-1]
            our_bank = float(last[our_seat]["reward"] or 0)
            opp_bank = float(last[1 - our_seat]["reward"] or 0)
            try:
                ft = last[our_seat]["observation"]["farms"][our_seat]["tiles"]
                weeds = sum(1 for row in ft for t in row
                            if isinstance(t, dict) and t.get("kind") == "WEED")
            except Exception:
                weeds = -1

        win = 1.0 if our_bank > opp_bank else (0.5 if our_bank == opp_bank else 0.0)
        reward = win + self.margin_shaping * (our_bank - opp_bank) / 1e5
        return dict(our_bank=our_bank, opp_bank=opp_bank, win=win,
                    reward=reward, weeds=weeds, tape_cells=ctrl.actions)


def _compile_worker(args):
    """Top-level (picklable) worker: compile ONE plan -> batch-tape string.

    Each worker owns its own light serve process; the plan is a picklable list of
    MacroAction. Used by a BOUNDED process pool so parallel compile never
    over-subscribes memory (see BatchRoller.parallel_compile)."""
    plan, exec_name, ref_seed = args
    exe = SeasonCbsController if exec_name == "seasoncbs" else PlanController
    env = MacroEnv(backend="serve", executor=lambda p: exe(p))
    try:
        cells = env.rollout(plan, ref_seed, opp_cells=None, our_seat=0)["tape_cells"]
    finally:
        env.close()
    return cells_to_batch_tape(cells, ref_seed)


class BatchRoller:
    """B4 fast path: ~10-20 ms/game via Rust ``kagg batch`` on PRE-BUILT tapes.

    The insight that avoids re-implementing the engine: OUR field ops
    (plant/water/harvest/move) are SEED-INDEPENDENT -- the initial board is a
    fixed NW quadrant (25 tiles, spawn (4,4)) and crop growth is deterministic.
    So a plan compiles to ONE tape (recorded via a single closed-loop serve pass,
    reusing the validated executor -- no dead-reckoning engine rewrite), and that
    tape is replayed across MANY seeds/opponents by `kagg batch` at ~7 ms/game
    each. Amortised over M games/plan the per-game cost -> 7 ms + compile/M, i.e.
    ~10-20 ms once M >= ~16. Opponent tapes are materialised once and cached.

    Reward is still the bit-exact engine's terminal bank; only the transport
    changed (open-loop batch instead of per-turn stdio).
    """

    def __init__(self, pool: Optional[OpponentPool] = None,
                 kagg: Optional[str] = None, workdir: Optional[str] = None,
                 compile_mode: str = "serve"):
        self.pool = pool or OpponentPool()
        self.compile_mode = compile_mode        # "serve" (default) or "deadreckon" (X4)
        # prefer the lean isolated kagg-engine (batch only), else deployed kagg
        cands = [kagg] if kagg else [
            os.path.join(ROOT, ".local", "scratch", "bandit_target", "release",
                         "kagg-engine.exe"),
            os.path.join(ROOT, ".local", "scratch", "bandit_target", "release",
                         "kagg.exe"),
            os.path.join(ROOT, "rustengine", "kagg.exe")]
        self.kagg = next((c for c in cands if c and os.path.exists(c)), None)
        if not self.kagg:
            raise SystemExit("no kagg binary with `batch` found (build kagg-engine)")
        self.workdir = workdir or os.path.join(
            ROOT, ".local", "scratch", "rl_tapes")
        os.makedirs(self.workdir, exist_ok=True)
        self._compiler = MacroEnv(pool=self.pool, backend="serve")
        self._opp_cache: Dict[str, str] = {}
        self._pool = None                   # persistent bounded compile pool
        self._pool_n = 0
        self.pool.prewarm_destbreso()       # ONE parquet pass, not per-lookup

    def _get_pool(self, n):
        """A PERSISTENT bounded process pool (workers stay warm across iters,
        avoiding Windows re-spawn overhead). Hard-capped for memory safety."""
        n = max(1, min(int(n), (os.cpu_count() or 4) - 1, 6))
        if self._pool is None or self._pool_n != n:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=n)
            self._pool_n = n
        return self._pool

    def compile_plan_tape(self, plan, path: str, ref_seed: int = 42) -> str:
        """Record the plan's (seed-independent) field+market tape to ``path``."""
        if self.compile_mode == "deadreckon":   # X4: no engine call
            dr = DeadReckonCompiler(self._compiler.executor)
            cells = dr.compile(plan, ref_seed)
        else:
            cells = self._compiler.rollout(plan, ref_seed, opp_cells=None,
                                           our_seat=0)["tape_cells"]
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(cells_to_batch_tape(cells, ref_seed))
        return path

    def parallel_compile(self, plans, n_workers: int = 3, ref_seed: int = 42,
                         exec_name: str = "seasoncbs"):
        """Compile MANY plans -> tape paths using a BOUNDED process pool.

        The plan-compile (serve) is the serial bottleneck; this parallelises it
        WITHOUT killing the box: n_workers is hard-capped (default 3) and the
        pool is torn down each call. The kagg batch play (all-cores) runs AFTER,
        so compile and batch never contend. Returns list of tape file paths.
        """
        n = max(1, min(int(n_workers), (os.cpu_count() or 4) - 1, 6))  # HARD CAP
        paths = [os.path.join(self.workdir, f"plan_{os.getpid()}_{i}.tape")
                 for i in range(len(plans))]
        if n <= 1 or len(plans) <= 1 or self.compile_mode == "deadreckon":
            for pl, p in zip(plans, paths):     # serial fallback
                self.compile_plan_tape(pl, p, ref_seed)
            return paths
        args = [(pl, exec_name, ref_seed) for pl in plans]
        try:
            tapes = list(self._get_pool(n).map(_compile_worker, args))
        except Exception:                       # any pool issue -> safe serial
            self._pool = None
            for pl, p in zip(plans, paths):
                self.compile_plan_tape(pl, p, ref_seed)
            return paths
        for t, p in zip(tapes, paths):
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(t)
        return paths

    def rollout_many(self, plan_tapes, per_plan_games, our_seat: int = 0,
                     threads: int = 0):
        """Play MANY pre-compiled plan tapes vs their opponents in ONE batch call.

        ``per_plan_games`` = list (aligned to plan_tapes) of [(entry, seed), ...].
        Returns list (aligned) of per-plan result lists. One kagg batch call
        maximises core use across all games."""
        import subprocess
        pass_tape = os.path.join(self.workdir, "pass.tape")
        if not os.path.exists(pass_tape):
            with open(pass_tape, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(cells_to_batch_tape([], 42))
        jobs, meta = [], []
        for pi, (ptape, games) in enumerate(zip(plan_tapes, per_plan_games)):
            for entry, seed in games:
                ot = self.opp_tape(entry) or pass_tape
                a, b = (ptape, ot) if our_seat == 0 else (ot, ptape)
                jobs.append(f"{seed}\t{a}\t{b}")
                meta.append((pi, entry))
        jobs_path = os.path.join(self.workdir, f"jobsmany_{os.getpid()}.tsv")
        with open(jobs_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(jobs) + "\n")
        th = str(threads or os.cpu_count() or 4)
        out = subprocess.run([self.kagg, "batch", jobs_path, th],
                             capture_output=True, text=True, cwd=ROOT)
        res = [[] for _ in plan_tapes]
        for line in out.stdout.splitlines():
            f = line.split("\t")
            if len(f) < 4 or f[1] == "ERR":
                continue
            i, seed, b0, b1 = int(f[0]), int(f[1]), float(f[2]), float(f[3])
            pi, entry = meta[i]
            our = b0 if our_seat == 0 else b1
            opp = b1 if our_seat == 0 else b0
            win = 1.0 if our > opp else (0.5 if our == opp else 0.0)
            res[pi].append(dict(seed=seed, our_bank=our, opp_bank=opp, win=win,
                                team=entry.get("team"), rating=entry.get("rating")))
        return res

    def opp_tape(self, entry) -> Optional[str]:
        """Materialise an opponent (destbreso cells) to a cached batch tape."""
        if entry.get("kind") == "league_tape":     # T1: a pre-compiled self tape
            p = entry.get("path")
            return p if p and os.path.exists(p) else None
        if entry.get("kind") in (None, "pass", "frozen_self", "agent"):
            return None
        key = f"{entry.get('kind')}:{entry.get('episode_id')}:{entry.get('seat')}"
        if key in self._opp_cache:
            return self._opp_cache[key]
        cells = self.pool.cells_for(entry)
        if not cells:
            return None
        p = os.path.join(self.workdir, f"opp_{abs(hash(key))%10**10}.tape")
        seed = int(entry.get("seed", 42)) & 0x7fffffff or 42
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(cells_to_batch_tape(cells, seed))
        self._opp_cache[key] = p
        return p

    def rollout_batch(self, plan, games, ref_seed: int = 42, our_seat: int = 0,
                      threads: int = 0):
        """Play one plan vs many (opponent_entry, seed) games in ONE batch.

        ``games`` = list of (entry, seed). Returns list of dict(seed, our_bank,
        opp_bank, win, reward, team, rating).
        """
        import subprocess
        plan_tape = os.path.join(self.workdir, f"plan_{os.getpid()}.tape")
        self.compile_plan_tape(plan, plan_tape, ref_seed)
        pass_tape = os.path.join(self.workdir, "pass.tape")
        if not os.path.exists(pass_tape):
            with open(pass_tape, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(cells_to_batch_tape([], ref_seed))
        jobs, meta = [], []
        for entry, seed in games:
            ot = self.opp_tape(entry) or pass_tape
            a, b = (plan_tape, ot) if our_seat == 0 else (ot, plan_tape)
            jobs.append(f"{seed}\t{a}\t{b}")
            meta.append((entry, seed))
        jobs_path = os.path.join(self.workdir, f"jobs_{os.getpid()}.tsv")
        with open(jobs_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(jobs) + "\n")
        th = str(threads or os.cpu_count() or 4)
        out = subprocess.run([self.kagg, "batch", jobs_path, th],
                             capture_output=True, text=True, cwd=ROOT)
        res = []
        for line in out.stdout.splitlines():
            f = line.split("\t")
            if len(f) < 4 or f[1] == "ERR":
                continue
            i, seed, b0, b1 = int(f[0]), int(f[1]), float(f[2]), float(f[3])
            entry, _ = meta[i]
            our = b0 if our_seat == 0 else b1
            opp = b1 if our_seat == 0 else b0
            win = 1.0 if our > opp else (0.5 if our == opp else 0.0)
            res.append(dict(seed=seed, our_bank=our, opp_bank=opp, win=win,
                            reward=win, team=entry.get("team"),
                            rating=entry.get("rating")))
        return res


class DeadReckonCompiler:
    """X4: compile a plan -> tape with NO engine call (dead-reckoning).

    Maintains a Python belief of the FIELD state (owned tiles, mover positions,
    tile plant/water/yield, shed, seeds, money) and steps it forward from the
    executor's own actions, mirroring the engine's crop rules (WATER yield window
    [ceil((max_yield_day+1)/2), max_yield_day]; HARVEST at age>=first_yield_day).
    It does NOT model the seed's weed RNG or market prices -- weeds are rare when
    we water, and sells are priced by the REAL engine at batch time -- so the
    field tape it emits is seed-independent, exactly like the serve-compiled one.

    Fidelity is validated against the serve compile before this is used as the
    default; ``BatchRoller(compile_mode="deadreckon")`` opts in.
    """

    def __init__(self, executor_factory):
        self._factory = executor_factory

    def compile(self, plan, ref_seed: int = 42) -> List[dict]:
        ctrl = self._factory(plan)
        st = _DRState()
        cells = []
        for step in range(EPISODE_STEPS - 1):
            obs = st.obs()
            act = ctrl.agent(obs)
            st.apply(act)
            cells.append(act)
            st.tick()
        return cells


class _DRState:
    """Minimal engine belief for dead-reckoning (field mechanics only)."""
    def __init__(self):
        self.day = 0
        self.hour = 0
        self.money = 3000.0
        self.tiles = [["LOCKED"] * 10 for _ in range(10)]
        for y in range(5):
            for x in range(5):
                self.tiles[y][x] = None            # NW quadrant owned
        self.unlocked = ["NW"]
        self.farmer = [4, 4]
        self.hands = []
        self.hires_today = 0
        self.seeds = {}
        self.shed = {}
        self.inv = [{}]                            # farmer + hands inventories

    def obs(self):
        farms = [{"money": self.money, "farmer": list(self.farmer),
                  "hands": [list(h) for h in self.hands], "tiles": self.tiles,
                  "unlocked_quadrants": self.unlocked, "hires_today": self.hires_today},
                 {"money": 3000, "farmer": [4, 4], "hands": [], "tiles": self.tiles,
                  "unlocked_quadrants": ["NW"]}]
        return {"day": self.day, "hour": self.hour, "player": 0, "farms": farms,
                "market": {"inventory": {}, "prices": {}},
                "private": {"shed": self.shed, "seeds": self.seeds,
                            "inventories": self.inv}}

    def _move(self, pos, verb):
        x, y = pos
        if verb == "NORTH":
            y = max(0, y - 1)
        elif verb == "SOUTH":
            y = min(9, y + 1)
        elif verb == "EAST":
            x = min(9, x + 1)
        elif verb == "WEST":
            x = max(0, x - 1)
        return [x, y]

    def _unit(self, movers, i, op):
        if not op:
            return
        verb = op[0]
        pos = movers[i]
        x, y = pos
        if verb in ("NORTH", "SOUTH", "EAST", "WEST"):
            movers[i] = self._move(pos, verb)
            return
        t = self.tiles[y][x] if self.tiles[y][x] != "LOCKED" else None
        if verb == "PLANT" and len(op) >= 2:
            crop = op[1]
            if self.tiles[y][x] is None and self.seeds.get(crop, 0) > 0 \
                    and _CROPS and crop in _CROPS:
                self.seeds[crop] -= 1
                cd = _CROPS[crop]
                self.tiles[y][x] = {"kind": "PLANT", "crop": crop,
                                    "planted_day": self.day, "watered_today": False,
                                    "yield_units": 0 if cd["ongoing"] else 1,
                                    "fertilized_until_day": -1}
        elif verb == "WATER" and isinstance(t, dict) and t.get("kind") == "PLANT":
            if not t["watered_today"]:
                t["watered_today"] = True
                cd = _CROPS[t["crop"]] if _CROPS else {}
                if cd and not cd["ongoing"]:
                    age = self.day - t["planted_day"]
                    ws = (cd["max_yield_day"] + 1) // 2
                    if ws <= age <= cd["max_yield_day"]:
                        t["yield_units"] = min(cd["max_yield"], t["yield_units"] + 1)
        elif verb == "HARVEST" and isinstance(t, dict):
            yu = t.get("yield_units", 0)
            crop = t.get("crop")
            cd = _CROPS.get(crop) if _CROPS and crop else None
            if yu > 0 and cd and self.day - t.get("planted_day", 0) >= cd["first_yield_day"]:
                self.shed[crop] = self.shed.get(crop, 0) + yu
                t["yield_units"] = 0
                if not cd["ongoing"]:
                    self.tiles[y][x] = None

    def apply(self, act):
        movers = [self.farmer] + self.hands
        self._unit(movers, 0, act.get("farmer"))
        for hi, h in enumerate(act.get("hands") or []):
            if hi + 1 < len(movers):
                self._unit(movers, hi + 1, h)
        self.farmer = movers[0]
        self.hands = movers[1:]
        for o in (act.get("market") or []):
            if not o:
                continue
            v = o[0]
            if v == "HIRE":
                self.money -= 1
                self.hands.append([4, 4])
                self.inv.append({})
                self.hires_today += 1
            elif v == "BUY_LAND":
                n = len(self.unlocked) - 1
                if n < 3:
                    self.money -= (1000, 2000, 4000)[n]
                    quad = ["NE", "SW", "SE"][n]
                    self.unlocked.append(quad)
                    ox, oy = {"NE": (5, 0), "SW": (0, 5), "SE": (5, 5)}[quad]
                    for yy in range(oy, oy + 5):
                        for xx in range(ox, ox + 5):
                            self.tiles[yy][xx] = None
            elif v == "BUY_SEED" and len(o) >= 3:
                self.seeds[o[1]] = self.seeds.get(o[1], 0) + int(o[2])
            elif v == "SELL" and len(o) >= 3:
                self.shed[o[1]] = max(0, self.shed.get(o[1], 0) - int(o[2]))
            elif v == "BUY_PRODUCT" and len(o) >= 3:
                self.shed[o[1]] = self.shed.get(o[1], 0) + int(o[2])

    def tick(self):
        self.hour += 1
        if self.hour >= TURNS_PER_DAY:
            self.hour = 0
            self.day += 1
            self.farmer = [4, 4]
            self.hands = [[4, 4] for _ in self.hands]
            self.hires_today = 0
            for row in self.tiles:
                for t in row:
                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        t["watered_today"] = False


def cells_to_batch_tape(cells: List[dict], seed: int) -> str:
    """Serialise recorded action cells to the Rust ``kagg batch`` tape format.

    Line 0 = ``SEED <n>``; each line = ``farmer<TAB>hand;hand<TAB>order;order``
    with space-separated tokens (matches ``service::parse_action_line``).
    """
    def toks(op):
        return " ".join(str(t) for t in op) if op else "PASS"
    lines = [f"SEED {seed}"]
    for i in range(EPISODE_STEPS - 1):
        c = cells[i] if i < len(cells) else {"farmer": ["PASS"], "hands": [],
                                             "market": []}
        farmer = toks(c.get("farmer") or ["PASS"])
        hands = ";".join(toks(h) for h in (c.get("hands") or []))
        mkt = ";".join(" ".join(str(t) for t in o)
                       for o in (c.get("market") or []))
        lines.append(f"{farmer}\t{hands}\t{mkt}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def _demo_plan() -> List[MA.MacroAction]:
    """A small hand plan: buy seed + hire early, plant melon+carrot, sell late."""
    plan = []
    for day in range(MA.__dict__.get("MAX_DAYS", 30) if False else 30):
        ma = MA.MacroAction()
        if day == 0:
            ma.hire = 4
            ma.buy_seed = {"MELON": 8, "CARROT": 12}
            ma.plant = {"MELON": 6, "CARROT": 8}
        elif day < 6:
            ma.buy_seed = {"CARROT": 6}
            ma.plant = {"CARROT": 6}
        elif day >= 8:
            ma.sell = {"MELON": 50, "CARROT": 80}
        plan.append(ma)
    return plan


def _smoke() -> int:
    env = MacroEnv(margin_shaping=1.0)
    plan = _demo_plan()
    # 1) vs PASS
    r0 = env.rollout(plan, seed=42, opp_cells=None, our_seat=0)
    assert r0["our_bank"] >= 0, r0
    print(f"[env][smoke] vs PASS  our={r0['our_bank']:.0f} opp={r0['opp_bank']:.0f} "
          f"win={r0['win']} weeds={r0['weeds']} cells={len(r0['tape_cells'])}")
    assert len(r0["tape_cells"]) >= 700, "did not record a full tape"
    # tape serialises to the batch format
    tape = cells_to_batch_tape(r0["tape_cells"], 42)
    assert tape.startswith("SEED 42\n") and tape.count("\n") >= 719
    # 2) vs a real destbreso opponent if the pool + parquet are present
    pool = env.pool
    db = next((e for e in pool.entries if e.get("kind") == "destbreso_tape"), None)
    if db is not None:
        cells = pool.cells_for(db)
        if cells:
            r1 = env.rollout(plan, seed=int(db.get("seed", 42)) & 0x7fffffff,
                             opp_cells=cells, our_seat=0)
            print(f"[env][smoke] vs {db.get('team')} (r{db.get('rating'):.0f})  "
                  f"our={r1['our_bank']:.0f} opp={r1['opp_bank']:.0f} win={r1['win']}")
        else:
            print("[env][smoke] destbreso cells not materialised (parquet absent) -- OK")
    else:
        print("[env][smoke] no destbreso opponent in pool -- vs-PASS only, OK")
    print("[env][smoke] OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    # default: one demo rollout vs PASS
    env = MacroEnv()
    r = env.rollout(_demo_plan(), seed=args.seed)
    print(json.dumps({k: v for k, v in r.items() if k != "tape_cells"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
