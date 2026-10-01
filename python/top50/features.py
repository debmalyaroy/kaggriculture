"""Per-game and per-turn features of the top-50 teams' games (queue Q34, step 1 of 2).

    python python/top50/features.py [--root data/top50/fetch] [--procs 24]

Reads data/top50/{index.tsv, tapes/sNN/*.json, trace/sNN.tsv} (python/top50/extract.py) and writes
  data/top50/feat/games.parquet    one row per (game, top seat): result, world, opening / plan hashes,
                                   rival-copy signal, economy (farm by day, buys), sales (units, revenue,
                                   timing), money by day
  data/top50/feat/sells.parquet    per turn and item where the top player held stock: what they could
                                   see and whether they sold (the "reactive shell" decision)
Fills are estimated from the shed: units that left the shed on a turn with a SELL order for that item
(capped by the order). Everything a player could not see (rival shed, seed) is kept out of sells.parquet.
"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOP = os.path.join(RL, "data", "top50", "fetch")  # --root; set in main() before the pool starts (fork)
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SHOP_ITEMS = {"BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"], "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
              "YARN_STORE": ["WOOL"], "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
              "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]}
NI = len(ITEMS)
SELL_ROWS_PER_GAME = 400  # cap: held-stock turns sampled per game (all sell turns kept)


def h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def unit_ops(a):
    """farmer + hands unit actions as tuples (the economy side)."""
    if not isinstance(a, dict):
        return []
    ops = []
    f = a.get("farmer")
    if f:
        ops.append(tuple(f))
    for x in a.get("hands") or []:
        if x:
            ops.append(tuple(x))
    return ops


def market_ops(a):
    return [tuple(x) for x in (a.get("market") or [])] if isinstance(a, dict) else []


def load_trace(path):
    S, F, W = [], [], []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            x = ln.rstrip("\n").split("\t")
            if x[0] == "S":
                S.append(x[1:])
            elif x[0] == "F":
                F.append(x[1:])
            elif x[0] == "W":
                W.append(x[1:])
    return S, F, W


def shard(k):
    tr = os.path.join(TOP, "trace", f"s{k:02d}.tsv")
    if not os.path.exists(tr):
        return None
    S, F, W = load_trace(tr)
    sd = {}
    for x in S:
        sd.setdefault(x[0], []).append(x)
    fd = {}
    for x in F:
        fd.setdefault(x[0], []).append(x)
    games, sells = [], []
    rng = np.random.default_rng(k)
    for w in W:
        gid, seed, seat = w[0], int(w[1]), int(w[2])
        w1, w2, i1, i2 = w[3], w[4], w[5], w[6]
        rep = [float(w[7]), float(w[8])]
        exact = w[11] == "1"
        tape = json.load(open(os.path.join(TOP, "tapes", f"s{k:02d}", f"{gid}.json"), encoding="utf-8"))
        g_extra = {k2: tape.get(k2) for k2 in ("team", "team_id", "sub", "rating", "opp_team_id", "opp_rating", "end")}
        acts = tape["actions"]
        own = [p[seat] if len(p) > seat else None for p in acts]
        riv = [p[1 - seat] if len(p) > 1 - seat else None for p in acts]
        s = np.array([[float(v) for v in x[3:]] for x in sd.get(gid, [])])  # step-major, cols below
        if len(s) < 700:
            continue
        # columns: 0 money_me 1 money_rv, 2..10 shed_me, 11..19 shed_rv, 20..28 mkt_inv, 29..37 mkt_px
        steps = len(s)
        money_me, money_rv = s[:, 0], s[:, 1]
        shed_me, inv, px = s[:, 2:2 + NI], s[:, 20:20 + NI], s[:, 29:29 + NI]
        g = {"id": gid, "seed": seed, "seat": seat, "world1": w1, "world2": w2, "idle1": i1, "idle2": i2,
             "moved1": int(w1 != i1), "moved2": int(w2 != i2), "exact": int(exact),
             "bank": rep[seat], "bank_rv": rep[1 - seat], "margin": rep[seat] - rep[1 - seat],
             "win": 1.0 if rep[seat] > rep[1 - seat] else 0.0 if rep[seat] < rep[1 - seat] else 0.5, **g_extra}
        # opening and plan hashes (economy side only, so a changed sale order does not split a plan)
        g["open_d0_2"] = h([unit_ops(a) for a in own[:72]])
        g["open_d0_5"] = h([unit_ops(a) for a in own[:144]])
        g["plan_d6_11"] = h([unit_ops(a) for a in own[144:288]])
        g["plan_d0_23"] = h([unit_ops(a) for a in own[:576]])
        g["mkt_d0_5"] = h([[m for m in market_ops(a) if m and m[0] != "SELL"] for a in own[:144]])
        # per-day fingerprints: economy (unit actions), buys (non-SELL market) and sales (SELL orders),
        # so two games can be compared day by day (first day they differ = where the plan branched)
        g["dh_unit"] = ",".join(h([unit_ops(a) for a in own[24 * d:24 * d + 24]])[:8] for d in range(30))
        g["dh_buy"] = ",".join(h([[m for m in market_ops(a) if m and m[0] != "SELL"] for a in own[24 * d:24 * d + 24]])[:8] for d in range(30))
        g["dh_sell"] = ",".join(h([[m for m in market_ops(a) if m and m[0] == "SELL"] for a in own[24 * d:24 * d + 24]])[:8] for d in range(30))
        # rival copy signal: share of day-0 turns where the rival's unit actions equal ours
        g["copy_d0"] = float(np.mean([unit_ops(a) == unit_ops(b) for a, b in zip(own[:24], riv[:24])]))
        g["copy_d0_5"] = float(np.mean([unit_ops(a) == unit_ops(b) for a, b in zip(own[:144], riv[:144])]))
        g["cash_eq1"] = int(money_me[0] == money_rv[0])
        # first divergence from the rival (step), a proxy for when a copy race starts or ends
        dv = next((t for t, (a, b) in enumerate(zip(own, riv)) if unit_ops(a) != unit_ops(b)), 720)
        g["diverge_step"] = dv
        # economy: farm composition on days 6, 12, 18, 24, 29
        fr = {int(x[2]): [float(v) for v in x[3:]] for x in fd.get(gid, [])}
        for d in (6, 12, 18, 24, 29):
            v = fr.get(d)
            if v is None:
                continue
            g[f"land_d{d}"], g[f"hands_d{d}"] = v[0], v[1]
            for i, c in enumerate(CROPS):
                g[f"{c.lower()}_tiles_d{d}"] = v[2 + i]
            for i, an in enumerate(ANIMALS):
                g[f"{an.lower()}_d{d}"] = v[7 + i]
        # buys and structures by phase
        phases = [(0, 144), (144, 288), (288, 432), (432, 576), (576, 720)]
        for pi, (a0, a1) in enumerate(phases):
            seeds, animals, hires, land, builds = {c: 0 for c in CROPS}, {x: 0 for x in ANIMALS}, 0, 0, 0
            for a in own[a0:a1]:
                for m in market_ops(a):
                    if not m:
                        continue
                    if m[0] == "BUY_SEED" and len(m) >= 3 and m[1] in seeds:
                        seeds[m[1]] += int(m[2] or 0)
                    elif m[0] == "BUY_ANIMAL" and len(m) >= 3 and m[1] in animals:
                        animals[m[1]] += int(m[2] or 0)
                    elif m[0] == "HIRE":
                        hires += 1
                    elif m[0] == "BUY_LAND":
                        land += 1
                for u in unit_ops(a):
                    if u and u[0] in ("BUILD_COOP", "BUILD_PASTURE"):
                        builds += 1
            for c in CROPS:
                g[f"seed_{c.lower()}_p{pi}"] = seeds[c]
            for x in ANIMALS:
                g[f"buy_{x.lower()}_p{pi}"] = animals[x]
            g[f"hire_p{pi}"], g[f"land_p{pi}"], g[f"build_p{pi}"] = hires, land, builds
        # sales: fills estimated from the shed on turns with a SELL order
        sold = np.zeros((steps, NI))
        ordered = np.zeros((steps, NI))
        for t in range(min(steps, len(own))):
            for m in market_ops(own[t]):
                if m and m[0] == "SELL" and len(m) >= 3 and m[1] in ITEMS:
                    ordered[t, ITEMS.index(m[1])] += int(m[2] or 0)
        prev = np.vstack([np.zeros((1, NI)), shed_me[:-1]])
        drop = np.clip(prev - shed_me, 0, None)
        sold = np.where(ordered > 0, np.minimum(drop, ordered), 0)
        pxp = np.vstack([px[:1], px[:-1]])
        rev = sold * pxp
        tot = sold.sum(0)
        g["units_total"] = float(tot.sum())
        g["revenue_est"] = float(rev.sum())
        for i, it in enumerate(ITEMS):
            g[f"sold_{it.lower()}"] = float(tot[i])
            g[f"rev_{it.lower()}"] = float(rev[:, i].sum())
            g[f"unsold_{it.lower()}"] = float(shed_me[-1, i])
        hours = np.arange(steps) % 24
        day = np.arange(steps) // 24
        su = sold.sum(1)
        if su.sum() > 0:
            g["sell_hour_mean"] = float((hours * su).sum() / su.sum())
            g["sell_share_last_day"] = float(su[day >= 29].sum() / su.sum())
            g["sell_share_d24_29"] = float(su[day >= 24].sum() / su.sum())
            g["sell_turns"] = int((su > 0).sum())
            g["units_per_sell_turn"] = float(su.sum() / max(1, (su > 0).sum()))
        # money at the end of days
        for d in (5, 11, 17, 23, 26, 28):
            t = min(steps - 1, 24 * (d + 1) - 1)
            g[f"money_d{d}"], g[f"money_rv_d{d}"] = money_me[t], money_rv[t]
        games.append(g)
        # sell/hold decisions (observable only): turns where we held stock of an item
        shop_items = set(sum((SHOP_ITEMS.get(x, []) for x in w2.split("|") if x), []))
        run_mean = np.cumsum(px, 0) / np.arange(1, steps + 1)[:, None]
        dinv = np.vstack([np.zeros((1, NI)), np.diff(inv, axis=0)])
        dpx = np.vstack([np.zeros((1, NI)), np.diff(px, axis=0)])
        cand = [(t, i) for t in range(1, steps) for i in range(NI - 1) if prev[t, i] > 0]
        pos = [c for c in cand if sold[c] > 0]
        neg = [c for c in cand if sold[c] == 0]
        if len(neg) > SELL_ROWS_PER_GAME:
            neg = [neg[j] for j in rng.choice(len(neg), SELL_ROWS_PER_GAME, replace=False)]
        for (t, i) in pos + neg:
            sells.append((gid, i, t, t // 24, t % 24, 720 - t, prev[t, i], inv[t - 1, i], pxp[t, i], pxp[t, i] / max(1.0, run_mean[t - 1, i]),
                          dpx[t - 1, i], dinv[t - 1, i], money_me[t - 1] - money_rv[t - 1], int(ITEMS[i] in shop_items),
                          float(sold[t, i] > 0), sold[t, i], ordered[t, i]))
    return games, sells


def main():
    global TOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--root", default=TOP)
    a = ap.parse_args()
    TOP = os.path.abspath(a.root)
    shards = sorted(int(f[1:3]) for f in os.listdir(os.path.join(TOP, "trace")) if f.endswith(".tsv"))
    games, sells = [], []
    with Pool(a.procs) as p:
        for r in p.imap_unordered(shard, shards):
            if r:
                games += r[0]
                sells += r[1]
                print(f"[feat] {len(games)} games, {len(sells)} sell rows", flush=True)
    os.makedirs(os.path.join(TOP, "feat"), exist_ok=True)
    g = pd.DataFrame(games)
    idx = pd.read_csv(os.path.join(TOP, "index.tsv"), sep="\t")
    keys = [k for k in ("id", "seat", "seed") if k in idx.columns]
    idx["id"] = idx["id"].astype(str)
    extra = [c for c in idx.columns if c not in g.columns or c in keys]
    g = g.merge(idx[extra], on=keys, how="left")
    g.to_parquet(os.path.join(TOP, "feat", "games.parquet"))
    cols = ["id", "item", "step", "day", "hour", "hours_left", "stock", "mkt_inv", "px", "px_rel", "dpx", "dinv", "money_gap",
            "shop_item", "sold", "units", "ordered"]
    s = pd.DataFrame(sells, columns=cols)
    s["item"] = s["item"].map(dict(enumerate(ITEMS)))
    s.merge(g[["id", "team"]], on="id", how="left").to_parquet(os.path.join(TOP, "feat", "sells.parquet"))
    print(f"[feat] -> {TOP}/feat: {len(g)} games, {len(s)} sell rows")


if __name__ == "__main__":
    main()
