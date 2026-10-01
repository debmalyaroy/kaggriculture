"""Synthetic self-play generator: (day-boundary state, final margin) pairs.

The value-net fuel for the reopened planner lane (operator 2026-09-06:
game tree over enumerated worlds, pruned by our move x opponent move;
Colab-ready, synthetic-only — everything derives from seed + rules).

Each job: pick a seed (world known from data/worlds/seed_bank.json), pick
two tapes from the pool (our builds' tapes + clone-winner routes +
mutated variants for diversity), play on kagg, and record for every day
boundary d8..d29 the OBSERVABLE state vector of both seats plus the final
bank margin. One 720-turn game yields ~22 labelled states per seat.

Output: data/selfplay/states_<shard>.jsonl
Row: {"seed", "world", "day", "f": [...features...], "y_margin", "y_win"}

Runs identically on this box or Colab (the kagg musl binary is the same
artifact the submission carries; pass --kagg to point at it).

    python src/trackp/harness/selfplay_gen.py --games 200 --shard 0
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

ITEMS = ("WOOL", "MILK", "EGG", "WHEAT", "MELON", "CARROT", "TOMATO",
         "FERTILIZER")




def board_feats(farm):
    """v2 board features (2026-09-07): standing value the money-only view
    was blind to -- the measured cause of the value net's 0.77 sign
    ceiling. Per seat: plants, standing yield units, weeds, animals,
    pending care bonus, hands."""
    plants = weeds = animals = yiel = care = 0
    for row in (farm.get("tiles") or []):
        for t in (row if isinstance(row, list) else [row]):
            if not isinstance(t, dict):
                continue
            k = t.get("kind")
            if k == "PLANT":
                plants += 1
                yiel += int(t.get("yield_units") or 0)
            elif k == "WEED":
                weeds += 1
            if t.get("animal"):
                animals += 1
                care += int(t.get("pending_care_bonus") or 0)
    hands = len(farm.get("hands") or [])
    return [plants / 50.0, yiel / 100.0, weeds / 20.0, animals / 20.0,
            care / 50.0, hands / 12.0]


def state_features(js, me, step):
    """Observable day-boundary state for seat `me` from a serve obs."""
    farms = js.get("farms") or [{}, {}]
    market = js.get("market") or {}
    prices = market.get("prices") or {}
    inv = market.get("inventory") or {}
    f = [step / 720.0,
         float(farms[me].get("money") or 0) / 1e5,
         float(farms[1 - me].get("money") or 0) / 1e5]
    for it in ITEMS:
        f.append(float(prices.get(it) or 0) / 100.0)
        f.append(float(inv.get(it) or 0) / 20000.0)
    shed = (js.get("private") or [{}, {}])
    try:
        sh = shed[me].get("shed") or {}
    except Exception:                                          # noqa: BLE001
        sh = {}
    for it in ITEMS:
        f.append(float(sh.get(it) or 0) / 100.0)
    f += board_feats(farms[me]) + board_feats(farms[1 - me])
    return [round(x, 5) for x in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--kagg", default=None)
    a = ap.parse_args()
    import kaggriculture.engine.serve_match as SM
    if a.kagg:
        SM.KAGG = a.kagg
        os.environ["KAGG_BIN"] = a.kagg
    from kaggriculture.trackp import routes_io as R
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    pool = [h["id"] for h in hits]
    # ALL families, not just clones (operator 2026-09-06: every economy):
    # strongest fresh non-clone winners, one per team
    idx = R.load_index()
    seen_teams = set()
    for k, v in sorted(idx["routes"].items(),
                       key=lambda t: -(t[1].get("bank") or 0)):
        if (v.get("engine") == "1.32.7" and v.get("won")
                and (v.get("bank") or 0) >= 100000
                and str(v.get("date", "")) >= "2026-09-01"
                and v.get("team") not in seen_teams
                and k not in pool):
            pool.append(k)
            seen_teams.add(v.get("team"))
        if len(seen_teams) >= 40:
            break
    rng = random.Random(1000 + a.shard)
    outd = os.path.join(ROOT, "data", "selfplay")
    os.makedirs(outd, exist_ok=True)
    outp = os.path.join(outd, f"states_{a.shard}.jsonl")
    seed_world = {s: w for w, ss in bank.items() for s in ss}
    seeds = list(seed_world)
    n_rows = 0
    with open(outp, "a", encoding="utf-8") as out:
        srv = SM.Serve()
        try:
            for gi in range(a.games):
                seed = rng.choice(seeds)
                ta, tb = rng.sample(pool, 2)
                try:
                    A = R.load_route(ta)
                    B = R.load_route(tb)
                except Exception:                              # noqa: BLE001
                    continue
                # GENGAME (2026-09-07): the whole game in ONE IPC --
                # the engine plays 720 steps in ~7 ms; the per-step
                # loop spent ~1 s/game on 719 stdio round-trips.
                la = chr(31).join(SM.action_to_line(
                    A[i] if i < len(A) else None) for i in range(719))
                lb = chr(31).join(SM.action_to_line(
                    B[i] if i < len(B) else None) for i in range(719))
                js = srv.cmd("GENGAME " + str(seed) + chr(30) + la
                             + chr(30) + lb)
                if "error" in js:
                    print(f"  SERVE-ERROR g{gi}: "
                          f"{str(js['error'])[:120]}", flush=True)
                    continue
                rows = []
                for ob in js.get("days") or []:
                    stp = int(ob.get("step") or 0)
                    for me in (0, 1):
                        rows.append((me, stp,
                                     state_features(ob, me, stp)))
                js = js.get("final") or {}
                farms = js.get("farms") or [{}, {}]
                banks = [float(farms[i].get("money") or 0) for i in (0, 1)]
                if gi < 3 or (gi + 1) % 25 == 0:
                    print(f"  g{gi} seed {seed} world "
                          f"{seed_world.get(seed)} banks "
                          f"{banks[0]:,.0f} vs {banks[1]:,.0f}", flush=True)
                if gi == 2 and all(b == 0 for b in banks):
                    raise SystemExit("SANITY: three games, all banks zero — "
                                     "action lines are not being played")
                for me, step, f in rows:
                    out.write(json.dumps(
                        {"seed": seed, "world": seed_world.get(seed),
                         "day": step // 24, "f": f,
                         "y_margin": round(banks[me] - banks[1 - me], 1),
                         "y_win": 1 if banks[me] > banks[1 - me] else
                         (0 if banks[me] < banks[1 - me] else 0.5)},
                        separators=(",", ":")) + "\n")
                    n_rows += 1
                if (gi + 1) % 25 == 0:
                    print(f"  {gi+1}/{a.games} games, {n_rows} rows",
                          flush=True)
        finally:
            srv.close()
    print(f"{n_rows} labelled states -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
