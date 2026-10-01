"""Leak audit: is this build's economy the top-10 economy? (Tier 0, 2026-09-03)

Plays N games (default 6 seeds, one DISTINCT world each, rotating the gauntlet
opponent from serve_gate.OPPS and alternating the seat -- playing one seed from
both seats reproduces the game to the dollar and measures nothing twice) on the
OFFICIAL vendored engine -- the full action
stream and every per-turn observation (incl. the private shed/seed block)
are needed, and only env.steps carries them -- then measures, per game, the
leaks the 2026-09-03 audit found (docs/history/plan-2800-2026-09-03.md section 1) and
compares each to the hardcoded TOP-10 MEDIANS from that audit. A metric that
is worse than the top-10 value by more than 25% in the costly direction is
flagged REGRESSION.

    python src/leak_audit.py agents/v42.0_bandit.py [-n 6] [--workers 6]
    python src/leak_audit.py A.py --seeds 1,2,3 --opps 3

Output: a table on stdout and .local/leak_audit/<agent>.json. Exit code 1 when
any metric's median regresses (so it can stand in a build chain as a test).
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(ROOT, ".local", "leak_audit")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER")
TURNS_PER_DAY = 24

# TOP-10 medians, 2026-09-03 audit (plan section 1). `worse` = the costly
# direction: "high" means a larger value than the reference is the leak.
REFERENCE = {
    "feed_wheat":   (183, "high", "BUY_PRODUCT WHEAT units"),
    "fert_bought":  (62,  "high", "BUY_PRODUCT FERTILIZER units"),
    "pass":         (513, "high", "explicit PASS ops (farmer + hands)"),
    "feed":         (326, "low",  "FEED ops"),
    "care":         (333, "low",  "CARE ops"),
    "collect":      (338, "low",  "COLLECT_FERTILIZER ops"),
    "pastures":     (17,  "low",  "BUILD_PASTURE ops"),
    "cows":         (9,   "low",  "BUY_ANIMAL COW"),
    "sheep":        (5,   "low",  "BUY_ANIMAL SHEEP"),
    "seed_held":    (3,   "high", "unplanted seeds, mean of days 15/20/24"),
    "milk_held":    (0,   "high", "shed MILK, mean over days 8-12 turns"),
    "starving_d10": (0,   "high", "animals consecutive_unfed>=1 at day 10"),
    "tiles_d28":    (27,  "high", "planted tiles at day 28"),
    "shed_700":     (24,  "low",  "shed product units at step 700"),
}
# Reported, never flagged: no top-10 reference in the audit or not a leak.
INFO_KEYS = ("shed_719", "bank", "idle_units", "seed_d15", "seed_d20",
             "seed_d24", "won")
TOLERANCE = 0.25


# ------------------------------------------------------------------ measure

def _ops(action):
    """[farmer_op, hand_op, ...] with absent entries normalised to ["PASS"]."""
    action = action if isinstance(action, dict) else {}
    farmer = action.get("farmer")
    hands = action.get("hands") or []
    out = [farmer if isinstance(farmer, list) and farmer else ["PASS"]]
    for h in hands:
        out.append(h if isinstance(h, list) and h else ["PASS"])
    return out


def _tiles(farm):
    for row in (farm.get("tiles") or []):
        for t in (row if isinstance(row, list) else [row]):
            if isinstance(t, dict):
                yield t


def _shed_products(private):
    shed = (private or {}).get("shed") or {}
    return sum(float(shed.get(p, 0) or 0) for p in PRODUCTS)


def metrics_from_steps(steps, seat):
    """Per-game leak metrics for `seat` out of a replay's `steps`.

    `steps[k][seat]` carries "observation" (public farms/market/town plus the
    seat's private block) and "action" (the action applied to reach step k).
    Works on plain dicts and on kaggle_environments Structs alike.
    """
    counts = {}
    market = {}
    idle = 0
    by_step = {}
    for k, st in enumerate(steps):
        if not st or seat >= len(st):
            continue
        obs = st[seat].get("observation") or {}
        pub = st[0].get("observation") or {}
        farms = pub.get("farms") or []
        if len(farms) > seat:
            step_no = int(pub.get("step", k) or 0)
            by_step[step_no] = (farms[seat], obs.get("private") or {})
        if k == 0:
            continue
        ops = _ops(st[seat].get("action"))
        for op in ops:
            counts[op[0]] = counts.get(op[0], 0) + 1
        prev = steps[k - 1][0].get("observation") or {}
        pf = (prev.get("farms") or [])
        units = 1 + len((pf[seat].get("hands") or []) if len(pf) > seat else [])
        idle += max(0, units - sum(1 for op in ops if op[0] != "PASS"))
        act = st[seat].get("action")
        for order in ((act or {}).get("market") or []) if isinstance(act, dict) else []:
            if isinstance(order, list) and len(order) >= 2:
                try:
                    qty = float(order[2]) if len(order) > 2 else 1.0
                except (TypeError, ValueError):
                    qty = 0.0
                key = f"{order[0]}:{order[1]}"
                market[key] = market.get(key, 0.0) + qty

    def at(step_no):
        return by_step.get(step_no, ({}, {}))

    def seeds_at(day):
        _, priv = at(day * TURNS_PER_DAY)
        return sum(float(v or 0) for v in (priv.get("seeds") or {}).values())

    milk = [float(((at(s)[1]).get("shed") or {}).get("MILK", 0) or 0)
            for s in range(8 * TURNS_PER_DAY, 13 * TURNS_PER_DAY) if s in by_step]
    farm10, _ = at(10 * TURNS_PER_DAY)
    starving = sum(1 for t in _tiles(farm10)
                   if t.get("animal") and (t.get("consecutive_unfed") or 0) >= 1)
    farm28, _ = at(28 * TURNS_PER_DAY)
    tiles28 = sum(1 for t in _tiles(farm28)
                  if t.get("kind") == "PLANT" and t.get("crop"))
    last = max(by_step) if by_step else 0
    final_farm, _ = at(last)
    m = {
        "feed_wheat": market.get("BUY_PRODUCT:WHEAT", 0.0),
        "fert_bought": market.get("BUY_PRODUCT:FERTILIZER", 0.0),
        "pass": counts.get("PASS", 0),
        "idle_units": idle,
        "feed": counts.get("FEED", 0),
        "care": counts.get("CARE", 0),
        "collect": counts.get("COLLECT_FERTILIZER", 0),
        "pastures": counts.get("BUILD_PASTURE", 0),
        "cows": market.get("BUY_ANIMAL:COW", 0.0),
        "sheep": market.get("BUY_ANIMAL:SHEEP", 0.0),
        "chickens": market.get("BUY_ANIMAL:CHICKEN", 0.0),
        "seeds_bought": {k.split(":")[1]: v for k, v in market.items()
                         if k.startswith("BUY_SEED:")},
        "seed_d15": seeds_at(15), "seed_d20": seeds_at(20),
        "seed_d24": seeds_at(24),
        "milk_held": statistics.mean(milk) if milk else 0.0,
        "starving_d10": starving,
        "tiles_d28": tiles28,
        "shed_700": _shed_products(at(700)[1]),
        "shed_719": _shed_products(at(719)[1]),
        "bank": float(final_farm.get("money") or 0),
    }
    m["seed_held"] = (m["seed_d15"] + m["seed_d20"] + m["seed_d24"]) / 3.0
    return m


def regression(value, ref, worse, tol=TOLERANCE):
    """True when `value` is worse than `ref` by more than `tol` (25%) in the
    costly direction; the tolerance never drops below 1 unit so a zero
    reference (milk held, starving animals) flags real leaks, not rounding."""
    margin = max(tol * abs(ref), 1.0)
    if worse == "high":
        return value > ref + margin
    return value < ref - margin


def summarise(games):
    """Median per metric over games + flags. Returns (table_rows, any_flag)."""
    rows = []
    any_flag = False
    for key, (ref, worse, desc) in REFERENCE.items():
        vals = [g["metrics"][key] for g in games]
        med = statistics.median(vals) if vals else 0.0
        flag = regression(med, ref, worse)
        any_flag |= flag
        rows.append({"metric": key, "desc": desc, "top10": ref, "ours": med,
                     "worse": worse, "regression": flag,
                     "per_game": vals})
    return rows, any_flag


# --------------------------------------------------------------------- play

def play(job):
    """Worker: one official-engine game -> per-game metrics for `seat`."""
    agent, opp, seed, seat = job
    os.chdir(ROOT)
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()          # inherited from the parent (env marker)
    import kaggriculture.engine.serve_match as SM
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    from kaggle_environments import make
    left, right = (agent, opp) if seat == 0 else (opp, agent)
    env = make("kaggriculture", configuration={
        "episodeSteps": 720, "seed": seed, "actTimeout": 60,
        "runTimeout": 100000}, debug=False)
    t0 = time.time()
    env.run([SM.load_agent(left), SM.load_agent(right)])
    m = metrics_from_steps(env.steps, seat)
    final = env.steps[-1]
    own = float(final[seat].get("reward") or 0)
    other = float(final[1 - seat].get("reward") or 0)
    m["won"] = 1 if own > other else (0.5 if own == other else 0)
    return {"opp": opp, "seed": seed, "seat": seat, "own": own, "opp_bank": other,
            "secs": round(time.time() - t0, 1), "metrics": m}


def jobs_for(agent, n, seeds, opps, both_seats=False):
    """The (agent, opp, seed, seat) list: one DISTINCT world per game.

    Playing the same (opponent, seed) from both seats looks like two games and
    is one: the engine starts the two farms identical and both sides here are
    deterministic, so the swap reproduces the cell to the dollar (measured
    2026-09-03: 3/3 pairs in this audit and 162/166 cells on the top-100 band
    panel came back byte-identical). A median over 6 such games is a median
    over 3. So the default rotates the SEED, alternating the seat for the
    asymmetry check; `both_seats` restores the old mirrored pairing.
    """
    agent = os.path.abspath(agent)
    if both_seats:
        out = [(agent, opps[i % len(opps)], seed, seat)
               for i, seed in enumerate(seeds) for seat in (0, 1)]
    else:
        out = [(agent, opps[i % len(opps)], seed, i % 2)
               for i, seed in enumerate(seeds)]
    return out[:n] if n else out


def audit(agent, n=6, seeds=None, n_opps=3, workers=6, both_seats=False):
    import kaggriculture.engine.engine_check as engine_check
    import kaggriculture.engine.serve_gate as SG
    engine_check.require(verbose=True)
    opps = [os.path.join(ROOT, o) for o in SG.OPPS[:n_opps]]
    if not seeds:
        seeds = list(range(1, (n // 2 if both_seats else n) + 1))
    jobs = jobs_for(agent, n, list(seeds), opps, both_seats)
    t0 = time.time()
    print(f"{os.path.basename(agent)}: {len(jobs)} official-engine games "
          f"({len({(j[1], j[2]) for j in jobs})} distinct worlds) on "
          f"{workers} worker(s)...", flush=True)
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        games = list(ex.map(play, jobs, chunksize=1))
    print(f"  done in {time.time() - t0:.0f}s")
    rows, any_flag = summarise(games)
    name = os.path.basename(agent)
    print(f"\nLEAK AUDIT {name}  ({len(games)} games; medians vs top-10 "
          f"medians of the 2026-09-03 audit; flag = >25% worse)")
    print(f"  {'metric':<13} {'top10':>7} {'ours':>8}  {'flag':<10} per-game")
    for r in rows:
        pg = " ".join(f"{v:.0f}" for v in r["per_game"])
        print(f"  {r['metric']:<13} {r['top10']:>7} {r['ours']:>8.1f}  "
              f"{'REGRESSION' if r['regression'] else 'ok':<10} {pg}")
    info = {k: statistics.median([g["metrics"][k] for g in games])
            for k in INFO_KEYS}
    sb = {}
    for g in games:
        for c, v in g["metrics"]["seeds_bought"].items():
            sb.setdefault(c, []).append(v)
    info["seeds_bought"] = {c: statistics.median(v) for c, v in sb.items()}
    info["chickens"] = statistics.median([g["metrics"]["chickens"] for g in games])
    print("  info: " + ", ".join(
        f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in info.items()))
    print(f"  games: " + ", ".join(
        f"{os.path.basename(g['opp']).replace('.py', '')} s{g['seed']} "
        f"seat{g['seat']} {'W' if g['metrics']['won'] == 1 else 'L'} "
        f"{g['own']:.0f}/{g['opp_bank']:.0f}" for g in games))
    verdict = "REGRESSION" if any_flag else "CLEAN"
    print(f"LEAK VERDICT: {verdict} ({sum(r['regression'] for r in rows)} "
          f"metric(s) flagged)")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{os.path.splitext(name)[0]}.json")
    json.dump({"agent": agent, "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "engine": "official (vendor)", "verdict": verdict,
               "table": rows, "info": info, "games": games},
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}")
    return verdict, rows, games


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent")
    ap.add_argument("-n", type=int, default=6,
                    help="games; one distinct world each")
    ap.add_argument("--seeds", default=None, help="comma list; default 1..n")
    ap.add_argument("--opps", type=int, default=3,
                    help="gauntlet opponents from serve_gate.OPPS (rotating)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--both-seats", action="store_true",
                    help="play every (opponent, seed) from both seats -- "
                         "halves the distinct worlds; only for an asymmetry "
                         "check on a NON-deterministic agent")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    verdict, _, _ = audit(args.agent, n=args.n, seeds=seeds, n_opps=args.opps,
                          workers=args.workers, both_seats=args.both_seats)
    return 1 if verdict == "REGRESSION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
