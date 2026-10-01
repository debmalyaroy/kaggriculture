"""Relationship miner #4 -- counterfactual regret on the Rust engine.

The only truly CAUSAL rung of the ladder: take a recorded game, change one
thing at a decision point, replay the remainder bit-exactly, and measure the
final-bank delta. Two counterfactual kinds:

  takeover   from turn t our seat switches from the recorded actions to the
             closed-loop planner (kagg serve, opponent = recorded tape).
             Positive delta at t = "the planner would have salvaged this
             game from here"; the t-profile localizes WHERE play went wrong.

  sell-shift all SELL orders of one day moved +/- SHIFT_TURNS (kagg batch,
             both seats tapes). The causal value of sell timing, per day.

Caveats, stated plainly: the opponent is OPEN-LOOP in every counterfactual
(they cannot react to the change), and P4.1 measured open-loop rankings as
only weakly predictive of closed-loop outcomes -- so regrets are evidence
for WHERE to look, never ship decisions. Official-engine confirmation stays
mandatory for anything acted on.

Usage:
  python src/trackp/regret.py --replay <file.json> --seat 0
  python src/trackp/regret.py --episode 93017130 --seat 0   (staged lookup)
  python src/trackp/regret.py --smoke                        (self-test)
Output: models/trackp/regret/<episode>_s<seat>.json
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import json
import os
import subprocess

try:
    from . import common
    from .serve_env import ServeEnv, seat_view
except ImportError:
    import sys
    from kaggriculture.trackp import common
    from kaggriculture.trackp.serve_env import ServeEnv, seat_view

OUT_DIR = os.path.join(common.MODELS, "regret")
os.makedirs(OUT_DIR, exist_ok=True)
TAKEOVER_DAYS = (4, 8, 12, 16, 20, 24)
SHIFT_TURNS = 24  # sell-shift = one full day


def _planner_ns():
    tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "planner_template.py")
    ns: dict = {}
    with open(tpl, encoding="utf-8") as fh:
        exec(compile(fh.read(), "planner_regret", "exec"), ns)  # noqa: S102
    return ns


def takeover_profile(rep: dict, seat: int) -> dict:
    """Replay to turn t on recorded actions, then hand over to the planner."""
    seed = common.replay_seed(rep)
    mine = common.replay_actions(rep, seat)
    theirs = common.replay_actions(rep, 1 - seat)
    opp_tape = os.path.join(OUT_DIR, "_opp.tape")
    with open(opp_tape, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"SEED {seed}\n")
        for a in theirs:
            fh.write(common.action_to_tape_lines(a) + "\n")
    base = common.final_banks(rep)
    base_gap = base[seat] - base[1 - seat]
    rows = []
    env = ServeEnv()
    try:
        for day in TAKEOVER_DAYS:
            t0 = day * common.TURNS_PER_DAY
            ns = _planner_ns()
            obs = env.reset(seed, opp_tape=opp_tape,
                            opp_seat=1 if seat == 0 else 0)
            t = 0
            while not obs.get("done"):
                if t < t0 and t < len(mine):
                    a = mine[t]
                else:
                    a = ns["agent"](seat_view(obs, seat))
                # serve places our action at the free seat itself (the OPP
                # declaration at RESET fixed which seat replays the tape)
                obs = env.step(a)
                t += 1
            b_me = float(obs["farms"][seat]["money"])
            b_op = float(obs["farms"][1 - seat]["money"])
            rows.append({"takeover_day": day,
                         "cf_gap": round(b_me - b_op, 0),
                         "recorded_gap": round(base_gap, 0),
                         "regret": round((b_me - b_op) - base_gap, 0)})
    finally:
        env.close()
    return {"kind": "takeover", "rows": rows}


def sell_shift_profile(rep: dict, seat: int) -> dict:
    """Move each day's SELL orders one day earlier/later; kagg batch replay."""
    seed = common.replay_seed(rep)
    mine = common.replay_actions(rep, seat)
    theirs = common.replay_actions(rep, 1 - seat)
    base = common.final_banks(rep)
    base_gap = base[seat] - base[1 - seat]

    def write_tape(actions, path):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"SEED {seed}\n")
            for a in actions:
                fh.write(common.action_to_tape_lines(a) + "\n")

    opp_tape = os.path.join(OUT_DIR, "_ss_opp.tape")
    write_tape(theirs, opp_tape)

    jobs, meta = [], []
    for day in range(2, 29):
        lo, hi = day * 24, (day + 1) * 24
        has_sell = any(o and o[0] == "SELL"
                       for t in range(lo, min(hi, len(mine)))
                       for o in (mine[t].get("market") or []))
        if not has_sell:
            continue
        for shift, tag in ((-SHIFT_TURNS, "earlier"), (SHIFT_TURNS, "later")):
            acts = copy.deepcopy(mine)
            moved = []
            for t in range(lo, min(hi, len(acts))):
                sells = [o for o in (acts[t].get("market") or [])
                         if o and o[0] == "SELL"]
                if sells:
                    acts[t]["market"] = [o for o in acts[t]["market"]
                                         if not (o and o[0] == "SELL")]
                    moved.append((t, sells))
            for t, sells in moved:
                nt = min(len(acts) - 1, max(0, t + shift))
                mk = list(acts[nt].get("market") or [])
                acts[nt]["market"] = (sells + mk)[:10]
            tp = os.path.join(OUT_DIR, f"_ss_{day}_{tag}.tape")
            write_tape(acts, tp)
            jobs.append((seed, tp, opp_tape) if seat == 0
                        else (seed, opp_tape, tp))
            meta.append((day, tag))
    if not jobs:
        return {"kind": "sell_shift", "rows": []}
    jf = os.path.join(OUT_DIR, "_ss_jobs.tsv")
    with open(jf, "w", encoding="utf-8", newline="\n") as fh:
        for s, a, b in jobs:
            fh.write(f"{s}\t{a}\t{b}\n")
    out = subprocess.run([common.KAGG, "batch", jf], capture_output=True,
                         text=True, timeout=600)
    rows = []
    for line in out.stdout.strip().splitlines():
        p = line.split("\t")
        if p[1] == "ERR":
            continue
        i = int(p[0])
        b0, b1 = float(p[2]), float(p[3])
        gap = (b0 - b1) if seat == 0 else (b1 - b0)
        my_bank = b0 if seat == 0 else b1
        day, tag = meta[i]
        # Open-loop tapes are cascade-coupled: a moved sell can starve later
        # buys (cash) or fire before the goods exist (shed) and derail the
        # whole run -- the overlay-transplant lesson. When the counterfactual
        # bank collapses far below the recorded one, the delta measures
        # FRAGILITY, not sell-timing value; flag it so nobody reads it raw.
        fragile = my_bank < 0.7 * base[seat]
        rows.append({"day": day, "shift": tag,
                     "cf_gap": round(gap, 0),
                     "delta_vs_recorded": round(gap - base_gap, 0),
                     "fragile": fragile})
    solid = [r for r in rows if not r["fragile"]]
    solid.sort(key=lambda r: -abs(r["delta_vs_recorded"]))
    fragile = [r for r in rows if r["fragile"]]
    return {"kind": "sell_shift", "rows": solid,
            "fragile_discarded": len(fragile)}


def mine_replay(path: str, seat: int) -> dict:
    rep = common.load_replay(path)
    eid = rep.get("info", {}).get("EpisodeId") or \
        os.path.splitext(os.path.basename(path))[0]
    result = {"episode": str(eid), "seat": seat,
              "engine": rep.get("module_version"),
              "recorded_banks": common.final_banks(rep),
              "takeover": takeover_profile(rep, seat),
              "sell_shift": sell_shift_profile(rep, seat)}
    out = os.path.join(OUT_DIR, f"{eid}_s{seat}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({"episode": str(eid),
                      "takeover": result["takeover"]["rows"],
                      "top_sell_shifts": result["sell_shift"]["rows"][:5]},
                     indent=1))
    return result


def find_replay(eid: str) -> str:
    for pat in (f"data/sameday/_stage/{eid}/{eid}.json",
                f"data/ourgames/_stage/{eid}/*.json",
                f"data/mine/episode-{eid}-replay.json",
                f"data/trackp/replays/{eid}.json"):
        import glob as _g
        hits = _g.glob(os.path.join(common.ROOT, pat))
        if hits:
            return hits[0]
    raise SystemExit(f"no local replay for episode {eid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="")
    ap.add_argument("--episode", default="")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        import glob as _g
        cands = sorted(_g.glob(os.path.join(
            common.ROOT, "data", "sameday", "_stage", "*", "*.json")))
        pick = next((c for c in cands
                     if os.path.getsize(c) > 3_000_000), None)
        if not pick:
            raise SystemExit("no staged replay for smoke")
        mine_replay(pick, 0)
    elif a.replay:
        mine_replay(a.replay, a.seat)
    elif a.episode:
        mine_replay(find_replay(a.episode), a.seat)
