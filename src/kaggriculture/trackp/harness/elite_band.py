"""ELITE BAND: play a candidate against >=2500-rated teams' recorded games.

B4 (2026-09-05): after the team_score backfill the index holds 6,874
episodes with BOTH action tracks where one seat belongs to a team rated
>=2500 today (201 teams; Crop Dusta 194 cells, Hanserong 248, ...). This
is the "measure vs 2500+" instrument at scale -- the hard band's 56 cells
cannot rank above its saturation, this can.

Per sampled cell, two phases on the Rust engine:
  CERTIFY  force both recorded tracks; the recorded banks must reproduce
           to the dollar or the cell is dropped (tape not faithful).
  SCORE    the candidate replaces the NON-elite seat and plays the elite
           tape in its own world. Credited win only while the elite tape
           still banks >= 60%% of its recorded bank.

Sampling is stratified by team (max --per-team cells each) so one prolific
team cannot dominate the verdict.

    python src/trackp/harness/elite_band.py <agent.py> --cells 120
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import kaggriculture.engine.serve_match as SM  # noqa: E402
from kaggriculture.trackp import routes_io as R                              # noqa: E402

CELLS = os.path.join(ROOT, ".local", "hardband", "elite_cells.json")
OUT = os.path.join(ROOT, "models", "trackp", "elite_band.json")
COLLAPSE = 0.60


def play_tapes(srv, seed, acts0, acts1):
    js = srv.cmd(f"RESET {int(seed)}")
    n = min(len(acts0), len(acts1))
    for i in range(n):
        if js.get("done") or int(js.get("step") or 0) >= 719:
            break
        la = SM.action_to_line(acts0[i])
        lb = SM.action_to_line(acts1[i])
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
    return [float(f.get("money") or 0) for f in js["farms"]]


def play_agent(srv, agent, opp_acts, seed, seat):
    js = srv.cmd(f"RESET {int(seed)}")
    while not js.get("done") and int(js.get("step") or 0) < 719:
        i = int(js.get("step") or 0)
        mine = SM.action_to_line(agent(SM.obs_for(seat, js)))
        theirs = SM.action_to_line(opp_acts[min(i, len(opp_acts) - 1)])
        la, lb = (mine, theirs) if seat == 0 else (theirs, mine)
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
    b = [float(f.get("money") or 0) for f in js["farms"]]
    return b[seat], b[1 - seat]


def sample(cells, n, per_team, seed=23):
    rng = random.Random(seed)
    by = defaultdict(list)
    for c in cells:
        by[c["team"]].append(c)
    pool = []
    for t, cs in by.items():
        rng.shuffle(cs)
        pool.extend(cs[:per_team])
    rng.shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent")
    ap.add_argument("--cells", type=int, default=120)
    ap.add_argument("--per-team", type=int, default=3)
    a = ap.parse_args()
    cells = json.load(open(CELLS, encoding="utf-8"))
    picks = sample(cells, a.cells, a.per_team)
    print(f"elite band: {len(picks)} sampled cells "
          f"(of {len(cells)}, max {a.per_team}/team)")

    srv = SM.Serve()
    wins = losses = uncert = uncred = 0
    margins = []
    try:
        for c in picks:
            ep, es = str(c["episode"]), int(c["elite_seat"])
            try:
                elite = R.load_route(f"{ep}_s{es}")
                other = R.load_route(f"{ep}_s{1 - es}")
            except Exception:                                  # noqa: BLE001
                uncert += 1
                continue
            a0, a1 = (elite, other) if es == 0 else (other, elite)
            banks = play_tapes(srv, c["seed"], a0, a1)
            rec_e = float(c.get("elite_bank") or 0)
            rec_o = float(c.get("other_bank") or 0)
            if (abs(banks[es] - rec_e) >= 1
                    or abs(banks[1 - es] - rec_o) >= 1):
                uncert += 1
                continue
            agent = SM.load_agent(os.path.abspath(a.agent))
            mine, theirs = play_agent(srv, agent, elite, c["seed"], 1 - es)
            if mine > theirs and theirs >= COLLAPSE * rec_e:
                wins += 1
            elif mine > theirs:
                uncred += 1
            else:
                losses += 1
            margins.append(mine - theirs)
            print(f"  ep{ep} {c['team'][:18]:<18} ({c['team_score']:.0f}) "
                  f"{'WIN ' if mine > theirs else 'loss'} "
                  f"{mine:>9,.0f} vs {theirs:>9,.0f}", flush=True)
    finally:
        srv.close()
    scored = wins + losses + uncred
    margins.sort()
    med = margins[len(margins) // 2] if margins else 0.0
    print(f"\n{os.path.basename(a.agent)} vs ELITE (>=2500): "
          f"CREDITED {wins}/{scored} (+{uncred} uncredited), "
          f"median margin {med:+,.0f}, {uncert} uncertified/skipped")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"agent": os.path.basename(a.agent), "wins": wins,
               "losses": losses, "uncredited": uncred, "scored": scored,
               "median_margin": med, "uncertified": uncert},
              open(OUT, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
