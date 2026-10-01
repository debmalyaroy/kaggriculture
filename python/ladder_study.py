"""Learn from the players who beat us on the real ladder (operator 2026-09-27: "you have the top players' tapes --
simulate and understand what they do in a particular situation").

    python python/ladder_study.py [--tapedump target-v64/release/tapedump] [--threads 12]

1. Target set: every real ladder loss (all our submissions, own team excluded) to a 2600+ team, plus every loss to a
   sub-2300 team by under $1,000; the matching wins are kept as the control set.
2. tapedump replays each game exactly (the ladder banks reproduce) and dumps both seats per day.
3. Where the gap forms: day-by-day money gap, and the final-days sale ledger (days 24-29): units, realized price and
   revenue per product for us vs them -- who sold what, when, at what price.
Writes data/ladder_study/{sets.json, days.jsonl, report.json} and prints the summary.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
from collections import defaultdict

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAD = os.path.join(RL, "data", "ladder")
OUT = os.path.join(RL, "data", "ladder_study")


def load_games():
    rows = []
    for f in glob.glob(os.path.join(LAD, "*", "games.tsv")):
        sub = os.path.basename(os.path.dirname(f))
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["opp_team"] == "Debmalya":
                continue
            r["sub"] = sub
            r["tape"] = os.path.join(LAD, sub, "tapes", f"{r['episode']}_{r['seat']}.json")
            rows.append(r)
    return [r for r in rows if os.path.exists(r["tape"])]


def classify(r):
    R, m = float(r["opp_rating"] or 0), float(r["margin"])
    if R >= 2600:
        return "top_loss" if r["result"] == "L" else "top_win"
    if R < 2300 and abs(m) < 1000:
        return "close_loss" if r["result"] == "L" else "close_win"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapedump", default=os.path.join(RL, "target-v64", "release", "tapedump"))
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "tapes"), exist_ok=True)
    games = load_games()
    sets = defaultdict(list)
    for r in games:
        k = classify(r)
        if k:
            sets[k].append(r)
            link = os.path.join(OUT, "tapes", f"{r['sub']}__{os.path.basename(r['tape'])}")
            if not os.path.exists(link):
                os.symlink(r["tape"], link)
    json.dump({k: [dict(sub=r["sub"], episode=r["episode"], seat=r["seat"], opp=r["opp_team"], rating=r["opp_rating"],
                        margin=r["margin"]) for r in v] for k, v in sets.items()}, open(os.path.join(OUT, "sets.json"), "w"), indent=1)
    print("[study] sets:", {k: len(v) for k, v in sets.items()}, flush=True)
    days_path = os.path.join(OUT, "days.jsonl")
    with open(days_path, "w") as fo:
        subprocess.run([a.tapedump, "--tapes", os.path.join(OUT, "tapes"), "--threads", str(a.threads)], stdout=fo, check=True)
    # key -> seat -> day -> row
    D = defaultdict(lambda: defaultdict(dict))
    for line in open(days_path):
        d = json.loads(line)
        D[d["key"]][d["seat"]][d["day"]] = d
    kind = {}
    for k, v in sets.items():
        for r in v:
            kind[f"{r['sub']}__{r['episode']}_{r['seat']}"] = (k, int(r["seat"]))
    rep = {}
    for k in sets:
        gap_by_day = defaultdict(list)
        led = defaultdict(lambda: defaultdict(float))  # product -> us_units, them_units, us_rev, them_rev
        timing = defaultdict(lambda: defaultdict(float))  # product -> (who, day) -> units
        n = 0
        for key, seats in D.items():
            kk = kind.get(os.path.splitext(os.path.basename(str(key)))[0]) or kind.get(str(key))
            if not kk or kk[0] != k:
                continue
            us, them = kk[1], 1 - kk[1]
            if us not in seats or them not in seats:
                continue
            n += 1
            for day in range(30):
                a_, b_ = seats[us].get(day), seats[them].get(day)
                if a_ and b_:
                    gap_by_day[day].append(a_["money"] - b_["money"])
            for day in range(24, 30):
                for who, s in (("us", us), ("them", them)):
                    row = seats[s].get(day)
                    if not row:
                        continue
                    for p, u in row.get("sells", {}).items():
                        led[p][who + "_units"] += u
                        led[p][who + "_rev"] += row.get("sell_rev", {}).get(p, 0.0)
                        timing[p][f"{who}_d{day}"] += u
        med = {d: sorted(v)[len(v) // 2] for d, v in sorted(gap_by_day.items()) if v}
        rep[k] = {"games": n, "median_money_gap_by_day": med,
                  "final_days_ledger": {p: {kx: round(vx / max(1, n), 1) for kx, vx in v.items()} for p, v in led.items()},
                  "final_days_timing": {p: {kx: round(vx / max(1, n), 1) for kx, vx in sorted(v.items())} for p, v in timing.items()}}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)
    for k, v in rep.items():
        print(f"\n[study] {k}: {v['games']} games")
        print("  median money gap (us - them) by day:", {d: round(g) for d, g in v["median_money_gap_by_day"].items() if d % 3 == 2 or d >= 24})
        for p, L in sorted(v["final_days_ledger"].items(), key=lambda x: -abs(x[1].get("them_rev", 0) - x[1].get("us_rev", 0))):
            print(f"  {p:12s} per game, days 24-29: us {L.get('us_units', 0):6.1f} u ${L.get('us_rev', 0):7.0f} | them {L.get('them_units', 0):6.1f} u ${L.get('them_rev', 0):7.0f}")


if __name__ == "__main__":
    main()
