"""Study the top-N ladder players game by game: per team and per submission, what they choose in which
situation, and what goes with winning.

Input: `data/field/top40/days.jsonl` from `tapedump` (exact replays of GM's replays: state at the start of
each day + the choices made during it, both seats) and GM's episodes.csv / teams.csv.

    python -m kaggriculture.bandit.top_study [--top 40]

Writes data/field/top40/games.parquet (one row per top-player game side) and study.json, prints the report.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import pandas as pd

from kaggriculture.paths import ROOT

GM = r"D:\gm_dataset"
BASE = os.path.join(ROOT, "data", "field", "top40")
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def teams(top, lb=None):
    if lb:
        s = open(lb, encoding="utf-8").read()
        rows = json.loads(s[s.index("["):])[:top]
        return {str(r["teamId"]): (i, r["teamName"], float(r["score"])) for i, r in enumerate(rows)}
    rows = sorted(csv.DictReader(open(os.path.join(GM, "teams.csv"), encoding="utf-8")), key=lambda r: -float(r["ladder_score"] or 0))
    return {r["team_id"]: (i, r["team_name"], float(r["ladder_score"])) for i, r in enumerate(rows[:top])}


def games(top, lb=None):
    tm = teams(top, lb)
    ep = {r["episode_id"]: r for r in csv.DictReader(open(os.path.join(GM, "episodes.csv"), encoding="utf-8"))}
    by = collections.defaultdict(dict)
    for l in open(os.path.join(BASE, "days.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        eid, ts = d["key"].split("_")
        if d["seat"] != int(ts):
            continue  # the top player's side only (a top-vs-top game has a tape per side)
        by[(eid, d["seat"])][d["day"]] = d
    rows = []
    for (eid, s), days in by.items():
        e = ep.get(eid)
        if not e or e[f"team_{s}"] not in tm:
            continue
        rank, name, score = tm[e[f"team_{s}"]]
        o = 1 - s
        D = [days.get(i, {}) for i in range(31)]
        last = max(days)
        end = days[last]
        shops = next((x["shops"] for x in D if x and x["shops"].count("|") >= 1), "")
        w = "|".join(shops.split("|")[:2])
        g = {"eid": eid, "seat": s, "team": name, "rank": rank, "score": score, "sub": e[f"sub_{s}"],
             "opp_team": e[f"team_{o}"], "opp_top": e[f"team_{o}"] in tm, "rating": float(e[f"rating_{s}"] or 0),
             "opp_rating": float(e[f"rating_{o}"] or 0), "date": e["end_time"][:10], "world": w, "shop1": w.split("|")[0],
             "bank": end["bank_end"], "rival_bank": end["rival_end"],
             "win": 1.0 if end["bank_end"] > end["rival_end"] else 0.0 if end["bank_end"] < end["rival_end"] else 0.5,
             "copy_d0": D[0].get("copy", 0), "copy_d1_5": sum(D[i].get("copy", 0) for i in range(1, 6)) / 5}
        g["group"] = "COPY" if g["copy_d0"] >= 0.9 else "DIFF" if g["copy_d0"] <= 0.1 else "PARTIAL"
        for dd in (3, 6, 10, 15, 20, 25, 29):
            x = D[dd] if D[dd] else end
            g[f"money_d{dd}"] = x.get("money", 0)
            g[f"lead_d{dd}"] = x.get("money", 0) - x.get("rival_money", 0)
            g[f"animals_d{dd}"] = sum(x.get("animals", {}).values())
            g[f"plants_d{dd}"] = sum(x.get("plants", {}).values())
            g[f"quads_d{dd}"] = x.get("quads", 1)
        tot = collections.Counter()
        sold = collections.Counter()
        rev = collections.Counter()
        late = collections.Counter()
        hires = []
        first = {}
        for i in range(31):
            x = D[i]
            if not x:
                continue
            b = x.get("buys", {})
            hires.append(b.get("HIRE", 0))
            for k, v in b.items():
                tot[k] += v
                first.setdefault(k, i)
            for k, v in x.get("sells", {}).items():
                sold[k] += v
                # exact revenue when the dump has it (counterfactual accounting), else units x price
                rev[k] += x["sell_rev"].get(k, 0) if "sell_rev" in x else v * x.get("sell_px", {}).get(k, 0)
                if i >= 28:
                    late[k] += v
            for k, v in x.get("ops", {}).items():
                if k.startswith("PLANT "):
                    tot["PLANTED " + k[6:]] += v
        for c in CROPS:
            g[f"planted_{c}"] = tot.get("PLANTED " + c, 0)
        for a in ANIMALS:
            g[f"bought_{a}"] = tot.get("BUY_ANIMAL " + a, 0)
            g[f"first_{a}"] = first.get("BUY_ANIMAL " + a, 99)
        g["first_land"] = first.get("BUY_LAND", 99)
        g["lands"] = tot.get("BUY_LAND", 0)
        g["hires_total"] = sum(hires)
        g["hires_peak"] = max(hires or [0])
        g["hires_d0_9"] = sum(hires[:10])
        g["hires_d10_19"] = sum(hires[10:20])
        g["hires_d20_29"] = sum(hires[20:30])
        allsold = sum(sold.values())
        for p in PRODS:
            g[f"sold_{p}"] = sold.get(p, 0)
            g[f"rev_{p}"] = rev.get(p, 0)
        g["revenue"] = sum(rev.values())
        sp = collections.Counter()
        for i in range(31):
            for k, v in (D[i].get("spend", {}) if D[i] else {}).items():
                sp[k] += v
        g["spend"] = sum(sp.values())
        for k in ("HIRE", "BUY_LAND", "BUY_PRODUCT WHEAT", "BUY_ANIMAL COW", "BUY_ANIMAL SHEEP", "BUY_ANIMAL GOOSE"):
            g["spend_" + k.split()[-1].lower() if " " in k else "spend_" + k.lower()] = sp.get(k, 0)
        g["spend_seeds"] = sum(v for k, v in sp.items() if k.startswith("BUY_SEED"))
        g["late_share"] = sum(late.values()) / allsold if allsold else 0
        rows.append(g)
    return pd.DataFrame(rows)


def report(df):
    out = {}
    print(f"{len(df)} top-player game sides, {df.team.nunique()} teams, {df['sub'].nunique()} submissions")
    print(f"overall win {df.win.mean():.3f}; median bank {df.bank.median():.0f}; group {dict(df.group.value_counts())}")
    t = df.groupby(["rank", "team"]).agg(n=("win", "size"), win=("win", "mean"), bank=("bank", "median"), subs=("sub", "nunique"),
                                         copy=("group", lambda s: (s == "COPY").mean()), animals10=("animals_d10", "median"),
                                         plants10=("plants_d10", "median"), hires=("hires_total", "median"), lands=("lands", "median"),
                                         late=("late_share", "median")).reset_index()
    print("\nper team (rank = ladder rank):")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out["teams"] = t.to_dict("records")
    # what goes with winning, within team (removes team strength): standardised difference winners - losers
    feats = [c for c in df.columns if c.split("_")[0] in ("money", "lead", "animals", "plants", "quads", "planted", "bought", "first",
                                                          "hires", "sold", "rev") or c in ("revenue", "late_share", "lands", "first_land")]
    dec = df[df.win != 0.5]
    rows = []
    for f in feats:
        z = []
        for _, g in dec.groupby("team"):
            if g.win.nunique() < 2 or g[f].std() == 0:
                continue
            z.append((g[g.win == 1][f].mean() - g[g.win == 0][f].mean()) / g[f].std())
        if z:
            rows.append((f, sum(z) / len(z), len(z)))
    rows.sort(key=lambda r: -abs(r[1]))
    print("\nwithin-team winner-vs-loser gap (std units, averaged over teams), top 25:")
    for f, v, n in rows[:25]:
        print(f"  {f:22} {v:+.2f}  ({n} teams)")
    out["win_gaps"] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--base", default=None, help="data dir with days.jsonl (default data/field/top40)")
    ap.add_argument("--leaderboard", default=None, help="leaderboard JSON for the current top N")
    a = ap.parse_args()
    global BASE
    if a.base:
        BASE = a.base
    df = games(a.top, a.leaderboard)
    df.to_parquet(os.path.join(BASE, "games.parquet"))
    rep = report(df)
    json.dump(rep, open(os.path.join(BASE, "study.json"), "w", encoding="utf-8"), indent=1, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
