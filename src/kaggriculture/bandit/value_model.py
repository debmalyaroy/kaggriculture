"""Value model for search: P(win) from a player's situation at the start of a day.

Cash lead alone is no value before day 15 (AUC 0.51 day 3 .. 0.64 day 15 on top-40 games): early money
becomes animals, crops, land. This model reads the whole situation from `tapedump` player-days (both
seats of every game) and is validated on HELD-OUT TEAMS (GroupKFold by the team of the tape), per day.

    python -m kaggriculture.bandit.value_model [--days data/field/week50/days.jsonl ...] [--folds 5]

Gate B0 (docs/history/search-plan-2026-09-26.md): AUC >= 0.75 at day 10 on held-out teams.
Writes data/field/value/model.joblib and report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from kaggriculture.paths import ROOT

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
PRODS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def feats(d):
    g = lambda m, k: float(m.get(k, 0))
    px = d.get("prices", {})
    shed_val = sum(g(d["shed"], p) * g(px, p) for p in PRODS)
    f = [d["day"], d["money"] / 1000, d["rival_money"] / 1000, (d["money"] - d["rival_money"]) / 1000, shed_val / 1000,
         d["crew"], d["quads"], d["land"], d["weeds"]]
    f += [g(d["plants"], c) for c in CROPS] + [g(d["ripe"], c) for c in CROPS]
    f += [g(d["animals"], a) for a in ANIMALS] + [g(d["structs"], s) for s in ("COOP", "PASTURE")]
    f += [g(d["seeds"], c) for c in CROPS] + [g(d["shed"], a) for a in ANIMALS]
    f += [sum(d["r_plants"].values()), sum(d["r_animals"].values())]
    f += [g(d["r_plants"], c) for c in CROPS] + [g(d["r_animals"], a) for a in ANIMALS]
    shops = d["shops"].split("|") if d["shops"] else []
    f += [float(shops.count(s)) for s in SHOPS]
    f += [g(px, p) for p in PRODS]
    # value of what is growing / grazing at today's prices, and our-minus-rival board differences
    prod = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
    grow = sum(g(d["plants"], c) * g(px, c) for c in CROPS) / 1000
    ripe = sum(g(d["ripe"], c) * g(px, c) for c in CROPS) / 1000
    herd = sum(g(d["animals"], a) * g(px, prod[a]) for a in ANIMALS) / 1000
    rgrow = sum(g(d["r_plants"], c) * g(px, c) for c in CROPS) / 1000
    rherd = sum(g(d["r_animals"], a) * g(px, prod[a]) for a in ANIMALS) / 1000
    f += [grow, ripe, herd, rgrow, rherd, grow - rgrow, herd - rherd,
          (d["money"] - d["rival_money"]) / 1000 + (grow - rgrow) + (herd - rherd) * 3]
    return f


def load(paths):
    X, y, grp, day = [], [], [], []
    for path in paths:
        base = os.path.dirname(path)
        team = {}
        meta = os.path.join(base, "meta.csv")
        if os.path.exists(meta):
            for r in csv.DictReader(open(meta, encoding="utf-8")):
                team[r["key"]] = r.get("top_team") or r.get("team") or ""
        for l in open(path, encoding="utf-8"):
            d = json.loads(l)
            if d["bank_end"] == d["rival_end"]:
                continue
            k = d["key"]
            top_seat = int(k.split("_")[1])
            # group = the team behind this seat (the tape's top player, or "rival of <team>" for the other side)
            t = team.get(k, "?")
            grp.append(t if d["seat"] == top_seat else "rival:" + k.split("_")[0])
            X.append(feats(d))
            y.append(1 if d["bank_end"] > d["rival_end"] else 0)
            day.append(d["day"])
    return np.array(X, dtype=np.float32), np.array(y), np.array(grp), np.array(day)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=[os.path.join(ROOT, "data", "field", "week50", "days.jsonl")])
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X, y, grp, day = load(a.days)
    # rival sides are one-off groups; tie each to its game so a game never straddles folds
    print(f"{len(y)} player-days, {len(set(grp))} groups", flush=True)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(a.folds).split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    rep = {}
    lead = X[:, 3]
    for d in (3, 6, 10, 15, 20, 25, 29):
        s = day == d
        if s.sum() > 50 and len(set(y[s])) == 2:
            rep[d] = {"auc_model": roc_auc_score(y[s], pred[s]), "auc_cash_lead": roc_auc_score(y[s], lead[s]), "n": int(s.sum())}
            print(f"day {d:2}: model AUC {rep[d]['auc_model']:.3f}  (cash lead {rep[d]['auc_cash_lead']:.3f}, n {rep[d]['n']})", flush=True)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0).fit(X, y)
    out = os.path.join(ROOT, "data", "field", "value")
    os.makedirs(out, exist_ok=True)
    import joblib
    joblib.dump(m, os.path.join(out, "model.joblib"))
    gate = rep.get(10, {}).get("auc_model", 0) >= 0.75
    json.dump({"by_day": rep, "gate_B0_day10_auc_ge_0.75": gate}, open(os.path.join(out, "report.json"), "w"), indent=1)
    print("gate B0 (day 10 AUC >= 0.75 on held-out teams):", "PASS" if gate else "FAIL", flush=True)


if __name__ == "__main__":
    main()
