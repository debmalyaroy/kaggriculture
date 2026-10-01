"""WHEN do top players sell / buy? Interpretable scenario rules from their own turns (bcdump records).

    python python/tpp/scenarios.py --data data/tpp/d1 [--depth 4] [--out data/tpp/scenarios.json]

For each order type (SELL x product, BUY_SEED x crop, BUY_PRODUCT, BUY_ANIMAL, HIRE, BUY_LAND) a shallow decision
tree on plain game quantities -- day, hour, price, our shed stock, market stock, cash, rival cash, plants/animals --
says in which situations they place it. Each leaf is printed as a rule: IF ... THEN they place it on X% of such
turns (N turns). Trees are fitted on 95% of the games and scored on the held-out 5% (game % 20 == 0).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402

ITEMS = "WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER GOOSE COW SHEEP".split()
CROPS = ITEMS[:5]
NAMES = ["HIRE", "BUY_LAND"] + [f"SELL {x}" for x in ITEMS] + [f"BUY_SEED {x}" for x in CROPS] + [f"BUY_PRODUCT {x}" for x in ITEMS] + [f"BUY_ANIMAL {x}" for x in ("GOOSE", "COW", "SHEEP")]


def features(r):
    g = r["glob"].astype(np.float64)
    ex = lambda v, s: np.exp(v * s) - 1  # noqa: E731
    F = {"day": g[:, 2] * 30, "hour": g[:, 1] * 24, "cash": ex(g[:, 3], 10), "rival_cash": ex(g[:, 4], 10),
         "cash_lead": g[:, 7] * 1e4, "hands": g[:, 78] * 16}
    for i, it in enumerate(ITEMS):
        F[f"shed_{it}"] = np.round(ex(g[:, 24 + i], 5))
        F[f"price_{it}"] = g[:, 36 + i] * 200
        F[f"mkt_{it}"] = np.round(ex(g[:, 48 + i], 6))
    for i, c in enumerate(CROPS):
        F[f"seeds_{c}"] = np.round(ex(g[:, 60 + i], 4))
        F[f"plants_{c}"] = np.round(g[:, 65 + i] * 30)
        F[f"rival_plants_{c}"] = np.round(g[:, 81 + i] * 30)
    F["ripe_plants"] = np.round(g[:, 70] * 30)
    F["empty_tiles"] = np.round(g[:, 72] * 100)
    for i, an in enumerate(("GOOSE", "COW", "SHEEP")):
        F[f"animals_{an}"] = np.round(g[:, 74 + i] * 10)
    names = list(F)
    return np.stack([F[k] for k in names], 1), names


def rules(tree, names, min_rate):
    t = tree.tree_
    out = []

    def walk(n, conds):
        if t.children_left[n] == -1:
            v = t.value[n][0]
            rate = v[1] / v.sum() if len(v) > 1 else 0.0
            out.append((rate, int(t.n_node_samples[n]), " AND ".join(conds) or "always"))
            return
        f, th = names[t.feature[n]], t.threshold[n]
        walk(t.children_left[n], conds + [f"{f} <= {th:.1f}"])
        walk(t.children_right[n], conds + [f"{f} > {th:.1f}"])

    walk(0, [])
    return sorted([o for o in out if o[0] >= min_rate], key=lambda x: -x[0] * np.log1p(x[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tpp/d1")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=600000)
    ap.add_argument("--out", default="data/tpp/scenarios.json")
    a = ap.parse_args()
    parts = D.open_parts(a.data)
    per = max(1, a.max_rows // len(parts))
    r = np.concatenate([np.asarray(p[np.linspace(0, len(p) - 1, min(per, len(p))).astype(int)]) for p in parts])
    X, names = features(r)
    held = r["game"] % 20 == 0
    ids = r["mlab"][:, :, 0]
    rep = {}
    for k, nm in enumerate(NAMES):
        y = (ids == k).any(1)
        if y.sum() < 300:
            continue
        tr = DecisionTreeClassifier(max_depth=a.depth, min_samples_leaf=200, class_weight=None).fit(X[~held], y[~held])
        base = y.mean()
        pred = tr.predict_proba(X[held])[:, 1]
        # held-out lift: how much better the rule's rate separates than the base rate
        top = pred >= np.quantile(pred, 0.9)
        rs = rules(tr, names, max(0.15, 3 * base))
        rep[nm] = {"base_rate": float(base), "held_top10pct_rate": float(y[held][top].mean()), "rules": [{"rate": float(x[0]), "turns": x[1], "when": x[2]} for x in rs[:5]]}
        print(f"\n== {nm}: placed on {base:.1%} of turns; held-out: the rule's top 10% of turns -> {y[held][top].mean():.0%}")
        for rate, n, cond in rs[:4]:
            print(f"   {rate:5.0%} of {n:6d} turns | IF {cond}")
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
