"""Top-50 play analysis (queue Q39, follows Q38 features). Writes data/top50/fetch/report/report.json.

    python python/top50/report.py [--root data/top50/fetch]

Sections (all from data/top50/fetch/feat/{games,sells}.parquet):
  teams      per team: games, win rate by opponent band, margin, worlds seen, opening variety, plan families,
             revenue mix by item, sales timing, copy games and results vs copies
  bands      by the team's rating at game time and by the opponent's band
  worlds     per world (first two shops): games, win rate, revenue mix, plan families; how much the plan
             depends on the world (conditional entropy); realized vs idle world (did play move the world?)
  determinism  same team + same world + same opening: identical economy? first day the plans diverge
  shells     per team: sell / hold model from what the player could see (depth-4 tree rules + depth-8 accuracy
             on held-out games vs the majority baseline), feature importance, cross-team transfer
  plans      plan families (k-means on the day-12 / day-24 farm and the buys by phase), centroids, by world
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RBINS = [0, 2100, 2300, 2500, 2700, 2900, 99999]
RBANDS = ["lt2100", "2100-2300", "2300-2500", "2500-2700", "2700-2900", "2900plus"]
ITEMS = ["wheat", "carrot", "tomato", "strawberry", "melon", "egg", "milk", "wool", "fertilizer"]
SHELL_X = ["day", "hour", "hours_left", "stock", "mkt_inv", "px", "px_rel", "dpx", "dinv", "money_gap", "shop_item"]


def entropy(xs):
    c = Counter(xs)
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def cond_entropy(xs, ys):
    """H(X | Y) in bits."""
    by = {}
    for x, y in zip(xs, ys):
        by.setdefault(y, []).append(x)
    n = len(xs)
    return sum(len(v) / n * entropy(v) for v in by.values()) if n else 0.0


def rate(s):
    return round(float(s.mean()), 3) if len(s) else None


def first_diff(a, b):
    for d, (x, y) in enumerate(zip(a.split(","), b.split(","))):
        if x != y:
            return d
    return 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RL, "data", "top50", "fetch"))
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    g = pd.read_parquet(os.path.join(root, "feat", "games.parquet"))
    s = pd.read_parquet(os.path.join(root, "feat", "sells.parquet"))
    g["r"] = pd.to_numeric(g.get("rating", g.get("r")), errors="coerce")
    g["opp_r"] = pd.to_numeric(g.get("opp_rating", g.get("opp_r")), errors="coerce")
    g["rband"] = pd.cut(g.r, RBINS, right=False, labels=RBANDS).astype(str)
    g["oband"] = pd.cut(g.opp_r, RBINS, right=False, labels=RBANDS).astype(str)
    g["is_copy"] = (g.copy_d0 >= 0.9).astype(int)
    rev_cols = [f"rev_{i}" for i in ITEMS]
    g["rev_sum"] = g[rev_cols].sum(1).replace(0, np.nan)
    rep = {"n_games": int(len(g)), "n_teams": int(g.team.nunique()), "exact_share": rate(g.exact),
           "moved_world1_share": rate(g.moved1), "moved_world2_share": rate(g.moved2)}

    # ---- plan families: k-means on the economy vector
    econ = [c for c in g.columns if any(c.startswith(p) for p in ("wheat_tiles_d", "carrot_tiles_d", "tomato_tiles_d", "strawberry_tiles_d",
                                                                   "melon_tiles_d", "goose_d", "cow_d", "sheep_d", "land_d", "hands_d"))
            and (c.endswith("_d12") or c.endswith("_d24"))]
    econ += [c for c in g.columns if c.startswith("seed_") or c.startswith("buy_")]
    X = g[econ].fillna(0).to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=12, n_init=5, random_state=0).fit(Z)
    g["plan"] = km.labels_
    cent = pd.DataFrame(km.cluster_centers_ * sd + mu, columns=econ)
    rep["plans"] = {"features": econ, "families": [
        {"id": int(k), "games": int((g.plan == k).sum()), "win": rate(g[g.plan == k].win),
         "top_features": {c: round(float(cent.loc[k, c]), 1) for c in cent.columns[np.argsort(-np.abs(km.cluster_centers_[k]))[:8]]},
         "revenue_mix": {i: round(float((g[g.plan == k][f"rev_{i}"] / g[g.plan == k].rev_sum).mean()), 3) for i in ITEMS}}
        for k in range(12)]}

    # ---- teams
    teams = []
    for tm, x in g.groupby("team"):
        by_band = {b: {"games": int(len(y)), "win": rate(y.win)} for b, y in x.groupby("oband")}
        opens = x.open_d0_2.value_counts()
        mix = {i: round(float((x[f"rev_{i}"] / x.rev_sum).mean()), 3) for i in ITEMS}
        # determinism: same world + same opening -> identical economy through day 23?
        det, div = [], []
        for _, y in x.groupby(["world2", "open_d0_5"]):
            if len(y) >= 2:
                det.append(float(y.plan_d0_23.value_counts().iloc[0] / len(y)))
                rows = y.dh_unit.tolist()
                div += [first_diff(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, min(len(rows), i + 6))]
        teams.append({
            "team": tm, "games": int(len(x)), "rating_max": round(float(x.r.max()), 1) if x.r.notna().any() else None,
            "win": rate(x.win), "margin_mean": round(float(x.margin.mean()), 0), "by_opp_band": by_band,
            "worlds": int(x.world2.nunique()), "openings_d0_2": int(len(opens)), "top_opening_share": round(float(opens.iloc[0] / len(x)), 3),
            "plans": {int(k): int(v) for k, v in x.plan.value_counts().items()},
            "H_plan": round(entropy(x.plan), 2), "H_plan_given_world": round(cond_entropy(x.plan.tolist(), x.world2.tolist()), 2),
            "same_world_same_open_identical_d0_23": round(float(np.mean(det)), 3) if det else None,
            "diverge_day_median": float(np.median(div)) if div else None, "diverge_pairs": len(div),
            "revenue_mix": mix, "sell_hour_mean": round(float(x.sell_hour_mean.mean()), 1),
            "sell_share_last_day": rate(x.sell_share_last_day), "sell_share_d24_29": rate(x.sell_share_d24_29),
            "units_per_sell_turn": round(float(x.units_per_sell_turn.mean()), 1),
            "unsold_value_units": round(float(x[[f"unsold_{i}" for i in ITEMS if i != "fertilizer"]].sum(1).mean()), 1),
            "copy_games": int(x.is_copy.sum()), "win_vs_copy": rate(x[x.is_copy == 1].win), "win_vs_noncopy": rate(x[x.is_copy == 0].win),
            "diverge_from_rival_step_median": float(x.diverge_step.median()),
            "land_d12": round(float(x.get("land_d12", pd.Series([0])).mean()), 2), "hands_d12": round(float(x.get("hands_d12", pd.Series([0])).mean()), 2)})
    rep["teams"] = sorted(teams, key=lambda t: -(t["rating_max"] or 0))

    # ---- bands
    rep["bands"] = {
        "by_team_rating": {b: {"games": int(len(y)), "win": rate(y.win), "margin": round(float(y.margin.mean()), 0),
                               "revenue_mix": {i: round(float((y[f"rev_{i}"] / y.rev_sum).mean()), 3) for i in ITEMS},
                               "sell_share_d24_29": rate(y.sell_share_d24_29)} for b, y in g.groupby("rband")},
        "by_opp_band": {b: {"games": int(len(y)), "win": rate(y.win), "margin": round(float(y.margin.mean()), 0),
                            "copy_share": rate(y.is_copy), "plans": {int(k): int(v) for k, v in y.plan.value_counts().head(4).items()}}
                        for b, y in g.groupby("oband")}}

    # ---- worlds
    worlds = []
    for w, y in g.groupby("world2"):
        worlds.append({"world": w, "games": int(len(y)), "teams": int(y.team.nunique()), "win": rate(y.win),
                       "revenue_mix": {i: round(float((y[f"rev_{i}"] / y.rev_sum).mean()), 3) for i in ITEMS},
                       "plans": {int(k): int(v) for k, v in y.plan.value_counts().head(3).items()},
                       "moved_from_idle": rate(y.moved2)})
    rep["worlds"] = sorted(worlds, key=lambda x: -x["games"])
    rep["world_dependence"] = {"H_plan": round(entropy(g.plan), 3), "H_plan_given_world2": round(cond_entropy(g.plan.tolist(), g.world2.tolist()), 3),
                               "H_plan_given_team": round(cond_entropy(g.plan.tolist(), g.team.tolist()), 3),
                               "H_plan_given_team_world2": round(cond_entropy(g.plan.tolist(), (g.team + "|" + g.world2).tolist()), 3)}
    # did top players land in their better worlds more often than chance? win rate when the world moved vs not
    rep["world_steering"] = {"win_moved2": rate(g[g.moved2 == 1].win), "win_not_moved2": rate(g[g.moved2 == 0].win),
                             "n_moved2": int(g.moved2.sum()), "n_not_moved2": int((g.moved2 == 0).sum())}

    # ---- shells (sales decompile)
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier, export_text
    s = s.dropna(subset=["team"])
    s["gid"] = s["id"]
    shells, models = [], {}
    for tm, x in s.groupby("team"):
        if x.gid.nunique() < 12 or x.sold.sum() < 200:
            continue
        ids = sorted(x.gid.unique())
        test = set(ids[::4])
        tr, te = x[~x.gid.isin(test)], x[x.gid.isin(test)]
        row = {"team": tm, "rows": int(len(x)), "sell_rate": round(float(x.sold.mean()), 3), "per_item": {}}
        for depth in (4, 8):
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=40, class_weight="balanced", random_state=0).fit(tr[SHELL_X], tr.sold)
            row[f"bal_acc_d{depth}"] = round(float(balanced_accuracy_score(te.sold, m.predict(te[SHELL_X]))), 3)
            if depth == 8:
                models[tm] = m
                row["importance"] = {c: round(float(v), 3) for c, v in sorted(zip(SHELL_X, m.feature_importances_), key=lambda z: -z[1])[:6]}
            else:
                row["rules_d4"] = export_text(m, feature_names=SHELL_X, max_depth=4)[:3000]
        for it, y in x.groupby("item"):
            if y.sold.sum() >= 50 and y.gid.nunique() >= 8:
                ytr, yte = y[~y.gid.isin(test)], y[y.gid.isin(test)]
                if len(yte) and yte.sold.nunique() > 1:
                    m = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=0).fit(ytr[SHELL_X], ytr.sold)
                    row["per_item"][it] = round(float(balanced_accuracy_score(yte.sold, m.predict(yte[SHELL_X]))), 3)
        shells.append(row)
    rep["shells"] = sorted(shells, key=lambda r: -r["bal_acc_d8"])
    top_teams = [r["team"] for r in rep["teams"] if r["team"] in models][:12]
    rep["shell_transfer"] = {"teams": top_teams, "matrix": [[
        round(float(balanced_accuracy_score(s[s.team == b].sold, models[a2].predict(s[s.team == b][SHELL_X]))), 3) for b in top_teams] for a2 in top_teams]}
    out = os.path.join(root, "report")
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    g[["id", "team", "world2", "plan", "win", "margin", "is_copy", "rband", "oband"]].to_csv(os.path.join(out, "games_plans.tsv"), sep="\t", index=False)
    print(f"[report] {len(g)} games, {g.team.nunique()} teams, {len(shells)} shells -> {out}/report.json")
    print(json.dumps({k: rep[k] for k in ("exact_share", "moved_world2_share", "world_dependence", "world_steering")}, indent=1))


if __name__ == "__main__":
    main()
