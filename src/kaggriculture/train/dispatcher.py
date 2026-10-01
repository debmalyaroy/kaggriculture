"""E4.1 / E4.2 -- world -> economy dispatcher, and market-timing stats.

E4.1  A small CART (sklearn ``DecisionTreeClassifier``) that maps a game's
      WORLD -- the day-6 shop signature "S1|S2" -- to the ECONOMY CLASS the
      winner ran. This is the runtime routing decision: shops are observed by
      day 6, so a dispatcher can pick the economy the field's winners used in
      that world. Reports stratified 5-fold CV accuracy against the majority
      baseline (a tree that beats the base rate means the world genuinely
      predicts the winning economy).

E4.2  Descriptive market-timing: for each product, WHEN winners sell it --
      the unit-weighted day distribution and the peak selling day. A first cut
      for a sell-tranche predictor (house rule: winners restrict supply and
      capture scarcity, so timing per product is the signal).

Reads the corpus written by ``kaggriculture.data.bc_corpus`` (winner seats).

    python -m kaggriculture.train.dispatcher            # both E4.1 and E4.2
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

import kaggriculture.train.macro_actions as MA

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CORPUS = os.path.join(ROOT, "data", "bc_corpus")
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP",
         "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
SHOP_IDX = {s: i for i, s in enumerate(SHOPS)}
ECON_CLASSES = ("LAND_RUSH", "ANIMAL_ECON", "CROP_FOCUS", "BALANCED")


# --------------------------------------------------------------------------- #
# E4.1 : world -> winner economy class
# --------------------------------------------------------------------------- #
def _sig_features(sig):
    """Two categorical shop ids -> two ordinal features (8 = absent/unknown)."""
    parts = (sig or "").split("|")
    a = SHOP_IDX.get(parts[0], 8) if len(parts) >= 1 else 8
    b = SHOP_IDX.get(parts[1], 8) if len(parts) >= 2 else 8
    return [a, b]


# the economy aggregate axes clustered into a "winner economy class"
ECON_AXES = ("animals", "land", "plant", "hire")


def load_episodes(macro_path):
    """Per winning-episode: world_sig + the 4-axis economy aggregate.

    Robust to a --both corpus: keeps, per episode, the seat with the higher
    final_return (the winner); ties keep seat 0. Economy classes are derived
    data-driven (KMeans) from the aggregates rather than hand-thresholded,
    because absolute thresholds collapse to one class (nearly every winner
    buys animals) and teach the tree nothing.
    """
    tbl = pq.read_table(macro_path, columns=(
        ["episode_id", "seat", "world_sig", "final_return",
         "buy_land", "hire_bucket"]
        + [f"buyanimal_{a}" for a in MA.ANIMALS]
        + [f"plant_{s}" for s in MA.SEEDS]))
    d = tbl.to_pydict()
    n = len(d["episode_id"])
    animal_cols = [d[f"buyanimal_{a}"] for a in MA.ANIMALS]
    plant_cols = [d[f"plant_{s}"] for s in MA.SEEDS]
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, "", -1e18])  # an,land,plant,hire,sig,ret
    for i in range(n):
        key = (d["episode_id"][i], d["seat"][i])
        rec = agg[key]
        rec[0] += sum(c[i] for c in animal_cols)
        rec[1] += d["buy_land"][i]
        rec[2] += sum(c[i] for c in plant_cols)
        rec[3] += d["hire_bucket"][i]
        rec[4] = d["world_sig"][i]
        rec[5] = d["final_return"][i]
    best = {}
    for (ep, seat), rec in agg.items():
        if ep not in best or rec[5] > best[ep][5]:
            best[ep] = rec
    X, econ, sigs = [], [], []
    for ep, rec in best.items():
        an, land, plant, hire, sig, _ret = rec
        X.append(_sig_features(sig))
        econ.append([an, land, plant, hire])
        sigs.append(sig)
    return np.array(X), np.array(econ, dtype=float), sigs


def _econ_labels(econ, k=2):
    """KMeans the winner economy aggregates -> a class name per episode,
    each cluster named for its dominant standardised axis."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    Z = StandardScaler().fit_transform(econ)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    # name each cluster by the axis where its centroid is most positive
    names = {}
    for c in range(k):
        centroid = km.cluster_centers_[c]
        axis = ECON_AXES[int(np.argmax(centroid))]
        base = {"animals": "ANIMAL", "land": "LAND",
                "plant": "CROP", "hire": "LABOR"}[axis]
        # disambiguate duplicate names with a rank suffix
        nm = base
        j = 2
        while nm in names.values():
            nm = f"{base}{j}"; j += 1
        names[c] = nm
    return np.array([names[c] for c in km.labels_])


def train_dispatcher(macro_path, verbose=True):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    X, econ, _sigs = load_episodes(macro_path)
    y = _econ_labels(econ)
    classes, counts = np.unique(y, return_counts=True)
    base = counts.max() / counts.sum()
    clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15,
                                 random_state=0)
    # need >= n_splits per class; drop tiny classes' effect via k adjust
    min_class = counts.min()
    k = max(2, min(5, int(min_class)))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    clf.fit(X, y)
    out = {
        "n_episodes": int(len(y)),
        "econ_class_balance": {c: int(n) for c, n in zip(classes, counts)},
        "majority_baseline": round(float(base), 3),
        "cv_folds": k,
        "cv_accuracy_mean": round(float(scores.mean()), 3),
        "cv_accuracy_std": round(float(scores.std()), 3),
        "train_accuracy": round(float(clf.score(X, y)), 3),
        "features": ["shop1_id", "shop2_id"],
    }
    if verbose:
        print("=== E4.1 world -> winner economy dispatcher (CART) ===")
        print(f"episodes: {out['n_episodes']}  classes: "
              f"{out['econ_class_balance']}")
        print(f"majority baseline: {out['majority_baseline']:.3f}")
        print(f"5-fold CV accuracy: {out['cv_accuracy_mean']:.3f} "
              f"+/- {out['cv_accuracy_std']:.3f}  (k={k})")
        lift = out["cv_accuracy_mean"] - out["majority_baseline"]
        print(f"lift over baseline: {lift:+.3f}  "
              f"({'world predicts economy' if lift > 0.01 else 'no signal'})")
    return out


# --------------------------------------------------------------------------- #
# E4.2 : winners' per-product sell-timing (descriptive)
# --------------------------------------------------------------------------- #
def market_timing(micro_path, verbose=True):
    """Unit-weighted day distribution of each product's SELLs by winners."""
    tbl = pq.read_table(micro_path,
                        columns=["seat", "final_return", "day", "action_json"])
    d = tbl.to_pydict()
    n = len(d["day"])
    # winner rows only: final_return >= 1.0 (a win); draws (0.5) excluded
    day_units = {p: defaultdict(float) for p in MA.PRODUCTS}
    tot_units = defaultdict(float)
    for i in range(n):
        if d["final_return"][i] < 1.0:
            continue
        try:
            act = json.loads(d["action_json"][i])
        except (ValueError, TypeError):
            continue
        day = int(d["day"][i])
        for o in (act.get("market") or []):
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" \
                    and o[1] in day_units:
                try:
                    q = max(0, int(o[2]))
                except (TypeError, ValueError):
                    continue
                day_units[o[1]][day] += q
                tot_units[o[1]] += q
    stats = {}
    for p in MA.PRODUCTS:
        dd = day_units[p]
        tot = tot_units[p]
        if tot <= 0:
            stats[p] = {"total_units": 0}
            continue
        days = sorted(dd)
        wmean = sum(day * u for day, u in dd.items()) / tot
        peak = max(dd, key=dd.get)
        # median day (unit-weighted)
        cum = 0.0
        med = days[-1]
        for day in days:
            cum += dd[day]
            if cum >= tot / 2.0:
                med = day
                break
        stats[p] = {"total_units": int(tot), "peak_day": int(peak),
                    "mean_day": round(wmean, 1), "median_day": int(med)}
    if verbose:
        print("\n=== E4.2 winners' sell-timing per product ===")
        print(f"{'product':<11} {'units':>9} {'peak_d':>7} "
              f"{'mean_d':>7} {'med_d':>6}")
        for p in MA.PRODUCTS:
            s = stats[p]
            if s["total_units"] == 0:
                print(f"{p:<11} {'0':>9}")
                continue
            print(f"{p:<11} {s['total_units']:>9,} {s['peak_day']:>7} "
                  f"{s['mean_day']:>7} {s['median_day']:>6}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out", default=os.path.join("models", "dispatcher.json"))
    args = ap.parse_args()
    corpus = args.corpus if os.path.isabs(args.corpus) \
        else os.path.join(ROOT, args.corpus)
    macro_path = os.path.join(corpus, "macro.parquet")
    micro_path = os.path.join(corpus, "micro.parquet")
    if not os.path.exists(macro_path):
        print(f"no corpus at {macro_path} -- run "
              f"`python -m kaggriculture.data.bc_corpus` first")
        return 1
    disp = train_dispatcher(macro_path)
    timing = market_timing(micro_path)
    out_path = args.out if os.path.isabs(args.out) \
        else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"when": "2026-09-18", "dispatcher": disp,
                   "sell_timing": timing}, fh, indent=1)
    print(f"\n-> {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
