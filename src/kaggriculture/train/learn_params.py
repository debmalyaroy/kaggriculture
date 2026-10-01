"""Learn what actually predicts winning, from a dataset of top-ladder episodes.

Input: `data/episodes.csv` from src/kaggriculture/data/build_dataset.py -- one row per player per
episode, with what they built and whether they won.

Two views, deliberately:

1. **Paired winner-minus-loser deltas.** Both players in an episode faced the
   same seed, the same town, the same market. Differencing within an episode
   removes all of that, so the median delta is a clean estimate of "winners did
   more of X". This is the robust view and the one to trust.

2. **Logistic regression** (pure numpy, no sklearn) on standardised features,
   predicting the win. Handles correlated features that the paired view reads
   one at a time -- e.g. it can tell "more hands" from "more hands *because*
   more land".

Neither is causal. A feature can predict winning because it *causes* winning or
because good agents happen to do it. The output is a prior for
`src/kaggriculture/train/tune.py`, not a conclusion.

    python -m kaggriculture.train.learn_params
    python -m kaggriculture.train.learn_params --top 25 --min-episodes 30
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import math
import os
import sys


SKIP = {"episode_id", "date", "player", "won", "final_bank", "opp_bank", "margin"}
# Features whose sign we can act on directly, mapped to the PARAMS they inform.
ACTIONABLE = {
    "peak_COW": "target_cow", "peak_SHEEP": "target_sheep", "peak_GOOSE": "target_goose",
    "peak_MELON": "target_melon", "peak_STRAWBERRY": "target_strawberry",
    "peak_TOMATO": "target_tomato", "peak_WHEAT": "target_wheat",
    "peak_CARROT": "target_carrot", "peak_herd": "max_herd",
    "peak_hands": "hands_max", "move_frac": "travel_weight (inverse)",
    "quadrants": "land_min_day (earlier)", "first_animal_day": "animal_cash_buffer (inverse)",
}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("date",):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = 0.0
    return rows


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def paired_deltas(rows, features):
    """Median (winner - loser) per feature, plus how often the winner was higher."""
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)
    out = {f: [] for f in features}
    n_ep = 0
    for _eid, pair in by_ep.items():
        if len(pair) != 2:
            continue
        w = pair[0] if pair[0]["won"] else pair[1]
        l = pair[1] if pair[0]["won"] else pair[0]
        if w["won"] == l["won"]:
            continue                       # tie or malformed
        n_ep += 1
        for f in features:
            out[f].append(w[f] - l[f])
    stats = {}
    for f, ds in out.items():
        if not ds:
            continue
        higher = sum(1 for d in ds if d > 0)
        lower = sum(1 for d in ds if d < 0)
        decided = higher + lower
        stats[f] = {
            "median_delta": median(ds),
            "winner_higher_pct": 100.0 * higher / decided if decided else 50.0,
            "n": len(ds),
        }
    return stats, n_ep


def logistic(rows, features, iters=4000, lr=0.25, l2=1e-3):
    import numpy as np
    X = np.array([[r[f] for f in features] for r in rows], dtype=float)
    y = np.array([r["won"] for r in rows], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        g = Xs.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    p = 1.0 / (1.0 + np.exp(-Xs @ w))
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    return dict(zip(features, w[:-1])), acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "episodes.csv"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-episodes", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"no dataset at {args.data}\n"
                 f"  build one first:  python -m kaggriculture.data.build_dataset --days 3 --per-day 50\n"
                 f"  (needs the kaggle CLI on a machine with network)")
    rows = load(args.data)
    features = [k for k in rows[0] if k not in SKIP and k != "date"]
    print(f"{len(rows)} player-rows, {len(rows)//2} episodes, {len(features)} features\n")

    stats, n_ep = paired_deltas(rows, features)
    if n_ep < args.min_episodes:
        print(f"WARNING: only {n_ep} decided episodes -- results are noise below "
              f"~{args.min_episodes}. Fetch more with build_dataset.py.\n")

    print("=" * 74)
    print("PAIRED VIEW  (winner minus loser, same seed / town / market)")
    print("=" * 74)
    print(f"  {'feature':<26}{'median Δ':>12}{'winner higher':>16}{'n':>6}")
    ranked = sorted(stats.items(),
                    key=lambda kv: -abs(kv[1]["winner_higher_pct"] - 50.0))
    for f, s in ranked[:args.top]:
        flag = "  <-- actionable" if f in ACTIONABLE else ""
        print(f"  {f:<26}{s['median_delta']:>12.2f}{s['winner_higher_pct']:>15.0f}%"
              f"{s['n']:>6}{flag}")

    try:
        coefs, acc = logistic(rows, features)
    except ImportError:
        print("\n(numpy missing -- skipping the logistic model)")
        return
    print("\n" + "=" * 74)
    print(f"LOGISTIC MODEL  (standardised; in-sample accuracy {acc*100:.1f}%)")
    print("=" * 74)
    print(f"  {'feature':<26}{'coefficient':>14}   direction")
    for f, c in sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:args.top]:
        arrow = "more is better" if c > 0 else "less is better"
        print(f"  {f:<26}{c:>14.3f}   {arrow}")

    print("\n" + "=" * 74)
    print("SUGGESTED PRIORS for src/kaggriculture/train/tune.py")
    print("=" * 74)
    any_sug = False
    for feat, knob in ACTIONABLE.items():
        s = stats.get(feat)
        if not s or abs(s["winner_higher_pct"] - 50.0) < 12:
            continue
        any_sug = True
        direction = "raise" if s["median_delta"] > 0 else "lower"
        print(f"  {knob:<32} {direction:>5}  "
              f"(winners differ by {s['median_delta']:+.1f}, "
              f"{s['winner_higher_pct']:.0f}% of episodes)")
    if not any_sug:
        print("  nothing clears the noise floor yet -- gather more episodes.")
    print("\nThese are priors, not conclusions. Feed them to tune.py and let")
    print("self-play decide; several 'obvious' signals have measured negative.")


if __name__ == "__main__":
    main()
