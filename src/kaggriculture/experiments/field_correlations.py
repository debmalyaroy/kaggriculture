"""Field-wide correlation study: every traced game play, not just ours.

The backfills made this possible for the first time: 12,000+ episodes carry
per-turn 49-field traces for BOTH seats, joined to outcomes and team ratings.
This mines that corpus for what actually correlates with winning -- across
the whole field -- and, separately, for what the ELITE (2900+) do differently
from the mid-field: the direct evidence base for Track P's design.

Outputs, ranked by |correlation| with the game result (win=1/draw=.5/loss=0):
  * engineered per-play features (development timing, labour, economy shape,
    market interaction, product mix) vs outcome and vs final bank;
  * elite-vs-mid contrast: feature means at rating >= ELITE vs the 2300-2700
    band, with a pooled-SD effect size, over WINS ONLY (how winners win,
    tier by tier);
  * day-by-day money-gap decisiveness for the whole field.

    python src/experiments/field_correlations.py
"""
from kaggriculture.paths import ROOT
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

OUT = os.path.join(ROOT, "models", "lab", "field_correlations.json")
ELITE = 2900.0
MID = (2300.0, 2700.0)
TPD = 24                     # turns per day


def day_idx(d):
    return min(719, d * TPD)


def play_features(arr, opp):
    """Engineered features for one seat's play. `arr` is (T, 49); columns per
    turn_features._LAYOUT: 0-2 time, 3-11 price, 12-20 mkt_inv, 21-27 us,
    28-34 them, 35-43 shed, 44-48 seeds."""
    f = {}
    T = len(arr)
    if T < 700:
        return None
    money = arr[:, 21] * 1e5
    o_money = opp[:, 21] * 1e5 if opp is not None else arr[:, 28] * 1e5
    hands = arr[:, 22] * 20
    quads = arr[:, 23] * 4
    crops = arr[:, 24] * 100
    animals = arr[:, 25] * 100
    dry = arr[:, 26] * 100
    prices = arr[:, 3:12]

    # development timing
    f["first_hire_day"] = float(np.argmax(hands > 0) / TPD) if (hands > 0).any() else 30.0
    f["hands_day10"] = float(hands[day_idx(10)])
    f["hands_day20"] = float(hands[day_idx(20)])
    f["max_hands"] = float(hands.max())
    f["quads_day8"] = float(quads[day_idx(8)])
    f["quads_day15"] = float(quads[day_idx(15)])
    f["land_all4_day"] = float(np.argmax(quads >= 4) / TPD) if (quads >= 4).any() else 30.0
    f["animals_day12"] = float(animals[day_idx(12)])
    f["animals_day25"] = float(animals[day_idx(25)])
    f["crops_day6"] = float(crops[day_idx(6)])
    f["crops_day15"] = float(crops[day_idx(15)])
    f["crops_peak"] = float(crops.max())
    f["dry_share"] = float((dry.sum() + 1) / (crops.sum() + 1))

    # economy shape
    f["min_cash_d2_10"] = float(money[day_idx(2):day_idx(10)].min())
    f["cash_day5"] = float(money[day_idx(5)])
    f["cash_day10"] = float(money[day_idx(10)])
    f["cash_day16"] = float(money[day_idx(16)])
    f["cash_day23"] = float(money[day_idx(23)])
    f["first_10k_day"] = float(np.argmax(money >= 10000) / TPD) if (money >= 10000).any() else 30.0
    f["first_30k_day"] = float(np.argmax(money >= 30000) / TPD) if (money >= 30000).any() else 30.0
    f["late_growth"] = float(money[-1] - money[day_idx(20)])

    # market interaction (prices are shared -- how the pair's play shaped them)
    f["price_idx_day10"] = float(prices[day_idx(10)].mean())
    f["price_idx_day20"] = float(prices[day_idx(20)].mean())
    f["price_idx_day28"] = float(prices[day_idx(28)].mean())
    f["price_crash"] = float(prices[day_idx(8)].mean() - prices[day_idx(25)].mean())
    f["min_price_idx"] = float(prices.mean(axis=1).min())

    # opponent gap dynamics
    gap = money - o_money
    f["gap_day10"] = float(gap[day_idx(10)])
    f["gap_day16"] = float(gap[day_idx(16)])
    f["gap_day23"] = float(gap[day_idx(23)])
    lead = gap > 0
    f["lead_share"] = float(lead.mean())
    f["last_lead_change_day"] = float(
        (np.where(np.diff(lead.astype(int)) != 0)[0][-1] / TPD)
        if (np.diff(lead.astype(int)) != 0).any() else 0.0)

    # shed / product mix (columns 35-43: WHEAT..FERTILIZER)
    shed = arr[:, 35:44] * 100
    f["shed_peak"] = float(shed.sum(axis=1).max())
    f["shed_end"] = float(shed[-1].sum())          # unsold at the end = waste
    hi_val = shed[:, 3:5].sum(axis=1) + shed[:, 6:8].sum(axis=1)  # STRAW+MELON+MILK+WOOL
    f["hival_share_day20"] = float((hi_val[day_idx(20)] + 1)
                                   / (shed[day_idx(20)].sum() + 1))
    return f


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    import kaggriculture.data.routes as R
    import kaggriculture.data.turn_features as TF
    from kaggriculture.measure.band_panel import board_ratings

    ratings = board_ratings()
    idx = R.load_index()["routes"]
    files = sorted(os.listdir(TF.TRACE_DIR))
    rows = []
    loaded = 0
    for fn in files:
        ep = os.path.splitext(fn)[0]
        tr = TF.load(ep)
        if not tr:
            continue
        loaded += 1
        for seat in (0, 1):
            arr = tr.get(seat)
            rec = idx.get(f"{ep}_s{seat}")
            if arr is None or rec is None or rec.get("bank") is None:
                continue
            feats = play_features(np.asarray(arr, dtype=np.float64),
                                  np.asarray(tr.get(1 - seat), dtype=np.float64)
                                  if tr.get(1 - seat) is not None else None)
            if feats is None:
                continue
            bank = float(rec["bank"])
            oppb = float(rec["opp_bank"])
            feats["_score"] = 1.0 if bank > oppb else (0.5 if bank == oppb else 0.0)
            feats["_bank"] = bank
            feats["_rating"] = float(ratings.get(rec.get("team")) or 0.0)
            rows.append(feats)
        if loaded % 2000 == 0:
            print(f"  {loaded:,} episodes loaded, {len(rows):,} plays",
                  flush=True)

    print(f"\n{len(rows):,} game plays from {loaded:,} episodes")
    names = [k for k in rows[0] if not k.startswith("_")]
    score = np.array([r["_score"] for r in rows])
    bank = np.array([r["_bank"] for r in rows])
    rating = np.array([r["_rating"] for r in rows])

    table = []
    for nm in names:
        v = np.array([r[nm] for r in rows])
        table.append({"feature": nm, "corr_win": corr(v, score),
                      "corr_bank": corr(v, bank)})
    table.sort(key=lambda t: -abs(t["corr_win"]))
    print(f"\n=== correlation with WINNING (whole field, n={len(rows):,}) ===")
    print(f"{'feature':<22}{'r(win)':>9}{'r(bank)':>9}")
    for t in table:
        print(f"{t['feature']:<22}{t['corr_win']:>9.3f}{t['corr_bank']:>9.3f}")

    # elite contrast: winners only, elite tier vs mid tier
    elite_i = (rating >= ELITE) & (score == 1.0)
    mid_i = (rating >= MID[0]) & (rating <= MID[1]) & (score == 1.0)
    contrast = []
    for nm in names:
        v = np.array([r[nm] for r in rows])
        a, b = v[elite_i], v[mid_i]
        if len(a) < 30 or len(b) < 30:
            continue
        sp = math.sqrt((a.std() ** 2 + b.std() ** 2) / 2) or 1e-9
        contrast.append({"feature": nm, "elite": float(a.mean()),
                         "mid": float(b.mean()),
                         "effect": float((a.mean() - b.mean()) / sp)})
    contrast.sort(key=lambda t: -abs(t["effect"]))
    print(f"\n=== how ELITE wins differ (rating>={ELITE:.0f}, n={elite_i.sum():,} "
          f"vs mid {MID[0]:.0f}-{MID[1]:.0f}, n={mid_i.sum():,}) ===")
    print(f"{'feature':<22}{'elite':>10}{'mid':>10}{'effect':>8}")
    for t in contrast[:18]:
        print(f"{t['feature']:<22}{t['elite']:>10.2f}{t['mid']:>10.2f}"
              f"{t['effect']:>8.2f}")

    json.dump({"n_plays": len(rows), "n_episodes": loaded,
               "correlations": table, "elite_contrast": contrast,
               "elite_n": int(elite_i.sum()), "mid_n": int(mid_i.sum())},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
