"""Rating-band referee panel: model, sampler, and the adoption backtest.

CROWN-2 Phase A. The current panel is built from OUR LOSSES -- it answers
"does this candidate fix what beat us?", a repair signal. But the ladder
pays wins against whatever MATCHMAKING serves, and matchmaking pairs by
rating: a panel whose family mix matches the rating band we actually play in
should predict realized ladder score better than a loss-biased one.

Adoption is gated by a BACKTEST on our own history, no new games needed:
for every submission with >= 30 named-opponent games, split its games
chronologically; from the first half build each panel weighting (loss-family
vs band-family); predict the second half's expected score as the weighted
mean of first-half per-family win rates; compare |predicted - realized|.
Adopt the band panel only if it wins on MAE and rank correlation.

    python src/experiments/band_panel.py            # A1 model + A3 backtest
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

OUT = os.path.join(ROOT, "models", "lab", "band_panel.json")
MIN_GAMES = 30
BAND_QUANTILES = (0.10, 0.90)       # the matchmaking band, per own rating


def board_ratings():
    import kaggriculture.data.episodes as E
    rows = E.leaderboard_rows(verbose=False) or []
    return {r["team"]: float(r["score"]) for r in rows
            if r.get("team") and r.get("score")}


def own_sub_ratings():
    """{submission_ref: public score} from the live submissions table."""
    p = subprocess.run(["kaggle", "competitions", "submissions",
                        "kaggriculture"], capture_output=True, text=True,
                       timeout=300)
    out = p.stdout or ""
    scores = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d{6,})\s", line)
        if not m:
            continue
        nums = re.findall(r"(\d{3,4}\.\d)\s*$", line)
        if nums:
            scores[m.group(1)] = float(nums[-1])
    return scores


def family_labeler():
    """The same opponent-family labeling stage_tapes uses (opp_sold distance
    to the counter-agent library finals) -- shared so panel families and tape
    families can never drift apart."""
    import kaggriculture.agentbuild.counter_agent as CA
    lib, teams = CA.build_library(exclude_id="")
    finals = []
    for sched in lib:
        cum = {i: 0 for i in CA.TRACKED}
        for sells in sched.values():
            for item, q in sells:
                cum[item] += q
        finals.append(cum)

    def label(g):
        opp = {i: int((g.get("opp_sold") or {}).get(i, 0) or 0)
               for i in CA.TRACKED}
        total = sum(opp.values())
        if total < 50:
            return "unknown"
        ds = sorted((sum(abs(opp[i] - f[i]) for i in CA.TRACKED), n)
                    for n, f in enumerate(finals))
        best, who = ds[0]
        return teams[who] if best <= 0.35 * total else "NOVEL"
    return label


def game_rows():
    gidx = json.load(open(os.path.join(ROOT, "data", "ourgames",
                                       "index.json"), encoding="utf-8"))
    return list(gidx["games"].values())


def band_model(rows, ratings, own_scores):
    """A1: opponent-rating quantiles faced, bucketed by own rating."""
    by_sub = defaultdict(list)
    for g in rows:
        r = ratings.get(g.get("opponent"))
        if r:
            by_sub[str(g["submission"])].append(r)
    buckets = defaultdict(list)
    for sub, opps in by_sub.items():
        own = own_scores.get(sub)
        if own is None or len(opps) < MIN_GAMES:
            continue
        buckets[round(own / 200) * 200].extend(opps)
    model = {}
    for b, opps in sorted(buckets.items()):
        opps.sort()
        lo = opps[int(len(opps) * BAND_QUANTILES[0])]
        hi = opps[int(len(opps) * BAND_QUANTILES[1])]
        model[str(b)] = {"lo": lo, "hi": hi, "n": len(opps),
                         "median": opps[len(opps) // 2]}
    return model


def band_for(model, own_rating):
    if not model:
        return (2300.0, 2600.0)
    key = min(model, key=lambda k: abs(int(k) - own_rating))
    return model[key]["lo"], model[key]["hi"]


def score(g):
    return 1.0 if g.get("won") else (0.5 if g.get("tied") else 0.0)


def backtest(rows, ratings, own_scores, label):
    """A3: which panel weighting predicts the second half better?"""
    by_sub = defaultdict(list)
    for g in rows:
        by_sub[str(g["submission"])].append(g)
    results = []
    for sub, games in by_sub.items():
        if len(games) < MIN_GAMES:
            continue
        games.sort(key=lambda g: int(g["episode"]))
        half = len(games) // 2
        first, second = games[:half], games[half:]
        fam_first = defaultdict(list)
        for g in first:
            fam_first[label(g)].append(score(g))
        winrate = {f: statistics.mean(v) for f, v in fam_first.items()}

        # Weighting 1 -- LOSS panel (mimics stage_tapes): families of the
        # first half's losses, weighted by loss count.
        loss_w = Counter(label(g) for g in first
                         if not g.get("won") and not g.get("tied"))
        # Weighting 2 -- BAND panel: families of first-half opponents whose
        # rating sits in this sub's matchmaking band.
        own = own_scores.get(sub)
        model = band_model(rows, ratings, own_scores)
        lo, hi = band_for(model, own if own else 2400)
        band_w = Counter(label(g) for g in first
                         if lo <= (ratings.get(g.get("opponent")) or 0) <= hi)

        realized = statistics.mean(score(g) for g in second)
        row = {"sub": sub, "games": len(games), "realized": round(realized, 4)}
        for name, w in (("loss", loss_w), ("band", band_w)):
            tot = sum(n for f, n in w.items() if f in winrate)
            if tot == 0:
                row[name] = None
                continue
            pred = sum(winrate[f] * n for f, n in w.items()
                       if f in winrate) / tot
            row[name] = round(pred, 4)
        results.append(row)
    return results


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    ratings = board_ratings()
    own = own_sub_ratings()
    rows = game_rows()
    label = family_labeler()
    named = sum(1 for g in rows if ratings.get(g.get("opponent")))
    print(f"{len(rows)} games, {named} with a rated opponent "
          f"({len(ratings)} teams on the board); {len(own)} own submissions "
          f"scored")

    model = band_model(rows, ratings, own)
    print("\nA1 matchmaking bands (opponent rating 10th-90th pct by own "
          "rating bucket):")
    for b, m in sorted(model.items(), key=lambda kv: int(kv[0])):
        print(f"  own ~{b}: {m['lo']:.0f}..{m['hi']:.0f} "
              f"(median {m['median']:.0f}, n={m['n']})")

    results = backtest(rows, ratings, own, label)
    usable = [r for r in results if r["loss"] is not None
              and r["band"] is not None]
    print(f"\nA3 backtest over {len(usable)} submissions "
          f"(chronological half-split):")
    print(f"{'sub':<10}{'games':>6}{'realized':>10}{'loss-pred':>11}"
          f"{'band-pred':>11}")
    for r in usable:
        print(f"{r['sub']:<10}{r['games']:>6}{r['realized']:>10.3f}"
              f"{r['loss']:>11.3f}{r['band']:>11.3f}")
    mae_loss = statistics.mean(abs(r["loss"] - r["realized"]) for r in usable)
    mae_band = statistics.mean(abs(r["band"] - r["realized"]) for r in usable)
    sp_loss = spearman([r["loss"] for r in usable],
                       [r["realized"] for r in usable])
    sp_band = spearman([r["band"] for r in usable],
                       [r["realized"] for r in usable])
    print(f"\nMAE:      loss {mae_loss:.4f}   band {mae_band:.4f}")
    print(f"Spearman: loss {sp_loss:.3f}   band {sp_band:.3f}")
    adopt = mae_band < mae_loss and sp_band >= sp_loss
    verdict = ("ADOPT the band panel" if adopt
               else "KEEP the loss panel (band did not predict better)")
    print(f"VERDICT: {verdict}")
    json.dump({"model": model, "backtest": usable,
               "mae": {"loss": mae_loss, "band": mae_band},
               "spearman": {"loss": sp_loss, "band": sp_band},
               "adopt_band": bool(adopt)},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
