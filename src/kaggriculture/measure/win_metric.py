"""The competition's currency: WINS. Margin is only a proxy, and a bad one.

The ladder scores win / draw / loss. $1 of margin and $10,000 of margin are
paid identically, so any decision made on mean margin is denominated in a
currency the competition does not use.

The failure is not merely cosmetic. A margin shift converts to wins only
through the DENSITY of the margin distribution near zero, so two changes with
the same mean margin can be worth wildly different amounts of win rate:

  measured over 1,748 real games (data/ourgames, 2026-08-13):
    W/D/L 1044/26/678, win rate 0.597
    median |margin| $8,136, sd $23,850
    27.6% of games are decided by less than $3,000
    240 of our 678 losses were by less than $3,000

  so a uniform +$1,000/game is worth +5.3pp of win rate, while +$200,000
  concentrated in a tenth of the games is worth ~0.

That is why `mean_margin` must never be the decision statistic: it is a mean
over a variable whose value is a step function at zero. This module supplies
the replacements.

Use `score()` for a single game, `expected_score()` to summarise, `flips()` to
convert a per-game effect into wins EXACTLY, and `paired_test()` to decide
whether a difference is real. `dollars_to_wins()` exists only to restate
legacy dollar claims; do not build new criteria on it.

    python src/win_metric.py            # print the exchange table
"""
from kaggriculture.paths import ROOT
import json
import math
import os

OURGAMES = os.path.join(ROOT, "data", "ourgames", "index.json")

WIN, DRAW, LOSS = 1.0, 0.5, 0.0


# --------------------------------------------------------------- the metric --

def score(bank, opp_bank):
    """The only thing the ladder pays: 1 win, 0.5 draw, 0 loss."""
    a, b = float(bank), float(opp_bank)
    if a > b:
        return WIN
    if a < b:
        return LOSS
    return DRAW


def expected_score(scores):
    """Mean score = win rate with draws counted as half. The headline number."""
    scores = list(scores)
    return sum(scores) / len(scores) if scores else float("nan")


def summarise(pairs):
    """(w, d, l, expected_score) for an iterable of (bank, opp_bank)."""
    w = d = l = 0
    for bank, opp in pairs:
        s = score(bank, opp)
        if s == WIN:
            w += 1
        elif s == DRAW:
            d += 1
        else:
            l += 1
    n = w + d + l
    return w, d, l, ((w + 0.5 * d) / n if n else float("nan"))


# ------------------------------------------------- effects, priced in wins --

def flips(baseline_margins, deltas):
    """EXACT win value of a per-game effect -- the right way to price a change.

    `baseline_margins[i]` is our margin in game i without the change;
    `deltas[i]` is the change's effect on that same game. Returns
    (gained, lost, net_score_delta_per_game).

    This is exact where a mean-margin average is not, because it applies the
    step at zero per game instead of averaging dollars that mostly cannot
    matter. Always prefer this when per-game numbers exist.
    """
    gained = lost = 0
    net = 0.0
    n = 0
    for base, d in zip(baseline_margins, deltas):
        before, after = score(base, 0.0), score(base + d, 0.0)
        net += after - before
        if after > before:
            gained += 1
        elif after < before:
            lost += 1
        n += 1
    return gained, lost, (net / n if n else 0.0)


def dollars_to_wins(margins, dollars):
    """Win-rate value of a UNIFORM +$dollars/game shift, from real margins.

    For restating legacy dollar-denominated claims only. It assumes the effect
    lands equally in every game, which is exactly the assumption that makes
    mean margin misleading -- so treat the result as an upper bound and use
    `flips()` whenever per-game numbers exist.
    """
    margins = [float(m) for m in margins]
    if not margins:
        return float("nan")
    n = len(margins)
    if dollars >= 0:
        return sum(1 for m in margins if -dollars < m <= 0) / n
    return -sum(1 for m in margins if 0 < m <= -dollars) / n


# ------------------------------------------------------------- is it real? --

def paired_test(a_scores, b_scores):
    """Paired comparison in win units. Replaces the $3,000 margin floor.

    Same opponents, same seeds, both seats, so the games pair up. Only the
    DISCORDANT pairs -- where one build won and the other did not -- carry
    information, which is McNemar's exact test. Returns a dict with the score
    difference, the discordant counts, the two-sided exact p-value, and a
    normal-approximation 95% CI on the difference.
    """
    a, b = list(a_scores), list(b_scores)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    win_a = sum(1 for x, y in zip(a, b) if x > y)      # A better on this pair
    win_b = sum(1 for x, y in zip(a, b) if y > x)
    disc = win_a + win_b
    # two-sided exact binomial on the discordant pairs
    if disc == 0:
        p = 1.0
    else:
        k = min(win_a, win_b)
        tail = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2 ** disc)
        p = min(1.0, 2 * tail)
    diff = (sum(a) - sum(b)) / n if n else 0.0
    # CI on the paired per-game score difference
    if n > 1:
        mu = diff
        var = sum(((x - y) - mu) ** 2 for x, y in zip(a, b)) / (n - 1)
        half = 1.959964 * math.sqrt(var / n)
    else:
        half = float("nan")
    return {"n_pairs": n, "score_a": expected_score(a),
            "score_b": expected_score(b), "score_diff": diff,
            "better_a": win_a, "better_b": win_b, "discordant": disc,
            "p_value": p, "ci95": (diff - half, diff + half),
            "significant": bool(p < 0.05)}


def min_detectable(n_pairs, base_rate=0.6, alpha=0.05, power=0.8,
                   discordance=None):
    """Smallest score difference `n_pairs` paired games can detect.

    The honest replacement for "differences under ~$3k are noise": a floor in
    the currency that pays, that shrinks as evidence accumulates instead of
    being a fixed dollar guess.

    This is a PLANNING tool -- it answers "how many games should I run?". It is
    NOT a post-hoc veto: an actual paired test on observed data can be
    significant below this floor when the effect is one-directional. The 2026-
    08-13 timing A/B is the worked example: -14.29pp against a 16.34pp floor,
    yet 6 blocks worse and 0 better gives an exact p of 0.031. Trust
    `paired_test()`, not this, once the games exist.

    `discordance` is the fraction of pairs on which the two builds disagree,
    and it drives the power entirely. PASS THE OBSERVED VALUE when you have
    it: two builds differing in one flag agree on most games, so discordance
    is small and the paired test is far more powerful than the default
    neutral prior (2p(1-p) ~= 0.48) suggests. The default is deliberately
    conservative, not representative.
    """
    if n_pairs < 2:
        return float("nan")
    z_a, z_b = 1.959964, 0.8416212
    pdisc = (max(1.0 / n_pairs, float(discordance))
             if discordance is not None
             else max(0.05, 2 * base_rate * (1 - base_rate)))
    return (z_a + z_b) * math.sqrt(pdisc / n_pairs)


# --------------------------------------------------------------- live data --

def parse_eval(out, strict=True):
    """Parse evaluate.py's stable RESULT lines. THE only way to read it.

    Three call sites in refresh_cycle used to scrape the human table's `win%`
    column with a regex; when that table gained a column every tournament cell
    silently parsed as 0/0, which would have handed the crown gate garbage
    without raising anything. `strict` makes a parse failure loud instead.

    Returns [{opponent, score, wins, draws, losses, games, bank, opp_bank,
    margin}].
    """
    rows = []
    for line in (out or "").splitlines():
        if not line.startswith("RESULT\t"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            continue
        try:
            rows.append({"opponent": f[1], "score": float(f[2]),
                         "wins": int(f[3]), "draws": int(f[4]),
                         "losses": int(f[5]), "games": int(f[6]),
                         "bank": float(f[7]), "opp_bank": float(f[8]),
                         "margin": float(f[9])})
        except ValueError:
            continue
    if strict and not rows:
        raise ValueError(
            "no RESULT lines in evaluate.py output -- refusing to report a "
            "silent 0/0. Check the child process actually ran (engine_check, "
            "missing agent path, or a traceback above).")
    return rows


def totals(rows):
    """(weighted score, wins, games) over parse_eval rows."""
    wins = sum(r["wins"] + 0.5 * r["draws"] for r in rows)
    games = sum(r["games"] for r in rows)
    return (wins / games if games else float("nan")), wins, games


def our_margins(subs=None, path=OURGAMES):
    """Real margins from our ladder games -- the empirical distribution."""
    try:
        idx = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for g in (idx.get("games") or {}).values():
        if g.get("bank") is None or g.get("opp_bank") is None:
            continue
        if subs and str(g.get("submission")) not in subs:
            continue
        out.append(float(g["bank"]) - float(g["opp_bank"]))
    return out


def main():
    m = our_margins()
    if not m:
        print("no games indexed yet")
        return 0
    w = sum(1 for x in m if x > 0)
    d = sum(1 for x in m if x == 0)
    print(f"{len(m)} real games: W/D/L {w}/{d}/{len(m) - w - d}, "
          f"expected score {(w + 0.5 * d) / len(m):.3f}")
    print("\nuniform margin shift -> win-rate value (the exchange rate):")
    for dollars in (500, 1000, 2000, 3000, 5000, 10000, 20000):
        v = dollars_to_wins(m, dollars)
        print(f"  +${dollars:>6,}/game  ->  {100 * v:>5.2f}pp of win rate "
              f"({round(v * len(m)):>3} of {len(m)} games flip)")
    print("\nminimum DETECTABLE score difference, by paired sample size:")
    for n in (28, 56, 84, 168, 336, 1000):
        print(f"  n={n:>5} pairs  ->  {100 * min_detectable(n):>5.2f}pp")
    print("\nNote: the exchange rate assumes a UNIFORM shift. Use flips() with "
          "per-game numbers whenever you have them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
