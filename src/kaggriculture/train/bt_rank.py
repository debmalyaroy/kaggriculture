"""Bradley-Terry ranking from played games -- the ladder's own math.

Reads the episode sink (or any (a, b, winner) rows) and fits BT strengths
by minorization-maximization. Use for panel analysis and slot A/B, where
opponents vary; with identical rosters it reduces to win-count ordering.

    python -m kaggriculture.train.bt_rank --days 3
"""
import argparse
import math
import os
import sys
from collections import defaultdict

import kaggriculture.data.sink as sink  # noqa: E402


def fit(games, iters=200):
    """games: list of (winner, loser). Returns {player: strength}."""
    players = sorted({p for g in games for p in g})
    wins = defaultdict(int)
    pairs = defaultdict(int)
    for w, l in games:
        wins[w] += 1
        pairs[(w, l)] += 1
        pairs[(l, w)] += 0
    s = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            num = wins[p]
            den = 0.0
            for q in players:
                if q == p:
                    continue
                n_pq = pairs[(p, q)] + pairs[(q, p)]
                if n_pq:
                    den += n_pq / (s[p] + s[q])
            new[p] = (num / den) if den > 0 else s[p]
        norm = sum(new.values()) / max(1, len(new))
        s = {p: v / norm for p, v in new.items()}
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--contains", default=None,
                    help="only games where either side's name contains this")
    args = ap.parse_args()
    games = []
    for r in sink.rows(days=args.days):
        a = os.path.basename(str(r.get("agent") or ""))
        b = os.path.basename(str(r.get("opponent") or ""))
        if args.contains and args.contains not in a and args.contains not in b:
            continue
        bank, opp = r.get("bank"), r.get("opp_bank")
        if bank is None or opp is None or bank == opp:
            continue
        games.append((a, b) if bank > opp else (b, a))
    if not games:
        print("no decisive games in the sink for this filter")
        return 0
    s = fit(games)
    print(f"{len(games)} decisive games, {len(s)} players (BT strength, "
          f"log-scale rating):")
    for p, v in sorted(s.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {p[:44]:<46} {v:>8.3f}  {400 * math.log10(max(v, 1e-9)) + 2000:>7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
