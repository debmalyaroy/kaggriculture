"""Contested-dump field forensics: who sold first, and what it was worth.

The relay layer exists on one bet -- that when both farms are about to dump
the same product, selling FIRST is worth money, because the second seller
prints into a market the first one just pushed down. Every measurement of
that bet so far has been simulated (paired ablations against clones). This
reads the bet off REAL ladder replays.

For each episode it reconstructs, per product, every *collision*: our SELL
and theirs landing within `--window` turns of each other. For each collision
it records who was first, the market price each side actually printed at, and
the money the second seller gave up versus printing at the first seller's
price. Aggregated, that is the relay's real-world win rate (`first%`) and its
real-world stake ($/game), measured on the ladder rather than in self-play.

It also measures the premise directly: `--impact` regresses the per-turn
price change on the units sold into that product, so "does racing matter at
all" is answered by the same data rather than assumed.

    python src/contested_dumps.py --losses            # every v24.1 loss
    python src/contested_dumps.py --replay X.json --seat 0
    python src/contested_dumps.py --losses --wins 12 --impact

Diagnostic only -- nothing here gates a build. It tells us whether to keep
spending agent complexity on the race.
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys

# Windows' console codec is cp1252 and opponent team names are arbitrary
# UTF-8 (a CJK name crashed this tool mid-run on 2026-09-04, after 22 of its
# episodes had already printed). An analysis tool must never die on a name.
try:                                        # pragma: no cover - tty only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAWBERRY", "POTATO",
            "MILK", "WOOL", "EGG", "FERTILIZER")


# ------------------------------------------------------------------- read --

def sells_and_prices(steps):
    """(sells, prices) -- sells[seat] = [(turn, product, qty)], prices[turn]."""
    sells = {0: [], 1: []}
    prices = []
    for i, step in enumerate(steps):
        if not step:
            prices.append({})
            continue
        obs = step[0].get("observation") or {}
        prices.append(dict((obs.get("market") or {}).get("prices") or {}))
        for seat in (0, 1):
            if seat >= len(step):
                continue
            for order in ((step[seat].get("action") or {}).get("market")
                          or []):
                if (isinstance(order, (list, tuple)) and len(order) >= 3
                        and order[0] == "SELL"):
                    sells[seat].append((i, str(order[1]), int(order[2] or 0)))
    return sells, prices


def price_at(prices, turn, product):
    """The price the agent printed at: the quote it acted on, that turn."""
    for t in (turn, turn + 1, turn - 1):
        if 0 <= t < len(prices) and product in prices[t]:
            return float(prices[t][product])
    return None


# -------------------------------------------------------------- collisions --

def collisions(steps, seat, window=6):
    """Every same-product sell of ours that races one of theirs.

    A collision is charged once per pair of nearest orders, so a long
    liquidation tail of ten orders against their one does not inflate the
    count ten-fold.
    """
    sells, prices = sells_and_prices(steps)
    mine, theirs = sells[seat], sells[1 - seat]
    out = []
    used = set()
    for (t, prod, qty) in mine:
        cands = [(abs(t2 - t), j, t2, q2)
                 for j, (t2, p2, q2) in enumerate(theirs)
                 if p2 == prod and abs(t2 - t) <= window and j not in used]
        if not cands:
            continue
        _, j, t2, q2 = min(cands)
        used.add(j)
        pm, pt = price_at(prices, t, prod), price_at(prices, t2, prod)
        if pm is None or pt is None:
            continue
        # Same-turn orders are a genuine TIE -- both print against the same
        # quote and the engine's order does not favour a seat. Crediting the
        # seat-0 tie-break made every self-match read 100% first.
        first_us = None if t == t2 else t < t2
        if first_us is True:
            loss_to_second, our_edge = q2 * (pm - pt), 0.0
        elif first_us is False:
            loss_to_second = qty * (pt - pm)
            our_edge = -loss_to_second
        else:
            loss_to_second, our_edge = 0.0, 0.0
        out.append({"turn": t, "opp_turn": t2, "product": prod,
                    "qty": qty, "opp_qty": q2, "our_price": pm,
                    "opp_price": pt, "first": first_us,
                    "second_gave_up": round(loss_to_second, 1),
                    "our_edge": round(our_edge, 1),
                    "lead": t2 - t})
    return out


def impact(steps):
    """Price change per unit sold, de-trended within the day.

    A raw regression of dp% on quantity is confounded: prices drift over a
    day and sells cluster where the price is already moving, which is how a
    first pass produced *positive* slopes for STRAWBERRY and MILK (sell more,
    price rises). Every turn of the episode is a sample -- zero-flow turns
    included -- and both quantity and dp% are de-meaned within (product, day)
    so the slope reads only the within-day, cross-turn relationship.
    """
    sells, prices = sells_and_prices(steps)
    flow = {}
    for seat in (0, 1):
        for (t, prod, qty) in sells[seat]:
            flow[(t, prod)] = flow.get((t, prod), 0) + qty
    # Two windows per turn: the FORWARD one (t -> t+2) is the effect a sell
    # could cause, the BACKWARD one (t-2 -> t) is a placebo -- a sell cannot
    # move a price that already moved. If both slopes carry the same sign the
    # regression is reading sell TIMING, not sell impact.
    cells, plac = {}, {}
    for t in range(2, len(prices) - 2):
        day = t // 24
        for prod in prices[t]:
            p0, p1 = price_at(prices, t, prod), price_at(prices, t + 2, prod)
            pb = price_at(prices, t - 2, prod)
            if p0 is None or p1 is None or not p0:
                continue
            q = float(flow.get((t, prod), 0))
            cells.setdefault((prod, day), []).append((q, (p1 - p0) / p0))
            if pb:
                plac.setdefault((prod, day), []).append((q, (p0 - pb) / pb))
    def _slopes(src):
        pts = {}
        for (prod, _day), rows in src.items():
            if len(rows) < 4 or all(r[0] == 0 for r in rows):
                continue
            mq = sum(r[0] for r in rows) / len(rows)
            mp = sum(r[1] for r in rows) / len(rows)
            pts.setdefault(prod, []).extend(
                [(r[0] - mq, r[1] - mp) for r in rows])
        res = {}
        for prod, rows in pts.items():
            if len(rows) < 24:
                continue
            sxy = sum(x * y for x, y in rows)
            sxx = sum(x * x for x, _ in rows) or 1e-9
            res[prod] = (100.0 * sxy / sxx, len(rows))
        return res

    fwd, back = _slopes(cells), _slopes(plac)
    out = {}
    for prod, (slope, n) in fwd.items():
        qs = [flow.get(k, 0) for k in flow if k[1] == prod]
        out[prod] = {"slope_pct_per_unit": slope, "n": n,
                     "placebo_pct_per_unit": back.get(prod, (0.0, 0))[0],
                     "mean_qty": (sum(qs) / len(qs)) if qs else 0.0}
    return out


# ------------------------------------------------------------------ driver --

def our_games(subs, want_wins=0):
    """(episode, seat, margin, opponent, won) for our recent games."""
    idx = json.load(open(os.path.join(ROOT, "data", "ourgames",
                                      "index.json"), encoding="utf-8"))
    rows = [g for g in idx["games"].values()
            if str(g.get("submission")) in subs]
    losses = [g for g in rows if not g["won"] and not g.get("tied")]
    wins = sorted((g for g in rows if g["won"]),
                  key=lambda g: -(g["bank"] - g["opp_bank"]))[:want_wins]
    return [{"episode": str(g["episode"]), "seat": int(g.get("seat") or 0),
             "margin": float(g["bank"]) - float(g["opp_bank"]),
             "opponent": str(g.get("opponent") or "?"),
             "won": bool(g["won"])} for g in losses + wins]


def fetch(ep, dest):
    path = os.path.join(dest, f"{ep}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    import kaggriculture.data.sameday as sameday
    os.makedirs(dest, exist_ok=True)
    return sameday._grab_direct(ep, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--losses", action="store_true",
                    help="every loss of the submissions in --subs")
    ap.add_argument("--wins", type=int, default=0,
                    help="also include the N biggest wins, as a control")
    ap.add_argument("--subs", default="55477892,55477866")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--impact", action="store_true")
    ap.add_argument("--dest",
                    default=os.path.join(ROOT, ".local", "lossreplays"))
    ap.add_argument("--json", dest="as_json")
    args = ap.parse_args()

    jobs = []
    if args.replay:
        for p in glob.glob(args.replay):
            jobs.append({"episode": os.path.splitext(
                os.path.basename(p))[0], "seat": args.seat, "margin": None,
                "opponent": "?", "won": None, "path": p})
    if args.losses or args.wins:
        subs = set(s.strip() for s in args.subs.split(",") if s.strip())
        for g in our_games(subs, args.wins):
            g["path"] = None
            jobs.append(g)
    if not jobs:
        print("nothing to do: pass --replay or --losses")
        return 1

    tot = {"n": 0, "first": 0, "tie": 0, "edge": 0.0, "conceded": 0.0,
           "games": 0}
    per_game, imp_acc = [], {}
    for g in jobs:
        try:
            path = g["path"] or fetch(g["episode"], args.dest)
            steps = json.load(open(path, encoding="utf-8"))["steps"]
        except Exception as exc:                                # noqa: BLE001
            print(f"{g['episode']}: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        cs = collisions(steps, g["seat"], args.window)
        if args.impact:
            for prod, st in impact(steps).items():
                a = imp_acc.setdefault(prod, {"s": 0.0, "n": 0, "q": 0.0,
                                              "p": 0.0})
                a["s"] += st["slope_pct_per_unit"] * st["n"]
                a["p"] += st["placebo_pct_per_unit"] * st["n"]
                a["n"] += st["n"]
                a["q"] += st["mean_qty"] * st["n"]
        first = sum(1 for c in cs if c["first"] is True)
        tie = sum(1 for c in cs if c["first"] is None)
        edge = sum(c["our_edge"] for c in cs)
        conceded = sum(c["second_gave_up"] for c in cs)
        tot["n"] += len(cs)
        tot["first"] += first
        tot["tie"] += tie
        tot["edge"] += edge
        tot["conceded"] += conceded
        tot["games"] += 1
        decided = len(cs) - tie
        per_game.append({"episode": g["episode"], "opponent": g["opponent"],
                         "margin": g["margin"], "won": g["won"],
                         "collisions": len(cs), "first": first, "tie": tie,
                         "our_edge": round(edge, 1),
                         "we_cost_them": round(conceded, 1),
                         "median_lead": (sorted(c["lead"] for c in cs)
                                         [len(cs) // 2] if cs else None)})
        tag = ("WIN " if g["won"] else "LOSS") if g["won"] is not None else "?  "
        mg = f"{g['margin']:+,.0f}" if g["margin"] is not None else "?"
        print(f"{tag} {g['episode']} vs {g['opponent'][:20]:20} "
              f"margin {mg:>10}  races {len(cs):>3} tie {tie:>3} "
              f"first {first:>3} "
              f"({(100.0 * first / decided) if decided else 0:.0f}% of decided)"
              f"  we_lost {edge:>+9,.0f}  they_lost {conceded:>+9,.0f}",
              flush=True)

    n, gm = tot["n"], max(1, tot["games"])
    dec = n - tot["tie"]
    print(f"\n=== {tot['games']} games, {n} contested dumps "
          f"({tot['tie']} exact ties, {dec} decided) ===")
    if dec:
        print(f"we sold first: {tot['first']}/{dec} "
              f"({100.0 * tot['first'] / dec:.1f}% of decided races)  "
              f"-- the relay's REAL win rate")
        print(f"our edge when we lost the race: {tot['edge']:+,.0f} "
              f"({tot['edge'] / gm:+,.0f}/game)")
        print(f"what being first cost THEM: {tot['conceded']:+,.0f} "
              f"({tot['conceded'] / gm:+,.0f}/game)")
        print(f"net race value: {(tot['conceded'] + tot['edge']) / gm:+,.0f}"
              f"/game")
    if imp_acc:
        print("\nprice impact (per unit sold, % of price) -- placebo is the "
              "SAME regression on the price move that already happened;\n"
              "a placebo near the estimate means the number is sell TIMING, "
              "not impact:")
        for prod, a in sorted(imp_acc.items(),
                              key=lambda kv: -abs(kv[1]["s"] / kv[1]["n"])):
            s, p = a["s"] / a["n"], a["p"] / a["n"]
            # A sell cannot move a price that already moved, so ANY large
            # placebo -- either sign -- means the regressor is entangled with
            # the market's own cycle and the forward slope is not impact.
            flag = "ENTANGLED" if abs(p) > 0.4 * abs(s) else "clean"
            print(f"  {prod:<12} {s:+.4f} %/unit   placebo {p:+.4f}  "
                  f"[{flag}]  (mean order {a['q'] / a['n']:.0f} units, "
                  f"n={a['n']})")
    if args.as_json:
        json.dump({"total": tot, "games": per_game}, open(args.as_json, "w",
                  encoding="utf-8"), indent=1)
        print(f"-> {args.as_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
