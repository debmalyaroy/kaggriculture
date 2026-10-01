"""Find routes matching the winning FARM COMPOSITION (discussion 736214).

Luka Duvanov's teardown of the top open-loop agent (V16-RC5): the binding
constraint is ACTIONS, not tiles, so the winning shape is a SMALL herd and a
restrained footprint, not a maxed one:
  * ~8 cows, ~4 sheep, NO geese
  * only 2 land expansions (never the $4,000 4th quadrant)
  * fertiliser SPENT on strawberries, not sold (its town drain is zero)

Our own v29 forensics show the opposite failure mode (sells ~200 fertiliser,
likely over-animaled). This ranks the fresh WINNING pool by how closely each
route matches the winning shape, so crown_eval can test whether a
fundamentals-matched route actually out-rates our incumbent.

    python src/compose_filter.py --top 12
"""
from kaggriculture.paths import ROOT
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def composition(actions):
    cows = sheep = geese = land = fert_sold = 0
    # HIRE, BUY_LAND, BUY_ANIMAL, BUY_*, SELL are all MARKET orders in this
    # engine (kaggriculture._parse_order), not farmer/hands unit actions.
    for turn in actions:
        if not isinstance(turn, dict):
            continue
        for o in (turn.get("market") or []):
            if not (isinstance(o, list) and o):
                continue
            op = o[0]
            if op == "BUY_ANIMAL" and len(o) > 1:
                a = o[1]
                cows += a == "COW"
                sheep += a == "SHEEP"
                geese += a in ("GOOSE", "GEESE")
            elif op == "BUY_LAND":
                land += 1
            elif op == "SELL" and len(o) >= 3 and o[1] == "FERTILIZER":
                try:
                    fert_sold += max(0, int(o[2]))
                except (TypeError, ValueError):
                    pass
    return {"cows": cows, "sheep": sheep, "geese": geese, "land": land,
            "fert_sold": fert_sold}


def match_score(c):
    """Lower = closer to the winning shape (8 cows / 4 sheep / 0 geese /
    <=2 land / ~0 fertiliser sold)."""
    return (abs(c["cows"] - 8) + abs(c["sheep"] - 4) + 3 * c["geese"]
            + 2 * max(0, c["land"] - 2) + c["fert_sold"] / 20.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-bank", type=float, default=90000.0)
    ap.add_argument("--recent-only", action="store_true", default=True)
    args = ap.parse_args()

    import kaggriculture.data.routes as R
    idx = R.load_index()
    # candidate pool: recent winning routes with a real bank
    dates = sorted({r.get("date", "") for r in idx["routes"].values()})
    recent = set(dates[-6:])
    rows = []
    scanned = 0
    for rid, rec in idx["routes"].items():
        if not rec.get("won") or float(rec.get("bank", 0)) < args.min_bank:
            continue
        if args.recent_only and rec.get("date") not in recent:
            continue
        try:
            c = composition(R.load_route(rid))
        except Exception:                                          # noqa: BLE001
            continue
        scanned += 1
        rows.append((match_score(c), rid, rec, c))
    rows.sort(key=lambda t: (t[0], -float(t[2].get("bank", 0))))
    print("scanned %d recent winning routes; closest to the winning shape:"
          % scanned)
    out = []
    for score, rid, rec, c in rows[:args.top]:
        print("  %-14s match %5.1f  bank %8.0f  %dC/%dS/%dG land%d fert%d  %s"
              % (rid, score, float(rec.get("bank", 0)), c["cows"], c["sheep"],
                 c["geese"], c["land"], c["fert_sold"], rec.get("team")))
        out.append({"id": rid, "match": score, "bank": rec.get("bank"),
                    "team": rec.get("team"), **c})
    # our incumbent's own shape for contrast
    try:
        inc = composition(R.load_route("92513718_s1"))
        print("\nincumbent 92513718_s1 shape: %dC/%dS/%dG land%d fert_sold%d "
              "(match %.1f)" % (inc["cows"], inc["sheep"], inc["geese"],
                                inc["land"], inc["fert_sold"],
                                match_score(inc)))
    except Exception:                                              # noqa: BLE001
        pass
    json.dump({"top": out}, open(os.path.join(
        ROOT, "models", "factory", "compose_filter.json"), "w",
        encoding="utf-8"), indent=1)
    print("\nroute ids:", " ".join(r["id"] for r in out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
