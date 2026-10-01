"""Probe the loss tapes of python/rshell/loss_rca.py for the class-specific mechanisms (analysis only).

    python python/rshell/loss_probe.py [--rca data/rshell/public25/rca]

  ENDGAME:STRAWBERRY  first step each side sells STRAWBERRY on days 27-29, and who sold first
  ECONOMY:TOMATO      PLANT TOMATO / BUY_SEED TOMATO counts per side (does our agent plant tomatoes at all?)
  ECONOMY:WOOL        FEED / CARE / COLLECT counts per side on sheep days
  SPEND:*             spend composition per side: buys by op with the day's quote (BUY_* priced from the tape's own
                      market via the rca ledgers is not available here, so counts by op + land / hires / animals)
"""
import argparse
import collections as C
import json
import os
import statistics as S

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ops(tape, seat):
    out = []
    for t, pair in enumerate(tape["actions"]):
        a = pair[seat] or {}
        for u in [a.get("farmer") or []] + list(a.get("hands") or []):
            if u:
                out.append((t, "U", tuple(str(x) for x in u)))
        for m in a.get("market") or []:
            if m:
                out.append((t, "M", tuple(str(x) for x in m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rca", default=os.path.join(RL, "data", "rshell", "public25", "rca"))
    a = ap.parse_args()
    rows = [x for x in json.load(open(os.path.join(a.rca, "rca.json")))["losses"] if "class" in x]
    tape = lambda k: json.load(open(os.path.join(a.rca, "tapes", k + ".json")))  # noqa: E731
    rep = {}
    # ENDGAME strawberry: who sells first on days 27..29
    first = C.Counter()
    lead = []
    for x in [r for r in rows if r["class"] == "ENDGAME" and r["product"] == "STRAWBERRY"]:
        tp = tape(x["key"])
        us, th = x["seat"], 1 - x["seat"]
        def f(seat, lo):
            for t, kind, o in ops(tp, seat):
                if t >= lo and kind == "M" and o[0] == "SELL" and o[1] == "STRAWBERRY" and (len(o) < 3 or o[2] != "0"):
                    return t
            return None
        for lo in (648, 672, 696):
            a_, b_ = f(us, lo), f(th, lo)
            if a_ is None and b_ is None:
                continue
            first[(lo // 24, "us" if (a_ is not None and (b_ is None or a_ < b_)) else "them" if (b_ is not None and (a_ is None or b_ < a_)) else "same")] += 1
            if a_ is not None and b_ is not None:
                lead.append(a_ - b_)
    rep["endgame_strawberry_first_seller"] = {f"day{k[0]}:{k[1]}": v for k, v in sorted(first.items())}
    rep["endgame_strawberry_our_delay_steps_median"] = S.median(lead) if lead else None
    # ECONOMY tomato: our tomato plantings
    tom = []
    for x in [r for r in rows if r["class"] == "ECONOMY" and r["product"] == "TOMATO"]:
        tp = tape(x["key"])
        cnt = lambda seat, pred: sum(1 for t, k, o in ops(tp, seat) if pred(k, o))  # noqa: E731
        plant = lambda k, o: k == "U" and o[0] == "PLANT" and len(o) > 1 and o[1] == "TOMATO"  # noqa: E731
        seed = lambda k, o: k == "M" and o[0] == "BUY_SEED" and o[1] == "TOMATO"  # noqa: E731
        land = lambda k, o: k == "M" and o[0] == "BUY_LAND"  # noqa: E731
        first_t = min([t for t, k, o in ops(tp, 1 - x["seat"]) if plant(k, o)] or [None]) if any(plant(k, o) for _, k, o in ops(tp, 1 - x["seat"])) else None
        tom.append((cnt(x["seat"], plant), cnt(1 - x["seat"], plant), cnt(x["seat"], seed), cnt(1 - x["seat"], seed), cnt(x["seat"], land), cnt(1 - x["seat"], land), first_t))
    if tom:
        rep["economy_tomato"] = {"our_plant_median": S.median(t[0] for t in tom), "their_plant_median": S.median(t[1] for t in tom),
                                 "games_we_plant_none": sum(t[0] == 0 for t in tom), "n": len(tom),
                                 "their_first_tomato_plant_day_median": S.median([t[6] // 24 for t in tom if t[6] is not None] or [0]),
                                 "land_buys_us_them": (sum(t[4] for t in tom), sum(t[5] for t in tom))}
    # ECONOMY wool: FEED / CARE / COLLECT
    wool = []
    for x in [r for r in rows if r["class"] == "ECONOMY" and r["product"] == "WOOL"]:
        tp = tape(x["key"])
        c = lambda seat, op: sum(1 for t, k, o in ops(tp, seat) if k == "U" and o[0] == op)  # noqa: E731
        wool.append({op: (c(x["seat"], op), c(1 - x["seat"], op)) for op in ("FEED", "CARE", "COLLECT", "SHEAR", "PICKUP")})
    if wool:
        rep["economy_wool_ops_us_them_median"] = {op: (S.median(w[op][0] for w in wool), S.median(w[op][1] for w in wool)) for op in wool[0]}
    # SPEND: counts of market buys by op
    for p in ("WHEAT", "CARROT", "FERTILIZER"):
        X = [r for r in rows if r["class"] == "SPEND" and r["product"] == p]
        if not X:
            continue
        diff = C.Counter()
        for x in X:
            tp = tape(x["key"])
            for seat, sgn in ((x["seat"], -1), (1 - x["seat"], 1)):
                for t, k, o in ops(tp, seat):
                    if k == "M" and o[0].startswith(("BUY", "HIRE")):
                        diff[" ".join(o[:2])] += sgn * (int(o[2]) if len(o) > 2 and o[2].lstrip("-").isdigit() else 1)
        rep[f"spend_{p}_them_minus_us_units_per_game"] = {k: round(v / len(X), 1) for k, v in diff.most_common() if abs(v / len(X)) >= 0.2}
    json.dump(rep, open(os.path.join(a.rca, "probe.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
