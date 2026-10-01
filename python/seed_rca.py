"""Per-turn RCA of a bad seed vs a good seed of the same world, same opponent (operator 29 Sep).

    PYTHONPATH=<repo>/src KAGG_BIN=... python python/seed_rca.py --cand .local/pub/v63.10_rl/main.py \
        --opp .local/field80/field/<agent>.py --bad 820024 --good 820252 --seat 0 [--out .local/rca_seed/<name>.md]

For each seed: the game on the faithful harness (Python opponent on the Rust `kagg serve` engine,
loss_forensics.capture_game), then per day: shops unlocked (the realised world beyond the first two), both
monies and the margin, both seats' units SOLD per product that day (from their SELL orders x the shed they had)
and revenue at the step's quote, animals bought, land bought. Prints where the two seeds diverge: the first
day the realised shops differ, the day the margin turns, and the per-product revenue gaps after that day.
"""
import argparse
import collections
import json
import os

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def play(cand, opp, seed, seat):
    from kaggriculture.bandit.gate import loss_forensics as LF
    ours = LF.load_pyagent(cand)
    recs, stream, us, them = LF.capture_game(ours, opp, seed, seat)
    return recs, stream, us, them


def per_day(recs, stream):
    days = collections.defaultdict(lambda: {"shops": None, "margin": 0.0, "sold_us": collections.Counter(), "sold_them": collections.Counter(),
                                            "rev_us": collections.Counter(), "rev_them": collections.Counter(), "buys_us": collections.Counter(), "buys_them": collections.Counter()})
    for i, r in enumerate(recs):
        d = r["step"] // 24
        D = days[d]
        if i < len(stream):
            town = (stream[i] or {}).get("town") or {}
            D["shops"] = tuple(town.get("unlocked_shops") or ())
        D["margin"] = r["our_money"] - r["opp_money"]
        for who, act, shed in (("us", r["our_act"], r["our_shed"]), ("them", r["opp_act"], r["opp_shed"])):
            left = dict(shed)
            for o in (act or {}).get("market") or []:
                if not isinstance(o, list) or len(o) < 2:
                    continue
                if o[0] == "SELL" and len(o) >= 3 and o[1] in PRODUCTS:
                    q = min(int(o[2]), max(0, int(left.get(o[1], 0))))
                    left[o[1]] = left.get(o[1], 0) - q
                    D[f"sold_{who}"][o[1]] += q
                    D[f"rev_{who}"][o[1]] += q * r["prices"].get(o[1], 0)
                elif o[0] in ("BUY_ANIMAL", "BUY_LAND", "BUY_PRODUCT"):
                    D[f"buys_{who}"][f"{o[0]} {o[1] if len(o) > 1 else ''}".strip()] += int(o[2]) if len(o) >= 3 else 1
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--bad", type=int, required=True)
    ap.add_argument("--good", type=int, required=True)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for lab, seed in (("bad", a.bad), ("good", a.good)):
        recs, stream, us, them = play(a.cand, a.opp, seed, a.seat)
        res[lab] = (per_day(recs, stream), us, them)
    L = [f"# Seed RCA: {os.path.basename(a.opp)} seat {a.seat}: bad {a.bad} vs good {a.good}\n"]
    for lab in ("bad", "good"):
        days, us, them = res[lab]
        L.append(f"## {lab} seed: final us {us:.0f} them {them:.0f} margin {us - them:+.0f}")
        prev = ()
        for d in sorted(days):
            D = days[d]
            new = [s for s in (D["shops"] or ()) if list(D["shops"]).count(s) > list(prev).count(s)]
            prev = D["shops"] or prev
            sold = {p: (D["sold_us"][p], D["sold_them"][p]) for p in PRODUCTS if D["sold_us"][p] or D["sold_them"][p]}
            L.append(f"- day {d:2d}: margin {D['margin']:+8.0f} | new shops {new} | sold us/them {sold} | buys us {dict(D['buys_us'])} them {dict(D['buys_them'])}")
        L.append("")
    # late revenue gaps by product (days 20-29)
    for lab in ("bad", "good"):
        days = res[lab][0]
        g = collections.Counter()
        for d in range(20, 30):
            for p in PRODUCTS:
                g[p] += days[d]["rev_them"][p] - days[d]["rev_us"][p] if d in days else 0
        L.append(f"{lab}: days 20-29 revenue gap (them - us) by product: {dict((k, round(v)) for k, v in g.most_common())}")
    txt = "\n".join(L)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
