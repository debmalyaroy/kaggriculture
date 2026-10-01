"""Sale timing in the public-25 losses: town-draw phase (step % 4), hour of day, order slot, first seller per
sale-day, for the price / endgame classes of python/rshell/loss_rca.py. Analysis only.

    python python/rshell/loss_timing.py [--rca data/rshell/public25/rca]
"""
import argparse
import collections as C
import json
import os
import statistics as S

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sells(tp, seat, item):
    out = []
    for t, pair in enumerate(tp["actions"]):
        for i, m in enumerate((pair[seat] or {}).get("market") or []):
            if m and m[0] == "SELL" and m[1] == item and (len(m) < 3 or int(m[2]) > 0):
                out.append((t, i, int(m[2]) if len(m) > 2 else 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rca", default=os.path.join(RL, "data", "rshell", "public25", "rca"))
    a = ap.parse_args()
    rows = [x for x in json.load(open(os.path.join(a.rca, "rca.json")))["losses"] if x.get("class")]
    rep = {}
    for cls, item in [("PRICE", "MILK"), ("ENDGAME", "MILK"), ("SPEND", "MILK"), ("PRICE", "STRAWBERRY"), ("ENDGAME", "WOOL"), ("PRICE", "WHEAT")]:
        X = [x for x in rows if x["class"] == cls and x["product"] == item]
        mod = {"us": C.Counter(), "them": C.Counter()}
        hr = {"us": C.Counter(), "them": C.Counter()}
        slot = {"us": [], "them": []}
        units = {"us": 0, "them": 0}
        firsts = C.Counter()
        for x in X:
            tp = json.load(open(os.path.join(a.rca, "tapes", x["key"] + ".json")))
            su, st = sells(tp, x["seat"], item), sells(tp, 1 - x["seat"], item)
            for who, ss in (("us", su), ("them", st)):
                for t, i, q in ss:
                    mod[who][t % 4] += 1
                    hr[who][t % 24] += 1
                    slot[who].append(i)
                    units[who] += q
            du, dt = {}, {}
            for t, _, _ in su:
                du.setdefault(t // 24, t)
            for t, _, _ in st:
                dt.setdefault(t // 24, t)
            for d in set(du) | set(dt):
                u, v = du.get(d), dt.get(d)
                firsts["us" if u is not None and (v is None or u < v) else "them" if v is not None and (u is None or v < u) else "same"] += 1
        n = {w: sum(mod[w].values()) for w in mod}
        share = lambda c, tot: {k: round(v / max(1, tot), 2) for k, v in sorted(c.items())}  # noqa: E731
        rep[f"{cls}:{item}"] = {"games": len(X), "sell_orders": n, "ordered_units": units,
                               "step_mod4_share": {w: share(mod[w], n[w]) for w in mod},
                               "top_hours": {w: [h for h, _ in hr[w].most_common(6)] for w in hr},
                               "median_slot": {w: S.median(slot[w] or [0]) for w in slot},
                               "first_seller_per_sale_day": dict(firsts)}
    json.dump(rep, open(os.path.join(a.rca, "timing.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
