"""Per-game, per-turn RCA of every loss against a tape (operator 29 Sep: "detailed RCA on why we are losing").

    python python/loss_rca.py --rec DIR --money FILE --days FILE --out DIR [--name top]

Inputs (one tapeplay run of the candidate over the loss tapes):
  --rec    tapeplay --record DIR     (both seats' actions as played; tape "seat" = OUR seat)
  --money  tapeplay --money-dump F   (id, step, our money, their money after each step)
  --days   tapedump over --rec       (per seat per day: sells / sell_rev / shed / plants / animals)
Per lost game:
  arc        margin (us - them) on days 5,10,15,20,24,26,28,29 and the day the final deficit opened
  class      A behind by day 10 | B behind by day 20 | C level to day 20, lost days 20-26 | D ahead on day 26, lost at the end
  turns      the 8 steps with the largest drop in margin, each with both seats' market orders and money change
             (a drop = their money rose more than ours: their sale, our spend, their sale into the premium we waited for)
  products   days 20-29 per product: units, revenue, average price, us vs them; stock left on day 29
  cause      the product whose revenue gap explains most of the loss, and whether it is a VOLUME gap (they sold more
             units: production / farm plan) or a PRICE gap (same units, higher price: sale timing)
  verdict    sale-fixable (price-driven, within reach), plan-fixable (volume-driven), or out of reach (deficit larger
             than every product gap we can close by timing)
Writes OUT/<name>_rca.jsonl (one line per game), OUT/<name>_rca.md (summary + worst/representative games).
"""
import argparse
import collections
import glob
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--money", required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="loss")
    a = ap.parse_args()
    money = collections.defaultdict(dict)
    for l in open(a.money, encoding="utf-8"):
        i, t, x, y = l.rstrip("\n").split("\t")
        money[i][int(t)] = (float(x), float(y))
    days = collections.defaultdict(lambda: collections.defaultdict(dict))
    for l in open(a.days, encoding="utf-8"):
        d = json.loads(l)
        k = os.path.basename(str(d["key"])).replace(".json", "").replace("__r", "")
        days[k][d["seat"]][d["day"]] = d
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.rec, "*.json"))):
        g = os.path.basename(f).replace("__r.json", "").replace(".json", "")
        if g not in money:
            continue
        m = money[g]
        last = max(m)
        fin = m[last][0] - m[last][1]
        if fin >= 0:
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        # tapeplay --record writes opp_team verbatim (team names may carry backslashes / control chars): drop it
        txt = re.sub(r'"opp_team":(null|".*?"),"actions"', '"actions"', txt, count=1)
        rec = json.loads(txt)
        us = rec["seat"]
        acts = rec["actions"]
        gap = {t: v[0] - v[1] for t, v in m.items()}
        at = lambda day: gap.get(min(day * 24, last), 0.0)  # noqa: E731
        arc = {d: round(at(d)) for d in (5, 10, 15, 20, 24, 26, 28, 29)}
        # the day the final deficit opened: last day the margin was still >= 0
        opened = next((d for d in range(29, -1, -1) if at(d) >= 0), None)
        cls = "A" if at(10) < -2000 else "B" if at(20) < -2000 else ("D" if at(26) >= 0 else "C")
        # decisive turns: largest single-step drops in margin
        drops = []
        for t in range(2, last + 1):
            if t in gap and t - 1 in gap:
                dm = gap[t] - gap[t - 1]
                if dm < 0:
                    drops.append((dm, t))
        drops.sort()
        turns = []
        for dm, t in drops[:8]:
            st = t - 1                                   # money after step t reflects the action at step t-1
            pair = acts[st] if st < len(acts) else [None, None]
            mk = lambda s: [o for o in ((pair[s] or {}).get("market") or []) if o]  # noqa: E731
            du = m[t][0] - m[t - 1][0]
            dt = m[t][1] - m[t - 1][1]
            turns.append({"step": st, "day": st // 24, "hour": st % 24, "margin_drop": round(dm), "our_cash": round(du), "their_cash": round(dt),
                          "our_orders": mk(us), "their_orders": mk(1 - us)})
        # products, days 20-29
        S = days.get(g, {})
        prod = {}
        for who, s in (("us", us), ("them", 1 - us)):
            for d in range(20, 30):
                row = S.get(s, {}).get(d, {})
                for p, u in (row.get("sells") or {}).items():
                    e = prod.setdefault(p, {"us_u": 0, "us_r": 0.0, "them_u": 0, "them_r": 0.0, "us_left": 0, "them_left": 0})
                    e[who + "_u"] += u
                    e[who + "_r"] += (row.get("sell_rev") or {}).get(p, 0.0)
            for p, u in ((S.get(s, {}).get(29, {}) or {}).get("shed") or {}).items():
                e = prod.setdefault(p, {"us_u": 0, "us_r": 0.0, "them_u": 0, "them_r": 0.0, "us_left": 0, "them_left": 0})
                e[who + "_left"] += u
        gaps = []
        for p, e in prod.items():
            rg = e["them_r"] - e["us_r"]
            up = e["us_r"] / e["us_u"] if e["us_u"] else 0.0
            tp = e["them_r"] / e["them_u"] if e["them_u"] else 0.0
            vol = (e["them_u"] - e["us_u"]) * (up or tp)       # revenue explained by the unit difference
            price = e["us_u"] * (tp - up) if e["us_u"] and e["them_u"] else 0.0   # by the price difference at our volume
            gaps.append({"product": p, "rev_gap": round(rg), "volume_part": round(vol), "price_part": round(price),
                         "us": [e["us_u"], round(up, 1)], "them": [e["them_u"], round(tp, 1)], "left": [e["us_left"], e["them_left"]]})
        gaps.sort(key=lambda x: -x["rev_gap"])
        top = gaps[0] if gaps else None
        pos = [x for x in gaps if x["rev_gap"] > 0]
        vol_total = sum(max(0, x["volume_part"]) for x in pos)
        price_total = sum(max(0, x["price_part"]) for x in pos)
        if -fin <= price_total and price_total >= vol_total:
            verdict = "sale-fixable (price gap on the same volume covers the deficit)"
        elif -fin <= vol_total + price_total:
            verdict = "plan-fixable (they sold MORE units: production / farm plan)" if vol_total > price_total else "sale+plan (both needed)"
        else:
            verdict = "out of reach late (deficit larger than all late product gaps: formed earlier or in costs)"
        rows.append({"game": g, "final": round(fin), "class": cls, "arc": arc, "deficit_opened_day": opened, "cause": top, "product_gaps": gaps[:6],
                     "price_total": round(price_total), "volume_total": round(vol_total), "verdict": verdict, "turns": turns})
    with open(os.path.join(a.out, f"{a.name}_rca.jsonl"), "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    # summary
    n = len(rows)
    cls = collections.Counter(r["class"] for r in rows)
    ver = collections.Counter(r["verdict"].split(" (")[0] for r in rows)
    cause = collections.Counter((r["cause"] or {}).get("product") for r in rows)
    typ = collections.Counter(("price" if (r["cause"] or {}).get("price_part", 0) >= (r["cause"] or {}).get("volume_part", 0) else "volume") for r in rows)
    L = [f"# Loss RCA: {a.name} ({n} lost games)\n",
         "## Where the game is lost\n", "| class | games |", "|---|---|"]
    names = {"A": "behind by day 10 (opening)", "B": "behind by day 20 (mid-game economy)", "C": "level until day 20, lost days 20-26", "D": "ahead on day 26, lost in the final days"}
    for c in "ABCD":
        L.append(f"| {names[c]} | {cls.get(c, 0)} |")
    L += ["", "## Verdict", "| verdict | games |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ver.most_common()]
    L += ["", "## Main cause product (largest late revenue gap)", "| product | games |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cause.most_common()]
    L += ["", f"Main-cause type: {dict(typ)} (volume = they sold more units; price = same units, better price)", ""]
    hours = collections.Counter()
    for r in rows:
        for t in r["turns"][:3]:
            hours[(t["day"] // 5 * 5, t["hour"])] += 1
    L += ["## When the decisive turns happen (top-3 drops per game, day bucket x hour)", ", ".join(f"d{d}+ h{h}: {c}" for (d, h), c in hours.most_common(15)), ""]
    L += ["## Representative games (largest loss per class) with the decisive turns", ""]
    for c in "ABCD":
        rs = sorted([r for r in rows if r["class"] == c], key=lambda r: r["final"])
        for r in rs[:2] + rs[-1:]:
            L.append(f"### {r['game']} (class {c}): final {r['final']:+d}; arc {r['arc']}; deficit opened day {r['deficit_opened_day']}")
            L.append(f"verdict: {r['verdict']}; price gap {r['price_total']}, volume gap {r['volume_total']}")
            for x in r["product_gaps"][:4]:
                L.append(f"- {x['product']}: gap ${x['rev_gap']:+d} (volume {x['volume_part']:+d}, price {x['price_part']:+d}); us {x['us'][0]}u @{x['us'][1]} | them {x['them'][0]}u @{x['them'][1]}; left {x['left']}")
            for t in r["turns"][:5]:
                L.append(f"  - step {t['step']} (d{t['day']} h{t['hour']}): margin {t['margin_drop']:+d} (us {t['our_cash']:+d}, them {t['their_cash']:+d}); ours {t['our_orders'][:4]} | theirs {t['their_orders'][:4]}")
            L.append("")
    open(os.path.join(a.out, f"{a.name}_rca.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
