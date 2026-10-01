"""Per-product realized SELL price in REAL ladder games (us vs winner), by phase.

Uses data/ourgames index (seat, won) + staged replays. For each of our LOSS games,
walks both players' actions and the market prices at each step, accumulating
price-weighted units sold per product and per phase (early<240, mid 240-599,
late>=600). Shows exactly which products and phases we realize lower prices than
the winner -- the config target for the sell rails.

Usage: python src/analyze_real_sells.py --submission 56260608 [--wins]
"""
from kaggriculture.paths import ROOT
import argparse, glob, json, os
from collections import defaultdict



def phase(step):
    return "early" if step < 240 else ("mid" if step < 600 else "late")


def find_replay(ep):
    for p in (f"data/ourgames/_stage/{ep}/episode-{ep}-replay.json",
              f"data/ourgames/_stage/{ep}/{ep}.json",
              f".local/lossreplays/{ep}.json"):
        fp = os.path.join(ROOT, p)
        if os.path.exists(fp):
            return fp
    g = glob.glob(os.path.join(ROOT, f"data/ourgames/_stage/{ep}/*.json"))
    return g[0] if g else None


def sells_of(action):
    for m in (action.get("market") or []):
        if m and m[0] == "SELL":
            yield m[1], (int(m[2]) if len(m) > 2 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="56260608")
    ap.add_argument("--wins", action="store_true", help="analyze WINS instead of losses")
    a = ap.parse_args()
    G = json.load(open(os.path.join(ROOT, "data/ourgames/index.json")))["games"]
    games = [(ep, v) for ep, v in G.items() if str(v.get("submission")) == a.submission
             and (str(v.get("won")) == ("True" if a.wins else "False"))]
    # acc[who][phase][prod] = [price*units, units]
    acc = {s: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])) for s in ("us", "them")}
    ng = 0
    for ep, v in games:
        fp = find_replay(ep)
        if not fp:
            continue
        try:
            r = json.load(open(fp)); steps = r["steps"]
        except Exception:
            continue
        seat = int(v.get("seat", 0)); opp = 1 - seat
        for si, s in enumerate(steps):
            obs = s[seat].get("observation", {}) if isinstance(s, list) else {}
            prices = ((obs.get("market") or {}).get("prices")) or {}
            for who, sidx in (("us", seat), ("them", opp)):
                act = s[sidx].get("action") or {}
                if not isinstance(act, dict):
                    continue
                for prod, u in sells_of(act):
                    pr = float(prices.get(prod, 0) or 0)
                    acc[who][phase(si)][prod][0] += pr * max(u, 0)
                    acc[who][phase(si)][prod][1] += max(u, 0)
        ng += 1
    print(f"submission {a.submission}: {ng} {'WIN' if a.wins else 'LOSS'} games\n")
    for ph in ("early", "mid", "late"):
        prods = sorted(set(acc["us"][ph]) | set(acc["them"][ph]))
        if not prods:
            continue
        print(f"[{ph}]  {'product':11} {'us $/u':>7} {'us units':>9} | {'them $/u':>8} {'them u':>8}")
        for p in prods:
            uu = acc["us"][ph][p]; tt = acc["them"][ph][p]
            up = uu[0] / uu[1] if uu[1] else 0; tp = tt[0] / tt[1] if tt[1] else 0
            print(f"       {p:11} {up:7.1f} {uu[1]:9.0f} | {tp:8.1f} {tt[1]:8.0f}")
        print()


if __name__ == "__main__":
    main()
