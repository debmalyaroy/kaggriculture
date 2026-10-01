"""How do top players trade WHEAT? (operator 29 Sep: "sell high, buy low; which step they do market manipulation")

    python python/wheat_study.py [--files 400] [--workers 2] [--out data/opp/wheat_study.json]

Slim corpus (crates/corpus/src/slim.rs): steps[t].a = actions that led to state t, steps[t].m = market at state t.
Per seat-game, split by the seat's rating after the game (>= 2500 = top, else field):
  flow      wheat BUY_PRODUCT / SELL units by (day, hour), and by step % 4 (town drain phase)
  same-step orders in one step containing both a wheat BUY and a wheat SELL: order sequence, quantities, step
  trips     each bought unit matched FIFO to the seat's later wheat sales within 48 steps: buy price vs sell price
  context   wheat price and market stock at their buys / sells vs the day's mean
Prices come from steps[t].m (the market the action at t saw, i.e. state t).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_RE = re.compile(r'"WHEAT":\s*(-?\d+)')


def split_steps(txt):
    """[(actions_text, inventory_text, prices_text)] per step, by plain string splitting (no backtracking)."""
    out = []
    for ch in txt.split('{"a":[')[1:]:
        j = ch.find('],"m":')
        if j < 0:
            continue
        m = ch[j + 6: j + 900]
        ii, pi = m.find('"inventory"'), m.find('"prices"')
        inv = m[ii: pi] if ii >= 0 and pi > ii else ""
        pr = m[pi: m.find("}", pi) + 1] if pi >= 0 else ""
        out.append((ch[:j], inv, pr))
    return out


def one(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    acc = {"top": defaultdict(float), "field": defaultdict(float), "ours": defaultdict(float)}
    same = {"top": defaultdict(int), "field": defaultdict(int), "ours": defaultdict(int)}
    trips = {"top": [], "field": [], "ours": []}
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g, columns=["rating_after_0", "rating_after_1", "team_name_0", "team_name_1", "slim"]).to_pylist():
            steps = split_steps(r["slim"] or "")
            if len(steps) < 700:
                continue
            px = []
            inv = []
            for a, i, p in steps:
                mp, mi = W_RE.search(p), W_RE.search(i)
                px.append(int(mp.group(1)) if mp else 0)
                inv.append(int(mi.group(1)) if mi else 10000)
            for seat in (0, 1):
                rt = r[f"rating_after_{seat}"] or 0
                cls = "ours" if str(r.get(f"team_name_{seat}")) == "Debmalya" else "top" if rt >= 2500 else "field"
                A = acc[cls]
                A["games"] += 1
                buys = []  # (step, qty, price)
                sells = []
                for t in range(1, len(steps)):
                    a = steps[t][0]
                    if "WHEAT" not in a:
                        continue
                    try:
                        pair = json.loads("[" + a + "]")
                    except Exception:  # noqa: BLE001
                        continue
                    act = pair[seat] if len(pair) > seat and isinstance(pair[seat], dict) else {}
                    st = t - 1                     # the step the orders were issued at
                    p0 = px[st] if st < len(px) else 0
                    seq = []
                    for o in act.get("market") or []:
                        if isinstance(o, list) and len(o) >= 3 and o[1] == "WHEAT" and o[0] in ("BUY_PRODUCT", "SELL"):
                            try:
                                q = int(o[2])
                            except Exception:  # noqa: BLE001
                                continue
                            if q <= 0:
                                continue
                            q = min(q, 500)
                            seq.append(("B" if o[0] == "BUY_PRODUCT" else "S", q))
                            d, h = st // 24, st % 24
                            if o[0] == "BUY_PRODUCT":
                                A[f"buy_d{d}"] += q; A[f"buy_h{h}"] += q; A[f"buy_ph{st % 4}"] += q
                                A["buy_units"] += q; A["buy_cost"] += q * p0
                                buys.append([st, q, p0])
                            else:
                                A[f"sell_d{d}"] += q; A[f"sell_h{h}"] += q; A[f"sell_ph{st % 4}"] += q
                                A["sell_units"] += q; A["sell_rev"] += q * p0
                                sells.append([st, q, p0])
                    kinds = {k for k, _ in seq}
                    if kinds == {"B", "S"}:
                        key = f"d{st // 24}h{st % 24}:" + ",".join(f"{k}{q}" for k, q in seq)
                        same[cls][key] += 1
                        A["same_step"] += 1
                # FIFO trips: a bought unit sold within 48 steps
                bq = [b[:] for b in buys]
                for s in sells:
                    need = s[1]
                    for b in bq:
                        if need <= 0:
                            break
                        if b[1] <= 0 or b[0] > s[0] or s[0] - b[0] > 48:
                            continue
                        m = min(need, b[1])
                        trips[cls].append((b[0], s[0], m, b[2], s[2]))
                        b[1] -= m
                        need -= m
    return acc, same, trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RL, "data", "opp", "wheat_study.json"))
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(RL, "data", "slim", "s1", "source=gm", "date=*", "*.parquet")))
    step = max(1, len(fs) // a.files)
    fs = fs[::step][: a.files]
    acc = {"top": defaultdict(float), "field": defaultdict(float), "ours": defaultdict(float)}
    same = {"top": defaultdict(int), "field": defaultdict(int), "ours": defaultdict(int)}
    trips = {"top": [], "field": [], "ours": []}
    with ProcessPoolExecutor(a.workers) as ex:
        for A, S, T in ex.map(one, fs):
            for c in ("top", "field", "ours"):
                for k, v in A[c].items():
                    acc[c][k] += v
                for k, v in S[c].items():
                    same[c][k] += v
                trips[c] += T[c]
    rep = {}
    for c in ("top", "field", "ours"):
        A = acc[c]
        n = max(1, A["games"])
        T = trips[c]
        tu = sum(t[2] for t in T)
        rep[c] = {
            "player_games": int(A["games"]),
            "per_game": {"buy_units": A["buy_units"] / n, "sell_units": A["sell_units"] / n,
                         "avg_buy_px": A["buy_cost"] / max(1, A["buy_units"]), "avg_sell_px": A["sell_rev"] / max(1, A["sell_units"]),
                         "same_step_buy_and_sell": A["same_step"] / n},
            "buy_by_day": [round(A[f"buy_d{d}"] / n, 2) for d in range(30)],
            "sell_by_day": [round(A[f"sell_d{d}"] / n, 2) for d in range(30)],
            "buy_by_hour": [round(A[f"buy_h{h}"] / n, 2) for h in range(24)],
            "sell_by_hour": [round(A[f"sell_h{h}"] / n, 2) for h in range(24)],
            "buy_by_drain_phase": [round(A[f"buy_ph{p}"] / n, 2) for p in range(4)],
            "sell_by_drain_phase": [round(A[f"sell_ph{p}"] / n, 2) for p in range(4)],
            "round_trips": {"units_per_game": tu / n, "avg_buy_px": sum(t[2] * t[3] for t in T) / max(1, tu),
                            "avg_sell_px": sum(t[2] * t[4] for t in T) / max(1, tu),
                            "avg_hold_steps": sum(t[2] * (t[1] - t[0]) for t in T) / max(1, tu),
                            "profit_per_game": sum(t[2] * (t[4] - t[3]) for t in T) / n},
            "top_same_step_patterns": sorted(same[c].items(), key=lambda x: -x[1])[:25],
        }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    for c in ("top", "field", "ours"):
        r = rep[c]
        print(f"\n== {c}: {r['player_games']} player-games")
        print("  per game:", {k: round(v, 2) for k, v in r["per_game"].items()})
        print("  round trips (bought, resold within 48 steps):", {k: round(v, 2) for k, v in r["round_trips"].items()})
        print("  buys by day :", r["buy_by_day"])
        print("  sells by day:", r["sell_by_day"])
        print("  buys by hour :", r["buy_by_hour"])
        print("  sells by hour:", r["sell_by_hour"])
        print("  by drain phase (step%4) buy/sell:", r["buy_by_drain_phase"], r["sell_by_drain_phase"])
        print("  top same-step buy+sell patterns:", r["top_same_step_patterns"][:12])


if __name__ == "__main__":
    main()
