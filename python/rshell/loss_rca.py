"""Root-cause every loss of the public-25 tournament (python/rshell/public25.py). Analysis only.

    PYTHONPATH=~/kaggriculture/src KAGG_BIN=... python python/rshell/loss_rca.py --cand DIR/main.py [--workers 12]

1. REPLAY each lost game on the same harness (capture_game: both action streams + money / sheds / market every
   step) and write it as a tape (OUT/tapes/<opp>__<seed>_<seat>.json; `seat` = OUR seat). A replay that does not
   reproduce the tournament's banks is flagged (non-deterministic opponent).
2. EXACT per-day ledgers with the v62 `tapedump` (per seat per day: farm, crew, land, sales units / avg price /
   revenue per product -- revenue exact by counterfactual -- spending, rival-on-our-squares share).
3. DIAGNOSE each loss: margin, the day the lead was lost for good, revenue gap per product, spending gap,
   production gap (units sold), realized price gap on shared products, copy share, unsold stock at the end.
   Class (first match):
     SPEND    we spent more on land / animals / crew / seed and did not earn it back (spend gap > margin)
     ECONOMY  they sold materially more units (production gap) of the deciding product
     PRICE    similar units, lower realized price on the deciding product (sale timing / race)
     ENDGAME  lead lost on the last day with stock left unsold or sold into a glut
   Writes OUT/rca.json (per loss + aggregates) and OUT/rca_summary.txt.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def replay(task):
    import contextlib
    import io
    from kaggriculture.bandit.gate import loss_forensics as LF
    cand, opp, opp_path, seed, seat, out = task
    ours = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ours = LF.load_pyagent(cand)
            recs, _, us, them = LF.capture_game(ours, opp_path, seed, seat)
        acts = []
        for r in recs:
            a = [r["our_act"], r["opp_act"]] if seat == 0 else [r["opp_act"], r["our_act"]]
            acts.append(a)
        key = f"{opp}__{seed}_{seat}"
        money = [(r["step"], r["our_money"], r["opp_money"]) for r in recs if r["step"] % 24 == 0]
        final_shed = {"ours": recs[-1]["our_shed"], "opp": recs[-1]["opp_shed"]}
        json.dump({"id": key, "seed": seed, "seat": seat, "actions": acts}, open(os.path.join(out, "tapes", key + ".json"), "w"))
        return {"key": key, "opp": opp, "seed": seed, "seat": seat, "us": us, "them": them, "money_by_day": money, "final_shed": final_shed}
    except Exception as exc:  # noqa: BLE001
        return {"key": f"{opp}__{seed}_{seat}", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        try:
            proc = (ours.__globals__.get("_BIN") or {}).get("proc") if ours else None
            if proc is not None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--games", default=os.path.join(RL, "data", "rshell", "public25", "games.jsonl"))
    ap.add_argument("--field", default=os.path.join(RL, ".local", "ext", "field25"))
    ap.add_argument("--tapedump", default=os.path.expanduser("~/krl/target-v62/release/tapedump"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(RL, "data", "rshell", "public25", "rca"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tapes"), exist_ok=True)
    games = [json.loads(l) for l in open(a.games)]
    losses = [g for g in games if g.get("gap", 0) < 0]
    world_of = {(g["seed"]): g["world"] for g in games}
    print(f"[rca] {len(losses)} losses of {len(games)} games", flush=True)
    # 1. replay
    reps = []
    tasks = [(os.path.abspath(a.cand), g["opp"], os.path.join(a.field, g["opp"] + ".py"), g["seed"], g["seat"], a.out) for g in losses]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(replay, t) for t in tasks]), 1):
            reps.append(f.result())
            if i % 100 == 0:
                print(f"[rca] replayed {i}/{len(tasks)}", flush=True)
    orig = {(g["opp"], g["seed"], g["seat"]): g for g in losses}
    for r in reps:
        if "error" in r:
            continue
        o = orig[(r["opp"], r["seed"], r["seat"])]
        r["reproduced"] = abs(r["us"] - o["us"]) < 0.5 and abs(r["them"] - o["them"]) < 0.5
    # 2. exact ledgers
    dump = subprocess.run([a.tapedump, "--tapes", os.path.join(a.out, "tapes"), "--threads", str(a.workers)], capture_output=True, text=True)
    open(os.path.join(a.out, "days.jsonl"), "w").write(dump.stdout)
    days = defaultdict(dict)
    for l in dump.stdout.splitlines():
        if l.startswith("{"):
            d = json.loads(l)
            days[d["key"]][(d["seat"], d["day"])] = d
    # 3. diagnose
    rows = []
    for r in reps:
        if "error" in r:
            rows.append(r)
            continue
        key, s = r["key"], r["seat"]
        dd = days.get(key, {})
        def tot(seat, fld):
            # SUM over the 30 days (a dict comprehension here once kept only the last day's value per product)
            c = Counter()
            for d in range(30):
                for k, v in ((dd.get((seat, d)) or {}).get(fld) or {}).items():
                    c[k] += v
            return c
        rev_us, rev_th = tot(s, "sell_rev"), tot(1 - s, "sell_rev")
        units_us, units_th = tot(s, "sells"), tot(1 - s, "sells")
        buys_us, buys_th = tot(s, "buys"), tot(1 - s, "buys")
        margin = r["them"] - r["us"]
        gaps = {p: rev_th.get(p, 0) - rev_us.get(p, 0) for p in PRODUCTS}
        prod, gap = max(gaps.items(), key=lambda kv: kv[1])
        du, dt_ = units_us.get(prod, 0), units_th.get(prod, 0)
        px_us = rev_us.get(prod, 0) / du if du else 0.0
        px_th = rev_th.get(prod, 0) / dt_ if dt_ else 0.0
        rev_total_gap = sum(rev_th.values()) - sum(rev_us.values())
        spend_gap = (sum(rev_us.values()) - r["us"]) - (sum(rev_th.values()) - r["them"])  # our net spend minus theirs
        lead_lost = None
        for step, mu, mo in r["money_by_day"]:
            if mu >= mo:
                lead_lost = step // 24
        last_day = dd.get((s, 29), {})
        copy = sum((dd.get((s, d), {}).get("copy") or 0) for d in range(30)) / 30
        unsold = {p: n for p, n in (r["final_shed"]["ours"] or {}).items() if n and p not in ("WHEAT", "FERTILIZER")}
        if spend_gap > margin and spend_gap > 0.5 * abs(rev_total_gap):
            cls = "SPEND"
        elif dt_ > 1.15 * du + 2:
            cls = "ECONOMY"
        elif (lead_lost or 0) >= 28 and (sum(unsold.values()) > 0 or px_us < px_th):
            cls = "ENDGAME"
        else:
            cls = "PRICE"
        rows.append({**{k: r[k] for k in ("key", "opp", "seed", "seat", "us", "them", "reproduced")}, "world": world_of.get(r["seed"]),
                     "margin": margin, "class": cls, "product": prod, "product_gap": gap, "units_us": du, "units_them": dt_,
                     "px_us": round(px_us, 1), "px_them": round(px_th, 1), "rev_gap": rev_total_gap, "spend_gap": spend_gap,
                     "lead_lost_day": lead_lost, "copy_share": round(copy, 3), "unsold_end": unsold,
                     "buys_us": dict(buys_us), "buys_them": dict(buys_th), "last_day_sells_us": last_day.get("sells")})
    ok = [x for x in rows if "class" in x]
    agg = {"losses": len(losses), "analysed": len(ok), "not_reproduced": sum(not x["reproduced"] for x in ok),
           "by_class": Counter(x["class"] for x in ok), "by_product": Counter(x["product"] for x in ok),
           "by_opp": Counter(x["opp"] for x in ok), "by_world": Counter(x["world"] for x in ok).most_common(15),
           "margin_under_500": sum(x["margin"] < 500 for x in ok), "margin_under_1000": sum(x["margin"] < 1000 for x in ok),
           "lead_lost_day": Counter(x["lead_lost_day"] for x in ok).most_common(10),
           "class_x_product": Counter(f"{x['class']}:{x['product']}" for x in ok).most_common(15)}
    json.dump({"aggregate": agg, "losses": rows}, open(os.path.join(a.out, "rca.json"), "w"), indent=1, default=str)
    print("[rca] " + json.dumps(agg, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
