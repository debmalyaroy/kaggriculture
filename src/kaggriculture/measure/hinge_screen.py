"""Hinge-seller screen: do CARROT/TOMATO/EGG-selling routes beat our base?

Diagnosis (2026-08-20, own-games forensics): our flagship sells ZERO units
of the 1.32.7 scarcity trio while opponents' egg volume tracks their wins.
This screen ranks the fresh mined pool by hinge-trio sell share (from the
obsfeat sidecars -- no replay loads) and plays the top sellers against the
elite panel on the Rust batch engine, with the incumbent base in the same
run for a paired read. Output: a ranked verdict in
models/factory/hinge_screen.json -- evidence for reserving tournament slots
for hinge-sellers, or for dropping the idea.

    python src/hinge_screen.py --top 10 --seeds 12
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

OBSDIR = os.path.join(ROOT, "data", "obsfeat")
# obs_features.flat layout: shop_mix occupies [2:11) in PRODUCTS order
PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
            "CARROT", "WHEAT", "FERTILIZER")
HINGE = {"EGG", "TOMATO", "CARROT"}
MIX0 = 2
H_IDX = [MIX0 + i for i, p in enumerate(PRODUCTS) if p in HINGE]


def hinge_share(rid):
    p = os.path.join(OBSDIR, rid + ".json")
    if not os.path.exists(p):
        return None
    try:
        v = json.load(open(p, encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(v, list) or len(v) < 74:
        return None
    return sum(v[i] for i in H_IDX)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--panel-size", type=int, default=6)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--min-bank", type=float, default=30000.0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    import kaggriculture.data.routes as R
    import kaggriculture.pipeline.sell_search as SS
    import kaggriculture.measure.win_metric as WM
    idx = R.load_index()

    base_id, base_src = SS.default_base()
    base_rec = idx["routes"][base_id]
    print(f"incumbent base: {base_id} (from {base_src})")

    # Rank fresh, winning-side routes by hinge-trio sell share x bank.
    rows = []
    for rid, rec in idx["routes"].items():
        if rec.get("window") != "fit" or float(rec.get("bank", 0)) < \
                args.min_bank:
            continue
        hs = hinge_share(rid)
        if hs is None or hs <= 0.02:
            continue
        rows.append((hs, float(rec.get("bank", 0)), rid, rec))
    rows.sort(key=lambda r: (-r[0] * r[1], r[2]))
    picks = rows[:args.top]
    if not picks:
        raise SystemExit("no hinge-selling routes in the fresh pool")
    print(f"{len(rows)} hinge-selling fresh routes; screening top "
          f"{len(picks)}:")
    for hs, bank, rid, rec in picks:
        print(f"  {rid:<32} hinge-share {hs:.3f}  bank {bank:>9,.0f}  "
              f"team {rec.get('team')}")

    os.makedirs(SS.WORK, exist_ok=True)
    panel = SS.pick_panel(idx, base_rec, args.panel_size)
    print(f"panel: {[r['id'] for r in panel]}")
    panel_tapes = [SS.write_tape(R.load_route(r["id"]),
                                 os.path.join(SS.WORK, f"hs_opp_{i}.tape"))
                   for i, r in enumerate(panel)]
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    cand_tapes = [SS.write_tape(R.load_route(base_id),
                                os.path.join(SS.WORK, "hs_base.tape"))]
    for i, (_, _, rid, _) in enumerate(picks):
        cand_tapes.append(SS.write_tape(R.load_route(rid),
                                        os.path.join(SS.WORK,
                                                     f"hs_c{i}.tape")))
    evals = SS.batch_eval(cand_tapes, panel_tapes, seeds, args.threads)
    base_ev = evals[0]

    print(f"\nincumbent: {base_ev['score']:.4f} on the panel "
          f"({base_ev['n']} games)")
    out_rows = []
    for i, (hs, bank, rid, rec) in enumerate(picks):
        ev = evals[i + 1]
        cells = sorted(set(ev["cells"]) & set(base_ev["cells"]))
        t = WM.paired_test([ev["cells"][k] for k in cells],
                           [base_ev["cells"][k] for k in cells])
        beats = t["score_diff"] > 0 and t["significant"]
        print(f"  {rid:<32} {ev['score']:.4f}  diff "
              f"{t['score_diff']:+.4f}  discordant "
              f"{t['better_a']}-{t['better_b']}  p={t['p_value']:.4f}"
              f"{'  ** SIGN-BETTER **' if beats else ''}")
        out_rows.append({"id": rid, "team": rec.get("team"),
                         "hinge_share": round(hs, 4), "bank": bank,
                         "score": ev["score"], "diff": t["score_diff"],
                         "p": t["p_value"], "sign_better": beats})
    n_better = sum(1 for r in out_rows if r["sign_better"])
    verdict = (f"{n_better}/{len(out_rows)} hinge-sellers sign-tested better "
               f"than the incumbent on the field panel")
    print(f"\nVERDICT: {verdict}")
    json.dump({"when": __import__("datetime").datetime.now()
               .isoformat(timespec="seconds"),
               "incumbent": {"id": base_id, "score": base_ev["score"]},
               "panel": [r["id"] for r in panel], "seeds": args.seeds,
               "rows": out_rows, "verdict": verdict},
              open(os.path.join(ROOT, "models", "factory",
                                "hinge_screen.json"), "w",
                   encoding="utf-8"), indent=1)
    print("wrote models/factory/hinge_screen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
