"""Round-robin candidates against the public-agent gauntlet on `kagg serve`.

The gauntlet (data/gauntlet/) holds REACTIVE agents: the strongest public
notebook artifacts (V16-RC5, Kaito v27, Rayk C94/C95, Adaptive-R1) plus our
own proven builds. Unlike frozen tape panels, these respond to what happens,
so a paired win rate here measures play, not memorization of one recording.
Offline results still only RANK candidates -- the ladder VALIDATES
(.local/memory/offline-measure-doesnt-predict-ladder.md).

    python src/gauntlet.py --candidates A.py B.py --seeds 8
    python src/gauntlet.py --candidates A.py --opponents data/gauntlet/*.py

Every candidate plays every opponent from BOTH seats on the same seeds
(paired cells). Draws count 0.5. Per-opponent records are printed and the
whole result is written to models/gauntlet/ as JSON. Agents are re-exec'd
fresh for every game so cross-game module state cannot leak.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.engine.engine_check as engine_check  # noqa: E402
import kaggriculture.engine.serve_match as SM  # noqa: E402

# Verify the engine before believing any number (house rule). serve_match
# games run on the Rust engine, whose own equivalence gate is
# models/serve_equiv.json; this checks the official interpreter the gate
# compares against.
engine_check.require()

OUT_DIR = os.path.join(ROOT, "models", "gauntlet")
DEFAULT_OPP_DIR = os.path.join(ROOT, "data", "gauntlet")


def _fresh(path):
    return SM.load_agent(path)


def run(candidates, opponents, seeds, srv=None, verbose=True):
    own = srv is None
    if own:
        srv = SM.Serve()
    results = {}
    try:
        for cand in candidates:
            cname = os.path.basename(cand)
            per_opp = {}
            for opp in opponents:
                oname = os.path.basename(opp)
                if os.path.abspath(opp) == os.path.abspath(cand):
                    continue
                w = l = d = 0
                margins = []
                for seed in seeds:
                    # seat 0: candidate first
                    a, b = _fresh(cand), _fresh(opp)
                    m0, m1 = SM.run_match(a, b, seed, srv=srv)
                    if m0 > m1:
                        w += 1
                    elif m0 < m1:
                        l += 1
                    else:
                        d += 1
                    margins.append(m0 - m1)
                    # seat 1: candidate second, same seed (paired)
                    a, b = _fresh(opp), _fresh(cand)
                    n0, n1 = SM.run_match(a, b, seed, srv=srv)
                    if n1 > n0:
                        w += 1
                    elif n1 < n0:
                        l += 1
                    else:
                        d += 1
                    margins.append(n1 - n0)
                games = w + l + d
                score = (w + 0.5 * d) / games if games else 0.0
                per_opp[oname] = {
                    "w": w, "l": l, "d": d, "score": round(score, 4),
                    "mean_margin": round(sum(margins) / len(margins), 1)
                    if margins else 0.0,
                }
                if verbose:
                    print(f"  {cname:34s} vs {oname:26s} "
                          f"{w:3d}-{l:<3d}(d{d}) score {score:.3f} "
                          f"margin {per_opp[oname]['mean_margin']:+,.0f}")
            total_w = sum(v["w"] for v in per_opp.values())
            total_l = sum(v["l"] for v in per_opp.values())
            total_d = sum(v["d"] for v in per_opp.values())
            n = total_w + total_l + total_d
            overall = (total_w + 0.5 * total_d) / n if n else 0.0
            # A candidate is only as good as its worst strong matchup: losing
            # records are what the ladder punishes (win conversion, not margin).
            losing = sorted(k for k, v in per_opp.items() if v["score"] < 0.5)
            results[cname] = {
                "path": cand, "overall": round(overall, 4),
                "w": total_w, "l": total_l, "d": total_d,
                "losing_matchups": losing, "per_opponent": per_opp,
            }
            if verbose:
                print(f"{cname}: {total_w}-{total_l}-{total_d} "
                      f"overall {overall:.3f} losing vs: {losing or 'none'}")
    finally:
        if own:
            srv.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--seeds", default="8",
                    help="an int N (seeds 101..100+N) or a comma list")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    opponents = args.opponents or sorted(
        glob.glob(os.path.join(DEFAULT_OPP_DIR, "*.py")))
    if not opponents:
        raise SystemExit("no gauntlet opponents found; populate data/gauntlet/")
    if "," in args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(101, 101 + int(args.seeds)))

    print(f"gauntlet: {len(args.candidates)} candidate(s) x "
          f"{len(opponents)} opponent(s) x {len(seeds)} seed(s) x 2 seats")
    results = run(args.candidates, opponents, seeds)

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"gauntlet_{stamp}{args.tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"when": stamp, "seeds": seeds,
                   "opponents": [os.path.basename(o) for o in opponents],
                   "results": results}, fh, indent=1)
    print(f"\nwritten: {os.path.relpath(out, ROOT)}")
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["overall"])
    print("\nranking:")
    for name, r in ranked:
        print(f"  {r['overall']:.3f}  {name}  "
              f"({r['w']}-{r['l']}-{r['d']}; losing: "
              f"{', '.join(r['losing_matchups']) or 'none'})")


if __name__ == "__main__":
    main()
