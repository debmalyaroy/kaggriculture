"""Verdict on two harness.py runs: paired, sign-tested, in WIN units.

`harness.py --json A.json` writes one row per (opponent, seed, seat) cell. Two
runs over the SAME cell list -- one for the compiled artefact, one for a
reference agent via `--ref` -- pair cell for cell, which is the only thing that
makes `win_metric.paired_test` (McNemar exact on the discordant pairs) valid.

    python src/trackp/compiled/verdict.py CAND.json --ref REF.json

Reports, per opponent and overall: score (draws 0.5), W-D-L, median banks, and
the paired test against the reference. Dollar deltas are printed for context
only -- the ladder pays wins, and `docs/history/plan-2800-2026-09-03.md` is explicit
that a mean margin cannot be converted into wins without the per-game
distribution.

Cells whose opponent did not finish, or banked <= its starting money, are DEAD
and are excluded from every number (harness.py's own guard; the count is
reported so an excluded cell can never pass silently).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


import kaggriculture.measure.win_metric as win_metric  # noqa: E402


def key(r):
    return (r["opp"], r["seed"], r["seat"])


def clean(rows):
    return [r for r in rows
            if r["status"] == "DONE" and r["opp_status"] == "DONE"
            and r["opp_bank"] > 3000.0]


def block(name, rows):
    if not rows:
        print(f"{name:<26} (no clean cells)")
        return
    pairs = [(r["bank"], r["opp_bank"]) for r in rows]
    w, d, lo, sc = win_metric.summarise(pairs)
    om = sorted(r["bank"] for r in rows)[len(rows) // 2]
    pm = sorted(r["opp_bank"] for r in rows)[len(rows) // 2]
    print(f"{name:<26} {len(rows):>5} {w:>3}-{d}-{lo:<3} {sc:>6.3f} "
          f"{om:>9,.0f} {pm:>9,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cand", help="harness --json output for the candidate")
    ap.add_argument("--ref", default=None,
                    help="harness --json output for the agent it would replace")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--ref-label", default="reference")
    args = ap.parse_args()

    with open(args.cand, encoding="utf-8") as fh:
        cand = json.load(fh)
    dead_c = len(cand) - len(clean(cand))
    cand = clean(cand)

    print(f"== {args.label} ==   ({dead_c} dead-opponent cells excluded)")
    print(f"{'opponent':<26} {'cells':>5} {'W-D-L':>9} {'score':>6} "
          f"{'own med':>9} {'opp med':>9}")
    for opp in sorted({r["opp"] for r in cand}):
        block(opp, [r for r in cand if r["opp"] == opp])
    block("ALL", cand)

    if not args.ref:
        return 0

    with open(args.ref, encoding="utf-8") as fh:
        ref = json.load(fh)
    dead_r = len(ref) - len(clean(ref))
    ref = clean(ref)
    print(f"\n== {args.ref_label} ==   ({dead_r} dead-opponent cells excluded)")
    print(f"{'opponent':<26} {'cells':>5} {'W-D-L':>9} {'score':>6} "
          f"{'own med':>9} {'opp med':>9}")
    for opp in sorted({r["opp"] for r in ref}):
        block(opp, [r for r in ref if r["opp"] == opp])
    block("ALL", ref)

    # ---- paired ---------------------------------------------------------
    # Only cells that survived cleaning on BOTH sides pair up; a cell dropped
    # on one side must be dropped on the other or the pairing is a lie.
    rmap = {key(r): r for r in ref}
    both = [(c, rmap[key(c)]) for c in cand if key(c) in rmap]
    print(f"\n== PAIRED  {args.label} vs {args.ref_label} =="
          f"   ({len(both)} of {len(cand)} candidate cells pair)")
    print(f"{'opponent':<26} {'pairs':>5} {'A':>6} {'B':>6} {'diff':>7} "
          f"{'A+':>3} {'B+':>3} {'p':>7}  verdict")
    groups = {opp: [(c, r) for c, r in both if c["opp"] == opp]
              for opp in sorted({c["opp"] for c, _ in both})}
    groups["ALL"] = both
    for name, g in groups.items():
        if not g:
            continue
        a = [win_metric.score(c["bank"], c["opp_bank"]) for c, _ in g]
        b = [win_metric.score(r["bank"], r["opp_bank"]) for _, r in g]
        t = win_metric.paired_test(a, b)
        verdict = ("A BETTER" if t["significant"] and t["score_diff"] > 0 else
                   "A WORSE" if t["significant"] and t["score_diff"] < 0 else
                   "not resolved")
        print(f"{name:<26} {t['n_pairs']:>5} {t['score_a']:>6.3f} "
              f"{t['score_b']:>6.3f} {t['score_diff']:>+7.3f} "
              f"{t['better_a']:>3} {t['better_b']:>3} {t['p_value']:>7.4f}  "
              f"{verdict}")
    n = len(both)
    if n:
        md = win_metric.min_detectable(n)
        print(f"\n  min detectable score difference at {n} pairs: {md:.3f} "
              f"(planning only -- a one-directional observed effect can be "
              f"significant below it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
