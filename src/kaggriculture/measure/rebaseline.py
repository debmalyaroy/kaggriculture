"""B4.0 RE-BASELINE on the FAITHFUL serve harness (Workstream B4.0).

The operator's core correction: prior rankings were measured on a STATIC /
mis-ranking substrate and are unreliable. Now that the serve invocation layer
is fixed (S5/S3/S4, verified reactive-exact vs official -- see
`serve-fidelity-fix-2026-09-18`), this runs the TRUE head-to-head ranking of
the strong references and our recent agents so ship decisions gate against a
ranking we can trust.

Round-robin, BOTH seats, N seeds, scored by win_metric (the ladder currency:
win/draw/loss, NOT coin margin). Serve-only: the harness is now trusted, so we
do not pay the 12x official tax for a whole tournament. Paired McNemar vs the
top ~2500 reference is reported for each candidate.

    python -m kaggriculture.measure.rebaseline --seeds 6

Default field: the two LIVE ~2500 tapes (the BAR) + v56y (claimed 2072) +
v57 + v58 (claimed 463 on the ladder -- but that was the Kaggle FALLBACK, C2;
this self-contained tape build is what its embedded economy actually plays).
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import itertools
import json
import os

from kaggriculture.engine import serve_match as sm
from kaggriculture.measure import win_metric as wm

DEFAULT_FIELD = {
    "ref_v46_2517": ".local/panel/ref2500/ref_v46_chassis_2517.py",
    "ref_k0006_2494": ".local/panel/ref2500/ref_k0006_2494.py",
    "v56y": "agents/v56y_trackp.py",
    "v57": "agents/v57_trackp.py",
    "v58": "agents/v58_bandit.py",
}
BAR = "ref_v46_2517"  # the strong reference to paired-test everyone against


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def play_pair(name_a, path_a, name_b, path_b, seeds, srv):
    """Both seats x seeds. Returns (a_scores, b_scores) aligned per game and
    the per-game (a_bank, b_bank) list for margin inspection."""
    a_scores, b_scores, banks = [], [], []
    pa, pb = _abs(path_a), _abs(path_b)
    for seed in seeds:
        # seat arrangement 1: A=player0, B=player1
        ba, bb = sm.run_match(sm.load_agent(pa), sm.load_agent(pb), seed, srv)
        a_scores.append(wm.score(ba, bb)); b_scores.append(wm.score(bb, ba))
        banks.append((ba, bb))
        # seat arrangement 2: B=player0, A=player1 (swap seats, same seed)
        bb2, ba2 = sm.run_match(sm.load_agent(pb), sm.load_agent(pa), seed, srv)
        a_scores.append(wm.score(ba2, bb2)); b_scores.append(wm.score(bb2, ba2))
        banks.append((ba2, bb2))
    return a_scores, b_scores, banks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", nargs="*", default=None,
                    help="name=path pairs; default = the built-in field")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--out", default=".local/rebaseline_2026-09-18.json")
    args = ap.parse_args()

    field = dict(DEFAULT_FIELD)
    if args.agents:
        field = {}
        for tok in args.agents:
            k, _, v = tok.partition("=")
            field[k] = v
    # keep only existing files
    field = {k: v for k, v in field.items() if os.path.exists(_abs(v))}
    seeds = [args.seed0 + i for i in range(max(1, args.seeds))]
    names = list(field)

    totals = {n: [] for n in names}          # all game scores per agent
    h2h = {n: {m: None for m in names} for n in names}
    paired = {}

    srv = sm.Serve()
    try:
        for na, nb in itertools.combinations(names, 2):
            a_sc, b_sc, banks = play_pair(na, field[na], nb, field[nb], seeds, srv)
            totals[na] += a_sc
            totals[nb] += b_sc
            ea, eb = wm.expected_score(a_sc), wm.expected_score(b_sc)
            h2h[na][nb] = round(ea, 3)
            h2h[nb][na] = round(eb, 3)
            print(f"{na:>16} vs {nb:<16}  {ea:.3f} - {eb:.3f}  "
                  f"({len(a_sc)} games)", flush=True)
            if BAR in (na, nb) and na != nb:
                cand = nb if na == BAR else na
                cand_sc = b_sc if na == BAR else a_sc
                bar_sc = a_sc if na == BAR else b_sc
                try:
                    p = wm.paired_test(cand_sc, bar_sc)
                except Exception:  # noqa: BLE001
                    p = None
                paired[cand] = {"cand_score": round(wm.expected_score(cand_sc), 3),
                                "vs_bar": BAR, "paired": str(p)}
    finally:
        srv.close()

    ranking = sorted(names, key=lambda n: wm.expected_score(totals[n]), reverse=True)
    print("\n=== RE-BASELINE RANKING (faithful serve, both seats, "
          f"{len(seeds)} seeds) ===", flush=True)
    for n in ranking:
        print(f"  {wm.expected_score(totals[n]):.3f}  {n:>16}  "
              f"({len(totals[n])} games)", flush=True)
    print("\n=== vs the ~2500 BAR ({}), paired ===".format(BAR), flush=True)
    for n in ranking:
        if n in paired:
            print(f"  {n:>16}: {paired[n]}", flush=True)

    out = {
        "when": "2026-09-18", "substrate": "faithful serve (S5/S3/S4 fixed)",
        "seeds": seeds, "field": field, "bar": BAR,
        "ranking": [{"agent": n, "score": round(wm.expected_score(totals[n]), 4),
                     "games": len(totals[n])} for n in ranking],
        "head_to_head": h2h, "paired_vs_bar": paired,
    }
    os.makedirs(os.path.dirname(_abs(args.out)), exist_ok=True)
    with open(_abs(args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
