"""Strict-future veto gate: a frozen candidate vs episodes recorded AFTER it.

Chronological validation in the style of Kaito Fukami's v27 protocol: freeze
the candidate, note the newest episode id in the route index, then -- once
newer episodes have been ingested (the hourly scrape) -- replay the candidate
against the routes the live field ACTUALLY played after the freeze. Because
those games did not exist when the candidate was chosen, they cannot have
been selected for, which is the failure mode that made tape-panel crowns
anti-predictive (.local/memory/offline-measure-doesnt-predict-ladder.md).

This is a VETO gate, not a crown: opponents are still frozen recordings, so
a PASS means "not overfit to yesterday's panel", never "will rate X".

    python src/strict_future.py --freeze                 # record the cutoff
    python src/strict_future.py --candidate A.py         # gate vs post-cutoff
    python src/strict_future.py --candidate A.py --cutoff 95803242 --min-team-score 2400
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import csv
import datetime as dt
import glob
import json
import os
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.engine.serve_match as SM  # noqa: E402


def _sign_test(w, l):
    """Two-sided exact binomial sign test on wins vs losses (draws excluded)."""
    import math
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)

OUT_DIR = os.path.join(ROOT, "models", "strict_future")
FREEZE_FILE = os.path.join(OUT_DIR, "freeze.json")

_TAPE_TMPL = '''import base64, copy, json, zlib
_T = json.loads(zlib.decompress(base64.b85decode("{payload}")))
def agent(obs, config=None):
    s = int(obs.get("step", 0) or 0)
    return copy.deepcopy(_T[min(s, len(_T) - 1)])
'''


def _epi(rec):
    try:
        return int(rec.get("episode") or 0)
    except (TypeError, ValueError):
        return 0


def newest_episode(idx=None):
    idx = idx or R.load_index()
    return max((_epi(r) for r in idx["routes"].values()), default=0)


def team_scores():
    """TeamName -> public score from the newest leaderboard CSV in data/lb."""
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "lb",
                                          "*publicleaderboard*.csv")),
                   key=os.path.getmtime)
    if not paths:
        return {}
    out = {}
    with open(paths[-1], encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["TeamName"]] = float(row["Score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def freeze():
    os.makedirs(OUT_DIR, exist_ok=True)
    cut = newest_episode()
    payload = {"cutoff_episode": cut,
               "when": dt.datetime.now().isoformat(timespec="seconds")}
    with open(FREEZE_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"frozen at episode {cut} -- gate later against episodes > {cut}")
    return payload


def gate(candidate, cutoff=None, min_team_score=2400.0, max_tapes=40,
         engine="1.32.7"):
    idx = R.load_index()
    if cutoff is None:
        if not os.path.exists(FREEZE_FILE):
            raise SystemExit("no freeze recorded; run --freeze first "
                             "or pass --cutoff")
        with open(FREEZE_FILE, encoding="utf-8") as fh:
            cutoff = int(json.load(fh)["cutoff_episode"])
    scores = team_scores()

    pool = []
    for rec in idx["routes"].values():
        if rec.get("engine") != engine or _epi(rec) <= cutoff:
            continue
        team = rec.get("team") or ""
        if scores and scores.get(team, 0.0) < min_team_score:
            continue
        if not os.path.exists(R.route_path(rec["id"])):
            continue
        pool.append(rec)
    # newest first, one per team so a single replay-happy team cannot
    # dominate the verdict
    pool.sort(key=_epi, reverse=True)
    seen, tapes = set(), []
    for rec in pool:
        if rec.get("team") in seen:
            continue
        seen.add(rec.get("team"))
        tapes.append(rec)
        if len(tapes) >= max_tapes:
            break
    if len(tapes) < 5:
        raise SystemExit(
            f"only {len(tapes)} post-cutoff tapes (episode > {cutoff}, "
            f"team score >= {min_team_score}); wait for the next ingest")

    cand_agent_src = open(candidate, encoding="utf-8").read()
    wins = losses = draws = 0
    rows = []
    srv = SM.Serve()
    tmpdir = tempfile.mkdtemp(prefix="strictfuture_")
    try:
        for rec in tapes:
            actions = R.load_route(rec["id"])
            payload = base64.b85encode(zlib.compress(
                json.dumps(actions, separators=(",", ":")).encode("utf-8"),
                9)).decode("ascii")
            tape_path = os.path.join(tmpdir, f"t_{rec['id']}.py")
            with open(tape_path, "w", encoding="utf-8") as fh:
                fh.write(_TAPE_TMPL.format(payload=payload))
            seed = _epi(rec) % 100000  # deterministic, tape-specific
            w = l = d = 0
            for seat in (0, 1):
                a = SM.load_agent(candidate if seat == 0 else tape_path)
                b = SM.load_agent(tape_path if seat == 0 else candidate)
                m0, m1 = SM.run_match(a, b, seed, srv=srv)
                mine, theirs = (m0, m1) if seat == 0 else (m1, m0)
                if mine > theirs:
                    w += 1
                elif mine < theirs:
                    l += 1
                else:
                    d += 1
            wins += w
            losses += l
            draws += d
            rows.append({"route": rec["id"], "team": rec.get("team"),
                         "team_score": scores.get(rec.get("team") or ""),
                         "w": w, "l": l, "d": d})
            print(f"  {rec['id']:16s} {str(rec.get('team'))[:22]:22s} "
                  f"{w}-{l}-{d}")
    finally:
        srv.close()

    p = _sign_test(wins, losses)
    score = (wins + 0.5 * draws) / max(1, wins + losses + draws)
    verdict = "PASS" if score >= 0.5 else "FAIL"
    result = {"candidate": os.path.basename(candidate),
              "cutoff_episode": cutoff, "tapes": len(tapes),
              "w": wins, "l": losses, "d": draws,
              "score": round(score, 4), "sign_p": round(float(p), 5),
              "verdict": verdict,
              "when": dt.datetime.now().isoformat(timespec="seconds"),
              "rows": rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"gate_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"\n{verdict}: {wins}-{losses}-{draws} over {len(tapes)} "
          f"post-cutoff tapes (score {score:.3f}, sign p {p:.4f})")
    print(f"written: {os.path.relpath(out, ROOT)}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--candidate")
    ap.add_argument("--cutoff", type=int, default=None)
    ap.add_argument("--min-team-score", type=float, default=2400.0)
    ap.add_argument("--max-tapes", type=int, default=40)
    args = ap.parse_args()
    if args.freeze:
        freeze()
        return
    if not args.candidate:
        raise SystemExit("pass --candidate or --freeze")
    gate(args.candidate, cutoff=args.cutoff,
         min_team_score=args.min_team_score, max_tapes=args.max_tapes)


if __name__ == "__main__":
    main()
