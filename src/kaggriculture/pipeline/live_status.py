"""Live status of the ACTIVE pair: games, W/L, last-20/last-50, by band.

Quota-free: reads Kaggle's internal EpisodeService, which returns every
episode's final banks and post-game ratings as small JSON. No replay
downloads (those are budgeted -- a quota hold blanks the whole scrape).

The endpoint is normally browser-authenticated (XSRF + cookies), but it
also accepts the API token over HTTP Basic, so this runs headless:

    python src/live_status.py                  # the active pair
    python src/live_status.py --subs 55978480  # named submissions
    python src/live_status.py --json           # machine-readable

`ListEpisodesForSubmission` does NOT exist (404) -- the method is
`ListEpisodes` and the body is {"submissionId": <id>}.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import urllib.error
import urllib.request

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ENDPOINT = ("https://www.kaggle.com/api/i/"
            "competitions.EpisodeService/ListEpisodes")
STATE = os.path.join(ROOT, ".local", "live_status.json")
# Opponent-rating bands. The operator's targets: win ALL below 2000, 90%
# in 2000-2500, 70% above 2500 (docs/history/plan-2800-2026-09-03.md section 2).
BANDS = ((0, 2000, 1.00), (2000, 2500, 0.90), (2500, 9999, 0.70))


def _auth_header():
    p = os.path.expanduser("~/.kaggle/kaggle.json")
    c = json.load(open(p, encoding="utf-8"))
    tok = f"{c['username']}:{c['key']}".encode()
    return "Basic " + base64.b64encode(tok).decode()


def episodes(submission_id, timeout=90):
    """[{my, op, my_rating, op_rating_pre, end}] newest first."""
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps({"submissionId": int(submission_id)}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": _auth_header()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    rows = []
    for e in payload.get("episodes") or []:
        agents = e.get("agents") or []
        me = next((a for a in agents
                   if str(a.get("submissionId")) == str(submission_id)), None)
        op = next((a for a in agents
                   if str(a.get("submissionId")) != str(submission_id)), None)
        if not me or not op or me.get("reward") is None or op.get("reward") is None:
            continue                      # still running, or an errored episode
        rows.append({"my": float(me["reward"]), "op": float(op["reward"]),
                     "my_rating": me.get("updatedScore"),
                     "op_rating_pre": op.get("initialScore"),
                     "episode": e.get("id"),
                     "my_seat": agents.index(me),
                     "end": e.get("endTime") or e.get("createTime")})
    rows.sort(key=lambda r: str(r["end"] or ""), reverse=True)
    return rows


def active_pair():
    """[(ref, description)] for the 2 submissions that are currently active."""
    p = subprocess.run(["kaggle", "competitions", "submissions",
                        "kaggriculture", "-v"], capture_output=True, text=True,
                       timeout=300, encoding="utf-8", errors="replace")
    out, seen = [], 0
    for ln in (p.stdout or "").splitlines():
        ref = ln.split(",", 1)[0].strip()
        if ref.isdigit():
            desc = ln.split(",")[3] if ln.count(",") >= 3 else ""
            out.append((ref, desc.strip('"')[:70]))
            seen += 1
            if seen == 2:                 # only the latest 2 stay active
                break
    return out


def _tally(rows):
    w = sum(1 for r in rows if r["my"] > r["op"])
    lo = sum(1 for r in rows if r["my"] < r["op"])
    d = len(rows) - w - lo
    return w, lo, d


def _pct(w, n):
    return f"{100.0 * w / n:5.1f}%" if n else "    --"


def summarise(rows):
    out = {"n": len(rows)}
    if not rows:
        return out
    w, l, d = _tally(rows)
    out.update(games=len(rows), w=w, l=l, d=d,
               rating=rows[0].get("my_rating"),
               peak=max((r["my_rating"] for r in rows
                         if r["my_rating"] is not None), default=None),
               med_my=statistics.median(r["my"] for r in rows),
               med_op=statistics.median(r["op"] for r in rows),
               last=rows[0]["end"])
    for k, sl in (("last20", rows[:20]), ("last50", rows[:50]),
                  ("first10", rows[-10:])):
        ww, ll, dd = _tally(sl)
        out[k] = {"n": len(sl), "w": ww, "l": ll, "d": dd,
                  "med_my": statistics.median(r["my"] for r in sl),
                  "med_op": statistics.median(r["op"] for r in sl)}
    bands = {}
    for lo, hi, bar in BANDS:
        sel = [r for r in rows if r["op_rating_pre"] is not None
               and lo <= float(r["op_rating_pre"]) < hi]
        ww, ll, dd = _tally(sel)
        bands[f"{lo}-{hi}"] = {"n": len(sel), "w": ww, "l": ll, "bar": bar,
                               "pass": (ww / len(sel) >= bar) if sel else None}
    out["bands"] = bands
    # UPSET losses -- to an opponent rated BELOW us at match time. The
    # operator's sustain condition (2026-09-05): never lose downward; these
    # are the rating killers (11 of v44.1's first 24 losses were upsets).
    out["upsets"] = sum(
        1 for r in rows
        if r["my"] is not None and r["op"] is not None
        and r["my"] < r["op"]
        and r.get("my_rating") is not None
        and r.get("op_rating_pre") is not None
        and float(r["op_rating_pre"]) < float(r["my_rating"]))
    # Own-collapse vs out-banked: a loss under 75k is our economy failing;
    # a loss at 90k+ is the opponent simply banking more.
    losses = [r for r in rows if r["my"] < r["op"]]
    out["loss_kinds"] = {
        "collapse_lt75k": sum(1 for r in losses if r["my"] < 75000),
        "outbanked_ge90k": sum(1 for r in losses if r["my"] >= 90000),
        "total": len(losses)}
    return out


def render(pair):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"LIVE PAIR STATUS  {now} IST", "=" * 72]
    for ref, desc, s in pair:
        if not s.get("n"):
            lines += [f"\n{ref}  {desc}", "  no completed episodes yet"]
            continue
        rate = s.get("rating")
        lines += [f"\n{ref}  {desc}",
                  f"  rating {rate:.0f} (peak {s['peak']:.0f})   "
                  f"TOTAL {s['w']}-{s['l']}"
                  + (f"-{s['d']}" if s['d'] else "")
                  + f" of {s['games']}  {_pct(s['w'], s['games'])}"]
        for k, lbl in (("last20", "last 20"), ("last50", "last 50"),
                       ("first10", "first 10")):
            b = s[k]
            lines.append(f"  {lbl:<8} {b['w']}-{b['l']}"
                         + (f"-{b['d']}" if b['d'] else "")
                         + f"  {_pct(b['w'], b['n'])}"
                         f"   bank {b['med_my']:>8,.0f} vs {b['med_op']:>8,.0f}")
        lines.append("  by opponent rating band (operator targets):")
        for band, b in s["bands"].items():
            if not b["n"]:
                continue
            mark = "" if b["pass"] is None else (" PASS" if b["pass"] else " MISS")
            lines.append(f"    {band:<10} {b['w']}-{b['l']}  "
                         f"{_pct(b['w'], b['n'])}  (target "
                         f"{100 * b['bar']:.0f}%){mark}")
        lk = s["loss_kinds"]
        if lk["total"]:
            lines.append(f"  losses: {lk['total']} total, "
                         f"{lk['collapse_lt75k']} own-collapse (<75k), "
                         f"{lk['outbanked_ge90k']} out-banked (>=90k)")
        ups = s.get("upsets")
        if ups is not None:
            flag = "  <-- SUSTAIN FAIL (target 0)" if ups else "  (target 0) OK"
            lines.append(f"  UPSET losses (to lower-rated opponents): "
                         f"{ups}{flag}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subs", nargs="*", default=None,
                    help="submission ids (default: the active pair)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.subs:
        targets = [(s, "") for s in args.subs]
    else:
        targets = active_pair()

    pair = []
    for ref, desc in targets:
        try:
            rows = episodes(ref)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            pair.append((ref, desc, {"n": 0, "error": str(exc)}))
            continue
        pair.append((ref, desc, summarise(rows)))

    if args.json:
        print(json.dumps([{"ref": r, "desc": d, **s} for r, d, s in pair],
                         indent=1, default=str))
    else:
        print(render(pair))
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump({"at": dt.datetime.now().isoformat(),
               "pair": [{"ref": r, "desc": d, **s} for r, d, s in pair]},
              open(STATE, "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
