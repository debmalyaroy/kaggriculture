"""Our real ladder games: download ALL of a submission's episodes and analyse them.

    python -m kaggriculture.winplan.ourladder 56518820            # fetch + report
    python -m kaggriculture.winplan.ourladder 56518820 --no-fetch  # report only

* Episode list from Kaggle's EpisodeService (every episode, both agents'
  ratings and rewards, end time) -> data/winplan/ourladder/<sub>/episodes.json
* Every replay downloaded (kaggleusercontent direct, CLI fallback, the
  project's download quota respected) and kept gzipped next to it.
* Per game: seat, opponent team + submission + rating at the time, result,
  margin, the day the bank gap opened for good, per-item realized revenue for
  both seats (fills capped to the shed, priced at the step's quote), and the
  farm-layout similarity (the field's clone detector) on day 15.
* Report: totals, last 20, last 50, by opponent rating band, and every loss
  in detail (items where the rival out-earned us, broke day, clone or not).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import statistics as st
import urllib.request

from kaggriculture.paths import ROOT

OUT = os.path.join(ROOT, "data", "winplan", "ourladder")
ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
        "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
BANDS = [(0, 2000), (2000, 2200), (2200, 2400), (2400, 2600), (2600, 2800), (2800, 3000), (3000, 9999)]


def list_episodes(sub):
    from kaggriculture.pipeline.live_status import ENDPOINT, _auth_header
    import time
    req = urllib.request.Request(ENDPOINT, data=json.dumps({"submissionId": int(sub)}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": _auth_header()})
    for i in range(8):                       # 429s come in bursts: back off up to ~4 min in total
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                eps = json.loads(r.read()).get("episodes") or []
            if eps:
                return eps
        except Exception as exc:                                     # noqa: BLE001
            print(f"  EpisodeService: {exc}; retry {i + 1}/8")
        time.sleep(5 + 7 * i)
    raise RuntimeError(f"EpisodeService returned no episodes for {sub} (rate-limited or auth)")


def list_episodes_cli(sub):
    """Fallback when EpisodeService is throttled: ids + times from the CLI, no agents/ratings."""
    import csv
    import io
    import subprocess
    out = subprocess.run(["kaggle", "competitions", "episodes", str(sub), "-v"],
                         capture_output=True, text=True, timeout=300).stdout
    eps = [{"id": int(r["id"]), "endTime": r.get("endTime"), "createTime": r.get("createTime"), "agents": None}
           for r in csv.DictReader(io.StringIO(out)) if (r.get("id") or "").isdigit()]
    if not eps:
        raise RuntimeError(f"CLI listed no episodes for {sub}")
    return eps


def our_team(sub, eps):
    """The team name present in every replay (our own), from the downloaded files."""
    names = collections.Counter()
    for e in eps:
        p = os.path.join(OUT, str(sub), f"{e['id']}.json.gz")
        if os.path.exists(p):
            with gzip.open(p, "rb") as fh:
                t = (json.loads(fh.read()).get("info") or {}).get("TeamNames") or []
            names.update(set(t))
    return names.most_common(1)[0][0] if names else None


def fetch(sub, eps, log=print):
    import kaggriculture.data.sameday as sameday
    d = os.path.join(OUT, str(sub))
    os.makedirs(d, exist_ok=True)
    got = 0
    for e in eps:
        eid = str(e["id"])
        gz = os.path.join(d, f"{eid}.json.gz")
        if os.path.exists(gz):
            continue
        tmp = os.path.join(d, "_tmp")
        os.makedirs(tmp, exist_ok=True)
        try:
            path = sameday._grab_direct(eid, tmp)
        except Exception as exc:                                     # noqa: BLE001
            log(f"  ! {eid}: {str(exc)[:120]}")
            continue
        with open(path, "rb") as fh, gzip.open(gz, "wb", compresslevel=6) as out:
            out.write(fh.read())
        os.remove(path)
        got += 1
        if got % 10 == 0:
            log(f"  downloaded {got}")
    return got


def _similarity(farms, me):
    own, rival = farms[me], farms[1 - me]
    if own.get("unlocked_quadrants") != rival.get("unlocked_quadrants"):
        return 0.0
    m = t = 0
    for a, b in zip([x for row in own["tiles"] for x in row], [x for row in rival["tiles"] for x in row]):
        sa = (a.get("crop"), a.get("animal")) if isinstance(a, dict) else (None, None)
        sb = (b.get("crop"), b.get("animal")) if isinstance(b, dict) else (None, None)
        if sa != (None, None) or sb != (None, None):
            t += 1; m += sa == sb
    return round(m / t, 3) if t >= 8 else 0.0


def _revenue(steps, seat):
    """Realized revenue per item: step t's action is steps[t+1][seat].action;
    fills capped to the seat's shed at step t, priced at step t's quote."""
    rev = collections.Counter(); units = collections.Counter()
    for t in range(len(steps) - 1):
        obs = steps[t][seat].get("observation") or {}
        shared = steps[t][0].get("observation") or {}
        prices = (shared.get("market") or {}).get("prices") or {}
        shed = dict(((obs.get("private") or {}).get("shed")) or {})
        act = steps[t + 1][seat].get("action")
        if not isinstance(act, dict):
            continue
        for o in act.get("market") or []:
            if not (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in BASE):
                continue
            try:
                q = max(0, min(int(o[2]), int(shed.get(o[1], 0))))
            except (TypeError, ValueError):
                continue
            shed[o[1]] = shed.get(o[1], 0) - q
            units[o[1]] += q; rev[o[1]] += q * float(prices.get(o[1], 0) or 0)
    return rev, units


def analyse(sub, ep, team=None):
    path = os.path.join(OUT, str(sub), f"{ep['id']}.json.gz")
    with gzip.open(path, "rb") as fh:
        rp = json.loads(fh.read())
    info = rp.get("info") or {}
    teams = list(info.get("TeamNames") or ["?", "?"])
    agents = ep.get("agents") or []
    if agents:
        seat = next(i for i, a in enumerate(agents) if str(a.get("submissionId")) == str(sub))
        me, op = agents[seat], agents[1 - seat]
    else:                                   # CLI fallback: seat by team name, no ratings
        if teams.count(team) != 1:
            raise ValueError("self-play or our team not in the replay")
        seat = teams.index(team)
        me, op = {}, {}
    rewards = [float(r or 0) for r in (rp.get("rewards") or [0, 0])]
    steps = rp.get("steps") or []
    gap = []
    for t in range(0, len(steps), 24):
        farms = ((steps[t][0].get("observation") or {}).get("farms")) or []
        if len(farms) == 2:
            gap.append(float(farms[seat].get("money", 0)) - float(farms[1 - seat].get("money", 0)))
    margin = rewards[seat] - rewards[1 - seat]
    broke = None
    if margin < 0:
        broke = 0
        for d in range(len(gap) - 1, -1, -1):
            if gap[d] >= 0:
                broke = d + 1
                break
    sim = None
    if len(steps) > 360:
        farms = ((steps[360][0].get("observation") or {}).get("farms")) or []
        if len(farms) == 2:
            sim = _similarity(farms, seat)
    r_us, u_us = _revenue(steps, seat)
    r_op, u_op = _revenue(steps, 1 - seat)
    return {"id": ep["id"], "end": ep.get("endTime") or ep.get("createTime"), "seat": seat,
            "opp_team": teams[1 - seat] if len(teams) > 1 else "?", "opp_sub": op.get("submissionId"),
            "opp_rating": op.get("initialScore"), "our_rating_before": me.get("initialScore"),
            "our_rating_after": me.get("updatedScore"),
            "us": rewards[seat], "them": rewards[1 - seat], "margin": margin,
            "result": "W" if margin > 0 else "L" if margin < 0 else "T",
            "broke_day": broke, "layout_sim_d15": sim, "module_version": info.get("module_version") or rp.get("version"),
            "rev_us": {k: round(v) for k, v in r_us.items()}, "rev_op": {k: round(v) for k, v in r_op.items()},
            "units_us": dict(u_us), "units_op": dict(u_op)}


def _line(rows, label):
    w = sum(r["result"] == "W" for r in rows); l = sum(r["result"] == "L" for r in rows); t = len(rows) - w - l
    pct = 100 * (w + 0.5 * t) / len(rows) if rows else 0
    med = st.median([r["margin"] for r in rows]) if rows else 0
    rated = [r["opp_rating"] for r in rows if r["opp_rating"]]
    opp = st.mean(rated) if rated else 0
    return f"{label:10s} games {len(rows):4d}  W {w:3d}  L {l:3d}  T {t:2d}  win {pct:5.1f}%  median margin {med:+9,.0f}  mean opp rating {opp:7.1f}"


def report(sub, rows):
    rows = sorted(rows, key=lambda r: str(r["end"]))
    print(f"\n=== submission {sub}: {len(rows)} games (engine {sorted({str(r['module_version']) for r in rows})}) ===")
    print(_line(rows, "ALL"))
    print(_line(rows[-50:], "LAST 50"))
    print(_line(rows[-20:], "LAST 20"))
    if rows:
        print(f"rating: first game {rows[0]['our_rating_before']} -> latest {rows[-1]['our_rating_after']}")
    print("\n--- by opponent rating band ---")
    for lo, hi in BANDS:
        b = [r for r in rows if r["opp_rating"] and lo <= r["opp_rating"] < hi]
        if b:
            print(_line(b, f"{lo}-{hi if hi < 9999 else '+'}"))
    clones = [r for r in rows if (r["layout_sim_d15"] or 0) >= 0.95]
    print(f"\n--- clone mirrors (layout similarity >= 0.95 on day 15): {len(clones)}/{len(rows)} ---")
    if clones:
        print(_line(clones, "clones"))
        others = [r for r in rows if r not in clones]
        if others:
            print(_line(others, "non-clone"))
    losses = [r for r in rows if r["result"] == "L"]
    print(f"\n--- losses in detail ({len(losses)}) ---")
    print(f"{'end':19s} {'opp team':24s} {'opp rt':>7s} {'margin':>9s} {'broke':>5s} {'sim':>5s}  where the rival out-earned us (item: $diff)")
    agg = collections.Counter()
    for r in losses:
        diffs = {i: r["rev_op"].get(i, 0) - r["rev_us"].get(i, 0) for i in set(r["rev_op"]) | set(r["rev_us"])}
        agg.update({k: v for k, v in diffs.items()})
        top = sorted(((i, d) for i, d in diffs.items() if d > 0), key=lambda x: -x[1])[:3]
        print(f"{str(r['end'])[:19]:19s} {str(r['opp_team'])[:24]:24s} {r['opp_rating'] or 0:7.1f} {r['margin']:+9,.0f} "
              f"{str(r['broke_day']):>5s} {str(r['layout_sim_d15']):>5s}  " + ", ".join(f"{i}:{d:+,.0f}" for i, d in top))
    if losses:
        print("\nsum over losses, rival revenue minus ours, per item:")
        for i, d in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"   {i:11s} {d:+10,.0f}  ({d / len(losses):+8,.0f}/loss)")
        bd = [r["broke_day"] for r in losses if r["broke_day"] is not None]
        if bd:
            print(f"broke day (last day we were ahead): median {st.median(bd)}, distribution {sorted(bd)}")
        by_team = collections.Counter(r["opp_team"] for r in losses)
        print("losses by opponent team:", ", ".join(f"{t} x{n}" for t, n in by_team.most_common()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)
    d = os.path.join(OUT, str(a.sub)); os.makedirs(d, exist_ok=True)
    epath = os.path.join(d, "episodes.json")
    if a.no_fetch:
        eps = json.load(open(epath))
    else:
        try:
            eps = list_episodes(a.sub)
        except RuntimeError as exc:
            print(f"{exc} -> falling back to the CLI list (no ratings)")
            eps = list_episodes_cli(a.sub)
        json.dump(eps, open(epath, "w"))
        print(f"{len(eps)} episodes listed for {a.sub}")
        fetch(a.sub, eps)
    done = [e for e in eps if os.path.exists(os.path.join(d, f"{e['id']}.json.gz"))]
    team = our_team(a.sub, done)
    rows = []
    for e in done:
        # the validation self-play episode has our submission in both seats
        if e.get("agents") and len({str(x.get("submissionId")) for x in e["agents"]}) != 2:
            continue
        try:
            rows.append(analyse(a.sub, e, team))
        except ValueError:
            continue
        except Exception as exc:                                     # noqa: BLE001
            print(f"  ! analyse {e['id']}: {exc}")
    print(f"{len(done)}/{len(eps)} replays on disk; {len(rows)} ladder games (validation self-play excluded); team {team!r}")
    if any(r["opp_rating"] is None for r in rows):
        # EpisodeService throttled: fall back to each opponent team's CURRENT
        # leaderboard score (not its rating at game time; flagged in the rows)
        from kaggriculture.data import episodes as E
        lb = {r["team"]: float(r["score"]) for r in E.leaderboard_rows() if r.get("score") not in (None, "")}
        for r in rows:
            if r["opp_rating"] is None and r["opp_team"] in lb:
                r["opp_rating"], r["opp_rating_src"] = lb[r["opp_team"]], "leaderboard-now"
        print(f"opponent ratings from the current leaderboard for {sum(r.get('opp_rating_src') == 'leaderboard-now' for r in rows)} games")
    json.dump(rows, open(os.path.join(d, "games.json"), "w"), indent=1)
    report(a.sub, rows)


if __name__ == "__main__":
    main()
