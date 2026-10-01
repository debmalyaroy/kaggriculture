"""Exact shared-opening detection via action-stream prefix hashes.

Two seats whose canonicalized action streams hash identically at turn N
played byte-equivalent openings for N straight turns -- an OBSERVATION of
copying/consensus, not a similarity threshold (technique from the public
Kaggriculture Episodes dataset discussion; implementation ours).

Two jobs:
  * CORPUS: hash every indexed route's prefixes at the checkpoint turns,
    cached by route id (routes are immutable, so the cache never staleness-
    checks; it only grows). ~13 ms/route on first build, delta afterwards.
  * RELEASE CHECK (wired into refresh_cycle.stage_build): every built agent
    records how exposed its OPENING is -- how many field routes and teams
    share its exact prefix at day 1/3/6/10. A widely-shared opening is a
    near-mirror magnet and a copyability signal; the number lands in the
    cycle log, the report and models/lab/opening_hashes.jsonl, so exposure
    is a tracked series from the next release onwards.

    python src/stream_hashes.py --build            # corpus cache (delta)
    python src/stream_hashes.py --check 92513718_s1
"""
from kaggriculture.paths import ROOT
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CACHE = os.path.join(ROOT, "models", "lab", "stream_hashes.json")
LOG = os.path.join(ROOT, "models", "lab", "opening_hashes.jsonl")
# Checkpoint turns: end of day 1, 2, 3 (first shop), 6 (second shop), 10.
CHECKPOINTS = (24, 48, 72, 144, 240)


def _canon_turn(turn):
    """Canonical serialization of one turn's actions. Sorting is deliberately
    NOT applied -- order is part of the behavior (sell-first matters)."""
    t = turn if isinstance(turn, dict) else {}
    return json.dumps({"f": t.get("farmer"), "h": t.get("hands"),
                       "m": t.get("market")}, separators=(",", ":"),
                      sort_keys=True)


def prefix_hashes(actions, checkpoints=CHECKPOINTS):
    """{turn: sha1-16} cumulative prefix hashes at each checkpoint."""
    h = hashlib.sha1()
    out = {}
    marks = set(checkpoints)
    last = max(checkpoints)
    for t, turn in enumerate(actions[:last]):
        h.update(_canon_turn(turn).encode("utf-8"))
        if (t + 1) in marks:
            out[t + 1] = h.hexdigest()[:16]
    return out


def load_cache():
    if os.path.exists(CACHE):
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except ValueError:
            pass
    return {}


def build_corpus(log=print):
    import kaggriculture.data.routes as R
    idx = R.load_index()["routes"]
    cache = load_cache()
    todo = [rid for rid in idx if rid not in cache]
    log(f"stream-hash corpus: {len(cache):,} cached, {len(todo):,} to hash")
    done = 0
    for rid in todo:
        try:
            cache[rid] = prefix_hashes(R.load_route(rid))
        except Exception:                                       # noqa: BLE001
            cache[rid] = {}
        done += 1
        if done % 2000 == 0:
            json.dump(cache, open(CACHE, "w", encoding="utf-8"))
            log(f"  {done:,}/{len(todo):,}")
    json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    return cache


def opening_exposure(actions, cache=None, idx=None):
    """How shared is this action stream's opening, per checkpoint?

    Returns {turn: {"routes": n, "teams": m}} counting OTHER field routes
    whose exact prefix hash matches.
    """
    if cache is None:
        cache = load_cache()
    if idx is None:
        import kaggriculture.data.routes as R
        idx = R.load_index()["routes"]
    mine = prefix_hashes(actions)
    out = {}
    for t, h in mine.items():
        routes_n, teams = 0, set()
        for rid, hh in cache.items():
            if hh.get(str(t), hh.get(t)) == h:
                routes_n += 1
                team = (idx.get(rid) or {}).get("team")
                if team and team != "?":
                    teams.add(team)
        out[t] = {"routes": routes_n, "teams": len(teams)}
    return out


def check_route(rid, log=print):
    import datetime as dt
    import kaggriculture.data.routes as R
    cache = build_corpus(log=lambda *_: None)     # delta build, cheap
    idx = R.load_index()["routes"]
    exposure = opening_exposure(R.load_route(rid), cache, idx)
    for t in sorted(exposure):
        e = exposure[t]
        log(f"  opening @turn {t:>3} (day {t // 24}): shared with "
            f"{e['routes']:,} route(s) / {e['teams']} team(s)")
    row = {"date": dt.date.today().isoformat(), "route": rid,
           "exposure": {str(k): v for k, v in exposure.items()}}
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return exposure


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", metavar="ROUTE_ID")
    args = ap.parse_args()
    if args.build:
        build_corpus()
        return 0
    if args.check:
        check_route(args.check)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
