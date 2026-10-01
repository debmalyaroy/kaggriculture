"""Harvest current episode ids from the leaderboard's top teams -- no browser.

The whole chain is the authenticated Kaggle Python API:

    competition_leaderboard_view(slug, page_size, page_token)  -> team_id per team
    competition_team_submissions(team_id)                      -> submission ids
    competition_list_episodes(submission_id)                   -> episodes (id,
                                                                  create_time,
                                                                  end_time, agents)

Filtered to recent episodes (default: last 36h) above a score floor, this is
the same-day foreign-route source that beats the archive's ~1-day lag, and it
is fully scriptable -- no Chrome, no scraping, no reverse-engineered internal
API. The ids print to stdout / --out for src/kaggriculture/data/sameday.py to replay + ingest.

    python -m kaggriculture.data.leaderboard_harvest --top 200 --hours 36 --out ids.json
    python -m kaggriculture.data.leaderboard_harvest --top 200 | python -m kaggriculture.data.sameday
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _api():
    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def top_teams(api, slug, top):
    import contextlib
    import io
    import re as _re
    teams, token = [], None
    while len(teams) < top:
        # kagglesdk returns a PLAIN LIST and only PRINTS the continuation
        # token to stdout ("Next Page Token = ..."), so every harvest was
        # silently capped at one page (100 teams) however large --top was
        # (found 2026-08-13: --top 200 walked exactly 100). Capture stdout
        # and parse the token out of it.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = api.competition_leaderboard_view(
                slug, page_size=min(100, top - len(teams)), page_token=token)
        if not rows:
            break
        for r in rows:
            tid = getattr(r, "team_id", None)
            if tid:
                try:
                    sc = float(getattr(r, "score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    sc = 0.0
                teams.append({"team_id": tid,
                              "team": getattr(r, "team_name", "?"),
                              "score": sc})
        m = _re.search(r"Next Page Token\s*=\s*(\S+)", buf.getvalue())
        token = m.group(1) if m else None
        if not token:
            break
    return teams[:top]


def _epoch(v):
    """Epoch seconds from whatever the SDK hands back -- ASSUMING UTC.

    The SDK returns NAIVE datetimes that are UTC wall-clock, and both
    `datetime.timestamp()` and `fromisoformat(...).timestamp()` interpret a
    naive value as LOCAL time. On an IST box that read every episode as 5.5 h
    older than it was, so the hourly harvest's 3 h window matched NOTHING --
    every sip from 2026-08-12 to 2026-08-14 returned 0 ids, silently, and the
    daily release inherited a full day's delta (the 36 h default window had
    been absorbing the shift, which is why narrowing the window is what broke
    it). Naive datetimes are pinned to UTC before conversion.
    """
    if v is None:
        return 0.0
    if isinstance(v, dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        return v.timestamp()
    for attr in ("seconds",):
        if hasattr(v, attr):
            try:
                return float(getattr(v, attr)() if callable(getattr(v, attr))
                             else getattr(v, attr))
            except Exception:                                      # noqa: BLE001
                pass
    try:
        return (dt.datetime.fromisoformat(str(v)[:19])
                .replace(tzinfo=dt.timezone.utc).timestamp())
    except Exception:                                              # noqa: BLE001
        return 0.0


def harvest(top=200, hours=36, min_score=2600.0, per_team_subs=2, verbose=True):
    api = _api()
    cutoff = time.time() - hours * 3600
    teams = top_teams(api, "kaggriculture", top)
    if verbose:
        print(f"[lb] {len(teams)} teams; window {hours}h; score floor "
              f"{min_score}", file=sys.stderr)
    ids, seen_sub = [], set()
    throttled = 0
    for i, t in enumerate(teams):
        if t["score"] and t["score"] < min_score:
            continue
        # The per-team listing endpoint 429s in bursts (~80 misses logged
        # 2026-08-12, i.e. whole teams silently skipped that sweep). Retry
        # with backoff before giving a team up, and stretch the inter-call
        # pause while the API is telling us to slow down.
        subs, exc_msg = None, ""
        for attempt in range(3):
            try:
                subs = api.competition_team_submissions(t["team_id"]) or []
                break
            except Exception as exc:                               # noqa: BLE001
                exc_msg = str(exc).splitlines()[0]
                if "429" in exc_msg and attempt < 2:
                    throttled += 4
                    time.sleep(4 * (attempt + 1))
                else:
                    break
        throttled = max(0, throttled - 1)      # pressure decays on success
        if subs is None:
            if verbose:
                print(f"  ! team {t['team']}: {exc_msg}", file=sys.stderr)
            continue
        subs = sorted(subs, key=lambda s: str(getattr(s, "date_submitted", "")),
                      reverse=True)[:per_team_subs]
        for s in subs:
            sid = getattr(s, "id", None)
            if not sid or sid in seen_sub:
                continue
            seen_sub.add(sid)
            try:
                eps = api.competition_list_episodes(sid) or []
            except Exception as exc:                               # noqa: BLE001
                # Never swallow this silently: an endpoint change here would
                # zero the whole harvest while looking exactly like a quiet
                # ladder (the silent-zero bug class, again).
                if verbose:
                    print(f"  ! episodes({sid}): "
                          f"{str(exc).splitlines()[0][:100]}", file=sys.stderr)
                continue
            for e in eps:
                end = _epoch(getattr(e, "end_time", None) or
                             getattr(e, "create_time", None))
                if end >= cutoff and getattr(e, "id", None):
                    ids.append(str(e.id))
        if verbose and (i + 1) % 25 == 0:
            print(f"  [lb] {i + 1}/{len(teams)} teams, {len(ids)} recent "
                  f"episode ids so far", file=sys.stderr)
        # base politeness pause, stretched while the API is throttling us
        time.sleep(0.05 if not throttled else min(2.0, 0.25 * throttled))
    ids = sorted(set(ids))
    if verbose:
        print(f"[lb] {len(ids)} unique recent episode ids from "
              f"{len(seen_sub)} submissions", file=sys.stderr)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--hours", type=float, default=36.0,
                    help="keep episodes ended within this many hours")
    ap.add_argument("--min-score", type=float, default=2600.0)
    ap.add_argument("--per-team-subs", type=int, default=2)
    ap.add_argument("--out", default=None,
                    help="write ids here (JSON list); default stdout")
    args = ap.parse_args()
    ids = harvest(top=args.top, hours=args.hours, min_score=args.min_score,
                  per_team_subs=args.per_team_subs)
    payload = json.dumps(ids)
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        open(out, "w", encoding="utf-8").write(payload)
        print(f"wrote {len(ids)} ids -> {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
