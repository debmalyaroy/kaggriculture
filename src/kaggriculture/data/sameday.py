"""Same-day episode scrape: routes fresher than the archive.

The daily episode datasets lag the ladder by ~2 days. But
`kaggle competitions replay <id>` fetches ANY episode immediately (verified
on our own games and on foreign episodes). The missing piece is a source of
*current* episode ids for teams other than us -- the public archive does not
carry them until it is published.

This tool ingests episode ids harvested from the leaderboard's episode
viewer (a JSON list produced by tools/harvest_episode_ids via the browser,
or any newline/JSON list on stdin/--ids-file), replays each within a byte
budget, extracts routes + obs-features, and folds them into the index with
today's date -- so the crown tournament can select on data hours old rather
than days.

    python -m kaggriculture.data.sameday --ids-file data/sameday/ids.json --max-gb 6
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402

WORK = os.path.join(ROOT, "data", "sameday")
STAGE = os.path.join(WORK, "_stage")
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def harvest_own(subs):
    """Same-day foreign episode ids for free: the episodes our own active
    submissions played today. The opponent seat of each is a current route,
    hours old, with no dependence on the archive's 2-day lag."""
    import kaggriculture.data.episodes as E
    ids = []
    for sub in subs:
        try:
            ids += E._own_episode_ids(str(sub), verbose=False)
        except Exception:                                          # noqa: BLE001
            continue
    return sorted(set(ids), reverse=True)


def load_ids(args):
    if args.harvest_own:
        subs = args.harvest_own.split(",")
        ids = harvest_own(subs)
        print(f"harvested {len(ids)} episode ids from submissions {subs}")
        return ids
    if args.ids_file:
        raw = open(args.ids_file, encoding="utf-8").read()
    else:
        raw = sys.stdin.read()
    raw = raw.strip()
    try:
        ids = json.loads(raw)
        if isinstance(ids, dict):
            ids = ids.get("ids") or list(ids.values())
    except Exception:                                              # noqa: BLE001
        ids = [x for x in raw.replace(",", "\n").split() if x.strip().isdigit()]
    return [str(i) for i in ids if str(i).strip().isdigit()]


# ---- replay-quota ledger (2026-08-16) --------------------------------------
# The replay endpoints throttle on a ROLLING window shared by every fetcher we
# run. The nightly pattern that broke two mornings in a row: the 02:xx archive
# mine burned the whole window, the 04:30 release found its loss-tape fetches
# 429'd, and the recovery rerun then fetched for hours on a dead quota. So the
# shared chokepoint (this function -- sameday, ourgames, routes.py and the
# cycle's tape fetches all come through here) enforces two rules for
# NON-release callers:
#   * a pre-release QUIET WINDOW (03:00-06:00 local): no replay downloads at
#     all -- that headroom belongs to the 04:30 release's loss tapes;
#   * a rolling 24h budget (_QUOTA_CAP downloads) so overnight bulk jobs can
#     never exhaust the window the morning needs.
# The release itself (KAGG_RELEASE=1, set by daily_release) bypasses both but
# still records its downloads. A hold raises RuntimeError("QUOTA-HOLD ...");
# callers with a CLI fallback must NOT fall through on that prefix -- the CLI
# burns the same quota slower.
_QUOTA_LEDGER = os.path.join(ROOT, "data", "sameday", "quota_ledger.json")
_QUOTA_CAP = int(os.environ.get("KAGG_QUOTA_CAP", "2000"))
_QUIET_HOURS = (3, 6)          # local [start, end) -- release headroom
_QUOTA_LOCK = None

# ---- 429-storm circuit breaker (2026-08-18) ---------------------------------
# The quota ledger protects OUR budget, but it cannot see a server-side 429
# storm: on 2026-08-17 Kaggle throttled the episode endpoints for 30+ hours
# and every hourly run still fired ~2,000 listing/fetch attempts into the
# wall -- 48k requests/day that plausibly PERPETUATE the throttle. The
# breaker: a run that fails _BREAKER_TRIP grabs in a row with nothing kept
# gives up and writes a cooldown marker; while the marker is fresh (<55 min,
# i.e. the next hourly runs) a run probes with _BREAKER_PROBE attempts only.
# The first successful download clears the marker and full width returns.
# The release (KAGG_RELEASE=1) is never probe-capped -- it has its own time
# budget and its fetches are the ones the quota exists to protect.
_BREAKER_TRIP = 40
_BREAKER_PROBE = 25
_COOLDOWN = os.path.join(ROOT, "data", "sameday", "fetch_cooldown.json")


def _cooldown_active(max_age_s=3300):
    """A fresh trip marker => this run should only probe. Returns age or None."""
    try:
        age = time.time() - float(
            json.load(open(_COOLDOWN, encoding="utf-8"))["tripped"])
    except (OSError, ValueError, KeyError):
        return None
    return f"tripped {age / 60:.0f} min ago" if 0 <= age < max_age_s else None


def _cooldown_trip(reason):
    try:
        with open(_COOLDOWN, "w", encoding="utf-8") as fh:
            json.dump({"tripped": time.time(), "reason": reason}, fh)
    except OSError:
        pass


def _cooldown_clear():
    try:
        os.remove(_COOLDOWN)
    except OSError:
        pass


def _quota_allow():
    """Gate one replay download; raise RuntimeError('QUOTA-HOLD ...') to hold.

    CHECK-ONLY (fixed 2026-08-16 evening): the first version recorded the
    ledger event HERE, before the download -- so an afternoon of real Kaggle
    429s burned the whole 2,000 budget on failed attempts and the guard then
    locked out fetching even after Kaggle's own quota recovered ("2000
    fetches at 123/s, 0.00 GB streamed"). The ledger must measure DOWNLOADS,
    not attempts: `_quota_record()` is called by `_grab_direct` only after
    the bytes are on disk.
    """
    release = os.environ.get("KAGG_RELEASE") == "1"
    now = time.time()
    if not release:
        hour = time.localtime(now).tm_hour
        if _QUIET_HOURS[0] <= hour < _QUIET_HOURS[1]:
            raise RuntimeError(
                "QUOTA-HOLD pre-release quiet window "
                f"({_QUIET_HOURS[0]:02d}:00-{_QUIET_HOURS[1]:02d}:00): replay "
                "quota is reserved for the 04:30 release")
        events = _quota_events(now)
        if len(events) >= _QUOTA_CAP:
            raise RuntimeError(
                f"QUOTA-HOLD rolling 24h budget spent ({len(events)}/"
                f"{_QUOTA_CAP} downloads): leaving headroom for the release")


def _quota_lock():
    global _QUOTA_LOCK
    if _QUOTA_LOCK is None:
        import threading
        _QUOTA_LOCK = threading.Lock()
    return _QUOTA_LOCK


def _quota_events(now=None):
    now = now or time.time()
    try:
        with open(_QUOTA_LEDGER, encoding="utf-8") as fh:
            events = json.load(fh).get("events", [])
    except (OSError, ValueError):
        events = []
    return [t for t in events if now - float(t) < 24 * 3600]


def _quota_record():
    """One SUCCESSFUL replay download; every fetch path records here."""
    with _quota_lock():
        now = time.time()
        events = _quota_events(now)
        events.append(now)
        tmp = _QUOTA_LEDGER + ".tmp"
        os.makedirs(os.path.dirname(_QUOTA_LEDGER), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"events": events}, fh)
        os.replace(tmp, _QUOTA_LEDGER)


def _grab_direct(eid, dest):
    """Fetch the replay straight from kaggleusercontent.

    Measured 2026-08-12: `kaggle competitions replay` served ~5 episodes in 30
    minutes and 429s readily once hit (the same endpoint that rate-limited the
    ourgames mine), while this URL returned a 34 MB replay in seconds. Same
    bytes, no CLI, no zip step. Kept as the primary path with the CLI as
    fallback so a change on Kaggle's side degrades instead of breaking.
    """
    import gzip
    import io
    import urllib.request
    _quota_allow()
    url = f"https://www.kaggleusercontent.com/episodes/{eid}.json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
        clen = resp.headers.get("Content-Length")
        if clen and len(raw) != int(clen):
            raise RuntimeError(
                f"truncated download {eid}: {len(raw)}/{clen} bytes")
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    # A replay is one JSON object; a mid-stream cutoff (the 13 legacy files
    # all died at byte 15,990,784) leaves a tail that cannot end in '}'.
    # Refuse it here so retry/CLI-fallback handles it, instead of a corrupt
    # .json poisoning every future ingest.
    if not raw.rstrip().endswith(b"}"):
        raise RuntimeError(f"truncated replay body {eid} ({len(raw)} bytes)")
    path = os.path.join(dest, f"{eid}.json")
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, path)           # atomic: never a half-written .json
    _quota_record()                 # count DOWNLOADS, never attempts
    return path


def grab(eid):
    dest = os.path.join(STAGE, str(eid))
    os.makedirs(dest, exist_ok=True)
    hits = [f for f in os.listdir(dest) if f.endswith(".json")]
    if not hits:
        try:
            path = _grab_direct(eid, dest)
            return eid, path, os.path.getsize(path)
        except Exception as exc:                                   # noqa: BLE001
            direct_err = f"{type(exc).__name__}: {str(exc)[:60]}"
            if str(exc).startswith("QUOTA-HOLD"):
                # The hold is a BUDGET decision, not a transport failure --
                # the CLI spends the same quota slower. Skip it entirely.
                return eid, None, direct_err
        # TIMEOUT IS LOAD-BEARING (2026-08-14): this CLI is the endpoint that
        # hangs silently when throttled. Without it, one stuck child blocked
        # one pool thread, the executor's exit waited on that thread forever,
        # and a single zombie instance silently ate SIX consecutive hourly
        # triggers (16:35 -> 23:35) while looking alive.
        try:
            r = subprocess.run(["kaggle", "competitions", "replay", str(eid),
                                "-p", dest], capture_output=True, env=ENV,
                               timeout=300)
        except subprocess.TimeoutExpired:
            return eid, None, f"direct[{direct_err}] cli[timeout 300s]"
        if r.returncode != 0:
            return eid, None, (f"direct[{direct_err}] cli["
                               + (r.stderr or b"").decode("utf-8", "replace")[:60]
                               + "]")
        for f in os.listdir(dest):
            if f.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(dest, f)) as z:
                    z.extractall(dest)
        hits = [f for f in os.listdir(dest) if f.endswith(".json")]
    path = os.path.join(dest, hits[0]) if hits else None
    size = os.path.getsize(path) if path else 0
    return eid, path, size


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--harvest-own", default=None,
                    help="comma-separated submission refs; harvest the "
                         "episodes they played today (opponent seats = "
                         "same-day foreign routes). Zero-browser source.")
    ap.add_argument("--max-gb", type=float, default=6.0)
    ap.add_argument("--jobs", type=int, default=6,
                    help="STARTING fetch width; adapts upward on "
                         "success and halves on throttling")
    ap.add_argument("--max-jobs", type=int, default=24,
                    help="ceiling for the adaptive fetch width")
    ap.add_argument("--min-score", type=float, default=2600.0,
                    help="skip episodes whose avg agent score is below this")
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    os.makedirs(STAGE, exist_ok=True)
    ids = load_ids(args)
    idx = R.load_index()
    todo = [e for e in ids if e not in idx["routes"]
            and f"{e}_s0" not in idx["routes"]][:2000]
    print(f"{len(ids)} ids in, {len(todo)} not yet in the index")
    if not todo:
        return 0
    cd = _cooldown_active()
    if cd and os.environ.get("KAGG_RELEASE") != "1":
        todo = todo[:_BREAKER_PROBE]
        print(f"fetch cooldown active ({cd}) -- probing with "
              f"{len(todo)} attempt(s) only", flush=True)

    budget = args.max_gb * 1e9
    spent = kept = streak = 0
    # ADAPTIVE CONCURRENCY (src/fetch_pool.py). Fetching is latency-bound --
    # 1.30 MB on the wire in ~2.7-3.1 s -- so width is the whole lever, and it
    # costs no CPU (which matters: the engine side is capped at 3 workers). A
    # fixed high width is how you get rate-limited, and a 429 storm already
    # cost a release once, so the width ramps on success and halves on a burst
    # of throttle signals. `--jobs` is now the STARTING width, `--max-jobs` the
    # ceiling.
    import kaggriculture.data.fetch_pool as fetch_pool
    apool = fetch_pool.AdaptivePool(start=args.jobs,
                                    ceiling=max(args.jobs, args.max_jobs))

    def _grab_tracked(e):
        apool.acquire()
        try:
            out = grab(e)
            apool.report(ok=bool(out and out[1]),
                         error=None if (out and out[1]) else "empty result")
            return out
        except Exception as exc:                                   # noqa: BLE001
            apool.report(ok=False, error=exc)
            raise
        finally:
            apool.release()

    with ThreadPoolExecutor(max_workers=max(args.jobs, args.max_jobs)) as pool:
        futs = {pool.submit(_grab_tracked, e): e for e in todo}
        for fut in as_completed(futs):
            if spent >= budget:
                # Every id was queued up front, and the executor's __exit__
                # waits for the whole queue -- so simply breaking kept
                # downloading all 2,000 episodes and threw the bytes away
                # (measured 2026-08-12: 20 GB of useful data, then GBs more
                # discarded, replays piling up in _stage). Cancel what has not
                # started so --max-gb is a real cap, not just a stop-ingesting
                # line.
                for pending in futs:
                    pending.cancel()
                print(f"budget {args.max_gb} GB reached -- cancelled pending "
                      f"downloads", flush=True)
                break
            eid, path, size = fut.result()
            if not path:
                streak += 1
                if streak >= _BREAKER_TRIP and kept == 0:
                    for pending in futs:
                        pending.cancel()
                    _cooldown_trip(f"{streak} consecutive failed fetches, "
                                   f"0 kept")
                    print(f"fetch breaker TRIPPED: {streak} consecutive "
                          f"failures, nothing kept -- cancelled pending, "
                          f"cooling down (next runs probe "
                          f"{_BREAKER_PROBE})", flush=True)
                    break
                continue
            if streak or cd:
                _cooldown_clear()          # the endpoint works again
                cd = None
            streak = 0
            spent += size if isinstance(size, int) else 0
            try:
                n = R.ingest_replay(path, source="sameday") \
                    if hasattr(R, "ingest_replay") else _ingest(path, idx)
                kept += n
                # Persist as we go. _ingest only mutates the in-memory index
                # while already having written the route file and obs-features
                # sidecar to disk, so a run killed before the final save left
                # every route it fetched orphaned: on disk, invisible to the
                # index, and re-downloaded next time. A scheduled scrape gets
                # interrupted (reboot, budget, the 06:30 release needing the
                # API), so checkpoint instead of betting on a clean exit.
                if kept and kept % 25 == 0:
                    R.save_index(idx)
                    print(f"  [checkpoint] {kept} route(s) indexed, "
                          f"{spent / 1e9:.2f} GB streamed", flush=True)
            except Exception as exc:                               # noqa: BLE001
                print(f"  ! {eid}: {str(exc).splitlines()[0]}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
    R.save_index(idx)
    print(f"same-day: kept {kept} route(s) from {len(todo)} episodes, "
          f"{spent / 1e9:.2f} GB streamed, 0 kept on disk")
    # What the controller learned, so a bad ceiling shows up in the log rather
    # than as an unexplained slowdown.
    apool.log_stats("adaptive fetch")
    return 0


def _episode_date(path, data=None):
    """The replay's real play date -- so a historical episode harvested from
    an old submission is NOT mislabelled as fresh (which would corrupt the
    freshness-ranked crown). Falls back to today only if unknown."""
    import datetime as dt
    import json as _json
    try:
        d = data if data is not None \
            else _json.load(open(path, encoding="utf-8"))
        for k in ("EndTime", "CreateTime", "endTime", "createTime"):
            v = (d.get("info") or {}).get(k) or d.get(k)
            if v:
                return str(v)[:10]
    except Exception:                                              # noqa: BLE001
        pass
    return dt.date.today().isoformat()


def _ingest(path, idx):
    """Fold both seats of a replay into the index, dated by the replay's own
    timestamp, with observation-features extracted NOW -- the replay is
    deleted after this call, so obs-features (reactivity, shop-mix, cash
    shape, microstructure) that no action stream carries must be captured
    here or lost. Stored in an obs_features sidecar keyed by route id.

    The replay is parsed exactly ONCE here and the dict passed everywhere.
    The old flow parsed the same 15-30 MB file five times (meta, date, this
    function, one extract_actions per seat) on the ingest thread, and that --
    not the parallel downloads -- was the fetch bottleneck: ~15 s/episode of
    pure redundant json.load (measured 2026-08-12)."""
    import json as _json
    import kaggriculture.measure.opponents as opponents
    import kaggriculture.data.obs_features as obs_features
    replay = _json.load(open(path, encoding="utf-8"))
    meta = opponents.replay_meta(path, data=replay)
    if not meta:
        return 0
    date = _episode_date(path, data=replay)
    import re as _re
    _m = _re.search(r"(\d{6,})", str(meta.get("episode") or ""))
    episode_num = _m.group(1) if _m else str(meta.get("episode"))
    meta["episode"] = episode_num
    sidecar = os.path.join(ROOT, "data", "obsfeat")
    os.makedirs(sidecar, exist_ok=True)
    # replay_meta has no "teams" key, so the old meta.get("teams") fallback
    # labeled EVERY same-day route "?" -- which starved anything joining
    # routes to leaderboard ratings (the rating scorer found labels on only
    # 1,503/11k routes, 2026-08-13). The replay itself carries the names.
    teams = list(((replay.get("info") or {}).get("TeamNames")) or [])
    # The replay's OWN engine version. This was hardcoded "1.32.6" (2026-08-20
    # bug), so every sameday-ingested route mis-reported its engine -- which
    # made the whole index look 1.32.6 while the ladder ran 1.32.7 and nearly
    # triggered a wrong engine revert. Read the real module_version; fall back
    # to unknown, never to a guessed version.
    engine_ver = str(replay.get("module_version")
                     or (replay.get("info") or {}).get("module_version")
                     or "unknown")
    added = 0
    for seat in (0, 1):
        rid = f"{meta['episode']}_s{seat}"
        if rid in idx["routes"]:
            continue
        acts = opponents.extract_actions(path, seat, data=replay)
        R.save_route(rid, acts)
        try:
            of = obs_features.flat(obs_features.obs_features(replay, seat))
            _json.dump(of, open(os.path.join(sidecar, rid + ".json"), "w"),
                       separators=(",", ":"))
            has_obs = True
        except Exception:                                          # noqa: BLE001
            has_obs = False
        idx["routes"][rid] = {
            "id": rid, "episode": meta["episode"], "seat": seat,
            "team": teams[seat] if seat < len(teams) else "?",
            "bank": meta["rewards"][seat], "opp_bank": meta["rewards"][1 - seat],
            "date": date, "window": "fit",
            "won": meta["winner"] == seat, "engine": engine_ver,
            "source": "sameday", "obs": has_obs,
            # The episode's own RNG seed, which was being discarded entirely.
            # One integer, and without it a replay cannot even be approximately
            # reproduced once the file is deleted.
            "seed": ((replay.get("info") or {}).get("seed")),
            "trace": False}
        added += 1

    # PER-TURN OBSERVATION TRACES (2026-08-13). The route above records what
    # the farm DID; nothing recorded the state it did it in, so there was no
    # (state, action) corpus for any offline learning -- and the replay is
    # deleted on the next line but one. Reconstruction is not a fallback:
    # replaying recorded actions under the recorded seed diverges 25%.
    # Measured 39 KB per episode compressed, both seats.
    try:
        import kaggriculture.data.turn_features as TF
        if not os.path.exists(TF.path_for(meta["episode"])):
            TF.capture(replay, meta["episode"])
        for seat in (0, 1):
            rid = f"{meta['episode']}_s{seat}"
            if rid in idx["routes"]:
                idx["routes"][rid]["trace"] = True
    except Exception as exc:                                       # noqa: BLE001
        # Never let trace capture cost us a route: the index entry is the
        # thing the crown needs, the trace is an enrichment.
        print(f"  ! trace capture {meta.get('episode')}: "
              f"{type(exc).__name__}: {str(exc)[:60]}")
    return added


if __name__ == "__main__":
    raise SystemExit(main())


# --- ID HARVEST NOTE ---------------------------------------------------------
# The replay-by-id path above works for ANY episode and is the reusable core.
# The one input it needs is a list of *current* foreign episode ids, which the
# public archive does not expose until it publishes (the 2-day lag).
#
# Harvest options, robust -> fragile:
#   1. Our own opponents: `kaggle competitions episodes <our_sub>` lists the
#      episodes WE played today; the opponent seat of each is a same-day
#      foreign route. Zero browser, always available -- wire via ourgames.
#   2. Browser: open a top team's episode viewer on the leaderboard; episode
#      ids appear in the /competitions/kaggriculture/leaderboard?dialog=episodes
#      URLs and in the GetLeaderboard/episode API responses (endpoint
#      confirmed: competitions.LeaderboardService/GetLeaderboard, but the
#      protobuf-json body shape needs pinning before automation).
#
# Option 1 gives same-day foreign routes for free and is implemented as the
# default source below.
