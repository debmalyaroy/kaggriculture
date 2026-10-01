"""Mine fresh 719-turn routes off the public episode archive.

    python -m kaggriculture.data.routes --mine --top 200 --per-team 3 --max-gb 8
    python -m kaggriculture.data.routes --report
    python -m kaggriculture.data.routes --list
    python -m kaggriculture.data.routes --build 90161060_s1 --out agents/v15_route.py

The default slice is the **top 200**, not the top 30. Two reasons. The rating
floor at rank 200 is ~2,540 against ~2,900 at rank 30, so the extra 170 teams
are still far above the 687 median -- these are real routes, not filler. And a
30-team panel is small enough that one replay-heavy family dominates it, which
is the failure mode the medoid rule exists to prevent; 200 gives that rule
something to work with.

Why this exists, and why *fresh* is the whole point. Kaito Fukami's v21.1
write-up measures one agent with three different routes against the same
validation window:

    old public v21 / Dennis medoid ..... 19/46
    current Konstantin medoid .......... 40/46
    current Richard Silence medoid ..... 41/46

Same safety layers, same everything else. The top three current families differ
at about 110 channel moments and sit more than 1,200 moments from the stale
one. A route is a perishable asset, so mining one is a thing you do on a
cadence, not once.

How it works, and why it is shaped like this
--------------------------------------------
The daily `manifest.csv` carries `episode_id, create_time, avg_score,
min_score, agent_count, size_bytes` and **no team**. Team identity lives inside
the replay, under `info.TeamNames`, so there is no way to pick a specific
player's games without downloading. A replay is ~27 MB and a day is ~21 GB, so
this streams: download one, extract the route, delete it, move on. Disk stays
flat regardless of how much is mined.

Ranking is by `min_score` rather than `avg_score` -- a high average can be one
strong player beating a weak one, and half of such a game is a route we do not
want. `min_score` high means *both* seats were strong.

**One medoid route per submission.** A team that replays heavily would
otherwise supply a dozen near-identical routes and dominate the panel by sheer
count. The medoid is the route closest to that team's own centre, so it is a
typical game rather than a lucky one.

**Disjoint windows.** Newest 6 per team are the outer holdout and are never
loaded during selection; the next 3 validate; the rest are fit-only donors.
Selecting and validating on the same games is how you pick a route that is
excellent at the games you already have.

Provenance: this reads Kaggle's own published episode archive, which is what
the whole field mines and credits. Payloads decoded out of competitors'
*notebooks* remain data-only under SECURITY.md -- never imported, never run,
never shipped.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import copy
import glob
import gzip
import json
import os
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.data.episodes as E  # noqa: E402
import kaggriculture.measure.opponents as OPP  # noqa: E402

WORK = os.path.join(ROOT, "data", "routes")
STAGE = os.path.join(WORK, "_stage")
INDEX = os.path.join(WORK, "index.json")

# Window sizes, newest first, per submission. Straight from the protocol the
# current top-30 notebooks describe, so results are comparable to theirs.
OUTER, VALIDATION = 6, 3


# --------------------------------------------------------------------- store

def load_index():
    if os.path.exists(INDEX):
        try:
            with open(INDEX, encoding="utf-8") as fh:
                return json.load(fh)
        except ValueError:
            pass
    return {"routes": {}, "mined": [], "updated": None}


def save_index(idx):
    """Merge-then-write under a lock, so two concurrent writers (a same-day
    scrape and a breadth mine) cannot stomp each other's additions -- the bug
    that lost 196 same-day routes on 2026-08-11. We re-read the on-disk index
    inside the lock and union routes before writing.
    """
    os.makedirs(WORK, exist_ok=True)
    lock = INDEX + ".lock"
    for _ in range(600):                       # up to ~60s
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        disk = load_index()
        merged = dict(disk.get("routes", {}))
        merged.update(idx.get("routes", {}))   # our newer entries win per key
        idx["routes"] = merged
        idx["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = INDEX + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=1, default=str)
        os.replace(tmp, INDEX)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def route_path(route_id):
    # A route id becomes a filename, so it must survive being one. "?" is legal
    # in a JSON key and illegal in a Windows path, and the difference killed a
    # 12 GB mine several hundred downloads in.
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(route_id))
    return os.path.join(WORK, f"{safe or 'unknown'}.json.gz")


def save_route(route_id, actions):
    os.makedirs(WORK, exist_ok=True)
    with gzip.open(route_path(route_id), "wt", encoding="utf-8") as fh:
        json.dump(actions, fh, separators=(",", ":"))
    return os.path.getsize(route_path(route_id))


def load_route(route_id):
    with gzip.open(route_path(route_id), "rt", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ extract

def _signature(actions):
    """A cheap, order-sensitive summary used to find a team's medoid route.

    Counting ops per channel rather than diffing 719 dicts: the medoid only has
    to be *typical*, and a full pairwise diff over a few hundred routes would
    cost more than the download did.
    """
    counts = {}
    for turn in actions:
        if not isinstance(turn, dict):
            continue
        ops = [turn.get("farmer")] + list(turn.get("hands") or [])
        for op in ops:
            if isinstance(op, list) and op:
                counts[f"u:{op[0]}"] = counts.get(f"u:{op[0]}", 0) + 1
        for order in turn.get("market") or []:
            if isinstance(order, list) and order:
                key = f"m:{order[0]}:{order[1] if len(order) > 1 else ''}"
                qty = float(order[2]) if len(order) > 2 else 1.0
                counts[key] = counts.get(key, 0) + qty
    return counts


def _distance(a, b):
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _value_after_key(raw, key):
    """Decode one JSON value out of a byte buffer without parsing the rest."""
    start = raw.find(key)
    if start < 0:
        return None
    colon = raw.find(b":", start + len(key))
    if colon < 0:
        return None
    text = raw[colon + 1:].decode("utf-8", errors="ignore").lstrip()
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def header(replay_path):
    """Team names, rewards and episode id from the first ~1 MB of a replay.

    A replay is ~27 MB and `json.load` on one costs a second or two and about
    ten times its size in memory. Everything needed to decide *whether we want
    this game at all* -- which teams played and who won -- sits in `info` near
    the top of the file, so read a prefix and stop.

    Technique from `llccqq624/kaggriculture-replay-data-miner`.
    """
    raw = b""
    try:
        with open(replay_path, "rb") as fh:
            for limit in (131_072, 524_288, 1_048_576, 4_194_304):
                raw += fh.read(limit - len(raw))
                teams = _value_after_key(raw, b'"TeamNames"')
                rewards = _value_after_key(raw, b'"rewards"')
                if teams is not None and rewards is not None:
                    break
                if len(raw) < limit:
                    break                      # file is shorter than the window
    except OSError:
        return {}
    return {
        "teams": list(teams or []),
        "rewards": [float(r or 0) for r in (rewards or [])],
        "episode": _value_after_key(raw, b'"EpisodeId"'),
        "engine": _value_after_key(raw, b'"module_version"'),
    }


def _episode_from_name(path):
    """Episode id out of a filename, whichever of the two shapes it has."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return stem                                   # 90161060.json
    for part in stem.split("-"):
        if part.isdigit():
            return part                               # episode-90161060-replay.json
    return "".join(c for c in stem if c.isalnum() or c in "._-") or "unknown"


def extract(replay_path):
    """Pull every usable route out of one replay. Returns a list of dicts."""
    with open(replay_path, encoding="utf-8") as fh:
        replay = json.load(fh)

    info = replay.get("info") or {}
    teams = list(info.get("TeamNames") or [])
    # The ternary here used to bind across the `or`, so `str((A or B) if cond
    # else "?")` returned "?" for every file whose name had no "-" -- which is
    # every file downloaded straight from the daily dataset (`<id>.json`), as
    # opposed to the `episode-<id>-replay.json` the CLI produces. The route
    # then tried to write to "?_s0.json.gz" and the whole mine died on an
    # invalid filename.
    episode = str(info.get("EpisodeId") or _episode_from_name(replay_path))
    rewards = [float(r or 0) for r in (replay.get("rewards") or [0, 0])]
    statuses = list(replay.get("statuses") or [])
    engine = replay.get("module_version")
    created = info.get("EndTime") or replay.get("created")

    out = []
    for seat in range(len(rewards)):
        if seat < len(statuses) and statuses[seat] not in ("DONE", "ACTIVE"):
            continue                      # a crashed seat is not a route
        actions = OPP.extract_actions(replay_path, seat)
        if not actions or len(actions) < 700:
            continue
        mine = rewards[seat]
        theirs = rewards[1 - seat] if len(rewards) > 1 else 0.0
        out.append({
            "id": f"{episode}_s{seat}",
            "episode": episode,
            "seat": seat,
            "team": teams[seat] if seat < len(teams) else "?",
            "opponent": teams[1 - seat] if len(teams) > 1 else "?",
            "bank": mine,
            "opp_bank": theirs,
            "won": mine > theirs,
            "steps": len(actions),
            "engine": engine,
            "created": created,
            "_actions": actions,
        })
    return out


# --------------------------------------------------------------------- mine

def _manifest_rows(days, verbose=True):
    """Episodes from the most recent daily datasets, best first."""
    index = E.fetch_index(WORK)
    if not index:
        raise SystemExit("the episode index manifest was empty -- is the "
                         "kaggle CLI authenticated?")
    chosen = index[-days:] if days else index
    rows = []
    for day_row in reversed(chosen):
        date = day_row["date"]
        dataset = day_row.get("daily_dataset_slug") or E.DAILY_DATASET.format(date=date)
        if "/" not in dataset:
            dataset = f"kaggle/{dataset}"
        day_dir = os.path.join(WORK, "idx", date)
        try:
            man = E._download_dataset_file(dataset, "manifest.csv", day_dir)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {date}: {exc}")
            continue
        if not man:
            continue
        for r in E._read_csv(man):
            r["_date"], r["_dataset"] = date, dataset
            rows.append(r)
        if verbose:
            print(f"  {date}: {len(E._read_csv(man))} episodes indexed")

    def key(r):
        # Both seats strong. A high average can be one good player farming a
        # weak one, and half of that game is a route we do not want.
        for k in ("min_score", "avg_score"):
            if r.get(k):
                try:
                    return float(r[k])
                except ValueError:
                    pass
        return 0.0

    # Per-day quality floor, not an absolute one. The whole field's rating
    # level climbs daily, so one number taken from *today's* leaderboard and
    # applied to *historical* games silently rejects most of the archive: at a
    # floor of 2,843 only 34% of 3,761 indexed episodes qualified, and the split
    # was 683 on the newest day against 47 four days earlier. That is not a
    # quality signal, it is calendar drift -- and it is why 44 top-100 teams
    # looked unreachable when their games were simply older.
    #
    # Ranking within a day is stable, so each episode carries its own day's
    # percentile and callers threshold on that instead.
    by_day = {}
    for r in rows:
        by_day.setdefault(r.get("_date"), []).append(r)
    for _date, day_rows in by_day.items():
        day_rows.sort(key=key, reverse=True)
        n = max(1, len(day_rows))
        for i, r in enumerate(day_rows):
            r["_day_pct"] = 1.0 - (i / n)      # 1.0 = best of that day

    rows.sort(key=key, reverse=True)
    return rows


_WIRE_RATIO = 13.0     # conservative end of the measured 13-16x json/gzip ratio


def _wire_bytes(path):
    """Bytes this replay actually cost on the network.

    Both transports move the replay compressed (gzip on kaggleusercontent, zip
    via the dataset CLI) and land it decompressed on disk, so the on-disk size
    overstates the network cost by the measured ratio. Deliberately uses the
    CONSERVATIVE end of the range, so the budget still under-spends rather than
    over-spends if a replay happens to compress badly.
    """
    try:
        return int(os.path.getsize(path) / _WIRE_RATIO)
    except OSError:
        return 0


def _fetch_one(row, dest_dir):
    """One episode replay, by the fastest transport that works.

    kaggleusercontent FIRST (2026-08-13). The daily-dataset CLI path was the
    only transport here, and it is the throttled one: sameday's own record is
    ~5 episodes per 30 minutes once rate-limited (~360 s each), against a
    MEASURED 3.05 s for the direct endpoint. Archive-era ids resolve there
    too -- verified on 2026-08-01 episodes, 12 days old -- so the CLI is now
    only the fallback, exactly as in sameday and refresh_cycle.stage_tapes.

    A direct 404 falls through to the CLI so the caller still gets the
    exception it needs to tombstone a pruned file permanently.
    """
    eid = str(row.get("episode_id") or "").strip()
    if eid:
        try:
            import kaggriculture.data.sameday as sameday
            path = sameday._grab_direct(eid, dest_dir)
            if path and os.path.exists(path):
                return path
        except Exception as exc:                                   # noqa: BLE001
            if str(exc).startswith("QUOTA-HOLD"):
                # Budget decision, not a transport failure -- the CLI burns
                # the same rolling quota slower. Do not fall through.
                raise
            pass                        # fall through to the CLI transport
    fname = E._episode_file(row)
    if not fname:
        return None
    try:
        return E._download_dataset_file(row["_dataset"], fname, dest_dir)
    except Exception as exc:                                       # noqa: BLE001
        # One retry for transient transport failures (ConnectionAborted
        # 10053 observed when 8 workers burst against the API); permanent
        # errors (404) re-raise immediately so the caller can tombstone.
        if "404" in str(exc):
            raise
        time.sleep(3)
        return E._download_dataset_file(row["_dataset"], fname, dest_dir)


def mine(top=200, per_team=3, max_gb=8.0, jobs=8, days=3,
         min_rating=None, min_pct=0.35, rescan=False, verbose=True):
    os.makedirs(STAGE, exist_ok=True)
    idx = load_index()
    already = set(idx.get("mined") or [])

    # Rank -> team and rank -> score in one read, so the quality floor is the
    # score at rank `top` rather than a constant that goes stale. This field
    # grew 517 teams in 22 hours; a hardcoded floor would already be wrong.
    board = E.leaderboard_rows(verbose=verbose) or []
    ranked = {row["team"]: row for row in board[:top] if row["team"]}
    teams = set(ranked)
    if min_rating is None:
        min_rating = board[min(top, len(board)) - 1]["score"] if board else 0.0
    if verbose:
        print(f"\ntarget: top-{top} teams"
              + (f" ({len(teams)} resolved, rating floor "
                 f"{min_rating:,.1f} at rank {min(top, len(board))})"
                 if teams else " (leaderboard unavailable -- keeping every "
                 "strong episode)"))
        idx["leaderboard"] = {"captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "teams": len(board), "top": top,
                              "floor": min_rating}

    rows = _manifest_rows(days, verbose=verbose)
    rows = [r for r in rows if str(r.get("episode_id") or "") not in already]
    budget = int(max_gb * 1024 ** 3)
    if verbose:
        print(f"{len(rows)} candidate episode(s); budget {max_gb:.1f} GB, "
              f"{jobs} download(s) at a time")
        print(f"want {top} x {per_team} = {top * per_team} route(s); a replay "
              f"yields up to 2, so this needs ~{top * per_team // 2} usable "
              f"downloads at ~27 MB each\n")

    spent, kept, skipped, seen_per_team = 0, 0, 0, {}
    t0 = time.time()

    def enough():
        if not teams:
            return kept >= top * per_team
        return all(seen_per_team.get(t, 0) >= per_team for t in teams)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending, cursor = {}, 0
        while cursor < len(rows) or pending:
            while len(pending) < jobs and cursor < len(rows) and spent < budget:
                row = rows[cursor]; cursor += 1
                dest = os.path.join(STAGE, f"w{cursor % (jobs * 2)}")
                pending[pool.submit(_fetch_one, row, dest)] = row
            if not pending:
                break
            done = next(as_completed(list(pending)))
            row = pending.pop(done)
            try:
                path = done.result()
            except Exception as exc:                               # noqa: BLE001
                msg = str(exc)
                # A 404 is PERMANENT: the file is gone from that daily
                # dataset (observed for 2026-08-01/05 files on 2026-08-12).
                # Without a tombstone the backfill retries it every single
                # run forever. Mark it mined so it never comes back.
                if "404 Client Error" in msg or "Not Found" in msg:
                    idx.setdefault("mined", []).append(
                        str(row.get("episode_id")))
                    print(f"  ! {row.get('episode_id')}: 404 (gone from "
                          f"dataset) -- tombstoned, will not retry")
                else:
                    print(f"  ! {row.get('episode_id')}: "
                          f"{msg.splitlines()[0] if msg else exc}")
                continue
            if not path or not os.path.exists(path):
                continue
            # BUDGET IS WIRE BYTES, NOT DISK BYTES (2026-08-13). This counted
            # os.path.getsize(path) -- the DECOMPRESSED json -- while both
            # transports move the file compressed: measured 1.30-1.64 MB on
            # the wire against 20.8 MB decompressed, a ~13-16x ratio. So an
            # "8 GB" budget was really ~0.5 GB of traffic, and the miner was
            # stopping an order of magnitude early. Staged files are deleted
            # immediately after ingest (below), so disk is bounded by
            # jobs x one replay and was never what this protected.
            spent += _wire_bytes(path)
            # Peek at the header before committing to a 27 MB parse. Most
            # downloaded episodes are played by teams we did not ask for, and
            # rejecting those in ~10 ms instead of ~2 s is most of the loop.
            head = header(path)
            # Reject on the header when nobody in this game is still wanted --
            # either not in the slice, or already at their per-team cap. The
            # top of the manifest is dominated by a handful of very active
            # teams, so without the cap check most of the budget is spent
            # re-parsing 27 MB for routes that are then thrown away.
            wanted_here = {t for t in (head.get("teams") or [])
                           if (not teams or t in teams)
                           and seen_per_team.get(t, 0) < per_team}
            if head.get("teams") and not wanted_here:
                try:
                    os.remove(path)
                except OSError:
                    pass
                idx.setdefault("mined", []).append(str(row.get("episode_id")))
                skipped += 1
                continue
            try:
                found = extract(path)
            except Exception as exc:                               # noqa: BLE001
                print(f"  ! {row.get('episode_id')}: extract failed: {exc}")
                found = []
            finally:
                # Delete before anything else can fail. This is the invariant
                # that keeps a 21 GB/day archive minable on a laptop.
                try:
                    os.remove(path)
                except OSError:
                    pass

            for rec in found:
                team = rec["team"]
                if teams and team not in teams:
                    continue
                if seen_per_team.get(team, 0) >= per_team:
                    continue
                # A route mine that quietly absorbs mid-field games produces a
                # panel that looks full and teaches nothing. The manifest
                # already ranked this episode; carry that rating onto the route
                # and refuse anything below the floor.
                rating = 0.0
                for k in ("min_score", "avg_score"):
                    if row.get(k):
                        try:
                            rating = float(row[k]); break
                        except ValueError:
                            pass
                # Percentile-within-its-own-day, so an older game is judged
                # against the field it actually played in.
                pct = float(row.get("_day_pct") or 0.0)
                if min_pct and pct < min_pct:
                    continue
                if min_rating and rating and rating < min_rating:
                    continue
                rec["rating"] = rating
                rec["rank"] = (ranked.get(team) or {}).get("rank")
                rec["team_score"] = (ranked.get(team) or {}).get("score")
                actions = rec.pop("_actions")
                rec["sig"] = _signature(actions)
                rec["bytes"] = save_route(rec["id"], actions)
                rec["date"] = row.get("_date")
                idx["routes"][rec["id"]] = rec
                seen_per_team[team] = seen_per_team.get(team, 0) + 1
                kept += 1
            idx.setdefault("mined", []).append(str(row.get("episode_id")))

            if kept and kept % 10 == 0:
                save_index(idx)
            if verbose:
                print(f"  [{spent / 1024**3:>5.2f}/{max_gb:.1f} GB wire] "
                      f"{row.get('episode_id')}  +{len(found)} route(s)  "
                      f"kept {kept}  skipped {skipped}  teams {len(seen_per_team)}")
            if spent >= budget or enough():
                break

    save_index(idx)
    if verbose:
        print(f"\nmined {kept} route(s) from {len(seen_per_team)} team(s) in "
              f"{(time.time() - t0) / 60:.1f} min, {spent / 1024**3:.2f} GB "
              f"streamed, 0 bytes kept on disk")
        assign_windows(idx, verbose=False)
        save_index(idx)
    return idx


# ------------------------------------------------------------------ windows

def assign_windows(idx=None, verbose=True):
    """Split each team's routes, newest first, into outer / validation / fit.

    The published protocol assumes ~17 routes per submission, at which point the
    fixed 6/3/rest split is right. With fewer it is not: a team with exactly six
    routes puts all six in the outer holdout and leaves **nothing to fit on**,
    which reads as "mined successfully" and yields no usable route. So the split
    scales below the threshold, and `thin` records that it had to.
    """
    idx = idx or load_index()
    by_team = {}
    for rid, rec in idx["routes"].items():
        by_team.setdefault(rec.get("team", "?"), []).append(rec)
    thin = []
    for team, recs in by_team.items():
        recs.sort(key=lambda r: (r.get("date") or "", r.get("episode") or ""),
                  reverse=True)
        n = len(recs)
        if n >= OUTER + VALIDATION + 1:
            outer, valid = OUTER, VALIDATION
        else:
            # Keep the shape (holdout largest, then validation, then fit) but
            # never let fit reach zero while any route exists.
            outer = max(1, n // 3) if n >= 3 else (1 if n > 1 else 0)
            valid = max(1, n // 3) if n >= 3 else 0
            if outer + valid >= n:
                outer, valid = max(0, n - 1), 0
            thin.append((team, n))
        for i, rec in enumerate(recs):
            rec["window"] = ("outer" if i < outer else
                             "validation" if i < outer + valid else "fit")
    if verbose:
        counts = {}
        for rec in idx["routes"].values():
            counts[rec.get("window")] = counts.get(rec.get("window"), 0) + 1
        print("windows: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if thin:
            print(f"  ! {len(thin)} team(s) below the {OUTER + VALIDATION + 1}-route "
                  f"threshold, so their split was scaled: "
                  + ", ".join(f"{t}({n})" for t, n in thin[:6])
                  + (" ..." if len(thin) > 6 else ""))
            print(f"    mine more per team for the full {OUTER}/{VALIDATION}/rest "
                  f"protocol.")
    return idx


def medoids(idx=None, window="fit"):
    """One representative route per team: the one closest to that team's centre."""
    idx = idx or load_index()
    by_team = {}
    for rec in idx["routes"].values():
        if window and rec.get("window") != window:
            continue
        by_team.setdefault(rec.get("team", "?"), []).append(rec)
    out = []
    for team, recs in sorted(by_team.items()):
        if len(recs) == 1:
            out.append(recs[0])
            continue
        best, best_cost = None, None
        for a in recs:
            cost = sum(_distance(a.get("sig") or {}, b.get("sig") or {})
                       for b in recs if b is not a)
            if best_cost is None or cost < best_cost:
                best, best_cost = a, cost
        out.append(best)
    return out


# ------------------------------------------------------------------- report

def report():
    idx = load_index()
    routes = list(idx["routes"].values())
    if not routes:
        print("no routes mined yet -- run: python -m kaggriculture.data.routes --mine")
        return 1
    assign_windows(idx, verbose=False)
    save_index(idx)

    by_team = {}
    for rec in routes:
        by_team.setdefault(rec.get("team", "?"), []).append(rec)
    lb = idx.get("leaderboard") or {}
    print(f"{len(routes)} route(s) from {len(by_team)} team(s), "
          f"updated {idx.get('updated')}")
    if lb:
        print(f"leaderboard captured {lb.get('captured')}: top-{lb.get('top')} "
              f"of {lb.get('teams')} teams, rating floor {lb.get('floor', 0):,.1f}")
    print()
    print(f"{'rank':>5} {'team':<28}{'n':>3}{'won':>5}{'best bank':>12}{'median':>11}"
          f"{'outer':>7}{'valid':>7}{'fit':>5}")
    print("-" * 84)
    def _rank(recs):
        r = [x.get("rank") for x in recs if x.get("rank")]
        return min(r) if r else 10**6
    for team, recs in sorted(by_team.items(), key=lambda kv: _rank(kv[1])):
        banks = [r["bank"] for r in recs]
        w = {"outer": 0, "validation": 0, "fit": 0}
        for r in recs:
            w[r.get("window", "fit")] = w.get(r.get("window", "fit"), 0) + 1
        rank = _rank(recs)
        print(f"{(rank if rank < 10**6 else '?'):>5} {team[:27]:<28}{len(recs):>3}"
              f"{sum(1 for r in recs if r['won']):>5}"
              f"{max(banks):>12,.0f}{statistics.median(banks):>11,.0f}"
              f"{w['outer']:>7}{w['validation']:>7}{w['fit']:>5}")

    engines = {}
    for rec in routes:
        engines[rec.get("engine")] = engines.get(rec.get("engine"), 0) + 1
    print(f"\nengine versions in the mine: "
          + ", ".join(f"{k}x{v}" for k, v in sorted(engines.items(), key=str)))
    meds = medoids(idx)
    print(f"medoid routes available (fit window): {len(meds)}")
    for rec in sorted(meds, key=lambda r: -r["bank"])[:10]:
        print(f"  {rec['id']:<18} {rec['team'][:24]:<26} "
              f"${rec['bank']:>10,.0f}  vs ${rec['opp_bank']:>10,.0f}"
              f"  {'WON' if rec['won'] else 'lost'}")
    return 0


def build(route_id, out_path, weed_catchup=8, impact_slots=True,
          mirror_tiebreak=True, floor_guard=False,
          endgame_pull=False, guard_ratio=0.0, swap_advance=False,
          sell_first=True, feed_pin=True, premium_lead=False,
          deposit_advance=False, floor_seller=False, branches=None,
          verbose=True):
    """Render one mined route into a single self-contained agent."""
    import base64
    import zlib
    import datetime as dt
    import kaggriculture.engine.tape_runtime as tape_runtime

    if endgame_pull or swap_advance:
        raise SystemExit(
            "endgame_pull/swap_advance repaid themselves through the retired "
            "_safe_market clamp; enabling them now would sell pulled "
            "quantities twice. Reimplement explicit subtraction first.")

    idx = load_index()
    rec = idx["routes"].get(route_id)
    if rec is None:
        # Allow "best", or a team name, as well as a route id.
        if route_id in ("best", "auto"):
            # Only ever pick from the fit window. The outer and validation
            # windows exist to answer "is this route any good", and a route
            # chosen because it scored well in them cannot then be measured by
            # them -- that is selecting and validating on the same games, which
            # is exactly what the split was built to prevent.
            assign_windows(idx, verbose=False)
            save_index(idx)
            fit = [r for r in idx["routes"].values() if r.get("window") == "fit"]
            pool = [r for r in fit if r.get("won")] or fit
            if not pool:
                raise SystemExit(
                    "no routes in the fit window -- mine more per team "
                    "(python -m kaggriculture.data.routes --report shows the split)")
            rec = max(pool, key=lambda r: r.get("bank", 0))
        else:
            hits = [r for r in idx["routes"].values()
                    if r.get("team", "").lower() == route_id.lower()]
            if not hits:
                raise SystemExit(f"no route {route_id!r}; try --list")
            rec = max(hits, key=lambda r: r.get("bank", 0))
    actions = load_route(rec["id"])

    payload = base64.b85encode(zlib.compress(
        json.dumps(actions, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
    branchpack = ""
    if branches:
        bmap = {}
        for shop, bid in branches.items():
            try:
                bmap[shop] = load_route(bid)
            except OSError:
                pass
        if bmap:
            branchpack = base64.b85encode(zlib.compress(
                json.dumps(bmap, separators=(",", ":")).encode("utf-8"),
                9)).decode("ascii")
    src = tape_runtime.TEMPLATE.format(
        label=os.path.basename(out_path),
        route_id=rec["id"], team=rec.get("team", "?"),
        episode=rec.get("episode", "?"), seat=rec.get("seat", 0),
        bank=rec.get("bank", 0.0), opp_bank=rec.get("opp_bank", 0.0),
        built=dt.date.today().isoformat(),
        payload=payload, weed_catchup=int(weed_catchup),
        impact_slots=bool(impact_slots),
        mirror_tiebreak=bool(mirror_tiebreak),
        floor_guard=bool(floor_guard),
        endgame_pull=bool(endgame_pull),
        guard_ratio=float(guard_ratio),
        swap_advance=bool(swap_advance),
        sell_first=bool(sell_first),
        feed_pin=bool(feed_pin),
        premium_lead=bool(premium_lead),
        deposit_advance=bool(deposit_advance),
        floor_seller=bool(floor_seller),
        branchpack=branchpack)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    if verbose:
        try:
            shown = os.path.relpath(out_path, ROOT)
        except ValueError:                    # e.g. out_path on another drive
            shown = out_path
        print(f"built {shown} "
              f"({os.path.getsize(out_path):,} bytes)")
        print(f"  route   {rec['id']}  {rec.get('team')}  "
              f"${rec.get('bank', 0):,.0f} vs ${rec.get('opp_bank', 0):,.0f}")
        print(f"  window  {rec.get('window')}   rank {rec.get('rank', '?')}")
        print(f"\nNext: python -m kaggriculture.measure.evaluate {os.path.relpath(out_path, ROOT)} "
              f"--vs agents/v14_market.py -n 4")
    return out_path


def select(n=6, seeds=2, seed0=51000, workers=None, out=None, extra=None,
           verbose=True):
    """Build the top candidate routes and pick the winner by playing them.

    `--build best` ranks by the bank the route recorded, and that is a bad
    selector: $151k banked against a weak opponent is not better than $144k
    banked against a strong one, and the recorded number also carries whatever
    the *opponent* did that game. Rank correlates better but still measures the
    player, not this particular trajectory.

    So: build the candidates, play them against each other and against the
    incumbent, both seats, and promote on the result. Candidates are drawn from
    the **fit** window only; the validation and outer windows stay unopened so
    there is something left to check the winner against.
    """
    import itertools
    from concurrent.futures import ProcessPoolExecutor

    idx = assign_windows(load_index(), verbose=False)
    save_index(idx)
    fit = [r for r in idx["routes"].values() if r.get("window") == "fit"]
    pool = [r for r in fit if r.get("won")] or fit
    if not pool:
        raise SystemExit("no fit-window routes -- mine more per team")

    # Seed the shortlist by team rank (their converged skill), breaking ties on
    # margin rather than bank -- margin is the part of the recorded game that
    # was actually about this route.
    pool.sort(key=lambda r: (r.get("rank") or 10 ** 6,
                             -(r.get("bank", 0) - r.get("opp_bank", 0))))
    short = pool[:max(2, n)]

    stage = os.path.join(WORK, "_cand")
    os.makedirs(stage, exist_ok=True)
    built = []
    for rec in short:
        path = os.path.join(stage, f"cand_{rec['id']}.py")
        build(rec["id"], path, verbose=False)
        built.append((rec, path))
    contenders = [p for _, p in built] + list(extra or [])
    if verbose:
        print(f"{len(built)} candidate(s) from the fit window, "
              f"{len(extra or [])} incumbent(s), {seeds} seed(s) x 2 seats\n")

    jobs, meta = [], []
    for a, b in itertools.combinations(contenders, 2):
        for s in range(seeds):
            jobs.append((a, b, seed0 + s)); meta.append((a, b))
            jobs.append((b, a, seed0 + s)); meta.append((b, a))

    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_play_pair, jobs))

    wins = {p: 0.0 for p in contenders}
    games = {p: 0 for p in contenders}
    banks = {p: [] for p in contenders}
    for (left, right), (r0, r1) in zip(meta, results):
        games[left] += 1; games[right] += 1
        banks[left].append(r0); banks[right].append(r1)
        if r0 > r1:
            wins[left] += 1
        elif r1 > r0:
            wins[right] += 1
        else:
            wins[left] += 0.5; wins[right] += 0.5

    rows = []
    for rec, path in built:
        rows.append({"rec": rec, "path": path,
                     "win": wins[path] / max(1, games[path]),
                     "bank": statistics.mean(banks[path]) if banks[path] else 0.0,
                     "games": games[path]})
    for path in (extra or []):
        rows.append({"rec": None, "path": path,
                     "win": wins[path] / max(1, games[path]),
                     "bank": statistics.mean(banks[path]) if banks[path] else 0.0,
                     "games": games[path]})
    rows.sort(key=lambda r: (-r["win"], -r["bank"]))

    if verbose:
        print(f"{'route':<16}{'rank':>5} {'team':<24}{'win%':>7}{'mean bank':>12}{'n':>4}")
        print("-" * 70)
        for r in rows:
            rec = r["rec"]
            name = rec["id"] if rec else os.path.basename(r["path"])
            rank = (rec.get("rank") if rec else None) or "-"
            team = (rec.get("team", "?")[:23] if rec else "incumbent")
            print(f"{name:<16}{rank:>5} {team:<24}{100 * r['win']:>6.0f}%"
                  f"{r['bank']:>12,.0f}{r['games']:>4}")

    winner = rows[0]
    if winner["rec"] is None:
        if verbose:
            print(f"\nthe incumbent won -- no candidate beat "
                  f"{os.path.basename(winner['path'])}. Nothing promoted.")
        return None
    if out:
        build(winner["rec"]["id"], out, verbose=False)
        if verbose:
            print(f"\nwrote {os.path.relpath(out, ROOT)} from "
                  f"{winner['rec']['id']} ({winner['rec'].get('team')}, "
                  f"rank {winner['rec'].get('rank')}) -- "
                  f"{100 * winner['win']:.0f}% over {winner['games']} games")
            print(f"\nNext: python -m kaggriculture.measure.evaluate {os.path.relpath(out, ROOT)} "
                  f"--vs agents/v14_market.py -n 4")
    return winner


def splice(field_id, market_id):
    """A route with one donor's field channel and another's market channel.

    Two reasons this is the right move rather than a clever one.

    **Mechanically**, a verbatim route cannot beat the team it came from. In the
    top-100 panel our margin against that team was exactly $0 -- both sides run
    the same 719 actions, so the episode is a perfect mirror and there is no
    edge to find. Splicing removes that ceiling: the result is nobody's recorded
    game, so no opponent on the ladder mirrors it.

    **Empirically**, the channels are separable and unequal. prvsiyan's public
    finding is that "the farmer and hand tapes contribute almost nothing; the
    market tape carries the frontier gain", and several public agents already
    store the unit and market traces separately for exactly this reason. If the
    market channel is where the value is, it should be free to come from the
    best market donor rather than riding along with whoever had the best field
    play.

    The risk is obvious and is what the safety stack is for: a market channel
    fitted to a *different* field channel will ask to sell produce this farm has
    not grown yet. `_safe_market` clamps every SELL to the projected shed and
    drops the empties, so a mismatch costs an order slot rather than desyncing
    the route. Whether the trade is worth it is a measurement, not an argument
    -- hence `splice_search`.
    """
    field = load_route(field_id)
    market = load_route(market_id)
    n = min(len(field), len(market))
    out = []
    for t in range(max(len(field), len(market))):
        f = field[t] if t < len(field) else {}
        m = market[t] if t < len(market) else {}
        out.append({
            "farmer": copy.deepcopy(f.get("farmer")) if isinstance(f, dict) else ["PASS"],
            "hands": copy.deepcopy(f.get("hands") or []) if isinstance(f, dict) else [],
            "market": copy.deepcopy(m.get("market") or []) if isinstance(m, dict) else [],
        })
    return out, n


def splice_search(n_field=4, n_market=4, seeds=2, seed0=45000, workers=None,
                  out=None, extra=None, verbose=True):
    """Try field x market combinations and keep the one that wins."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor
    import base64
    import datetime as dt
    import zlib
    import kaggriculture.engine.tape_runtime as tape_runtime

    idx = assign_windows(load_index(), verbose=False)
    save_index(idx)
    fit = [r for r in idx["routes"].values() if r.get("window") == "fit"]
    pool = [r for r in fit if r.get("won")] or fit
    if len(pool) < 2:
        raise SystemExit("need at least two fit-window routes to splice")
    pool.sort(key=lambda r: (r.get("rank") or 10 ** 6,
                             -(r.get("bank", 0) - r.get("opp_bank", 0))))
    fields = pool[:max(1, n_field)]
    markets = pool[:max(1, n_market)]

    stage = os.path.join(WORK, "_splice")
    os.makedirs(stage, exist_ok=True)
    built = []
    for f, m in itertools.product(fields, markets):
        tape, _n = splice(f["id"], m["id"])
        label = f"{f['id']}+{m['id']}"
        path = os.path.join(stage, f"sp_{f['id']}__{m['id']}.py")
        payload = base64.b85encode(zlib.compress(
            json.dumps(tape, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
        src = tape_runtime.TEMPLATE.format(
            label=label, route_id=label,
            team=f"field {f.get('team')} / market {m.get('team')}",
            episode=f"{f.get('episode')}+{m.get('episode')}", seat=f.get("seat", 0),
            bank=f.get("bank", 0.0), opp_bank=m.get("bank", 0.0),
            built=dt.date.today().isoformat(), payload=payload,
            weed_catchup=8, impact_slots=True, mirror_tiebreak=True,
            floor_guard=False, endgame_pull=False,
            guard_ratio=0.0, swap_advance=False,
            sell_first=True, feed_pin=True, premium_lead=False,
            deposit_advance=False, floor_seller=False,
            branchpack="")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
        built.append(({"field": f, "market": m, "label": label}, path))

    contenders = [p for _, p in built] + list(extra or [])
    if verbose:
        print(f"{len(built)} splice(s) from {len(fields)} field x "
              f"{len(markets)} market donors, plus {len(extra or [])} incumbent(s)")
        print(f"{seeds} seed(s) x 2 seats\n")

    jobs, meta = [], []
    for a, b in itertools.combinations(contenders, 2):
        for s in range(seeds):
            jobs.append((a, b, seed0 + s)); meta.append((a, b))
            jobs.append((b, a, seed0 + s)); meta.append((b, a))
    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_play_pair, jobs))

    wins = {p: 0.0 for p in contenders}
    games = {p: 0 for p in contenders}
    banks = {p: [] for p in contenders}
    for (left, right), (r0, r1) in zip(meta, results):
        games[left] += 1; games[right] += 1
        banks[left].append(r0); banks[right].append(r1)
        if r0 > r1:
            wins[left] += 1
        elif r1 > r0:
            wins[right] += 1
        else:
            wins[left] += 0.5; wins[right] += 0.5

    rows = []
    for info, path in built:
        rows.append({"info": info, "path": path, "win": wins[path] / max(1, games[path]),
                     "bank": statistics.mean(banks[path]) if banks[path] else 0.0,
                     "games": games[path]})
    for path in (extra or []):
        rows.append({"info": None, "path": path, "win": wins[path] / max(1, games[path]),
                     "bank": statistics.mean(banks[path]) if banks[path] else 0.0,
                     "games": games[path]})
    rows.sort(key=lambda r: (-r["win"], -r["bank"]))

    if verbose:
        print(f"{'field donor':<22}{'market donor':<22}{'win%':>7}{'mean bank':>12}{'n':>4}")
        print("-" * 68)
        for r in rows:
            if r["info"]:
                f, m = r["info"]["field"], r["info"]["market"]
                a = f"{f.get('team','?')[:20]}"
                b = f"{m.get('team','?')[:20]}"
                mark = "  <- same donor" if f["id"] == m["id"] else ""
            else:
                a, b, mark = os.path.basename(r["path"])[:20], "(incumbent)", ""
            print(f"{a:<22}{b:<22}{100 * r['win']:>6.0f}%{r['bank']:>12,.0f}"
                  f"{r['games']:>4}{mark}")

    best = rows[0]
    if best["info"] is None:
        if verbose:
            print("\nno splice beat the incumbent -- nothing written.")
        return None
    if out:
        import shutil as _sh
        _sh.copy(best["path"], out)
        f, m = best["info"]["field"], best["info"]["market"]
        if verbose:
            print(f"\nwrote {os.path.relpath(out, ROOT)}")
            print(f"  field channel  {f['id']}  {f.get('team')} (rank {f.get('rank')})")
            print(f"  market channel {m['id']}  {m.get('team')} (rank {m.get('rank')})")
            print(f"  {100 * best['win']:.0f}% over {best['games']} games")
    return best


def _play_pair(job):
    left, right, seed = job
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": 720, "seed": seed, "actTimeout": 60,
        "runTimeout": 100000}, debug=False)
    env.run([os.path.abspath(left), os.path.abspath(right)])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def list_routes():
    idx = load_index()
    for rid, rec in sorted(idx["routes"].items(),
                           key=lambda kv: -kv[1].get("bank", 0)):
        print(f"{rid:<18} {rec.get('window', '?'):<11} {rec.get('team', '?')[:24]:<26}"
              f" ${rec.get('bank', 0):>10,.0f}  {rec.get('date', '')}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--splice", metavar="FIELD:MARKET",
                    help="build one route from another's field channel and a "
                         "second's market channel")
    ap.add_argument("--splice-search", action="store_true",
                    help="try field x market combinations and keep the winner")
    ap.add_argument("--n-field", type=int, default=4)
    ap.add_argument("--n-market", type=int, default=4)
    ap.add_argument("--select", action="store_true",
                    help="build the top candidates and pick the winner by "
                         "playing them, rather than by recorded bank")
    ap.add_argument("--top-n", type=int, default=6, help="--select: candidates")
    ap.add_argument("--seeds", type=int, default=2, help="--select: seeds per pair")
    ap.add_argument("--vs", nargs="*", default=None,
                    help="--select: incumbents to include in the tournament")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--build", metavar="ROUTE",
                    help="render a route into an agent. Takes a route id, a "
                         "team name, or 'best'.")
    ap.add_argument("--out", default=os.path.join(ROOT, "agents", "v15_route.py"))
    ap.add_argument("--weed-catchup", type=int, default=8,
                    help="turns the route may replay one behind after a DIG")
    ap.add_argument("--endgame-pull", action="store_true",
                    help="in the last days, sell held stock ahead of schedule")
    ap.add_argument("--floor-guard", action="store_true",
                    help="hold a SELL back when the market is already floored")
    ap.add_argument("--mirror-tiebreak", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="near-mirror market relay v2: measured +202/game vs a "
                         "clone of our own base, byte-identical outside the "
                         "mirror (2026-08-11). On by default; "
                         "--no-mirror-tiebreak disables")
    ap.add_argument("--guard-ratio", type=float, default=0.0,
                    help="floor-guard trigger as a fraction of base price; "
                         "0 keeps the pure $1-floor trigger")
    ap.add_argument("--swap-advance", action="store_true",
                    help="while the guard holds a depressed product, advance "
                         "an equal value of an already-scheduled healthy SELL")
    ap.add_argument("--premium-lead", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="one-turn conservation sell lead on WOOL/MILK/MELON/"
                         "STRAWBERRY vs ANY opponent (C94/V16-RC5 class), "
                         "demand-parity gated, repaid via the relay ledger. "
                         "OFF by default until paired evidence lands")
    ap.add_argument("--deposit-advance", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="let the premium lead convert ONE idle shed-adjacent "
                         "carrier to DROP so the pulled item is sellable this "
                         "turn (requires --premium-lead). OFF until paired "
                         "evidence lands")
    ap.add_argument("--floor-seller", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="mid-game floored premium seller (day 14+, per-item "
                         "price floors, free slots only, ledger-subtracted; "
                         "disc 737879 class). OFF until paired evidence lands")
    ap.add_argument("--no-impact-slots", action="store_true",
                    help="build without price-impact SELL ranking (for an A/B)")
    ap.add_argument("--top", type=int, default=200,
                    help="how many leaderboard teams to mine routes from")
    ap.add_argument("--per-team", type=int, default=3,
                    help="routes per team; 200x3 needs ~300 usable downloads")
    ap.add_argument("--max-gb", type=float, default=8.0, help="download budget")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel downloads; 20 drops the Kaggle connection")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--breadth", action="store_true",
                    help="identifier-coverage mine: widen to --top 400 teams "
                         "at --min-pct 0.10 (the mid-ladder lookalike zone "
                         "the classifier confuses), fewer routes per team. "
                         "Overridden by explicit --top/--min-pct/--per-team.")
    ap.add_argument("--min-pct", type=float, default=0.35,
                    help="keep episodes in the top fraction of THEIR OWN day. "
                         "Absolute floors reject the archive as the field's "
                         "rating level drifts up; 0 disables.")
    ap.add_argument("--rescan", action="store_true",
                    help="re-consider episodes already seen (use after "
                         "loosening a filter)")
    ap.add_argument("--min-rating", type=float, default=None,
                    help="refuse routes from episodes rated below this. "
                         "Default: the live score at rank --top, so the floor "
                         "tracks the field instead of going stale. 0 disables.")
    ap.add_argument("--clean-stage", action="store_true",
                    help="remove any replay left behind by an interrupted run")
    args = ap.parse_args()

    if args.clean_stage:
        if os.path.isdir(STAGE):
            n = len(glob.glob(os.path.join(STAGE, "**", "*.json"), recursive=True))
            shutil.rmtree(STAGE)
            print(f"removed {n} staged replay(s)")
        return 0
    if args.splice_search:
        extra = [os.path.join(ROOT, v) for v in (args.vs or [])]
        splice_search(n_field=args.n_field, n_market=args.n_market,
                      seeds=args.seeds, workers=args.workers, out=args.out,
                      extra=[e for e in extra if os.path.exists(e)])
        return 0
    if args.splice:
        import base64, datetime as _dt, zlib, tape_runtime
        if args.endgame_pull or args.swap_advance:
            raise SystemExit(
                "endgame_pull/swap_advance repaid themselves through the "
                "retired _safe_market clamp; enabling them would sell pulled "
                "quantities twice. Reimplement explicit subtraction first.")
        fid, _, mid = args.splice.partition(":")
        tape, _n = splice(fid, mid)
        payload = base64.b85encode(zlib.compress(
            json.dumps(tape, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
        idx = load_index()
        f = idx["routes"].get(fid, {}); m = idx["routes"].get(mid, {})
        src = tape_runtime.TEMPLATE.format(
            label=os.path.basename(args.out), route_id=f"{fid}+{mid}",
            team=f"field {f.get('team','?')} / market {m.get('team','?')}",
            episode=f"{f.get('episode','?')}+{m.get('episode','?')}",
            seat=f.get("seat", 0), bank=f.get("bank", 0.0),
            opp_bank=m.get("bank", 0.0), built=_dt.date.today().isoformat(),
            payload=payload, weed_catchup=args.weed_catchup,
            impact_slots=not args.no_impact_slots,
            mirror_tiebreak=args.mirror_tiebreak,
            floor_guard=args.floor_guard,
            endgame_pull=args.endgame_pull,
            guard_ratio=args.guard_ratio,
            swap_advance=args.swap_advance,
            sell_first=True, feed_pin=True,
            premium_lead=args.premium_lead,
            deposit_advance=args.deposit_advance,
            floor_seller=args.floor_seller,
            branchpack="")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
        print(f"wrote {os.path.relpath(args.out, ROOT)} "
              f"({os.path.getsize(args.out):,} bytes)")
        print(f"  field  {fid}  {f.get('team','?')}")
        print(f"  market {mid}  {m.get('team','?')}")
        return 0
    if args.select:
        extra = [os.path.join(ROOT, v) for v in (args.vs or [])]
        select(n=args.top_n, seeds=args.seeds, workers=args.workers,
               out=args.out, extra=[e for e in extra if os.path.exists(e)])
        return 0
    if args.build:
        build(args.build, args.out, weed_catchup=args.weed_catchup,
              impact_slots=not args.no_impact_slots,
              mirror_tiebreak=args.mirror_tiebreak,
              floor_guard=args.floor_guard,
              endgame_pull=args.endgame_pull,
              guard_ratio=args.guard_ratio,
              swap_advance=args.swap_advance,
              premium_lead=args.premium_lead,
              deposit_advance=args.deposit_advance,
              floor_seller=args.floor_seller)
        return 0
    if args.mine:
        top = args.top
        per_team = args.per_team
        min_pct = args.min_pct
        if args.breadth:
            # Explicit values win; breadth only fills the unchanged defaults.
            if top == 200:
                top = 400
            if per_team == 3:
                per_team = 2
            if min_pct == 0.35:
                min_pct = 0.10
        mine(top=top, per_team=per_team, max_gb=args.max_gb,
             jobs=args.jobs, days=args.days, min_rating=args.min_rating,
             min_pct=min_pct, rescan=args.rescan)
        return report()
    if args.list:
        return list_routes()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
