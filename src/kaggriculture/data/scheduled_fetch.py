"""Incremental, timestamped episode fetch — safe to run on a schedule.

Designed to be run hourly. Each run:

  1. reads `data/fetch_state.json` for where the last run stopped
  2. pulls any *new* daily top-episode dataset published since then
  3. pulls your own new episodes played since the last run's timestamp
  4. featurises each replay and appends rows to `data/episodes.csv`
  5. writes a new state file and appends to `data/fetch_log.csv`

Everything is keyed on episode id, so an interrupted run costs nothing and a
double-run duplicates nothing. Replays are deleted after featurising, so disk
stays flat regardless of how long the schedule has been running.

Note on cadence: Kaggle publishes the *top-episode* dataset once a day, so the
hourly value comes mostly from your own agents' new episodes. The daily pull is
skipped automatically until a genuinely new date appears.

    python -m kaggriculture.data.scheduled_fetch                 # one incremental run
    python -m kaggriculture.data.scheduled_fetch --per-day 40 --own 40
    python -m kaggriculture.data.scheduled_fetch --status        # what would happen, no fetch
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.data.episodes as ep  # noqa: E402
import kaggriculture.data.build_dataset as bd  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402

DATA = os.path.join(ROOT, "data")
STATE = os.path.join(DATA, "fetch_state.json")
CSV = os.path.join(DATA, "episodes.csv")
LOG = os.path.join(DATA, "fetch_log.csv")
WORK = os.path.join(ROOT, "data", "episodes", "sched")


def now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_run": None, "last_daily_date": None, "seen": []}


def save_state(st):
    os.makedirs(DATA, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)          # atomic: a killed run never corrupts state


def log_run(row):
    os.makedirs(DATA, exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_at", "daily_date", "top_new",
                                          "own_new", "rows_added", "seconds", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def append_rows(rows):
    os.makedirs(DATA, exist_ok=True)
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bd.feature_names())
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def ingest(path, date, seen, added):
    eid = os.path.basename(path).replace(".json", "")
    if eid in seen:
        return 0
    try:
        rows = bd.rows_from_replay(path, date)
    except Exception as exc:                                       # noqa: BLE001
        pr.warn(f"parse {eid}: {exc}", 3)
        return 0
    append_rows(rows)
    seen.add(eid)
    added.append(eid)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-day", type=int, default=30, help="top episodes per new day")
    ap.add_argument("--own", type=int, default=25, help="own episodes per run")
    ap.add_argument("--days", type=int, default=3, help="daily datasets to consider")
    ap.add_argument("--top-n", type=int, default=0,
                    help="only keep episodes played by the top-N leaderboard teams")
    ap.add_argument("--jobs", type=int, default=6,
                    help="parallel replay downloads (each spawns its own kaggle "
                         "CLI process; 6-10 is the sweet spot)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    t0 = now()
    pr.reset()
    st = load_state()
    seen = set(st.get("seen", []))
    pr.log(f"scheduled fetch  ({t0.isoformat()})")
    pr.log(f"last run       : {st.get('last_run') or 'never'}", 1)
    pr.log(f"last daily set : {st.get('last_daily_date') or 'none'}", 1)
    pr.log(f"episodes known : {len(seen):,}", 1)
    pr.log(f"plan           : up to {args.per_day} episodes/day over "
           f"{args.days} day(s), plus up to {args.own:,} of your own", 1)
    pr.log(f"parallelism    : {args.jobs} concurrent downloads", 1)
    if args.status:
        pr.log("--status: nothing fetched", 1)
        return 0

    os.makedirs(WORK, exist_ok=True)
    top_new = own_new = rows_added = 0
    note = ""
    wanted = set()
    if args.top_n:
        pr.log(f"resolving the top-{args.top_n} leaderboard teams ...", 1)
        wanted = {t.lower() for t in ep.leaderboard_top(args.top_n)}
        if not wanted:
            note = "leaderboard names unresolved; no team filter applied"

    # ---- 1. new daily top-episode dataset --------------------------------
    pr.log("fetching the daily-episode index ...", 1)
    try:
        index = ep.fetch_index(WORK)
        pr.log(f"index lists {len(index)} published day(s)", 1)
    except ep.KaggleError as exc:
        note = f"index unavailable: {exc}".replace("\n", " ")[:180]
        pr.warn(note, 1)
        index = []

    added_ids = []
    fresh_days = [r for r in index
                  if r.get("date")
                  and not (st.get("last_daily_date")
                           and r["date"] <= st["last_daily_date"])]
    if index and not fresh_days:
        pr.log(f"no daily dataset newer than {st.get('last_daily_date')} "
               f"-- nothing new to pull on the ladder side", 1)
    for day_row in fresh_days:
        date = day_row.get("date")
        dataset = day_row.get("daily_dataset_slug") or f"kaggriculture-episodes-{date}"
        if "/" not in dataset:
            dataset = f"kaggle/{dataset}"
        day_dir = os.path.join(WORK, date)
        pr.log(f"day {date}: downloading manifest ...", 1)
        try:
            man = ep._download_dataset_file(dataset, "manifest.csv", day_dir)
        except ep.KaggleError as exc:
            pr.warn(str(exc).splitlines()[0], 2)
            continue
        if not man:
            pr.warn(f"{date}: no manifest.csv in {dataset}", 2)
            continue
        manifest = ep._read_csv(man)
        manifest.sort(key=ep._score_of, reverse=True)
        top_score = ep._score_of(manifest[0]) if manifest else 0.0
        pr.log(f"day {date}: {len(manifest):,} episodes listed, "
               f"top score {top_score:.1f}; taking up to {args.per_day}", 1)
        # Pick the whole batch first, then fetch it in parallel. Downloading
        # one at a time wasted ~1-2s of kaggle-CLI startup per 27 MB file --
        # strictly serial dead time that dwarfed the transfer on a fast link.
        skipped_seen = skipped_team = 0
        batch = []
        for mrow in manifest:
            if len(batch) >= args.per_day:
                break
            fname = ep._episode_file(mrow)
            if not fname:
                continue
            if fname.replace(".json", "") in seen:
                skipped_seen += 1
                pr.heartbeat(f"skipping already-ingested episodes "
                             f"({skipped_seen} so far)", key="skip-seen")
                continue
            if wanted:
                teams = {t.lower() for t in ep.episode_teams(mrow)}
                # If the manifest does not expose team names we cannot filter
                # before downloading; fall back to taking the highest-rated
                # episodes, which is what the dataset is ranked by anyway.
                if teams and not (teams & wanted):
                    skipped_team += 1
                    continue
            batch.append((fname, mrow))

        pr.log(f"day {date}: fetching {len(batch)} episodes "
               f"{args.jobs} at a time", 1)
        tick = pr.Ticker(total=len(batch), label="ladder episodes", indent=2)
        taken = 0

        def _grab(item):
            """Download one replay. Runs in a worker thread; no shared state."""
            fname, mrow = item
            local = os.path.join(day_dir, fname)
            if os.path.exists(local):          # a previous run died mid-flight
                return fname, mrow, local, None
            try:
                ep._download_dataset_file(dataset, fname, day_dir)
            except ep.KaggleError as exc:
                return fname, mrow, None, str(exc).splitlines()[0]
            return fname, mrow, (local if os.path.exists(local) else None), None

        # Downloads run ahead in the pool while this thread featurises and
        # deletes whatever has already landed -- network and CPU overlap
        # instead of taking turns.
        if batch:
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                futures = [pool.submit(_grab, item) for item in batch]
                for fut in as_completed(futures):
                    fname, mrow, local, err = fut.result()
                    if err:
                        pr.warn(f"{fname}: {err}", 3)
                        continue
                    if not local:
                        continue
                    mb = os.path.getsize(local) / 1e6
                    n_rows = ingest(local, date, seen, added_ids)
                    rows_added += n_rows
                    taken += 1
                    top_new += 1
                    tick.step(f"{fname}  {mb:.1f} MB",
                              extra=f"score {ep._score_of(mrow):.0f}  +{n_rows} rows")
                    try:
                        os.remove(local)
                    except OSError:
                        pass
        st["last_daily_date"] = date
        tick.done(f"[skipped {skipped_seen} already known"
                  + (f", {skipped_team} off-leaderboard" if skipped_team else "")
                  + "]")

    # ---- 2. our own new episodes ----------------------------------------
    pr.log("fetching your own episodes ...", 1)
    try:
        mine = ep.fetch_own_episodes(WORK, limit=args.own, verbose=True,
                                     jobs=args.jobs)
    except Exception as exc:                                       # noqa: BLE001
        mine = []
        note = (note + f" | own: {exc}")[:180]
        pr.warn(f"own episodes: {exc}", 1)
    for path in mine:
        n = ingest(path, t0.date().isoformat(), seen, added_ids)
        if n:
            own_new += 1
            rows_added += n
        try:
            os.remove(path)
        except OSError:
            pass
    pr.log(f"own new episodes ingested: {own_new}", 1)

    # ---- 3. persist ------------------------------------------------------
    st["last_run"] = t0.isoformat()
    st["seen"] = sorted(seen)[-20000:]        # bounded; ids are ~8 bytes each
    save_state(st)
    secs = (now() - t0).total_seconds()
    log_run({"run_at": t0.isoformat(), "daily_date": st.get("last_daily_date"),
             "top_new": top_new, "own_new": own_new, "rows_added": rows_added,
             "seconds": int(secs), "note": note})
    pr.log(f"done: +{rows_added:,} rows ({top_new} ladder + {own_new} own "
           f"episodes) in {pr.hms(secs)} -> {os.path.relpath(CSV, ROOT)}")
    if note:
        pr.log(f"note: {note}", 1)
    if rows_added:
        pr.log("refresh the report:  python -m kaggriculture.pipeline.eda_report", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
