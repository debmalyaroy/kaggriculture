@echo off
rem Recurring same-day episode scrape (KaggricultureSameDay) -- HOURLY.
rem
rem The public episode datasets lag the ladder by ~2 days, but replays are
rem servable immediately (kaggleusercontent fast path inside sameday.py).
rem This harvests current episode ids from the top of the leaderboard through
rem the authenticated API, replays them within a byte budget, and folds the
rem routes into the index dated TODAY.
rem
rem HOURLY MODE (2026-08-12): small sips all day long, so the 04:xx release
rem finds the index already current and its fetch stage is a near no-op
rem (refresh_cycle --no-fetch relies on this). Budget is per-run small; the
rem index dedupes, so every run is delta-only and duplicates are never
rem re-downloaded (grab() also reuses _stage leftovers).
rem
rem Guard: skip while a release is running -- concurrent index writes are
rem lock-safe (save_index merges), but the bandwidth contention is pointless.
rem
rem Self-contained in the same style as daily_release.bat: absolute python
rem path, quoted env, explicit Kaggle credential dir, append-only log.
cd /d D:\codebase\kaggriculture
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KAGGLE_CONFIG_DIR=C:\Users\debma\.kaggle"
if not exist data\logs mkdir data\logs
if not exist data\sameday mkdir data\sameday
set "LOG=data\logs\sameday_scrape.log"

rem ---- release-in-progress guard --------------------------------------------
rem Logic lives in release_running.ps1: inline batch->powershell quoting broke
rem ('^|' + escaped quotes mangled -> exit 255 before the first log line,
rem observed on every hourly run 07:35-10:35 on 2026-08-13).
set "RELCOUNT=0"
for /f %%N in ('powershell -NoProfile -ExecutionPolicy Bypass -File scripts\release_running.ps1') do set "RELCOUNT=%%N"
if not "%RELCOUNT%"=="0" (
    echo ===== sameday_scrape SKIPPED - release running - %DATE% %TIME% ===== >> "%LOG%"
    exit /b 0
)
echo ===== sameday_scrape start %DATE% %TIME% ===== >> "%LOG%"

rem 1. Harvest current episode ids. Window 4h with hourly runs = deliberate
rem    overlap, so an episode can never slip between two sips. (The window was
rem    3h against a timezone bug that read every episode 5.5h old -- fixed
rem    2026-08-14 in leaderboard_harvest._epoch; every sip since 08-12 had
rem    silently returned 0 ids.)
"C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\data\leaderboard_harvest.py --top 200 --hours 4 --out data\sameday\ids_sched.json >> "%LOG%" 2>&1

rem 2. Replay + ingest. Operator order 2026-08-14: DO NOT budget the size --
rem    disk is plentiful and the fetch is delta-only anyway (the index
rem    dedupes). --max-gb stays only as a runaway-fetch safety valve, set
rem    where an honest hourly delta can never reach it.
"C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\data\sameday.py --ids-file data\sameday\ids_sched.json --max-gb 20 --jobs 6 --min-score 2600 >> "%LOG%" 2>&1

rem 3. Our own pair's new games (delta; tape material stays current).
"C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\data\ourgames.py --all >> "%LOG%" 2>&1

rem 3a. Rating snapshot: one row per active submission into
rem     data\lb\rating_track.jsonl -- the dashboard's rating-vs-time curves.
rem     One cheap submissions call; no replays, no quota.
"C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\measure\rating_track.py >> "%LOG%" 2>&1

rem 3b. Collapse alarm: trailing-window health of both active submissions.
rem     ALARM lines also append to data\logs\collapse_alarm.log.
"C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\measure\collapse_alarm.py >> "%LOG%" 2>&1

rem 4. EVENING (21:xx) only: historical-archive backfill. It lived at 02:xx
rem    until 2026-08-16, which burned the ROLLING replay quota right before
rem    the 04:30 release -- two mornings in a row the release found its
rem    loss-tape fetches 429'd (Aug 15 HELD; Aug 16 tape-fallback + a 75-min
rem    throttled fetch crawl on the rerun). 21:xx gives the window 7+ hours
rem    to recover before the release, and sameday._quota_allow() now also
rem    enforces a hard 24h budget plus an 03:00-06:00 quiet window on every
rem    non-release replay download, so this class of morning cannot recur.
rem    Old days feed the identifier, never crown candidates -- the evening
rem    slot loses nothing.
set "HH=%TIME:~0,2%"
if "%HH%"==" 2" set "HH=02"
rem Nightly pre-ranker recall probe: the funnel gate closes itself when the
rem trailing cycle-audit dips below 0.75, but a closed gate runs no funnel
rem and so generates no new audit points -- without this probe it could
rem never reopen. Engine idle, ~10 min. NOTE: comments must stay OUTSIDE the
rem parenthesized block below -- a ')' inside a block comment closes the
rem block early in cmd.
if "%HH%"=="21" (
    "C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\data\routes.py --mine --top 200 --days 0 --max-gb 12 --jobs 8 >> "%LOG%" 2>&1
    "C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\data\routes.py --mine --breadth --max-gb 8 --jobs 8 >> "%LOG%" 2>&1
    "C:\ProgramData\anaconda3\python.exe" -X utf8 src\kaggriculture\engine\rust_prerank.py --recall 8 >> "%LOG%" 2>&1
)

rem 1.32.7 rollout watcher: scans the replays just fetched; performs the
rem atomic engine swap the hour the ladder flips. No-op afterward.
"C:\ProgramData\anaconda3\python.exe" -X utf8 scripts\engine_swap_1327.py --check >> "%LOG%" 2>&1

rem Heartbeat: the release's fetch-skip decision keys on this file's mtime,
rem NOT on index.json (an empty-but-successful sip leaves the index mtime old
rem and used to trigger a pointless 75-minute release-time re-fetch).
echo %DATE% %TIME% > data\sameday\heartbeat.txt

echo ===== sameday_scrape exit %ERRORLEVEL% at %DATE% %TIME% ===== >> "%LOG%"
