# Delta pipeline: new Kaggle daily + GM data → sanitized corpus (2026-09-25)

A single command pulls whatever is new since the last run into the RL corpus. It extracts the
data on Kaggle, downloads it, files it, dedups it and recomputes ranks:

```powershell
python python/delta.py run          # or: .\scripts\delta.ps1  (same, with logging)
```

It is safe to re-run at any time. It only does what is new, and it resumes an interrupted run
from the recorded phase.

## Pieces

| file | role |
|---|---|
| `kaggle/delta/notebook_template.py` | The extraction notebook (one code cell). It runs the Rust `corpus-extract` (dataset `debmalya84/kaggriculture-rl-bin`) in slim mode, keeps engine 1.32.7 only, writes `slim/` plus **`slim/manifest.txt`** (every output file) and `run.json` (rc, files, bytes, **`finished_utc`**, plan). |
| `kaggle/kernels/krl-delta-daily/`, `krl-delta-gm/` | Generated from the template on every run. Stable slugs, so each run is a new version of the same two **private** notebooks. |
| `python/delta.py` | The orchestrator (below). |
| `python/corpus_index.py` | Builds the dedup + rank index (see `docs/corpus_index.md`). Also run by `delta.py`. |
| `scripts/delta.ps1` | Wrapper that runs `delta.py run` and appends to `data/kaggle_out/delta.log`. |
| `data/kaggle_out/delta_state.json` | State: GM version and per-day coverage already extracted, daily days done, each kernel's phase, history. |
| `data/kaggle_out/delta_report.md` | One section per run: what was filed and pruned, and the index headline. |

## What `run` does

1. **Plan**
   - *Daily*: every `kaggle/kaggriculture-episodes-YYYY-MM-DD` dataset from 2026-08-15 on that
     we have not filed. It must have files and must have been untouched for 30 min, since new
     ones are still uploading around 00:05 UTC.
   - *GM*: only when `georgymamarin/kaggriculture-episodes` has a **newer version** than the one
     we extracted. Then every day whose `replay_coverage` (the dataset's `daily_stats.csv`) rose by
     ≥ 2 points since our extraction, or which we lack, is re-extracted as one `--since/--until`
     window. GM fills in recent days over several versions: the version of 2026-09-23 had
     21 Sep at 42%, 22 Sep at 18% and 23 Sep at 0.1%.
2. **Push**: generate the notebook and push it. If Kaggle's 5 CPU sessions are busy, it retries
   every 2 min.
3. **Wait**: poll the kernel status every 2 min, for up to 8 h.
4. **Download**: fetch `run.json` and `manifest.txt` first. Refuse the output if the run failed or
   its `finished_utc` predates our push, so a stale version is never filed. Then fetch exactly
   the files the manifest lists that are missing locally, in batches of 40 by `--file-pattern`,
   for up to 8 rounds.
   - This replaces the old whole-output page walk, which stalled at 3100 of 3228 files on
     GM Sep 1–14.
5. **File**: move the parts to `data/slim/s1/source={official|gm}/date=…/part-<RUNID>-….parquet`
   and the ledgers to `ledger/`. The run's JSON and log go to `runs/`.
   RUNID = `yyyymmddThhmmZ-krl-delta-<kind>-<sha7>`.
6. **Prune**: for every day this run filed, delete the older parts of that day. A GM re-extract
   comes from a fuller GM version; a daily re-run re-reads the same dataset.
7. **Dedup + rerank**: `corpus_index.py` rebuilds `data/slim/s1/index/episodes.parquet`.
   - One row per `episode_id`; the newest run wins.
   - Per seat: `rating_pre`, `rating_post`, `band` and daily leaderboard `rank` / `rank_pct`.
   - Validation episodes are flagged.
8. **Report**: append to `delta_report.md`.

## Sanitize rules (what reaches the corpus)
- Engine 1.32.7 only. The ladder cut over on 2026-08-15 at 01:38–01:41 UTC; the extractor drops
  other versions per episode (GM: `episode_features.csv`; daily: the replay's `module_version`).
- No duplicate episodes in the index (key `episode_id`; seeds are NOT unique).
- A re-extracted day keeps only its newest parts on disk.
- Validation self-play episodes are kept but flagged (`episode_type`).

## Commands

```
python python/delta.py bootstrap   # ONE-TIME baseline (done 2026-09-25 03:04Z: GM version 2026-09-23 23:22, 40 GM days, 41 daily days)
python python/delta.py plan        # dry run: what would be extracted
python python/delta.py status      # state + both notebooks' Kaggle status
python python/delta.py run         # the whole cycle (resumable)
```

## Scheduling (not installed)
- New daily datasets appear around 00:05 UTC; GM publishes irregularly (a new version every
  ~1–2 days).
- A daily run at about 01:00 UTC catches the daily set. A second run at about 13:00 UTC catches
  GM updates.
- Suggested Task Scheduler action: `powershell -NoProfile -ExecutionPolicy Bypass -File
  D:\codebase\kaggriculture\kaggriculture-rl\scripts\delta.ps1`. Installing a scheduled task is
  left to the operator.

## Failure modes
- **Kernel ERROR/CANCELLED**: the kind is marked `failed` in the state. Fix the cause, set its
  phase back to `planned` in `delta_state.json`, and run again.
- **Timeout (8 h)**: the kernel stays `pushed`; the next `run` resumes the wait.
- **Files still missing after 8 rounds**: the run raises. Re-running resumes the download (the
  phase stays `pushed`).
- **The old per-range notebooks** (`krl-daily-a..f`, `krl-gm-aug/sep1/sep2/aug15`) and
  `scripts/backfill_watch.ps1` remain for one-off historical backfills; the delta pipeline
  supersedes them for new data.

## Shared Kaggle machinery (2026-09-25)

The push / status / manifest-download / stale-output logic now lives in `python/kjob.py`, shared with
`python/league_kaggle.py` and `python/gate_kaggle.py` (see docs/queue.md). `delta.py` behaviour is unchanged;
it runs as queue job Q01 every 12 h and its completion re-runs features -> day_obs -> cache -> BC fine-tune.
