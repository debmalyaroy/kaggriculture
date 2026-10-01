"""Stage 1 of the pipeline: pull episode replays off Kaggle.

Two sources, both official:

* **Top-player games** -- Kaggle publishes a daily dataset of the highest-rated
  episodes (`kaggle/kaggriculture-episodes-<date>`), announced in competition
  discussion 731215: "Each day we order episodes by the average rating of the
  agents playing, then download up to 20 GB of replays and make a new daily
  dataset." An index dataset (`kaggle/kaggriculture-episodes-index`) lists the
  daily datasets with their episode counts and score summaries.

* **Your own games** -- `kaggle competitions submissions` -> `episodes` ->
  `replay`, i.e. the episodes your own submissions have actually played.

Size discipline matters: each replay is ~27 MB and a daily dataset is ~21 GB
for ~790 episodes. Nothing here ever downloads a whole daily dataset; it reads
that dataset's `manifest.csv` (a few hundred KB), ranks the episodes, and pulls
individual `<episode_id>.json` files until a byte budget is reached.

Requires the `kaggle` CLI, authenticated. This module never touches your
credentials -- it shells out to `kaggle`, which reads them itself.
"""
from kaggriculture.paths import ROOT
import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import kaggriculture.pipeline.progress as pr  # noqa: E402

INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
DAILY_DATASET = "kaggle/kaggriculture-episodes-{date}"
COMPETITION = "kaggriculture"
BYTES_PER_REPLAY = 27 * 1024 * 1024      # observed average


class KaggleError(RuntimeError):
    pass


_KAGGLE_CMD = None


def _probe(cmd):
    """True if `cmd` runs the Kaggle CLI. Probe with --version, then --help:
    some builds lack --version but every build has --help, and --help needs no
    credentials."""
    for flag in ("--version", "--help"):
        try:
            r = subprocess.run(cmd + [flag], capture_output=True, text=True,
                               timeout=60)
        except Exception:                                          # noqa: BLE001
            return False
        # returncode only: "No module named kaggle" contains the word "kaggle"
        # and would otherwise read as success.
        if r.returncode == 0:
            return True
    return False


def kaggle_cmd():
    """How to invoke the Kaggle CLI on this machine.

    `pip install kaggle` drops kaggle.exe into a Scripts directory Windows very
    often does not have on PATH -- and when pip says "Defaulting to user
    installation" that directory is the *user* one
    (%APPDATA%\\Python\\PythonXY\\Scripts), not the interpreter's own.

    Note that `python -m kaggle` does NOT work -- the package ships no
    __main__.py ("'kaggle' is a package and cannot be directly executed").

    We try, in order: PATH; the Scripts dir belonging to *this* interpreter;
    the per-user Scripts dir; sysconfig's script paths (user and default
    schemes); `python -m kaggle.cli`; and finally the installed `kaggle`
    console-script entry point resolved through importlib.metadata, which
    works regardless of which module that entry point happens to live in.
    """
    global _KAGGLE_CMD
    if _KAGGLE_CMD is not None:
        return _KAGGLE_CMD

    candidates = []
    exe = shutil.which("kaggle")
    if exe:
        candidates.append([exe])

    names = ["kaggle.exe", "kaggle"] if os.name == "nt" else ["kaggle"]
    seen = set()
    dirs = [os.path.join(os.path.dirname(sys.executable), "Scripts"),
            os.path.dirname(sys.executable)]

    # Per-user install ("Defaulting to user installation because normal
    # site-packages is not writeable") puts the scripts under USER_BASE, not
    # under the interpreter, e.g. %APPDATA%\\Python\\Python313\\Scripts.
    try:
        import site
        base = getattr(site, "USER_BASE", None)
        if base:
            dirs.append(os.path.join(base, "Scripts"))
            dirs.append(os.path.join(base, "bin"))
        usp = site.getusersitepackages()
        if isinstance(usp, str):
            usp = [usp]
        for d in usp or []:
            dirs.append(os.path.join(os.path.dirname(d), "Scripts"))
            dirs.append(os.path.join(os.path.dirname(d), "bin"))
    except Exception:                                              # noqa: BLE001
        pass

    try:
        import sysconfig
        available = set(sysconfig.get_scheme_names())
        schemes = [s_ for s_ in ("nt_user", "posix_user") if s_ in available]
        for key in ("scripts", "purelib"):
            for scheme in [None] + schemes:
                try:
                    d = (sysconfig.get_path(key) if scheme is None
                         else sysconfig.get_path(key, scheme=scheme))
                except Exception:                                  # noqa: BLE001
                    continue
                if not d:
                    continue
                dirs.append(d if key == "scripts"
                            else os.path.join(os.path.dirname(d), "Scripts"))
    except Exception:                                              # noqa: BLE001
        pass
    for d in dirs:
        if not d or d in seen:
            continue
        seen.add(d)
        for n in names:
            cand = os.path.join(d, n)
            if os.path.exists(cand):
                candidates.append([cand])

    candidates.append([sys.executable, "-m", "kaggle.cli"])
    # Last resort: run whatever the `kaggle` console-script entry point points
    # at. Version-agnostic -- it does not care which module holds the callable.
    candidates.append([sys.executable, "-c",
                       "import sys;from importlib.metadata import entry_points;"
                       "e=[x for x in entry_points(group='console_scripts')"
                       " if x.name=='kaggle'];"
                       "sys.exit('no kaggle console script') if not e"
                       " else e[0].load()()"])

    for cmd in candidates:
        if _probe(cmd):
            _KAGGLE_CMD = cmd
            return cmd

    raise KaggleError(
        "the Kaggle CLI is installed but could not be invoked.\n"
        "  tried: " + " | ".join(" ".join(c) for c in candidates) + "\n"
        "  `python -m kaggle` never works -- the package has no __main__.py.\n"
        "  Check the install belongs to THIS interpreter (" + sys.executable +
        "):\n"
        "        python -c \"import kaggle;print(kaggle.__file__)\"\n"
        "  and that the CLI module imports:\n"
        "        python -m kaggle.cli --version")


def _run(args, timeout=900):
    """Run a kaggle CLI command, returning stdout.

    `args` starts with the literal "kaggle"; it is swapped for whatever actually
    works on this machine (see kaggle_cmd).
    """
    if args and args[0] == "kaggle":
        args = kaggle_cmd() + list(args[1:])
    pr.vlog("$ " + " ".join(args), 2)
    t0 = time.time()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise KaggleError(
            "the Kaggle CLI is not installed.\n"
            "  pip install kaggle\n"
            "then authenticate (see docs/history/evaluation.md section 3).") from None
    except subprocess.TimeoutExpired:
        raise KaggleError(f"timed out: {' '.join(args)}") from None
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        hint = ""
        low = err.lower()
        if "401" in err or "unauthor" in low or "credential" in low:
            hint = ("\n  -> not authenticated. Run `kaggle auth login`, or put your "
                    "token in ~/.kaggle/access_token (see docs/history/evaluation.md).")
        elif "403" in err or "forbidden" in low:
            hint = ("\n  -> you have not accepted the competition rules. Open "
                    "https://www.kaggle.com/competitions/kaggriculture and click "
                    "'Join Competition'.")
        raise KaggleError(f"`{' '.join(args)}` failed:\n{err}{hint}")
    pr.vlog(f"  -> ok in {time.time() - t0:.1f}s, {len(p.stdout):,} bytes", 2)
    return p.stdout


def _unzip_inplace(directory):
    """Kaggle sometimes wraps single-file downloads in a zip."""
    for name in list(os.listdir(directory)):
        if not name.endswith(".zip"):
            continue
        path = os.path.join(directory, name)
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(directory)
            os.remove(path)
        except zipfile.BadZipFile:
            pass


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _download_dataset_file(dataset, filename, dest_dir):
    """Fetch one file out of a Kaggle dataset. Returns the local path or None.

    A single episode is ~27 MB and the CLI prints nothing until it finishes, so
    announce the intent first -- otherwise a slow link looks like a hang.
    """
    os.makedirs(dest_dir, exist_ok=True)
    pr.vlog(f"downloading {dataset}:{filename}", 3)
    _run(["kaggle", "datasets", "download", "-d", dataset,
          "-f", filename, "-p", dest_dir, "--force"])
    _unzip_inplace(dest_dir)
    path = os.path.join(dest_dir, filename)
    return path if os.path.exists(path) else None


DIAG_DIR = os.path.join(ROOT,
                        ".local", "diag")

# Column names the leaderboard CSV has used across Kaggle CLI versions. Checked
# in order; anything else containing "team" or "name" is picked up by the
# fuzzy pass below.
_TEAM_KEYS = ("teamname", "teamnamenullable", "team", "name", "displayname",
              "teamdisplayname", "submittedby", "username")


def _leaderboard_invocations():
    """Every way the Kaggle CLI has accepted `competitions leaderboard`.

    The competition moved between positional and `-c` across versions, and
    `--csv` is spelled `-v` on some builds. Trying them in order costs one
    fast round-trip each and means this keeps working after a CLI upgrade.
    """
    c = COMPETITION
    return [
        ["kaggle", "competitions", "leaderboard", c, "-s", "--csv"],
        ["kaggle", "competitions", "leaderboard", "-c", c, "-s", "--csv"],
        ["kaggle", "competitions", "leaderboard", c, "--show", "-v"],
        ["kaggle", "competitions", "leaderboard", c, "-s"],
    ]


def _leaderboard_full(n, verbose=True):
    """The whole leaderboard, via the download endpoint. Returns [] on failure.

    `leaderboard --show` is capped at 20 rows however large you ask, so every
    request for a deeper slice silently returned the same 20 teams -- asking for
    the top 200 and getting 20 looks like "only 20 teams qualified" rather than
    "the CLI paginated". The download endpoint has no such cap: it ships a zip
    containing the full public leaderboard CSV, 2,425 rows today.
    """
    import zipfile
    dest = os.path.join(DIAG_DIR, "lb")
    os.makedirs(dest, exist_ok=True)
    try:
        _run(["kaggle", "competitions", "leaderboard", "-c", COMPETITION,
              "--download", "-p", dest], timeout=240)
    except KaggleError as exc:
        pr.vlog(f"leaderboard download failed: {exc}", 2)
        return []
    zips = sorted(glob.glob(os.path.join(dest, "*.zip")),
                  key=os.path.getmtime, reverse=True)
    if not zips:
        return []
    try:
        with zipfile.ZipFile(zips[0]) as z:
            member = next((m for m in z.namelist() if m.endswith(".csv")), None)
            if not member:
                return []
            rows = list(csv.DictReader(
                io.TextIOWrapper(z.open(member), "utf-8-sig")))
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        pr.vlog(f"leaderboard zip unreadable: {exc}", 2)
        return []

    names = []
    for r in rows[:n]:
        name = (r.get("TeamName") or r.get("teamName") or "").strip()
        if name:
            names.append(name)
    if verbose and names:
        pr.log(f"leaderboard: {len(names)} of top-{n} teams resolved from the "
               f"full CSV ({len(rows)} teams total)", 1)
    return names


def leaderboard_rows(n=None, verbose=False):
    """Full leaderboard rows: rank, team, score. [] if it cannot be read.

    Same download as `_leaderboard_full`, but keeps the scores. A miner wants
    the score at rank N to set its own quality floor -- hardcoding one goes
    stale the moment the field moves, and this field moved 517 teams in 22
    hours.
    """
    import zipfile
    dest = os.path.join(DIAG_DIR, "lb")
    os.makedirs(dest, exist_ok=True)
    try:
        _run(["kaggle", "competitions", "leaderboard", "-c", COMPETITION,
              "--download", "-p", dest], timeout=240)
    except KaggleError:
        return []
    zips = sorted(glob.glob(os.path.join(dest, "*.zip")),
                  key=os.path.getmtime, reverse=True)
    if not zips:
        return []
    try:
        with zipfile.ZipFile(zips[0]) as z:
            member = next((m for m in z.namelist() if m.endswith(".csv")), None)
            if not member:
                return []
            raw = list(csv.DictReader(
                io.TextIOWrapper(z.open(member), "utf-8-sig")))
    except (zipfile.BadZipFile, OSError, KeyError):
        return []

    out = []
    for i, r in enumerate(raw, 1):
        name = (r.get("TeamName") or r.get("teamName") or "").strip()
        try:
            score = float(r.get("Score") or r.get("score") or 0)
        except ValueError:
            score = 0.0
        out.append({"rank": i, "team": name, "score": score})
    if verbose:
        pr.log(f"leaderboard: {len(out)} teams, top {out[0]['score']:.1f}", 1)
    return out[:n] if n else out


def _save_diag(name, text):
    """Keep raw CLI output so a parsing failure is debuggable after the fact."""
    try:
        os.makedirs(DIAG_DIR, exist_ok=True)
        path = os.path.join(DIAG_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path
    except OSError:
        return None


def _names_from_csv(text, n):
    """Team names out of CSV output, whatever the header happens to be called."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return [], []
    header = [h for h in (rows[0].keys() or []) if h]
    lower = {h.lower().replace("_", "").replace(" ", ""): h for h in header}

    col = None
    for key in _TEAM_KEYS:
        if key in lower:
            col = lower[key]
            break
    if col is None:                       # fuzzy: any column mentioning a team
        for norm, orig in lower.items():
            if "team" in norm or norm.endswith("name"):
                col = orig
                break
    if col is None:
        return [], header

    names = []
    for r in rows:
        v = (r.get(col) or "").strip()
        if v and v not in names:
            names.append(v)
        if len(names) >= n:
            break
    return names, header


def _names_from_table(text, n):
    """Team names out of the CLI's aligned-table output (no --csv).

    The table looks like:
        teamId  teamName            submissionDate       score
        12345   Some Team Name      2026-08-03 11:02:00  1204.6
    Names can contain spaces, so anchor on the columns either side: drop the
    leading id and the trailing date/score.
    """
    import re
    names = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line or set(line) <= set("- "):
            continue
        if line.lower().lstrip().startswith(("teamid", "rank", "#")):
            continue
        m = re.match(r"^\s*\d+\s+(.*?)\s{2,}\d{4}-\d{2}-\d{2}", line)
        if not m:
            m = re.match(r"^\s*\d+\s+(.*?)\s{2,}[-\d.]+\s*$", line)
        if m:
            v = m.group(1).strip()
            if v and v not in names:
                names.append(v)
        if len(names) >= n:
            break
    return names


def leaderboard_top(n=20, verbose=True):
    """Team names of the current top-n on the leaderboard.

    Returns [] if the leaderboard cannot be read or parsed -- callers must treat
    an empty list as "no team filter", never as "nobody qualifies". Raw CLI
    output is saved under .local/diag/ so a parse failure can be diagnosed
    without re-running the whole download.
    """
    # `--show` is capped at 20 rows no matter what you ask for, so anything
    # deeper has to come from the full CSV. Try that first when it is needed.
    if n > 20:
        names = _leaderboard_full(n, verbose=verbose)
        if names:
            return names
        pr.warn(f"could not read the full leaderboard; falling back to the "
                f"capped --show path, which returns at most 20 teams", 1)

    tried, raw = [], ""
    for cmd in _leaderboard_invocations():
        try:
            raw = _run(cmd, timeout=120)
        except KaggleError as exc:
            tried.append((" ".join(cmd[1:]), str(exc).splitlines()[0]))
            continue
        if not (raw or "").strip():
            tried.append((" ".join(cmd[1:]), "empty output"))
            continue

        names, header = _names_from_csv(raw, n)
        if not names:
            names = _names_from_table(raw, n)
        if names:
            if verbose:
                pr.log(f"leaderboard: {len(names)} of top-{n} teams resolved "
                       f"(e.g. {', '.join(names[:3])})", 1)
            return names
        tried.append((" ".join(cmd[1:]),
                      f"parsed 0 names; header={header or 'none'}"))

    path = _save_diag("leaderboard_raw.txt", raw or "(no output)")
    if verbose:
        pr.warn("could not resolve leaderboard team names:", 1)
        for cmd, why in tried:
            pr.log(f"  {cmd}  ->  {why}", 2)
        if path:
            pr.log(f"raw output saved to {os.path.relpath(path, os.path.dirname(DIAG_DIR))}",
                   2)
        pr.log("continuing WITHOUT a team filter: the daily dataset is already "
               "ranked by rating, so the top episodes are the top players' "
               "games anyway.", 1)
    return []


def episode_teams(row):
    """Team names recorded in a daily-manifest row, if the schema exposes them."""
    out = []
    for key in ("team_names", "teams", "agent_names", "players",
                "team_name_0", "team_name_1", "team0", "team1"):
        v = row.get(key)
        if v:
            out += [t.strip() for t in str(v).replace("|", ",").split(",") if t.strip()]
    return out


def fetch_index(work_dir):
    """Return the rows of the index manifest, newest day last."""
    d = os.path.join(work_dir, "index")
    path = _download_dataset_file(INDEX_DATASET, "manifest.csv", d)
    if not path:
        raise KaggleError(f"could not download manifest.csv from {INDEX_DATASET}")
    rows = _read_csv(path)
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def _score_of(row):
    """Best-effort episode score from a daily manifest row (schema may vary)."""
    for key in ("avg_score", "average_score", "avg_rating", "score",
                "top_avg_score", "mean_score"):
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except ValueError:
                pass
    return 0.0


def _episode_file(row):
    for key in ("file", "filename", "path", "episode_file"):
        if row.get(key):
            return os.path.basename(row[key])
    for key in ("episode_id", "EpisodeId", "id"):
        if row.get(key):
            return f"{row[key]}.json"
    return None


def fetch_top_episodes(work_dir, days=2, per_day=4, max_bytes=None, verbose=True,
                       jobs=6):
    """Download the highest-rated replays from the most recent daily datasets.

    Returns a list of local .json paths.
    """
    out_dir = os.path.join(work_dir, "top")
    os.makedirs(out_dir, exist_ok=True)
    budget = max_bytes if max_bytes is not None else per_day * days * BYTES_PER_REPLAY

    index = fetch_index(work_dir)
    if not index:
        raise KaggleError("index manifest was empty")
    chosen_days = index[-days:] if days else index
    if verbose:
        pr.log(f"index lists {len(index)} day(s); using "
               f"{', '.join(r['date'] for r in chosen_days)}", 1)

    downloaded, spent = [], 0
    for day_row in reversed(chosen_days):
        date = day_row["date"]
        dataset = day_row.get("daily_dataset_slug") or DAILY_DATASET.format(date=date)
        if "/" not in dataset:
            dataset = f"kaggle/{dataset}"
        day_dir = os.path.join(out_dir, date)
        try:
            man = _download_dataset_file(dataset, "manifest.csv", day_dir)
        except KaggleError as exc:
            print(f"  ! skipping {date}: {exc}")
            continue
        if not man:
            print(f"  ! {date}: no manifest.csv")
            continue
        rows = _read_csv(man)
        rows.sort(key=_score_of, reverse=True)
        if verbose:
            print(f"  {date}: {len(rows)} episodes, top score "
                  f"{_score_of(rows[0]):.1f}" if rows else f"  {date}: empty")

        batch = []
        for row in rows:
            if len(batch) >= per_day:
                break
            fname = _episode_file(row)
            if fname:
                batch.append((fname, row))

        def _grab(item):
            fname, row = item
            local = os.path.join(day_dir, fname)
            if os.path.exists(local):
                return fname, row, local, None
            try:
                _download_dataset_file(dataset, fname, day_dir)
            except KaggleError as exc:
                return fname, row, None, str(exc).splitlines()[0]
            return fname, row, (local if os.path.exists(local) else None), None

        taken = 0
        if batch:
            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                for fut in as_completed([pool.submit(_grab, b) for b in batch]):
                    if spent >= budget:
                        break
                    fname, row, local, err = fut.result()
                    if err:
                        pr.warn(f"{fname}: {err}", 3)
                        continue
                    if local and os.path.exists(local):
                        size = os.path.getsize(local)
                        spent += size
                        taken += 1
                        downloaded.append(local)
                        if verbose:
                            pr.log(f"+ {fname}  {size/1e6:.1f} MB  "
                                   f"(score {_score_of(row):.1f})", 2)
    if verbose:
        print(f"  downloaded {len(downloaded)} top-player replays "
              f"({spent/1e6:.0f} MB)")
    return downloaded


_API = None


def kaggle_api():
    """An authenticated in-process KaggleApi, or None.

    Used only as a fallback when the CLI wrapper is broken (see
    _own_submission_rows). Credentials are read by the kaggle package itself,
    exactly as the CLI does -- nothing here reads or copies them.
    """
    global _API
    if _API is not None:
        return _API or None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        _API = api
    except Exception as exc:                                       # noqa: BLE001
        pr.vlog(f"in-process Kaggle API unavailable: {exc}", 2)
        _API = False
        return None
    return _API


def _obj_rows(objs):
    """Turn API result objects into plain dicts, whatever their attribute names."""
    rows = []
    for o in objs or []:
        if isinstance(o, dict):
            rows.append({str(k): o[k] for k in o})
            continue
        d = {}
        for name in dir(o):
            if name.startswith("_"):
                continue
            try:
                v = getattr(o, name)
            except Exception:                                      # noqa: BLE001
                continue
            if callable(v):
                continue
            d[name] = v
        rows.append(d)
    return rows


def _call_api(fn, *args, **maybe_kwargs):
    """Call an API method with only the keyword arguments it actually accepts.

    The kaggle package changes these signatures between releases; passing an
    argument a build does not know is precisely the bug we are working around.
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {k: v for k, v in maybe_kwargs.items() if k in params}
    return fn(*args, **kw)


def _own_submission_rows(verbose=True):
    """Rows describing your submissions. CLI first, in-process API as fallback.

    kaggle 1.8.3 ships a broken `competitions submissions` CLI: cli.py calls
    KaggleApi.competition_submissions(page_number=...), which that build's API
    does not accept, so every invocation dies with

        TypeError: KaggleApi.competition_submissions() got an unexpected
        keyword argument 'page_number'

    The data is fine -- only the CLI wrapper is wrong -- so call the API
    directly, passing only the keywords this build's signature declares.
    """
    try:
        out = _run(["kaggle", "competitions", "submissions", COMPETITION, "--csv"])
        rows = list(csv.DictReader(io.StringIO(out)))
        if rows:
            return rows
        if verbose:
            pr.log("CLI returned no submission rows; trying the API directly", 2)
    except KaggleError as exc:
        first = str(exc)
        broken = ("unexpected keyword argument" in first
                  or "page_number" in first
                  or "TypeError" in first)
        if verbose:
            if broken:
                pr.warn("this kaggle build's `competitions submissions` CLI is "
                        "broken (known bug in 1.8.3); using the API directly", 1)
            else:
                pr.warn(f"submissions CLI failed: {first.splitlines()[0]}", 1)

    api = kaggle_api()
    if api is None:
        pr.warn("could not reach the Kaggle API in-process either", 1)
        return []
    try:
        subs = _call_api(api.competition_submissions, COMPETITION,
                         page_size=100, page_token=None)
    except Exception as exc:                                       # noqa: BLE001
        pr.warn(f"API competition_submissions failed: {exc}", 1)
        return []
    rows = _obj_rows(subs)
    if verbose:
        pr.log(f"API returned {len(rows)} submission(s)", 2)
    return rows


def _own_episode_ids(sub_id, verbose=True):
    """Episode ids played by one submission. CLI first, API fallback."""
    try:
        out = _run(["kaggle", "competitions", "episodes", str(sub_id), "-v"])
        ids = []
        for r in csv.DictReader(io.StringIO(out)):
            for key in ("id", "episodeId", "EpisodeId"):
                if r.get(key) and str(r[key]).isdigit():
                    ids.append(str(r[key]))
                    break
        if ids:
            return ids
    except KaggleError as exc:
        if verbose:
            pr.warn(f"episodes CLI for {sub_id}: {str(exc).splitlines()[0]}", 2)

    api = kaggle_api()
    if api is None:
        return []
    for name in ("competition_list_episodes", "competitions_list_episodes",
                 "competition_episodes"):
        fn = getattr(api, name, None)
        if not fn:
            continue
        try:
            rows = _obj_rows(_call_api(fn, submission_id=int(sub_id),
                                       submissionId=int(sub_id)))
        except Exception:                                          # noqa: BLE001
            continue
        ids = []
        for r in rows:
            for key in ("id", "episodeId", "EpisodeId"):
                v = r.get(key)
                if v is not None and str(v).isdigit():
                    ids.append(str(v))
                    break
        if ids:
            return ids
    return []


def fetch_all_own_episodes(work_dir, verbose=True, jobs=6):
    """Every episode every submission of yours has played (no cap)."""
    return fetch_own_episodes(work_dir, limit=10 ** 6, verbose=verbose, jobs=jobs)


def fetch_own_episodes(work_dir, limit=4, verbose=True, jobs=6):
    """Download replays of episodes your own submissions have played."""
    out_dir = os.path.join(work_dir, "mine")
    os.makedirs(out_dir, exist_ok=True)

    rows = _own_submission_rows(verbose=verbose)
    if not rows:
        if verbose:
            pr.log("no submissions found -- skipping own-game analysis", 1)
        return []

    sub_ids = []
    for r in rows:
        for key in ("ref", "id", "submissionId", "submission_id", "fileName"):
            v = r.get(key)
            if v is not None and str(v).isdigit():
                if str(v) not in sub_ids:
                    sub_ids.append(str(v))
                break
    if not sub_ids:
        pr.warn(f"could not find submission ids in the response "
                f"(keys seen: {sorted(rows[0])[:12]})", 1)
        return []

    if verbose:
        pr.log(f"{len(sub_ids)} submission(s) to enumerate", 1)
    episodes = []
    for i, sid in enumerate(sub_ids, 1):
        if verbose:
            pr.log(f"submission {sid} ({i}/{len(sub_ids)}) -- listing episodes", 2)
        found = _own_episode_ids(sid, verbose=verbose)
        if not found and verbose:
            pr.warn(f"no episodes listed for submission {sid}", 2)
        for eid in found:
            if eid not in episodes:
                episodes.append(eid)
        if len(episodes) >= limit:
            break

    if verbose:
        pr.log(f"{len(episodes)} episode(s) found; downloading "
               f"{min(limit, len(episodes))} ({jobs} at a time)", 1)
    wanted_ids = episodes[:limit]

    def _grab(eid):
        """One replay. Runs in a worker thread -- each call is its own kaggle
        CLI process, so concurrency here is safe and is most of the speed-up."""
        cand = os.path.join(out_dir, f"{eid}.json")
        if os.path.exists(cand):
            return eid, cand, None
        try:
            _run(["kaggle", "competitions", "replay", eid, "-p", out_dir])
        except KaggleError as exc:
            return eid, None, str(exc).splitlines()[0]
        _unzip_inplace(out_dir)
        return eid, (cand if os.path.exists(cand) else None), None

    downloaded = []
    tick = pr.Ticker(total=len(wanted_ids), label="own replays", indent=2)
    if wanted_ids:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for fut in as_completed([pool.submit(_grab, e) for e in wanted_ids]):
                eid, path, err = fut.result()
                if err:
                    pr.warn(f"replay {eid}: {err}", 3)
                    continue
                if not path:
                    continue
                downloaded.append(path)
                if verbose:
                    tick.step(f"episode {eid}  {os.path.getsize(path)/1e6:.1f} MB")
    if verbose:
        tick.done()
    return downloaded


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=os.path.join(
        ROOT, "data", "episodes"))
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--per-day", type=int, default=4)
    ap.add_argument("--own", type=int, default=4)
    ap.add_argument("--max-mb", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=6, help="parallel downloads")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    try:
        print("fetching top-player replays ...")
        top = fetch_top_episodes(args.work, args.days, args.per_day,
                                 int(args.max_mb * 1e6) if args.max_mb else None,
                                 jobs=args.jobs)
        print("fetching your own replays ...")
        mine = fetch_own_episodes(args.work, args.own, jobs=args.jobs)
    except KaggleError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"top": top, "mine": mine}, indent=1))


if __name__ == "__main__":
    main()
