"""The project's single source of truth: models, runs, data, opponents.

Everything else in the system reads and writes through here, so the dashboard,
the trainers, the submitter and the Elo ladder cannot disagree about what
exists or how good it is.

Three stores, all JSON under .local/registry/ so they survive a crash and can
be inspected with a text editor:

    models.json     one entry per agent: config, lineage, Elo, test status,
                    latency, submission history
    runs.json       one entry per job the dashboard or CLI has launched:
                    command, status, timing, exit code, log path
    opponents.json  the leaderboard-topper clones built from replay data

Design note: the Elo ladder in models/elo/ladder.json stays the authority on
ratings -- this module reads it rather than duplicating it. A registry that
keeps its own copy of a number another tool owns is a registry that goes stale.

    python -m kaggriculture.data.registry --list
    python -m kaggriculture.data.registry --json
    python -m kaggriculture.data.registry --sync        # re-scan agents/ and the ladder
"""
from kaggriculture.paths import ROOT
import datetime as dt
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(ROOT, ".local", "registry")
MODELS = os.path.join(REG, "models.json")
RUNS = os.path.join(REG, "runs.json")
OPPONENTS = os.path.join(REG, "opponents.json")
LADDER = os.path.join(ROOT, "models", "elo", "ladder.json")
LOGDIR = os.path.join(ROOT, "data", "logs", "runs")

MIN_GAMES = 16          # below this a rating is not actionable; see dashboard.py


# --------------------------------------------------------------- primitives --

def _now():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def _read(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)          # atomic: a killed writer never truncates


# ------------------------------------------------------------------ models --

def load_models():
    return _read(MODELS, {})


def save_models(m):
    _write(MODELS, m)


def register_model(name, **fields):
    """Create or update a model entry. Unknown fields are kept verbatim, so a
    later tool can attach data this module has never heard of."""
    m = load_models()
    entry = m.get(name, {"name": name, "created": _now()})
    entry.update(fields)
    entry["updated"] = _now()
    m[name] = entry
    save_models(m)
    return entry


def ladder():
    return _read(LADDER, {"rating": {}, "games": {}, "wins": {}, "history": []})


def sync():
    """Re-scan agents/ and the Elo ladder so the registry matches reality.

    Cheap and idempotent -- the dashboard calls it on every page load, because
    a registry that only updates when someone remembers to update it is worse
    than no registry.
    """
    lad = ladder()
    rating, games, wins = lad.get("rating", {}), lad.get("games", {}), lad.get("wins", {})
    m = load_models()

    for path in sorted(glob.glob(os.path.join(ROOT, "agents", "*.py"))):
        name = os.path.basename(path)
        if name.startswith("__"):
            continue
        e = m.get(name, {"name": name, "created": _now()})
        e["path"] = os.path.join("agents", name)
        e["exists"] = True
        e["bytes"] = os.path.getsize(path)
        e["mtime"] = dt.datetime.fromtimestamp(
            os.path.getmtime(path)).replace(microsecond=0).isoformat()
        g = int(games.get(name, 0))
        e["elo"] = float(rating.get(name, 0.0))
        e["games"] = g
        e["wins"] = float(wins.get(name, 0.0))
        e["win_rate"] = (e["wins"] / g) if g else 0.0
        e["confidence"] = (400.0 / (g ** 0.5)) if g else None
        e["rated"] = g >= MIN_GAMES
        digest = file_digest(path)
        if digest and digest != e.get("sha256"):
            # The file changed: its old measurements describe a different agent.
            e["sha256"] = digest
            e["params"] = snapshot_params(path)
            e["params_changed"] = _now()
        elif "params" not in e:
            e["params"] = snapshot_params(path)
        m[name] = e

    for name, e in list(m.items()):
        if not os.path.exists(os.path.join(ROOT, "agents", name)):
            e["exists"] = False          # keep the history, flag the absence
            # DERIVED FIELDS MUST BE DEFINITE, NOT None. Entries for files that
            # are gone (a deleted agent, a non-agent name like "pipeline", a
            # candidate that lived in .local/) never pass through the loop
            # above, so `rated` stayed None -- and `None == False` is False, so
            # any consumer comparing the flag to a boolean silently disagrees.
            # That is what failed test_system's registry check. An absent model
            # is certainly not rated; say so.
            e["rated"] = bool(e.get("rated") or False)
            e.setdefault("games", 0)
            e.setdefault("wins", 0.0)
            e.setdefault("win_rate", 0.0)

    save_models(m)
    return m




# ------------------------------------------------------- model detail store --
#
# The registry used to hold only what the Elo ladder knew: a rating and a game
# count. That is not a model record -- it cannot answer "what was in this
# agent", "what did it actually score against the ladder tapes", or "is this
# the file we submitted". Those three questions are the whole reason to keep a
# registry, so they are stored explicitly.

def file_digest(path):
    """sha256 of the agent file: the only reliable identity for a submission."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def snapshot_params(path):
    """The agent's PARAMS block, so a record survives the file being replaced."""
    try:
        import kaggriculture.pipeline.params as paramio
        return paramio.load(path)
    except Exception:                                              # noqa: BLE001
        return None


def record_eval(name, opponent, win_rate, margin, bank, seeds=None,
                matches=None, note=None):
    """Attach one measured result to a model. Newest first, de-duplicated by
    (opponent, seeds) so re-running a comparison updates rather than piles up."""
    m = load_models()
    entry = m.get(name) or {"name": name, "created": _now()}
    evals = [e for e in entry.get("evals", [])
             if not (e.get("opponent") == opponent and e.get("seeds") == seeds)]
    evals.insert(0, {
        "opponent": opponent, "win_rate": round(float(win_rate), 4),
        "margin": round(float(margin), 1), "bank": round(float(bank), 1),
        "seeds": seeds, "matches": matches, "note": note, "when": _now(),
    })
    entry["evals"] = evals[:40]
    entry["updated"] = _now()
    m[name] = entry
    save_models(m)
    return entry


def record_submission(name, **fields):
    """Mark a model as submitted: when, to what, and what the ladder said."""
    m = load_models()
    entry = m.get(name) or {"name": name, "created": _now()}
    subs = entry.get("submissions", [])
    row = {"when": _now()}
    row.update(fields)
    subs.insert(0, row)
    entry["submissions"] = subs[:20]
    entry["updated"] = _now()
    m[name] = entry
    save_models(m)
    return entry


def card(name):
    """Everything the registry knows about one model, for printing."""
    m = sync()
    return m.get(name)


def print_card(name):
    e = card(name)
    if not e:
        print(f"no such model: {name}")
        return 1
    print("=" * 74)
    print(f"{e['name']}")
    print("=" * 74)
    for key in ("path", "created", "updated", "bytes", "sha256", "config",
                "base", "description"):
        if e.get(key):
            print(f"  {key:<12} {e[key]}")
    if e.get("tags"):
        print(f"  {'tags':<12} {', '.join(e['tags'])}")
    print(f"  {'elo':<12} {e.get('elo', 0):.1f}  "
          f"games {e.get('games', 0)}  win rate {100 * (e.get('win_rate') or 0):.0f}%"
          f"  {'rated' if e.get('rated') else 'unrated'}")
    if e.get("overrides"):
        print("  overrides")
        for k, v in e["overrides"].items():
            print(f"    {k:<24} {v}")
    if e.get("evals"):
        print("  measured")
        for ev in e["evals"][:12]:
            print(f"    vs {os.path.basename(str(ev['opponent'])):<34} "
                  f"win {100 * ev['win_rate']:>5.0f}%  margin {ev['margin']:>+10,.0f}"
                  f"  bank {ev['bank']:>9,.0f}   {ev['when'][:16]}")
    if e.get("submissions"):
        print("  submissions")
        for sub in e["submissions"]:
            bits = "  ".join(f"{k}={v}" for k, v in sub.items() if k != "when")
            print(f"    {sub['when'][:16]}  {bits}")
    return 0


def ranked():
    """Models newest-Elo-first, unrated last. What every table in the UI uses."""
    m = sync()
    rows = list(m.values())
    rows.sort(key=lambda r: (-(r.get("elo") or 0.0), -(r.get("games") or 0)))
    return rows


def best(require_rated=True):
    """Top model, or None. `require_rated` enforces the games threshold."""
    for r in ranked():
        if not r.get("exists"):
            continue
        if require_rated and not r.get("rated"):
            continue
        return r
    return None


# -------------------------------------------------------------------- runs --

def load_runs():
    return _read(RUNS, [])


def save_runs(r):
    _write(RUNS, r)


def start_run(kind, command, meta=None):
    """Record a job as started; returns its id. The dashboard's job runner and
    the CLI both call this, so run history is complete however work is launched."""
    runs = load_runs()
    rid = f"{kind}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(runs) + 1}"
    os.makedirs(LOGDIR, exist_ok=True)
    runs.append({
        "id": rid, "kind": kind, "command": command, "status": "running",
        "started": _now(), "finished": None, "exit_code": None,
        "log": os.path.join("data", "logs", "runs", rid + ".log"),
        "meta": meta or {},
    })
    save_runs(runs[-500:])          # bounded; the logs keep the detail
    return rid


def finish_run(rid, exit_code, meta=None):
    runs = load_runs()
    for r in runs:
        if r["id"] == rid:
            r["status"] = "ok" if exit_code == 0 else "failed"
            r["exit_code"] = exit_code
            r["finished"] = _now()
            if meta:
                r["meta"].update(meta)
            break
    save_runs(runs)


def run_log_path(rid):
    return os.path.join(LOGDIR, rid + ".log")


def reap_orphans():
    """Mark rows still 'running' from a previous process as orphaned.

    finish_run only ever fires from the dashboard's output pump, so a server
    that is killed -- or a machine that reboots -- leaves its in-flight rows
    at 'running' forever. They then show amber in the UI with a Stop button
    that cannot work, because the process they name died with the old server,
    and clear-runs deliberately preserves running rows. Nothing on the box has
    a live handle to them, so at startup they are orphans by definition.

    Returns the ids that were reaped.
    """
    runs = load_runs()
    reaped = []
    for r in runs:
        if r.get("status") == "running":
            r["status"] = "orphaned"
            r["finished"] = _now()
            r["exit_code"] = None
            reaped.append(r["id"])
    if reaped:
        save_runs(runs)
    return reaped


# --------------------------------------------------------------- opponents --

def load_opponents():
    return _read(OPPONENTS, {})


def register_opponent(name, **fields):
    o = load_opponents()
    e = o.get(name, {"name": name, "created": _now()})
    e.update(fields)
    e["updated"] = _now()
    o[name] = e
    _write(OPPONENTS, o)
    return e


# -------------------------------------------------------------------- data --

def data_status():
    """What the download has actually produced. Shown on the dashboard so
    'have enough data?' is answerable without opening a terminal."""
    csv = os.path.join(ROOT, "data", "episodes.csv")
    state = _read(os.path.join(ROOT, "data", "fetch_state.json"), {})
    rows = 0
    if os.path.exists(csv):
        try:
            with open(csv, encoding="utf-8") as f:
                rows = max(0, sum(1 for _ in f) - 1)
        except OSError:
            rows = 0
    replays = glob.glob(os.path.join(ROOT, "data", "episodes", "**", "*.json"),
                        recursive=True)
    return {
        "player_rows": rows,
        "episodes": rows // 2,
        "last_run": state.get("last_run"),
        "last_daily_date": state.get("last_daily_date"),
        "seen": len(state.get("seen", [])),
        "replays_on_disk": len(replays),
        "csv_mtime": (dt.datetime.fromtimestamp(os.path.getmtime(csv))
                      .replace(microsecond=0).isoformat()
                      if os.path.exists(csv) else None),
    }


def _posix(path):
    """Forward slashes everywhere the path leaves Python.

    A Windows path embedded in a JavaScript string literal loses its
    backslashes -- "agents\\agent_v4.py" becomes "agentsagent_v4.py", because
    \\a is not an escape JS knows and it drops the backslash. Python opens
    "agents/agent_v4.py" perfectly well on Windows, so normalising on the way
    out removes a whole class of bug rather than escaping around it.
    """
    return path.replace("\\", "/") if isinstance(path, str) else path


def snapshot():
    """Everything the dashboard needs in one call."""
    models = ranked()
    for m in models:
        for key in ("path", "base"):
            if m.get(key):
                m[key] = _posix(m[key])
    opponents = list(load_opponents().values())
    for o in opponents:
        if o.get("path"):
            o["path"] = _posix(o["path"])
    return {
        "generated": _now(),
        "models": models,
        "runs": list(reversed(load_runs()))[:60],
        "opponents": opponents,
        "data": data_status(),
        "min_games": MIN_GAMES,
    }


def sync_kaggle(verbose=True):
    """Pull public scores off Kaggle and attach them to the models that earned them.

    The registry has always stored local Elo and nothing else, so the one number
    the competition actually ranks on -- the public score -- lived only in the
    Kaggle web UI. Elo 944 next to public 873.7 is the comparison that tells you
    whether local measurement is tracking the ladder at all; without the second
    column there is nothing to check against.

    Matching is by submission id first (record_submission stores the ref), then
    by the agent name appearing in the upload message, which is the convention
    every submission from this repo has followed.
    """
    import csv as _csv
    import io as _io
    import subprocess as _sp
    import kaggriculture.data.episodes as _ep

    try:
        out = _sp.run(_ep.kaggle_cmd() + ["competitions", "submissions",
                                          "kaggriculture", "--csv"],
                      capture_output=True, text=True, timeout=180)
    except Exception as exc:                                       # noqa: BLE001
        if verbose:
            print(f"could not reach Kaggle: {exc}")
        return {}
    if out.returncode != 0:
        if verbose:
            print(f"kaggle CLI failed: {(out.stderr or out.stdout)[:200]}")
        return {}

    rows = [r for r in _csv.DictReader(_io.StringIO(out.stdout)) if r.get("ref")]
    models = load_models()
    by_ref = {}
    for name, entry in models.items():
        for sub in entry.get("submissions") or []:
            if sub.get("ref"):
                by_ref[str(sub["ref"])] = name

    hits = {}
    for row in rows:
        ref = str(row.get("ref") or "").strip()
        score = (row.get("publicScore") or "").strip()
        if not score:
            continue
        name = by_ref.get(ref)
        if not name:
            desc = (row.get("description") or "")
            for candidate in models:
                stem = candidate[:-3] if candidate.endswith(".py") else candidate
                if stem and stem in desc:
                    name = candidate
                    break
        if not name:
            continue
        entry = models[name]
        try:
            value = float(score)
        except ValueError:
            continue
        best = entry.get("public_score")
        if best is None or value > float(best):
            entry["public_score"] = value
            entry["public_score_ref"] = ref
            entry["public_score_when"] = row.get("date")
        for sub in entry.get("submissions") or []:
            if str(sub.get("ref")) == ref:
                sub["public_score"] = value
                sub["status"] = str(row.get("status") or sub.get("status") or "")
        entry["updated"] = _now()
        hits[name] = value

    if hits:
        save_models(models)
    if verbose:
        if not hits:
            print(f"{len(rows)} Kaggle submission(s) read; none matched a "
                  f"registered model by ref or by name in the message")
        for name, value in sorted(hits.items(), key=lambda kv: -kv[1]):
            local = models[name].get("elo")
            print(f"  {name:<34} public {value:>8.1f}"
                  + (f"   local elo {local:>6.0f}" if local else ""))
    return hits


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--sync-kaggle", action="store_true",
                    help="pull public scores off Kaggle into the model records")
    ap.add_argument("--card", default=None,
                    help="print everything known about one model")
    args = ap.parse_args()

    if args.sync_kaggle:
        sync_kaggle()
        return 0

    if args.card:
        name = os.path.basename(args.card)
        return print_card(name)

    if args.json:
        print(json.dumps(snapshot(), indent=1, default=str))
        return 0

    snap = snapshot()
    d = snap["data"]
    print(f"data      : {d['episodes']:,} episodes ({d['player_rows']:,} rows), "
          f"{d['replays_on_disk']} replay(s) on disk, last fetch {d['last_run'] or 'never'}")
    print(f"opponents : {len(snap['opponents'])}")
    print(f"runs      : {len(snap['runs'])} recorded\n")
    print(f"{'model':<40} {'elo':>7} {'games':>6} {'win%':>6}  status")
    print("-" * 78)
    for r in snap["models"]:
        status = []
        if not r.get("exists"):
            status.append("missing")
        status.append("rated" if r.get("rated") else f"needs {MIN_GAMES - r.get('games', 0)} games")
        if r.get("tests"):
            status.append(r["tests"])
        print(f"{r['name']:<40} {r.get('elo', 0):>7.0f} {r.get('games', 0):>6} "
              f"{r.get('win_rate', 0):>5.0%}  {', '.join(status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
