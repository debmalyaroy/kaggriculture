"""Local control panel for the whole project. One command, no dependencies.

    python dashboard/serve.py                 ->  http://127.0.0.1:8787
    python dashboard/serve.py --port 9000
    python dashboard/serve.py --allow-submit  ->  enables the live submit button

What it gives you, all driving the same tools the CLI uses -- there is no second
implementation to drift:

    Models      every agent with Elo, games, confidence, config lineage, test
                status; buttons to evaluate, Elo-rate, post-mortem and submit
    Data        episode counts and last fetch, with a download button
    Runs        every job ever launched, with live streaming logs
    Losses      post-mortems: where the bank gap opened and what caused it
    Opponents   leaderboard-topper tapes and clones, and a button to play them
    Configs     the model factory: build any config into an agent

Deliberate choices
------------------
*Standard library only.* http.server and threads. Adding FastAPI would mean a
pip install standing between you and your own dashboard, on a machine where the
last three problems were all environment problems.

*Binds to 127.0.0.1.* Not configurable to 0.0.0.0. This process runs arbitrary
commands by design; it must not be reachable from the network.

*Submit needs two independent gates*, as it does everywhere else: the server
must be started with --allow-submit, and the browser must send the literal
string SUBMIT. Without the flag the button is disabled and the endpoint refuses.
Five submissions a day, only the latest two active -- an accidental click is
expensive.
"""
from kaggriculture.paths import ROOT
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.registry as registry  # noqa: E402

ALLOW_SUBMIT = False
BUILD = None                      # short hash of this server + page, shown in the UI
_JOBS = {}                          # run id -> {"proc":, "thread":, "kind":}
_LOCK = threading.Lock()
PY = sys.executable


# ------------------------------------------------------------- job plumbing --

# Directories a path parameter can legitimately point into.
_DIRS = ("agents", "configs", "opponents", "tools", "tests", "dashboard", "data")


def _fix_path(value):
    """Repair a path that lost its separator, and normalise slashes.

    A Windows path embedded in a JavaScript string literal silently loses its
    backslashes -- "agents\\agent_v4.py" arrives as "agentsagent_v4.py" and the
    command then runs against a file that does not exist. The page no longer
    produces those, but a browser holding a cached copy still can, so the repair
    lives on the server where one fix covers every client.

    Also recovers by basename: if the file is not where the caller said, but a
    file of that name exists under agents/, use it and carry on.
    """
    if not isinstance(value, str) or not value:
        return value
    v = value.replace("\\", "/")
    if os.path.exists(os.path.join(ROOT, v)) or os.path.isabs(v):
        return v
    for d in _DIRS:
        if v.startswith(d) and not v.startswith(d + "/"):
            candidate = d + "/" + v[len(d):]
            if os.path.exists(os.path.join(ROOT, candidate)):
                return candidate
    base = os.path.basename(v)
    for d in _DIRS:
        candidate = os.path.join(d, base)
        if os.path.exists(os.path.join(ROOT, candidate)):
            return candidate.replace("\\", "/")
    return v


def _cmd_for(kind, p):
    """Map a UI action to an actual command. The single place actions are defined."""
    p = dict(p or {})
    for key in ("agent", "vs", "config", "base"):
        if key in p:
            p[key] = _fix_path(p[key])
    g = lambda k, d=None: p.get(k, d)                              # noqa: E731

    if kind == "pipeline":
        cmd = [PY, "-u", "scripts/run_pipeline.py"]
        if g("on_demand"):
            cmd.append("--on-demand")
        if g("fresh_only"):
            cmd.append("--fresh-only")
        return cmd
    if kind == "download":
        return [PY, "-u", "scripts/download_data.py",
                "--days", str(int(g("days", 3))),
                "--per-day", str(int(g("per_day", 60))),
                "--own", str(int(g("own", 0))),
                "--jobs", str(int(g("jobs", 8)))] + \
               (["--any-top"] if g("any_top") else []) + \
               (["--verbose"] if g("verbose") else [])
    if kind == "download_check":
        return [PY, "-u", "scripts/download_data.py", "--check"]
    if kind == "download_diagnose":
        return [PY, "-u", "scripts/download_data.py", "--leaderboard"]
    if kind == "submissions_diagnose":
        return [PY, "-u", "scripts/download_data.py", "--submissions"]
    if kind == "test_fast":
        return [PY, "-u", "-c",
                "import sys;sys.path.insert(0,'tests');import test_agents as t;"
                "t.test_agents_are_legal_and_fast();print('legality + latency ok')"]
    if kind == "test_full":
        return [PY, "-u", "tests/test_agents.py"]
    if kind == "evaluate":
        return [PY, "-u", "src/kaggriculture/measure/evaluate.py", g("agent"),
                "--vs", g("vs", "agents/v2_tuned.py"),
                "-n", str(int(g("n", 8))),
                "--workers", str(int(g("workers", 0)) or _workers()),
                "--seed0", str(int(g("seed0", 0)))]
    if kind == "ab_paired":
        # One-off paired A/B with the sign-test verdict computed and stored
        # (models/ab/). The instrument behind the demand-model and adaptive
        # sell-timing decisions, as a button.
        return [PY, "-u", "src/kaggriculture/measure/ab_test.py", g("a"), g("b"),
                "-n", str(int(g("n", 8))),
                "--seeds", g("seeds", "60000,150000")]
    if kind == "elo":
        base = [PY, "-u", "src/kaggriculture/measure/elo.py", "--rounds", str(int(g("rounds", 2))),
                "--workers", str(int(g("workers", 0)) or _workers())]
        return base + (["--only", g("agent")] if g("agent") else [])
    if kind == "improve":
        return [PY, "-u", "src/kaggriculture/pipeline/improve.py",
                "--minutes", str(float(g("minutes", 30))),
                "--elo1", str(float(g("elo1", 25))),
                "--workers", str(int(g("workers", 0)) or _workers())] + \
               (["--base", g("agent")] if g("agent") else [])
    if kind == "build":
        return [PY, "-u", "src/kaggriculture/agentbuild/build_agent.py", g("config"), "--smoke"]
    if kind == "build_all":
        return [PY, "-u", "src/kaggriculture/agentbuild/build_agent.py", "configs/*.json", "--smoke"]
    if kind == "optimize":
        cmd = [PY, "-u", "src/kaggriculture/train/optimize.py",
               "--generations", str(int(g("generations", 8))),
               "--popsize", str(int(g("popsize", 12))),
               "--seeds", str(int(g("seeds", 3))),
               "--workers", str(int(g("workers", 0)) or _workers()),
               "--objective", str(g("objective", "win")),
               "--run", str(g("run", "main"))]
        if g("base"):
            cmd += ["--base", g("base")]
        if g("out"):
            cmd += ["--out", _fix_path(g("out"))]
        if g("vs"):
            cmd += ["--vs"] + [_fix_path(x) for x in str(g("vs")).split(",") if x.strip()]
        if g("resume"):
            cmd += ["--resume"]
        if g("budget_min"):
            cmd += ["--budget-min", str(float(g("budget_min")))]
        return cmd
    if kind == "notebooks":
        return [PY, "-u", "src/kaggriculture/data/notebooks.py", "--top", str(int(g("top", 25)))] +                (["--refresh"] if g("refresh") else [])
    if kind == "notebooks_index":
        return [PY, "-u", "src/kaggriculture/data/notebooks.py", "--index"]
    if kind == "profile_replay":
        return [PY, "-u", "src/kaggriculture/data/replay_profile.py", _fix_path(g("path", ".local/top/*.json")),
                "--compact"]
    if kind == "test_contract":
        return [PY, "-u", "tests/test_contract.py"]
    if kind == "model_card":
        return [PY, "-u", "src/kaggriculture/data/registry.py", "--card", g("agent", "")]
    if kind == "opponents_tapes":
        return [PY, "-u", "src/kaggriculture/measure/opponents.py", "--build-tapes"]
    if kind == "opponents_clones":
        return [PY, "-u", "src/kaggriculture/measure/opponents.py", "--build-clones",
                "--top", str(int(g("top", 5)))]
    if kind == "opponents_play":
        cmd = [PY, "-u", "src/kaggriculture/measure/opponents.py", "--play", g("agent"),
               "--n", str(int(g("n", 2))),
               "--workers", str(int(g("workers", 0)) or _workers())]
        # An empty "vs" means the whole set, which is the old behaviour.
        vs = [s for s in str(g("vs", "")).split(",") if s.strip()]
        if vs:
            cmd += ["--vs"] + [s.strip() for s in vs]
        return cmd
    if kind == "routes_mine":
        return [PY, "-u", "src/kaggriculture/data/routes.py", "--mine",
                "--top", str(int(g("top", 30))),
                "--per-team", str(int(g("per_team", 6))),
                "--max-gb", str(float(g("max_gb", 8))),
                "--days", str(int(g("days", 3))),
                "--jobs", str(int(g("jobs", 8)))]
    if kind == "routes_report":
        return [PY, "-u", "src/kaggriculture/data/routes.py", "--report"]
    if kind == "routes_select":
        return [PY, "-u", "src/kaggriculture/data/routes.py", "--select",
                "--top-n", str(int(g("top_n", 8))),
                "--seeds", str(int(g("seeds", 2))),
                "--out", _fix_path(g("out", "agents/v17_route.py"))]
    if kind == "panel":
        return [PY, "-u", "src/kaggriculture/measure/panel.py", "--agent", g("agent"),
                "--top", str(int(g("top", 100))),
                "--seeds", str(int(g("seeds", 3))),
                "--workers", str(int(g("workers", 0)) or _workers())]
    if kind == "routes_build":
        return [PY, "-u", "src/kaggriculture/data/routes.py", "--build", g("route", "best"),
                "--out", _fix_path(g("out", "agents/v15_route.py"))]
    if kind == "autopilot":
        cmd = [PY, "-u", "src/kaggriculture/pipeline/autopilot.py",
               "--stages", g("stages", "fetch,train,build,gate"),
               "--line", ("route" if g("line") == "route" else "params"),
               "--top", str(int(g("top", 200))),
               "--per-team", str(int(g("per_team", 3))),
               "--jobs", str(int(g("jobs", 8)))]
        if g("quick"):
            cmd.append("--quick")
        # --submit-live is deliberately unreachable from the browser: the
        # submit stage shells out to src/submit.py, which asks for a typed
        # SUBMIT on a console the dashboard does not own.
        return cmd
    if kind == "autopilot_status":
        return [PY, "-u", "src/kaggriculture/pipeline/autopilot.py", "--status"]
    if kind == "autopilot_notebook":
        return [PY, "-u", "src/kaggriculture/pipeline/autopilot.py", "--emit-notebook"]
    if kind == "loss_scan":
        return [PY, "-u", "src/kaggriculture/measure/loss_analysis.py", "--agent", g("agent"),
                "--vs", g("vs", "agents/v2_tuned.py"),
                "--scan", str(int(g("scan", 6)))]
    if kind == "adaptive_set":
        cmd = [PY, "-u", "src/kaggriculture/train/adaptive.py", "--agent", g("agent"),
               "--mode", g("mode", "both")]
        if g("ensemble_k"):
            cmd += ["--ensemble-k", str(int(g("ensemble_k")))]
        if g("arbiter"):
            cmd += ["--arbiter", str(g("arbiter"))]
        if g("ab"):
            cmd += ["--ab", "--elo1", str(float(g("elo1", 25)))]
        return cmd
    if kind == "adaptive_list":
        return [PY, "-u", "src/kaggriculture/train/adaptive.py", "--list"]
    if kind == "select_build":
        cmd = [PY, "-u", "src/kaggriculture/train/select.py", "--build",
               "--k", str(int(g("k", 4)))]
        if g("spread_guard"):
            cmd += ["--spread-guard"]
        return cmd
    if kind == "arbiter":
        cmd = [PY, "-u", "src/kaggriculture/train/train_arbiter.py",
               "--minutes", str(float(g("minutes", 45)))]
        if g("agent"):
            cmd += ["--base", g("agent")]
        return cmd
    if kind == "pbt":
        cmd = [PY, "-u", "src/kaggriculture/train/pbt.py",
               "--population", str(int(g("population", 6))),
               "--minutes", str(float(g("minutes", 60))),
               "--seeds", str(int(g("seeds", 2)))]
        if g("agent"):
            cmd += ["--base", g("agent")]
        if g("resume"):
            cmd += ["--resume"]
        return cmd
    if kind == "pbt_harvest":
        return [PY, "-u", "src/kaggriculture/train/pbt.py", "--harvest", str(int(g("n", 4)))]
    if kind == "pbt_status":
        return [PY, "-u", "src/kaggriculture/train/pbt.py", "--status"]
    if kind == "xgb_collect":
        return [PY, "-u", "src/kaggriculture/train/train_xgb.py", "--collect",
                str(int(g("episodes", 30))), "--base", g("agent")]
    if kind == "xgb_train":
        return [PY, "-u", "src/kaggriculture/train/train_xgb.py", "--train",
                "--rounds", str(int(g("rounds", 200)))]
    if kind == "xgb_build":
        cmd = [PY, "-u", "src/kaggriculture/train/train_xgb.py", "--build", "--base", g("agent")]
        if g("ab"):
            cmd += ["--ab"]
        return cmd
    if kind == "xgb_status":
        return [PY, "-u", "src/kaggriculture/train/train_xgb.py", "--status"]
    if kind == "engine_check":
        return [PY, "-u", "src/kaggriculture/engine/engine_check.py"]
    if kind == "conformance":
        cmd = [PY, "-u", "src/kaggriculture/engine/conformance.py"]
        if g("agent"):
            cmd += ["--agent", g("agent")]
        return cmd
    if kind == "kaggle_audit":
        return [PY, "-u", "src/kaggriculture/engine/kaggle_env.py", "--audit"]
    if kind == "kaggle_verify":
        return [PY, "-u", "src/kaggriculture/engine/kaggle_env.py", "--verify",
                "--agent", g("agent"), "--vs", g("vs", "agents/v2_tuned.py"),
                "--seed", str(int(g("seed", 21)))]
    if kind == "eda":
        return [PY, "-u", "src/kaggriculture/pipeline/eda_report.py"]
    if kind == "scheduler_install":
        return ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                "src/install_scheduler.ps1"]
    if kind == "select_show":
        return [PY, "-u", "src/kaggriculture/train/select.py", "--show", "--spread-guard"]
    if kind == "improve_status":
        return [PY, "-u", "src/kaggriculture/pipeline/improve.py", "--status"]
    if kind == "parity":
        return [PY, "-u", "src/kaggriculture/engine/parity.py", "--matches", str(int(g("matches", 3)))]
    if kind == "test_system":
        return [PY, "-u", "tests/test_system.py"]
    if kind == "refresh_dashboard":
        return [PY, "-u", "src/kaggriculture/pipeline/dashboard.py"]
    if kind == "model_graph":
        return [PY, "-u", "src/kaggriculture/agentbuild/model_graph.py", g("agent")]
    if kind == "notebook_build":
        return [PY, "-u", "src/kaggriculture/agentbuild/build_notebook.py", "--agent", g("agent")]
    if kind == "notebook_push":
        # Pushes the write-up kernel. Entering it in the competition is still a
        # separate, manual step -- see the command the tool prints.
        return [PY, "-u", "src/kaggriculture/agentbuild/build_notebook.py", "--agent", g("agent"), "--push"]
    if kind == "submit_dry":
        return [PY, "-u", "src/kaggriculture/pipeline/submit.py", "--agent", g("agent"),
                "--seeds", str(int(g("seeds", 6))), "--dry-run"]
    if kind == "submit_live":
        return [PY, "-u", "src/kaggriculture/pipeline/submit.py", "--agent", g("agent"),
                "--seeds", str(int(g("seeds", 6)))] + \
               (["--message", g("message")] if g("message") else [])

    # ------------------------- ops / release / engine (added 2026-08-16) --
    if kind == "daily_release_dry":
        return [PY, "-u", "scripts/daily_release.py", "--no-submit"]
    if kind == "daily_release_live":                       # gated: LIVE_KINDS
        return [PY, "-u", "scripts/daily_release.py"]
    if kind == "release_resume":                           # gated: LIVE_KINDS
        return [PY, "-u", "scripts/daily_release.py", "--resume-publish"]
    if kind == "engine_swap_status":
        return [PY, "-u", "scripts/engine_swap_1327.py", "--status"]
    if kind == "engine_swap_check":
        return [PY, "-u", "scripts/engine_swap_1327.py", "--check"]
    if kind == "ourgames_all":
        return [PY, "-u", "src/kaggriculture/data/ourgames.py", "--all",
                "--jobs", str(int(g("jobs", 6)))]
    if kind == "ourgames_by_sub":
        return [PY, "-u", "src/kaggriculture/data/ourgames.py", "--by-submission"]
    if kind == "sameday_fetch":
        return [PY, "-u", "src/kaggriculture/data/sameday.py", "--jobs", str(int(g("jobs", 6))),
                "--max-gb", str(float(g("max_gb", 8)))]
    if kind == "backfill_ingest":
        return [PY, "-u", "src/kaggriculture/data/backfill_ingest.py",
                "--jobs", str(int(g("jobs", 6)))]
    if kind == "run_all_tests":
        return [PY, "-u", "tests/run_all.py"] + \
               (["--fast"] if g("fast", 1) else [])

    # ---------------------------------------------------- Track P (ditto) --
    if kind == "trackp_pipeline":
        cmd = [PY, "-u", "src/kaggriculture/trackp/pipeline.py"]
        if g("stages"):
            cmd += ["--stages", str(g("stages"))]
        if g("weekly"):
            cmd += ["--weekly"]
        return cmd
    if kind == "trackp_phase":
        return [PY, "-u", "src/kaggriculture/trackp/phases.py", "--run",
                str(g("phase", "0"))]
    if kind == "trackp_phases_status":
        return [PY, "-u", "src/kaggriculture/trackp/phases.py", "--status"]
    if kind == "trackp_arena":
        return [PY, "-u", "src/kaggriculture/trackp/arena.py",
                g("agent", "agents/planner_v0.py"),
                "--vs", g("vs", "agents/v26.0_route.py"),
                "-n", str(int(g("n", 2))), "--latency"]
    if kind == "trackp_guard":
        return [PY, "-u", "src/kaggriculture/trackp/guard.py"]
    if kind == "trackp_graduation":
        return [PY, "-u", "src/kaggriculture/trackp/graduation.py",
                "--panel-n", str(int(g("panel_n", 6))),
                "--holdout-n", str(int(g("holdout_n", 3)))]
    if kind == "trackp_verdict":
        return [PY, "-u", "src/kaggriculture/trackp/verdict_model.py"]
    if kind == "trackp_twins":
        return [PY, "-u", "src/kaggriculture/trackp/twins.py"]
    if kind == "trackp_winprob":
        return [PY, "-u", "src/kaggriculture/trackp/winprob.py"]
    if kind == "trackp_insight":
        return [PY, "-u", "src/kaggriculture/trackp/insight.py"]
    if kind == "trackp_regret":
        cmd = [PY, "-u", "src/kaggriculture/trackp/regret.py",
               "--seat", str(int(g("seat", 0)))]
        if g("episode"):
            cmd += ["--episode", str(g("episode"))]
        else:
            cmd += ["--smoke"]
        return cmd
    if kind == "promote_planner":                          # gated: LIVE_KINDS
        return [PY, "-u", "src/kaggriculture/pipeline/submit.py",
                "--agent", "agents/planner_v0.py",
                "--seeds", str(int(g("seeds", 6)))]
    raise ValueError(f"unknown action: {kind}")


# Kinds that upload to Kaggle: both gates apply (server flag + typed SUBMIT).
LIVE_KINDS = {"submit_live", "daily_release_live", "release_resume",
              "promote_planner"}


def _workers():
    return max(1, (os.cpu_count() or 2) - 1)


def _pump(rid, proc, log_path):
    """Stream a job's output to its log file so the browser can tail it."""
    with io.open(log_path, "a", encoding="utf-8", errors="replace") as f:
        for line in iter(proc.stdout.readline, ""):
            f.write(line)
            f.flush()
    code = proc.wait()
    registry.finish_run(rid, code)
    with _LOCK:
        _JOBS.pop(rid, None)


def launch(kind, params):
    # Child processes print notebook titles and agent names that are not
    # cp1252-encodable; a Windows console default turns that into a crash in a
    # tool that otherwise worked.
    """Start a job. Returns (run_id, command)."""
    cmd = _cmd_for(kind, params or {})
    rid = registry.start_run(kind, " ".join(str(c) for c in cmd), meta=params)
    log_path = registry.run_log_path(rid)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with io.open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")

    # One dict, not two -- this was written twice and the first was discarded,
    # so PYTHONUNBUFFERED and KAGG_VERBOSE never reached a child. Streaming
    # survived only because every command already passes -u.
    env = dict(os.environ,
               PYTHONUNBUFFERED="1",
               KAGG_VERBOSE=os.environ.get("KAGG_VERBOSE", "0"),
               PYTHONUTF8="1",
               PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            text=True, bufsize=1, env=env)
    t = threading.Thread(target=_pump, args=(rid, proc, log_path), daemon=True)
    t.start()
    with _LOCK:
        _JOBS[rid] = {"proc": proc, "thread": t, "kind": kind}
    return rid, cmd


def stop(rid):
    with _LOCK:
        job = _JOBS.get(rid)
    if not job:
        return False
    job["proc"].terminate()
    return True


# ------------------------------------------------------------------ payload --

_LIVE_CACHE = {"t": 0.0, "data": {}}


def _kaggle_live():
    """agent file -> live Kaggle status: ref, public score, ACTIVE/PENDING/
    retired, ladder W-L. Cached 5 min (one CLI call + one local file)."""
    now = time.time()
    if _LIVE_CACHE["data"] and now - _LIVE_CACHE["t"] < 300:
        return _LIVE_CACHE["data"]
    out = {}
    try:
        import csv as _csv
        import re as _re
        r = subprocess.run(["kaggle", "competitions", "submissions",
                            "kaggriculture", "-v"], capture_output=True,
                           text=True, timeout=90)
        rows = list(_csv.DictReader(io.StringIO(r.stdout)))
        rows.sort(key=lambda x: x.get("date") or "", reverse=True)
        for i, row in enumerate(rows):
            m = _re.match(r"(\S+\.py)", row.get("description") or "")
            if not m:
                continue
            fname = m.group(1)
            status = (row.get("status") or "").split(".")[-1]
            live = ("PENDING" if status == "PENDING" else "ACTIVE") \
                if i < 2 else "retired"
            try:
                pub = float(row.get("publicScore") or "")
            except ValueError:
                pub = None
            if fname not in out:              # newest submission wins
                out[fname] = {"ref": row.get("ref"), "public": pub,
                              "live": live}
    except Exception:                                              # noqa: BLE001
        pass
    try:
        with io.open(os.path.join(ROOT, "data", "ourgames", "index.json"),
                     encoding="utf-8") as fh:
            d = json.load(fh)
        games = list(d["games"].values()) if isinstance(d["games"], dict) \
            else d["games"]
        per = {}
        for g in games:
            rec = per.setdefault(str(g.get("submission")), [0, 0])
            if g.get("won"):
                rec[0] += 1
            elif not g.get("tied"):
                rec[1] += 1
        for e in out.values():
            wl = per.get(str(e["ref"]))
            if wl:
                e["w"], e["l"] = wl
    except Exception:                                              # noqa: BLE001
        pass
    _LIVE_CACHE.update(t=now, data=out)
    return out


_DATA_CACHE = {"t": 0.0, "data": None}


def _live_data_stats():
    """The REAL data layer, not the retired pre-restructure downloader's:
    route index, staged replays, traces, our-games -- what the pipelines
    actually eat. Cached 60 s (a few listdirs + two json headers)."""
    now = time.time()
    if _DATA_CACHE["data"] and now - _DATA_CACHE["t"] < 60:
        return _DATA_CACHE["data"]
    out = {}

    def _count(path):
        try:
            return len(os.listdir(os.path.join(ROOT, path)))
        except OSError:
            return 0
    try:
        with io.open(os.path.join(ROOT, "data", "routes", "index.json"),
                     encoding="utf-8") as fh:
            idx = json.load(fh)
        out["route_index"] = len(idx.get("routes") or [])
        out["route_index_updated"] = idx.get("updated")
    except Exception:                                              # noqa: BLE001
        out["route_index"] = 0
    try:
        with io.open(os.path.join(ROOT, "data", "ourgames", "index.json"),
                     encoding="utf-8") as fh:
            og = json.load(fh)
        g = og.get("games") or {}
        out["our_games"] = len(g)
        out["our_games_updated"] = og.get("updated")
    except Exception:                                              # noqa: BLE001
        out["our_games"] = 0
    out["staged_replays"] = _count("data/sameday/_stage")
    out["traces_v1"] = _count("data/turntrace")
    out["traces_v2"] = _count("data/trackp/traces_v2")
    out["obsfeat"] = _count("data/obsfeat")
    try:
        out["engine"] = json.load(io.open(os.path.join(
            ROOT, "models", "engine_version.json"),
            encoding="utf-8")).get("engine")
    except Exception:                                              # noqa: BLE001
        out["engine"] = "1.32.6"
    out["fetch"] = _fetch_volumes()
    _DATA_CACHE.update(t=now, data=out)
    return out


def snapshot():
    snap = registry.snapshot()
    snap["data_live"] = _live_data_stats()
    # LIVE ladder truth on every model row + newest models first: local Elo
    # ranks history, but the operator's first question is "what is on the
    # board NOW and how is it doing" (operator request 2026-08-16).
    live = _kaggle_live()
    for m in snap.get("models") or []:
        e = live.get(m.get("name"))
        if e:
            m["live"] = e["live"]
            if e.get("public") is not None:
                m["public_score"] = e["public"]
            if e.get("w") is not None:
                m["ladder_w"], m["ladder_l"] = e["w"], e["l"]
    (snap.get("models") or []).sort(
        key=lambda m: (m.get("live") in ("ACTIVE", "PENDING"),
                       m.get("created") or ""), reverse=True)
    snap["allow_submit"] = ALLOW_SUBMIT
    snap["workers"] = _workers()
    snap["root"] = ROOT
    snap["build"] = BUILD
    snap["files"] = generated_files()
    # Rating-vs-time curves (collected hourly by src/rating_track.py since
    # 2026-08-16 -- Kaggle only exposes the CURRENT score, so history exists
    # only from collection start). Ratings CONVERGE with games played; the
    # curves make retirement-age artifacts visible instead of reading as
    # decline (the v25 lesson).
    curves = {}
    try:
        with open(os.path.join(ROOT, "data", "lb", "rating_track.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                curves.setdefault(
                    f"{r.get('agent', '?')} ({r.get('ref', '?')})", []).append(
                    [r.get("ts"), r.get("score"), r.get("games")])
    except OSError:
        pass
    snap["rating_curves"] = {k: v[-500:] for k, v in curves.items()}

    cfgs = []
    cfg_dir = os.path.join(ROOT, "configs")
    if os.path.isdir(cfg_dir):
        for name in sorted(os.listdir(cfg_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(cfg_dir, name), encoding="utf-8") as f:
                    c = json.load(f)
                cfgs.append({"file": f"configs/{name}", "name": c.get("name", name),
                             "description": c.get("description", ""),
                             "params": c.get("params", {}),
                             "tags": c.get("tags", [])})
            except (OSError, ValueError) as exc:
                cfgs.append({"file": f"configs/{name}", "name": name,
                             "description": f"INVALID: {exc}", "params": {}})
    snap["configs"] = cfgs

    try:
        import kaggriculture.measure.loss_analysis as loss_analysis
        snap["losses"] = [{k: v for k, v in r.items() if k != "series"}
                          for r in loss_analysis.stored_reports()[:40]]
    except Exception:                                              # noqa: BLE001
        snap["losses"] = []

    with _LOCK:
        snap["active"] = sorted(_JOBS)
    return snap


def generated_files():
    """Reports and knowledge graphs the dashboard can open in place."""
    out = []
    for pattern in ("docs/*.html", "docs/*.md", "agents/*.html"):
        import glob as _glob
        for path in sorted(_glob.glob(os.path.join(ROOT, pattern))):
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            out.append({"path": rel, "name": os.path.basename(rel),
                        "bytes": os.path.getsize(path),
                        "kind": "graph" if rel.startswith("agents/") else "doc"})
    return out


def tail(path, n=400):
    if not os.path.exists(path):
        return ""
    try:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return ""


# ------------------------------------------------------------------- server --

class Handler(BaseHTTPRequestHandler):
    server_version = "kaggriculture-dashboard"

    def log_message(self, fmt, *a):                                # quieter
        pass

    def _send(self, code, body, ctype="application/json", filename=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if filename:
            # Downloads get a real filename rather than "snapshot" or the path.
            safe = os.path.basename(str(filename)).replace('"', "")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{safe}"')
        # Belt and braces. A stale page is indistinguishable from a broken one
        # -- every "the buttons do nothing" report so far has been a browser
        # serving HTML from before the fix.
        self.send_header("Cache-Control",
                         "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html")
        if u.path == "/api/snapshot":
            return self._send(200, json.dumps(snapshot(), default=str))
        if u.path == "/api/status":
            if q.get("force"):
                _STATUS_CACHE["t"] = 0.0
            return self._send(200, json.dumps(status_payload(), default=str))
        if u.path == "/api/log":
            rid = (q.get("id") or [""])[0]
            if not rid or "/" in rid or "\\" in rid:
                return self._send(400, json.dumps({"error": "bad id"}))
            with _LOCK:
                running = rid in _JOBS
            return self._send(200, json.dumps({
                "id": rid, "running": running,
                "text": tail(registry.run_log_path(rid))}))
        if u.path == "/api/params":
            name = (q.get("agent") or [""])[0]
            path = _fix_path(name)
            full = os.path.join(ROOT, path)
            if not os.path.exists(full):
                return self._send(404, json.dumps({"error": f"no such agent: {name}"}))
            try:
                import kaggriculture.pipeline.params as paramio
                data = {"agent": path, "params": paramio.load(full)}
                doc = ""
                with io.open(full, encoding="utf-8") as fh:
                    head = fh.read(4000)
                if head.lstrip().startswith('"""'):
                    doc = head.split('"""')[1]
                data["doc"] = doc.strip()
                return self._send(200, json.dumps(data, default=str))
            except Exception as exc:                               # noqa: BLE001
                return self._send(500, json.dumps({"error": str(exc)}))

        if u.path == "/api/files":
            return self._send(200, json.dumps({"files": generated_files()}))

        if u.path == "/api/file":
            rel = _fix_path((q.get("path") or [""])[0])
            full = os.path.abspath(os.path.join(ROOT, rel))
            # Never serve outside the project, and only the types we generate.
            if not full.startswith(os.path.abspath(ROOT)) or \
                    not full.endswith((".html", ".md", ".json", ".csv")) or \
                    not os.path.exists(full):
                return self._send(404, "not found", "text/plain")
            ctype = ("text/html" if full.endswith(".html") else
                     "application/json" if full.endswith(".json") else "text/plain")
            with io.open(full, encoding="utf-8", errors="replace") as fh:
                return self._send(200, fh.read(), ctype)

        if u.path == "/api/artifact":
            return self._artifact(q)

        if u.path == "/api/loss":
            f = (q.get("file") or [""])[0]
            path = os.path.join(ROOT, ".local", "losses", os.path.basename(f))
            if not os.path.exists(path):
                return self._send(404, json.dumps({"error": "no such report"}))
            with io.open(path, encoding="utf-8") as fh:
                return self._send(200, fh.read())
        return self._send(404, json.dumps({"error": "not found"}))

    def _artifact(self, q):
        """Download a model as the `main.py` Kaggle actually wants.

        /api/file deliberately allowlists .html/.md/.json/.csv, which meant the
        one file the whole project exists to produce -- the submission -- could
        not be fetched from the dashboard at all. This is a separate endpoint
        rather than a wider allowlist: it resolves a *registered model name*,
        never a caller-supplied path, so widening what can be downloaded does
        not widen where it can be read from.

        `?format=tar` returns the submission.tar.gz some Kaggle flows expect.
        """
        name = os.path.basename((q.get("model") or [""])[0] or "")
        fmt = ((q.get("format") or ["py"])[0] or "py").lower()
        if not name:
            return self._send(400, json.dumps({"error": "model is required"}))
        if not name.endswith(".py"):
            name += ".py"

        src = os.path.join(ROOT, "agents", name)
        if not os.path.exists(src):
            # build/main.py is a legitimate target too -- it is what was packaged.
            if name == "main.py" and os.path.exists(os.path.join(ROOT, "build", "main.py")):
                src = os.path.join(ROOT, "build", "main.py")
            else:
                return self._send(404, json.dumps(
                    {"error": f"no such model: {name}"}))

        try:
            with open(src, "rb") as fh:
                blob = fh.read()
        except OSError as exc:
            return self._send(500, json.dumps({"error": str(exc)}))

        if fmt in ("tar", "targz", "tar.gz"):
            import io as _io
            import tarfile
            buf = _io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                info = tarfile.TarInfo("main.py")
                info.size = len(blob)
                info.mtime = int(os.path.getmtime(src))
                tar.addfile(info, _io.BytesIO(blob))
            return self._send(200, buf.getvalue(), "application/gzip",
                              filename="submission.tar.gz")

        return self._send(200, blob, "text/x-python", filename="main.py")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, json.dumps({"error": "bad json"}))

        if u.path == "/api/run":
            kind = body.get("kind")
            params = body.get("params") or {}
            # every kind that UPLOADS to Kaggle carries the same two gates
            if kind in LIVE_KINDS:
                if not ALLOW_SUBMIT:
                    return self._send(403, json.dumps({
                        "error": "live submit is disabled. Restart the server "
                                 "with --allow-submit if you mean it."}))
                if body.get("confirm") != "SUBMIT":
                    return self._send(403, json.dumps({
                        "error": "type SUBMIT to confirm"}))
            try:
                rid, cmd = launch(kind, params)
            except (ValueError, OSError) as exc:
                return self._send(400, json.dumps({"error": str(exc)}))
            return self._send(200, json.dumps({"id": rid, "command": cmd}))

        if u.path == "/api/stop":
            return self._send(200, json.dumps({"stopped": stop(body.get("id"))}))

        if u.path == "/api/stop-all":
            with _LOCK:
                ids = list(_JOBS)
            for rid in ids:
                stop(rid)
            return self._send(200, json.dumps({"stopped": len(ids)}))

        if u.path == "/api/clear-runs":
            keep = [r for r in registry.load_runs() if r.get("status") == "running"]
            registry.save_runs(keep)
            return self._send(200, json.dumps({"kept": len(keep)}))

        return self._send(404, json.dumps({"error": "not found"}))


# ------------------------------------------------------------ status page --

_STATUS_CACHE = {"t": 0.0, "data": None}
_TASKS = ("KaggricultureSameDay", "KaggricultureRefreshCycle",
          "KaggricultureTrackP")


def _sched_tasks():
    names = "','".join(_TASKS)
    cmd = ("Get-ScheduledTask -TaskName '" + names + "' | ForEach-Object { "
           "$i = $_ | Get-ScheduledTaskInfo; [pscustomobject]@{"
           "name=$_.TaskName; state=[string]$_.State; "
           "last=[string]$i.LastRunTime; next=[string]$i.NextRunTime; "
           "result=$i.LastTaskResult} } | ConvertTo-Json")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=40)
        d = json.loads(out.stdout)
        return d if isinstance(d, list) else [d]
    except Exception as exc:                                       # noqa: BLE001
        return [{"error": str(exc)}]


def _log_runs(path, n=3):
    """Last n start/exit header pairs from a ===== structured log."""
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return []
    try:
        with open(full, "rb") as fh:
            fh.seek(max(0, os.path.getsize(full) - 120_000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines()
             if "=====" in ln and (" start" in ln or " exit" in ln
                                   or "RESUME" in ln)]
    return lines[-n * 2:]


def _fetch_volumes(path="data/logs/sameday_scrape.log"):
    """What the LAST hourly run actually downloaded/ingested: the operator's
    'how much data came in' question, answered per run."""
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return {}
    try:
        with open(full, "rb") as fh:
            fh.seek(max(0, os.path.getsize(full) - 400_000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    runs = text.split("===== sameday_scrape start")
    last = runs[-1] if len(runs) > 1 else text
    out = {"new_episode_ids": 0, "replays_fetched": 0, "rate_per_s": None,
           "throttle_signals": 0, "final_width": None, "errors_429": 0}
    import re as _re
    for m in _re.finditer(r"(\d+) episode\(s\), (\d+) new", last):
        out["new_episode_ids"] += int(m.group(2))
    m = _re.search(r"adaptive fetch: (\d+) fetches at ([\d.]+)/s, final "
                   r"width (\d+).*?(\d+) throttle", last)
    if m:
        out["replays_fetched"] = int(m.group(1))
        out["rate_per_s"] = float(m.group(2))
        out["final_width"] = int(m.group(3))
        out["throttle_signals"] = int(m.group(4))
    out["errors_429"] = last.count("429")
    m = _re.search(r"ingest(?:ed)?[:\s]+(\d+)", last)
    if m:
        out["ingested"] = int(m.group(1))
    return out


def _read_json(path, default=None):
    try:
        with io.open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                              # noqa: BLE001
        return default


def status_payload():
    now = time.time()
    if _STATUS_CACHE["data"] and now - _STATUS_CACHE["t"] < 30:
        return _STATUS_CACHE["data"]

    grad = _read_json("models/trackp/graduation_report.json", {}) or {}
    conds = {k: (grad.get(k) or {}).get("pass")
             for k in ("c1_crown_margin", "c2_holdout", "c3_guard",
                       "c4_sim_to_real")}
    grad_path = os.path.join(ROOT, "models", "trackp",
                             "graduation_report.json")
    age_h = ((now - os.path.getmtime(grad_path)) / 3600
             if os.path.exists(grad_path) else None)
    all4 = bool(grad.get("graduated_1_to_4"))
    fresh = age_h is not None and age_h < 48
    if all4 and fresh:
        promo = {"verdict": "READY -- OPERATOR GO REQUIRED",
                 "detail": "conditions 1-4 pass on a fresh report; promote "
                           "by submitting agents/planner_v0.py (rotates the "
                           "route out; rollback = resubmit the route)"}
    else:
        failing = [k for k, v in conds.items() if not v]
        promo = {"verdict": "NOT YET",
                 "detail": ("stale graduation report" if all4 and not fresh
                            else "failing: " + ", ".join(failing or ["?"]))}

    pipe = _read_json("models/trackp/pipeline_report.json", {}) or {}
    stages = [{"stage": k, "ok": v.get("ok"), "secs": v.get("secs"),
               "note": v.get("skipped") or v.get("error") or ""}
              for k, v in (pipe.get("stages") or {}).items()]

    data = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_tasks": _sched_tasks(),
        "job_logs": {
            "hourly_fetch": _log_runs("data/logs/sameday_scrape.log"),
            "daily_release": _log_runs("data/logs/daily_release.log"),
            "trackp_daily": _log_runs("data/logs/trackp_daily.log"),
        },
        "hourly_fetch_volumes": _fetch_volumes(),
        "engine": {
            "version": (_read_json("models/engine_version.json", {})
                        or {}).get("engine", "1.32.6"),
            "swap_log": _log_runs("data/logs/engine_swap.log", 2),
        },
        "last_release": _read_json("data/logs/last_release.json", {}),
        "trackp": {
            "pipeline_run": {"date": pipe.get("date"),
                             "total_secs": pipe.get("total_secs"),
                             "failed": [s["stage"] for s in stages
                                        if s["ok"] is False]},
            "stages": stages,
            "phase_board": (_read_json("models/trackp/phase_report.json",
                                       {}) or {}).get("phases"),
            "graduation": {"conditions": conds, "all_1_to_4": all4,
                           "report_age_hours": round(age_h, 1)
                           if age_h is not None else None},
            "phase0": _read_json("models/trackp/phase0_arena.json", {}),
        },
        "promotion": promo,
    }
    _STATUS_CACHE.update(t=now, data=data)
    return data


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggriculture control panel</title><style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#0f1115;color:#e6e8eb}
header{padding:14px 22px;border-bottom:1px solid #232733;display:flex;gap:18px;
       align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#0f1115;z-index:5}
h1{font-size:16px;margin:0;font-weight:600}
.pill{font-size:12px;color:#8b93a1;background:#181b22;border:1px solid #232733;
      border-radius:99px;padding:2px 10px}
nav{display:flex;gap:2px;padding:0 16px;border-bottom:1px solid #232733;flex-wrap:wrap}
nav button{background:none;border:0;color:#8b93a1;padding:10px 14px;cursor:pointer;
           font:inherit;border-bottom:2px solid transparent}
nav button.on{color:#e6e8eb;border-bottom-color:#4f8ef7}
main{padding:20px 22px 60px;max-width:1200px}
section{display:none}section.on{display:block}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:#8b93a1;
   margin:24px 0 8px;font-weight:600}
table{border-collapse:collapse;width:100%;margin-top:6px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #1e222b;vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8b93a1}
td.num{text-align:right;font-variant-numeric:tabular-nums}
td.mono,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
tr.top{background:#12261d}
button.act{background:#1c2231;border:1px solid #2c3446;color:#cfd6e4;border-radius:6px;
           padding:4px 10px;font:inherit;font-size:12px;cursor:pointer;margin:2px 3px 2px 0}
button.act:hover{background:#243049;border-color:#3d5a8f}
button.act:disabled{opacity:.4;cursor:not-allowed}
button.danger{border-color:#5c3030;color:#e8b0b0}
button.danger:hover{background:#3a1f1f}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600}
.green{background:#123a2a;color:#4ad598}.amber{background:#3a3212;color:#e3c04a}
.grey{background:#26292f;color:#98a0ad}.red{background:#3a1c1c;color:#e08a8a}
.card{border:1px solid #232733;border-radius:8px;padding:12px 16px;margin:8px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
label{font-size:12px;color:#8b93a1}
input,select{background:#161922;border:1px solid #2c3446;color:#e6e8eb;border-radius:5px;
             padding:4px 7px;font:inherit;font-size:12px}
input[type=number]{width:80px}
pre{background:#0b0d11;border:1px solid #1e222b;border-radius:8px;padding:12px;
    overflow:auto;max-height:460px;font-size:12px;white-space:pre-wrap;word-break:break-word}
.muted{color:#8b93a1;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{border:1px solid #232733;border-radius:8px;padding:10px 12px}
.stat b{display:block;font-size:20px;font-weight:600}
.stat span{font-size:11px;color:#8b93a1;text-transform:uppercase;letter-spacing:.05em}
a{color:#6ea8ff}
</style></head><body>
<header>
  <h1>Kaggriculture</h1>
  <span class="pill" id="hdr-data">loading</span>
  <span class="pill" id="hdr-best"></span>
  <span class="pill" id="hdr-jobs"></span>
  <span class="pill" id="hdr-submit"></span>
  <span class="pill" id="hdr-build" title="server build; if this does not change after a restart, your browser is serving a cached page (Ctrl-F5)"></span>
</header>
<nav>
  <button data-t="status">Status</button>
  <button data-t="models" class="on">Models</button>
  <button data-t="data">Data</button>
  <button data-t="opponents">Opponents</button>
  <button data-t="losses">Losses</button>
  <button data-t="configs">Configs</button>
  <button data-t="reports">Reports</button>
  <button data-t="ops">Ops</button>
  <button data-t="runs">Runs</button>
</nav>
<main>

<section id="status">
  <h2 style="margin:8px 0">System status <span class="pill" id="st-when"></span>
      <button onclick="loadStatus(true)">refresh</button></h2>
  <div id="st-promo" style="padding:12px 16px;border-radius:8px;margin:10px 0;
       font-weight:600;border:1px solid #232733;background:#181b22"></div>
  <h3>Scheduled jobs</h3><div id="st-jobs"></div>
  <h3>Last runs (from logs)</h3><div id="st-logs" style="font:12px/1.5 ui-monospace,monospace"></div>
  <h3>Engine</h3><div id="st-engine"></div>
  <h3>Track P — daily pipeline stages</h3><div id="st-stages"></div>
  <h3>Track P — roadmap gates</h3><div id="st-phases"></div>
  <h3>Last release</h3><pre id="st-release" style="font-size:12px"></pre>
</section>

<section id="ops">
  <h2 style="margin:8px 0">Operations — every BAU action</h2>
  <h3>Release &amp; publish</h3>
  <p>
    <button onclick="run('daily_release_dry')">Daily release (dry, no upload)</button>
    <button onclick="run('daily_release_live')" class="danger">Daily release + PUBLISH</button>
    <button onclick="run('release_resume')" class="danger">Resume publish (after a killed cycle)</button>
    <button onclick="run('promote_planner')" class="danger">PROMOTE Track P planner (submit)</button>
  </p>
  <h3>Data &amp; fetch</h3>
  <p>
    <button onclick="run('ourgames_all')">Refresh our games</button>
    <button onclick="run('ourgames_by_sub')">Per-submission record</button>
    <button onclick="run('sameday_fetch')">Same-day fetch (8 GB)</button>
    <button onclick="run('backfill_ingest')">Drain ingest backlog</button>
  </p>
  <h3>Engine 1.32.7</h3>
  <p>
    <button onclick="run('engine_swap_status')">Swap watcher status</button>
    <button onclick="run('engine_swap_check')">Check ladder + swap if flipped</button>
  </p>
  <h3>Track P</h3>
  <p>
    <button onclick="run('trackp_phases_status')">Roadmap status</button>
    <button onclick="run('trackp_phase',{phase:prompt('Phase number 0-4','0')})">Run a phase</button>
    <button onclick="run('trackp_pipeline')">Full Track P pipeline</button>
    <button onclick="run('trackp_arena')">Planner vs route (arena)</button>
    <button onclick="run('trackp_guard')">Guard verdict</button>
    <button onclick="run('trackp_graduation')">Graduation report</button>
  </p>
  <h3>Track P — relationship miners</h3>
  <p>
    <button onclick="run('trackp_verdict')">Verdict model</button>
    <button onclick="run('trackp_twins')">Twin studies</button>
    <button onclick="run('trackp_winprob')">Win-prob curves</button>
    <button onclick="run('trackp_insight')">Four-pass insight</button>
    <button onclick="run('trackp_regret',{episode:prompt('Episode id (blank = smoke)','')||undefined})">Counterfactual regret</button>
  </p>
  <h3>Tests</h3>
  <p>
    <button onclick="run('run_all_tests',{fast:1})">All suites (fast)</button>
    <button onclick="run('run_all_tests',{fast:0})">All suites (full)</button>
  </p>
</section>

<section id="models" class="on">
  <h2>Autopilot &mdash; fetch, train, build, gate</h2>
  <p class="muted">One button, five stages, <b>never overlapping</b>: mine fresh
    routes from the top-200 &rarr; re-derive the sale-timing prior and search the
    parameter vector &rarr; package and run the submission contract &rarr; play
    the candidate against the incumbent on <i>seeds it was never tuned on</i>.
    <b>Nothing is promoted unless it wins that gate</b>, and nothing is uploaded
    without a human typing SUBMIT at a console.</p>
  <p class="muted">The architecture never changes &mdash;
    <code>agents/v1_heuristic.py</code> is the only policy source, and a cycle
    rewrites its <code>PARAMS</code> and <code>SELL_SCHEDULE</code> blocks
    against the new data. That is what makes two cycles comparable.
    Budgets come from the core count, so this runs the same stages on Kaggle or
    Colab with fewer trials rather than shorter ones.</p>
  <div class="row">
    <label>stages <select id="ap-stages">
      <option value="fetch,train,build,gate">fetch &rarr; train &rarr; build &rarr; gate</option>
      <option value="fetch,train,build,gate,submit">...and submit (asks first)</option>
      <option value="train,build,gate">train &rarr; build &rarr; gate (reuse data)</option>
      <option value="fetch">fetch only</option>
    </select></label>
    <label>line <select id="ap-line">
      <option value="params">params &mdash; v1_heuristic, new PARAMS</option>
      <option value="route">route &mdash; tape runtime, new mined route</option>
    </select></label>
    <label>top teams <input type="number" id="ap-top" value="200" min="10" max="500"></label>
    <label>per team <input type="number" id="ap-per" value="3" min="1"></label>
    <label>jobs <input type="number" id="ap-jobs" value="8" min="1" max="12"></label>
    <label><input type="checkbox" id="ap-quick"> quick (~15 min)</label>
    <button class="act" onclick="run('autopilot',{stages:v('ap-stages'),line:v('ap-line'),top:+v('ap-top'),per_team:+v('ap-per'),jobs:+v('ap-jobs'),quick:c('ap-quick')})">Run autopilot</button>
    <button class="act" onclick="run('autopilot_status')">Status</button>
    <button class="act" onclick="run('autopilot_notebook')">Emit Kaggle/Colab notebook</button>
  </div>

  <h2>Ladder</h2>
  <p class="muted">Ranked by local Elo. A rating under <b id="ming">16</b> games is not
    actionable &mdash; the interval is wider than the gaps between our agents.
    Win rate is the score; coin margin is only a tie-break.</p>
  <table><thead><tr><th>#</th><th>Model</th><th>Live</th><th>Elo</th><th>Public</th><th>Ladder W-L</th><th>Games</th><th>Win%</th>
    <th>Conf</th><th>Worst ms</th><th>Evals</th><th>Actions</th></tr></thead>
    <tbody id="tb-models"></tbody></table>
  <h2>A/B &mdash; paired head-to-head with the verdict computed</h2>
  <p class="muted">Same seeds, both seats, exact binomial sign test on wins
    &mdash; the instrument behind the demand-model fix (22&ndash;10, p&asymp;0.03)
    and the adaptive sell-timing retirement, as one button. Verdicts are
    stored in <code>models/ab/</code>. Single-match results are worthless;
    this is the honest minimum.</p>
  <div class="row">
    <label>A <select id="ab-a"></select></label>
    <label>B <select id="ab-b"></select></label>
    <label>games/seed set <input type="number" id="ab-n" value="8" min="2" style="width:4em"></label>
    <button onclick="run('ab_paired',{a:$('ab-a').value,b:$('ab-b').value,n:+$('ab-n').value||8})">Run A/B</button>
  </div>
  <h2>Rating curves &mdash; the active pair over time</h2>
  <p class="muted">Ratings <b>converge with games played</b>; a submission
    retired young reads lower than it truly was (v25.0_bandit died at 4 hours
    and looked like decline). Collected hourly by <code>rating_track.py</code>
    since 2026-08-16 &mdash; Kaggle exposes only the current score, so curves
    start at collection start. Compare submissions at <i>equal window age</i>,
    never at retirement.</p>
  <div id="rating-curves"><span class="muted">no snapshots yet</span></div>
  <h2>Bulk</h2>
  <div class="row">
    <label>Elo rounds <input type="number" id="elo-rounds" value="2" min="1"></label>
    <button class="act" onclick="run('elo',{rounds:+v('elo-rounds')})">Rate everything</button>
    <button class="act" onclick="run('test_fast')">Fast tests</button>
    <button class="act" onclick="run('test_full')">Full tests</button>
    <button class="act" onclick="run('refresh_dashboard')">Rebuild static dashboard</button>
  </div>
  <h2>Ensemble in use</h2>
  <p class="muted">Pick which ensemble definition to build from. Each config is a
    different committee shape; building writes a new agent, so switching never
    disturbs a model you have already rated.</p>
  <div class="row">
    <label>config <select id="ens-config"></select></label>
    <button class="act" onclick="run('build',{config:v('ens-config')})">Build this ensemble</button>
    <label>arbiter <select id="ens-arbiter">
      <option value="table">table (bucketed)</option>
      <option value="xgb">xgb (gradient-boosted)</option>
    </select></label>
    <button class="act" onclick="ensArbiter()">Apply arbiter</button>
  </div>

  <h2>Kaggle-exact checks</h2>
  <p class="muted">Most harnesses raise actTimeout to 60 s for throughput. These
    two run at Kaggle's real 1 s, where an overrun means a silently substituted
    action rather than a slow turn. <b>Engine check</b> answers a different
    question: is this even the engine the ladder scores with? A Kaggle notebook
    image has shipped COW at 600 against the ladder's 400 and dropped
    SELL FERTILIZER silently, so numbers from it describe a different game.</p>
  <div class="row">
    <label>agent <select id="ke-agent"></select></label>
    <button class="act" onclick="run('engine_check')">Engine check</button>
    <button class="act" onclick="run('conformance',{agent:v('ke-agent')})">Conformance vs AGENTS.md</button>
    <button class="act" onclick="run('kaggle_verify',{agent:v('ke-agent')})">Verify strict episode</button>
    <button class="act" onclick="run('kaggle_audit')">Audit every call site</button>
  </div>

  <h2>CEM policy search</h2>
  <p class="muted">Samples whole parameter vectors, keeps the elite fraction,
    refits, repeats &mdash; scored on paired both-seat matches against the
    opponents you pick. Use this rather than coordinate descent when the knobs
    are coupled, which the cash-flow ones are. Resumable; state in
    <code>.local/cem/</code>.</p>
  <div class="row">
    <label>base <select id="cem-base"></select></label>
    <label>vs <select id="cem-vs"></select></label>
    <label>generations <input type="number" id="cem-gen" value="8" min="1"></label>
    <label>pop <input type="number" id="cem-pop" value="12" min="4"></label>
    <label>seeds <input type="number" id="cem-seeds" value="3" min="1"></label>
    <label>objective <select id="cem-obj">
      <option value="win">win rate</option>
      <option value="margin">margin</option>
      <option value="bank">bank</option></select></label>
    <button class="act" onclick="run('optimize',{base:v('cem-base'),vs:v('cem-vs'),generations:+v('cem-gen'),popsize:+v('cem-pop'),seeds:+v('cem-seeds'),objective:v('cem-obj'),out:'agents/v9_cem.py'})">Search</button>
    <button class="act" onclick="run('optimize',{resume:1,generations:+v('cem-gen'),out:'agents/v9_cem.py'})">Resume</button>
  </div>

  <h2>Public notebooks</h2>
  <p class="muted">The ranked competitors publish their agent inside a notebook,
    base85+zlib encoded. This pulls them all and decodes the payloads as data
    (never executed). Decoded agents land in
    <code>.local/kernels/_agents/</code>; index in <code>data/notebooks.csv</code>.</p>
  <div class="row">
    <label>top <input type="number" id="nb-top" value="25" min="1"></label>
    <button class="act" onclick="run('notebooks',{top:+v('nb-top')})">Harvest</button>
    <button class="act" onclick="run('notebooks',{top:+v('nb-top'),refresh:1})">Re-pull all</button>
    <button class="act" onclick="run('notebooks_index')">Show index</button>
    <button class="act" onclick="run('test_contract')">Submission contract</button>
  </div>

  <h2>Population-Based Training</h2>
  <p class="muted">A league of workers that exploit and explore each other.
    Single-chain search cannot produce diverse members; this can, which is what
    the ensemble and the arbiter have been short of.</p>
  <div class="row">
    <label>population <input type="number" id="pbt-n" value="6" min="2"></label>
    <label>minutes <input type="number" id="pbt-min" value="60" min="1"></label>
    <label>seeds <input type="number" id="pbt-seeds" value="2" min="1"></label>
    <button class="act" onclick="run('pbt',{population:+v('pbt-n'),minutes:+v('pbt-min'),seeds:+v('pbt-seeds'),agent:v('adapt-base')})">Run PBT</button>
    <button class="act" onclick="run('pbt',{population:+v('pbt-n'),minutes:+v('pbt-min'),resume:1})">Resume</button>
    <button class="act" onclick="run('pbt_harvest',{n:4})">Harvest 4</button>
    <button class="act" onclick="run('pbt_status')">Status</button>
  </div>

  <h2>XGBoost arbiter</h2>
  <p class="muted">Trains offline, ships as plain nested lists &mdash; no xgboost
    at run time. Collect, train, then build.</p>
  <div class="row">
    <label>base <select id="xgb-base"></select></label>
    <label>episodes <input type="number" id="xgb-eps" value="30" min="4"></label>
    <button class="act" onclick="run('xgb_collect',{agent:v('xgb-base'),episodes:+v('xgb-eps')})">Collect</button>
    <label>rounds <input type="number" id="xgb-rounds" value="200" min="10"></label>
    <button class="act" onclick="run('xgb_train',{rounds:+v('xgb-rounds')})">Train</button>
    <button class="act" onclick="run('xgb_build',{agent:v('xgb-base')})">Build agent</button>
    <button class="act" onclick="run('xgb_build',{agent:v('xgb-base'),ab:1})">Build + SPRT</button>
    <button class="act" onclick="run('xgb_status')">Status</button>
  </div>

  <h2>Adaptive play</h2>
  <p class="muted">Off plays the same policy whatever the score.
    <b>bandit</b> runs Exp3 over the committee, reweighting members from an
    in-game wealth signal so the vote leans on whatever is working in this game.
    <b>risk</b> presses when behind late and protects a lead when ahead &mdash;
    on a skill ladder a narrow loss and a wide loss score identically, so
    protecting a losing position is strictly worse than gambling out of it.
    Switching writes a new agent; nothing is edited in place.</p>
  <div class="row">
    <label>mode <select id="adapt-mode">
      <option value="off">off</option><option value="bandit">bandit</option>
      <option value="risk">risk</option><option value="both" selected>both</option>
    </select></label>
    <label>base <select id="adapt-base"></select></label>
    <label>committee k <input type="number" id="adapt-k" value="8" min="0"></label>
    <button class="act" onclick="run('adaptive_set',{agent:v('adapt-base'),mode:v('adapt-mode'),ensemble_k:+v('adapt-k')})">Build variant</button>
    <button class="act" onclick="run('adaptive_set',{agent:v('adapt-base'),mode:v('adapt-mode'),ensemble_k:+v('adapt-k'),ab:1})">Build + SPRT it</button>
    <button class="act" onclick="run('adaptive_list')">Show modes</button>
  </div>

  <h2>Ensemble and arbiter</h2>
  <div class="row">
    <label>committee size <input type="number" id="sel-k" value="4" min="2"></label>
    <label><input type="checkbox" id="sel-guard" checked> drop near-duplicates</label>
    <button class="act" onclick="run('select_build',{k:+v('sel-k'),spread_guard:c('sel-guard')})">Build best-of ensemble</button>
    <label>arbiter minutes <input type="number" id="arb-min" value="45" min="1"></label>
    <button class="act" onclick="run('arbiter',{minutes:+v('arb-min'),agent:v('adapt-base')})">Learn arbiter (CEM)</button>
    <button class="act" onclick="run('parity',{matches:3})">Forecaster parity</button>
    <button class="act" onclick="run('test_system')">System tests</button>
    <button class="act" onclick="run('select_show')">Show committee choice</button>
    <button class="act" onclick="run('improve_status')">Improve status</button>
  </div>

  <h2>Improve (SPRT-gated)</h2>
  <p class="muted">Proposes candidates and keeps only statistically significant gains.
    Most candidates are rejected &mdash; that is the point.</p>
  <div class="row">
    <label>minutes <input type="number" id="imp-min" value="30" min="1"></label>
    <label>elo1 <input type="number" id="imp-elo" value="25" min="1"></label>
    <label>base <select id="imp-base"></select></label>
    <button class="act" onclick="run('improve',{minutes:+v('imp-min'),elo1:+v('imp-elo'),agent:v('imp-base')})">Run improve</button>
  </div>
</section>

<section id="data">
  <h2>Episode data</h2>
  <div class="grid" id="data-stats"></div>
  <h2>Download</h2>
  <p class="muted">Replays stream: fetched, reduced to ~60 numbers, deleted. Disk does
    not grow. <code>per-day</code> is per daily dataset, so the total is per-day &times; days.</p>
  <div class="row">
    <label>days <input type="number" id="dl-days" value="3" min="1"></label>
    <label>per-day <input type="number" id="dl-per" value="60" min="1"></label>
    <label>own (0=all) <input type="number" id="dl-own" value="0" min="0"></label>
    <label>jobs <input type="number" id="dl-jobs" value="8" min="1"></label>
    <label><input type="checkbox" id="dl-any"> any-top</label>
    <button class="act" onclick="run('download',{days:+v('dl-days'),per_day:+v('dl-per'),own:+v('dl-own'),jobs:+v('dl-jobs'),any_top:c('dl-any')})">Download</button>
  </div>
  <div class="row">
    <button class="act" onclick="run('download_check')">Preflight</button>
    <button class="act" onclick="run('download_diagnose')">Diagnose leaderboard</button>
    <button class="act" onclick="run('submissions_diagnose')">Diagnose own games</button>
    <button class="act" onclick="run('scheduler_install')">Install hourly fetch</button>
  </div>
</section>

<section id="opponents">
  <h2>Route mine (current top-30)</h2>
  <p class="muted">A route is a perishable asset. The same agent measured
    <b>19/46</b> on last week's route and <b>40&ndash;41/46</b> on a current one, so
    this refreshes rather than accumulates. Replays stream &mdash; downloaded,
    reduced to a 719-turn route, deleted &mdash; so disk stays flat against a
    21&nbsp;GB/day archive. One medoid route per team, and the newest six per
    team are held out and never opened during selection.</p>
  <div class="row">
    <label>teams <input type="number" id="rt-top" value="30" min="1"></label>
    <label>per team <input type="number" id="rt-per" value="6" min="1"></label>
    <label>budget GB <input type="number" id="rt-gb" value="8" min="1" step="0.5"></label>
    <label>days <input type="number" id="rt-days" value="3" min="1"></label>
    <label>jobs <input type="number" id="rt-jobs" value="8" min="1" max="12"></label>
    <button class="act" onclick="run('routes_mine',{top:+v('rt-top'),per_team:+v('rt-per'),max_gb:+v('rt-gb'),days:+v('rt-days'),jobs:+v('rt-jobs')})">Mine routes</button>
    <button class="act" onclick="run('routes_report')">Route report</button>
  </div>
  <p class="muted"><b>Select</b> builds the top candidates and picks the winner by
    playing them &mdash; recorded bank does not predict head-to-head strength, and
    the candidate with the highest bank in our tournament had the second-worst win
    rate. <b>Panel</b> then plays the winner against one held-out opponent per
    top-N team, drawn from the outer and validation windows that selection never
    opened.</p>
  <div class="row">
    <label>candidates <input type="number" id="rt-n" value="8" min="2"></label>
    <label>seeds <input type="number" id="rt-seeds" value="2" min="1"></label>
    <button class="act" onclick="run('routes_select',{top_n:+v('rt-n'),seeds:+v('rt-seeds')})">Select by playing</button>
    <label>panel agent <select id="pn-agent"></select></label>
    <label>top-N <input type="number" id="pn-top" value="100" min="10" max="500"></label>
    <button class="act" onclick="run('panel',{agent:v('pn-agent'),top:+v('pn-top'),seeds:+v('rt-seeds')})">Run held-out panel</button>
  </div>

  <h2>Leaderboard-topper opponents</h2>
  <p class="muted">Tapes replay a topper's recorded actions with a repair layer;
    clones fit our heuristic to their decisions so they react. Beating a tape is
    necessary, not sufficient. Leave <b>vs</b> on <i>all</i> to play the whole
    set, or pick one topper to simulate against just them.</p>
  <div class="row">
    <button class="act" onclick="run('opponents_tapes')">Build tapes from replays</button>
    <button class="act" onclick="run('opponents_clones',{top:5})">Fit clones (slow)</button>
    <label>agent <select id="opp-agent"></select></label>
    <label>vs <select id="opp-vs"></select></label>
    <label>seeds <input type="number" id="opp-n" value="2" min="1"></label>
    <button class="act" onclick="run('opponents_play',{agent:v('opp-agent'),n:+v('opp-n'),vs:v('opp-vs')})">Play</button>
  </div>
  <table><thead><tr><th>Name</th><th>Kind</th><th>Bank</th><th>Detail</th></tr></thead>
    <tbody id="tb-opps"></tbody></table>
</section>

<section id="losses">
  <h2>Post-mortems</h2>
  <p class="muted">Ranked by how much money each cause is worth, not by when it
    happened. A loss is rarely one bad turn.</p>
  <div class="row">
    <label>agent <select id="loss-agent"></select></label>
    <label>vs <select id="loss-vs"></select></label>
    <label>seeds <input type="number" id="loss-n" value="6" min="1"></label>
    <button class="act" onclick="run('loss_scan',{agent:v('loss-agent'),vs:v('loss-vs'),scan:+v('loss-n')})">Scan for losses</button>
  </div>
  <table><thead><tr><th>Match</th><th>Result</th><th>Margin</th><th>Top cause</th><th></th></tr></thead>
    <tbody id="tb-losses"></tbody></table>
  <pre id="loss-detail" style="display:none"></pre>
</section>

<section id="configs">
  <h2>Model factory</h2>
  <p class="muted">Every agent should be reproducible from a file you can read and
    diff. Unknown knob names are a hard error, so a typo cannot silently build a
    null edit.</p>
  <div class="row"><button class="act" onclick="run('build_all')">Build all configs</button></div>
  <table><thead><tr><th>Config</th><th>Description</th><th>Overrides</th><th></th></tr></thead>
    <tbody id="tb-configs"></tbody></table>
</section>

<section id="reports">
  <h2>Generated reports and knowledge graphs</h2>
  <p class="muted">Everything the tools write. Opens in place &mdash; no need to
    hunt through the folder.</p>
  <div class="row">
    <button class="act" onclick="run('refresh_dashboard')">Rebuild Elo dashboard</button>
    <button class="act" onclick="run('eda')">Rebuild EDA report</button>
    <button class="act" onclick="refresh()">Refresh list</button>
  </div>
  <table><thead><tr><th>File</th><th>Kind</th><th>Size</th><th></th></tr></thead>
    <tbody id="tb-files"></tbody></table>
  <iframe id="viewer" style="display:none;width:100%;height:620px;border:1px solid #232733;border-radius:8px;margin-top:12px;background:#fff"></iframe>
  <pre id="viewer-text" style="display:none"></pre>
</section>

<section id="runs">
  <h2>Runs</h2>
  <div class="row">
    <button class="act" onclick="refresh()">Refresh</button>
    <button class="act danger" onclick="stopAll()">Stop all running</button>
    <button class="act" onclick="clearRuns()">Clear finished</button>
    <span class="muted" id="run-note"></span>
  </div>
  <table><thead><tr><th>When</th><th>Kind</th><th>Status</th><th>Command</th><th></th></tr></thead>
    <tbody id="tb-runs"></tbody></table>
  <h2>Log <span class="muted" id="log-id"></span></h2>
  <pre id="log">select a run</pre>
</section>

</main>
<script>
let S={}, watching=null, logTimer=null;
const $=id=>document.getElementById(id);
const v=id=>$(id).value, c=id=>$(id).checked;
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));
// For values that land inside a JS string literal in an onclick attribute.
// A raw Windows backslash silently disappears there: 'agents\\a.py' becomes
// 'agentsa.py', and the command runs against a file that does not exist.
const jesc=s=>esc(String(s==null?'':s).replace(/\\/g,'/').replace(/'/g,"\\'"));

document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); $(b.dataset.t).classList.add('on');
  if(b.dataset.t==='status') loadStatus();
});

function stTbl(rows,cols){
  if(!rows||!rows.length)return '<i>none</i>';
  let h='<table style="border-collapse:collapse;font-size:12px">'+
    '<tr>'+cols.map(c=>'<th style="text-align:left;padding:3px 10px;color:#8b93a1">'+esc(c)+'</th>').join('')+'</tr>';
  for(const r of rows)h+='<tr>'+cols.map(c=>'<td style="padding:3px 10px;border-top:1px solid #232733">'+esc(r[c])+'</td>').join('')+'</tr>';
  return h+'</table>';
}
async function loadStatus(force){
  const d=await api('/api/status'+(force?'?force=1':''));
  $('st-when').textContent=d.when;
  const p=d.promotion||{};
  const ready=(p.verdict||'').startsWith('READY');
  $('st-promo').style.borderColor=ready?'#2f8f4e':'#232733';
  $('st-promo').style.color=ready?'#7ee2a0':'#e6e8eb';
  $('st-promo').innerHTML='PROMOTE TRACK P? &nbsp; '+esc(p.verdict)+
    '<div style="font-weight:400;color:#8b93a1;margin-top:4px">'+esc(p.detail)+'</div>'+
    (ready?'<div style="margin-top:8px"><button class="danger" onclick="run(\'promote_planner\')">Promote now</button></div>':'');
  $('st-jobs').innerHTML=stTbl(d.scheduled_tasks,['name','state','last','next','result']);
  let lg='';
  const vol=d.hourly_fetch_volumes||{};
  lg+='<b>hourly_fetch — last run volumes</b><br>'+
      esc(`new episode ids: ${vol.new_episode_ids??'-'} · replays fetched: ${vol.replays_fetched??'-'}`+
          (vol.rate_per_s?` @ ${vol.rate_per_s}/s (width ${vol.final_width}, ${vol.throttle_signals} throttle)`:'')+
          ` · 429s: ${vol.errors_429??0}`+(vol.ingested!=null?` · ingested: ${vol.ingested}`:''))+'<br><br>';
  for(const [k,v] of Object.entries(d.job_logs||{}))
    lg+='<b>'+esc(k)+'</b><br>'+(v||[]).map(esc).join('<br>')+'<br><br>';
  $('st-logs').innerHTML=lg;
  $('st-engine').innerHTML='version <b>'+esc(d.engine?.version)+'</b> &nbsp; '+
    '<span style="color:#8b93a1;font-size:12px">'+(d.engine?.swap_log||[]).map(esc).join(' · ')+'</span>';
  $('st-stages').innerHTML=stTbl((d.trackp?.stages||[]).map(s=>({stage:s.stage,
    ok:s.ok===false?'FAIL':(s.ok?'ok':'?'),secs:s.secs,note:s.note})),['stage','ok','secs','note']);
  $('st-phases').innerHTML=stTbl((d.trackp?.phase_board||[]).map(x=>({phase:x.phase,
    name:x.name,gate:(x.gate&&x.gate.passed)?'PASS':'open',needs:x.gate?x.gate.needs:''})),
    ['phase','name','gate','needs']);
  $('st-release').textContent=JSON.stringify(d.last_release||{},null,1);
}

async function api(path,opts){const r=await fetch(path,opts);return r.json();}

const LIVE_KINDS=['submit_live','daily_release_live','release_resume','promote_planner'];
async function run(kind,params){
  const body={kind,params:params||{}};
  if(LIVE_KINDS.includes(kind)){
    const t=prompt('This uploads to Kaggle. 5 submissions a day, only the latest 2 active.\n'+
                   (kind==='promote_planner'?'PROMOTING THE PLANNER ROTATES THE ROUTE OUT.\n':'')+
                   'Type SUBMIT to confirm:');
    if(t!=='SUBMIT'){return;}
    body.confirm=t;
  }
  const res=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
                                  body:JSON.stringify(body)});
  if(res.error){alert(res.error);return;}
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x=>x.classList.remove('on'));
  document.querySelector('nav button[data-t=runs]').classList.add('on');$('runs').classList.add('on');
  watch(res.id); refresh();
}

function watch(id){
  watching=id; $('log-id').textContent=id;
  if(logTimer)clearInterval(logTimer);
  const pull=async()=>{
    const r=await api('/api/log?id='+encodeURIComponent(id));
    $('log').textContent=r.text||'(no output yet)';
    $('log').scrollTop=$('log').scrollHeight;
    if(!r.running){clearInterval(logTimer);logTimer=null;refresh();}
  };
  pull(); logTimer=setInterval(pull,1200);
}

function ensArbiter(){
  const a=$('ens-arbiter').value, base=$('adapt-base').value;
  if(!base){alert('no model selected');return;}
  // The arbiter is a parameter, so switching it is a build like any other.
  run('adaptive_set',{agent:base,mode:adaptiveMode(),arbiter:a});
}
function adaptiveMode(){const e=document.getElementById('adapt-mode');return e?e.value:'both';}
function opts(sel,list,cur){
  const el=$(sel); if(!el)return;
  const keep=el.value||cur;
  el.innerHTML=list.map(m=>`<option value="${esc(String(m.path||m).replace(/\\/g,'/'))}">${esc(m.name||m)}</option>`).join('');
  if(keep)el.value=keep;
}

async function refresh(){
  S=await api('/api/snapshot');
  const d=S.data||{};
  $('hdr-data').textContent=`${(d.episodes||0).toLocaleString()} episodes`;
  const best=(S.models||[]).find(m=>m.exists);
  $('hdr-best').textContent=best?`best ${best.name} ${Math.round(best.elo)}`:'no models';
  $('hdr-jobs').textContent=(S.active||[]).length?`${S.active.length} running`:'idle';
  $('hdr-submit').textContent=S.allow_submit?'submit enabled':'submit locked';
  $('hdr-build').textContent='build '+(S.build||'?');
  if(window.__build && window.__build!==S.build){
    $('hdr-build').textContent='build '+S.build+' - RELOAD (Ctrl-F5)';
    $('hdr-build').style.background='#3a1c1c';
  }
  window.__build=S.build;
  $('ming').textContent=S.min_games;

  const live=(S.models||[]).filter(m=>m.exists);
  opts('imp-base',live); opts('opp-agent',live); opts('loss-agent',live);
  opts('pn-agent',live);
  opts('loss-vs',live); opts('adapt-base',live); opts('ke-agent',live);
  opts('cem-base',live);
  // CEM is only worth running against opponents that mean something: the
  // recorded ladder tapes first, then our own agents.
  // "all" first, so the default stays the old whole-set behaviour.
  const oppvs=$('opp-vs');
  if(oppvs){const keep=oppvs.value;
    oppvs.innerHTML='<option value="">all opponents</option>'+
      (S.opponents||[]).map(o=>`<option value="${esc(o.name)}">${esc(o.name)}</option>`).join('');
    if(keep)oppvs.value=keep;}
  const cemvs=$('cem-vs');
  if(cemvs){const keep=cemvs.value;
    const tapes=(S.opponents||[]).map(o=>({name:o.name,path:o.path}));
    const rows=tapes.concat(live.map(m=>({name:m.name,path:m.path})));
    cemvs.innerHTML=rows.map(r=>`<option value="${esc(String(r.path).replace(/\\/g,'/'))}">${esc(r.name)}</option>`).join('');
    if(keep)cemvs.value=keep;}
  opts('xgb-base',live.filter(m=>/ensemble|vsel|vadapt|vxgb/.test(m.name)).length
       ? live.filter(m=>/ensemble|vsel|vadapt|vxgb/.test(m.name)) : live);
  opts('ab-a',live); opts('ab-b',live);
  const ecfg=$('ens-config');
  if(ecfg){const keep=ecfg.value;
    ecfg.innerHTML=(S.configs||[]).map(cf=>`<option value="${esc(String(cf.file).replace(/\\/g,'/'))}">${esc(cf.name)}</option>`).join('');
    if(keep)ecfg.value=keep;}

  $('tb-models').innerHTML=(S.models||[]).map((m,i)=>{
    const rated=m.rated?'<span class="badge green">rated</span>'
      :`<span class="badge amber">${m.games||0} games</span>`;
    const gone=m.exists?'':' <span class="badge red">missing</span>';
    const conf=m.confidence?('&plusmn;'+Math.round(m.confidence)):'&plusmn;&infin;';
    const a=jesc(m.path||('agents/'+m.name));
    const acts=m.exists?`
      <button class="act" onclick="run('evaluate',{agent:'${a}'})">Eval</button>
      <button class="act" onclick="run('elo',{agent:'${a}',rounds:2})">Elo</button>
      <button class="act" onclick="run('loss_scan',{agent:'${a}',scan:6})">Losses</button>
      <button class="act" onclick="showParams('${a}')">Params</button>
      <button class="act" onclick="run('model_graph',{agent:'${a}'})">Graph</button>
      <button class="act" onclick="dl('${jesc(m.name)}')">Download</button>
      <button class="act" onclick="run('adaptive_set',{agent:'${a}',mode:adaptiveMode()})">Adaptive</button>
      <button class="act" onclick="run('submit_dry',{agent:'${a}'})">Validate</button>
      <button class="act" onclick="run('notebook_build',{agent:'${a}'})">Notebook</button>
      <button class="act danger" ${S.allow_submit?'':'disabled'}
        onclick="run('submit_live',{agent:'${a}'})">Submit</button>`:'';
    // Local Elo and the public score side by side: the only way to see
    // whether local measurement is tracking the ladder or drifting from it.
    const pub=m.public_score?`<b>${Math.round(m.public_score)}</b>`:'<span class="muted">-</span>';
    const lat=m.worst_ms?(Math.round(m.worst_ms)+(m.worst_ms>250?' !':'')):'<span class="muted">-</span>';
    const nev=(m.evals||[]).length;
    const evs=nev?`${nev}`:'<span class="muted">-</span>';
    const liveB=m.live==='ACTIVE'?'<span class="badge green">ACTIVE</span>'
      :m.live==='PENDING'?'<span class="badge amber">PENDING</span>'
      :m.live==='retired'?'<span class="badge">retired</span>'
      :'<span class="muted">-</span>';
    const lwl=(m.ladder_w!=null)?`${m.ladder_w}-${m.ladder_l}`:'<span class="muted">-</span>';
    return `<tr class="${m.live==='ACTIVE'?'top':''}"><td>${i+1}</td>
      <td class="mono">${esc(m.name)}${gone}</td>
      <td>${liveB}</td>
      <td class="num"><b>${Math.round(m.elo||0)}</b></td>
      <td class="num">${pub}</td>
      <td class="num">${lwl}</td>
      <td class="num">${m.games||0}</td>
      <td class="num">${Math.round((m.win_rate||0)*100)}%</td>
      <td class="num">${conf} ${rated}</td>
      <td class="num">${lat}</td>
      <td class="num">${evs}</td><td>${acts}</td></tr>`;
  }).join('');

  // Rating-vs-time sparkline per tracked submission (inline SVG, no deps).
  const rc=S.rating_curves||{};
  const rcKeys=Object.keys(rc).filter(k=>(rc[k]||[]).length>0);
  const rcDiv=$('rating-curves');
  if(rcDiv&&rcKeys.length){
    rcDiv.innerHTML=rcKeys.map(k=>{
      const pts=rc[k];
      const ys=pts.map(p=>+p[1]).filter(v=>!isNaN(v));
      if(!ys.length)return'';
      const lo=Math.min(...ys),hi=Math.max(...ys),span=Math.max(1,hi-lo);
      const W=460,H=64,n=pts.length;
      const xy=pts.map((p,i)=>{
        const x=n>1?i*(W-10)/(n-1)+5:W/2;
        const y=H-8-((+p[1]-lo)/span)*(H-16);
        return `${x.toFixed(1)},${y.toFixed(1)}`;}).join(' ');
      const last=pts[pts.length-1];
      return `<div style="margin:6px 0">
        <span class="mono">${esc(k)}</span>
        <span class="muted">now <b>${Math.round(+last[1])}</b>
          (${last[2]||0} games, ${esc(String(last[0]).slice(5,16))})
          range ${Math.round(lo)}&ndash;${Math.round(hi)}</span><br>
        <svg width="${W}" height="${H}" style="background:rgba(128,128,128,.08);border-radius:4px">
          ${n>1?`<polyline points="${xy}" fill="none" stroke="currentColor" stroke-width="1.5"/>`
                :`<circle cx="${W/2}" cy="${H/2}" r="2.5" fill="currentColor"/>`}
        </svg></div>`;
    }).join('')||'<span class="muted">no snapshots yet</span>';
  }

  const dl=S.data_live||{};
  const fv=dl.fetch||{};
  $('data-stats').innerHTML=[
    ['route index',(dl.route_index||0).toLocaleString()],
    ['staged replays',(dl.staged_replays||0).toLocaleString()],
    ['our games',(dl.our_games||0).toLocaleString()],
    ['traces v1',(dl.traces_v1||0).toLocaleString()],
    ['traces v2',(dl.traces_v2||0).toLocaleString()],
    ['obs features',(dl.obsfeat||0).toLocaleString()],
    ['engine',dl.engine||'?'],
    ['index updated',dl.route_index_updated?String(dl.route_index_updated).replace('T',' ').slice(0,16):'never'],
    ['last fetch: new ids',(fv.new_episode_ids??0).toLocaleString()],
    ['last fetch: replays',(fv.replays_fetched??0).toLocaleString()+(fv.rate_per_s?(' @ '+fv.rate_per_s+'/s'):'')],
  ].map(([k,val])=>`<div class="stat"><b>${esc(val)}</b><span>${k}</span></div>`).join('');

  $('tb-opps').innerHTML=(S.opponents||[]).length?(S.opponents||[]).map(o=>
    `<tr><td class="mono">${esc(o.name)}</td><td>${esc(o.kind||'')}</td>
     <td class="num">$${Math.round(o.bank||0).toLocaleString()}</td>
     <td class="muted">${o.kind==='clone'?Math.round((o.agreement||0)*100)+'% agreement':(o.steps||0)+' steps'}</td></tr>`
  ).join(''):'<tr><td colspan="4" class="muted">none yet &mdash; build tapes from downloaded replays</td></tr>';

  $('tb-losses').innerHTML=(S.losses||[]).length?(S.losses||[]).map(l=>
    `<tr><td class="mono">${esc(l.file||'')}</td>
     <td><span class="badge ${l.result==='lost'?'red':'green'}">${esc(l.result)}</span></td>
     <td class="num">${Math.round(l.margin||0).toLocaleString()}</td>
     <td class="muted">${esc((l.causes&&l.causes[0]&&l.causes[0].cause)||'nothing structural')}</td>
     <td><button class="act" onclick="showLoss('${jesc(l.file)}')">Open</button></td></tr>`
  ).join(''):'<tr><td colspan="5" class="muted">no post-mortems yet</td></tr>';

  $('tb-configs').innerHTML=(S.configs||[]).map(cf=>
    `<tr><td class="mono">${esc(cf.name)}</td><td class="muted">${esc(cf.description)}</td>
     <td class="mono muted">${esc(JSON.stringify(cf.params))}</td>
     <td><button class="act" onclick="run('build',{config:'${jesc(cf.file)}'})">Build</button></td></tr>`
  ).join('')||'<tr><td colspan="4" class="muted">no configs</td></tr>';

  $('tb-runs').innerHTML=(S.runs||[]).map(r=>{
    const cls=r.status==='ok'?'green':(r.status==='running'?'amber':(r.status==='orphaned'?'grey':'red'));
    return `<tr><td class="muted">${esc((r.started||'').replace('T',' ').slice(5,16))}</td>
      <td>${esc(r.kind)}</td><td><span class="badge ${cls}">${esc(r.status)}</span></td>
      <td class="mono muted">${esc((r.command||'').slice(0,110))}</td>
      <td><button class="act" onclick="watch('${jesc(r.id)}')">Log</button>
      ${r.status==='running'?`<button class="act danger" onclick="stopRun('${jesc(r.id)}')">Stop</button>`:''}</td></tr>`;
  }).join('')||'<tr><td colspan="5" class="muted">nothing has run yet</td></tr>';
  $('run-note').textContent=`${(S.runs||[]).length} recorded, ${S.workers} workers available`;

  $('tb-files').innerHTML=(S.files||[]).map(f=>
    `<tr><td class="mono">${esc(f.path)}</td><td>${esc(f.kind)}</td>
     <td class="num">${Math.round(f.bytes/1024)} KB</td>
     <td><button class="act" onclick="openFile('${jesc(f.path)}')">Open</button></td></tr>`
  ).join('')||'<tr><td colspan="4" class="muted">nothing generated yet</td></tr>';
}

function openFile(path){
  const isHtml=path.endsWith('.html');
  $('viewer').style.display=isHtml?'block':'none';
  $('viewer-text').style.display=isHtml?'none':'block';
  if(isHtml){$('viewer').src='/api/file?path='+encodeURIComponent(path);}
  else{fetch('/api/file?path='+encodeURIComponent(path)).then(r=>r.text())
        .then(t=>{$('viewer-text').textContent=t;});}
}

// Download the agent as the main.py Kaggle wants. Navigating rather than
// fetching lets the browser do the save dialog and the filename itself.
function dl(name){ window.location='/api/artifact?model='+encodeURIComponent(name); }

async function showParams(agent){
  const d=await api('/api/params?agent='+encodeURIComponent(agent));
  if(d.error){alert(d.error);return;}
  const lines=[d.agent,'',(d.doc||'').trim(),'','PARAMS:'];
  Object.keys(d.params||{}).sort().forEach(k=>lines.push('  '+k.padEnd(26)+JSON.stringify(d.params[k])));
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x=>x.classList.remove('on'));
  document.querySelector('nav button[data-t=reports]').classList.add('on');
  $('reports').classList.add('on');
  $('viewer').style.display='none';
  $('viewer-text').style.display='block';
  $('viewer-text').textContent=lines.join('\n');
}

async function stopAll(){await api('/api/stop-all',{method:'POST',
  headers:{'Content-Type':'application/json'},body:'{}'});refresh();}
async function clearRuns(){await api('/api/clear-runs',{method:'POST',
  headers:{'Content-Type':'application/json'},body:'{}'});refresh();}

async function stopRun(id){await api('/api/stop',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});refresh();}

async function showLoss(file){
  const l=await api('/api/loss?file='+encodeURIComponent(file));
  const lines=[`${(l.result||'').toUpperCase()}  $${Math.round(l.final_bank||0).toLocaleString()} vs $${Math.round(l.opp_bank||0).toLocaleString()}  (${Math.round(l.margin||0).toLocaleString()})`,
    `agent ${l.agent||'?'}   vs ${l.opponent||'?'}   seed ${l.seed}`,
    `worst day ${l.worst_day} (${Math.round(l.worst_day_swing||0).toLocaleString()} that day)`,''];
  (l.causes||[]).forEach(x=>{lines.push(`~$${Math.round(x.weight).toLocaleString()}  ${x.cause}`);
                             lines.push(`            ${x.detail}`);});
  if(!(l.causes||[]).length)lines.push('No structural cause stood out: decided by small margins.');
  $('loss-detail').style.display='block'; $('loss-detail').textContent=lines.join('\n');
}

refresh(); setInterval(()=>{if(!logTimer)refresh();},8000);
</script></body></html>"""


def main():
    global ALLOW_SUBMIT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--allow-submit", action="store_true",
                    help="enable the live submit button (still needs a typed SUBMIT)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    ALLOW_SUBMIT = args.allow_submit
    global BUILD
    BUILD = hashlib.sha1(
        (PAGE + str(os.path.getmtime(os.path.abspath(__file__)))).encode()
    ).hexdigest()[:8]

    # Rows left at "running" belong to a server that is no longer here; this
    # process is the only thing that could ever finish them.
    orphans = registry.reap_orphans()

    url = f"http://127.0.0.1:{args.port}"
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Kaggriculture control panel  ->  {url}")
    print(f"  project      : {ROOT}")
    if orphans:
        print(f"  reaped       : {len(orphans)} orphaned run(s) from a previous server")
    print(f"  python       : {PY}")
    print(f"  workers      : {_workers()}")
    print(f"  live submit  : {'ENABLED' if ALLOW_SUBMIT else 'locked (start with --allow-submit)'}")
    print(f"  build        : {BUILD}   (shown in the header; if the browser "
          f"shows a different one, hard-refresh)")
    print("  bound to 127.0.0.1 only -- this process runs commands by design")
    print("\nCtrl-C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        with _LOCK:
            for job in _JOBS.values():
                job["proc"].terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
