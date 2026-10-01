"""Pick the best agent, validate it the way Kaggle will, and submit it.

Everything here follows the competition Overview page:

* **Ranking is win/loss only.** "The actual coin difference in a match does not
  affect the rating change - only the win, loss, or tie outcome matters." So
  candidates are ranked by **win rate**, not by bank size; bank is a tie-break.
* **A Validation Episode is run on upload**, in which "your agent plays against
  a copy of itself to ensure it runs without errors". We run exactly that
  locally first, under the stock configuration, so an Error submission never
  burns one of the day's slots.
* **Daily submission limit is 5**, and **only the latest 2 submissions are
  active** -- they are also what the final leaderboard is computed from. That
  makes submitting a *worse* agent actively harmful: it can push a better one
  out of the active pair. The script shows you today's usage and what the
  active pair will look like afterwards, before asking.
* **Submission size limit is 100 MiB**; agents run on 1.6 vCPU / 6.5 GiB RAM,
  which is slower than a dev box, so the turn-latency check keeps a wide margin
  under the 1-second actTimeout.

Nothing is submitted without your explicit confirmation.

Usage
-----
    python -m kaggriculture.pipeline.submit                      # rank, validate, confirm, submit
    python -m kaggriculture.pipeline.submit --dry-run            # everything except the upload
    python -m kaggriculture.pipeline.submit --agent agents/v2_tuned.py   # skip the ranking
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import datetime as _dt
import glob
import importlib.util
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (enables vendor fallback)
import kaggriculture.data.episodes as ep  # noqa: E402  (CLI resolution + API fallback)

COMPETITION = "kaggriculture"
SIZE_LIMIT = 100 * 1024 * 1024          # 100 MiB, per the FAQ
LATENCY_LIMIT_MS = 250.0                # generous margin under a 1 s actTimeout
STOCK_CONFIG = {"episodeSteps": 720, "actTimeout": 1, "runTimeout": 1200}
# v33 lesson (2026-08-27): Kaggle's validation mirror banked 7,777 while the
# identical file read 96k locally, and the old gate passed it because it only
# checked completion. The floor is seed-sensitive, so it must fail on EVERY
# retry seed to block a submission.
VALIDATION_BANK_FLOOR = 60_000
VALIDATION_SEEDS = (17, 51, 93)


def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def ask(prompt, default=""):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  (no interactive input available)")
        return default


def _mute_stdin():
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
    except OSError:
        pass


# ------------------------------------------------------------------ ranking
def _play(job):
    left, right, seed = job
    from kaggle_environments import make
    cfg = dict(STOCK_CONFIG)
    cfg["actTimeout"] = 60          # rank quickly; latency is checked separately
    cfg["runTimeout"] = 100000
    cfg["seed"] = seed
    env = make(COMPETITION, configuration=cfg, debug=False)
    env.run([left, right])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


def rank_agents(candidates, seeds, workers):
    """Round-robin, both seats. Returns rows sorted best-first by win rate."""
    jobs, meta = [], []
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if i >= j:
                continue
            for s in seeds:
                jobs.append((a, b, s)); meta.append((a, b))
                jobs.append((b, a, s)); meta.append((b, a))
    if not jobs:
        return [{"agent": candidates[0], "win_pct": 100.0, "mean_bank": 0.0, "n": 0}]

    with ProcessPoolExecutor(max_workers=workers, initializer=_mute_stdin) as ex:
        results = list(ex.map(_play, jobs))

    wins, games, banks = {a: 0 for a in candidates}, {a: 0 for a in candidates}, \
                         {a: [] for a in candidates}
    for (left, right), (r0, r1) in zip(meta, results):
        games[left] += 1; games[right] += 1
        banks[left].append(r0); banks[right].append(r1)
        if r0 > r1:
            wins[left] += 1
        elif r1 > r0:
            wins[right] += 1
        else:
            wins[left] += 0.5; wins[right] += 0.5

    rows = [{"agent": a,
             "win_pct": 100.0 * wins[a] / max(1, games[a]),
             "mean_bank": statistics.mean(banks[a]) if banks[a] else 0.0,
             "n": games[a]} for a in candidates]
    # Win rate is what the ladder scores; bank only breaks ties.
    rows.sort(key=lambda r: (-r["win_pct"], -r["mean_bank"]))
    return rows


# --------------------------------------------------------------- validation
def validate(agent_path):
    """Reproduce Kaggle's upload checks locally.

    Returns (ok, findings, metrics). The metrics dict carries the numbers the
    findings only ever rendered into prose -- size, latency, the self-play
    banks -- so the registry can store what this run actually measured instead
    of someone re-typing it from the console later.
    """
    findings, ok = [], True
    metrics = {}

    # Lab instruments never ship. v22_agent --lab-force-arms stamps its output
    # because that build deliberately bypasses the ARM GUARD to generate
    # counterfactual evidence -- uploading one would re-run the v23.1 collapse.
    head = open(agent_path, encoding="utf-8", errors="replace").read(400)
    if "LAB BUILD -- NEVER SUBMIT" in head:
        return False, ["FAIL  this is a stamped LAB BUILD (guard-bypassing "
                       "instrument); it must never be uploaded"], metrics

    size = os.path.getsize(agent_path)
    metrics["bytes"] = size
    if size > SIZE_LIMIT:
        ok = False
        findings.append(f"FAIL  size {size/1e6:.1f} MB exceeds the 100 MiB limit")
    else:
        findings.append(f"ok    size {size/1024:.0f} KiB (limit 100 MiB)")

    spec = importlib.util.spec_from_file_location("cand", agent_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                       # noqa: BLE001
        return False, findings + [f"FAIL  the file does not import: {exc}"], metrics
    if not hasattr(mod, "agent"):
        return False, findings + ["FAIL  no `agent` function defined"], metrics
    findings.append("ok    defines agent()")

    # Kaggle's Validation Episode: the agent against a copy of itself.
    # Run with a bank floor: seed-to-seed spread is real, so the floor only
    # blocks when EVERY retry seed lands below it.
    from kaggle_environments import make
    worst, total, n = 0.0, 0.0, 0
    two_arg = mod.agent.__code__.co_argcount > 1

    def timed(obs, cfg=None):
        nonlocal worst, total, n
        t0 = time.time()
        out = mod.agent(obs, cfg) if two_arg else mod.agent(obs)
        dt = time.time() - t0
        worst = max(worst, dt); total += dt; n += 1
        return out

    attempts, env = [], None
    for seed in VALIDATION_SEEDS:
        cfg = dict(STOCK_CONFIG)
        cfg["actTimeout"] = 60      # measure the latency instead of forfeiting on it
        cfg["runTimeout"] = 100000
        cfg["seed"] = seed
        env = make(COMPETITION, configuration=cfg, debug=True)
        try:
            env.run([timed, agent_path])
        except Exception as exc:                                   # noqa: BLE001
            return False, findings + [f"FAIL  self-play episode raised: {exc}"], metrics

        statuses = [s["status"] for s in env.steps[-1]]
        rewards = [float(s["reward"] or 0) for s in env.steps[-1]]
        attempts.append({"seed": seed, "statuses": statuses,
                         "banks": [round(r, 1) for r in rewards]})
        if any(s not in ("DONE", "ACTIVE") for s in statuses):
            metrics["selfplay_status"] = list(statuses)
            metrics["selfplay_banks"] = [round(r, 1) for r in rewards]
            return False, findings + [
                f"FAIL  self-play episode ended {statuses} -- Kaggle would "
                f"mark this submission Error"], metrics
        if min(rewards) >= VALIDATION_BANK_FLOOR:
            break

    statuses = attempts[-1]["statuses"]
    rewards = [float(b) for b in attempts[-1]["banks"]]
    metrics["selfplay_status"] = list(statuses)
    metrics["selfplay_banks"] = attempts[-1]["banks"]
    metrics["selfplay_attempts"] = attempts
    if min(rewards) < VALIDATION_BANK_FLOOR:
        ok = False
        tried = ", ".join(f"seed {a['seed']}: {min(a['banks']):,.0f}" for a in attempts)
        findings.append(
            f"FAIL  self-play bank below the {VALIDATION_BANK_FLOOR:,} floor on "
            f"every seed ({tried}) -- the v33 collapse shipped through a "
            f"completion-only gate; a healthy shell mirrors at 90k+")
    else:
        findings.append(f"ok    self-play validation episode completed "
                        f"({statuses[0]}), banks {rewards[0]:,.0f} / {rewards[1]:,.0f}"
                        f" (seed {attempts[-1]['seed']}, attempt {len(attempts)})")

    # Action-stream structure (the Kaito-v47 failure class: locally perfect,
    # on-ladder PASS-only). Checked on the last completed episode.
    def _idle(action):
        if not isinstance(action, dict):
            return True
        farmer = action.get("farmer") or []
        hands = action.get("hands") or []
        market = action.get("market") or []
        busy = ([farmer] if farmer and farmer[0] != "PASS" else []) \
            + [h for h in hands if h and h[0] != "PASS"] + list(market)
        return not busy

    acts = [(step[0].get("action") if step and isinstance(step[0], dict) else None)
            for step in env.steps[1:]]
    first_act = next((a for a in acts if a is not None), None)
    sold = sorted({str(o[1]) for a in acts if isinstance(a, dict)
                   for o in (a.get("market") or [])
                   if isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL"})
    run_len, longest_idle = 0, 0
    for a in acts:
        run_len = run_len + 1 if _idle(a) else 0
        longest_idle = max(longest_idle, run_len)
    metrics["selfplay_longest_idle"] = longest_idle
    metrics["selfplay_sell_items"] = sold
    # A single idle turn 0 is a legitimate strategy (Kaileh57's top-15
    # program passes turn 0 -- it doubles as a fingerprint dodge, leaving
    # money=3000 at step 1). The v47 fallback signature is a DEAD OPENING,
    # not a quiet first turn: fail only when a whole day opens idle.
    opening_idle = 0
    for a in acts[:24]:
        if not _idle(a):
            break
        opening_idle += 1
    if longest_idle >= len(acts):
        ok = False
        findings.append("FAIL  the whole episode is PASS -- fallback cascade")
    elif opening_idle >= 24:
        ok = False
        findings.append("FAIL  the entire first day is PASS/empty -- the "
                        "exception-to-fallback signature")
    else:
        turn0 = "idle turn 0 (deliberate), " if (first_act is None or
                                                 _idle(first_act)) else ""
        findings.append(f"ok    action stream: {turn0}longest idle run "
                        f"{longest_idle} turns, sells {len(sold)} product classes "
                        f"({', '.join(sold) or 'none'})")

    mean_ms, worst_ms = 1000 * total / max(1, n), 1000 * worst
    metrics["mean_ms"] = round(mean_ms, 2)
    metrics["worst_ms"] = round(worst_ms, 2)
    if worst_ms > LATENCY_LIMIT_MS:
        ok = False
        findings.append(f"FAIL  worst turn {worst_ms:.0f} ms; Kaggle allows 1000 ms "
                        f"but runs on 1.6 vCPU, so we require < {LATENCY_LIMIT_MS:.0f} ms here")
    else:
        findings.append(f"ok    turn latency mean {mean_ms:.1f} ms, worst "
                        f"{worst_ms:.1f} ms (limit 1000 ms on slower hardware)")
    return ok, findings, metrics


# ------------------------------------------------------------------- kaggle
def submission_status():
    """(submissions_today, active_pair_descriptions) or (None, []) if unknown."""
    try:
        out = subprocess.run(["kaggle", "competitions", "submissions",
                              COMPETITION, "--csv"],
                             capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, []
    if out.returncode != 0:
        return None, []
    rows = list(csv.DictReader(io.StringIO(out.stdout)))
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    n_today = sum(1 for r in rows
                  if str(r.get("date", "")).startswith(today))
    latest = []
    for r in rows[:2]:
        latest.append(f"{r.get('date', '?')}  {r.get('description') or r.get('fileName', '?')}")
    return n_today, latest



def _record(best, main_py, metrics, msg, how):
    """Write what we just shipped into the model registry.

    record_submission() has existed since the registry was written and had zero
    call sites, so the submission history was maintained by hand -- 2 of 32
    models carried one, and only one of those had a real Kaggle ref. Everything
    here is already on the stack; not storing it was pure loss.
    """
    try:
        import kaggriculture.data.registry as registry
        name = os.path.basename(best)
        row = {"artefact": os.path.relpath(main_py, ROOT),
               "message": msg, "how": how,
               "bytes": metrics.get("bytes"),
               "mean_ms": metrics.get("mean_ms"),
               "worst_ms": metrics.get("worst_ms"),
               "selfplay_banks": metrics.get("selfplay_banks"),
               "sha256": _sha256(main_py),
               "status": "SUBMITTED - pending validation episode"}
        ref = _latest_submission_ref()
        if ref:
            row["ref"] = ref
        registry.record_submission(name, **row)
        # Latency is a property of the model, not of one upload, so it also
        # goes on the record itself where the dashboard column reads it.
        fields = {k: metrics[k] for k in ("mean_ms", "worst_ms") if k in metrics}
        if fields:
            registry.register_model(name, latency_ms=metrics.get("worst_ms"),
                                    **fields)
        print(f"  recorded in the model registry as {name}"
              + (f" (ref {ref})" if ref else ""))
    except Exception as exc:                                       # noqa: BLE001
        print(f"  (registry not updated: {exc})")


def _wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _now_iso():
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def describe(agent_path, metrics=None, notebook=None):
    """Build the submission description from what we actually measured.

    Kaggle shows this string next to the score forever, and it is the only
    context you get when you come back in a week and see four rows of
    `main.py`. Typing it by hand means it drifts from the evidence, so it is
    assembled from the registry: architecture, where the route came from, the
    measured results, and the latency the validation episode just recorded.
    """
    name = os.path.basename(agent_path)
    parts = []
    try:
        import kaggriculture.data.registry as registry
        card = registry.card(name) or {}
    except Exception:                                              # noqa: BLE001
        card = {}

    prov = card.get("route_provenance") or {}
    if prov:
        parts.append(
            f"{name}: open-loop 719-turn route + safety stack. "
            f"Route {prov.get('route_id', '?')} from team "
            f"{prov.get('team', '?')} (rank {prov.get('rank_at_mine') or prov.get('rank', '?')} "
            f"at capture), episode {prov.get('episode', '?')} seat "
            f"{prov.get('seat', '?')}, {prov.get('window', '?')} window, mined "
            f"from Kaggle's public episode archive.")
    else:
        base = card.get("base") or "agents/v1_heuristic.py"
        parts.append(f"{name}: closed-loop planner, {os.path.basename(base)} "
                     f"with a searched PARAMS block.")

    if card.get("description"):
        parts.append(str(card["description"]))

    evals = card.get("evals") or []
    if evals:
        shown = []
        for e in evals[:6]:
            opp = os.path.basename(str(e.get("opponent", "?")))
            win = e.get("win_rate")
            margin = e.get("margin")
            if win is None or margin is None:
                continue
            shown.append(f"{opp} {100 * float(win):.0f}%/{float(margin):+,.0f}")
        if shown:
            parts.append("Measured (win%/margin, both seats): " + "; ".join(shown) + ".")

    if card.get("elo") and card.get("games"):
        parts.append(f"Local Elo {float(card['elo']):.0f} over "
                     f"{int(card['games'])} matches "
                     f"({100 * float(card.get('win_rate') or 0):.0f}%).")

    if metrics:
        parts.append(
            f"Validation: self-play {metrics.get('selfplay_status', ['?'])[0]}, "
            f"banks {'/'.join(f'{b:,.0f}' for b in (metrics.get('selfplay_banks') or []))}, "
            f"turn latency mean {metrics.get('mean_ms', '?')} ms / worst "
            f"{metrics.get('worst_ms', '?')} ms, {metrics.get('bytes', 0):,} bytes.")

    if notebook:
        parts.append(f"Write-up with EDA and provenance: {notebook}")

    text = " ".join(parts)
    # Kaggle truncates very long descriptions in some views; keep it readable.
    return text[:1900]


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_submission_ref():
    """The id Kaggle just assigned, if the API will tell us."""
    try:
        out = subprocess.run(ep.kaggle_cmd() + ["competitions", "submissions",
                                                COMPETITION, "--csv"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return None
        rows = list(csv.DictReader(io.StringIO(out.stdout)))
        return (rows[0].get("ref") or "").strip() or None if rows else None
    except Exception:                                              # noqa: BLE001
        return None


def measure_all(exclude=()):
    """Run the validation episode for every agent and store what it measured.

    The registry carried a latency figure for 2 of 32 models, which meant the
    one hard limit on the submission contract was invisible for everything
    else. This fills the column in without spending a submission slot: it is
    the same self-play episode `validate()` runs before an upload.
    """
    hr("Measuring every agent (no upload)")
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "agents", "*.py")))
             if os.path.basename(p) not in set(exclude)]
    print(f"  {len(paths)} agent(s); each plays one 720-turn self-play episode\n")
    import kaggriculture.data.registry as registry
    done, failed = 0, []
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        try:
            ok, findings, metrics = validate(path)
        except Exception as exc:                                   # noqa: BLE001
            failed.append((name, str(exc)[:80]))
            print(f"  [{i:>2}/{len(paths)}] {name:<44} ERROR {str(exc)[:40]}")
            continue
        if "worst_ms" not in metrics:
            failed.append((name, "no latency measured"))
            print(f"  [{i:>2}/{len(paths)}] {name:<44} "
                  f"{[f for f in findings if f.startswith('FAIL')][:1]}")
            continue
        registry.register_model(
            name, latency_ms=metrics.get("worst_ms"),
            mean_ms=metrics.get("mean_ms"), worst_ms=metrics.get("worst_ms"),
            selfplay_banks=metrics.get("selfplay_banks"),
            validated=_now_iso(),
            contract_ok=bool(ok))
        done += 1
        flag = "" if metrics["worst_ms"] <= LATENCY_LIMIT_MS else "  OVER LIMIT"
        print(f"  [{i:>2}/{len(paths)}] {name:<44} mean {metrics['mean_ms']:>6.1f} ms  "
              f"worst {metrics['worst_ms']:>7.1f} ms{flag}")
    print(f"\n  {done} measured, {len(failed)} failed")
    for name, why in failed:
        print(f"    - {name}: {why}")
    return 0


# --------------------------------------------------------------------- main
def _cli_is_broken(proc):
    """True when the failure is the CLI wrapper, not the submission itself."""
    text = ((proc.stdout or "") + (proc.stderr or "")).lower()
    return any(k in text for k in ("unexpected keyword argument", "typeerror",
                                   "traceback (most recent call last)"))


def _submit_via_api(main_py, message):
    """Upload through the in-process Kaggle API, passing only what it accepts."""
    api = ep.kaggle_api()
    if api is None:
        return False, ("could not authenticate the Kaggle API in-process either. "
                       "Check %USERPROFILE%\\.kaggle\\kaggle.json exists.")
    for name in ("competition_submit", "competition_submit_cli"):
        fn = getattr(api, name, None)
        if not fn:
            continue
        try:
            out = ep._call_api(fn, main_py, message, COMPETITION,
                               file_name=main_py, message=message,
                               competition=COMPETITION, quiet=False)
            return True, f"{name} accepted the upload: {str(out)[:160]}"
        except TypeError:
            try:
                out = fn(main_py, message, COMPETITION)
                return True, f"{name} accepted the upload: {str(out)[:160]}"
            except Exception as exc:                               # noqa: BLE001
                return False, f"{name} failed: {type(exc).__name__}: {exc}"
        except Exception as exc:                                   # noqa: BLE001
            return False, f"{name} failed: {type(exc).__name__}: {exc}"
    return False, "this kaggle build exposes no competition_submit method"


def main():
    # The ladder's engine, or nothing this prints means anything.
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None,
                    help="submit this file; default is to rank agents/ and pick the best")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except the upload")
    ap.add_argument("--exclude", nargs="*", default=["v0_baseline.py"])
    ap.add_argument("--notebook", default=None,
                    help="a kernel ref to cite in the description, e.g. "
                         "user/slug -- use with src/kaggriculture/agentbuild/build_notebook.py")
    ap.add_argument("--describe", action="store_true",
                    help="print the auto-generated description and exit")
    ap.add_argument("--allow-untested-fix", action="store_true",
                    help="operator override for the intraday-fix gate "
                         "(a y>0 build without a sign-tested verdict)")
    ap.add_argument("--measure-all", action="store_true",
                    help="validate every agent and store its latency, no upload")
    args = ap.parse_args()

    if args.describe:
        target = args.agent or "agents/v17_route.py"
        path = os.path.join(ROOT, target) if not os.path.isabs(target) else target
        print(describe(path, notebook=args.notebook))
        return 0

    if args.measure_all:
        return measure_all(args.exclude)

    hr("Kaggriculture -- submit the best agent")

    # ---- 1. choose ----------------------------------------------------
    if args.agent:
        best = os.path.abspath(os.path.join(ROOT, args.agent))
        if not os.path.exists(best):
            sys.exit(f"no such agent: {best}")
        print(f"  agent chosen explicitly: {os.path.relpath(best, ROOT)}")
    else:
        cands = [p for p in sorted(glob.glob(os.path.join(ROOT, "agents", "*.py")))
                 if os.path.basename(p) not in args.exclude]
        if not cands:
            sys.exit("no candidate agents in agents/")
        print(f"  ranking {len(cands)} candidates by WIN RATE "
              f"({args.seeds} seeds, both seats, round robin)")
        print("  (the ladder scores wins, not coin margin -- see the Evaluation "
              "section of the competition overview)\n")
        seeds = [args.seed0 + i for i in range(args.seeds)]
        t0 = time.time()
        rows = rank_agents(cands, seeds, args.workers)
        print(f"  {'agent':<34}{'win%':>8}{'mean bank':>13}{'games':>8}")
        print("  " + "-" * 63)
        for r in rows:
            print(f"  {os.path.basename(r['agent']):<34}{r['win_pct']:>7.0f}%"
                  f"{r['mean_bank']:>13,.0f}{r['n']:>8}")
        print(f"\n  ranked in {time.time() - t0:.0f}s")
        best = rows[0]["agent"]
        print(f"  best: {os.path.basename(best)}")

    # ---- 1b. intraday-fix gate (2026-08-16) ----------------------------
    # A v{x}.{y>0} build retires a LIVE submission mid-climb (only the
    # latest 2 stay active), and ratings converge with games played --
    # v25.1 shipped untested and cut down v25.0_bandit at 4 hours. A fix
    # must present a fresh sign-tested PASS from src/intraday_gate.py.
    m = re.search(r"v(\d+)\.(\d+)_", os.path.basename(best))
    if m and int(m.group(2)) > 0 and not args.allow_untested_fix:
        vpath = os.path.join(ROOT, "models", "intraday_gate",
                             os.path.basename(best) + ".json")
        why = None
        try:
            vd = json.load(open(vpath, encoding="utf-8"))
            if not vd.get("passed"):
                why = "verdict is FAIL (no sign-tested edge)"
            elif time.time() - os.path.getmtime(vpath) > 24 * 3600:
                why = "verdict is older than 24h"
            elif os.path.getmtime(best) > os.path.getmtime(vpath):
                why = "agent was rebuilt AFTER the verdict (stale)"
        except (OSError, ValueError):
            why = "no verdict found"
        if why:
            print(f"\n  intraday-fix gate: {why}")
            print(f"  run: python src/intraday_gate.py "
                  f"{os.path.relpath(best, ROOT)}")
            print("\n  REFUSING TO SUBMIT -- an untested fix does not retire "
                  "a climbing submission.")
            return 1
        print("  intraday-fix gate: PASS (fresh sign-tested edge)")

    # ---- 2. validate --------------------------------------------------
    hr("Validation (mirrors Kaggle's upload checks)")
    ok, findings, metrics = validate(best)
    for f in findings:
        print("  " + f)
    if not ok:
        print("\n  REFUSING TO SUBMIT -- fix the failures above first.")
        return 1

    # ---- 3. package ---------------------------------------------------
    build = os.path.join(ROOT, "build")
    os.makedirs(build, exist_ok=True)
    main_py = os.path.join(build, "main.py")
    shutil.copy(best, main_py)
    print(f"\n  packaged -> build/main.py ({os.path.getsize(main_py):,} bytes)")

    # ---- 4. quota / active-pair warning -------------------------------
    hr("Before you confirm")
    n_today, latest = submission_status()
    if n_today is None:
        print("  ! could not read your submission history (is the kaggle CLI "
              "installed and authenticated?)")
    else:
        print(f"  submissions today: {n_today}/5")
        if n_today >= 5:
            print("  ! daily limit reached -- Kaggle will reject this until UTC midnight")
        if latest:
            print("  currently active (latest 2 count for the final leaderboard):")
            for line in latest:
                print(f"    - {line}")
    print("\n  NOTE: only your latest 2 submissions stay active, and the final")
    print("  leaderboard is computed from them. Submitting a weaker agent can")
    print("  push a stronger one out of that pair.")

    msg = args.message or describe(best, metrics, notebook=args.notebook)
    print("\n  description Kaggle will show:")
    for line in _wrap(msg, 74):
        print(f"    {line}")
    # Resolve the CLI the same way every other tool does. This line used to
    # hand the literal string "kaggle" to subprocess, which fails on any machine
    # where pip put kaggle.exe somewhere PATH does not look -- reported as
    # "the kaggle CLI is not installed" on a machine where it plainly was.
    try:
        cmd = ep.kaggle_cmd() + ["competitions", "submit", COMPETITION,
                                 "-f", main_py, "-m", msg]
    except ep.KaggleError as exc:
        print(f"\n  ! {exc}")
        return 1
    print(f"\n  command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n  --dry-run: stopping here. build/main.py is ready.")
        # Latency and the self-play banks are properties of the agent, not of
        # an upload, and a dry run has just measured both. Store them -- this
        # is how every model gets a latency figure without spending a slot.
        try:
            import kaggriculture.data.registry as registry
            registry.register_model(
                os.path.basename(best),
                latency_ms=metrics.get("worst_ms"),
                mean_ms=metrics.get("mean_ms"),
                worst_ms=metrics.get("worst_ms"),
                selfplay_banks=metrics.get("selfplay_banks"),
                validated=_now_iso())
            print(f"  latency recorded in the registry: mean "
                  f"{metrics.get('mean_ms')} ms, worst {metrics.get('worst_ms')} ms")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  (registry not updated: {exc})")
        return 0

    confirm = ask("\n  Type SUBMIT to upload this agent to Kaggle: ")
    if confirm != "SUBMIT":
        print("  not submitted. build/main.py is ready if you want to send it later.")
        return 0

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        print("  ! the Kaggle CLI vanished between resolving and running it")
        return 1
    print(p.stdout or p.stderr)

    if p.returncode != 0 and _cli_is_broken(p):
        # kaggle 1.8.3 ships CLI wrappers that pass arguments its own API does
        # not accept. The upload endpoint is fine; only the wrapper is wrong, so
        # call the API directly rather than making the user downgrade.
        print("\n  the CLI wrapper failed with a version bug; using the API "
              "directly ...")
        ok, detail = _submit_via_api(main_py, msg)
        print("  " + detail)
        if ok:
            print("\n  submitted. Watch it at:\n"
                  "  https://www.kaggle.com/competitions/kaggriculture/submissions")
            return 0
        return 1

    if p.returncode != 0:
        print("  ! submission failed. Common causes: you have not clicked "
              "'Join Competition' in a browser, or the daily limit of 5 is used up.")
        return 1
    _record(best, main_py, metrics, msg, "cli")
    print("  submitted. Track it with:")
    print("    kaggle competitions submissions kaggriculture")
    print("    kaggle competitions leaderboard kaggriculture -s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
