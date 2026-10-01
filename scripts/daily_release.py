"""Daily release: fetch everything current, retrain every model on it, crown a
fresh base, and upload BOTH the bandit and the route from two separate PRIVATE
notebooks -- maintaining each notebook's version history.

Order of operations (each step trains/builds on the output of the last, so the
shipped agent is always trained on the latest data):

  1. FETCH   own game plays (ourgames --all) + same-day scrape + leaderboard
             top-200 harvest (fresh_data), delta-only.
  2. CYCLE   refresh_cycle.py: retrains identifier + gates + surrogate, mines
             fresh routes, referees a tournament, crowns a base, and builds the
             v{N} route + bandit. It never submits.
  3. GATE    submit.py --dry-run on each built agent (size, self-play Validation
             Episode, latency). A failure here aborts the upload.
  4. NOTEBOOKS  build + push two PRIVATE kernels, one per agent, same stable
             slugs every day so Kaggle accumulates version history.
  5. UPLOAD  submit both agents to the competition, each citing its notebook.

Safeguard: if the cycle HELD (no fresh crown) or either gate fails, nothing is
uploaded and the current live pair is left in place. The last-shipped pair is
recorded in data/logs/last_release.json so a regression can be reverted.

    python scripts/daily_release.py                 # full daily release
    python scripts/daily_release.py --no-submit     # everything but the upload
    python scripts/daily_release.py --fetch-only     # just refresh the data
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

SRC = os.path.join(ROOT, "src")
LOGS = os.path.join(ROOT, "data", "logs")
# KAGG_RELEASE marks every child of the release as release-context: the
# replay-quota guard in sameday._quota_allow() waves those through (and only
# those) during the pre-release quiet window and past the rolling 24h budget.
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
           KAGG_RELEASE="1")

# Two stable private slugs -- pushing to the same slug each day is what keeps
# the version history. One notebook per agent, as required.
BANDIT_SLUG = "kaggriculture-adaptive-bandit-private"
BANDIT_TITLE = "Kaggriculture | Adaptive Bandit (private)"
ROUTE_SLUG = "kaggriculture-route-refresh-private"
ROUTE_TITLE = "Kaggriculture | Route Refresh (private)"


def run(cmd, timeout=None, capture=False):
    """Run a child; a TIMEOUT is a FAILED CHILD (code 124), never a raise.

    Same lesson as refresh_cycle.run() (2026-08-16): an uncaught
    TimeoutExpired from ONE slow child aborts the whole unattended release.
    Callers that must abort on failure already check the returned code.
    """
    print("$ " + " ".join(cmd), flush=True)
    try:
        if capture:
            p = subprocess.run(cmd, cwd=ROOT, env=ENV, timeout=timeout,
                               capture_output=True, text=True)
            print((p.stdout or "")[-2000:], flush=True)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        return (subprocess.run(cmd, cwd=ROOT, env=ENV,
                               timeout=timeout).returncode, "")
    except subprocess.TimeoutExpired as exc:
        print(f"child TIMED OUT after {timeout}s (treated as failed, the "
              f"release continues): {' '.join(cmd[:3])}", flush=True)
        out = ""
        if capture:
            out = ((exc.stdout or b"").decode("utf-8", "replace")
                   if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        return 124, out


def active_refs():
    """The refs of our currently active submissions, to harvest their games."""
    try:
        import kaggriculture.pipeline.refresh_cycle as RC
        return RC.stage_detect()
    except Exception:                                              # noqa: BLE001
        return []


FETCH_FRESH_HOURS = 2.0
HEARTBEAT = os.path.join(ROOT, "data", "sameday", "heartbeat.txt")


def _freshness_age_hours():
    """Hours since the hourly track last COMPLETED, or since the index moved.

    The old test keyed on index.json's mtime alone, which lied in both
    directions: an hourly run that completed but found nothing new leaves the
    mtime old (2026-08-16: the 06:44 rerun saw "2.0h stale" and burned 75+
    minutes re-fetching a corpus that was actually current), and the hourly
    task deliberately skips while a release runs, ageing the index further.
    The hourly script now writes a heartbeat at the end of EVERY successful
    run -- including empty ones -- and that is the primary freshness signal;
    the index mtime remains as the fallback for a missing heartbeat.
    """
    import time
    ages = []
    for p in (HEARTBEAT,
              os.path.join(ROOT, "data", "routes", "index.json")):
        try:
            ages.append((time.time() - os.path.getmtime(p)) / 3600.0)
        except OSError:
            pass
    return min(ages) if ages else None


def stage_fetch(refs):
    """Bring the corpus current -- but only if the hourly track has not.

    The hourly KaggricultureSameDay task already does every part of this:
    leaderboard harvest, same-day replay ingest, `ourgames.py --all`, and the
    02:xx archive mine. Running it again here duplicated 60-120 minutes onto
    the release's critical path.

    A blind `--skip-fetch` would be wrong in the other direction: if the hourly
    task has been failing, the release would silently crown on a stale index.
    So the skip is conditional on measured freshness and says which way it
    went, rather than being a flag someone has to remember to pass.
    """
    age = _freshness_age_hours()
    if age is not None and age <= FETCH_FRESH_HOURS:
        print(f"\n===== 1. FETCH -- SKIPPED (hourly track fresh, "
              f"{age:.1f}h old) =====")
        return
    why = "no freshness signal" if age is None else f"{age:.1f}h stale"
    print(f"\n===== 1. FETCH ({why}; ourgames + same-day + leaderboard) =====")
    # TIME-BUDGETED (2026-08-16): this stage sits on the release's critical
    # path, and the 06:44 rerun spent 75+ minutes here on a throttled quota.
    # The release needs FRESHNESS (today's loss tapes, today's routes), not
    # COMPLETENESS -- completeness is the hourly track's job. Loss-tape
    # material first, then a capped harvest; a timeout is a failed child now,
    # never a dead release.
    run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "data", "ourgames.py"), "--all"],
        timeout=1500)
    sub_arg = ",".join(refs) if refs else ""
    cmd = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "data", "fresh_data.py")]
    if sub_arg:
        cmd += ["--subs", sub_arg]
    cmd += ["--lb-gb", "4"]
    run(cmd, timeout=3600)


def newest_pair():
    """The freshest built (route, second) agent files under the v{x}.{y}
    scheme (x = daily release, y = intra-day fix; bare v{x} == v{x}.0).

    SECOND-SLOT RULE (2026-08-16): the cycle decides who occupies slot 2 --
    the bandit only when it beat the route sign-tested, otherwise a diversity
    route (v{x}.{y}_route2.py). Its decision is written to
    models/second_slot.json; that marker WINS over any filename glob when its
    version matches the newest route, so a bandit file that lost its seat is
    never published just for existing.
    """
    def newest(glob_pat):
        best, bestv = None, (-1, -1)
        for p in glob.glob(os.path.join(ROOT, "agents", glob_pat)):
            m = re.search(r"v(\d+)(?:\.(\d+))?_", os.path.basename(p))
            if not m:
                continue
            v = (int(m.group(1)), int(m.group(2) or 0))
            if v > bestv:
                best, bestv = p, v
        return best, bestv
    # SEAT ARCHITECTURE (operator law, 2026-09-02, reaffirmed 2026-09-03):
    # the live pair is ALWAYS {bandit, trackp} -- a tape+adaptive-chassis seat
    # and a closed-loop seat. The v*_route.py family has not been a seat since
    # v34 (2026-08-27). Globbing it here made this function return
    # route=v34.0_route.py (nine releases stale) as half the pair, so a
    # publish chain run would have shipped a dead agent. Prefer trackp; fall
    # back to route only if no trackp build exists at all.
    route, rv = newest("v*_trackp.py")
    if route is None:
        route, rv = newest("v*_route.py")
    second, sv = newest("v*_bandit.py")
    marker = os.path.join(ROOT, "models", "second_slot.json")
    try:
        with open(marker, encoding="utf-8") as fh:
            m = json.load(fh)
        mv = tuple(int(x) for x in str(m.get("version", "")).split("."))
        mv = (mv + (0,))[:2]
        chosen = m.get("second")
        if chosen:
            chosen = os.path.join(ROOT, chosen) if not os.path.isabs(chosen) \
                else chosen
        if mv == rv and chosen and os.path.exists(chosen):
            second, sv = chosen, mv
        elif mv == rv and not chosen:
            second, sv = None, (-1, -1)   # cycle said: route-only release
    except (OSError, ValueError, TypeError):
        pass                              # no/old marker: glob behaviour
    return route, second, rv, sv


def gate(agent_path):
    code, out = run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "submit.py"),
                     "--agent", os.path.relpath(agent_path, ROOT), "--dry-run"],
                    timeout=1800, capture=True)
    ok = "stopping here" in out and "REFUSING TO SUBMIT" not in out
    return ok


def build_and_push_notebook(agent_path, slug, title):
    """Build + push the private kernel; returns (ok, version_number)."""
    import kaggriculture.agentbuild.build_notebook as B
    out_dir = os.path.join(ROOT, "notebooks", slug)
    B.build(agent_path, out_dir=out_dir, slug=slug, title=title, private=True)
    p = subprocess.run(["kaggle", "kernels", "push", "-p", out_dir],
                       capture_output=True, text=True, timeout=900, env=ENV)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[-400:], flush=True)
    m = re.search(r"Kernel version (\d+)", out)
    return ("successfully pushed" in out, int(m.group(1)) if m else None)


def already_live(agent_path):
    """True if this exact agent is already one of our active submissions.

    Only the latest 2 submissions stay active and a fresh upload starts at a
    provisional rating that takes ~24h to develop. So re-uploading an unchanged
    agent is not free: it evicts a developed rating and replaces it with a
    provisional copy of the same code. On a day the crown gate HOLDs, the pair
    is unchanged, and this is exactly what would happen to the live route.

    Submission descriptions carry "(sha xxxxxxxx)". The prefix LENGTH has
    varied across releases (8 hex in the v23 pair, 12 in newer messages), so
    compare prefix-wise rather than for an exact token -- an exact match on 12
    silently returned False for the live v23_route, which is precisely the
    unchanged-agent case this guard exists to catch.
    """
    import hashlib
    import re as _re
    full = hashlib.sha256(open(agent_path, "rb").read()).hexdigest()
    try:
        p = subprocess.run(["kaggle", "competitions", "submissions",
                            "kaggriculture"], capture_output=True, text=True,
                           timeout=300, env=ENV)
        out = (p.stdout or "")
    except Exception:                                              # noqa: BLE001
        return False
    lines = [ln for ln in out.splitlines() if _re.match(r"^\d{6,}\s", ln)]
    for ln in lines[:2]:                       # the active pair only
        for tok in _re.findall(r"sha\s+([0-9a-f]{8,64})", ln):
            n = min(len(tok), len(full))
            if n >= 8 and tok[:n] == full[:n]:
                return True
    return False


def submit_agent(agent_path, notebook_slug, version):
    """Submit FROM the notebook version's main.py output (never a direct
    file upload): wait for the kernel run to COMPLETE, verify its main.py is
    byte-identical to the built agent, then submit -k <ref> -v <version>."""
    import hashlib
    import time as _t
    if already_live(agent_path):
        print(f"{os.path.basename(agent_path)} is already an ACTIVE submission "
              f"(same sha) -- not re-uploading; that would evict its developed "
              f"rating for a provisional copy of identical code")
        return "skipped-already-live"
    ref = f"debmalya84/{notebook_slug}"
    # `kernels status` reports COMPLETE for the PREVIOUS version's run while
    # the freshly-pushed one is still executing, and `kernels output` then
    # serves the stale main.py (bit the 2026-08-13 v24.1 ship: bandit v6
    # returned v5's output, sha check refused). The only trustworthy signal
    # is the output itself, so poll until main.py's sha matches the agent.
    want = hashlib.sha256(open(agent_path, "rb").read()).hexdigest()
    outdir = os.path.join(ROOT, ".local", "nbout", notebook_slug)
    got = None
    # POLL CADENCE (2026-08-13): was a flat 30 s, which adds ~15 s of pure
    # waiting to every check and made this the largest remaining stage once
    # compute was optimised. Kernel runs finish in a wide range, so poll
    # tightly at first and back off -- same ~30 min ceiling, far less idle
    # time when the run finishes early.
    delays = [10] * 12 + [20] * 12 + [30] * 40      # ~10 s .. ~30 min
    for i, delay in enumerate(delays):
        s = subprocess.run(["kaggle", "kernels", "status", ref],
                           capture_output=True, text=True, env=ENV).stdout
        if "ERROR" in s or "CANCEL" in s:
            print(f"kernel run failed for {ref}: {s.strip()[-120:]}")
            return False
        subprocess.run(["kaggle", "kernels", "output", ref, "-p", outdir],
                       capture_output=True, env=ENV, timeout=600)
        mp = os.path.join(outdir, "main.py")
        if os.path.exists(mp):
            got = hashlib.sha256(open(mp, "rb").read()).hexdigest()
            if got == want:
                break
        _t.sleep(delay)
    if got != want:
        print(f"{ref}: output main.py sha {(got or 'missing')[:12]} != agent "
              f"{want[:12]} after the wait -- not submitting")
        return False
    msg = (f"{os.path.basename(agent_path)} from notebook "
           f"{notebook_slug} v{version} (sha {want[:12]})")
    p = subprocess.run(["kaggle", "competitions", "submit", "kaggriculture",
                        "-k", ref, "-v", str(version), "-f", "main.py",
                        "-m", msg],
                       capture_output=True, text=True, timeout=600, env=ENV)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[-300:], flush=True)
    return p.returncode == 0


def keep_awake(enable):
    """Hold the machine awake for the duration of the run (Windows).

    A scheduled task can WAKE the machine (WakeToRun + wake timers), but a
    woken machine re-enters sleep/hibernate on its idle timeout -- which
    would kill a multi-hour pipeline partway. ES_SYSTEM_REQUIRED with
    ES_CONTINUOUS pins it awake until we release (or the process exits,
    which also releases it). Display may sleep; the system does not.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:                                              # noqa: BLE001
        pass


def release_held():
    """Operator convergence hold (docs/history/pair-improvement-plan.md, lever L1).

    `models/release_hold.json`: {"from": "YYYY-MM-DD", "until": "YYYY-MM-DD"}.
    Between the two dates (inclusive) the pipeline still fetches, trains,
    runs the tournament and crowns -- candidates stay warm -- but the publish
    chain (gate -> notebooks -> upload) is skipped so the LIVE pair ages
    undisturbed: ratings converge with games played, and v27 was retired at
    18h still climbing. Dated and self-expiring. Returns the reason or None.
    """
    p = os.path.join(ROOT, "models", "release_hold.json")
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p, encoding="utf-8"))
        d0 = dt.date.fromisoformat(str(m.get("from", "")))
        d1 = dt.date.fromisoformat(str(m.get("until", "")))
    except (OSError, ValueError) as exc:
        print(f"release_hold.json unreadable ({exc}) -- ignored")
        return None
    today = dt.date.today()
    if d0 <= today <= d1:
        return (f"operator convergence hold {d0} .. {d1} "
                f"(models/release_hold.json)")
    if today > d1:
        print(f"release hold expired ({d1}) -- normal publish")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-submit", action="store_true",
                    help="fetch, train, crown, gate, push notebooks -- but do "
                         "not upload to the competition")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="assume the daily fetch already ran; go straight to "
                         "the cycle")
    ap.add_argument("--resume-publish", action="store_true",
                    help="the cycle already built today's pair but the run "
                         "died before publishing (e.g. the 2026-08-15 outer "
                         "timeout): skip fetch+cycle AND the HELD guard and "
                         "go straight to gate -> notebooks -> upload")
    args = ap.parse_args()
    keep_awake(True)
    import atexit
    atexit.register(keep_awake, False)
    os.makedirs(LOGS, exist_ok=True)
    started = dt.datetime.now().isoformat(timespec="seconds")
    print(f"daily_release {started}")

    refs = active_refs()
    print(f"active submission refs: {refs}")

    if not args.skip_fetch and not args.resume_publish:
        stage_fetch(refs)
    if args.fetch_only:
        return 0

    print("\n===== 2. CYCLE (train all models on latest + crown) =====")
    # --no-fetch: stage 1 above just brought the index current; the cycle
    # re-fetching everything cost ~55 duplicated minutes per release
    # (measured 2026-08-12) and doubled the 429 exposure. The hourly
    # KaggricultureSameDay scrape keeps data fresh between releases too.
    #
    # --include-foreign: authorised by the operator 2026-08-12. It lets routes
    # decoded from other teams' PUBLIC notebooks (source == "notebook", e.g.
    # boatlee V16-RC2 at public 3094) compete in the crown tournament. If one
    # wins the gate this job will build the pair from it and UPLOAD it, so the
    # release can ship another team's published route under our name. That is
    # the intended behaviour here; remove the flag to go back to
    # mined-archive-only crowning.
    # Daily Rust-engine parity (CROWN-2): a few episodes through both engines,
    # bit-identical per step. On mismatch the pre-ranker's recall evidence is
    # revoked so the funnel gate closes; the release itself continues (every
    # decision runs on the official engine). Mirrors run_pipeline.py -- this
    # entrypoint is the one the 04:30 schedule actually uses.
    kagg = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
    if os.path.exists(kagg) and not args.resume_publish:
        code, _ = run([sys.executable,
                       os.path.join(ROOT, "tests", "test_rust_engine.py"),
                       "--episodes", "1", "--replays", "2"], timeout=1800)
        if code != 0:
            recall = os.path.join(ROOT, "models", "lab",
                                  "preranker_recall.json")
            if os.path.exists(recall):
                os.remove(recall)
            print("ALARM: Rust engine parity FAILED -- pre-ranker evidence "
                  "revoked; funnel stays closed", flush=True)

    if not args.resume_publish:
        # 18000s, raised from 14400: the 2026-08-15 run spent 76 min riding
        # out a replay-quota throttle in tape backoff and the old cap killed
        # it AFTER the pair was built but BEFORE publish.
        run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "refresh_cycle.py"),
             "--no-fetch", "--include-foreign"], timeout=18000)

    route, bandit, rv, bv = newest_pair()
    print(f"\nnewest built: route=v{rv} ({route}), bandit=v{bv} ({bandit})")

    # HELD guard: if the cycle produced no fresh crown today, its report says so
    # and the newest agents are unchanged from before. Do not re-upload stale.
    report_path = os.path.join(ROOT, "data", "refresh",
                               dt.date.today().isoformat(), "REPORT.md")
    held = False
    if os.path.exists(report_path) and not args.resume_publish:
        r = open(report_path, encoding="utf-8").read()
        held = "Holding." in r or "No loss tapes" in r or "Nothing new" in r
    if held:
        print("\nCYCLE HELD -- no fresh crown today. Leaving the live pair in "
              "place; nothing uploaded.")
        json.dump({"time": started, "status": "held"},
                  open(os.path.join(LOGS, "last_release.json"), "w"), indent=1)
        return 0

    hold = None if args.resume_publish else release_held()
    if hold:
        print(f"\nRELEASE HOLD -- {hold}. Fetch/train/tournament/crown ran; "
              "the live pair stays in place, nothing published.")
        json.dump({"time": started, "status": "hold", "reason": hold},
                  open(os.path.join(LOGS, "last_release.json"), "w"), indent=1)
        return 0

    print("\n===== 3. GATE (submit.py self-play validation) =====")
    if not (route and gate(route)):
        sys.exit("route failed the submission gate -- aborting upload")
    if bandit and not gate(bandit):
        sys.exit("bandit failed the submission gate -- aborting upload")

    print("\n===== 4. NOTEBOOKS (two private kernels, version history) =====")
    bv = rv_nb = None
    if bandit:
        _, bv = build_and_push_notebook(bandit, BANDIT_SLUG, BANDIT_TITLE)
    _, rv_nb = build_and_push_notebook(route, ROUTE_SLUG, ROUTE_TITLE)

    if args.no_submit:
        print("\n--no-submit: notebooks pushed, agents gated, NOT uploaded.")
        return 0

    print("\n===== 5. UPLOAD (from the notebook versions, never main.py) =====")
    # Quota floor: Kaggle's 5/day submission quota resets at 00:00 UTC
    # (05:30 IST). The schedule targets a ~06:00 IST push, so on a fast run
    # the pipeline can reach this stage BEFORE the reset -- and on a day whose
    # previous-UTC-day quota is exhausted, that submission is rejected
    # outright. Hold the upload until 00:05 UTC, never longer than 3h (a
    # pathological clock or a mid-day manual run must not sleep the pipeline).
    import datetime as _dt
    import time as _t
    now = _dt.datetime.now(_dt.timezone.utc)
    nxt = now.replace(hour=0, minute=5, second=0, microsecond=0)
    if nxt <= now:
        nxt += _dt.timedelta(days=1)          # late in the UTC day: the reset
    if (nxt - now) <= _dt.timedelta(hours=3):  # ahead is tonight's midnight
        wait = (nxt - now).total_seconds()
        print(f"quota floor: {now:%H:%M} UTC, next reset {nxt:%H:%M} UTC -- "
              f"holding upload {wait / 60:.0f} min")
        _t.sleep(wait)
    results = {}
    if bandit and bv:
        results["bandit"] = submit_agent(bandit, BANDIT_SLUG, bv)
    if rv_nb:
        results["route"] = submit_agent(route, ROUTE_SLUG, rv_nb)

    json.dump({"time": started, "status": "released",
               "route": os.path.basename(route),
               "bandit": os.path.basename(bandit) if bandit else None,
               "results": results},
              open(os.path.join(LOGS, "last_release.json"), "w"), indent=1)
    print(f"\ndaily_release done: {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
