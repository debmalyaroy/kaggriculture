"""Intraday-fix gate: a v{x}.{y>0} agent cannot ship without a fresh
sign-tested PASS verdict from src/intraday_gate.py.

Fast suite -- no games. The refusals happen BEFORE self-play validation, so
each subprocess call returns in seconds.
"""
from kaggriculture.paths import ROOT
import json
import os
import subprocess
import sys

import kaggriculture.measure.intraday_gate as IG  # noqa: E402

FIX = os.path.join(ROOT, "agents", "v25.1_bandit.py")
VDIR = os.path.join(ROOT, "models", "intraday_gate")
VPATH = os.path.join(VDIR, os.path.basename(FIX) + ".json")


def _submit_dry(*extra):
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "submit.py"),
         "--agent", os.path.relpath(FIX, ROOT), "--dry-run", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    failures = []

    # parse_version
    for name, want in (("v27.1_bandit.py", (27, 1, "bandit")),
                       ("v27_route.py", (27, 0, "route")),
                       ("v27.0_route2.py", (27, 0, "route2")),
                       ("notaversion.py", None)):
        got = IG.parse_version(name)
        if got != want:
            failures.append(f"parse_version({name}) = {got}, want {want}")

    # default_baseline: the fix's baseline is the version it retires
    base = IG.default_baseline(FIX)
    if not (base and os.path.basename(base) == "v25.0_bandit.py"):
        failures.append(f"default_baseline(v25.1_bandit) = {base}")

    if not os.path.exists(FIX):
        failures.append(f"fixture agent missing: {FIX}")
    else:
        saved = None
        if os.path.exists(VPATH):
            saved = open(VPATH, encoding="utf-8").read()
        try:
            # no verdict -> refuse
            if os.path.exists(VPATH):
                os.remove(VPATH)
            code, out = _submit_dry()
            if code == 0 or "intraday-fix gate" not in out:
                failures.append("missing verdict did not refuse")

            # stale verdict (agent newer than verdict) -> refuse
            os.makedirs(VDIR, exist_ok=True)
            with open(VPATH, "w", encoding="utf-8") as fh:
                json.dump({"passed": True}, fh)
            old = os.path.getmtime(FIX) - 60
            os.utime(VPATH, (old, old))
            code, out = _submit_dry()
            if code == 0 or ("stale" not in out and "older than 24h" not in out):
                failures.append("stale verdict did not refuse")

            # failing verdict -> refuse
            with open(VPATH, "w", encoding="utf-8") as fh:
                json.dump({"passed": False}, fh)
            code, out = _submit_dry()
            if code == 0 or "FAIL" not in out:
                failures.append("failing verdict did not refuse")
        finally:
            if saved is not None:
                with open(VPATH, "w", encoding="utf-8") as fh:
                    fh.write(saved)
            elif os.path.exists(VPATH):
                os.remove(VPATH)

    if failures:
        print("FAIL test_intraday_gate:")
        for f in failures:
            print("  -", f)
        return 1
    print("ok  test_intraday_gate (7 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
