"""Serve-substrate gate: the tournament may only leave the official engine
on exact-bank evidence, and any mismatch closes the gate.

Fast suite -- no games (the real equivalence audit plays them; this pins the
gate logic and the CLI/RESULT contracts around it).
"""
from kaggriculture.paths import ROOT
import json
import os
import sys



def main():
    failures = []
    import kaggriculture.pipeline.refresh_cycle as RC

    saved = None
    if os.path.exists(RC.SERVE_EQUIV):
        saved = open(RC.SERVE_EQUIV, encoding="utf-8").read()
    try:  # the engine pin is generated data (models/ is created on demand); 1.32.6 = the ladder engine
        eng = json.load(open(os.path.join(ROOT, "models", "engine_version.json"), encoding="utf-8"))["engine"]
    except (OSError, ValueError, KeyError):
        eng = "1.32.6"

    def put(d):
        os.makedirs(os.path.dirname(RC.SERVE_EQUIV), exist_ok=True)
        with open(RC.SERVE_EQUIV, "w", encoding="utf-8") as fh:
            json.dump(d, fh)

    try:
        # green audit -> open
        put({"matches": 12, "mismatches": 0, "engine": eng,
             "serve_s_per_game": 0.6, "official_s_per_game": 10.0})
        ok, why = RC.serve_allowed()
        if not ok:
            failures.append(f"green audit refused: {why}")

        # one mismatch -> closed
        put({"matches": 50, "mismatches": 1, "engine": eng})
        ok, _ = RC.serve_allowed()
        if ok:
            failures.append("mismatch did not close the gate")

        # stale engine -> closed
        put({"matches": 50, "mismatches": 0, "engine": "0.0.0"})
        ok, _ = RC.serve_allowed()
        if ok:
            failures.append("wrong-engine audit did not close the gate")

        # thin evidence -> closed
        put({"matches": 5, "mismatches": 0, "engine": eng})
        ok, _ = RC.serve_allowed()
        if ok:
            failures.append("thin audit (<12) did not close the gate")
    finally:
        if saved is not None:
            with open(RC.SERVE_EQUIV, "w", encoding="utf-8") as fh:
                fh.write(saved)
        elif os.path.exists(RC.SERVE_EQUIV):
            os.remove(RC.SERVE_EQUIV)

    # contracts: the tournament swaps scripts, so serve_match must speak
    # evaluate's CLI and RESULT lines, and the kill-switch must exist.
    sm = open(os.path.join(ROOT, "src", "kaggriculture", "engine", "serve_match.py"),
              encoding="utf-8").read()
    for needle in ('"--vs"', '"--seed0"', '"--no-record"', "RESULT\\t",
                   "def eval_mode"):
        if needle not in sm:
            failures.append(f"serve_match missing: {needle}")
    rc = open(os.path.join(ROOT, "src", "kaggriculture", "pipeline", "refresh_cycle.py"),
              encoding="utf-8").read()
    for needle in ("def serve_allowed", "def serve_spot_check",
                   "serve_spot_check(picks", "src/kaggriculture/engine/serve_match.py"):
        if needle not in rc:
            failures.append(f"refresh_cycle missing: {needle}")

    if failures:
        print("FAIL test_serve_substrate:")
        for f in failures:
            print("  -", f)
        return 1
    print("ok  test_serve_substrate (8 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
