"""Second-slot rule: the cycle's marker decides who ships in slot 2.

Fast suite -- no games. Covers:
  * newest_pair honors models/second_slot.json when its version matches
  * marker with second=None means a route-only release
  * a stale-version marker falls back to the filename glob
  * the adaptive sell-timing layer is present and wired in the template
"""
from kaggriculture.paths import ROOT
import importlib.util
import json
import os
import sys

MARKER = os.path.join(ROOT, "models", "second_slot.json")


def _load_daily_release():
    spec = importlib.util.spec_from_file_location(
        "daily_release", os.path.join(ROOT, "scripts", "daily_release.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    failures = []
    dr = _load_daily_release()
    saved = None
    if os.path.exists(MARKER):
        saved = open(MARKER, encoding="utf-8").read()

    try:
        # Baseline: glob behaviour without a marker.
        if os.path.exists(MARKER):
            os.remove(MARKER)
        route, second, rv, sv = dr.newest_pair()
        # SEAT ARCHITECTURE (operator law 2026-09-02): the pair is
        # {bandit, trackp}. This assertion used to demand a *_route.py first
        # seat, which is why newest_pair kept returning a nine-releases-stale
        # v34.0_route.py as half the pair. trackp is the first seat now; route
        # remains the fallback only while no trackp build exists.
        if not (route and route.endswith(("_trackp.py", "_route.py"))):
            failures.append(f"glob first seat wrong: {route}")
        glob_second = second

        # 1. Marker with a matching version redirects slot 2.
        target = os.path.join(ROOT, "agents", "v25.0_route.py")
        with open(MARKER, "w", encoding="utf-8") as fh:
            json.dump({"version": f"{rv[0]}.{rv[1]}",
                       "second": os.path.relpath(target, ROOT)}, fh)
        _, second, _, sv = dr.newest_pair()
        if os.path.abspath(second or "") != os.path.abspath(target):
            failures.append(f"marker redirect ignored: {second}")

        # 2. Marker says route-only.
        with open(MARKER, "w", encoding="utf-8") as fh:
            json.dump({"version": f"{rv[0]}.{rv[1]}", "second": None}, fh)
        _, second, _, _ = dr.newest_pair()
        if second is not None:
            failures.append(f"route-only marker ignored: {second}")

        # 3. Stale marker version falls back to the glob.
        with open(MARKER, "w", encoding="utf-8") as fh:
            json.dump({"version": "1.0",
                       "second": os.path.relpath(target, ROOT)}, fh)
        _, second, _, _ = dr.newest_pair()
        if second != glob_second:
            failures.append(f"stale marker did not fall back: {second}")
    finally:
        if saved is not None:
            with open(MARKER, "w", encoding="utf-8") as fh:
                fh.write(saved)
        elif os.path.exists(MARKER):
            os.remove(MARKER)

    # 4. Adaptive sell-timing is RETIRED (2026-08-18): the layer stays in the
    # template (OFF) but the cycle must NOT flip it on anywhere. Five
    # measurements, zero wins -- re-adding the flip needs fresh evidence.
    tpl = open(os.path.join(ROOT, "src", "kaggriculture", "engine", "tape_runtime.py"),
               encoding="utf-8").read()
    if "_ADAPT_SELL = False" not in tpl:
        failures.append("template lost the (OFF) adaptive layer default")
    rc = open(os.path.join(ROOT, "src", "kaggriculture", "pipeline", "refresh_cycle.py"),
              encoding="utf-8").read()
    if '"_ADAPT_SELL = True"' in rc.replace("'", '"'):
        failures.append("adaptive flip re-added to the cycle without "
                        "sign-tested evidence (retired 2026-08-18)")

    # 5. refresh_cycle exposes the stage, the 3-tuple contract and the
    # operator force helper (dated, self-expiring).
    for needle in ("def stage_second", "bandit_passed", "second_slot.json",
                   "paired_test", "def second_slot_force",
                   "second_slot_force()", "factory::"):
        if needle not in rc:
            failures.append(f"refresh_cycle missing: {needle}")

    # 6. second_slot_force honors dates: valid today, expired yesterday.
    import datetime as _dt
    import kaggriculture.pipeline.refresh_cycle as RC
    fp = os.path.join(ROOT, "models", "second_slot_force.json")
    saved_force = open(fp, encoding="utf-8").read() if os.path.exists(fp) \
        else None
    try:
        today = _dt.date.today().isoformat()
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"force": "bandit", "until": today}, fh)
        if RC.second_slot_force() != "bandit":
            failures.append("valid force marker not honored")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"force": "bandit",
                       "until": (_dt.date.today()
                                 - _dt.timedelta(days=1)).isoformat()}, fh)
        if RC.second_slot_force() is not None:
            failures.append("EXPIRED force marker still honored")
        with open(fp, "w", encoding="utf-8") as fh:
            json.dump({"force": "everything", "until": today}, fh)
        if RC.second_slot_force() is not None:
            failures.append("unknown force kind honored")
    finally:
        if saved_force is not None:
            with open(fp, "w", encoding="utf-8") as fh:
                fh.write(saved_force)
        elif os.path.exists(fp):
            os.remove(fp)

    # 7. release_held honors its window: inside -> held, outside -> not.
    hp = os.path.join(ROOT, "models", "release_hold.json")
    saved_hold = open(hp, encoding="utf-8").read() if os.path.exists(hp) \
        else None
    try:
        today = _dt.date.today()
        with open(hp, "w", encoding="utf-8") as fh:
            json.dump({"from": today.isoformat(),
                       "until": today.isoformat()}, fh)
        if not dr.release_held():
            failures.append("in-window hold not honored")
        with open(hp, "w", encoding="utf-8") as fh:
            json.dump({"from": (today - _dt.timedelta(days=3)).isoformat(),
                       "until": (today - _dt.timedelta(days=2)).isoformat()},
                      fh)
        if dr.release_held() is not None:
            failures.append("EXPIRED hold still holding releases")
    finally:
        if saved_hold is not None:
            with open(hp, "w", encoding="utf-8") as fh:
                fh.write(saved_hold)
        elif os.path.exists(hp):
            os.remove(hp)

    if failures:
        print("FAIL test_second_slot:")
        for f in failures:
            print("  -", f)
        return 1
    print("ok  test_second_slot (7 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
