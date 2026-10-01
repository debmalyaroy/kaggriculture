"""Replay-quota guard + release freshness: the morning-killer class, pinned.

Fast suite -- no network. Covers:
  * rolling 24h budget holds non-release downloads at the cap
  * the pre-release quiet window holds non-release downloads
  * KAGG_RELEASE=1 bypasses both
  * a QUOTA-HOLD is not retried through the CLI transport (prefix contract)
  * release freshness prefers the hourly heartbeat over index mtime
"""
from kaggriculture.paths import ROOT
import json
import os
import sys
import time



def main():
    failures = []
    saved_env = os.environ.pop("KAGG_RELEASE", None)
    import kaggriculture.data.sameday as sameday

    led = sameday._QUOTA_LEDGER + ".test"
    os.makedirs(os.path.dirname(led), exist_ok=True)
    saved_ledger, sameday._QUOTA_LEDGER = sameday._QUOTA_LEDGER, led
    saved_quiet = sameday._QUIET_HOURS
    try:
        # budget cap
        sameday._QUIET_HOURS = (99, 99)          # disable the window branch
        with open(led, "w", encoding="utf-8") as fh:
            json.dump({"events": [time.time()] * sameday._QUOTA_CAP}, fh)
        try:
            sameday._quota_allow()
            failures.append("cap not enforced")
        except RuntimeError as e:
            if not str(e).startswith("QUOTA-HOLD"):
                failures.append(f"cap raised wrong prefix: {e}")

        # stale events roll off: a day-old ledger must not hold
        with open(led, "w", encoding="utf-8") as fh:
            json.dump({"events": [time.time() - 25 * 3600]
                       * sameday._QUOTA_CAP}, fh)
        try:
            sameday._quota_allow()
        except RuntimeError as e:
            failures.append(f"stale events still held: {e}")

        # quiet window
        hour = time.localtime().tm_hour
        sameday._QUIET_HOURS = (hour, hour + 1)
        try:
            sameday._quota_allow()
            failures.append("quiet window not enforced")
        except RuntimeError as e:
            if "quiet window" not in str(e):
                failures.append(f"window raised wrong reason: {e}")

        # release bypass (window active AND ledger full)
        os.environ["KAGG_RELEASE"] = "1"
        with open(led, "w", encoding="utf-8") as fh:
            json.dump({"events": [time.time()] * sameday._QUOTA_CAP}, fh)
        try:
            sameday._quota_allow()
        except RuntimeError as e:
            failures.append(f"release bypass failed: {e}")
        os.environ.pop("KAGG_RELEASE", None)

        # ATTEMPTS never consume budget (the 2026-08-16 evening bug: an
        # afternoon of 429'd attempts burned the cap with zero data) --
        # only _quota_record() does.
        sameday._QUIET_HOURS = (99, 99)
        with open(led, "w", encoding="utf-8") as fh:
            json.dump({"events": [time.time()] * 5}, fh)
        sameday._quota_allow()
        sameday._quota_allow()
        n = len(json.load(open(led, encoding="utf-8"))["events"])
        if n != 5:
            failures.append(f"_quota_allow consumed budget ({n} != 5)")
        sameday._quota_record()
        n = len(json.load(open(led, encoding="utf-8"))["events"])
        if n != 6:
            failures.append(f"_quota_record did not record ({n} != 6)")
    finally:
        sameday._QUOTA_LEDGER = saved_ledger
        sameday._QUIET_HOURS = saved_quiet
        if os.path.exists(led):
            os.remove(led)
        if saved_env is not None:
            os.environ["KAGG_RELEASE"] = saved_env

    # 429-storm breaker (2026-08-18): a trip marker makes the next runs
    # probe instead of firing 2,000 attempts into a server-side throttle;
    # clearing it restores full width.
    cdp = sameday._COOLDOWN + ".test"
    saved_cd, sameday._COOLDOWN = sameday._COOLDOWN, cdp
    try:
        if sameday._cooldown_active() is not None:
            failures.append("cooldown active with no marker")
        sameday._cooldown_trip("test storm")
        if sameday._cooldown_active() is None:
            failures.append("fresh trip marker not detected")
        with open(cdp, "w", encoding="utf-8") as fh:
            json.dump({"tripped": time.time() - 4000}, fh)
        if sameday._cooldown_active() is not None:
            failures.append("stale trip marker (>55min) still cooling")
        sameday._cooldown_clear()
        if os.path.exists(cdp):
            failures.append("_cooldown_clear left the marker")
    finally:
        sameday._COOLDOWN = saved_cd
        if os.path.exists(cdp):
            os.remove(cdp)

    # the QUOTA-HOLD prefix is load-bearing at three CLI-fallback sites
    for rel, needle in (("src/kaggriculture/data/sameday.py", 'startswith("QUOTA-HOLD")'),
                        ("src/kaggriculture/data/ourgames.py", 'startswith("QUOTA-HOLD")'),
                        ("src/kaggriculture/data/routes.py", 'startswith("QUOTA-HOLD")')):
        txt = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if needle not in txt:
            failures.append(f"{rel} lost the QUOTA-HOLD CLI-skip")

    # freshness: heartbeat wins over a stale index
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "daily_release", os.path.join(ROOT, "scripts", "daily_release.py"))
    dr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dr)
    hb = dr.HEARTBEAT
    saved_hb = None
    if os.path.exists(hb):
        saved_hb = os.path.getmtime(hb)
    try:
        os.makedirs(os.path.dirname(hb), exist_ok=True)
        with open(hb, "w", encoding="utf-8") as fh:
            fh.write("test\n")
        age = dr._freshness_age_hours()
        if age is None or age > 0.1:
            failures.append(f"fresh heartbeat not honored (age={age})")
    finally:
        if saved_hb is not None:
            os.utime(hb, (saved_hb, saved_hb))
        else:
            os.remove(hb)

    if failures:
        print("FAIL test_quota_guard:")
        for f in failures:
            print("  -", f)
        return 1
    print("ok  test_quota_guard (7 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
