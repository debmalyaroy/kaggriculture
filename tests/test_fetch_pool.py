"""Contract tests for the adaptive fetch controller.

A width controller that misbehaves is expensive in both directions: too narrow
wastes an hour of the release path, too wide triggers the 429 storm that once
left a cycle with nothing built. And it runs unattended, so the properties have
to be asserted rather than eyeballed in a log.

    python tests/test_fetch_pool.py
"""
import os
import sys
import threading

import kaggriculture.data.fetch_pool as FP  # noqa: E402


def test_404_is_not_throttling():
    """A pruned archive file is permanent, not a rate limit. Reading it as
    throttling would narrow the pool over data that does not exist."""
    assert not FP.looks_throttled("404 Client Error: Not Found")
    assert not FP.looks_throttled(Exception("404"))
    assert FP.looks_throttled("429 Too Many Requests")
    assert FP.looks_throttled("ConnectionAborted 10053")
    assert FP.looks_throttled("503 Service Unavailable")
    print("throttle detection: 429/503/aborted yes, 404 no")


def test_widens_on_clean_run():
    p = FP.AdaptivePool(start=4, ceiling=16, verbose=False)
    start = p.width
    for _ in range(FP.CLEAN_TO_WIDEN):
        p.report(ok=True)
    assert p.width > start, (start, p.width)
    print(f"clean run widened {start} -> {p.width}")


def test_narrows_on_throttle_burst():
    p = FP.AdaptivePool(start=16, ceiling=32, verbose=False)
    for _ in range(16):
        p.acquire()                     # simulate all slots in flight
        p.release()
    start = p.width
    for _ in range(10):
        p.report(ok=False, error="429 Too Many Requests")
    assert p.width < start, (start, p.width)
    print(f"throttle burst narrowed {start} -> {p.width}")


def test_never_below_floor_or_above_ceiling():
    p = FP.AdaptivePool(start=4, ceiling=6, floor=2, verbose=False)
    for _ in range(200):
        p.report(ok=True)
    assert p.width <= 6, p.width
    for _ in range(200):
        p.report(ok=False, error="429")
    assert p.width >= 2, p.width
    print(f"width stayed within [2, 6]; ended at {p.width}")


def test_404s_do_not_narrow():
    """The failure that must NOT shrink the pool."""
    p = FP.AdaptivePool(start=12, ceiling=24, verbose=False)
    start = p.width
    for _ in range(30):
        p.report(ok=False, error="404 Client Error: Not Found")
    assert p.width == start, (start, p.width)
    print(f"30 consecutive 404s left width at {p.width} (correct)")


def test_narrow_does_not_deadlock_with_work_in_flight():
    """Reclaiming slots must never block behind in-flight requests."""
    p = FP.AdaptivePool(start=8, ceiling=8, floor=2, verbose=False)
    for _ in range(8):
        p.acquire()                     # every slot held, none released
    done = threading.Event()

    def hammer():
        for _ in range(10):
            p.report(ok=False, error="429")
        done.set()

    t = threading.Thread(target=hammer, daemon=True)
    t.start()
    assert done.wait(timeout=5), "narrow() deadlocked while work was in flight"
    print("narrow with all slots held: no deadlock")


def test_narrow_takes_effect_under_full_load():
    """The bug: narrowing while every permit is held used to reclaim zero.

    Throttling happens precisely when the pool is saturated, so a narrow that
    only works when slots are free is a narrow that never works when it matters.
    The shortfall is carried as debt and swallowed by subsequent releases.
    """
    p = FP.AdaptivePool(start=8, ceiling=8, floor=2, verbose=False)
    for _ in range(8):
        p.acquire()                     # fully saturated, nothing free
    before = p.width
    for _ in range(10):
        p.report(ok=False, error="429 Too Many Requests")
    assert p.width < before, (before, p.width, "narrow had no effect")
    assert p.stats()["debt"] > 0, "shortfall was not recorded as debt"
    # As workers finish, the debt must actually consume their permits.
    for _ in range(8):
        p.release()
    got = 0
    while p._sem.acquire(blocking=False):
        got += 1
    assert got <= p.width, (got, p.width, "pool handed back more than its width")
    print(f"saturated narrow {before} -> {p.width}; permits after drain "
          f"{got} <= width {p.width}")


def test_stats_are_reported():
    p = FP.AdaptivePool(start=4, ceiling=8, verbose=False)
    for _ in range(5):
        p.report(ok=True)
    p.report(ok=False, error="429")
    s = p.stats()
    assert s["completed"] == 6 and s["throttles"] == 1, s
    print(f"stats: {s}")


if __name__ == "__main__":
    for fn in (test_404_is_not_throttling, test_widens_on_clean_run,
               test_narrow_takes_effect_under_full_load,
               test_narrows_on_throttle_burst,
               test_never_below_floor_or_above_ceiling,
               test_404s_do_not_narrow,
               test_narrow_does_not_deadlock_with_work_in_flight,
               test_stats_are_reported):
        fn()
    print("\nall fetch-pool checks passed")
