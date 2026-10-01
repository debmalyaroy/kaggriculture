"""The harvest window must read SDK timestamps as UTC.

The bug this pins down: the Kaggle SDK returns NAIVE datetimes in UTC, and
`datetime.timestamp()` interprets naive values as LOCAL time. On an IST box
(UTC+5:30) every episode therefore looked 5.5 h older than it was, and the
hourly harvest's 3 h window matched NOTHING -- every scheduled sip from
2026-08-12 to 2026-08-14 returned 0 ids while exiting 0. The daily release
then had to fetch a whole day's delta itself.

The failure mode is what makes this worth a suite: a timezone offset in a
filter produces a silent, plausible-looking zero, not an error.

    python tests/test_harvest_epoch.py
"""
import datetime as dt
import os
import sys
import time

import kaggriculture.data.leaderboard_harvest as LH  # noqa: E402


def test_naive_datetime_is_utc():
    """An episode that ended 30 min ago (UTC wall-clock, naive) must land
    inside a 1 h window regardless of the box's timezone."""
    ended = dt.datetime.utcfromtimestamp(time.time() - 1800)     # naive UTC
    got = LH._epoch(ended)
    age = time.time() - got
    assert 1700 < age < 1900, (
        f"naive UTC datetime aged {age / 3600:.2f}h -- a timezone is leaking "
        f"into the window filter (IST shift would read ~5.8h)")
    print(f"naive UTC datetime read as {age / 60:.1f} min old (correct)")


def test_aware_datetime_passthrough():
    ended = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
    age = time.time() - LH._epoch(ended)
    assert 500 < age < 700, age
    print(f"aware datetime read as {age / 60:.1f} min old (correct)")


def test_iso_string_is_utc():
    ended = dt.datetime.utcfromtimestamp(time.time() - 3600)
    age = time.time() - LH._epoch(ended.isoformat())
    assert 3500 < age < 3700, age
    print(f"ISO string read as {age / 60:.1f} min old (correct)")


def test_none_and_garbage():
    assert LH._epoch(None) == 0.0
    assert LH._epoch("not a date") == 0.0
    print("None/garbage -> 0.0 (filtered out, never crashes)")


if __name__ == "__main__":
    for fn in (test_naive_datetime_is_utc, test_aware_datetime_passthrough,
               test_iso_string_is_utc, test_none_and_garbage):
        fn()
    print("\nall harvest-epoch checks passed")
