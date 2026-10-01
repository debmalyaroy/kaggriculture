"""Fetch delta latest data. The mandatory first step of ANY model build.

Every model -- the 05:00 daily, a mid-day version, an on-demand rebuild --
must be trained on current data. This fetches only the DELTA (nothing already
in the index is re-downloaded) from three sources, cheapest/freshest first,
then retrains the identifier so the new data is in the label space:

  1. same-day  -- our active submissions' games from today; opponent seats
                  are foreign routes hours old (ahead of the archive lag)
  2. archive   -- the newest published daily dataset, incremental
  3. breadth   -- a small mid-ladder pass for identifier coverage

Idempotent and safe to call repeatedly: same-day and archive both skip ids
already indexed, and the index write is locked. Returns the count of new
routes so a caller can decide whether a rebuild is even warranted.

    python -m kaggriculture.data.fresh_data                 # auto-detect active subs
    python -m kaggriculture.data.fresh_data --subs 55397322,55396717
    python -m kaggriculture.data.fresh_data --quick         # same-day only (fast)
"""
from kaggriculture.paths import ROOT
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402

ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def _run(cmd, timeout=5400):
    """A child's TIMEOUT is a failed child, never a dead fetch (2026-08-16:
    an identifier retrain overran its budget mid cache-rebuild and the
    uncaught TimeoutExpired killed the whole fresh_data pass)."""
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run(cmd, env=ENV, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  child TIMED OUT after {timeout}s (continuing): "
              + " ".join(str(c) for c in cmd[1:3]), flush=True)


def active_subs():
    try:
        out = subprocess.run(["kaggle", "competitions", "submissions",
                              "kaggriculture"], capture_output=True, env=ENV,
                             timeout=120).stdout.decode("utf-8", "replace")
    except Exception:                                              # noqa: BLE001
        return []
    refs = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d{8})\s", line)
        if m and "COMPLETE" in line:
            refs.append(m.group(1))
        if len(refs) == 2:
            break
    return refs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subs", default=None,
                    help="comma-separated active refs; default auto-detect")
    ap.add_argument("--top", type=int, default=200,
                    help="leaderboard teams to harvest same-day episodes from")
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--min-score", type=float, default=2600.0)
    ap.add_argument("--lb-gb", type=float, default=12.0,
                    help="download budget for the leaderboard same-day pull")
    ap.add_argument("--quick", action="store_true",
                    help="same-day only (skip archive + breadth)")
    ap.add_argument("--no-retrain", action="store_true",
                    help="skip the identifier retrain")
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    before = len(R.load_index()["routes"])
    subs = (args.subs.split(",") if args.subs else active_subs())

    # Same-day LEADERBOARD harvest (top teams' current episodes via the
    # authenticated Kaggle API -- no browser). This is the freshest and
    # largest foreign-route source, ahead of the archive's ~1-day lag.
    ids_file = os.path.join(ROOT, "data", "sameday", "lb_ids.json")
    print(f"[fresh] leaderboard harvest (top {args.top}, {args.hours}h)")
    _run([sys.executable, "src/kaggriculture/data/leaderboard_harvest.py", "--top",
          str(args.top), "--hours", str(args.hours), "--min-score",
          str(args.min_score), "--out",
          os.path.relpath(ids_file, ROOT)], timeout=2400)
    if os.path.exists(ids_file):
        _run([sys.executable, "src/kaggriculture/data/sameday.py", "--ids-file", ids_file,
              "--max-gb", str(args.lb_gb), "--jobs", "6"], timeout=10800)

    if subs:
        print(f"[fresh] same-day delta from our own submissions {subs}")
        _run([sys.executable, "src/kaggriculture/data/sameday.py", "--harvest-own",
              ",".join(subs), "--max-gb", "3", "--jobs", "6"], timeout=3600)

    if not args.quick:
        # Newest archive days only (--days 2). The historical backfill
        # (--days 0, every published dataset) was moved OUT of the release
        # path 2026-08-12: old days feed the identifier, never crown
        # candidates (archive lags ~2 days), and the backfill cost the
        # morning release ~20-30 min. It now runs in the hourly
        # KaggricultureSameDay scrape's 02:xx off-peak slot instead.
        print("[fresh] archive delta (newest 2 days, incremental)")
        _run([sys.executable, "src/kaggriculture/data/routes.py", "--mine", "--top", "200",
              "--days", "2", "--max-gb", "8", "--jobs", "8"])
        print("[fresh] breadth delta (identifier coverage)")
        _run([sys.executable, "src/kaggriculture/data/routes.py", "--mine", "--breadth",
              "--max-gb", "3", "--jobs", "8"], timeout=3600)

    idx = R.load_index()
    R.assign_windows(idx, verbose=False)
    R.save_index(idx)
    added = len(idx["routes"]) - before
    print(f"[fresh] +{added} new routes (delta only; nothing re-downloaded)")

    if not args.no_retrain and added > 0:
        print("[fresh] retraining identifier on the enriched index")
        _run([sys.executable, "src/kaggriculture/train/train_identifier.py"], timeout=1800)
    print(f"[fresh] done. Index at {len(idx['routes'])} routes; models built "
          f"from here are trained on current data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
