#!/usr/bin/env python
"""(1) DELTA BUILDER  --  local, NO gdrive push.

Fetches the day's new ladder + our-own games, folds them into the persistent
trackp delta corpus, and tars ONLY the shards that are new since the last run.
Prints the tar.gz path for YOU to upload to gdrive:kaggriculture/delta/.

    python scripts/trackp/delta_build.py                 # full daily delta
    python scripts/trackp/delta_build.py --no-fetch       # rebuild tar from existing shards
    python scripts/trackp/delta_build.py --limit 2000     # cap episodes (test)

Flow:  scheduled_fetch (incremental) -> trackp_corpus (append to delta dir,
manifest-resumable) -> diff shard set -> tar.gz only-new -> print path.

The delta dir persists at data/trackp_delta ; each run appends and we tar the
new shards only, so uploads stay small (a day of games is ~50-200 MB).
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA_DIR = os.path.join(ROOT, "data", "trackp_delta")
OUT_DIR = os.path.join(ROOT, ".local", "delta")
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")


def _run(cmd, optional=False):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT, env=ENV).returncode
    if rc != 0 and not optional:
        raise SystemExit(f"step failed (rc={rc}): {cmd}")
    if rc != 0:
        print(f"  (optional step rc={rc} -- continuing)", flush=True)
    return rc


def _shards(d):
    return set(os.path.basename(p) for p in glob.glob(os.path.join(d, "shard_*.parquet")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-fetch", action="store_true", help="skip the data fetch")
    ap.add_argument("--limit", type=int, default=0, help="cap #episodes (0=all new)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DELTA_DIR, exist_ok=True)

    # 1) pull the day's new data (incremental + timestamped; our own games too)
    if not a.no_fetch:
        _run([PY, "-m", "kaggriculture.data.scheduled_fetch"], optional=True)
        _run([PY, "-m", "kaggriculture.data.ourgames", "--all"], optional=True)

    before = _shards(DELTA_DIR)

    # 2) fold new episodes into the delta corpus (manifest-resumable: only new eps)
    cmd = [PY, "-m", "kaggriculture.data.trackp_corpus", "--out", DELTA_DIR,
           "--rows-per-shard", str(a.rows_per_shard), "--jobs", str(a.jobs)]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    _run(cmd)

    after = _shards(DELTA_DIR)
    new = sorted(after - before)
    if not new:
        print("\nNO NEW SHARDS -- nothing to upload (no fresh episodes since last run).")
        return

    # 3) tar ONLY the new shards + companions, so the upload is delta-sized
    epoch = time.strftime("%Y%m%d_%H%M%S")            # epoch = YYYYMMDD_HHMMSS
    tar_path = os.path.join(OUT_DIR, f"bc_corpus_delta_{epoch}.tar.gz")
    companions = [f for f in ("manifest.json", "vocab.json", "norm.json", "index.json")
                  if os.path.exists(os.path.join(DELTA_DIR, f))]
    with tarfile.open(tar_path, "w:gz") as tf:
        for f in new + companions:
            tf.add(os.path.join(DELTA_DIR, f), arcname=f)
    size_mb = os.path.getsize(tar_path) / 1e6

    print("\n" + "=" * 66)
    print(f"DELTA READY: {len(new)} new shards  ({size_mb:.1f} MB)")
    print(f"UPLOAD THIS -> gdrive:kaggriculture/delta/")
    print(f"PATH: {tar_path}")
    print("=" * 66)


if __name__ == "__main__":
    main()
