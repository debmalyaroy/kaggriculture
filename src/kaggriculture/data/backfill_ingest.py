"""Ingest the staged-replay backlog into the route index, in parallel.

data/sameday/_stage accumulated ~2,900 full replays (~22 GB) whose episodes
never made it into the route index -- interrupted scrapes leave the file but
not the index entry, and the next run skips re-downloading what is already on
disk. Un-indexed, those episodes contribute NOTHING: no routes for the crown
pool, no bank/outcome for the policy corpus join, no obs-features, and their
traces cannot be labelled. This drains the backlog through the SAME ingest
the hourly scrape uses (sameday._ingest), so every enrichment lands: route
files, index rows with outcomes, obs-feature sidecars, per-turn traces.

Parallel by SHARD: each worker ingests its slice against its own in-memory
index copy and saves at the end -- safe because routes.save_index merges
under a lock rather than overwriting. Replays are left on disk (they also
feed the Rust differential harness); delete them separately if disk matters.

    python src/backfill_ingest.py --jobs 6
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _shard(args_):
    paths, shard_id = args_
    import kaggriculture.data.routes as R
    import kaggriculture.data.sameday as sameday
    idx = R.load_index()
    added = failed = 0
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        try:
            added += sameday._ingest(p, idx)
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            if failed <= 5:
                print(f"  [w{shard_id}] ! {os.path.basename(p)}: "
                      f"{type(e).__name__}: {str(e)[:70]}", flush=True)
        if i % 100 == 0:
            R.save_index(idx)
            print(f"  [w{shard_id}] {i}/{len(paths)} "
                  f"({i / (time.time() - t0):.1f}/s)", flush=True)
    R.save_index(idx)
    return added, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import kaggriculture.data.routes as R
    idx = R.load_index()
    todo = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "sameday", "_stage",
                                           "*", "*.json"))):
        ep = os.path.splitext(os.path.basename(p))[0]
        if ep.isdigit() and f"{ep}_s0" not in idx["routes"]:
            todo.append(p)
    if args.limit:
        todo = todo[:args.limit]
    print(f"staged replays not in the index: {len(todo):,}", flush=True)
    if not todo:
        return 0

    shards = [(todo[k::args.jobs], k) for k in range(args.jobs)]
    from multiprocessing import Pool
    t0 = time.time()
    added = failed = 0
    with Pool(args.jobs) as pool:
        for a, f in pool.imap_unordered(_shard, shards):
            added += a
            failed += f
    idx = R.load_index()
    print(f"\ningested {added:,} route(s) ({failed} replay(s) failed) in "
          f"{(time.time() - t0) / 60:.1f} min; index now "
          f"{len(idx['routes']):,} routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
