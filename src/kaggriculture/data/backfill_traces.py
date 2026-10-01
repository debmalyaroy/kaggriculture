"""Historical backfill of per-turn traces from every replay still on disk.

Capture was forward-only by the original decision, which left the corpus at 22
bootstrap episodes while ~2,900 full replays (~22 GB) sat un-mined in
data/sameday/_stage. This walks every replay directory, captures the 49-field
trace for both seats of every episode that does not already have one, and
leaves the replays where they are -- it reads, it never deletes.

Measured on this box: 0.45 s to parse a 32 MB replay + 0.08 s to extract both
seats, so the full backfill is ~26 CPU-minutes; at 6 workers it is ~5 minutes.
Workers are PROCESSES because json.load holds the GIL, and each worker holds at
most one parsed replay (~300 MB peak), so 6 workers fit comfortably in 16 GB.
This is CPU+disk only -- no network -- so it cannot contend with the hourly
scrape's rate limits; it only shares the disk.

    python src/backfill_traces.py                 # everything not yet captured
    python src/backfill_traces.py --jobs 4 --limit 100
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

REPLAY_GLOBS = (
    os.path.join(ROOT, "data", "sameday", "_stage", "*", "*.json"),
    os.path.join(ROOT, "data", "sameday", "_stage", "*.json"),
    os.path.join(ROOT, ".local", "lossreplays", "*.json"),
    os.path.join(ROOT, ".local", "lossreplays", "*", "*.json"),
    os.path.join(ROOT, "data", "episodes", "*", "*.json"),
)


def find_replays():
    """{episode_id: path}, newest file wins on duplicates."""
    out = {}
    for pat in REPLAY_GLOBS:
        for f in glob.glob(pat):
            ep = os.path.splitext(os.path.basename(f))[0]
            if not ep.isdigit():        # manifests, indexes, id lists
                continue
            out[ep] = f
    return out


def _one(job):
    """Worker: parse one replay, capture both seats. Returns (ep, bytes|err)."""
    ep, path = job
    import kaggriculture.data.turn_features as TF
    try:
        replay = json.load(open(path, encoding="utf-8"))
        steps = replay.get("steps") or []
        if len(steps) < 100:            # truncated download, not an episode
            return ep, "short:%d" % len(steps)
        return ep, TF.capture(replay, ep)
    except Exception as e:                                      # noqa: BLE001
        return ep, "err:%s" % type(e).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true",
                    help="re-capture even if a trace file exists")
    args = ap.parse_args()

    import kaggriculture.data.turn_features as TF
    have = {os.path.splitext(f)[0]
            for f in os.listdir(TF.TRACE_DIR)} if os.path.isdir(
                TF.TRACE_DIR) else set()
    replays = find_replays()
    todo = sorted((ep, p) for ep, p in replays.items()
                  if args.force or ep not in have)
    if args.limit:
        todo = todo[:args.limit]
    print(f"replays on disk: {len(replays):,}; already captured: "
          f"{len(have):,}; to backfill: {len(todo):,}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    from multiprocessing import Pool
    t0 = time.time()
    ok = bad = total_kb = 0
    with Pool(args.jobs) as pool:
        for i, (ep, res) in enumerate(pool.imap_unordered(_one, todo,
                                                          chunksize=4), 1):
            if isinstance(res, int) and res > 0:
                ok += 1
                total_kb += res / 1024
            else:
                bad += 1
                if bad <= 10:
                    print(f"  skip {ep}: {res}", flush=True)
            if i % 250 == 0 or i == len(todo):
                dt = time.time() - t0
                print(f"  {i:,}/{len(todo):,} "
                      f"({i / dt:.1f}/s, eta {(len(todo) - i) / (i / dt):.0f}s)",
                      flush=True)
    dt = time.time() - t0
    print(f"\ncaptured {ok:,} episodes ({total_kb / 1024:.0f} MB of traces), "
          f"{bad} skipped, in {dt / 60:.1f} min")
    n_now = len(glob.glob(os.path.join(TF.TRACE_DIR, "*.npz")))
    print(f"trace corpus: {n_now:,} episodes at {TF.TRACE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
