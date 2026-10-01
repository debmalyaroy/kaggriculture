"""P0.1 -- recapture every on-disk full replay into trace v2.

Sources (full replays only; derived summaries are skipped by size):
  data/sameday/_stage/<eid>/<eid>.json     the staged-ingest cache
  data/mine/episode-<eid>-replay.json      early mined episodes
  data/trackp/replays/<eid>.json           Track P's own fetches

Idempotent: an episode with an existing trace file is skipped, so this runs
safely as a pipeline stage. Parallel across processes (JSON parse bound).

Usage: python src/trackp/recapture.py [--jobs 6] [--limit N]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))

from kaggriculture.trackp import common, trace_v2  # noqa: E402


def _size_ok(f: str) -> bool:
    """The hourly ingest deletes staged replays CONCURRENTLY with any scan;
    a vanished file is a skip, never a crash."""
    try:
        return os.path.getsize(f) > 2_000_000
    except OSError:
        return False


def discover() -> dict:
    """episode id -> replay path, newest source winning."""
    out = {}
    for f in glob.glob(os.path.join(common.ROOT, "data", "mine",
                                    "episode-*-replay.json")):
        m = re.search(r"episode-(\d+)-replay", os.path.basename(f))
        if m and _size_ok(f):
            out[m.group(1)] = f
    for f in glob.glob(os.path.join(common.ROOT, "data", "sameday", "_stage",
                                    "*", "*.json")):
        base = os.path.splitext(os.path.basename(f))[0]
        if base.isdigit() and _size_ok(f):
            out[base] = f
    for f in glob.glob(os.path.join(common.REPLAYS, "*.json")):
        base = os.path.splitext(os.path.basename(f))[0]
        if base.isdigit() and _size_ok(f):
            out[base] = f
    return out


def _one(eid: str, path: str) -> tuple:
    try:
        rep = common.load_replay(path)
        X, meta = trace_v2.extract(rep)
        meta["uid"] = meta.get("episode")
        meta["episode"] = int(eid)
        trace_v2.save(trace_v2.trace_path(eid), X, meta)
        return eid, "ok", meta.get("engine")
    except Exception as e:  # noqa: BLE001 -- a bad file must not kill the run
        return eid, f"err {type(e).__name__}: {e}", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    todo = {e: p for e, p in discover().items()
            if not os.path.exists(trace_v2.trace_path(e))}
    items = sorted(todo.items())
    if args.limit:
        items = items[:args.limit]
    print(f"recapture: {len(items)} episodes to trace (v{trace_v2.VERSION}, "
          f"dim {trace_v2.DIM})", flush=True)
    ok = err = 0
    engines = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_one, e, p) for e, p in items]
        for i, fu in enumerate(as_completed(futs), 1):
            eid, status, eng = fu.result()
            if status == "ok":
                ok += 1
                engines[eng] = engines.get(eng, 0) + 1
            else:
                err += 1
                print(f"  {eid}: {status}", flush=True)
            if i % 250 == 0:
                print(f"  progress {i}/{len(items)} ok={ok} err={err}",
                      flush=True)
    print(f"recapture done: ok={ok} err={err} engines={engines}", flush=True)
    summary = {"ok": ok, "err": err, "engines": engines,
               "total_traces": len(glob.glob(
                   os.path.join(common.TRACES, "*.npz")))}
    with open(os.path.join(common.MODELS, "recapture_last.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
