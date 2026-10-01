"""Re-fetch old episodes to fill ingest-time enrichments they predate.

The index's OUTCOMES are 100% complete, but enrichments only exist from the
day their capture code landed: obs-features (85%), per-turn traces (34% of
episodes), seeds (35%), team labels (50%). The replays were deleted at
ingest, so nothing local can fill the gaps -- but kaggleusercontent still
serves every episode by id, so the gaps are REFILLABLE, not lost.

For each episode with any missing enrichment: fetch the replay (adaptive
width, 429-aware), then capture whatever is absent -- trace (both seats,
with seed), obs-feature sidecars, and the index row's seed/team/trace/obs
fields -- and delete the replay. Delta by construction: fully-enriched
episodes are never fetched again, so re-running is cheap and idempotent.

    python src/backfill_enrich.py                # everything missing
    python src/backfill_enrich.py --limit 200 --jobs 6
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

STAGE = os.path.join(ROOT, "data", "sameday", "_enrich")
OBSFEAT = os.path.join(ROOT, "data", "obsfeat")


def needs(idx, trace_files):
    """{episode: [route ids]} for episodes missing ANY enrichment."""
    out = {}
    for rid, r in idx["routes"].items():
        ep = str(r.get("episode") or rid.rsplit("_s", 1)[0])
        if not ep.isdigit():
            continue                      # foreign/notebook artifacts
        missing = (ep not in trace_files
                   or r.get("seed") is None
                   or not r.get("obs")
                   or not r.get("team") or r.get("team") == "?")
        if missing:
            out.setdefault(ep, []).append(rid)
    return out


def enrich(path, ep, rids, idx):
    """Capture every absent enrichment from one fetched replay."""
    import kaggriculture.data.obs_features as obs_features
    import kaggriculture.data.turn_features as TF
    replay = json.load(open(path, encoding="utf-8"))
    if len(replay.get("steps") or []) < 100:
        return 0
    seed = (replay.get("info") or {}).get("seed")
    teams = list(((replay.get("info") or {}).get("TeamNames")) or [])
    if not os.path.exists(TF.path_for(ep)):
        TF.capture(replay, ep)
    done = 0
    for rid in rids:
        r = idx["routes"].get(rid)
        if r is None:
            continue
        seat = int(r.get("seat") or rid[-1])
        if not r.get("obs"):
            try:
                of = obs_features.flat(obs_features.obs_features(replay, seat))
                os.makedirs(OBSFEAT, exist_ok=True)
                json.dump(of, open(os.path.join(OBSFEAT, rid + ".json"), "w"),
                          separators=(",", ":"))
                r["obs"] = True
            except Exception:                                   # noqa: BLE001
                pass
        if r.get("seed") is None and seed is not None:
            r["seed"] = seed
        if (not r.get("team") or r.get("team") == "?") and seat < len(teams):
            r["team"] = teams[seat]
        r["trace"] = os.path.exists(TF.path_for(ep))
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import kaggriculture.data.fetch_pool as fetch_pool
    import kaggriculture.data.routes as R
    import kaggriculture.data.sameday as sameday
    import kaggriculture.data.turn_features as TF

    idx = R.load_index()
    trace_files = ({os.path.splitext(f)[0] for f in os.listdir(TF.TRACE_DIR)}
                   if os.path.isdir(TF.TRACE_DIR) else set())
    todo = needs(idx, trace_files)
    eps = sorted(todo, reverse=True)      # newest first: most model-relevant
    if args.limit:
        eps = eps[:args.limit]
    print(f"episodes missing enrichment: {len(todo):,}; fetching "
          f"{len(eps):,}", flush=True)
    if not eps:
        return 0

    os.makedirs(STAGE, exist_ok=True)
    pool = fetch_pool.AdaptivePool(start=args.jobs,
                                   ceiling=max(args.jobs, 16))

    def fetch(ep):
        dest = os.path.join(STAGE, ep)
        os.makedirs(dest, exist_ok=True)
        pool.acquire()
        try:
            sameday._grab_direct(ep, dest)
            j = next((os.path.join(dest, f) for f in os.listdir(dest)
                      if f.endswith(".json")), None)
            pool.report(ok=bool(j), error=None if j else "empty")
            return ep, j
        except Exception as exc:                                # noqa: BLE001
            pool.report(ok=False, error=exc)
            return ep, None
        finally:
            pool.release()

    t0 = time.time()
    ok = failed = rows = 0
    with ThreadPoolExecutor(max_workers=16) as tp:
        futs = [tp.submit(fetch, ep) for ep in eps]
        for i, fut in enumerate(as_completed(futs), 1):
            ep, path = fut.result()
            if not path:
                failed += 1
            else:
                try:
                    rows += enrich(path, ep, todo[ep], idx)
                    ok += 1
                except Exception as exc:                        # noqa: BLE001
                    failed += 1
                    if failed <= 8:
                        print(f"  ! {ep}: {type(exc).__name__}: "
                              f"{str(exc)[:70]}", flush=True)
                shutil.rmtree(os.path.join(STAGE, ep), ignore_errors=True)
            if i % 100 == 0:
                R.save_index(idx)
                dt_ = time.time() - t0
                print(f"  {i:,}/{len(eps):,} ({i / dt_:.1f}/s, "
                      f"eta {(len(eps) - i) / (i / dt_) / 60:.0f} min)",
                      flush=True)
    R.save_index(idx)
    shutil.rmtree(STAGE, ignore_errors=True)
    n_tr = len(os.listdir(TF.TRACE_DIR))
    print(f"\nenriched {ok:,} episodes / {rows:,} route rows "
          f"({failed} failed) in {(time.time() - t0) / 60:.1f} min; "
          f"trace corpus now {n_tr:,} episodes")
    pool.log_stats("enrich fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
