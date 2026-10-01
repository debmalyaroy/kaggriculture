"""Cached per-route feature vectors, invalidated on feature-code change.

`train_identifier.build_dataset()` re-derived features for all 11,961 routes on
every retrain, and loaded each route from disk TWICE -- once for the clustering
signature, once for the prefix vectors. Measured per route: `load_route`
13.4 ms (x2), `route_features` 2.8 ms, `prefix_features` 5.5 ms across the six
checkpoints. That is ~6.5 min of an 8.1 min retrain, spent re-deriving vectors
for routes that have not changed, when only ~900 of 11,961 are new each day.

Routes are immutable once mined, so this is pure waste. Cached, the same work
is ~0.5 min and stays flat as the corpus grows.

STALENESS IS THE REAL RISK HERE, NOT SPEED. A cache that survived an edit to
`features.py` would silently train tomorrow's identifier on yesterday's feature
semantics -- the same silent-drift class as reading `farms[me]["coins"]` when
the engine key is `money`, where nothing raises and the number is merely wrong.
So the cache carries a hash of the feature module's source, and any mismatch
discards the WHOLE cache rather than mixing two generations of vectors.

    import kaggriculture.data.feature_cache as FC
    route_vecs, prefix_vecs = FC.build(ids, checkpoints)
"""
from kaggriculture.paths import ROOT
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache", "features")
CACHE = os.path.join(CACHE_DIR, "route_features.npz")


def feature_hash():
    """Identity of the feature CODE, not the data.

    Covers features.py and routes.py: the first defines the vectors, the second
    defines how a route is loaded and so what those vectors are computed over.
    """
    h = hashlib.sha256()
    for mod in (features, R):
        path = getattr(mod, "__file__", None)
        if not path or not os.path.exists(path):
            return "unknown"
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def load(checkpoints):
    """(route dict, prefix dict, hash) from cache, or empty dicts if stale."""
    want = feature_hash()
    if not os.path.exists(CACHE):
        return {}, {}, want
    try:
        z = np.load(CACHE, allow_pickle=False)
        if str(z["feature_hash"]) != want:
            print(f"feature cache STALE (code changed) -- discarding "
                  f"{len(z['ids']):,} entries and rebuilding")
            return {}, {}, want
        if list(z["checkpoints"]) != list(checkpoints):
            print("feature cache checkpoints differ -- discarding")
            return {}, {}, want
        # Materialise each array ONCE. An NpzFile decompresses on every
        # __getitem__, so `z["prefix"][n]` inside a comprehension re-inflated
        # the whole 78 MB array on all 11,912 iterations -- which MemoryError'd
        # and fell back to a silent full rebuild, i.e. the cache looked like it
        # worked while doing nothing.
        ids = [str(v) for v in z["ids"]]
        route_arr = z["route"]
        prefix_arr = z["prefix"]
        route = {i: route_arr[n] for n, i in enumerate(ids)}
        prefix = {i: prefix_arr[n] for n, i in enumerate(ids)}
        return route, prefix, want
    except Exception as exc:                                    # noqa: BLE001
        print(f"feature cache unreadable ({type(exc).__name__}) -- rebuilding")
        return {}, {}, want


def save(route, prefix, checkpoints, want):
    if not route:
        return
    ids = sorted(route)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE + ".tmp"
    # Write-then-rename: a cache truncated by an interrupt is worse than none,
    # because the next run would train on a partial corpus without noticing.
    # Pass an open HANDLE, not a path -- np.savez_compressed appends ".npz" to
    # any path not already ending in it, so a "...npz.tmp" path silently became
    # "...npz.tmp.npz" and the rename then found nothing.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, ids=np.array(ids),
            route=np.array([route[i] for i in ids], dtype=np.float32),
            prefix=np.array([prefix[i] for i in ids], dtype=np.float32),
            checkpoints=np.array(list(checkpoints)),
            feature_hash=np.array(want))
    os.replace(tmp, CACHE)


def build(ids, checkpoints, verbose=True):
    """Feature vectors for `ids`, computing only what the cache lacks.

    Returns (route_vecs, prefix_vecs) as dicts keyed by route id; prefix_vecs
    values are (len(checkpoints), d) arrays. Ids whose route fails to load are
    omitted from both, exactly as the uncached path skipped them.
    """
    route, prefix, want = load(checkpoints)
    missing = [i for i in ids if i not in route or i not in prefix]
    if verbose:
        print(f"feature cache: {len(ids) - len(missing):,} hit, "
              f"{len(missing):,} to build")
    built = 0
    for rid in missing:
        try:
            acts = R.load_route(rid)                 # ONE load, not two
            rv, _ = features.route_features(acts)
            pv = [features.prefix_features(acts, t) for t in checkpoints]
        except Exception:                                       # noqa: BLE001
            continue
        route[rid] = np.asarray(rv, dtype=np.float32)
        prefix[rid] = np.asarray(pv, dtype=np.float32)
        built += 1
        if verbose and built % 500 == 0:
            print(f"  built {built:,}/{len(missing):,}", flush=True)
    if built:
        save(route, prefix, checkpoints, want)
    return route, prefix


def main():
    """Warm the cache for every indexed route."""
    idx = R.load_index()
    ids = list(idx["routes"])
    import kaggriculture.train.train_identifier as TI
    route, prefix = build(ids, TI.PREFIX_TURNS)
    print(f"cache holds {len(route):,} routes; "
          f"{os.path.getsize(CACHE) / 1e6:.1f} MB at {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
