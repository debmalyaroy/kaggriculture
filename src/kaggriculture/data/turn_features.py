"""Per-turn observation traces -- the data the pipeline currently destroys.

Every ingest parses a ~21-31 MB replay, keeps the 719-turn ACTION schedule, and
deletes the observations. Routes therefore carry what a farm DID but nothing
about the state it did it in, so there is no (state, action) corpus for offline
learning. Reconstruction is not available either: replaying the recorded
actions under the recorded seed diverges 25% (tested 2026-08-13), and the seed
was not even being stored. So the traces have to be captured here, while the
replay is still on disk.

WIDTH. An earlier note said "275-dim per turn". That number is the *prefix*
feature width -- a cumulative summary computed at six checkpoints -- and it is
the wrong object for per-turn capture: recomputing a cumulative summary 720
times is both expensive and redundant. What a policy conditions on at turn t is
the STATE at turn t, which is ~50 numbers. Measured footprint at that width:
~141 KB per seat, ~282 KB per episode, ~254 MB/day at current scrape rates.

INFORMATION BOUNDARY. `farms` is public for both seats, so the opponent's
money, tiles, animals and hand count are all fair game. `private` (shed, seeds,
per-hand inventories) is per-seat, so only OUR side of it is recorded -- reading
the opponent's private block would be a train/serve skew bug, the same class
that was caught and reverted in the 2026-08-11 identifier audit.

    import kaggriculture.data.turn_features as TF
    arr = TF.trace(replay, seat)      # (T, TF.WIDTH) float32
"""
from kaggriculture.paths import ROOT
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

TRACE_DIR = os.path.join(ROOT, "data", "turntrace")

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
            "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
# Base prices from the engine's marketParams, used only to normalise; a price
# is meaningful relative to its own base, not on an absolute scale.
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
        "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
EQUILIBRIUM = 10000.0
SHED_CAP = 100.0

# Field layout, in order. Kept explicit so a change is visible in a diff and
# so `feature_names()` can never drift from what `turn_vector()` emits.
_LAYOUT = (
    ("time", 3),            # day/30, hour/24, step/720
    ("price", 9),           # per product, / its base price
    ("mkt_inv", 9),         # per product, / equilibrium
    ("us", 7),              # money, hands, quads, crops, animals, dry, starving
    ("them", 7),            # same, from the public opponent farm
    ("shed", 9),            # OUR shed only (private is per-seat)
    ("seeds", 5),           # OUR seeds only
)
WIDTH = sum(n for _, n in _LAYOUT)          # 49


def feature_names():
    out = []
    for group, n in _LAYOUT:
        if group in ("price", "mkt_inv", "shed"):
            out += [f"{group}_{p}" for p in PRODUCTS]
        elif group == "seeds":
            out += [f"seeds_{c}" for c in CROPS]
        elif group == "time":
            out += ["day", "hour", "step"]
        else:
            out += [f"{group}_{k}" for k in
                    ("money", "hands", "quads", "crops", "animals",
                     "dry", "starving")]
    assert len(out) == WIDTH, (len(out), WIDTH)
    return out


def _farm_stats(farm):
    """The seven public numbers, from the 10x10 NESTED tile grid.

    `tiles` is rows-of-cells and the two cell kinds carry disjoint keys, so a
    flat walk silently reports zero crops and zero animals for every farm --
    the trap that produced an all-zeros forensics table on 2026-08-13.
    """
    crops = animals = dry = starving = 0
    for row in (farm.get("tiles") or []):
        for t in (row if isinstance(row, list) else [row]):
            if not isinstance(t, dict):
                continue
            if t.get("kind") == "PLANT" and t.get("crop"):
                crops += 1
                if (t.get("consecutive_unwatered") or 0) >= 1:
                    dry += 1
            elif t.get("kind") == "PASTURE" and t.get("animal"):
                animals += 1
                if (t.get("consecutive_unfed") or 0) >= 1:
                    starving += 1
    return [float(farm.get("money") or 0) / 1e5,
            len(farm.get("hands") or []) / 20.0,
            len(farm.get("unlocked_quadrants") or []) / 4.0,
            crops / 100.0, animals / 100.0, dry / 100.0, starving / 100.0]


def turn_vector(obs_pub, obs_seat, seat):
    """One turn's state vector, or None if the step carries no farm state.

    `obs_pub` supplies the shared market/town and BOTH farms; `obs_seat` is the
    observation handed to `seat`, which is the only place that seat's private
    block is valid.
    """
    farms = obs_pub.get("farms") or []
    if len(farms) < 2:
        return None
    market = obs_pub.get("market") or {}
    prices = market.get("prices") or {}
    inv = market.get("inventory") or {}
    priv = (obs_seat or {}).get("private") or {}
    shed = priv.get("shed") or {}
    seeds = priv.get("seeds") or {}
    v = [
        float(obs_pub.get("day") or 0) / 30.0,
        float(obs_pub.get("hour") or 0) / 24.0,
        float(obs_pub.get("step") or 0) / 720.0,
    ]
    v += [float(prices.get(p, 0) or 0) / BASE[p] for p in PRODUCTS]
    v += [float(inv.get(p, 0) or 0) / EQUILIBRIUM for p in PRODUCTS]
    v += _farm_stats(farms[seat])
    v += _farm_stats(farms[1 - seat])
    v += [float(shed.get(p, 0) or 0) / SHED_CAP for p in PRODUCTS]
    v += [float(seeds.get(c, 0) or 0) / 20.0 for c in CROPS]
    return v


def trace(replay, seat):
    """(T, WIDTH) float32 trace for `seat`, or None if unusable."""
    steps = replay.get("steps") or []
    rows = []
    for st in steps:
        if not st:
            continue
        obs_pub = (st[0].get("observation") or {})
        obs_seat = ((st[seat].get("observation") or {})
                    if seat < len(st) else {})
        v = turn_vector(obs_pub, obs_seat, seat)
        if v is not None:
            rows.append(v)
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def path_for(episode):
    return os.path.join(TRACE_DIR, f"{episode}.npz")


def save(episode, seat_traces, seed=None, meta=None):
    """One file per EPISODE holding both seats, written atomically.

    Both seats together because they share the market path; a partial file
    would be worse than none, since training would silently see a truncated
    episode -- hence write-then-rename.
    """
    if not seat_traces:
        return None
    os.makedirs(TRACE_DIR, exist_ok=True)
    dest = path_for(episode)
    tmp = dest + ".tmp"
    payload = {f"s{s}": a for s, a in seat_traces.items() if a is not None}
    if not payload:
        return None
    payload["width"] = np.array(WIDTH)
    payload["seed"] = np.array(-1 if seed is None else int(seed))
    if meta:
        for k, val in meta.items():
            payload[f"meta_{k}"] = np.array(str(val))
    # np.savez_compressed appends ".npz" to any path not already ending in it,
    # so pass a HANDLE -- a "....npz.tmp" path silently became ".npz.tmp.npz"
    # and the rename then found nothing (hit while building feature_cache.py).
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, dest)
    return dest


def load(episode):
    """{seat: (T, WIDTH) array} plus 'seed', or None."""
    p = path_for(episode)
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p, allow_pickle=False)
        # Materialise each array ONCE: an NpzFile decompresses on every
        # __getitem__, so touching a key inside a loop re-inflates it.
        out = {}
        for k in z.files:
            if k.startswith("s") and k[1:].isdigit():
                out[int(k[1:])] = z[k]
        out["seed"] = int(z["seed"]) if "seed" in z.files else -1
        return out
    except Exception:                                           # noqa: BLE001
        return None


def capture(replay, episode, seats=(0, 1)):
    """Extract and persist both seats' traces for one replay. Returns bytes."""
    seed = ((replay.get("info") or {}).get("seed"))
    traces = {s: trace(replay, s) for s in seats}
    dest = save(episode, traces, seed=seed)
    try:
        return os.path.getsize(dest) if dest else 0
    except OSError:
        return 0


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--episode")
    args = ap.parse_args()
    replay = json.load(open(args.replay, encoding="utf-8"))
    ep = args.episode or str((replay.get("info") or {}).get("EpisodeId")
                             or os.path.splitext(
                                 os.path.basename(args.replay))[0])
    n = capture(replay, ep)
    got = load(ep)
    print(f"width {WIDTH} ({len(feature_names())} named)")
    for s in (0, 1):
        if got and s in got:
            print(f"  seat {s}: {got[s].shape} float32")
    print(f"stored seed: {got.get('seed') if got else None}")
    print(f"{n / 1024:.0f} KB at {path_for(ep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
