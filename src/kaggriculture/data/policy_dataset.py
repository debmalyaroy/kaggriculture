"""Assemble the offline (state, action, return) corpus from turn traces.

`turn_features` captures the 49-field state per turn; the route index holds the
719-turn ACTION schedule; the game record holds the outcome. This joins them
into the arrays an offline policy needs, which is the first thing that was
impossible before traces existed.

Two deliberate choices, both about not lying to a learner:

* **Target is SCORE, not margin.** A policy trained to maximise dollars
  optimises a currency the ladder does not pay -- $1 and $10,000 are the same
  result, and 27.6% of our 1,748 real games are decided under $3,000
  (src/win_metric.py).
* **Actions are the SELL SCHEDULE, not raw ops.** The blend policy's job is to
  emit a bounded schedule delta, not to generate farming from scratch: the route
  is the coordinate system every adaptive layer is written in. So the action per
  turn is the per-product sell vector, which is what a delta acts on.

    python src/policy_dataset.py --out models/lab/policy_dataset.npz
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.turn_features as TF  # noqa: E402
import kaggriculture.measure.win_metric as WM  # noqa: E402

PRODUCTS = TF.PRODUCTS
OUT = os.path.join(ROOT, "models", "lab", "policy_dataset.npz")


def sell_vector(turn_action):
    """Per-product units sold this turn, from the recorded market orders."""
    v = [0.0] * len(PRODUCTS)
    for o in (turn_action.get("market") or []):
        if (isinstance(o, (list, tuple)) and len(o) >= 3
                and o[0] == "SELL" and o[1] in PRODUCTS):
            try:
                v[PRODUCTS.index(o[1])] += float(o[2])
            except (TypeError, ValueError):
                continue
    return v


REPLAY_DIRS = (os.path.join(ROOT, ".local", "lossreplays"),
               os.path.join(ROOT, "data", "sameday", "_stage"))


def _actions_for(rid, ep, seat):
    """The 719-turn action stream for one seat.

    Route index first, because that is where the forward path puts every
    ingested episode. But OUR OWN games are tracked in data/ourgames and their
    routes are not in the route index, so fall back to extracting from a replay
    still on disk -- otherwise a bootstrap corpus from local replays yields
    nothing, which is exactly what happened on the first run.
    """
    import kaggriculture.data.routes as R
    try:
        return R.load_route(rid)
    except Exception:                                           # noqa: BLE001
        pass
    import kaggriculture.measure.opponents as opponents
    for d in REPLAY_DIRS:
        for cand in (os.path.join(d, f"{ep}.json"),
                     os.path.join(d, str(ep), f"{ep}.json")):
            if os.path.exists(cand):
                try:
                    return opponents.extract_actions(cand, seat)
                except Exception:                               # noqa: BLE001
                    continue
    return None


def build(max_episodes=None, verbose=True):
    import kaggriculture.data.routes as R
    ridx = R.load_index()
    try:
        gidx = json.load(open(os.path.join(ROOT, "data", "ourgames",
                                           "index.json"), encoding="utf-8"))
        games = {str(g["episode"]): g for g in gidx["games"].values()}
    except (OSError, ValueError, KeyError):
        games = {}

    X, A, Y, EP = [], [], [], []
    files = sorted(glob.glob(os.path.join(TF.TRACE_DIR, "*.npz")))
    if max_episodes:
        files = files[:max_episodes]
    used = skipped = 0
    for f in files:
        ep = os.path.splitext(os.path.basename(f))[0]
        tr = TF.load(ep)
        if not tr:
            skipped += 1
            continue
        for seat in (0, 1):
            arr = tr.get(seat)
            rid = f"{ep}_s{seat}"
            rec = ridx["routes"].get(rid) or {}
            if arr is None:
                skipped += 1
                continue
            acts = _actions_for(rid, ep, seat)
            if not acts:
                skipped += 1
                continue
            # Outcome for THIS seat, in score units.
            g = games.get(ep)
            if g is not None and int(g.get("seat") or 0) == seat:
                y = WM.score(g["bank"], g["opp_bank"])
            elif g is not None:
                # the record is for the OTHER seat; invert it
                y = WM.score(g["opp_bank"], g["bank"])
            elif rec.get("bank") is not None and rec.get("opp_bank") is not None:
                y = WM.score(rec["bank"], rec["opp_bank"])
            else:
                skipped += 1
                continue
            n = min(len(arr), len(acts))
            for t in range(n):
                X.append(arr[t])
                A.append(sell_vector(acts[t]))
                Y.append(y)
                EP.append(ep)
            used += 1
    if not X:
        return None
    out = {"X": np.asarray(X, dtype=np.float32),
           "A": np.asarray(A, dtype=np.float32),
           "Y": np.asarray(Y, dtype=np.float32),
           "episodes": np.asarray(EP)}
    if verbose:
        print(f"{used} trajectories from {len(files)} trace file(s) "
              f"({skipped} skipped)")
        print(f"X {out['X'].shape} (state)  A {out['A'].shape} (sell vector)  "
              f"Y {out['Y'].shape} (score)")
        uniq = sorted(set(EP))
        print(f"{len(uniq)} distinct episodes; score mean "
              f"{float(out['Y'].mean()):.3f}")
    return out


def save(data, path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, path)
    return path


def load(path=OUT):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    # Materialise once -- an NpzFile decompresses on every __getitem__.
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    data = build(max_episodes=args.max_episodes)
    if data is None:
        print("no traces yet -- capture accrues on every ingest "
              "(src/turn_features.py). Nothing to build.")
        return 0
    p = save(data, args.out)
    print(f"-> {p} ({os.path.getsize(p) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
