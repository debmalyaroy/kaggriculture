"""Day-6 feature extraction for the economy-dispatcher key.

Operator order (2026-09-05): include the FULL dispatcher; fix the key. The
oracle prize is +17.4pp at 201-team scale (0.970 vs 0.796) and neither
market conditions nor opening family key it. This extracts everything an
agent can OBSERVE by step 145 (the existing branch socket) in each grid
cell's recorded world, so a classifier can learn obs(day<=6) -> best
economy:

  * prices at steps 72 and 144 (9 + 9, ratio to base);
  * market inventories at 144 (9, /10000);
  * town shop unlock sequence (first 3 shops, one-hot by product family);
  * opponent day-3 opening family hash bucket (16 buckets);
  * opponent visible field at 144: crop counts (5) + animal count.

Rows join models/trackp/dispatch_grid.json (margins per economy) into
models/trackp/dispatch_features.npz for the GPU classifier.

    python src/trackp/harness/dispatch_features.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import kaggriculture.engine.serve_match as SM  # noqa: E402
from kaggriculture.trackp import routes_io as R                              # noqa: E402

GRID = os.path.join(ROOT, "models", "trackp", "dispatch_grid.json")
OUT = os.path.join(ROOT, "models", "trackp", "dispatch_features.npz")
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
        "MELON": 80, "EGG": 40, "MILK": 90, "WOOL": 110, "FERTILIZER": 50}
PRODUCTS = tuple(BASE)
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


def cell_features(srv, seed, opp_acts, opp_seat, fam_hash):
    js = srv.cmd(f"RESET {int(seed)}")
    snap = {}
    # drive the world with the OPPONENT's real actions and PASS for us --
    # the observable day-6 state depends almost entirely on the world seed
    # and their play; our probe must not require knowing our own tape
    empty = {"farmer": ["PASS"], "hands": [], "market": []}
    while not js.get("done") and int(js.get("step") or 0) < 145:
        i = int(js.get("step") or 0)
        if i in (72, 144):
            snap[i] = js
        o = SM.action_to_line(opp_acts[min(i, len(opp_acts) - 1)])
        m = SM.action_to_line(empty)
        la, lb = (o, m) if opp_seat == 0 else (m, o)
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
    snap.setdefault(144, js)
    snap.setdefault(72, snap[144])
    v = []
    for step in (72, 144):
        prices = (snap[step].get("market") or {}).get("prices") or {}
        v += [float(prices.get(p, 0) or 0) / BASE[p] for p in PRODUCTS]
    inv = (snap[144].get("market") or {}).get("inventory") or {}
    v += [float(inv.get(p, 0) or 0) / 10000.0 for p in PRODUCTS]
    shops = ((snap[144].get("town") or {}).get("unlocked_shops") or [])[:3]
    for k in range(3):
        s = str(shops[k]) if k < len(shops) else ""
        v += [1.0 if key in s.upper() else 0.0
              for key in ("SHEEP", "COW", "GOOSE", "SEED", "TOOL")]
    v += [((int(fam_hash, 16) >> b) & 1) * 1.0 for b in range(16)]
    farm = (snap[144].get("farms") or [{}, {}])[opp_seat]
    crops = {c: 0 for c in CROPS}
    animals = 0
    for row in (farm.get("tiles") or []):
        for t in (row if isinstance(row, list) else [row]):
            if isinstance(t, dict):
                if t.get("crop") in crops:
                    crops[t["crop"]] += 1
                if t.get("animal"):
                    animals += 1
    v += [crops[c] / 50.0 for c in CROPS] + [animals / 20.0]
    return v


def main():
    import numpy as np
    grid = json.load(open(GRID, encoding="utf-8"))
    X, Y, teams = [], [], []
    srv = SM.Serve()
    try:
        for i, c in enumerate(grid["cells"]):
            ep, es = str(c["episode"]), int(c["elite_seat"])
            try:
                opp = R.load_route(f"{ep}_s{es}")
                v = cell_features(srv, c["seed"], opp, es, c["family"])
            except Exception as exc:                           # noqa: BLE001
                print(f"  cell {ep}: {type(exc).__name__}", flush=True)
                continue
            X.append(v)
            Y.append([m if m is not None else -1e9
                      for m in c["margins"]])
            teams.append(c["team"])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(grid['cells'])} cells", flush=True)
    finally:
        srv.close()
    np.savez_compressed(OUT, X=np.asarray(X, dtype=np.float32),
                        Y=np.asarray(Y, dtype=np.float32),
                        teams=np.asarray(teams),
                        bases=np.asarray(grid["bases"]))
    print(f"{len(X)} cells x {len(X[0])} features -> "
          f"{os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
