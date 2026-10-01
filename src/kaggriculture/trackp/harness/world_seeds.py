"""Seed->world bank: manufacture any shop-pair world on demand.

Operator direction (2026-09-06): stop depending on what the field happened
to play — the rules are known, so enumerate the worlds ourselves. The
world is the ordered pair of the first two town shops, fixed by the seed.
This scans seeds on the Rust engine (PASS-only drive to step 150, where
both unlocks are visible) and writes a bank of seeds per world.

Downstream: the factory's (1+lambda) tail evolution picks its episode
seeds per world from this bank, so every one of the 64 worlds gets owned
tails searched in ITS OWN worlds — no tape required to reach a world.

    python src/trackp/harness/world_seeds.py --seeds 2000
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.engine.serve_match as SM  # noqa: E402

OUT = os.path.join(ROOT, "data", "worlds", "seed_bank.json")
PASS_LINE = json.dumps({"farmer": ["PASS"], "hands": [], "market": []},
                       separators=(",", ":"))


def world_of_seed(srv, seed):
    js = srv.cmd(f"RESET {seed}")
    for _ in range(150):
        js = srv.cmd(f"STEP2 {PASS_LINE}\x1e{PASS_LINE}")
        if js.get("done"):
            break
    shops = (js.get("town") or {}).get("unlocked_shops") or []
    return "|".join(shops[:2]) if len(shops) >= 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--start", type=int, default=1)
    a = ap.parse_args()
    bank = defaultdict(list)
    if os.path.exists(OUT):
        for k, v in json.load(open(OUT, encoding="utf-8")).items():
            bank[k] = v
    srv = SM.Serve()
    try:
        for seed in range(a.start, a.start + a.seeds):
            if any(seed in v for v in bank.values()):
                continue
            w = world_of_seed(srv, seed)
            if w:
                bank[w].append(seed)
            if seed % 250 == 0:
                print(f"  seed {seed}: {len(bank)} worlds, "
                      f"{sum(len(v) for v in bank.values())} banked",
                      flush=True)
    finally:
        srv.close()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(bank), open(OUT, "w", encoding="utf-8"))
    print(f"\n{len(bank)} distinct worlds, "
          f"{sum(len(v) for v in bank.values())} seeds -> {OUT}")
    for w, v in sorted(bank.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {w:36s} {len(v):>4} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
