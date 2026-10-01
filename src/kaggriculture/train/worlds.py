"""T2/T3 -- world generator + conditioning for the macro policy.

The macro policy is world-BLIND by default (it samples with world_bucket=0). Two
fixes live here:

  * T3 WORLD GENERATOR -- extract the day-6 shop-world frequency the ladder
    actually visits (from the BC corpus `world_sig` column) so training samples
    TARGET worlds by their real frequency instead of a uniform bucket.
  * T2 WORLD-CONDITIONING -- map a `world_sig` to the same bucket the policy's
    feature builder uses (bc_warmup._world_bucket), so the policy can condition
    on a target world at train time and on the OBSERVED day-6 world at inference
    (the reactive/live seat re-plans at D6; the compiled tape picks the modal
    world).

Architectural honesty: for a COMPILED (open-loop) tape the plan is committed
before the world reveals, so this teaches a world-PRIOR-conditioned plan; true
per-game adaptation is the trackp live seat / the bandit's D6 rails.

    python -m kaggriculture.train.worlds --build     # write data/worlds/world_freq.json
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import json
import os
import random
import sys

import kaggriculture.train.bc_warmup as BC

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

WORLD_FREQ = os.path.join(ROOT, "data", "worlds", "world_freq.json")


def build_world_freq() -> dict:
    """Count day-6 world_sig frequency across the macro corpus. Returns {sig: n}."""
    import pyarrow.parquet as pq
    from collections import Counter
    c = Counter()
    for f in ("macro.parquet", "macro_destbreso.parquet"):
        p = os.path.join(BC.MACRO_DIR, f)
        if not os.path.exists(p):
            continue
        pf = pq.ParquetFile(p)
        names = set(pf.schema_arrow.names)
        if "world_sig" not in names or "day" not in names:
            continue
        for b in pf.iter_batches(batch_size=8192, columns=["world_sig", "day"]):
            d = b.to_pydict()
            for sig, day in zip(d["world_sig"], d["day"]):
                if float(day) == 0 and sig:      # one count per episode-seat
                    c[sig] += 1
    freq = dict(c.most_common())
    os.makedirs(os.path.dirname(WORLD_FREQ), exist_ok=True)
    with open(WORLD_FREQ, "w", encoding="utf-8") as fh:
        json.dump(freq, fh, indent=1)
    return freq


SEED_BANK = os.path.join(ROOT, "data", "worlds", "seed_bank.json")


def build_seed_bank(n_probe: int = 120, keep: int = 32) -> dict:
    """G3: realize candidate seeds to day 6 (PASS vs PASS on the fast Rust serve),
    record each seed's day-6 shop-world, and keep a WORLD-DIVERSE subset so the
    gate covers many worlds instead of clustering on a few. (Caveat: the PASS
    world can differ from a contested world — this is a coverage aid, not an
    exact per-seed world label.)"""
    import json
    import random
    try:
        from kaggriculture.engine.serve_match import Serve
    except Exception:
        return {}
    srv = Serve()
    rng = random.Random(0)
    by_world = {}
    try:
        for _ in range(n_probe):
            seed = rng.randint(1, 2**31 - 1)
            js = srv.cmd(f"RESET {seed}")
            for _ in range(6 * 24):          # step PASS to day 6
                if js.get("day", 0) >= 6:
                    break
                js = srv.cmd("STEP2 PASS\x1ePASS")
                if "error" in js:
                    break
            shops = (js.get("town", {}) or {}).get("unlocked_shops", []) or []
            sig = "|".join(sorted(shops[:2]))
            by_world.setdefault(sig, []).append(seed)
    finally:
        srv.close()
    # round-robin across worlds -> diverse coverage
    seeds, worlds = [], list(by_world)
    i = 0
    while len(seeds) < keep and any(by_world.values()):
        w = worlds[i % len(worlds)]
        if by_world[w]:
            seeds.append(by_world[w].pop())
        i += 1
    out = {"seeds": seeds, "worlds_covered": len(by_world),
           "by_world": {w: len(v) for w, v in by_world.items()}}
    os.makedirs(os.path.dirname(SEED_BANK), exist_ok=True)
    with open(SEED_BANK, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


class WorldSampler:
    """Samples a target world_sig by ladder frequency and maps it to a bucket."""

    def __init__(self, path: str = WORLD_FREQ):
        self.freq = {}
        if os.path.exists(path):
            try:
                self.freq = json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError):
                self.freq = {}
        if not self.freq:
            self.freq = {"": 1}                  # graceful: world-blind fallback
        self.sigs = list(self.freq.keys())
        self.weights = [max(1, int(v)) for v in self.freq.values()]

    def sample_bucket(self, rng: random.Random) -> int:
        sig = rng.choices(self.sigs, weights=self.weights, k=1)[0]
        return BC._world_bucket(sig)

    def modal_bucket(self) -> int:
        sig = max(self.freq, key=lambda s: self.freq[s]) if self.freq else ""
        return BC._world_bucket(sig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed-bank", action="store_true",
                    help="G3: build a world-diverse seed bank for the gate")
    args = ap.parse_args()
    if args.seed_bank:
        b = build_seed_bank()
        print(f"[worlds] seed bank: {len(b.get('seeds', []))} seeds across "
              f"{b.get('worlds_covered')} worlds -> {os.path.relpath(SEED_BANK, ROOT)}")
        return 0
    freq = build_world_freq()
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:10]
    print(f"[worlds] {len(freq)} distinct day-6 worlds; top:")
    for sig, n in top:
        print(f"   {n:>6}  {sig}")
    print(f"[worlds] wrote {os.path.relpath(WORLD_FREQ, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
