"""Dev-only RNG decoupling + shop forcing for the vendored interpreter.

The engine derives ONE `random.Random((seed * 1_000_003) ^ day)` per day and
consumes it for BOTH weed spawns (`.random()`, called once per EMPTY tile --
a count the players control) and the town shop draw (`.choice()`). So a
candidate that digs or plants differently from its control shifts the shop
unlock sequence mid-A/B, and "same seed" is not the same world (reported
independently: discussion 737663; Michael Timbs, 8th, runs exactly this
harness locally).

`activate()` monkeypatches the vendored module with a proxy Random whose
`.random()` stream is untouched (weeds stay seed-faithful) while `.choice()`
draws from an INDEPENDENT stream keyed `day + 1` -- Biolatti's proposed
split, applied locally. Optionally, `forced_shops` pins the entire unlock
sequence outright for fixed-replay evals and shop-sensitivity sweeps.

STRICTLY an eval-noise instrument. It changes the world model, so:
* NEVER active in `ladder_parity.py`, `submit.py` validation, serve
  equivalence audits, or anything whose job is to mirror the ladder.
* Never baked into a build. Activation is per-process and explicit.

    from kaggriculture.engine.forced_env import activate
    activate()                             # split streams only
    activate(forced_shops=["BAKERY", "YARN_STORE", ...])   # pin the draw

    python src/forced_env.py --selftest    # proves the split + the pin
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_MULT = 1_000_003


class _SplitRandom:
    """`.random()` = the engine's own stream; `.choice()` = its own stream.

    The day function uses `.random()` only for weeds and `.choice()` only
    for the shop draw, so splitting at the method level reproduces
    Biolatti's two-rng proposal without touching engine source.
    """

    def __init__(self, key, forced_shops, drawn_counter):
        self._weeds = random.Random(key)
        self._shops = random.Random(key + 1)
        self._forced = forced_shops
        self._drawn = drawn_counter

    def random(self):
        return self._weeds.random()

    def choice(self, seq):
        if self._forced is not None:
            i = self._drawn[0]
            self._drawn[0] += 1
            if i < len(self._forced):
                pick = self._forced[i]
                if pick in seq:
                    return pick
            # Forced list exhausted or invalid entry: stable fallback.
        return self._shops.choice(seq)

    def __getattr__(self, name):                 # any future method use
        return getattr(self._weeds, name)


def activate(forced_shops=None):
    """Patch the vendored interpreter in THIS process. Returns undo()."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments.envs.kaggriculture import kaggriculture as mod

    real_random = mod.random
    drawn = [0]

    class _Proxy:
        def Random(self, key=None):
            # The day function is the only caller that seeds with the
            # multiplied key; everything else gets the real class.
            if isinstance(key, int) and key >= _MULT - (1 << 20):
                return _SplitRandom(key, forced_shops, drawn)
            return real_random.Random(key)

        def __getattr__(self, name):
            return getattr(real_random, name)

    mod.random = _Proxy()

    def undo():
        mod.random = real_random

    return undo


def _selftest():
    """Same seed, different play -> same shops under the patch."""
    import json                                                  # noqa: F401
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make

    def shops_after(digger, patched, forced=None):
        undo = activate(forced) if patched else None
        try:
            env = make("kaggriculture",
                       configuration={"episodeSteps": 240, "actTimeout": 60,
                                      "runTimeout": 100000, "seed": 99},
                       info={"seed": 99})

            def busy(obs, cfg=None):
                # Perturb the empty-tile count relative to the idle twin.
                farmer = ["PLANT", "WHEAT"] if digger and obs["step"] % 3 == 0 \
                    else ["PASS"]
                hands = [["PASS"] for _ in
                         (obs["farms"][obs["player"]].get("hands") or [])]
                mkt = [["BUY_SEED", "WHEAT", 1]] if digger and obs["step"] < 40 \
                    else []
                return {"farmer": farmer, "hands": hands, "market": mkt}

            env.run([busy, busy])
            town = env.state[0].observation.get("town") or {}
            return list(town.get("unlocked_shops") or [])
        finally:
            if undo:
                undo()

    base = shops_after(False, False)
    moved = shops_after(True, False)
    print(f"unpatched idle:   {base}")
    print(f"unpatched digger: {moved}")
    split_a = shops_after(False, True)
    split_b = shops_after(True, True)
    print(f"patched idle:     {split_a}")
    print(f"patched digger:   {split_b}")
    pin = ["BAKERY", "YARN_STORE", "PET_CAFE"]
    pinned = shops_after(True, True, forced=pin)
    print(f"pinned {pin}: {pinned}")
    ok = (split_a == split_b) and pinned[:len(pin)] == pin[:len(pinned)]
    coupled = base != moved
    print(f"\ncoupling visible unpatched: {coupled}; "
          f"split stable: {split_a == split_b}; pin honored: "
          f"{pinned[:len(pin)] == pin[:len(pinned)]}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
