"""Fast forward model: what will this farm be worth in H turns?

Why this is a forecaster and not a simulator
--------------------------------------------
The obvious move is to re-implement the game so training can run without the
official interpreter. The reference implementations that do this are how you
end up tuning an agent for a game nobody is playing: the copy drifts from the
real rules, the drift is silent, and every number you measure afterwards is
about your copy. We already vendor the official `kaggle-environments`
interpreter, and that stays the ground truth for every result.

What search and RL actually need is narrower and much safer to build: a cheap
estimate of *how much this position is worth*, so a candidate plan can be
ranked without playing the game out. That is a value function, and unlike a
simulator its error is directly measurable -- `src/kaggriculture/engine/parity.py` plays real
episodes and reports how far the forecast was from what happened.

    from kaggriculture.engine.fastsim import forecast
    v = forecast(obs, config, horizon=72)      # ~3 in-game days ahead

    python -m kaggriculture.engine.fastsim --demo             # forecast a real position
    python -m kaggriculture.engine.parity --matches 4         # how wrong is it?

Read the parity number before trusting a forecast for anything. It is published
rather than assumed for exactly the reason above.
"""
from kaggriculture.paths import ROOT
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_AGENT = None


def _tables():
    """CROPS/ANIMALS/prices from the agent source, never a second copy.

    Importing the module rather than restating the tables means the forecaster
    cannot drift from the policy that uses it -- the failure mode this whole
    file exists to avoid.
    """
    global _AGENT
    if _AGENT is None:
        path = os.path.join(ROOT, "agents", "v1_heuristic.py")
        spec = importlib.util.spec_from_file_location("_fastsim_tables", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _AGENT = mod
    return _AGENT


def _iter_tiles(farm):
    """Every occupied tile as a dict.

    `farm["tiles"]` is a 10x10 grid whose cells are None, the string "LOCKED",
    or a dict. Anything that is not a dict carries no value to forecast.
    """
    grid = farm.get("tiles") or []
    for row in grid:
        cells = row if isinstance(row, (list, tuple)) else [row]
        for t in cells:
            if isinstance(t, dict):
                yield t


def _price(obs, item):
    mod = _tables()
    try:
        return float(mod._price(obs, item))
    except Exception:                                              # noqa: BLE001
        return 0.0


def farm_value(obs, seat=None):
    """Liquidation value now: cash, plus everything sellable at current prices.

    Deliberately excludes standing crops and livestock -- those are counted by
    `forecast` as future income, and counting them twice is how a value function
    learns to hoard.
    """
    seat = obs["player"] if seat is None else seat
    farm = obs["farms"][seat]
    total = float(farm.get("money", 0) or 0)
    private = obs.get("private") or {}
    for store in (private.get("shed") or {},):
        for item, qty in store.items():
            total += _price(obs, item) * float(qty or 0)
    for inv in (private.get("inventories") or []):
        for item, qty in (inv or {}).items():
            total += _price(obs, item) * float(qty or 0)
    return total


def _labour_capacity(obs, P, seat):
    farm = obs["farms"][seat]
    hands = 1 + len(farm.get("hands") or [])
    return hands * 24.0 * float(P.get("capacity_util", 0.85))


def forecast(obs, config=None, horizon=72, params=None, seat=None):
    """Estimated liquidation value `horizon` turns from now.

    Three income streams, each capped by the thing that actually limits it:

      crops    -- each planted tile yields its remaining harvests, but only as
                  many as the crew has unit-turns to water and harvest
      animals  -- each animal yields per day from its first yield day, minus
                  the feed it eats, and only while the crew can feed it
      holdings -- what is already sellable, at current prices

    The cap is the point. A forecast that assumes every tile is tended is
    exactly the mistake the agent itself used to make: it plants a field it
    cannot water and reports a fortune it will never collect.
    """
    mod = _tables()
    P = params or mod.PARAMS
    seat = obs["player"] if seat is None else seat
    farm = obs["farms"][seat]

    turns_per_day = 24
    if config:
        try:
            turns_per_day = int(config["turnsPerDay"])
        except (KeyError, TypeError, ValueError):
            turns_per_day = 24
    days = max(0.0, horizon / float(turns_per_day))

    total_days = 30
    if config:
        try:
            total_days = max(1, int(config["episodeSteps"]) // turns_per_day)
        except (KeyError, TypeError, ValueError):
            total_days = 30
    days_left = max(0.0, min(days, total_days - float(obs.get("day", 0) or 0)))

    value = farm_value(obs, seat)
    capacity = _labour_capacity(obs, P, seat) * days_left
    spent = 0.0

    crop_income = 0.0
    animal_income = 0.0
    feed_cost = 0.0

    for tile in _iter_tiles(farm):
        crop = tile.get("crop") if tile.get("kind") == "PLANT" else None
        if crop and crop in mod.CROPS:
            cd = mod.CROPS[crop]
            age = float(tile.get("age", 0) or 0)
            got = float(tile.get("harvests", 0) or 0)
            left = max(0.0, cd["max_yield"] - got)
            reachable = min(left, max(0.0, (cd["max_yield_day"] - age + days_left)) / 1.0)
            cost = float(P.get("cost_per_crop_day", 2.0)) * days_left
            if spent + cost <= capacity:
                spent += cost
                crop_income += reachable * _price(obs, crop) * 0.9

        animal = tile.get("animal")
        if animal and animal in mod.ANIMALS:
            ad = mod.ANIMALS[animal]
            age = float(tile.get("age", 0) or 0)
            producing = max(0.0, days_left - max(0.0, ad.get("first_yield_day", 0) - age))
            cost = float(P.get("cost_per_animal_day", 4.0)) * days_left
            if spent + cost <= capacity:
                spent += cost
                product = ad.get("product") or ad.get("yield")
                amount = ad.get("yield_amount", ad.get("amount", 1))
                animal_income += producing * _price(obs, product) * float(amount or 1)
                feed_cost += producing * _price(obs, "WHEAT") * float(
                    ad.get("feed", ad.get("feed_per_day", 1)) or 1)

    return {
        "value_now": value,
        "crop_income": crop_income,
        "animal_income": animal_income,
        "feed_cost": feed_cost,
        "labour_used": spent,
        "labour_available": capacity,
        "labour_bound": spent >= capacity - 1e-9,
        "horizon": horizon,
        "forecast": value + crop_income + animal_income - feed_cost,
    }


def _demo():
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 240, "seed": 9,
                              "actTimeout": 60, "runTimeout": 100000})
    seen = {}

    def spy(obs, cfg):
        if obs.get("step") in (24, 96, 168):
            seen[obs["step"]] = (obs, cfg)
        return _tables().agent(obs, cfg)

    env.run([spy, "agents/v2_tuned.py"])
    for step, (obs, cfg) in sorted(seen.items()):
        f = forecast(obs, cfg, horizon=72)
        print(f"turn {step:>4}  now ${f['value_now']:>9,.0f}  "
              f"+72 turns ${f['forecast']:>9,.0f}   "
              f"(crops ${f['crop_income']:>8,.0f}, animals ${f['animal_income']:>8,.0f}, "
              f"feed -${f['feed_cost']:>7,.0f})"
              f"{'  [labour-bound]' if f['labour_bound'] else ''}")
    final = env.steps[-1][0]["reward"]
    print(f"\nactual final bank: ${float(final or 0):,.0f}")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--horizon", type=int, default=72)
    args = ap.parse_args()
    if args.demo:
        return _demo()
    print(__doc__)
    print("run with --demo, then check src/kaggriculture/engine/parity.py for the error it makes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
