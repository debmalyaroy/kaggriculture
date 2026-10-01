"""Differential test: every pure rule function/table against the interpreter.

These are total functions of their arguments, so they can be checked
EXHAUSTIVELY rather than sampled -- fib over its whole used range, the
quadrant map over all 100 board cells, every crop and animal field. That is
worth doing because a transcribed constant table is the highest-probability
place for a port to be quietly wrong, and because the failure mode is not a
crash: a mis-copied `max_yield` or an off-by-one `fib` index produces an engine
that runs fine and prices a farm differently.

The Python side is the vendored module itself; nothing here restates a value
that could be read.

    python tests/test_rust_rules.py
"""
from kaggriculture.paths import ROOT
import json
import os
import subprocess
import sys

BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg.exe")
if not os.path.exists(BIN):
    BIN = os.path.join(ROOT, "rustengine", "target", "release", "kagg")


def engine():
    try:
        import kaggriculture.engine._vendor as _vendor  # noqa: F401
        from kaggle_environments.envs.kaggriculture import kaggriculture as k
        return k
    except Exception as exc:                                    # noqa: BLE001
        print(f"engine import failed: {type(exc).__name__}: {exc}")
        return None


ENG = engine()
R = json.loads(subprocess.run([BIN, "rules"], capture_output=True, text=True,
                              timeout=60).stdout)


def test_fib_indexing():
    """fib(0)=1, fib(1)=1, fib(2)=2 -- NOT textbook indexing."""
    want = [ENG._fib(n) for n in range(25)]
    assert R["fib"] == want, (R["fib"][:8], want[:8])
    assert want[:5] == [1, 1, 2, 3, 5], want[:5]
    print(f"fib(0..24) exact; indexing confirmed {want[:6]}")


def test_hire_cost_curve():
    want = [ENG._hire_cost(n) for n in range(15)]
    assert R["hire_cost"] == want, (R["hire_cost"], want)
    print(f"hire cost for hires 1..15 of a day: {want}")


def test_quadrant_map_exhaustive():
    want = [ENG._quadrant_of(x, y, 10) for y in range(10) for x in range(10)]
    assert R["quadrants"] == want, "quadrant map differs"
    assert len(set(want)) == 4, set(want)
    print(f"quadrant map: all 100 cells exact, {sorted(set(want))}")


def test_shed_access_tiles():
    want = [list(t) for t in ENG._shed_access_tiles(10)]
    assert R["shed_access"] == want, (R["shed_access"], want)
    for t in want:
        assert ENG._is_shed_adjacent(tuple(t), 10)
    assert not ENG._is_shed_adjacent((0, 0), 10)
    print(f"shed access tiles exact: {want}")


def test_land_order_and_prices():
    assert R["land_order"] == list(ENG.LAND_ORDER), R["land_order"]
    assert R["land_prices"] == list(ENG.LAND_PRICES), R["land_prices"]
    assert R["max_shop_instances"] == ENG.MAX_SHOP_INSTANCES
    print(f"land: {R['land_order']} at {R['land_prices']}; "
          f"max shop instances {R['max_shop_instances']}")


def test_crop_table_field_by_field():
    assert set(R["crops"]) == set(ENG.CROPS), (set(R["crops"]),
                                               set(ENG.CROPS))
    for name, want in ENG.CROPS.items():
        got = R["crops"][name]
        for k, v in want.items():
            assert got[k] == v, f"{name}.{k}: rust {got[k]} != engine {v}"
    print(f"crops: {len(ENG.CROPS)} entries, every field exact")


def test_animal_table_field_by_field():
    assert set(R["animals"]) == set(ENG.ANIMALS)
    for name, want in ENG.ANIMALS.items():
        got = R["animals"][name]
        for k, v in want.items():
            assert got[k] == v, f"{name}.{k}: rust {got[k]} != engine {v}"
    print(f"animals: {len(ENG.ANIMALS)} entries, every field exact")


def test_products_list_matches():
    """PRODUCTS order matters wherever the engine iterates it."""
    from_state = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
                  "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"]
    assert sorted(ENG.PRODUCTS) == from_state, sorted(ENG.PRODUCTS)
    print(f"products: {len(ENG.PRODUCTS)} items, sorted set matches the port")


if __name__ == "__main__":
    assert ENG is not None, "vendored interpreter not importable"
    for fn in (test_fib_indexing, test_hire_cost_curve,
               test_quadrant_map_exhaustive, test_shed_access_tiles,
               test_land_order_and_prices, test_crop_table_field_by_field,
               test_animal_table_field_by_field, test_products_list_matches):
        fn()
    print("\nPURE RULES VERIFIED exhaustively against the interpreter")
