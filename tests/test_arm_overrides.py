"""Contract tests for arm market overrides (v23.1 postmortem, 2026-08-11).

An arm may only retime SELL orders. If an override turn ever loses a
structural order (BUY_LAND, HIRE, BUY_ANIMAL, BUY_SEED, BUY_PRODUCT) from
the base plan, the arm wrecks the farm economy it is grafted onto: the
raw-replace override dropped a BUY_LAND and halved the bank in 14 of 19
ladder losses of v23.1_bandit.

Run with:  python tests/test_arm_overrides.py
"""
import os
import sys


import kaggriculture.agentbuild.v22_agent as V  # noqa: E402

STRUCT = ("BUY_LAND", "HIRE", "BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT")


def _count(orders, op):
    return sum(1 for o in orders if isinstance(o, list) and o and o[0] == op)


def _mk(n_turns, at):
    route = [{"farmer": ["PASS"], "hands": [], "market": []}
             for _ in range(n_turns)]
    for t, market in at.items():
        route[t]["market"] = market
    return route


def test_structural_orders_survive():
    n = V.PREFIX_END + 10
    base = _mk(n, {V.PREFIX_END + 2: [
        ["SELL", "WOOL", 16], ["SELL", "WHEAT", 2], ["BUY_LAND"],
        ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
        ["HIRE"]]})
    # the arm drops the whole turn (the v23.1 failure shape)
    arm = _mk(n, {V.PREFIX_END + 2: [["SELL", "WHEAT", 3]]})
    diff = V.market_overrides(base, arm)
    key = str(V.PREFIX_END + 2)
    assert key in diff, "differing turn must be overridden"
    merged = diff[key]
    b = base[V.PREFIX_END + 2]["market"]
    for op in STRUCT:
        assert _count(merged, op) >= _count(b, op), \
            f"override lost structural order {op}: {merged}"
    assert _count(merged, "SELL") == 1 and ["SELL", "WHEAT", 3] in merged, \
        "arm SELL retiming must be applied"
    assert len(merged) <= 10, "order cap"


def test_arm_never_adds_structural_orders():
    n = V.PREFIX_END + 10
    base = _mk(n, {V.PREFIX_END + 1: [["SELL", "MILK", 3]]})
    arm = _mk(n, {V.PREFIX_END + 1: [
        ["BUY_ANIMAL", "COW", 2], ["BUY_SEED", "MELON", 12],
        ["SELL", "MILK", 3]]})
    diff = V.market_overrides(base, arm)
    merged = diff.get(str(V.PREFIX_END + 1))
    if merged is not None:
        for op in ("BUY_ANIMAL", "BUY_SEED"):
            assert _count(merged, op) == 0, \
                f"arm smuggled structural order {op}: {merged}"


def test_identical_turns_not_overridden():
    n = V.PREFIX_END + 5
    m = {V.PREFIX_END + 1: [["SELL", "MILK", 3], ["HIRE"]]}
    assert V.market_overrides(_mk(n, m), _mk(n, m)) == {}


def test_cap_prefers_base_structure():
    n = V.PREFIX_END + 5
    base = _mk(n, {V.PREFIX_END + 1: [["SELL", "WOOL", 4]] + [["HIRE"]] * 9})
    arm = _mk(n, {V.PREFIX_END + 1: [["SELL", "WOOL", 2], ["SELL", "MILK", 2],
                                     ["SELL", "EGG", 2]]})
    merged = V.market_overrides(base, arm)[str(V.PREFIX_END + 1)]
    assert len(merged) <= 10
    assert _count(merged, "HIRE") == 9, "base structure wins under the cap"
    assert _count(merged, "SELL") == 1, "one slot left for arm sells"


if __name__ == "__main__":
    test_structural_orders_survive()
    test_arm_never_adds_structural_orders()
    test_identical_turns_not_overridden()
    test_cap_prefers_base_structure()
    print("all arm-override checks passed")
