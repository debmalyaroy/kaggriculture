"""Contract tests for the Kaggriculture agents.

Run with:  python -m pytest tests -q     (or just: python tests/test_agents.py)

These check the things the competition will punish silently: malformed action
dicts, illegal ops, per-turn latency against the 1-second actTimeout, and
crashes on degenerate observations.
"""
from kaggriculture.paths import ROOT
import os
import sys
import time

sys.path.insert(0, os.path.join(ROOT, "agents"))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (puts .local/vendor on sys.path when kaggle-environments is not installed)

UNIT_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
    "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE",
    "DIG", "FEED", "CARE", "COLLECT_FERTILIZER",
}
MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
AGENT_FILES = ["agents/v0_baseline.py", "agents/v1_heuristic.py", "agents/v2_tuned.py"]

# The SHIPPED agents were never in this list, so the contract tests -- legality,
# the 10-order cap, hand alignment, latency against the 1 s actTimeout -- only
# ever covered three historical heuristics. The submissions are the artefacts
# that must not break, so the newest v24+ pair is tested too. Discovered rather
# than hardcoded, so tomorrow's release is covered without editing this file.
def _live_agents():
    import glob
    import re
    found = {}
    for p in glob.glob(os.path.join(ROOT, "agents", "v2[4-9]*.py")):
        name = os.path.basename(p)
        m = re.match(r"v(\d+)\.?(\d*)_(bandit|route)", name)
        if not m:
            continue
        kind = m.group(3)
        key = (int(m.group(1)), int(m.group(2) or 0))
        if kind not in found or key > found[kind][0]:
            found[kind] = (key, os.path.join("agents", name))
    return [v[1] for v in found.values()]


AGENT_FILES += _live_agents()


def _load(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        os.path.basename(path)[:-3], os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_action(action, n_hands):
    assert isinstance(action, dict), f"action must be a dict, got {type(action)}"
    assert set(action) <= {"farmer", "hands", "market"}, f"unexpected keys: {set(action)}"

    farmer = action.get("farmer", ["PASS"])
    assert isinstance(farmer, list) and farmer, "farmer op must be a non-empty list"
    assert farmer[0] in UNIT_OPS, f"unknown farmer op {farmer[0]}"

    hands = action.get("hands", [])
    assert isinstance(hands, list), "hands must be a list"
    assert len(hands) <= max(n_hands, 0) + 1, f"{len(hands)} hand ops for {n_hands} hands"
    for op in hands:
        assert isinstance(op, list) and op, "hand op must be a non-empty list"
        assert op[0] in UNIT_OPS, f"unknown hand op {op[0]}"

    market = action.get("market", [])
    assert isinstance(market, list), "market must be a list"
    assert len(market) <= 10, f"{len(market)} market orders exceeds maxMarketOrdersPerTurn"
    for order in market:
        assert isinstance(order, list) and order, "market order must be a non-empty list"
        assert order[0] in MARKET_OPS, f"unknown market op {order[0]}"
        if order[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
            assert len(order) == 3, f"{order[0]} needs item and count: {order}"
            assert isinstance(order[2], int) and order[2] > 0, f"bad count in {order}"


def run_episode(path, opponent="starter", seed=11, steps=720):
    """Play a full match, validating every action and timing every turn."""
    from kaggle_environments import make

    mod = _load(path)
    worst = 0.0
    total = 0.0
    turns = 0

    def wrapped(obs, config=None):
        nonlocal worst, total, turns
        t0 = time.time()
        try:
            action = mod.agent(obs) if mod.agent.__code__.co_argcount == 1 else mod.agent(obs, config)
        except Exception as exc:                                  # noqa: BLE001
            raise AssertionError(f"{path} raised on day {obs['day']} hour {obs['hour']}: {exc}")
        dt = time.time() - t0
        worst = max(worst, dt)
        total += dt
        turns += 1
        check_action(action, len(obs["farms"][obs["player"]]["hands"]))
        return action

    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000},
               debug=True)
    env.run([wrapped, opponent])
    final = env.steps[-1]
    return {
        "reward": float(final[0]["reward"] or 0),
        "status": final[0]["status"],
        "worst_turn_s": worst,
        "mean_turn_s": total / max(1, turns),
    }


def test_agents_are_legal_and_fast():
    for path in AGENT_FILES:
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"skip {path} (not built yet)")
            continue
        r = run_episode(path)
        assert r["status"] == "DONE", f"{path} finished with status {r['status']}"
        assert r["worst_turn_s"] < 0.5, (
            f"{path} worst turn {r['worst_turn_s']*1000:.0f} ms is too close to the "
            f"1 s actTimeout")
        print(f"{path:<28} ${r['reward']:>10,.0f}  "
              f"mean {r['mean_turn_s']*1000:5.1f} ms  worst {r['worst_turn_s']*1000:6.1f} ms")


def test_beats_builtin_baselines():
    from kaggle_environments import make
    for path in AGENT_FILES:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            continue
        for opp in ("pass", "random", "starter"):
            env = make("kaggriculture",
                       configuration={"episodeSteps": 720, "seed": 7,
                                      "actTimeout": 60, "runTimeout": 100000})
            env.run([full, opp])
            mine, theirs = (float(env.steps[-1][i]["reward"] or 0) for i in (0, 1))
            assert mine > theirs, f"{path} lost to {opp}: {mine:,.0f} vs {theirs:,.0f}"
            print(f"{path:<28} beats {opp:<8} ${mine:>10,.0f} vs ${theirs:>9,.0f}")


if __name__ == "__main__":
    test_agents_are_legal_and_fast()
    test_beats_builtin_baselines()
    print("\nall checks passed")
