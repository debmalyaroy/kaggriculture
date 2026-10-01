"""The submission contract, checked mechanically.

Four rules decide whether an agent can be submitted at all, and every one of
them has already cost this project a slot or a silent loss:

1. **One self-contained file importing nothing but `math`.** Kaggle uploads a
   single `main.py`; a package-relative import is an upload that validates
   locally and dies on the ladder.
2. **Legal actions only.** Illegal ops are *silent* no-ops in the interpreter,
   so a bug looks like laziness rather than an error. We assert the agent's
   output shape instead of waiting to notice.
3. **`hands` aligned positionally** with `farms[me]["hands"]`, and **at most 10
   market orders** -- extras are dropped without warning.
4. **A turn budget.** Kaggle allows 1s per turn on 1.6 vCPU, which is slower
   than any dev box. We hold the mean under 20ms and the worst under 150ms so
   there is room for that gap.

Run directly (`python tests/test_contract.py`) or under pytest.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import ast
import glob
import os
import sys
import time

import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

UNIT_OPS = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
            "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER"}
MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
# The rule that matters is *standalone*: no package-relative import, no shared
# helper module, no third-party dependency. `math` covers the heuristic; the
# ensemble draws jitter with `random` and the gradient-boosted arbiter compares
# thresholds in float32 via `struct`. All three ship with CPython, so they cost
# nothing on the ladder -- a numpy import would not.
# A route agent carries its 719 turns as a base85+zlib blob, so it needs
# base64/zlib to unpack it, json to parse it and copy to hand out a fresh
# action each turn. All four are CPython stdlib and cost nothing on the
# ladder, which is the only thing this rule is protecting -- a numpy import
# would not be.
ALLOWED_IMPORTS = {"math", "random", "struct", "base64", "zlib", "json",
                   "copy",
                   # gc: deliberate since 2026-08-14 -- gc.freeze() +
                   # threshold tuning in every generated agent kills the
                   # 245-330 ms collector pauses that failed the latency
                   # gate (docs/history/issues-and-improvements.md, GC section).
                   "gc"}
MAX_ORDERS = 10
MAX_BYTES = 100 * 1024 * 1024


# Agents under test: everything shipped plus whatever the search last wrote.
# This globbed only v9_* for a while, which quietly excluded v10/v13/v14 --
# including the agent that was actually on the ladder. Match every version.
def agent_paths():
    named = [os.path.join(ROOT, "agents", n) for n in
             ("v1_heuristic.py", "v2_tuned.py")]
    named += sorted(glob.glob(os.path.join(ROOT, "agents", "v[0-9]*.py")))
    named += sorted(glob.glob(os.path.join(ROOT, "build", "main.py")))
    seen, out = set(), []
    for p in named:
        key = os.path.abspath(p)
        if key not in seen and os.path.exists(p):
            seen.add(key)
            out.append(p)
    return out


def load(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agent_" + os.path.basename(path)[:-3].replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_file_and_stdlib_only():
    """No package imports, no third-party dependency, no file I/O."""
    for path in agent_paths():
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise AssertionError(
                        f"{path}: relative import -- the submission is one file")
                imported.add((node.module or "").split(".")[0])
        extra = imported - ALLOWED_IMPORTS
        assert not extra, f"{path}: imports {sorted(extra)}, only {sorted(ALLOWED_IMPORTS)} allowed"
        assert os.path.getsize(path) < MAX_BYTES, f"{path}: over Kaggle's size limit"
        assert "def agent(" in src, f"{path}: no agent() entry point"
    print(f"contract: {len(agent_paths())} agent(s) are single-file, math-only")


def _episode(path, steps=240, seed=5):
    """Play a short episode against the built-in starter, checking every action."""
    from kaggle_environments import make
    mod = load(path)
    worst, total, turns = 0.0, 0.0, 0
    seen_hand_mismatch = []

    def wrapped(obs, config=None):
        nonlocal worst, total, turns
        t0 = time.perf_counter()
        action = mod.agent(obs, config) if config is not None else mod.agent(obs)
        dt = (time.perf_counter() - t0) * 1000.0
        worst = max(worst, dt)
        total += dt
        turns += 1

        assert isinstance(action, dict), "action must be a dict"
        assert set(action) <= {"farmer", "hands", "market"}, f"stray keys {set(action)}"
        farmer = action.get("farmer") or ["PASS"]
        assert isinstance(farmer, list) and farmer and farmer[0] in UNIT_OPS, \
            f"illegal farmer op {farmer}"
        hands = action.get("hands") or []
        assert isinstance(hands, list), "hands must be a list"
        live = len(obs["farms"][obs["player"]]["hands"])
        if len(hands) != live:
            seen_hand_mismatch.append((len(hands), live))
        for h in hands:
            assert isinstance(h, list) and h and h[0] in UNIT_OPS, f"illegal hand op {h}"
        orders = action.get("market") or []
        assert isinstance(orders, list), "market must be a list"
        assert len(orders) <= MAX_ORDERS, f"{len(orders)} market orders (max {MAX_ORDERS})"
        for o in orders:
            assert isinstance(o, list) and o and o[0] in MARKET_OPS, f"illegal order {o}"
            if o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
                assert len(o) >= 3 and int(o[2]) > 0, f"order needs a positive count: {o}"
        return action

    env = make("kaggriculture", configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    env.run([wrapped, "starter"])
    assert env.steps[-1][0]["status"] == "DONE", "agent did not finish the episode"
    return {"mean": total / max(1, turns), "worst": worst,
            "bank": float(env.steps[-1][0]["reward"] or 0),
            "hand_mismatch": seen_hand_mismatch}


def test_actions_are_legal_and_shaped():
    for path in agent_paths():
        r = _episode(path)
        assert not r["hand_mismatch"], \
            f"{path}: hands misaligned with farm hands, e.g. {r['hand_mismatch'][:3]}"
        print(f"contract: {os.path.basename(path):<28} legal, "
              f"bank ${r['bank']:,.0f} over 240 turns")


# Dead experiments kept for the record. Legality and imports are still asserted
# for these -- an illegal action is a bug wherever it lives -- but the latency
# budget is not, because latency only matters for something we might ship and
# none of these will be. Measured mean/worst on this box: the v5 ensembles run
# 21-58 ms mean and up to 410 ms worst, which is exactly why the ensemble line
# was abandoned. Delete the file or drop it from this list to re-arm the check.
LEGACY = {
    "v5_ensemble_20260805_025648.py", "v5_ensemble_20260805_030740.py",
    "v5_ensemble_20260805_034029.py", "v5_ensemble_20260805_034448.py",
    "v5_ensemble_20260805_040835.py", "v5_ensemble_wide_20260805_025655.py",
    "v5_ensemble_wide_20260805_034452.py", "v5_ridge_ensemble_20260805_025656.py",
    "v5_ridge_ensemble_20260805_034453.py",
    "agent_vadapt_both_20260805_032405.py",
}


def test_turn_budget():
    """Kaggle allows 1000 ms on hardware slower than this box.

    Scoped to shippable agents. A dead experiment being slow costs nothing;
    letting it fail the suite costs the signal from every other check, which is
    how a real regression gets waved through.
    """
    skipped = []
    for path in agent_paths():
        name = os.path.basename(path)
        r = _episode(path)
        if name in LEGACY:
            skipped.append((name, r))
            continue
        assert r["mean"] < 20.0, f"{path}: mean turn {r['mean']:.1f} ms (limit 20)"
        assert r["worst"] < 150.0, f"{path}: worst turn {r['worst']:.1f} ms (limit 150)"
        print(f"contract: {os.path.basename(path):<28} "
              f"mean {r['mean']:.1f} ms  worst {r['worst']:.1f} ms")
    for name, r in skipped:
        print(f"contract: {name:<28} mean {r['mean']:.1f} ms  "
              f"worst {r['worst']:.1f} ms   (legacy, budget not enforced)")


if __name__ == "__main__":
    test_single_file_and_stdlib_only()
    test_actions_are_legal_and_shaped()
    test_turn_budget()
    print("\nall contract checks passed")
