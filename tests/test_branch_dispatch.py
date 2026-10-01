"""The runtime branch dispatch: it must fire, and it must not leak.

Two failures this exists to prevent.

1. **Cross-episode leak.** The first branch runtime swapped the MODULE-GLOBAL
   `_ROUTE` when it dispatched. `kagg serve` and the ladder both reuse an
   agent process across episodes, so game 2 would open playing game 1's
   branch -- from turn 0, with none of the board that branch assumes. It was
   recorded as a dormant hazard in the 2026-08-31 audit ("dead now, live if
   config changes"); enabling a branchpack is exactly that config change, so
   the guarantee is asserted here rather than remembered.

2. **A branchpack the runtime cannot play.** Illegal ops are silent no-ops,
   so a branch of the wrong length, with an over-long market queue, or
   carrying its own opening instead of the base's, does not raise -- it just
   plays 648 turns of nothing and looks like a weak agent.
   `v22_agent.validate_branchpack` refuses all three at build time.

    python tests/test_branch_dispatch.py
"""
from kaggriculture.paths import ROOT
import copy
import glob
import json
import os
import sys


BRANCH_STEP = 72          # day 3 dawn: the first town shop unlock
FAILED = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


# ------------------------------------------------------------------ builder

def test_validator():
    import kaggriculture.agentbuild.v22_agent as V
    print("validate_branchpack (BRANCH GUARD)")
    base = [{"farmer": ["PASS"], "hands": [], "market": []}
            for _ in range(720)]
    good = [dict(t) for t in base]
    good[100] = {"farmer": ["WATER"], "hands": [], "market": []}

    lines = V.validate_branchpack(base, {"YARN_STORE": good})
    check(bool(lines) and "1 distinct routes" in lines[0],
          "a well-formed pack passes and reports its distinct-route count")

    def refuses(pack, what):
        try:
            V.validate_branchpack(base, pack)
        except SystemExit as exc:
            check("BRANCH GUARD" in str(exc), f"refuses {what}")
            return
        check(False, f"refuses {what}")

    refuses({"A": good[:700]}, "a branch that is not 720 turns")
    over = [dict(t) for t in good]
    over[300] = {"farmer": ["PASS"], "hands": [],
                 "market": [["SELL", "WHEAT", 1]] * 11}
    refuses({"A": over}, "a turn queueing more than 10 market orders")
    own_prefix = [dict(t) for t in good]
    own_prefix[0] = {"farmer": ["DIG"], "hands": [], "market": []}
    refuses({"A": own_prefix}, "a branch that does not share the base prefix")
    refuses({"A": [dict(t) for t in base]},
            "a pack whose every branch IS the base (cannot dispatch)")
    refuses({}, "an empty pack")


# ------------------------------------------------------------------ runtime

def newest_agent():
    import re
    files = glob.glob(os.path.join(ROOT, "agents", "v*_bandit.py"))
    if not files:
        return None

    def ver(p):
        m = re.match(r"v(\d+)(?:\.(\d+))?", os.path.basename(p))
        return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)

    return max(files, key=ver)


def load(path):
    ns = {}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    return ns


def obs(step, shops, seat=0, inv=None):
    """The few fields the dispatch block reads, in engine shape."""
    return {"step": step, "player": seat, "day": step // 24,
            "farms": [{"money": 3000, "hands": [], "tiles": [],
                       "farmer": [0, 0]} for _ in range(2)],
            "market": {"prices": {}, "inventory": dict(inv or {})},
            "town": {"unlocked_shops": list(shops)},
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}


def test_runtime(path):
    """The SHOP-LABEL dispatch path (`_BRANCH_ON == "shop"`).

    Skips loudly on a drain-configured agent rather than reporting failures
    for a mode it does not implement: these observations carry an empty
    market inventory, so a drain dispatcher correctly never fires, and every
    check here would otherwise pass VACUOUSLY or fail spuriously. The drain
    mode has its own suite below — between them every shipped mode is
    covered, and a NEW mode with no suite shows up as a loud skip, not as
    silent green.
    """
    ns = load(path)
    mode = ns.get("_BRANCH_ON", "shop")
    if mode != "shop":
        print(f"runtime dispatch ({os.path.relpath(path, ROOT)}): "
              f"_BRANCH_ON={mode!r} -- skipped (see the drain suite)")
        return
    print(f"runtime dispatch ({os.path.relpath(path, ROOT)})")
    route = ns["_ROUTE"]
    check(len(route) == 720, "base route is 720 turns")

    branch = [copy.deepcopy(t) for t in route]
    branch[BRANCH_STEP] = {"farmer": ["DIG"], "hands": [], "market": []}
    ns["_BRANCHES"] = {"YARN_STORE": branch}
    ns["_STATE"] = {0: {}, 1: {}}

    ns["agent"](obs(0, []))
    ns["agent"](obs(BRANCH_STEP, ["YARN_STORE"]))
    st = ns["_STATE"][0]
    check(st.get("route") is branch,
          "dispatch stores the picked branch in the PER-EPISODE state")
    check(ns["_ROUTE"] is route and ns["_ROUTE"] == route,
          "the module-global _ROUTE is NOT swapped (no cross-episode leak)")
    check(not ns["_STATE"][1].get("route"),
          "the other seat's state is untouched")

    # A new episode in the same process: step 0 must clear the branch.
    ns["agent"](obs(0, []))
    check("route" not in ns["_STATE"][0],
          "step 0 of the next episode clears the dispatched route")

    # ... and it must clear even from a poisoned state (serve reuse can hand
    # us a step-0 observation with the previous episode's dict still live).
    ns["_STATE"][0] = {"last": 700, "route": branch, "branch_done": True}
    ns["agent"](obs(0, []))
    check("route" not in ns["_STATE"][0] and
          "branch_done" not in ns["_STATE"][0],
          "a stale end-of-episode state is discarded at step 0")

    # An unmapped shop must leave the base in place, not crash or half-switch.
    ns["_STATE"] = {0: {}, 1: {}}
    ns["agent"](obs(0, []))
    ns["agent"](obs(BRANCH_STEP, ["PET_CAFE"]))
    check(ns["_STATE"][0].get("branch_done") is True
          and "route" not in ns["_STATE"][0],
          "an unmapped first shop decides once and keeps the base")

    # The decision is taken once: a later shop unlock must not re-dispatch.
    ns["_STATE"] = {0: {}, 1: {}}
    ns["agent"](obs(0, []))
    ns["agent"](obs(BRANCH_STEP, ["PET_CAFE"]))
    ns["agent"](obs(144, ["PET_CAFE", "YARN_STORE"]))
    check("route" not in ns["_STATE"][0],
          "a second unlock does not re-dispatch mid-game")

    # With no branchpack the agent must behave exactly as before.
    ns["_BRANCHES"] = {}
    ns["_STATE"] = {0: {}, 1: {}}
    ns["agent"](obs(0, []))
    ns["agent"](obs(BRANCH_STEP, ["YARN_STORE"]))
    check("branch_done" not in ns["_STATE"][0],
          "an empty branchpack costs nothing at runtime")



DRAIN_STEP = 145          # day 6 h1: the demand tick the drain rule reads


def test_runtime_drain(path):
    """The DRAIN dispatch path — the one v44.0 actually ships.

    `test_runtime` above exercises `_BRANCH_ON == "shop"`. A drain-configured
    agent never fires on those observations (their market inventory is empty),
    so on 2026-09-04 every runtime check passed VACUOUSLY against v44.0 while
    two failed outright — the leak, isolation and decide-once properties were
    untested for the mode we ship. This covers them.

    Baseline drain is exactly 1 unit/turn (the town centre); each unlocked shop
    listing a product adds 1 more, so >= 2 means demand for it.
    """
    ns = load(path)
    if ns.get("_BRANCH_ON") != "drain":
        print(f"runtime drain ({os.path.relpath(path, ROOT)}): "
              f"_BRANCH_ON={ns.get('_BRANCH_ON')!r} -- skipped")
        return
    print(f"runtime drain dispatch ({os.path.relpath(path, ROOT)})")
    route = ns["_ROUTE"]
    branch = [copy.deepcopy(t) for t in route]
    branch[DRAIN_STEP] = {"farmer": ["DIG"], "hands": [], "market": []}
    ns["_BRANCHES"] = {"SHEEP": branch}

    def run(dw, dm, seat=0):
        """Two turns straddling the tick, with a chosen wool/milk drain."""
        pre = {"WOOL": 10000, "MILK": 10000}
        post = {"WOOL": 10000 - dw, "MILK": 10000 - dm}
        ns["agent"](obs(0, [], seat, pre))
        ns["agent"](obs(DRAIN_STEP - 1, [], seat, pre))
        ns["agent"](obs(DRAIN_STEP, [], seat, post))

    ns["_STATE"] = {0: {}, 1: {}}
    run(dw=3, dm=1)
    check(ns["_STATE"][0].get("route") is branch,
          "drain >=2 on WOOL dispatches to SHEEP in PER-EPISODE state")
    check(ns["_ROUTE"] is route and ns["_ROUTE"] == route,
          "drain: module-global _ROUTE is NOT swapped (no cross-episode leak)")
    check(not ns["_STATE"][1].get("route"),
          "drain: the other seat's state is untouched")

    # A baseline drain (1/turn) is NOT demand: keep the base, decide once.
    ns["_STATE"] = {0: {}, 1: {}}
    run(dw=1, dm=1)
    check(ns["_STATE"][0].get("branch_done") is True
          and "route" not in ns["_STATE"][0],
          "drain: baseline 1/turn decides once and keeps the base")

    # An unmapped class (COW here) must keep the base, not half-switch.
    ns["_STATE"] = {0: {}, 1: {}}
    run(dw=1, dm=3)
    check(ns["_STATE"][0].get("branch_done") is True
          and "route" not in ns["_STATE"][0],
          "drain: an unmapped class keeps the base")

    # Step 0 of the next episode clears a dispatched branch...
    ns["_STATE"] = {0: {}, 1: {}}
    run(dw=3, dm=1)
    ns["agent"](obs(0, [], 0, {"WOOL": 10000, "MILK": 10000}))
    check("route" not in ns["_STATE"][0] and "_pre" not in ns["_STATE"][0],
          "drain: step 0 clears the dispatched route and the pre-tick reading")

    # ...and clears from a poisoned end-of-episode state (serve reuse).
    ns["_STATE"][0] = {"last": 700, "route": branch, "branch_done": True,
                       "_pre": [1, 1]}
    ns["agent"](obs(0, [], 0, {"WOOL": 10000, "MILK": 10000}))
    check(all(k not in ns["_STATE"][0]
              for k in ("route", "branch_done", "_pre")),
          "drain: a stale end-of-episode state is discarded at step 0")

    # Reading only AFTER the tick (no pre-sample) must not dispatch on noise.
    ns["_STATE"] = {0: {}, 1: {}}
    ns["agent"](obs(0, [], 0, {"WOOL": 10000, "MILK": 10000}))
    ns["agent"](obs(DRAIN_STEP, [], 0, {"WOOL": 9990, "MILK": 10000}))
    check("route" not in ns["_STATE"][0],
          "drain: a missing pre-tick sample never dispatches")

    # With no branchpack the drain path must cost nothing.
    ns["_BRANCHES"] = {}
    ns["_STATE"] = {0: {}, 1: {}}
    run(dw=3, dm=1)
    check("route" not in ns["_STATE"][0],
          "drain: an empty branchpack costs nothing at runtime")

def main():
    test_validator()
    path = newest_agent()
    if path is None:
        print("  skip  no agents/v*_bandit.py to exercise")
    else:
        test_runtime(path)
        test_runtime_drain(path)
    print()
    if FAILED:
        print(f"FAILED {len(FAILED)}:")
        for m in FAILED:
            print("  -", m)
        return 1
    print("test_branch_dispatch: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
