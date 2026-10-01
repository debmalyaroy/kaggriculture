"""Do our agents conform to the competition's own AGENTS.md and README.md?

    python -m kaggriculture.engine.conformance                    # audit every agent
    python -m kaggriculture.engine.conformance --agent agents/v14_market.py
    python -m kaggriculture.engine.conformance --refresh           # re-download the docs first
    python -m kaggriculture.engine.conformance --json

The competition ships two documents as *competition data*
(`kaggle competitions files kaggriculture`): `AGENTS.md`, the agent contract,
and `README.md`, the full rules. They are the authority, and they get revised --
the CARE bonus text changed between the version this project first read and the
2026-08-04 revision, which is exactly the kind of drift that leaves an agent
optimising against a rule nobody enforces any more.

Two different things are checked, and conflating them would be misleading:

**Contract** -- rules an agent can *violate*. A violation is silent: the
interpreter no-ops an illegal op rather than raising, so a broken agent looks
lazy instead of broken. These are hard pass/fail.

**Coverage** -- documented observation fields an agent never reads. Ignoring a
field is not a violation; it is a strategy gap with a citation, and worth
seeing next to the contract results rather than buried in a notebook.

Config defaults are read from the live interpreter and compared against the
table in README.md, so a Kaggle image running different constants shows up here
as well as in `src/kaggriculture/engine/engine_check.py`.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

DOCS = os.path.join(ROOT, ".local", "compdocs")

# Documented configuration defaults, README.md "Configuration Defaults".
CONFIG_DEFAULTS = {
    "episodeSteps": 720, "boardSize": 10, "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24, "shedCapacity": 100,
    "weedSpawnChance": 0.005, "townShopUnlockInterval": 3,
    "townShopSellInterval": 4, "townCenterSellInterval": 12,
}

# Ops AGENTS.md lists. Anything else is a silent no-op on the ladder.
UNIT_OPS = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
            "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER"}
MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}

# Observation fields AGENTS.md documents, with the citation that makes each one
# worth reading. Absence is a gap, not a fault.
COVERAGE = [
    ("hires_today", "farms[me]", "drives the next HIRE cost (fib resets daily)"),
    ("unlocked_quadrants", "farms[me]", "which quadrants can be acted in"),
    ("consecutive_unwatered", "tile", "two missed refreshes turn a plant into a weed"),
    ("max_lifespan_step", "tile", "decay starts one day past it; yield_units falls to 0, then WEED"),
    ("fertilized_until_day", "tile", "FERTILIZE doubles the watering bonus for 3 days"),
    ("pending_care_bonus", "tile", "banked CARE, paid in full on the next production day"),
    ("consecutive_unfed", "tile", "two missed feeds and the animal escapes, unrecoverably"),
    ("yield_units", "tile", "harvestable units standing on the tile"),
    ("unlocked_shops", "town", "who is consuming, and therefore what the price will do"),
]

# "Shed-adjacent" is the four centre tiles, AGENTS.md line 63.
SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}


def fetch_docs(force=False):
    """Download AGENTS.md / README.md from the competition if we lack them."""
    os.makedirs(DOCS, exist_ok=True)
    got = {}
    for name in ("AGENTS.md", "README.md"):
        path = os.path.join(DOCS, name)
        if os.path.exists(path) and not force:
            got[name] = path
            continue
        import kaggriculture.data.episodes as E
        try:
            subprocess.run(E.kaggle_cmd() + ["competitions", "download", "-c",
                                             "kaggriculture", "-f", name,
                                             "-p", DOCS, "--force"],
                           capture_output=True, timeout=180)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! could not fetch {name}: {exc}")
        if os.path.exists(path):
            got[name] = path
    return got


def check_config():
    """The live interpreter against README.md's Configuration Defaults table."""
    from kaggle_environments import make
    cfg = make("kaggriculture", debug=False).configuration
    rows = []
    for key, want in sorted(CONFIG_DEFAULTS.items()):
        got = cfg.get(key)
        ok = (abs(float(got) - float(want)) < 1e-9
              if isinstance(want, float) and got is not None else got == want)
        rows.append({"key": key, "want": want, "got": got, "ok": bool(ok)})
    return rows


def check_agent(path, steps=240, seed=5):
    """Play the agent and audit every action against the documented contract."""
    from kaggle_environments import make
    import importlib.util

    name = os.path.basename(path)
    spec = importlib.util.spec_from_file_location("conf_" + name[:-3].replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    result = {"agent": name, "violations": [], "ok": True, "turns": 0}
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                                       # noqa: BLE001
        result["ok"] = False
        result["violations"].append(f"does not import: {exc}")
        return result
    if not hasattr(mod, "agent"):
        result["ok"] = False
        result["violations"].append("no agent() function -- AGENTS.md requires one")
        return result

    two_arg = mod.agent.__code__.co_argcount > 1
    seen = {"orders_max": 0, "shed_ops_off_centre": 0, "drop_place": 0}

    def audited(obs, cfg=None):
        act = mod.agent(obs, cfg) if two_arg else mod.agent(obs)
        result["turns"] += 1
        me = int(obs["player"])
        farm = obs["farms"][me]

        if not isinstance(act, dict):
            result["violations"].append(f"turn {result['turns']}: not a dict")
            return {"farmer": ["PASS"], "hands": [], "market": []}
        extra = set(act) - {"farmer", "hands", "market"}
        if extra:
            result["violations"].append(f"unknown action keys: {sorted(extra)}")

        farmer = act.get("farmer") or ["PASS"]
        if not (isinstance(farmer, list) and farmer and farmer[0] in UNIT_OPS):
            result["violations"].append(f"illegal farmer op: {farmer}")
        hands = act.get("hands") or []
        want = len(farm.get("hands") or [])
        if len(hands) != want:
            result["violations"].append(
                f"hands misaligned: {len(hands)} given, {want} hired "
                f"-- AGENTS.md requires one op per hand, in hands order")
        for op in hands:
            if not (isinstance(op, list) and op and op[0] in UNIT_OPS):
                result["violations"].append(f"illegal hand op: {op}")

        orders = act.get("market") or []
        seen["orders_max"] = max(seen["orders_max"], len(orders))
        if len(orders) > CONFIG_DEFAULTS["maxMarketOrdersPerTurn"]:
            result["violations"].append(
                f"{len(orders)} market orders -- over maxMarketOrdersPerTurn, "
                f"the extras are silently dropped")
        for o in orders:
            if not (isinstance(o, list) and o and o[0] in MARKET_OPS):
                result["violations"].append(f"illegal market op: {o}")
            elif o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"):
                if len(o) < 3:
                    result["violations"].append(f"{o[0]} without a quantity: {o}")
                else:
                    try:
                        if int(o[2]) <= 0:
                            result["violations"].append(f"non-positive quantity: {o}")
                    except (TypeError, ValueError):
                        result["violations"].append(f"non-numeric quantity: {o}")

        # DROP/PLACE-into-shed only works from the four centre tiles. Issuing it
        # elsewhere is legal but a wasted unit-turn, so it is counted, not failed.
        positions = [tuple(farm.get("farmer") or (0, 0))] + \
                    [tuple(p) for p in (farm.get("hands") or [])]
        for i, op in enumerate([farmer] + list(hands)):
            if isinstance(op, list) and op and op[0] == "DROP":
                seen["drop_place"] += 1
                if i < len(positions) and positions[i] not in SHED_TILES:
                    seen["shed_ops_off_centre"] += 1
        return act

    env = make("kaggriculture", debug=False, configuration={
        "episodeSteps": steps, "seed": seed, "actTimeout": 60, "runTimeout": 100000})
    try:
        env.run([audited, "starter"])
    except Exception as exc:                                       # noqa: BLE001
        result["violations"].append(f"episode raised: {exc}")

    status = env.steps[-1][0]["status"]
    if status not in ("DONE", "ACTIVE"):
        result["violations"].append(f"episode ended {status}")
    result["status"] = status
    result["bank"] = float(env.steps[-1][0]["reward"] or 0)
    result["max_orders"] = seen["orders_max"]
    result["drop_ops"] = seen["drop_place"]
    result["drop_off_centre"] = seen["shed_ops_off_centre"]
    # De-duplicate: one structural mistake repeats on every one of 240 turns.
    uniq, kinds = [], set()
    for v in result["violations"]:
        key = re.sub(r"\d+", "#", v)
        if key not in kinds:
            kinds.add(key)
            uniq.append(v)
    result["violations"] = uniq
    result["ok"] = not uniq
    return result


def check_coverage(path):
    """Which documented observation fields does this agent actually read?"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    rows = []
    for field, where, why in COVERAGE:
        rows.append({"field": field, "where": where, "why": why,
                     "used": field in src})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download AGENTS.md / README.md first")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    docs = fetch_docs(force=args.refresh)
    report = {"docs": {k: os.path.getsize(v) for k, v in docs.items()}}

    report["config"] = check_config()
    if args.agent:
        paths = [os.path.join(ROOT, args.agent)]
    else:
        paths = [p for p in sorted(glob.glob(os.path.join(ROOT, "agents", "v*.py")))]
        build = os.path.join(ROOT, "build", "main.py")
        if os.path.exists(build):
            paths.append(build)

    report["agents"] = [check_agent(p, steps=args.steps) for p in paths]
    report["coverage"] = {os.path.basename(p): check_coverage(p) for p in paths}

    if args.json:
        print(json.dumps(report, indent=1, default=str))
        return 0 if all(a["ok"] for a in report["agents"]) else 1

    print("Competition docs (kaggle competitions files kaggriculture)")
    for name, size in report["docs"].items():
        print(f"  {name:<12}{size:>8,} bytes")
    if not report["docs"]:
        print("  ! neither doc could be fetched -- checks below use the "
              "constants recorded in this file")

    print("\nConfiguration against README.md's defaults table")
    bad = [r for r in report["config"] if not r["ok"]]
    for r in report["config"]:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"  {mark} {r['key']:<24} {str(r['got']):<10}"
              + ("" if r["ok"] else f"(documented {r['want']})"))
    if bad:
        print("  ! this interpreter does not match the published defaults")

    print(f"\nAgent contract ({args.steps} turns vs starter, both channels audited)")
    for a in report["agents"]:
        mark = "ok  " if a["ok"] else "FAIL"
        note = (f"max {a.get('max_orders', 0)} order(s)/turn, "
                f"{a.get('drop_ops', 0)} DROP"
                + (f", {a['drop_off_centre']} off the shed tiles"
                   if a.get("drop_off_centre") else ""))
        print(f"  {mark} {a['agent']:<40}{note}")
        for v in a["violations"]:
            print(f"       - {v}")

    print("\nDocumented observation fields, and whether we read them")
    fields = COVERAGE
    names = [os.path.basename(p) for p in paths]
    head = "  {:<24}".format("field") + "".join(f"{n[:10]:>12}" for n in names[:5])
    print(head)
    for i, (field, _where, why) in enumerate(fields):
        cells = ""
        for n in names[:5]:
            used = next((r["used"] for r in report["coverage"][n]
                         if r["field"] == field), False)
            cells += f"{'yes' if used else '-':>12}"
        print(f"  {field:<24}{cells}")
        if not any(next((r['used'] for r in report['coverage'][n]
                         if r['field'] == field), False) for n in names):
            print(f"      unread everywhere -- {why}")

    failures = [a for a in report["agents"] if not a["ok"]]
    print(f"\n{len(report['agents']) - len(failures)}/{len(report['agents'])} "
          f"agent(s) conform to the documented contract")
    return 1 if failures or bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
