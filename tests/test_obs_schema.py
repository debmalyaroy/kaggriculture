"""Every observation key an agent reads must EXIST in the real engine.

The bug this exists to prevent, found 2026-08-13: `_price_hold` read
`farms[me]["coins"]` while the engine's key is `money`, so the value was
always 0.0, the cash-floor guard always tripped, and the whole layer was dead
code -- silently, because `_get(..., default)` cannot tell a missing key from
a legitimately zero one. Illegal ops are silent no-ops in this engine and
missing keys are silent zeros; both make bugs look like laziness.

This walks the AST of every shipped agent, collects the literal key each
`_get(<known-container>, "<key>", ...)` / `<container>.get("<key>")` reads,
and checks it against the schema observed in a real replay.

    python tests/test_obs_schema.py
    python tests/test_obs_schema.py --replay path/to/episode.json
"""
from kaggriculture.paths import ROOT
import argparse
import ast
import glob
import json
import os
import sys


# Containers we can check, and how to reach a sample of each in a replay.
# Anything not listed here is skipped rather than guessed at.
CONTAINERS = {
    "obs": lambda o: o,
    "farm": lambda o: (o.get("farms") or [{}])[0],
    "farms_elem": lambda o: (o.get("farms") or [{}])[0],
    "private": lambda o: o.get("private") or {},
    "market": lambda o: o.get("market") or {},
    "prices": lambda o: (o.get("market") or {}).get("prices") or {},
    "inventory": lambda o: (o.get("market") or {}).get("inventory") or {},
    "shed": lambda o: (o.get("private") or {}).get("shed") or {},
    "town": lambda o: o.get("town") or {},
}

# `tiles` is a 10x10 NESTED grid whose cells are of two kinds -- PLANT and
# PASTURE -- with disjoint key sets. Sampling one cell yields one kind's keys
# and reports the other kind's legitimate reads as bugs, so the tile schema is
# the UNION over every cell of every sampled step.
TILE_KINDS = ("PLANT", "PASTURE")

# Variable names in agent source that hold each container. Derived by reading
# the agents: these names are assigned from the matching obs path.
VARMAP = {
    "obs": "obs", "farm": "farm", "farms_elem": "farm",
    "private": "private", "market": "market",
    "market_obs": "market", "prices": "prices", "inventory": "inventory",
    "shed": "shed", "town": "town", "tile": "tile", "t": "tile",
}

# Keys an agent legitimately keeps in its OWN state dicts, not in obs.
SKIP_CONTAINERS = {"state", "spec", "rec", "slot", "ledger", "held", "action"}

# Narrow, documented waivers. These are agents ALREADY UPLOADED to Kaggle:
# rewriting the local file to fix the read would make it disagree with the
# artefact that actually ran on the ladder, which is worse than the bug. The
# read sits in `_price_hold`, which is dead code in these builds (`_HOLD` is
# False), so it changed no behaviour. The generator is fixed, so every build
# from now on is clean -- and this test fails on any NEW instance.
WAIVED = {
    ("v24.1_bandit.py", "coins"):
        "shipped 2026-08-13 as submission 55477892; dead code (_HOLD=False)",
    ("v24.1g_bandit.py", "coins"):
        "GRU sibling of the shipped v24.1 build; dead code (_HOLD=False)",
}


def replay_schema(path):
    data = json.load(open(path, encoding="utf-8"))
    steps = data["steps"]
    seen = {name: set() for name in CONTAINERS}
    seen["tile"] = set()
    for i in (1, len(steps) // 3, len(steps) // 2, len(steps) - 2):
        if not (0 <= i < len(steps)) or not steps[i]:
            continue
        obs = steps[i][0].get("observation") or {}
        if not obs.get("farms"):
            continue
        for name, reach in CONTAINERS.items():
            try:
                d = reach(obs)
            except Exception:                                   # noqa: BLE001
                continue
            if isinstance(d, dict):
                seen[name] |= set(d.keys())
        for farm in (obs.get("farms") or []):
            for row in (farm.get("tiles") or []):
                for cell in (row if isinstance(row, list) else [row]):
                    if isinstance(cell, dict):
                        seen["tile"] |= set(cell.keys())
    return seen


def reads_of(path):
    """[(container_var, key, lineno)] for every literal .get / _get read."""
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [], f"SyntaxError: {exc}"
    out = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):                             # noqa: N802
            fn = node.func
            # _get(container, "key", default)
            if (isinstance(fn, ast.Name) and fn.id == "_get"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                cont = node.args[0]
                name = (cont.id if isinstance(cont, ast.Name) else
                        _subject(cont))
                if name:
                    out.append((name, node.args[1].value, node.lineno))
            # container.get("key")
            if (isinstance(fn, ast.Attribute) and fn.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                name = _subject(fn.value)
                if name:
                    out.append((name, node.args[0].value, node.lineno))
            self.generic_visit(node)

    def _subject(n):
        # `farms[me] if me < len(farms) else {}` is how every agent reaches
        # its own farm, so an IfExp must resolve to its body and a subscript
        # of `farms` must resolve to the FARM schema, not the list's. Without
        # both, this checker misses the very bug it was written for.
        if isinstance(n, ast.IfExp):
            return _subject(n.body)
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Subscript):
            # `obs["farms"][seat]` is a FARM, not an obs -- without this hop
            # the chain resolves to `obs` and every farm key reads as a bug.
            inner = n.value
            if (isinstance(inner, ast.Subscript)
                    and isinstance(inner.slice, ast.Constant)
                    and inner.slice.value == "farms"):
                return "farms_elem"
            base = _subject(inner)
            return "farms_elem" if base == "farms" else base
        return None

    V().visit(tree)
    return out, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay")
    ap.add_argument("--agents", default=os.path.join(ROOT, "agents", "*.py"))
    args = ap.parse_args()

    rp = args.replay
    if not rp:
        cands = sorted(glob.glob(os.path.join(ROOT, ".local", "lossreplays",
                                              "*.json")))
        cands = [c for c in cands if os.path.getsize(c) > 1_000_000]
        if not cands:
            print("SKIP: no staged replay to read the schema from "
                  "(pass --replay)")
            return 0
        rp = cands[0]
    schema = replay_schema(rp)
    print(f"schema from {os.path.relpath(rp, ROOT)}")
    for name in ("obs", "farm", "private", "market", "tile"):
        print(f"  {name:<9} {sorted(schema[name])}")

    bad = 0
    checked = 0
    for path in sorted(glob.glob(args.agents)):
        reads, err = reads_of(path)
        if err:
            print(f"\n{os.path.basename(path)}: {err}")
            bad += 1
            continue
        hits = []
        for var, key, line in reads:
            cont = VARMAP.get(var)
            if not cont or var in SKIP_CONTAINERS:
                continue
            known = schema.get(cont) or set()
            if not known:
                continue
            checked += 1
            if key not in known:
                hits.append((line, var, key, sorted(known)))
        waived = [h for h in hits
                  if (os.path.basename(path), h[2]) in WAIVED]
        hits = [h for h in hits if h not in waived]
        for line, var, key, _known in waived:
            print(f"{os.path.basename(path)}: line {line} {var}[{key!r}] "
                  f"-- WAIVED: {WAIVED[(os.path.basename(path), key)]}")
        if hits:
            print(f"\n{os.path.basename(path)}: "
                  f"{len(hits)} key(s) NOT in the engine schema")
            for line, var, key, known in hits:
                print(f"  line {line}: {var}[{key!r}] -- engine has {known}")
            bad += len(hits)

    print(f"\nchecked {checked} key reads across "
          f"{len(glob.glob(args.agents))} agents")
    if bad:
        print(f"FAIL: {bad} bad key read(s) -- each one is a silent zero")
        return 1
    print("PASS: every checked key exists in the engine observation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
