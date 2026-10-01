"""P1 -- build the single-file planner agent from the template.

Splices between sentinels:
  PARAMS  -- dict overrides (P4.2 search output)
  PRIOR   -- opponent family dump prior [(step, product, qty), ...]
  L1      -- exported macro-policy MLP weights (P3.5) or None

The emitted file is the template itself with those blocks replaced -- the
template IS the agent; the builder never edits logic. A build stamp records
provenance. Usage:
  python src/trackp/build_agent.py --out agents/planner_v0.py
      [--params models/trackp/params.json] [--prior] [--l1 weights.json]
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..")))

from kaggriculture.trackp import common  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "planner_template.py")


def _splice(text: str, begin: str, end: str, payload: str) -> str:
    b = text.index(begin) + len(begin)
    e = text.index(end)
    return text[:b] + "\n" + payload + "\n" + text[e:]


def load_family_prior(top_k: int = 1) -> list:
    """Fold models/lab/family_dumps_v2.json (DATA file) into a flat prior.

    Uses the most-supported class(es); entries become (step, product, qty)
    weighted by support. Missing file -> empty prior (the agent degrades to
    the observed-rate-only projection).
    """
    path = os.path.join(common.ROOT, "models", "lab", "family_dumps_v2.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    members = d.get("members", {})
    classes = d.get("classes", {})
    def _support(cid):
        v = members.get(cid, 0) if isinstance(members, dict) else 0
        return v if isinstance(v, (int, float)) else len(v)

    if isinstance(classes, dict):
        ranked = sorted(classes.items(), key=lambda kv: -_support(kv[0]))
    else:
        ranked = list(enumerate(classes))
    prior = []
    for cid, rows in ranked[:top_k]:
        for r in rows:
            t, product, qty, support = r[0], r[1], r[2], r[3]
            w = float(qty) * float(support)
            if w >= 1.0:
                prior.append((int(t), str(product), round(w, 1)))
    prior.sort()
    return prior


def build(out: str, params_path: str = "", use_prior: bool = True,
          l1_path: str = "", stamp_extra: str = "") -> str:
    with open(TEMPLATE, encoding="utf-8") as fh:
        text = fh.read()

    if params_path:
        with open(params_path, encoding="utf-8") as fh:
            overrides = json.load(fh)
        # Merge onto template defaults: exec the template's PARAMS block.
        import re
        m = re.search(r"# --- PARAMS BEGIN ---\n(.*?)# --- PARAMS END ---",
                      text, re.S)
        ns = {}
        exec(m.group(1), {}, ns)  # noqa: S102 -- our own template
        merged = ns["PARAMS"]
        unknown = [k for k in overrides if k not in merged]
        if unknown:
            raise SystemExit(f"unknown PARAMS keys: {unknown}")
        merged.update(overrides)
        payload = "PARAMS = " + json.dumps(merged, indent=1)
        text = _splice(text, "# --- PARAMS BEGIN ---", "# --- PARAMS END ---",
                       payload)

    prior = load_family_prior() if use_prior else []
    text = _splice(text, "# --- PRIOR BEGIN ---", "# --- PRIOR END ---",
                   "FAMILY_PRIOR = " + json.dumps(prior))

    if l1_path:
        with open(l1_path, encoding="utf-8") as fh:
            w = json.load(fh)
        text = _splice(text, "# --- L1 BEGIN ---", "# --- L1 END ---",
                       "L1_WEIGHTS = " + json.dumps(w))

    # Engine selection follows models/engine_version.json (the swap script's
    # state): agents built after the 1.32.7 ladder flip project prices with
    # the hinge curve automatically.
    is_1327 = common._ver_tuple(common.engine_version()) >= (1, 32, 7)
    text = _splice(text, "# --- ENGINE BEGIN ---", "# --- ENGINE END ---",
                   f"ENGINE_1327 = {is_1327}")

    stamp = (f"# TRACKP BUILD: template=planner_template.py "
             f"prior_rows={len(prior)} l1={'yes' if l1_path else 'no'} "
             f"params={'tuned' if params_path else 'default'} {stamp_extra}\n")
    text = stamp + text
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    # contract self-check: compiles, imports only math
    import ast
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names]
                     if isinstance(node, ast.Import) else [node.module])
            for n in names:
                if n != "math":
                    raise SystemExit(f"contract violation: import {n}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default="")
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--l1", default="")
    a = ap.parse_args()
    p = build(a.out, a.params, not a.no_prior, a.l1)
    print("built", p)
