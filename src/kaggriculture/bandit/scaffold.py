"""Bandit-owned base scaffold + base-tape writer.

Separated from trackp (2026-09-19): the bandit's base economy and the tape
format its build consumes are the BANDIT's own concern. `our_route()` loads the
base economy (the v45 bandit's recorded `_ROUTE`, the scaffold the checkpoint
branches were evolved against); `write_base_tape()` emits it in the raw
`.tape` format `build_rust_bandit.parse_tape` reads. Nothing here imports
`kaggriculture.trackp`.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import base64
import json
import os
import re
import zlib

# The base economy lives in a shipped bandit agent as a compressed _ROUTE blob.
BASE_AGENT = os.path.join(ROOT, "agents", "v45.0_bandit.py")


def our_route(agent_path: str = BASE_AGENT):
    """The base opening/economy as a list of {farmer,hands,market} rows (720)."""
    src = open(agent_path, encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    if not m:
        raise ValueError(f"no _ROUTE blob in {agent_path}")
    return json.loads(zlib.decompress(base64.b85decode(m.group(1))).decode("utf-8"))


def _tape_line(row):
    f = " ".join(str(t) for t in (row.get("farmer") or ["PASS"]))
    h = ";".join(" ".join(str(t) for t in op) for op in (row.get("hands") or []) if op)
    m = ";".join(" ".join(str(t) for t in op) for op in (row.get("market") or []) if op)
    return f"{f}\t{h}\t{m}"


def write_base_tape(path: str, rows=None):
    """Write the base economy to a raw `.tape` (farmer<TAB>hands<TAB>market)."""
    rows = rows if rows is not None else our_route()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows[:719]:
            fh.write(_tape_line(r) + "\n")
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="write the bandit base tape from the base economy")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "bandit", "base.tape"))
    ap.add_argument("--agent", default=BASE_AGENT)
    a = ap.parse_args()
    p = write_base_tape(a.out, our_route(a.agent))
    print(f"wrote {p} ({os.path.getsize(p):,} bytes, {len(our_route(a.agent))} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
