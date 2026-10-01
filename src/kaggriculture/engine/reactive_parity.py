"""Reactive-agent serve<->official parity gate (Workstream A2).

`models/serve_equiv.json` historically certified only TAPE-SHELL agents
(v43.0_bandit, v42.1_trackp, raw_*), which by construction IGNORE the
observation and therefore CANNOT expose a serve invocation-layer bug. That is
the blind spot that let the step-719 mis-rank (S5) sit undetected while the
gate read green. This module certifies GENUINELY REACTIVE agents: each
candidate plays a fixed reactive opponent on BOTH the Rust serve substrate and
the official vendored engine, on BOTH seats, over several seeds; the final
banks must match EXACTLY. It writes `serve_equiv.json` with a `reactive_agents`
roster so `serve_allowed()` can require reactive coverage, not just tape shells.

    python -m kaggriculture.engine.reactive_parity --seeds 3            # default panel
    python -m kaggriculture.engine.reactive_parity --agents a.py b.py --opponent o.py

Vetting (D2.2): each candidate must load, run a full 720-step episode on serve
without crashing, and be DETERMINISTIC (two serve runs on the same seed agree)
-- a non-reproducible agent would produce false parity mismatches unrelated to
serve fidelity, so it is excluded from the gate and reported.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import json
import os
import sys

from kaggriculture.engine import serve_match as sm

SERVE_EQUIV = os.path.join(ROOT, "models", "serve_equiv.json")
ENGINE_VERSION = os.path.join(ROOT, "models", "engine_version.json")

# Default reactive panel: agents that READ obs and adapt (NOT tape shells).
# Diverse families so an obs bug in any consumed field surfaces somewhere.
DEFAULT_PANEL = [
    "agents/agent_vadapt_both_20260805_225549.py",
    "agents/agent_vadapt_risk_20260805_040840.py",
    "agents/agent_vadapt_both_20260805_032405.py",
    "agents/agent_vadapt_risk_20260805_032413.py",
    "agents/agent_v4_optimal_20260805_014340.py",
    "agents/v0_baseline.py",
    "agents/ml_ridge.py",
]
DEFAULT_OPPONENT = "agents/agent_vadapt_risk_20260805_040840.py"


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def vet(path, srv, seed=7):
    """Load + full-episode + determinism check. Returns (ok, reason)."""
    ap = _abs(path)
    if not os.path.exists(ap):
        return False, "missing file"
    try:
        b1 = sm.run_match(sm.load_agent(ap), sm.load_agent(ap), seed, srv)
        b2 = sm.run_match(sm.load_agent(ap), sm.load_agent(ap), seed, srv)
    except Exception as exc:  # noqa: BLE001 - report any load/run failure
        return False, f"{type(exc).__name__}: {exc}"
    if b1 != b2:
        return False, f"non-deterministic ({b1} != {b2})"
    return True, "ok"


def parity_rows(agent, opponent, seeds, seed0, srv):
    """serve vs official, both seats, per seed. Yields row dicts."""
    pa, po = _abs(agent), _abs(opponent)
    for k, seed in enumerate(seeds):
        for seat in (0, 1):
            # seat 0: candidate is player 0; seat 1: candidate is player 1.
            if seat == 0:
                serve = sm.run_match(sm.load_agent(pa), sm.load_agent(po), seed, srv)
                off = sm.official_banks(pa, po, seed)
            else:
                s = sm.run_match(sm.load_agent(po), sm.load_agent(pa), seed, srv)
                o = sm.official_banks(po, pa, seed)
                serve, off = (s[1], s[0]), (o[1], o[0])
            yield {
                "a": os.path.basename(agent), "b": os.path.basename(opponent),
                "seat": seat, "seed": seed,
                "serve": [serve[0], serve[1]], "official": [off[0], off[1]],
                "match": serve == off,
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", nargs="*", default=None,
                    help="reactive candidates (default: built-in panel)")
    ap.add_argument("--opponent", default=None,
                    help="fixed reactive opponent (default: built-in)")
    ap.add_argument("--seeds", type=int, default=2,
                    help="seeds per agent (each played on both seats)")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--out", default=SERVE_EQUIV)
    ap.add_argument("--dry-run", action="store_true",
                    help="do not write serve_equiv.json")
    args = ap.parse_args()

    panel = args.agents or DEFAULT_PANEL
    opponent = args.opponent or DEFAULT_OPPONENT
    seeds = [args.seed0 + i for i in range(max(1, args.seeds))]

    engine = "unknown"
    try:
        engine = json.load(open(ENGINE_VERSION, encoding="utf-8")).get("engine", "unknown")
    except (OSError, ValueError):
        pass

    srv = sm.Serve()
    rows, vetted, excluded = [], [], []
    try:
        # Vet the opponent too -- a bad opponent poisons every pairing.
        ok, why = vet(opponent, srv)
        if not ok:
            print(f"FATAL: opponent {opponent} failed vetting: {why}", flush=True)
            return 2
        for a in panel:
            if os.path.basename(a) == os.path.basename(opponent):
                continue
            ok, why = vet(a, srv)
            if not ok:
                excluded.append((a, why))
                print(f"EXCLUDE {os.path.basename(a)}: {why}", flush=True)
                continue
            vetted.append(a)
            for row in parity_rows(a, opponent, seeds, args.seed0, srv):
                rows.append(row)
                tag = "MATCH" if row["match"] else "MISMATCH"
                print(f"{tag}\t{row['a']}\tseat{row['seat']}\tseed{row['seed']}"
                      f"\tserve={row['serve']}\toff={row['official']}", flush=True)
    finally:
        srv.close()

    matches = sum(1 for r in rows if r["match"])
    mismatches = sum(1 for r in rows if not r["match"])
    reactive_agents = sorted({r["a"] for r in rows if r["match"]}
                             - {r["a"] for r in rows if not r["match"]})

    out = {
        "matches": matches, "mismatches": mismatches, "games": len(rows),
        "engine": engine,
        "when": dt.datetime.now().replace(microsecond=0).isoformat(),
        "written_by": "src/kaggriculture/engine/reactive_parity.py",
        "reactive_agents": reactive_agents,
        "reactive_covered": len(reactive_agents),
        "opponent": os.path.basename(opponent),
        "excluded": [{"agent": os.path.basename(a), "reason": w} for a, w in excluded],
        "rows": rows,
    }
    print(f"\nREACTIVE PARITY: {matches} match / {mismatches} mismatch over "
          f"{len(rows)} games; {len(reactive_agents)} clean reactive agents "
          f"({', '.join(os.path.basename(a) for a in vetted)})", flush=True)

    if not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {args.out}", flush=True)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
