"""Realized-world map: which shops a (seed, opponent, seat) cell ACTUALLY
unlocks under a given prefix.

The 2026-09-07 finding: shop unlocks draw from the same per-day RNG as
weeds, so the world is a function of the seed AND both action streams AND
seat order — data/worlds/seed_bank.json (PASS-only drive) labels a world
real play does not visit (measured: seed 65 is BRUNCH_SPOT|BAKERY under
PASS, PET_CAFE|BAKERY vs one donor, FARMERS_MARKET|BRUNCH_SPOT in
mirror). Any per-world evolution must group by THIS map, not the bank.

The first shop is realized by t~73 and the second by t~145, so a map
built from a fixed prefix stays valid for mutants that only touch steps
past the relevant split (74 for day-3 keying, 146 for day-6 keying).
"""
from __future__ import annotations

import json
import os
import sys



def tape_str(actions, SM):
    return chr(31).join(SM.action_to_line(
        actions[i] if i < len(actions) else None) for i in range(719))


def build_map(our_actions, panel_paths, seeds, progress=True):
    """{(opp_idx, seed, seat): "S1|S2"} by playing our tape vs each panel
    tape in both seats — one GENGAME per cell (~20 ms)."""
    import kaggriculture.engine.serve_match as SM
    ours = tape_str(our_actions, SM)
    opp_tapes = []
    for p in panel_paths:
        lines = open(p, encoding="utf-8").read().splitlines()
        if lines and lines[0].startswith("SEED"):
            lines = lines[1:]           # write_tape header
        opp_tapes.append(chr(31).join(
            (lines[i] if i < len(lines) else lines[-1])
            for i in range(719)))
    out = {}
    srv = SM.Serve()
    try:
        total = len(panel_paths) * len(seeds) * 2
        done = 0
        for oi, ot in enumerate(opp_tapes):
            for s in seeds:
                for seat in (0, 1):
                    a, b = (ours, ot) if seat == 0 else (ot, ours)
                    js = srv.cmd("GENGAME " + str(s) + chr(30) + a
                                 + chr(30) + b)
                    days = js.get("days") or []
                    if not days or "error" in js:
                        continue
                    shops = ((days[0].get("town") or {})
                             .get("unlocked_shops") or [])
                    if len(shops) >= 2:
                        out[(oi, s, seat)] = f"{shops[0]}|{shops[1]}"
                    done += 1
                    if progress and done % 2000 == 0:
                        print(f"  realized map {done}/{total}", flush=True)
    finally:
        srv.close()
    return out
