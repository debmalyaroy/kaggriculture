"""P3.3 -- the league: mined-tape anchors, past selves, exploiters, PFSP.

Composition (fixed by the plan): 40% anchors / 30% past selves / 20% current
policy / 10% exploiters. Opponents are sampled by prioritized fictitious
self-play: weight ~ w*(1-w) + eps toward ~50% win-rate matchups.

Anchors are REAL ladder seats (single-seat tapes from 1.32.6 replays), the
non-drifting yardstick: checkpoint selection uses ANCHOR score only, never
self-play score.
"""
from __future__ import annotations

import glob
import json
import os
import random

import numpy as np

try:
    from . import common, trace_v2
except ImportError:  # script mode: python src/trackp/league.py
    import sys
    from kaggriculture.trackp import common, trace_v2

ANCHORS = os.path.join(common.DATA, "anchors")
SNAPSHOTS = os.path.join(common.MODELS, "snapshots")
EXPLOITERS = os.path.join(common.MODELS, "exploiters")
for _d in (ANCHORS, SNAPSHOTS, EXPLOITERS):
    os.makedirs(_d, exist_ok=True)

MANIFEST = os.path.join(ANCHORS, "anchors.json")


def build_anchors(n: int = 60, engine: str = None,
                  per_team_cap: int = 4) -> dict:
    """Pick the strongest, team-diverse ladder seats and write their tapes."""
    if engine is None:
        engine = common.engine_version()  # follow the ladder
    try:
        from .recapture import discover
    except ImportError:
        from kaggriculture.trackp.recapture import discover
    paths = discover()
    rows = []
    for f in glob.glob(os.path.join(common.TRACES, "*.npz")):
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except Exception:
            continue
        if meta.get("engine") != engine:
            continue
        eid = str(meta.get("episode"))
        if eid not in paths:
            continue
        banks = meta.get("banks") or [0, 0]
        teams = meta.get("teams") or ["?", "?"]
        for seat in (0, 1):
            rows.append((float(banks[seat]), eid, seat, teams[seat]))
    rows.sort(reverse=True)
    picked, per_team = [], {}
    for bank, eid, seat, team in rows:
        if per_team.get(team, 0) >= per_team_cap:
            continue
        picked.append((bank, eid, seat, team))
        per_team[team] = per_team.get(team, 0) + 1
        if len(picked) >= n:
            break
    if len(picked) < 8 and engine:
        # Transition window after an engine rebalance: almost no traces on
        # the new version yet. An empty anchor pool would break search/PPO/
        # validity outright, which is worse than slightly-stale anchors --
        # their ACTIONS still replay fine; only their recorded banks carry
        # old economics. Fall back to all engines and say so loudly.
        print(f"ANCHOR FALLBACK: only {len(picked)} anchors on engine "
              f"{engine}; rebuilding without the engine filter until the "
              f"new-engine corpus grows", flush=True)
        return build_anchors(n=n, engine="", per_team_cap=per_team_cap)
    manifest = []
    for bank, eid, seat, team in picked:
        rep = common.load_replay(paths[eid])
        tape = os.path.join(ANCHORS, f"{eid}_s{seat}.tape")
        common.replay_to_tape(rep, seat, tape)
        manifest.append({"tape": tape, "seed": common.replay_seed(rep),
                         "seat": seat, "bank": bank, "team": team,
                         "episode": eid})
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return {"anchors": len(manifest),
            "teams": len({m['team'] for m in manifest}),
            "bank_range": [manifest[-1]["bank"], manifest[0]["bank"]]
            if manifest else None}


def load_anchors() -> list:
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


class League:
    """Opponent pool with PFSP sampling and per-member running win rates."""

    MIX = {"anchor": 0.4, "self": 0.3, "current": 0.2, "exploiter": 0.1}

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.members = []
        for m in load_anchors():
            self.members.append({"kind": "anchor", "ref": m, "w": 0.5,
                                 "n": 0})
        for f in sorted(glob.glob(os.path.join(SNAPSHOTS, "*.json"))):
            self.members.append({"kind": "self", "ref": f, "w": 0.5, "n": 0})
        for f in sorted(glob.glob(os.path.join(EXPLOITERS, "*.json"))):
            self.members.append({"kind": "exploiter", "ref": f, "w": 0.5,
                                 "n": 0})

    def sample(self):
        """One opponent, honoring the kind mix, PFSP within the kind."""
        kinds = [k for k in self.MIX
                 if k == "current" or any(m["kind"] == k
                                          for m in self.members)]
        weights = [self.MIX[k] for k in kinds]
        kind = self.rng.choices(kinds, weights)[0]
        if kind == "current":
            return {"kind": "current", "ref": None}
        pool = [m for m in self.members if m["kind"] == kind]
        pw = [max(0.02, m["w"] * (1.0 - m["w"])) + 0.02 for m in pool]
        return self.rng.choices(pool, pw)[0]

    def record(self, member, my_score: float):
        """my_score in {0, 0.5, 1} from the CURRENT policy's side."""
        if member is None or member.get("kind") == "current":
            return
        m = member
        m["n"] += 1
        # member's win rate vs us moves opposite to our score
        m["w"] += (1.0 - my_score - m["w"]) / min(m["n"], 50)

    def anchor_members(self):
        return [m for m in self.members if m["kind"] == "anchor"]

    def exploiter_edge(self):
        ex = [m for m in self.members if m["kind"] == "exploiter"
              and m["n"] >= 5]
        if not ex:
            return None
        return max(m["w"] for m in ex)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-anchors", type=int, default=0)
    a = ap.parse_args()
    if a.build_anchors:
        print(json.dumps(build_anchors(a.build_anchors), indent=1))
