"""Did the bandit's in-game commits correlate with wins? Field evidence.

Downloads v22.1's ladder replays, detects in each whether an arm committed
(the agent's market channel deviating from the base route's schedule after
turn 192 in a way that matches an arm override), and tabulates outcomes for
committed vs uncommitted games -- the first live measurement of in-game
adaptation on this ladder.

    python -m kaggriculture.measure.commit_audit --submission 55397322 --agent agents/v22_1_bandit.py
"""
from kaggriculture.paths import ROOT
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.episodes as E  # noqa: E402

STAGE = os.path.join(ROOT, "models", "v22", "audit_stage")


def load_agent_module(path):
    spec = importlib.util.spec_from_file_location("audit_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def our_actions(replay, seat):
    """The action our seat actually took at each step, off-by-one corrected."""
    steps = replay.get("steps") or []
    out = []
    for i in range(1, len(steps)):
        out.append(steps[i][seat].get("action") or {})
    return out


def detect_commit(actions, mod):
    """Which arm's overrides do the played market orders match, if any?"""
    votes = defaultdict(int)
    base = mod._ROUTE
    for t, act in enumerate(actions):
        if t < 192 or t >= 710:
            continue
        played = act.get("market") or []
        scheduled = (base[t].get("market") or []) if t < len(base) else []
        for arm, overrides in mod._ARMS.items():
            ov = overrides.get(str(t))
            if ov is None:
                continue
            # A commit shows as the played sells matching the override where
            # it differs from base. Compare sell multisets loosely (clamping
            # changes quantities, so match on items and order presence).
            ov_sells = sorted((o[1]) for o in ov
                              if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL")
            base_sells = sorted((o[1]) for o in scheduled
                                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL")
            played_sells = sorted((o[1]) for o in played
                                  if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL")
            if ov_sells != base_sells:
                if played_sells == ov_sells:
                    votes[arm] += 1
                elif played_sells == base_sells:
                    votes[arm] -= 1
    if not votes:
        return None
    arm, score = max(votes.items(), key=lambda kv: kv[1])
    return arm if score >= 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", default="55397322")
    ap.add_argument("--agent", default=os.path.join("agents", "v22_1_bandit.py"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    mod = load_agent_module(os.path.join(ROOT, args.agent)
                            if not os.path.isabs(args.agent) else args.agent)
    idx_path = os.path.join(ROOT, "data", "ourgames", "index.json")
    games = json.load(open(idx_path, encoding="utf-8"))["games"]
    mine = {eid: g for eid, g in games.items()
            if str(g.get("submission")) == str(args.submission)}
    print(f"{len(mine)} indexed games for {args.submission}")
    os.makedirs(STAGE, exist_ok=True)

    def grab(eid):
        dest = os.path.join(STAGE, str(eid))
        os.makedirs(dest, exist_ok=True)
        hits = [f for f in os.listdir(dest) if f.endswith(".json")]
        if not hits:
            subprocess.run(["kaggle", "competitions", "replay", str(eid),
                            "-p", dest], check=True, capture_output=True)
            for f in os.listdir(dest):
                if f.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(dest, f)) as z:
                        z.extractall(dest)
            hits = [f for f in os.listdir(dest) if f.endswith(".json")]
        return os.path.join(dest, hits[0]) if hits else None

    tab = defaultdict(lambda: [0, 0, 0.0])       # key -> [wins, games, margin]
    rows = list(mine.items())[:args.limit]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(grab, eid): (eid, g) for eid, g in rows}
        for fut in as_completed(futs):
            eid, g = futs[fut]
            try:
                path = fut.result()
                replay = json.load(open(path, encoding="utf-8"))
            except Exception as exc:                               # noqa: BLE001
                print(f"  ! {eid}: {str(exc).splitlines()[0]}")
                continue
            acts = our_actions(replay, int(g["seat"]))
            arm = detect_commit(acts, mod)
            os.remove(path)
            key = arm or "no-commit"
            tab[key][0] += 1 if g["won"] else 0
            tab[key][1] += 1
            tab[key][2] += float(g.get("margin") or 0)

    print(f"\n{'condition':<12} {'games':>6} {'wins':>6} {'win%':>6} {'mean margin':>12}")
    for key, (w, n, m) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
        print(f"{key:<12} {n:>6} {w:>6} {100*w/max(1,n):>5.0f}% {m/max(1,n):>+12,.0f}")


if __name__ == "__main__":
    main()
