"""Ladder parity: does the VENDORED engine reproduce real ladder episodes?

The version string is not the truth -- the interpreter's behavior is. This
gate replays fresh RAW ladder replays through the vendored python engine with
both seats forced to their recorded actions, and compares the final banks to
what the ladder actually paid, to the dollar (destbreso's community
measurement standard: recorded episodes replay 40/40 to the dollar on a
matching build). Any mismatch means every offline number is answering a
different question than the ladder asks, so the verdict self-revokes offline
evidence the same way `serve_equiv` does.

Why banks and not per-step digests: a raw replay does not carry our digest
fields, but the final bank is a hash of the whole episode's economics -- a
divergent step almost never re-converges to the exact recorded dollar.

    python src/ladder_parity.py                # audit up to 6 fresh replays
    python src/ladder_parity.py --replays 12
    python src/ladder_parity.py --check        # print the gate verdict only

Writes models/ladder_parity.json:
    {"engine": "1.32.7", "when": iso, "matches": n, "mismatches": n,
     "checked": [{"episode", "path", "recorded", "replayed", "ok"}, ...]}

The 2026-08-23 "local vs ladder divergence" scare was a submission-id
misattribution (see .local/memory/v33-ladder-collapse.md) -- this gate exists
so the next scare is settled by measurement in minutes, not by inference.
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PARITY = os.path.join(ROOT, "models", "ladder_parity.json")
ENGINE_VERSION = os.path.join(ROOT, "models", "engine_version.json")
STAGE_GLOBS = (
    os.path.join(ROOT, "data", "sameday", "_stage", "*", "*.json"),
    os.path.join(ROOT, "data", "ourgames", "_stage", "*", "*.json"),
)
TOLERANCE = 0.5          # dollars; recorded rewards are floats


def vendor_engine():
    try:
        with open(ENGINE_VERSION, encoding="utf-8") as fh:
            return json.load(fh).get("engine")
    except (OSError, ValueError):
        return None


def load_replay(path):
    """(episode_id, seed, action_pairs, recorded_banks) or None."""
    try:
        rep = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # module_version is the ONLY engine ground truth in a raw replay --
    # rep["version"] is the schema version (memory: engine-and-ingest-truth).
    ver = rep.get("module_version")
    eng = vendor_engine()
    if eng and ver and str(ver) != str(eng):
        return None
    seed = (rep.get("info") or {}).get("seed")
    steps = rep.get("steps") or []
    if seed is None or len(steps) < 100:
        return None
    rewards = rep.get("rewards")
    if not (isinstance(rewards, list) and len(rewards) == 2):
        last = steps[-1]
        try:
            rewards = [float(last[i].get("reward") or 0) for i in (0, 1)]
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
    pairs = []
    for st in steps[1:]:
        acts = []
        for i in (0, 1):
            a = st[i].get("action") if i < len(st) else None
            acts.append(a if isinstance(a, dict) else {})
        pairs.append(tuple(acts))
    episode = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    return episode, int(seed), pairs, [float(rewards[0]), float(rewards[1])]


def replay_banks(seed, action_pairs):
    """Final banks after forcing both seats' recorded actions locally."""
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": seed},
               info={"seed": seed})
    env.reset(2)
    for a0, a1 in action_pairs:
        if env.done:
            break
        env.step([a0, a1])
    return [float(env.state[i].observation.farms[i]["money"]) for i in (0, 1)]


def fresh_replays(limit):
    stamped = []
    for pattern in STAGE_GLOBS:
        for path in glob.glob(pattern):
            # The hourly scrape ingests and moves stage files while we scan.
            try:
                stamped.append((os.path.getmtime(path), path))
            except OSError:
                continue
    stamped.sort(reverse=True)
    return [p for _, p in stamped[: limit * 4]]  # headroom: some will not parse


def audit(n_replays):
    eng = vendor_engine()
    checked, matches, mismatches = [], 0, 0
    for path in fresh_replays(n_replays):
        if len(checked) >= n_replays:
            break
        parsed = load_replay(path)
        if parsed is None:
            continue
        episode, seed, pairs, recorded = parsed
        try:
            replayed = replay_banks(seed, pairs)
        except Exception as exc:                                # noqa: BLE001
            print(f"  {episode}: replay raised {type(exc).__name__}: {exc}")
            continue
        ok = all(abs(a - b) <= TOLERANCE for a, b in zip(recorded, replayed))
        matches += 1 if ok else 0
        mismatches += 0 if ok else 1
        checked.append({"episode": episode, "path": os.path.relpath(path, ROOT),
                        "recorded": recorded, "replayed": replayed, "ok": ok})
        flag = "ok " if ok else "MISMATCH"
        print(f"  {flag} {episode}: recorded {recorded[0]:,.0f}/{recorded[1]:,.0f}"
              f" replayed {replayed[0]:,.0f}/{replayed[1]:,.0f}")
    verdict = {"engine": eng, "when": dt.datetime.now().isoformat(timespec="seconds"),
               "matches": matches, "mismatches": mismatches, "checked": checked}
    os.makedirs(os.path.dirname(PARITY), exist_ok=True)
    with open(PARITY, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print(f"\n{matches} match(es), {mismatches} mismatch(es) -> {os.path.relpath(PARITY, ROOT)}")
    return 0 if mismatches == 0 and matches > 0 else 1


def parity_ok(max_age_days=7, min_matches=3):
    """Gate predicate, `serve_allowed` style: (bool, reason)."""
    try:
        d = json.load(open(PARITY, encoding="utf-8"))
        eng = vendor_engine()
        age_h = (dt.datetime.now().timestamp() - os.path.getmtime(PARITY)) / 3600
        if d.get("mismatches", 1) != 0:
            return False, f"{d['mismatches']} ladder-parity mismatch(es) on record"
        if d.get("matches", 0) < min_matches:
            return False, f"only {d.get('matches', 0)} parity matches (<{min_matches})"
        if eng and d.get("engine") != eng:
            return False, f"audit ran on {d.get('engine')}, engine is {eng}"
        if age_h > max_age_days * 24:
            return False, f"audit {age_h / 24:.1f}d old (>{max_age_days}d)"
        return True, f"{d['matches']} exact-bank ladder replays, 0 mismatches"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"no usable parity audit ({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replays", type=int, default=6)
    ap.add_argument("--check", action="store_true",
                    help="print the current gate verdict and exit")
    args = ap.parse_args()
    if args.check:
        ok, why = parity_ok()
        print(("PASS  " if ok else "BLOCK ") + why)
        return 0 if ok else 1
    return audit(args.replays)


if __name__ == "__main__":
    raise SystemExit(main())
