"""Play an agent against a held-out panel of the current top-N teams.

    python -m kaggriculture.measure.panel --agent agents/v17_route.py --top 100
    python -m kaggriculture.measure.panel --agent agents/v17_route.py --top 30 --seeds 4
    python -m kaggriculture.measure.panel --list

This is the honest test. `src/kaggriculture/data/routes.py --select` picks a route by playing
candidates against each other, but those candidates all come from the **fit**
window. This builds opponents from the **outer and validation** windows -- games
that selection never opened -- so a route cannot look good here merely because
it was chosen here.

One opponent per team, their strongest held-out game, so the panel is breadth
across the top-N rather than depth on whoever happened to replay most. Every
pairing is played in both seats on common seeds: shared-market games are not
symmetric and a one-seat test can reverse the apparent winner.

What a result here does and does not mean: these opponents execute fixed
recorded trajectories and do not react to us. That makes this a policy screen,
not a prediction of a rating. It is still the closest thing we have to the
ladder, because every opponent is a real game a real top-100 team actually
played.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.data.routes as R  # noqa: E402

STAGE = os.path.join(ROOT, "data", "panel")


def _mute(*_a):
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def play(job):
    left, right, seed = job
    from kaggle_environments import make
    env = make("kaggriculture", configuration={
        "episodeSteps": 720, "seed": seed, "actTimeout": 60,
        "runTimeout": 100000}, debug=False)
    env.run([left, right])
    final = env.steps[-1]
    return float(final[0]["reward"] or 0), float(final[1]["reward"] or 0)


def assemble(top=100, verbose=True):
    """One held-out opponent per top-N team. Returns [(path, meta), ...]."""
    idx = R.assign_windows(R.load_index(), verbose=False)
    R.save_index(idx)
    held = [r for r in idx["routes"].values()
            if r.get("window") in ("outer", "validation")
            and (r.get("rank") or 10 ** 9) <= top]
    by_team = {}
    for r in held:
        by_team.setdefault(r["team"], []).append(r)
    chosen = sorted((max(v, key=lambda x: x.get("bank", 0))
                     for v in by_team.values()),
                    key=lambda r: r.get("rank") or 10 ** 9)
    os.makedirs(STAGE, exist_ok=True)
    out = []
    for rec in chosen:
        path = os.path.join(STAGE, f"opp_r{(rec.get('rank') or 999):03d}_{rec['id']}.py")
        if not os.path.exists(path):
            R.build(rec["id"], path, verbose=False)
        out.append((path, rec))
    if verbose:
        print(f"{len(held)} held-out route(s) from top-{top} teams "
              f"-> {len(out)} opponent(s), one per team")
    return out


def _ascii(text):
    """Console-safe team name; a cp1252 console cannot print arbitrary
    Unicode and an unguarded print kills the whole run."""
    return str(text).encode("ascii", "replace").decode("ascii")


def run(agent, top=100, seeds=3, seed0=64000, workers=None, verbose=True):
    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()

    panel = assemble(top, verbose=verbose)
    if not panel:
        raise SystemExit(f"no held-out routes from top-{top} teams -- "
                         f"mine more (python -m kaggriculture.data.routes --mine)")
    me = os.path.abspath(agent)
    jobs, tag = [], []
    for path, _rec in panel:
        o = os.path.abspath(path)
        for s in range(seeds):
            jobs.append((me, o, seed0 + s)); tag.append((path, 0))
            jobs.append((o, me, seed0 + s)); tag.append((path, 1))

    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    if verbose:
        print(f"{len(jobs)} episode(s) on {workers} workers "
              f"({seeds} seed(s) x 2 seats x {len(panel)} opponents)\n")
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as ex:
        results = list(ex.map(play, jobs))

    acc = {}
    for (path, seat), (r0, r1) in zip(tag, results):
        mine, theirs = (r0, r1) if seat == 0 else (r1, r0)
        a = acc.setdefault(path, {"w": 0, "n": 0, "m": [], "t": []})
        a["n"] += 1
        a["w"] += int(mine > theirs)
        a["m"].append(mine)
        a["t"].append(theirs)

    meta = {p: rec for p, rec in panel}
    rows = []
    for path, a in acc.items():
        rec = meta[path]
        rows.append({"rank": rec.get("rank"), "team": rec.get("team", "?"),
                     "route": rec["id"], "window": rec.get("window"),
                     "win": a["w"] / a["n"], "n": a["n"],
                     "mine": statistics.mean(a["m"]),
                     "theirs": statistics.mean(a["t"])})
    rows.sort(key=lambda r: (r["rank"] or 10 ** 9))

    # Persist BEFORE printing. A panel run costs half an hour; a console
    # encoding error must never be able to destroy it. This exact crash
    # (UnicodeEncodeError on a non-Latin-1 team name, cp1252 console) threw
    # away a completed 70-opponent run on 2026-08-07.
    try:
        os.makedirs(os.path.join(ROOT, ".local", "dbg"), exist_ok=True)
        auto = os.path.join(ROOT, ".local", "dbg",
                            "panel_" + os.path.basename(agent).replace(".py", "") + ".json")
        with open(auto, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    except OSError:
        pass

    if verbose:
        print(f"{os.path.basename(agent)} vs the top-{top} held-out panel\n")
        print(f"{'rank':>5} {'team':<26}{'win%':>7}{'our bank':>12}"
              f"{'theirs':>12}{'margin':>11}  window")
        print("-" * 84)
        for r in rows:
            print(f"{r['rank'] or '?':>5} {_ascii(r['team'])[:25]:<26}"
                  f"{100 * r['win']:>6.0f}%{r['mine']:>12,.0f}"
                  f"{r['theirs']:>12,.0f}{r['mine'] - r['theirs']:>11,.0f}"
                  f"  {r['window']}")
        wins = sum(r["win"] * r["n"] for r in rows)
        n = sum(r["n"] for r in rows)
        beaten = sum(1 for r in rows if r["win"] > 0.5)
        print("-" * 84)
        print(f"OVERALL  {wins:.0f}/{n} games = {100 * wins / n:.1f}%")
        print(f"         beaten outright: {beaten}/{len(rows)} opponents")
        print(f"         mean margin {statistics.mean(r['mine'] - r['theirs'] for r in rows):+,.0f}")
        lost = [r for r in rows if r["win"] <= 0.5]
        if lost:
            print(f"\n  not beaten ({len(lost)}):")
            for r in sorted(lost, key=lambda r: r["win"]):
                print(f"    rank {r['rank']:>4}  {_ascii(r['team'])[:28]:<30}"
                      f"{100 * r['win']:>4.0f}%  margin "
                      f"{r['mine'] - r['theirs']:>+10,.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "v17_route.py"))
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=64000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--json", default=None, help="also write the rows here")
    ap.add_argument("--list", action="store_true",
                    help="show the panel without playing it")
    args = ap.parse_args()

    if args.list:
        for path, rec in assemble(args.top):
            print(f"  rank {rec.get('rank'):>4}  {_ascii(rec['team'])[:28]:<30}"
                  f"{rec['id']:<16} {rec.get('window')}")
        return 0

    agent = args.agent if os.path.isabs(args.agent) else os.path.join(ROOT, args.agent)
    rows = run(agent, top=args.top, seeds=args.seeds, seed0=args.seed0,
               workers=args.workers)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.json}")

    # Only our own agents belong in the model registry. Pointing this tool at a
    # panel member is a legitimate calibration -- it is how we learned that the
    # rank-1 team's own route scores 26.4% -- but recording the result would
    # invent a model with no file in agents/ and no ladder rating, which is
    # exactly the inconsistency test_system checks for.
    if os.path.dirname(os.path.abspath(agent)) != os.path.join(ROOT, "agents"):
        print(f"\n(not recorded: {os.path.basename(agent)} is not one of "
              f"our agents -- the registry tracks agents/ only)")
        return 0

    try:
        import kaggriculture.data.registry as registry
        wins = sum(r["win"] * r["n"] for r in rows)
        n = sum(r["n"] for r in rows)
        registry.record_eval(
            os.path.basename(agent), f"top-{args.top} held-out panel",
            wins / max(1, n),
            statistics.mean(r["mine"] - r["theirs"] for r in rows),
            statistics.mean(r["mine"] for r in rows),
            seeds=args.seeds, matches=n,
            note=f"{len(rows)} opponents, outer+validation windows only")
    except Exception as exc:                                       # noqa: BLE001
        print(f"(registry not updated: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
