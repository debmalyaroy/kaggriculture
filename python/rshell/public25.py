"""v63.5_rl vs the top-25 public agents (+ our live v62.1 / v63) in all 64 worlds (operator 2026-09-27).

    PYTHONPATH=~/kaggriculture/src KAGG_BIN=... python python/rshell/public25.py --cand DIR/main.py [--per-world 1]
        [--split heldout] [--workers 24] [--out data/rshell/public25]

The public agents are PYTHON notebooks, so they play on the main repo's faithful harness (Python agent on the
Rust `kagg serve` engine, exact vs the official engine; kaggriculture.bandit.gate.loss_forensics.capture_game).
The candidate is its own tarball's main.py over a native agent-stdio build. Seeds: the 64-world bank
(data/worlds/w64_bank.json, one world = the first two shops), both seats. Rows stream to OUT/games.jsonl
(resumable); OUT/report.json has W/L/D and score per opponent and per world.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = RL  # one repo since the 2026-10-01 merge


def play(task):
    import contextlib
    import io
    from kaggriculture.bandit.gate import loss_forensics as LF
    cand, opp, opp_path, world, seed, seat = task
    row = {"opp": opp, "world": world, "seed": seed, "seat": seat}
    t0 = time.time()
    ours = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ours = LF.load_pyagent(cand)
            _, _, us, them = LF.capture_game(ours, opp_path, seed, seat)
        row.update(us=us, them=them, gap=us - them)
    except Exception as exc:  # noqa: BLE001
        row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}")
    finally:
        # the candidate's main.py keeps its agent-stdio child in a module global and never ends it; every game
        # loads a fresh module, so without this each game leaked a process (2,047 of them = 114 GB on 27 Sep)
        try:
            g = ours.__globals__
            proc = (g.get("_BIN") or {}).get("proc")
            if proc is not None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
    row["secs"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--opps", default=os.path.join(RL, ".local", "ext", "top25_public.json"))
    ap.add_argument("--field", default=os.path.join(REPO, "data", "winplan", "field"))
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--per-world", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(RL, "data", "rshell", "public25"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    opps = json.load(open(a.opps, encoding="utf-8"))
    bank = json.load(open(os.path.join(RL, "data", "worlds", "w64_bank.json")))[a.split]
    seeds = [(w, s) for w in sorted(bank) for s in bank[w][: a.per_world]]
    path = os.path.join(a.out, "games.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if "gap" in r:
                done.add((r["opp"], r["seed"], r["seat"]))
    tasks = [(os.path.abspath(a.cand), o, os.path.join(a.field, o + ".py"), w, s, seat)
             for o in opps for w, s in seeds for seat in (0, 1) if (o, s, seat) not in done]
    print(f"[public25] {len(opps)} opponents x {len(seeds)} seeds x 2 seats; {len(tasks)} to play, {len(done)} done", flush=True)
    t0 = time.time()
    with open(path, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(play, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"[public25] {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    rows = [json.loads(l) for l in open(path)]
    errs = [r for r in rows if "error" in r]
    per = defaultdict(lambda: [0, 0, 0])
    worlds = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if "gap" not in r:
            continue
        k = 0 if r["gap"] > 0 else 1 if r["gap"] < 0 else 2
        per[r["opp"]][k] += 1
        worlds[r["world"]][k] += 1
    sc = lambda v: (v[0] + 0.5 * v[2]) / max(1, sum(v))  # noqa: E731
    rep = {"cand": a.cand, "games": sum(sum(v) for v in per.values()), "errors": len(errs),
           "score": sc([sum(v[i] for v in per.values()) for i in range(3)]),
           "per_opponent": {o: {"W": v[0], "L": v[1], "D": v[2], "score": sc(v)} for o, v in sorted(per.items(), key=lambda kv: sc(kv[1]))},
           "per_world": {w: {"W": v[0], "L": v[1], "D": v[2], "score": sc(v)} for w, v in sorted(worlds.items(), key=lambda kv: sc(kv[1]))}}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(f"[public25] score {rep['score']:.3f} over {rep['games']} games, {len(errs)} errors", flush=True)
    for o, v in rep["per_opponent"].items():
        print(f"[public25]   {o:48s} W {v['W']:3d} L {v['L']:3d} D {v['D']:3d}  {v['score']:.3f}", flush=True)


if __name__ == "__main__":
    main()
