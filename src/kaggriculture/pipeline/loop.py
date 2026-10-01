"""The improvement loop: mine, learn, search, gate, repeat.

One command that keeps the agent improving against real ladder data. Each cycle:

    1. MINE      stream more top-rated episodes; replay each sampled state
                 through the current agent and record where it disagrees
    2. DERIVE    turn the field's measured sale timing into a SELL_SCHEDULE prior
    3. SEARCH    CEM on the schedule, then on the parameter vector, scored on
                 paired both-seat matches against the tapes and the incumbent
    4. GATE      promote only if the candidate beats the incumbent on a fresh
                 seed set it was not tuned on
    5. PUBLISH   refresh the Kaggle dataset so the notebook sees the new data

**Stages run one at a time, never concurrently.** That is not tidiness: with
the box saturated by a 300-episode mine, an episode recording of the same agent
on the same seed came back at $16,348 instead of $80,289, because turns that
miss actTimeout get a substituted action. Overlapping a search with a download
does not just slow the search, it corrupts it.

    python -m kaggriculture.pipeline.loop --cycles 3
    python -m kaggriculture.pipeline.loop --cycles 1 --skip mine
    python -m kaggriculture.pipeline.loop --status
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402

PY = sys.executable
STATE = os.path.join(ROOT, ".local", "loop", "state.json")
LOGS = os.path.join(ROOT, "data", "logs")
INCUMBENT = os.path.join(ROOT, "agents", "v9_cem.py")


def _run(name, args, timeout=None):
    os.makedirs(LOGS, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = os.path.join(LOGS, f"loop-{name}-{stamp}.log")
    print(f"  [{name}] {' '.join(str(a) for a in args[1:4])} ...  -> {os.path.relpath(log, ROOT)}")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=ROOT, timeout=timeout,
                              env=dict(os.environ, PYTHONUTF8="1"))
    dur = time.time() - t0
    tail = ""
    try:
        tail = "\n".join(open(log, encoding="utf-8", errors="replace")
                         .read().splitlines()[-4:])
    except OSError:
        pass
    print(f"  [{name}] exit {proc.returncode} in {dur / 60:.1f} min")
    if tail:
        for line in tail.splitlines():
            print(f"      {line[:110]}")
    return proc.returncode, log


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except ValueError:
            pass
    return {"cycles": 0, "history": [], "incumbent": os.path.relpath(INCUMBENT, ROOT)}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w", encoding="utf-8"), indent=1)


def gate(candidate, incumbent, seeds, seed0, workers):
    """Promote only on a seed set the candidate was never tuned on."""
    rc, log = _run("gate", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "measure", "evaluate.py"),
                            candidate, "--vs", incumbent, "-n", str(seeds),
                            "--seed0", str(seed0), "--workers", str(workers)])
    win = margin = None
    try:
        for line in open(log, encoding="utf-8", errors="replace"):
            if os.path.basename(incumbent) in line and "%" in line:
                parts = line.split()
                win = float(parts[-4].rstrip("%")) / 100.0
                margin = float(parts[-1].replace(",", ""))
    except (OSError, ValueError, IndexError):
        pass
    return win, margin


def cycle(args, st):
    n = st["cycles"] + 1
    incumbent = os.path.join(ROOT, st["incumbent"])
    print(f"\n{'=' * 72}\ncycle {n}   incumbent {os.path.relpath(incumbent, ROOT)}\n{'=' * 72}")
    record = {"cycle": n, "started": dt.datetime.now().replace(microsecond=0).isoformat(),
              "incumbent": st["incumbent"]}

    if "mine" not in args.skip:
        _run("mine", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "data", "mine_top.py"),
                      "--episodes", str(args.episodes), "--days", "7",
                      "--jobs", str(args.jobs), "--sample-every", "6",
                      "--agent", incumbent])

    if "derive" not in args.skip:
        _run("derive", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "pipeline", "schedule.py"),
                        "--derive", "--base", incumbent,
                        "--out", os.path.join(ROOT, ".local", "loop", "seeded.py")])

    candidates = []
    seeded = os.path.join(ROOT, ".local", "loop", "seeded.py")
    if "schedule" not in args.skip and os.path.exists(seeded):
        out = os.path.join(ROOT, "agents", f"v11_schedule_c{n}.py")
        rc, _ = _run("schedule", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "pipeline", "schedule.py"),
                                  "--optimise", "--base", seeded, "--out", out,
                                  "--generations", str(args.generations),
                                  "--seeds", str(args.seeds),
                                  "--workers", str(args.workers),
                                  "--vs", incumbent])
        if rc == 0 and os.path.exists(out):
            candidates.append(out)

    if "params" not in args.skip:
        out = os.path.join(ROOT, "agents", f"v10_params_c{n}.py")
        rc, _ = _run("params", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "train", "optimize.py"),
                                "--base", incumbent, "--out", out,
                                "--run", f"loop{n}",
                                "--generations", str(args.generations),
                                "--seeds", str(args.seeds),
                                "--workers", str(args.workers),
                                "--vs", incumbent])
        if rc == 0 and os.path.exists(out):
            candidates.append(out)

    promoted = None
    for cand in candidates:
        win, margin = gate(cand, incumbent, args.gate_seeds,
                           args.seed0 + 1000 * n, args.workers)
        print(f"  gate: {os.path.basename(cand):<28} "
              f"win {('%.0f%%' % (100 * win)) if win is not None else '?':>5}  "
              f"margin {margin if margin is not None else float('nan'):>+10,.0f}")
        record.setdefault("gates", []).append(
            {"candidate": os.path.relpath(cand, ROOT), "win": win, "margin": margin})
        # Win rate is what the ladder scores; margin only breaks ties. The bar
        # is a real edge on unseen seeds, not a coin flip.
        if win is not None and margin is not None and win > 0.55 and margin > 0:
            promoted = cand
            break

    if promoted:
        st["incumbent"] = os.path.relpath(promoted, ROOT)
        record["promoted"] = st["incumbent"]
        try:
            registry.register_model(
                os.path.basename(promoted),
                description=f"promoted by src/kaggriculture/pipeline/loop.py cycle {n}",
                tags=["loop", "promoted"])
        except Exception:                                          # noqa: BLE001
            pass
        print(f"  PROMOTED -> {st['incumbent']}")
    else:
        record["promoted"] = None
        print("  nothing cleared the gate; incumbent stands")

    if "publish" not in args.skip:
        _run("publish", [PY, "-u", os.path.join(ROOT, "src", "kaggriculture", "data", "kaggle_dataset.py"),
                         "--update", "--note", f"loop cycle {n}"])

    record["finished"] = dt.datetime.now().replace(microsecond=0).isoformat()
    st["cycles"] = n
    st["history"].append(record)
    save_state(st)
    return promoted


def status():
    st = load_state()
    print(f"cycles run : {st['cycles']}")
    print(f"incumbent  : {st['incumbent']}")
    for r in st["history"][-8:]:
        gates = ", ".join(
            f"{os.path.basename(g['candidate'])} "
            f"{100 * (g['win'] or 0):.0f}%/{(g['margin'] or 0):+,.0f}"
            for g in r.get("gates", []))
        print(f"  cycle {r['cycle']:>2}  {r['started'][:16]}  "
              f"promoted={r.get('promoted') or '-'}   {gates}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=60, help="episodes to mine per cycle")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gate-seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["mine", "derive", "schedule", "params", "publish"])
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return 0
    st = load_state()
    for _ in range(args.cycles):
        cycle(args, st)
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
