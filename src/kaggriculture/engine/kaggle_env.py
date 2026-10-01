"""The one place an episode is created, configured exactly as Kaggle does.

The competition's own defaults, read from the installed environment rather than
restated here:

    episodeSteps 720   actTimeout 1   runTimeout 1200   turnsPerDay 24
    boardSize 10       startingMoney 3000               shedCapacity 100

`actTimeout` is the one that bites. Most of this project's harnesses raise it to
60 so a slow debug run does not get killed mid-episode -- which is convenient
and also means those runs cannot detect an agent that would time out on Kaggle.
An agent that overruns is not slow there, it is *wrong*: the engine substitutes
a default action and you lose games you would otherwise win, with nothing in the
logs to say why.

So there are two ways to build an episode and they are named for what they are:

    strict_env(seed)    exactly Kaggle's configuration. Use for anything whose
                        result you intend to believe.
    relaxed_env(seed)   actTimeout raised, for debugging and bulk search where
                        throughput matters more than fidelity.

    python -m kaggriculture.engine.kaggle_env --show      # the official defaults
    python -m kaggriculture.engine.kaggle_env --audit     # who in this repo diverges, and how
    python -m kaggriculture.engine.kaggle_env --verify    # play one strict episode end to end
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

# Raised only where fidelity is deliberately traded for throughput.
RELAXED_ACT_TIMEOUT = 60
RELAXED_RUN_TIMEOUT = 100000


def official_defaults():
    """Kaggle's configuration, straight from the installed environment."""
    from kaggle_environments import make
    return dict(make("kaggriculture").configuration)


def strict_env(seed=None, steps=None):
    """An episode configured exactly as the competition runs it.

    Only `seed` is set, and only because the competition sets one per episode
    too. Nothing else is overridden -- that is the whole point.
    """
    from kaggle_environments import make
    cfg = {}
    if seed is not None:
        cfg["seed"] = seed
    if steps is not None:                  # shortened episodes for smoke tests
        cfg["episodeSteps"] = steps
    return make("kaggriculture", configuration=cfg) if cfg else make("kaggriculture")


def relaxed_env(seed=None, steps=720, act_timeout=RELAXED_ACT_TIMEOUT):
    """Faster-to-debug episode. Results from here are indicative, not final."""
    from kaggle_environments import make
    return make("kaggriculture",
                configuration={"episodeSteps": steps, "seed": seed,
                               "actTimeout": act_timeout,
                               "runTimeout": RELAXED_RUN_TIMEOUT})


def run_strict(left, right, seed=None, steps=None):
    """One strict episode. Returns (reward_left, reward_right, statuses)."""
    env = strict_env(seed, steps)
    env.run([left, right])
    final = env.steps[-1]
    return (float(final[0]["reward"] or 0), float(final[1]["reward"] or 0),
            [s["status"] for s in final])


# ------------------------------------------------------------------- audit --

_MAKE = re.compile(r"make\(\s*[\"']kaggriculture[\"']", re.S)
_ACT = re.compile(r"[\"']actTimeout[\"']\s*:\s*([0-9.]+)")
_RUN = re.compile(r"[\"']runTimeout[\"']\s*:\s*([0-9.]+)")
_STEPS = re.compile(r"[\"']episodeSteps[\"']\s*:\s*([0-9]+)")


def audit():
    """Every place this repo builds an episode, and how it differs from Kaggle.

    Divergence is not automatically a bug -- a 48-turn smoke test is supposed to
    be short -- but it should be visible rather than buried in a dict literal
    three files away from the number it changes.
    """
    official = official_defaults()
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        if rel.startswith((".local", "build", "opponents")) or "__pycache__" in rel:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        if not _MAKE.search(src):
            continue
        act = _ACT.search(src)
        run = _RUN.search(src)
        steps = _STEPS.search(src)
        diffs = []
        if act and float(act.group(1)) != float(official["actTimeout"]):
            diffs.append(f"actTimeout {act.group(1)} (Kaggle {official['actTimeout']})")
        if run and float(run.group(1)) != float(official["runTimeout"]):
            diffs.append(f"runTimeout {run.group(1)} (Kaggle {official['runTimeout']})")
        if steps and int(steps.group(1)) != int(official["episodeSteps"]):
            diffs.append(f"episodeSteps {steps.group(1)} (Kaggle {official['episodeSteps']})")
        rows.append({"file": rel, "diffs": diffs})
    return official, rows


def verify(seed=7, steps=None, left="agents/v2_tuned.py", right="random"):
    """Play a strict episode and confirm both agents survived the real timeout."""
    import time
    t0 = time.time()
    a, b, statuses = run_strict(left, right, seed, steps)
    elapsed = time.time() - t0
    ok = all(s == "DONE" for s in statuses)
    return {"left": left, "right": right, "seed": seed,
            "rewards": [a, b], "statuses": statuses, "seconds": round(elapsed, 1),
            "ok": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--agent", default="agents/v2_tuned.py")
    ap.add_argument("--vs", default="random")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.verify:
        res = verify(args.seed, args.steps, args.agent, args.vs)
        if args.json:
            print(json.dumps(res, indent=1))
        else:
            print(f"strict episode, seed {res['seed']}, {res['seconds']}s")
            print(f"  {res['left']}: ${res['rewards'][0]:,.0f}  [{res['statuses'][0]}]")
            print(f"  {res['right']}: ${res['rewards'][1]:,.0f}  [{res['statuses'][1]}]")
            print("\n" + ("both agents finished under Kaggle's real 1s actTimeout"
                          if res["ok"] else
                          "SOMETHING DID NOT FINISH -- an agent overran the real "
                          "timeout and would silently lose games on Kaggle"))
        return 0 if res["ok"] else 1

    official, rows = audit()
    if args.json:
        print(json.dumps({"official": official, "callsites": rows}, indent=1))
        return 0

    if args.show or not args.audit:
        print("Kaggle's configuration for this competition\n" + "-" * 60)
        for k, v in official.items():
            print(f"  {k:<24} {json.dumps(v)}")
        print("\nactTimeout is 1 second. An agent that overruns is not slow, it is")
        print("wrong: the engine substitutes a default action and the loss is silent.")
        if not args.audit:
            return 0

    print("\nWhere this repo builds episodes\n" + "-" * 60)
    clean = [r for r in rows if not r["diffs"]]
    dirty = [r for r in rows if r["diffs"]]
    for r in dirty:
        print(f"  {r['file']}")
        for d in r["diffs"]:
            print(f"      {d}")
    for r in clean:
        print(f"  {r['file']}   (matches Kaggle)")
    print(f"\n{len(clean)} exact, {len(dirty)} relaxed.")
    print("Relaxed is fine for bulk search and debugging -- it is how a 720-turn")
    print("match finishes in 7 seconds. Anything you intend to *believe* should")
    print("go through strict_env(), and src/kaggriculture/engine/official_eval.py already does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
