"""Is the interpreter on this machine the one the ladder scores with?

Run this before anything that produces a number, especially off this box.

    python -m kaggriculture.engine.engine_check            # report and exit 0/1
    python -m kaggriculture.engine.engine_check --json     # machine-readable
    python -m kaggriculture.engine.engine_check --quiet    # exit code only

Why this exists: a public notebook (`muneeb2405`) probed the engine inside a
Kaggle notebook image with real `env.step` calls and found it was **not** the
engine the ladder runs. It had `startingMoney` 2000 against the ladder's 3000,
`COW` at 600 against 400, `farmHandCostMult` 10 against 1, `MELON`'s glut
exponent at 0.90 against 3.60 -- and `SELL FERTILIZER` was silently dropped.

None of that raises. An agent tuned in that image tunes for a different game
and reports the result with total confidence, which is worse than crashing. So
every measurement path calls `require()` first and every remote runner calls it
before it is allowed to write a checkpoint.

Two layers, because either alone would miss something:

* **named constants** -- the five that have actually been observed drifting,
  checked by name so the failure message says which one and by how much;
* **a fingerprint** over the whole economy (crops, animals, market curves, land
  prices, board and episode shape), so a drift in a constant nobody has
  thought about still trips the check instead of passing quietly.

The fingerprint is expected to change when Kaggle ships a legitimate engine
update. When it does, re-baseline deliberately -- `--update-baseline` -- and
say so in BUILD_JOURNAL.md, because every measurement taken before that point
was taken against a different game.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402  (puts the vendored interpreter on the path)

BASELINE = os.path.join(ROOT, "data", "engine_baseline.json")

# The constants that have been seen to differ between images. Checked by name
# so a mismatch names itself rather than showing up as an opaque hash change.
EXPECT = {
    "config.startingMoney": 3000,
    "config.farmHandCostMult": 1,
    "config.episodeSteps": 720,
    "config.turnsPerDay": 24,
    "config.boardSize": 10,
    "ANIMALS.COW.cost": 400,
    "ANIMALS.SHEEP.cost": 500,
    "ANIMALS.GOOSE.cost": 300,
    "MARKET_PARAMS.MELON.above_target": 3.6,
    "MARKET_PARAMS.STRAWBERRY.base": 120,
    "MARKET_PARAMS.FERTILIZER.base": 100,
    "FARM_HAND_COST_MULT": 1,
    "PRICE_FLOOR": 1,
    "MARKET_I0": 10000,
}


class EngineMismatch(RuntimeError):
    """The local interpreter is not the one the ladder scores with."""


def _env():
    from kaggle_environments import make
    return make("kaggriculture", debug=False)


def observed():
    """Every constant we care about, flattened to dotted keys."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    cfg = _env().configuration
    out = {}
    for key in ("startingMoney", "farmHandCostMult", "episodeSteps",
                "turnsPerDay", "boardSize"):
        out[f"config.{key}"] = cfg.get(key)
    for name, spec in sorted(K.ANIMALS.items()):
        for field, value in sorted(spec.items()):
            out[f"ANIMALS.{name}.{field}"] = value
    for name, spec in sorted(K.CROPS.items()):
        for field, value in sorted(spec.items()):
            out[f"CROPS.{name}.{field}"] = value
    for name, spec in sorted(K.MARKET_PARAMS.items()):
        for field, value in sorted(spec.items()):
            out[f"MARKET_PARAMS.{name}.{field}"] = value
    out["FARM_HAND_COST_MULT"] = K.FARM_HAND_COST_MULT
    out["PRICE_FLOOR"] = K.PRICE_FLOOR
    out["MARKET_I0"] = K.MARKET_I0
    out["LAND_PRICES"] = list(K.LAND_PRICES)
    out["PRODUCTS"] = list(K.PRODUCTS)
    return out


def fingerprint(obs=None):
    obs = observed() if obs is None else obs
    blob = json.dumps(obs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def probe_fertilizer_sell():
    """Can a fertilizer actually be sold? In one Kaggle image it could not.

    A named constant would not have caught that -- the order was accepted and
    silently dropped, which is the interpreter's normal response to an illegal
    op. So this is a behavioural probe: seed the shed and watch the money.
    """
    from kaggle_environments import make
    env = make("kaggriculture", debug=False,
               configuration={"episodeSteps": 4, "actTimeout": 60})
    env.reset(2)

    def seller(obs, config=None):
        me = int(obs["player"])
        hands = obs["farms"][me].get("hands") or []
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands],
                "market": [["SELL", "FERTILIZER", 3]]}

    def idle(obs, config=None):
        me = int(obs["player"])
        hands = obs["farms"][me].get("hands") or []
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands],
                "market": []}

    # Put fertilizer in seat 0's shed before the first market resolves.
    env.state[0].observation.private["shed"]["FERTILIZER"] = 3
    before = float(env.state[0].observation.farms[0]["money"])
    env.step([seller(env.state[0].observation), idle(env.state[1].observation)])
    after = float(env.state[0].observation.farms[0]["money"])
    shed = int((env.state[0].observation.private["shed"] or {})
               .get("FERTILIZER", 0) or 0)
    return {"money_before": before, "money_after": after,
            "shed_left": shed, "sold": after > before and shed < 3}


def check(strict=True):
    """Return a report. `strict` also compares the whole-economy fingerprint."""
    obs = observed()
    report = {"ok": True, "problems": [], "fingerprint": fingerprint(obs),
              "python": sys.version.split()[0],
              "interpreter": os.path.dirname(
                  sys.modules["kaggle_environments"].__file__)
              if "kaggle_environments" in sys.modules else "?"}

    for key, want in EXPECT.items():
        got = obs.get(key)
        if got is None:
            report["problems"].append(f"{key}: missing from this engine")
        elif isinstance(want, float):
            if abs(float(got) - want) > 1e-9:
                report["problems"].append(f"{key}: {got} (ladder has {want})")
        elif got != want:
            report["problems"].append(f"{key}: {got} (ladder has {want})")

    try:
        fert = probe_fertilizer_sell()
        report["fertilizer_sell"] = fert
        if not fert["sold"]:
            report["problems"].append(
                "SELL FERTILIZER did not move money or shed -- this engine "
                "drops the order silently, as one Kaggle image did")
    except Exception as exc:                                   # noqa: BLE001
        report["problems"].append(f"fertilizer probe failed: {exc}")

    baseline = None
    if os.path.exists(BASELINE):
        try:
            baseline = json.load(open(BASELINE, encoding="utf-8"))
        except ValueError:
            report["problems"].append(f"{BASELINE} is not valid JSON")
    report["baseline"] = baseline
    if strict and baseline:
        if baseline.get("fingerprint") != report["fingerprint"]:
            drift = sorted(
                k for k in set(obs) | set(baseline.get("constants") or {})
                if obs.get(k) != (baseline.get("constants") or {}).get(k))
            report["drift"] = drift
            report["problems"].append(
                f"economy fingerprint differs from the baseline recorded "
                f"{baseline.get('recorded', '?')}; {len(drift)} constant(s) "
                f"changed: {', '.join(drift[:6])}"
                + (" ..." if len(drift) > 6 else ""))

    report["ok"] = not report["problems"]
    return report


_CHECKED = None


def require(verbose=False):
    """Raise unless this engine is the ladder's. Call before measuring.

    Cached per process, and skipped entirely when KAGG_ENGINE_CHECKED is set,
    so the pool workers that a search spawns by the hundred do not each pay for
    an environment construction and a probe episode. The parent checks once and
    marks the environment; the workers inherit it.
    """
    global _CHECKED
    if _CHECKED is not None:
        return _CHECKED
    if os.environ.get("KAGG_ENGINE_CHECKED") == "1":
        _CHECKED = {"ok": True, "problems": [], "cached": "inherited"}
        return _CHECKED
    report = check()
    if not report["ok"]:
        raise EngineMismatch(
            "this interpreter is not the one the ladder scores with:\n  - "
            + "\n  - ".join(report["problems"])
            + "\n\nAny number produced here describes a different game. "
              "Run `python -m kaggriculture.engine.engine_check` for the full report.")
    os.environ["KAGG_ENGINE_CHECKED"] = "1"
    _CHECKED = report
    if verbose:
        print(f"engine ok  ({report['fingerprint'][:16]})")
    return report


def write_baseline():
    obs = observed()
    import datetime as dt
    payload = {"recorded": dt.datetime.now().replace(microsecond=0).isoformat(),
               "fingerprint": fingerprint(obs), "constants": obs}
    os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
    with open(BASELINE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    ap.add_argument("--update-baseline", action="store_true",
                    help="re-record the fingerprint (say so in BUILD_JOURNAL.md)")
    args = ap.parse_args()

    if args.update_baseline:
        payload = write_baseline()
        print(f"baseline recorded {payload['recorded']}")
        print(f"  fingerprint {payload['fingerprint'][:16]}  "
              f"({len(payload['constants'])} constants)")
        return 0

    report = check()
    if args.json:
        print(json.dumps(report, indent=1, default=str))
        return 0 if report["ok"] else 1
    if not args.quiet:
        print(f"interpreter   {report['interpreter']}")
        print(f"python        {report['python']}")
        print(f"fingerprint   {report['fingerprint'][:16]}")
        fert = report.get("fertilizer_sell") or {}
        if fert:
            print(f"SELL FERT     {'works' if fert.get('sold') else 'DROPPED'}"
                  f"  (money {fert.get('money_before')} -> {fert.get('money_after')},"
                  f" shed {fert.get('shed_left')} left of 3)")
        if report["ok"]:
            print("\nOK -- this is the engine the ladder scores with.")
        else:
            print(f"\nMISMATCH ({len(report['problems'])}):")
            for p in report["problems"]:
                print(f"  - {p}")
            print("\nNumbers produced on this engine describe a different "
                  "game. Fix the environment before measuring anything.")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
