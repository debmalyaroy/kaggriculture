"""Local smoke test -- does an agent RUN, exception-free, for a full episode?

The cheapest gate: load a single-file agent module, play it against PASS on the
vendored engine for a full 720-turn episode from BOTH seats, and count any Python
exception raised inside ``agent(obs)``. Zero exceptions + a finite bank = the
agent is structurally sound and safe to carry into the heavier evaluations
(``measure.eval_harness``) or a package step. This is the "zero runtime
environment errors" check every agent must pass before it goes anywhere.

    python -m kaggriculture.measure.smoke_test agents/v56y_baseline.py
    python -m kaggriculture.measure.smoke_test agents/A.py agents/B.py --seeds 3
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import importlib.util
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(ROOT, "vendor"))

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def load_agent(path: str):
    """Import a single-file agent module and return its ``agent`` callable."""
    spec = importlib.util.spec_from_file_location(
        "smoke_agent_" + os.path.basename(path).replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "agent", None)
    if fn is None:
        raise AttributeError(f"{path} has no top-level agent(obs)")
    return fn


def _wrap_counting(fn, errors):
    """Wrap an agent so exceptions are counted, not fatal (engine sees PASS)."""
    def safe(obs, config=None):
        try:
            import inspect
            n = len(inspect.signature(fn).parameters)
            return fn(obs) if n == 1 else fn(obs, config)
        except Exception as e:                # noqa: BLE001 -- counting harness
            errors.append(f"{type(e).__name__}: {e}\n"
                          + traceback.format_exc(limit=2))
            return PASS
    return safe


def smoke(agent_path: str, seeds=(3, 4, 5), steps: int = 720) -> dict:
    from kaggle_environments import make
    fn = load_agent(agent_path)
    errors, banks, timings, worst = [], [], [], 0.0
    for seed in seeds:
        for seat in (0, 1):
            per_turn = []

            def timed(obs, c=None, _f=fn, _e=errors, _pt=per_turn):
                import inspect
                t = time.time()
                try:
                    n = len(inspect.signature(_f).parameters)
                    r = _f(obs) if n == 1 else _f(obs, c)
                except Exception as ex:      # noqa: BLE001
                    _e.append(f"{type(ex).__name__}: {ex}")
                    r = PASS
                _pt.append((time.time() - t) * 1000.0)
                return r
            agents = ([timed, lambda o, c=None: PASS] if seat == 0
                      else [lambda o, c=None: PASS, timed])
            t0 = time.time()
            env = make("kaggriculture", configuration={"seed": seed},
                       debug=False)
            env.run(agents)
            timings.append((time.time() - t0) / max(steps, 1) * 1000.0)
            if per_turn:
                worst = max(worst, max(per_turn))    # G4: worst single turn
            bank = float(env.steps[-1][seat]["reward"] or 0)
            banks.append(bank)
    n_games = len(seeds) * 2
    # G4: PASS also requires the worst single turn under the 1 s actTimeout
    ok = (len(errors) == 0 and all(b == b for b in banks)   # b==b: not NaN
          and worst < 1000.0)
    res = dict(agent=os.path.relpath(agent_path, ROOT), games=n_games,
               errors=len(errors), banks=banks,
               mean_bank=sum(banks) / max(len(banks), 1),
               mean_turn_ms=sum(timings) / max(len(timings), 1),
               worst_turn_ms=worst, pass_=ok)
    tag = "PASS" if ok else "FAIL"
    print(f"[smoke] {res['agent']}: {tag}  {n_games} games  "
          f"{len(errors)} errors  mean_bank {res['mean_bank']:.0f}  "
          f"~{res['mean_turn_ms']:.1f} ms/turn  worst {worst:.0f} ms"
          f"{'  ⚠>1s' if worst >= 1000 else ''}")
    for e in errors[:3]:
        print("   !", e.splitlines()[0])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()
    seeds = tuple(range(3, 3 + args.seeds))
    all_ok = True
    for a in args.agents:
        r = smoke(a, seeds=seeds, steps=args.steps)
        all_ok = all_ok and r["pass_"]
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
