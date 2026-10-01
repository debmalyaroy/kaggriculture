"""Run every test suite and report one verdict.

There was no single entrypoint, which is part of how `tests/test_agents.py`
stayed broken for days after the 2026-08-10 restructure: nothing ran it, so
nothing noticed. This discovers suites rather than listing them, so a new
`tests/test_*.py` is included the moment it exists and cannot be forgotten.

Ordered fast-to-slow so a structural break (stale paths, bad metric, bad gate)
surfaces in seconds rather than after the episode-playing suites.

    python tests/run_all.py            # everything
    python tests/run_all.py --fast     # skip suites that play episodes
    python tests/run_all.py --list
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

# Suites that play full episodes; minutes rather than seconds.
SLOW = {"test_agents", "test_system", "test_contract", "test_rust_engine",
        "test_compiled_agent", "test_bandit_parity"}
# Cheap structural suites first: if paths or the currency are broken, every
# later number is meaningless anyway.
ORDER = ["test_paths", "test_win_metric", "test_gates", "test_tourney_halving",
         "test_obs_schema", "test_branch_dispatch", "test_rust_rng",
         "test_rust_market"]


def discover():
    found = []
    for p in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        found.append(os.path.splitext(os.path.basename(p))[0])
    ranked = [n for n in ORDER if n in found]
    ranked += [n for n in found if n not in ORDER and n not in SLOW]
    ranked += [n for n in found if n in SLOW]
    return ranked


def run_one(name, timeout):
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, f"{name}.py")],
                       capture_output=True, text=True, cwd=ROOT, env=ENV,
                       timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    dt = time.time() - t0
    skipped = "SKIP" in out and p.returncode == 0 and "passed" not in out.lower()
    return p.returncode, dt, out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true",
                    help="skip suites that play full episodes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    suites = discover()
    if args.fast:
        suites = [s for s in suites if s not in SLOW]
    if args.list:
        for s in suites:
            print(f"{'SLOW' if s in SLOW else 'fast'}  {s}")
        return 0

    print(f"running {len(suites)} suite(s)"
          f"{' (fast only)' if args.fast else ''}\n")
    fails, skips = [], []
    total = 0.0
    for s in suites:
        tag = "SLOW" if s in SLOW else "fast"
        print(f"  {s:<26} [{tag}] ", end="", flush=True)
        try:
            rc, dt, out, skipped = run_one(s, args.timeout)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            fails.append((s, "timed out"))
            continue
        total += dt
        if rc != 0:
            print(f"FAIL ({dt:.1f}s)")
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-4:]
            for ln in tail:
                print(f"      {ln[:150]}")
            fails.append((s, tail[-1] if tail else f"exit {rc}"))
        elif skipped:
            print(f"skipped ({dt:.1f}s)")
            skips.append(s)
        else:
            print(f"pass ({dt:.1f}s)")

    print(f"\n{len(suites) - len(fails) - len(skips)} passed, "
          f"{len(skips)} skipped, {len(fails)} failed in {total:.1f}s")
    if skips:
        print(f"skipped: {', '.join(skips)}")
    if fails:
        print("\nFAILURES:")
        for s, why in fails:
            print(f"  {s}: {why[:160]}")
        return 1
    print("ALL SUITES GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
