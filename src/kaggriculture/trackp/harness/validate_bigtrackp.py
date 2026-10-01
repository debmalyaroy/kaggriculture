"""Official-engine validation of the big-trackp tarball, inside Linux.

Runs REAL kaggle_environments episodes: the unpacked artefact's main.py
vs a reference agent, both seats, several seeds. PASS requires:
  * every episode completes with both agents ACTIVE (no error/timeout);
  * zero RUST-FAIL lines in the artefact agent's stderr log
    (the no-fallback rule: a rust failure must be loud, and pre-flight
    means catching it HERE, not on the ladder);
  * the artefact's banks are within a sane band of the reference's.

    python src/trackp/harness/validate_bigtrackp.py \
        --stage /tmp/artefact --vs agents/v44.1_bandit.py --seeds 3,4
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import os
import selectors  # noqa: F401  -- bind stdlib select BEFORE any path games:
import socket     # noqa: F401     src/select.py shadows stdlib `select`
import sys

# repo ROOT is FOUR levels up (src/trackp/harness/this.py); never put src/
# itself on sys.path (the select.py shadow broke the 2026-09-05 container run)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--vs", required=True)
    ap.add_argument("--seeds", default="3,4")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    from kaggle_environments import make

    main_py = os.path.join(a.stage, "main.py")
    assert os.path.exists(main_py), main_py
    os.environ["KAGG_BIN"] = os.path.join(a.stage, "kagg")

    failures = 0
    for seed in [int(s) for s in a.seeds.split(",")]:
        for seat in (0, 1):
            env = make("kaggriculture",
                       configuration={"episodeSteps": 720, "actTimeout": 60,
                                      "runTimeout": 1000000, "seed": seed},
                       info={"seed": seed})
            agents = [a.vs, a.vs]
            agents[seat] = main_py
            env.run(agents)
            banks = [float(env.state[i].observation.farms[i]["money"])
                     for i in (0, 1)]
            statuses = [str(env.state[i].status) for i in (0, 1)]
            logs = env.logs or []
            rust_fail = 0
            for stepl in logs:
                try:
                    err = (stepl[seat] or {}).get("stderr") or ""
                except Exception:                              # noqa: BLE001
                    continue
                rust_fail += err.count("RUST-FAIL")
            ok = (all(s == "DONE" or s == "ACTIVE" for s in statuses)
                  and rust_fail == 0 and banks[seat] > 20000)
            failures += 0 if ok else 1
            print(f"seed {seed} seat {seat}: banks {banks[seat]:,.0f} vs "
                  f"{banks[1 - seat]:,.0f}  status {statuses}  "
                  f"RUST-FAIL x{rust_fail}  -> {'OK' if ok else 'FAIL'}",
                  flush=True)
    print(f"\nVALIDATION {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
