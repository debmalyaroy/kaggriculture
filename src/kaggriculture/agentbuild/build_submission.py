"""Package an agent as a Kaggle submission (`build/main.py`).

Verifies the agent is legal and fast before copying, so a submission that would
be silently penalised never leaves the machine.

Usage:
    python -m kaggriculture.agentbuild.build_submission --agent agents/v2_tuned.py
    python -m kaggriculture.agentbuild.build_submission --agent agents/v2_tuned.py --skip-tests
"""
from kaggriculture.paths import ROOT
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="agents/v2_tuned.py")
    ap.add_argument("--out", default="build")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    src = os.path.join(ROOT, args.agent)
    if not os.path.exists(src):
        sys.exit(f"no such agent: {src}")

    if not args.skip_tests:
        import test_agents
        print(f"validating {args.agent} ...")
        r = test_agents.run_episode(args.agent, opponent="starter", seed=11)
        assert r["status"] == "DONE", f"episode ended {r['status']}"
        assert r["worst_turn_s"] < 0.5, f"worst turn {r['worst_turn_s']:.3f}s too slow"
        print(f"  ok: ${r['reward']:,.0f}  mean {r['mean_turn_s']*1000:.1f} ms  "
              f"worst {r['worst_turn_s']*1000:.1f} ms")

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "main.py")
    shutil.copy(src, dst)
    size = os.path.getsize(dst)
    print(f"\nwrote {dst}  ({size:,} bytes, from {args.agent})")
    print("submit with:\n"
          f"  kaggle competitions submit kaggriculture -f {args.out}/main.py -m \"<message>\"")


if __name__ == "__main__":
    main()
