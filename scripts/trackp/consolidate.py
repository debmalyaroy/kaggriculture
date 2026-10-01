#!/usr/bin/env python
"""(9) CONSOLIDATION / DISTILLATION POLISH.

The "add epochs back later" trick: run the bulk of RL at --ppo-epochs 1 (fast),
then, before a deadline drop, take the latest checkpoint and do a SHORT run with
MORE gradient passes per rollout (epochs 3-4) + a firmer teacher anchor. This
recovers the sample-efficiency a 2nd epoch would have given -- but only on the
FINAL policy, where it matters -- for a few M cheap steps.

Writes to a SEPARATE file (…_polished.pt) so the main lineage is never clobbered;
gate the polished vs the raw checkpoint before submitting.

    python scripts/trackp/consolidate.py                       # polish latest ckpts/policy_rl.pt
    python scripts/trackp/consolidate.py --resume ckpts/policy_rl.pt --budget 10000000 --epochs 4
    python scripts/trackp/consolidate.py --teacher ema          # anchor to a slow copy of self
"""
from __future__ import annotations
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")


def _gstep(ckpt):
    import torch
    try:
        return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("global_step", 0))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resume", default=os.path.join(ROOT, "ckpts", "policy_rl.pt"),
                    help="checkpoint to polish (default: the live RL ckpt)")
    ap.add_argument("--out", default=None, help="output (default: <resume>_polished.pt)")
    ap.add_argument("--budget", type=int, default=10_000_000, help="extra env-steps of polish")
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs during polish (3-4 typical)")
    ap.add_argument("--kl-c", type=float, default=0.8, help="firmer teacher anchor during polish")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--teacher", choices=["bc", "ema", "refresh"], default="bc")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="anchor the polish to this checkpoint (e.g. an improved local BC)")
    a = ap.parse_args()
    if not os.path.exists(a.resume):
        raise SystemExit(f"checkpoint not found: {a.resume}")
    out = a.out or a.resume.replace(".pt", "_polished.pt")
    start = _gstep(a.resume)
    target = start + a.budget                       # rl_selfplay's --steps is ABSOLUTE gstep
    print(f"[consolidate] {a.resume} @ {start:,} -> +{a.budget:,} steps "
          f"(epochs={a.epochs}, kl_c={a.kl_c}, teacher={a.teacher}) -> {out}")
    # seed the polished file from the resume so it continues that lineage's gstep
    import shutil
    shutil.copy2(a.resume, out)
    cmd = [PY, "-m", "kaggriculture.trackp.rl_selfplay", "--resume", out, "--out", out,
           "--steps", str(target), "--envs", str(a.envs), "--ppo-epochs", str(a.epochs),
           "--kl-c", str(a.kl_c), "--teacher-mode", a.teacher,
           "--max-hours", "999", "--ckpt-secs", "120"]
    if a.teacher_ckpt:
        cmd += ["--teacher-ckpt", a.teacher_ckpt]
    rc = subprocess.run(cmd, cwd=ROOT, env=ENV).returncode
    if rc != 0:
        raise SystemExit(f"polish run failed rc={rc}")
    print("\n" + "=" * 60)
    print(f"POLISHED: {out}")
    print("Gate it vs the raw checkpoint before submitting, e.g.:")
    print(f"  python scripts/trackp/submission_build.py --weights {os.path.basename(out)}")
    print(f"  (compare against {os.path.basename(a.resume)} via the panel gate)")
    print("=" * 60)


if __name__ == "__main__":
    main()
