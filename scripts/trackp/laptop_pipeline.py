#!/usr/bin/env python
"""(3) LAPTOP PIPELINE  --  sync weights -> BC on delta -> short RL -> STOP + show path.

Produces a decorrelated SIDE-LINEAGE candidate on the laptop, in parallel with the
box's main RL run. It does NOT touch the box. Steps:

  1. rclone-pull the current base weights from gdrive:kaggriculture/ckpt
  2. (optional) rclone-pull + extract the newest delta tar from gdrive:.../delta
  3. BC fine-tune the BASE on the delta corpus            -> policy_bc_delta.pt
  4. short local RL self-play warm-started from that BC   -> policy_rl_laptop.pt
  5. STOP and print the final weights path (YOU upload it to gdrive if you want it)

Laptop is memory-bound (RTX 4060 8 GB, RAM reaper ~2.6 GB): defaults are safe
(64 envs, 20M steps, 1 BC epoch). Raise with flags if you have headroom.

    python scripts/trackp/laptop_pipeline.py
    python scripts/trackp/laptop_pipeline.py --rl-steps 40000000 --envs 96
    python scripts/trackp/laptop_pipeline.py --skip-rl        # just refresh BC on delta
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")
GDRIVE = os.environ.get("GDRIVE", "gdrive:kaggriculture")
WORK = os.path.join(ROOT, ".local", "laptop")
DELTA = os.path.join(ROOT, "data", "trackp_delta")


def run(cmd, optional=False):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT, env=ENV).returncode
    if rc != 0 and not optional:
        raise SystemExit(f"step failed (rc={rc})")
    return rc


def rclone(*args, optional=True):
    if not _have("rclone"):
        print("  (rclone not found -- skipping gdrive step; place files manually)")
        return 1
    return run(["rclone", *args], optional=optional)


def _have(exe):
    from shutil import which
    return which(exe) is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="policy_bc.pt",
                    help="which gdrive weight to fine-tune (policy_bc.pt | policy_rl.pt)")
    ap.add_argument("--pull-delta", action="store_true",
                    help="also pull+extract the newest delta tar from gdrive")
    ap.add_argument("--bc-epochs", type=int, default=1)
    ap.add_argument("--rl-steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--skip-rl", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    # 1) pull base weights from gdrive
    base_local = os.path.join(WORK, a.base)
    rclone("copyto", f"{GDRIVE}/ckpt/{a.base}", base_local)
    if not os.path.exists(base_local):
        raise SystemExit(f"base weights not found: {base_local} "
                         f"(pull {GDRIVE}/ckpt/{a.base} manually)")

    # 2) optional: pull + extract newest delta tar
    if a.pull_delta:
        os.makedirs(DELTA, exist_ok=True)
        tmp = os.path.join(WORK, "delta_tars")
        os.makedirs(tmp, exist_ok=True)
        rclone("copy", f"{GDRIVE}/delta", tmp, "--include", "*.tar.gz")
        tars = sorted(glob.glob(os.path.join(tmp, "*.tar.gz")))
        if tars:
            print(f"[delta] extracting {os.path.basename(tars[-1])}")
            with tarfile.open(tars[-1]) as tf:
                tf.extractall(DELTA)

    if not glob.glob(os.path.join(DELTA, "shard_*.parquet")):
        raise SystemExit(f"no delta shards in {DELTA} -- run delta_build.py or --pull-delta")

    # 3) BC fine-tune the base on the delta corpus
    bc_out = os.path.join(WORK, "policy_bc_delta.pt")
    run([PY, "-m", "kaggriculture.trackp.bc_train", "--out", bc_out,
         "--resume", base_local, "--corpus", DELTA, "--epochs", str(a.bc_epochs),
         "--bs", "512", "--workers", "4", "--ckpt-every", "500", "--max-hours", "999"])
    print(f"[bc-delta] -> {bc_out}")

    if a.skip_rl:
        _final(bc_out); return

    # 4) short local RL warm-started from the delta-BC
    rl_out = os.path.join(WORK, "policy_rl_laptop.pt")
    run([PY, "-m", "kaggriculture.trackp.rl_selfplay", "--resume", bc_out, "--out", rl_out,
         "--steps", str(a.rl_steps), "--envs", str(a.envs), "--max-hours", "999",
         "--ckpt-secs", "120"])
    _final(rl_out)


def _final(path):
    print("\n" + "=" * 66)
    print("LAPTOP LINEAGE COMPLETE -- weights ready.")
    print(f"PATH: {path}")
    print(f"metrics: {os.path.join(os.path.dirname(path), 'metrics.jsonl')}")
    print("Upload to gdrive:kaggriculture/ckpt/ if you want to gate/submit it.")
    print("=" * 66)


if __name__ == "__main__":
    main()
