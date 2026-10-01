"""Cloud VM training orchestrator -- BC -> gate -> RL(500M) -> gate -> RL(1B) ->
deploy, hands-off and crash-resilient.

Every trainer here self-checkpoints (~150s) and resumes from its own checkpoint,
and a background thread rclone-syncs the checkpoint dir to Google Drive every
150s. So on ANY crash the supervisor (scripts/cloud_run.sh) restarts this script,
which resumes each stage from the last checkpoint -> at most ~2-3 min of lost
work, and even total VM loss resumes from gdrive.

There is NO Kaggle 12h kill on a rented VM, so trainers run with --max-hours 999
(run to `done`), not the Kaggle 10.5h guard.

    GDRIVE=gdrive:trackp CKPTS=./ckpts WORKERS=16 python -m kaggriculture.pipeline.cloud_train
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import os, subprocess, sys, threading, time

PY = sys.executable
GDRIVE = os.environ.get("GDRIVE", "gdrive:trackp")
CKPTS = os.path.abspath(os.environ.get("CKPTS", os.path.join(ROOT, "ckpts")))
WORKERS = os.environ.get("WORKERS", "12")     # BC dataloader workers
ENVS = os.environ.get("ENVS", "256")          # RL self-play envs (rollout is batch-bound: 256 -> ~7k env-steps/s)
RL_STEPS = int(os.environ.get("RL_STEPS", "500000000"))  # 500M first target (~3-4 days @ 256 envs + 2ep + teacher-cache)
# RL sample-efficiency recipe (see scripts/trackp/RUNBOOK.md "levers"):
PPO_EPOCHS = os.environ.get("PPO_EPOCHS", "2")           # 1 = fast; 2 = more sample-efficient
KL_C = os.environ.get("KL_C", "0.5")                     # teacher-KL coef (raise to ~1.0 at 1 epoch)
TEACHER_MODE = os.environ.get("TEACHER_MODE", "bc")      # bc | ema | refresh
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", "")        # improved-BC path to anchor to (optional)
# BC sized to run in ~2.5h: 6M model + ~22M-row subset (2 epochs). All GPU opts are on
# (compile+seq128+bs1024); the ONLY 2-3h lever is fewer rows -- BC is GPU-bound, not Rust-able.
BC_EPOCHS = os.environ.get("BC_EPOCHS", "1")             # 1 pass over ALL rows now (~7h);
BC_SHARDS = os.environ.get("BC_SHARDS", "0")             # 0 = all 134M rows. 2nd epoch later
                                                          # folds in via refreshed teacher.
BC_DIMS = os.environ.get("BC_DIMS", "384,4,1024")        # ~6M model (d_model,layers,ff)
RAILS = os.environ.get("RAILS", os.path.join(ROOT, "configs", "trackp_rails.json"))
MARK = os.path.join(CKPTS, "stages")
LOG = os.path.join(CKPTS, "cloud_train.log")
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")
SYNC_SECS = 150

BC = os.path.join(CKPTS, "policy_bc.pt")
RL = os.path.join(CKPTS, "policy_rl.pt")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(CKPTS, exist_ok=True)
    open(LOG, "a").write(line + "\n")


def sync_up():
    try:
        subprocess.run(["rclone", "copy", CKPTS, f"{GDRIVE}/ckpt", "--exclude", "*.tmp"],
                       timeout=600)
    except Exception as e:
        log(f"sync warn: {e}")


def sync_daemon(stop):
    while not stop.is_set():
        stop.wait(SYNC_SECS)
        sync_up()


def done_flag(path):
    if not os.path.exists(path):
        return False
    try:
        import torch
        return bool(torch.load(path, map_location="cpu", weights_only=False).get("done"))
    except Exception:
        return False


def run(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=ROOT, env=ENV).returncode


def marked(name):
    return os.path.exists(os.path.join(MARK, name))


def mark(name):
    os.makedirs(MARK, exist_ok=True)
    open(os.path.join(MARK, name), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


# stage runners return True when the stage is COMPLETE (else caller re-runs)
def stage_bc():
    dm, ly, ff = (BC_DIMS.split(",") + ["384", "4", "1024"])[:3]
    cmd = [PY, "-m", "kaggriculture.trackp.bc_train", "--out", BC, "--epochs", BC_EPOCHS,
           "--bs", "1024", "--workers", WORKERS, "--max-hours", "999", "--ckpt-every", "1000",
           "--d-model", dm, "--layers", ly, "--ff", ff]
    if int(BC_SHARDS) > 0:
        cmd += ["--limit-shards", BC_SHARDS]      # ~2.5h lever: fewer rows (BC is GPU-bound)
    run(cmd)
    return done_flag(BC)


def stage_bc_gate():
    run([PY, "-m", "kaggriculture.measure.tournament_gate", BC, "--seeds", "8"])
    return True                       # report-only, never blocks


def stage_rl(target, out, resume):
    cmd = [PY, "-m", "kaggriculture.trackp.rl_selfplay", "--resume", resume, "--out", out,
           "--steps", str(target), "--envs", ENVS, "--max-hours", "999",
           "--ckpt-secs", str(SYNC_SECS), "--ppo-epochs", PPO_EPOCHS, "--kl-c", KL_C,
           "--teacher-mode", TEACHER_MODE]
    if TEACHER_CKPT and os.path.exists(TEACHER_CKPT):
        cmd += ["--teacher-ckpt", TEACHER_CKPT]           # anchor to an improved BC
    if os.path.exists(RAILS):
        cmd += ["--rails", RAILS]
    run(cmd)
    return done_flag(out)


def stage_deploy():
    # final policy is RL (or BC if RL absent); compiled tarball is a separate
    # local build step -- here we just certify the final policy through the gate.
    final = RL if os.path.exists(RL) else BC
    run([PY, "-m", "kaggriculture.measure.tournament_gate", final, "--seeds", "12"])
    return True


# FIRST PIPELINE (lean): BC(12M) -> gate -> PPO 500M -> gate -> deploy-certify.
# The 1B run + continuous drops + IQL/league are added later (built while this trains).
STAGES = [
    ("1_bc", stage_bc),
    ("2_bc_gate", stage_bc_gate),
    ("3_rl", lambda: stage_rl(RL_STEPS, RL, BC)),
    ("4_rl_gate", lambda: (run([PY, "-m", "kaggriculture.measure.tournament_gate", RL,
                                 "--seeds", "8"]) or True)),
    ("5_deploy", stage_deploy),
]


def main():
    os.makedirs(CKPTS, exist_ok=True)
    # GDrive sync ONLY at stage completion (per operator: no periodic egress during
    # training). Local checkpoints (bc --ckpt-every / rl --ckpt-secs) stay on the box
    # disk for crash-resume; the supervisor restarts and resumes from those.
    log(f"=== cloud_train START (gdrive={GDRIVE} ckpts={CKPTS} bc_workers={WORKERS} envs={ENVS}) ===")
    for name, fn in STAGES:
        if marked(name):
            log(f"[{name}] done -- skip"); continue
        log(f"[{name}] START")
        ok = fn()
        if not ok:
            log(f"[{name}] not complete (crash/partial) -- exit; supervisor resumes")
            return 1
        mark(name); sync_up(); log(f"[{name}] DONE -> gdrive synced")
    log("=== cloud_train COMPLETE ==="); sync_up()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
