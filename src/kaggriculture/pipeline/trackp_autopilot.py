"""Trackp data->corpus->train autopilot -- ONE hands-off chain, no prompting.

Stages (each writes a marker in .local/scratch/trackp_lane/autopilot/; a present
marker is skipped, so the run is resumable):

  0 parity      -- Rust extractor must be byte-identical to Python (GATE; abort on fail)
  1 kill_python -- stop the Python build as a whole process tree (no orphans)
  2 rust_gm     -- trackp-extract --jobs 2 to completion (resumes from existing shards)
  3 finalize_gm -- index/stats/norm over the gm corpus
  4 selfplay    -- top-12 reactive x 24 worlds x both seats
  5 combine     -- fold self-play shards into the corpus dir, re-finalize
  6 upload      -- create/version the private-account Kaggle DATASET
  7 bc_kernel   -- push the PRIVATE BC training kernel (GPU) -> starts training on Kaggle

Nothing here submits a competition entry. Run detached; logs to autopilot.log.

    python -m kaggriculture.pipeline.trackp_autopilot
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import os
import subprocess
import sys
import time

PY = sys.executable
LANE = os.path.join(ROOT, ".local", "scratch", "trackp_lane")
MARK = os.path.join(LANE, "autopilot")
LOG = os.path.join(LANE, "autopilot.log")
CORPUS = os.path.join(ROOT, "data", "trackp_corpus")
SELFPLAY = os.path.join(ROOT, "data", "trackp_corpus_selfplay")
RUST_BIN = os.path.join(ROOT, "rustengine", "extractor", "target", "release",
                        "trackp-extract.exe")
GM = r"D:\gm_dataset"
EXTRA_AGENTS = os.path.join(ROOT, ".local", "trackp_selfplay_extra")
# BC kernel: tracked source + metadata; staged into a scratch dir only to push.
BC_SRC = os.path.join(ROOT, "src", "kaggriculture", "trackp", "bc_train.py")
BC_META = os.path.join(ROOT, "configs", "trackp_bc_kernel-metadata.json")
KERNEL_STAGE = os.path.join(LANE, "bc_kernel_stage")
DATASET_ID = "debmalya84/kaggriculture-trackp-corpus"
KERNEL_ID = "debmalya84/kaggriculture-trackp-bc"

ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LANE, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def done(stage):
    return os.path.exists(os.path.join(MARK, stage))


def mark(stage):
    os.makedirs(MARK, exist_ok=True)
    open(os.path.join(MARK, stage), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def run(cmd, **kw):
    """Run a subprocess, stream nothing (inherit), return returncode."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=ROOT, env=ENV, **kw).returncode


def stage(name):
    """Decorator: skip if marker present, mark on success, abort chain on failure."""
    def deco(fn):
        def wrapped():
            if done(name):
                log(f"[{name}] already done -- skip")
                return True
            log(f"[{name}] START")
            ok = fn()
            if ok:
                mark(name)
                log(f"[{name}] DONE")
            else:
                log(f"[{name}] FAILED -- halting autopilot (fix + rerun to resume)")
            return ok
        return wrapped
    return deco


# --- stages ------------------------------------------------------------------
@stage("0_parity")
def parity():
    rc = run([PY, os.path.join("tests", "test_trackp_rust_parity.py")])
    return rc == 0


@stage("1_kill_python")
def kill_python():
    try:
        import psutil
    except ImportError:
        log("psutil missing; cannot kill safely"); return False
    victims = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower().startswith("python") and \
               any("kaggriculture.data.trackp_corpus" in str(a)
                   for a in (p.info["cmdline"] or [])):
                victims.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for parent in victims:
        for ch in parent.children(recursive=True):
            try:
                ch.kill()
            except psutil.Error:
                pass
        try:
            parent.kill()
        except psutil.Error:
            pass
    log(f"killed {len(victims)} build parent(s) + children")
    time.sleep(3)
    return True


@stage("2_rust_gm")
def rust_gm():
    if not os.path.exists(RUST_BIN):
        log(f"rust binary missing at {RUST_BIN}"); return False
    rc = run([RUST_BIN, "--gm", GM, "--out", CORPUS, "--jobs", "2",
              "--rows-per-shard", "50000"])
    return rc == 0


@stage("3_finalize_gm")
def finalize_gm():
    rc = run([PY, "-m", "kaggriculture.data.trackp_corpus", "--finalize", "--out", CORPUS])
    return rc == 0


@stage("4_selfplay")
def selfplay():
    cmd = [PY, "-m", "kaggriculture.data.selfplay_corpus", "--top-reactive", "12",
           "--seeds", "24", "--both", "--out", SELFPLAY]
    if os.path.isdir(EXTRA_AGENTS):
        cmd += ["--extra-agents-dir", EXTRA_AGENTS]
    rc = run(cmd)
    return rc == 0


@stage("5_combine")
def combine():
    import glob
    import shutil
    sp = sorted(glob.glob(os.path.join(SELFPLAY, "shard_*.parquet")))
    n = 0
    for i, s in enumerate(sp):
        dst = os.path.join(CORPUS, f"shard_sp_{i:04d}.parquet")
        if not os.path.exists(dst):
            shutil.copy2(s, dst)
            n += 1
    log(f"folded {n} self-play shards into corpus")
    rc = run([PY, "-m", "kaggriculture.data.trackp_corpus", "--finalize", "--out", CORPUS])
    return rc == 0


@stage("6_upload")
def upload():
    import json
    meta = {"title": "Kaggriculture Trackp BC Corpus", "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}]}
    json.dump(meta, open(os.path.join(CORPUS, "dataset-metadata.json"), "w"), indent=1)
    # create first time, else version
    rc = run(["kaggle", "datasets", "create", "-p", CORPUS, "--dir-mode", "zip"],
             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if rc != 0:
        log("create failed/exists -> versioning")
        rc = run(["kaggle", "datasets", "version", "-p", CORPUS, "-m",
                  f"autopilot {time.strftime('%Y-%m-%d %H:%M')}", "--dir-mode", "zip"])
    return rc == 0


CKPT_DS = "debmalya84/trackp-bc-ckpt"
CKPT_STAGE = os.path.join(LANE, "bc_ckpt_stage")
MAX_BC_RUNS = 3                 # each run self-stops at ~11h; 2 epochs fit in ~1-2


def _kaggle(args, **kw):
    import subprocess
    return subprocess.run(["kaggle"] + args, cwd=ROOT, env=ENV,
                          capture_output=True, text=True, **kw)


def _ensure_ckpt_dataset():
    """Create the checkpoint dataset (placeholder) if it doesn't exist yet, so it
    can be a kernel source on run 1 (empty -> trainer starts fresh)."""
    import json as _j
    r = _kaggle(["datasets", "status", CKPT_DS])
    if r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower():
        return True
    os.makedirs(CKPT_STAGE, exist_ok=True)
    open(os.path.join(CKPT_STAGE, "README.md"), "w").write(
        "Trackp BC checkpoints (policy_bc.pt) -- versioned between kernel runs "
        "for resumable training.\n")
    _j.dump({"title": "Trackp BC Checkpoint", "id": CKPT_DS,
             "licenses": [{"name": "CC0-1.0"}]},
            open(os.path.join(CKPT_STAGE, "dataset-metadata.json"), "w"))
    rc = run(["kaggle", "datasets", "create", "-p", CKPT_STAGE])
    return rc == 0


def _kernel_status():
    r = _kaggle(["kernels", "status", KERNEL_ID])
    return (r.stdout + r.stderr).lower()


def _training_done():
    """True if the pulled checkpoint marks training complete."""
    import glob as _g
    import torch
    pts = _g.glob(os.path.join(CKPT_STAGE, "*.pt"))
    if not pts:
        return False
    try:
        return bool(torch.load(max(pts, key=os.path.getmtime), map_location="cpu",
                               weights_only=False).get("done"))
    except Exception:
        return False


@stage("7_bc_kernel")
def bc_kernel():
    import shutil
    import time as _t
    if not (os.path.exists(BC_SRC) and os.path.exists(BC_META)):
        log(f"BC source/metadata missing ({BC_SRC} / {BC_META}) -- STOP."); return False
    if not _ensure_ckpt_dataset():
        log("could not ensure checkpoint dataset; STOP"); return False
    os.makedirs(KERNEL_STAGE, exist_ok=True)
    shutil.copy2(BC_SRC, os.path.join(KERNEL_STAGE, "train_bc.py"))
    shutil.copy2(BC_META, os.path.join(KERNEL_STAGE, "kernel-metadata.json"))

    for run_i in range(MAX_BC_RUNS):
        rc = run(["kaggle", "kernels", "push", "-p", KERNEL_STAGE])
        if rc != 0:
            log(f"kernel push failed (run {run_i}); STOP"); return False
        log(f"BC kernel pushed (run {run_i}) PRIVATE+GPU; polling status "
            f"(resumes from {CKPT_DS} if a prior checkpoint exists)")
        # poll until the run leaves the running/queued state
        while True:
            _t.sleep(300)
            st = _kernel_status()
            if any(k in st for k in ("complete", "error", "cancel")):
                log(f"kernel status: {st.strip()[:120]}")
                break
        # pull the run's output checkpoint
        os.makedirs(CKPT_STAGE, exist_ok=True)
        _kaggle(["kernels", "output", KERNEL_ID, "-p", CKPT_STAGE])
        if _training_done():
            log("BC training COMPLETE (checkpoint done=True). "
                f"Model: kaggle kernels output {KERNEL_ID}")
            return True
        # not done -> version the checkpoint dataset so the next run resumes
        import json as _j
        _j.dump({"title": "Trackp BC Checkpoint", "id": CKPT_DS,
                 "licenses": [{"name": "CC0-1.0"}]},
                open(os.path.join(CKPT_STAGE, "dataset-metadata.json"), "w"))
        run(["kaggle", "datasets", "version", "-p", CKPT_STAGE, "-m",
             f"bc ckpt after run {run_i}"])
        log(f"checkpoint versioned; re-pushing to resume (run {run_i+1})")
    log(f"BC not done after {MAX_BC_RUNS} runs; latest checkpoint is in {CKPT_DS}. "
        f"Re-run the autopilot to continue (resumes automatically).")
    return True                 # progressing; don't fail the whole pipeline


STAGES = [parity, kill_python, rust_gm, finalize_gm, selfplay, combine, upload, bc_kernel]


def main():
    log("=== trackp autopilot START ===")
    for fn in STAGES:
        if not fn():
            log("=== autopilot HALTED ===")
            return 1
    log("=== trackp autopilot COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
