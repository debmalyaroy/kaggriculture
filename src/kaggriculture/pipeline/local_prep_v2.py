"""Local v2 prep (runs on this box, before renting the VM): wait for the Rust
re-extract, finalize, self-play (16 agents), combine, then push corpus+code to
Google Drive so the VM can pull it. Resumable via markers; safe to relaunch.

    GDRIVE=gdrive:trackp python -m kaggriculture.pipeline.local_prep_v2
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import glob, os, shutil, subprocess, sys, time

PY = sys.executable
GDRIVE = os.environ.get("GDRIVE", "gdrive:trackp")
CORPUS = os.path.join(ROOT, "data", "trackp_corpus_v2")
SELFPLAY = os.path.join(ROOT, "data", "trackp_corpus_v2_selfplay")
OURGAMES = os.path.join(ROOT, "data", "trackp_corpus_v2_ourgames")
EXTRA = os.path.join(ROOT, ".local", "trackp_selfplay_extra")
RUST_BIN = os.path.join(ROOT, "rustengine", "extractor", "target", "release",
                        "trackp-extract.exe")
MARK = os.path.join(ROOT, ".local", "scratch", "trackp_lane", "prep_v2")
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def marked(n):
    return os.path.exists(os.path.join(MARK, n))


def mark(n):
    os.makedirs(MARK, exist_ok=True)
    open(os.path.join(MARK, n), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))


def run(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=ROOT, env=ENV).returncode


def extract_alive():
    try:
        import psutil
        return any("trackp-extract" in (p.info["name"] or "").lower()
                   for p in psutil.process_iter(["name"]))
    except Exception:
        return False


def stage_wait_extract():
    log("waiting for Rust v2 re-extract to finish (sustained-absence check) ...")
    gone = 0
    while True:
        if extract_alive():
            gone = 0
        else:
            gone += 1
            if gone >= 3:            # 3 polls (~3min) w/o extractor -> truly done,
                break                # not a momentary restart gap
        time.sleep(60)
    n = len(glob.glob(os.path.join(CORPUS, "shard_*.parquet")))
    log(f"re-extract finished: {n} shards in {CORPUS}")
    return n > 0


def stage_gm_delta():
    """Incremental Rust extract into the SAME dir: dedup skips the ~done base
    episodes and appends the fresh gm delta (replays_2026-09e + updated CSVs).
    Safe now that the base extract has finished (no concurrent shard_r writer)."""
    if not os.path.exists(RUST_BIN):
        log("rust extractor missing -- skip delta (base corpus still valid)"); return True
    while extract_alive():                       # never run concurrently with a base extract
        log("gm_delta: a base extract is still running -- waiting 60s")
        time.sleep(60)
    return run([RUST_BIN, "--gm", r"D:\gm_dataset", "--out", CORPUS, "--jobs", "2",
                "--rows-per-shard", "50000"]) == 0


def stage_finalize_gm():
    return run([PY, "-m", "kaggriculture.data.trackp_corpus", "--finalize", "--out", CORPUS]) == 0


def stage_selfplay():
    cmd = [PY, "-m", "kaggriculture.data.selfplay_corpus", "--top-reactive", "12",
           "--seeds", "24", "--both", "--out", SELFPLAY]
    if os.path.isdir(EXTRA):
        cmd += ["--extra-agents-dir", EXTRA]
    return run(cmd) == 0


def stage_combine():
    total = 0
    for src, tag in ((SELFPLAY, "sp"), (OURGAMES, "og")):
        shards = sorted(glob.glob(os.path.join(src, "shard_*.parquet")))
        for i, s in enumerate(shards):
            dst = os.path.join(CORPUS, f"shard_{tag}_{i:04d}.parquet")
            if not os.path.exists(dst):
                shutil.copy2(s, dst)
        log(f"folded {len(shards)} {tag} shards into corpus")
        total += len(shards)
    return run([PY, "-m", "kaggriculture.data.trackp_corpus", "--finalize", "--out", CORPUS]) == 0


def stage_push_gdrive():
    try:
        have = subprocess.run(["rclone", "version"], capture_output=True).returncode == 0
    except FileNotFoundError:            # rclone not on PATH (Windows raises, no returncode)
        have = False
    if not have:
        log("rclone not installed/configured -- SKIP push (upload manually). Corpus is at "
            f"{CORPUS} ; code under src/ scripts/ configs/ rustengine/ (excl target)."); return True
    run(["rclone", "copy", CORPUS, f"{GDRIVE}/corpus", "--transfers", "8", "--progress"])
    # stage code (src + scripts + configs + rustengine sources + vendor) for the VM
    for sub in ("src", "scripts", "configs", "tests", "agents", "vendor"):
        p = os.path.join(ROOT, sub)
        if os.path.isdir(p):
            run(["rclone", "copy", p, f"{GDRIVE}/code/{sub}"])
    run(["rclone", "copy", os.path.join(ROOT, "rustengine"), f"{GDRIVE}/code/rustengine",
         "--exclude", "target/**", "--exclude", "*.exe"])
    log(f"pushed corpus + code to {GDRIVE}")
    return True


STAGES = [("1_wait", stage_wait_extract), ("2_gm_delta", stage_gm_delta),
          ("3_finalize", stage_finalize_gm), ("4_selfplay", stage_selfplay),
          ("5_combine", stage_combine), ("6_push", stage_push_gdrive)]


def main():
    log(f"=== local_prep_v2 START (gdrive={GDRIVE}) ===")
    for name, fn in STAGES:
        if marked(name):
            log(f"[{name}] done -- skip"); continue
        log(f"[{name}] START")
        if not fn():
            log(f"[{name}] FAILED -- stop (rerun to resume)"); return 1
        mark(name); log(f"[{name}] DONE")
    log("=== local_prep_v2 COMPLETE -- corpus+code on gdrive; ready to rent ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
