#!/usr/bin/env python
"""LOCAL PIPELINE (trackp v2 delta flywheel).

Five stages, each runnable alone or chained with `all`:

  1. fetch    kaggle dataset georgymamarin/kaggriculture-episodes -> D:\\gm_dataset
              (ONLY new replay files -- name-diff vs local, byte-budget safe).
  2. delta    transform the NEW episodes into TRACKP CORPUS V2 shards under
              data/trackp_corpus_v2/delta/  (shard_<YYYYMMDD_HHmmss>_NNNN.parquet)
              and record the batch + training status in delta/index.json.
  3. selfplay top-N public agents x M worlds on the FAST serve path (+clone play)
              -> data/trackp_corpus_v2/self_play/ (same timestamped naming) and
              self_play/index.json.
  4. train    BC on the UNTRAINED batches from BOTH indices (skips already-trained),
              resuming the latest checkpoint in .local/checkpoints (kept in sync
              with gdrive:kaggriculture/ckpt), writing policy_bc_<YYYYMMDD_HHmmss>.pt.
  5. upload   push the new weight to gdrive:kaggriculture/ckpt.

GDRIVE: this laptop has no rclone and the operator rule is "you upload". So the
checkpoint sync (down) and the weight upload (up) are RCLONE-OPTIONAL: if rclone
is present they run; otherwise the pipeline prints the exact gdrive path/command
and expects you to place the base policy_bc.pt into .local/checkpoints yourself.

    python scripts/trackp/local_pipeline.py all --top-n 10 --seeds 10 --clone
    python scripts/trackp/local_pipeline.py fetch
    python scripts/trackp/local_pipeline.py delta   [--limit 5000]
    python scripts/trackp/local_pipeline.py selfplay --top-n 10 --seeds 10 --clone --jobs 12
    python scripts/trackp/local_pipeline.py train   [--epochs 1]
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")

GM = r"D:\gm_dataset"                                   # raw kaggle dataset (replays_*.parquet)
DATASET = "georgymamarin/kaggriculture-episodes"
V2 = os.path.join(ROOT, "data", "trackp_corpus_v2")    # base v2 corpus (shard_og_*)
DELTA_DIR = os.path.join(V2, "delta")
SELFPLAY_DIR = os.path.join(V2, "self_play")
BASE_NORM = os.path.join(V2, "norm.json")
CKPT_DIR = os.path.join(ROOT, ".local", "checkpoints")
GDRIVE = "gdrive:kaggriculture/ckpt"
BC_DIMS = ("384", "4", "1024")                         # MUST match the remote policy_bc.pt (6M model)


# ---------------------------------------------------------------- helpers ----
def _ts():
    return time.strftime("%Y%m%d_%H%M%S")


def _have(exe):
    return shutil.which(exe) is not None


def _run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=ENV, **kw)


def _load_index(dir_, kind):
    p = os.path.join(dir_, "index.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"kind": kind, "updated": None, "batches": [], "fetched_files": []}


def _save_index(dir_, idx):
    os.makedirs(dir_, exist_ok=True)
    idx["updated"] = _ts()
    json.dump(idx, open(os.path.join(dir_, "index.json"), "w"), indent=2)


def rclone_sync_down():
    """Mirror gdrive:kaggriculture/ckpt -> .local/checkpoints (rclone-optional)."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    if _have("rclone"):
        _run(["rclone", "copy", GDRIVE, CKPT_DIR, "--include", "*.pt"])
    else:
        print(f"  (no rclone) place the base weights yourself:\n"
              f"     rclone copy {GDRIVE} {CKPT_DIR} --include '*.pt'\n"
              f"     -- or download policy_bc.pt from {GDRIVE} into {CKPT_DIR}")


def rclone_upload(path):
    """Push one weight -> gdrive:kaggriculture/ckpt (rclone-optional)."""
    if _have("rclone"):
        _run(["rclone", "copy", path, GDRIVE])
        print(f"[upload] {os.path.basename(path)} -> {GDRIVE}")
    else:
        print(f"[upload] no rclone on this laptop -- upload it yourself:\n"
              f"     rclone copyto {path} {GDRIVE}/{os.path.basename(path)}\n"
              f"     (or drag {path} into the gdrive kaggriculture/ckpt folder)")


# ------------------------------------------------------------- 1. fetch ------
def cmd_fetch(a):
    os.makedirs(GM, exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "files", DATASET], capture_output=True,
                       text=True, env=ENV)
    remote = sorted(set(re.findall(r"replays_[0-9A-Za-z\-]+\.parquet", r.stdout)))
    if not remote:
        raise SystemExit(f"[fetch] no replay files listed (kaggle auth/access?):\n{r.stdout[:400]}")
    local = {os.path.basename(p) for p in glob.glob(os.path.join(GM, "replays_*.parquet"))}
    want = a.files.split(",") if a.files else [f for f in remote if f not in local]
    print(f"[fetch] remote={len(remote)} local={len(local)} -> download {len(want)}: {want}")
    got = []
    for f in want:
        cmd = ["kaggle", "datasets", "download", DATASET, "-f", f, "-p", GM]
        if a.force:
            cmd.append("--force")
        if _run(cmd).returncode != 0:
            print(f"  (skip {f}: download failed)"); continue
        z = os.path.join(GM, f + ".zip")
        if os.path.exists(z):
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(GM)
            os.remove(z)
        got.append(f)
    print(f"[fetch] added {len(got)} replay file(s) to {GM}")
    return got


# ------------------------------------------------------------- 2. delta ------
def cmd_delta(a):
    import kaggriculture.data.trackp_corpus as TC
    os.makedirs(DELTA_DIR, exist_ok=True)
    eng, banks = TC.load_lookups()
    keep = {e for e, v in eng.items() if v == TC.ENGINE_KEEP}     # current-engine episodes
    del eng
    done = TC._done_episodes(V2)[0] | TC._done_episodes(DELTA_DIR)[0]
    todo = keep - done                                            # NEW episodes only
    if a.limit:
        todo = set(list(todo)[:a.limit])
    print(f"[delta] engine-keep={len(keep)} already-ingested={len(done)} -> new={len(todo)}")
    if not todo:
        print("[delta] nothing new to transform."); return None
    ts = _ts(); prefix = f"shard_{ts}_"
    before = set(glob.glob(os.path.join(DELTA_DIR, prefix + "*.parquet")))
    nr, ne, proc = TC._stream_to_shards(todo, banks, DELTA_DIR, a.rows_per_shard,
                                        prefix, 0, limit=a.limit)
    shards = sorted(os.path.basename(p) for p in
                    set(glob.glob(os.path.join(DELTA_DIR, prefix + "*.parquet"))) - before)
    idx = _load_index(DELTA_DIR, "delta")
    idx["batches"].append({"ts": ts, "kind": "delta", "source": "kaggle:" + DATASET,
                           "shards": shards, "episodes": ne, "rows": nr, "trained_by": []})
    idx["fetched_files"] = sorted(os.path.basename(p) for p in
                                  glob.glob(os.path.join(GM, "replays_*.parquet")))
    _save_index(DELTA_DIR, idx)
    print(f"[delta] batch {ts}: {ne} episodes, {nr:,} rows -> {len(shards)} shards in {DELTA_DIR}")
    return ts


# ----------------------------------------------------------- 3. selfplay -----
def _selfplay_run(tmp, a, mirror):
    cmd = [PY, "-m", "kaggriculture.data.selfplay_corpus",
           "--top-reactive", str(a.top_n), "--seeds", str(a.seeds), "--both",
           "--jobs", str(a.jobs), "--out", tmp]
    if mirror:
        cmd.append("--mirror")
    _run(cmd)


def cmd_selfplay(a):
    os.makedirs(SELFPLAY_DIR, exist_ok=True)
    ts = _ts(); prefix = f"shard_{ts}_"
    moved = []
    runs = [("cross", False)] + ([("clone", True)] if a.clone else [])
    for tag, mirror in runs:
        tmp = tempfile.mkdtemp(prefix=f"sp_{tag}_", dir=os.path.join(ROOT, ".local", "scratch"))
        try:
            _selfplay_run(tmp, a, mirror)
            for k, src in enumerate(sorted(glob.glob(os.path.join(tmp, "shard_*.parquet")))):
                dst = os.path.join(SELFPLAY_DIR, f"{prefix}{tag}_{k:04d}.parquet")
                shutil.move(src, dst); moved.append(os.path.basename(dst))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not moved:
        print("[selfplay] no shards produced (empty pool?)"); return None
    # rows/episodes tally
    import pyarrow.parquet as pq
    rows = sum(pq.read_metadata(os.path.join(SELFPLAY_DIR, m)).num_rows for m in moved)
    idx = _load_index(SELFPLAY_DIR, "self_play")
    idx["batches"].append({"ts": ts, "kind": "self_play",
                           "source": {"top_n": a.top_n, "seeds": a.seeds, "clone": a.clone},
                           "shards": moved, "episodes": None, "rows": int(rows), "trained_by": []})
    _save_index(SELFPLAY_DIR, idx)
    print(f"[selfplay] batch {ts}: {rows:,} rows -> {len(moved)} shards in {SELFPLAY_DIR}")
    return ts


# ------------------------------------------------------------- 4. train ------
def _latest_ckpt():
    cks = sorted(glob.glob(os.path.join(CKPT_DIR, "policy_bc_*.pt")))
    if cks:
        return cks[-1]                              # ts-sorted == chronological
    base = os.path.join(CKPT_DIR, "policy_bc.pt")
    return base if os.path.exists(base) else None


def _untrained(dir_, kind):
    """[(index, batch)] whose trained_by is empty, + the resolved shard paths."""
    idx = _load_index(dir_, kind)
    out = []
    for b in idx["batches"]:
        if not b.get("trained_by"):
            paths = [os.path.join(dir_, s) for s in b["shards"] if os.path.exists(os.path.join(dir_, s))]
            if paths:
                out.append((b, paths))
    return idx, out


def cmd_train(a):
    rclone_sync_down()
    base = _latest_ckpt()
    if not base:
        raise SystemExit(f"[train] no base checkpoint in {CKPT_DIR}. Put policy_bc.pt there "
                         f"(from {GDRIVE}) or install rclone.")
    d_idx, d_batches = _untrained(DELTA_DIR, "delta")
    s_idx, s_batches = _untrained(SELFPLAY_DIR, "self_play")
    all_batches = d_batches + s_batches
    shards = [p for _, paths in all_batches for p in paths]
    if not shards:
        print("[train] no untrained batches in either index -- nothing to do."); return None
    ts = _ts(); out = os.path.join(CKPT_DIR, f"policy_bc_{ts}.pt")
    print(f"[train] resume {os.path.basename(base)} on {len(all_batches)} untrained batch(es), "
          f"{len(shards)} shards -> {os.path.basename(out)}")
    # stage untrained shards (hardlink; copy across volumes) + pin BASE norm/vocab
    stage = tempfile.mkdtemp(prefix="bc_stage_", dir=os.path.join(ROOT, ".local", "scratch"))
    try:
        for i, src in enumerate(shards):
            dst = os.path.join(stage, f"shard_{i:05d}.parquet")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        if os.path.exists(BASE_NORM):
            shutil.copy2(BASE_NORM, os.path.join(stage, "norm.json"))   # fine-tune: keep base norm
        v = os.path.join(V2, "vocab.json")
        if os.path.exists(v):
            shutil.copy2(v, os.path.join(stage, "vocab.json"))
        cmd = [PY, "-m", "kaggriculture.trackp.bc_train", "--corpus", stage,
               "--resume", base, "--out", out, "--epochs", str(a.epochs),
               "--bs", str(a.bs), "--workers", str(a.workers),
               "--d-model", BC_DIMS[0], "--layers", BC_DIMS[1], "--ff", BC_DIMS[2]]
        rc = _run(cmd).returncode
        if rc != 0 or not os.path.exists(out):
            raise SystemExit(f"[train] bc_train failed (rc={rc})")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    # mark batches trained + save both indices
    for b, _ in d_batches:
        b["trained_by"].append(os.path.basename(out))
    for b, _ in s_batches:
        b["trained_by"].append(os.path.basename(out))
    _save_index(DELTA_DIR, d_idx); _save_index(SELFPLAY_DIR, s_idx)
    print(f"[train] done -> {out}; marked {len(all_batches)} batch(es) trained_by={os.path.basename(out)}")
    rclone_upload(out)                              # 5. upload (rclone-optional)
    return out


# --------------------------------------------------------------- all ---------
def cmd_all(a):
    cmd_fetch(a)
    cmd_delta(a)
    cmd_selfplay(a)
    cmd_train(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--top-n", type=int, default=10, dest="top_n", help="top-N public agents (selfplay)")
        p.add_argument("--seeds", type=int, default=10, help="M worlds/seeds (selfplay)")
        p.add_argument("--clone", action="store_true", help="also add clone-vs-clone play")
        p.add_argument("--jobs", type=int, default=12, help="parallel serve workers")
        p.add_argument("--limit", type=int, default=0, help="cap #episodes (delta; 0=all new)")
        p.add_argument("--rows-per-shard", type=int, default=50_000)
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--bs", type=int, default=1024)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--files", default="", help="fetch: explicit comma-sep replay files")
        p.add_argument("--force", action="store_true", help="fetch: re-download named files")

    for name in ("fetch", "delta", "selfplay", "train", "all"):
        common(sub.add_parser(name))
    a = ap.parse_args()
    {"fetch": cmd_fetch, "delta": cmd_delta, "selfplay": cmd_selfplay,
     "train": cmd_train, "all": cmd_all}[a.stage](a)


if __name__ == "__main__":
    main()
