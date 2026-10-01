"""Parity gate: the Rust extractor's token/action blobs must be BYTE-IDENTICAL
to the Python `trackp_corpus` encoder on the same real replays.

Strategy (memory-frugal, safe to run while the Python build is going): the
already-built Python corpus in ``data/trackp_corpus`` IS the reference (it is
literally `encode_tokens`/`encode_action` output). We pick a few episodes that
are wholly contained inside one existing Python shard, have Rust re-extract
exactly those (``--only``), and assert every (episode, seat, step) row matches
on tokens bytes, action bytes, and all metadata columns.

    C:/ProgramData/anaconda3/envs/llm/python.exe tests/test_trackp_rust_parity.py

Env knobs: TRACKP_RUST_BIN (extractor exe), TRACKP_PY_CORPUS (reference dir).
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_CORPUS = os.environ.get("TRACKP_PY_CORPUS", os.path.join(ROOT, "data", "trackp_corpus"))
RUST_BIN = os.environ.get(
    "TRACKP_RUST_BIN",
    os.path.join(ROOT, "rustengine", "extractor", "target", "release", "trackp-extract.exe"))
GM = os.environ.get("GM_DATASET", r"D:\gm_dataset")
TOK_W = 14
ACT_W = 56
N_EPISODES = 5

KEYCOLS = ("day", "split", "world_family", "n_tokens")  # step is in the row key


def _rows_by_key(table):
    """Map (episode_id, seat, step) -> full row dict from a corpus parquet table."""
    d = table.to_pydict()
    out = {}
    n = table.num_rows
    for i in range(n):
        key = (d["episode_id"][i], int(d["seat"][i]), int(d["step"][i]))
        out[key] = {
            "day": int(d["day"][i]), "split": int(d["split"][i]),
            "world_family": int(d["world_family"][i]), "n_tokens": int(d["n_tokens"][i]),
            "rtg": float(d["rtg"][i]), "rating": float(d["rating"][i]),
            "tokens": d["tokens"][i], "action": d["action"][i],
        }
    return out


def _pick_interior_episodes(shard_path, k):
    """Episodes wholly inside a shard = every episode except the first/last by
    first-appearance order (only boundary episodes can be split across shards)."""
    t = pq.read_table(shard_path, columns=["episode_id"])
    ids = t.column("episode_id").to_pylist()
    order = []
    seen = set()
    for e in ids:
        if e not in seen:
            seen.add(e)
            order.append(e)
    interior = order[1:-1]
    if len(interior) < k:
        raise SystemExit(f"shard {shard_path} has too few interior episodes "
                         f"({len(interior)}); pick a fuller shard")
    return interior[:k]


def main():
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"FAIL: Rust binary not built at {RUST_BIN}")
    shards = sorted(glob.glob(os.path.join(PY_CORPUS, "shard_*.parquet")))
    if not shards:
        raise SystemExit(f"FAIL: no Python reference shards in {PY_CORPUS}")
    # a stable, complete, not-being-written shard: oldest by mtime
    shards.sort(key=os.path.getmtime)
    ref_shard = shards[0]
    eids = _pick_interior_episodes(ref_shard, N_EPISODES)
    print(f"[parity] ref shard: {os.path.basename(ref_shard)}")
    print(f"[parity] episodes: {eids}")

    # reference rows for exactly those episodes (may span this + next shard)
    ref = {}
    eid_set = set(eids)
    ref_sources = [shards[0]] + ([shards[1]] if len(shards) > 1 else [])
    for sp in ref_sources:
        for key, row in _rows_by_key(pq.read_table(sp)).items():
            if key[0] in eid_set:
                ref[key] = row
    print(f"[parity] reference rows: {len(ref)}")

    # Rust extracts exactly those episodes into a temp dir (or reuse an existing
    # output dir via TRACKP_RUST_OUT to skip the slow re-scan).
    reuse = os.environ.get("TRACKP_RUST_OUT")
    if reuse and glob.glob(os.path.join(reuse, "shard_*.parquet")):
        out = reuse
        print(f"[parity] reusing rust output: {out}")
    else:
        out = tempfile.mkdtemp(prefix="trackp_rust_parity_")
        cmd = [RUST_BIN, "--gm", GM, "--out", out, "--jobs", "1",
               "--only", ",".join(str(e) for e in eids)]
        print(f"[parity] running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit(f"FAIL: rust extractor exited {r.returncode}")
    rust_shards = sorted(glob.glob(os.path.join(out, "shard_*.parquet")))
    if not rust_shards:
        print(r.stderr[-2000:])
        raise SystemExit("FAIL: rust wrote no shard")
    rust = {}
    for sp in rust_shards:
        rust.update(_rows_by_key(pq.read_table(sp)))
    print(f"[parity] rust rows: {len(rust)}")

    # compare
    ref_keys = {k for k in ref if k[0] in eid_set}
    rust_keys = {k for k in rust if k[0] in eid_set}
    if ref_keys != rust_keys:
        only_ref = sorted(ref_keys - rust_keys)[:10]
        only_rust = sorted(rust_keys - ref_keys)[:10]
        raise SystemExit(f"FAIL: row-key mismatch. only-ref={only_ref} only-rust={only_rust} "
                         f"(ref={len(ref_keys)} rust={len(rust_keys)})")

    tok_bad = act_bad = meta_bad = 0
    for key in sorted(ref_keys):
        a, b = ref[key], rust[key]
        if a["tokens"] != b["tokens"]:
            tok_bad += 1
            if tok_bad <= 3:
                pa_ = np.frombuffer(a["tokens"], np.int32).reshape(-1, TOK_W)
                pb_ = np.frombuffer(b["tokens"], np.int32).reshape(-1, TOK_W)
                diff = np.argwhere(pa_ != pb_)[:5]
                print(f"  TOKENS differ at {key}: {len(diff)} cells e.g. {diff.tolist()}")
                for (ti, tj) in diff[:3]:
                    print(f"    tok[{ti},{tj}] py={pa_[ti, tj]} rust={pb_[ti, tj]}  "
                          f"pyrow={pa_[ti].tolist()} rustrow={pb_[ti].tolist()}")
        if a["action"] != b["action"]:
            act_bad += 1
            if act_bad <= 3:
                pa_ = np.frombuffer(a["action"], np.int32)
                pb_ = np.frombuffer(b["action"], np.int32)
                diff = np.argwhere(pa_ != pb_).ravel()[:8]
                print(f"  ACTION differ at {key}: idx {diff.tolist()} "
                      f"py={pa_[diff].tolist()} rust={pb_[diff].tolist()}")
        for c in KEYCOLS:
            if a[c] != b[c]:
                meta_bad += 1
                if meta_bad <= 5:
                    print(f"  META {c} differ at {key}: py={a[c]} rust={b[c]}")
        if abs(a["rtg"] - b["rtg"]) > 1e-6 or abs(a["rating"] - b["rating"]) > 1e-2:
            meta_bad += 1
            if meta_bad <= 5:
                print(f"  FLOAT differ at {key}: rtg py={a['rtg']} rust={b['rtg']} "
                      f"rating py={a['rating']} rust={b['rating']}")

    print(f"[parity] compared {len(ref_keys)} rows: tokens_bad={tok_bad} "
          f"action_bad={act_bad} meta_bad={meta_bad}")
    if tok_bad or act_bad or meta_bad:
        raise SystemExit("FAIL: Rust extractor is NOT byte-identical to Python")
    print("[parity] PASS -- Rust extractor is byte-identical to Python encoder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
