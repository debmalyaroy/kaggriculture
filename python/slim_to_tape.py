"""Slim-corpus episodes -> tapes for `tapeplay` (same tape JSON as replay_to_tape.py).

    python python/slim_to_tape.py data/slim/s1/source=official/date=2026-09-24 --out DIR [--n 50]

A slim record keeps both seats' raw actions per step (steps[t].a = the pair that produced state
t) and info.seed, so any corpus episode replays exactly in `tapeplay --verify`.
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="a slim parquet file or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    files = [a.src] if a.src.endswith(".parquet") else sorted(glob.glob(os.path.join(a.src, "**", "*.parquet"), recursive=True))
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for f in files:
        t = pq.ParquetFile(f).read(columns=["episode_id", "slim"])
        for eid, js in zip(t.column("episode_id").to_pylist(), t.column("slim").to_pylist()):
            if js is None:
                continue
            d = json.loads(js)
            steps = d["steps"]
            acts = [list(steps[t + 1].get("a") or [None, None]) for t in range(len(steps) - 1)]
            acts = [[x if isinstance(x, dict) else None for x in (p + [None, None])[:2]] for p in acts]
            tape = {"id": eid, "seed": d["info"]["seed"], "seat": 0, "rewards": d.get("rewards"), "actions": acts}
            with open(os.path.join(a.out, f"{eid}.json"), "w", encoding="utf-8") as fh:
                json.dump(tape, fh)
            n += 1
            if n >= a.n:
                print(f"wrote {n} tapes -> {a.out}")
                return
    print(f"wrote {n} tapes -> {a.out}")


if __name__ == "__main__":
    main()
