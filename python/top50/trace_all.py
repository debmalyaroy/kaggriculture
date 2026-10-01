"""Exact replay + per-step state of every fetched top-50 tape (queue Q37, follows Q36).

    python python/top50/trace_all.py [--root data/top50/fetch] [--threads 16]

Runs `trace --seats tape` (crates/runner/src/bin/trace.rs) on each tapes/sNN shard -> trace/sNN.tsv, and
reports how many games replay exactly (the engine must reproduce the recorded banks to the dollar).
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TRACE = os.path.join(BIN, "trace" + (".exe" if os.name == "nt" else ""))

MIN_FREE_GB = float(os.environ.get("KRL_MIN_FREE_GB", "15"))


def disk_ok(path):
    """Stop before the box's disk gets low (operator rule: never overflow it)."""
    import shutil as _sh
    free = _sh.disk_usage(path).free / 2 ** 30
    if free < MIN_FREE_GB:
        print(f"[disk] only {free:.1f} GB free under {path} (< {MIN_FREE_GB:.0f} GB): stopping", flush=True)
        return False
    return True



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RL, "data", "top50", "fetch"))
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    os.makedirs(os.path.join(root, "trace"), exist_ok=True)
    shards = sorted(d for d in os.listdir(os.path.join(root, "tapes")) if d.startswith("s"))

    def one(sh):
        if not disk_ok(root):
            return f"[trace] {sh}: skipped (disk)"
        r = subprocess.run([TRACE, "--tapes", os.path.join(root, "tapes", sh), "--out", os.path.join(root, "trace", f"{sh}.tsv"),
                            "--seats", "tape"], capture_output=True, text=True)
        return r.stderr.strip()

    n = ex_ = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for line in ex.map(one, shards):
            print(line, flush=True)
            try:
                parts = line.split()
                n += int(parts[1])
                ex_ += int(parts[3])
            except (IndexError, ValueError):
                pass
    print(f"[trace_all] {n} tapes, {ex_} exact ({100.0 * ex_ / max(1, n):.1f}%)")


if __name__ == "__main__":
    main()
