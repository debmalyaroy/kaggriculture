"""Put back the parts of PPO's training pool that a rebuild of data/tapes/train deletes (queue Q35).

    python python/restore_pool.py [--threads 16]

The delta job Q20 rebuilds data/tapes/train with band_tapes.py (an rmtree of the folder), which also
removes what other jobs installed inside it. On 26 Sep 16:24 IST that dropped the 1,348 top-player tapes,
the hard-tape links and our ladder games for hours. This runs after Q20 and Q22 (follow) and restores:
  1. train/top           every game rated 2700+ at the time, 15 Aug-17 Sep (band_tapes.py --top)
  2. train/hard_x2       the 796 hard tapes (data/tapes/hard: v63's losses and wins by < $3k) three times
                         (__l0, __l1, __l2), and the late hard tapes (data/tapes/hard_late) twice (__l1, __l2)
  3. train/ladder(+_copy_x) and data/tapes/ladder_val   python/ladder_split.py (our ladder games)
Each step is skipped when its output is already complete.
"""
import argparse
import glob
import os
import subprocess
import sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPES = os.path.join(RL, "data", "tapes")
TRAIN = os.path.join(TAPES, "train")


def link(src_dir, n_copies, first, out):
    os.makedirs(out, exist_ok=True)
    made = 0
    for f in sorted(glob.glob(os.path.join(src_dir, "*.json"))):
        base = os.path.basename(f)[:-5]
        for k in range(first, first + n_copies):
            dst = os.path.join(out, f"{base}__l{k}.json")
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(f), dst)
                made += 1
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    top = os.path.join(TRAIN, "top")
    if len(glob.glob(os.path.join(top, "**", "*.json"), recursive=True)) < 1000:
        subprocess.run([sys.executable, os.path.join(RL, "python", "band_tapes.py"), "--top", "--min-rating", "2700",
                        "--since", "2026-08-15", "--until", "2026-09-17", "--out", top], cwd=RL, check=True)
    print(f"[pool] train/top: {len(glob.glob(os.path.join(top, '**', '*.json'), recursive=True))} tapes", flush=True)
    hx = os.path.join(TRAIN, "hard_x2")
    n1 = link(os.path.join(TAPES, "hard"), 3, 0, hx)
    n2 = link(os.path.join(TAPES, "hard_late"), 2, 1, hx)
    print(f"[pool] train/hard_x2: +{n1} hard, +{n2} hard_late links; {len(os.listdir(hx))} in total", flush=True)
    if not os.path.isdir(os.path.join(TRAIN, "ladder")):
        subprocess.run([sys.executable, os.path.join(RL, "python", "ladder_split.py"), "--threads", str(a.threads)], cwd=RL, check=True)
    print(f"[pool] train/ladder: {len(glob.glob(os.path.join(TRAIN, 'ladder', '*.json')))} tapes, "
          f"ladder_copy_x {len(glob.glob(os.path.join(TRAIN, 'ladder_copy_x', '*.json')))} links", flush=True)
    lx = os.path.join(TRAIN, "losses_x")
    if not os.path.isdir(lx) and os.path.isdir(os.path.join(TAPES, "losses")):
        subprocess.run([sys.executable, os.path.join(RL, "python", "loss_tapes.py")], cwd=RL, check=True)
    for d in sorted(os.listdir(TRAIN)):
        p = os.path.join(TRAIN, d)
        if os.path.isdir(p):
            print(f"[pool]   {d}: {len(glob.glob(os.path.join(p, '**', '*.json'), recursive=True))}")


if __name__ == "__main__":
    main()
