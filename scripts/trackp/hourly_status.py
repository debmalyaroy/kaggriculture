#!/usr/bin/env python
"""(6b) HOURLY STATUS  --  box job status + live Kaggle submission performance.

Read-only. Two blocks:
  1. Remote box training status (delegates to box_status.py: stage / progress% /
     sps / ETA / GPU/RAM / alive procs).
  2. Kaggle submission performance -- the N most recent submissions (default 2,
     which is exactly the pool the ladder keeps active), status + publicScore.
     Printed only if `kaggle competitions submissions` returns any row.

Designed to be fired hourly by a scheduler; also runnable by hand:

    python scripts/trackp/hourly_status.py
    python scripts/trackp/hourly_status.py --rl-target 1000000000 --subs 3

NEVER submits. `kaggle competitions submissions` is a read-only list call.
"""
from __future__ import annotations
import argparse, csv, io, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "kaggriculture"


def _box(rl_target: int) -> str:
    """Run box_status.py and return its captured block (SSH status pull)."""
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts/trackp", "box_status.py"),
             "--rl-target", str(rl_target)],
            capture_output=True, text=True, timeout=90)
        out = (r.stdout or "").strip()
        return out or (r.stderr or "no response from box").strip()
    except subprocess.TimeoutExpired:
        return "box status: SSH timed out (box unreachable or busy)"
    except Exception as e:
        return f"box status: error ({e})"


def _subs(n: int) -> str:
    """The n most-recent Kaggle submissions, newest first. Read-only list call."""
    try:
        r = subprocess.run(["kaggle", "competitions", "submissions", COMP, "-v"],
                           capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return "kaggle CLI not found -- pip install kaggle (skipping submission perf)"
    except subprocess.TimeoutExpired:
        return "kaggle submissions: timed out"
    except Exception as e:
        return f"kaggle submissions: error ({e})"
    txt = (r.stdout or "").strip()
    if not txt or "," not in txt:
        return "no submissions found" if r.returncode == 0 else (r.stderr or "kaggle error").strip()
    rows = list(csv.DictReader(io.StringIO(txt)))
    if not rows:
        return "no submissions found"
    lines = []
    for row in rows[:n]:
        ref = row.get("ref", "?")
        date = (row.get("date", "") or "")[:16]
        status = (row.get("status", "") or "").replace("SubmissionStatus.", "")
        score = row.get("publicScore") or "-"
        desc = (row.get("description", "") or "").strip().replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        lines.append(f"  #{ref}  {date}  {status:<8}  public={score:<8}  {desc}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rl-target", type=int, default=1_000_000_000)
    ap.add_argument("--subs", type=int, default=2, help="how many recent submissions to show")
    a = ap.parse_args()

    print("#" * 64)
    print(_box(a.rl_target))
    print("-" * 64)
    print(f"Kaggle submissions (latest {a.subs}):")
    print(_subs(a.subs))
    print("#" * 64)


if __name__ == "__main__":
    main()
