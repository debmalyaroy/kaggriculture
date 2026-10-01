"""Publish this project's mined data as a Kaggle dataset, and keep it current.

Two different datasets matter here and it is worth not confusing them.

**Kaggle's own daily episode datasets already exist** and are mountable in a
notebook with no download at all:

    kaggle/kaggriculture-episodes-index          manifest of every day
    kaggle/kaggriculture-episodes-2026-08-05     ~740 replays, ~20 GB

Verify with `kaggle datasets files kaggle/kaggriculture-episodes-index`. They
are not easy to find by search because they are published under the `kaggle`
org rather than surfaced on the competition page.

**Ours does not exist until this tool makes it.** It carries the *derived*
artefacts a notebook needs and cannot recompute cheaply: the mined policy diff,
the field's market schedule, the top trajectories, our agent and its tapes.
Small (a few MB), so a version can be pushed after every mine.

    python -m kaggriculture.data.kaggle_dataset --create        # first time, private
    python -m kaggriculture.data.kaggle_dataset --update        # new version, with a note
    python -m kaggriculture.data.kaggle_dataset --status
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.episodes as E  # noqa: E402

STAGE = os.path.join(ROOT, "data", "dataset")
SLUG = "kaggriculture-ladder-mine"
TITLE = "Kaggriculture ladder mine"

# (source, name in the dataset). Missing sources are skipped, not fatal.
PAYLOAD = [
    ("data/policy_diff.csv", "policy_diff.csv"),
    ("data/market_schedule.csv", "market_schedule.csv"),
    ("data/top_trajectory.csv", "top_trajectory.csv"),
    ("data/episodes.csv", "episodes.csv"),
    ("data/notebooks.csv", "notebooks.csv"),
    ("data/mined_episodes.json", "mined_episodes.json"),
    ("agents/v9_cem.py", "agent_v9_cem.py"),
    ("docs/history/agent-v9.md", "agent_v9.md"),
    ("BUILD_JOURNAL.md", "BUILD_JOURNAL.md"),
]


def _safe(text):
    """Kaggle's CLI echoes dataset titles; Windows consoles are cp1252."""
    enc = sys.stdout.encoding or "utf-8"
    return str(text).encode(enc, "replace").decode(enc, "replace")


def _kaggle(args, timeout=900):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(E.kaggle_cmd() + args, capture_output=True,
                          timeout=timeout, env=env)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    return proc.returncode, (out + err).strip()


def username():
    for path in (os.path.expanduser("~/.kaggle/kaggle.json"),
                 os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json")):
        if path and os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))["username"]
            except Exception:                                      # noqa: BLE001
                pass
    return os.environ.get("KAGGLE_USERNAME", "USERNAME")


def stage(private=True):
    """Copy the payload into a clean folder with dataset-metadata.json."""
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE, exist_ok=True)
    included = []
    for src, name in PAYLOAD:
        full = os.path.join(ROOT, src)
        if not os.path.exists(full):
            continue
        shutil.copy2(full, os.path.join(STAGE, name))
        included.append((name, os.path.getsize(full)))
    meta = {
        "title": TITLE,
        "id": f"{username()}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": bool(private),
    }
    with open(os.path.join(STAGE, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    readme = [
        "# Kaggriculture ladder mine", "",
        "Derived artefacts from the public episode archive, produced by",
        "`src/kaggriculture/data/mine_top.py`. The raw replays are Kaggle's own daily datasets",
        "(`kaggle/kaggriculture-episodes-<date>`); this holds only what was",
        "computed from them.", "",
        "| file | what it is |", "|---|---|",
        "| `policy_diff.csv` | our agent's action against the winning action, on the winners' own game states, aggregated by op |",
        "| `market_schedule.csv` | units of each product the field sells, per day and per hour |",
        "| `top_trajectory.csv` | hands, herd, tiles and cash by day for the winning seat |",
        "| `episodes.csv` | ~60 features per player per episode |",
        "| `agent_v9_cem.py` | the agent those diffs were computed against |",
        "", f"Generated {dt.datetime.now().replace(microsecond=0).isoformat()}.",
    ]
    with open(os.path.join(STAGE, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(readme))
    return included, meta


def create(private=True):
    included, meta = stage(private)
    if not included:
        print("nothing to publish -- run src/kaggriculture/data/mine_top.py first")
        return 1
    rc, out = _kaggle(["datasets", "create", "-p", STAGE, "--dir-mode", "zip"])
    print(out)
    if rc == 0:
        print(f"\ncreated {meta['id']}  (private={meta['isPrivate']})")
    return rc


def update(note=None):
    included, meta = stage()
    if not included:
        print("nothing to publish -- run src/kaggriculture/data/mine_top.py first")
        return 1
    note = note or f"mine refresh {dt.datetime.now().replace(microsecond=0).isoformat()}"
    rc, out = _kaggle(["datasets", "version", "-p", STAGE, "-m", note,
                       "--dir-mode", "zip"])
    print(out)
    if rc == 0:
        print(f"\nnew version of {meta['id']}: {note}")
        for name, size in included:
            print(f"  {name:<26}{size:>10,} bytes")
    return rc


def status():
    ident = f"{username()}/{SLUG}"
    rc, out = _kaggle(["datasets", "files", ident], timeout=180)
    if rc != 0:
        print(f"{ident}: not created yet")
        print("  python -m kaggriculture.data.kaggle_dataset --create")
    else:
        print(f"{ident}:")
        print(_safe(out))
    print("\nKaggle's own episode datasets (mountable in a notebook, no download):")
    for ident in ("kaggle/kaggriculture-episodes-index",
                  "kaggle/kaggriculture-episodes-2026-08-05"):
        rc, out = _kaggle(["datasets", "files", ident], timeout=180)
        head = [l for l in out.splitlines() if l and "Next Page" not in l][:3]
        print(f"  {ident:<46}{'OK' if rc == 0 else 'MISSING'}   {head[-1][:40] if head else ''}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--public", action="store_true", help="publish publicly (default private)")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()
    if args.create:
        return create(private=not args.public)
    if args.update:
        return update(args.note)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
