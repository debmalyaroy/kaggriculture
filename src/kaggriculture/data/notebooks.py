"""Download the competition's public notebooks -- and the agents inside them.

This turned out to be the highest-value data source in the whole project and
nothing was fetching it. The top of this leaderboard publishes its work: the
notebooks carry the agent's `main.py` as a base85+zlib payload, and decoding
one showed that the leaders are not shipping a policy at all, they are shipping
a recorded 720-turn trajectory with a slip-recovery layer. No amount of replay
featurising would have told us that.

What it does, in one pass:

  1. list the competition's public kernels, newest and highest-voted first
  2. `kaggle kernels pull` each one into data/kernels/<ref>/
  3. flatten notebooks to .py, decode any embedded agent payload
     (src/kaggriculture/data/notebook_extract.py), and summarise what came out
  4. write an index to data/notebooks.csv so the next run only fetches new work

    python -m kaggriculture.data.notebooks --top 20
    python -m kaggriculture.data.notebooks --refresh          # re-pull everything
    python -m kaggriculture.data.notebooks --index            # just show what we have
    python -m kaggriculture.data.notebooks --sweep            # every listed kernel: new + updated
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.data.episodes as E  # noqa: E402  (kaggle CLI resolution lives there)
import kaggriculture.data.notebook_extract as NX  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402

COMPETITION = "kaggriculture"
KDIR = os.path.join(ROOT, "data", "kernels")
SRC = os.path.join(KDIR, "_src")
AGENTS = os.path.join(KDIR, "_agents")
INDEX = os.path.join(ROOT, "data", "notebooks.csv")


def _kaggle(args, timeout=300):
    """Run the kaggle CLI with a UTF-8 pipe.

    Notebook titles on this competition contain emoji, and on Windows the CLI
    inherits a cp1252 stdout and dies with a UnicodeEncodeError before printing
    a single row. Forcing the child's IO encoding is the whole fix.
    """
    cmd = E.kaggle_cmd() + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    return proc.returncode, out, err


def list_kernels(top=20, sort="voteCount"):
    """Public notebooks attached to the competition, best first."""
    rc, out, err = _kaggle(["kernels", "list", "--competition", COMPETITION,
                            "--sort-by", sort, "--page-size", str(max(1, top)),
                            "--csv"])
    if rc != 0 or not out.strip():
        pr.warn(f"kernels list failed ({sort}): {(err or out).splitlines()[:1]}", 1)
        return []
    rows = list(csv.DictReader(io.StringIO(out)))
    return [r for r in rows if r.get("ref")]


def pull(ref, refresh=False):
    """Pull one kernel's source. Returns its directory."""
    dest = os.path.join(KDIR, ref.replace("/", "_"))
    if os.path.isdir(dest) and not refresh and os.listdir(dest):
        return dest, "cached"
    os.makedirs(dest, exist_ok=True)
    rc, out, err = _kaggle(["kernels", "pull", ref, "-p", dest, "-m"])
    if rc != 0:
        return dest, f"failed: {(err or out).strip().splitlines()[:1]}"
    return dest, "pulled"


def flatten(path):
    """Notebook -> flat .py in data/kernels/_src, so grep and diff work."""
    os.makedirs(SRC, exist_ok=True)
    with open(path, encoding="utf-8", errors="replace") as fh:
        nb = json.load(fh)
    chunks = []
    for i, cell in enumerate(nb.get("cells", [])):
        body = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            chunks.append(f'\n#%% [markdown] cell {i}\n"""\n{body}\n"""')
        else:
            chunks.append(f"\n#%% [code] cell {i}\n{body}")
    out = os.path.join(SRC, os.path.basename(path)[:-6] + ".py")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chunks))
    return out


def harvest(top=20, refresh=False, verbose=True):
    os.makedirs(KDIR, exist_ok=True)
    seen, rows = {}, []
    for sort in ("voteCount", "dateRun"):
        for row in list_kernels(top=top, sort=sort):
            seen.setdefault(row["ref"], row)
    if not seen:
        pr.warn("no public notebooks listed -- is the kaggle CLI authenticated?", 1)
        return []
    pr.log(f"{len(seen)} public notebook(s) on {COMPETITION}", 1)

    for ref, meta in seen.items():
        dest, how = pull(ref, refresh=refresh)
        rows.append(_process(ref, meta, dest, how, verbose))
    _write_index(rows, merge=False)
    return rows


def _process(ref, meta, dest, how, verbose=True):
    """Flatten + decode one pulled kernel; return its index row."""
    note = how
    flat, decoded = [], []
    for nb in sorted(glob.glob(os.path.join(dest, "*.ipynb"))):
        flat.append(flatten(nb))
    flat += sorted(glob.glob(os.path.join(dest, "*.py")))
    for f in flat:
        try:
            # Report what this notebook HAS decoded, not what this run
            # happened to create. Diffing the glob meant a re-run showed
            # decoded=0 for everything already on disk, so the index
            # understated our coverage: four notebooks with payloads sitting
            # in _agents/ all read as undecoded.
            stem = os.path.splitext(os.path.basename(f))[0]
            NX.process(f, AGENTS)
            decoded += sorted(
                glob.glob(os.path.join(AGENTS, glob.escape(stem) + "__*")))
        except Exception as exc:                      # noqa: BLE001
            note = f"decode error: {exc}"
    row = ({
        "ref": ref,
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "votes": meta.get("totalVotes", ""),
        "last_run": meta.get("lastRunTime", ""),
        "status": note,
        "files": len(flat),
        "decoded": len(decoded),
        "decoded_paths": ";".join(os.path.basename(d) for d in decoded),
    })
    if verbose:
        pr.log(_safe(f"{ref:<62} {note:<8} files={len(flat)} decoded={len(decoded)}"), 2)
    return row


def _write_index(rows, merge=True):
    """Write the index; with `merge`, rows for refs not in `rows` are kept."""
    if not rows:
        return
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    keep = []
    if merge and os.path.exists(INDEX):
        mine = {r["ref"] for r in rows}
        with open(INDEX, encoding="utf-8") as fh:
            keep = [r for r in csv.DictReader(fh) if r.get("ref") not in mine]
    with open(INDEX, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep + rows)
    pr.log(f"index -> {os.path.relpath(INDEX, ROOT)} ({len(keep) + len(rows)} rows)", 1)


def list_all(page_size=100, max_pages=50):
    """EVERY public kernel on the competition (all pages), newest run first."""
    out = {}
    for page in range(1, max_pages + 1):
        rc, txt, err = _kaggle(["kernels", "list", "--competition", COMPETITION, "--sort-by", "dateRun",
                                "--page", str(page), "--page-size", str(page_size), "--csv"])
        rows = [r for r in csv.DictReader(io.StringIO(txt)) if r.get("ref")] if rc == 0 else []
        if not rows:
            break
        new = 0
        for r in rows:
            if r["ref"] not in out:
                out[r["ref"]] = r
                new += 1
        if new == 0:
            break
    return out


def sweep(verbose=True):
    """Pull every public kernel we lack, and re-pull any whose lastRunTime is newer than our copy.

    A re-pull first ARCHIVES the old copy to data/kernels/_versions/<name>/<old lastRun>/ -- the
    version that played the replays we hold is exactly the old one. Each copy records the lastRunTime
    it was pulled at in `.lastrun`."""
    listed = list_all()
    pr.log(f"{len(listed)} public notebook(s) listed on {COMPETITION}", 1)
    rows, counts = [], {"new": 0, "updated": 0, "same": 0, "failed": 0}
    for ref, meta in listed.items():
        dest = os.path.join(KDIR, ref.replace("/", "_"))
        stamp_file = os.path.join(dest, ".lastrun")
        remote = (meta.get("lastRunTime") or "").strip()
        have = os.path.isdir(dest) and any(f != ".lastrun" for f in os.listdir(dest))
        local = open(stamp_file).read().strip() if os.path.exists(stamp_file) else ""
        if have and not local:
            # pulled before stamps existed: compare with the pull time (file mtime, UTC)
            import datetime as _dt
            mt = max(os.path.getmtime(os.path.join(dest, f)) for f in os.listdir(dest))
            local = _dt.datetime.utcfromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
        if have and remote and remote[:19] <= local[:19]:
            counts["same"] += 1
            continue
        if have:
            import shutil
            arch = os.path.join(KDIR, "_versions", ref.replace("/", "_"), (local[:19] or "unknown").replace(":", "").replace(" ", "T"))
            os.makedirs(os.path.dirname(arch), exist_ok=True)
            if not os.path.exists(arch):
                shutil.move(dest, arch)
            else:
                shutil.rmtree(dest)
        dest, how = pull(ref, refresh=True)
        if how.startswith("failed"):
            counts["failed"] += 1
            pr.warn(_safe(f"{ref}: {how}"), 2)
            continue
        with open(stamp_file, "w") as fh:
            fh.write(remote)
        counts["updated" if have else "new"] += 1
        rows.append(_process(ref, meta, dest, "updated" if have else "pulled", verbose))
    _write_index(rows, merge=True)
    pr.log(f"sweep: {counts}", 1)
    return counts


def _safe(text):
    """Titles on this competition contain emoji and Windows consoles are cp1252."""
    enc = (sys.stdout.encoding or "utf-8")
    return str(text).encode(enc, "replace").decode(enc, "replace")


def show_index():
    if not os.path.exists(INDEX):
        print("no notebook index yet -- run: python -m kaggriculture.data.notebooks")
        return
    with open(INDEX, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            print(_safe(f"{row['votes']:>4} votes  {row['ref']:<62} "
                        f"decoded={row['decoded']}  {row['title'][:50]}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--refresh", action="store_true", help="re-pull cached kernels")
    ap.add_argument("--index", action="store_true", help="print the index and exit")
    ap.add_argument("--sweep", action="store_true",
                    help="ALL listed kernels: pull new ones, re-pull updated ones (old copy archived)")
    args = ap.parse_args()
    if args.index:
        show_index()
        return
    if args.sweep:
        sweep()
        return
    harvest(top=args.top, refresh=args.refresh)


if __name__ == "__main__":
    main()
