#!/usr/bin/env python
"""(8) AGENT FETCH  --  download top-scored opponent agents into a folder, with an
index. Handles agents that are a single .py AND agents that PACK A BINARY
(C++/Rust exe, .so, or a submission.tar.gz) -- the index records each agent's
TYPE + ENTRYPOINT so the box knows how to run it. league_refresh.py --pack then
tars this folder for upload; the RL model (torch weights) is a separate path.

Two modes:

  # rank the local crown panel + copy the top-N reactive agents' full payloads
  python scripts/trackp/agent_fetch.py --top 16

  # add ONE specific agent from a Kaggle kernel URL/slug + version (with its score)
  python scripts/trackp/agent_fetch.py --url debmalya84/some-agent --version 4 --score 61234

Layout produced under .local/top_agents/ :
  <name>/<files...>          one folder per agent (py, and/or binary, and/or tar)
  index.json                 [{name, score, entry, type, files, source, url, version, added}]

`type` is one of: py | binary | tar | bundle. `entry` is what a runner invokes
(the .py, the executable, or the tarball to unpack).
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, stat, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))
OUT = os.path.join(ROOT, ".local", "top_agents")
INDEX = os.path.join(OUT, "index.json")
BIN_EXT = {".exe", ".bin", ".so", ".out", ""}          # "" = extension-less unix exe
TAR_EXT = {".gz", ".tgz", ".tar", ".zip"}


def _now():
    return time.strftime("%Y%m%d_%H%M%S")


def _safe(name):
    """Filesystem-safe, COLLISION-PROOF name: truncated + a hash of the full name
    so distinct agents whose names share a 48-char prefix never overwrite."""
    import hashlib
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if len(base) <= 48:
        return base
    h = hashlib.sha1(name.encode()).hexdigest()[:6]
    return base[:41] + "_" + h


def _load_index():
    if os.path.exists(INDEX):
        try:
            return json.load(open(INDEX))
        except Exception:
            pass
    return {"generated": _now(), "agents": []}


def _save_index(idx):
    idx["generated"] = _now()
    os.makedirs(OUT, exist_ok=True)
    json.dump(idx, open(INDEX, "w"), indent=2)


def _upsert(idx, entry):
    idx["agents"] = [a for a in idx["agents"] if a["name"] != entry["name"]]
    idx["agents"].append(entry)
    idx["agents"].sort(key=lambda a: -(a.get("score") or 0))


def _classify(files):
    """Pick (type, entry) from a set of payload files. A binary or a tarball wins
    over a .py (that .py is likely just a thin loader for the binary)."""
    def ext(f):
        return os.path.splitext(f)[1].lower()
    exe = [f for f in files if ext(f) in {".exe", ".bin", ".out", ".so"} or
           (ext(f) == "" and _is_exec(f))]
    tars = [f for f in files if ext(f) in TAR_EXT]
    pys = [f for f in files if ext(f) == ".py"]
    if len(files) > 1 and (exe or tars):
        return "bundle", (exe[0] if exe else tars[0])
    if exe:
        return "binary", exe[0]
    if tars:
        return "tar", tars[0]
    if pys:
        return "py", pys[0]
    return "unknown", (files[0] if files else "")


def _is_exec(path):
    try:
        return bool(os.stat(path).st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except Exception:
        return False


def fetch_top(n):
    from kaggriculture.data import selfplay_corpus as SC
    agents = SC.top_reactive_agents(n)                 # [(name, path, rating)]
    idx = _load_index()
    added = 0
    for name, path, rating in agents:
        # key the folder on the FILE stem (unique per payload), not the registry
        # name (route families share one name across MOON/MUNIB/... variants)
        safe = _safe(os.path.splitext(os.path.basename(path))[0])
        dst_dir = os.path.join(OUT, safe)
        os.makedirs(dst_dir, exist_ok=True)
        payload = _copy_payload(path, dst_dir)         # .py + any sibling binary
        typ, entry = _classify([os.path.join(dst_dir, f) for f in payload])
        _upsert(idx, {"name": safe, "display": name, "score": round(float(rating), 1),
                      "type": typ, "entry": os.path.relpath(entry, OUT) if entry else "",
                      "files": payload, "source": "crown_panel",
                      "url": None, "version": None, "added": _now()})
        added += 1
        print(f"  [{rating:7.1f}] {safe:40s} {typ:7s} entry={os.path.basename(entry)}")
    _save_index(idx)
    print(f"[agent-fetch] top-{n}: {added} agents -> {OUT}  (index.json updated)")


def _copy_payload(src_path, dst_dir):
    """Copy an agent's payload into dst_dir. If src is a .py, also grab same-stem
    sibling binaries (agent.py + agent.bin). If src is a dir or archive, copy all."""
    out = []
    if os.path.isdir(src_path):
        for f in os.listdir(src_path):
            s = os.path.join(src_path, f)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst_dir, f)); out.append(f)
    else:
        stem = os.path.splitext(os.path.basename(src_path))[0]
        d = os.path.dirname(src_path)
        for s in glob.glob(os.path.join(d, stem + ".*")) + [src_path]:
            if os.path.isfile(s) and not s.endswith((".html", ".pyc")):
                bn = os.path.basename(s)
                if bn not in out:
                    shutil.copy2(s, os.path.join(dst_dir, bn)); out.append(bn)
    return out


def fetch_url(slug, version, score, name=None):
    """Download a specific Kaggle KERNEL (source + outputs, incl. any binary) and
    index it. `slug` is user/kernel-name (a full URL is trimmed to that)."""
    slug = slug.rstrip("/").split("kaggle.com/")[-1].replace("code/", "")
    name = name or slug.split("/")[-1]
    safe = _safe(name)
    dst_dir = os.path.join(OUT, safe)
    os.makedirs(dst_dir, exist_ok=True)
    if not _have("kaggle"):
        raise SystemExit("kaggle CLI not found: pip install kaggle + set ~/.kaggle/kaggle.json")
    # source (the agent code, versioned) + outputs (compiled binary / submission.tar.gz)
    pull = ["kaggle", "kernels", "pull", slug, "-p", dst_dir, "-m"]
    if version:
        pull += ["-v", str(version)]
    _run(pull, optional=True)
    _run(["kaggle", "kernels", "output", slug, "-p", dst_dir], optional=True)
    files = [f for f in os.listdir(dst_dir)
             if os.path.isfile(os.path.join(dst_dir, f)) and f != "kernel-metadata.json"]
    typ, entry = _classify([os.path.join(dst_dir, f) for f in files])
    idx = _load_index()
    _upsert(idx, {"name": safe, "score": (round(float(score), 1) if score is not None else None),
                  "type": typ, "entry": os.path.relpath(entry, OUT) if entry else "",
                  "files": files, "source": "kaggle-kernel",
                  "url": f"https://www.kaggle.com/code/{slug}", "version": version,
                  "added": _now()})
    _save_index(idx)
    print(f"[agent-fetch] {safe}: type={typ} entry={os.path.basename(entry)} "
          f"score={score} files={len(files)} -> {dst_dir}")


def _run(cmd, optional=False):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0 and not optional:
        raise SystemExit(f"failed rc={rc}")
    return rc


def _have(exe):
    from shutil import which
    return which(exe) is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=0, help="copy the top-N crown-panel reactive agents")
    ap.add_argument("--url", default=None, help="a Kaggle kernel slug/URL to add")
    ap.add_argument("--version", default=None, help="kernel version for --url")
    ap.add_argument("--score", type=float, default=None, help="public score for --url")
    ap.add_argument("--name", default=None, help="override the stored name for --url")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.url:
        fetch_url(a.url, a.version, a.score, a.name)
    if a.top:
        fetch_top(a.top)
    if not a.url and not a.top:
        ap.error("give --top N and/or --url <slug>")
    print(f"\nindex: {INDEX}   (feed to: python scripts/trackp/league_refresh.py --pack)")


if __name__ == "__main__":
    main()
