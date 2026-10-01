#!/usr/bin/env python
"""(2) LEAGUE OPPONENT REFRESH  --  roster of top-scored opponent agents.

Three modes (epoch = YYYYMMDD_HHMMSS):

  # LOCAL: build/refresh the roster from the fetched top-agents folder (or, if
  # absent, the crown panel), and copy files into models/trackp/league/
  python scripts/trackp/league_refresh.py --n 16

  # LOCAL: pack .local/top_agents/ (roster + all agent payloads, BINARIES included)
  # into top_agents_<epoch>.tar.gz  -> upload to gdrive:kaggriculture/top_agents/
  python scripts/trackp/league_refresh.py --pack

  # BOX: extract a pulled top_agents_<epoch>.tar.gz and build the consumable roster
  python scripts/trackp/league_refresh.py --from-tar .local/top_agents/top_agents_<epoch>.tar.gz

Handles heterogeneous agents: a bundle may carry a C++/Rust BINARY or a
submission.tar.gz alongside its .py; the index's `type`+`entry` tells the runner
how to invoke each. (The RL model is torch weights on the ckpt path -- separate.)
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))
TOP = os.path.join(ROOT, ".local", "top_agents")
TOP_INDEX = os.path.join(TOP, "index.json")
OUT_JSON = os.path.join(ROOT, "models", "trackp", "league_agents.json")
OUT_DIR = os.path.join(ROOT, "models", "trackp", "league")
PACK_DIR = os.path.join(ROOT, ".local", "top_agents")


def _epoch():
    return time.strftime("%Y%m%d_%H%M%S")


def _roster_from_index():
    idx = json.load(open(TOP_INDEX))
    return [{"name": a["name"], "score": a.get("score"), "type": a.get("type"),
             "entry": a.get("entry"), "files": a.get("files", []),
             "url": a.get("url"), "version": a.get("version")}
            for a in idx.get("agents", [])]


def _roster_from_panel(n):
    from kaggriculture.data import selfplay_corpus as SC
    out = []
    for name, path, rating in SC.top_reactive_agents(n):
        dst = os.path.join(OUT_DIR, os.path.basename(path))
        try:
            if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(dst):
                shutil.copy2(path, dst)
        except Exception:
            pass
        out.append({"name": name, "score": round(float(rating), 1), "type": "py",
                    "entry": os.path.relpath(dst, ROOT), "files": [os.path.basename(dst)],
                    "url": None, "version": None})
    return out


def build_roster(n):
    """Prefer the fetched top-agents index (handles binaries); fall back to the
    crown panel (py-only). Writes models/trackp/league_agents.json."""
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(TOP_INDEX):
        roster = _roster_from_index()[:n]
        src = "top_agents/index.json"
    else:
        roster = _roster_from_panel(n)
        src = "crown_panel"
    payload = {"updated": _epoch(), "source": src, "n": len(roster), "agents": roster}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w"), indent=2)
    print(f"[league] roster {len(roster)} agents (from {src}) -> {OUT_JSON}")
    for r in roster[:10]:
        sc = f"{r['score']:.0f}" if r.get("score") is not None else "  ? "
        print(f"  {sc:>7}  {r['name']:32s} {r.get('type'):7s} {r.get('entry')}")
    return payload


def pack():
    """Tar the whole top_agents folder (index + every agent payload, binaries and
    tars included, file MODES preserved) for upload to gdrive:kaggriculture/top_agents/."""
    if not os.path.exists(TOP_INDEX):
        raise SystemExit(f"no {TOP_INDEX} -- run agent_fetch.py first")
    build_roster(9999)                                   # refresh roster json inside the pack
    ep = _epoch()
    tar_path = os.path.join(PACK_DIR, f"top_agents_{ep}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:           # tarfile preserves exec mode bits
        for p in sorted(glob.glob(os.path.join(TOP, "*"))):
            if os.path.abspath(p) == os.path.abspath(tar_path):
                continue
            tf.add(p, arcname=os.path.relpath(p, TOP))
    size_mb = os.path.getsize(tar_path) / 1e6
    print("\n" + "=" * 66)
    print(f"TOP AGENTS PACKED: {size_mb:.1f} MB")
    print(f"UPLOAD THIS -> gdrive:kaggriculture/top_agents/")
    print(f"PATH: {tar_path}")
    print("=" * 66)
    return tar_path


def from_tar(tar_path):
    """BOX side: extract a pulled top_agents tar into .local/top_agents/ (restoring
    binary exec bits) and rebuild the consumable roster."""
    os.makedirs(TOP, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(TOP)                               # restores modes -> binaries stay executable
    print(f"[league] extracted {tar_path} -> {TOP}")
    build_roster(9999)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=16, help="roster size")
    ap.add_argument("--pack", action="store_true", help="tar top_agents for upload")
    ap.add_argument("--from-tar", default=None, help="BOX: extract a pulled top_agents tar")
    a = ap.parse_args()
    if a.from_tar:
        from_tar(a.from_tar)
    elif a.pack:
        pack()
    else:
        build_roster(a.n)


if __name__ == "__main__":
    main()
