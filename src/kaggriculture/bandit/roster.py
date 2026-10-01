"""Tournament roster for the bandit track: EVERY public agent we have that plays a full game.

    python -m kaggriculture.bandit.roster [--workers 6]   -> models/bandit/roster.json

Sources: data/winplan/field (curated shims), data/kernels/_agents (payloads extracted from the public
notebooks), opponents/pub_*.py. Each candidate plays one full game vs agents/v61.1_bandit.py on the
faithful harness (Python agents on the Rust serve engine, kaggriculture.winplan.gate._play); it is kept
if the game finishes without an error. Exact duplicates (same source sha256) are dropped. Our own live
agents (v61, v61.1, v62, v62.1) are added by the tournament, not here.
"""
import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from kaggriculture.paths import ROOT

OUT = os.path.join(ROOT, "models", "bandit", "roster.json")
REF = os.path.join(ROOT, "agents", "v61.1_bandit.py")


def sources():
    c = [(os.path.basename(f)[:-3], f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "winplan", "field", "*.py")))]
    c += [(os.path.basename(f)[:-3], f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "kernels", "_agents", "*.py")))]
    c += [(os.path.basename(f)[:-3], f) for f in sorted(glob.glob(os.path.join(ROOT, "opponents", "pub_*.py")))]
    return c


def real_sha(path):
    """Field shims load main.py from data/winplan/agents/...; hash what actually runs."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    if "_d = '" in txt and "main.py" in txt:
        d = txt.split("_d = '", 1)[1].split("'", 1)[0].replace("\\\\", "\\")
        p = os.path.join(d, "main.py")
        if os.path.exists(p):
            txt = open(p, encoding="utf-8", errors="replace").read()
    return hashlib.sha256(txt.encode()).hexdigest()


def check(task):
    from kaggriculture.winplan.gate import _play
    name, path = task
    row = _play((REF, name, path, "check", 12345, 0))
    return name, path, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seen, cands = {}, []
    for name, path in sources():
        h = real_sha(path)
        if h in seen:
            continue
        seen[h] = name
        cands.append((name, path))
    print(f"[roster] {len(cands)} unique candidates (from {len(sources())} files)", flush=True)
    ok, bad = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(check, c) for c in cands]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                name, path, row = f.result(timeout=900)
            except Exception as exc:  # noqa: BLE001
                bad.append({"error": str(exc)[:160]})
                continue
            (bad if "error" in row else ok).append({"name": name, "path": os.path.relpath(path, ROOT), **{k: row.get(k) for k in ("us", "them", "error", "secs")}})
            if i % 20 == 0:
                print(f"[roster] {i}/{len(cands)} checked; {len(ok)} runnable", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"runnable": sorted(ok, key=lambda r: r["name"]), "failed": bad}, open(OUT, "w"), indent=1)
    print(f"[roster] runnable {len(ok)}, failed {len(bad)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
