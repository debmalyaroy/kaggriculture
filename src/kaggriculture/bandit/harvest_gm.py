"""Harvest the top-rated gm economies into panel tapes + base-tape seeds.

Selects the highest-rated PUBLIC+COMPLETED winner seats from the gm dataset
(deduped to one per submission so the panel is DIVERSE agents, not many games of
one), streams the replay shards RAM-leanly (one replay at a time), extracts each
winner's 720-turn action stream, and writes it as a tape into
`.local/candidates/strong_tapes/` (build_panel reads these as evolution
opponents). The strongest few double as base-tape SEEDS for the sell_search.

    python -m kaggriculture.bandit.harvest_gm --top 30 --min-rating 2800
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse, csv, glob, json, os, sys

import pyarrow.parquet as pq
import kaggriculture.pipeline.sell_search as SS

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    csv.field_size_limit(10 ** 8)
except (AttributeError, OSError, OverflowError):
    pass

GM = r"D:\gm_dataset"
OUT = os.path.join(ROOT, ".local", "candidates", "strong_tapes")
MANIFEST = os.path.join(ROOT, "models", "bandit", "harvested.json")


def select(top, min_rating):
    """Top winner seats, one per submission id, by rating. -> {eid: (seat, rating, bank, sub)}."""
    rows = []
    with open(os.path.join(GM, "episodes.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("type") != "EPISODE_TYPE_PUBLIC" or r.get("state") != "COMPLETED":
                continue
            for s in (0, 1):
                try:
                    rat = float(r.get(f"rating_{s}") or 0)
                    bank = float(r.get(f"bank_{s}") or 0)
                    obank = float(r.get(f"bank_{1 - s}") or 0)
                except (TypeError, ValueError):
                    continue
                if rat >= min_rating and bank >= obank:        # a winning seat
                    rows.append((rat, bank, str(r["episode_id"]), s,
                                 str(r.get(f"sub_{s}") or "")))
    rows.sort(key=lambda x: -x[0])
    picked, seen = {}, set()
    for rat, bank, eid, seat, sub in rows:
        key = sub or f"{eid}:{seat}"
        if key in seen:
            continue
        seen.add(key)
        picked[eid] = (seat, rat, bank, sub)
        if len(picked) >= top:
            break
    return picked


def extract(picked, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    want = dict(picked)
    done = {}
    for sh in sorted(glob.glob(os.path.join(GM, "replays_*.parquet"))):
        if not want:
            break
        pf = pq.ParquetFile(sh)
        for b in pf.iter_batches(batch_size=4, columns=["episode_id", "replay_json"]):
            d = b.to_pydict()
            for eid, rj in zip(d["episode_id"], d["replay_json"]):
                eid = str(eid)
                if eid not in want:
                    continue
                seat, rat, bank, sub = want.pop(eid)
                try:
                    steps = json.loads(rj)["steps"]
                except (ValueError, TypeError, KeyError):
                    continue
                acts = []
                for c in steps:
                    cell = c[seat] if seat < len(c) else None
                    a = (cell or {}).get("action") if isinstance(cell, dict) else None
                    acts.append(a or {"farmer": ["PASS"], "hands": [], "market": []})
                name = f"gm_{int(rat)}_{eid}.tape"
                SS.write_tape(acts[:719], os.path.join(OUT, name))
                done[eid] = {"tape": name, "seat": seat, "rating": rat,
                             "bank": bank, "sub": sub}
                if verbose:
                    print(f"[harvest] {name}  rating={rat:.0f} bank={bank:.0f}", flush=True)
            if not want:
                break
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(done, open(MANIFEST, "w", encoding="utf-8"), indent=1)
    print(f"[harvest] wrote {len(done)} tapes -> {os.path.relpath(OUT, ROOT)}; "
          f"manifest {os.path.relpath(MANIFEST, ROOT)}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-rating", type=float, default=2800.0)
    a = ap.parse_args()
    picked = select(a.top, a.min_rating)
    print(f"[harvest] selected {len(picked)} distinct high-scorers "
          f"(>={a.min_rating:.0f}); scanning shards...", flush=True)
    extract(picked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
