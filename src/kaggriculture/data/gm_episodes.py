"""Ingest the georgymamarin `kaggriculture-episodes` dataset — FILTERED + DELTA.

The dataset (~20 GB) is monthly replay shards (`replays_2026-*.parquet`, each
`episode_id`+`replay_json`) + `episodes.csv` (per-episode `rating_N`/`bank_N`/`type`).
This module:

  * FILTERS on ingest — only `type=EPISODE_TYPE_PUBLIC`, `state=COMPLETED`, and only
    seats whose `rating_N >= --min-rating` (the strong plays are the BC targets);
  * STREAMS the shards (never loads a 9.8 GB file whole — the box is 16.9 GB) via a
    pyarrow dataset, distils each kept seat's 720 turns into daily MacroActions
    (return + rating conditioned), and appends to
    `data/bc_corpus/macro_gm.parquet` (loaded automatically by bc_warmup);
  * DELTA — a ledger (`data/bc_corpus/gm_ingested.json`) records finished shards +
    ingested episode_ids, so the NEXT run (after `kaggle datasets download --force`)
    processes ONLY new shards / new episodes. A frozen month never re-ingests.

    python -m kaggriculture.data.gm_episodes --min-rating 2200
    python -m kaggriculture.data.gm_episodes --min-rating 2200 --rebuild   # ignore ledger
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

import kaggriculture.train.macro_actions as MA
from kaggriculture.data.bc_corpus import world_sig, split_of, TURNS_PER_DAY

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    csv.field_size_limit(10 ** 8)
except (AttributeError, OSError, OverflowError):
    pass

def _resolve_data_dir():
    """Where the georgymamarin dataset lives. Priority: KAGG_GM_DATA env → the
    Chrome/manual download at D:\\gm_dataset (whichever holds episodes.csv) →
    the in-repo scratch copy. Keeping this a function (not a frozen constant)
    lets the autopilot/download supervisor drop shards anywhere and be found."""
    env = os.environ.get("KAGG_GM_DATA")
    cands = [env] if env else []
    cands += [r"D:\gm_dataset",
              os.path.join(ROOT, ".local", "scratch", "datasets", "gm_episodes")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "episodes.csv")):
            return c
    return os.path.join(ROOT, ".local", "scratch", "datasets", "gm_episodes")


DATA = _resolve_data_dir()
DATASET_SLUG = "georgymamarin/kaggriculture-episodes"
OUT = os.path.join(ROOT, "data", "bc_corpus", "macro_gm.parquet")
LEDGER = os.path.join(ROOT, "data", "bc_corpus", "gm_ingested.json")
FEATURES = list(MA.FEATURE_NAMES)


def load_ledger():
    if os.path.exists(LEDGER):
        try:
            d = json.load(open(LEDGER, encoding="utf-8"))
            d.setdefault("episodes", [])
            d.setdefault("shards", {})
            return d
        except (OSError, ValueError):
            pass
    return {"episodes": [], "shards": {}}


def save_ledger(led):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    json.dump(led, open(LEDGER, "w", encoding="utf-8"), indent=1)


def _remote_manifest():
    """{filename: size_bytes} for the dataset, via the Kaggle API. [] on any failure
    (no creds / no quota / offline) — the caller treats that as 'download nothing'."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        owner, slug = DATASET_SLUG.split("/")
        out = {}
        page = api.dataset_list_files(owner, slug)
        for f in getattr(page, "files", []) or []:
            sz = getattr(f, "totalBytes", None) or getattr(f, "size", None)
            out[str(getattr(f, "name", f))] = int(sz) if sz else None
        return out
    except Exception:
        return {}


def delta_download(dest=None, patterns=("replays_", "episodes.csv"), verbose=True):
    """DELTA download: pull only files whose on-disk size != the Kaggle manifest size
    (missing files always download; a frozen month whose shard already matches is
    skipped). Returns the list of files fetched. No-op (returns []) when the Kaggle
    API is unavailable — safe to call unconditionally from the autopilot.

        python -m kaggriculture.data.gm_episodes --delta-download
    """
    dest = dest or DATA
    os.makedirs(dest, exist_ok=True)
    remote = _remote_manifest()
    if not remote:
        if verbose:
            print("[gm] delta-download: Kaggle API unavailable — skipped (0 files)")
        return []
    want = [n for n in remote if any(p in n for p in patterns)]
    todo = []
    for n in want:
        lp = os.path.join(dest, n)
        rs = remote[n]
        if not os.path.exists(lp):
            todo.append(n)
        elif rs and os.path.getsize(lp) != rs:
            todo.append(n)                       # shard grew / changed → re-pull
    if verbose:
        print(f"[gm] delta-download: {len(want)} candidate files, "
              f"{len(todo)} to fetch (rest match on-disk size)")
    if not todo:
        return []
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    owner, slug = DATASET_SLUG.split("/")
    got = []
    for n in todo:
        try:
            api.dataset_download_file(owner, slug, n, path=dest, force=True, quiet=not verbose)
            # kaggle writes <n>.zip for large files; unzip in place
            zp = os.path.join(dest, n + ".zip")
            if os.path.exists(zp):
                import zipfile
                with zipfile.ZipFile(zp) as z:
                    z.extractall(dest)
                os.remove(zp)
            got.append(n)
            if verbose:
                print(f"[gm]   fetched {n}")
        except Exception as e:
            if verbose:
                print(f"[gm]   {n} failed: {str(e)[:120]}")
    return got


def load_filter(min_rating):
    """episodes.csv → {episode_id: {seat: (rating, bank)}} for strong PUBLIC games."""
    p = os.path.join(DATA, "episodes.csv")
    keep = {}
    if not os.path.exists(p):
        return keep
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("type") != "EPISODE_TYPE_PUBLIC" or row.get("state") != "COMPLETED":
                continue
            seats = {}
            for s in (0, 1):
                try:
                    rat = float(row.get(f"rating_{s}") or 0)
                    bank = float(row.get(f"bank_{s}") or 0)
                except (TypeError, ValueError):
                    continue
                if rat >= min_rating:
                    seats[s] = (rat, bank)
            if seats:
                keep[str(row["episode_id"])] = seats
    return keep


def _macro_rows(eid, seat, rating, bank, steps, rewards):
    sig = world_sig(steps)
    split = split_of(eid)
    try:
        ret = 1.0 if rewards[seat] >= rewards[1 - seat] else 0.0
    except (TypeError, IndexError):
        ret = 1.0
    n_days = (len(steps) + TURNS_PER_DAY - 1) // TURNS_PER_DAY
    rows = []
    for d in range(n_days):
        cells = [c[seat] for c in steps[d * TURNS_PER_DAY:(d + 1) * TURNS_PER_DAY]
                 if seat < len(c) and c[seat]]
        if not cells:
            continue
        ma = MA.day_macro(cells)
        vec = ma.to_vector()
        row = {"episode_id": str(eid), "seat": seat, "day": float(d),
               "split": split, "world_sig": sig, "final_return": ret,
               "primary_class": ma.primary_class(), "rating": float(rating),
               "bank_return": float(bank), "source": "gm"}
        row.update({FEATURES[i]: float(vec[i]) for i in range(len(FEATURES))})
        rows.append(row)
    return rows


def ingest(min_rating=2200.0, rebuild=False, flush=200_000, verbose=True):
    keep = load_filter(min_rating)
    if not keep:
        print(f"[gm] no episodes.csv under {DATA} (download the dataset first)")
        return 0
    led = {"episodes": [], "shards": {}} if rebuild else load_ledger()
    done_eps = set(led["episodes"])
    shards = sorted(glob.glob(os.path.join(DATA, "replays_*.parquet")))
    print(f"[gm] filter: {len(keep)} strong episodes ≥{min_rating}; "
          f"{len(shards)} shards; {len(done_eps)} already ingested")

    buf = []
    W = {"writer": None}                             # streaming ParquetWriter holder
    existing = OUT if (os.path.exists(OUT) and not rebuild) else None
    tmp = OUT + ".tmp"

    def _flush():
        if not buf:
            return
        tbl = pa.Table.from_pylist(buf)
        if W["writer"] is None:
            W["writer"] = pq.ParquetWriter(tmp, tbl.schema)
        W["writer"].write_table(tbl)
        buf.clear()

    for sh in shards:
        name = os.path.basename(sh)
        sig = f"{os.path.getsize(sh)}"
        if not rebuild and led["shards"].get(name) == sig:
            continue                                 # frozen shard already ingested
        n_sh = 0
        pf = pq.ParquetFile(sh)
        for b in pf.iter_batches(batch_size=8, columns=["episode_id", "replay_json"]):
            d = b.to_pydict()
            for eid, rj in zip(d["episode_id"], d["replay_json"]):
                eid = str(eid)
                if eid not in keep or eid in done_eps:
                    continue
                try:
                    r = json.loads(rj)
                    steps, rewards = r["steps"], r.get("rewards") or [0, 0]
                except (ValueError, TypeError, KeyError):
                    continue
                for seat, (rat, bank) in keep[eid].items():
                    buf.extend(_macro_rows(eid, seat, rat, bank, steps, rewards))
                done_eps.add(eid)
                led["episodes"].append(eid)
                n_sh += 1
                if len(buf) >= flush:
                    _flush()
        led["shards"][name] = sig
        if verbose:
            print(f"[gm]   {name}: +{n_sh} episodes")
    _flush()
    if W["writer"] is not None:
        W["writer"].close()
        if existing and os.path.exists(existing):     # merge delta into the corpus
            merged = pa.concat_tables(
                [pq.read_table(existing), pq.read_table(tmp)],
                promote_options="default")
            pq.write_table(merged, OUT)
            os.remove(tmp)
        else:
            os.replace(tmp, OUT)
    save_ledger(led)
    tot = pq.read_table(OUT, columns=["episode_id"]).num_rows if os.path.exists(OUT) else 0
    print(f"[gm] done — ingested {len(done_eps)} strong episode-seats total; "
          f"macro_gm.parquet = {tot:,} rows. Ledger: {len(led['shards'])} shards.")
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-rating", type=float, default=2200.0)
    ap.add_argument("--rebuild", action="store_true", help="ignore the ledger")
    ap.add_argument("--delta-download", action="store_true",
                    help="pull only new/changed shards from Kaggle, then ingest")
    args = ap.parse_args()
    print(f"[gm] data dir: {DATA}")
    if args.delta_download:
        delta_download()
    ingest(min_rating=args.min_rating, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
