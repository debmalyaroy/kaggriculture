"""Build a per-player feature dataset from top-ladder episodes.

Kaggle publishes the highest-rated episodes daily and says outright they are for
"IL/BC, bootstrapping RL, or just gathering statistics". This script turns them
into one CSV row per player per episode: what that player built, hired, bought
and sold, plus whether they won.

The download is the expensive part -- ~27 MB per replay, ~21 GB per day -- so
episodes are processed **streaming**: download one, extract ~60 numbers, delete
the JSON, move on. Disk high-water mark stays at one replay. The run is
resumable; episode ids already in the CSV are skipped.

Requires the `kaggle` CLI, authenticated, on a machine with network.

    python -m kaggriculture.data.build_dataset --days 3 --per-day 60
    python -m kaggriculture.data.build_dataset --days 5 --per-day 100 --out data/episodes.csv
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.data.episodes as ep  # noqa: E402
import kaggriculture.data.replay_analysis as ra  # noqa: E402

CROPS = ra.CROPS
ANIMALS = ra.ANIMALS
PRODUCTS = ra.PRODUCTS
OPS = ["NORTH", "SOUTH", "EAST", "WEST", "PASS", "WATER", "HARVEST", "PLANT",
       "FEED", "CARE", "PICKUP", "PLACE", "DROP", "DIG", "FERTILIZE",
       "BUILD_COOP", "BUILD_PASTURE", "COLLECT_FERTILIZER"]
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def feature_names():
    f = ["episode_id", "date", "player", "won", "final_bank", "opp_bank", "margin",
         "peak_hands", "mean_hands_late", "first_animal_day", "peak_herd",
         "peak_crops", "quadrants", "land_day1", "land_day2", "land_day3",
         "move_frac", "pass_frac", "productive_frac"]
    f += [f"peak_{k}" for k in list(CROPS) + list(ANIMALS)]
    f += [f"tiledays_{k}" for k in list(CROPS) + list(ANIMALS)]
    f += [f"sold_{p}" for p in PRODUCTS]
    f += [f"op_{o}" for o in OPS]
    return f


def rows_from_replay(path, date):
    rep = ra.profile_replay(path)
    if not rep:
        return []
    out = []
    for p, prof in enumerate(rep["players"]):
        mine = rep["rewards"][p]
        theirs = rep["rewards"][1 - p] if len(rep["rewards"]) > 1 else 0.0
        ops = prof["unit_ops"]
        tot = sum(ops.values()) or 1
        mv = sum(v for k, v in ops.items() if k in MOVES)
        land = prof["land_days"] + [None, None, None]
        row = {
            "episode_id": rep.get("episode_id"), "date": date, "player": p,
            "won": 1 if mine > theirs else 0,
            "final_bank": mine, "opp_bank": theirs, "margin": mine - theirs,
            "peak_hands": prof["peak_hands"],
            "mean_hands_late": round(prof["mean_hands_late"], 2),
            "first_animal_day": prof["first_animal_day"]
            if prof["first_animal_day"] is not None else -1,
            "peak_herd": prof["peak_herd"], "peak_crops": prof["peak_crops"],
            "quadrants": max(prof["quadrants_by_day"].values() or [1]),
            "land_day1": land[0] if land[0] is not None else -1,
            "land_day2": land[1] if land[1] is not None else -1,
            "land_day3": land[2] if land[2] is not None else -1,
            "move_frac": round(mv / tot, 4),
            "pass_frac": round(ops.get("PASS", 0) / tot, 4),
            "productive_frac": round((tot - mv - ops.get("PASS", 0)) / tot, 4),
        }
        for k in list(CROPS) + list(ANIMALS):
            row[f"peak_{k}"] = prof["peak"].get(k, 0)
            row[f"tiledays_{k}"] = round(prof["tile_days"].get(k, 0), 1)
        for prod in PRODUCTS:
            row[f"sold_{prod}"] = prof["sold_qty"].get(prod, 0)
        for o in OPS:
            row[f"op_{o}"] = round(ops.get(o, 0) / tot, 4)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--per-day", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "episodes.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "episodes", "stream"))
    ap.add_argument("--keep", action="store_true", help="do not delete replays")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    seen = set()
    if os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                seen.add(str(r.get("episode_id")))
        print(f"resuming: {len(seen)//2} episodes already in {args.out}")

    fields = feature_names()
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if new:
        writer.writeheader()

    try:
        index = ep.fetch_index(args.work)
    except ep.KaggleError as exc:
        sys.exit(f"ERROR: {exc}")
    days = index[-args.days:]
    print(f"days: {', '.join(r['date'] for r in days)}")

    total, t0 = 0, time.time()
    for day_row in reversed(days):
        date = day_row["date"]
        dataset = day_row.get("daily_dataset_slug") or f"kaggriculture-episodes-{date}"
        if "/" not in dataset:
            dataset = f"kaggle/{dataset}"
        day_dir = os.path.join(args.work, date)
        try:
            man = ep._download_dataset_file(dataset, "manifest.csv", day_dir)
        except ep.KaggleError as exc:
            print(f"  ! {date}: {exc}")
            continue
        if not man:
            continue
        manifest = ep._read_csv(man)
        manifest.sort(key=ep._score_of, reverse=True)
        print(f"  {date}: {len(manifest)} episodes available")

        taken = 0
        for mrow in manifest:
            if taken >= args.per_day:
                break
            fname = ep._episode_file(mrow)
            if not fname:
                continue
            eid = fname.replace(".json", "")
            if eid in seen:
                taken += 1
                continue
            local = os.path.join(day_dir, fname)
            try:
                ep._download_dataset_file(dataset, fname, day_dir)
            except ep.KaggleError as exc:
                print(f"    ! {fname}: {exc}")
                continue
            if not os.path.exists(local):
                continue
            try:
                for row in rows_from_replay(local, date):
                    writer.writerow(row)
                    total += 1
                fh.flush()
                taken += 1
                seen.add(eid)
            except Exception as exc:                               # noqa: BLE001
                print(f"    ! parse {fname}: {exc}")
            finally:
                if not args.keep:
                    try:
                        os.remove(local)          # streaming: keep disk flat
                    except OSError:
                        pass
            if taken % 10 == 0:
                print(f"    {taken}/{args.per_day}  ({total} rows, "
                      f"{time.time()-t0:.0f}s)", flush=True)
    fh.close()
    print(f"\nwrote {total} new rows to {args.out}  ({time.time()-t0:.0f}s)")
    print("next:  python -m kaggriculture.train.learn_params")


if __name__ == "__main__":
    main()
