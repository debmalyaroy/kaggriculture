"""Route library v2, A1 (full dataset): mine route-0-compatible winning seats from the whole GM dataset (LAPTOP).

    python python/routes/gm_routes.py [--gm D:/gm_dataset] [--min-rating 2400] [--procs 6]

The dataset's stream_hashes.csv hashes each seat's action stream (sha256 over canonical JSON actions + NUL, first
16 hex): a seat whose stream_h136 equals route 0's played route 0's exact actions (farm AND market) through step
135. Candidates: those seats in PUBLIC games that they WON, rated >= --min-rating after the game, not our own team
(16655505), engine 1.32.7 (read from the replay), steps 1..143 farmer + hands equal route 0 and opening purchases
equal (checked on the tape), deduplicated by full-stream hash (stream_h719). Reads only the needed row groups
(episode_id statistics). Writes data/routes/gm_cands/<episode>_<seat>.json (seat = the player) + manifest.json.
The GM dataset is read, never modified.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python"))
from replay_to_tape import convert  # noqa: E402

OUT = os.path.join(RL, "data", "routes", "gm_cands")
OUR_TEAM = "16655505"
ENGINE = "1.32.7"
canon = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"))  # noqa: E731
R0 = json.load(open(os.path.join(RL, "configs", "bases", "v61.1", "routes.json")))["0"]
R0U = [canon([a.get("farmer") or ["PASS"], a.get("hands") or []]) for a in R0[:144]]
BUYS = ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE")
R0B = [canon(sorted(canon(m) for m in (a.get("market") or []) if m and m[0] in BUYS)) for a in R0[:144]]
WANT = {}


def init(w):
    global WANT
    WANT = w


def ok(tape):
    s = tape["seat"]
    acts = tape["actions"]
    for i in range(1, 144):
        a = (acts[i][s] if i < len(acts) and len(acts[i]) > s else None) or {}
        if canon([a.get("farmer") or ["PASS"], a.get("hands") or []]) != R0U[i]:
            return False
        if canon(sorted(canon(m) for m in (a.get("market") or []) if m and m[0] in BUYS)) != R0B[i]:
            return False
    return True


def work(args):
    path, rg = args
    f = pq.ParquetFile(path)
    ids = f.read_row_group(rg, columns=["episode_id"]).column("episode_id").to_pylist()
    hit = [i for i, e in enumerate(ids) if e in WANT]
    if not hit:
        return []
    t = f.read_row_group(rg, columns=["episode_id", "replay_json"])
    out = []
    for i in hit:
        eid = ids[i]
        rp = json.loads(t.column("replay_json")[i].as_py())
        if str(rp.get("module_version")) != ENGINE:
            continue
        for seat, lab in WANT[eid]:
            tape = convert(rp, seat)
            tape["seat"] = seat
            if not ok(tape):
                continue
            tape.update(lab)
            key = f"{eid}_{seat}"
            json.dump(tape, open(os.path.join(OUT, key + ".json"), "w", encoding="utf-8"))
            out.append({"key": key, **lab})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm", default="D:/gm_dataset")
    ap.add_argument("--min-rating", type=float, default=2400)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    m = hashlib.sha256()
    for x in R0[:136]:
        m.update(canon(x).encode() + b"\0")
    h136 = m.hexdigest()[:16]
    seats, h719 = {}, {}
    for row in csv.DictReader(open(os.path.join(a.gm, "stream_hashes.csv"))):
        if row["stream_h136"] == h136:
            seats[(int(row["episode_id"]), int(row["seat"]))] = row["stream_h719"]
    want, seen = {}, set()
    for row in csv.DictReader(open(os.path.join(a.gm, "episodes.csv"), encoding="utf-8")):
        if row["type"] != "EPISODE_TYPE_PUBLIC":
            continue
        e = int(row["episode_id"])
        for s in (0, 1):
            if (e, s) not in seats or row[f"team_{s}"] == OUR_TEAM:
                continue
            b, o = float(row[f"bank_{s}"] or 0), float(row[f"bank_{1 - s}"] or 0)
            rt = float(row[f"rating_{s}"] or 0)
            if b <= o or rt < a.min_rating or seats[(e, s)] in seen:
                continue
            seen.add(seats[(e, s)])
            want.setdefault(e, []).append((s, {"team_id": row[f"team_{s}"], "rating": rt, "margin": b - o, "opp_rating": float(row[f"rating_{1 - s}"] or 0)}))
    print(f"[gm-routes] route-0 prefix seats {len(seats)}; wins rated >= {a.min_rating:.0f}, distinct streams: {sum(len(v) for v in want.values())}", flush=True)
    import bisect
    ids_sorted = sorted(want)
    jobs, skipped = [], 0
    for fn in sorted(os.listdir(a.gm)):
        if not fn.startswith("replays_") or not fn.endswith(".parquet"):
            continue
        path = os.path.join(a.gm, fn)
        md = pq.ParquetFile(path).metadata
        col = [md.schema.column(i).name for i in range(md.num_columns)].index("episode_id")
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(col).statistics
            if st is not None and st.has_min_max:
                j = bisect.bisect_left(ids_sorted, st.min)
                inside = j < len(ids_sorted) and ids_sorted[j] <= st.max
            else:
                inside = True
            if not inside:
                skipped += 1
                continue
            jobs.append((path, rg))
    print(f"[gm-routes] {len(jobs)} row groups to read ({skipped} skipped)", flush=True)
    rows = []
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(want,)) as ex:
        for i, r in enumerate(ex.map(work, jobs, chunksize=8), 1):
            rows += r
            if i % 500 == 0:
                print(f"[gm-routes] {i}/{len(jobs)} groups, {len(rows)} candidates", flush=True)
    json.dump(rows, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print(f"[gm-routes] {len(rows)} candidate tapes -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
