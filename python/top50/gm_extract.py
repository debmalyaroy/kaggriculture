"""Top-100 teams' games from the local GM dataset (D:/gm_dataset, georgymamarin/kaggriculture-episodes) -> tapes.

    python python/top50/gm_extract.py [--gm D:/gm_dataset] [--leaderboard .local/opp/leaderboard.csv] [--top 100] [--procs 12]

Runs on the LAPTOP (the dataset lives there; ~26 GB of replay parquet shards). Reads episodes.csv, keeps public
completed games where either seat's team_id is in the leaderboard's top --top, then reads only those rows
from the replay shards (pyarrow filter on episode_id), converts each replay to a tape per top seat
(python/replay_to_tape.py format + team, team_id, sub, rating, opp_* labels) and writes
  data/top50/gm_local/tapes/sNN/<episode>_<seat>.json, data/top50/gm_local/index.tsv, data/top50/gm_local/READY
The box picks it up after upload (aws/sync_up.sh does not carry data: rsync data/top50/gm_local).
The GM dataset is read, never modified.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python"))
from replay_to_tape import convert  # noqa: E402

OUT = os.path.join(RL, "data", "top50", "gm_local")
SHARDS = 24
WANT = {}
ENGINE = "1.32.7"  # the ladder's engine since 16 Aug (models/engine_version.json); older replays follow other rules


def init(w):
    global WANT
    WANT = w


def work(args):
    path, rg = args
    want = WANT
    f = pq.ParquetFile(path)
    t = f.read_row_group(rg, columns=["episode_id"])
    ids = t.column("episode_id").to_pylist()
    hit = [i for i, e in enumerate(ids) if e in want]
    if not hit:
        return 0
    t = f.read_row_group(rg, columns=["episode_id", "replay_json"])
    n = 0
    for i in hit:
        eid = ids[i]
        rp = json.loads(t.column("replay_json")[i].as_py())
        if str(rp.get("module_version")) != ENGINE:
            continue
        for seat, lab in want[eid]:
            tape = convert(rp, seat)
            tape.update(lab)
            tape["engine"] = rp.get("module_version")
            d = os.path.join(OUT, "tapes", f"s{eid % SHARDS:02d}")
            os.makedirs(d, exist_ok=True)
            json.dump(tape, open(os.path.join(d, f"{eid}_{seat}.json"), "w", encoding="utf-8"))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm", default="D:/gm_dataset")
    ap.add_argument("--leaderboard", default=os.path.join(RL, ".local", "opp", "leaderboard.csv"))
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    lb = pd.read_csv(a.leaderboard).sort_values("Score", ascending=False).head(a.top)
    name = dict(zip(lb.TeamId.astype(int), lb.TeamName))
    top = set(name)
    e = pd.read_csv(os.path.join(a.gm, "episodes.csv"))
    e = e[(e.type == "EPISODE_TYPE_PUBLIC") & (e.state == "COMPLETED")]
    e = e[(e.team_0.isin(top) | e.team_1.isin(top)) & (e.end_time >= "2026-08-15")]  # 1.32.7 era; checked per replay too
    want, rows = {}, []
    for r in e.itertuples():
        for s in (0, 1):
            tid = getattr(r, f"team_{s}")
            if tid in top:
                o = 1 - s
                lab = {"team": name[tid], "team_id": int(tid), "sub": int(getattr(r, f"sub_{s}")), "rating": float(getattr(r, f"rating_{s}")),
                       "opp_team_id": int(getattr(r, f"team_{o}")), "opp_sub": int(getattr(r, f"sub_{o}")), "opp_rating": float(getattr(r, f"rating_{o}")),
                       "end": r.end_time, "source": "gm"}
                want.setdefault(int(r.episode_id), []).append((s, lab))
                rows.append({"id": f"{r.episode_id}_{s}", "eid": r.episode_id, "seat": s, **lab})
    print(f"[gm] top {a.top}: {len(want)} episodes, {len(rows)} player-games, {len({x['team_id'] for x in rows})} teams", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.tsv"), sep="\t", index=False)
    # skip row groups by the parquet's own episode_id statistics (min/max per group): only groups that can
    # hold a wanted episode are opened
    import bisect
    ids = sorted(want)
    jobs, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(a.gm, "replays_*.parquet"))):
        md = pq.ParquetFile(path).metadata
        col = md.schema.names.index("episode_id")
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(col).statistics
            if st is not None and st.has_min_max:
                i = bisect.bisect_left(ids, st.min)
                if i >= len(ids) or ids[i] > st.max:
                    skipped += 1
                    continue
            jobs.append((path, rg))
    print(f"[gm] {len(jobs)} row groups to read ({skipped} skipped by statistics)", flush=True)
    done = 0
    with ProcessPoolExecutor(a.procs, initializer=init, initargs=(want,)) as ex:
        for i, n in enumerate(ex.map(work, jobs, chunksize=16), 1):
            done += n
            if i % 500 == 0:
                print(f"[gm] {i}/{len(jobs)} row groups, {done} tapes", flush=True)
    open(os.path.join(OUT, "READY"), "w").write(f"{done}\n")
    print(f"[gm] -> {OUT}: {done} tapes")


if __name__ == "__main__":
    main()
