"""EVERY game with a top player from GM's dataset -> PPO opponent tapes (operator 2026-09-28: "any game where any top
player is involved you have to fetch and train PPO"; 23,965 GM games have a 2500+ seat).

    python python/gm_top_tapes.py [--gm D:/gm_dataset] [--min-rating 2500] [--since 2026-08-15] [--workers 4]

For each public 1.32.7 game (engine cut-over 2026-08-15) whose seat N was rated >= --min-rating (GM's rating_N, right
after the game), one tape per top seat: the top player's recorded stream is the opponent and `seat` = OUR seat (the
other one), python/replay_to_tape.py format + opp_team / opp_rating / band / date. Our own games and every game in a
held-out yardstick (the band gate set, data/ladder_study) are skipped. Tapes go to OUT/shard_XX/<episode>_<topseat>.json
(XX = episode % 24): PPO rotates one shard per iteration (ppo.py --tapes-shards) so no rollout loads 25k tapes.
Resumable: existing tapes are skipped.
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NSHARD = 24


def band(r):
    return "2700plus" if r >= 2700 else "2600-2700" if r >= 2600 else "2500-2600" if r >= 2500 else "2400-2500"


def work(task):
    # Memory-light (the dataset scanner's readahead took ~10 GB per worker and crashed the laptop, 28 Sep): read the
    # episode ids first, then one row group (~2 games) at a time, only where a wanted id sits.
    import gc
    import pyarrow.parquet as pq
    from replay_to_tape import convert
    shard, want, out, g0, g1 = task
    pf = pq.ParquetFile(shard)
    n = 0
    for g in range(g0, min(g1, pf.metadata.num_row_groups)):
        gids = pf.read_row_group(g, columns=["episode_id"]).column(0).to_pylist()
        if not any(int(e) in want for e in gids):
            continue
        tb = pf.read_row_group(g, columns=["episode_id", "replay_json"])
        rows = list(zip(tb.column(0).to_pylist(), tb.column(1).to_pylist()))
        del tb
        for eid, rj in rows:
            seats = want.get(int(eid))
            if not seats:
                continue
            todo = [(s, info) for s, info in seats if not os.path.exists(os.path.join(out, f"shard_{int(eid) % NSHARD:02d}", f"{eid}_{s}.json"))]
            if not todo:
                continue
            try:
                rp = json.loads(rj)
            except Exception:  # noqa: BLE001
                continue
            if len(rp.get("steps", [])) < 700:
                continue
            for s, info in todo:
                t = convert(rp, 1 - s)
                t.update(id=f"{eid}_{s}", opp_team=info["team"], opp_rating=info["rating"], band=band(info["rating"]), date=info["date"])
                d = os.path.join(out, f"shard_{int(eid) % NSHARD:02d}")
                os.makedirs(d, exist_ok=True)
                dst = os.path.join(d, f"{eid}_{s}.json")
                with open(dst + ".tmp", "w") as fh:  # atomic: a crash never leaves a truncated tape for PPO
                    json.dump(t, fh, separators=(",", ":"))
                os.replace(dst + ".tmp", dst)
                n += 1
        del rows
        gc.collect()
    return f"{os.path.basename(shard)}[{g0}:{g1}]", n


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--gm", default="D:/gm_dataset")
    ap.add_argument("--min-rating", type=float, default=2500)
    ap.add_argument("--since", default="2026-08-15")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=200, help="row groups per task (~2 games each)")
    ap.add_argument("--out", default=os.path.join(RL, "data", "tapes", "top_all"))
    a = ap.parse_args()
    e = pd.read_csv(os.path.join(a.gm, "episodes.csv"))
    e = e[e["type"].astype(str).str.contains("PUBLIC") & (e["end_time"].astype(str) >= a.since)]
    ef = pd.read_csv(os.path.join(a.gm, "episode_features.csv"), usecols=["episode_id", "engine_version"]).drop_duplicates("episode_id")
    e = e.merge(ef, on="episode_id", how="left")
    e = e[e["engine_version"].astype(str) == "1.32.7"]
    held = set()
    for f in glob.glob(os.path.join(RL, "data", "tapes", "band", "*", "*.json")):
        held.add(int(os.path.basename(f).split("_")[0]))
    ls = os.path.join(RL, "data", "ladder_study", "sets.json")
    if os.path.exists(ls):
        for v in json.load(open(ls)).values():
            held.update(int(r["episode"]) for r in v)
    want = {}
    for r in e.itertuples():
        if int(r.episode_id) in held:
            continue
        for s in (0, 1):
            rt, tm = getattr(r, f"rating_{s}"), str(getattr(r, f"team_{s}"))
            other = str(getattr(r, f"team_{1 - s}"))
            if pd.notna(rt) and rt >= a.min_rating and tm != "Debmalya" and other != "Debmalya":
                want.setdefault(int(r.episode_id), []).append((s, {"team": tm, "rating": float(rt), "date": str(r.end_time)[:10]}))
    print(f"[gm-top] {len(want)} games with a >= {a.min_rating:.0f} player since {a.since} (1.32.7, held-out and our games skipped), "
          f"{sum(len(v) for v in want.values())} top seats", flush=True)
    shards = sorted(glob.glob(os.path.join(a.gm, "replays_*.parquet")))
    os.makedirs(a.out, exist_ok=True)
    tot = 0
    with ProcessPoolExecutor(a.workers) as ex:
        import pyarrow.parquet as pq
        tasks = []
        for sh in shards:  # row-group chunks, so the 9.8 GB file does not pin one worker for hours
            ng = pq.ParquetFile(sh).metadata.num_row_groups
            tasks += [(sh, want, a.out, g, g + a.chunk) for g in range(0, ng, a.chunk)]
        print(f"[gm-top] {len(tasks)} chunks of {a.chunk} row groups over {len(shards)} files", flush=True)
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            name, n = f.result()
            tot += n
            print(f"[gm-top] {name}: {n} tapes (total {tot})", flush=True)
    print(f"[gm-top] done: {tot} new tapes -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
