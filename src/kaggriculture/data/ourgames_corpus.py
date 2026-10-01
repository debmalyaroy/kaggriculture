"""Feed our live-submission ladder replays into the v2 token corpus.

The gm dataset stores replays as a parquet ``replay_json`` column; our own games
(downloaded by ``data.ourgames`` into ``data/ourgames/_stage/<eid>/<eid>.json``)
are RAW Kaggle episode replays -- a different container, same underlying game.
Mapping:
  seat        = info.TeamNames.index(<our team>)   (both seats extracted)
  banks/rtg   = replay.rewards                      (gm uses agents.csv)
  engine      = replay.module_version               (keep 1.32.7)
  (obs, act)  = steps[t][seat].observation / .action   (SAME index -- verified)
Encoding is the IDENTICAL v2 TC.encode_tokens / encode_action, so the output
shards are schema-identical to the gm + self-play corpus and combine directly.

    python -m kaggriculture.data.ourgames_corpus --out data/trackp_corpus_v2_ourgames
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse, glob, json, os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import kaggriculture.data.trackp_corpus as TC

STAGE = os.path.join(ROOT, "data", "ourgames", "_stage")
OUT_DIR = os.path.join(ROOT, "data", "trackp_corpus_v2_ourgames")
SCHEMA = TC.SHARD_SCHEMA


def _rows(replay, eid):
    steps = replay.get("steps") or []
    if len(steps) < 24:
        return
    if str(replay.get("module_version")) != TC.ENGINE_KEEP:      # 1.32.7 only
        return
    rewards = [float(r or 0) for r in (replay.get("rewards") or [0, 0])]
    if len(rewards) < 2:
        return
    b0, b1 = rewards[0], rewards[1]
    win = {0: 1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0)}
    win[1] = 1.0 - win[0] if win[0] != 0.5 else 0.5
    sp = TC.split_of(eid)
    wf = TC.world_family(steps)
    for seat in (0, 1):                                          # both real-ladder seats
        for t in range(len(steps)):
            cell = steps[t][seat] if seat < len(steps[t]) else None
            if not cell:
                continue
            obs = cell.get("observation") or {}
            if obs.get("player") != seat:
                continue
            toks = TC.encode_tokens(obs, seat)
            acta = TC.encode_action(cell.get("action") or {})
            yield (int(eid) if str(eid).isdigit() else abs(hash(eid)) % (10 ** 15),
                   seat, int(obs.get("day", 0)), int(obs.get("step", t)), int(sp),
                   float(win[seat]), 2700.0, int(wf), int(toks.shape[0]),
                   toks.tobytes(), acta.tobytes())


def download_raw(subs, jobs=4):
    """Download + KEEP raw episode replays for the given submissions (the
    ourgames analyser deletes them; we need the raw JSON for encoding)."""
    import kaggriculture.data.episodes as E
    import kaggriculture.data.sameday as sameday
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.makedirs(STAGE, exist_ok=True)
    for sub in subs:
        ids = E._own_episode_ids(str(sub), verbose=False)
        print(f"[ourgames_corpus] submission {sub}: {len(ids)} episode(s)")

        def grab(eid):
            dest = os.path.join(STAGE, str(eid))
            os.makedirs(dest, exist_ok=True)
            have = glob.glob(os.path.join(dest, "*.json"))
            if have:
                return eid, have[0], None
            try:
                return eid, sameday._grab_direct(str(eid), dest), None
            except Exception as e:
                return eid, None, str(e)[:80]
        got = 0
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for fut in as_completed({pool.submit(grab, e): e for e in ids}):
                eid, path, err = fut.result()
                if path:
                    got += 1
                elif err:
                    print(f"  ! {eid}: {err}")
        print(f"[ourgames_corpus] submission {sub}: kept {got} raw replays")


def build(stage=STAGE, out_dir=OUT_DIR, rows_per_shard=50_000):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(stage, "**", "*.json"), recursive=True))
    print(f"[ourgames_corpus] {len(files)} replay files under {stage}")
    done, shard_start = TC._done_episodes(out_dir)
    names = SCHEMA.names
    buf = {k: [] for k in names}
    shard_i = shard_start
    n_rows = n_eps = n_skip = 0
    t0 = time.time()

    def flush():
        nonlocal shard_i
        if not buf["episode_id"]:
            return
        pq.write_table(pa.table(buf, schema=SCHEMA),
                       os.path.join(out_dir, f"shard_og_{shard_i:04d}.parquet"),
                       compression="zstd", compression_level=10)
        shard_i += 1
        for k in buf:
            buf[k].clear()

    for f in files:
        eid = os.path.splitext(os.path.basename(f))[0]
        if str(eid) in done:
            continue
        try:
            replay = json.load(open(f, encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue
        used = False
        for row in _rows(replay, eid):
            for k, v in zip(names, row):
                buf[k].append(v)
            n_rows += 1
            used = True
            if len(buf["episode_id"]) >= rows_per_shard:
                flush()
        if used:
            n_eps += 1
        else:
            n_skip += 1
    flush()
    TC._build_index(out_dir)
    TC._write_stats_norm(out_dir)
    total = pq.read_metadata(os.path.join(out_dir, "index.parquet")).num_rows \
        if os.path.exists(os.path.join(out_dir, "index.parquet")) else n_rows
    json.dump(dict(source="ourgames", n_rows=int(total), n_eps=n_eps, n_skip=n_skip,
                   token_layout_version=TC.TOKEN_LAYOUT_VERSION,
                   built=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[ourgames_corpus] +{n_rows} rows / {n_eps} eps ({n_skip} skipped) "
          f"in {time.time()-t0:.1f}s; total {total} -> {out_dir}")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    ap.add_argument("--submissions", default=None,
                    help="comma ids: raw-download + KEEP their replays before building")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.submissions:
        download_raw([s.strip() for s in a.submissions.split(",") if s.strip()], a.jobs)
    build(a.stage, a.out, a.rows_per_shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
