"""Training caches (memory-mapped, so a 4 GB corpus never has to fit in RAM).

    python python/learn/cache.py corpus     # data/obs/s1/day_obs -> data/train/corpus_*.npy
    python python/learn/cache.py league     # data/leagues/rand/batch-*.{tsv,obs} -> data/train/league_*.npy

Corpus cache, one row per (episode, seat):
  corpus_obs.npy      float16 [E, 30, 93]   dayobs vectors (0 on days the game never reached)
  corpus_rnext.npy    float16 [E, 30, 9]    aux target: slog(rival net units sold next day), 0 if unknown
  corpus_meta.npz     episode_id, seat, score, days, opp_band, own_band, end_date (YYYYMMDD int)
League cache, one row per game (learner seat only):
  league_obs.npy      float16 [G, 30, 93]
  league_meta.npz     seed, seat, score, sched [G, 30] int16, opp_kind (int code), batch
Bands (opponent's rating at game time, from data/slim/s1/index/episodes.parquet):
  0 <2100, 1 2100-2300, 2 2300-2500, 3 2500-2700, 4 2700+, -1 unknown.
Both builds are full rebuilds from the source files (minutes); the queue re-runs them after
every delta.
"""
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(RL, "data", "train")
BANDS = ["<2100", "2100-2300", "2300-2500", "2500-2700", "2700+"]
NF = 93  # = crates/dayobs N (FEAT_VERSION 2)
KINDS = ["v611", "fixed", "randknobs", "randsched"]


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def build_corpus(src=os.path.join(RL, "data", "obs", "s1", "day_obs")):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"no day_obs parquet under {src}")
    ocols = None
    # pass 1: the (episode, seat) keys, in file order
    keys = {}
    for f in files:
        t = pq.ParquetFile(f).read(columns=["episode_id", "seat"])
        for e, s in zip(t.column("episode_id").to_numpy(), t.column("seat").to_numpy()):
            k = (int(e), int(s))
            if k not in keys:
                keys[k] = len(keys)
    E = len(keys)
    os.makedirs(OUT, exist_ok=True)
    obs = np.lib.format.open_memmap(os.path.join(OUT, "corpus_obs.npy.part"), mode="w+", dtype=np.float16, shape=(E, 30, NF))
    rn = np.lib.format.open_memmap(os.path.join(OUT, "corpus_rnext.npy.part"), mode="w+", dtype=np.float16, shape=(E, 30, 9))
    ep = np.zeros(E, np.int64)
    seat = np.zeros(E, np.int8)
    score = np.full(E, np.nan, np.float32)
    days = np.zeros(E, np.int8)
    date = np.zeros(E, np.int32)
    for k, i in keys.items():
        ep[i], seat[i] = k
    for n, f in enumerate(files):
        pf = pq.ParquetFile(f)
        if ocols is None:
            ocols = [c for c in pf.schema_arrow.names if c.startswith("o_")]
            rcols = [c for c in pf.schema_arrow.names if c.startswith("y_rival_next_")]
            assert len(ocols) == NF and len(rcols) == 9, (len(ocols), len(rcols))
        t = pf.read(columns=["episode_id", "seat", "day", "score", "end_date"] + ocols + rcols)
        e_ = t.column("episode_id").to_numpy()
        s_ = t.column("seat").to_numpy()
        d_ = t.column("day").to_numpy()
        idx = np.array([keys[(int(a), int(b))] for a, b in zip(e_, s_)])
        X = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in ocols], 1).astype(np.float32)
        R = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in rcols], 1).astype(np.float64)
        R = np.nan_to_num(slog(R), nan=0.0)
        obs[idx, d_] = X.astype(np.float16)
        rn[idx, d_] = R.astype(np.float16)
        score[idx] = t.column("score").to_numpy()
        np.maximum.at(days, idx, (d_ + 1).astype(np.int8))
        dd = np.array([int(x.replace("-", "")) if x else 0 for x in t.column("end_date").to_pylist()], np.int32)
        date[idx] = dd
        if n % 200 == 0:
            print(f"[cache] corpus {n}/{len(files)} files", flush=True)
    obs.flush()
    rn.flush()
    del obs, rn
    # opponent / own band from the index
    ix = pd.read_parquet(os.path.join(RL, "data", "slim", "s1", "index", "episodes.parquet"),
                         columns=["episode_id", "band_0", "band_1"]).drop_duplicates("episode_id").set_index("episode_id")
    code = {b: i for i, b in enumerate(BANDS)}
    b0 = ix["band_0"].map(code).reindex(ep).to_numpy()
    b1 = ix["band_1"].map(code).reindex(ep).to_numpy()
    own = np.where(seat == 0, b0, b1)
    opp = np.where(seat == 0, b1, b0)
    own = np.nan_to_num(own.astype(np.float64), nan=-1).astype(np.int8)
    opp = np.nan_to_num(opp.astype(np.float64), nan=-1).astype(np.int8)
    for a in ("corpus_obs.npy", "corpus_rnext.npy"):
        os.replace(os.path.join(OUT, a + ".part"), os.path.join(OUT, a))
    np.savez(os.path.join(OUT, "corpus_meta.npz"), episode_id=ep, seat=seat, score=score, days=days,
             opp_band=opp, own_band=own, end_date=date)
    print(f"[cache] corpus: {E} (episode, seat) rows; opp band known {(opp >= 0).mean():.1%}; "
          f"dates {date[date > 0].min()}..{date.max()} -> {OUT}")


FAMILIES = ["v611", "escalated", "afr", "randknobs", "learner"]
AFR_PROFILES = {25, 26, 27, 28, 29, 30, 31}  # configs/profiles/v2.json ids with afr_on


def family(kind, desc):
    """Opponent family label for the aux head (league ground truth): v61.1, a fixed escalated profile,
    a fixed front-running (AFR) profile, random clone-lineage knobs, or a random-schedule learner."""
    if kind == "v611":
        return 0
    if kind == "fixed":
        return 2 if desc.isdigit() and int(desc) in AFR_PROFILES else 1
    if kind == "randknobs":
        return 3
    return 4 if kind == "randsched" else -1


def build_league(src=os.path.join(RL, "data", "leagues", "rand"), prefix="league"):
    tsvs = sorted(glob.glob(os.path.join(src, "batch-*.tsv")))
    if not tsvs:
        sys.exit(f"no league batches under {src}")
    rows = []
    for f in tsvs:
        b = int(os.path.basename(f)[6:11])
        for line in open(f, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            rows.append((b, int(p[0]), int(p[1]), p[2], [int(x) for x in p[4].split(",")], float(p[7]), p[3]))
    G = len(rows)
    os.makedirs(OUT, exist_ok=True)
    obs = np.lib.format.open_memmap(os.path.join(OUT, f"{prefix}_obs.npy.part"), mode="w+", dtype=np.float16, shape=(G, 30, NF))
    i = 0
    for f in tsvs:
        n = sum(1 for _ in open(f, encoding="utf-8"))
        raw = np.fromfile(f[:-4] + ".obs", dtype="<f4")
        assert raw.size == n * 30 * NF, (f, raw.size, n)
        obs[i:i + n] = raw.reshape(n, 30, NF).astype(np.float16)
        i += n
    obs.flush()
    del obs
    os.replace(os.path.join(OUT, f"{prefix}_obs.npy.part"), os.path.join(OUT, f"{prefix}_obs.npy"))
    np.savez(os.path.join(OUT, f"{prefix}_meta.npz"),
             batch=np.array([r[0] for r in rows], np.int32), seed=np.array([r[1] for r in rows], np.int64),
             seat=np.array([r[2] for r in rows], np.int8),
             opp_kind=np.array([KINDS.index(r[3]) if r[3] in KINDS else -1 for r in rows], np.int8),
             sched=np.array([r[4] for r in rows], np.int16), score=np.array([r[5] for r in rows], np.float32),
             opp_fam=np.array([family(r[3], r[6]) for r in rows], np.int8))
    print(f"[cache] league: {G} games from {len(tsvs)} batches -> {OUT}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("corpus", "all"):
        build_corpus()
    if what in ("league", "all"):
        build_league()
