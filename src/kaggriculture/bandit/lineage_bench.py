"""Lineage-matched 2500+ benchmark: GM games in which one player opened exactly like one of OUR real games
(stream_hashes h100 equal to a Debmalya game's) against an opponent rated >= --min-rating. Our candidate
takes the lineage player's seat; the opponent replays its recorded moves, which were its real reactions to
an agent playing our line (so open-loop is close to real play, unlike top-player tapes vs anyone).

    python -m kaggriculture.bandit.lineage_bench --min-rating 2500 [--since 2026-08-15] [--workers 6]

Writes tapes to data/field/lineage2500/tapes/<eid>_<opp_seat>.json (tape "seat" = our seat) and meta.csv.
Then: tapeplay --tapes data/field/lineage2500/tapes --profiles ... --pa N --base configs/bandit/bases/v61.1
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from kaggriculture.bandit.top_field import GM, shard
from kaggriculture.paths import ROOT

US = 16655505  # team "Debmalya"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rating", type=float, default=2500)
    ap.add_argument("--since", default="2026-08-15")  # engine 1.32.7 on the ladder from Aug 15
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = os.path.join(ROOT, "data", "field", f"lineage{int(a.min_rating)}")
    os.makedirs(os.path.join(out, "tapes"), exist_ok=True)
    e = pd.read_csv(os.path.join(GM, "episodes.csv"))
    e = e[(e.state == "COMPLETED") & (e.type == "EPISODE_TYPE_PUBLIC") & (e.end_time.str[:10] >= a.since)]
    h = pd.read_csv(os.path.join(GM, "stream_hashes.csv"))
    ours = e[(e.team_0 == US) | (e.team_1 == US)]
    hi = h.set_index(["episode_id", "seat"])
    keys = [(r.episode_id, 0 if r.team_0 == US else 1) for r in ours.itertuples()]
    our_h = {hi.loc[k, "stream_h100"] for k in keys if k in hi.index}
    m = h[h.stream_h100.isin(our_h)].merge(e[["episode_id", "team_0", "team_1", "rating_0", "rating_1", "bank_0", "bank_1", "end_time"]], on="episode_id")
    m["team"] = [x if s == 0 else y for s, x, y in zip(m.seat, m.team_0, m.team_1)]
    m["opp_rating"] = [y if s == 0 else x for s, x, y in zip(m.seat, m.rating_0, m.rating_1)]
    m = m[(m.team != US) & (m.opp_rating >= a.min_rating)]
    # one lineage seat per game (if both sides share our line, keep seat 0's)
    m = m.sort_values(["episode_id", "seat"]).drop_duplicates("episode_id")
    m["opp_seat"] = 1 - m.seat
    m.to_csv(os.path.join(out, "meta.csv"), index=False)
    held = {f.split(".")[0] for f in os.listdir(os.path.join(out, "tapes"))}
    want = {int(r.episode_id): [int(r.opp_seat)] for r in m.itertuples() if f"{r.episode_id}_{r.opp_seat}" not in held}
    print(f"{len(m)} lineage games vs >= {a.min_rating}; {len(held)} tapes held, {len(want)} to extract", flush=True)
    shards = sorted(glob.glob(os.path.join(GM, "replays_*.parquet")))
    with ProcessPoolExecutor(a.workers) as ex:
        for path, n in ex.map(shard, [(p, want, os.path.join(out, "tapes")) for p in shards]):
            print(f"  {os.path.basename(path)}: {n} tapes", flush=True)
    print(f"{len(os.listdir(os.path.join(out, 'tapes')))} tapes in {out}", flush=True)


if __name__ == "__main__":
    main()
