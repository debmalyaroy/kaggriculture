"""Top-50 teams' games -> tapes + state traces, for the play analysis (queue Q33).

    python python/top50/extract.py [--teams 50] [--since 2026-09-15] [--shards 24] [--threads 24]

Top teams = the N highest peak ratings (best submission) among teams that played since --since (corpus index,
data/slim/s1/index/episodes.parquet). Every public 720-step game any of them played (all dates) becomes
one tape per top seat (seat = the top player's seat; a game between two top teams gives two tapes),
then `trace --seats tape` replays each exactly and writes the per-step state (target-dev/release/trace).
  data/top50/index.tsv        id eid seat team sub rating opp_team opp_rating opp_band date seed file
  data/top50/teams.tsv        team latest_rating games
  data/top50/tapes/sNN/*.json
  data/top50/trace/sNN.tsv    W / S / F rows (see crates/runner/src/bin/trace.rs)
"""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLIM = os.path.join(RL, "data", "slim", "s1")
OUT = os.path.join(RL, "data", "top50")  # --out
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TRACE = os.path.join(BIN, "trace" + (".exe" if os.name == "nt" else ""))
BINS = [0, 2100, 2300, 2500, 2700, 99999]
BANDS = ["lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=50)
    ap.add_argument("--since", default="2026-09-15")
    ap.add_argument("--sub-min", type=float, default=2700, help="keep a team's submissions whose peak rating reached this")
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--leaderboard", default=None, help="instead of peak ratings: the top --teams teams of this leaderboard CSV "
                    "(kaggle competitions leaderboard -d), matched by team name; every game of theirs in the corpus, any date")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = os.path.abspath(a.out)
    ix = pd.read_parquet(os.path.join(SLIM, "index", "episodes.parquet"))
    ix = ix[(ix.episode_type == "EPISODE_TYPE_PUBLIC") & (ix.n_steps >= 720)]
    seats = []
    for s in (0, 1):
        seats.append(pd.DataFrame({"eid": ix.episode_id, "seat": s, "team": ix[f"team_name_{s}"], "sub": ix[f"submission_id_{s}"],
                                   "r": ix[f"rating_pre_{s}"].fillna(ix[f"rating_{s}"]), "opp_team": ix[f"team_name_{1 - s}"],
                                   "opp_r": ix[f"rating_pre_{1 - s}"].fillna(ix[f"rating_{1 - s}"]), "date": ix.end_date,
                                   "time": ix.end_time, "seed": ix.seed, "file": ix.file}))
    d = pd.concat(seats).dropna(subset=["team"])
    # Ratings are per SUBMISSION and every new submission restarts at 600, so a team's latest rating is
    # often a fresh submission mid-climb. Rank teams by their best submission's peak rating since --since,
    # and keep every game of each team's strong submissions (peak >= --sub-min), ramp-up games included
    # (they are the team's play against weaker bands).
    recent = d[d.date >= a.since].dropna(subset=["r"])
    sub_peak = d.dropna(subset=["r"]).groupby(["team", "sub"]).r.max()
    if a.leaderboard:
        lb = pd.read_csv(a.leaderboard).sort_values("Score", ascending=False).head(a.teams)
        latest = pd.Series(lb.Score.values, index=lb.TeamName.values)
        strong = {k for k in sub_peak.index if k[0] in set(latest.index)}  # every submission of these teams
    else:
        latest = recent.groupby("team").r.max().sort_values(ascending=False).head(a.teams)
        strong = {k for k, v in sub_peak.items() if v >= a.sub_min and k[0] in set(latest.index)}
    top = d[[(t, s) in strong for t, s in zip(d.team, d["sub"])]].drop_duplicates(["eid", "seat"]).copy()
    top["sub_peak"] = [sub_peak.get((t, s)) for t, s in zip(top.team, top["sub"])]
    top["opp_band"] = pd.cut(top.opp_r, BINS, right=False, labels=BANDS).astype(str)
    top = top.sort_values(["file", "eid", "seat"]).reset_index(drop=True)
    top["id"] = top.eid.astype(str) + "_" + top.seat.astype(str)
    top["shard"] = top.index % a.shards
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "trace"))
    pd.DataFrame({"team": latest.index, "peak_rating": latest.values.round(1),
                  "games": [int((top.team == t).sum()) for t in latest.index]}).to_csv(os.path.join(OUT, "teams.tsv"), sep="\t", index=False)
    print(f"[top50] {len(latest)} teams (peak rating {latest.iloc[-1]:.0f}-{latest.iloc[0]:.0f}), {len(strong)} strong submissions, {len(top)} player-games, "
          f"{top.eid.nunique()} episodes", flush=True)
    rows = []
    for f, g in top.groupby("file"):
        want = set(g.eid)
        t = pq.read_table(os.path.join(SLIM, f), columns=["episode_id", "slim"], filters=[("episode_id", "in", list(want))], partitioning=None).to_pandas()
        t = t[t.episode_id.isin(want)].drop_duplicates("episode_id").set_index("episode_id")
        for r in g.itertuples():
            if r.eid not in t.index or t.loc[r.eid, "slim"] is None:
                continue
            j = json.loads(t.loc[r.eid, "slim"])
            steps = j["steps"]
            acts = [list(steps[i + 1].get("a") or [None, None]) for i in range(len(steps) - 1)]
            acts = [[x if isinstance(x, dict) else None for x in (p + [None, None])[:2]] for p in acts]
            dd = os.path.join(OUT, "tapes", f"s{r.shard:02d}")
            os.makedirs(dd, exist_ok=True)
            json.dump({"id": r.id, "seed": j["info"]["seed"], "seat": int(r.seat), "rewards": j.get("rewards"), "actions": acts,
                       "team": r.team, "rating": None if pd.isna(r.r) else float(r.r), "opp_team": r.opp_team,
                       "opp_rating": None if pd.isna(r.opp_r) else float(r.opp_r), "end": str(r.date)},
                      open(os.path.join(dd, f"{r.id}.json"), "w", encoding="utf-8"))
            rows.append(r)
    idx = pd.DataFrame(rows)[["id", "eid", "seat", "team", "sub", "sub_peak", "r", "opp_team", "opp_r", "opp_band", "date", "seed", "file"]]
    idx.to_csv(os.path.join(OUT, "index.tsv"), sep="\t", index=False)
    print(f"[top50] wrote {len(idx)} tapes; tracing in {a.shards} shards", flush=True)

    def one(k):
        src = os.path.join(OUT, "tapes", f"s{k:02d}")
        if not os.path.isdir(src):
            return ""
        r = subprocess.run([TRACE, "--tapes", src, "--out", os.path.join(OUT, "trace", f"s{k:02d}.tsv"), "--seats", "tape"],
                           capture_output=True, text=True)
        return r.stderr.strip()

    with ThreadPoolExecutor(a.threads) as ex:
        for line in ex.map(one, range(a.shards)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
