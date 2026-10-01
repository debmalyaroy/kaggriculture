"""Top-N ladder players' games from the gm dataset -> tapeplay tapes (one per top-player side).

For every COMPLETED game a current top-N team played (teams.csv ladder_score), writes a tape where the
TOP player's recorded actions are the fixed side and OUR agent takes the rival's seat, so
`tapeplay --tapes DIR` answers "what would we bank in that world against what that top player did".
A game between two top-N teams yields two tapes (one per side).

    python -m kaggriculture.bandit.top_field --top 40 [--workers 5]
    python -m kaggriculture.bandit.top_field --top 50 --leaderboard .local/field/week/top50_lb.json --out data/field/week50

Out: data/field/top40/tapes/<eid>_<topseat>.json and data/field/top40/meta.csv
(eid, top_seat, our_seat, top_team, rival_team, top_rating, rival_rating, top_bank, rival_bank, end_time).
Reads shards row-group by row-group, one replay body at a time (RAM lean).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

from kaggriculture.paths import ROOT

GM = r"D:\gm_dataset"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def select(top, lb=None):
    """-> (teams {team_id: (name, score)}, games {eid: row}). `lb` = a leaderboard JSON (kaggle competitions
    leaderboard --format json: teamId/teamName/score) to take the CURRENT top N instead of GM's ladder_score."""
    if lb:
        s = open(lb, encoding="utf-8").read()
        rows = json.loads(s[s.index("["):])[:top]
        teams = {str(r["teamId"]): (r["teamName"], float(r["score"])) for r in rows}
    else:
        with open(os.path.join(GM, "teams.csv"), newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("ladder_score")]
        rows.sort(key=lambda r: -float(r["ladder_score"]))
        teams = {r["team_id"]: (r["team_name"], float(r["ladder_score"])) for r in rows[:top]}
    games = {}
    with open(os.path.join(GM, "episodes.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["state"] == "COMPLETED" and r.get("type", "EPISODE_TYPE_PUBLIC") == "EPISODE_TYPE_PUBLIC" and (r["team_0"] in teams or r["team_1"] in teams):
                games[int(r["episode_id"])] = r
    return teams, games


def tape(rp, our_seat, eid):
    steps = rp["steps"]
    acts = [[steps[t + 1][s].get("action") if isinstance(steps[t + 1][s].get("action"), dict) else None for s in (0, 1)]
            for t in range(len(steps) - 1)]
    return {"id": eid, "seed": rp["info"]["seed"], "seat": our_seat, "rewards": rp.get("rewards"), "actions": acts}


def shard(job):
    path, want, out = job
    pf = pq.ParquetFile(path)
    n = 0
    for g in range(pf.num_row_groups):
        ids = pf.read_row_group(g, columns=["episode_id"]).column(0).to_pylist()
        hit = [i for i, e in enumerate(ids) if e in want]
        if not hit:
            continue
        body = pf.read_row_group(g, columns=["replay_json"]).column(0)
        for i in hit:
            eid = ids[i]
            rp = json.loads(body[i].as_py())
            if not (rp.get("info") or {}).get("seed"):
                continue
            for top_seat in want[eid]:
                with open(os.path.join(out, f"{eid}_{top_seat}.json"), "w") as fh:
                    json.dump(tape(rp, 1 - top_seat, eid), fh, separators=(",", ":"))
                n += 1
    return path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--leaderboard", default=None, help="leaderboard JSON: take the current top N from it")
    a = ap.parse_args()
    base = a.out or os.path.join(ROOT, "data", "field", f"top{a.top}")
    out = os.path.join(base, "tapes")
    os.makedirs(out, exist_ok=True)
    teams, games = select(a.top, a.leaderboard)
    want = {eid: [s for s in (0, 1) if r[f"team_{s}"] in teams] for eid, r in games.items()}
    done = {f.split(".")[0] for f in os.listdir(out)}
    todo = {e: [s for s in ss if f"{e}_{s}" not in done] for e, ss in want.items()}
    todo = {e: ss for e, ss in todo.items() if ss}
    print(f"top {a.top}: {len(teams)} teams, {len(games)} games, {sum(map(len, want.values()))} sides; {len(done)} tapes held, "
          f"{sum(map(len, todo.values()))} to extract", flush=True)
    shards = sorted(glob.glob(os.path.join(GM, "replays_*.parquet")))
    with ProcessPoolExecutor(a.workers) as ex:
        for path, n in ex.map(shard, [(p, todo, out) for p in shards]):
            print(f"  {os.path.basename(path)}: {n} tapes", flush=True)
    held = {f.split(".")[0] for f in os.listdir(out)}
    with open(os.path.join(base, "meta.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "eid", "top_seat", "our_seat", "top_team", "rival_team", "top_score", "top_rating", "rival_rating",
                    "top_bank", "rival_bank", "end_time"])
        for eid, r in sorted(games.items()):
            for s in want[eid]:
                if f"{eid}_{s}" not in held:
                    continue
                o = 1 - s
                w.writerow([f"{eid}_{s}", eid, s, o, teams[r[f"team_{s}"]][0], r[f"team_{o}"], teams[r[f"team_{s}"]][1],
                            r[f"rating_{s}"], r[f"rating_{o}"], r[f"bank_{s}"], r[f"bank_{o}"], r["end_time"]])
    print(f"{len(held)} tapes in {out}; meta {os.path.join(base, 'meta.csv')}", flush=True)


if __name__ == "__main__":
    main()
