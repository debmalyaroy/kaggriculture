"""Band-labelled ladder opponents for the objective gate (docs/ARCHITECTURE.md s4.4, PLAN s20 G1/O2).

    python python/band_tapes.py [--since 2026-09-18] [--per-band 220] [--per-team 2] [--out data/tapes/band]   (the gate set)
    python python/band_tapes.py --since 2026-09-10 --until 2026-09-17 --per-band 600 --per-team 3 --out data/tapes/train   (PPO opponents)
    python python/band_tapes.py --top --min-rating 2700 --since 2026-08-15 --until 2026-09-17 --out data/tapes/train/top
        (EVERY game played at 2700+ at the time, no per-team cap: how each strong player plays across many worlds and
        rivals; a team's older low-rated submissions are not "top play", hence the per-game rating, not the team's)
    NOTE: rebuilding data/tapes/train deletes train/top (rmtree of --out): rebuild top afterwards.

From the corpus index: public 720-step games since --since. For each rating band of the OPPONENT
(<2100, 2100-2300, 2300-2500, 2500-2700, 2700+ by its rating before the game), up to --per-band seats,
at most --per-team per team (newest first, distinct seeds), so no single player dominates a band.
Every seed (= world) is used once across the whole set, as on the ladder. Each becomes a tape (python/slim_to_tape.py format) with `seat` = OUR seat (the other one), plus
labels: opp_team, opp_rating, band, date. `tapeplay --guarded` then plays a candidate in our seat
against that player's recorded stream on its own seed, through the guards that keep a stream legal
against a different us (measured exact on 60/60 real games when nothing is off).
Writes <out>/<band>/<episode>_<opp_seat>.json and <out>/index.tsv. Re-run on each delta for new players.
"""
import argparse
import json
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIM = os.path.join(RL, "data", "slim", "s1")
BINS = [0, 2100, 2300, 2500, 2700, 99999]
BANDS = ["lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-18")
    ap.add_argument("--until", default="9999-12-31", help="last date (inclusive); the PPO training set uses a window before the gate's")
    ap.add_argument("--per-band", type=int, default=220)
    ap.add_argument("--per-team", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RL, "data", "tapes", "band"))
    ap.add_argument("--top", action="store_true", help="instead of bands: ALL games rated >= --min-rating at the time "
                    "(labelled band 'top'); per-band/per-team caps are not applied")
    ap.add_argument("--top-teams", type=int, default=0, help="--top: also restrict to the N teams with the highest latest rating")
    ap.add_argument("--min-rating", type=float, default=2700, help="--top: the player's rating before the game")
    ap.add_argument("--exclude", nargs="*", default=[os.path.join(RL, "data", "tapes", "band", "index.tsv"), os.path.join(RL, "data", "tapes", "train", "index.tsv")],
                    help="index.tsv files whose seeds are never reused (the gate's worlds stay unseen; no world twice)")
    a = ap.parse_args()
    ix = pd.read_parquet(os.path.join(SLIM, "index", "episodes.parquet"))
    ix = ix[(ix.episode_type == "EPISODE_TYPE_PUBLIC") & (ix.n_steps >= 720) & (ix.end_date >= a.since) & (ix.end_date <= a.until)]
    seats = []
    for s in (0, 1):
        seats.append(pd.DataFrame({"eid": ix.episode_id, "opp_seat": s, "team": ix[f"team_name_{s}"], "file": ix.file,
                                   "r": ix[f"rating_pre_{s}"].fillna(ix[f"rating_{s}"]), "date": ix.end_date,
                                   "time": ix.end_time, "seed": ix.seed}))
    d = pd.concat(seats).dropna(subset=["r", "team"])
    d["band"] = pd.cut(d.r, BINS, right=False, labels=BANDS)
    d = d.sort_values("time", ascending=False)
    # every match on its own world: a seed is used once across ALL bands (ladder games are one seed
    # each; a recorded stream is only valid on the seed it was played on). Rarest band picks first.
    pick, used = [], set()
    for f in a.exclude or []:
        if os.path.exists(f) and os.path.dirname(os.path.abspath(f)) != os.path.abspath(a.out):
            used |= set(pd.read_csv(f, sep="	").seed)
    if a.top:
        g = d[(d.r >= a.min_rating) & ~d.seed.isin(used)]
        if a.top_teams:
            latest = d.sort_values("time").groupby("team").r.last().sort_values(ascending=False)
            g = g[g.team.isin(set(latest.head(a.top_teams).index))]
        g = g.drop_duplicates("seed").assign(band="top")
        vc = g.team.value_counts()
        print(f"[tapes] top: {len(g)} games rated {a.min_rating:.0f}+, {len(vc)} teams, most per team {vc.iloc[0] if len(vc) else 0} "
              f"({vc.index[0] if len(vc) else '-'}), teams with 10+ games {(vc >= 10).sum()}", flush=True)
        pick.append(g)
    for band in ([] if a.top else reversed(BANDS)):
        g = d[(d.band == band) & ~d.seed.isin(used)].drop_duplicates("seed")
        g = g.groupby("team", sort=False).head(a.per_team).head(a.per_band)
        used |= set(g.seed)
        pick.append(g)
    pick = pd.concat(pick)
    shutil.rmtree(a.out, ignore_errors=True)
    rows, by_file = [], {}
    for r in pick.itertuples():
        by_file.setdefault(r.file, []).append(r)
    for f, rs in by_file.items():
        want = {r.eid for r in rs}
        t = pq.read_table(os.path.join(SLIM, f), columns=["episode_id", "slim"], filters=[("episode_id", "in", list(want))], partitioning=None).to_pandas()
        t = t[t.episode_id.isin(want)].drop_duplicates("episode_id").set_index("episode_id")
        for r in rs:
            if r.eid not in t.index or t.loc[r.eid, "slim"] is None:
                continue
            g = json.loads(t.loc[r.eid, "slim"])
            steps = g["steps"]
            acts = [list(steps[i + 1].get("a") or [None, None]) for i in range(len(steps) - 1)]
            acts = [[x if isinstance(x, dict) else None for x in (p + [None, None])[:2]] for p in acts]
            tape = {"id": f"{r.eid}_{r.opp_seat}", "seed": g["info"]["seed"], "seat": 1 - r.opp_seat, "rewards": g.get("rewards"),
                    "actions": acts, "opp_team": r.team, "opp_rating": float(r.r), "band": r.band, "date": r.date}
            dd = os.path.join(a.out, r.band)
            os.makedirs(dd, exist_ok=True)
            json.dump(tape, open(os.path.join(dd, f"{tape['id']}.json"), "w", encoding="utf-8"))
            rows.append((tape["id"], r.band, r.team, f"{r.r:.0f}", r.date, g["info"]["seed"]))
    with open(os.path.join(a.out, "index.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("id\tband\topp_team\topp_rating\tdate\tseed\n")
        for x in rows:
            fh.write("\t".join(map(str, x)) + "\n")
    cnt = pd.Series([x[1] for x in rows]).value_counts().reindex(BANDS).fillna(0).astype(int)
    print(f"[band_tapes] {len(rows)} opponent tapes since {a.since}: " + ", ".join(f"{b} {n}" for b, n in cnt.items()) + f" -> {a.out}")


if __name__ == "__main__":
    main()
