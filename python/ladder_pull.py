"""Our own ladder games -> tapes + a results table (queue Q26; option C: study v63's real losses).

    python python/ladder_pull.py [--subs 56567216,56538921] [--team Debmalya] [--sleep 3]

Per submission: list its episodes (kaggle competitions episodes), download each new replay (one at a
time, --sleep between, backing off on 429), convert it to a tape (python/replay_to_tape.py format +
opp_team / opp_rating / band) in data/ladder/<sub>/tapes/<episode>_<seat>.json and delete the raw
replay. data/ladder/<sub>/games.tsv: episode, date, seat, our bank, their bank, margin, result,
opponent team and its latest known rating (data/slim/s1/index/episodes.parquet; blank if unknown).
Resumable: an episode with a tape is never downloaded again.
"""
import argparse
import csv
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_to_tape import convert  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RL, "data", "ladder")
KAGGLE = os.environ.get("KAGGLE_BIN") or os.path.join(RL, ".venv", "bin", "kaggle")
BANDS = [(2100, "lt2100"), (2300, "2100-2300"), (2500, "2300-2500"), (2700, "2500-2700"), (1e9, "2700plus")]


def kaggle(*args, tries=6):
    wait = 30
    for _ in range(tries):
        r = subprocess.run([KAGGLE if os.path.exists(KAGGLE) else "kaggle", *args], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        if "429" in r.stdout + r.stderr:
            print(f"[ladder] 429, waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 600)
            continue
        raise RuntimeError(f"kaggle {' '.join(args)}: {(r.stdout + r.stderr)[-400:]}")
    raise RuntimeError(f"kaggle {' '.join(args)}: still rate limited")


def ratings():
    try:
        import pandas as pd
        d = pd.read_parquet(os.path.join(RL, "data", "slim", "s1", "index", "episodes.parquet"),
                            columns=["end_time", "team_name_0", "team_name_1", "rating_post_0", "rating_post_1", "rating_max", "rating_min"])
    except Exception as e:  # noqa: BLE001
        print(f"[ladder] no rating index ({e})")
        return {}
    d = d.sort_values("end_time")
    out = {}
    for s in (0, 1):
        x = d[["team_name_%d" % s, "rating_post_%d" % s]].dropna()
        out.update(dict(zip(x["team_name_%d" % s], x["rating_post_%d" % s])))
    return out


def band(r):
    if r is None:
        return "unknown"
    return next(b for lim, b in BANDS if r < lim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="56567216,56538921")
    ap.add_argument("--team", default="Debmalya")
    ap.add_argument("--sleep", type=float, default=3.0)
    a = ap.parse_args()
    rt = ratings()
    for sub in a.subs.split(","):
        d = os.path.join(OUT, sub)
        td = os.path.join(d, "tapes")
        os.makedirs(td, exist_ok=True)
        eps = [row["id"] for row in csv.DictReader(io.StringIO(kaggle("competitions", "episodes", sub, "-v")))
               if "COMPLETED" in (row.get("state") or "") and (row.get("id") or "").isdigit()]
        have = {os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(td, "*.json"))}
        new = [e for e in eps if e not in have]
        print(f"[ladder] {sub}: {len(eps)} completed episodes, {len(new)} new", flush=True)
        for e in new:
            with tempfile.TemporaryDirectory() as tmp:
                kaggle("competitions", "replay", e, "-p", tmp, "-q")
                f = glob.glob(os.path.join(tmp, "*.json"))
                if not f:
                    continue
                rp = json.load(open(f[0], encoding="utf-8"))
            names = rp.get("info", {}).get("TeamNames") or []
            if a.team not in names:
                continue
            seat = names.index(a.team)
            t = convert(rp, seat)
            opp = names[1 - seat]
            r = rt.get(opp)
            t.update(opp_team=opp, opp_rating=r, band=band(r), statuses=rp.get("statuses"), sub=sub)
            json.dump(t, open(os.path.join(td, f"{e}_{seat}.json"), "w", encoding="utf-8"))
            time.sleep(a.sleep)
        rows = []
        for f in sorted(glob.glob(os.path.join(td, "*.json"))):
            t = json.load(open(f, encoding="utf-8"))
            rw = t.get("rewards") or [0, 0]
            s = t["seat"]
            us, them = float(rw[s] or 0), float(rw[1 - s] or 0)
            res = "W" if us > them else "L" if us < them else "D"
            rows.append((t["id"], s, us, them, us - them, res, t.get("opp_team"), t.get("opp_rating") or "", t.get("band")))
        with open(os.path.join(d, "games.tsv"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("episode\tseat\tus\tthem\tmargin\tresult\topp_team\topp_rating\tband\n")
            for x in rows:
                fh.write("\t".join(str(v) for v in x) + "\n")
        w = sum(r[5] == "W" for r in rows)
        l = sum(r[5] == "L" for r in rows)
        by = {}
        for r in rows:
            b = by.setdefault(r[8], [0, 0, 0])
            b["WLD".index(r[5])] += 1
        print(f"[ladder] {sub}: {len(rows)} games, {w} W / {l} L; by opponent band (W/L/D): {by}", flush=True)


if __name__ == "__main__":
    main()
