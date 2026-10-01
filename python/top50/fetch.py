"""Fetch the current top-50 teams' recent games straight from Kaggle (queue Q36). No notebook.

    python python/top50/fetch.py [--top 50] [--days 7] [--per-team 80] [--jobs 4] [--max-queries 800]

1. Leaderboard: `kaggle competitions leaderboard -d` -> the top --top teams by score (team id, name).
2. Submissions: a breadth-first crawl of Kaggle's EpisodeService (ListEpisodes by submission id; API-token
   auth, no replay download, so no replay quota). Every game it lists names both players' submission id,
   team id and ratings, so the crawl walks from submissions we know (ours, and the teams' submissions in
   the corpus index) out to every top team's current submissions. It stops when every top team has a
   submission with games in the window, or at --max-queries.
3. Games: for each top team, its games from the last --days days (newest first, at most --per-team,
   spread over its submissions), downloaded one replay at a time (`kaggle competitions replay`, ~32 MB,
   backoff on 429), converted at once to a tape (python/replay_to_tape.py format; seat = the top team's
   seat, two tapes when both seats are top teams) and the replay deleted.
Writes data/top50/fetch/{leaderboard.csv, crawl_subs.json, crawl_episodes.json, index.tsv, tapes/sNN/*.json}.
Re-running resumes: tapes already on disk are skipped, the crawl is merged.
"""
import argparse
import base64
import datetime as D
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python"))
from replay_to_tape import convert  # noqa: E402

OUT = os.path.join(RL, "data", "top50", "fetch")
KAGGLE = os.environ.get("KAGGLE_BIN") or os.path.join(RL, ".venv", "bin", "kaggle")
ENDPOINT = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
OUR_SUBS = [56581792, 56581889, 56567216, 56538921, 56538836, 56524454]
SHARDS = 24

MIN_FREE_GB = float(os.environ.get("KRL_MIN_FREE_GB", "15"))


def disk_ok(path):
    """Stop before the box's disk gets low (operator rule: never overflow it)."""
    import shutil as _sh
    free = _sh.disk_usage(path).free / 2 ** 30
    if free < MIN_FREE_GB:
        print(f"[disk] only {free:.1f} GB free under {path} (< {MIN_FREE_GB:.0f} GB): stopping", flush=True)
        return False
    return True



def kaggle(*args, tries=7):
    wait = 30
    for _ in range(tries):
        r = subprocess.run([KAGGLE if os.path.exists(KAGGLE) else "kaggle", *args], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        if "429" in r.stdout + r.stderr:
            print(f"[fetch] 429, waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 600)
            continue
        raise RuntimeError(f"kaggle {' '.join(args)}: {(r.stdout + r.stderr)[-300:]}")
    raise RuntimeError(f"kaggle {' '.join(args)}: still rate limited")


def auth():
    """kaggle.json (username + key) -> Basic; the newer ~/.kaggle/access_token (the box) -> Bearer."""
    p = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(p):
        c = json.load(open(p, encoding="utf-8"))
        return "Basic " + base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    return "Bearer " + open(os.path.expanduser("~/.kaggle/access_token"), encoding="utf-8").read().strip()


def list_episodes(sid, hdr):
    req = urllib.request.Request(ENDPOINT, data=json.dumps({"submissionId": int(sid)}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": hdr})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()).get("episodes") or []
        except Exception:  # noqa: BLE001
            time.sleep(3 + 6 * i)
    return None


def leaderboard(top):
    with tempfile.TemporaryDirectory() as tmp:
        kaggle("competitions", "leaderboard", "kaggriculture", "-d", "-p", tmp)
        z = glob.glob(os.path.join(tmp, "*.zip"))[0]
        zipfile.ZipFile(z).extractall(tmp)
        lb = pd.read_csv(glob.glob(os.path.join(tmp, "*.csv"))[0])
    lb.to_csv(os.path.join(OUT, "leaderboard.csv"), index=False)
    return lb.sort_values("Score", ascending=False).head(top).reset_index(drop=True)


def crawl(top_ids, seeds, since, max_queries):
    hdr = auth()
    fs, fe = os.path.join(OUT, "crawl_subs.json"), os.path.join(OUT, "crawl_episodes.json")
    subs = json.load(open(fs)) if os.path.exists(fs) else {}
    eps = json.load(open(fe)) if os.path.exists(fe) else {}
    queried, frontier = set(), [int(s) for s in seeds]

    def covered():
        return {r["team"] for r in subs.values() if r["team"] in top_ids and r.get("last", "") >= since and int(r["sub"]) in queried}

    best, since_gain = 0, 0
    while frontier and len(queried) < max_queries and len(covered()) < len(top_ids) and since_gain < 60:
        batch = []
        for s in frontier:
            if s not in queried and s not in batch:
                batch.append(s)
            if len(batch) == 6:
                break
        frontier = [s for s in frontier if s not in batch]
        if not batch:
            break
        with ThreadPoolExecutor(3) as ex:
            res = list(ex.map(lambda s: list_episodes(s, hdr), batch))
        c = len(covered())
        best, since_gain = (c, 0) if c > best else (best, since_gain + len(batch))
        for sid, rows in zip(batch, res):
            queried.add(sid)
            for e in rows or []:
                ag = e.get("agents") or []
                t = str(e.get("endTime") or e.get("createTime") or "")
                eps[str(e["id"])] = {"id": e["id"], "end": t, "type": e.get("type"), "state": e.get("state"),
                                     "agents": [{k: a.get(k) for k in ("submissionId", "teamId", "reward", "initialScore", "updatedScore")} for a in ag]}
                for a in ag:
                    s = a.get("submissionId")
                    if s is None:
                        continue
                    r = subs.setdefault(str(s), {"sub": s, "team": a.get("teamId"), "rating": None, "last": "", "n": 0})
                    sc = a.get("updatedScore") or a.get("initialScore")
                    if sc is not None and t >= r["last"]:
                        r["rating"], r["last"] = sc, t
                    r["n"] += 1
        # next: unqueried submissions of top teams first (newest first), then other high-rated ones as bridges
        cand = [r for r in subs.values() if int(r["sub"]) not in queried]
        mine = sorted((r for r in cand if r["team"] in top_ids), key=lambda r: r["last"], reverse=True)
        bridge = sorted((r for r in cand if r["team"] not in top_ids and (r["rating"] or 0) >= 2600), key=lambda r: -(r["rating"] or 0))
        frontier = [int(r["sub"]) for r in mine] + [int(r["sub"]) for r in bridge[:30]] + frontier
        if len(queried) % 24 == 0:
            print(f"[fetch] crawl: {len(queried)} queried, {len(subs)} submissions, {len(eps)} games, "
                  f"top teams covered {len(covered())}/{len(top_ids)}", flush=True)
            json.dump(subs, open(fs, "w"))
            json.dump(eps, open(fe, "w"))
    json.dump(subs, open(fs, "w"))
    json.dump(eps, open(fe, "w"))
    print(f"[fetch] crawl done: {len(queried)} queried, top teams covered {len(covered())}/{len(top_ids)}", flush=True)
    return subs, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--per-team", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-queries", type=int, default=800)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lb = leaderboard(a.top)
    top_ids = set(int(x) for x in lb.TeamId)
    name = dict(zip(lb.TeamId.astype(int), lb.TeamName))
    print(f"[fetch] top {len(lb)}: {lb.Score.min():.0f}-{lb.Score.max():.0f}", flush=True)
    since = (D.datetime.utcnow() - D.timedelta(days=a.days)).strftime("%Y-%m-%d")
    # seeds: our submissions + the top teams' submissions known to the corpus index (newest first)
    seeds = list(OUR_SUBS)
    try:
        ix = pd.read_parquet(os.path.join(RL, "data", "slim", "s1", "index", "episodes.parquet"),
                             columns=["team_name_0", "team_name_1", "submission_id_0", "submission_id_1", "end_time"])
        for s in (0, 1):
            g = ix[ix[f"team_name_{s}"].isin(set(lb.TeamName))].sort_values("end_time", ascending=False)
            seeds += [int(x) for x in g[f"submission_id_{s}"].dropna().unique()[:200]]
    except Exception as e:  # noqa: BLE001
        print(f"[fetch] corpus seeds unavailable: {e}", flush=True)
    subs, eps = crawl(top_ids, seeds, since, a.max_queries)
    # games per top team in the window
    want = {}
    for e in eps.values():
        if e["end"][:10] < since or "COMPLETED" not in str(e.get("state") or "COMPLETED") or "PUBLIC" not in str(e.get("type") or "PUBLIC"):
            continue
        for seat, ag in enumerate(e["agents"]):
            if ag.get("teamId") in top_ids:
                want.setdefault(ag["teamId"], []).append((e["end"], e["id"], seat, ag.get("submissionId"),
                                                           ag.get("initialScore"), e["agents"][1 - seat] if len(e["agents"]) > 1 else {}))
    jobs = []
    for tid, rows in want.items():
        rows.sort(reverse=True)
        jobs += [(tid,) + r for r in rows[:a.per_team]]
    print(f"[fetch] {len(jobs)} player-games to fetch for {len(want)} teams in the last {a.days} days; "
          f"teams with none: {[name[t] for t in top_ids - set(want)]}", flush=True)
    by_ep = {}
    for j in jobs:
        by_ep.setdefault(j[2], []).append(j)
    have = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(OUT, "tapes", "*", "*.json"))}
    todo = [(ep, js) for ep, js in by_ep.items() if any(f"{ep}_{j[3]}" not in have for j in js)]
    print(f"[fetch] {len(todo)} replays to download ({len(by_ep) - len(todo)} already on disk)", flush=True)

    def one(item):
        ep, js = item
        if not disk_ok(OUT):
            return 0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kaggle("competitions", "replay", str(ep), "-p", tmp, "-q")
                f = glob.glob(os.path.join(tmp, "*.json"))
                if not f:
                    return 0
                rp = json.load(open(f[0], encoding="utf-8"))
        except Exception as ex:  # noqa: BLE001
            print(f"[fetch] {ep}: {ex}", flush=True)
            return 0
        n = 0
        for (tid, end, _, seat, sub, rating, opp) in js:
            t = convert(rp, seat)
            t.update(team=name[tid], team_id=tid, sub=sub, rating=rating, opp_team_id=opp.get("teamId"), opp_sub=opp.get("submissionId"),
                     opp_rating=opp.get("initialScore"), end=end, team_names=rp.get("info", {}).get("TeamNames"))
            d = os.path.join(OUT, "tapes", f"s{int(ep) % SHARDS:02d}")
            os.makedirs(d, exist_ok=True)
            json.dump(t, open(os.path.join(d, f"{ep}_{seat}.json"), "w", encoding="utf-8"))
            n += 1
        return n

    done = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, n in enumerate(ex.map(one, todo), 1):
            done += n
            if i % 25 == 0:
                print(f"[fetch] {i}/{len(todo)} replays, {done} tapes", flush=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "tapes", "*", "*.json"))):
        t = json.load(open(f, encoding="utf-8"))
        rw = t.get("rewards") or [0, 0]
        s = t["seat"]
        rows.append({"id": os.path.basename(f)[:-5], "eid": str(t["id"]), "seat": s, "team": t.get("team"), "team_id": t.get("team_id"),
                     "sub": t.get("sub"), "r": t.get("rating"), "opp_team_id": t.get("opp_team_id"), "opp_sub": t.get("opp_sub"),
                     "opp_r": t.get("opp_rating"), "end": t.get("end"), "seed": t.get("seed"),
                     "bank": rw[s], "bank_rv": rw[1 - s], "shard": os.path.basename(os.path.dirname(f))})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.tsv"), sep="\t", index=False)
    print(f"[fetch] -> data/top50/fetch: {len(rows)} tapes; per team " + str(pd.DataFrame(rows).team.value_counts().describe().round(1).to_dict()))


if __name__ == "__main__":
    main()
