"""Ladder intelligence: who is where, what the top teams do, their routes.

* crawl(): BFS over Kaggle's EpisodeService (API-token auth, no replay quota)
  from our submissions out to the top of the ladder -> submissions with
  ratings, and every episode seen with both agents' banks and ratings.
* download_top(): replays of recent games where BOTH seats rate >= floor,
  spread across teams (per-team cap), within a count budget.
* economy(): per player-game revenue by product (walking the engine's own
  price curve), realized price vs base, land timing, hires, seed/animal spend.
* extract_routes(): each top player's action stream as a kagg tape, indexed
  by the first two shops unlocked (the world the route was played in).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import statistics as st
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from kaggriculture.paths import ROOT
from kaggriculture.winplan import paths as P


def _list_episodes(sid, auth):
    from kaggriculture.pipeline.live_status import ENDPOINT
    req = urllib.request.Request(ENDPOINT, data=json.dumps({"submissionId": int(sid)}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": auth})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()).get("episodes") or []
        except Exception:                                            # noqa: BLE001
            time.sleep(3 + 5 * i)                                    # 429s happen; back off
    return []


def adopt_legacy():
    """Fold this morning's .local/crawl output (crawl json + replays) into data/winplan."""
    import shutil
    legacy = os.path.join(ROOT, ".local", "crawl")
    for name in ("subs.json", "episodes.json"):
        src, dst = os.path.join(legacy, name), os.path.join(P.CRAWL, name)
        if os.path.exists(src) and (not os.path.exists(dst) or os.path.getsize(dst) < os.path.getsize(src)):
            shutil.copy2(src, dst)
    for f in glob.glob(os.path.join(legacy, "replays", "episode-*-replay.json")):
        dst = os.path.join(P.REPLAYS, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)


def crawl(seed_subs, min_rating=2850, max_queries=250, log=print):
    """Merges into the existing crawl files; never overwrites them with less.
    Raises if the service returned nothing (rate-limited / auth), so the task
    fails visibly instead of reporting an empty success."""
    from kaggriculture.pipeline.live_status import _auth_header
    auth = _auth_header()
    try:
        subs = json.load(open(os.path.join(P.CRAWL, "subs.json")))
        eps = json.load(open(os.path.join(P.CRAWL, "episodes.json")))
    except (OSError, ValueError):
        subs, eps = {}, {}
    n_eps0 = len(eps)
    queried, frontier = set(), list(seed_subs)
    while frontier and len(queried) < max_queries:
        batch = [s for s in frontier if s not in queried][:6]
        frontier = [s for s in frontier if s not in batch]
        if not batch:
            break
        with ThreadPoolExecutor(3) as ex:
            results = list(ex.map(lambda s: _list_episodes(s, auth), batch))
        for sid, rows in zip(batch, results):
            queried.add(sid)
            for e in rows:
                ag = e.get("agents") or []
                t = str(e.get("endTime") or e.get("createTime") or "")
                eps[str(e["id"])] = {"id": e["id"], "end": t, "agents": [
                    {k: a.get(k) for k in ("submissionId", "teamId", "reward", "initialScore", "updatedScore")} for a in ag]}
                for a in ag:
                    s = a.get("submissionId")
                    if s is None:
                        continue
                    rec = subs.setdefault(str(s), {"sub": s, "team": a.get("teamId"), "rating": None, "t": "", "n": 0})
                    sc = a.get("updatedScore") or a.get("initialScore")
                    if sc is not None and t >= rec["t"]:
                        rec["rating"], rec["t"] = sc, t
                    rec["n"] += 1
        cand = sorted((r for r in subs.values() if int(r["sub"]) not in queried and (r["rating"] or 0) >= min_rating),
                      key=lambda r: -(r["rating"] or 0))
        frontier = [int(r["sub"]) for r in cand] + frontier
        if len(queried) % 30 == 0:
            log(f"crawl: queried {len(queried)}, submissions {len(subs)}, episodes {len(eps)}")
    if len(eps) == n_eps0 and not n_eps0:
        raise RuntimeError("EpisodeService returned no episodes (rate-limited or auth); crawl files untouched")
    json.dump(subs, open(os.path.join(P.CRAWL, "subs.json"), "w"))
    json.dump(eps, open(os.path.join(P.CRAWL, "episodes.json"), "w"))
    top = [e for e in eps.values() if len(e["agents"]) == 2 and all(
        (a.get("initialScore") or 0) >= min_rating and a.get("reward") is not None for a in e["agents"])]
    return {"queried": len(queried), "submissions": len(subs), "episodes": len(eps), "top_vs_top": len(top)}


def download_top(floor=2950, budget=300, per_team=25, jobs=5, log=print):
    eps = json.load(open(os.path.join(P.CRAWL, "episodes.json")))
    subs = json.load(open(os.path.join(P.CRAWL, "subs.json")))
    team = {k: v["team"] for k, v in subs.items()}
    top = sorted((e for e in eps.values() if len(e["agents"]) == 2 and all(
        (a.get("initialScore") or 0) >= floor and a.get("reward") is not None for a in e["agents"])),
        key=lambda e: e["end"], reverse=True)
    have = {os.path.basename(f).split("-")[1] for f in glob.glob(os.path.join(P.REPLAYS, "episode-*-replay.json"))}
    per, pick = collections.Counter(), []
    for e in top:
        ts = [team.get(str(a["submissionId"])) for a in e["agents"]]
        if str(e["id"]) in have:
            per.update(ts); continue
        if any(per[t] >= per_team for t in ts):
            continue
        pick.append(e); per.update(ts)
        if len(pick) >= budget:
            break

    def dl(e):
        return subprocess.run(["kaggle", "competitions", "replay", str(e["id"]), "-p", P.REPLAYS],
                              capture_output=True, text=True).returncode
    fails = 0
    with ThreadPoolExecutor(jobs) as ex:
        for i, rc in enumerate(ex.map(dl, pick), 1):
            fails += rc != 0
            if i % 25 == 0:
                log(f"replays: {i}/{len(pick)} downloaded ({fails} failed)")
    return {"eligible": len(top), "downloaded": len(pick) - fails, "failed": fails,
            "on_disk": len(glob.glob(os.path.join(P.REPLAYS, "episode-*-replay.json"))), "teams": len(per)}


def _price_fn():
    src = open(os.path.join(P.VENDOR, "kaggle_environments", "envs", "kaggriculture", "kaggriculture.py"), encoding="utf-8").read()
    ns = {"math": math}
    exec(src[src.index("MARKET_I0 = "):src.index("# (dx, dy)")], ns)
    a = src.index("def market_price("); exec(src[a:src.index("\ndef ", a + 10)], ns)
    return ns["market_price"], {k: v["base"] for k, v in ns["MARKET_PARAMS"].items()}


def _economy_one(f, market_price, base):
    rep = json.load(open(f, encoding="utf-8")); S = rep["steps"]; names = rep["info"]["TeamNames"]
    params = (rep.get("configuration") or {}).get("marketParams")
    shops = (S[-1][0]["observation"].get("town") or {}).get("unlocked_shops", [])[:2]
    out = []
    for p in (0, 1):
        rev, units, spend, land, hires = collections.Counter(), collections.Counter(), collections.Counter(), [], 0
        for i in range(len(S) - 1):
            a = S[i + 1][p].get("action") or {}
            inv = dict(S[i][0]["observation"]["market"]["inventory"])
            shed = dict((S[i][p]["observation"].get("private") or {}).get("shed") or {})
            for o in a.get("market") or []:
                if not o:
                    continue
                if o[0] == "BUY_LAND": land.append(i); continue
                if o[0] == "HIRE": hires += 1; continue
                if len(o) < 3 or not isinstance(o[2], int) or o[2] <= 0:
                    continue
                if o[0] == "SELL":
                    q = min(o[2], shed.get(o[1], 0))
                    for _ in range(q):
                        pr = market_price(o[1], inv[o[1]], params); rev[o[1]] += pr; units[o[1]] += 1
                        if pr > 1: inv[o[1]] += 1
                    shed[o[1]] = shed.get(o[1], 0) - q
                elif o[0] == "BUY_SEED":
                    spend["seeds"] += o[2] * {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(o[1], 0)
                elif o[0] == "BUY_ANIMAL":
                    spend["animals"] += o[2] * {"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(o[1], 0)
        out.append({"episode": rep["info"]["EpisodeId"], "team": names[p], "bank": rep["rewards"][p],
                    "won": rep["rewards"][p] > rep["rewards"][1 - p], "shops": "|".join(shops),
                    "rev": dict(rev), "rev_total": sum(rev.values()),
                    "px_ratio": {k: rev[k] / units[k] / base[k] for k in units if units[k]},
                    "land": land, "hires": hires, "spend": dict(spend)})
    return out


def economy(log=print):
    market_price, base = _price_fn()
    rows = []
    for f in sorted(glob.glob(os.path.join(P.REPLAYS, "episode-*-replay.json"))):
        try:
            rows += _economy_one(f, market_price, base)
        except Exception as exc:                                     # noqa: BLE001
            log(f"economy: skip {os.path.basename(f)}: {exc}")
    json.dump(rows, open(os.path.join(P.DATA, "economy.json"), "w"))
    teams = collections.defaultdict(list)
    for r in rows:
        teams[r["team"]].append(r)
    prods = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    win = [r for r in rows if r["won"]]
    summary = {"games": len(rows) // 2, "players": len(rows), "teams": len(teams),
               "winners_rev_by_product": {k: round(st.mean([r["rev"].get(k, 0) for r in win])) for k in prods} if win else {},
               "winners_px_ratio": {k: round(st.mean([r["px_ratio"][k] for r in win if k in r["px_ratio"]]), 2)
                                    for k in prods if any(k in r["px_ratio"] for r in win)},
               "winners_first_land_step": round(st.mean([r["land"][0] for r in win if r["land"]])) if win else None,
               "winners_land_buys": round(st.mean([len(r["land"]) for r in win]), 2) if win else None,
               "winners_bank": round(st.mean([r["bank"] for r in win])) if win else None}
    json.dump(summary, open(os.path.join(P.DATA, "economy_summary.json"), "w"), indent=1)
    return summary


def extract_routes(log=print):
    from kaggriculture.trackp import common as C
    out_dir = os.path.join(P.DATA, "routes"); os.makedirs(out_dir, exist_ok=True)
    index = []
    for f in sorted(glob.glob(os.path.join(P.REPLAYS, "episode-*-replay.json"))):
        try:
            rep = json.load(open(f, encoding="utf-8"))
        except ValueError:
            continue
        shops = (rep["steps"][-1][0]["observation"].get("town") or {}).get("unlocked_shops", [])[:2]
        for seat in (0, 1):
            eid = rep["info"]["EpisodeId"]
            tape = os.path.join(out_dir, f"{eid}_s{seat}.tape")
            if not os.path.exists(tape):
                C.replay_to_tape(rep, seat, tape)
            index.append({"tape": P.rel(tape), "episode": eid, "seat": seat, "team": rep["info"]["TeamNames"][seat],
                          "bank": rep["rewards"][seat], "won": rep["rewards"][seat] > rep["rewards"][1 - seat],
                          "shops": "|".join(shops)})
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=0)
    by_world = collections.Counter(r["shops"] for r in index if r["won"])
    return {"routes": len(index), "winning_routes": sum(r["won"] for r in index),
            "worlds_covered": len(by_world), "top_worlds": by_world.most_common(6)}
