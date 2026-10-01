"""Identify which known agent a ladder opponent was running.

    python -m kaggriculture.winplan.fingerprint <sub> [--games L|all] [--steps 240] [--field DIR ...]

Feeds the opponent's OWN recorded observations (from our downloaded replay)
to every candidate agent, step by step, and compares the candidate's action
with what the opponent actually did. Agents are deterministic given their
observation history, so an exact match up to step N is the same agent (or a
clone identical up to N). Reports, per opponent, the best candidates by first
divergence step and agreement rate.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from kaggriculture.paths import ROOT

OUR = os.path.join(ROOT, "data", "winplan", "ourladder")
FIELDS = [os.path.join(ROOT, "data", "winplan", "field")]


def _norm(a):
    if not isinstance(a, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    f = a.get("farmer") or ["PASS"]
    return {"farmer": [str(x) for x in f],
            "hands": [[str(x) for x in h] if isinstance(h, list) else [] for h in (a.get("hands") or [])],
            "market": [[str(x) for x in o] if isinstance(o, list) else [] for o in (a.get("market") or [])]}


def _match(task):
    cand_path, obs_seq, act_seq, cfg = task
    sys.path.insert(0, os.path.join(ROOT, "vendor"))
    from kaggle_environments.agent import get_last_callable
    from kaggle_environments.utils import structify
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn = get_last_callable(open(cand_path, encoding="utf-8").read(), path=cand_path)
    except Exception as exc:                                         # noqa: BLE001
        return cand_path, -1, 0.0, f"load {type(exc).__name__}"
    two = getattr(fn, "__code__", None) is None or fn.__code__.co_argcount > 1
    first, agree = None, 0
    for t, (o, want) in enumerate(zip(obs_seq, act_seq)):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ob = structify(copy.deepcopy(o))
                got = fn(ob, structify(dict(cfg))) if two else fn(ob)
        except Exception as exc:                                     # noqa: BLE001
            return cand_path, t if first is None else first, agree / max(1, t), f"err@{t} {type(exc).__name__}"
        if _norm(got) == _norm(want):
            agree += 1
        elif first is None:
            first = t
    n = len(act_seq)
    return cand_path, n if first is None else first, agree / max(1, n), ""


def opponent_stream(sub, game, steps):
    with gzip.open(os.path.join(OUR, str(sub), f"{game['id']}.json.gz"), "rb") as fh:
        rp = json.loads(fh.read())
    s = 1 - game["seat"]
    st = rp["steps"]
    n = min(steps, len(st) - 1)
    obs = []
    for t in range(n):
        o = copy.deepcopy(st[t][s].get("observation") or {})
        shared = st[t][0].get("observation") or {}
        for k, v in shared.items():                                  # seat 1's obs omits shared keys in replays
            o.setdefault(k, v)
        o["player"] = s
        obs.append(o)
    acts = [st[t + 1][s].get("action") for t in range(n)]
    cfg = rp.get("configuration") or {}
    return obs, acts, cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--games", default="L")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--field", action="append")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    fields = a.field or FIELDS
    cands = [os.path.join(d, f) for d in fields for f in sorted(os.listdir(d)) if f.endswith(".py")]
    rows = json.load(open(os.path.join(OUR, str(a.sub), "games.json")))
    games = [r for r in rows if a.games == "all" or r["result"] == a.games]
    out = {}
    for g in sorted(games, key=lambda r: r["end"]):
        obs, acts, cfg = opponent_stream(a.sub, g, a.steps)
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(_match, [(c, obs, acts, cfg) for c in cands]))
        res.sort(key=lambda r: (-r[1], -r[2]))
        out[str(g["id"])] = {"opp_team": g["opp_team"], "result": g["result"], "margin": g["margin"],
                             "top": [{"agent": os.path.basename(p)[:-3], "first_diverge": fd, "agree": round(ag, 3), "note": nt}
                                     for p, fd, ag, nt in res[:5]]}
        best = res[0]
        print(f"{g['result']} {g['margin']:+8,.0f} {g['opp_team'][:18]:18s} -> {os.path.basename(best[0])[:-3][:48]:48s} "
              f"diverge@{best[1]:4d} agree {best[2]:.2f}" + (f"  | 2nd {os.path.basename(res[1][0])[:-3][:30]} @{res[1][1]}" if len(res) > 1 else ""), flush=True)
    json.dump(out, open(os.path.join(OUR, str(a.sub), f"fingerprint_{a.games}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
