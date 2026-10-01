"""Build (and optionally push) a Kaggle Notebook that runs the analysis on-platform.

Why this exists: the daily episode datasets are ~25 MB a replay and 142 GB in
total. Downloading them to a laptop is the slow, expensive way to look at them.
On Kaggle the same datasets are **mounted read-only at /kaggle/input** -- no
download, no egress, and the notebook runs on Kaggle's compute.

The notebook is self-contained: our agent is embedded as a base85+zlib payload
(the same packaging the top competitors use for their submissions), so the
notebook has no dependency on this repo. It:

  1. walks every mounted episode file, ranked by the agents' rating
  2. hands our agent *their* observation at sampled turns and records what it
     would have done instead
  3. aggregates the field's market schedule -- units of each product sold per
     day and hour -- which is the artefact the top write-ups say carries the
     remaining edge
  4. writes policy_diff.csv / market_schedule.csv / top_trajectory.csv to
     /kaggle/working so they can be downloaded in one go

    python -m kaggriculture.agentbuild.build_kaggle_notebook --agent agents/v9_cem.py
    python -m kaggriculture.agentbuild.build_kaggle_notebook --push          # private by default
    python -m kaggriculture.agentbuild.build_kaggle_notebook --push --days 2026-08-05 2026-08-04
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import json
import os
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(ROOT, "notebooks", "kaggle_mine")
SLUG = "kaggriculture-policy-diff-vs-top-ladder"


def encode_agent(path):
    raw = open(path, "rb").read()
    blob = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    return [blob[i:i + 110] for i in range(0, len(blob), 110)]


ANALYSIS = r'''
# ---------------------------------------------------------------- analysis --
import csv, json, os, glob, importlib.util
from collections import defaultdict

TPD = 24
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER"]
SAMPLE_EVERY = 6

spec = importlib.util.spec_from_file_location("our_agent", AGENT_PATH)
AGENT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AGENT)
CONFIG = {"episodeSteps": 720, "turnsPerDay": TPD}


def seat_observation(steps, t, seat):
    obs = dict(steps[t][seat].get("observation") or {})
    shared = steps[t][0].get("observation") or {}
    for key in ("farms", "market", "town", "day", "hour", "step"):
        if key not in obs and key in shared:
            obs[key] = shared[key]
    obs["player"] = seat
    return obs


def op_class(action):
    if not isinstance(action, list) or not action:
        return "NONE"
    return "MOVE" if action[0] in MOVES else action[0]


def manifest_rows():
    """Every mounted episode, best-rated first."""
    rows = []
    for man in glob.glob("/kaggle/input/**/manifest.csv", recursive=True):
        base = os.path.dirname(man)
        with open(man, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                eid = r.get("episode_id")
                if not eid:
                    continue
                path = os.path.join(base, f"{eid}.json")
                if not os.path.exists(path):
                    continue
                try:
                    score = float(r.get("avg_score") or 0)
                except ValueError:
                    score = 0.0
                rows.append((score, path))
    if not rows:                      # dataset without a manifest
        for path in glob.glob("/kaggle/input/**/*.json", recursive=True):
            rows.append((0.0, path))
    rows.sort(reverse=True)
    return rows


diff = defaultdict(float)
sched = defaultdict(float)
traj = []
n_done = 0

for score, path in manifest_rows()[:MAX_EPISODES]:
    try:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
    except Exception:
        continue
    steps = rep.get("steps") or []
    rewards = rep.get("rewards") or [0, 0]
    if len(steps) < 100:
        continue
    winner = 0 if float(rewards[0] or 0) >= float(rewards[1] or 0) else 1
    episode = os.path.basename(path).split(".")[0]

    for t in range(0, len(steps) - 1, SAMPLE_EVERY):
        obs = seat_observation(steps, t, winner)
        if "farms" not in obs:
            continue
        theirs = steps[t + 1][winner].get("action")
        if not isinstance(theirs, dict):
            continue
        try:
            ours = AGENT.agent(obs, CONFIG)
        except Exception:
            continue
        tops = [theirs.get("farmer")] + list(theirs.get("hands") or [])
        oops = [ours.get("farmer")] + list(ours.get("hands") or [])
        for i in range(max(len(tops), len(oops))):
            a = op_class(tops[i] if i < len(tops) else None)
            b = op_class(oops[i] if i < len(oops) else None)
            diff["unit|%s|%s" % (a, b)] += 1
        tm, om = defaultdict(float), defaultdict(float)
        for o in theirs.get("market") or []:
            if isinstance(o, list) and o:
                tm[o[0]] += float(o[2]) if len(o) > 2 else 1.0
        for o in ours.get("market") or []:
            if isinstance(o, list) and o:
                om[o[0]] += float(o[2]) if len(o) > 2 else 1.0
        for op in set(tm) | set(om):
            diff["market|%s|theirs" % op] += tm.get(op, 0.0)
            diff["market|%s|ours" % op] += om.get(op, 0.0)

    for t in range(len(steps) - 1):
        day, hour = t // TPD, t % TPD
        for seat in (0, 1):
            act = steps[t + 1][seat].get("action") if seat < len(steps[t + 1]) else None
            if not isinstance(act, dict):
                continue
            for o in act.get("market") or []:
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in PRODUCTS:
                    sched[(day, hour, o[1])] += float(o[2]) / 2.0

    for t in range(0, len(steps), TPD):
        obs0 = steps[t][0].get("observation") or {}
        farms = obs0.get("farms") or []
        if winner >= len(farms):
            continue
        farm = farms[winner]
        counts = defaultdict(int)
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    counts[tile.get("crop") or tile.get("animal") or tile.get("kind")] += 1
        traj.append(dict(episode=episode, score=score, day=obs0.get("day", t // TPD),
                         money=farm.get("money", 0), hands=len(farm.get("hands") or []),
                         quads=len(farm.get("unlocked_quadrants") or []),
                         **{k: counts.get(k, 0) for k in
                            ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                             "GOOSE", "COW", "SHEEP", "COOP", "PASTURE", "WEED")}))
    n_done += 1
    if n_done % 25 == 0:
        print(f"  {n_done} episodes analysed", flush=True)

print(f"analysed {n_done} episodes")

with open("/kaggle/working/policy_diff.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["kind", "theirs", "ours", "count", "per_episode"])
    for key, v in sorted(diff.items(), key=lambda kv: -kv[1]):
        kind, a, b = key.split("|")
        w.writerow([kind, a, b, round(v, 1), round(v / max(1, n_done), 2)])

with open("/kaggle/working/market_schedule.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["day", "hour", "product", "units_per_episode"])
    for (day, hour, item), v in sorted(sched.items()):
        w.writerow([day, hour, item, round(v / max(1, n_done), 4)])

if traj:
    with open("/kaggle/working/top_trajectory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj[0]))
        w.writeheader()
        w.writerows(traj)

# ------------------------------------------------------------------ report --
unit = defaultdict(lambda: defaultdict(float))
market = defaultdict(lambda: defaultdict(float))
for key, v in diff.items():
    kind, a, b = key.split("|")
    (unit if kind == "unit" else market)[a][b] += v / max(1, n_done)
total = sum(sum(v.values()) for v in unit.values()) or 1.0
agree = sum(v.get(k, 0.0) for k, v in unit.items())
print("\nunit-op agreement: %.1f%%" % (100 * agree / total))
print("\n%-20s%9s  what we did instead" % ("they did", "per ep"))
for their, ours in sorted(unit.items(), key=lambda kv: -sum(kv[1].values())):
    tot = sum(ours.values())
    if tot < 0.5:
        continue
    wrong = sorted(((v, k) for k, v in ours.items() if k != their), reverse=True)[:3]
    print("%-20s%9.0f  %3.0f%% match   %s" % (
        their, tot, 100 * ours.get(their, 0) / tot,
        "  ".join("%s %.0f%%" % (k, 100 * v / tot) for v, k in wrong)))
print("\n%-20s%11s%10s   ratio" % ("market op", "theirs/ep", "ours/ep"))
for op, sides in sorted(market.items()):
    th, ou = sides.get("theirs", 0.0), sides.get("ours", 0.0)
    print("%-20s%11.0f%10.0f   %5.2fx" % (op, th, ou, (ou / th) if th else 0))
'''


def build(agent_path, days, max_episodes):
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = encode_agent(agent_path)
    literal = "\n".join(f"    '{p}'" for p in parts)

    header = f'''"""Kaggriculture: what would my agent do in the leaders' games?

Runs entirely on Kaggle. The daily episode datasets are mounted read-only at
/kaggle/input, so nothing is downloaded -- this walks {max_episodes} of the
highest-rated episodes, replays each sampled game state through our agent, and
reports where our action differs from the action that actually won.

Outputs (in /kaggle/working):
  policy_diff.csv       our action vs theirs, aggregated by op
  market_schedule.csv   units of each product the field sells per day and hour
  top_trajectory.csv    hands / herd / tiles / cash by day for the winning seat

The agent below is embedded as a base85+zlib payload so this notebook is
self-contained.
"""
import base64, zlib

MAX_EPISODES = {max_episodes}
AGENT_PATH = "/kaggle/working/our_agent.py"

_AGENT_B85 = (
{literal}
)
with open(AGENT_PATH, "wb") as fh:
    fh.write(zlib.decompress(base64.b85decode(_AGENT_B85)))
print("agent written:", AGENT_PATH)
'''
    script = header + ANALYSIS
    script_path = os.path.join(OUT_DIR, f"{SLUG}.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)

    sources = [f"kaggle/kaggriculture-episodes-{d}" for d in days]
    meta = {
        "id": f"{kaggle_user()}/{SLUG}",
        "title": "Kaggriculture: policy diff vs the top ladder",
        "code_file": f"{SLUG}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": sources,
        "competition_sources": ["kaggriculture"],
        "kernel_sources": [],
    }
    meta_path = os.path.join(OUT_DIR, "kernel-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return script_path, meta_path, meta


def kaggle_user():
    for path in (os.path.expanduser("~/.kaggle/kaggle.json"),
                 os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json")):
        if path and os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))["username"]
            except Exception:                                      # noqa: BLE001
                pass
    return os.environ.get("KAGGLE_USERNAME", "USERNAME")


def push():
    import kaggriculture.data.episodes as E
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = E.kaggle_cmd() + ["kernels", "push", "-p", OUT_DIR]
    proc = subprocess.run(cmd, capture_output=True, env=env)
    out = (proc.stdout or b"").decode("utf-8", "replace")
    err = (proc.stderr or b"").decode("utf-8", "replace")
    return proc.returncode, out + err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "v9_cem.py"))
    ap.add_argument("--days", nargs="+",
                    default=["2026-08-05", "2026-08-04", "2026-08-03"])
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--push", action="store_true",
                    help="upload as a PRIVATE notebook under your account")
    args = ap.parse_args()

    script, meta, m = build(args.agent, args.days, args.max_episodes)
    print(f"notebook  -> {os.path.relpath(script, ROOT)}  ({os.path.getsize(script):,} bytes)")
    print(f"metadata  -> {os.path.relpath(meta, ROOT)}")
    print(f"id        : {m['id']}   private={m['is_private']}")
    print(f"inputs    : {', '.join(m['dataset_sources'])}")
    if args.push:
        rc, out = push()
        print(out.strip())
        if rc != 0:
            return 1
    else:
        print("\nnot pushed. To run it on Kaggle's compute:")
        print("  python -m kaggriculture.agentbuild.build_kaggle_notebook --push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
