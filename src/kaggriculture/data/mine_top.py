"""Stream the top-rated ladder episodes and ask our agent what it would do.

The ask was "top 50 teams x 1000 runs each". That data does not exist: the
whole public archive is ~5,700 episodes across every team and every day, 142 GB
at ~25 MB a replay. So this takes the other route -- rank every episode by its
agents' rating, pull the best ones, and **do the work while the replay is in
memory**, then delete it. Disk stays flat and nothing is capped by storage.

Three things come out of one pass:

  1. **policy diff** -- our agent is handed *their* observation at sampled
     turns, and we record what it would have done against what they did. This
     is the direct answer to "where does my bot disagree with the leaders".
  2. **market schedule** -- units of each product sold per day and per hour,
     averaged over the field. Two independent top-ten write-ups conclude the
     remaining edge at the top is sale *timing*, so this is the artefact worth
     having.
  3. **farm trajectory** -- hands, herd, tiles and cash by day, to compare the
     shape of the build.

    python -m kaggriculture.data.mine_top --episodes 40 --agent agents/v9_cem.py
    python -m kaggriculture.data.mine_top --episodes 100 --days 4 --jobs 6
    python -m kaggriculture.data.mine_top --report            # re-read what is already mined
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402
import kaggriculture.data.episodes as E  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402

WORK = os.path.join(ROOT, "data", "mine")
OUT_DIFF = os.path.join(ROOT, "data", "policy_diff.csv")
OUT_SCHED = os.path.join(ROOT, "data", "market_schedule.csv")
OUT_TRAJ = os.path.join(ROOT, "data", "top_trajectory.csv")
SEEN = os.path.join(ROOT, "data", "mined_episodes.json")

TPD = 24
UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
            "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER"]

_LOCK = threading.Lock()


def load_agent(path):
    spec = importlib.util.spec_from_file_location("mined_agent", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rank_episodes(days, limit, verbose=True):
    """Episode ids ordered by the rating of the agents that played them."""
    index = E.fetch_index(os.path.join(WORK, "idx"))
    index.sort(key=lambda r: r.get("date", ""))
    chosen = index[-days:] if days else index
    rows = []
    for day_row in reversed(chosen):
        date = day_row["date"]
        dataset = day_row.get("daily_dataset_slug") or f"kaggriculture-episodes-{date}"
        if "/" not in dataset:
            dataset = f"kaggle/{dataset}"
        try:
            man = E._download_dataset_file(dataset, "manifest.csv",
                                           os.path.join(WORK, "idx", date))
        except Exception as exc:                                  # noqa: BLE001
            pr.warn(f"{date}: {exc}", 1)
            continue
        if not man:
            continue
        for r in E._read_csv(man):
            r["_date"] = date
            r["_dataset"] = dataset
            rows.append(r)
    rows.sort(key=E._score_of, reverse=True)
    if verbose and rows:
        pr.log(f"{len(rows):,} episodes across {len(chosen)} day(s); "
               f"best avg rating {E._score_of(rows[0]):.0f}", 1)
    return rows[:limit]


def seat_observation(steps, t, seat):
    """A seat's observation, with the shared keys the replay only stores once."""
    obs = dict(steps[t][seat].get("observation") or {})
    shared = steps[t][0].get("observation") or {}
    for key in ("farms", "market", "town", "day", "hour", "step"):
        if key not in obs and key in shared:
            obs[key] = shared[key]
    obs["player"] = seat
    return obs


def op_class(action):
    """Bucket a unit op so disagreements aggregate into something readable."""
    if not isinstance(action, list) or not action:
        return "NONE"
    op = action[0]
    return "MOVE" if op in MOVES else op


def mine_one(path, agent_mod, config, sample_every):
    """One replay -> (diff rows, schedule rows, trajectory rows)."""
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    steps = rep["steps"]
    rewards = rep.get("rewards") or [0, 0]
    winner = 0 if float(rewards[0] or 0) >= float(rewards[1] or 0) else 1
    episode = os.path.basename(path).split(".")[0].replace("episode-", "").replace("-replay", "")

    diff = defaultdict(int)
    sched = defaultdict(float)
    traj = []
    n = len(steps)

    for t in range(0, n - 1, sample_every):
        obs = seat_observation(steps, t, winner)
        if "farms" not in obs:
            continue
        nxt = steps[t + 1]
        theirs = nxt[winner].get("action") if winner < len(nxt) else None
        if not isinstance(theirs, dict):
            continue
        try:
            ours = agent_mod.agent(obs, config)
        except Exception:                                          # noqa: BLE001
            continue

        their_ops = [theirs.get("farmer")] + list(theirs.get("hands") or [])
        our_ops = [ours.get("farmer")] + list(ours.get("hands") or [])
        for i in range(max(len(their_ops), len(our_ops))):
            a = op_class(their_ops[i] if i < len(their_ops) else None)
            b = op_class(our_ops[i] if i < len(our_ops) else None)
            diff[f"unit|{a}|{b}"] += 1

        their_mkt = defaultdict(float)
        our_mkt = defaultdict(float)
        for o in theirs.get("market") or []:
            if isinstance(o, list) and o:
                their_mkt[o[0]] += float(o[2]) if len(o) > 2 else 1.0
        for o in ours.get("market") or []:
            if isinstance(o, list) and o:
                our_mkt[o[0]] += float(o[2]) if len(o) > 2 else 1.0
        for op in set(their_mkt) | set(our_mkt):
            diff[f"market|{op}|theirs"] += their_mkt.get(op, 0.0)
            diff[f"market|{op}|ours"] += our_mkt.get(op, 0.0)

    # market schedule and trajectory come from the record alone
    for t in range(n - 1):
        day, hour = t // TPD, t % TPD
        nxt = steps[t + 1]
        for seat in (0, 1):
            if seat >= len(nxt):
                continue
            act = nxt[seat].get("action")
            if not isinstance(act, dict):
                continue
            for o in act.get("market") or []:
                if not isinstance(o, list) or len(o) < 3 or o[0] != "SELL":
                    continue
                if o[1] in PRODUCTS:
                    sched[(day, hour, o[1])] += float(o[2]) / 2.0

    # Sample at midday, not at hour 0. The crew is released overnight, so an
    # hour-0 sample reads hands as 0 on every day of every episode -- 9,000
    # rows of the column carried no information at all -- and money is read
    # before the day's selling, about 13% low.
    for t in range(TPD // 2, n, TPD):
        obs0 = steps[t][0].get("observation") or {}
        farms = obs0.get("farms") or []
        if winner >= len(farms):
            continue
        farm = farms[winner]
        counts = defaultdict(int)
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    key = tile.get("crop") or tile.get("animal") or tile.get("kind")
                    counts[key] += 1
        traj.append({"episode": episode, "day": obs0.get("day", t // TPD),
                     "money": farm.get("money", 0), "hands": len(farm.get("hands") or []),
                     "quads": len(farm.get("unlocked_quadrants") or []),
                     **{k: counts.get(k, 0) for k in
                        ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                         "GOOSE", "COW", "SHEEP", "COOP", "PASTURE", "WEED")}})
    return diff, sched, traj, rewards, winner


def run(episodes, days, agent_path, jobs, sample_every, verbose=True):
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DIFF), exist_ok=True)
    seen = set()
    if os.path.exists(SEEN):
        try:
            seen = set(json.load(open(SEEN, encoding="utf-8")))
        except ValueError:
            seen = set()

    rows = rank_episodes(days, episodes * 3, verbose=verbose)
    todo = [r for r in rows if str(r.get("episode_id")) not in seen][:episodes]
    if not todo:
        pr.log("nothing new to mine", 1)
        return
    pr.log(f"mining {len(todo)} episode(s) with {jobs} worker(s); "
           f"replays are deleted after use", 1)

    agent_mod = load_agent(agent_path)
    config = {"episodeSteps": 720, "turnsPerDay": TPD}
    diff_tot = defaultdict(float)
    sched_tot = defaultdict(float)
    traj_all = []
    done = {"n": 0, "bytes": 0}
    tick = pr.Ticker(total=len(todo), label="episodes", indent=2)

    def _one(row):
        eid = str(row.get("episode_id"))
        dest = os.path.join(WORK, eid)
        os.makedirs(dest, exist_ok=True)
        # Kaggle drops connections under heavy parallelism -- at 20 workers a
        # visible share of pulls die with RemoteDisconnected. Retry with a
        # backoff rather than losing the episode.
        path = None
        for attempt in range(4):
            try:
                path = E._download_dataset_file(row["_dataset"], f"{eid}.json", dest)
                if path and os.path.exists(path):
                    break
            except Exception:                                      # noqa: BLE001
                path = None
            time.sleep(1.5 * (attempt + 1))
        try:
            if not path or not os.path.exists(path):
                return None
            size = os.path.getsize(path)
            out = mine_one(path, agent_mod, config, sample_every)
        except Exception as exc:                                   # noqa: BLE001
            pr.warn(f"episode {eid}: {exc}", 3)
            return None
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        return eid, size, out

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for res in pool.map(_one, todo):
            if not res:
                continue
            eid, size, (diff, sched, traj, rewards, winner) = res
            with _LOCK:
                for k, v in diff.items():
                    diff_tot[k] += v
                for k, v in sched.items():
                    sched_tot[k] += v
                traj_all += traj
                seen.add(eid)
                done["n"] += 1
                done["bytes"] += size
                tick.step(f"{eid}  ${rewards[winner]:,.0f}  "
                          f"({done['bytes'] / 1e9:.1f} GB streamed, 0 kept)")
                # Checkpoint as we go. A mine that only writes at the end throws
                # away an hour of streaming if anything interrupts it, and these
                # runs are long by construction.
                if done["n"] % 10 == 0:
                    with open(SEEN, "w", encoding="utf-8") as fh:
                        json.dump(sorted(seen), fh)
                    _write_diff(diff_tot, done["n"])
                    _write_schedule(sched_tot, done["n"])
                    _write_traj(traj_all)
    tick.done()

    with open(SEEN, "w", encoding="utf-8") as fh:
        json.dump(sorted(seen), fh)
    _write_diff(diff_tot, done["n"])
    _write_schedule(sched_tot, done["n"])
    _write_traj(traj_all)
    report()


def _write_diff(diff, n_eps):
    rows = []
    for key, v in sorted(diff.items(), key=lambda kv: -kv[1]):
        kind, a, b = key.split("|")
        rows.append({"kind": kind, "theirs": a, "ours": b, "count": round(v, 1),
                     "per_episode": round(v / max(1, n_eps), 2)})
    with open(OUT_DIFF, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "theirs", "ours", "count", "per_episode"])
        w.writeheader()
        w.writerows(rows)


def _write_schedule(sched, n_eps):
    with open(OUT_SCHED, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "hour", "product", "units_per_episode"])
        for (day, hour, item), v in sorted(sched.items()):
            w.writerow([day, hour, item, round(v / max(1, n_eps), 4)])


def _write_traj(traj):
    if not traj:
        return
    with open(OUT_TRAJ, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj[0]))
        w.writeheader()
        w.writerows(traj)


def report():
    if not os.path.exists(OUT_DIFF):
        print("nothing mined yet -- run: python -m kaggriculture.data.mine_top --episodes 40")
        return
    print("\n" + "=" * 74)
    print("WHERE OUR AGENT DISAGREES WITH THE LEADERS  (their state, our action)")
    print("=" * 74)
    unit = defaultdict(lambda: defaultdict(float))
    market = defaultdict(lambda: defaultdict(float))
    with open(OUT_DIFF, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "unit":
                unit[r["theirs"]][r["ours"]] += float(r["per_episode"])
            else:
                market[r["theirs"]][r["ours"]] += float(r["per_episode"])
    total = sum(sum(v.values()) for v in unit.values()) or 1.0
    agree = sum(v.get(k, 0.0) for k, v in unit.items())
    print(f"  unit-op agreement: {100 * agree / total:.1f}%  "
          f"({total:.0f} sampled unit-turns per episode)\n")
    print(f"  {'they did':<20}{'total':>8}  we did instead")
    for their, ours in sorted(unit.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(ours.values())
        if tot < 0.5:
            continue
        wrong = sorted(((v, k) for k, v in ours.items() if k != their), reverse=True)[:3]
        detail = "  ".join(f"{k} {100 * v / tot:.0f}%" for v, k in wrong)
        print(f"  {their:<20}{tot:>8.0f}  {100 * ours.get(their, 0) / tot:>3.0f}% match   {detail}")
    if market:
        print(f"\n  {'market op':<20}{'theirs/ep':>11}{'ours/ep':>10}   ratio")
        for op, sides in sorted(market.items()):
            th, ou = sides.get("theirs", 0.0), sides.get("ours", 0.0)
            ratio = (ou / th) if th else float("inf")
            print(f"  {op:<20}{th:>11.0f}{ou:>10.0f}   {ratio:>5.2f}x")
    if os.path.exists(OUT_SCHED):
        print(f"\n  market schedule -> {os.path.relpath(OUT_SCHED, ROOT)}")
    if os.path.exists(OUT_TRAJ):
        print(f"  trajectories    -> {os.path.relpath(OUT_TRAJ, ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "v9_cem.py"))
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=6,
                    help="run our agent on every Nth turn (1 = every turn)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    run(args.episodes, args.days, args.agent, args.jobs, args.sample_every)


if __name__ == "__main__":
    main()
