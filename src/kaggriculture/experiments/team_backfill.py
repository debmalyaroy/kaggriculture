"""Backfill team attribution for already-indexed routes.

sameday._ingest labeled every same-day route team "?" until 2026-08-13
(replay_meta has no teams key); the replays are deleted, but the leaderboard
API still knows which episodes each team's submissions played. Walk the top
teams, map episode_id -> team, and relabel index routes whose team is
missing. Uses the 429-backoff already in leaderboard_harvest.

    python src/experiments/team_backfill.py --top 200
"""
from kaggriculture.paths import ROOT
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.leaderboard_harvest as LH  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--per-team-subs", type=int, default=3)
    args = ap.parse_args()

    idx = R.load_index()
    missing = {str(r["episode"]) for r in idx["routes"].values()
               if r.get("team") in (None, "", "?") and r.get("episode")}
    print(f"{len(missing)} episodes with unattributed routes")

    api = LH._api()
    teams = LH.top_teams(api, "kaggriculture", args.top)
    # Episode objects carry agents[].submissionId + index (SEAT) + reward, so
    # attribution is per-seat exact -- naive episode->team labeling would
    # have stamped both seats with one team's name.
    seat2team = {}
    throttled = 0
    for i, t in enumerate(teams):
        subs = None
        for attempt in range(3):
            try:
                subs = api.competition_team_submissions(t["team_id"]) or []
                break
            except Exception as exc:                               # noqa: BLE001
                if "429" in str(exc) and attempt < 2:
                    throttled += 4
                    time.sleep(4 * (attempt + 1))
                else:
                    break
        throttled = max(0, throttled - 1)
        if subs is None:
            continue
        sub_ids = {int(getattr(s, "id", 0) or 0) for s in subs}
        subs = sorted(subs, key=lambda s: str(getattr(s, "date_submitted",
                                                      "")), reverse=True)
        for s in subs[:args.per_team_subs]:
            sid = getattr(s, "id", None)
            if not sid:
                continue
            try:
                eps = api.competition_list_episodes(sid) or []
            except Exception:                                      # noqa: BLE001
                continue
            for e in eps:
                eid = str(getattr(e, "id", "") or "")
                if eid not in missing:
                    continue
                for a in (getattr(e, "agents", None) or []):
                    a = a if isinstance(a, dict) else getattr(
                        a, "__dict__", {})
                    a = {k.lstrip("_"): v for k, v in a.items()}
                    if int(a.get("submissionId") or a.get("submission_id")
                           or 0) in sub_ids:
                        seat = int(a.get("index") or 0)
                        seat2team[(eid, seat)] = (t["team"],
                                                  a.get("reward"))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(teams)} teams, {len(seat2team)} seats "
                  f"attributed so far", flush=True)
        time.sleep(0.05 if not throttled else min(2.0, 0.25 * throttled))

    fixed = skipped = 0
    for rec in idx["routes"].values():
        if rec.get("team") not in (None, "", "?"):
            continue
        hit = seat2team.get((str(rec.get("episode")), int(rec.get("seat", 0))))
        if not hit:
            continue
        team, reward = hit
        # sanity: the API reward must match the route's recorded bank
        if (reward is not None and rec.get("bank") is not None
                and abs(float(reward) - float(rec["bank"])) > 1.0):
            skipped += 1
            continue
        rec["team"] = team
        fixed += 1
    R.save_index(idx)
    print(f"attributed {fixed} routes (per-seat exact; {skipped} skipped on "
          f"reward mismatch) from top-{args.top} teams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
