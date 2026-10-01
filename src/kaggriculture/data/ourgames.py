"""Mine our own ladder games and find out why we lose the ones we lose.

    python -m kaggriculture.data.ourgames --submission 55314513
    python -m kaggriculture.data.ourgames --submission 55314513 --limit 80 --jobs 6
    python -m kaggriculture.data.ourgames --report

Every other measurement in this project is a proxy. The panel plays fixed
recorded tapes that do not react; the tournament plays candidates against each
other. This plays nothing -- it reads what actually happened on the ladder, to
the agent that is actually submitted, against the opponents it actually drew.

That matters because the proxies have been optimistic. `v18_route` scores 95.2%
against a 70-opponent held-out panel and is converging near 2,350 on the ladder,
behind a copied notebook at ~2,500. One of those numbers is wrong about the
world and it is not the ladder.

Streams like every other miner: fetch a replay, reduce it, delete it. For each
game it records who we played, both banks, the winner, and -- for losses -- a
per-product cash-flow comparison, because the losses so far have all been the
same shape and it is worth knowing whether that generalises.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import collections
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.episodes as E  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402

WORK = os.path.join(ROOT, "data", "ourgames")
STAGE = os.path.join(WORK, "_stage")
INDEX = os.path.join(WORK, "index.json")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
            "WOOL", "FERTILIZER")


def _ascii(text):
    """Console-safe rendering of a team name.

    Team names are arbitrary Unicode and a Windows console is cp1252, so an
    unguarded print raises UnicodeEncodeError *inside the download loop* and
    kills the whole mine. That is how a 116-episode run silently stopped at
    20 -- the failure looked like "no more episodes", not like a crash.
    """
    return str(text).encode("ascii", "replace").decode("ascii")


def load_index():
    if os.path.exists(INDEX):
        try:
            with open(INDEX, encoding="utf-8") as fh:
                return json.load(fh)
        except ValueError:
            pass
    return {"games": {}, "updated": None}


def save_index(idx):
    import time
    os.makedirs(WORK, exist_ok=True)
    idx["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=1, default=str)
    os.replace(tmp, INDEX)


def _sold(replay, seat):
    """Units sold per product, and the turn of each sale, for one seat."""
    steps = replay.get("steps") or []
    per_item = collections.Counter()
    by_day = collections.Counter()
    for i in range(len(steps)):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if not (nxt and seat < len(nxt)):
            continue
        act = nxt[seat].get("action")
        if not isinstance(act, dict):
            continue
        for o in act.get("market") or []:
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                    and o[1] in PRODUCTS):
                try:
                    n = max(0, int(o[2]))
                except (TypeError, ValueError):
                    continue
                per_item[o[1]] += n
                by_day[i // 24] += n
    return per_item, by_day


def analyse(path, me_name):
    """One game, reduced. Returns None if we are not in it."""
    with open(path, encoding="utf-8") as fh:
        replay = json.load(fh)
    info = replay.get("info") or {}
    teams = list(info.get("TeamNames") or [])
    if me_name not in teams:
        return None
    seat = teams.index(me_name)
    rewards = [float(r or 0) for r in (replay.get("rewards") or [0, 0])]
    statuses = list(replay.get("statuses") or [])
    mine, theirs = rewards[seat], rewards[1 - seat]

    my_sold, my_days = _sold(replay, seat)
    their_sold, their_days = _sold(replay, 1 - seat)

    # Cash by day for both seats. Knowing *that* we lose is not actionable;
    # knowing the day the gap opens is, because the route is fixed and every
    # day maps to a known block of it.
    steps = replay.get("steps") or []
    money, opp_money = [], []
    for t in range(0, len(steps), 24):
        obs = (steps[t][0] or {}).get("observation") or {}
        farms = obs.get("farms") or []
        if len(farms) < 2:
            continue
        money.append(float(farms[seat].get("money", 0) or 0))
        opp_money.append(float(farms[1 - seat].get("money", 0) or 0))
    gap = [a - b for a, b in zip(money, opp_money)]
    # The day the deficit starts and never recovers.
    broke = None
    if gap:
        final = gap[-1]
        if final < 0:
            for d in range(len(gap) - 1, -1, -1):
                if gap[d] >= 0:
                    broke = d + 1
                    break
            else:
                broke = 0
    return {
        "money_by_day": money, "opp_money_by_day": opp_money,
        "gap_by_day": gap, "broke_day": broke,
        "episode": str(info.get("EpisodeId") or R._episode_from_name(path)),
        "seat": seat,
        "opponent": teams[1 - seat] if len(teams) > 1 else "?",
        "bank": mine, "opp_bank": theirs, "margin": mine - theirs,
        "won": mine > theirs, "tied": mine == theirs,
        "status": statuses[seat] if seat < len(statuses) else "?",
        "opp_status": statuses[1 - seat] if len(statuses) > 1 else "?",
        "sold": dict(my_sold), "opp_sold": dict(their_sold),
        "sold_total": sum(my_sold.values()),
        "opp_sold_total": sum(their_sold.values()),
    }


def mine(submission, limit=60, jobs=6, verbose=True):
    os.makedirs(STAGE, exist_ok=True)
    idx = load_index()
    me = E.leaderboard_rows(verbose=False)
    me_name = None
    try:
        import kaggriculture.data.registry as registry  # noqa: F401
        me_name = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"),
                                 encoding="utf-8")).get("username")
    except Exception:                                              # noqa: BLE001
        pass
    # The replay stores the *team* name, which need not equal the username.
    board = {r["team"] for r in me}
    team = None
    for row in me:
        if row["team"].lower() == str(me_name or "").lower():
            team = row["team"]
            break
    if team is None:
        team = "Debmalya" if "Debmalya" in board else (me_name or "")
    if verbose:
        print(f"our team on the leaderboard: {team!r}")

    ids = E._own_episode_ids(str(submission), verbose=False)
    todo = [e for e in ids if e not in idx["games"]][:limit]
    if verbose:
        print(f"submission {submission}: {len(ids)} episode(s), "
              f"{len(todo)} new, downloading {jobs} at a time\n")
    if not todo:
        return idx

    def grab(eid):
        dest = os.path.join(STAGE, eid)
        os.makedirs(dest, exist_ok=True)
        # kaggleusercontent first: `kaggle competitions replay` 429s on every
        # episode once hit at any rate (measured 2026-08-11, this whole mine
        # returned "failed:" for all 44 episodes), while the direct URL served
        # 31.9 MB in 10.2 s. Same helper sameday.py uses, imported rather than
        # copied so the fallback logic has one home.
        try:
            import kaggriculture.data.sameday as sameday
            path = sameday._grab_direct(eid, dest)
            return eid, path, None
        except Exception as direct_exc:                            # noqa: BLE001
            direct_err = f"direct: {type(direct_exc).__name__}"
            if str(direct_exc).startswith("QUOTA-HOLD"):
                # Budget decision, not a transport failure; the CLI spends
                # the same quota slower. Skip it entirely.
                return eid, None, str(direct_exc)[:100]
        try:
            E._run(["kaggle", "competitions", "replay", str(eid), "-p", dest])
        except Exception as exc:                                   # noqa: BLE001
            return eid, None, f"{direct_err}; cli: {str(exc).splitlines()[0]}"
        E._unzip_inplace(dest)
        hits = [os.path.join(dest, f) for f in os.listdir(dest)
                if f.endswith(".json")]
        return eid, (hits[0] if hits else None), None

    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(grab, e): e for e in todo}
        for fut in as_completed(futures):
            eid, path, err = fut.result()
            if err or not path:
                if verbose:
                    print(f"  ! {eid}: {err or 'no replay file'}")
                continue
            try:
                row = analyse(path, team)
            except Exception as exc:                               # noqa: BLE001
                row = None
                if verbose:
                    print(f"  ! {eid}: {exc}")
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if row:
                # Which of our submissions played it -- so a per-model record
                # can be read back without re-downloading anything.
                row["submission"] = str(submission)
                idx["games"][eid] = row
                done += 1
                if verbose:
                    mark = "WON " if row["won"] else ("TIE " if row["tied"] else "LOST")
                    print(f"  {mark} {row['bank']:>10,.0f} vs {row['opp_bank']:>10,.0f}"
                          f"  {row['margin']:>+10,.0f}  {_ascii(row['opponent'])[:26]}")
            if done % 10 == 0:
                save_index(idx)
    save_index(idx)
    if verbose:
        print(f"\nmined {done} game(s); replays deleted")
    return idx


def report(idx=None):
    idx = idx or load_index()
    games = list(idx["games"].values())
    if not games:
        print("no games mined yet -- run: python -m kaggriculture.data.ourgames --submission <id>")
        return 1
    wins = [g for g in games if g["won"]]
    losses = [g for g in games if not g["won"] and not g["tied"]]
    ties = [g for g in games if g["tied"]]
    print(f"{len(games)} ladder game(s): {len(wins)}W {len(losses)}L {len(ties)}T "
          f"= {100 * len(wins) / len(games):.1f}% win rate\n")

    bad = [g for g in games if g["status"] not in ("DONE", "ACTIVE")]
    if bad:
        print(f"!! {len(bad)} game(s) where OUR agent did not finish cleanly: "
              f"{collections.Counter(g['status'] for g in bad)}")

    print(f"{'':<10}{'n':>4}{'mean bank':>12}{'mean opp':>12}{'mean margin':>13}"
          f"{'mean sold':>11}{'opp sold':>10}")
    for label, rows in (("wins", wins), ("losses", losses), ("ties", ties)):
        if not rows:
            continue
        print(f"{label:<10}{len(rows):>4}"
              f"{statistics.mean(g['bank'] for g in rows):>12,.0f}"
              f"{statistics.mean(g['opp_bank'] for g in rows):>12,.0f}"
              f"{statistics.mean(g['margin'] for g in rows):>+13,.0f}"
              f"{statistics.mean(g['sold_total'] for g in rows):>11,.0f}"
              f"{statistics.mean(g['opp_sold_total'] for g in rows):>10,.0f}")

    if losses:
        print(f"\nworst {min(8, len(losses))} loss(es):")
        for g in sorted(losses, key=lambda g: g["margin"])[:8]:
            print(f"  {g['margin']:>+11,.0f}  {g['bank']:>10,.0f} vs "
                  f"{g['opp_bank']:>10,.0f}  seat{g['seat']}  {_ascii(g['opponent'])[:28]}")

        print("\nper-product sales, wins vs losses (units/game):")
        print(f"{'product':<13}{'in wins':>10}{'in losses':>11}{'opp (wins)':>12}"
              f"{'opp (loss)':>12}")
        for p in PRODUCTS:
            w = statistics.mean([g["sold"].get(p, 0) for g in wins]) if wins else 0
            l = statistics.mean([g["sold"].get(p, 0) for g in losses])
            ow = statistics.mean([g["opp_sold"].get(p, 0) for g in wins]) if wins else 0
            ol = statistics.mean([g["opp_sold"].get(p, 0) for g in losses])
            if max(w, l, ow, ol) < 1:
                continue
            print(f"{p:<13}{w:>10,.0f}{l:>11,.0f}{ow:>12,.0f}{ol:>12,.0f}")

        opps = collections.Counter(_ascii(g["opponent"]) for g in losses)
        print(f"\nopponents we lose to: "
              + ", ".join(f"{k} x{v}" for k, v in opps.most_common(8)))

        # When does the deficit open? The route is fixed, so a day maps to a
        # known block of it -- this says which block to look at.
        broke = [g["broke_day"] for g in losses if g.get("broke_day") is not None]
        if broke:
            print(f"\nday the deficit opens and never closes "
                  f"(n={len(broke)}): median {statistics.median(broke):.0f}, "
                  f"range {min(broke)}-{max(broke)}")
            hist = collections.Counter(b // 5 * 5 for b in broke)
            for bucket in sorted(hist):
                print(f"  days {bucket:>2}-{bucket + 4}: "
                      + "#" * hist[bucket] + f" {hist[bucket]}")

        # Mean cash gap per day, wins vs losses: where do the curves separate?
        def curve(rows):
            n = max((len(g.get("gap_by_day") or []) for g in rows), default=0)
            return [statistics.mean([g["gap_by_day"][d] for g in rows
                                     if len(g.get("gap_by_day") or []) > d])
                    for d in range(n)]
        cw, cl = curve(wins) if wins else [], curve(losses)
        if cw and cl:
            print(f"\nmean cash gap by day (us minus them):")
            print(f"{'day':>4}{'in wins':>12}{'in losses':>12}")
            for d in range(0, min(len(cw), len(cl)), 3):
                print(f"{d:>4}{cw[d]:>+12,.0f}{cl[d]:>+12,.0f}")
    return 0


def our_submissions(verbose=True):
    """Every submission we have ever made, newest first."""
    rows = E._own_submission_rows()
    out = []
    for r in rows:
        ref = r.get("ref") or r.get("id")
        if ref is None:
            continue
        out.append({"ref": str(ref),
                    "score": r.get("publicScore"),
                    "date": str(r.get("date") or "")[:19],
                    "status": str(r.get("status") or "")})
    if verbose:
        print(f"{len(out)} submission(s) on the account:")
        for s in out:
            print(f"  {s['ref']:>10}  score {str(s['score']):>8}  {s['date']}")
        print()
    return out


def mine_all(limit=60, jobs=6, verbose=True):
    """Mine our own games for *every* submission, not just the live one.

    Old submissions stop being scored but their episodes stay downloadable,
    so this is how the per-model record gets built once and kept.
    """
    idx = load_index()
    subs = our_submissions(verbose=verbose)
    for s in subs:
        if verbose:
            print(f"--- submission {s['ref']} (score {s['score']}) ---")
        try:
            idx = mine(s["ref"], limit=limit, jobs=jobs, verbose=verbose)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! submission {s['ref']}: {exc}")
    return idx


def by_submission(idx=None):
    """Per-model win rate, straight out of the index."""
    idx = idx or load_index()
    games = list(idx["games"].values())
    seen = collections.defaultdict(list)
    for g in games:
        seen[str(g.get("submission"))].append(g)
    scores = {s["ref"]: s["score"] for s in our_submissions(verbose=False)}
    print(f"{'submission':>12} {'public':>8} {'games':>6} {'W':>4} {'L':>4} "
          f"{'T':>3} {'win%':>6} {'mean bank':>11}")
    for sub, rows in sorted(seen.items(),
                            key=lambda kv: -len(kv[1])):
        w = sum(1 for g in rows if g.get("won"))
        t = sum(1 for g in rows if g.get("tied"))
        bank = sum(g.get("bank") or 0 for g in rows) / max(1, len(rows))
        print(f"{sub:>12} {str(scores.get(sub, '-')):>8} {len(rows):>6} "
              f"{w:>4} {len(rows) - w - t:>4} {t:>3} "
              f"{100.0 * w / max(1, len(rows)):>5.1f}% {bank:>11,.0f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--submission", default=None)
    ap.add_argument("--all", action="store_true",
                    help="mine our games for every submission we ever made")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--by-submission", action="store_true",
                    help="per-model win rate from the games already mined")
    args = ap.parse_args()
    if args.by_submission:
        return by_submission()
    if args.all:
        idx = mine_all(limit=args.limit, jobs=args.jobs)
        print()
        return report(idx)
    if args.submission:
        idx = mine(args.submission, limit=args.limit, jobs=args.jobs)
        print()
        return report(idx)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
