"""E0.3 -- the opponent-pool manifest for Slot-2 self-play / league training.

Writes ``models/opponent_pool.json``: the roster an RL / league trainer plays
against so the policy is measured against a FIXED, strong, diverse field
(house rule 6: never train on an unweighted ladder win rate). Four sources:

  1. ``replay_tape`` -- high-rated seats mined from the top-100 replay corpus,
     recorded as POINTERS (episode_id + seat + source part); the action stream
     is materialised on demand with ``measure.opponents.extract_actions``.
     These are the field's real strategies as scripted tapes. Ratings are not
     carried in the replay payload, so each team is weighted by its empirical
     WIN-SHARE over the corpus (``rating_source: "win_proxy"``) -- the corpus
     is already a top-cohort scrape, so every seat here is strong.
  2. ``ref_tape`` -- the two held-out reference agents in
     ``.local/panel/ref2500/`` (ratings parsed from their filenames).
  3. ``agent`` -- ``agents/v57_trackp.py`` (our current strongest seat).
  4. ``frozen_self`` -- a placeholder the trainer overwrites with the latest
     policy checkpoint (self-play anchor).

Weights are normalised across the whole pool so a sampler can draw directly.

    python -m kaggriculture.data.opponent_pool --limit 800 --top 24
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import pyarrow.parquet as pq

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PARTS = os.path.join(ROOT, ".local", "scratch", "gm", "top100_replays_parts")
PANEL = os.path.join(ROOT, ".local", "panel", "ref2500")
DESTBRESO = os.path.join(ROOT, ".local", "scratch", "datasets", "destbreso",
                         "matchups_all.parquet")
OUT = os.path.join(ROOT, "models", "opponent_pool.json")
OUT_FULL = os.path.join(ROOT, "models", "opponent_pool_full.json")


def _team_names(rep):
    info = rep.get("info") or {}
    names = info.get("TeamNames") or [a.get("Name") for a in
                                      (info.get("Agents") or [])]
    return names if isinstance(names, list) and len(names) == 2 else None


def mine_replay_seats(limit, top):
    """Rank teams by corpus win-share; keep one representative seat each."""
    wins = defaultdict(int)
    games = defaultdict(int)
    best = {}          # team -> (margin, episode_id, seat, part)
    scanned = 0
    for f in sorted(glob.glob(os.path.join(PARTS, "*.parquet"))):
        part = os.path.basename(f)
        try:
            pf = pq.ParquetFile(f)
        except Exception:                                   # noqa: BLE001
            continue
        for b in pf.iter_batches(batch_size=8,
                                 columns=["episode_id", "replay_json"]):
            for eid, rj in zip(b.column("episode_id").to_pylist(),
                               b.column("replay_json").to_pylist()):
                try:
                    rep = json.loads(rj)
                except (ValueError, TypeError):
                    continue
                scanned += 1
                rewards = rep.get("rewards") or []
                names = _team_names(rep)
                if len(rewards) != 2 or not names:
                    continue
                try:
                    r0, r1 = float(rewards[0]), float(rewards[1])
                except (TypeError, ValueError):
                    continue
                w = 0 if r0 >= r1 else 1
                margin = abs(r0 - r1)
                for s in (0, 1):
                    games[names[s]] += 1
                wins[names[w]] += 1
                key = names[w]
                if key not in best or margin > best[key][0]:
                    best[key] = (margin, str(eid), w, part)
                if limit and scanned >= limit:
                    break
            if limit and scanned >= limit:
                break
        if limit and scanned >= limit:
            break

    ranked = sorted(wins.items(),
                    key=lambda kv: (kv[1] / max(1, games[kv[0]]), kv[1]),
                    reverse=True)
    seats = []
    for team, w in ranked[:top]:
        g = games[team]
        _m, eid, seat, part = best[team]
        seats.append({
            "kind": "replay_tape", "team": team,
            "episode_id": eid, "seat": seat, "source_part": part,
            "win_share": round(w / max(1, g), 3), "games": g,
            "rating": None, "rating_source": "win_proxy",
            "materialise": "measure.opponents.extract_actions(<part-row>, seat)",
        })
    return seats, scanned


def mine_destbreso_seats(path, min_rating):
    """One representative tape per distinct team whose rating >= min_rating.

    The destbreso matchups carry the OPPONENT's full 720-turn action tape plus a
    real ladder ``opponent_rating`` (104-3139). Each kept team is a scripted
    replay-tape opponent (POINTER: episode_id + opponent_seat + the parquet;
    materialise the row's ``opponent_actions``). Representative = the team's
    highest-rated row (ties: larger recorded bank)."""
    if not os.path.exists(path):
        return []
    best = {}      # team -> (rating, bank, episode_id, seat, seed)
    pf = pq.ParquetFile(path)
    cols = ["episode_id", "opponent_seat", "opponent_team", "opponent_rating",
            "recorded_bank_opponent", "seed"]
    for b in pf.iter_batches(batch_size=8192, columns=cols):
        eids = b.column("episode_id").to_pylist()
        seats = b.column("opponent_seat").to_pylist()
        teams = b.column("opponent_team").to_pylist()
        rats = b.column("opponent_rating").to_pylist()
        banks = b.column("recorded_bank_opponent").to_pylist()
        seeds = b.column("seed").to_pylist()
        for eid, seat, team, rat, bank, seed in zip(
                eids, seats, teams, rats, banks, seeds):
            if rat is None or float(rat) < min_rating:
                continue
            cur = best.get(team)
            key = (float(rat), float(bank or 0))
            if cur is None or key > (cur[0], cur[1]):
                best[team] = (float(rat), float(bank or 0), str(eid),
                              int(seat or 0), int(seed or 0))
    seats = []
    for team, (rat, bank, eid, seat, seed) in sorted(
            best.items(), key=lambda kv: kv[1][0], reverse=True):
        seats.append({
            "kind": "destbreso_tape", "team": team,
            "episode_id": eid, "seat": seat, "seed": seed,
            "recorded_bank": bank,
            "rating": round(rat, 1), "rating_source": "ladder",
            "source_parquet": os.path.relpath(path, ROOT),
            "materialise": "row of matchups_all.parquet where episode_id==<> "
                           "and opponent_seat==<seat>; json.loads(opponent_actions)",
        })
    return seats


def panel_tapes():
    out = []
    if not os.path.isdir(PANEL):
        return out
    for fn in sorted(os.listdir(PANEL)):
        if not fn.endswith(".py"):
            continue
        m = re.search(r"(\d{4})", fn)
        out.append({
            "kind": "ref_tape",
            "path": os.path.relpath(os.path.join(PANEL, fn), ROOT),
            "rating": int(m.group(1)) if m else None,
            "rating_source": "filename",
        })
    return out


def build(limit, top):
    seats, scanned = mine_replay_seats(limit, top)
    entries = []
    entries += seats
    entries += panel_tapes()
    v57 = os.path.join(ROOT, "agents", "v57_trackp.py")
    if os.path.exists(v57):
        entries.append({"kind": "agent",
                        "path": os.path.relpath(v57, ROOT),
                        "rating": None, "rating_source": None})
    entries.append({"kind": "frozen_self", "path": None,
                    "note": "trainer overwrites with the latest checkpoint",
                    "rating": None, "rating_source": "self"})

    # weight: replay/ref by rating-or-winshare proxy, agents/self a fixed share
    raw = []
    for e in entries:
        if e["kind"] == "replay_tape":
            raw.append(max(0.05, e["win_share"]))
        elif e["kind"] == "ref_tape" and e.get("rating"):
            raw.append(e["rating"] / 2500.0)
        elif e["kind"] == "agent":
            raw.append(0.6)
        elif e["kind"] == "frozen_self":
            raw.append(1.0)       # self-play the dominant partner
        else:
            raw.append(0.3)
    tot = sum(raw) or 1.0
    for e, r in zip(entries, raw):
        e["weight"] = round(r / tot, 4)

    manifest = {
        "when": "2026-09-18", "scanned_replays": scanned,
        "n_entries": len(entries),
        "counts": {k: sum(1 for e in entries if e["kind"] == k)
                   for k in ("replay_tape", "ref_tape", "agent",
                             "frozen_self")},
        "note": "rating_source=win_proxy where the replay payload carries no "
                "rating; the corpus is a top-cohort scrape so every replay "
                "seat is high-rated. Weights are normalised for a sampler.",
        "entries": entries,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def build_full(limit, top, min_rating):
    """Augmented pool = the base pool (top-100 replay/ref/agent/self) PLUS the
    destbreso ladder-rated tapes (>= min_rating), rating-weighted. Written to
    ``models/opponent_pool_full.json``; the base ``opponent_pool.json`` is left
    untouched for provenance."""
    seats, scanned = mine_replay_seats(limit, top)
    dest = mine_destbreso_seats(DESTBRESO, min_rating)
    entries = list(seats) + list(dest) + panel_tapes()
    v57 = os.path.join(ROOT, "agents", "v57_trackp.py")
    if os.path.exists(v57):
        entries.append({"kind": "agent",
                        "path": os.path.relpath(v57, ROOT),
                        "rating": None, "rating_source": None})
    entries.append({"kind": "frozen_self", "path": None,
                    "note": "trainer overwrites with the latest checkpoint",
                    "rating": None, "rating_source": "self"})

    # weight: destbreso/ref by rating (rating/2500 clipped), replay by win-share,
    # agents/self a fixed share.
    raw = []
    for e in entries:
        if e["kind"] == "replay_tape":
            raw.append(max(0.05, e["win_share"]))
        elif e["kind"] == "destbreso_tape":
            raw.append(max(0.05, min(2.0, e["rating"] / 2500.0)))
        elif e["kind"] == "ref_tape" and e.get("rating"):
            raw.append(e["rating"] / 2500.0)
        elif e["kind"] == "agent":
            raw.append(0.6)
        elif e["kind"] == "frozen_self":
            raw.append(1.0)
        else:
            raw.append(0.3)
    tot = sum(raw) or 1.0
    for e, r in zip(entries, raw):
        e["weight"] = round(r / tot, 5)

    n2500 = sum(1 for e in dest if e["rating"] >= 2500)
    n2700 = sum(1 for e in dest if e["rating"] >= 2700)
    n2900 = sum(1 for e in dest if e["rating"] >= 2900)
    manifest = {
        "when": "2026-09-18", "scanned_replays": scanned,
        "destbreso_min_rating": min_rating,
        "n_entries": len(entries),
        "counts": {k: sum(1 for e in entries if e["kind"] == k)
                   for k in ("replay_tape", "destbreso_tape", "ref_tape",
                             "agent", "frozen_self")},
        "destbreso_coverage": {"ge_2500": n2500, "ge_2700": n2700,
                               "ge_2900": n2900,
                               "min_rating": min(
                                   (e["rating"] for e in dest), default=None),
                               "max_rating": max(
                                   (e["rating"] for e in dest), default=None)},
        "note": "destbreso_tape entries carry a REAL ladder rating and are "
                "rating-weighted; materialise opponent_actions from the parquet "
                "row (episode_id+opponent_seat). Base pool (win_proxy replay "
                "tapes) preserved in opponent_pool.json.",
        "entries": entries,
    }
    with open(OUT_FULL, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--top", type=int, default=24,
                    help="replay-tape seats to keep")
    ap.add_argument("--full", action="store_true",
                    help="also write opponent_pool_full.json with destbreso tapes")
    ap.add_argument("--min-rating", type=float, default=2500.0,
                    help="destbreso teams with rating >= this join the full pool")
    args = ap.parse_args()
    m = build(args.limit, args.top)
    print(f"opponent pool -> {os.path.relpath(OUT, ROOT)}")
    print(f"  {m['n_entries']} entries: {m['counts']}")
    top5 = [e for e in m["entries"] if e["kind"] == "replay_tape"][:5]
    for e in top5:
        print(f"  replay_tape  win_share={e['win_share']:.3f}  "
              f"w={e['weight']:.3f}  {e['team'][:30]}")
    if args.full:
        mf = build_full(args.limit, args.top, args.min_rating)
        print(f"\nfull pool -> {os.path.relpath(OUT_FULL, ROOT)}")
        print(f"  {mf['n_entries']} entries: {mf['counts']}")
        print(f"  destbreso coverage: {mf['destbreso_coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
