"""War Room feed for the live pair: status, rating, W/L and the last 20 results per submission.

    python -m kaggriculture.pipeline.live_board [--no-download] [--cap 200]
    scripts/live_board_loop.sh                             # the same, every hour, forever

Submission status comes from `kaggle competitions submissions`; games from Kaggle's EpisodeService
via `live_status.episodes` (small JSON). Then every finished game of the live pair not yet on disk
is downloaded, unparsed, to data/live_replays/<submission>/<episode>.json (counted in the shared
replay quota ledger, see sameday._grab_direct).

Outputs (the board's shared database documents, ready to write as-is):
    live.json          board/live   -- one entry per active submission (rendered by the Live panel)
    status_patch.json  board/status -- `rating` + `rating_note` only, merged with an update so the
                                       hand-written headline is never overwritten

Prints one line (`LIVE_BOARD ok ...` or `LIVE_BOARD error ...`) so an hourly monitor can react.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys

from kaggriculture.paths import ROOT
from kaggriculture.pipeline.live_status import BANDS, episodes, summarise

OUT = os.path.join(ROOT, ".local", "live_board")
REPLAYS = os.path.join(ROOT, "data", "live_replays")   # data/live_replays/<submission>/<episode>.json
N_ACTIVE = 2   # only the latest 2 submissions stay active
N_LAST = 20


def submissions():
    """Latest N_ACTIVE rows of `kaggle competitions submissions` as dicts."""
    p = subprocess.run(["kaggle", "competitions", "submissions", "-c", "kaggriculture", "--csv"],
                       capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
    text = p.stdout or ""
    start = text.find("ref,")
    rows = list(csv.DictReader(io.StringIO(text[start:]))) if start >= 0 else []
    return rows[:N_ACTIVE]


def label(desc):
    """The agent name at the front of a description: 'v62.1_bandit: ...' -> 'v62.1'."""
    head = (desc or "").split(":", 1)[0].strip()
    return head.split("_", 1)[0] or head[:20]


def entry(sub):
    ref = sub["ref"]
    status = (sub.get("status") or "").replace("SubmissionStatus.", "")
    score = sub.get("publicScore") or ""
    e = {"ref": ref, "name": label(sub.get("description")), "desc": (sub.get("description") or "")[:160],
         "submitted": (sub.get("date") or "")[:16].replace(" ", "T") + "Z", "status": status,
         "score": float(score) if score else None}
    try:
        rows = episodes(ref)
    except Exception as exc:  # noqa: BLE001 -- 429 etc.: skip, the next hourly run retries
        e["error"] = str(exc)[:200]
        return e
    e["_episodes"] = [r["episode"] for r in rows if r.get("episode")]
    s = summarise(rows)
    e.update(games=s.get("games", 0), w=s.get("w", 0), l=s.get("l", 0), d=s.get("d", 0),
             rating=s.get("rating"), peak=s.get("peak"), upsets=s.get("upsets", 0))
    if rows:
        e["bands"] = {k: {"n": b["n"], "w": b["w"], "l": b["l"]} for k, b in s["bands"].items()}
        l20 = s["last20"]
        e["last20_wld"] = [l20["w"], l20["l"], l20["d"]]
    e["last"] = [{"r": "W" if r["my"] > r["op"] else "L" if r["my"] < r["op"] else "D",
                  "my": round(r["my"]), "op": round(r["op"]),
                  "opr": round(float(r["op_rating_pre"])) if r.get("op_rating_pre") is not None else None,
                  "rat": round(float(r["my_rating"])) if r.get("my_rating") is not None else None,
                  "t": str(r.get("end") or "")[:19]}
                 for r in rows[:N_LAST]]
    return e


def download(items, cap):
    """Pull every finished game of the live pair that is not on disk yet. No parsing.

    Uses sameday._grab_direct, so each download is counted in the shared replay quota ledger and
    obeys its holds (rolling 24h cap, pre-release quiet window). A hold stops this run; the next
    hour picks up where it left off because files already on disk are skipped."""
    from kaggriculture.data.sameday import _grab_direct
    got, held, failed = 0, "", 0
    for e in items:
        dest = os.path.join(REPLAYS, e["ref"])
        os.makedirs(dest, exist_ok=True)
        for eid in e.get("_episodes", []):
            if got >= cap or held:
                break
            if os.path.exists(os.path.join(dest, f"{eid}.json")):
                continue
            try:
                _grab_direct(eid, dest)
                got += 1
            except RuntimeError as exc:
                if str(exc).startswith("QUOTA-HOLD"):
                    held = str(exc)[:80]
                else:
                    failed += 1
            except Exception:  # noqa: BLE001 -- retried next hour
                failed += 1
        e["replays_on_disk"] = sum(1 for f in os.listdir(dest) if f.endswith(".json"))
    return got, failed, held


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-download", action="store_true", help="board files only, skip replay downloads")
    ap.add_argument("--cap", type=int, default=200, help="max replay downloads per run")
    a = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        subs = submissions()
        if not subs:
            raise RuntimeError("no submissions returned by the kaggle CLI")
        items = [entry(s) for s in subs]
    except Exception as exc:  # noqa: BLE001
        print(f"LIVE_BOARD error {now} {exc}", flush=True)
        return 1
    dl = "downloads skipped"
    if not a.no_download:
        got, failed, held = download(items, a.cap)
        dl = f"downloaded {got}, failed {failed}" + (f", held: {held}" if held else "")
    try:   # a skipped listing (429) keeps the last good numbers, marked stale, instead of blanking them
        prev = {p["ref"]: p for p in json.load(open(os.path.join(OUT, "live.json"), encoding="utf-8"))["subs"]}
    except (OSError, ValueError, KeyError):
        prev = {}
    for e in items:
        e.pop("_episodes", None)
        p = prev.get(e["ref"])
        if e.get("error") and p and p.get("games"):
            for k in ("games", "w", "l", "d", "rating", "peak", "upsets", "bands", "last20_wld", "last",
                      "replays_on_disk"):
                if k in p:
                    e[k] = p[k]
            e["stale_since"] = p.get("stale_since") or p.get("fresh_at")
        elif not e.get("error"):
            e["fresh_at"] = now
    os.makedirs(OUT, exist_ok=True)
    json.dump({"updated": now, "bands": [f"{lo}-{hi}" for lo, hi, _ in BANDS], "subs": items},
              open(os.path.join(OUT, "live.json"), "w", encoding="utf-8"), indent=1)

    def rt(e):
        r = e.get("rating") if e.get("rating") is not None else e.get("score")
        return f"{r:.0f}" if isinstance(r, (int, float)) else "--"

    def rec(e):
        if not e.get("games"):
            return "no games yet"
        return f"{e['w']}-{e['l']}" + (f"-{e['d']}" if e.get("d") else "") + f" of {e['games']}"

    patch = {"rating": " / ".join(rt(e) for e in items),
             "rating_note": " · ".join(f"{e['name']} ({e['ref']}) {rt(e)} {e['status'].lower()}, {rec(e)}"
                                       + (f", upsets {e['upsets']}" if e.get("upsets") else "")
                                       for e in items) + f" · as of {now[11:16]} UTC (auto, hourly)"}
    json.dump(patch, open(os.path.join(OUT, "status_patch.json"), "w", encoding="utf-8"), indent=1)
    errs = [e["ref"] for e in items if e.get("error")]
    print(f"LIVE_BOARD {'partial' if errs else 'ok'} {now} " + "; ".join(
        f"{e['name']} {rt(e)} {rec(e)}" for e in items) + f"; {dl}" + (f" errors={errs}" if errs else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
