"""Collapse alarm: flag a v23.1-style failure hours before the rating shows.

Runs after each hourly ourgames delta. Primary signal is outcome-based and
model-free (robust): the trailing window of our ACTIVE submissions' games.
v23.1's arm-commit collapse halved banks immediately -- a trailing-10 bank
mean under the floor, or a win rate under the floor, would have flagged it
within ~2 hours of play. Secondary: the lab rating scorer (v0, coarse) if
its model file exists.

Writes one status line per submission to stdout (-> sameday_scrape.log) and
appends ALARM lines to data/logs/collapse_alarm.log so a glance (or a grep)
answers "is the pair healthy".

    python src/collapse_alarm.py
"""
from kaggriculture.paths import ROOT
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW = 10
# The trigger is EXPECTED SCORE (win=1, draw=0.5, loss=0) -- the only thing the
# ladder pays. The old absolute bank floor (60,000) was a false-alarm
# generator: bank is set largely by the OPPONENT's market flooding, measured at
# corr(late-game price, our bank) = +0.88 over 22 replays, so a perfectly
# healthy agent drawn against heavy sellers banks 54k and would have tripped
# it, while a broken agent in a rich market could clear it. Bank is now logged
# as context only.
SCORE_FLOOR = 0.35        # matchmaking keeps healthy agents near ~50%+
BANK_CONTEXT_ONLY = True
# The upload Validation Episode is SELF-PLAY, so the opponent-flooding caveat
# above does not apply to it: a mirror banking under this floor means the
# uploaded artifact is broken on Kaggle's runtime regardless of what the same
# bytes did locally (v33 shipped at 7,777 vs 96k local, 2026-08-21).
VALIDATION_BANK_FLOOR = 30_000
OUR_TEAM = "Debmalya"
ALARM_LOG = os.path.join(ROOT, "data", "logs", "collapse_alarm.log")


def active_refs():
    import subprocess
    import re
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(["kaggle", "competitions", "submissions",
                        "kaggriculture"], capture_output=True, text=True,
                       timeout=300, env=env)
    refs = []
    for line in (p.stdout or "").splitlines():
        m = re.match(r"\s*(\d{6,})\s", line)
        if m and "COMPLETE" in line:
            refs.append(m.group(1))
        if len(refs) == 2:
            break
    return refs


def check_validation(ref, idx):
    """Alarm if the upload validation mirror banked below the floor."""
    mirrors = [g for g in idx["games"].values()
               if str(g.get("submission")) == str(ref)
               and g.get("opponent") == OUR_TEAM]
    if not mirrors:
        return None
    bank = min(float(g.get("bank") or 0) for g in mirrors)
    if bank >= VALIDATION_BANK_FLOOR:
        return f"sub {ref}: validation mirror banked {bank:,.0f} -- healthy"
    alarm = (f"ALARM {dt.datetime.now():%Y-%m-%d %H:%M} sub {ref}: VALIDATION "
             f"mirror banked {bank:,.0f} (< {VALIDATION_BANK_FLOOR:,}) -- the "
             f"uploaded artifact is broken on Kaggle's runtime; replace it")
    os.makedirs(os.path.dirname(ALARM_LOG), exist_ok=True)
    with open(ALARM_LOG, "a", encoding="utf-8") as fh:
        fh.write(alarm + "\n")
    return alarm


def check(ref):
    path = os.path.join(ROOT, "data", "ourgames", "index.json")
    idx = json.load(open(path, encoding="utf-8"))
    vline = check_validation(ref, idx)
    rows = [g for g in idx["games"].values()
            if str(g.get("submission")) == str(ref)
            and g.get("opponent") != OUR_TEAM]
    rows.sort(key=lambda g: str(g.get("episode")))
    tail = rows[-WINDOW:]
    prefix = (vline + "\n") if vline else ""
    if len(tail) < 5:
        return prefix + f"sub {ref}: only {len(tail)} recent games -- too few to judge"
    import kaggriculture.measure.win_metric as WM
    bank = sum(float(g.get("bank") or 0) for g in tail) / len(tail)
    # Draws are half a point, not a loss -- the old `g.get("won")` count
    # silently scored every draw as 0.
    score = WM.expected_score(
        WM.score(g.get("bank") or 0, g.get("opp_bank") or 0) for g in tail
        if g.get("bank") is not None and g.get("opp_bank") is not None)
    if score != score:                                  # no usable banks
        score = sum(1 for g in tail if g.get("won")) / len(tail)
    line = (f"sub {ref}: trailing-{len(tail)} expected score {score:.2f} "
            f"(bank {bank:,.0f}, context only)")
    if score < SCORE_FLOOR:
        alarm = (f"ALARM {dt.datetime.now():%Y-%m-%d %H:%M} {line} "
                 f"(floor: score {SCORE_FLOOR:.2f})")
        os.makedirs(os.path.dirname(ALARM_LOG), exist_ok=True)
        with open(ALARM_LOG, "a", encoding="utf-8") as fh:
            fh.write(alarm + "\n")
        return prefix + alarm
    return prefix + line + " -- healthy"


def main():
    for ref in active_refs():
        try:
            print(check(ref), flush=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"sub {ref}: alarm check failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
