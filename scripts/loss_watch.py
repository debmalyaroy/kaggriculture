"""HOURLY loss watch: pull fresh loss tapes, autopsy, refresh upset cells.

Operator order (2026-09-05): the hourly loss tapes must be downloaded and
analyzed continuously; fixes go into the BANDIT lane, and common/logical
fixes lift into trackp. This runs every hour from the scheduler so the
cadence never depends on anyone remembering it.

Sequence (light, serialized, ledger-guarded):
  1. live_status -- both submissions, bands, upsets;
  2. loss_autopsy --max-pulls 6 -- fresh loss replays within the ledger;
  3. refresh upset cells from the live records (feeds the gauntlet);
  4. one-line verdict appended to data/logs/loss_watch.log.

    python scripts/loss_watch.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import datetime as dt
import json
import os
import subprocess
import sys

LOG = os.path.join(ROOT, "data", "logs", "loss_watch.log")


def sh(args, timeout=1200):
    return subprocess.run([sys.executable] + args, cwd=ROOT, timeout=timeout,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="minutes")
    live = sh([os.path.join(ROOT, "src", "kaggriculture", "pipeline", "live_status.py")])
    auto = sh([os.path.join(ROOT, "src", "kaggriculture", "measure", "loss_autopsy.py"),
               "--max-pulls", "6"], timeout=1800)

    # refresh upset cells for the gauntlet
    ups_line = ""
    try:
        import kaggriculture.pipeline.live_status as LS
        cells = []
        for sid, _desc in LS.active_pair():
            for r in LS.episodes(sid):
                if (r["my"] is not None and r["op"] is not None
                        and r["my"] < r["op"]
                        and r.get("my_rating") is not None
                        and r.get("op_rating_pre") is not None
                        and float(r["op_rating_pre"]) < float(r["my_rating"])):
                    cells.append({"episode": r["episode"], "sub": sid,
                                  "my_seat": r["my_seat"],
                                  "my_rating": float(r["my_rating"]),
                                  "op_rating": float(r["op_rating_pre"]),
                                  "my": r["my"], "op": r["op"]})
        json.dump(cells, open(os.path.join(ROOT, ".local", "hardband",
                                           "upset_cells.json"), "w",
                              encoding="utf-8"), indent=1)
        ups_line = f"upset cells {len(cells)}"
    except Exception as exc:                                   # noqa: BLE001
        ups_line = f"upset refresh failed: {type(exc).__name__}"

    # ---- the FULL operator report (operator order 2026-09-05: hourly
    # status must ALWAYS carry loss-tape analysis + the bandit/trackp fix
    # routing -- never just ratings)
    reg = {}
    regp = os.path.join(ROOT, "models", "trackp", "fix_registry.json")
    if os.path.exists(regp):
        reg = json.load(open(regp, encoding="utf-8"))
    mech = []
    try:
        rep = json.load(open(os.path.join(ROOT, ".local",
                                          "loss_autopsy.json"),
                             encoding="utf-8"))
        an = [r for r in rep.get("reports", []) if "skipped" not in r]
        for r in an[:8]:
            m = r.get("margin")
            own = r.get("our_bank")
            kind = ("narrow" if m is not None and m > -3000 else
                    "collapse" if own is not None and own < 75000 else
                    "outbanked")
            wtag = r.get("world") or "?"
            ttag = (" TAIL:" + r["tail_donor"]) if r.get("tail_donor") else ""
            mech.append(f"    ep{r.get('episode')} {kind:<9} "
                        f"margin {m:+,.0f} decisive d{r.get('decisive_day')}"
                        f"  [{wtag}{ttag}]")
    except Exception as exc:                                   # noqa: BLE001
        mech.append(f"    (autopsy aggregation failed: {type(exc).__name__})")

    # BT-implied rating from the games actually played (one-sided MLE vs
    # known opponent ratings; the displayed rating lags the K-schedule)
    bt_lines = []
    try:
        import math
        import kaggriculture.pipeline.live_status as LS2
        def _mle(games):
            if not games:
                return None
            s = 10 ** (sum(r for _, r in games) / len(games) / 400)
            W = sum(w for w, _ in games)
            si = [10 ** (r / 400) for _, r in games]
            for _ in range(500):
                d = sum(1.0 / (s + x) for x in si)
                s2 = max(1e-9, W / d) if d else s
                if abs(math.log(s2 / s)) < 1e-10:
                    s = s2
                    break
                s = s2
            return 400 * math.log10(s)
        for sid, desc in LS2.active_pair():
            rows = [r for r in LS2.episodes(sid)
                    if r["my"] is not None and r["op"] is not None
                    and r.get("op_rating_pre") is not None]
            rows.sort(key=lambda r: r.get("end") or 0)
            g = [(1.0 if r["my"] > r["op"] else
                  (0.5 if r["my"] == r["op"] else 0.0),
                  float(r["op_rating_pre"])) for r in rows]
            if g:
                opp30 = sum(r for _, r in g[-30:]) / len(g[-30:])
                bt_lines.append(
                    f"  {sid} BT-implied: all {_mle(g):.0f}  "
                    f"last30 {_mle(g[-30:]):.0f}  last10 {_mle(g[-10:]):.0f}"
                    f"  (opp mean last30 {opp30:.0f}, n={len(g)})")
    except Exception as exc:                                   # noqa: BLE001
        bt_lines.append(f"  (BT-implied failed: {type(exc).__name__})")
    lines = [f"===== HOURLY OPERATOR REPORT {stamp} =====", "",
             "-- LIVE --", live.stdout.strip(), "",
             "-- BT-IMPLIED (from actual games) --"] + bt_lines + ["",
             f"-- LOSS TAPES ({ups_line}; autopsy exit "
             f"{auto.returncode}) --"] + mech + ["", "-- FIX ROUTING --"]
    for f in reg.get("findings", []):
        lines.append(f"  * {f['finding']}")
        lines.append(f"      bandit: {f['bandit']}")
        lines.append(f"      trackp: {f['trackp']}")
    block = "\n".join(lines) + "\n"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("\n" + block)
        tail = "\n".join(auto.stdout.splitlines()[-25:])
        fh.write(tail + "\n")
    print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
