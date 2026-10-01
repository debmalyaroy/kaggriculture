"""Our ladder losses -> PPO training at 5x, a conversion gate, and a per-loss anatomy (queue Q61).

    python python/loss_tapes.py [--weight 5] [--threads 16]

Every game any of our submissions lost or drew on the ladder (data/ladder/<sub>/tapes, python/ladder_pull.py)
becomes a target: an RL candidate must turn these into wins against the same recorded opponent.
  data/tapes/losses/<id>.json               every loss / draw (the conversion gate's set)
  data/tapes/train/losses_x/<id>__lK.json   links, --weight copies, so PPO trains on them (it reads data/tapes/train)
  data/tapes/losses/anatomy.tsv + anatomy.json
      per loss: sub, opponent team, opponent group (day-0 COPY / PARTIAL / DIFFERENT), world, final margin, the day we
      last led and the lead then, and the item where the rival out-earned us most (exact replay: trace bin)
Note: these include held-out ladder games (odd episodes), so the held-out ladder check is no longer fully
out-of-sample for losses; the band gate (real players' tapes) stays independent.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RL, "python"))
from ladder_split import groups  # noqa: E402

BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TRACE = os.path.join(BIN, "trace" + (".exe" if os.name == "nt" else ""))
ITEMS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
LOSSES = os.path.join(RL, "data", "tapes", "losses")
LINKS = os.path.join(RL, "data", "tapes", "train", "losses_x")
TAB = chr(9)


def revenue(S, acts, seat):
    """per-item estimated sale revenue of `seat` from the trace S rows (step-major; shed drop on SELL turns x price)."""
    s = np.array([[float(v) for v in x[3:]] for x in S])
    shed, px = s[:, 2:11], s[:, 29:38]
    prev = np.vstack([np.zeros((1, 9)), shed[:-1]])
    ordered = np.zeros_like(shed)
    for t in range(min(len(acts), len(shed))):
        for m in ((acts[t][seat] or {}).get("market") or []) if isinstance(acts[t][seat], dict) else []:
            if m and m[0] == "SELL" and len(m) >= 3 and m[1] in ITEMS:
                ordered[t, ITEMS.index(m[1])] += int(m[2] or 0)
    sold = np.where(ordered > 0, np.minimum(np.clip(prev - shed, 0, None), ordered), 0)
    pxp = np.vstack([px[:1], px[:-1]])
    return (sold * pxp).sum(0), s[:, 1], s[:, 0]  # revenue by item, rival money, our money


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    for d in (LOSSES, LINKS):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    picked, seen = [], set()
    for f in sorted(glob.glob(os.path.join(RL, "data", "ladder", "*", "tapes", "*.json"))):
        t = json.load(open(f, encoding="utf-8"))
        rw = t.get("rewards") or [0, 0]
        s = t["seat"]
        if (rw[s] or 0) > (rw[1 - s] or 0):
            continue
        tid = os.path.basename(f)[:-5]
        if tid in seen:
            continue
        seen.add(tid)
        shutil.copy(f, LOSSES)
        for k in range(a.weight):
            os.symlink(os.path.join(LOSSES, tid + ".json"), os.path.join(LINKS, f"{tid}__l{k}.json"))
        picked.append((tid, f.split(os.sep)[-3], t.get("opp_team"), t.get("opp_rating"), (rw[s] or 0) - (rw[1 - s] or 0)))
    print(f"[losses] {len(picked)} losses / draws of our submissions -> {LOSSES}; {len(picked) * a.weight} training links", flush=True)
    grp = groups(sorted(glob.glob(os.path.join(LOSSES, "*.json"))), a.threads)
    tr = os.path.join(LOSSES, "_trace.tsv")
    subprocess.run([TRACE, "--tapes", LOSSES, "--out", tr, "--seats", "both"], capture_output=True, text=True)
    S, W = {}, {}
    for ln in open(tr, encoding="utf-8"):
        x = ln.rstrip("\n").split(TAB)
        if x[0] == "S":
            S.setdefault((x[1], int(x[2])), []).append(x[1:])
        elif x[0] == "W":
            W[x[1]] = x[5]
    os.remove(tr)
    rows = []
    for tid, sub, opp, opp_r, margin in picked:
        t = json.load(open(os.path.join(LOSSES, tid + ".json"), encoding="utf-8"))
        s = t["seat"]
        if (tid, s) not in S or (tid, 1 - s) not in S:
            continue
        rev_us, _, money_us = revenue(S[(tid, s)], t["actions"], s)
        rev_them, _, _ = revenue(S[(tid, 1 - s)], t["actions"], 1 - s)
        money_them = np.array([float(x[3]) for x in S[(tid, 1 - s)]])
        gap = money_us - money_them
        led = np.where(gap > 0)[0]
        last_led = int(led[-1]) if len(led) else -1
        d = rev_them - rev_us
        worst = int(np.argmax(d))
        rows.append({"id": tid, "sub": sub, "opp_team": opp, "opp_rating": opp_r, "group": grp.get(tid, "?"), "world": W.get(tid, ""),
                     "margin": margin, "last_led_day": last_led // 24 if last_led >= 0 else -1,
                     "max_lead": float(gap.max()), "max_lead_day": int(np.argmax(gap)) // 24,
                     "item_rival_outearned": ITEMS[worst], "item_gap": round(float(d[worst]), 0)})
    with open(os.path.join(LOSSES, "anatomy.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        if rows:
            fh.write(TAB.join(rows[0]) + "\n")
            for r in rows:
                fh.write(TAB.join(str(v) for v in r.values()) + "\n")
    summ = {"losses": len(rows), "by_sub": Counter(r["sub"] for r in rows), "by_group": Counter(r["group"] for r in rows),
            "by_item_rival_outearned": Counter(r["item_rival_outearned"] for r in rows),
            "by_last_led_day": Counter(("never" if r["last_led_day"] < 0 else f"d{r['last_led_day'] // 5 * 5}-{r['last_led_day'] // 5 * 5 + 4}") for r in rows),
            "close_under_3k": sum(abs(r["margin"]) < 3000 for r in rows),
            "led_at_day_24_plus": sum(r["last_led_day"] >= 24 for r in rows),
            "top_worlds": Counter(r["world"] for r in rows).most_common(8)}
    json.dump(summ, open(os.path.join(LOSSES, "anatomy.json"), "w"), indent=1, default=dict)
    print(json.dumps(summ, indent=1, default=dict))


if __name__ == "__main__":
    main()
