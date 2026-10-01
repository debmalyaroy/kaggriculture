"""Opponent sell-timing dataset: WHEN will the opponent dump, per item class.

Operator direction (2026-09-06): learn to predict the situations where the
opponent is about to SELL (and later: buy), so our agent can act FIRST.
The two measured mechanisms this targets:
  * settlement race -- in 20 live lost worlds the opponent's terminal dump
    started first 15 times, ours never (race-first flush is the static fix;
    this predictor is the precise one);
  * scarcity-peak selling -- mid-game dumps into price peaks decide games
    (fix registry #1, contested_dumps price impact).

Labels are GROUND TRUTH: every staged replay has both action tracks, so
"opponent sells >= K units of item X within the next H steps" is read
directly from their recorded actions. No inference, no oracle keying.

Rows (one per step 96..719 per episode, OUR seat's point of view):
  features: the observable state -- day frac, step-in-day, our bank, their
    bank, price vector, price ratio to base, their visible board (animals,
    tiles by crop age), their recent sell volume (6/24-step windows),
    market inventory drain, regime multipliers;
  labels: for each item class (WOOL MILK EGG WHEAT MELON CARROT TOMATO
    FERTILIZER), 1 if the OPPONENT sells >= 5 units of it within the next
    12 steps, else 0; plus "any burst >= 10 units within 12 steps".

Split BY EPISODE (never by row). Output: data/opp_sell/{train,val}.jsonl
plus meta.json (feature names, base rates).

    python src/trackp/harness/opp_sell_dataset.py --limit 3000
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

ITEMS = ("WOOL", "MILK", "EGG", "WHEAT", "MELON", "CARROT", "TOMATO",
         "FERTILIZER")
HORIZON = 12        # steps ahead for the label window
BURST = 5           # units that count as a dump
FROM_STEP = 96      # day 4 on; the opening is scripted everywhere
OUT = os.path.join(ROOT, "data", "opp_sell")


def sell_units(act, item=None):
    n = 0
    for o in (act or {}).get("market") or []:
        if isinstance(o, list) and len(o) >= 3 and str(o[0]).upper() == "SELL":
            if item is None or str(o[1]).upper() == item:
                try:
                    n += int(o[2])
                except Exception:                              # noqa: BLE001
                    n += 1
    return n


def episode_rows(rep):
    steps = rep.get("steps") or []
    if len(steps) < 720:
        return
    # pre-compute opponent sell units per (step, item) for both seats
    per = [[{it: sell_units(steps[t][s].get("action"), it) for it in ITEMS}
            for t in range(720)] for s in range(2)]
    for me in (0, 1):
        opp = 1 - me
        for t in range(FROM_STEP, 720 - HORIZON):
            ob = steps[t][me].get("observation") or {}
            farms = ob.get("farms") or [{}, {}]
            market = ob.get("market") or {}
            prices = market.get("prices") or {}
            mf, of = farms[me] or {}, farms[opp] or {}
            # RUNTIME-OBSERVABLE ONLY (2026-09-06 audit): the agent
            # never sees opponent actions. Opponent selling shows up in
            # market.inventory (sells add units) and prices; our own sells
            # are known, so opp volume ~= d_inventory - our_sells.
            inv = market.get("inventory") or {}
            f = [t / 720.0, (t % 24) / 24.0,
                 float(mf.get("money") or 0) / 1e5,
                 float(of.get("money") or 0) / 1e5]
            for it in ITEMS:
                f.append(float(prices.get(it) or 0) / 100.0)
            for w in (6, 24):
                lo = max(0, t - w)
                obw = (steps[lo][me].get("observation") or {})
                inv0 = (obw.get("market") or {}).get("inventory") or {}
                pr0 = (obw.get("market") or {}).get("prices") or {}
                my_sold = sum(sum(per[me][u].values())
                              for u in range(lo, t))
                d_inv = sum(float(inv.get(it) or 0)
                            - float(inv0.get(it) or 0) for it in ITEMS)
                f.append((d_inv - my_sold) / (10.0 * w))
                for it in ("WOOL", "MILK", "WHEAT", "MELON"):
                    f.append((float(prices.get(it) or 0)
                              - float(pr0.get(it) or 0)) / 50.0)
            for it in ITEMS:
                lo6 = max(0, t - 6)
                ob6 = (steps[lo6][me].get("observation") or {})
                inv6 = (ob6.get("market") or {}).get("inventory") or {}
                d_it = (float(inv.get(it) or 0) - float(inv6.get(it) or 0)
                        - per[me][lo6].get(it, 0))
                f.append(d_it / 60.0)
            lab = {}
            for it in ITEMS:
                fut = sum(per[opp][u][it] for u in range(t, t + HORIZON))
                lab[it] = 1 if fut >= BURST else 0
            fut_all = sum(sum(per[opp][u].values())
                          for u in range(t, t + HORIZON))
            lab["ANY"] = 1 if fut_all >= 2 * BURST else 0
            yield {"f": [round(x, 5) for x in f], "y": lab, "t": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=3000,
                    help="max episodes to ingest")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=4,
                    help="keep every Nth step row (class balance + size)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    src_dirs = [os.path.join(ROOT, ".local", "lossreplays"),
                os.path.join(ROOT, ".local", "hardband", "replays"),
                os.path.join(ROOT, "data", "episodes")]
    eps = []
    for d in src_dirs:
        if os.path.isdir(d):
            eps += [os.path.join(d, x) for x in os.listdir(d)
                    if x.endswith(".json")]
    random.Random(7).shuffle(eps)
    eps = eps[:a.limit]
    n_val = max(1, int(len(eps) * a.val_frac))
    val_set = set(eps[:n_val])
    counts = {"train": 0, "val": 0}
    pos = {k: 0 for k in (*ITEMS, "ANY")}
    fh = {k: open(os.path.join(OUT, f"{k}.jsonl"), "w", encoding="utf-8")
          for k in counts}
    for i, p in enumerate(eps):
        try:
            rep = json.load(open(p, encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        split = "val" if p in val_set else "train"
        for j, row in enumerate(episode_rows(rep)):
            if j % a.stride:
                continue
            fh[split].write(json.dumps(row, separators=(",", ":")) + "\n")
            counts[split] += 1
            for k, v in row["y"].items():
                pos[k] += v if split == "train" else 0
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(eps)} episodes; rows {counts}", flush=True)
    for h in fh.values():
        h.close()
    base = {k: (pos[k] / counts["train"] if counts["train"] else 0.0)
            for k in pos}
    meta = {"items": list(ITEMS), "horizon": HORIZON, "burst": BURST,
            "episodes": len(eps), "rows": counts, "base_rates": base}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
