"""Opponent LINEAGES (tape families) and their sale decision tables, for the agent to fingerprint and out-play
(operator 28 Sep: "cluster based on lineage; some players who are using RL cannot be clustered").

    python python/opp_lineage.py [--sig data/opp/sig_v2.npz] [--min-games 15] [--max 150] [--out configs/opp/lineages_v1.json]

Lineage key = hash of the seat's exact action stream over days 0-5 (python/opp_corpus_sig.py pfx6; tape bots replay
it move for move). Per lineage with >= --min-games games:
  p_step[item][720]  P(a sale of item at that step)   -- its decision table: WHAT it sells at WHICH step
  lot[item]          median order size
  consistency        mean |2p - 1| over the steps where the lineage ever sells (1 = fixed schedule, 0 = coin flip)
Lineages below --min-consistency are dropped (adaptive / RL players repeat an opening but not a schedule).
A BACKGROUND entry (the pooled table of everything else) is appended last: when the rival's observed sales fit
the background better than any lineage, the agent treats it as unpredictable and does not pre-empt.
Same JSON shape as python/opp_cluster2.py ("clusters": [...]) so crates/agent/src/preempt.rs loads either.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default=os.path.join(RL, "data", "opp", "sig_v2.npz"))
    ap.add_argument("--key", default="pfx6")
    ap.add_argument("--min-games", type=int, default=15)
    ap.add_argument("--min-consistency", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(RL, "configs", "opp", "lineages_v1.json"))
    a = ap.parse_args()
    z = np.load(a.sig, allow_pickle=False)
    items = [str(x) for x in z["items"]]
    NI = len(items)
    n = len(z["eid"])
    key = z[a.key]
    ev_packed = z["ev"]
    cnt = Counter(key.tolist())
    cand = [k for k, c in cnt.most_common() if c >= a.min_games]
    print(f"[lineage] {n} player-games, {len(cnt)} distinct {a.key} keys, {len(cand)} with >= {a.min_games} games", flush=True)
    win = (z["bank"] > z["obank"])
    out, used = [], np.zeros(n, bool)
    for k in cand:
        idx = np.where(key == k)[0]
        ev = np.unpackbits(ev_packed[idx], axis=1)[:, : NI * 720].reshape(len(idx), NI, 720).astype(np.float32)
        p = ev.mean(0)
        active = p > 0.02
        cons = float(np.abs(2 * p[active] - 1).mean()) if active.any() else 0.0
        if cons < a.min_consistency:
            continue
        lots = z["lot_med"][idx]
        out.append({"key": int(k), "n": int(len(idx)), "consistency": round(cons, 3), "win_rate": round(float(win[idx].mean()), 3),
                    "mean_rating": round(float(z["rating"][idx][z["rating"][idx] > 0].mean()), 0) if (z["rating"][idx] > 0).any() else 0.0,
                    "teams": Counter(z["team"][idx].tolist()).most_common(3),
                    "p_step": {it: [round(float(x), 2) for x in p[j]] for j, it in enumerate(items)},
                    "lot": {it: round(float(np.median(lots[:, j][lots[:, j] > 0])) if (lots[:, j] > 0).any() else 0.0, 1) for j, it in enumerate(items)}})
        used[idx] = True
        if len(out) >= a.max:
            break
    # background: everyone not in a kept lineage (adaptive, RL, rare lineages)
    rest = np.where(~used)[0]
    bg = np.zeros((NI, 720), np.float64)
    lots_bg = z["lot_med"][rest]
    for s0 in range(0, len(rest), 20000):
        ch = rest[s0:s0 + 20000]
        bg += np.unpackbits(ev_packed[ch], axis=1)[:, : NI * 720].reshape(len(ch), NI, 720).sum(0)
    bg /= max(1, len(rest))
    out.append({"key": 0, "background": True, "n": int(len(rest)), "consistency": 0.0,
                "p_step": {it: [round(float(x), 2) for x in bg[j]] for j, it in enumerate(items)},
                "lot": {it: round(float(np.median(lots_bg[:, j][lots_bg[:, j] > 0])) if (lots_bg[:, j] > 0).any() else 0.0, 1) for j, it in enumerate(items)}})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"items": items, "key": a.key, "player_games": int(n), "covered": int(used.sum()), "clusters": out}, open(a.out, "w"))
    print(f"[lineage] kept {len(out) - 1} lineages covering {used.sum()} of {n} player-games ({used.mean():.0%}); background {len(rest)} -> {a.out}", flush=True)
    for c in out[:25]:
        if c.get("background"):
            continue
        peaks = {it: [t for t, v in enumerate(c["p_step"][it]) if v >= 0.8][:6] for it in items}
        peaks = {k: v for k, v in peaks.items() if v}
        print(f"  lineage {c['key'] & 0xffffff:06x}: {c['n']:5d} games, consistency {c['consistency']:.2f}, win {c['win_rate']:.2f}, rating {c['mean_rating']:.0f} | sure sales (p>=0.8) first steps {peaks}", flush=True)


if __name__ == "__main__":
    main()
