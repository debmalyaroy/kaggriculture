"""Rust vs Python parity of the endgame controller: recompute every logged decision (KRL_ENDG_LOG from the Rust
agent) with the exported JSON nets in numpy and compare utilities and picks.

    python python/endg/parity.py --endg weights/endg/v1/endg.json --log data/endg/parity.jsonl
"""
import argparse
import json

import numpy as np


def run(m, x):
    sd = np.asarray(m["std"])
    h = np.clip((np.asarray(x, np.float64) - np.asarray(m["mean"])) / np.maximum(sd, 1e-6), -8.0, 8.0)
    h[sd <= 2e-3] = 0.0   # endg.rs Mlp::run guard
    for i, l in enumerate(m["layers"]):
        h = np.asarray(l["w"]) @ h + np.asarray(l["b"])
        if i + 1 < len(m["layers"]):
            h = np.maximum(h, 0)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endg", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    c = json.load(open(a.endg))
    k = len(c["proposals"])
    sg = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    n = bad = 0
    du = 0.0
    for line in open(a.log):
        d = json.loads(line)
        o = run(c["objective"], d["x"])
        u = sg(o[:k]) - sg(o[k:2 * k])
        u[0] = 0.0
        pr = run(c["proposal"], d["x"])
        cands = list(np.argsort(-pr)[:c["top_m"]]) if c["top_m"] < k else list(range(k))
        if 0 not in cands:
            cands.append(0)
        best = max(cands, key=lambda j: (u[j], -j))
        pick = best if u[best] > c["gate"] else 0
        n += 1
        bad += pick != d["pick"]
        if d["u"]:
            du = max(du, float(np.abs(np.asarray(d["u"]) - u).max()))
    print(f"[endg-parity] {n} decisions, {bad} pick mismatches, max |u_rust - u_py| = {du:.2e}")


if __name__ == "__main__":
    main()
