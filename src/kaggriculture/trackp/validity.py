"""P4.1 -- open-loop vs closed-loop ranking validity.

Question: when the Rust engine ranks opponents/candidates OPEN-LOOP
(tape vs tape, kagg batch), does that ranking agree with CLOSED-LOOP
outcomes (the adaptive planner playing them on serve)? Measured ONCE and
recorded before any Rust fitness is trusted for closed-loop decisions
(the same discipline as the crown funnel's recall gate).

Method: for K anchors,
  open(a)   = mean of a's final bank over pairings vs R reference tapes
              (kagg batch)
  closed(a) = mean gap the CURRENT planner achieves vs a on kagg serve
Rank correlation (Spearman) between -open and closed (a stronger anchor
should be harder for the planner) + top-overlap. Writes
models/trackp/validity_report.json.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

try:
    from . import common, league, rollouts
except ImportError:
    import sys
    from kaggriculture.trackp import common, league, rollouts


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if len(a) < 3:
        return 0.0
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c)


def run(k: int = 20, refs: int = 4, jobs: int = 8) -> dict:
    anchors = league.load_anchors()
    if len(anchors) < k + refs:
        k = max(4, len(anchors) - refs)
    subject = anchors[:k]
    reference = anchors[k:k + refs]

    # open-loop: batch of subject vs reference tapes
    jobs_path = os.path.join(common.DATA, "tapes", "validity_jobs.tsv")
    os.makedirs(os.path.dirname(jobs_path), exist_ok=True)
    lines = []
    meta = []
    for si, s in enumerate(subject):
        for r in reference:
            lines.append(f"{s['seed']}\t{s['tape']}\t{r['tape']}")
            meta.append(si)
    with open(jobs_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    out = subprocess.run([common.KAGG, "batch", jobs_path],
                         capture_output=True, text=True, timeout=600)
    open_score = np.zeros(k)
    open_n = np.zeros(k)
    for line in out.stdout.strip().splitlines():
        p = line.split("\t")
        i = int(p[0])
        if p[1] == "ERR":
            continue
        open_score[meta[i]] += float(p[2])  # subject seat 0 bank
        open_n[meta[i]] += 1
    open_n[open_n == 0] = 1
    open_score /= open_n

    # closed-loop: planner vs each subject on serve
    runner = rollouts.Runner(jobs)
    try:
        cjobs = [{"seed": s["seed"], "me": {},
                  "opp": {"kind": "tape", "tape": s["tape"]}}
                 for s in subject]
        res = runner.run(cjobs)
    finally:
        runner.close()
    closed = np.zeros(k)
    for i, r in enumerate(res):
        if "banks" in r:
            closed[i] = r["banks"][0] - r["banks"][1]  # planner gap vs anchor

    # a stronger anchor (higher open bank) should mean a lower planner gap
    rho = spearman(-open_score, closed)
    ord_open = np.argsort(-open_score)[: max(3, k // 4)]
    ord_hard = np.argsort(closed)[: max(3, k // 4)]
    overlap = len(set(ord_open.tolist()) & set(ord_hard.tolist())) / len(
        ord_open)
    rep = {"k": k, "refs": refs, "spearman": round(rho, 3),
           "top_overlap": round(overlap, 3),
           "interpretation": "open-loop strength predicts closed-loop "
                             "difficulty" if rho > 0.4 else
                             "WEAK -- do not trust Rust fitness for "
                             "closed-loop decisions without official "
                             "confirmation"}
    with open(os.path.join(common.MODELS, "validity_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run(k=a.k, jobs=a.jobs)
