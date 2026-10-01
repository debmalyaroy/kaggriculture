#!/usr/bin/env python
"""Review a gameplay replay logged by submission_build.py.

    python scripts/trackp/replay_view.py .local/submissions/<stamp>/replays/selfplay_seed3.jsonl
    python scripts/trackp/replay_view.py <replay.jsonl> --actions      # every non-PASS action

Prints the money trajectory (sampled), the decisive swing, final banks/winner,
and (optionally) every turn the agent actually acted -- so you can SEE how it
played, not just the summary numbers.
"""
from __future__ import annotations
import argparse, json


def _fmt_act(a):
    if not a:
        return "PASS"
    parts = []
    if a.get("farmer"):
        parts.append("F:" + " ".join(map(str, a["farmer"])))
    if a.get("hands"):
        parts.append("H:" + "|".join(" ".join(map(str, h)) for h in a["hands"]))
    if a.get("market"):
        parts.append("M:" + "|".join(" ".join(map(str, m)) for m in a["market"]))
    return "  ".join(parts) if parts else "PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("replay")
    ap.add_argument("--every", type=int, default=48, help="money-curve sample stride (steps)")
    ap.add_argument("--actions", action="store_true", help="print every non-PASS turn (seat 0)")
    ap.add_argument("--seat", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.replay) if l.strip()]
    meta = rows[0].get("meta", {}) if rows else {}
    steps = [r for r in rows if "step" in r]
    final = next((r for r in rows if r.get("final")), None)

    print(f"replay: {a.replay}")
    print(f"seed={meta.get('seed')}  turns={len(steps)}")
    print("\n money curve (seat0 / seat1):")
    prev = None
    for r in steps:
        if r["step"] % a.every == 0:
            m = r["money"]
            swing = "" if prev is None else f"  Δ={m[0]-prev:+.0f}"
            print(f"  step {r['step']:>3} d{r.get('day')}h{r.get('hour'):<2}  "
                  f"{m[0]:>9.0f} / {m[1]:<9.0f}{swing}")
            prev = m[0]

    if final:
        m = final["money"]; w = final["winner"]
        who = "DRAW" if w == -1 else f"seat{w} wins"
        print(f"\n FINAL: {m[0]:.0f} / {m[1]:.0f}   -> {who}")

    if a.actions:
        key = f"a{a.seat}"
        acted = [r for r in steps if _fmt_act(r.get(key)) != "PASS"]
        print(f"\n seat{a.seat} acted on {len(acted)}/{len(steps)} turns:")
        for r in acted:
            print(f"  d{r.get('day')}h{r.get('hour'):<2} step{r['step']:>3}: {_fmt_act(r.get(key))}")


if __name__ == "__main__":
    main()
