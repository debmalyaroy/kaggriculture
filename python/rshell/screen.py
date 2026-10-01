"""Verify every chain switch: only settings that earn their place stay ON (operator order 2026-09-27).

    python python/rshell/screen.py [--agent "<tapeplay/selfplay args>"] [--threads 24] [--offset 50] [--out data/rshell/screen]

Candidates: each stage of layers::knobs::MARKET_STAGES and ECON_STAGES switched off for the whole game
(--chain-off), and each whole-game on/off knob the profiles never set (knob-over: r85_feed_on, e410_on,
r51_input_on, v9_fert_on, term_on). Knobs the PPO profiles set (rsa/ev/dp/mp/mpx/cxd/v44y/afr/v92/tsell) are
PPO's per-day choice and are not overridden here.
Each candidate = the agent + that one switch off, paired against the agent itself on
  closed loop  64 worlds x 1 seed (bank train split, rotation --offset) x 7 lineage opponents x both seats
  open loop    training ladder half, 600 training band tapes, our real losses split A
RULE: a switch goes OFF only if turning it off is no worse (better - worse >= 0) on the closed-loop panel AND
on each open-loop set, and strictly better on at least one; otherwise it stays ON.
Then the chosen set is applied TOGETHER and re-checked on fresh seeds (rotation --offset + 1); while the
combination is worse anywhere, the switch with the weakest single evidence is put back ON.
Writes OUT/screen.json (every verdict), OUT/chain_off.txt (the final comma list), OUT/knob_over.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel as P  # noqa: E402

MARKET = ["sales_first", "r36", "r37", "v9_race", "ctrtable", "overflow", "or2", "r127", "preguard", "e335", "t62a", "wb3", "fx", "bd", "sm", "mg", "ig"]
ECON = ["v219", "v231", "v233", "r51_warehouse", "r85", "r95", "r97", "courier", "carrot", "opening", "ca", "ch", "sr", "hd2", "cs", "y", "wl", "pipe", "e402"]
KNOBS = ["r85_feed_on", "e410_on", "r51_input_on", "v9_fert_on", "term_on"]


def evaluate(agent, seeds, sets, threads, ref):
    cl, worlds = P.closed_loop(agent, seeds, threads, opponents=P.FIT)
    ol = P.open_loop(agent, sets, threads)
    bad = [k for k, w in worlds.items() if str(w).startswith("MISMATCH")]
    d = {"closed": P.score_delta(cl, ref[0])[1]} if ref else {}
    if ref:
        for k in sets:
            d[k] = P.score_delta(ol[k], ref[1][k])[1]
    return (cl, ol), d, bad


def net(d):
    return {k: v["better"] - v["worse"] for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default=P.REF)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--offset", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(P.RL, "data", "rshell", "screen"))
    ap.add_argument("--only", default=None, help="comma list of candidate names to screen (e.g. a confirmation pass)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = P.bank_seeds("train", 1, a.offset)
    sets = P.tape_sets("A")
    print(f"[screen] agent: {a.agent}\n[screen] {len(seeds)} worlds x {len(P.OPPONENTS)} opponents x 2 seats; tapes " +
          ", ".join(f"{k} {len(v)}" for k, v in sets.items()), flush=True)
    ref, _, bad = evaluate(a.agent, seeds, sets, a.threads, None)
    if bad:
        print(f"[screen] WARNING world mismatch on {len(bad)} games (reference)", flush=True)
    cands = [(n, f"--chain-off {n}") for n in MARKET + ECON]
    for k in KNOBS:
        f = os.path.join(a.out, f"ko_{k}.json")
        json.dump({k: False}, open(f, "w"))
        cands.append((k, f"--knob-over {f}"))
    if a.only:
        keep = set(a.only.split(","))
        cands = [c for c in cands if c[0] in keep]
    rep = {"agent": a.agent, "offset": a.offset, "single": {}}
    off = []
    for name, extra in cands:
        _, d, bad = evaluate(f"{a.agent} {extra}", seeds, sets, a.threads, ref)
        n = net(d)
        ok = all(v >= 0 for v in n.values()) and any(v > 0 for v in n.values())
        rep["single"][name] = {"delta": d, "net": n, "off": ok, "world_mismatch": len(bad)}
        if ok:
            off.append(name)
        print(f"[screen] {name:14s} off: " + " ".join(f"{k} {v:+d}" for k, v in n.items()) + ("  -> OFF" if ok else ""), flush=True)
    # the combination, on fresh seeds; back off the weakest while worse anywhere
    seeds2 = P.bank_seeds("train", 1, a.offset + 1)
    ref2, _, _ = evaluate(a.agent, seeds2, sets, a.threads, None)
    strength = lambda n: sum(rep["single"][n]["net"].values())  # noqa: E731
    while off:
        mk = []
        ko = {k: False for k in off if k in KNOBS}
        stages = [k for k in off if k not in KNOBS]
        if stages:
            mk.append("--chain-off " + ",".join(stages))
        if ko:
            f = os.path.join(a.out, "knob_over.json")
            json.dump(ko, open(f, "w"))
            mk.append(f"--knob-over {f}")
        _, d, _ = evaluate(f"{a.agent} {' '.join(mk)}", seeds2, sets, a.threads, ref2)
        n = net(d)
        print(f"[screen] combination {off}: " + " ".join(f"{k} {v:+d}" for k, v in n.items()), flush=True)
        rep.setdefault("combination", []).append({"off": list(off), "net": n, "delta": d})
        if all(v >= 0 for v in n.values()):
            break
        weakest = min(off, key=strength)
        off.remove(weakest)
    rep["final_off"] = off
    stages = [k for k in off if k not in KNOBS]
    open(os.path.join(a.out, "chain_off.txt"), "w").write(",".join(stages))
    json.dump({k: False for k in off if k in KNOBS}, open(os.path.join(a.out, "knob_over.json"), "w"))
    json.dump(rep, open(os.path.join(a.out, "screen.json"), "w"), indent=1)
    print(f"[screen] final OFF: {off or 'none (every switch stays ON)'} -> {a.out}/screen.json", flush=True)


if __name__ == "__main__":
    main()
