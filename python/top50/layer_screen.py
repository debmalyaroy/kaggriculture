"""Tune v63's own end-game / race layers on our real losses (queue Q63).

    python python/top50/layer_screen.py [--threads 24] [--top 4]

225 of our 292 ladder losses came after we last led in days 25-29, 240 by under $3,000, 205 against copies:
the end-game sale race. v63.1_rl leaves room there: the terminal planner starts at step 712 (it can start at 695,
the last day) with 512 simulations, the terminal sell layer starts at 672, market pressure is off, and the worst
turn is ~86 ms of a 1 s budget. Each variant = profile 35 with a few end-game knobs changed (appended to the v64rl
table), played as v63.1_rl plays (--pa K --group K,K,Kc where Kc = the variant + lead32 for COPY opponents).
  SELECTION: our real losses (data/tapes/losses: how many turn into wins) + the training half of our ladder games
             + 600 training-band tapes, paired vs v63.1_rl
  TEST (top --top): the band gate (991 real players' tapes) and the held-out ladder half, plus the worst turn time
Writes data/gates/layer_screen.json and data/gates/layer_cands.json (the table).
"""
import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_screen import RL, paired, play  # noqa: E402

V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
OUT_TABLE = os.path.join(RL, "data", "gates", "layer_cands.json")
LEAD32 = {"ev_h": 32, "dp_h": 32, "mp_h": 32}
VARIANTS = [
    ("term700", {"term_start": 700}), ("term696", {"term_start": 696}), ("term704", {"term_start": 704}),
    ("sims1024", {"term_sims": 1024}), ("pass3", {"term_passes": 3}), ("props12", {"term_props": 12}),
    ("term700_s1024", {"term_start": 700, "term_sims": 1024}), ("term696_s1024_p3", {"term_start": 696, "term_sims": 1024, "term_passes": 3}),
    ("tsell648", {"tsell_from": 648}), ("tsell624", {"tsell_from": 624}), ("tsell_w48", {"tsell_window": 48}),
    ("tsell_min25", {"tsell_min": 25.0}), ("tsell_min100", {"tsell_min": 100.0}),
    ("tsell648_w48", {"tsell_from": 648, "tsell_window": 48}),
    ("press552", {"press_on": True, "press_from": 552}), ("press600", {"press_on": True, "press_from": 600}),
    ("rsa16", {"rsa_look": 16}), ("rsa8", {"rsa_look": 8}),
    ("v92w8", {"v92_ext_window": 8}), ("v92top3", {"v92_top": 3}),
    ("end_combo", {"term_start": 700, "term_sims": 1024, "tsell_from": 648, "tsell_window": 48}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--top", type=int, default=4)
    a = ap.parse_args()
    t = json.load(open(V64, encoding="utf-8"))
    base = dict(t["profiles"][35])
    ids = {}
    for name, kn in VARIANTS:
        for tag, extra in (("", {}), ("_copy", LEAD32)):
            p = dict(base)
            p.update(kn)
            p.update(extra)
            p["name"] = f"c35_{name}{tag}"
            ids[name + tag] = len(t["profiles"])
            t["profiles"].append(p)
    t["note"] = "v64rl + end-game layer variants of profile 35 (python/top50/layer_screen.py); *_copy = + lead32"
    json.dump(t, open(OUT_TABLE, "w", encoding="utf-8"), indent=1)
    arm = lambda n: ["--profiles", OUT_TABLE, "--guarded", "--pa", str(ids[n]), "--group", f"{ids[n]},{ids[n]},{ids[n + '_copy']}"]  # noqa: E731
    ref_args = ["--profiles", OUT_TABLE, "--guarded", "--pa", "35", "--group", "35,35,36"]
    losses = sorted(glob.glob(os.path.join(RL, "data", "tapes", "losses", "*.json")))
    sel = sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", "ladder", "*.json")))
    band_tr = []
    for d in ("lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"):
        band_tr += sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", d, "*.json")))
    random.Random(7).shuffle(band_tr)
    sel += band_tr[:600]
    ref_l, ref_s = play(ref_args, losses, a.threads), play(ref_args, sel, a.threads)
    rep = {"table": OUT_TABLE, "losses": len(losses), "ref_loss_wins": int(sum(v == 1 for v in ref_l.values())), "selection": {}, "test": {}}
    print(f"[layers] {len(losses)} real losses (v63.1_rl wins {rep['ref_loss_wins']}), selection {len(sel)} tapes, {len(VARIANTS)} variants", flush=True)
    for name, _ in VARIANTS:
        pl, ps = paired(play(arm(name), losses, a.threads), ref_l), paired(play(arm(name), sel, a.threads), ref_s)
        rep["selection"][name] = {"losses": pl, "sel": ps, "score": pl["better"] - pl["worse"] + ps["better"] - ps["worse"]}
        print(f"[layers] {name:18s}: losses +{pl['better']}/-{pl['worse']} (wins {pl['wins']}) | selection +{ps['better']}/-{ps['worse']}", flush=True)
    top = sorted(rep["selection"], key=lambda n: rep["selection"][n]["score"], reverse=True)[:a.top]
    vf = sorted(glob.glob(os.path.join(RL, "data", "tapes", "ladder_val", "**", "*.json"), recursive=True))
    ref_v = play(ref_args, vf, a.threads)
    for name in top:
        t0 = time.time()
        x = paired(play(arm(name), vf, a.threads), ref_v)
        k = ids[name]
        r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", str(k), "--cand-group", f"{k},{k},{ids[name + '_copy']}",
                            "--profiles", OUT_TABLE, "--ref-profile", "35", "--ref-group", "35,35,36", "--name", f"layers-{name}", "--threads", str(a.threads)],
                           cwd=RL, capture_output=True, text=True)
        line = (r.stdout.strip().splitlines() or [""])[-1]
        m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", line)
        rep["test"][name] = {"ladder": x, "band": line, "losses_below_2500": int(m.group(1)) if m else None, "band_paired": (int(m.group(3)), int(m.group(4))) if m else None}
        print(f"[layers] TEST {name}: held-out ladder +{x['better']}/-{x['worse']} | band {rep['test'][name]['band_paired']}, losses<2500 {rep['test'][name]['losses_below_2500']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(rep, open(os.path.join(RL, "data", "gates", "layer_screen.json"), "w"), indent=1)
    print("[layers] -> data/gates/layer_screen.json")


if __name__ == "__main__":
    main()
