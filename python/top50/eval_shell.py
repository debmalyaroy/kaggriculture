"""Evaluate a learned sales shell against v63.1_rl (our live best), paired on the same games.

    python python/top50/eval_shell.py --shell weights/shell/NAME/shell.json [--threads 16]

Candidate = v63.1_rl + the shell (profiles v64rl, profile 35, group 35,35,36, --shell). Reference = v63.1_rl.
The per-decision tau from training does not survive being applied every turn (one-turn labels compound),
so the setting is chosen on whole games:
  1. sweep tau x direction (either way / only sell more / only sell less) on the held-out half of our real
     ladder games (data/tapes/ladder_val), paired vs v63.1_rl; the best by (better - worse) is written
     back into shell.json (tau, mode) so DAgger and any build use it
  2. the band gate on 991 real players' tapes (python/band_gate.py) for that best setting
Writes weights/shell/NAME/eval.json.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_screen import RL, paired, play  # noqa: E402

V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
TAUS = (0.7, 0.8, 0.9, 0.95)
MODES = {0: "either", 1: "sell_more", 2: "sell_less"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    shell = os.path.abspath(a.shell)
    wd = os.path.dirname(shell)
    out = os.path.join(wd, "eval.json")
    model = json.load(open(shell))
    # choose on the TRAINING half of our ladder games (never the test set); test on the held-out half + band gate
    tr = os.path.join(RL, "data", "tapes", "train", "ladder")
    sel_files = sorted(glob.glob(os.path.join(tr, "*.json")))
    # + 600 real players' tapes from PPO's training bands (never the band gate's tapes): the setting must
    # help on the ladder we meet AND on the wider field
    import random
    band_tr = sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", "*-*", "*.json")) + glob.glob(os.path.join(RL, "data", "tapes", "train", "lt2100", "*.json"))
                     + glob.glob(os.path.join(RL, "data", "tapes", "train", "2700plus", "*.json")))
    random.Random(7).shuffle(band_tr)
    sel_files += band_tr[:600]
    va = os.path.join(RL, "data", "tapes", "ladder_val")
    grp = {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(va, "index.tsv"), encoding="utf-8"), delimiter="\t")}
    files = sorted(glob.glob(os.path.join(va, "**", "*.json"), recursive=True))
    base = ["--profiles", V64, "--guarded", "--pa", "35", "--group", "35,35,36"]
    ref_sel = play(base, sel_files, a.threads)
    ref = play(base, files, a.threads)
    rep = {"shell": shell, "selection_set": tr, "sweep": []}
    best = None
    for groups in ([], [2]):
        for mode in MODES:
            for tau in TAUS:
                v = dict(model, tau=tau, mode=mode, groups=groups)
                f = os.path.join(wd, f"shell_g{''.join(map(str, groups)) or 'all'}_m{mode}_t{int(tau * 100)}.json")
                json.dump(v, open(f, "w"))
                r = paired(play(base + ["--shell", f], sel_files, a.threads), ref_sel)
                r.update(tau=tau, mode=MODES[mode], groups="COPY" if groups else "all")
                rep["sweep"].append(r)
                print(f"[eval_shell] {r['groups']:4s} {MODES[mode]:9s} tau {tau}: +{r['better']}/-{r['worse']} p {r['p']} (selection half)", flush=True)
                if best is None or r["better"] - r["worse"] > best[0]["better"] - best[0]["worse"]:
                    best = (r, mode, tau, groups)
    r, mode, tau, groups = best
    model.update(tau=tau, mode=mode, groups=groups)
    json.dump(model, open(shell, "w"))
    cand = play(base + ["--shell", shell], files, a.threads)
    rep["chosen"] = {"tau": tau, "mode": MODES[mode], "groups": "COPY" if groups else "all", "all": paired(cand, ref)}
    for G in ("COPY", "PARTIAL", "DIFFERENT"):
        rep["chosen"][G] = paired({k: v for k, v in cand.items() if grp.get(k) == G}, ref)
    x = rep["chosen"]["all"]
    print(f"[eval_shell] chosen {'COPY-only' if groups else 'all'} {MODES[mode]} tau {tau}: held-out ladder +{x['better']}/-{x['worse']} p {x['p']}", flush=True)
    r2 = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", "35", "--cand-group", "35,35,36",
                         "--cand-shell", shell, "--profiles", V64, "--ref-profile", "35", "--ref-group", "35,35,36",
                         "--name", "shell-" + os.path.basename(wd), "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
    rep["band_gate"] = (r2.stdout.strip().splitlines() or [r2.stderr[-400:]])[-1]
    print(r2.stdout[-1500:], flush=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(f"[eval_shell] -> {out}")


if __name__ == "__main__":
    main()
