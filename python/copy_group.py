"""Copy-race profiles played only against the opponent group they help (queue Q31).

    python python/copy_group.py [--threads 16]

Q30 (python/copy_screen.py) found lead32 (profile 35 with 32 h lead-sell windows) beats v63 on the held-out
ladder half, +12/-2, mostly against COPY opponents, but is even on the 988-tape band gate (+2/-5). Here the
agent plays v63 (35) on day 0 and, from the day-1 boundary (step 25), a fixed profile per day-0 opponent
group (tapeplay --group D,P,C: the shield's DIFFERENT / PARTIAL / COPY signal, as in the live agent).
Every arm is paired vs plain v63 on the held-out ladder half (all groups, and per group), then the arms
that beat v63 there go through the band gate. Uses data/gates/copy_cands.json from Q30 (ids 36+).
Writes data/gates/copy_group.json.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy_screen import OUT_TABLE, RL, paired, play  # noqa: E402

# name -> (DIFFERENT, PARTIAL, COPY); ids from copy_screen.VARIANTS: 36 lead32, 37 lead40, 62 lead32_rsa16
ARMS = {
    "lead32_copy": (35, 35, 36),
    "lead32_copy_partial": (35, 36, 36),
    "lead40_copy": (35, 35, 37),
    "lead32rsa16_copy": (35, 35, 62),
    "lead32_all_g": (36, 36, 36),  # control: lead32 from step 25 (day 0 v63); should track Q30's lead32
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    if not os.path.exists(OUT_TABLE):
        raise SystemExit(f"{OUT_TABLE} missing: run python/copy_screen.py (Q30) first")
    va_dir = os.path.join(RL, "data", "tapes", "ladder_val")
    grp = {r["id"]: r["group"] for r in csv.DictReader(open(os.path.join(va_dir, "index.tsv"), encoding="utf-8"), delimiter="\t")}
    files = sorted(glob.glob(os.path.join(va_dir, "**", "*.json"), recursive=True))
    base = ["--profiles", OUT_TABLE, "--guarded", "--pa", "35"]
    ref = play(base, files, a.threads)
    print(f"[cgroup] held-out ladder half: {len(ref)} games; v63 wins {int(sum(v == 1 for v in ref.values()))}", flush=True)
    rep = {"table": OUT_TABLE, "held_out": {}, "band_gate": {}}
    for name, g in ARMS.items():
        c = play(base + ["--group", ",".join(map(str, g))], files, a.threads)
        x = {"groups": list(g), "all": paired(c, ref)}
        for G in ("COPY", "PARTIAL", "DIFFERENT"):
            x[G] = paired({k: v for k, v in c.items() if grp.get(k) == G}, ref)
        rep["held_out"][name] = x
        print(f"[cgroup] {name:20s} {g}: all +{x['all']['better']}/-{x['all']['worse']} p {x['all']['p']}; "
              + "; ".join(f"{G} +{x[G]['better']}/-{x[G]['worse']}" for G in ("COPY", "PARTIAL", "DIFFERENT")), flush=True)
    for name, x in rep["held_out"].items():
        if name == "lead32_all_g" or x["all"]["better"] <= x["all"]["worse"]:
            continue
        r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand-profile", "35", "--cand-group", ",".join(map(str, x["groups"])),
                            "--name", f"cgroup-{name}", "--profiles", OUT_TABLE, "--ref-profile", "35", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        rep["band_gate"][name] = (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1]
        print(r.stdout[-1500:], flush=True)
    json.dump(rep, open(os.path.join(RL, "data", "gates", "copy_group.json"), "w"), indent=1)
    print("[cgroup] -> data/gates/copy_group.json")


if __name__ == "__main__":
    main()
