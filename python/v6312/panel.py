"""A0 fast screening panel on the in-process Rust engine: CAND vs our real adaptive releases, both seats.

    python .local/v6312/panel.py --cand-stage data/builds/v63.11_rl/stage [--cand-extra "--rshell X"] --out OUT.tsv [--seeds 3] [--threads 8]
Rows: opp  seat  seed  world  us  them  gap. World = the bank label (seed-realised worlds can differ in play)."""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from args import stage_args
RL = os.environ.get("KRL_RL") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SP = os.environ.get("KRL_SELFPLAY") or os.path.join(RL, "target-v6312", "release", "selfplay.exe")
OPPS = ["v63.6_rl", "v63.7_rl", "v63.8_rl", "v63.10_rl", "v63.11_rl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-stage", required=True); ap.add_argument("--cand-extra", default=""); ap.add_argument("--cand-drop", default="", help="regex removed from the candidate args, e.g. '--policy \S+'")
    ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split", default="heldout"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--opps", default=",".join(OPPS)); ap.add_argument("--worlds", default=""); ap.add_argument("--sp", default=SP)
    a = ap.parse_args()
    bank = json.load(open(os.path.join(RL, "data/worlds/w64_bank.json")))[a.split]
    keep = set(a.worlds.split(",")) if a.worlds else None
    w_of = {s: w for w, ss in bank.items() for s in ss[:a.seeds] if keep is None or w in keep}
    seeds = ",".join(str(s) for s in w_of)
    cb, ca = stage_args(a.cand_stage)
    if a.cand_drop:
        import re; ca = re.sub(a.cand_drop, "", ca)
    ca = (ca + " " + a.cand_extra).strip()
    with open(a.out, "w") as f:
        for o in a.opps.split(","):
            ob, oa = stage_args(os.path.join(RL, "data/builds", o, "stage"))
            for seat in (0, 1):
                A, B = ((cb, ca), (ob, oa)) if seat == 0 else ((ob, oa), (cb, ca))
                p = subprocess.run([a.sp, "--a", A[0], "--b", B[0], "--a-args", A[1], "--b-args", B[1], "--seeds", seeds,
                                    "--threads", str(a.threads)], capture_output=True, text=True, cwd=RL)
                n = 0
                for l in p.stdout.splitlines():
                    x = l.split("\t")
                    if len(x) < 3 or not x[0].lstrip("-").isdigit(): continue
                    s = int(x[0]); b0, b1 = float(x[1]), float(x[2]); us, th = (b0, b1) if seat == 0 else (b1, b0)
                    f.write(f"{o}\t{seat}\t{s}\t{w_of.get(s,'?')}\t{us:.0f}\t{th:.0f}\t{us-th:.0f}\n"); n += 1
                f.flush()
                print(o, seat, n, "games", p.stderr.strip().splitlines()[-4:] if p.returncode else "", flush=True)

if __name__ == "__main__":
    main()
