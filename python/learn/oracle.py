"""Branch oracle (expert iteration; PLAN s20 L3, queue Q19). Runs continuously beside PPO.

    python python/learn/oracle.py [--games 64] [--days 0,1,2,3,12,20,23,26] [--threads 16]

Loop: take PPO's current weights (the newest weights/ppo/<run>/current.bin), play --games greedy games
against the PPO opponent mix (30% real players' tapes), and on each key day replay the identical game
once per allowed candidate profile with that day forced (ppo-rollout --oracle-days; every game is
deterministic given its seed and choices, and forcing the policy's own choice reproduces the base game
to the dollar). Writes data/oracle/o_<UTC>.{tsv,traj,oracle}; keeps the newest --keep batches.
python/learn/ppo.py trains on these labels (only decisions where the choice changed the margin).
Measured 2026-09-26: the day's profile changes the margin on ~16% of decisions and flips W/L on ~1.5%,
which is why plain PPO barely moves.
"""
import argparse
import datetime as dt
import glob
import os
import subprocess
import sys
import time

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
ROLL = os.path.join(BIN, "ppo-rollout" + (".exe" if os.name == "nt" else ""))
OUT = os.path.join(RL, "data", "oracle")
PPO = os.path.join(RL, "weights", "ppo")
SHIELD = os.path.join(RL, "configs", "shield", "v1.json")
TAPES = os.path.join(RL, "data", "tapes", "train")
CANDS = "0,2,12,13,19,21,22,23,24,29,30,31,32,33,34,35"  # rl3: + the v92 variants and v63 (35)  # our lineage + escalated/AFR variants (9, 11 are masked)


def current_weights():
    runs = sorted(d for d in glob.glob(os.path.join(PPO, "ppo-*")) if os.path.exists(os.path.join(d, "current.bin")))
    return os.path.join(runs[-1], "current.bin") if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=128)
    # measured 2026-09-26 over all 30 days (96 games): days 0-5 and 7-11 never change the game (shops not yet
    # unlocked; the opening plays itself); D6 matters on 66% of decisions, D12 30%, D15+ 50-96%, W/L flips at D13-D27
    ap.add_argument("--days", default="6,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28")
    ap.add_argument("--cands", default=CANDS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("KRL_THREADS") or 16))
    ap.add_argument("--keep", type=int, default=60)
    ap.add_argument("--tape-frac", type=float, default=0.7)
    ap.add_argument("--learner-args", default="", help="the shipped agent's learner extras for the branches (--shell F --rshell F --knob-over F --chain-off S): labels must describe the agent that ships")
    ap.add_argument("--tapes-dir", default=None, help="real-player tape pool (default data/tapes/train); 28 Sep: data/tapes/leaders = decisions labelled in the top players' situations")
    ap.add_argument("--max-margin", type=float, default=3000.0, help="branch only games lost or won by less than this")
    ap.add_argument("--once", action="store_true")
    # ppo2 (27 Sep): a second oracle for the ppo2 run -- its own label dir, weights from that run's root,
    # closed-loop copy games (--mix), the 64-world bank, mid-day slots in --days ("25.13")
    ap.add_argument("--out", default=None, help="label dir (default data/oracle)")
    ap.add_argument("--root", default=None, help="PPO run root whose newest current.bin is branched (default weights/ppo)")
    ap.add_argument("--mix", default=None, help="opponent mix for the non-tape games, e.g. mirror:40,v63:40,rand:20")
    ap.add_argument("--seed-bank", default=None)
    ap.add_argument("--seed-base", type=int, default=40_000_000)
    a = ap.parse_args()
    global OUT, PPO
    if a.out:
        OUT = a.out
    if a.root:
        PPO = a.root
    os.makedirs(OUT, exist_ok=True)
    seed0 = a.seed_base + int(time.time()) % 1_000_000 * 64  # oracle seeds: their own range, fresh per batch
    while True:
        w = current_weights()
        if not w:
            print("[oracle] no PPO weights yet; waiting", flush=True)
            time.sleep(60)
            continue
        tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = os.path.join(OUT, f"o_{tag}")
        # a private copy: PPO rewrites current.bin every iteration
        wcopy = prefix + ".weights.bin"
        open(wcopy, "wb").write(open(w, "rb").read())
        cmd = [ROLL, "--weights", wcopy, "--out", prefix, "--games", str(a.games), "--seed0", str(seed0), "--threads", str(a.threads),
               "--profiles", os.environ.get("KRL_PROFILES") or os.path.join(RL, "configs", "profiles", "rl3.json"), "--greedy", "--oracle-days", a.days, "--oracle-cands", a.cands]
        if os.path.exists(SHIELD):
            cmd += ["--shield", SHIELD]
        tapes = os.path.abspath(a.tapes_dir) if a.tapes_dir else TAPES
        if os.path.isdir(tapes):  # the objective is measured against real players: most oracle games are tapes
            cmd += ["--tapes", tapes, "--tape-frac", str(a.tape_frac)]
        cmd += ["--oracle-max-margin", str(a.max_margin)]
        cmd += a.learner_args.split()
        if a.mix:
            cmd += ["--mix", a.mix]
        if a.seed_bank:
            cmd += ["--seed-bank", a.seed_bank]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RL, capture_output=True, text=True)
        os.remove(wcopy)
        if r.returncode != 0:
            sys.exit(f"[oracle] ppo-rollout failed rc {r.returncode}: {r.stderr[-800:]}")
        rows = [ln.split("\t") for ln in open(prefix + ".oracle", encoding="utf-8").read().splitlines()]
        useful = 0
        for _, _, cells in rows:
            ms = [float(c.split(":")[1]) for c in cells.split(",")]
            useful += max(ms) != min(ms)
        print(f"[oracle] {tag}: {a.games} games, {len(rows)} decisions, {useful} where the choice matters, "
              f"{time.time() - t0:.0f}s", flush=True)
        seed0 += a.games
        batches = sorted(glob.glob(os.path.join(OUT, "o_*.oracle")))
        for old in batches[:-a.keep]:
            for ext in (".oracle", ".tsv", ".traj"):
                try:
                    os.remove(old[:-7] + ext)
                except OSError:
                    pass
        if a.once:
            return


if __name__ == "__main__":
    main()
