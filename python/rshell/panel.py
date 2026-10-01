"""The evaluation panel for reactive shell v2 and chain-setting searches (python/rshell/cmaes.py, screen.py).

CLOSED LOOP on the 64-world seed bank (data/worlds/w64_bank.json: world = the first two unlocked shops, realized
by play; within our agent family it depends only on the seed, 192/192 held-out seeds verified on our engine):
every seed is played from BOTH seats against our own lineage, which reacts:
  v63 (rl3 35), v62.1 (rl3 19), v62 (rl3 13), v61.1 (rl3 0), v63.1_rl (v64rl 35 + group), MIRROR = the reference
  agent itself (i790 + big1 shell: the copy of our public replays), RAND = a random clone-lineage knob vector.
OPEN LOOP on real players' tapes: the training half of our ladder games, 600 training band tapes, half of our
real losses (split A; split B is held out for the final test).
A game scores win 1 / draw 0.5 / loss 0 for OUR seat; comparisons are paired (same seed, seat, opponent).
"""
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the v2 flags (--a-args, --rshell, --chain-off, --knob-over) exist only in the target-v2 build; the box exports
# KRL_BIN=target-dev for older jobs, whose binaries silently ignore unknown flags -> always override here
os.environ["KRL_BIN"] = os.environ.get("KRL_V2_BIN") or os.path.join(RL, "target-v2", "release")
sys.path.insert(0, os.path.join(RL, "python"))
from copy_screen import paired, play as tape_play  # noqa: E402

BIN = os.environ["KRL_BIN"]
EXE = ".exe" if os.name == "nt" else ""
SELFPLAY = os.path.join(BIN, "selfplay" + EXE)
BANK = os.path.join(RL, "data", "worlds", "w64_bank.json")
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
SHIELD = os.path.join(RL, "configs", "shield", "v1.json")
RUN = os.path.join(RL, "weights", "ppo", "ppo-gru64-f2-p36-from-init-20260926T0344Z")
BIG1 = os.path.join(RL, "weights", "shell", "big1", "shell.json")


def ppo_args(it=790, shell=BIG1, extra=""):
    pol = os.path.join(RUN, "snapshots", f"i{it:06d}.bin")
    s = f"--profiles {RL3} --policy {pol} --shield {SHIELD}"
    if shell:
        s += f" --shell {shell}"
    return (s + " " + extra).strip()


REF = ppo_args()
OPPONENTS = {
    "v63": f"--profiles {RL3} --pa 35",
    "v62.1": f"--profiles {RL3} --pa 19",
    "v62": f"--profiles {RL3} --pa 13",
    "v61.1": f"--profiles {RL3} --pa 0",
    "v63.1_rl": f"--profiles {V64} --pa 35 --group 35,35,36",
    "mirror": REF,
    "rand": "RAND",
}
# FITNESS opponents: everything except the MIRROR. The reference playing itself draws almost every game, so ANY
# deviation turns draws into wins against it (2026-09-27: the chain-off fallback was +312/-48 vs mirror but
# -10 to -21 vs every other lineage opponent and worse on the band). The mirror stays in reports only.
FIT = {k: v for k, v in OPPONENTS.items() if k != "mirror"}


def bank_seeds(split="train", per_world=1, offset=0):
    """One (or per_world) seed per world, rotating through the bank with `offset` (a generation number)."""
    b = json.load(open(BANK))[split]
    out = []
    for w in sorted(b):
        s = b[w]
        for j in range(per_world):
            out.append((w, s[(offset * per_world + j) % len(s)]))
    return out


def _selfplay(a_args, b_args, seeds, threads, rand_seat=None):
    cmd = [SELFPLAY, "--seeds", ",".join(str(s) for s in seeds), "--threads", str(threads), "--a-args", a_args, "--b-args", b_args]
    if rand_seat is not None:
        cmd += ["--rand-b", "7", "--rand-seat", str(rand_seat)]
    r = subprocess.run(cmd, cwd=RL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"selfplay rc {r.returncode}: {r.stderr[-600:]}")
    out = {}
    for ln in r.stdout.splitlines():
        x = ln.split("\t")
        if len(x) >= 6:
            out[int(x[0])] = (float(x[1]), float(x[2]), x[5])
    return out


def closed_loop(agent, seeds, threads=24, opponents=None, plain=None):
    """{(opp, seed, seat): score of `agent` in that seat} over `seeds` [(world, seed)] x opponents x both seats."""
    opponents = opponents or OPPONENTS
    plain = plain or f"--profiles {RL3} --pa 0"
    sd = [s for _, s in seeds]
    world = {s: w for w, s in seeds}
    jobs = []
    for name, opp in opponents.items():
        for seat in (0, 1):
            if opp == "RAND":
                jobs.append((name, seat, (agent, plain, 1) if seat == 0 else (plain, agent, 0)))
            else:
                jobs.append((name, seat, (agent, opp, None) if seat == 0 else (opp, agent, None)))
    per = max(1, -(-threads // len(jobs)))
    res, worlds = {}, {}

    def one(j):
        name, seat, (a, b, rs) = j
        return name, seat, _selfplay(a, b, sd, per, rs)

    with ThreadPoolExecutor(min(len(jobs), max(1, threads))) as ex:
        for name, seat, out in ex.map(one, jobs):
            for s, (b0, b1, w) in out.items():
                m = (b0 - b1) if seat == 0 else (b1 - b0)
                res[(name, s, seat)] = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
                worlds[(name, s, seat)] = w
                if w != world.get(s) and name != "rand":
                    worlds[(name, s, seat)] = f"MISMATCH {w} != {world.get(s)}"
    return res, worlds


def tape_sets(split="A"):
    """Open-loop sets: training ladder half, 600 training band tapes (seeded), our real losses split A/B."""
    lad = sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", "ladder", "*.json")))
    band = []
    for d in ("lt2100", "2100-2300", "2300-2500", "2500-2700", "2700plus"):
        band += sorted(glob.glob(os.path.join(RL, "data", "tapes", "train", d, "*.json")))
    random.Random(7).shuffle(band)
    losses = sorted(glob.glob(os.path.join(RL, "data", "tapes", "losses", "*.json")))
    half = lambda f: "A" if int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 2 == 0 else "B"  # noqa: E731
    pub = sorted(glob.glob(os.path.join(RL, "data", "rshell", "public25", "rca", "tapes", "*.json")))
    pub += sorted(glob.glob(os.path.join(RL, "data", "rshell", "public25", "wins", "tapes", "*.json")))
    out = {"ladder": lad, "band": band[:600], "losses": [f for f in losses if half(f) == split]}
    if pub:
        # the top-25 public agents' recorded games against v63.5_rl (losses + a sample of wins), open loop
        out["public"] = [f for f in pub if half(f) == split]
    # our REAL ladder games vs the players who beat us + the close games we won (python/ladder_study.py), split by game:
    # half A enters fitness (CMA), half B is the held-out real-ladder yardstick (fix_test)
    rl = sorted(glob.glob(os.path.join(RL, "data", "ladder_study", "tapes", "*.json")))
    if rl:
        out["real_ladder"] = [f for f in rl if half(f) == split]
    # KRL_TOP_N=N: N of the top-player tapes (python/gm_top_tapes.py, 2500+ players' recorded games, 28 Sep), split
    # A/B by game like the rest (A = fitness, B = held-out), a fixed seeded sample so every generation sees the same set
    n_top = int(os.environ.get("KRL_TOP_N", "0") or 0)
    if n_top:
        top = sorted(glob.glob(os.path.join(RL, "data", "tapes", "top_all", "shard_*", "*.json")))
        top = [f for f in top if half(f) == split]
        random.Random(11).shuffle(top)
        out["top"] = top[:n_top]
    return out


def open_loop(agent, sets, threads=24):
    args = agent.split() + ["--guarded"]
    return {name: tape_play(args, files, threads) for name, files in sets.items()}


def compare(c, r):
    return paired(c, r)


def score_delta(c, r):
    """Paired (better - worse) of a candidate result dict vs a reference dict (same keys)."""
    p = paired(c, r)
    return p["better"] - p["worse"], p
