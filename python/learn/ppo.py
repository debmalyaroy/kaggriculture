"""Continuous recurrent PPO for the macro policy (tasks P7.1-P7.5, queue Q16).

    python python/learn/ppo.py [--games 1024] [--threads 14]

Loop (runs until stopped; every step resumable):
  1. export the current weights (atomic) -> ppo-rollout plays --games games (learner samples its
     profile each day; opponents: v61.1, fixed escalated/AFR profiles, random knobs, frozen
     snapshots of this run);
  2. PPO update over whole 30-day sequences: gamma 1, GAE lambda .95, clip .2, value loss, entropy
     bonus, KL(pi || BC teacher) with a decaying coefficient;
  3. checkpoint (model + optimizer + iteration, atomic) + metrics line;
  4. every --val-every iterations: greedy policy vs v61.1 on the fixed validation seeds
     (500000.., both seats); the best is kept as best.bin; two consecutive drops > 5 points below
     the best -> roll back to the best and halve the learning rate;
  5. the BC teacher is re-read from weights/bc/LATEST each iteration and hot-swapped when a newer
     one appears (the daily delta fine-tune).
Run dir: weights/ppo/<PPORUN>/ {ckpt.pt, current.bin, best.bin, snapshots/iNNNNNN.bin, metrics.jsonl,
HEARTBEAT}. A new run starts only if no unfinished run exists; `--new` forces one.

ppo2 options (27 Sep; all off by default, so an existing run resumes unchanged):
  --mid-days 25-29     a second decision at hour 13 on those days; each game then has 30 + n decision slots in
                       chronological order (crates/dayobs slot_of) and exports carry a trailer the Rust side reads
  --anchor-net W       KL anchor = this policy (a .bin export or a .pt checkpoint) instead of --anchor-prior / BC
  --init-from W.bin    warm start from an export (as well as a ckpt.pt)
  --lr-half-life N     lr = base lr * 0.5 ** (iterations since start / N); a rollback halves the base lr
  --mix, --seed-bank, --tape-frac   the rollout opponent mix, 64-world seed bank, and tape share
  --rshell, --chain-off, --knob-over  passed to every learner rollout and validation (reactive shell v2)
  --oracle-dir D       read branch-oracle labels from D (default data/oracle)
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MacroNet, NF, profile_names  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BC = os.path.join(RL, "weights", "bc")
PPO = os.path.join(RL, "weights", "ppo")
ROLL = os.path.join(os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release"), "ppo-rollout" + (".exe" if os.name == "nt" else ""))
W = NF + 5  # per day: obs[NF], action, logp, value logit, allowed-profile mask (low, high u32 bits in two f32 slots)


MID_MAGIC = 0x4D494431  # crates/policy MID_MAGIC ("MID1")


def parse_days(s):
    out = []
    for part in (s or "").split(","):
        if not part.strip():
            continue
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def slot_days(mid):
    """Day of each decision slot (crates/dayobs slot_days): day d, then d again for a mid day."""
    v = []
    for d in range(30):
        v.append(d)
        if d in mid:
            v.append(d)
    return v


def mask_bits(t):
    """The 64-bit allowed-profile mask from the two f32 slots (rl3 has 36 profiles: one u32 is not enough)."""
    lo = np.ascontiguousarray(t[..., NF + 3]).view("<u4").astype(np.uint64)
    hi = np.ascontiguousarray(t[..., NF + 4]).view("<u4").astype(np.uint64)
    return (lo | (hi << np.uint64(32))).astype(np.int64)
# validation: its own seed range (the gate uses 500,000.. per opponent), 800 greedy games vs --val-opp
VAL_SEED0, VAL_N = 600_000, 800
# v3 (2026-09-26): best/rollback on a composite, not head-to-head alone. A checkpoint level with v63 head-to-head was
# significantly WORSE than v63 against the fixed panel in the tournament (i070/i080: 33-34 better / 73-76 worse).
#   val_combo = (h2h score vs v63 - 0.5) + (panel: cand - v63, same seeds) + (hard tapes: cand - v63, same seeds)
VAL_VERSION = 4
# v4 (2026-09-26 16:10 IST): + our own ladder games, held-out half (python/ladder_split.py: odd episodes), each played
# once by tapeplay against the recorded opponent (the sim reproduces our agents' real margins to the dollar on
# 537/549), paired vs v63 and weighted 2x: every one of v63's ladder losses is there (all against COPY opponents).
#   val_combo = (h2h - 0.5) + panel delta + hard delta + 2 * ladder delta
LADDER_VAL = os.path.join(RL, "data", "tapes", "ladder_val")
LADDER_W = 2.0
TAPEPLAY = os.path.join(os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release"), "tapeplay" + (".exe" if os.name == "nt" else ""))
VAL_PANEL = ["fixed:0", "fixed:2", "fixed:12", "fixed:13", "fixed:19", "fixed:22", "fixed:30", "fixed:31"]  # the tournament panel minus v63
VAL_PANEL_N, VAL_PANEL_SEED0 = 150, 700_000
VAL_HARD_N, VAL_HARD_SEED0 = 800, 800_000
HARD_TAPES = os.path.join(RL, "data", "tapes", "hard")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ORACLE = os.path.join(RL, "data", "oracle")
N_SLOTS = 30  # decision slots per game (30 + mid days), set in main
_ORACLE_CACHE = {}


def oracle_rows(max_files, sigma, beta, n_act, min_gain=0.0):
    """Labels from the branch oracle (python/learn/oracle.py): for each decision where the forced profile
    changed the learner's margin, the observation sequence of that game, the day, a target distribution
    over the candidates (softmax of beta * Phi(margin / sigma)), the allowed mask and a weight (how much
    the best candidate beat the policy's own choice). Returns numpy arrays or None."""
    files = sorted(glob.glob(os.path.join(ORACLE, "o_*.oracle")))[-max_files:]
    for f in list(_ORACLE_CACHE):
        if f not in files:
            del _ORACLE_CACHE[f]
    for f in files:
        if f in _ORACLE_CACHE:
            continue
        pre = f[:-7]
        try:
            tsv = [ln.split("\t") for ln in open(pre + ".tsv", encoding="utf-8").read().splitlines()]
            raw = np.fromfile(pre + ".traj", dtype="<f4")
            traj = raw.reshape(len(tsv), len(raw) // (len(tsv) * W), W)
            if traj.shape[1] != N_SLOTS:  # labels from a run with a different decision layout
                continue
        except (OSError, ValueError, ZeroDivisionError):
            continue
        at = {int(r[0]): i for i, r in enumerate(tsv)}
        obs, day, tgt, allow, wt, phs = [], [], [], [], [], []
        for ln in open(f, encoding="utf-8").read().splitlines():
            seed, d, cells = ln.split("\t")
            c = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in cells.split(",")]
            ms = np.array([m for _, m in c])
            if ms.max() == ms.min() or int(seed) not in at:
                continue
            g, d = at[int(seed)], int(d)
            bits = int(mask_bits(traj[g, d]).reshape(-1)[0])
            ok = np.array([(bits >> k) & 1 == 1 for k, _ in c])
            if not ok[1:].any():
                continue
            phi = 0.5 * (1 + np.array([math.erf(m / (sigma * 2 ** 0.5)) for m in ms]))
            p = np.zeros(n_act, np.float32)
            z = (phi - phi[ok].max()) / max(phi[ok].max() - phi[ok].min(), 1e-9)  # 0 = best, -1 = worst allowed
            e = np.where(ok, np.exp(beta * z), 0.0)
            for (k, _), v in zip(c, e / e.sum()):
                p[k] = v
            ph = np.full(n_act, np.nan, np.float32)  # Phi(margin/sigma) of each branched, allowed candidate (regret metric)
            for (k, _), v, o in zip(c, phi, ok):
                if o:
                    ph[k] = v
            obs.append(traj[g, :, :NF])
            day.append(d)
            tgt.append(p)
            allow.append([(bits >> k) & 1 == 1 for k in range(n_act)])
            wt.append(float(phi[ok].max() - phi[0]))
            phs.append(ph)
        _ORACLE_CACHE[f] = (obs, day, tgt, allow, wt, phs)
    parts = [v for f in files if f in _ORACLE_CACHE for v in [_ORACLE_CACHE[f]] if v[0]]
    if not parts:
        return None
    cat = lambda i, dt_: np.array([x for v in parts for x in v[i]], dtype=dt_)  # noqa: E731
    out = cat(0, np.float32), cat(1, np.int64), cat(2, np.float32), cat(3, bool), cat(4, np.float32), cat(5, np.float32)
    keep = out[4] >= min_gain  # material decisions only: the policy's own choice lost >= min_gain Phi to the best
    return tuple(x[keep] for x in out) if keep.any() else None


def bc_latest():
    try:
        return open(os.path.join(BC, "LATEST")).read().strip()
    except OSError:
        return None


def load_net(n_act, ckpt, dev):
    s = torch.load(ckpt, map_location=dev, weights_only=False)
    net = MacroNet(n_act).to(dev)
    net.load_state_dict(s["model"])
    return net, s


MID = []  # mid-day decision days of this run (set in main from --mid-days / the checkpoint)


def export_bin(net, path):
    tmp_dir = path + ".d"
    net.export(tmp_dir, {"note": "ppo export", "mid_days": MID})
    wb = os.path.join(tmp_dir, "weights.bin")
    if MID:
        # trailer [days..., n, MID_MAGIC] (crates/policy): the Rust agent then also decides at hour 13 of these days
        tail = np.array([float(d) for d in MID] + [float(len(MID))], dtype="<f4").tobytes() + np.array([MID_MAGIC], dtype="<u4").tobytes()
        with open(wb, "ab") as fh:
            fh.write(tail)
    os.replace(wb, path)


def net_from_bin(path, n_act, dev):
    """A MacroNet from a weights.bin export (EXPORT_ORDER; the training-only fam head stays at its init)."""
    from model import EXPORT_ORDER
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) >= 2 and raw[-1:].view("<u4")[0] == MID_MAGIC:
        n = int(raw[-2])
        raw = raw[:len(raw) - 2 - n]
    net = MacroNet(n_act).to(dev)
    sd = net.state_dict()
    at = 0
    for k in EXPORT_ORDER:
        n = sd[k].numel()
        sd[k] = torch.from_numpy(raw[at:at + n].copy()).reshape(sd[k].shape).to(dev)
        at += n
    if at != len(raw):
        raise ValueError(f"{path}: {len(raw)} floats, the export order needs {at}")
    net.load_state_dict(sd)
    return net


INIT_BIAS = 0.0  # --init-bias: logit offset of every new (option) row vs the base row it copies


def expand_net(src, n_act, init_map, dev):
    """src (a MacroNet with <= n_act actions) as an n_act-action net: rows 0..old_n-1 kept, each new row copies the
    row --init-map names (new:old), its bias lowered by INIT_BIAS. Without the offset, k exact copies of a base row
    tie with it: sampling splits the base row's mass k+1 ways (the policy mostly plays options it never chose) and
    greedy argmax lands on an arbitrary copy (PPO3 27 Sep: val combo -0.58 at iteration 10 vs i910's +0.12)."""
    old_n = src.state_dict()["pi.weight"].shape[0]
    if old_n == n_act:
        return src
    rows = {int(k): int(v) for k, v in (x.split(":") for x in init_map.split(","))} if init_map else {}
    net = MacroNet(n_act).to(dev)
    own = net.state_dict()
    for k, v in src.state_dict().items():
        if k in ("pi.weight", "pi.bias"):
            t = own[k].clone()
            t[:old_n] = v
            for new_row in range(old_n, n_act):
                t[new_row] = v[rows.get(new_row, 0)]
                if k == "pi.bias":
                    t[new_row] -= INIT_BIAS
            own[k] = t
        elif k in own and own[k].shape == v.shape:
            own[k] = v
    net.load_state_dict(own)
    return net


def load_any(path, n_act, dev):
    if path.endswith(".bin"):
        return net_from_bin(path, n_act, dev)
    return load_net(n_act, path, dev)[0]


GATE_SHARE = 16  # cores a concurrent gate / tournament / panel job gets while PPO runs (ops/queue.py)


def share(threads):
    """PPO yields GATE_SHARE cores while the queue reports a gate or rc lane job running
    (data/ops/lanes.json, rewritten by the runner every pass); otherwise it takes them all."""
    try:
        lanes = json.load(open(os.path.join(RL, "data", "ops", "lanes.json")))
    except (OSError, ValueError):
        return threads
    # the oracle lane is NOT counted: each oracle batch ends on one thread (the last games' branch replays),
    # so reserving 16 cores for it left ~25% of the box idle (measured 2026-09-26); the kernel shares cores
    # when both are busy
    busy = lanes.get("gate", 0) + lanes.get("rc", 0)
    return max(2, threads - GATE_SHARE * busy)


SHIELD = os.path.join(RL, "configs", "shield", "v1.json")
PROFILES = os.environ.get("KRL_PROFILES") or os.path.join(RL, "configs", "profiles", "rl3.json")  # the action table (KRL_PROFILES: PPO3 = rl4)
TRAIN_TAPES = os.path.join(RL, "data", "tapes", "train")


EXTRA = []  # --rshell / --chain-off / --knob-over for every learner rollout (set in main)
TRAIN_EXTRA = []  # --mix / --seed-bank for training rollouts only (set in main)
TAPE_FRAC = None  # --tape-frac (None = KRL_TAPE_FRAC or 0.3)
TAPE_SHARDS = []  # --tapes-shards DIR: its shard_XX folders, one added per rollout call (rotating)
_SHARD_I = [0]


def start_rollout(weights, prefix, games, seed0, threads, snaps=None, greedy=False, opp=None):
    """Launch ppo-rollout in the background; finish_rollout collects it."""
    threads = share(threads)
    cmd = [ROLL, "--weights", weights, "--out", prefix, "--games", str(games), "--seed0", str(seed0), "--threads", str(threads), "--profiles", PROFILES]
    if os.path.exists(SHIELD):  # the opponent-group shield masks the policy's choice in training and play alike
        cmd += ["--shield", SHIELD]
    if os.path.isdir(TRAIN_TAPES) and not opp:  # real ladder players (a window before the gate's), 30% of games
        tf = str(TAPE_FRAC) if TAPE_FRAC is not None else os.environ.get("KRL_TAPE_FRAC", "0.3")
        pool = TRAIN_TAPES
        if TAPE_SHARDS:
            pool = pool + "," + TAPE_SHARDS[_SHARD_I[0] % len(TAPE_SHARDS)]
            _SHARD_I[0] += 1
        cmd += ["--tapes", pool, "--tape-frac", tf]
    cmd += EXTRA
    if not opp and not greedy:
        cmd += TRAIN_EXTRA
    if snaps:
        cmd += ["--snaps", snaps]
    if greedy:
        cmd += ["--greedy"]
    if opp:
        cmd += ["--opp", opp]
    return subprocess.Popen(cmd, cwd=RL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), prefix


def vplay(agent, prefix, n, seed0, threads, opp=None, tapes=None):
    """One greedy validation rollout; agent = ["--weights", W] (played under the shield) or ["--learner-fixed", K].
    Returns {seed: score}; files are removed."""
    cmd = [ROLL, *agent, "--out", prefix, "--games", str(n), "--seed0", str(seed0), "--threads", str(share(threads)), "--greedy", "--profiles", PROFILES]
    if agent[0] == "--weights" and os.path.exists(SHIELD):
        cmd += ["--shield", SHIELD]
    if agent[0] == "--weights":
        cmd += EXTRA
    if opp:
        cmd += ["--opp", opp]
    if tapes:
        cmd += ["--tapes", tapes, "--tape-frac", "1.0"]
    r = subprocess.run(cmd, cwd=RL, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"validation rollout failed rc {r.returncode}: {r.stderr[-800:]}")
    out = {int(x[0]): float(x[6]) for x in (ln.split("	") for ln in open(prefix + ".tsv", encoding="utf-8").read().splitlines())}
    for f in (prefix + ".tsv", prefix + ".traj"):
        try:
            os.remove(f)
        except OSError:
            pass
    return out


def tape_eval(agent, threads):
    """Every held-out ladder tape once through tapeplay (guarded opponent); agent = ["--policy", W] (under the
    shield) or ["--pa", K]. Returns {tape id: W/D/L score}."""
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    import tempfile
    files = sorted(glob.glob(os.path.join(LADDER_VAL, "**", "*.json"), recursive=True))
    if not files:
        return {}
    n = max(1, min(share(threads), len(files)))
    shards = [files[i::n] for i in range(n)]
    extra = ["--shield", SHIELD] if agent[0] == "--policy" and os.path.exists(SHIELD) else []
    if agent[0] == "--policy":
        extra += EXTRA

    def one(sh):
        d = tempfile.mkdtemp(prefix="ladval-")
        try:
            for f in sh:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--profiles", PROFILES, "--guarded", *agent, *extra], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ladder validation failed rc {r.returncode}: {r.stderr[-500:]}")
            out = {}
            for ln in r.stdout.splitlines():
                x = ln.split("\t")
                if len(x) >= 6:
                    m = float(x[4]) - float(x[5])
                    out[x[0]] = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)

    res = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for r in ex.map(one, shards):
            res.update(r)
    return res


def finish_rollout(proc, prefix):
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ppo-rollout failed rc {proc.returncode}: {err[-800:]}")
    tsv = [l.split("\t") for l in open(prefix + ".tsv", encoding="utf-8").read().splitlines()]
    raw = np.fromfile(prefix + ".traj", dtype="<f4")
    traj = raw.reshape(len(tsv), len(raw) // max(1, len(tsv) * W), W)
    return tsv, traj, err.strip().splitlines()[-1] if err.strip() else ""


def rollout(weights, prefix, games, seed0, threads, snaps=None, greedy=False, opp=None):
    return finish_rollout(*start_rollout(weights, prefix, games, seed0, threads, snaps, greedy, opp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0, help="entropy bonus (0: an entropy bonus erodes the BC anchor)")
    ap.add_argument("--kl0", type=float, default=0.3, help="initial KL-to-teacher coefficient")
    ap.add_argument("--no-pipeline", action="store_true", help="play iteration i+1's games only after update i (the box then "
                    "idles ~40% of each iteration); by default they are played during update i with the pre-update weights, "
                    "and PPO's ratio uses the logp they were actually sampled with")
    ap.add_argument("--torch-threads", type=int, default=16, help="learner threads (4 left the box ~95% idle during updates)")
    ap.add_argument("--policy-days", default="0-29", help="days whose choice trains the policy (pg, entropy, KL), e.g. "
                    "'6-29': a day whose profile never changes the game (before the shops unlock; measured by the branch "
                    "oracle over all 30 days) only adds noise to the gradient and spends the KL budget. The value head "
                    "still learns on every day")
    ap.add_argument("--oracle-steps", type=int, default=8, help="supervised steps per iteration on branch-oracle labels (0 = off)")
    ap.add_argument("--oracle-coef", type=float, default=1.0)
    ap.add_argument("--oracle-min-gain", type=float, default=0.01, help="train only on decisions where the best branched "
                    "profile beat the policy's own choice by >= this much Phi(margin/sigma) (0.01 ~ $75 near an even game). "
                    "Measured 2026-09-26: the median gain was 0.0004 (~$3) with ~5 candidates tied, and those near-tie rows "
                    "spread the policy (entropy 2.0 -> 2.4) away from v63 for nothing")
    ap.add_argument("--oracle-beta", type=float, default=8.0, help="target sharpness: exp(beta * z), z = 0 for the best allowed "
                    "candidate and -1 for the worst (Phi(margin/sigma) scaled by the decision's own spread)")
    ap.add_argument("--oracle-files", type=int, default=30, help="newest oracle batches to train on")
    ap.add_argument("--anchor-prior", default="35:0.5,19:0.1,31:0.1,34:0.1",
                    help="KL anchor: a fixed prior over profiles (id:mass, the rest spread evenly) instead of the BC "
                         "teacher. Against v62.1 only profiles 19 and 31 hold (0.505; every other fixed profile ~0.04), "
                         "while the BC teacher is near uniform, so anchoring to it pinned the policy to losing mixes. "
                         "'' = anchor to the BC teacher")
    ap.add_argument("--val-opp", default="fixed:35", help="validation opponent: fixed:35 = v63 (rl3), the agent to beat "
                    "(v611 saturated at 0.99, so best/rollback could not see progress)")
    ap.add_argument("--kl-target", type=float, default=0.3, help="adaptive KL leash: the coefficient is raised when the "
                    "measured KL to the BC teacher exceeds 1.5x this and lowered below 1/1.5x")
    ap.add_argument("--sigma", type=float, default=5000.0, help="margin scale of the reward's Phi(margin / sigma) term")
    ap.add_argument("--phi-mix", type=float, default=0.5, help="reward = (1-m) * W/D/L score + m * Phi(margin / sigma)")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10**9)
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--init-from", default=None, help="new run: warm-start from this PPO ckpt.pt (a smaller action set is "
                    "expanded: new profiles start from --init-map rows)")
    ap.add_argument("--init-map", default="32:0,33:19,34:19,35:19", help="new-profile row <- existing row (rl3: p0_v92 <- 0, v92 variants of 19 and v63 <- 19)")
    ap.add_argument("--root", default=None, help="run root (default weights/ppo); the smoke check uses a scratch root")
    ap.add_argument("--val-n", type=int, default=VAL_N)
    ap.add_argument("--mid-days", default="", help="ppo2: days with a second (hour-13) decision, e.g. 25-29 (new runs; a resumed run keeps its own)")
    ap.add_argument("--init-bias", type=float, default=0.0, help="logit offset of new option rows below their base row (warm start AND anchor), e.g. 4")
    ap.add_argument("--init-n", type=int, default=0, help="action count of a .bin --init-from when it differs from the table (e.g. 36)")
    ap.add_argument("--anchor-n", type=int, default=0, help="action count of --anchor-net when it differs from the table (e.g. 36)")
    ap.add_argument("--anchor-net", default=None, help="ppo2: KL anchor = this policy (.bin export or .pt ckpt), e.g. the best checkpoint i790")
    ap.add_argument("--lr-reset", type=float, default=0.0, help="resume: replace the stored base lr and restart the half-life schedule here (0 = keep)")
    ap.add_argument("--lr-half-life", type=float, default=0.0, help="ppo2: lr halves every N iterations from the run start (0 = constant)")
    ap.add_argument("--mix", default=None, help="ppo2: rollout opponent mix, e.g. mirror:25,snap:15,v611:10,fixed:20,v63:10,rand:20")
    ap.add_argument("--seed-bank", default=None, help="ppo2: 64-world seed bank (data/worlds/w64_bank.json) for non-tape training games")
    ap.add_argument("--tapes-dir", default=None, help="real-player tape pool for training games (default data/tapes/train); 28 Sep: data/tapes/leaders = 2500+ players + our real losses to them")
    ap.add_argument("--tapes-shards", default=None, help="folder of shard_XX tape folders (python/gm_top_tapes.py: every GM game with a 2500+ player); one shard joins --tapes-dir per rollout, rotating")
    ap.add_argument("--tape-frac", type=float, default=None, help="share of training games against real players' tapes (default 0.3)")
    ap.add_argument("--shell", default=None, help="v1 learned sales shell for the learner (the shipped agent plays with big1)")
    ap.add_argument("--rshell", default=None, help="reactive shell v2 config for the learner (crates/agent/src/rshell.rs); off by default")
    ap.add_argument("--chain-off", default=None, help="whole-game chain stages off for the learner (knobs::stage_bits names)")
    ap.add_argument("--knob-over", default=None, help="knob overrides on every profile for the learner (JSON file)")
    ap.add_argument("--oracle-dir", default=None, help="branch-oracle label dir (default data/oracle)")
    ap.add_argument("--seed-snaps", default="", help="new run: copy these policy exports into snapshots/ as lineage_*.bin (older PPO checkpoints as opponents)")
    a = ap.parse_args()
    globals()["INIT_BIAS"] = a.init_bias
    global MID, N_SLOTS, EXTRA, TRAIN_EXTRA, TAPE_FRAC, ORACLE, TRAIN_TAPES
    for k, v in (("--shell", a.shell), ("--rshell", a.rshell), ("--chain-off", a.chain_off), ("--knob-over", a.knob_over)):
        if v:
            EXTRA += [k, v]
    if a.mix:
        TRAIN_EXTRA += ["--mix", a.mix]
    if a.seed_bank:
        TRAIN_EXTRA += ["--seed-bank", a.seed_bank]
    TAPE_FRAC = a.tape_frac
    if a.tapes_shards:
        TAPE_SHARDS[:] = sorted(glob.glob(os.path.join(os.path.abspath(a.tapes_shards), "shard_*")))
        print(f"[ppo] top-player tape shards: {len(TAPE_SHARDS)} under {a.tapes_shards}", flush=True)
    if a.tapes_dir:
        TRAIN_TAPES = os.path.abspath(a.tapes_dir)
    if a.oracle_dir:
        ORACLE = a.oracle_dir
    torch.set_num_threads(a.torch_threads)
    global PPO
    if a.root:
        PPO = a.root
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_act = len(profile_names())

    # ---- run dir: resume the unfinished run, else start from the newest BC teacher
    # only a run with this action count can resume (the rl3 table has 36 profiles; older runs had 32)
    runs = sorted(d for d in glob.glob(os.path.join(PPO, "ppo-*")) if os.path.exists(os.path.join(d, "ckpt.pt"))
                  and not os.path.exists(os.path.join(d, "STOPPED")) and f"-p{n_act}-" in os.path.basename(d))
    teacher_id = bc_latest() or "none"
    if not a.anchor_prior and teacher_id == "none":
        sys.exit("no BC teacher (weights/bc/LATEST) and no --anchor-prior")
    if runs and not a.new:
        run = runs[-1]
        net, s = load_net(n_act, os.path.join(run, "ckpt.pt"), dev)
        it, lr, best, drops = s["iter"], s["lr"], s["best"], s["drops"]
        if best is not None and not isinstance(best, (int, float)):
            # 2026-09-26: a regret-metric local once shadowed `best` with a tensor and it was checkpointed
            print("[ppo] checkpoint best was not a number: reset", flush=True)
            best, drops = None, 0
        kl_c = s.get("kl_coef", a.kl0)
        # a new validation scale (opponent or harness fix): the old best is not comparable. VAL_VERSION 2:
        # fixed-profile opponents no longer collapse to profile 0 under the shield (2026-09-26)
        if (s.get("val_opp", "v611"), s.get("val_ver", 1)) != (a.val_opp, VAL_VERSION):
            print(f"[ppo] validation {s.get('val_opp', 'v611')} v{s.get('val_ver', 1)} -> {a.val_opp} v{VAL_VERSION}: best and drops reset", flush=True)
            best, drops = None, 0
        teacher_id = s.get("teacher", teacher_id)
        MID = list(s.get("mid", []))
        it0, base_lr = s.get("it0", 0), s.get("base_lr", lr)
        if a.lr_reset and base_lr > a.lr_reset * 1.0001:  # operator correction on a resumed run (applied once): new base lr, schedule restarts here
            it0, base_lr = it, a.lr_reset
            print(f"[ppo] lr reset to {base_lr:g} at iteration {it}", flush=True)
        print(f"[ppo] resumed {os.path.basename(run)} at iteration {it}" + (f" (mid-day decisions on days {MID})" if MID else ""), flush=True)
    else:
        src = "init" if a.init_from else teacher_id
        run = os.path.join(PPO, f"ppo-gru64-f2-p{n_act}-from-{src}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%MZ}")
        os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
        MID = parse_days(a.mid_days)
        if a.init_from and a.init_from.endswith(".bin"):
            src_n = a.init_n or n_act
            net = expand_net(net_from_bin(a.init_from, src_n, dev), n_act, a.init_map, dev)
            print(f"[ppo] new run warm-started from export {a.init_from} ({src_n} -> {n_act} actions, map {a.init_map})", flush=True)
        elif a.init_from:
            # warm start from an existing PPO checkpoint; a smaller action set is expanded row by row
            sd = torch.load(a.init_from, map_location=dev, weights_only=False)["model"]
            net = MacroNet(n_act).to(dev)
            own = net.state_dict()
            old_n = sd["pi.weight"].shape[0]
            rows = {int(k): int(v) for k, v in (x.split(":") for x in a.init_map.split(","))}
            for k, v in sd.items():
                if k in ("pi.weight", "pi.bias") and v.shape[0] != n_act:
                    t = own[k].clone()
                    t[:old_n] = v
                    for new_row, src_row in rows.items():
                        if old_n <= new_row < n_act:
                            t[new_row] = v[src_row] - (INIT_BIAS if k == "pi.bias" else 0.0)
                    own[k] = t
                elif k in own and own[k].shape == v.shape:
                    own[k] = v
            net.load_state_dict(own)
            print(f"[ppo] new run warm-started from {a.init_from} ({old_n} -> {n_act} actions, rows {rows})", flush=True)
        else:
            net, _ = load_net(n_act, os.path.join(BC, teacher_id, "ckpt.pt"), dev)
        s, it, lr, best, drops = None, 0, a.lr, None, 0
        it0, base_lr = 0, a.lr
        kl_c = a.kl0
    os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(run, "rollouts"), exist_ok=True)
    if s is None and a.seed_snaps:
        import shutil
        for f in [x for x in a.seed_snaps.split(",") if x.strip()]:
            shutil.copy(f, os.path.join(run, "snapshots", "lineage_" + os.path.basename(os.path.dirname(os.path.dirname(f)))[-14:] + "_" + os.path.basename(f)))
    N_SLOTS = 30 + len([d for d in MID if d < 30])
    SLOT_DAYS = slot_days(MID)
    # the BC teacher is only the KL anchor when no --anchor-prior is given; --anchor-net overrides both
    teacher = None
    if a.anchor_net:
        # an anchor with fewer actions (i790 plays rl3's 36) is expanded like the warm start: option rows copy their base
        teacher = expand_net(load_any(a.anchor_net, a.anchor_n or n_act, dev), n_act, a.init_map, dev)
        teacher.eval()
        teacher_id = "net:" + os.path.basename(os.path.dirname(os.path.dirname(a.anchor_net))) + "/" + os.path.basename(a.anchor_net)
        a.anchor_prior = ""
        print(f"[ppo] KL anchor = {a.anchor_net}", flush=True)
    elif not a.anchor_prior:
        teacher, _ = load_net(n_act, os.path.join(BC, teacher_id, "ckpt.pt"), dev)
        teacher.eval()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if s is not None and "opt" in s:
        opt.load_state_dict(s["opt"])
        for g in opt.param_groups:
            g["lr"] = lr
    ck = os.path.join(run, "ckpt.pt")

    def save():
        tmp = ck + ".tmp"
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it, "lr": lr, "best": best,
                    "drops": drops, "teacher": teacher_id, "kl_coef": kl_c, "val_opp": a.val_opp, "val_ver": VAL_VERSION,
                    "mid": MID, "it0": it0, "base_lr": base_lr}, tmp)
        os.replace(tmp, ck)

    def validate():
        p = os.path.join(run, "rollouts", "val")
        export_bin(net, os.path.join(run, "val.bin"))
        tsv, _, line = rollout(os.path.join(run, "val.bin"), p, a.val_n, VAL_SEED0, a.threads, greedy=True, opp=a.val_opp)
        sc = [float(r[6]) for r in tsv]
        # W/D/L: vs v63 most greedy games are exact mirrors (draws), so the mean alone hides wins vs losses
        wdl_ = f"W {sum(x == 1 for x in sc)} D {sum(x == 0.5 for x in sc)} L {sum(x == 0 for x in sc)}; "
        # the panel and the hard tapes, paired against v63 (profile 35) on the same seeds; the reference is
        # deterministic, so it is played once per validation version and cached in the run
        refp = os.path.join(run, f"val_ref_v{VAL_VERSION}.json")
        ref = json.load(open(refp)) if os.path.exists(refp) else None
        cand = {"panel": {}, "hard": {}}
        new_ref = {"panel": {}, "hard": {}}
        for opp in VAL_PANEL:
            cand["panel"][opp] = vplay(["--weights", os.path.join(run, "val.bin")], p + "_p", VAL_PANEL_N, VAL_PANEL_SEED0, a.threads, opp=opp)
            if ref is None:
                new_ref["panel"][opp] = vplay(["--learner-fixed", "35"], p + "_r", VAL_PANEL_N, VAL_PANEL_SEED0, a.threads, opp=opp)
        if os.path.isdir(HARD_TAPES):
            cand["hard"] = vplay(["--weights", os.path.join(run, "val.bin")], p + "_h", VAL_HARD_N, VAL_HARD_SEED0, a.threads, tapes=HARD_TAPES)
            if ref is None:
                new_ref["hard"] = vplay(["--learner-fixed", "35"], p + "_r", VAL_HARD_N, VAL_HARD_SEED0, a.threads, tapes=HARD_TAPES)
        cand["ladder"] = tape_eval(["--policy", os.path.join(run, "val.bin")], a.threads)
        if ref is None:
            new_ref["ladder"] = tape_eval(["--pa", "35"], a.threads)
        if ref is None:
            ref = {"panel": {o: {str(k): v for k, v in d.items()} for o, d in new_ref["panel"].items()}, "hard": {str(k): v for k, v in new_ref["hard"].items()},
                   "ladder": new_ref["ladder"]}
            json.dump(ref, open(refp, "w"))

        def delta(c, r):
            ks = [k for k in c if str(k) in r]
            b = sum(c[k] > r[str(k)] for k in ks)
            w = sum(c[k] < r[str(k)] for k in ks)
            return (float(np.mean([c[k] - r[str(k)] for k in ks])) if ks else 0.0), b, w
        pd_ = [delta(cand["panel"][o], ref["panel"][o]) for o in VAL_PANEL]
        d_panel = float(np.mean([x[0] for x in pd_]))
        pb, pw = sum(x[1] for x in pd_), sum(x[2] for x in pd_)
        d_hard, hb, hw = delta(cand["hard"], ref["hard"]) if cand["hard"] else (0.0, 0, 0)
        d_lad, lb, lw = delta(cand["ladder"], ref.get("ladder", {})) if cand["ladder"] else (0.0, 0, 0)
        combo = (float(np.mean(sc)) - 0.5) + d_panel + d_hard + LADDER_W * d_lad
        extra = {"val_panel_d": d_panel, "val_panel_bw": [pb, pw], "val_hard_d": d_hard, "val_hard_bw": [hb, hw],
                 "val_ladder_d": d_lad, "val_ladder_bw": [lb, lw], "val_ladder_n": len(cand["ladder"]), "val_combo": combo}
        return float(np.mean(sc)), wdl_ + line, extra

    pending = None
    while it < a.iters:
        # teacher hot-swap
        tl = bc_latest()
        if tl and tl != teacher_id and not a.anchor_prior and not a.anchor_net:
            teacher, _ = load_net(n_act, os.path.join(BC, tl, "ckpt.pt"), dev)
            teacher.eval()
            print(f"[ppo] teacher hot-swap {teacher_id} -> {tl}", flush=True)
            teacher_id = tl
        if a.lr_half_life > 0:
            lr = base_lr * 0.5 ** ((it - it0) / a.lr_half_life)
            for g in opt.param_groups:
                g["lr"] = lr
        t_it = time.time()
        cur = os.path.join(run, "current.bin")
        export_bin(net, cur)
        prefix = os.path.join(run, "rollouts", f"i{it:06d}")
        if pending is not None and pending[1] == prefix:
            tsv, traj, line = finish_rollout(*pending)
        else:
            if pending is not None:  # a stale background rollout (resume / rollback): discard it
                pending[0].kill()
            tsv, traj, line = rollout(cur, prefix, a.games, 20_000_000 + it * a.games, a.threads, snaps=os.path.join(run, "snapshots"))
        t_roll = time.time() - t_it
        pending = None
        if not a.no_pipeline:
            # iteration it+1's games start now with the pre-update weights (alternating files: the oracle and the
            # next export must not overwrite what this rollout is reading); they run during the update below
            wnext = os.path.join(run, f"behavior{(it + 1) % 2}.bin")
            export_bin(net, wnext)
            pending = start_rollout(wnext, os.path.join(run, "rollouts", f"i{it + 1:06d}"), a.games, 20_000_000 + (it + 1) * a.games,
                                    a.threads, snaps=os.path.join(run, "snapshots"))  # all cores: the update overlaps only ~9 s
        obs = torch.from_numpy(traj[:, :, :NF].copy()).to(dev)
        act = torch.from_numpy(traj[:, :, NF].astype(np.int64)).to(dev)
        lp_old = torch.from_numpy(traj[:, :, NF + 1].copy()).to(dev)
        # the shield's allowed-profile mask each choice was sampled under; the ratio must use the same
        # masked softmax, or a restricted day looks like a large policy change
        bits = mask_bits(traj)
        n_act = net.n_act if hasattr(net, "n_act") else int(net(obs[:1])["pi"].shape[-1])
        allowed = torch.from_numpy(((bits[..., None] >> np.arange(n_act)) & 1).astype(bool)).to(dev)
        allowed |= ~allowed.any(-1, keepdim=True)  # days never reached (mask 0): leave unmasked
        mask = (act >= 0).float()
        act = act.clamp(min=0)
        T = traj.shape[1]
        pdays = set(parse_days(a.policy_days))
        sd_ = SLOT_DAYS if T == len(SLOT_DAYS) else list(range(T))
        dsel = torch.tensor([1.0 if sd_[t] in pdays else 0.0 for t in range(T)], device=dev)
        pmask = mask * dsel[None, :]  # the policy terms train only on --policy-days
        wdl = torch.tensor([float(r[6]) for r in tsv], device=dev)
        margin = torch.tensor([float(r[4]) - float(r[5]) for r in tsv], device=dev)
        # reward: the W/D/L score blended with Phi(margin / sigma) -- the win currency made smooth, so a
        # near-miss loss is better than a rout and the gradient is not flat on lopsided worlds
        phi = 0.5 * (1 + torch.erf(margin / (a.sigma * 2 ** 0.5)))
        score = (1 - a.phi_mix) * wdl + a.phi_mix * phi
        with torch.no_grad():
            v = torch.sigmoid(net(obs, detach_v=True)["v"]) * mask
            last = (mask.sum(1) - 1).long().clamp(min=0)
            adv = torch.zeros_like(v)
            g = torch.zeros(len(tsv), device=dev)
            for t in range(T - 1, -1, -1):
                nxt = v[:, t + 1] if t < T - 1 else torch.zeros_like(g)
                r = torch.where(last == t, score, torch.zeros_like(score))
                nv = torch.where(last == t, torch.zeros_like(nxt), nxt)
                delta = (r + nv - v[:, t]) * mask[:, t]
                g = delta + 0.95 * g * mask[:, t]
                adv[:, t] = g
            ret = adv + v
            an = (adv - (adv * pmask).sum() / pmask.sum()) / (adv[pmask > 0].std() + 1e-6) * pmask
            if a.anchor_prior:
                pri = torch.full((n_act,), 0.0, device=dev)
                named = {int(k): float(v) for k, v in (x.split(":") for x in a.anchor_prior.split(","))}
                rest = max(1e-6, 1.0 - sum(named.values())) / max(1, n_act - len(named))
                for k in range(n_act):
                    pri[k] = named.get(k, rest)
                tl = torch.log(pri)[None, None].expand(obs.shape[0], obs.shape[1], n_act)
                tpi = F.log_softmax(tl.masked_fill(~allowed, -1e9), -1)
            else:
                tpi = F.log_softmax(teacher(obs)["pi"].masked_fill(~allowed, -1e9), -1)
        stats = []
        for _ in range(a.epochs):
            perm = torch.randperm(len(tsv), device=dev)
            for i in range(0, len(tsv), a.mb):
                b = perm[i:i + a.mb]
                o = net(obs[b], detach_v=True)
                logp_all = F.log_softmax(o["pi"].masked_fill(~allowed[b], -1e9), -1)
                lp = logp_all.gather(-1, act[b][..., None]).squeeze(-1)
                ratio = torch.exp(lp - lp_old[b])
                m, pm_ = mask[b], pmask[b]
                pg = -(torch.min(ratio * an[b], ratio.clamp(1 - a.clip, 1 + a.clip) * an[b]) * pm_).sum() / pm_.sum()
                vl = (((torch.sigmoid(o["v"]) - ret[b]) ** 2) * m).sum() / m.sum()
                p = logp_all.exp()
                ent = (-(p * logp_all).sum(-1) * pm_).sum() / pm_.sum()
                kl = ((p * (logp_all - tpi[b])).sum(-1) * pm_).sum() / pm_.sum()
                loss = pg + 0.5 * vl - a.ent * ent + kl_c * kl
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss; resume reloads the last checkpoint")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append([float(pg), float(vl), float(ent), float(kl)])
        t_upd = time.time() - t_it - t_roll
        # expert iteration: supervised steps toward the branch oracle's best profile on decisions where
        # the choice changed the outcome (plain PPO's signal is flat on the ~84% where it does not)
        ost = {}
        orc = oracle_rows(a.oracle_files, a.sigma, a.oracle_beta, n_act, a.oracle_min_gain) if a.oracle_steps > 0 else None
        if orc is not None:
            o_obs, o_day, o_tgt, o_allow, o_wt, o_ph = orc
            ces, accs = [], []
            for _ in range(a.oracle_steps):
                b = np.random.randint(0, len(o_day), size=min(256, len(o_day)))
                x = torch.from_numpy(o_obs[b]).to(dev)
                d_ = torch.from_numpy(o_day[b]).to(dev)
                pi = net(x)["pi"][torch.arange(len(b), device=dev), d_]
                al = torch.from_numpy(o_allow[b]).to(dev)
                logp = F.log_softmax(pi.masked_fill(~al, -1e9), -1)
                tg = torch.from_numpy(o_tgt[b]).to(dev)
                w_ = torch.from_numpy(o_wt[b]).to(dev) + 0.002  # gains are ~0.006 in Phi: a 0.05 floor made trivial $10 rows weigh as much as flips
                ce = (-(tg * logp).sum(-1) * w_).sum() / w_.sum()
                opt.zero_grad()
                (a.oracle_coef * ce).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                ces.append(float(ce))
                accs.append(float((logp.argmax(-1) == tg.argmax(-1)).float().mean()))
            ost = {"oracle_rows": int(len(o_day)), "oracle_ce": float(np.mean(ces)), "oracle_acc": float(np.mean(accs))}
            # regret (win-probability points x100) on up to 2,048 labelled decisions: how much Phi the policy gives up
            # against the oracle's best branched profile -- sampled (expected) and greedy -- and what always-v63 gives up.
            # Exact-best accuracy undercounts: many candidates tie.
            with torch.no_grad():
                b = np.random.RandomState(it).randint(0, len(o_day), size=min(2048, len(o_day)))
                pi = net(torch.from_numpy(o_obs[b]).to(dev))["pi"][torch.arange(len(b), device=dev), torch.from_numpy(o_day[b]).to(dev)]
                ph = torch.from_numpy(o_ph[b]).to(dev)
                have = ~torch.isnan(ph)
                pr = F.softmax(pi.masked_fill(~have, -1e9), -1)
                phz = torch.nan_to_num(ph, nan=0.0)
                obest = phz.masked_fill(~have, -1.0).max(-1).values
                ost["oracle_regret"] = float(100 * (obest - (pr * phz).sum(-1)).mean())
                ost["oracle_regret_greedy"] = float(100 * (obest - phz.gather(-1, pr.argmax(-1, keepdim=True)).squeeze(-1)).mean())
                h35 = have[:, 35] if ph.shape[1] > 35 else None
                if h35 is not None and bool(h35.any()):
                    ost["oracle_regret_v63"] = float(100 * (obest[h35] - phz[h35, 35]).mean())
                    ost["oracle_regret_greedy_on_v63rows"] = float(100 * (obest[h35] - phz[h35].gather(-1, pr[h35].argmax(-1, keepdim=True)).squeeze(-1)).mean())
        t_orc = time.time() - t_it - t_roll - t_upd
        it += 1
        st = np.mean(stats, 0)
        # adaptive KL leash to the BC anchor (replaces a coefficient that decayed to 0)
        kl_used = kl_c
        if st[3] > 1.5 * a.kl_target:
            kl_c = min(kl_c * 1.5, 50.0)
        elif st[3] < a.kl_target / 1.5:
            kl_c = max(kl_c / 1.5, 1e-3)
        with torch.no_grad():  # how much of the policy still sits on the anchor's choices
            pn = F.softmax(net(obs)["pi"].masked_fill(~allowed, -1e9), -1)
            m = mask > 0
            mass_p0 = float(pn[..., 0][m].mean())
            mass_teacher = float(pn.gather(-1, tpi.argmax(-1, keepdim=True)).squeeze(-1)[m].mean())
            mass_all = [round(float(x), 4) for x in pn[m].mean(0)]  # the policy's mean mass per profile (dashboard)
            played = np.bincount(traj[:, :, NF][traj[:, :, NF] >= 0].astype(np.int64), minlength=n_act)
            played = [round(float(x), 4) for x in played / max(1, played.sum())]  # what the rollouts actually chose
        rec = {"iter": it, "t": now(), "games": len(tsv), "sec_rollout_wait": round(t_roll, 1), "sec_update": round(t_upd, 1), "sec_oracle": round(t_orc, 1), "win": float(wdl.mean()), "reward": float(score.mean()),
               "mass_p0": mass_p0, "mass_teacher_argmax": mass_teacher, "mass": mass_all, "played": played, "kl_target": a.kl_target, "pg": st[0], "vl": st[1],
               "ent": st[2], **ost, "kl_teacher": st[3], "kl_coef": kl_used, "kl_coef_next": kl_c, "lr": lr, "teacher": teacher_id, "rollout": line}
        if it % a.val_every == 0:
            wv_h2h, vline, vx = validate()
            rec.update(val_win=wv_h2h, val=vline, **vx)
            wv = vx["val_combo"]  # best and rollback follow the composite (VAL_VERSION 3)
            export_bin(net, os.path.join(run, "snapshots", f"i{it:06d}.bin"))
            if best is None or wv > best:
                best, drops = wv, 0
                export_bin(net, os.path.join(run, "best.bin"))
                torch.save({"model": net.state_dict(), "iter": it}, os.path.join(run, "best.pt"))
            elif wv < best - 0.05:
                drops += 1
                if drops >= 2:
                    net.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=dev)["model"])
                    if pending is not None:
                        pending[0].kill()
                        pending = None
                    lr /= 2
                    base_lr /= 2
                    for g in opt.param_groups:
                        g["lr"] = lr
                    drops = 0
                    rec["rollback"] = f"to best ({best:.3f}), lr {lr:g}"
            else:
                drops = 0
        save()
        with open(os.path.join(run, "metrics.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        open(os.path.join(run, "HEARTBEAT"), "w").write(rec["t"])
        for f in glob.glob(prefix + ".*"):
            os.remove(f)
        print(f"[ppo] iter {it}: win {rec['win']:.3f}" + (f" val h2h {rec['val_win']:.3f} panel {rec['val_panel_d']:+.3f} hard {rec['val_hard_d']:+.3f} ladder {rec.get('val_ladder_d', 0):+.3f} {rec.get('val_ladder_bw')} combo {rec['val_combo']:+.3f} (best {best:+.3f})" if "val_win" in rec else "")
              + f" pg {st[0]:.4f} v {st[1]:.4f} ent {st[2]:.3f} kl {st[3]:.4f}", flush=True)


if __name__ == "__main__":
    main()
