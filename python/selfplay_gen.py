"""Interim league generator (until the Rust runner, task P4.1, replaces it).

Plays games on the Rust `kagg serve` engine through the repo's faithful harness (same
S3/S4/S5 call rules and last-callable loading as the gates), writes each game as a
Kaggle-format replay JSON, converts every finished batch with `corpus-extract --mode slim
--replays`, and deletes the raw JSON of that batch. Resumable: a batch whose slim output is in
the league's ledger is never replayed; the plan is deterministic given --seed.

    python python/selfplay_gen.py --league cross --games 3000 --workers 4
    python python/selfplay_gen.py --league clone --games 600 --workers 4

Output: data/slim/s1/source=league/league=<name>/ (corpus-extract's own resumable layout:
slim/date=.../part-<tag>.parquet + ledger/ + schema.json). TeamNames in each replay are the
agent names, so league games carry ground-truth identity labels.
"""
import argparse, contextlib, glob, io, json, os, random, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = RL  # one repo since the 2026-10-01 merge
sys.path.insert(0, os.path.join(REPO, "src"))
EXE = os.path.join(RL, "target", "release", "corpus-extract.exe")
OURS = {"v61.1": os.path.join(REPO, "agents", "v61.1_bandit.py"), "v61": os.path.join(REPO, "agents", "v61_bandit.py")}
FIELD = os.path.join(REPO, "data", "winplan", "field")
CONFIG = json.load(open(os.path.join(REPO, "data", "winplan", "ladder_config.json")))


def roster():
    r = dict(OURS)
    for f in sorted(os.listdir(FIELD)):
        if f.endswith(".py"):
            r[f[:-3]] = os.path.join(FIELD, f)
    return r


def play(task):
    """One game -> replay JSON file. Returns (path, banks) or (None, error)."""
    eid, a_name, a_path, b_name, b_path, seed, outdir = task
    from kaggriculture.bandit.gate import loss_forensics as LF, harness as H
    from kaggriculture.engine import serve_match as SM
    from kaggriculture.trackp.serve_env import ServeEnv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            A, B = LF.load_pyagent(a_path), LF.load_pyagent(b_path)
            E = ServeEnv(kagg_path=H.FRESH_KAGG)
            obs = E.reset(seed)
            steps = []
            prev = [{"farmer": ["PASS"], "hands": [], "market": []}] * 2
            while True:
                o0, o1 = SM.obs_for(0, obs), SM.obs_for(1, obs)
                steps.append([{"action": prev[0], "observation": json.loads(json.dumps(o0, default=lambda s: s.__dict__))},
                              {"action": prev[1], "observation": json.loads(json.dumps(o1, default=lambda s: s.__dict__))}])
                if obs.get("done") or obs["step"] >= SM.EPISODE_STEPS - 1:
                    break
                a0 = SM._call(A, SM.obs_for(0, obs)) or {"farmer": ["PASS"], "hands": [], "market": []}
                a1 = SM._call(B, SM.obs_for(1, obs)) or {"farmer": ["PASS"], "hands": [], "market": []}
                prev = [a0, a1]
                obs = E.step_both(a0, a1)
            E.close()
        banks = [float(obs["farms"][0]["money"]), float(obs["farms"][1]["money"])]
        rep = {"id": str(eid), "module_version": "1.32.7", "configuration": CONFIG,
               "info": {"EpisodeId": eid, "TeamNames": [a_name, b_name], "seed": seed},
               "rewards": banks, "statuses": ["DONE", "DONE"], "steps": steps}
        path = os.path.join(outdir, f"{eid}.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(rep, fh, separators=(",", ":"), default=lambda s: s.__dict__)
        os.replace(path + ".tmp", path)
        return path, banks
    except Exception as exc:                                         # noqa: BLE001
        return None, f"{a_name} vs {b_name} seed {seed}: {type(exc).__name__}: {str(exc)[:200]}"


def plan(league, n, seed0):
    rng = random.Random(seed0)
    names = sorted(roster())
    from kaggriculture.bandit.gate import harness as H
    worlds = [s for _, s in H.world_seeds(48)]
    out = []
    for i in range(n):
        seed = rng.choice(worlds) if i % 2 == 0 else rng.randrange(1, 2**31 - 1)
        if league == "clone":
            a = b = names[i % len(names)]
        else:
            a = rng.choice(["v61.1", "v61"]) if i % 3 == 0 else rng.choice(names)
            b = rng.choice(names)
            if rng.random() < 0.5:
                a, b = b, a
        out.append((a, b, seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["cross", "clone"], required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    ros = roster()
    out_root = os.path.join(RL, "data", "slim", "s1", "source=league", f"league={a.league}")
    tmp = os.path.join(RL, "data", "leagues", a.league, "incoming")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)
    done_path = os.path.join(RL, "data", "leagues", a.league, "done_batches.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    games = plan(a.league, a.games, a.seed)
    base = 9_000_000_000 if a.league == "cross" else 9_500_000_000
    for bi in range(0, len(games), a.batch):
        if bi in done:
            continue
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)
        tasks = [(base + a.seed % 1000 * 1_000_000 + bi + j, x, ros[x], y, ros[y], s, tmp)
                 for j, (x, y, s) in enumerate(games[bi:bi + a.batch])]
        t0 = time.time()
        ok = err = 0
        with ProcessPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(play, t) for t in tasks]):
                p, info = fut.result()
                if p:
                    ok += 1
                else:
                    err += 1
                    print("GAME-ERROR", info, flush=True)
        r = subprocess.run([EXE, "--mode", "slim", "--replays", tmp, "--out", out_root, "--threads", "2"],
                           capture_output=True, text=True)
        tail = [l for l in r.stderr.splitlines() if "done:" in l]
        if r.returncode != 0:
            print("EXTRACT-ERROR", r.stderr[-500:], flush=True)
            sys.exit(1)
        done.add(bi)
        json.dump(sorted(done), open(done_path, "w"))
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)
        print(f"BATCH {a.league} {bi // a.batch + 1}/{(len(games) + a.batch - 1) // a.batch} games ok {ok} err {err} "
              f"{time.time() - t0:.0f}s | {tail[-1].strip() if tail else ''}", flush=True)
    print(f"LEAGUE-DONE {a.league} {len(done)} batches", flush=True)


if __name__ == "__main__":
    main()
