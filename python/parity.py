"""Rust-port parity harness (tasks P2.3 / P2.5).

record: play the FULL Python v61.1 against roster opponents on the Rust serve engine and store
        v61.1's own-seat observations, one JSON per line, gzip:
        data/recordings/r1/v61.1/<RUNID>/<game>.jsonl.gz   (+ manifest.json)
check:  open-loop parity. For each recorded game, feed the observations to a truncated Python
        reference (python/ref/truncate.py cut) and to the Rust agent (agent-stdio), compare the
        actions step by step, report the first divergence per game and latency.

    python python/parity.py record --games 40 --workers 4
    python python/parity.py check --cut chassis [--run RUNID] [--games N]
"""
import argparse, contextlib, glob, gzip, io, json, os, random, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = RL  # one repo since the 2026-10-01 merge
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(RL, "python"))
OURS = os.path.join(REPO, "agents", "v61.1_bandit.py")
FIELD = os.path.join(REPO, "data", "winplan", "field")
REC = os.path.join(RL, "data", "recordings", "r1", "v61.1")
AGENT = os.environ.get("AGENT_EXE") or os.path.join(RL, "target-dev", "release", "agent-stdio.exe")
BASE = os.path.join(RL, "configs", "bases", "v61.1")


def _plain(o):
    return json.loads(json.dumps(o, default=lambda s: s.__dict__))


def play(task):
    gid, opp_name, opp_path, seed, seat, outdir = task
    from kaggriculture.bandit.gate import loss_forensics as LF, harness as H
    from kaggriculture.engine import serve_match as SM
    from kaggriculture.trackp.serve_env import ServeEnv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            me, opp = LF.load_pyagent(OURS), LF.load_pyagent(opp_path)
            ags = [me, opp] if seat == 0 else [opp, me]
            E = ServeEnv(kagg_path=H.FRESH_KAGG)
            obs = E.reset(seed)
            lines = []
            while not (obs.get("done") or obs["step"] >= SM.EPISODE_STEPS - 1):
                acts = []
                for s in (0, 1):
                    o = SM.obs_for(s, obs)
                    if s == seat:
                        lines.append(json.dumps(_plain(o), separators=(",", ":")))
                    acts.append(SM._call(ags[s], SM.obs_for(s, obs)) or {"farmer": ["PASS"], "hands": [], "market": []})
                obs = E.step_both(acts[0], acts[1])
            E.close()
        banks = [float(obs["farms"][0]["money"]), float(obs["farms"][1]["money"])]
        path = os.path.join(outdir, f"{gid}.jsonl.gz")
        with gzip.open(path + ".tmp", "wt") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(path + ".tmp", path)
        return gid, {"opp": opp_name, "seed": seed, "seat": seat, "banks": banks, "steps": len(lines)}
    except Exception as exc:                                                    # noqa: BLE001
        return gid, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def record(a):
    runid = time.strftime("%Y%m%dT%H%MZ", time.gmtime()) + f"-s{a.seed}"
    out = os.path.join(REC, runid)
    os.makedirs(out, exist_ok=True)
    from kaggriculture.bandit.gate import harness as H
    worlds = [s for _, s in H.world_seeds(48)]
    field = sorted(f for f in os.listdir(FIELD) if f.endswith(".py"))
    rng = random.Random(a.seed)
    tasks = []
    for i in range(a.games):
        f = field[i % len(field)] if i % 4 else "v61.1_bandit.py"
        p = OURS if f == "v61.1_bandit.py" else os.path.join(FIELD, f)
        seed = rng.choice(worlds) if i % 2 == 0 else rng.randrange(1, 2**31 - 1)
        tasks.append((f"g{i:04d}", f[:-3], p, seed, i % 2, out))
    man = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(play, t) for t in tasks]):
            gid, info = fut.result()
            man[gid] = info
            print("REC", gid, json.dumps(info), flush=True)
    json.dump(dict(sorted(man.items())), open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("RECORDED", out, sum("error" not in v for v in man.values()), "games", flush=True)


def _norm(x):
    return json.loads(json.dumps(x, default=lambda s: s.__dict__))


def check_game(task):
    path, cut, max_step = task
    from ref import truncate
    obs = [json.loads(l) for l in gzip.open(path, "rt") if l.strip()]
    ref = truncate.load_cut(cut)
    t0 = time.perf_counter()
    exp = []
    with contextlib.redirect_stdout(io.StringIO()):
        for o in obs:
            exp.append(_norm(ref(o, None)))
    py_s = time.perf_counter() - t0
    p = subprocess.run([AGENT, "--base", BASE, "--timing", "--cut", cut], input="\n".join(json.dumps(o) for o in obs) + "\n",
                       capture_output=True, text=True)
    got = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
    timing = [l for l in p.stderr.splitlines() if "turns" in l]
    fired = [l.split("fired ")[-1] for l in p.stderr.splitlines() if "fired" in l]
    first = None
    ndiff = 0
    for t, (e, g) in enumerate(zip(exp, got)):
        if e != g and (max_step is None or (obs[t].get("step") or 0) <= max_step):
            ndiff += 1
            if first is None:
                first = {"i": t, "step": obs[t].get("step"), "exp": e, "got": g}
    if len(exp) != len(got):
        ndiff += 1
    return os.path.basename(path), {"steps": len(exp), "diff": ndiff, "first": first, "py_ms": py_s * 1e3 / max(1, len(exp)),
                                    "rust": timing[-1].split("] ")[-1] if timing else p.stderr[-300:], "fired": fired[-1] if fired else ""}


def check(a):
    runs = sorted(glob.glob(os.path.join(REC, "*")))
    run = os.path.join(REC, a.run) if a.run else runs[-1]
    games = sorted(glob.glob(os.path.join(run, "*.jsonl.gz")))[: a.games or None]
    bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(check_game, (g, a.cut, a.max_step)) for g in games]):
            name, r = fut.result()
            bad += r["diff"] > 0
            print(f"{'OK  ' if not r['diff'] else 'DIFF'} {name} steps {r['steps']} diffs {r['diff']} py {r['py_ms']:.2f}ms/turn rust {r['rust']} | fired {r['fired']}", flush=True)
            if r["first"]:
                f = r["first"]
                print(f"     first @step {f['step']}\n     exp {json.dumps(f['exp'])}\n     got {json.dumps(f['got'])}", flush=True)
    print(f"PARITY cut={a.cut} run={os.path.basename(run)} games {len(games)} exact {len(games) - bad} diverged {bad}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--games", type=int, default=40)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--seed", type=int, default=611)
    c = sub.add_parser("check")
    c.add_argument("--cut", default="chassis")
    c.add_argument("--run")
    c.add_argument("--games", type=int, default=0)
    c.add_argument("--max-step", type=int, default=None, help="compare steps <= this only")
    c.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    record(a) if a.cmd == "record" else check(a)


if __name__ == "__main__":
    main()
