"""Closed-loop parity: replay each recorded game (same opponent, seed, seat) with the Rust agent
in place of the Python v61.1 and require the identical final banks (python/parity.py recorded the
Python agent's banks in manifest.json).

    python python/closed_loop.py --run 20260924T1835Z-s612 --workers 3 [--games N] [--exe PATH]
"""
import argparse, contextlib, io, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = RL  # one repo since the 2026-10-01 merge
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(RL, "python"))
REC = os.path.join(RL, "data", "recordings", "r1", "v61.1")
FIELD = os.path.join(REPO, "data", "winplan", "field")
OURS = os.path.join(REPO, "agents", "v61.1_bandit.py")


def play(task):
    gid, info, exe = task
    from kaggriculture.bandit.gate import loss_forensics as LF, harness as H
    from kaggriculture.engine import serve_match as SM
    from kaggriculture.trackp.serve_env import ServeEnv
    from rust_agent import RustAgent
    seat = info["seat"]
    opp_path = OURS if info["opp"] == "v61.1_bandit" else os.path.join(FIELD, info["opp"] + ".py")
    me = RustAgent(exe=exe)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            opp = LF.load_pyagent(opp_path)
            ags = [me, opp] if seat == 0 else [opp, me]
            E = ServeEnv(kagg_path=H.FRESH_KAGG)
            obs = E.reset(info["seed"])
            t_me = 0.0
            while not (obs.get("done") or obs["step"] >= SM.EPISODE_STEPS - 1):
                acts = []
                for s in (0, 1):
                    t0 = time.perf_counter()
                    a = SM._call(ags[s], SM.obs_for(s, obs)) or {"farmer": ["PASS"], "hands": [], "market": []}
                    if s == seat:
                        t_me = max(t_me, time.perf_counter() - t0)
                    acts.append(a)
                obs = E.step_both(acts[0], acts[1])
            E.close()
        banks = [float(obs["farms"][0]["money"]), float(obs["farms"][1]["money"])]
        return gid, {"banks": banks, "expected": info["banks"], "ok": banks == info["banks"], "max_call_ms": t_me * 1e3}
    except Exception as exc:                                                     # noqa: BLE001
        return gid, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        me.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exe", default=os.path.join(RL, "target-dev", "release", "agent-stdio.exe"))
    a = ap.parse_args()
    man = json.load(open(os.path.join(REC, a.run, "manifest.json")))
    items = [(g, i) for g, i in man.items() if "error" not in i][: a.games or None]
    ok = bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(play, (g, i, a.exe)) for g, i in items]):
            gid, r = fut.result()
            if r.get("ok"):
                ok += 1
            else:
                bad += 1
            print(("OK  " if r.get("ok") else "DIFF"), gid, json.dumps(r), flush=True)
    print(f"CLOSED-LOOP run={a.run} games {len(items)} identical {ok} different {bad}", flush=True)


if __name__ == "__main__":
    main()
