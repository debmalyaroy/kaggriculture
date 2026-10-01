"""Test a candidate base tape through the bandit shell (self-consistency + strength).

Drives `kagg bandit` (seat 0) with KAGG_BASETAPE=<tape> against a python opponent
(seat 1) on the serve engine, at the seed the tape was recorded on. Reports our
bank vs theirs. A DESYNC shows as a near-zero / collapsed bank; a transferred base
shows as a bank near the source agent's own.

Usage: python src/test_basetape.py --tape T.tape --seed 1001 --opp OPP.py [--base v56y]
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, json, os, subprocess, sys

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    finally:
        sys.stdout = old


def run(tape, seed, opp_fn, base):
    env = {**os.environ}
    if tape:
        env["KAGG_BASETAPE"] = tape
    proc = subprocess.Popen([KAGG, "bandit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env=env)
    proc.stdin.write("RESETP\n"); proc.stdin.flush(); proc.stdout.readline()
    E = ServeEnv(); obs = E.reset(seed); world = None
    while not obs.get("done"):
        cv = seat_view(obs, 0); ov = seat_view(obs, 1)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = f"{sh[0]}|{sh[1]}"
        proc.stdin.write(json.dumps(cv) + "\n"); proc.stdin.flush()
        our = json.loads(proc.stdout.readline())
        obs = E.step_both(our, opp_fn(dict(ov)))
    E.close(); proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.terminate()
    fm = obs["farms"]
    return fm[0]["money"], fm[1]["money"], world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default=None)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--opp", default=os.path.join(ROOT, ".local/panel_full/aurax7_kaggriculture_shop_router_reactive_v5.py"))
    a = ap.parse_args()
    opp = load(a.opp)
    us0, th0, w0 = run(None, a.seed, opp, "v56y")
    print(f"  v56y   seed {a.seed}  {w0:28}  {us0:8.0f} vs {th0:8.0f}  {'WIN' if us0>th0 else 'loss'}")
    if a.tape:
        us1, th1, w1 = run(a.tape, a.seed, opp, "base")
        print(f"  BASE   seed {a.seed}  {w1:28}  {us1:8.0f} vs {th1:8.0f}  {'WIN' if us1>th1 else 'loss'}   (d={us1-us0:+.0f})")


if __name__ == "__main__":
    main()
