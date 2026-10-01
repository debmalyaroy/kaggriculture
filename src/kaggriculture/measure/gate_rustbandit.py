"""Faithful gate: the REAL Rust `kagg bandit` vs a panel sample, win rate by env.

The Python build_bandit_harness output is NOT faithful to the shipped Rust bandit
(it sold 1434 wool vs the Rust agent's 7198). So gate the RUST binary directly.
Runs the bandit under two env settings (A vs B) against the same opponents/seeds
on ServeEnv, tallies wins, buckets by realized world. Use to test in-architecture
config levers (KAGG_COMPAT, KAGG_DEMANDSELL, ...) on a faithful testbed.

Usage: python src/gate_rustbandit.py --a "KAGG_COMPAT=1" --b "" --seeds 2001-2010 --opps a,b,c
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, json, os, subprocess, sys
from collections import defaultdict

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")
PANEL = os.path.join(ROOT, ".local", "panel_full")


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return getattr(m, "agent", None)
    except Exception:
        return None
    finally:
        sys.stdout = old


def parse_env(spec):
    e = {}
    for kv in spec.split(";"):
        kv = kv.strip()
        if "=" in kv:
            k, v = kv.split("=", 1); e[k] = v
    return e


def run(env_extra, seed, opp_fn):
    proc = subprocess.Popen([KAGG, "bandit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env={**os.environ, **env_extra})
    proc.stdin.write("RESETP\n"); proc.stdin.flush(); proc.stdout.readline()
    E = ServeEnv(); obs = E.reset(seed); world = None
    while not obs.get("done"):
        cv = seat_view(obs, 0); ov = seat_view(obs, 1)
        if world is None:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = "|".join(sorted(sh[:2]))
        proc.stdin.write(json.dumps(cv) + "\n"); proc.stdin.flush()
        our = json.loads(proc.stdout.readline())
        obs = E.step_both(our, opp_fn(dict(ov)))
    E.close(); proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.terminate()
    fm = obs["farms"]
    return fm[0]["money"], fm[1]["money"], world


def sc(u, t): return 1.0 if u > t else (0.5 if u == t else 0.0)


def seeds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="KAGG_COMPAT=1"); ap.add_argument("--b", default="")
    ap.add_argument("--seeds", default="2001-2010"); ap.add_argument("--opps", default=None)
    a = ap.parse_args()
    ea, eb = parse_env(a.a), parse_env(a.b)
    seeds = seeds_of(a.seeds)
    if a.opps:
        names = a.opps.split(",")
    else:
        names = ["daniilkrasnovvv_farm_top_solution_in_lb", "holeneckles_grandmaster",
                 "kaito_v43", "ahmedberatozer_kaggriculture_v41_review_candidate"]
    opps = [(n, load(os.path.join(PANEL, n + ".py"))) for n in names]
    opps = [(n, f) for n, f in opps if f]
    per = defaultdict(lambda: [0.0, 0.0, 0]); sA = sB = 0.0; N = 0
    for _, opp in opps:
        for s in seeds:
            au, at, w = run(ea, s, opp); bu, bt, _ = run(eb, s, opp)
            saA, saB = sc(au, at), sc(bu, bt); sA += saA; sB += saB; N += 1
            w = w or "?"; per[w][0] += saA; per[w][1] += saB; per[w][2] += 1
    print(f"A env='{a.a}'   B env='{a.b}'\n{'world':34} {'n':>3} {'A':>5} {'B':>5}  d")
    for w in sorted(per, key=lambda k: per[k][0] - per[k][1]):
        x, y, n = per[w]
        print(f"{w[:34]:34} {n:>3} {x:5.1f} {y:5.1f}  {x-y:+.1f}")
    print(f"\nN={N}  A={sA:.1f}  B={sB:.1f}  delta={sA-sB:+.1f}")


if __name__ == "__main__":
    main()
