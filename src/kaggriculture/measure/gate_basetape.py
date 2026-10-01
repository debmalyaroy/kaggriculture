"""Panel gate: bandit(v56y) vs a candidate base tape, both against the PANEL.

For each (opponent, seed) plays bandit-v56y and bandit-BASE against the same panel
opponent, tallies wins and mean score (win_metric convention: win=1, draw=.5).
The candidate ships only if it beats v56y on wins over the panel.

Usage: python src/gate_basetape.py --tape T.tape --seeds 1001,1002,1003 [--n-opp 8]
"""
from kaggriculture.paths import ROOT
import argparse, glob, importlib.util, io, json, os, subprocess, sys

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
KAGG = os.path.join(ROOT, "rustengine/kagg.exe")
PANEL = os.path.join(ROOT, ".local", "panel_full")


def load(p):
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        s = importlib.util.spec_from_file_location("o" + os.path.basename(p).replace(".", "_"), p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return getattr(m, "agent", None)
    except Exception:
        return None
    finally:
        sys.stdout = old


def run(tape, seed, opp_fn):
    env = {**os.environ}
    if tape:
        env["KAGG_BASETAPE"] = tape
    proc = subprocess.Popen([KAGG, "bandit"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1, env=env)
    proc.stdin.write("RESETP\n"); proc.stdin.flush(); proc.stdout.readline()
    E = ServeEnv(); obs = E.reset(seed)
    while not obs.get("done"):
        cv = seat_view(obs, 0); ov = seat_view(obs, 1)
        proc.stdin.write(json.dumps(cv) + "\n"); proc.stdin.flush()
        our = json.loads(proc.stdout.readline())
        obs = E.step_both(our, opp_fn(dict(ov)))
    E.close(); proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.terminate()
    fm = obs["farms"]
    return fm[0]["money"], fm[1]["money"]


def sc(us, th):
    return 1.0 if us > th else (0.5 if us == th else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", required=True)
    ap.add_argument("--seeds", default="1001,1002,1003")
    ap.add_argument("--n-opp", type=int, default=8)
    ap.add_argument("--opps", default=None, help="comma list of panel basenames; else a fixed diverse set")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.opps:
        names = a.opps.split(",")
    else:
        names = ["daniilkrasnovvv_farm_top_solution_in_lb", "holeneckles_grandmaster",
                 "kaito_v43", "boatlee_v29_r1_adaptive_market_hysteresis",
                 "ishivapatil_kaggriculture_cow_empire_v3", "dariushafshar_care_is_a_4_2x_multiplier_animal_econo",
                 "ahmedberatozer_kaggriculture_v41_review_candidate", "guruprasaathas111_kaggriculture_master_engine_v3"][:a.n_opp]
    opps = [(n, load(os.path.join(PANEL, n + ".py"))) for n in names]
    opps = [(n, f) for n, f in opps if f]
    v_score = b_score = 0.0; n = 0; vwins = bwins = 0
    print(f"{'opponent':44} {'seed':>5} {'v56y':>8} {'BASE':>8}  d")
    for n_, opp in opps:
        for s in seeds:
            vu, vt = run(None, s, opp); bu, bt = run(a.tape, s, opp)
            v_score += sc(vu, vt); b_score += sc(bu, bt); n += 1
            vwins += vu > vt; bwins += bu > bt
            print(f"{n_[:44]:44} {s:>5} {vu:8.0f} {bu:8.0f}  {bu-vu:+.0f}"
                  f"  {'V' if vu>vt else '.'}{'B' if bu>bt else '.'}")
    print(f"\nN={n}  v56y score={v_score:.1f} ({vwins}W)   BASE score={b_score:.1f} ({bwins}W)"
          f"   delta={b_score-v_score:+.1f}")


if __name__ == "__main__":
    main()
