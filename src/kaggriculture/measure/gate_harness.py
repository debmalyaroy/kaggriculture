"""Per-world gate: two python harnesses (A=candidate, B=baseline) vs the PANEL.

For each (opponent, seed) plays A and B against the same opp on ServeEnv, records
the realized D6 world, and tallies wins/score overall and PER WORLD. Shows which
world-forks help (positive delta) vs hurt (drop those forks and rebuild).

Usage: python src/gate_harness.py --a harness_v58.py --b harness_base.py
          --seeds 2001-2020 [--opps a,b,c]
"""
from kaggriculture.paths import ROOT
import argparse, importlib.util, io, os, sys
from collections import defaultdict

from kaggriculture.trackp.serve_env import ServeEnv, seat_view  # noqa: E402
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


def play(ag, opp, seed):
    E = ServeEnv(); obs = E.reset(seed); world = None
    while not obs.get("done"):
        cv = seat_view(obs, 0); ov = seat_view(obs, 1)
        if world is None and int(obs.get("step") or 0) >= 146:
            sh = (cv.get("town") or {}).get("unlocked_shops") or []
            if len(sh) >= 2: world = "|".join(sorted(sh[:8]))
        obs = E.step_both(ag(dict(cv)) or {"farmer": ["PASS"], "hands": [], "market": []},
                          opp(dict(ov)) or {"farmer": ["PASS"], "hands": [], "market": []})
    E.close(); fm = obs["farms"]
    return fm[0]["money"], fm[1]["money"], world


def sc(us, th):
    return 1.0 if us > th else (0.5 if us == th else 0.0)


def seeds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", default="2001-2016")
    ap.add_argument("--opps", default=None)
    a = ap.parse_args()
    A = load(a.a if os.path.isabs(a.a) else os.path.join(ROOT, ".local/livepool", a.a))
    B = load(a.b if os.path.isabs(a.b) else os.path.join(ROOT, ".local/livepool", a.b))
    seeds = seeds_of(a.seeds)
    if a.opps:
        names = a.opps.split(",")
    else:
        names = ["daniilkrasnovvv_farm_top_solution_in_lb", "holeneckles_grandmaster",
                 "kaito_v43", "boatlee_v29_r1_adaptive_market_hysteresis",
                 "ishivapatil_kaggriculture_cow_empire_v3",
                 "dariushafshar_care_is_a_4_2x_multiplier_animal_econo",
                 "ahmedberatozer_kaggriculture_v41_review_candidate",
                 "guruprasaathas111_kaggriculture_master_engine_v3"]
    opps = [(n, load(os.path.join(PANEL, n + ".py"))) for n in names]
    opps = [(n, f) for n, f in opps if f]
    per = defaultdict(lambda: [0.0, 0.0, 0])  # world -> [scoreA, scoreB, n]
    sA = sB = 0.0; N = 0
    for _, opp in opps:
        for s in seeds:
            au, at, w = play(A, opp, s); bu, bt, _ = play(B, opp, s)
            saA, saB = sc(au, at), sc(bu, bt)
            sA += saA; sB += saB; N += 1
            w = w or "?"; per[w][0] += saA; per[w][1] += saB; per[w][2] += 1
    print(f"{'world':40} {'n':>3} {'A':>5} {'B':>5}  d")
    for w in sorted(per, key=lambda k: per[k][1] - per[k][0]):
        a2, b2, n = per[w]
        print(f"{w[:40]:40} {n:>3} {a2:5.1f} {b2:5.1f}  {a2-b2:+.1f}")
    print(f"\nN={N}  A(cand)={sA:.1f}  B(base)={sB:.1f}  delta={sA-sB:+.1f}")


if __name__ == "__main__":
    main()
