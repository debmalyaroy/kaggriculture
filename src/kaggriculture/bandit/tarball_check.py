"""Official-engine check of a submission tarball, run INSIDE Linux (the kagg-harness image).

For each (opponent, seed, seat): the unpacked tarball's main.py plays, then (optionally) a
reference Python agent plays the same cell. Reports banks, bridge/fallback counts, worst turn;
with --ref, the verdict is exact bank equality cell by cell (use it for a profile-0 build vs the
Python v61.1).

    python python/tarball_check.py --stage /tmp/art --vs agents/v61_bandit.py \
        --seeds 3,4 --seats 0,1 --ref agents/v61.1_bandit.py
"""
import argparse, importlib.util, inspect, json, os, sys, time

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "vendor"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(mod):
    fns = [v for v in vars(mod).values() if callable(v) and getattr(v, "__module__", None) == mod.__name__]
    return getattr(mod, "agent", None) or fns[-1]


def wrap(fn, lat):
    try:
        takes = len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        takes = True

    def agent(observation, configuration):
        t0 = time.perf_counter()
        a = fn(observation, configuration) if takes else fn(observation)
        lat.append((time.perf_counter() - t0) * 1000.0)
        return a
    return agent


def cell(me_fn, opp_path, seed, seat, tag):
    from kaggle_environments import make
    lat = []
    me = wrap(me_fn, lat)
    opp = wrap(entry(load(opp_path, f"opp_{tag}_{seed}_{seat}")), [])
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([me, opp] if seat == 0 else [opp, me])
    last = env.steps[-1]
    return {"bank": float(last[seat]["reward"] or 0), "opp_bank": float(last[1 - seat]["reward"] or 0),
            "status": last[seat]["status"], "worst_ms": max(lat) if lat else 0.0,
            "mean_ms": sum(lat) / max(1, len(lat))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--vs", nargs="+", required=True)
    ap.add_argument("--seeds", default="3,4")
    ap.add_argument("--seats", default="0,1")
    ap.add_argument("--ref")
    a = ap.parse_args()
    rows = []
    for opp in a.vs:
        for seed in [int(s) for s in a.seeds.split(",")]:
            for seat in [int(s) for s in a.seats.split(",")]:
                old = sys.modules.pop("tar_main", None)
                if old is not None:
                    try:
                        old._kill("next cell")
                    except Exception:
                        pass
                mod = load(os.path.join(a.stage, "main.py"), "tar_main")
                r = cell(mod.agent, opp, seed, seat, "t")
                r.update(opp=os.path.basename(opp), seed=seed, seat=seat, stats=dict(mod.STATS))
                if a.ref:
                    ref = cell(entry(load(a.ref, f"ref_{seed}_{seat}")), opp, seed, seat, "r")
                    r["ref_bank"], r["ref_opp_bank"] = ref["bank"], ref["opp_bank"]
                    r["same"] = (ref["bank"], ref["opp_bank"]) == (r["bank"], r["opp_bank"])
                    r["ref_mean_ms"] = ref["mean_ms"]
                rows.append(r)
                print(json.dumps(r), flush=True)
    fb = sum(r["stats"]["fallback"] for r in rows)
    same = sum(bool(r.get("same")) for r in rows)
    print(f"TARBALL cells {len(rows)} fallback_turns {fb} worst_ms {max(r['worst_ms'] for r in rows):.1f}"
          + (f" identical_to_ref {same}/{len(rows)}" if a.ref else ""), flush=True)


if __name__ == "__main__":
    main()
