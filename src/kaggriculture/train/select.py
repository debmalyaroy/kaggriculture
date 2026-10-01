"""Pick the best models and fuse them into one ensemble agent.

Two jobs, both of which people get wrong by hand:

1. **Selection.** Take the top of the Elo ladder, but only models whose rating
   is actually distinguishable. A 750 on 4 games and a 720 on 40 games is not a
   30-point lead, it is one measured number and one rumour. Selection here
   requires a minimum game count and drops members whose confidence interval
   swallows the leader's.

2. **Fusion.** Write those models' parameter sets into one agent as an explicit
   committee (`ENSEMBLE_MEMBERS`), so the vote is between policies that each
   earned their place, rather than between random jitter around a single one.
   Jitter can only explore a ball around one policy; distinct models disagree
   in ways that carry information.

    python -m kaggriculture.train.select --show                  # what would be chosen, and why
    python -m kaggriculture.train.select --build                 # write the ensemble agent
    python -m kaggriculture.train.select --build --k 5 --min-games 16
    python -m kaggriculture.train.select --build --spread-guard  # refuse near-duplicate members

The result is verified against the actTimeout before it is accepted: an ensemble
that thinks for too long is a loss, not a stronger agent.
"""
from kaggriculture.paths import ROOT
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.pipeline.params as paramio  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402
from kaggriculture.measure.sprt import elo_interval  # noqa: E402

SRC = os.path.join(ROOT, "agents", "v1_heuristic.py")
MEMBERS_BEGIN = "# --- MEMBERS BEGIN"
MEMBERS_END = "# --- MEMBERS END ---"

# Knobs a committee member is allowed to differ on. Restricted to the ones that
# decide *which job a unit takes* -- the thing the vote is actually over.
# Letting members disagree about budgets would have them arguing about spending
# while only their movement gets used.
MEMBER_KNOBS = ("travel_weight", "fert_weight", "poach_penalty", "care_weight",
                "fert_collect_weight", "cost_per_crop_day", "cost_per_animal_day",
                "capacity_util", "shed_pressure", "assign_mode")


def candidates(min_games=16):
    """Rated, existing models, best first, with their confidence intervals."""
    out = []
    for m in registry.ranked():
        if not m.get("exists") or (m.get("games") or 0) < min_games:
            continue
        g, w = m["games"], m.get("wins", 0)
        lo, mid, hi = elo_interval(w, g)
        out.append({**m, "ci_lo": m["elo"] - (m.get("confidence") or 0),
                    "ci_hi": m["elo"] + (m.get("confidence") or 0),
                    "wr_lo": lo, "wr_hi": hi})
    return out


def choose(k=4, min_games=16, spread_guard=False, verbose=True):
    """Which models belong in the committee, and the reason for each verdict."""
    cands = candidates(min_games)
    if not cands:
        if verbose:
            pr.warn(f"no model has {min_games} rated games yet")
            pr.log("fix it:  python -m kaggriculture.measure.elo --rounds 4", 1)
        return [], []

    leader = cands[0]
    picked, rejected = [], []
    for m in cands:
        if len(picked) >= k:
            rejected.append((m, "committee is full"))
            continue
        # A member well below the leader drags the vote toward a worse policy.
        # "Well below" means its whole interval sits under the leader's point.
        if m is not leader and m["ci_hi"] < leader["elo"] - (leader.get("confidence") or 0):
            rejected.append((m, f"clearly weaker: {m['elo']:.0f}+/-"
                                f"{m.get('confidence') or 0:.0f} vs leader "
                                f"{leader['elo']:.0f}"))
            continue
        if spread_guard and picked:
            dup = _near_duplicate(m, picked)
            if dup:
                rejected.append((m, f"near-duplicate of {dup} on the voting knobs"))
                continue
        picked.append(m)
    return picked, rejected


def _member_params(model):
    path = os.path.join(ROOT, model.get("path") or f"agents/{model['name']}")
    P = paramio.load(path)
    return {k: P[k] for k in MEMBER_KNOBS if k in P}


def _near_duplicate(model, picked, tol=0.02):
    """True if this model's voting knobs match one already chosen.

    Two members that always vote identically add cost and no information.
    """
    mine = _member_params(model)
    for other in picked:
        theirs = _member_params(other)
        same = True
        for k, v in mine.items():
            w = theirs.get(k)
            if isinstance(v, (int, float)) and isinstance(w, (int, float)):
                denom = max(abs(v), abs(w), 1e-9)
                if abs(v - w) / denom > tol:
                    same = False
                    break
            elif v != w:
                same = False
                break
        if same:
            return other["name"]
    return None


def write_ensemble(picked, out_path, consensus=0.5, base=None, verbose=True):
    """Write an agent whose committee is exactly `picked`."""
    base = base or os.path.join(ROOT, picked[0].get("path") or
                                f"agents/{picked[0]['name']}")
    P = dict(paramio.load(base))
    members = [_member_params(m) for m in picked[1:]]      # [0] is the incumbent
    P["ensemble_k"] = len(members)
    P["ensemble_consensus"] = consensus

    doc = ["Ensemble of the top rated models, selected by src/kaggriculture/train/select.py.", ""]
    doc.append(f"Incumbent: {picked[0]['name']} "
               f"({picked[0]['elo']:.0f} Elo, {picked[0]['games']} games)")
    for m in picked[1:]:
        doc.append(f"Member   : {m['name']} "
                   f"({m['elo']:.0f} Elo, {m['games']} games)")
    doc.append("")
    doc.append("Unit ops are voted; market orders come from the incumbent alone,")
    doc.append("because they carry budget invariants that do not survive mixing.")

    paramio.write(SRC, out_path, P, header="selected ensemble",
                  module_doc="\n".join(doc) + "\n")

    # Inject the explicit committee into the sentinel block.
    with open(out_path, encoding="utf-8") as f:
        src = f.read()
    i, j = src.index(MEMBERS_BEGIN), src.index(MEMBERS_END)
    body = [MEMBERS_BEGIN + " (src/kaggriculture/train/select.py rewrites this block) ---",
            "# Each entry overrides PARAMS for one committee member. Selected from",
            "# the Elo ladder, so every member independently earned its place.",
            "ENSEMBLE_MEMBERS = ["]
    for m, mp in zip(picked[1:], members):
        body.append(f"    # {m['name']}  {m['elo']:.0f} Elo, {m['games']} games")
        body.append(f"    {json.dumps(mp)},")
    body.append("]")
    src = src[:i] + "\n".join(body) + "\n" + src[j:]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(src)

    rel = os.path.relpath(out_path, ROOT)
    registry.register_model(os.path.basename(out_path), path=rel,
                            built_by="select", ensemble_members=[m["name"] for m in picked],
                            description=f"ensemble of {len(picked)} rated models")
    if verbose:
        pr.log(f"wrote {rel} ({os.path.getsize(out_path):,} bytes)")
    return rel


def check_latency(path, steps=180, budget_ms=1000.0, verbose=True):
    """An ensemble that thinks too long is a loss. Measure before accepting it."""
    import importlib.util
    import statistics
    import time
    sys.path.insert(0, os.path.join(ROOT, "agents"))
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make

    spec = importlib.util.spec_from_file_location("cand", os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    times = []

    def timed(obs, cfg):
        t0 = time.perf_counter()
        a = mod.agent(obs, cfg)
        times.append((time.perf_counter() - t0) * 1000.0)
        return a

    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": 17,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([timed, "agents/v2_tuned.py"])
    times.sort()
    p95 = times[int(len(times) * 0.95)] if times else 0.0
    res = {"mean": statistics.mean(times) if times else 0.0,
           "p95": p95, "max": times[-1] if times else 0.0,
           "headroom": (budget_ms / p95) if p95 else float("inf"),
           "ok": times and times[-1] < budget_ms * 0.5}
    if verbose:
        pr.log(f"latency: mean {res['mean']:.2f} ms, p95 {res['p95']:.2f} ms, "
               f"max {res['max']:.2f} ms  ({res['headroom']:.0f}x headroom at p95)", 1)
        if not res["ok"]:
            pr.warn("too close to the 1000 ms actTimeout -- reduce --k", 1)
    return res


def show(k, min_games, spread_guard):
    picked, rejected = choose(k, min_games, spread_guard)
    if not picked:
        return 1
    print(f"\nCommittee ({len(picked)} of {k} slots)\n" + "-" * 66)
    for i, m in enumerate(picked):
        role = "incumbent" if i == 0 else "member"
        print(f"  {role:<10} {m['name']:<40} {m['elo']:>6.0f} "
              f"+/-{m.get('confidence') or 0:>3.0f}  {m['games']:>3}g")
        mp = _member_params(m)
        print(f"             {json.dumps(mp)}")
    if rejected:
        print(f"\nRejected\n" + "-" * 66)
        for m, why in rejected:
            print(f"  {m['name']:<40} {why}")
    print("\nMarket orders are never voted -- they carry budget invariants that "
          "do not\nsurvive mixing. Only unit ops go to the committee.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=4, help="committee size incl. incumbent")
    ap.add_argument("--min-games", type=int, default=16)
    ap.add_argument("--consensus", type=float, default=0.5)
    ap.add_argument("--spread-guard", action="store_true",
                    help="drop members that vote identically to one already picked")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()

    pr.reset()
    if args.show or not args.build:
        return show(args.k, args.min_games, args.spread_guard)

    picked, rejected = choose(args.k, args.min_games, args.spread_guard)
    if len(picked) < 2:
        pr.warn("need at least two rated models to form a committee")
        pr.log("rate more:  python -m kaggriculture.measure.elo --rounds 4", 1)
        return 1
    for m in picked:
        pr.log(f"member {m['name']}  {m['elo']:.0f} Elo, {m['games']} games", 1)
    for m, why in rejected[:6]:
        pr.log(f"rejected {m['name']}: {why}", 2)

    import datetime as dt
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(ROOT, args.out or f"agents/agent_vsel_{stamp}.py")
    rel = write_ensemble(picked, out, consensus=args.consensus)

    if not args.no_latency:
        res = check_latency(rel)
        registry.register_model(os.path.basename(out),
                                latency_ms=round(res["p95"], 2),
                                tests="latency ok" if res["ok"] else "LATENCY RISK")
        if not res["ok"]:
            return 1

    pr.log("")
    pr.log("next:")
    pr.log(f"python -m kaggriculture.measure.evaluate {rel} --vs {picked[0]['path']} -n 16", 1)
    pr.log(f"python -m kaggriculture.measure.elo --only {rel} --rounds 2", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
