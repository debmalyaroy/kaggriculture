"""Judge v23.1's REAL arm commits by paired counterfactual replay.

The ARM GUARD demands >=25 judged commits before arms may fire, but the
commits v23.1 made in its 83 real ladder games were never judged -- the
audit pipeline didn't exist. This builds it:

  for each sampled real game of v23.1 (55423269):
    opponent = their MINED route (both seats of our games are in the index),
                rebuilt as an agent -- the same reconstruction the crown
                tournament trusts;
    run v23.1 (arms ON) vs opponent  -- read _STATE for the actual commit;
    run v23.2 (arms OFF twin, same base/relay/identifier) vs opponent,
                same seed/seat -- the counterfactual control;
    margin_delta   = v23.1 bank - v23.2 bank   (what the commit cost/earned)
    commit_correct = offline classification of the opponent's FULL route
                     (t=480 prefix, the shipped identifier) == committed
                     class, decisively.

Appends rows to models/v22/commit_audit_rows.jsonl (train_gates.py's AUDIT)
and records identity-carrying sink rows as a side effect. Episodes run
in-process, sequentially -- one core, BSOD-safe.

    python src/experiments/commit_judge.py --games 24 --seeds 1
"""
from kaggriculture.paths import ROOT
import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

AUDIT = os.path.join(ROOT, "models", "v22", "commit_audit_rows.jsonl")
V231 = os.path.join(ROOT, "agents", "v23.1_bandit.py")
V232 = os.path.join(ROOT, "agents", "v23.2_bandit.py")


def load_agent(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_episode(mod_a, mod_b, seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"randomSeed": int(seed)},
               debug=False)
    env.run([mod_a.agent, mod_b.agent])
    r = env.state
    banks = [float(r[0].reward or 0), float(r[1].reward or 0)]
    statuses = [r[0].status, r[1].status]
    return banks, statuses


def classify_route(rid):
    """Decisive full-route class of the opponent, or None."""
    import kaggriculture.data.features as F
    import kaggriculture.data.routes as R
    import kaggriculture.train.train_identifier as TI
    idw = json.load(open(os.path.join(ROOT, "models", "v22", "identifier",
                                      "weights.json"), encoding="utf-8"))
    vec = F.prefix_features(R.load_route(rid), 480) + [480 / 720.0]
    p = TI.stdlib_predict(idw, vec)
    cls = max(range(len(p)), key=lambda i: p[i])
    return (cls, p[cls])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--rng-seed", type=int, default=7)
    ap.add_argument("--bandit", default=V231,
                    help="arms-ON build to judge (default: the historical "
                         "v23.1). For E4, a --lab-force-arms build of the "
                         "RETRAINED arms -- this is how new arms earn the "
                         "field evidence arms_allowed() demands")
    ap.add_argument("--control", default=V232,
                    help="arms-OFF twin (same base/relay/identifier)")
    ap.add_argument("--submission", default="55423269",
                    help="comma-separated submission refs whose real games "
                         "supply the opponents")
    ap.add_argument("--source-label", default="counterfactual-replay-2026-08-13")
    args = ap.parse_args()

    import kaggriculture.engine.engine_check as engine_check
    engine_check.require()
    import kaggriculture.data.routes as R
    import subprocess

    ridx = R.load_index()
    gidx = json.load(open(os.path.join(ROOT, "data", "ourgames",
                                       "index.json"), encoding="utf-8"))
    subs = {s.strip() for s in str(args.submission).split(",") if s.strip()}
    games = [g for g in gidx["games"].values()
             if str(g.get("submission")) in subs]
    # balanced sample: losses teach cost, wins teach gain. The dedup applies
    # per (episode, judged pair): judging DIFFERENT arms against an episode
    # already judged for v23.1's arms is new evidence, not a repeat.
    pair_tag = os.path.basename(args.bandit)
    already = set()
    if os.path.exists(AUDIT):
        for line in open(AUDIT, encoding="utf-8"):
            try:
                r = json.loads(line)
                already.add((str(r.get("episode")),
                             str(r.get("judged_pair")
                                 or os.path.basename(V231))))
            except Exception:                                      # noqa: BLE001
                continue
    games = [g for g in games if (str(g["episode"]), pair_tag) not in already]
    losses = [g for g in games if not g["won"] and not g.get("tied")]
    wins = [g for g in games if g["won"]]
    rng = random.Random(args.rng_seed)
    rng.shuffle(losses)
    rng.shuffle(wins)
    sample = losses[:args.games // 2] + wins[:args.games // 2]

    rows_out = []
    for gi, g in enumerate(sample):
        ep = str(g["episode"])
        our_seat = int(g.get("seat") or 0)
        opp_rid = f"{ep}_s{1 - our_seat}"
        if opp_rid not in ridx["routes"]:
            print(f"  [{gi + 1}/{len(sample)}] {ep}: opponent route not "
                  f"mined -- skip", flush=True)
            continue
        opp_agent_path = os.path.join(ROOT, ".local", "judge",
                                      f"opp_{opp_rid}.py")
        if not os.path.exists(opp_agent_path):
            os.makedirs(os.path.dirname(opp_agent_path), exist_ok=True)
            r = subprocess.run(
                [sys.executable, "src/kaggriculture/data/routes.py", "--build", opp_rid,
                 "--out", os.path.relpath(opp_agent_path, ROOT)],
                capture_output=True, cwd=ROOT)
            if r.returncode != 0:
                print(f"  [{gi + 1}] {ep}: opponent build failed -- skip",
                      flush=True)
                continue
        try:
            opp = load_agent(opp_agent_path, f"opp{gi}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [{gi + 1}] {ep}: opponent import failed: {exc}",
                  flush=True)
            continue

        for s in range(args.seeds):
            seed = 90000 + gi * 7 + s
            a231 = load_agent(args.bandit, f"a231_{gi}_{s}")   # fresh state per run
            b231, st231 = run_episode(a231, opp, seed)
            state = a231._STATE[0]
            arm = state.get("arm") or (state.get("arm_dead") and
                                       state.get("arm_class") is not None)
            committed = state.get("committed_at") is not None
            if not committed:
                print(f"  [{gi + 1}] {ep} seed {seed}: no commit "
                      f"(banks {b231[0]:,.0f})", flush=True)
                continue
            arm_name = state.get("arm") or "raj(abandoned)"
            arm_cls = state.get("arm_class")

            a232 = load_agent(args.control, f"a232_{gi}_{s}")
            b232, st232 = run_episode(a232, opp, seed)

            true_cls, true_p = classify_route(opp_rid)
            correct = (true_cls == arm_cls and true_p >= 0.6)
            delta = b231[0] - b232[0]
            # Record the OUTCOMES, not just the dollar delta. The ladder pays
            # win/draw/loss, so the question a gate must answer is "did the
            # commit change the RESULT?" -- `margin_delta` alone cannot say.
            # train_gates prefers `score_delta` when these fields are present.
            import kaggriculture.measure.win_metric as WM
            s_on = WM.score(b231[0], b231[1])
            s_off = WM.score(b232[0], b232[1])
            row = {"arm": str(arm_name), "arm_class": arm_cls,
                   "family": str(ridx["routes"][opp_rid].get("team") or
                                 f"cls{true_cls}"),
                   "opp_route": opp_rid, "episode": ep, "seed": seed,
                   "commit_correct": bool(correct),
                   "margin_delta": round(delta, 1),
                   "bank_on": round(b231[0], 1), "opp_bank_on": round(b231[1], 1),
                   "bank_off": round(b232[0], 1), "opp_bank_off": round(b232[1], 1),
                   "score_on": s_on, "score_off": s_off,
                   "score_delta": s_on - s_off,
                   "committed_at": state.get("committed_at"),
                   "judged_pair": pair_tag,
                   "source": args.source_label}
            rows_out.append(row)
            print(f"  [{gi + 1}/{len(sample)}] {ep}: commit@"
                  f"{state.get('committed_at')} cls{arm_cls} "
                  f"correct={correct} delta={delta:+,.0f} "
                  f"({b231[0]:,.0f} vs {b232[0]:,.0f})", flush=True)

    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    with open(AUDIT, "a", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    ok = sum(1 for r in rows_out if r["commit_correct"])
    print(f"\njudged {len(rows_out)} commits ({ok} correct, "
          f"{len(rows_out) - ok} wrong); appended to {AUDIT}")


if __name__ == "__main__":
    main()
