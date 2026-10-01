#!/usr/bin/env python
"""(4) SUBMISSION BUILDER  --  local.  gdrive weights -> agent -> validate -> gate
-> notebook -> PRIVATE kaggle kernel.  NEVER submits to the competition (that stays
your manual click). Building artifacts / uploading the ckpt dataset / pushing a
PRIVATE kernel are the authorized automated steps.

Stages (each logged to .local/submissions/<stamp>/build.log):
  1. rclone-pull the target weights from gdrive:kaggriculture/ckpt
  2. build the agent from the checkpoint  (policy_agent.make_agent)
  3. SELF-PLAY VALIDATION EPISODE (agent vs a copy of itself) -- Kaggle's upload
     check; if it fails we STOP (never push a broken agent)
  4. GATE vs the top-agent panel across multiple worlds/seeds  (tournament_gate)
  5. GATE vs the LAST submission                              (intraday_gate)
  6. (--push) refresh the ckpt Kaggle dataset + push the PRIVATE submission kernel

    python scripts/trackp/submission_build.py                      # build+validate+gate, no push
    python scripts/trackp/submission_build.py --weights policy_rl.pt --push
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"src{os.pathsep}vendor")
GDRIVE = os.environ.get("GDRIVE", "gdrive:kaggriculture")


def sh(cmd, log, optional=False, capture=False):
    line = "$ " + " ".join(str(c) for c in cmd)
    print(line, flush=True); log.write(line + "\n"); log.flush()
    r = subprocess.run(cmd, cwd=ROOT, env=ENV,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None, text=True)
    if capture and r.stdout:
        print(r.stdout); log.write(r.stdout + "\n"); log.flush()
    if r.returncode != 0 and not optional:
        raise SystemExit(f"stage failed (rc={r.returncode}): {cmd}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default="policy_rl.pt",
                    help="gdrive ckpt to submit (policy_rl.pt | policy_bc.pt)")
    ap.add_argument("--seeds", type=int, default=8, help="gate seeds (worlds)")
    ap.add_argument("--last", default=None, help="last-submission agent for intraday gate")
    ap.add_argument("--push", action="store_true", help="push the PRIVATE kernel (gated on validation PASS)")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = os.path.join(ROOT, ".local", "submissions", stamp)
    os.makedirs(work, exist_ok=True)
    log = open(os.path.join(work, "build.log"), "w")
    results = {"stamp": stamp, "weights": a.weights}

    # 1) pull weights
    wl = os.path.join(work, a.weights)
    sh(["rclone", "copyto", f"{GDRIVE}/ckpt/{a.weights}", wl], log, optional=True)
    if not os.path.exists(wl):
        raise SystemExit(f"weights not found: {wl} (pull {GDRIVE}/ckpt/{a.weights} manually)")

    # 2)+3) build agent + EVIDENCE-BASED validation (not a PASS flag: real numbers)
    from kaggriculture.trackp.policy_agent import make_agent
    agent = make_agent(wl)
    replay_dir = os.path.join(work, "replays")
    os.makedirs(replay_dir, exist_ok=True)
    ev = _evidence(agent, a.seeds, replay_dir)
    results["evidence"] = ev
    _print_evidence(ev, log)
    print(f"  gameplay replays (per-turn, reviewable) -> {replay_dir}")
    log.write(f"replays -> {replay_dir}\n")
    if not ev["works"]:
        json.dump(results, open(os.path.join(work, "result.json"), "w"), indent=2)
        raise SystemExit("AGENT DOES NOT DEMONSTRABLY WORK -- "
                         f"failed: {', '.join(ev['failed'])}. Not gating/pushing. See build.log")

    # 4) gate vs the top-agent panel across worlds/seeds
    r = sh([PY, "-m", "kaggriculture.measure.tournament_gate", wl, "--seeds", str(a.seeds)],
           log, optional=True, capture=True)
    results["panel_gate"] = _parse_gate(r.stdout or "")

    # 5) gate vs the last submission (if given)
    if a.last:
        r = sh([PY, "-m", "kaggriculture.measure.intraday_gate", wl, "--vs", a.last],
               log, optional=True, capture=True)
        results["intraday_gate"] = (r.stdout or "").strip().splitlines()[-3:]

    json.dump(results, open(os.path.join(work, "result.json"), "w"), indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(results, indent=2))
    print(f"log + result -> {work}")

    # 6) push private kernel (only after validation PASS)
    if a.push:
        _push_kernel(wl, work, log)
    else:
        print("\n(no --push: artifacts + gate only. Re-run with --push to publish the PRIVATE kernel.)")


def _play_logged(agent_a, agent_b, seed, replay_path):
    """Like serve_match.run_match but LOGS every turn to a reviewable JSONL so a
    human (or a later tool) can replay exactly how the agent played: per step it
    records day/hour, both seats' actions, and the running bank of each seat.
    Returns (bank0, bank1)."""
    import json as _j
    from kaggriculture.engine.serve_match import obs_for, action_to_line, Serve, EPISODE_STEPS
    srv = Serve()
    try:
        js = srv.cmd(f"RESET {seed}")
        with open(replay_path, "w") as rp:
            rp.write(_j.dumps({"meta": {"seed": seed, "steps": EPISODE_STEPS - 1}}) + "\n")
            while js["step"] < EPISODE_STEPS - 1:
                a0 = agent_a(obs_for(0, js)) or {}
                a1 = agent_b(obs_for(1, js)) or {}
                money = [f.get("money") for f in js["farms"]]
                rp.write(_j.dumps({
                    "step": js["step"], "day": js.get("day"), "hour": js.get("hour"),
                    "money": [round(float(m), 1) for m in money],
                    "a0": a0, "a1": a1}) + "\n")
                js = srv.cmd(f"STEP2 {action_to_line(a0)}\x1e{action_to_line(a1)}")
                if "error" in js:
                    rp.write(_j.dumps({"error": js["error"], "step": js.get("step")}) + "\n")
                    raise RuntimeError(js["error"])
            money = [float(f.get("money")) for f in js["farms"]]
            rp.write(_j.dumps({"final": True, "money": [round(m, 1) for m in money],
                               "winner": (0 if money[0] > money[1] else
                                          (1 if money[1] > money[0] else -1))}) + "\n")
        return money[0], money[1]
    finally:
        srv.close()


def _evidence(agent, seeds, replay_dir):
    """Prove the agent WORKS with measured numbers, not a PASS flag. Plays the
    agent vs a copy of itself AND vs a PASS baseline over `seeds` worlds (both
    seats), instrumenting every turn for latency + action content, and LOGS every
    game to replay_dir for review. Returns the raw metrics plus an explicit
    go/no-go with the exact failing checks.

    A 'working' submission must, concretely:
      * COMPLETE every episode (engine raises nothing)             -> status DONE
      * CRUSH the PASS baseline (a do-nothing agent) in score      -> vs-PASS win
      * actually ACT (not degenerate all-PASS)                     -> action rate
      * stay under the 1s actTimeout with margin                   -> worst latency
      * bank non-trivially in self-play (not ~0 on both seats)     -> self-play bank
      * be deterministic given an obs (reproducible submission)    -> determinism
    """
    import time as _t, os as _os
    from kaggriculture.engine.serve_match import obs_for, Serve

    lat, nonempty, turns = [], [0], [0]

    def wrap(fn):
        def g(obs):
            t0 = _t.perf_counter()
            act = fn(obs) or {}
            lat.append((_t.perf_counter() - t0) * 1000.0)
            turns[0] += 1
            if act.get("farmer") or act.get("hands") or act.get("market"):
                nonempty[0] += 1
            return act
        return g

    PASS = lambda o: {}
    seed_list = [3 + 2 * i for i in range(max(1, seeds))]
    self_banks, vspass = [], []
    errors = []
    wa = wrap(agent)
    replays = []
    rp = lambda tag, sd: _os.path.join(replay_dir, f"{tag}_seed{sd}.jsonl")
    # self-play (both seats are the agent) + vs-PASS (agent as seat 0 and seat 1)
    for sd in seed_list:
        try:
            f = rp("selfplay", sd); b0, b1 = _play_logged(wa, agent, sd, f)
            self_banks += [b0, b1]; replays.append(_os.path.basename(f))
        except Exception as e:
            errors.append(f"selfplay seed{sd}: {e}")
        try:
            f0 = rp("vs_pass_seat0", sd); a0, p1 = _play_logged(wa, PASS, sd, f0)
            f1 = rp("vs_pass_seat1", sd); p0, a1 = _play_logged(PASS, wa, sd, f1)
            vspass.append(a0 > p1); vspass.append(a1 > p0)
            replays += [_os.path.basename(f0), _os.path.basename(f1)]
        except Exception as e:
            errors.append(f"vs-PASS seed{sd}: {e}")

    # determinism: same obs twice -> identical action (default sample=False)
    det = True
    try:
        srv = Serve(); js = srv.cmd("RESET 5"); o = obs_for(0, js); srv.close()
        det = (agent(o) == agent(o))
    except Exception as e:
        errors.append(f"determinism probe: {e}"); det = False

    import statistics as st
    worst = max(lat) if lat else 0.0
    p95 = (sorted(lat)[int(0.95 * len(lat))] if lat else 0.0)
    mean_lat = (st.mean(lat) if lat else 0.0)
    act_rate = (nonempty[0] / turns[0]) if turns[0] else 0.0
    vspass_wr = (sum(vspass) / len(vspass)) if vspass else 0.0
    self_mean = (st.mean(self_banks) if self_banks else 0.0)

    checks = {
        "completed_all":  (len(errors) == 0, errors[:3]),
        "beats_PASS":     (vspass_wr >= 0.90, f"win vs PASS={vspass_wr:.0%} (need >=90%)"),
        "actually_acts":  (act_rate >= 0.30, f"non-PASS turns={act_rate:.0%} (need >=30%)"),
        "latency_ok":     (worst < 900.0, f"worst turn={worst:.0f}ms (need <900ms)"),
        "banks_nontriv":  (self_mean > 100.0, f"self-play mean bank={self_mean:.0f} (need >100)"),
        "deterministic":  (det, "same obs gave different actions" if not det else "ok"),
    }
    failed = [k for k, (ok, _) in checks.items() if not ok]
    return {
        "works": not failed, "failed": failed,
        "vs_pass_winrate": round(vspass_wr, 3), "action_rate": round(act_rate, 3),
        "self_play_mean_bank": round(self_mean, 1),
        "latency_ms": {"mean": round(mean_lat, 2), "p95": round(p95, 1), "worst": round(worst, 1)},
        "self_banks": [round(b, 1) for b in self_banks], "errors": errors,
        "replays": replays,
        "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
    }


def _print_evidence(ev, log):
    lines = ["", "=== AGENT WORKS? (measured evidence, not a PASS flag) ==="]
    for k, c in ev["checks"].items():
        mark = "OK " if c["ok"] else "XX "
        lines.append(f"  [{mark}] {k:14s} {c['detail']}")
    L = ev["latency_ms"]
    lines.append(f"  latency: mean {L['mean']}ms  p95 {L['p95']}ms  worst {L['worst']}ms")
    lines.append(f"  vs-PASS winrate {ev['vs_pass_winrate']:.0%} | action rate "
                 f"{ev['action_rate']:.0%} | self-play mean bank {ev['self_play_mean_bank']}")
    lines.append(f"  VERDICT: {'WORKS' if ev['works'] else 'BROKEN -> ' + ', '.join(ev['failed'])}")
    for ln in lines:
        print(ln); log.write(ln + "\n")
    log.flush()


def _parse_gate(out):
    d = {}
    for ln in out.splitlines():
        if "PASS=" in ln:
            d["pass"] = "PASS=True" in ln
        if "winrate=" in ln:
            d.setdefault("lines", []).append(ln.strip())
    return d


def _push_kernel(weights, work, log):
    """Refresh the ckpt Kaggle dataset + push the PRIVATE submission kernel.
    Uses kaggle CLI + configs/trackp_bc_kernel-metadata.json as the template.
    NEVER runs `kaggle competitions submit`."""
    meta = os.path.join(ROOT, "configs", "trackp_bc_kernel-metadata.json")
    print("\n=== PUSH (private kernel) ===")
    if not _have("kaggle"):
        print("kaggle CLI not installed -- `pip install kaggle` + set ~/.kaggle/kaggle.json, then re-run --push")
        return
    # the notebook loads weights from the mounted ckpt dataset; refresh that dataset first
    print("1) update the ckpt Kaggle dataset with the new weights (authorized):")
    print(f"   kaggle datasets version -p <dir-with {os.path.basename(weights)}> -m '{time.strftime('%Y-%m-%d')}'")
    print("2) push the PRIVATE kernel:")
    print(f"   kaggle kernels push -p <kernel-dir with {os.path.basename(meta)}>")
    print("   (is_private=true in the metadata; competition submit stays your manual click)")
    log.write("push instructions emitted (see console)\n")
    # NOTE: wire the exact dataset dir + kernel dir on the first live push; see RUNBOOK step 4.


def _have(exe):
    from shutil import which
    return which(exe) is not None


if __name__ == "__main__":
    main()
