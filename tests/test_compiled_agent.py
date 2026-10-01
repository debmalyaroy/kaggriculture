"""Differential test for the Track-P COMPILED agent (Phase A transport).

The compiled agent is two implementations of ONE policy:

  * `rustengine/src/policy.rs`   -- what the shipped binary runs
  * the `_fallback_agent()` inlined in `main.py` -- what runs when the
    subprocess bridge fails

At the DEFAULT search budget of 0 this asserts they emit BYTE-IDENTICAL actions
over full episodes. That is the correctness proof for the transport: if it
holds, a fallback costs speed and nothing else, and any bank difference between
the compiled artefact and the reference planner is a transport bug, not a
policy difference.

    python tests/test_compiled_agent.py                 # 3 seeds, both seats
    python tests/test_compiled_agent.py --seeds 3,4,5,6,7 --stage <dir>

**`--budget-ms N` (N > 0) turns the Phase-B searcher on, and then the two
streams SHOULD differ** -- the whole point of the search is to play something
better than the skeleton. In that mode the identity assertion is dropped and
the gate becomes transport-only (fallback 0, validator repairs 0, a full
episode). The equivalence guarantee is not weakened by this: it is still tested
at budget 0, which is a real path (`kagg play` bare, and the shipped binary
whenever the budget is configured to 0), and `search_selftest` proves inside
Rust that a zero-budget searcher is action-for-action the skeleton. Run both:

    python tests/test_compiled_agent.py                      # identity gate
    python tests/test_compiled_agent.py --budget-ms 100      # search smoke

Runs on the Rust `kagg serve` engine (bit-exact, and 200x faster than the
official interpreter here); the OFFICIAL-engine measurement lives in
src/trackp/compiled/harness.py.

Note: `serve` hands the agent 720 turns where the official interpreter hands
it 719 (see docs/history/issues-and-improvements.md Z1). That is harmless HERE -- both
implementations see the same observation on the same turn, which is all this
test compares -- but it is why the BANK printed below can differ from the
official-engine harness for a policy that acts on the final turn.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import importlib.util
import json
import os
import subprocess
import sys


import kaggriculture.engine.serve_match as serve_match  # noqa: E402


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def opponent_line(agent, obs):
    return serve_match.action_to_line(agent(obs))


def run_seed(stage, ref_path, seed, seat, srv, collect_diffs=True):
    """Play one episode driven by the COMPILED agent, asking the pure-Python
    reference for its action at every turn and comparing."""
    # Fresh module state per episode: the planner caches today's plan.
    for k in ("compiled_main", "ref_skel"):
        sys.modules.pop(k, None)
    bridge = load_module(os.path.join(stage, "main.py"), "compiled_main")
    ref = load_module(ref_path, "ref_skel")
    opp = serve_match.load_agent(os.path.join(ROOT, "agents", "v43.0_bandit.py"))

    js = srv.cmd(f"RESET {seed}")
    diffs = []
    ndiff = 0
    turns = 0
    while not js.get("done"):
        obs_me = serve_match.obs_for(seat, js)
        a_bridge = bridge.agent(obs_me)
        a_ref = ref.agent(obs_me)
        if serve_match.action_to_line(a_bridge) != serve_match.action_to_line(a_ref):
            ndiff += 1
            if collect_diffs and len(diffs) < 5:
                diffs.append((js["step"], a_bridge, a_ref))
        turns += 1
        opp_line = opponent_line(opp, serve_match.obs_for(1 - seat, js))
        me_line = serve_match.action_to_line(a_bridge)
        lines = [me_line, opp_line] if seat == 0 else [opp_line, me_line]
        js = srv.cmd(f"STEP2 {lines[0]}\x1e{lines[1]}")
        if "error" in js:
            raise RuntimeError(js["error"])
    banks = [float(f["money"]) for f in js["farms"]]
    return {
        "seed": seed, "seat": seat, "turns": turns, "diffs": diffs,
        "ndiff": ndiff,
        "bank": banks[seat], "opp_bank": banks[1 - seat],
        "stats": dict(bridge.STATS),
    }


def _stage_from_binary(kagg_bin):
    """Assemble a throwaway stage (freshly built main.py + ./kagg[.exe]) around
    an isolated binary so the identity gate can run without touching the shipped
    stage or rustengine/kagg.exe. The main.py fallback is rebuilt from the
    CURRENT build_econ_agent.py, so the gate compares the just-built Rust binary
    against today's Python planner."""
    import shutil
    trackp_dir = os.path.join(ROOT, "src", "kaggriculture", "trackp")
    # build_main.build() imports `build_econ_agent` by bare name; put its dir
    # (and build_main's own dir) on the path first.
    for p in (os.path.join(trackp_dir, "compiled"), trackp_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import build_main  # noqa: E402  (path set above)

    stage = os.path.join(ROOT, ".local", "scratch", "identity_stage")
    os.makedirs(stage, exist_ok=True)
    build_main.build(os.path.join(stage, "main.py"))
    ext = ".exe" if os.name == "nt" else ""
    shutil.copy2(kagg_bin, os.path.join(stage, "kagg" + ext))
    # G2: the compiled policy loads genome.json at runtime (from beside the exe
    # / cwd). Stage it so the gate exercises the REAL load path, not the
    # &SKELETON fallback -- both sides must agree via the same file.
    genome_json = os.path.join(trackp_dir, "genome.json")
    if os.path.exists(genome_json):
        shutil.copy2(genome_json, os.path.join(stage, "genome.json"))
    return stage


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default=os.path.join(
        ROOT, ".local", "candidates", "trackp_compiled", "stage_win"))
    ap.add_argument("--ref", default=os.path.join(
        ROOT, ".local", "candidates", "trackp_compiled", "skel_ref.py"))
    ap.add_argument("--seeds", default="3,4,5")
    ap.add_argument("--budget-ms", type=int, default=0,
                    help="Phase-B search budget per turn. 0 (default) runs the "
                         "skeleton and enforces the identity gate; >0 runs the "
                         "searcher and gates on transport only.")
    args = ap.parse_args()

    # Read by BOTH sides: bridge.py sizes its watchdog from it at import time
    # and passes it to `kagg play`, so it must be set before main.py is loaded.
    os.environ["TRACKP_BUDGET_MS"] = str(args.budget_ms)
    searching = args.budget_ms > 0

    # KAGG_BIN points the identity gate at an isolated build (e.g. one built
    # with a private CARGO_TARGET_DIR) WITHOUT touching the shipped stage or
    # rustengine/kagg.exe: we assemble a throwaway stage around it (a freshly
    # built main.py bridge + the binary as ./kagg[.exe]) so main.py's
    # __file__-relative lookup finds exactly this binary.
    kagg_bin = os.environ.get("KAGG_BIN", "").strip()
    if kagg_bin:
        if not os.path.exists(kagg_bin):
            print("FAIL: KAGG_BIN=%s does not exist" % kagg_bin)
            return 1
        args.stage = _stage_from_binary(kagg_bin)

    stage_bin = [os.path.join(args.stage, n) for n in ("kagg", "kagg.exe")]
    if not any(os.path.exists(b) for b in stage_bin):
        # No binary => the identity gate has NOTHING to compare. This is a
        # FAILURE, not a skip. A silent pass here is exactly how the shipped
        # binary and its Python fallback drifted into two DIFFERENT agents
        # (T1: linear vs quadrant-matched labour assignment). Build+stage a
        # binary (scripts/build_trackp_linux.ps1) or set KAGG_BIN to an
        # isolated kagg[.exe].
        print("FAIL: no staged artefact in %s and KAGG_BIN unset -- the "
              "identity gate cannot run and MUST NOT pass silently. Build+stage "
              "a binary or set KAGG_BIN to an isolated kagg[.exe]." % args.stage)
        return 1

    # ALWAYS regenerate the reference from the CURRENT build_econ_agent.py. A
    # STALE skel_ref (built before a policy change) is precisely what let the
    # linear-vs-quadrant drift hide: the gate would then compare the new binary
    # against an old planner and either mis-pass or mis-fail. The ref must be
    # today's Python planner.
    subprocess.check_call([sys.executable,
                           os.path.join(ROOT, "src", "kaggriculture", "trackp",
                                        "build_econ_agent.py"),
                           "--genome", "skeleton", "--out", args.ref])

    seeds = [int(s) for s in args.seeds.split(",")]
    srv = serve_match.Serve()
    bad = 0
    try:
        for seed in seeds:
            for seat in (0, 1):
                r = run_seed(args.stage, args.ref, seed, seat, srv,
                             collect_diffs=not searching)
                st = r["stats"]
                if searching:
                    # Diffs are EXPECTED here; the skeleton is the search's
                    # starting point, not its answer.
                    ok = st["fallback"] == 0 and st["repairs"] == 0
                else:
                    ok = not r["diffs"] and st["fallback"] == 0
                bad += 0 if ok else 1
                print(f"seed {seed} seat {seat}  turns {r['turns']:3d}  "
                      f"bank {r['bank']:>9.0f}  opp {r['opp_bank']:>9.0f}  "
                      f"bridge {st['bridge']} fallback {st['fallback']} "
                      f"repairs {st['repairs']}  "
                      f"worst {st['worst_ms']:.1f}ms  "
                      f"differs {r['ndiff']}  "
                      f"{'OK' if ok else 'FAIL'}")
                for step, a, b in r["diffs"]:
                    print(f"    step {step}\n      rust: {json.dumps(a)}"
                          f"\n      py  : {json.dumps(b)}")
                if st["reason"]:
                    print(f"    reason: {st['reason']}")
    finally:
        srv.close()
    mode = (f"SEARCH @ {args.budget_ms} ms (transport gate only; run with no "
            "--budget-ms for the identity gate)" if searching
            else "budget 0: IDENTITY gate (compiled policy == Python fallback)")
    print(f"mode: {mode}")
    print("PASS" if bad == 0 else f"FAIL ({bad} cells)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
