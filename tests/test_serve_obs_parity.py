"""A1.4 -- lock the serve observation as FIELD-FOR-FIELD faithful to official.

The bank-exact `--compare-official` check proves faithfulness END-TO-END (a
reactive agent that read a wrong obs would bank differently). This test locks
it at the FIELD level so any future drift in the obs contract (Rust
`service.rs::json_state` or Python `serve_match.obs_for`) is caught immediately,
not only when it happens to move a bank.

Method: drive BOTH the official vendored engine and `kagg serve` with a
deterministic PASS recorder, capture the per-seat observation each engine hands
the agent at every step, and assert the documented obs fields are identical.

Run: python tests/test_serve_obs_parity.py   (or via tests/run_all.py)
Needs a built rustengine/kagg(.exe) and the vendored engine importable.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "vendor"))

from kaggriculture.engine import serve_match as sm  # noqa: E402

FIELDS = ("step", "day", "hour", "player", "farms", "market", "town", "private")
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _plain(o):
    """Recursively convert Struct/_Struct/dict->dict, list->list, for value=="""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


def official_obs(seed):
    """{seat: [obs per step]} as the official engine hands them to agents."""
    from kaggle_environments import make
    rec = {0: [], 1: []}

    def recorder(obs):
        rec[obs["player"]].append({f: _plain(obs[f]) for f in FIELDS})
        return PASS

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([recorder, recorder])
    return rec


def serve_obs(seed):
    """{seat: [obs per step]} as serve hands them, driven with the SAME PASS."""
    rec = {0: [], 1: []}
    srv = sm.Serve()
    try:
        js = srv.cmd(f"RESET {seed}")
        while js["step"] < sm.EPISODE_STEPS - 1:
            for seat in (0, 1):
                o = sm.obs_for(seat, js)
                rec[seat].append({f: _plain(o[f]) for f in FIELDS})
            js = srv.cmd(f"STEP2 {sm.action_to_line(PASS)}\x1e{sm.action_to_line(PASS)}")
            if "error" in js:
                raise RuntimeError(js["error"])
    finally:
        srv.close()
    return rec


def check(seed):
    off = official_obs(seed)
    srv = serve_obs(seed)
    problems = []
    for seat in (0, 1):
        o, s = off[seat], srv[seat]
        if len(o) != len(s):
            problems.append(f"seat{seat}: step count {len(s)} serve != {len(o)} official")
            continue
        for step, (oo, ss) in enumerate(zip(o, s)):
            for f in FIELDS:
                if oo[f] != ss[f]:
                    problems.append(f"seat{seat} step{step} field '{f}' differs")
                    if len(problems) > 20:
                        return problems
    return problems


def main():
    seeds = [int(s) for s in (sys.argv[1:] or ["3", "4"])]
    all_problems = []
    for seed in seeds:
        p = check(seed)
        status = "OK" if not p else f"{len(p)} DIFF(s)"
        print(f"seed {seed}: obs parity {status}")
        for line in p[:10]:
            print("   ", line)
        all_problems += p
    if all_problems:
        print(f"\nFAIL: {len(all_problems)} obs field difference(s)")
        return 1
    print(f"\nPASS: serve obs is field-for-field faithful over {len(seeds)} seed(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
