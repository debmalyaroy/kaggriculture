"""Parity gate for the SHIPPED bandit (Rust `kagg mbandit`) vs its Python net.

Two audits (docs/history/bug-findings-2026-09-18.md §BANDIT) found the shipped Rust path
and the "action-faithful" Python fallback were DIFFERENT agents (C1/H3-H7): the
Rust `sells()` overlay, weed-repair, scarcity/front-run rails and hand-align had
no faithful twin in the fallback, so a transport failure silently played a
weaker agent. The resolution (approach B) is: the ECONOMY lives ONLY in
`kagg mbandit`; the Python fallback is demoted to a LOUD legal-PASS safety net.
"What we ship == what we measure" then holds because both are the Rust path.

This gate enforces exactly that, and CANNOT silently skip:

  G6.1 -- plays the shipped stage (Rust `kagg mbandit` via the driver) and the
          Python fallback on the SAME worlds and asserts:
            * the Rust bridge answered EVERY turn (STATS.fallback == 0) and banks
              are sane (it actually ran the economy), AND
            * the demoted Python fallback emits ONLY a legal PASS every turn
              (farmer PASS, each live hand PASS, no market orders).
          It FAILS (never skips) when no isolated binary is staged. Point it at
          an isolated build with KAGG_BIN (default:
          .local/scratch/bandit_target/release/kagg[.exe]) -- NEVER
          rustengine/kagg.exe, which another workload uses.

  G6.2 -- config round-trip: every knob in the shipped config is actually
          consumed by mbandit.rs (no dead knobs, no silent config lie).

    set KAGG_BIN=.local/scratch/bandit_target/release/kagg.exe
    PYTHONPATH=src PYTHONUTF8=1 python tests/test_bandit_parity.py

Build the isolated binary first (NEVER into rustengine/target):
    CARGO_TARGET_DIR=.local/scratch/bandit_target cargo build --release \
        --bin kagg --manifest-path rustengine/Cargo.toml
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys

# build_rust_bandit moved to the bandit build namespace (G0.3, 2026-09-18) and
# now uses proper absolute imports for its siblings, so no path hack is needed.
import kaggriculture.engine.serve_match as serve_match  # noqa: E402
from kaggriculture.bandit.build import build_rust_bandit  # noqa: E402

CONFIG = os.path.join(ROOT, "configs", "bandit_config_v581.json")
MBANDIT_RS = os.path.join(ROOT, "rustengine", "src", "mbandit.rs")
OPP = os.path.join(ROOT, "agents", "v43.0_bandit.py")
STAGE = os.path.join(ROOT, ".local", "scratch", "bandit_parity_stage")

# structural entries every guardrail carries; everything else is a tunable knob
_STRUCTURAL = {"name", "on"}
# top-level config keys consumed by the builder/driver/dispatch (not load_config)
_TOP_KEYS = {"_doc", "checkpoints", "branches", "guardrails", "base_tape"}


def _isolated_binary():
    """The isolated agent binary: KAGG_BIN, else the private-target default.
    NEVER rustengine/kagg.exe (that is another workload's live binary)."""
    env = os.environ.get("KAGG_BIN", "").strip()
    if env:
        return env
    name = "kagg.exe" if os.name == "nt" else "kagg"
    return os.path.join(ROOT, ".local", "scratch", "bandit_target", "release", name)


def load_module(path, name):
    for k in (name,):
        sys.modules.pop(k, None)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- G6.2: config round-trip (no dead knobs) -------------------------------

def consumed_from_rust():
    """Parse mbandit.rs for the guards + knobs it actually reads, so a dead
    config knob is caught against SOURCE, not a hand-maintained list."""
    src = open(MBANDIT_RS, encoding="utf-8").read()
    guards = set(re.findall(r'\.g\("([^"]+)"\)', src))
    knobs: dict[str, set] = {}
    for name, key in re.findall(r'\.knob\("([^"]+)",\s*"([^"]+)"', src):
        knobs.setdefault(name, set()).add(key)
    for name, key in re.findall(r'\.list\("([^"]+)",\s*"([^"]+)"', src):
        knobs.setdefault(name, set()).add(key)
    return guards, knobs


def check_config_roundtrip(cfg):
    guards, knobs = consumed_from_rust()
    bad = []
    for k in cfg:
        if k not in _TOP_KEYS:
            bad.append(f"top-level key '{k}' not consumed by builder/driver")
    for g in cfg.get("guardrails", []):
        name = g.get("name")
        if name not in guards:
            bad.append(f"guardrail '{name}' has no g(\"{name}\") in mbandit.rs")
            continue
        for key in g:
            if key in _STRUCTURAL:
                continue
            if key not in knobs.get(name, set()):
                bad.append(f"dead knob '{name}.{key}' -- not read by mbandit.rs")
    print(f"G6.2 config round-trip: guards={sorted(guards)}")
    print(f"                        knobs={ {k: sorted(v) for k, v in knobs.items()} }")
    if bad:
        for b in bad:
            print("  DEAD:", b)
        print("G6.2 FAIL")
        return False
    print("G6.2 PASS (every config knob is consumed)")
    return True


# ---- G6.1: shipped-Rust runs, Python net is legal-PASS ---------------------

def _is_legal_pass(action, n_hands):
    """The demoted fallback contract: farmer PASS, each live hand PASS
    (positionally aligned), and no market orders."""
    if not isinstance(action, dict):
        return False, "not a dict"
    if action.get("farmer") != ["PASS"]:
        return False, f"farmer={action.get('farmer')!r} (want ['PASS'])"
    hands = action.get("hands")
    if not isinstance(hands, list) or len(hands) != n_hands:
        return False, f"hands len {len(hands) if isinstance(hands, list) else '?'} != {n_hands}"
    for h in hands:
        if h != ["PASS"]:
            return False, f"hand {h!r} (want ['PASS'])"
    if action.get("market"):
        return False, f"market not empty: {action.get('market')!r}"
    return True, ""


def run_seed(stage, seed, seat, srv):
    """Drive one episode with the SHIPPED stage (Rust bridge) as `seat`, and at
    every turn also ask the demoted Python fallback and assert it is legal-PASS."""
    ship = load_module(os.path.join(stage, "main.py"), "bandit_ship")
    opp = serve_match.load_agent(OPP)
    js = srv.cmd(f"RESET {seed}")
    fb_bad = []
    sells = 0
    turns = 0
    try:
        while js["step"] < serve_match.EPISODE_STEPS - 1:
            obs_me = serve_match.obs_for(seat, js)
            n_hands = len(js["farms"][seat].get("hands") or [])
            a_bridge = ship.agent(obs_me)
            ok, why = _is_legal_pass(ship._py_fallback(obs_me), n_hands)
            if not ok and len(fb_bad) < 5:
                fb_bad.append((js["step"], why))
            sells += sum(1 for o in (a_bridge.get("market") or [])
                         if o and o[0] == "SELL")
            me_line = serve_match.action_to_line(a_bridge)
            opp_line = serve_match.action_to_line(opp(serve_match.obs_for(1 - seat, js)))
            lines = [me_line, opp_line] if seat == 0 else [opp_line, me_line]
            js = srv.cmd(f"STEP2 {lines[0]}\x1e{lines[1]}")
            if "error" in js:
                raise RuntimeError(js["error"])
            turns += 1
        banks = [float(f["money"]) for f in js["farms"]]
        return {"seed": seed, "seat": seat, "turns": turns,
                "bank": banks[seat], "opp_bank": banks[1 - seat],
                "sells": sells, "fb_bad": fb_bad, "stats": dict(ship.STATS)}
    finally:
        try:
            if ship._BIN.get("proc") is not None:
                ship._BIN["proc"].kill()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="3,4")
    ap.add_argument("--branches", default=None,
                    help="override config.branches (else the config's own)")
    ap.add_argument("--min-bank", type=float, default=20000.0,
                    help="a seat playing the Rust economy must bank at least this")
    args = ap.parse_args()

    # G6.2 first (no binary needed).
    roundtrip_ok = check_config_roundtrip(json.load(open(CONFIG, encoding="utf-8")))

    # The gate MUST NOT pass silently without a binary to compare -- that is
    # exactly how the Rust path and its fallback drifted apart. FAIL, don't skip.
    kbin = _isolated_binary()
    if not os.path.exists(kbin):
        print("FAIL: no isolated binary at %s and KAGG_BIN unset -- the parity "
              "gate cannot run and MUST NOT pass silently. Build one with\n"
              "  CARGO_TARGET_DIR=.local/scratch/bandit_target cargo build "
              "--release --bin kagg --manifest-path rustengine/Cargo.toml\n"
              "or set KAGG_BIN to an isolated kagg[.exe]." % kbin)
        return 1
    if os.path.abspath(kbin) == os.path.abspath(os.path.join(ROOT, "rustengine", "kagg.exe")):
        print("FAIL: KAGG_BIN points at rustengine/kagg.exe (another workload's "
              "live binary). Use an isolated build under .local/scratch.")
        return 1

    # Stage the shipped artefact WITHOUT the musl packaging (no Docker here):
    # write_stage() emits config/branches/base + the driver main.py; drop the
    # isolated binary in beside it so main.py's __file__-relative lookup finds it.
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE, ignore_errors=True)
    build_rust_bandit.write_stage(STAGE, CONFIG, args.branches, None)
    ext = ".exe" if os.name == "nt" else ""
    shutil.copy2(kbin, os.path.join(STAGE, "kagg" + ext))
    print(f"staged {STAGE} with isolated binary {kbin}")

    seeds = [int(s) for s in args.seeds.split(",")]
    srv = serve_match.Serve()
    bad = 0
    try:
        for seed in seeds:
            for seat in (0, 1):
                r = run_seed(STAGE, seed, seat, srv)
                st = r["stats"]
                bridge_ran = st["fallback"] == 0 and st["bridge"] > 0
                economy = r["sells"] > 0 and r["bank"] >= args.min_bank
                fb_ok = not r["fb_bad"]
                ok = bridge_ran and economy and fb_ok
                bad += 0 if ok else 1
                print(f"seed {seed} seat {seat}  turns {r['turns']:3d}  "
                      f"bank {r['bank']:>9.0f}  opp {r['opp_bank']:>9.0f}  "
                      f"sells {r['sells']:4d}  bridge {st['bridge']} "
                      f"fallback {st['fallback']}  {'OK' if ok else 'FAIL'}")
                if not bridge_ran:
                    print(f"    Rust bridge did NOT play every turn "
                          f"(fallback={st['fallback']}, reason={st['reason']!r})")
                if not economy:
                    print(f"    economy check failed (sells={r['sells']}, "
                          f"bank={r['bank']:.0f} < {args.min_bank:.0f})")
                for step, why in r["fb_bad"]:
                    print(f"    fallback NOT legal-PASS @ step {step}: {why}")
    finally:
        srv.close()

    ok_all = bad == 0 and roundtrip_ok
    print(f"\nG6.1 parity: {'PASS' if bad == 0 else f'FAIL ({bad} cells)'}")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
