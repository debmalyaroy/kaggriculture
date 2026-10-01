"""Build the CONFIG-DRIVEN RUST bandit harness artifact -- the daily pipeline
step. Given a config JSON + a branches JSON (+ an optional base tape), assemble a
ready-to-ship artifact directory:

    <out>/  main.py        driver: spawns `kagg mbandit config branches base`,
                           falls back to a LOUD legal-PASS safety net (the
                           economy lives ONLY in the Rust binary)
            kagg           the generic Rust harness binary -- a static
                           x86_64-unknown-linux-musl ELF (Kaggle runs Linux;
                           C2: NEVER the local Windows kagg.exe)
            config.json    active checkpoints + guard-rail toggles/knobs
            branches.json  the searched multi-checkpoint tapes
            base.tape      the base scaffold opening

The Rust binary is GENERIC: it reads config/branches/base from disk at runtime,
so a daily reship is package-only -- no recompile unless the harness LOGIC
(rustengine/src/mbandit.rs) changes, in which case pass --build.

Usage:
  python src/kaggriculture/bandit/build/build_rust_bandit.py \
      --config configs/bandit_config.json \
      --branches models/trackp/multi_ckpt_branches.json \
      --out .local/candidates/mbandit_rc [--base-tape PATH] [--build] [--tar]
"""
from kaggriculture.paths import ROOT
import argparse, json, os, shutil, subprocess, sys

from kaggriculture.bandit.build.build_bandit_harness import parse_tape  # noqa: E402

# The driver: same robust transport as the compiled bandit (spawn, threaded
# reader, per-turn watchdog, H8 re-spawn/MAX_MISSES), but it launches
# `kagg mbandit <config> <branches> <base>`. THE ECONOMY LIVES ONLY IN THE RUST
# BINARY -- that is what we ship AND what we measure. On any failure the driver
# degrades to a LOUD legal-PASS safety net (_py_fallback); it is NOT an
# action-faithful re-implementation of the tape/rails (an earlier build claimed
# it was, and it silently played a DIFFERENT, weaker agent -- bug C1/H3-H7).
# build_bandit_harness.py's Python economy remains a dev/measurement tool, but it
# is deliberately NOT embedded here as the ship fallback.
DRIVER = r'''"""Kaggriculture CONFIG-DRIVEN bandit (Rust mbandit harness + legal-PASS net).

main.py spawns `./kagg mbandit config.json branches.json base.tape` and speaks
one observation JSON per line. The ECONOMY lives ONLY in the Rust `kagg mbandit`
binary -- what we ship equals what we measure. If the bridge cannot run, main.py
degrades to a LOUD legal-PASS safety net (_py_fallback): it plays a legal PASS
every turn so the episode completes, but it does NOT reproduce the tape/rails
and is not the economy. A game that reaches the fallback is a transport failure
to investigate, not a strategy. Never raises.
"""
import json, os, subprocess, sys, threading, time

# Kaggle's agent loader exec's this file WITHOUT __file__ defined (get_last_
# callable). Referencing __file__ at import then raises NameError and the whole
# agent fails to load -> submission ERROR. Resolve the artefact dir defensively.
if "__file__" in globals():
    _HERE = os.path.dirname(os.path.abspath(__file__))
elif os.path.isdir("/kaggle_simulations/agent"):
    _HERE = "/kaggle_simulations/agent"
else:
    _HERE = os.getcwd()
_FILES = ("config.json", "branches.json", "base.tape")
UNIT_OPS = frozenset(("PASS","NORTH","SOUTH","EAST","WEST","PLANT","WATER",
    "HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","CARE",
    "COLLECT_FERTILIZER","DROP","PICKUP","PLACE"))
MARKET_OPS = frozenset(("HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"))
MARKET_NEEDS_ARGS = frozenset(("BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"))
TURN_BUDGET, FIRST_BUDGET = 0.40, 0.60
# H8: a cold/slow turn -- the step-0 branches.json load, one stalled rollout --
# must NOT condemn the whole game to the weak Python fallback. A miss RE-SPAWNS
# the subprocess (a fresh stream cannot desync) and only MAX_MISSES *consecutive*
# misses retire the bridge for good.
MAX_MISSES = 3
_BIN = {"proc": None, "q": None, "dead": False, "started": False, "misses": 0}
STATS = {"bridge": 0, "fallback": 0, "spawn_fail": 0, "timeouts": 0, "reason": ""}


def _binary():
    for root in (_HERE, "/kaggle_simulations/agent"):
        for nm in ("kagg", "kagg.exe"):
            p = os.path.join(root, nm)
            if os.path.exists(p):
                return p
    return None


def _reader(pipe, q):
    try:
        for line in pipe:
            q.append(line)
    except Exception:
        pass
    finally:
        q.append(None)


def _spawn():
    _BIN["started"] = True
    path = _binary()
    files = [os.path.join(_HERE, f) for f in _FILES]
    if not path or not all(os.path.exists(f) for f in files):
        _BIN["dead"] = True; STATS["spawn_fail"] += 1; STATS["reason"] = "missing binary/files"; return
    try:
        if not os.access(path, os.X_OK):
            try: os.chmod(path, 0o755)
            except OSError: pass
        proc = subprocess.Popen([path, "mbandit"] + files,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1, cwd=_HERE)
    except Exception as exc:
        _BIN["dead"] = True; STATS["spawn_fail"] += 1; STATS["reason"] = "spawn: %s" % exc; return
    q = []
    threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
    _BIN["proc"], _BIN["q"] = proc, q


def _retire(reason, hard=False):
    # H8: kill the current subprocess but, unless this is a hard/structural
    # failure or the MAX_MISSES-th CONSECUTIVE miss, leave the bridge eligible
    # to re-spawn on the next turn instead of dying for the rest of the game.
    STATS["reason"] = STATS["reason"] or reason
    proc = _BIN["proc"]; _BIN["proc"] = None; _BIN["q"] = None
    if proc is not None:
        try: proc.kill()
        except Exception: pass
    if hard:
        _BIN["dead"] = True; return
    _BIN["misses"] += 1
    if _BIN["misses"] >= MAX_MISSES:
        _BIN["dead"] = True


def _ask(payload, budget):
    proc, q = _BIN["proc"], _BIN["q"]
    if proc is None or q is None:
        return None
    try:
        proc.stdin.write(payload + "\n"); proc.stdin.flush()
    except Exception:
        _retire("stdin"); return None
    deadline = time.time() + budget
    while not q:
        if time.time() > deadline:
            STATS["timeouts"] += 1; _retire("timeout"); return None
        if proc.poll() is not None:
            _retire("exited"); return None
        time.sleep(0.0005)
    line = q.pop(0)
    if line is None:
        _retire("closed"); return None
    try:
        r = json.loads(line)
    except Exception:
        _retire("malformed"); return None
    return r if isinstance(r, dict) else None


def _clean_unit(op):
    if not isinstance(op, (list, tuple)) or not op:
        return None
    if not isinstance(op[0], str) or op[0] not in UNIT_OPS:
        return None
    out = [op[0]]
    for t in op[1:3]:
        if isinstance(t, bool):
            return None
        if isinstance(t, (int, float)):
            out.append(str(int(t)))
        elif isinstance(t, str):
            out.append(t)
        else:
            return None
    if op[0] in ("PLANT", "PLACE", "PICKUP") and len(out) < 2:
        return None
    return out


def _clean_order(o):
    if not isinstance(o, (list, tuple)) or not o or not isinstance(o[0], str) or o[0] not in MARKET_OPS:
        return None
    if o[0] not in MARKET_NEEDS_ARGS:
        return [o[0]]
    if len(o) < 3 or not isinstance(o[1], str) or not o[1]:
        return None
    try:
        n = int(o[2])
    except (TypeError, ValueError):
        return None
    return [o[0], o[1], str(n)] if n > 0 else None


def _validate(a, n_hands):
    if not isinstance(a, dict):
        return None
    farmer = _clean_unit(a.get("farmer")) or ["PASS"]
    hands = []
    raw = a.get("hands")
    if isinstance(raw, (list, tuple)):
        for op in raw[:n_hands]:
            hands.append(_clean_unit(op) or ["PASS"])
    while len(hands) < n_hands:
        hands.append(["PASS"])
    market = []
    raw = a.get("market")
    if isinstance(raw, (list, tuple)):
        for o in raw:
            if len(market) >= 10:
                break
            c = _clean_order(o)
            if c:
                market.append(c)
    return {"farmer": farmer, "hands": hands, "market": market}


def _n_hands(obs):
    try:
        return len(obs["farms"][int(obs["player"])]["hands"] or [])
    except Exception:
        return 0


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


_WARNED = {"said": False}


def _py_fallback(obs):
    # LOUD legal-PASS safety net (approach B). This is NOT the bandit economy:
    # the tape + rails run ONLY inside `kagg mbandit`, so what we ship equals
    # what we measure. When the Rust bridge cannot run we emit a legal PASS every
    # turn (farmer PASS, each live hand PASS, no market orders) so the episode
    # still completes without an illegal action -- but we do not pretend to
    # reproduce the economy. Reaching here is a transport failure to fix.
    if not _WARNED["said"]:
        _WARNED["said"] = True
        try:
            sys.stderr.write(
                "BANDIT: Rust mbandit bridge unavailable (%s) -- playing "
                "legal-PASS safety net, NOT the economy\n"
                % (STATS.get("reason") or "unknown"))
            sys.stderr.flush()
        except Exception:
            pass
    hands = [["PASS"] for _ in range(_n_hands(obs))]
    return {"farmer": ["PASS"], "hands": hands, "market": []}


def agent(obs, cfg=None):
    try:
        if _BIN["proc"] is None and not _BIN["dead"]:
            _spawn()                                  # H8: (re)spawn on demand
        if not _BIN["dead"]:
            # H8: the generous first-turn budget holds until the bridge has
            # actually answered ONCE, so a cold start or a re-spawn after a
            # miss still gets warm-up headroom -- not just literal step 0.
            budget = FIRST_BUDGET if STATS["bridge"] == 0 else TURN_BUDGET
            reply = _ask(json.dumps(_plain(dict(obs))), budget)
            if reply is not None:
                v = _validate(reply, _n_hands(obs))
                if v is not None:
                    _BIN["misses"] = 0                # H8: count CONSECUTIVE misses only
                    STATS["bridge"] += 1
                    return v
        STATS["fallback"] += 1
        return _py_fallback(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}


# H8: warm the subprocess at MODULE LOAD, off the per-turn clock, so the Rust
# binary's branches.json load happens during agent-load rather than on step 0
# (a large branches.json was blowing the first-turn budget and retiring the
# bridge for the whole game). Guarded: a failed warm-up just leaves the lazy
# (re)spawn in agent() to try again.
try:
    _spawn()
except Exception:
    pass
'''


def _cross_build_musl(target, image="rust:latest"):
    """C2: cross-build the static musl `kagg` ELF inside Docker, exactly as
    scripts/build_trackp_linux.ps1 does. Kaggle is Linux x86_64; a native
    Windows `cargo build` produces a PE that cannot exec on the ladder -- the
    driver watchdog kills it on turn 0 and the weak Python fallback plays the
    whole game (this shipped as v58, score 463)."""
    if shutil.which("docker") is None:
        sys.exit("C2: docker not found. The Kaggle binary must be a Linux musl "
                 "ELF, cross-built in Docker. Install Docker, or run "
                 "scripts/build_trackp_linux.ps1 on a box that has it.")
    mount = ROOT.replace("\\", "/") + ":/work"
    lin_target_dir = "/work/.local/build/rust-linux"
    # --bin kagg ONLY: the crate also hosts offline bins edited in parallel; a
    # shipped artefact must not fail to build because an unrelated target is
    # mid-change (mirrors scripts/build_trackp_linux.ps1).
    script = (
        "set -e\n"
        f"rustup target add {target} >/dev/null 2>&1 || true\n"
        "cd /work/rustengine\n"
        f"cargo build --release --bin kagg --target {target} "
        f"--target-dir {lin_target_dir}\n"
        f"strip {lin_target_dir}/{target}/release/kagg || true\n"
        f"ls -l {lin_target_dir}/{target}/release/kagg\n"
    )
    r = subprocess.run(["docker", "run", "--rm", "-v", mount, "-w", "/work",
                        image, "bash", "-c", script])
    if r.returncode != 0:
        sys.exit("C2: Docker musl cross-build failed (see output above)")


def _normalise_cfg(cfg):
    """The endgame dump is MANDATORY. D6 dispatch is auto-added UNLESS the config
    sets ``d6_optional: true`` (lets a sweep drop D6 to test dispatch structure)."""
    cps = [tuple(c) for c in cfg["checkpoints"]]
    if not cfg.get("d6_optional") and not any(d == 6 for _, d in cps):
        cps = [(146, 6)] + cps
    cfg["checkpoints"] = [list(c) for c in sorted({(s, d) for s, d in cps}, key=lambda c: c[0])]
    grs = cfg.get("guardrails", [])
    if not any(g.get("name") == "endgame" for g in grs):
        grs.append({"name": "endgame", "on": True, "final_dump": 700})
    for g in grs:
        if g.get("name") == "endgame":
            g["on"] = True
    cfg["guardrails"] = grs
    return cfg


def write_stage(out, config_path, branches=None, base_tape=None):
    """Write the shippable stage MINUS the Rust binary: config.json,
    branches.json, base.tape and main.py (driver + legal-PASS fallback ONLY --
    no embedded economy; the economy lives only in `kagg mbandit`). main() adds
    the cross-built musl `kagg`; the parity test drops in an isolated kagg[.exe].
    Returns (cfg, rows, brp, bt)."""
    os.makedirs(out, exist_ok=True)
    cfg = _normalise_cfg(json.load(open(config_path, encoding="utf-8")))
    stage_cfg = {"checkpoints": cfg["checkpoints"], "guardrails": cfg["guardrails"]}
    if cfg.get("tables"):
        # tables (sweep/esc_sweep thresholds, sell_priority, r37 curves, ...) are
        # read by the Rust harness at runtime; propagate them so they are tunable
        # by config. Absent tables leaves the Rust defaults (v581 bit-identical).
        stage_cfg["tables"] = cfg["tables"]
    json.dump(stage_cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    # branches
    brp = branches or cfg["branches"]
    brp = brp if os.path.isabs(brp) else os.path.join(ROOT, brp)
    if not os.path.exists(brp):
        sys.exit(f"branches not found: {brp}")
    shutil.copyfile(brp, os.path.join(out, "branches.json"))
    # base tape -- MUST come from base_tape/config["base_tape"]; the our_route()
    # /v45 fallback is REMOVED (it silently shipped the old v45 economy as v58).
    base_tape = base_tape or cfg.get("base_tape")
    if not base_tape:
        sys.exit("no base tape: set 'base_tape' in the config (or pass "
                 "--base-tape). The our_route()/v45 fallback has been removed.")
    bt = base_tape if os.path.isabs(base_tape) else os.path.join(ROOT, base_tape)
    if not os.path.exists(bt):
        sys.exit(f"base_tape not found: {bt}")
    rows = parse_tape(bt)[:719]
    with open(os.path.join(out, "base.tape"), "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            f = " ".join(str(t) for t in (r.get("farmer") or ["PASS"]))
            h = ";".join(" ".join(str(t) for t in op) for op in (r.get("hands") or []) if op)
            m = ";".join(" ".join(str(t) for t in op) for op in (r.get("market") or []) if op)
            fh.write(f"{f}\t{h}\t{m}\n")
    # main.py = driver + LOUD legal-PASS safety net (NO embedded economy).
    with open(os.path.join(out, "main.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(DRIVER)
    compile(DRIVER, os.path.join(out, "main.py"), "exec")
    return cfg, rows, brp, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "bandit_config.json"))
    ap.add_argument("--branches", default=None, help="override config.branches")
    ap.add_argument("--base-tape", default=None, help="raw .tape base (else our_route scaffold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true", help="cargo build --release first")
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    # 1-3 + main.py: config, branches, base.tape and the driver (legal-PASS net).
    cfg, rows, brp, bt = write_stage(out, a.config, a.branches, a.base_tape)

    # 4. Rust binary -- the SAME generic `kagg`, cross-built as a static
    # x86_64-unknown-linux-musl ELF for the Kaggle Linux runner. C2: we must
    # NEVER package the local Windows kagg.exe -- a PE cannot exec on the ladder
    # and the driver silently plays the weak Python fallback all game (v58, 463).
    target = "x86_64-unknown-linux-musl"
    kbin = os.path.join(ROOT, ".local", "build", "rust-linux",
                        target, "release", "kagg")
    if a.build or not os.path.exists(kbin):
        _cross_build_musl(target)
    if not os.path.exists(kbin):
        sys.exit(f"C2: no Linux musl binary at {kbin}; run with --build (needs Docker)")
    dst = os.path.join(out, "kagg")
    shutil.copyfile(kbin, dst)
    # C2: hard-fail if what we just packaged is not an ELF, so a Windows PE (or a
    # stray native build) can NEVER be shipped to the Linux ladder again.
    with open(dst, "rb") as fh:
        magic = fh.read(4)
    if magic != b"\x7fELF":
        raise RuntimeError(
            f"C2: packaged binary {dst} is not a Linux ELF (magic={magic!r}); "
            "refusing to ship a non-Linux kagg. Cross-build the musl target "
            "with Docker (scripts/build_trackp_linux.ps1 does the same).")
    try:                                   # exec bit for the ladder (Windows FS
        os.chmod(dst, 0o755)               # has none; driver also chmods at spawn)
    except OSError:
        pass

    # 5. main.py was written by write_stage() above (driver + legal-PASS net).
    # The economy is ONLY in the Rust binary; there is deliberately no embedded
    # Python economy fallback (that was bug C1 -- a different, weaker agent).

    n = sum(len(v) for v in json.load(open(os.path.join(out, "branches.json"))).values())
    print(f"built RUST harness artifact -> {out}", flush=True)
    print(f"  dispatch @ {[c[1] for c in cfg['checkpoints']]}; "
          f"rails ON = {[g['name'] for g in cfg['guardrails'] if g.get('on')]}; "
          f"{n} branches; base {len(rows)} rows", flush=True)
    if a.tar:
        import tarfile
        tp = out.rstrip("/\\") + ".tar.gz"
        with tarfile.open(tp, "w:gz") as t:
            for f in os.listdir(out):
                t.add(os.path.join(out, f), arcname=f)
        print(f"  tarball -> {tp}", flush=True)


if __name__ == "__main__":
    main()
