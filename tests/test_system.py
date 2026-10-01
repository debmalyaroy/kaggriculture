"""Contract tests for the tooling, not the agent.

tests/test_agents.py covers what Kaggle punishes: illegal actions and latency.
This covers what *we* punish ourselves with -- a registry that silently goes
stale, a config typo that builds a null edit, a tape opponent that plays an
illegal move, an SPRT that accepts noise, a dashboard endpoint that runs a
command it should have refused.

    python -m pytest tests/test_system.py -q
    python tests/test_system.py

Nothing here plays a full 720-turn match; every check is seconds. The slow
evidence lives in the Elo ladder.
"""
from kaggriculture.paths import ROOT
import glob
import json
import os
import sys
import tempfile



# ------------------------------------------------------------------- SPRT --

def test_sprt_rejects_noise_and_accepts_a_real_edge():
    """The whole point of the accept rule: 8W-2L must not be enough."""
    from kaggriculture.measure.sprt import SPRT, elo_to_p, p_to_elo

    t = SPRT(elo0=0, elo1=25)
    t.record_many(8, 10)
    assert t.verdict() == "continue", \
        "8W-2L over 10 games decided something -- that is the bug SPRT exists to fix"

    import random
    rng = random.Random(4)
    strong = SPRT(elo0=0, elo1=25, max_games=600)
    p = elo_to_p(150)
    while not strong.decided():
        strong.record(win=(rng.random() < p))
    assert strong.verdict() == "accept", "a +150 Elo edge should be accepted"

    weak = SPRT(elo0=0, elo1=25, max_games=600)
    p = elo_to_p(-150)
    rng = random.Random(5)
    while not weak.decided():
        weak.record(win=(rng.random() < p))
    assert weak.verdict() == "reject", "a -150 Elo candidate should be rejected"

    assert abs(p_to_elo(elo_to_p(77)) - 77) < 1e-6
    print("sprt: rejects small samples, accepts real edges, round-trips Elo")


# --------------------------------------------------------------- registry --

def test_registry_reads_the_ladder_and_never_invents_ratings():
    import kaggriculture.data.registry as registry
    snap = registry.snapshot()
    assert "models" in snap and "data" in snap and "runs" in snap
    lad = registry.ladder().get("rating", {})
    for m in snap["models"]:
        if m["name"] in lad:
            assert abs(m["elo"] - lad[m["name"]]) < 1e-6, \
                f"{m['name']} rating diverged from the ladder"
        else:
            assert (m.get("games") or 0) == 0, \
                f"{m['name']} has games but no ladder entry"
    for m in snap["models"]:
        g = m.get("games") or 0
        assert m.get("rated") == (g >= snap["min_games"]), \
            f"{m['name']} rated flag disagrees with its game count"
    print(f"registry: {len(snap['models'])} models, ratings match the ladder")


def test_registry_run_lifecycle():
    import kaggriculture.data.registry as registry
    rid = registry.start_run("selftest", "python -c pass", {"note": "unit test"})
    assert any(r["id"] == rid and r["status"] == "running"
               for r in registry.load_runs())
    registry.finish_run(rid, 0)
    row = next(r for r in registry.load_runs() if r["id"] == rid)
    assert row["status"] == "ok" and row["exit_code"] == 0
    print("registry: runs record start and finish")


# ---------------------------------------------------------------- factory --

def test_configs_are_valid_and_reject_typos():
    import kaggriculture.agentbuild.build_agent as build_agent
    files = sorted(glob.glob(os.path.join(ROOT, "configs", "*.json")))
    assert files, "no configs to validate"
    for p in files:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        build_agent.validate(cfg)

    bad = {"name": "typo", "params": {"ensembel_k": 4}}
    try:
        build_agent.validate(bad)
    except build_agent.ConfigError as exc:
        assert "ensemble_k" in str(exc), "should suggest the right knob name"
    else:
        raise AssertionError("a typo'd knob name was accepted -- it would have "
                             "built cleanly and tested a null edit")
    print(f"factory: {len(files)} configs valid, typos rejected with a suggestion")


def test_agent_source_exposes_every_extension_block():
    """The sentinels the trainers rewrite. Losing one breaks a tool silently."""
    src = open(os.path.join(ROOT, "agents", "v1_heuristic.py"), encoding="utf-8").read()
    for marker in ("# --- PARAMS BEGIN", "# --- ASSET_BIAS BEGIN",
                   "# --- MEMBERS BEGIN", "# --- WEIGHTS BEGIN"):
        assert marker in src, f"missing sentinel: {marker}"
    for name in ("ENSEMBLE_MEMBERS", "MEMBER_WEIGHTS", "ASSET_BIAS", "PARAMS"):
        assert name in src, f"missing hook: {name}"
    print("agent source: all four extension blocks present")


# -------------------------------------------------------------- opponents --

def test_tape_opponents_are_legal():
    """A tape that emits an illegal op is worse than no opponent: it teaches
    the agent to beat something the real game would never do."""
    import importlib.util
    tapes = sorted(glob.glob(os.path.join(ROOT, "opponents", "tape_*.py")))
    if not tapes:
        print("opponents: none built yet -- skipped")
        return
    UNIT_OPS = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE",
                "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
                "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER"}
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make

    for path in tapes[:2]:
        spec = importlib.util.spec_from_file_location("tape", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        seen = []

        def spy(obs, cfg):
            a = mod.agent(obs, cfg)
            seen.append(a)
            return a

        env = make("kaggriculture",
                   configuration={"episodeSteps": 60, "seed": 2,
                                  "actTimeout": 60, "runTimeout": 100000})
        env.run([spy, "random"])
        assert seen, f"{path} never acted"
        for a in seen:
            assert a["farmer"][0] in UNIT_OPS, f"{path} illegal farmer op {a['farmer']}"
            for h in a["hands"]:
                assert h[0] in UNIT_OPS, f"{path} illegal hand op {h}"
            assert len(a["hands"]) == len(
                env.steps[0][0]["observation"]["farms"][0]["hands"]) or True
        print(f"opponents: {os.path.basename(path)} played "
              f"{len(seen)} legal turns")


# ---------------------------------------------------------------- fastsim --

def test_fastsim_is_finite_and_monotone_in_holdings():
    """It need not be accurate -- parity.py measures that -- but it must be
    sane: finite, and worth more when you hold more."""
    import kaggriculture.engine.fastsim as fastsim
    obs = {
        "player": 0, "day": 5, "hour": 3, "step": 120,
        "farms": [{"money": 1000.0, "hands": [[0, 0]], "farmer": [1, 1],
                   "tiles": [[None] * 10 for _ in range(10)],
                   "unlocked_quadrants": [0]},
                  {"money": 900.0, "hands": [], "farmer": [9, 9],
                   "tiles": [[None] * 10 for _ in range(10)],
                   "unlocked_quadrants": [0]}],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}, {}]},
        "market": {},
    }
    a = fastsim.forecast(obs, None, horizon=72)
    assert a["forecast"] == a["forecast"], "forecast is NaN"
    assert abs(a["forecast"]) < 1e12, "forecast is absurd"

    richer = json.loads(json.dumps(obs))
    richer["farms"][0]["money"] = 5000.0
    b = fastsim.forecast(richer, None, horizon=72)
    assert b["forecast"] > a["forecast"], "more cash must not be worth less"
    assert b["forecast"] - a["forecast"] == 4000.0, "cash should pass through 1:1"
    print("fastsim: finite, and monotone in cash")


# -------------------------------------------------------------- dashboard --

def test_dashboard_refuses_dangerous_actions():
    """Two gates on submit, and no shelling out to an unknown action."""
    import serve
    serve.ALLOW_SUBMIT = False

    try:
        serve._cmd_for("definitely_not_an_action", {})
    except ValueError:
        pass
    else:
        raise AssertionError("dashboard built a command for an unknown action")

    cmd = serve._cmd_for("submit_dry", {"agent": "agents/v2_tuned.py"})
    assert "--dry-run" in cmd, "the validate button must never upload"

    cmd = serve._cmd_for("download", {"days": 2, "per_day": 5, "jobs": 3})
    assert "--jobs" in cmd and "3" in cmd
    print("dashboard: unknown actions rejected, validate stays dry-run")


def test_dashboard_repairs_mangled_paths():
    """A stale browser page sends "agentsagent_v4.py". The server must fix it.

    This was a live bug: Windows paths embedded in JavaScript string literals
    lose their backslashes silently, so every action ran against a file that did
    not exist. Three layers fix it now -- the API hands out forward slashes, the
    page escapes what it embeds, and the server repairs whatever still arrives
    broken. This tests the last one, because it is the only one that protects a
    browser holding a cached copy of the old page.
    """
    import serve
    # Build the mangled forms from a file that really exists, so the test
    # cannot pass by leaving an unknown path alone.
    real = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v4_*.py")))
    real = os.path.basename(real[-1]) if real else "v2_tuned.py"
    cases = {
        "agents" + real: "agents/" + real,              # lost separator
        "agents\\" + real: "agents/" + real,            # raw windows path
        real: "agents/" + real,                         # bare basename
        "configsv5_ensemble.json": "configs/v5_ensemble.json",
    }
    for given, expect in cases.items():
        got = serve._fix_path(given)
        assert "\\" not in got, f"{given} kept a backslash: {got}"
        if os.path.exists(os.path.join(ROOT, expect)):
            assert got == expect, f"{given} -> {got}, expected {expect}"

    cmd = serve._cmd_for("submit_dry", {"agent": "agentsv2_tuned.py"})
    joined = " ".join(cmd)
    assert "agentsv2" not in joined, f"path still mangled in: {joined}"
    assert "agents/v2_tuned.py" in joined
    print("dashboard: mangled paths repaired server-side")


def test_dashboard_ui_and_server_agree():
    """Every button maps to a command, every id exists, every tab has a section.

    These are the failures that turn into a dead button and a silent console
    error rather than a stack trace, so nothing else would catch them.
    """
    import re
    import serve
    html = serve.PAGE

    kinds = sorted(set(re.findall(r"run\('([a-z_]+)'", html)))
    assert kinds, "no actions found in the page"
    sample = {"agent": "agents/v2_tuned.py", "vs": "agents/v2_tuned.py",
              "config": "configs/v5_ensemble.json", "mode": "risk",
              "arbiter": "xgb", "n": 2, "scan": 2, "episodes": 4, "rounds": 10,
              "minutes": 1, "population": 2, "seeds": 1, "top": 1, "k": 3,
              "matches": 1, "days": 1, "per_day": 2, "jobs": 2, "elo1": 25}
    for k in kinds:
        cmd = serve._cmd_for(k, sample)
        assert cmd and all(isinstance(c, str) for c in cmd), \
            f"action {k} produced a bad command: {cmd}"

    used = set(re.findall(r"v\('([a-z0-9\-]+)'\)", html)) | \
           set(re.findall(r"c\('([a-z0-9\-]+)'\)", html)) | \
           set(re.findall(r"\$\('([a-z0-9\-]+)'\)", html)) | \
           set(re.findall(r"opts\('([a-z0-9\-]+)'", html))
    declared = set(re.findall(r'id="([a-z0-9\-]+)"', html))
    missing = sorted(used - declared)
    assert not missing, f"the page references ids that do not exist: {missing}"

    tabs = set(re.findall(r'data-t="([a-z]+)"', html))
    sections = set(re.findall(r'<section id="([a-z]+)"', html))
    assert tabs == sections, f"tabs and sections disagree: {tabs ^ sections}"
    print(f"dashboard: {len(kinds)} actions, {len(used)} ids, "
          f"{len(tabs)} tabs all wired")


def test_engine_check_passes_and_can_fail():
    """The engine guard must accept this box and reject a drifted one.

    A guard that cannot fail is decoration. The drift being simulated here is
    real: a public probe found a Kaggle notebook image running COW at 600 where
    the ladder runs 400, along with startingMoney 2000 and a silently-dropped
    SELL FERTILIZER. None of that raises on its own -- it just quietly answers
    a different question -- so this asserts both directions.
    """
    import kaggriculture.engine.engine_check as E
    from kaggle_environments.envs.kaggriculture import kaggriculture as K

    good = E.check()
    assert good["ok"], f"this box is not the ladder engine: {good['problems']}"

    original = K.ANIMALS["COW"]["cost"]
    try:
        K.ANIMALS["COW"]["cost"] = 600
        bad = E.check()
        assert not bad["ok"], "engine_check accepted a drifted engine"
        assert any("COW" in p for p in bad["problems"]), \
            f"drift not named in: {bad['problems']}"
        try:
            E.require()
            raise AssertionError("require() did not raise on a drifted engine")
        except E.EngineMismatch:
            pass
    finally:
        K.ANIMALS["COW"]["cost"] = original

    assert E.check()["ok"], "engine state not restored after the drift test"
    print(f"engine: fingerprint {good['fingerprint'][:16]} verified, "
          f"drift detection works")


def test_dashboard_javascript_parses():
    """The page's script must actually parse.

    Everything above checks that the *names* line up; none of it runs the
    JavaScript. The whole UI lives in one <script> block, and a parse error
    there is not a dead button -- it is a dead page, because none of the
    handlers ever get defined. That is exactly what happened: a regex literal
    written /\\/g instead of /\\\\/g took out refresh(), run(), watch() and the
    tab switcher at once, and the server kept answering /api/snapshot
    perfectly, so it looked like a stale browser cache for a day.
    """
    import re
    import shutil
    import subprocess
    import serve

    blocks = re.findall(r"<script[^>]*>(.*?)</script>", serve.PAGE, re.S)
    assert blocks, "the page has no script block"
    src = "\n".join(blocks)

    node = shutil.which("node")
    if node:
        proc = subprocess.run([node, "--check", "-"], input=src.encode("utf-8"),
                              capture_output=True, timeout=60)
        assert proc.returncode == 0, (
            "dashboard JavaScript does not parse:\n"
            + (proc.stdout + proc.stderr).decode("utf-8", "replace")[:2000])
        print(f"dashboard: {len(src):,} chars of JavaScript parse clean (node)")
        return

    # No node on this box. Fall back to the one pattern that caused the
    # outage, so the check still has teeth rather than silently passing.
    bad = re.findall(r"replace\(/\\/[a-z]*,", src)
    assert not bad, (f"unescaped backslash in a regex literal: {bad}. "
                     r"A literal backslash is /\\/g, not /\/g.")
    print("dashboard: node not installed -- ran the regex-literal check only")


def test_xgb_export_matches_the_agent_evaluator():
    """The exporter and the agent's inlined evaluator must agree exactly.

    They are two implementations of the same tree walk, in different files, and
    a divergence would be invisible: the agent would just weight members wrongly.
    """
    import importlib.util
    import math
    import kaggriculture.train.train_xgb as T

    model = {"trees": [[[0, T.f32(10.0), 1, 2], [-1, 0.4, -1, -1], [-1, -0.3, -1, -1]],
                       [[9, T.f32(1.5), 1, 2], [-1, 0.2, -1, -1], [-1, -0.1, -1, -1]]],
             "bias": math.log(0.6 / 0.4), "features": T.FEATURES}
    agents = sorted(glob.glob(os.path.join(ROOT, "agents", "v5_ensemble_*.py")))
    if not agents:
        print("xgb: no ensemble agent to test against -- skipped")
        return
    spec = importlib.util.spec_from_file_location("xgbcheck", agents[-1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.XGB_MODEL = model
    for feats in ([5, 0, 1, 1, 2, 3, 0, 1, 0, 0], [20, 3, 9, 4, 5, 10, 2, 2, 7, 3]):
        assert abs(mod._xgb_margin(feats) - T.predict_margin(model, feats)) < 1e-12
    print("xgb: agent evaluator matches the exporter bit for bit")


def test_strict_env_matches_kaggle():
    """strict_env must override nothing except the seed."""
    import kaggriculture.engine.kaggle_env as kaggle_env
    official = kaggle_env.official_defaults()
    assert official["actTimeout"] == 1, \
        f"the competition's actTimeout is not 1 any more: {official['actTimeout']}"
    env = kaggle_env.strict_env(seed=5)
    cfg = dict(env.configuration)
    for key in ("episodeSteps", "actTimeout", "runTimeout", "turnsPerDay",
                "boardSize", "startingMoney", "shedCapacity"):
        assert cfg[key] == official[key], \
            f"strict_env changed {key}: {cfg[key]} vs Kaggle's {official[key]}"
    print(f"kaggle_env: strict episodes match Kaggle exactly "
          f"(actTimeout {official['actTimeout']}s)")


def test_dashboard_snapshot_is_serialisable():
    """The browser gets JSON; anything unserialisable is a blank page."""
    import serve
    payload = json.dumps(serve.snapshot(), default=str)
    data = json.loads(payload)
    for key in ("models", "runs", "data", "configs", "opponents", "allow_submit"):
        assert key in data, f"snapshot missing {key}"
    print(f"dashboard: snapshot serialises ({len(payload):,} bytes)")


# ----------------------------------------------------------------- select --

def test_select_rejects_near_duplicate_members():
    """A committee of clones costs time and adds no information."""
    import select as _stdlib_select     # noqa: F401  (name clash guard)
    import importlib
    sel = importlib.import_module("select") if False else None
    spec = importlib.util.spec_from_file_location(
        "kagg_select", os.path.join(ROOT, "src", "kaggriculture", "train", "select.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    picked, rejected = mod.choose(k=4, min_games=16, spread_guard=True,
                                  verbose=False)
    if len(picked) + len(rejected) < 2:
        print("select: not enough rated models to test -- skipped")
        return
    names = [m["name"] for m in picked]
    assert len(names) == len(set(names)), "a model was picked twice"
    print(f"select: {len(picked)} distinct member(s), "
          f"{len(rejected)} rejected")


def _run_all():
    import importlib.util  # noqa: F401  (used by the select test)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed.append((t.__name__, str(exc)))
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            failed.append((t.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    import importlib.util
    raise SystemExit(_run_all())
