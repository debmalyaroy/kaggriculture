"""Track P suite: contract, isolation, math pins, guard logic, engine parity.

Fast by design: the single episode-playing check (serve==batch banks) uses
the Rust engine (~2 s) and SKIPs when the binary or tapes are missing.
"""
from kaggriculture.paths import ROOT
import ast
import glob
import json
import math
import os
import subprocess
import sys


from kaggriculture.trackp import common, guard, macro, trace_v2  # noqa: E402
from kaggriculture.trackp import planner_template as tpl  # noqa: E402

import numpy as np  # noqa: E402

PASSED = FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name} {detail}")


# ------------------------------------------------------------ 1. contract --
def test_contract():
    for path in [os.path.join(ROOT, "src", "kaggriculture", "trackp", "planner_template.py"),
                 os.path.join(ROOT, "agents", "planner_v0.py")]:
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        imps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imps += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imps.append(node.module)
        check(f"contract imports {os.path.basename(path)}",
              set(imps) <= {"math"}, str(imps))
        for s in ("# --- PARAMS BEGIN ---", "# --- PARAMS END ---",
                  "# --- L1 BEGIN ---", "# --- PRIOR BEGIN ---"):
            check(f"sentinel {s[:20]}.. in {os.path.basename(path)}",
                  s in src)


# ----------------------------------------------------------- 2. isolation --
BANDIT_ROUTE_MODULES = {
    "v22_agent", "refresh_cycle", "routes", "evaluate", "win_metric",
    "train_gates", "identifier", "features", "sameday", "ourgames",
    "stream_hashes", "turn_features", "rust_prerank", "feature_cache",
    "tune", "submit", "autopilot", "pipeline", "opponents", "elo",
    "registry", "policy_dataset", "backfill_traces", "backfill_ingest"}


def test_isolation():
    """The lane rule: nothing the planner runs on may import bandit/route code.

    Scope is deliberately `src/trackp/*.py` -- the LANE's own modules -- and not
    the tree below it. `src/trackp/compiled/verdict.py` imports `win_metric` on
    purpose: it is a JUDGE, not lane code, and CLAUDE.md's measurement
    discipline says verdicts are read in `win_metric`'s currency. Isolating the
    lane's code from the bandit lane and judging both lanes on one instrument
    are the same principle, not opposite ones. Where the lane needs the same
    arithmetic at RUN time it carries its own pinned copy -- see
    `test_lane_copies`.
    """
    bad = []
    for f in glob.glob(os.path.join(ROOT, "src", "kaggriculture", "trackp", "*.py")):
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                base = n.split(".")[0]
                if base in BANDIT_ROUTE_MODULES:
                    bad.append((os.path.basename(f), n))
    check("trackp imports no bandit/route module", not bad, str(bad))


# ---------------------------------- 2b. the lane's own copies do not drift --
def test_lane_copies():
    """The isolation rule forces Track P to carry its own route store reader
    and its own paired test. Duplication is only safe if it is PINNED, so both
    copies are compared here against the originals whenever those import.
    A skip (bandit lane absent) is reported, never silently passed."""
    from kaggriculture.trackp import routes_io, guard as G
    try:
        import kaggriculture.data.routes as R
        import kaggriculture.measure.win_metric as WM
    except Exception as e:                                     # noqa: BLE001
        print(f"  skip lane-copy pins (bandit lane not importable: "
              f"{type(e).__name__})")
        return
    ids = ["mv_Kaileh57_2026-08-29", "a/b?c", "", 17, "x" * 80]
    check("routes_io.route_path == routes.route_path",
          all(routes_io.route_path(i) == R.route_path(i) for i in ids))
    check("routes_io.INDEX == routes.INDEX", routes_io.INDEX == R.INDEX)
    empty = {"routes": {}, "mined": [], "updated": None}
    check("routes_io.load_index shape",
          set(routes_io.load_index()) >= set(empty))
    cases = [([1, 1, 1, 0], [0, 0, 0, 0]), ([0, 0], [0, 0]),
             ([1, 0.5, 0, 1, 1], [0, 1, 0, 0.5, 1]), ([1], [0]),
             ([0] * 8, [1] * 8)]
    for a, b in cases:
        mine, theirs = G.paired_test(a, b), WM.paired_test(a, b)
        same = all(
            (abs(mine[k] - theirs[k]) < 1e-12
             if isinstance(theirs[k], float) else mine[k] == theirs[k])
            for k in ("n_pairs", "score_a", "score_b", "score_diff",
                      "better_a", "better_b", "discordant", "p_value",
                      "significant"))
        check(f"guard.paired_test == win_metric.paired_test {a}/{b}", same,
              f"{mine} vs {theirs}")


# ------------------------------------------------------------ 3. math pins --
def test_price_model():
    # engine ground truth, pinned from `kagg prices` (verified 2026-08-14)
    check("quote at I0 = base",
          all(common.quote(p, 10000) == int(common.MARKET_PARAMS[p][0])
              for p in common.PRODUCTS))
    check("quote WHEAT 9000 = 57", common.quote("WHEAT", 9000) == 57)
    check("quote MELON 4000 = 326", common.quote("MELON", 4000) == 326)
    check("quote floor", common.quote("CARROT", 20000) == 1)
    check("hire fib", [common.fib_hire(i) for i in range(6)]
          == [1, 1, 2, 3, 5, 8])


def test_hinge_1327():
    """Engine 1.32.7 hinge pins (PR #1399), verified against the staged
    interpreter 2026-08-15: u + 8*max(0,u-1)^2, f(T,T)=1."""
    check("hinge at knee = 1", common._shape("hinge", 450, 450) == 1.0)
    check("hinge below knee linear",
          abs(common._shape("hinge", 225, 450) - 0.5) < 1e-12)
    u = 1000 / 200
    check("hinge deep scarcity",
          abs(common._shape("hinge", 1000, 200) - (u + 8 * (u - 1) ** 2))
          < 1e-9)
    # active table today must still be 1.32.6 until the ladder flips
    if common.engine_version() == "1.32.6":
        check("carrot quote pre-flip", common.quote("CARROT", 9000) == 43)
    # the 1.32.7 table itself, engine-state independent
    base, i0, t, bf, bt, af, at = common._MARKET_PARAMS_1327["TOMATO"]
    amp = bt * base / common._shape(bf, t, t)
    raw = base + amp * common._shape(bf, i0 - 9000, t)
    check("tomato 1327 quote pin", common._round_half_even(raw) == 3252,
          raw)
    # forced-1327 planner template agrees with the interpreter pins
    tpl_src = open(os.path.join(ROOT, "src", "kaggriculture", "trackp",
                                "planner_template.py"),
                   encoding="utf-8").read().replace(
        "ENGINE_1327 = False", "ENGINE_1327 = True", 1)
    ns = {}
    exec(compile(tpl_src, "tpl1327", "exec"), ns)  # noqa: S102
    check("template 1327 carrot 531", ns["_quote"]("CARROT", 9000) == 531)
    check("template 1327 egg 758", ns["_quote"]("EGG", 9000) == 758)


def test_town_drain():
    d = common.town_drain_per_day(["YARN_STORE", "BAKERY"])
    # YARN single-product x2 x6 fires = 12 wool + 1 center = 13
    check("drain wool 13", d["WOOL"] == 13, d["WOOL"])
    # BAKERY: EGG+WHEAT x1 x6 = 6 + 1 center = 7
    check("drain egg 7", d["EGG"] == 7, d["EGG"])
    check("drain fert 0", d["FERTILIZER"] == 0)


# ----------------------------------------------------- 4. trace v2 shapes --
def test_trace_shapes():
    check("DIM = fields", trace_v2.DIM == len(trace_v2.FIELDS))
    check("blocks sum", trace_v2.N_GLOBAL + 2 * trace_v2.N_FARM
          + 2 * trace_v2.N_PRIV + 2 * trace_v2.N_ACT == trace_v2.DIM)
    X = np.arange(2 * trace_v2.DIM, dtype=np.float32).reshape(2, -1)
    Y = trace_v2.seat_view(trace_v2.seat_view(X, 1), 1)
    check("seat_view involution", np.array_equal(X, Y))


# ------------------------------------------------------------- 5. macro --
def test_macro():
    check("N_LOGITS", macro.N_LOGITS == sum(k for _, k in macro.HEADS))
    check("FEAT_DIM 41 (32 base + 9 opponent)", macro.FEAT_DIM == 41)
    # synthetic day: 2 wheat plants + 1 hire + heavy selling
    X = np.zeros((48, trace_v2.DIM), dtype=np.float32)
    ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
    X[:24, ix["am_plant_WHEAT"]] = 0
    X[3, ix["am_plant_WHEAT"]] = 2
    X[5, ix["am_hire"]] = 1
    X[:, ix["m_hands"]] = 3
    X[10, ix["am_sell_WHEAT"]] = 10
    X[0, ix["pm_shed_total"]] = 10
    bins = macro.infer_day_bins(X, 0)
    check("mix_WHEAT inferred", bins["mix_WHEAT"] == 1, bins["mix_WHEAT"])
    check("labour bin", bins["labour"] == 2, bins["labour"])  # 3 hands -> bin2
    check("pace bin", bins["pace"] == 2, bins["pace"])  # 10/(10+10)=0.5 -> 2


def test_l1_feature_paths():
    """Runtime obs path == training trace path on a constructed state."""
    day, step = 5, 5 * 24
    obs = {"step": step, "day": day, "hour": 0, "player": 0,
           "farms": [
               {"money": 4000.0, "farmer": [4, 4], "hands": [[4, 4]],
                "hires_today": 0, "unlocked_quadrants": ["NW", "NE"],
                "tiles": [[None] * 10 for _ in range(10)]},
               {"money": 6000.0, "farmer": [4, 4], "hands": [],
                "hires_today": 0, "unlocked_quadrants": ["NW"],
                "tiles": [[None] * 10 for _ in range(10)]}],
           "market": {"inventory": {p: 10000 for p in common.PRODUCTS},
                      "prices": {p: int(common.MARKET_PARAMS[p][0])
                                 for p in common.PRODUCTS}},
           "town": {"unlocked_shops": ["YARN_STORE"]},
           "private": {"shed": {}, "seeds": {}, "inventories": [{}]}}
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": day,
        "watered_today": False, "consecutive_unwatered": 1,
        "yield_units": 1, "max_lifespan_step": 999,
        "fertilized_until_day": -1}
    f_obs = macro.l1_features_obs(obs)

    row = np.zeros(trace_v2.DIM, dtype=np.float32)
    ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
    row[ix["step"]] = step
    row[ix["day"]] = day
    row[ix["m_money"]] = 4000
    row[ix["t_money"]] = 6000
    row[ix["shopn_YARN_STORE"]] = 1
    row[ix["m_hands"]] = 1
    row[ix["m_quads"]] = 2
    row[ix["t_quads"]] = 1
    row[ix["m_crop_WHEAT"]] = 1
    for p in common.PRODUCTS:
        row[ix[f"mpx_{p}"]] = int(common.MARKET_PARAMS[p][0])
    d = common.town_drain_per_day(["YARN_STORE"])
    for p in common.PRODUCTS:
        row[ix[f"drain_{p}"]] = d[p]
    X = np.stack([row])
    f_tr = macro.l1_features_trace_full(X, 0)
    check("feature dims match", len(f_obs) == len(f_tr) == macro.FEAT_DIM,
          f"{len(f_obs)} vs {len(f_tr)}")
    diff = max(abs(a - b) for a, b in zip(f_obs, f_tr))
    check("l1 features obs==trace", diff < 1e-6, f"max diff {diff}")


# ------------------------------------------------------------- 6. guard --
def test_guard():
    rows = [{"generation": "g1", "bank_me": 10.0, "bank_opp": 5.0}] * 25
    g, scope = guard.newest_generation(rows)
    check("gen scope", len(g) == 25)
    check("sign tail all-wins", guard._binom_tail(25, 25) < 1e-6)
    check("sign tail 13/25", guard._binom_tail(13, 25) > 0.05)
    # generation scoping: old positive evidence + new losing generation
    old = [{"generation": "old", "bank_me": 10, "bank_opp": 5}] * 30
    new = [{"generation": "new", "bank_me": 1, "bank_opp": 5}] * 30
    g2, _ = guard.newest_generation(old + new)
    check("newest generation wins scoping",
          all(r["generation"] == "new" for r in g2))


# ------------------------------------------- 7. agent behaves on synth obs --
def test_agent_turn():
    sys.path.insert(0, os.path.join(ROOT, "agents"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "planner_v0_test", os.path.join(ROOT, "agents", "planner_v0.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obs = {"step": 0, "day": 0, "hour": 0, "player": 0,
           "farms": [
               {"money": 3000.0, "farmer": [4, 4], "hands": [],
                "hires_today": 0, "unlocked_quadrants": ["NW"],
                "tiles": [[(None if x < 5 and y < 5 else "LOCKED")
                           for x in range(10)] for y in range(10)]},
               {"money": 3000.0, "farmer": [4, 4], "hands": [],
                "hires_today": 0, "unlocked_quadrants": ["NW"],
                "tiles": [[(None if x < 5 and y < 5 else "LOCKED")
                           for x in range(10)] for y in range(10)]}],
           "market": {"inventory": {p: 10000 for p in common.PRODUCTS},
                      "prices": {p: int(common.MARKET_PARAMS[p][0])
                                 for p in common.PRODUCTS}},
           "town": {"unlocked_shops": []},
           "private": {"shed": {p: 0 for p in common.PRODUCTS},
                       "seeds": {c: 0 for c in common.CROP_NAMES},
                       "inventories": [{}]}}
    a = mod.agent(obs)
    check("agent returns dict", isinstance(a, dict))
    check("farmer is list", isinstance(a.get("farmer"), list))
    check("<=10 market orders", len(a.get("market", [])) <= 10)
    check("hands aligned", len(a.get("hands", [])) == 0)
    # crash-safety: garbage obs must not raise
    b = mod.agent({"step": 1})
    check("fail-soft on garbage obs", b["farmer"] == ["PASS"])


# ----------------------------------------- 8. serve == batch (needs kagg) --
def test_engine_paths():
    tapes = sorted(glob.glob(os.path.join(
        ROOT, "data", "trackp", "anchors", "*.tape")))
    if not os.path.exists(common.KAGG) or len(tapes) < 2:
        print("  SKIP engine paths (kagg or tapes missing)")
        return
    jobs = os.path.join(ROOT, "data", "trackp", "tapes", "_test_jobs.tsv")
    os.makedirs(os.path.dirname(jobs), exist_ok=True)
    with open(jobs, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"-\t{tapes[0]}\t{tapes[1]}\n")
    out = subprocess.run([common.KAGG, "batch", jobs], capture_output=True,
                         text=True, timeout=120)
    p = out.stdout.strip().split("\t")
    check("batch runs", len(p) >= 4 and p[1] != "ERR", out.stdout[:100])
    if len(p) >= 4 and p[1] != "ERR":
        from kaggriculture.trackp.serve_env import ServeEnv
        env = ServeEnv()
        seed = int(p[1])
        obs = env.reset(seed, opp_tape=tapes[1], opp_seat=1)
        with open(tapes[0], encoding="utf-8") as fh:
            lines = fh.read().splitlines()[1:]
        for line in lines:
            env._proc.stdin.write("STEP " + line + "\n")
            env._proc.stdin.flush()
            obs = json.loads(env._proc.stdout.readline())
            if obs.get("done"):
                break
        env.close()
        check("serve == batch banks",
              [obs["farms"][0]["money"], obs["farms"][1]["money"]]
              == [float(p[2]), float(p[3])])


# ------------------------------------------- 8b. relationship miners --
def test_miners():
    from kaggriculture.trackp import insight, twins, verdict_model, winprob  # noqa: F401
    X = np.zeros((720, trace_v2.DIM), dtype=np.float32)
    ix = {f: i for i, f in enumerate(trace_v2.FIELDS)}
    X[:, ix["m_money"]] = 5000
    X[:, ix["m_hands"]] = 2
    for p in common.PRODUCTS:
        X[:, ix[f"mpx_{p}"]] = common.MARKET_PARAMS[p][0]
    X[300, ix["am_sell_WHEAT"]] = 5
    X[200, ix["am_harvest"]] = 1
    X[220, ix["am_plant_WHEAT"]] = 1
    X[:, ix["drain_WOOL"]] = 13
    X[100, ix["at_sell_WOOL"]] = 4
    f = insight.play_features(X)
    for k in ("exec_alpha", "demand_capture", "incgap_WHEAT", "weed_diff",
              "idle_frac", "replant_latency", "sell_chunk_mean"):
        check(f"play feature {k}", k in f)
    check("replant latency 20", f["replant_latency"] == 20.0,
          f["replant_latency"])
    check("demand capture 0 (they sold the wool)",
          f["demand_capture"] == 0.0, f["demand_capture"])
    check("incgap wheat positive", f["incgap_WHEAT"] > 0)


# --------------------------------------------------- 9. forward logits pin --
def test_forward_logits():
    from kaggriculture.trackp import rollouts
    W = {"w1": [[1.0, 0.0], [0.0, 1.0]], "b1": [0.0, 0.0],
         "w2": [[1.0, 0.0], [0.0, 1.0]], "b2": [0.0, 0.0],
         "wo": [[1.0, 1.0]], "bo": [0.5]}
    out = rollouts._forward_logits(W, [0.5, -0.5])
    expect = math.tanh(math.tanh(0.5)) + math.tanh(math.tanh(-0.5)) + 0.5
    check("forward logits", abs(out[0] - expect) < 1e-9)


def main():
    for fn in (test_contract, test_isolation, test_lane_copies,
               test_price_model,
               test_hinge_1327, test_town_drain, test_trace_shapes,
               test_macro,
               test_l1_feature_paths, test_guard, test_agent_turn,
               test_miners, test_engine_paths, test_forward_logits):
        print(fn.__name__)
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            global FAILED
            FAILED += 1
            print(f"  FAIL {fn.__name__} raised {type(e).__name__}: {e}")
    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
