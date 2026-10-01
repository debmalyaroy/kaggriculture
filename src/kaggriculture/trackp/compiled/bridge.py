import json
import os
import subprocess
import sys
import threading
import time

# --------------------------------------------------------------- constants --

# ---------------------------------------------------------------- the search --
#
# Per-turn SEARCH budget handed to `kagg play`, in milliseconds. 0 = the plain
# skeleton policy, which is byte-identical to the Python fallback below.
#
# This is a WALL-CLOCK budget: the searcher checks the clock after every
# rollout, so a slower core buys fewer rollouts rather than more milliseconds.
# What a slower core does stretch is the fixed overhead (json.dumps, the pipe,
# the one rollout in flight when the deadline passes), which is why the
# watchdog below is sized as budget + slack and not as a constant.
#
# TRACKP_BUDGET_MS overrides it and is read by the BINARY too, so a measurement
# can sweep budgets against one packed artefact. Kaggle never sets it, so the
# shipped value is the constant.
#
# 0 -- THE SEARCH IS OFF, AND THAT IS A MEASUREMENT, NOT A DEFAULT.
#
# It shipped at 150 until 2026-09-03 22:40. The base-economy fix of that
# evening (docs/history/trackp-base-economy-2026-09-03.md: land as early as cash
# allows + the mined lean hire ramp) raised the skeleton path so far that the
# searcher, whose value function and knob space were fitted around the OLD
# weak skeleton, now hill-climbs AWAY from a better economy.
#
# Re-measured after the rebuild on the official vendored interpreter, the real
# tarball untarred inside Linux, 5 gauntlet opponents x 4 seeds x both seats =
# 40 cells per row, the SAME cells both times:
#
#     budget   W-D-L    own median   own mean   opp median   share   worst turn
#        0    0-0-40      63,857      64,791     135,090     33.0%     29.5 ms
#      150    0-0-40      48,543      50,036     115,840     29.7%    266.7 ms
#
# Paired cell for cell, the searcher is worth **-14,755 of own bank on the
# mean and is richer on only 11 of the 40 cells (two-sided sign test
# p = 0.0064)**, for exactly the same zero wins. It also BREAKS THE LATENCY
# BAR: 266.7 ms worst under the 6-worker load, past the 250 ms line that keeps
# 4x margin on the 1,000 ms actTimeout.
#
# Three further things budget 0 buys, each of which matters more than a search
# that measures negative:
#
#   1. **A blocked fork/exec on Kaggle costs NOTHING.** At budget 0 the
#      compiled path is byte-identical to the inlined Python fallback
#      (tests/test_compiled_agent.py, the identity gate). If the sandbox
#      refuses to spawn the binary, the agent plays the same game, action for
#      action. Nobody has proven Kaggle permits fork/exec; this makes that
#      question free to ask.
#   2. **It is REPRODUCIBLE.** A wall-clock budget makes the agent
#      non-deterministic, and src/determinism.py's rule is that a paired A/B
#      over a non-reproducible agent is INVALID, not merely noisy. Budget 0
#      returns the same bank for the same seed and seat, every time.
#   3. **34x latency margin under load** (29.5 ms worst over 28,760 timed
#      turns with 6 cells in flight; 9.1 ms on a quiet box) instead of ~4x.
#
# Latency by budget on the rebuilt binary, quiet box, 719 turns/cell:
#
#     budget    mean     p95    worst
#        0      1.06     1.3     9.1
#      100     99.2   103.9   105.2
#      150    148.2   155.6   159.8
#
# The overhead above a non-zero budget is a flat 7-10 ms, NOT a multiple of it:
# `budget_ms` is WALL CLOCK, checked after every rollout, so a slower core buys
# fewer rollouts rather than more milliseconds. That is exactly why raising the
# budget cannot rescue the searcher on Kaggle either -- 1.6 vCPU would run a
# WEAKER search than the 150 ms row above, and that row already measures
# negative here.
#
# DO NOT raise this without a fresh paired sign test against budget 0 on the
# gauntlet. The searcher is not deleted -- `--budget-ms N` and
# TRACKP_BUDGET_MS still turn it on for measurement, and search_selftest still
# proves a zero-budget searcher is action-for-action the skeleton.
SEARCH_BUDGET_MS = 0


def _budget_ms():
    try:
        v = int(os.environ.get("TRACKP_BUDGET_MS", "").strip())
    except (TypeError, ValueError):
        return SEARCH_BUDGET_MS
    return v if v >= 0 else SEARCH_BUDGET_MS


# Per-turn watchdog. actTimeout is 1.0 s. The search is told to finish in
# `budget`; 150 ms of slack covers the straggling rollout, the JSON round trip
# and a Kaggle core that is slower at the fixed work. Floor 0.25 s keeps the
# Phase-A (budget 0) behaviour exactly as it was measured; the 0.60 s ceiling
# keeps >= 40% of actTimeout in hand no matter what budget is configured.
#
# The FIRST turn also pays process start-up, hence a separate, larger budget.
TURN_BUDGET = min(0.60, max(0.25, _budget_ms() / 1000.0 + 0.15))

# The FIRST turn also pays process start-up, so it gets its own, larger budget.
#
# **It used to be 1.50 s and that was a BUG, found 2026-09-03 by finally firing
# the timeout path.** `actTimeout` is 1.0 s. A watchdog longer than the engine's
# own limit cannot protect anything: in the one situation it exists for -- the
# binary spawns but never answers -- it would sit on turn 1 for 1.5 s and hand
# the engine a turn it had already timed out, converting a harmless fallback
# into a possible forfeit. Measured directly: breakage case D (a `kagg` that
# reads the request and sleeps) produced a worst turn of **1,503 ms**.
#
# The old value rested on an ASSUMPTION written into the Phase-A doc -- "the
# engine has not started the clock on turn 1 in the same way" -- that was never
# measured. Nothing should rest on that.
#
# 0.50 s is sized from the measurement instead: spawn + the first day plan is
# under 30 ms on Linux with 6 cells in flight (the worst single turn of all
# 28,760 in the budget-0 gauntlet, first turns included), so this is ~16x the
# observed cost, ~8x on a core twice as slow, and still leaves HALF of
# actTimeout in hand.
FIRST_BUDGET = 0.50

# Every op the interpreter recognises. Anything else is a silent no-op on the
# engine side, which is exactly why it must be caught HERE: an illegal action
# looks like laziness, not like a bug.
UNIT_OPS = frozenset((
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST",
    "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE",
    "COLLECT_FERTILIZER", "DROP", "PICKUP", "PLACE",
))
MARKET_OPS = frozenset((
    "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL",
))
MARKET_NEEDS_ARGS = frozenset(("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"))
MAX_MARKET_ORDERS = 10

# Diagnostics the harness reads. Never used for a decision.
STATS = {
    "turns": 0,          # agent() calls
    "bridge": 0,         # answered by the compiled binary
    "fallback": 0,       # answered by the pure-Python planner
    "spawn_fail": 0,
    "timeouts": 0,
    "bad_reply": 0,
    "repairs": 0,        # actions the validator had to fix
    "worst_ms": 0.0,
    "reason": "",
    "budget_ms": 0,      # the search budget this episode actually ran at
}

_BIN = {"proc": None, "q": None, "dead": False, "started": False}


# ----------------------------------------------------------------- transport --

def _binary_path():
    """The compiled searcher, found RELATIVE TO THIS FILE.

    A submission tarball is unpacked into /kaggle_simulations/agent/, but the
    path is not contractual and hard-coding it makes the artefact untestable
    anywhere else. __file__ is authoritative in both places; the documented
    Kaggle path is only a fallback probe.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("kagg", "kagg.exe")
    roots = (here, "/kaggle_simulations/agent")
    for root in roots:
        for name in names:
            p = os.path.join(root, name)
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
    """Start the searcher once. Any failure marks the bridge dead for good."""
    _BIN["started"] = True
    path = _binary_path()
    if not path:
        _BIN["dead"] = True
        STATS["spawn_fail"] += 1
        STATS["reason"] = "binary not found"
        return
    try:
        if not os.access(path, os.X_OK):
            try:
                os.chmod(path, 0o755)
            except OSError:
                pass
        budget = _budget_ms()
        STATS["budget_ms"] = budget
        proc = subprocess.Popen(
            [path, "play", "--budget-ms", str(budget)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            cwd=os.path.dirname(os.path.abspath(path)),
        )
    except Exception as exc:                                  # noqa: BLE001
        _BIN["dead"] = True
        STATS["spawn_fail"] += 1
        STATS["reason"] = "spawn: %s" % (exc,)
        return
    q = []
    t = threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True)
    t.start()
    _BIN["proc"] = proc
    _BIN["q"] = q


def _kill(reason):
    """Retire the bridge permanently.

    A timed-out turn leaves a reply in flight, and without a request/response
    tag there is no way to re-pair the stream afterwards -- a late answer would
    silently become the NEXT turn's action. So one timeout retires the bridge
    for the rest of the episode and every later turn is planned in Python. That
    is a deliberate trade: a correct weaker agent over a fast desynchronised
    one.
    """
    _BIN["dead"] = True
    STATS["reason"] = STATS["reason"] or reason
    proc = _BIN["proc"]
    _BIN["proc"] = None
    _BIN["q"] = None
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:                                         # noqa: BLE001
        pass


def _ask(payload, budget):
    """One request/response round trip, or None."""
    proc = _BIN["proc"]
    q = _BIN["q"]
    if proc is None or q is None:
        return None
    try:
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()
    except Exception:                                         # noqa: BLE001
        _kill("stdin write failed")
        return None
    deadline = time.time() + budget
    while not q:
        if time.time() > deadline:
            STATS["timeouts"] += 1
            _kill("turn exceeded %.2fs" % budget)
            return None
        if proc.poll() is not None:
            _kill("process exited rc=%s" % proc.poll())
            return None
        time.sleep(0.0005)
    line = q.pop(0)
    if line is None:
        _kill("stdout closed")
        return None
    try:
        reply = json.loads(line)
    except Exception:                                         # noqa: BLE001
        STATS["bad_reply"] += 1
        _kill("malformed reply")
        return None
    if not isinstance(reply, dict):
        STATS["bad_reply"] += 1
        _kill("reply is not an object")
        return None
    return reply


# ----------------------------------------------------------------- validator --

def _clean_unit(op):
    """One unit action, repaired to something the interpreter accepts."""
    if not isinstance(op, (list, tuple)) or not op:
        return None
    name = op[0]
    if not isinstance(name, str) or name not in UNIT_OPS:
        return None
    out = [name]
    for tok in op[1:3]:
        if isinstance(tok, bool):
            return None
        if isinstance(tok, (int, float)):
            out.append(str(int(tok)))
        elif isinstance(tok, str):
            out.append(tok)
        else:
            return None
    if name in ("PLANT", "PLACE", "PICKUP") and len(out) < 2:
        return None
    return out


def _clean_order(order):
    if not isinstance(order, (list, tuple)) or not order:
        return None
    name = order[0]
    if not isinstance(name, str) or name not in MARKET_OPS:
        return None
    if name not in MARKET_NEEDS_ARGS:
        return [name]
    if len(order) < 3:
        return None
    item = order[1]
    if not isinstance(item, str) or not item:
        return None
    try:
        n = int(order[2])
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return [name, item, str(n)]


def _validate(action, n_hands):
    """Make ANY reply legal, or return None if it cannot be salvaged.

    Enforced here, not trusted from the binary: `hands` aligned POSITIONALLY
    with farms[me]["hands"], at most 10 market orders, known op names only.
    """
    if not isinstance(action, dict):
        return None
    repairs = 0

    farmer = _clean_unit(action.get("farmer"))
    if farmer is None:
        farmer = ["PASS"]
        repairs += 1

    raw = action.get("hands")
    hands = []
    if isinstance(raw, (list, tuple)):
        for op in raw[:n_hands]:
            c = _clean_unit(op)
            if c is None:
                c = ["PASS"]
                repairs += 1
            hands.append(c)
    while len(hands) < n_hands:
        hands.append(["PASS"])
        repairs += 1

    market = []
    raw = action.get("market")
    if isinstance(raw, (list, tuple)):
        for order in raw:
            if len(market) >= MAX_MARKET_ORDERS:
                repairs += 1
                break
            c = _clean_order(order)
            if c is None:
                repairs += 1
                continue
            market.append(c)

    STATS["repairs"] += repairs
    return {"farmer": farmer, "hands": hands, "market": market}


# --------------------------------------------------------------------- agent --

def _n_hands(obs):
    try:
        me = int(obs["player"])
        return len(obs["farms"][me]["hands"] or [])
    except Exception:                                         # noqa: BLE001
        return 0


def _plain(o):
    """kaggle-environments hands agents a Struct (a dict subclass). json.dumps
    copes, but a hostile/odd container must not be able to raise here."""
    if isinstance(o, dict):
        return dict((str(k), _plain(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _beacon(tag):
    """One line on stderr, which kaggle-environments captures into the
    episode's per-agent `logs`.

    This is the ONLY way to learn from the ladder whether the subprocess
    transport actually worked there. A blocked fork/exec would look exactly
    like a normal game -- the Python fallback covers for it and the agent
    still plays a legal episode -- so without a beacon we would ship a
    compiled agent and never know it ran interpreted. Two lines per episode,
    never on the hot path.
    """
    try:
        sys.stderr.write("TRACKP %s %s\n" % (tag, json.dumps(STATS)))
        sys.stderr.flush()
    except Exception:                                         # noqa: BLE001
        pass


def agent(obs, cfg=None):
    t0 = time.time()
    STATS["turns"] += 1
    action = None
    try:
        if not _BIN["started"]:
            _spawn()
        if not _BIN["dead"]:
            try:
                payload = json.dumps(obs, separators=(",", ":"))
            except Exception:                                 # noqa: BLE001
                payload = json.dumps(_plain(obs), separators=(",", ":"))
            budget = FIRST_BUDGET if STATS["bridge"] == 0 else TURN_BUDGET
            reply = _ask(payload, budget)
            if reply is not None:
                action = _validate(reply, _n_hands(obs))
        if action is not None:
            STATS["bridge"] += 1
        else:
            STATS["fallback"] += 1
            action = _validate(_fallback_agent(obs, cfg), _n_hands(obs))
    except Exception as exc:                                  # noqa: BLE001
        # Belt and braces. agent() raising costs the whole submission: the
        # upload Validation Episode marks it Error and the slot is spent.
        STATS["reason"] = STATS["reason"] or ("agent: %s" % (exc,))
        action = None
    if action is None:
        action = {"farmer": ["PASS"], "hands": [["PASS"]] * _n_hands(obs),
                  "market": []}
    dt = (time.time() - t0) * 1000.0
    if dt > STATS["worst_ms"]:
        STATS["worst_ms"] = dt
    if STATS["turns"] == 1:
        _beacon("start")
    elif int(_get_step(obs)) >= 717:
        _beacon("end")
    return action


def _get_step(obs):
    try:
        return obs["step"]
    except Exception:                                         # noqa: BLE001
        return 0


def _selftest():
    """`python main.py --selftest` -- prove the binary answers, without an
    engine. Prints the day-0 hour-0 action and the transport verdict."""
    obs = json.loads(sys.stdin.read())
    a = agent(obs)
    print(json.dumps(a))
    print(json.dumps(STATS), file=sys.stderr)
    return 0 if STATS["fallback"] == 0 else 1
