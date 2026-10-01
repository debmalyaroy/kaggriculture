"""Turn downloaded top-20 replays into opponents you can actually play against.

Until now every test was self-play: our agent against our own earlier agents.
That measures progress relative to ourselves and is blind to how the people
above us on the leaderboard actually play. This builds opponents out of their
recorded games.

Two kinds, and the difference matters
-------------------------------------
**tape** -- replay the recorded 720 actions in order, with a repair layer for
states where the recording no longer applies (the tile it wanted to harvest is
empty, the animal it wanted to feed is gone). This is exactly the architecture
of the 2600-Elo public agent we decoded: an open-loop trace plus a small
corrective policy. It is faithful to what that player did and costs nothing to
build -- but it does not *react*, so beating a tape is necessary, not
sufficient.

**clone** -- fit our own heuristic's parameters to best reproduce their recorded
decisions, giving an opponent that responds to states the tape never saw. Only
as expressive as our heuristic, so it cannot imitate a strategy our policy
cannot represent, but it will punish you for deviating in ways a tape cannot.

Build tapes first, play them, then build clones for anything that beats you.

    python -m kaggriculture.measure.opponents --build-tapes             # from data/episodes
    python -m kaggriculture.measure.opponents --build-clones --top 5
    python -m kaggriculture.measure.opponents --list
    python -m kaggriculture.measure.opponents --play agent_v4_optimal_*.py --n 4
"""
from kaggriculture.paths import ROOT
import argparse
import base64
import glob
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.registry as registry  # noqa: E402
import kaggriculture.pipeline.progress as pr  # noqa: E402

OPP_DIR = os.path.join(ROOT, "opponents")
REPLAYS = os.path.join(ROOT, "data", "episodes")


# ------------------------------------------------------------------ replays --

def find_replays(limit=None):
    hits = sorted(glob.glob(os.path.join(REPLAYS, "**", "*.json"), recursive=True))
    return hits[:limit] if limit else hits


def replay_meta(path, data=None):
    """Score, seat and length of a replay, without loading it twice.

    `data`: the already-parsed replay dict. A 15-30 MB replay costs seconds
    to parse; callers touching meta + actions + features must parse once and
    pass it here, or the parse dominates the whole ingest (sameday measured
    ~15 s/episode from five redundant loads, 2026-08-12)."""
    try:
        d = data if data is not None else json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    steps = d.get("steps") or []
    rewards = d.get("rewards") or []
    if not steps or len(rewards) < 2:
        return None
    try:
        r0, r1 = float(rewards[0] or 0), float(rewards[1] or 0)
    except (TypeError, ValueError):
        return None
    winner = 0 if r0 >= r1 else 1
    return {
        "path": path, "episode": os.path.basename(path)[:-5],
        "steps": len(steps), "rewards": [r0, r1], "winner": winner,
        "margin": abs(r0 - r1),
        "config": d.get("configuration", {}),
    }


def extract_actions(path, seat, data=None):
    """The action one seat played, turn by turn.

    `data`: optional pre-parsed replay dict -- see replay_meta.

    **The replay is offset by one.** `steps[i][p]["action"]` is the action that
    *produced* state i, not the action taken from it -- index 0 always holds a
    placeholder PASS and index 1 holds the opening move. Reading index i as
    turn i shifts the whole recording a turn early: every WATER fires before
    its plant exists and a taped $130k game replays as $24k, which is exactly
    what the first version of this tool measured.
    """
    if data is not None:
        d = data
    else:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    steps = d.get("steps") or []
    blank = {"farmer": ["PASS"], "hands": [], "market": []}
    out = []
    for i in range(len(steps)):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        a = nxt[seat].get("action") if (nxt and seat < len(nxt)) else None
        out.append(a if isinstance(a, dict) else dict(blank))
    return out


# --------------------------------------------------------------------- tape --

_TAPE_TEMPLATE = '''"""Replay-tape opponent: {label}

Built by src/kaggriculture/measure/opponents.py from episode {episode} (seat {seat}).
Final banks in the source game: {rewards}; this seat {result}.

An open-loop trace of {steps} turns plus a repair layer. The trace is the
player's recorded actions; the repair layer keeps the agent legal when the live
state has drifted from the recording -- units are elsewhere, a tile is already
harvested, the shed is full. This is the same architecture as the 2600-Elo
public submission, which is an open-loop tape with a small corrective policy.

It does not react to a new opponent. Beating it is necessary, not sufficient.
"""
import base64
import copy
import json
import zlib

_TRACE_B85 = "{payload}"
_TRACE = json.loads(zlib.decompress(base64.b85decode(_TRACE_B85)).decode("utf-8"))

UNIT_OPS = {{"NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
             "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
             "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER"}}
MARKET_OPS = {{"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}}


def _legal_unit(op):
    return isinstance(op, list) and op and op[0] in UNIT_OPS


MOVES = {{"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}}

# Where the recording *thinks* each unit is standing, replayed forward from the
# spawn tiles by applying the taped moves. Without this a tape desynchronises
# the first time a hire or a purchase lands differently -- every later op then
# fires on the wrong tile, which is why a taped 130k game replays as 24k.
_VIRT = {{"step": -1, "pos": [], "day": -1}}


def _spawn(size, taken):
    half = size // 2
    tiles = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]
    counts = [sum(1 for p in taken if tuple(p) == t) for t in tiles]
    best = min(range(4), key=lambda i: (counts[i], i))
    return list(tiles[best])


def _advance_virtual(action, size, n_units):
    """Apply one taped turn to the virtual positions."""
    pos = _VIRT["pos"]
    while len(pos) < n_units:
        pos.append(_spawn(size, pos))
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    for i, op in enumerate(ops):
        if i >= len(pos) or not _legal_unit(op):
            continue
        d = MOVES.get(op[0])
        if d is None:
            continue
        nx, ny = pos[i][0] + d[0], pos[i][1] + d[1]
        if 0 <= nx < size and 0 <= ny < size:
            pos[i] = [nx, ny]


def _toward(pos, target):
    if pos[0] < target[0]:
        return ["EAST"]
    if pos[0] > target[0]:
        return ["WEST"]
    if pos[1] < target[1]:
        return ["SOUTH"]
    if pos[1] > target[1]:
        return ["NORTH"]
    return None


def _repair(action, obs, step):
    """Keep the taped action legal *and on the right tile*.

    Index alignment alone is not enough. The recording assumes each unit is
    standing somewhere specific; when our cash or hire count drifts by one the
    positions drift with it, and a WATER fired from the wrong tile is a silent
    no-op. So the tape carries a virtual position per unit and any unit that
    has slipped walks back to it instead of miming the recording.
    """
    farm = obs["farms"][obs["player"]]
    size = len(farm["tiles"])
    units = [list(farm["farmer"])] + [list(p) for p in farm["hands"]]
    n_units = len(units)
    day = int(obs.get("day", 0) or 0)

    if step <= _VIRT["step"] or day != _VIRT["day"]:
        # New episode, or the engine reset every unit to the shed overnight.
        _VIRT["pos"] = [list(p) for p in units]
        _VIRT["day"] = day
    while len(_VIRT["pos"]) < n_units:
        _VIRT["pos"].append(_spawn(size, _VIRT["pos"]))
    _VIRT["step"] = step

    out = {{"farmer": ["PASS"], "hands": [], "market": []}}
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    chosen = []
    for i in range(n_units):
        op = ops[i] if i < len(ops) else None
        want = _VIRT["pos"][i] if i < len(_VIRT["pos"]) else units[i]
        if not _legal_unit(op):
            chosen.append(["PASS"])
            continue
        if op[0] in MOVES:
            chosen.append(list(op))
            continue
        if list(units[i]) == list(want):
            chosen.append(list(op))
        else:
            mv = _toward(units[i], want)
            chosen.append(mv if mv else list(op))
    out["farmer"] = chosen[0] if chosen else ["PASS"]
    out["hands"] = chosen[1:]

    money = float(farm.get("money", 0) or 0)
    for order in action.get("market") or []:
        if not isinstance(order, list) or not order or order[0] not in MARKET_OPS:
            continue
        if order[0] != "SELL" and money <= 0:
            continue
        out["market"].append(list(order))

    _advance_virtual(action, size, n_units)
    return out


def _step_of(obs, config=None):
    """Turn index, with a fallback.

    `step` is declared `"shared": true` in the base schema, and the core copies
    every shared property into each seat's observation before calling the agent
    (`core.py:__get_shared_state`). A two-seat probe confirms it: both seats see
    0..718. The seat-0-only claim this project carried for a while is true of
    the *stored replay* -- shared properties are stripped from non-first agents
    when the episode is written out -- not of the live observation.

    The day/hour fallback stays because it costs nothing and covers replaying a
    saved state through an agent, which is exactly what the miner does.
    """
    tpd = 24
    if config is not None:
        try:
            tpd = max(1, int(config["turnsPerDay"]))
        except Exception:
            tpd = 24
    step = obs.get("step")
    if step is None:
        step = int(obs.get("day", 0) or 0) * tpd + int(obs.get("hour", 0) or 0)
    return int(step or 0)


def agent(obs, config=None):
    step = _step_of(obs, config)
    idx = min(step, len(_TRACE) - 1)
    return _repair(copy.deepcopy(_TRACE[idx]), obs, step)
'''


def build_tape(replay_path, seat, out_dir=OPP_DIR, label=None):
    """Write one tape opponent. Returns its repo-relative path."""
    meta = replay_meta(replay_path)
    if not meta:
        return None
    actions = extract_actions(replay_path, seat)
    payload = base64.b85encode(
        zlib.compress(json.dumps(actions, separators=(",", ":")).encode("utf-8"), 9)
    ).decode("ascii")

    label = label or f"episode {meta['episode']} seat {seat}"
    name = f"tape_{meta['episode']}_s{seat}.py"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(_TAPE_TEMPLATE.format(
            label=label, episode=meta["episode"], seat=seat,
            rewards=f"${meta['rewards'][0]:,.0f} vs ${meta['rewards'][1]:,.0f}",
            result="won" if meta["winner"] == seat else "lost",
            steps=len(actions), payload=payload))

    rel = os.path.relpath(out, ROOT)
    registry.register_opponent(
        name, kind="tape", path=rel, episode=meta["episode"], seat=seat,
        steps=len(actions), bank=meta["rewards"][seat],
        opp_bank=meta["rewards"][1 - seat],
        won=(meta["winner"] == seat), bytes=os.path.getsize(out))
    return rel


def build_tapes(limit=None, winners_only=True, verbose=True):
    """One tape per replay, for the seat that won (that is the play worth facing)."""
    made = []
    paths = find_replays(limit)
    if verbose:
        pr.log(f"{len(paths)} replay(s) on disk")
    for p in paths:
        meta = replay_meta(p)
        if not meta:
            if verbose:
                pr.warn(f"unreadable or incomplete: {os.path.basename(p)}", 1)
            continue
        seats = [meta["winner"]] if winners_only else [0, 1]
        for seat in seats:
            rel = build_tape(p, seat)
            if rel:
                made.append(rel)
                if verbose:
                    pr.log(f"tape {os.path.basename(rel)}  "
                           f"${meta['rewards'][seat]:,.0f}  {meta['steps']} steps", 1)
    if verbose:
        pr.log(f"built {len(made)} tape opponent(s) in opponents/")
    return made


# -------------------------------------------------------------------- clone --

def fit_clone(replay_path, seat, base=None, verbose=True):
    """Fit our heuristic's parameters to reproduce one player's decisions.

    Scores a parameter set by how often our policy, shown the recorded
    observation, picks the same unit ops the human-tuned agent actually picked.
    Coordinate ascent over the spatial and valuation knobs, which are the ones
    that decide *which job* a unit takes -- the thing the recording lets us
    observe. Budget knobs are left alone: the recording shows what they bought,
    not what they could have afforded.
    """
    import importlib.util
    import kaggriculture.pipeline.params as paramio

    with open(replay_path, encoding="utf-8") as f:
        d = json.load(f)
    steps = d.get("steps") or []
    if not steps:
        return None

    base = base or _newest_agent()
    P = dict(paramio.load(base))
    src = os.path.join(ROOT, "agents", "v1_heuristic.py")
    work = os.path.join(ROOT, ".local", "opponents")
    os.makedirs(work, exist_ok=True)
    cand = os.path.join(work, "clone_cand.py")

    samples = []
    for i, step in enumerate(steps):
        if seat >= len(step):
            continue
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        obs = step[seat].get("observation")
        act = (nxt[seat].get("action") if nxt and seat < len(nxt) else None)
        if isinstance(obs, dict) and isinstance(act, dict) and "farms" in obs:
            samples.append((obs, act))
    if len(samples) < 20:
        if verbose:
            pr.warn(f"only {len(samples)} usable state/action pairs -- "
                    f"replay lacks per-seat observations", 1)
        return None
    samples = samples[::max(1, len(samples) // 200)]        # ~200 evenly spread

    def agreement(params):
        paramio.write(src, cand, params, header="clone candidate",
                      module_doc="opponent clone fitting candidate\n")
        spec = importlib.util.spec_from_file_location("clone_cand", cand)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        hits = total = 0
        for obs, act in samples:
            try:
                mine = mod.agent(obs, d.get("configuration"))
            except Exception:                                  # noqa: BLE001
                continue
            for key in ("farmer",):
                if mine.get(key) and act.get(key):
                    total += 1
                    hits += int(list(mine[key]) == list(act[key]))
            for a, b in zip(mine.get("hands") or [], act.get("hands") or []):
                total += 1
                hits += int(list(a) == list(b))
        return (hits / total) if total else 0.0

    knobs = ["travel_weight", "fert_weight", "poach_penalty", "care_weight",
             "fert_collect_weight", "capacity_util"]
    best = agreement(P)
    if verbose:
        pr.log(f"clone fit: starting agreement {best:.1%} over {len(samples)} states", 1)
    for k in knobs:
        v = P.get(k)
        if not isinstance(v, (int, float)):
            continue
        for mult in (0.6, 0.8, 1.25, 1.6):
            Q = dict(P)
            Q[k] = max(1e-6, v * mult)
            score = agreement(Q)
            if score > best:
                best, P = score, Q
    if verbose:
        pr.log(f"clone fit: final agreement {best:.1%}", 1)

    meta = replay_meta(replay_path)
    name = f"clone_{meta['episode']}_s{seat}.py"
    os.makedirs(OPP_DIR, exist_ok=True)
    out = os.path.join(OPP_DIR, name)
    paramio.write(src, out, P, header=f"clone of episode {meta['episode']} seat {seat}",
                  module_doc=(f"Fitted clone of the player in episode "
                              f"{meta['episode']} seat {seat}.\n"
                              f"Reproduces {best:.1%} of their recorded unit ops "
                              f"over {len(samples)} sampled states.\n"
                              f"Unlike the tape opponent this one reacts to new "
                              f"states, but it can only express strategies our "
                              f"own heuristic can represent.\n"))
    rel = os.path.relpath(out, ROOT)
    registry.register_opponent(name, kind="clone", path=rel,
                               episode=meta["episode"], seat=seat,
                               agreement=best, states=len(samples),
                               bank=meta["rewards"][seat])
    return rel


def _newest_agent():
    c = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
    return c[-1] if c else os.path.join(ROOT, "agents", "v2_tuned.py")


# --------------------------------------------------------------------- play --

def play(agent_path, opponents=None, n=2, workers=2, seed0=90000, verbose=True):
    """Play `agent_path` against the opponent set, both seats per seed."""
    from concurrent.futures import ProcessPoolExecutor
    import kaggriculture.engine._vendor as _vendor  # noqa: F401

    opps = opponents or [o["path"] for o in registry.load_opponents().values()
                         if os.path.exists(os.path.join(ROOT, o["path"]))]
    if not opps:
        pr.warn("no opponents built yet -- run --build-tapes first")
        return []

    jobs = []
    for opp in opps:
        for i in range(n):
            jobs.append((agent_path, opp, seed0 + i))
            jobs.append((opp, agent_path, seed0 + i))

    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_mute) as pool:
        for (a, b), (ra, rb) in zip([(j[0], j[1]) for j in jobs],
                                    pool.map(_play_job, jobs)):
            opp = b if a == agent_path else a
            mine = ra if a == agent_path else rb
            theirs = rb if a == agent_path else ra
            r = results.setdefault(opp, {"wins": 0, "games": 0, "mine": 0.0,
                                         "theirs": 0.0})
            r["games"] += 1
            r["wins"] += int(mine > theirs)
            r["mine"] += mine
            r["theirs"] += theirs

    rows = []
    for opp, r in sorted(results.items(), key=lambda kv: kv[1]["wins"] / max(1, kv[1]["games"])):
        row = {"opponent": os.path.basename(opp), "path": opp,
               "games": r["games"], "wins": r["wins"],
               "win_rate": r["wins"] / max(1, r["games"]),
               "mean_bank": r["mine"] / max(1, r["games"]),
               "opp_bank": r["theirs"] / max(1, r["games"])}
        rows.append(row)
        if verbose:
            pr.log(f"{row['opponent']:<34} {row['win_rate']:>5.0%}  "
                   f"${row['mean_bank']:>9,.0f} vs ${row['opp_bank']:>9,.0f}", 1)
    return rows


def _mute():
    try:
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    except OSError:
        pass


def _play_job(job):
    left, right, seed = job
    from kaggle_environments import make
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "seed": seed,
                              "actTimeout": 60, "runTimeout": 100000})
    env.run([left, right])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-tapes", action="store_true")
    ap.add_argument("--build-clones", action="store_true")
    ap.add_argument("--both-seats", action="store_true",
                    help="tape the loser too, not just the winning seat")
    ap.add_argument("--top", type=int, default=None, help="limit replays used")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--play", metavar="AGENT")
    ap.add_argument("--vs", nargs="+", default=None, metavar="OPPONENT",
                    help="play only these opponents (paths or basenames); "
                         "default is the whole set")
    ap.add_argument("--n", type=int, default=2, help="seeds per opponent")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=90000)
    args = ap.parse_args()

    pr.reset()
    if args.build_tapes:
        build_tapes(args.top, winners_only=not args.both_seats)
    if args.build_clones:
        for p in find_replays(args.top):
            meta = replay_meta(p)
            if meta:
                fit_clone(p, meta["winner"])
    if args.list or not (args.build_tapes or args.build_clones or args.play):
        opps = registry.load_opponents()
        if not opps:
            print("no opponents yet. Build some:\n"
                  "  python -m kaggriculture.measure.opponents --build-tapes")
            return 0
        print(f"{len(opps)} opponent(s)\n")
        print(f"{'name':<34} {'kind':<7} {'bank':>11}  detail")
        for o in sorted(opps.values(), key=lambda x: -(x.get("bank") or 0)):
            detail = (f"{o.get('agreement', 0):.0%} agreement" if o.get("kind") == "clone"
                      else f"{o.get('steps', 0)} steps")
            print(f"{o['name']:<34} {o.get('kind', ''):<7} "
                  f"${o.get('bank', 0):>10,.0f}  {detail}")
    if args.play:
        hits = sorted(glob.glob(os.path.join(ROOT, args.play)))
        agent = hits[-1] if hits else os.path.join(ROOT, args.play)
        # play() has always taken an opponent list; the CLI just never offered
        # one, so "simulate against this particular leaderboard topper" was not
        # expressible -- you got the whole set or nothing.
        chosen = None
        if args.vs:
            chosen = []
            known = {os.path.basename(o["path"]): o["path"]
                     for o in registry.load_opponents().values()}
            for want in args.vs:
                base = os.path.basename(want)
                if base in known:
                    chosen.append(known[base])
                elif os.path.exists(os.path.join(ROOT, want)):
                    chosen.append(want)
                else:
                    found = sorted(glob.glob(os.path.join(ROOT, "opponents", f"*{base}*")))
                    if found:
                        chosen.append(os.path.relpath(found[-1], ROOT))
                    else:
                        pr.warn(f"no opponent matching {want!r}", 1)
            if not chosen:
                pr.warn("none of --vs resolved; nothing to play", 1)
                return 1
        target = (", ".join(os.path.basename(c) for c in chosen) if chosen
                  else "the opponent set")
        pr.log(f"playing {os.path.relpath(agent, ROOT)} against {target}")
        play(agent, opponents=chosen, n=args.n, workers=args.workers,
             seed0=args.seed0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
