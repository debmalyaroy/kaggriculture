"""STAGE 0 of the behavioural-cloning plan: build the top-100 corpus.

Plan: `docs/history/bc-policy-plan-2026-09-04.md`, `docs/history/plan-2026-09-05.md`.

Why this exists. Every other route to a stronger agent is closed on
measurement (docs/history/hard-band-2026-09-04.md): layers are quantity-conserving,
dispatch gains nothing at any pool size, search cannot lift the runtime
policy, and mining a top team's TAPE transfers almost nothing -- rank 1's
replayed tape scores 0.071 on real 2300+ worlds, worse than our own base,
because a trace is not a decision rule. Behavioural cloning is the one route
that transfers the RULE rather than the trajectory.

What it does. For every top-100 episode we hold with both seats indexed, it
re-simulates the game locally from the stored seed and both action tracks,
captures the per-turn state vector for the top-100 seat, and pairs it with
the action that team actually played. No downloads: every seed is already in
the route index.

Two things make the corpus honest:

  * SPLIT BY TEAM, not just by episode. A model that has seen a team's other
    games is not being tested on generalisation.
  * TRAIN ONLY AFTER THE SCRIPT ENDS. Measured 2026-09-04, every top team
    plays a FIXED opening and then decides at runtime, at a team-specific
    day: Crop Dusta d3, Jesse Bullard d6, MtN d12, Wang H2O d18, while the
    mid-field is scripted end to end. Rows before a team's script-end day are
    that team's tape, not its policy, and teach nothing.

    python src/bc_corpus.py --scan          # script-end day per team
    python src/bc_corpus.py --build --top 100
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

# Opponent team names are arbitrary UTF-8 and the Windows console is cp1252;
# a CJK name has now crashed three separate analysis tools mid-run
# (band_panel, contested_dumps, and this one). An analysis tool must never die
# on a name.
try:                                        # pragma: no cover - tty only
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

OUT_DIR = os.path.join(ROOT, "data", "bc_corpus")
SEGMENTS = ((0, 72), (72, 145), (145, 288), (288, 432), (432, 576), (576, 720))
SEG_LABEL = ("d0-2", "d3-5", "d6-11", "d12-17", "d18-23", "d24-29")
# A segment counts as SCRIPTED when this fraction of the team's same-day games
# share byte-identical actions in it. 0.60 keeps teams that vary a little
# (Wang H2O runs 0.86) on the scripted side, which is the conservative
# direction: we would rather drop a few real decisions than train on tape.
SCRIPT_THRESHOLD = 0.60


def leaderboard_top(n):
    """Team names in the current top n, from the freshest LB csv on disk."""
    import csv
    import glob
    files = sorted(glob.glob(os.path.join(ROOT, ".local", "band_panel", "lb",
                                          "*.csv")))
    if not files:
        raise SystemExit("no leaderboard csv -- run band_panel.download_leaderboard()")
    out = {}
    with open(files[-1], encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if int(r["Rank"]) <= n:
                out[r["TeamName"]] = (int(r["Rank"]), float(r["Score"]))
    return out, os.path.basename(files[-1])


def script_end_day(idx, teams):
    """Per team: the first day from which its actions stop being identical.

    Uses ONE submission (the team's busiest single date) so that version
    churn between submissions cannot masquerade as runtime reactivity -- the
    uncontrolled version of this test made every team look adaptive.
    """
    from kaggriculture.trackp import routes_io as R
    by = defaultdict(lambda: defaultdict(list))
    for k, v in idx["routes"].items():
        if v.get("engine") != "1.32.7":
            continue
        t = str(v.get("team"))
        if t in teams:
            by[t][str(v.get("date"))].append(k)

    out = {}
    for t, days in by.items():
        date, ks = max(days.items(), key=lambda kv: len(kv[1]))
        if len(ks) < 5:
            continue                       # too few games on one day to judge
        acts = []
        for k in ks[:14]:
            try:
                acts.append(R.load_route(k))
            except Exception:                                    # noqa: BLE001
                continue
        if len(acts) < 5:
            continue
        shares = []
        for a, b in SEGMENTS:
            c = Counter(hashlib.sha256(
                json.dumps(A[a:b], sort_keys=True).encode()).hexdigest()[:12]
                for A in acts)
            shares.append(c.most_common(1)[0][1] / len(acts))
        end = 720
        for i, s in enumerate(shares):
            if s < SCRIPT_THRESHOLD:
                end = SEGMENTS[i][0]
                break
        out[t] = {"date": date, "games": len(acts), "shares": shares,
                  "script_end_step": end, "script_end_day": end // 24}
    return out


def cmd_scan(args):
    from kaggriculture.trackp import routes_io as R
    teams, src = leaderboard_top(args.top)
    idx = R.load_index()
    info = script_end_day(idx, set(teams))
    os.makedirs(OUT_DIR, exist_ok=True)
    for t, d in info.items():
        d["rank"], d["lb_score"] = teams[t]
    path = os.path.join(OUT_DIR, "script_end.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    rows = sorted(info.items(), key=lambda kv: kv[1]["rank"])
    print(f"leaderboard: {src}; {len(rows)} of top {args.top} judged\n")
    print(f"{'team':<24}{'rank':>5}{'games':>6}  " +
          "  ".join(f"{l:>7}" for l in SEG_LABEL) + "   script ends")
    for t, d in rows[:30]:
        print(f"{t[:22]:<24}{d['rank']:>5}{d['games']:>6}  " +
              "  ".join(f"{s:>7.2f}" for s in d["shares"]) +
              f"   day {d['script_end_day']}")
    reactive = [t for t, d in rows if d["script_end_step"] < 720]
    print(f"\n{len(reactive)} of {len(rows)} teams go reactive at some point; "
          f"{len(rows) - len(reactive)} are scripted end to end.")
    print(f"-> {os.path.relpath(path, ROOT)}")
    return 0


# ---------------------------------------------------------------- build ----

ACTION_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                   "EGG", "MILK", "WOOL", "FERTILIZER")


def sell_vector(action):
    """The SELL quantities the expert issued this turn, per product.

    This is the BC target. It matches the socket the agent already has: the
    embedded policy head is [50, 64, 64, 9] -> a 9-product sell vector, blended
    against the schedule with a bounded delta. Predicting the same shape means
    a trained model drops straight in.
    """
    out = [0.0] * len(ACTION_PRODUCTS)
    for o in (action or {}).get("market") or []:
        if not (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"):
            continue
        try:
            i = ACTION_PRODUCTS.index(o[1])
        except ValueError:
            continue
        try:
            out[i] += min(float(o[2] or 0), 100.0)
        except (TypeError, ValueError):
            pass
    return out


def simulate(seed, acts0, acts1, seat):
    """Replay one recorded game locally and capture (state, sell) per turn.

    Both action tracks are forced into the engine, so the world is the exact
    one that was played -- the seed makes it reproducible to the dollar (the
    hard band verified this on 56 of 58 cells).
    """
    import kaggriculture.engine._vendor as _vendor  # noqa: F401
    from kaggle_environments import make
    import kaggriculture.data.turn_features as TF
    env = make("kaggriculture",
               configuration={"episodeSteps": 720, "actTimeout": 60,
                              "runTimeout": 1000000, "seed": int(seed)},
               info={"seed": int(seed)})
    env.reset(2)
    X, Y = [], []
    n = min(len(acts0), len(acts1))
    for i in range(n):
        if env.done:
            break
        obs_pub = env.state[0].observation
        obs_seat = env.state[seat].observation
        v = TF.turn_vector(obs_pub, obs_seat, seat)
        if v is not None:
            X.append(v)
            Y.append(sell_vector(acts1[i] if seat == 1 else acts0[i]))
        env.step([acts0[i], acts1[i]])
    return X, Y


def simulate_rust(srv, seed, acts0, acts1, seat):
    """Same capture as simulate(), on the Rust engine via `kagg serve`.

    The serve state JSON carries day/hour/step, BOTH farms, the market and
    both seats' private blocks -- exactly the fields turn_vector reads. The
    engine is bit-identical to the official interpreter per step
    (tests/test_rust_engine.py, 50/50 episodes), and --verify-rust checks the
    produced ROWS against the official-engine path before this is trusted.
    """
    import kaggriculture.data.turn_features as TF
    from kaggriculture.engine.serve_match import action_to_line
    js = srv.cmd(f"RESET {int(seed)}")
    X, Y = [], []
    n = min(len(acts0), len(acts1))
    for i in range(n):
        # the official env (episodeSteps=720) is DONE at state step 719 --
        # 720 states, 719 transitions -- while serve accepts one more step;
        # mirror the official row count exactly
        if js.get("done") or int(js.get("step") or 0) >= 719:
            break
        obs_seat = {"private": (js.get("private") or [{}, {}])[seat]}
        v = TF.turn_vector(js, obs_seat, seat)
        if v is not None:
            X.append(v)
            Y.append(sell_vector(acts1[i] if seat == 1 else acts0[i]))
        la = action_to_line(acts0[i])
        lb = action_to_line(acts1[i])
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            raise RuntimeError(js["error"])
    return X, Y


_W = {}                                     # per-worker process state


def _worker_init():
    from kaggriculture.engine.serve_match import Serve
    from kaggriculture.trackp import routes_io as R
    _W["srv"] = Serve()
    _W["R"] = R


def _load_tracks(R, st, kme, kopp):
    """Both action tracks for a job: from the route index, or -- when kme is
    ("replay", path) -- straight from the raw replay file."""
    if isinstance(kme, tuple) and kme[0] == "replay":
        rep = json.load(open(kme[1], encoding="utf-8"))
        return ([(s[0] or {}).get("action") or {} for s in rep["steps"]],
                [(s[1] or {}).get("action") or {} for s in rep["steps"]])
    a_me = R.load_route(kme)
    a_op = R.load_route(kopp)
    return (a_me, a_op) if st == 0 else (a_op, a_me)


def _worker_job(job):
    """One (episode, seat) job in a worker process. Returns rows or an error."""
    ep, st, team, kme, kopp, seed = job
    R = _W["R"]
    try:
        acts0, acts1 = _load_tracks(R, st, kme, kopp)
        X, Y = simulate_rust(_W["srv"], seed, acts0, acts1, st)
        return (ep, st, team, X, Y, None)
    except Exception as exc:                                     # noqa: BLE001
        # a dead serve process poisons every later job in this worker
        try:
            _W["srv"].close()
        except Exception:                                        # noqa: BLE001
            pass
        from kaggriculture.engine.serve_match import Serve
        _W["srv"] = Serve()
        return (ep, st, team, None, None, f"{type(exc).__name__}: {exc}")


def cmd_verify_rust(args):
    """Prove the Rust capture equals the official-engine rows already on disk."""
    import numpy as np
    from kaggriculture.engine.serve_match import Serve
    from kaggriculture.trackp import routes_io as R
    p = sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.npz")))
    if not p:
        raise SystemExit("no shards to verify against -- build some first")
    z = np.load(p[0], allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    Xs, Ys = z["X"], z["Y"]
    script = json.load(open(os.path.join(OUT_DIR, "script_end.json"),
                            encoding="utf-8"))
    # contiguous (episode, seat) row groups
    groups, s = [], 0
    for i in range(1, len(meta) + 1):
        if i == len(meta) or (meta[i]["episode"], meta[i]["seat"]) != \
                (meta[s]["episode"], meta[s]["seat"]):
            groups.append((s, i))
            s = i
    idx = R.load_index()
    seats = defaultdict(dict)
    for k, v in idx["routes"].items():
        if v.get("engine") == "1.32.7":
            seats[str(v.get("episode"))][int(v.get("seat", 0))] = (k, v)
    srv = Serve()
    ok = bad = 0
    try:
        for a, b in groups[:args.limit or 6]:
            m = meta[a]
            ep, st, team = str(m["episode"]), int(m["seat"]), m["team"]
            sd = seats.get(ep) or {}
            if len(sd) != 2:
                continue
            kme, v = sd[st]
            a_me, a_op = R.load_route(kme), R.load_route(sd[1 - st][0])
            acts0, acts1 = (a_me, a_op) if st == 0 else (a_op, a_me)
            X, Y = simulate_rust(srv, v.get("seed"), acts0, acts1, st)
            start = int((script.get(team) or {}).get("script_end_step", 0))
            Xr = np.asarray(X[start:], dtype=np.float32)
            Yr = np.asarray(Y[start:], dtype=np.float32)
            same = (Xr.shape == Xs[a:b].shape and
                    np.array_equal(Xr, Xs[a:b]) and
                    np.array_equal(Yr, Ys[a:b]))
            ok += same
            bad += not same
            print(f"  ep {ep} seat {st} {team[:20]:<20} rows {b - a:>4} "
                  f"{'EXACT' if same else 'MISMATCH'}", flush=True)
            if not same and Xr.shape == Xs[a:b].shape:
                d = np.abs(Xr - Xs[a:b])
                print(f"    max |dX| {d.max():.6g} at feature "
                      f"{int(np.unravel_index(d.argmax(), d.shape)[1])}")
    finally:
        srv.close()
    print(f"\n{ok} exact, {bad} mismatched")
    return 0 if bad == 0 and ok else 1


def cmd_build(args):
    import numpy as np
    from kaggriculture.trackp import routes_io as R
    teams, _src = leaderboard_top(args.top)
    idx = R.load_index()
    script = {}
    p = os.path.join(OUT_DIR, "script_end.json")
    if os.path.exists(p):
        script = json.load(open(p, encoding="utf-8"))

    # every episode where a top-N team has a route on the current engine and
    # BOTH seats are indexed (so the game is exactly reproducible)
    seats = defaultdict(dict)
    for k, v in idx["routes"].items():
        if v.get("engine") != "1.32.7":
            continue
        seats[str(v.get("episode"))][int(v.get("seat", 0))] = (k, v)
    jobs = []
    for ep, sd in seats.items():
        if len(sd) != 2:
            continue
        for st, (k, v) in sd.items():
            t = str(v.get("team"))
            if t in teams:
                jobs.append((ep, st, t, k, sd[1 - st][0], v.get("seed")))
    jobs = [j for j in jobs if j[5] is not None]

    # REPLAY-DIRECT jobs: a raw replay file carries BOTH action tracks, the
    # seed and TeamNames -- no index rows needed. This is what lifts the
    # rank-1 corpus past the both-seats-indexed ceiling (194 episodes on
    # 2026-09-05) and turns every autopsy pull into training rows.
    for d in (args.replay_dirs or []):
        if not os.path.isdir(d):
            print(f"replay dir missing: {d}", flush=True)
            continue
        have = {j[0] for j in jobs}
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            ep = f[:-5]
            path = os.path.join(d, f)
            try:
                info = json.load(open(path, encoding="utf-8")).get("info") or {}
            except (OSError, ValueError):
                continue
            names = info.get("TeamNames") or []
            seed = info.get("seed")
            if seed is None or len(names) != 2:
                continue
            for st in (0, 1):
                t = str(names[st])
                if t in teams and (ep, st) not in have:
                    jobs.append((ep, st, t, ("replay", path), None, seed))

    if args.limit:
        jobs = jobs[:args.limit]

    # RESUME: skip (episode, seat) pairs already captured in shards on disk.
    # Keyed on the shard META, not a job counter, so a route index that grew
    # since the interrupted run cannot misalign the skip.
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    shard_i = 0
    for p in sorted(glob.glob(os.path.join(OUT_DIR, "shard_*.npz"))):
        z = np.load(p, allow_pickle=True)
        for m in json.loads(str(z["meta"])):
            done.add((str(m["episode"]), int(m["seat"])))
        shard_i = max(shard_i, int(os.path.basename(p)[6:10]) + 1)
    # a fully-scripted seat keeps ZERO rows, leaves no meta, and would
    # re-simulate every incremental run -- it gets its own done-marker file
    empty_p = os.path.join(OUT_DIR, "done_empty.json")
    if os.path.exists(empty_p):
        for e, s in json.load(open(empty_p, encoding="utf-8")):
            done.add((str(e), int(s)))
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if (str(j[0]), int(j[1])) not in done]
        print(f"resume: {before - len(jobs)} jobs already in "
              f"{shard_i} shard(s), skipping", flush=True)
    print(f"{len(jobs)} (episode, seat) jobs from {len(teams)} top-{args.top} "
          f"teams", flush=True)

    shard, kept, dropped, failed = [], 0, 0, 0
    empties = []

    def fold(n, ep, st, team, X, Y, err):
        """Fold one finished job into the current shard; write when full."""
        nonlocal shard, shard_i, kept, dropped, failed
        if err is not None:
            failed += 1
            if failed <= 3:
                print(f"  ep {ep}: {err}", flush=True)
        else:
            # TRAIN ONLY AFTER THE SCRIPT ENDS -- earlier rows are that
            # team's tape, not its policy.
            start = int((script.get(team) or {}).get("script_end_step", 0))
            for i in range(len(X)):
                if i < start:
                    dropped += 1
                    continue
                shard.append((X[i], Y[i], team, int(ep), st))
                kept += 1
            if start >= len(X):
                empties.append((str(ep), int(st)))
        if len(shard) >= args.shard or n == len(jobs):
            Xs = np.asarray([r[0] for r in shard], dtype=np.float32)
            Ys = np.asarray([r[1] for r in shard], dtype=np.float32)
            meta = [{"team": r[2], "episode": r[3], "seat": r[4]}
                    for r in shard]
            out = os.path.join(OUT_DIR, f"shard_{shard_i:04d}.npz")
            np.savez_compressed(out, X=Xs, Y=Ys,
                                meta=json.dumps(meta))
            print(f"  [{n}/{len(jobs)}] wrote {os.path.basename(out)}: "
                  f"{len(shard):,} rows (kept {kept:,}, pre-script dropped "
                  f"{dropped:,}, failed {failed})", flush=True)
            shard, shard_i = [], shard_i + 1

    if args.rust:
        # Rust engine via `kagg serve`, one serve process per worker. A job
        # keeps EPISODE granularity, so every row of one (episode, seat) run
        # stays in one contiguous shard block (Stage 1 keys sequences on it).
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init) as pool:
            for n, (ep, st, team, X, Y, err) in enumerate(
                    pool.imap_unordered(_worker_job, jobs, chunksize=1), 1):
                fold(n, ep, st, team, X, Y, err)
    else:
        for n, (ep, st, team, kme, kopp, seed) in enumerate(jobs, 1):
            try:
                acts0, acts1 = _load_tracks(R, st, kme, kopp)
                X, Y = simulate(seed, acts0, acts1, st)
                fold(n, ep, st, team, X, Y, None)
            except Exception as exc:                             # noqa: BLE001
                fold(n, ep, st, team, None, None,
                     f"{type(exc).__name__}: {exc}")
    if empties:
        prev = []
        if os.path.exists(empty_p):
            prev = json.load(open(empty_p, encoding="utf-8"))
        json.dump(sorted(set(map(tuple, prev)) | set(empties)),
                  open(empty_p, "w", encoding="utf-8"))
        print(f"  recorded {len(empties)} zero-row (episode, seat) markers")
    print(f"DONE. kept {kept:,} rows, dropped {dropped:,} pre-script, "
          f"{failed} episodes failed. -> {os.path.relpath(OUT_DIR, ROOT)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", action="store_true",
                    help="measure each team's script-end day (no simulation)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify-rust", action="store_true",
                    help="re-run shard_0000 groups on the Rust engine and "
                         "demand row-exact equality")
    ap.add_argument("--rust", action="store_true",
                    help="simulate on the Rust engine (kagg serve)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--replay-dirs", nargs="*", default=None,
                    help="also mine raw replay JSONs (both tracks + seed + "
                         "TeamNames are in the file; no index rows needed)")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=200000,
                    help="rows per output shard")
    a = ap.parse_args()
    if a.scan:
        return cmd_scan(a)
    if a.verify_rust:
        return cmd_verify_rust(a)
    if a.build:
        return cmd_build(a)
    ap.error("nothing to do: pass --scan or --build")


if __name__ == "__main__":
    raise SystemExit(main())
