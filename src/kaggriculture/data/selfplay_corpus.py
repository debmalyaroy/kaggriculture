"""Self-play corpus augmentation -- play the public agents against each other on
the Rust ``kagg serve`` engine, capture the per-turn trace, and extract rows in
the EXACT same token format as ``trackp_corpus`` (winner-seat by default).

Format correctness is by construction: the per-step observation dict is built
exactly like ``serve_match.obs_for`` (the faithful serve obs) and fed to the
SAME ``encode_tokens`` / ``encode_action`` used on the gm_dataset ladder
replays. Output goes to a SEPARATE dir with an IDENTICAL schema + companions, so
the BC loader can glob gm + self-play together and weight/subsample self-play.

    python -m kaggriculture.data.selfplay_corpus --smoke                 # 5 agents x 2 seeds
    python -m kaggriculture.data.selfplay_corpus --agents 169 --seeds 6  # full round-robin
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import kaggriculture.data.trackp_corpus as TC
from kaggriculture.engine.serve_match import (
    Serve, load_agent, obs_for, _call, action_to_line, EPISODE_STEPS)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

AGENTS_DIR = os.path.join(ROOT, "agents")
OUT_DIR = os.path.join(ROOT, "data", "trackp_corpus_selfplay")

SCHEMA = pa.schema([
    ("episode_id", pa.int64()), ("seat", pa.int8()), ("day", pa.int16()),
    ("step", pa.int16()), ("split", pa.int8()), ("rtg", pa.float32()),
    ("rating", pa.float32()), ("world_family", pa.int16()),
    ("n_tokens", pa.int16()), ("tokens", pa.binary()), ("action", pa.binary())])


# Agent ROSTER entries are (name, path, rating) -- PATHS, not loaded functions,
# so a spec list is picklable across the parallel worker pool (each worker loads
# its own copies). _load_specs turns them into (name, fn, rating) in-process.

def _load_specs(specs):
    out = []
    for name, path, rating in specs:
        try:
            out.append((name, load_agent(path), float(rating)))
        except Exception:
            continue
    return out


def list_agents(n=None):
    files = sorted(glob.glob(os.path.join(AGENTS_DIR, "*.py")))
    out = []
    for f in files:
        try:
            load_agent(f)                 # validates it exports agent(obs)
            out.append((os.path.basename(f)[:-3], f, 0.0))
        except Exception:
            continue                      # skip non-agents / broken files
        if n and len(out) >= n:
            break
    return out


def agents_from_dir(d):
    """Every *.py in ``d`` that exports agent(obs) -> [(name, path, 0.0)]."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.py"))):
        try:
            load_agent(f)
            out.append((os.path.basename(f)[:-3], f, 0.0))
        except Exception:
            continue
    return out


def top_reactive_agents(n=12, panel_path=None):
    """Top-``n`` REACTIVE ladder agents from the crown panel, by rating desc.

    Reactive == ``source == 'reactive-kernel'`` (real ``agent(obs)`` kernels, not
    static tapes). Skips ``tape_desync_risk`` refs and anything that won't load or
    isn't vetted. Returns [(name, agent_fn, rating)] -- the LOCKED self-play roster
    (top-12 reactive, per the trackp plan)."""
    panel_path = panel_path or os.path.join(ROOT, "models", "crown_panel.json")
    p = json.load(open(panel_path))
    refs = []
    for _band, lst in (p.get("bands") or {}).items():
        for r in lst:
            if r.get("source") != "reactive-kernel":
                continue
            if r.get("tape_desync_risk"):
                continue
            if r.get("vet") not in (None, "ok"):
                continue
            refs.append(r)
    refs.sort(key=lambda r: -float(r.get("rating", 0) or 0))
    seen, out = set(), []
    for r in refs:
        cands = [r.get("origin"), r.get("tape")]
        path = next((os.path.join(ROOT, c) for c in cands
                     if c and os.path.exists(os.path.join(ROOT, c))), None)
        if not path:
            continue
        key = os.path.basename(path)
        if key in seen:
            continue
        try:
            load_agent(path)              # validate; workers reload from path
        except Exception:
            continue
        seen.add(key)
        out.append((str(r.get("name", key))[:40], path, float(r.get("rating", 0) or 0)))
        if len(out) >= n:
            break
    return out


def _obs_dict(seat, js):
    """Plain-dict form of serve_match.obs_for (what encode_tokens reads)."""
    return {"step": js["step"], "day": js["day"], "hour": js["hour"],
            "player": seat, "farms": js["farms"], "market": js["market"],
            "town": js["town"], "private": js["private"][seat]}


def play_trace(agent_a, agent_b, seed, srv, max_turn_err=3):
    """Play one game on serve; return (banks, trace). trace[t]=(js, act0, act1).
    Faithful to serve_match.run_match (stops at EPISODE_STEPS-1, S5)."""
    js = srv.cmd(f"RESET {seed}")
    trace = []
    while js["step"] < EPISODE_STEPS - 1:
        a0 = _call(agent_a, obs_for(0, js))
        a1 = _call(agent_b, obs_for(1, js))
        trace.append((js, a0, a1))
        js = srv.cmd(f"STEP2 {action_to_line(a0)}\x1e{action_to_line(a1)}")
        if "error" in js:
            raise RuntimeError(js["error"])
    banks = [float(f.get("money") or 0) for f in js["farms"]]
    return banks, trace


def _world_family(trace):
    for js, _, _ in trace:
        if js.get("day") == 6:
            sh = ((js.get("town") or {}).get("unlocked_shops") or [])[:2]
            if sh:
                return 1 + int(hashlib.sha1("|".join(sorted(sh)).encode())
                               .hexdigest()[:6], 16) % 4095
    return 0


def extract_rows(trace, banks, epid, ratings=(0.0, 0.0), both=False):
    """Yield corpus rows for the winner seat (and loser if both)."""
    win = {0: 1.0 if banks[0] > banks[1] else (0.5 if banks[0] == banks[1] else 0.0)}
    win[1] = 1.0 - win[0] if win[0] != 0.5 else 0.5
    wf = _world_family(trace)
    sp = TC.split_of(epid)
    seats = [0, 1] if both else [0 if banks[0] >= banks[1] else 1]
    for seat in seats:
        for js, a0, a1 in trace:
            obs = _obs_dict(seat, js)
            act = a0 if seat == 0 else a1
            toks = TC.encode_tokens(obs, seat)
            acta = TC.encode_action(act or {})
            yield dict(episode_id=int(epid), seat=int(seat),
                       day=int(js.get("day", 0)), step=int(js.get("step", 0)),
                       split=int(sp), rtg=float(win[seat]),
                       rating=float(ratings[seat]), world_family=int(wf),
                       n_tokens=int(toks.shape[0]),
                       tokens=toks.tobytes(), action=acta.tobytes())


def _play_pairs(agents, seeds, both, out_dir, rows_per_shard, prefix, done,
                wid=0, njobs=1, limit_games=0, mirror=False):
    """Play the pair-share ``idx % njobs == wid`` on one serve, own ``prefix``
    shards, resume-aware. ``mirror`` pairs each agent with its own clone (i,i) --
    for exploiter diagnostics ("how our clone plays / how to beat it").
    Returns (n_rows, n_games, n_err)."""
    pairs = [(i, i) for i in range(len(agents))] if mirror \
        else list(itertools.combinations(range(len(agents)), 2))
    shard_i = TC._worker_shard_start(out_dir, prefix)
    buf = {k: [] for k in SCHEMA.names}
    srv = Serve(); n_rows = n_games = n_err = 0

    def flush():
        nonlocal shard_i
        if not buf["episode_id"]:
            return
        pq.write_table(pa.table(buf, schema=SCHEMA),
                       os.path.join(out_dir, f"{prefix}{shard_i:04d}.parquet"),
                       compression="zstd", compression_level=10)
        shard_i += 1
        for k in buf:
            buf[k].clear()

    try:
        for idx, (i, j) in enumerate(pairs):
            if idx % njobs != wid:
                continue
            (na, aa, ra), (nb, ab, rb) = agents[i], agents[j]
            for s in range(1, seeds + 1):
                for (x, y, nx, ny, rx, ry) in ((aa, ab, na, nb, ra, rb),
                                               (ab, aa, nb, na, rb, ra)):
                    epid = int(hashlib.sha1(f"{nx}|{ny}|{s}".encode())
                               .hexdigest()[:15], 16)
                    if str(epid) in done:
                        continue
                    try:
                        banks, trace = play_trace(x, y, s, srv)
                    except Exception:
                        n_err += 1
                        continue
                    for r in extract_rows(trace, banks, epid, ratings=(rx, ry), both=both):
                        for k in r:
                            buf[k].append(r[k])
                        n_rows += 1
                        if len(buf["episode_id"]) >= rows_per_shard:
                            flush()
                    n_games += 1
                    if limit_games and n_games >= limit_games:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        flush()
        srv.close()
    return n_rows, n_games, n_err


def _sp_worker(args):
    """Picklable pool worker: load agents from specs, play its pair-share."""
    wid, njobs, specs, seeds, both, out_dir, rows_per_shard, mirror = args
    agents = _load_specs(specs)
    done, _ = TC._done_episodes(out_dir)
    return _play_pairs(agents, seeds, both, out_dir, rows_per_shard,
                       f"shard_w{wid}_", done, wid=wid, njobs=njobs, mirror=mirror)


def _finalize(out_dir, n_specs, seeds, both, n_games, n_err, n_rows, t0):
    TC._build_index(out_dir)
    TC._write_stats_norm(out_dir)
    ip = os.path.join(out_dir, "index.parquet")
    total = pq.read_metadata(ip).num_rows if os.path.exists(ip) else n_rows
    json.dump(dict(source="selfplay", n_agents=n_specs, seeds=seeds,
                   both_seats=both, n_games_this_run=n_games, n_errors=n_err,
                   n_rows=int(total), token_layout_version=TC.TOKEN_LAYOUT_VERSION,
                   built=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    dt = time.time() - t0
    print(f"[selfplay] +{n_rows} rows / {n_games} games ({n_err} errs) in {dt:.1f}s "
          f"({n_games/max(dt,1e-6):.2f} games/s); total {total} rows -> {out_dir}")


def build_parallel(specs, jobs, seeds, both, out_dir, rows_per_shard, mirror=False):
    """Bounded-parallel self-play: ``jobs`` serve workers play concurrently, each
    its own pair-share + ``shard_w{wid}_`` files. THE speed lever -- the engine is
    microseconds, but each turn runs the Python agent over stdio, so N games in
    parallel is ~N x faster. Parent finalises the companions."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    npairs = len(specs) if mirror else len(specs) * (len(specs) - 1) // 2
    print(f"[selfplay] {len(specs)} agents; {jobs} workers; pairs={npairs}"
          f"{' (mirror)' if mirror else ''}; {seeds} seeds x 2 seats")
    try:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            res = list(ex.map(_sp_worker,
                              [(w, jobs, specs, seeds, both, out_dir, rows_per_shard, mirror)
                               for w in range(jobs)]))
    except Exception as e:
        print(f"[selfplay] pool interrupted: {e}; finalising partial")
        res = [(0, 0, 0)]
    nr = sum(r[0] for r in res); ng = sum(r[1] for r in res); ne = sum(r[2] for r in res)
    _finalize(out_dir, len(specs), seeds, both, ng, ne, nr, t0)


def build(n_agents=None, seeds=6, both=False, out_dir=OUT_DIR,
          rows_per_shard=50_000, limit_games=0, agents=None, jobs=1, mirror=False):
    os.makedirs(out_dir, exist_ok=True)
    specs = agents if agents is not None else list_agents(n_agents)
    if jobs and jobs > 1 and not limit_games:
        return build_parallel(specs, jobs, seeds, both, out_dir, rows_per_shard, mirror)
    t0 = time.time()
    npairs = len(specs) if mirror else len(specs) * (len(specs) - 1) // 2
    print(f"[selfplay] {len(specs)} agents (1 worker); {seeds} seeds; pairs={npairs}"
          f"{' (mirror)' if mirror else ''}")
    loaded = _load_specs(specs)
    done, _ = TC._done_episodes(out_dir)
    nr, ng, ne = _play_pairs(loaded, seeds, both, out_dir, rows_per_shard,
                             "shard_", done, limit_games=limit_games, mirror=mirror)
    _finalize(out_dir, len(specs), seeds, both, ng, ne, nr, t0)


def _smoke():
    out = os.path.join(ROOT, ".local", "scratch", "selfplay_smoke")
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    build(n_agents=5, seeds=2, both=False, out_dir=out, rows_per_shard=50_000,
          limit_games=8)
    # format check: schema + decode round-trip + shape parity vs a gm shard
    files = sorted(glob.glob(os.path.join(out, "shard_*.parquet")))
    assert files, "no self-play shard written"
    t = pq.read_table(files[0])
    assert t.schema.names == SCHEMA.names, ("schema mismatch", t.schema.names)
    r = t.slice(0, 1).to_pydict()
    nt = r["n_tokens"][0]
    toks = np.frombuffer(r["tokens"][0], np.int32).reshape(nt, TC.TOK_W)
    act = np.frombuffer(r["action"][0], np.int32)
    assert toks.shape == (nt, TC.TOK_W) and act.shape[0] == TC.ACT_W
    # compare to a real gm shard's schema if present
    gm = sorted(glob.glob(os.path.join(ROOT, "data", "trackp_corpus", "shard_*.parquet")))
    if gm:
        gschema = pq.read_schema(gm[0])
        assert gschema.names == SCHEMA.names, "gm vs selfplay schema mismatch!"
        assert [str(x) for x in gschema.types] == [str(x) for x in SCHEMA.types], \
            "gm vs selfplay dtype mismatch!"
        print("[smoke] schema MATCHES gm corpus:", SCHEMA.names)
    print(f"[smoke] OK tok0={toks.shape} act={act.shape} nt={nt} "
          f"rtg={r['rtg'][0]} wf={r['world_family'][0]}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", type=int, default=None, help="cap #agents (None=all)")
    ap.add_argument("--top-reactive", type=int, default=0,
                    help="use top-N REACTIVE crown-panel agents instead of agents/ dir")
    ap.add_argument("--extra-agents-dir", default=None,
                    help="also include every loadable agent(obs) *.py in this dir")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--both", action="store_true", help="keep loser rows too")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=min(3, os.cpu_count() or 2),
                    help="parallel serve workers (games run concurrently); kept "
                         "at 3 to stay gentle on a memory-constrained laptop")
    ap.add_argument("--mirror", action="store_true",
                    help="pair each agent with its own clone (exploiter diagnostics)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    agents = None
    if args.top_reactive:
        agents = top_reactive_agents(args.top_reactive)
        print(f"[selfplay] top-{args.top_reactive} reactive: "
              f"{[(n, round(r)) for n, _, r in agents]}")
    if args.extra_agents_dir and os.path.isdir(args.extra_agents_dir):
        extra = agents_from_dir(args.extra_agents_dir)
        have = {n for n, _, _ in (agents or [])}
        extra = [a for a in extra if a[0] not in have]
        agents = (agents or []) + extra
        print(f"[selfplay] +{len(extra)} extra agents from {args.extra_agents_dir}: "
              f"{[n for n, _, _ in extra]}")
    jobs = 1 if args.limit_games else args.jobs
    build(n_agents=args.agents, seeds=args.seeds, both=args.both, out_dir=args.out,
          rows_per_shard=args.rows_per_shard, limit_games=args.limit_games,
          agents=agents, jobs=jobs, mirror=args.mirror)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
