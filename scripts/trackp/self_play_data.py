#!/usr/bin/env python
"""(11) SELF-PLAY DATA GENERATOR.

Pairs of agents from the pool play each other across many worlds/seeds (clone vs
clone allowed) on the OFFICIAL engine, and every game is encoded into the EXACT
BC corpus format via trackp_corpus.episode_rows -- same tokenizer, same
winner-keep filtering (winner always; loser kept if rating >= 2100). Output is a
self-contained corpus dir BC can train on directly.

Agent pool (de-duped by path, --pool selects the sources; default "all"):
  crown       top reactive agents from models/crown_panel.json (real ladder
              ratings, faithful reactive kernels -- selfplay_corpus.top_reactive_agents)
  killers     the 5 agents we keep losing to (loss_forensics.KILLERS: tschinkel
              2945, alperen_rhythm, k0006 2494, nathanjacob, tetsutani)
  pub         agents/pub_*.py (the curated public head-to-head agents)
  top_agents  reactive agents in .local/top_agents/index.json (type=py; from agent_fetch)
plus any policy checkpoints you pass with --policies (play vs the field or a clone).

    python scripts/trackp/self_play_data.py --pairs 30 --seeds 8
    python scripts/trackp/self_play_data.py --pool crown,killers --pairs 40 --seeds 6
    python scripts/trackp/self_play_data.py --policies ckpts/policy_rl.pt --pairs 20 --seeds 6
    python scripts/trackp/self_play_data.py --all-pairs --seeds 4          # every combination

Output -> data/trackp_selfplay/  (shard_*.parquet + norm.json + index.parquet +
manifest.json).  Feed to BC with:  bc_train --corpus data/trackp_selfplay ...
(or as a teacher-refresh / delta source). Re-runs APPEND (shards continue).
"""
from __future__ import annotations
import argparse, glob, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "vendor"))
OUT_DIR = os.path.join(ROOT, "data", "trackp_selfplay")
TOP_INDEX = os.path.join(ROOT, ".local", "top_agents", "index.json")
DEFAULT_RATING = 2300.0                       # used when an agent has no score


def _load_pool(policies, sources):
    """[(name, agent_callable, rating)] from the selected sources, de-duped by
    resolved path. `sources` = subset of {top_agents, crown, killers, pub}."""
    import glob, json
    from kaggriculture.engine.serve_match import load_agent
    pool, seen = [], set()

    def _add(name, path, rating):
        if not path or not os.path.exists(path):
            return
        rp = os.path.realpath(path)
        if rp in seen:
            return
        try:
            ag = load_agent(path)
        except Exception as e:
            print(f"  skip {name} (load failed: {e})"); return
        seen.add(rp); pool.append((str(name)[:40], ag, float(rating or DEFAULT_RATING)))

    if "top_agents" in sources and os.path.exists(TOP_INDEX):
        idx = json.load(open(TOP_INDEX))
        for a in idx.get("agents", []):
            if a.get("type") != "py":
                print(f"  skip {a['name']} (type={a.get('type')} -- binary/tar not runnable as a callable)")
                continue
            _add(a["name"], os.path.join(os.path.dirname(TOP_INDEX), a["entry"]), a.get("score"))

    if "crown" in sources:                        # real ladder-rated reactive kernels
        try:
            from kaggriculture.data import selfplay_corpus as SC
            for name, path, rating in SC.top_reactive_agents(24):
                _add(name, path, rating)
        except Exception as e:
            print(f"  skip crown panel: {e}")

    if "killers" in sources:                       # the 5 agents we keep losing to
        try:
            from kaggriculture.bandit.gate.loss_forensics import KILLERS
            for name, path in KILLERS:
                _add(f"killer_{name}", path, 2600.0)
        except Exception as e:
            print(f"  skip killers: {e}")

    if "pub" in sources:                           # curated public head-to-head agents
        for p in sorted(glob.glob(os.path.join(ROOT, "agents", "pub_*.py"))):
            _add(os.path.splitext(os.path.basename(p))[0], p, None)

    for p in policies:
        try:
            from kaggriculture.trackp.policy_agent import make_agent
            pool.append((os.path.basename(p), make_agent(p), 2500.0))
        except Exception as e:
            print(f"  skip policy {p}: {e}")
    return pool


def _shard_start(out_dir):
    ex = sorted(glob.glob(os.path.join(out_dir, "shard_*.parquet")))
    if not ex:
        return 0
    return int(os.path.basename(ex[-1]).split("_")[1].split(".")[0]) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=30, help="number of (A,B) matchups to sample")
    ap.add_argument("--seeds", type=int, default=8, help="worlds/seeds played per matchup")
    ap.add_argument("--all-pairs", action="store_true", help="play EVERY agent combination (incl. self)")
    ap.add_argument("--pool", default="all",
                    help="comma-sep sources: crown,killers,pub,top_agents (or 'all')")
    ap.add_argument("--policies", default="", help="comma-sep policy ckpts to add to the pool")
    ap.add_argument("--rows-per-shard", type=int, default=50_000)
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()

    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    import kaggriculture.data.trackp_corpus as TC
    from kaggle_environments import make
    if hasattr(TC, "load_lookups"):
        try: TC.load_lookups()
        except Exception: pass

    ALL = ["crown", "killers", "pub", "top_agents"]
    sources = ALL if a.pool.strip().lower() == "all" else [s.strip() for s in a.pool.split(",") if s.strip()]
    pool = _load_pool([p for p in a.policies.split(",") if p], sources)
    if len(pool) < 1:
        raise SystemExit(f"empty agent pool (sources={sources}) -- check models/crown_panel.json / "
                         "agents/pub_*.py exist, run agent_fetch.py --top N, or pass --policies")
    print(f"[selfplay] pool={len(pool)} agents (sources={sources}); clone-vs-clone allowed")

    # matchups: every combination (incl. self) or a random sample (incl. self)
    K = len(pool)
    if a.all_pairs:
        matchups = [(i, j) for i in range(K) for j in range(i, K)]
    else:
        rng = random.Random(a.seed0)
        matchups = [(rng.randrange(K), rng.randrange(K)) for _ in range(a.pairs)]

    os.makedirs(OUT_DIR, exist_ok=True)
    names = TC.SHARD_SCHEMA.names
    shard_i = _shard_start(OUT_DIR)
    buf = {k: [] for k in names}
    n_rows = 0; n_eps = 0; eid = shard_i * 10_000_000 + a.seed0    # unique-ish, resumable
    t0 = time.time()

    def flush():
        nonlocal shard_i
        if not buf["episode_id"]:
            return
        pq.write_table(pa.table(buf, schema=TC.SHARD_SCHEMA),
                       os.path.join(OUT_DIR, f"shard_{shard_i:04d}.parquet"),
                       compression="zstd", compression_level=10)
        print(f"  wrote shard_{shard_i:04d} ({len(buf['episode_id'])} rows, {n_rows} total)")
        shard_i += 1
        for k in buf: buf[k].clear()

    for (i, j) in matchups:
        nA, agA, rA = pool[i]; nB, agB, rB = pool[j]
        for s in range(a.seeds):
            seed = a.seed0 + s + 7 * (i * K + j)
            eid += 1
            try:
                env = make("kaggriculture", configuration={"seed": seed}, debug=False)
                env.run([agA, agB])
            except Exception as e:
                print(f"  game {nA} vs {nB} seed{seed} FAILED: {e}"); continue
            b0 = float(env.steps[-1][0]["reward"] or 0); b1 = float(env.steps[-1][1]["reward"] or 0)
            rep = {"steps": env.steps}
            banks = {str(eid): {0: (b0, rA), 1: (b1, rB)}}
            got = 0
            for row in TC.episode_rows(eid, rep, banks):
                for k, v in zip(names, row): buf[k].append(v)
                got += 1
            n_rows += got; n_eps += 1
            if got:
                tag = "CLONE" if i == j else "vs"
                print(f"  {nA[:22]} {tag} {nB[:22]} seed{seed}: {b0:.0f}/{b1:.0f} -> +{got} rows")
            if len(buf["episode_id"]) >= a.rows_per_shard:
                flush()
    flush()

    # finalize into a BC-loadable corpus (norm/index/manifest)
    TC._write_stats_norm(OUT_DIR)
    TC._build_index(OUT_DIR)
    TC._write_meta(OUT_DIR, True, n_rows, n_eps, a.rows_per_shard, t0)
    print("\n" + "=" * 64)
    print(f"SELF-PLAY DATA: {n_eps} games -> {n_rows:,} rows in {OUT_DIR}")
    print(f"train BC on it:  bc_train --corpus {os.path.relpath(OUT_DIR, ROOT)} ...")
    print("=" * 64)


if __name__ == "__main__":
    main()
