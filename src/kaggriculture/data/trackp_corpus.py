"""Trackp BC corpus builder (T1.2/T1.3) -- stream D:\\gm_dataset replays into a
sharded token corpus for the end-to-end per-turn policy.

Grounded on the real 1.32.7 replay schema (see docs/history/trackp-token-spec.md,
TOKEN_LAYOUT_VERSION=1). Everything STREAMS: one replay batch at a time, rows
flushed to a ParquetWriter per shard, nothing held whole.

Per decision we store: packed int32 token array (the state), a packed int32
composite action, and lightweight metadata (episode/seat/day/step/rtg/rating/
world_family/split). Companions (manifest/vocab/index) are written alongside.

    python -m kaggriculture.data.trackp_corpus --smoke              # ~20 eps, validate + rate
    python -m kaggriculture.data.trackp_corpus --limit 0            # full 1.32.7 corpus
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:                                            # 2x faster JSON parse, no extra RAM
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

GM = r"D:\gm_dataset"
OUT_DIR = os.path.join(ROOT, "data", "trackp_corpus")
TOKEN_LAYOUT_VERSION = 2        # v2: verb-aware action args (no item collisions)
VAL_FRACTION = 0.15
LOSER_RATING_MIN = 2100.0
ENGINE_KEEP = "1.32.7"

# --- vocab (frozen; mirror in Rust obstoken.rs) ------------------------------
CROPS = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"]
ANIMALS = ["COW", "GOOSE", "SHEEP"]
PRODUCTS = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK",
            "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"]
SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]
MOVER_VERBS = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER",
               "HARVEST", "FEED", "CARE", "PLACE", "PICKUP", "DROP", "DIG",
               "FERTILIZE", "COLLECT_FERTILIZER", "BUILD_PASTURE", "BUILD_COOP"]
MARKET_VERBS = ["PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL",
                "HIRE", "BUY_LAND"]
KINDS = ["EMPTY", "PLANT", "PASTURE", "COOP", "WEED"]
MAX_HANDS = 12
TOK_W = 14                                   # fixed fields per token (int32)
MAX_TOKENS = 160
MAX_MARKET = 10
# action blob layout: [f_verb,f_arg] + 12*[verb,arg] + 10*[verb,arg1,arg2]
ACT_W = 2 + MAX_HANDS * 2 + MAX_MARKET * 3

_id = lambda lst: {v: i for i, v in enumerate(lst)}
CROP_ID, ANIM_ID, PROD_ID, SHOP_ID = _id(CROPS), _id(ANIMALS), _id(PRODUCTS), _id(SHOPS)
MV_ID, MK_ID, KIND_ID = _id(MOVER_VERBS), _id(MARKET_VERBS), _id(KINDS)

# token type tags
T_EMPTY, T_PLANT, T_PASTURE, T_COOP, T_WEED = 0, 1, 2, 3, 4
T_SHED, T_MARKET, T_HAND, T_FARMER, T_GAME, T_RTG = 6, 7, 8, 9, 10, 11
KIND_TOK = {"PLANT": T_PLANT, "PASTURE": T_PASTURE, "COOP": T_COOP, "WEED": T_WEED}


def split_of(eid) -> int:
    h = int(hashlib.sha1(str(eid).encode()).hexdigest()[:8], 16)
    return 1 if (h % 10000) / 10000.0 < VAL_FRACTION else 0    # 1=val 0=train


def _tok(type_, x=0, y=0, ca=0, cb=0, vals=()):
    row = [type_, x, y, ca, cb] + list(vals)
    return row + [0] * (TOK_W - len(row))


def world_family(steps) -> int:
    """day-6 unlocked_shops[:2] -> a small family id (stable hash)."""
    sig = ""
    for cell in steps:
        obs = (cell[0] or {}).get("observation") or {}
        if obs.get("day") == 6:
            sh = ((obs.get("town") or {}).get("unlocked_shops") or [])[:2]
            sig = "|".join(sorted(sh)); break
    if not sig:
        return 0
    return 1 + int(hashlib.sha1(sig.encode()).hexdigest()[:6], 16) % 4095


def encode_tokens(obs, seat) -> np.ndarray:
    farms = obs.get("farms") or []
    me = farms[seat] if seat < len(farms) else {}
    opp = farms[1 - seat] if len(farms) > 1 else {}
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    invs = priv.get("inventories") or []
    market = obs.get("market") or {}
    prices = market.get("prices") or {}
    minv = market.get("inventory") or {}
    shops = set((obs.get("town") or {}).get("unlocked_shops") or [])
    day = int(obs.get("day", 0)); hour = int(obs.get("hour", 0))
    step = int(obs.get("step", 0))
    toks = []
    # tiles (ours)
    tiles = me.get("tiles") or []
    for y, row in enumerate(tiles):
        for x, t in enumerate(row):
            if not isinstance(t, dict):
                continue
            kind = t.get("kind")
            tt = KIND_TOK.get(kind)
            if tt is None:
                continue
            crop = CROP_ID.get(t.get("crop"), -1) + 1
            anim = ANIM_ID.get(t.get("animal"), -1) + 1
            vals = [int(t.get("yield_units", 0)),
                    1 if t.get("watered_today") else 0,
                    1 if t.get("cared_today") else 0,
                    1 if t.get("fed_today") else 0,
                    int(t.get("consecutive_unwatered", 0)),
                    int(t.get("consecutive_unfed", 0)),
                    1 if t.get("fertilizer_available") else 0,
                    max(0, day - int(t.get("planted_day", day))),
                    max(0, day - int(t.get("placed_day", day)))]
            toks.append(_tok(tt, x, y, crop, anim, vals))
            if len(toks) >= MAX_TOKENS - 30:
                break
        if len(toks) >= MAX_TOKENS - 30:
            break
    # shed (one token per product held, compact into value slots)
    toks.append(_tok(T_SHED, vals=[int(shed.get(p, 0) or 0) for p in PRODUCTS[:9]]))
    # market: one token per product (price, inventory)
    for pi, p in enumerate(PRODUCTS):
        toks.append(_tok(T_MARKET, ca=pi,
                         vals=[int(prices.get(p, 0) or 0), int(minv.get(p, 0) or 0)]))
    # farmer + hands
    fp = me.get("farmer") or [0, 0]
    fcar = invs[0] if len(invs) > 0 else {}
    toks.append(_tok(T_FARMER, int(fp[0]), int(fp[1]),
                     vals=[sum(int(v) for v in (fcar or {}).values())]))
    for hi, hp in enumerate((me.get("hands") or [])[:MAX_HANDS]):
        car = invs[hi + 1] if len(invs) > hi + 1 else {}
        toks.append(_tok(T_HAND, int(hp[0]), int(hp[1]), ca=hi,
                         vals=[sum(int(v) for v in (car or {}).values())]))
    # game-state
    shopmulti = 0
    for s in shops:
        if s in SHOP_ID:
            shopmulti |= (1 << SHOP_ID[s])
    toks.append(_tok(T_GAME, vals=[day, hour, step // 24,
                     int(me.get("money", 0) or 0) // 16,
                     int(opp.get("money", 0) or 0) // 16,
                     len(me.get("hands") or []),
                     len(me.get("unlocked_quadrants") or []),
                     int(me.get("hires_today", 0) or 0), shopmulti]))
    a = np.zeros((min(len(toks), MAX_TOKENS), TOK_W), np.int32)
    for i, t in enumerate(toks[:MAX_TOKENS]):
        a[i] = t
    return a


def _mover_arg(verb, arg) -> int:
    """VERB-AWARE arg (layout v2): no cross-item collisions. PLANT->crop,
    PLACE->animal, PICKUP->product; each maps 1:1 within its verb."""
    if arg is None:
        return 0
    if verb == "PLANT":
        return CROP_ID.get(arg, -1) + 1
    if verb == "PLACE":
        return ANIM_ID.get(arg, -1) + 1
    if verb == "PICKUP":
        return PROD_ID.get(arg, -1) + 1
    return 0


def _market_arg(verb, arg) -> int:
    if arg is None:
        return 0
    if verb == "BUY_SEED":
        return CROP_ID.get(arg, -1) + 1
    if verb == "BUY_ANIMAL":
        return ANIM_ID.get(arg, -1) + 1
    if verb in ("BUY_PRODUCT", "SELL"):
        return PROD_ID.get(arg, -1) + 1
    return 0


def encode_action(act) -> np.ndarray:
    out = np.zeros(ACT_W, np.int32)
    fm = act.get("farmer")
    if isinstance(fm, list) and fm:
        out[0] = MV_ID.get(fm[0], 0)
        out[1] = _mover_arg(fm[0], fm[1] if len(fm) > 1 else None)
    o = 2
    for h in (act.get("hands") or [])[:MAX_HANDS]:
        if isinstance(h, list) and h:
            out[o] = MV_ID.get(h[0], 0)
            out[o + 1] = _mover_arg(h[0], h[1] if len(h) > 1 else None)
        o += 2
    o = 2 + MAX_HANDS * 2
    for m in (act.get("market") or [])[:MAX_MARKET]:
        if isinstance(m, list) and m:
            out[o] = MK_ID.get(m[0], 0)
            out[o + 1] = _market_arg(m[0], m[1] if len(m) > 1 else None)
            out[o + 2] = int(m[2]) if len(m) > 2 and str(m[2]).lstrip("-").isdigit() else 0
        o += 3
    return out


def load_lookups():
    """engine_version per (episode) and per-(episode,seat) bank+rating."""
    import csv
    eng = {}
    with open(os.path.join(GM, "episode_features.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            eng[r["episode_id"]] = r["engine_version"]
    banks = {}
    with open(os.path.join(GM, "agents.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            eid = r["episode_id"]; seat = int(r["agent_index"])
            try:
                bank = float(r["final_bank"] or 0)
                rat = float(r["rating_after"] or 0)
            except ValueError:
                continue
            banks.setdefault(eid, {})[seat] = (bank, rat)
    return eng, banks


def iter_replays(limit, keep=None):
    """Stream (episode_id, replay) newest shard first. If ``keep`` (a set of
    episode_id strings) is given, skip the expensive json.loads for any episode
    not in it -- so non-1.32.7 replays (e.g. the 9.8 GB 1.32.6-heavy Aug shard)
    are never parsed."""
    n = 0
    for f in sorted(glob.glob(os.path.join(GM, "replays_*.parquet")), reverse=True):
        try:
            pf = pq.ParquetFile(f)
        except Exception:
            continue
        for b in pf.iter_batches(batch_size=2, columns=["episode_id", "replay_json"]):
            eids = b.column("episode_id").to_pylist()
            rjs = b.column("replay_json").to_pylist()
            for eid, rj in zip(eids, rjs):
                if keep is not None and str(eid) not in keep:
                    continue
                try:
                    yield str(eid), _loads(rj)
                except (ValueError, TypeError):
                    continue
                n += 1
                if limit and n >= limit:
                    return


IDX_COLS = ("episode_id", "seat", "day", "step", "split", "rtg", "rating",
            "world_family")


def _done_episodes(out_dir):
    """Episode ids already present in written shards (resume support)."""
    done = set()
    mx = -1
    for sp in glob.glob(os.path.join(out_dir, "shard_*.parquet")):
        base = os.path.basename(sp)
        try:
            mx = max(mx, int(base[len("shard_"):-len(".parquet")]))
        except ValueError:
            pass
        try:
            t = pq.read_table(sp, columns=["episode_id"])
            done.update(str(e) for e in t.column("episode_id").to_pylist())
        except Exception:
            # truncated/corrupt shard from a mid-write reap -> drop & regenerate
            try:
                os.remove(sp)
                print(f"[trackp_corpus] removed corrupt shard {os.path.basename(sp)}")
            except OSError:
                pass
    return done, mx + 1


def _write_stats_norm(out_dir, sample_rows=200_000):
    """stats.json (from index) + norm.json (per-field mean/std over a TRAIN sample)."""
    ip = os.path.join(out_dir, "index.parquet")
    stats = {}
    if os.path.exists(ip):
        it = pq.read_table(ip, columns=["split", "rtg", "rating", "world_family"])
        import numpy as _np
        sp = _np.array(it.column("split")); rtg = _np.array(it.column("rtg"))
        rat = _np.array(it.column("rating")); wf = _np.array(it.column("world_family"))
        stats = dict(n_rows=int(sp.shape[0]),
                     train=int((sp == 0).sum()), val=int((sp == 1).sum()),
                     win_rate=float((rtg == 1.0).mean()),
                     rating_mean=float(rat.mean()), rating_p50=float(_np.median(rat)),
                     n_world_families=int(len(set(wf.tolist()))))
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    # norm: per-field mean/std over tokens of a train-shard sample
    import numpy as _np
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.parquet")))
    acc = _np.zeros(TOK_W); acc2 = _np.zeros(TOK_W); cnt = 0
    for sp_path in shards:
        t = pq.read_table(sp_path, columns=["split", "n_tokens", "tokens"])
        d = t.to_pydict()
        for i in range(t.num_rows):
            if d["split"][i] != 0:
                continue
            nt = d["n_tokens"][i]
            a = _np.frombuffer(d["tokens"][i], _np.int32).reshape(nt, TOK_W).astype(_np.float64)
            acc += a.sum(0); acc2 += (a * a).sum(0); cnt += nt
            if cnt >= sample_rows:
                break
        if cnt >= sample_rows:
            break
    if cnt > 0:
        mean = acc / cnt
        var = _np.maximum(acc2 / cnt - mean * mean, 1e-6)
        json.dump(dict(mean=mean.tolist(), std=_np.sqrt(var).tolist(),
                       n_tokens_sampled=int(cnt)),
                  open(os.path.join(out_dir, "norm.json"), "w"), indent=1)


def _build_index(out_dir):
    """Stream shard light-columns → index.parquet (shard_id, row_in_shard added).
    Reads only the metadata columns, one shard at a time — RAM stays flat."""
    shards = sorted(glob.glob(os.path.join(out_dir, "shard_*.parquet")))
    iw = None
    try:
        idx_schema = pa.schema([(c, pa.int64() if c in ("episode_id",) else
                                 (pa.float32() if c in ("rtg", "rating") else pa.int32()))
                                for c in IDX_COLS]
                               + [("shard_id", pa.int32()), ("row_in_shard", pa.int32())])
        iw = pq.ParquetWriter(os.path.join(out_dir, "index.parquet"), idx_schema,
                              compression="zstd")
        for sid, sp in enumerate(shards):
            t = pq.read_table(sp, columns=list(IDX_COLS))
            n = t.num_rows
            cols = {c: t.column(c) for c in IDX_COLS}
            cols["shard_id"] = pa.array([sid] * n, pa.int32())
            cols["row_in_shard"] = pa.array(list(range(n)), pa.int32())
            iw.write_table(pa.table(cols, schema=idx_schema))
    finally:
        if iw is not None:
            iw.close()


SHARD_SCHEMA = pa.schema([
    ("episode_id", pa.int64()), ("seat", pa.int8()), ("day", pa.int16()),
    ("step", pa.int16()), ("split", pa.int8()), ("rtg", pa.float32()),
    ("rating", pa.float32()), ("world_family", pa.int16()),
    ("n_tokens", pa.int16()), ("tokens", pa.binary()), ("action", pa.binary())])


def episode_rows(eid, rep, banks):
    """Yield corpus row-tuples (in SHARD_SCHEMA column order) for one episode:
    winner seat always, loser seat if rating >= LOSER_RATING_MIN."""
    steps = rep.get("steps") or []
    if len(steps) < 24:
        return
    bk = banks.get(str(eid), {})
    if len(bk) < 2:
        return
    wf = world_family(steps)
    b0 = bk.get(0, (0, 0))[0]; b1 = bk.get(1, (0, 0))[0]
    win = {0: 1.0 if b0 > b1 else (0.5 if b0 == b1 else 0.0)}
    win[1] = 1.0 - win[0] if win[0] != 0.5 else 0.5
    sp = split_of(eid)
    winner = 0 if b0 >= b1 else 1
    seats = [winner]
    loser = 1 - winner
    if bk.get(loser, (0, 0))[1] >= LOSER_RATING_MIN:
        seats.append(loser)
    for seat in seats:
        rat = bk.get(seat, (0, 0))[1]
        for t in range(len(steps)):
            cell = steps[t][seat] if seat < len(steps[t]) else None
            if not cell:
                continue
            obs = cell.get("observation") or {}
            if obs.get("player") != seat:          # obs is seat-specific
                continue
            toks = encode_tokens(obs, seat)
            acta = encode_action(cell.get("action") or {})
            yield (int(eid), seat, int(obs.get("day", 0)),
                   int(obs.get("step", t)), sp, float(win[seat]), float(rat),
                   int(wf), int(toks.shape[0]), toks.tobytes(), acta.tobytes())


def _stream_to_shards(keep, banks, out_dir, rows_per_shard, prefix, shard_start,
                      limit=0):
    """Consume a keep-filtered replay stream into shards named ``{prefix}{k}``.
    Returns (n_rows, n_eps, processed)."""
    names = SHARD_SCHEMA.names
    shard_i = shard_start
    buf = {k: [] for k in names}
    n_rows = 0; n_eps = 0; processed = 0

    def flush():
        nonlocal shard_i
        if not buf["episode_id"]:
            return
        pq.write_table(pa.table(buf, schema=SHARD_SCHEMA),
                       os.path.join(out_dir, f"{prefix}{shard_i:04d}.parquet"),
                       compression="zstd", compression_level=10)
        shard_i += 1
        for k in buf:
            buf[k].clear()

    for eid, rep in iter_replays(limit, keep=keep):
        processed += 1
        used = False
        for row in episode_rows(eid, rep, banks):
            for k, v in zip(names, row):
                buf[k].append(v)
            n_rows += 1; used = True
            if len(buf["episode_id"]) >= rows_per_shard:
                flush()
        if used:
            n_eps += 1
    flush()
    return n_rows, n_eps, processed


def _worker_shard_start(out_dir, prefix):
    start = 0
    for sp in glob.glob(os.path.join(out_dir, f"{prefix}*.parquet")):
        try:
            start = max(start, int(os.path.basename(sp)[len(prefix):-8]) + 1)
        except ValueError:
            pass
    return start


def _worker_main(args):
    """Top-level (picklable, spawn-safe) worker: process the hash-assigned share
    of the current-engine episodes into its own ``shard_w{wid}_*`` files."""
    wid, njobs, out_dir, rows_per_shard, limit = args
    eng, banks = load_lookups()
    keep = {e for e, v in eng.items() if v == ENGINE_KEEP}
    del eng
    done, _ = _done_episodes(out_dir)
    todo = {e for e in (keep - done)
            if int(hashlib.sha1(e.encode()).hexdigest()[:8], 16) % njobs == wid}
    prefix = f"shard_w{wid}_"
    start = _worker_shard_start(out_dir, prefix)
    return _stream_to_shards(todo, banks, out_dir, rows_per_shard, prefix, start,
                             limit=limit)


def build_parallel(njobs=2, out_dir=OUT_DIR, rows_per_shard=50_000, limit=0):
    """Bounded-parallel build: ``njobs`` workers each stream the files and
    process their hash-assigned episode share (batch_size=2 -> ~lean RAM/worker),
    writing own resume-aware shards. Parent finalises the companions."""
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    complete = True
    try:
        with ProcessPoolExecutor(max_workers=njobs) as ex:
            results = list(ex.map(_worker_main,
                                  [(w, njobs, out_dir, rows_per_shard, limit)
                                   for w in range(njobs)]))
    except Exception as e:                       # a worker died (e.g. reaped)
        print(f"[trackp_corpus] worker pool interrupted: {e}; finalising partial")
        results = []; complete = False
    n_rows = sum(r[0] for r in results); n_eps = sum(r[1] for r in results)
    _build_index(out_dir)
    _write_stats_norm(out_dir)
    _write_meta(out_dir, complete, n_rows, n_eps, rows_per_shard, t0)
    return dict(complete=complete)


def _write_meta(out_dir, complete, n_rows, n_eps, rows_per_shard, t0):
    vocab = dict(crops=CROPS, animals=ANIMALS, products=PRODUCTS, shops=SHOPS,
                 mover_verbs=MOVER_VERBS, market_verbs=MARKET_VERBS, kinds=KINDS,
                 max_hands=MAX_HANDS, tok_w=TOK_W, max_tokens=MAX_TOKENS,
                 act_w=ACT_W, max_market=MAX_MARKET)
    json.dump(vocab, open(os.path.join(out_dir, "vocab.json"), "w"), indent=1)
    ip = os.path.join(out_dir, "index.parquet")
    total = pq.read_metadata(ip).num_rows if os.path.exists(ip) else n_rows
    total_shards = len(glob.glob(os.path.join(out_dir, "shard_*.parquet")))
    manifest = dict(token_layout_version=TOKEN_LAYOUT_VERSION, engine=ENGINE_KEEP,
                    val_fraction=VAL_FRACTION, loser_rating_min=LOSER_RATING_MIN,
                    n_rows=int(total), n_shards=total_shards, complete=bool(complete),
                    rows_added_this_run=n_rows, eps_added_this_run=n_eps,
                    rows_per_shard=rows_per_shard, pyarrow=pa.__version__,
                    built=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(f"[trackp_corpus] +{n_rows} rows / +{n_eps} eps this run; "
          f"total {total} rows / {total_shards} shards in {time.time()-t0:.1f}s "
          f"complete={complete}  -> {out_dir}")
    return manifest


def build(limit=0, out_dir=OUT_DIR, rows_per_shard=200_000):
    os.makedirs(out_dir, exist_ok=True)
    eng, banks = load_lookups()
    keep = {e for e, v in eng.items() if v == ENGINE_KEEP}    # skip non-1.32.7 parse
    del eng                                                    # free RAM
    done, shard_start = _done_episodes(out_dir)                # resume support
    todo = keep - done
    print(f"[trackp_corpus] {len(keep)} @ {ENGINE_KEEP}; {len(done)} done; "
          f"{len(todo)} to process; resume shard #{shard_start}")
    t0 = time.time()
    n_rows, n_eps, processed = _stream_to_shards(
        todo, banks, out_dir, rows_per_shard, "shard_", shard_start, limit=limit)
    complete = (limit == 0) and (processed >= len(todo))       # all remaining done
    # companions -- index.parquet built by a streaming post-pass over the shards'
    # LIGHT columns only (never holds the corpus or the full index in RAM).
    _build_index(out_dir)
    _write_stats_norm(out_dir)
    return _write_meta(out_dir, complete, n_rows, n_eps, rows_per_shard, t0)


def _smoke():
    out = os.path.join(ROOT, ".local", "scratch", "trackp_corpus_smoke")
    m = build(limit=60, out_dir=out, rows_per_shard=50_000)
    # decode round-trip sanity on one row
    files = sorted(glob.glob(os.path.join(out, "shard_*.parquet")))
    assert files, "no shard written"
    t = pq.read_table(files[0])
    row0 = t.slice(0, 1).to_pydict()
    nt = row0["n_tokens"][0]
    toks = np.frombuffer(row0["tokens"][0], np.int32).reshape(nt, TOK_W)
    act = np.frombuffer(row0["action"][0], np.int32)
    assert toks.shape == (nt, TOK_W) and act.shape[0] == ACT_W
    assert os.path.exists(os.path.join(out, "index.parquet"))
    assert os.path.exists(os.path.join(out, "vocab.json"))
    print(f"[smoke] OK rows={m['n_rows']} shards={m['n_shards']} "
          f"tok0 shape={toks.shape} act shape={act.shape}")
    total = sum(os.path.getsize(f) for f in files)
    print(f"[smoke] shard bytes={total} ({total/max(m['n_rows'],1):.0f} B/row) "
          f"-> extrapolate 100M ≈ {total/max(m['n_rows'],1)*100e6/1e9:.1f} GB")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="cap #episodes (0=all)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--rows-per-shard", type=int, default=200_000)
    ap.add_argument("--jobs", type=int, default=1,
                    help="bounded-parallel workers (>1 uses build_parallel)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--finalize", action="store_true",
                    help="build index/stats/norm/meta over an EXISTING shard dir "
                         "(e.g. one produced by the Rust extractor) -- no extraction")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    if args.finalize:
        t0 = time.time()
        _build_index(args.out)
        _write_stats_norm(args.out)
        n = pq.read_metadata(os.path.join(args.out, "index.parquet")).num_rows
        _write_meta(args.out, complete=True, n_rows=n, n_eps=0,
                    rows_per_shard=args.rows_per_shard, t0=t0)
        print(f"[trackp_corpus] finalized {args.out}: {n} rows / "
              f"{len(glob.glob(os.path.join(args.out, 'shard_*.parquet')))} shards")
        return 0
    if args.jobs > 1:
        build_parallel(njobs=args.jobs, out_dir=args.out,
                       rows_per_shard=args.rows_per_shard)
    else:
        build(limit=args.limit, out_dir=args.out, rows_per_shard=args.rows_per_shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
