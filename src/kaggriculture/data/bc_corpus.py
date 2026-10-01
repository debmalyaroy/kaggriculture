"""E0.1 -- distil the top-100 replay corpus into a BC/RL training corpus.

Streams ``.local/scratch/gm/top100_replays_parts/*.parquet`` (episode_id,
replay_json) and, for the WINNER seat (``--both`` also the loser), writes two
parquet tables under ``data/bc_corpus/``:

  * MICRO rows -- one per (episode, seat, step): a compact per-turn
    ``obs_features`` vector, the raw action (kept as ``action_json`` for full
    fidelity plus a few decoded summaries), the game's ``final_return`` for
    that seat (SCORE, draws = 0.5 -- the currency the ladder actually pays),
    and the ``world_sig``.
  * MACRO rows -- one per (episode, seat, day): the day's aggregated
    ``macro_state`` and the distilled daily-plan action
    (``kaggriculture.train.macro_actions``), the ``primary_class`` label,
    ``final_return`` and ``world_sig``.

``world_sig`` = the day-6 ``unlocked_shops[:2]`` signature ("S1|S2"), the
field's known dispatch key (docs: realized world, 2026-09-07). The split
column is a DETERMINISTIC hash of episode_id, so a val episode never bleeds
into train.

Memory is bounded: replays are streamed one batch at a time and micro rows
are flushed to a ``ParquetWriter`` every ``--flush`` rows -- nothing is held
whole.

    # first corpus on an 800-replay subsample, winner seats:
    python -m kaggriculture.data.bc_corpus --limit 800
    # both seats, whole corpus, custom out:
    python -m kaggriculture.data.bc_corpus --both --out data/bc_corpus
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

import kaggriculture.train.macro_actions as MA

try:                                        # console may be cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PARTS = os.path.join(ROOT, ".local", "scratch", "gm", "top100_replays_parts")
DESTBRESO = os.path.join(ROOT, ".local", "scratch", "datasets", "destbreso",
                         "matchups_all.parquet")
OUT_DIR = os.path.join(ROOT, "data", "bc_corpus")
TURNS_PER_DAY = 24
VAL_FRACTION = 0.15          # by-episode holdout

PRODUCTS = MA.PRODUCTS
SEEDS = MA.SEEDS
PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")


# --------------------------------------------------------------------------- #
# streaming input
# --------------------------------------------------------------------------- #
def iter_replays(parts_dir, limit):
    n = 0
    for f in sorted(glob.glob(os.path.join(parts_dir, "*.parquet"))):
        try:
            pf = pq.ParquetFile(f)
        except Exception:                                   # noqa: BLE001
            continue
        for b in pf.iter_batches(batch_size=8,
                                 columns=["episode_id", "replay_json"]):
            eids = b.column("episode_id").to_pylist()
            rjs = b.column("replay_json").to_pylist()
            for eid, rj in zip(eids, rjs):
                try:
                    rep = json.loads(rj)
                except (ValueError, TypeError):
                    continue
                yield str(eid), rep
                n += 1
                if limit and n >= limit:
                    return


def split_of(episode_id: str) -> str:
    h = int(hashlib.sha1(str(episode_id).encode()).hexdigest()[:8], 16)
    return "val" if (h % 10000) / 10000.0 < VAL_FRACTION else "train"


def world_sig(steps) -> str:
    """Day-6 unlocked_shops[:2] joined 'S1|S2' (blank if never realised)."""
    for cell in steps:
        obs = (cell[0] or {}).get("observation") or {}
        if obs.get("day") == 6:
            sh = ((obs.get("town") or {}).get("unlocked_shops") or [])[:2]
            return "|".join(sh)
    # fall back to the last observed day's shops
    for cell in reversed(steps):
        obs = (cell[0] or {}).get("observation") or {}
        sh = ((obs.get("town") or {}).get("unlocked_shops") or [])[:2]
        if sh:
            return "|".join(sh)
    return ""


# --------------------------------------------------------------------------- #
# per-step feature extraction  (engine-independent, cheap, bounded)
# --------------------------------------------------------------------------- #
MICRO_FEATURES = (
    "day", "hour", "step", "money", "opp_money", "money_diff",
    "labor", "tiles_owned", "tiles_occupied", "shed_total", "seed_total",
    "mean_premium_price", "n_shops", "n_market_orders", "sell_units",
    "buy_orders",
)


def _count_tiles(farm):
    owned = occ = 0
    for row in (farm.get("tiles") or []):
        for t in row:
            if t == "LOCKED":
                continue
            owned += 1
            if t is not None:
                occ += 1
    return owned, occ


def micro_features(obs, seat):
    farms = obs.get("farms") or []
    me = farms[seat] if seat < len(farms) else {}
    opp = farms[1 - seat] if (1 - seat) < len(farms) else {}
    money = float((me or {}).get("money", 0) or 0)
    opp_money = float((opp or {}).get("money", 0) or 0)
    owned, occ = _count_tiles(me or {})
    priv = obs.get("private") or {}
    shed = priv.get("shed") or {}
    seeds = priv.get("seeds") or {}
    prices = (obs.get("market") or {}).get("prices") or {}
    prem = [float(prices.get(p, 0) or 0) for p in PREMIUM]
    act = obs  # not used; kept for clarity
    shops = (obs.get("town") or {}).get("unlocked_shops") or []
    return {
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "step": int(obs.get("step", 0) or 0),
        "money": money,
        "opp_money": opp_money,
        "money_diff": money - opp_money,
        "labor": len(me.get("hands") or []) if me else 0,
        "tiles_owned": owned,
        "tiles_occupied": occ,
        "shed_total": float(sum(int(v or 0) for v in shed.values())),
        "seed_total": float(sum(int(v or 0) for v in seeds.values())),
        "mean_premium_price": sum(prem) / len(prem) if prem else 0.0,
        "n_shops": len(shops),
    }


def _action_summary(action):
    """(n_market_orders, sell_units, buy_orders, farmer_verb) from an action."""
    market = action.get("market") or []
    n_orders = len(market)
    sell_units = buy_orders = 0
    for o in market:
        if isinstance(o, list) and o:
            if o[0] == "SELL" and len(o) >= 3:
                try:
                    sell_units += max(0, int(o[2]))
                except (TypeError, ValueError):
                    pass
            elif o[0].startswith("BUY") or o[0] == "HIRE":
                buy_orders += 1
    fm = action.get("farmer")
    farmer_verb = fm[0] if isinstance(fm, list) and fm else "PASS"
    return n_orders, sell_units, buy_orders, farmer_verb


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
def _micro_schema():
    fields = [("episode_id", pa.string()), ("seat", pa.int8()),
              ("split", pa.string()), ("world_sig", pa.string()),
              ("final_return", pa.float32())]
    for k in MICRO_FEATURES:
        fields.append((k, pa.float32()))
    fields += [("farmer_verb", pa.string()), ("action_json", pa.string())]
    return pa.schema(fields)


def _macro_schema():
    fields = [("episode_id", pa.string()), ("seat", pa.int8()),
              ("day", pa.int16()), ("split", pa.string()),
              ("world_sig", pa.string()), ("final_return", pa.float32()),
              ("primary_class", pa.string())]
    for k in MA.FEATURE_NAMES:
        fields.append((k, pa.float32()))
    return pa.schema(fields)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(limit, both, out_dir, flush_rows, verbose=True):
    import kaggriculture.measure.win_metric as WM
    os.makedirs(out_dir, exist_ok=True)
    micro_path = os.path.join(out_dir, "micro.parquet")
    macro_path = os.path.join(out_dir, "macro.parquet")
    m_schema = _micro_schema()
    macro_schema = _macro_schema()

    micro_writer = pq.ParquetWriter(micro_path, m_schema, compression="zstd")
    micro_buf = {k: [] for k in m_schema.names}
    macro_buf = {k: [] for k in macro_schema.names}
    n_micro = n_macro = n_games = n_skip = 0
    class_bal = Counter()
    split_bal = Counter()

    def flush_micro():
        nonlocal micro_buf
        if not micro_buf["episode_id"]:
            return
        tbl = pa.table({k: pa.array(v, type=m_schema.field(k).type)
                        for k, v in micro_buf.items()}, schema=m_schema)
        micro_writer.write_table(tbl)
        micro_buf = {k: [] for k in m_schema.names}

    for eid, rep in iter_replays(PARTS, limit):
        rewards = rep.get("rewards") or []
        steps = rep.get("steps") or []
        if len(rewards) != 2 or len(steps) < 2:
            n_skip += 1
            continue
        try:
            r0, r1 = float(rewards[0]), float(rewards[1])
        except (TypeError, ValueError):
            n_skip += 1
            continue
        winner = 0 if r0 >= r1 else 1
        seats = (winner,) if not both else (0, 1)
        sig = world_sig(steps)
        split = split_of(eid)
        n_games += 1

        for seat in seats:
            ret = WM.score(rewards[seat], rewards[1 - seat])
            split_bal[split] += 1
            # --- per-step micro rows ---
            for cell in steps:
                if seat >= len(cell) or not cell[seat]:
                    continue
                obs = cell[seat].get("observation") or {}
                action = cell[seat].get("action") or {}
                if not obs:
                    continue
                feats = micro_features(obs, seat)
                n_ord, sell_u, buy_o, fverb = _action_summary(action)
                feats["n_market_orders"] = float(n_ord)
                feats["sell_units"] = float(sell_u)
                feats["buy_orders"] = float(buy_o)
                micro_buf["episode_id"].append(eid)
                micro_buf["seat"].append(seat)
                micro_buf["split"].append(split)
                micro_buf["world_sig"].append(sig)
                micro_buf["final_return"].append(ret)
                for k in MICRO_FEATURES:
                    micro_buf[k].append(float(feats[k]))
                micro_buf["farmer_verb"].append(fverb)
                micro_buf["action_json"].append(json.dumps(
                    action, separators=(",", ":")))
                n_micro += 1
            if len(micro_buf["episode_id"]) >= flush_rows:
                flush_micro()

            # --- per-day macro rows ---
            n_days = (len(steps) + TURNS_PER_DAY - 1) // TURNS_PER_DAY
            for d in range(n_days):
                day_cells = [cell[seat] for cell in
                             steps[d * TURNS_PER_DAY:(d + 1) * TURNS_PER_DAY]
                             if seat < len(cell) and cell[seat]]
                if not day_cells:
                    continue
                ma = MA.day_macro(day_cells)
                cls = ma.primary_class()
                class_bal[cls] += 1
                vec = ma.to_vector()
                macro_buf["episode_id"].append(eid)
                macro_buf["seat"].append(seat)
                macro_buf["day"].append(d)
                macro_buf["split"].append(split)
                macro_buf["world_sig"].append(sig)
                macro_buf["final_return"].append(ret)
                macro_buf["primary_class"].append(cls)
                for i, name in enumerate(MA.FEATURE_NAMES):
                    macro_buf[name].append(float(vec[i]))
                n_macro += 1

    flush_micro()
    micro_writer.close()

    macro_tbl = pa.table(
        {k: pa.array(v, type=macro_schema.field(k).type)
         for k, v in macro_buf.items()}, schema=macro_schema)
    pq.write_table(macro_tbl, macro_path, compression="zstd")

    stats = {
        "games": n_games, "skipped": n_skip,
        "micro_rows": n_micro, "macro_rows": n_macro,
        "both_seats": both, "limit": limit,
        "class_balance": dict(class_bal.most_common()),
        "split_balance": dict(split_bal),
        "micro_path": micro_path, "macro_path": macro_path,
    }
    if verbose:
        print(f"\n=== E0.1 BC corpus  ({n_games} decided games, "
              f"{n_skip} skipped) ===")
        print(f"micro rows: {n_micro:,}  -> {micro_path} "
              f"({os.path.getsize(micro_path) / 1e6:.1f} MB)")
        print(f"macro rows: {n_macro:,}  -> {macro_path} "
              f"({os.path.getsize(macro_path) / 1e6:.1f} MB)")
        print(f"split (rows-per-seat): {dict(split_bal)}")
        print("\nMACRO primary-class balance (daily plans):")
        tot = sum(class_bal.values()) or 1
        for c, n in class_bal.most_common():
            print(f"  {c:<13} {n:>7,}  {100 * n / tot:5.1f}%")
    with open(os.path.join(out_dir, "stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    return stats


# --------------------------------------------------------------------------- #
# destbreso corpus -- high-rated opponent action TAPES (429-free)
# --------------------------------------------------------------------------- #
# Each row of ``matchups_all.parquet`` is one seat's (the OPPONENT's) FULL
# 720-turn action tape ({farmer,hands,market} per turn) plus its recorded final
# bank, the opponent's rating and the seed. The tape carries NO observations,
# and the world is realised by BOTH seats' play (docs 2026-09-07), so a single
# seat's obs cannot be faithfully rebuilt from its tape alone. We therefore:
#   * MACRO -- distilled FULLY & faithfully from the action stream (day_macro
#     reads only actions), the RL training target.
#   * MICRO -- the action-derived fields (day/hour/step + market-order summary +
#     farmer verb + the raw action_json) are exact; the obs-STATE fields
#     (money, tiles, shed, ...) are left NaN because they are not reconstructable
#     from one seat's tape. ``world_sig`` is "" for the same reason.
# Rows carry a ``source`` tag, the opponent ``rating`` and the raw
# ``bank_return`` (recorded_bank_opponent) so provenance is clear and the corpus
# is a concat-superset of the top-100 schema.
DESTBRESO_EXTRA = (("source", pa.string()), ("rating", pa.float32()),
                   ("bank_return", pa.float32()))


def _destbreso_micro_schema():
    return pa.schema(list(_micro_schema()) + list(DESTBRESO_EXTRA))


def _destbreso_macro_schema():
    return pa.schema(list(_macro_schema()) + list(DESTBRESO_EXTRA))


def iter_destbreso(path, min_rating, limit):
    """Yield (episode_id, seat, rating, bank_opp, bank_you, tape) for tapes
    whose opponent_rating >= min_rating. Streams one batch at a time."""
    pf = pq.ParquetFile(path)
    cols = ["episode_id", "opponent_seat", "opponent_team", "opponent_rating",
            "recorded_bank_opponent", "recorded_bank_yours", "opponent_actions"]
    n = 0
    for b in pf.iter_batches(batch_size=256, columns=cols):
        eids = b.column("episode_id").to_pylist()
        seats = b.column("opponent_seat").to_pylist()
        teams = b.column("opponent_team").to_pylist()
        rats = b.column("opponent_rating").to_pylist()
        bopp = b.column("recorded_bank_opponent").to_pylist()
        byou = b.column("recorded_bank_yours").to_pylist()
        acts = b.column("opponent_actions").to_pylist()
        for eid, seat, team, rat, bo, by, aj in zip(
                eids, seats, teams, rats, bopp, byou, acts):
            if rat is None or float(rat) < min_rating:
                continue
            try:
                tape = json.loads(aj)
            except (ValueError, TypeError):
                continue
            if not isinstance(tape, list) or len(tape) < 2:
                continue
            yield (str(eid), int(seat or 0), team, float(rat),
                   float(bo or 0), float(by or 0), tape)
            n += 1
            if limit and n >= limit:
                return


def build_destbreso(min_rating, limit, out_dir, flush_rows, verbose=True):
    import kaggriculture.measure.win_metric as WM
    os.makedirs(out_dir, exist_ok=True)
    micro_path = os.path.join(out_dir, "micro_destbreso.parquet")
    macro_path = os.path.join(out_dir, "macro_destbreso.parquet")
    m_schema = _destbreso_micro_schema()
    macro_schema = _destbreso_macro_schema()
    nan = float("nan")

    micro_writer = pq.ParquetWriter(micro_path, m_schema, compression="zstd")
    macro_writer = pq.ParquetWriter(macro_path, macro_schema, compression="zstd")
    micro_buf = {k: [] for k in m_schema.names}
    macro_buf = {k: [] for k in macro_schema.names}
    n_micro = n_macro = n_tapes = 0
    class_bal = Counter()
    band = Counter()
    split_bal = Counter()

    def flush(buf, writer, schema):
        if not buf["episode_id"]:
            return
        tbl = pa.table({k: pa.array(v, type=schema.field(k).type)
                        for k, v in buf.items()}, schema=schema)
        writer.write_table(tbl)
        for k in buf:
            buf[k] = []

    for eid, seat, team, rat, bo, by, tape in iter_destbreso(
            DESTBRESO, min_rating, limit):
        ret = WM.score(bo, by)          # 1 / 0.5 / 0 -- concat with final_return
        split = split_of(eid)
        n_tapes += 1
        split_bal[split] += 1
        band["2300" if rat < 2500 else "2500" if rat < 2700 else "2700"] += 1

        # --- per-step micro rows (obs-state = NaN; actions exact) ---
        for t, act in enumerate(tape):
            if not isinstance(act, dict):
                continue
            n_ord, sell_u, buy_o, fverb = _action_summary(act)
            micro_buf["episode_id"].append(eid)
            micro_buf["seat"].append(seat)
            micro_buf["split"].append(split)
            micro_buf["world_sig"].append("")
            micro_buf["final_return"].append(ret)
            for k in MICRO_FEATURES:
                if k == "day":
                    micro_buf[k].append(float(t // TURNS_PER_DAY))
                elif k == "hour":
                    micro_buf[k].append(float(t % TURNS_PER_DAY))
                elif k == "step":
                    micro_buf[k].append(float(t))
                elif k == "n_market_orders":
                    micro_buf[k].append(float(n_ord))
                elif k == "sell_units":
                    micro_buf[k].append(float(sell_u))
                elif k == "buy_orders":
                    micro_buf[k].append(float(buy_o))
                else:                       # obs-state -- not reconstructable
                    micro_buf[k].append(nan)
            micro_buf["farmer_verb"].append(fverb)
            micro_buf["action_json"].append(
                json.dumps(act, separators=(",", ":")))
            micro_buf["source"].append("destbreso")
            micro_buf["rating"].append(rat)
            micro_buf["bank_return"].append(bo)
            n_micro += 1
        if len(micro_buf["episode_id"]) >= flush_rows:
            flush(micro_buf, micro_writer, m_schema)

        # --- per-day macro rows (fully faithful from the action stream) ---
        n_days = (len(tape) + TURNS_PER_DAY - 1) // TURNS_PER_DAY
        for d in range(n_days):
            day_cells = [{"action": act} for act in
                         tape[d * TURNS_PER_DAY:(d + 1) * TURNS_PER_DAY]
                         if isinstance(act, dict)]
            if not day_cells:
                continue
            ma = MA.day_macro(day_cells)
            cls = ma.primary_class()
            class_bal[cls] += 1
            vec = ma.to_vector()
            macro_buf["episode_id"].append(eid)
            macro_buf["seat"].append(seat)
            macro_buf["day"].append(d)
            macro_buf["split"].append(split)
            macro_buf["world_sig"].append("")
            macro_buf["final_return"].append(ret)
            macro_buf["primary_class"].append(cls)
            for i, name in enumerate(MA.FEATURE_NAMES):
                macro_buf[name].append(float(vec[i]))
            macro_buf["source"].append("destbreso")
            macro_buf["rating"].append(rat)
            macro_buf["bank_return"].append(bo)
            n_macro += 1
        if len(macro_buf["episode_id"]) >= flush_rows:
            flush(macro_buf, macro_writer, macro_schema)

    flush(micro_buf, micro_writer, m_schema)
    flush(macro_buf, macro_writer, macro_schema)
    micro_writer.close()
    macro_writer.close()

    stats = {
        "source": "destbreso", "min_rating": min_rating, "limit": limit,
        "tapes": n_tapes, "micro_rows": n_micro, "macro_rows": n_macro,
        "rating_band": dict(band), "split_balance": dict(split_bal),
        "class_balance": dict(class_bal.most_common()),
        "micro_path": micro_path, "macro_path": macro_path,
        "note": "obs-state micro fields are NaN (single-seat tape; world "
                "realised by play); world_sig='' for the same reason. MACRO is "
                "fully faithful. bank_return = raw recorded_bank_opponent; "
                "final_return = WM.score(win/draw/loss).",
    }
    if verbose:
        print(f"\n=== destbreso corpus  (rating>={min_rating}, "
              f"{n_tapes:,} tapes) ===")
        print(f"micro rows: {n_micro:,}  -> {micro_path} "
              f"({os.path.getsize(micro_path) / 1e6:.1f} MB)")
        print(f"macro rows: {n_macro:,}  -> {macro_path} "
              f"({os.path.getsize(macro_path) / 1e6:.1f} MB)")
        print(f"rating band (tapes): {dict(band)}")
        print(f"split (tapes): {dict(split_bal)}")
        print("\nMACRO primary-class balance:")
        tot = sum(class_bal.values()) or 1
        for c, n in class_bal.most_common():
            print(f"  {c:<13} {n:>7,}  {100 * n / tot:5.1f}%")
    with open(os.path.join(out_dir, "stats_destbreso.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("top100", "destbreso"),
                    default="top100",
                    help="top100 replay parquet, or the destbreso action tapes")
    ap.add_argument("--limit", type=int, default=800,
                    help="max replays/tapes to scan (0 = all)")
    ap.add_argument("--min-rating", type=float, default=2300.0,
                    help="destbreso: keep tapes with opponent_rating >= this")
    ap.add_argument("--both", action="store_true",
                    help="emit loser seats too (default: winner only)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--flush", type=int, default=50000,
                    help="micro rows buffered before a parquet flush")
    args = ap.parse_args()
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if args.source == "destbreso":
        build_destbreso(args.min_rating, args.limit, out_dir, args.flush)
    else:
        build(args.limit, args.both, out_dir, args.flush)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
