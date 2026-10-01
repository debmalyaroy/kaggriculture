"""Extract BC training sets from the GM replay shards for the bandit's two NNs.

One streaming pass over `D:/gm_dataset/replays_*.parquet` (RAM-lean, batched)
emits, for high-rated WINNING seats:

  SALE set   X_sale (F feats) -> Y_sale (9 sell-fractions of shed, per product)
  OPP  set   X_opp  (F feats) -> y_opp  (1: does the OBSERVED opponent SELL a
             premium item within the next OPP_HORIZON steps -- a dump signal
             learnable from PUBLIC state only)

Everything the features use is OBSERVABLE at play time (market inventory/price,
our shed, both PUBLIC banks, day/step) -- so a policy trained on it can run in
the live agent. Labels come from what the strong agent actually did.

Run: python -m kaggriculture.bandit.nn.extract --min-rating 2700 --max-rows 1500000
Writes .local/nn/{sale_X,sale_Y,opp_X,opp_y}.npy + meta.json.
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.data.gm_episodes as GM
try:
    import orjson
    def _loads(s): return orjson.loads(s)
except Exception:  # fallback to stdlib
    def _loads(s): return json.loads(s)

# fixed product order + engine base prices (market.rs PARAMS)
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BASE = {"WHEAT":25.,"CARROT":35.,"TOMATO":60.,"STRAWBERRY":120.,"MELON":250.,
        "EGG":50.,"MILK":160.,"WOOL":200.,"FERTILIZER":100.}
PREMIUM = {"MILK","WOOL","STRAWBERRY","MELON"}
I0 = 10000.0
OPP_HORIZON = 4
DUMP_UNITS = 5      # opp label: >= this many premium units within horizon = a DUMP
SALE_MIN = 5        # sale label: >= this many units of a product = a meaningful sell
SALE_FRAC = 0.5     # ...or >= this fraction of the shed
OUT = os.path.join(ROOT, ".local", "nn")

# feature layout (MUST match rust mbandit.rs nn_feats EXACTLY):
#   BASE (32): per product [inv/I0, price/base, shed/50] (27) + globals
#     [day/30, step/720, me/1e5, opp/1e5, gap/1e5] (5)
#   TEMPORAL (75): opp_money_delta 1/2/4/8/16 (5) + my_money_delta_4 (1) +
#     town phase [step%4, step%24] (2) + opp sell-cadence 8/16 (2) + capacity
#     [n_hands/8, shed_total/300] (2) + per-product inv_delta 4/8/16 (27) +
#     price_delta 4/8/16 (27) + scarcity_margin (9)
NFEAT = len(PRODUCTS) * 3 + 5 + (5 + 1 + 2 + 2 + 2 + len(PRODUCTS) * 3 * 2 + len(PRODUCTS))  # 32 + 75 = 107

def _sells(action):
    out = {}
    for o in (action or {}).get("market", []) or []:
        if o and o[0] == "SELL" and len(o) >= 3:
            try: out[o[1]] = out.get(o[1], 0) + int(o[2])
            except (ValueError, TypeError): pass
    return out

def _mkt(obs):
    """observable snapshot mirroring rust record(): per-product inv[9], price[9]
    from the GLOBAL market (missing inv -> 0, non-positive price -> base)."""
    mkt = obs.get("market") or {}; inv = mkt.get("inventory") or {}; pr = mkt.get("prices") or {}
    invv = [float(inv.get(p, 0.0) or 0.0) for p in PRODUCTS]
    prv = []
    for p in PRODUCTS:
        x = float(pr.get(p, 0.0) or 0.0)
        prv.append(x if x > 0.0 else BASE[p])
    return invv, prv

def _obs_at(steps, t, seat):
    if t < 0: t = 0
    return ((steps[t][seat] or {}).get("observation")) or {}

def _feat(steps, t, seat):
    """Full 65-feature vector for `seat` at step `t`, plus that seat's shed."""
    obs = _obs_at(steps, t, seat)
    mkt = obs.get("market") or {}; inv = mkt.get("inventory") or {}; pr = mkt.get("prices") or {}
    shed = (obs.get("private") or {}).get("shed") or {}
    farms = obs.get("farms") or [{}, {}]
    me = float((farms[seat] or {}).get("money") or 0.0)
    opp = float((farms[1 - seat] or {}).get("money") or 0.0)
    f = []
    for p in PRODUCTS:
        f.append(float(inv.get(p, I0)) / I0)
        f.append(float(pr.get(p, BASE[p])) / BASE[p])
        f.append(float(shed.get(p, 0)) / 50.0)
    f += [float(obs.get("day", 0)) / 30.0, float(obs.get("step", 0)) / 720.0,
          me / 1e5, opp / 1e5, (me - opp) / 1e5]
    # --- temporal features (ORDER MUST MATCH rust nn_feats; 75 feats) ---
    # rust back(k) clamps to step 0 when t<k; history is recorded every step.
    def money(tt, sel_seat):
        o = _obs_at(steps, tt, seat).get("farms") or [{}, {}]
        return float((o[sel_seat] or {}).get("money") or 0.0)
    # opponent money deltas over 1/2/4/8/16
    for k in (1, 2, 4, 8, 16):
        f.append((money(t, 1 - seat) - money(max(0, t - k), 1 - seat)) / 1e4)
    f.append((money(t, seat) - money(max(0, t - 4), seat)) / 1e4)
    step = int(obs.get("step", 0))
    f.append((step % 4) / 4.0)
    f.append((step % 24) / 24.0)
    # opponent SELL CADENCE over last 8/16 (n = t+1 history entries)
    def cadence(k):
        nn = t + 1
        if nn < 2:
            return 0.0
        lo = max(0, nn - k)
        c = 0; d = 0
        for i in range(lo + 1, nn):  # i in [lo+1, t]
            d += 1
            if money(i, 1 - seat) > money(i - 1, 1 - seat):
                c += 1
        return c / d if d else 0.0
    f.append(cadence(8))
    f.append(cadence(16))
    # our capacity: labour on hand + total shed
    f.append(len((farms[seat] or {}).get("hands") or []) / 8.0)
    f.append(sum(float(shed.get(p, 0)) for p in PRODUCTS) / 300.0)
    # market inventory deltas over 4/8/16 per product
    inv_now, pr_now = _mkt(obs)
    snaps = {k: _mkt(_obs_at(steps, max(0, t - k), seat)) for k in (4, 8, 16)}
    for k in (4, 8, 16):
        inv_k, _ = snaps[k]
        for i in range(len(PRODUCTS)):
            f.append((inv_now[i] - inv_k[i]) / 1e4)
    # price deltas over 4/8/16 per product
    for k in (4, 8, 16):
        _, pr_k = snaps[k]
        for i, p in enumerate(PRODUCTS):
            f.append((pr_now[i] - pr_k[i]) / BASE[p])
    # scarcity margin per product
    for i, p in enumerate(PRODUCTS):
        f.append((pr_now[i] - 0.9 * BASE[p]) / BASE[p])
    return f, shed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-rating", type=float, default=2700.0)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = the COMPLETE dataset")
    ap.add_argument("--stride", type=int, default=1, help="subsample steps")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--chunk-rows", type=int, default=250_000,
                    help="rows per on-disk chunk (bounds RAM; flush + clear when full)")
    ap.add_argument("--chunks-dir", default=os.path.join(OUT, "chunks"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import pyarrow.parquet as pq
    keep = GM.load_filter(a.min_rating)     # {eid: {seat:(rating,bank)}}
    print(f"{len(keep)} strong episodes (>= {a.min_rating}) in episodes.csv", flush=True)
    # STREAM TO DISK: fill a bounded buffer of CH rows, FLUSH it to a numbered
    # chunk file when full, then clear. RAM stays ~CH rows regardless of total
    # dataset size, so the COMPLETE multi-shard dataset trains out-of-core.
    import shutil
    shutil.rmtree(a.chunks_dir, ignore_errors=True); os.makedirs(a.chunks_dir, exist_ok=True)
    CH = a.chunk_rows
    sX = np.zeros((CH, NFEAT), np.float32); sY = np.zeros((CH, len(PRODUCTS)), np.float32)
    oX = np.zeros((CH, NFEAT), np.float32); oy = np.zeros(CH, np.float32)
    n = 0            # rows in the current buffer
    chunk = 0        # chunks flushed
    total = 0        # rows across all chunks
    games = 0

    def flush(count):
        nonlocal chunk
        if count < 1:
            return
        np.save(os.path.join(a.chunks_dir, f"sale_X_{chunk:03d}.npy"), sX[:count])
        np.save(os.path.join(a.chunks_dir, f"sale_Y_{chunk:03d}.npy"), sY[:count])
        np.save(os.path.join(a.chunks_dir, f"opp_X_{chunk:03d}.npy"), oX[:count])
        np.save(os.path.join(a.chunks_dir, f"opp_y_{chunk:03d}.npy"), oy[:count])
        print(f"  flushed chunk {chunk:03d} ({count} rows; total {total})", flush=True)
        chunk += 1
    shards = sorted(glob.glob(os.path.join(GM._resolve_data_dir(), "replays_*.parquet")))
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for b in pf.iter_batches(batch_size=4, columns=["episode_id", "replay_json"]):
            d = b.to_pydict()
            for eid, rj in zip(d["episode_id"], d["replay_json"]):
                seats = keep.get(str(eid))
                if not seats:
                    continue
                try:
                    r = _loads(rj); steps = r["steps"]   # orjson: ~10x faster parse
                except Exception:
                    continue
                # winning seat among the kept (strong) seats = higher final bank
                W = max(seats, key=lambda s: seats[s][1])
                games += 1
                T = len(steps)
                for t in range(0, T, a.stride):
                    cell = steps[t]
                    obs = (cell[W] or {}).get("observation") or {}
                    if not obs.get("market"):
                        continue
                    feat, shed = _feat(steps, t, W)
                    # SALE label -> CLASSIFICATION (so the head has an AUC): per
                    # product, 1 if the strong agent sold a MEANINGFUL amount this
                    # step (>= SALE_FRAC of the shed, or >= SALE_MIN units), else 0.
                    sells = _sells((cell[W] or {}).get("action"))
                    sX[n] = feat
                    sY[n] = [1.0 if (sells.get(p, 0) >= SALE_MIN or
                                     sells.get(p, 0) >= SALE_FRAC * max(float(shed.get(p, 0)), 1.0))
                             else 0.0 for p in PRODUCTS]
                    # OPP label: does the OTHER seat DUMP (>= DUMP_UNITS premium
                    # units) within OPP_HORIZON steps -- a real dump, not a 1-unit sale
                    dump = 0
                    for k in range(1, OPP_HORIZON + 1):
                        if t + k >= T: break
                        os_ = _sells((steps[t + k][1 - W] or {}).get("action"))
                        if sum(os_.get(p, 0) for p in PREMIUM) >= DUMP_UNITS:
                            dump = 1; break
                    # opp head observes OUR perspective (matches inference: the
                    # live agent feeds its own nn_feats) and predicts whether the
                    # OPPONENT (1-W) dumps. opp_money_delta (their bank jumping =
                    # them selling) is the key signal. Features == the sale set's.
                    oX[n] = feat; oy[n] = dump
                    n += 1; total += 1
                    if n >= CH:          # buffer full -> flush to disk, clear
                        flush(n); n = 0
                if a.max_games and games >= a.max_games:
                    break
            if a.max_games and games >= a.max_games:
                break
        print(f"  {os.path.basename(shard)}: games={games} total_rows={total}", flush=True)
        if a.max_games and games >= a.max_games:
            break
    flush(n)  # final partial buffer
    json.dump({"products": PRODUCTS, "base": BASE, "nfeat": NFEAT, "opp_horizon": OPP_HORIZON,
               "games": games, "rows": total, "chunks": chunk,
               "chunks_dir": a.chunks_dir},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(f"\nCOMPLETE: {total} rows in {chunk} chunks from {games} games -> {a.chunks_dir}",
          flush=True)

if __name__ == "__main__":
    main()
