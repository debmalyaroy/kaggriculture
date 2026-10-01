"""Sale signatures of EVERY player-game in the corpus (GM + official daily, engine 1.32.7) for opponent clustering
(python/opp_cluster.py --sig). Operator 28 Sep: "we have so many games in GM's dataset -- cluster and learn the
different sale policies".

    python python/opp_corpus_sig.py [--slim data/slim/s1] [--workers 5] [--out data/opp/sig_v1.npz]

Reads the slim corpus (crates/corpus/src/slim.rs: steps[t].a = [action seat0, action seat1] that led to state t)
one parquet row group at a time (memory-light). Per (episode, seat):
  ev[item, day, hour]  1 if the seat placed a SELL of item at that step
  units[item, day]     units ordered (capped 200 per order; "sell all" orders count 200)
  lots                 median order size and share of "sell all" (>= 999) orders per item
  front[item]          share of the OTHER seat's sale runs of item that this seat pre-empted by 1-2 steps
                       (this seat sold item in the 1-2 steps before the other seat's run started)
plus team, submission id, rating after the game, bank, rival bank. Written as one compressed npz.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = ["WHEAT", "STRAWBERRY", "WOOL", "EGG", "MILK", "MELON", "CARROT", "TOMATO"]
NI = len(ITEMS)
import re  # noqa: E402
ACT_RE = re.compile(r'\{"a":\[(.*?)\],"m":')


def sells(a):
    out = []
    if not isinstance(a, dict):
        return out
    for o in a.get("market") or []:
        if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in ITEMS:
            try:
                q = int(o[2])
            except Exception:  # noqa: BLE001
                continue
            if q > 0:
                out.append((ITEMS.index(o[1]), q))
    return out


def one_file(path):
    import gc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = ["episode_id", "team_name_0", "team_name_1", "submission_id_0", "submission_id_1", "rating_after_0", "rating_after_1", "bank_0", "bank_1", "slim"]
    rows = []
    for g in range(pf.metadata.num_row_groups):
        tb = pf.read_row_group(g, columns=cols).to_pylist()
        for r in tb:
            # fast path: only the per-step action pairs ('{"a":[a0,a1],"m":' in the slim text) are parsed, and only
            # when they contain a SELL; the market / farm / private blobs are never decoded
            txt = r["slim"] or ""
            acts = ACT_RE.findall(txt)
            if len(acts) < 700:
                continue
            steps = acts
            # lineage keys: hash of the seat's exact action stream over days 0-2 and days 0-5 (tape bots replay it
            # move for move; RL / adaptive players do not repeat)
            import hashlib
            pf = [[hashlib.blake2b(digest_size=8), hashlib.blake2b(digest_size=8)] for _ in range(2)]
            for t in range(1, min(145, len(steps))):
                try:
                    pa = json.loads("[" + steps[t] + "]")
                except Exception:  # noqa: BLE001
                    pa = [None, None]
                for seat in (0, 1):
                    b = json.dumps(pa[seat] if len(pa) > seat else None, sort_keys=True, separators=(",", ":")).encode()
                    if t <= 72:
                        pf[seat][0].update(b)
                    pf[seat][1].update(b)
            ev = np.zeros((2, NI, 30, 24), np.uint8)
            units = np.zeros((2, NI, 30), np.float32)
            lots = [[[] for _ in range(NI)] for _ in range(2)]
            sale_steps = [[[] for _ in range(NI)] for _ in range(2)]
            for t in range(1, len(steps)):
                if '"SELL"' not in steps[t]:
                    continue
                try:
                    a = json.loads("[" + steps[t] + "]")
                except Exception:  # noqa: BLE001
                    continue
                st = t - 1
                d, h = min(st // 24, 29), st % 24
                for seat in (0, 1):
                    for i, q in sells(a[seat] if len(a) > seat else None):
                        ev[seat, i, d, h] = 1
                        units[seat, i, d] += min(q, 200)
                        lots[seat][i].append(q)
                        sale_steps[seat][i].append(st)
            for seat in (0, 1):
                if units[seat].sum() == 0:
                    continue
                front = np.zeros(NI, np.float32)
                for i in range(NI):
                    other = sorted(set(sale_steps[1 - seat][i]))
                    runs = [x for k, x in enumerate(other) if k == 0 or other[k - 1] < x - 1]
                    mine = set(sale_steps[seat][i])
                    if runs:
                        front[i] = sum(1 for x in runs if (x - 1) in mine or (x - 2) in mine) / len(runs)
                lot_med = np.array([np.median(l) if l else 0 for l in lots[seat]], np.float32)
                lot_all = np.array([np.mean(np.array(l) >= 999) if l else 0 for l in lots[seat]], np.float32)
                rows.append(dict(pfx3=int.from_bytes(pf[seat][0].digest(), "little", signed=True), pfx6=int.from_bytes(pf[seat][1].digest(), "little", signed=True),
                                 eid=int(r["episode_id"]), seat=seat, team=str(r[f"team_name_{seat}"]), opp_team=str(r[f"team_name_{1 - seat}"]),
                                 sub=int(r[f"submission_id_{seat}"] or 0), rating=float(r[f"rating_after_{seat}"] or 0),
                                 bank=float(r[f"bank_{seat}"] or 0), obank=float(r[f"bank_{1 - seat}"] or 0),
                                 ev=np.packbits(ev[seat].reshape(-1)), units=units[seat], lot_med=lot_med, lot_all=lot_all, front=front))
        del tb
        gc.collect()
    return os.path.basename(os.path.dirname(path)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", default=os.path.join(RL, "data", "slim", "s1"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(RL, "data", "opp", "sig_v2.npz"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.slim, "source=gm", "date=*", "*.parquet")) + glob.glob(os.path.join(a.slim, "source=official", "date=*", "*.parquet")))
    print(f"[sig] {len(files)} slim parquet files", flush=True)
    allrows = []
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_file, f) for f in files]
        for fu in as_completed(futs):
            d, rows = fu.result()
            allrows += rows
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"[sig] {done}/{len(files)} files, {len(allrows)} player-games", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out,
                        pfx3=np.array([r["pfx3"] for r in allrows], np.int64), pfx6=np.array([r["pfx6"] for r in allrows], np.int64),
                        eid=np.array([r["eid"] for r in allrows]), seat=np.array([r["seat"] for r in allrows], np.int8),
                        team=np.array([r["team"] for r in allrows]), opp_team=np.array([r["opp_team"] for r in allrows]),
                        sub=np.array([r["sub"] for r in allrows]), rating=np.array([r["rating"] for r in allrows], np.float32),
                        bank=np.array([r["bank"] for r in allrows], np.float32), obank=np.array([r["obank"] for r in allrows], np.float32),
                        ev=np.stack([r["ev"] for r in allrows]), units=np.stack([r["units"] for r in allrows]),
                        lot_med=np.stack([r["lot_med"] for r in allrows]), lot_all=np.stack([r["lot_all"] for r in allrows]),
                        front=np.stack([r["front"] for r in allrows]), items=np.array(ITEMS))
    print(f"[sig] {len(allrows)} player-games -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
