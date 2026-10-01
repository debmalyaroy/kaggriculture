"""Self-distillation: generate gameplays (tape + world generator) → keep the best
→ distil into BC macro rows the policy re-learns from (AlphaZero-style).

Self-play RL already generates games, and the league reuses them as opponents. This
closes the OTHER half: the current policy's BEST self-generated games (across diverse
worlds) become NEW behaviour-cloning targets, so the next BC learns from its own best
play — the corpus grows with self-made, high-quality gameplays, no external data.

Flow per call:
  1. sample K plans (with exploration) across worlds (WorldSampler);
  2. compile → tapes (BatchRoller) and play them (vs PASS + a few real opponents)
     across seeds → banks;
  3. keep the top `keep_frac` by bank (the winners);
  4. distil each kept plan's daily MacroActions into macro rows (return-conditioned,
     final_return=1) and APPEND to data/bc_corpus/macro_selfplay.parquet.
`bc_warmup._load_sequences` loads this file automatically, so the next `--fresh-bc`
trains on top-100 + destbreso + the self-play corpus.

    python -m kaggriculture.train.selfplay_corpus --games 200 --keep 0.3
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import os
import random
import sys

import kaggriculture.train.macro_actions as MA
import kaggriculture.train.bc_warmup as BC

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OUT = os.path.join(ROOT, "data", "bc_corpus", "macro_selfplay.parquet")
FEATURES = list(MA.FEATURE_NAMES)


def _rows_from_plan(plan, eid, world_bucket, bank):
    """Distil a winning plan into return-conditioned macro rows (macro.parquet schema)."""
    rows = []
    for day, ma in enumerate(plan):
        if day >= BC.MAX_DAYS:
            break
        vec = ma.to_vector()
        row = {"episode_id": str(eid), "seat": 0, "day": float(day),
               "split": "train", "world_sig": f"selfplay_{world_bucket}",
               "final_return": 1.0, "primary_class": ma.primary_class(),
               "rating": 2600.0, "bank_return": float(bank), "source": "selfplay"}
        row.update({FEATURES[i]: float(vec[i]) for i in range(len(FEATURES))})
        rows.append(row)
    return rows


def generate(model, log_std, norm, device, br, worlds, rng,
             n_games=200, keep_frac=0.3, opp_pool=None, gpp=6):
    """Generate → score → keep winners → append distilled rows. Returns #rows added."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kaggriculture.train.macro_rl import sample_plan

    plans = []
    for _ in range(n_games):
        wb = worlds.sample_bucket(rng)
        plan, _, _ = sample_plan(model, log_std, norm, wb, rtg=1.0, rating=0.85,
                                 device=device, rng_t=None, greedy=False,
                                 n_days=BC.MAX_DAYS)
        plans.append((plan, wb))
    tapes = br.parallel_compile([p for p, _ in plans], n_workers=3)
    games = []
    for _ in plans:
        g = []
        for _k in range(gpp):
            e = rng.choice(opp_pool) if opp_pool else {"kind": "pass"}
            g.append((e, rng.randint(1, 2**31 - 1)))
        games.append(g)
    res = br.rollout_many(tapes, games, our_seat=0)
    # EVAL-BASED keep: score by WIN RATE vs real opponents (bank as tiebreak), and
    # only distil games that actually WIN — the same signal the gate measures, so we
    # never train the policy on its own losing games.
    scored = []
    for i, (plan, wb) in enumerate(plans):
        wins = [r["win"] for r in res[i]] or [0]
        banks = [r["our_bank"] for r in res[i]] or [0]
        scored.append((plan, wb, float(np.mean(wins)), float(np.mean(banks))))
    scored.sort(key=lambda x: (-x[2], -x[3]))
    n_keep = max(1, int(len(scored) * keep_frac))
    keep = [(p, wb, bank) for (p, wb, win, bank) in scored[:n_keep] if win > 0.5]
    thresh = min((s[2] for s in scored[:n_keep]), default=0)
    rows = []
    for j, (plan, wb, bank) in enumerate(keep):
        rows.extend(_rows_from_plan(plan, f"sp_{os.getpid()}_{j}", wb, bank))
    if not keep:
        print("[selfplay] no self-play game WON vs real opponents this pass — "
              "nothing distilled (policy not yet competitive).")
    if not rows:
        return 0
    # append (merge with existing self-play parquet)
    table = pa.Table.from_pylist(rows)
    if os.path.exists(OUT):
        try:
            table = pa.concat_tables([pq.read_table(OUT), table], promote_options="default")
        except Exception:
            pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pq.write_table(table, OUT)
    print(f"[selfplay] generated {n_games}, kept {len(keep)} (bank≥{thresh:.0f}), "
          f"+{len(rows)} rows → {os.path.relpath(OUT, ROOT)} "
          f"(total {table.num_rows})")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--keep", type=float, default=0.3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    import torch
    from kaggriculture.train.macro_env import BatchRoller, OpponentPool
    from kaggriculture.train.worlds import WorldSampler
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    ckpt = os.path.join(ROOT, "models", "rl", "rl_policy.pt")
    ckpt = ckpt if os.path.exists(ckpt) else BC.OUT_PATH
    model, ck = BC.load_policy(ckpt, device)
    ls = ck.get("log_std")
    ls = ls.to(device) if hasattr(ls, "to") else torch.zeros(MA.VECTOR_LEN, device=device)
    pool = OpponentPool()
    br = BatchRoller(pool=pool)
    opp = [e for e in pool.entries if e.get("kind") == "destbreso_tape"][:20]
    n = generate(model, ls, ck["norm"], device, br, WorldSampler(),
                 random.Random(0), n_games=args.games, keep_frac=args.keep,
                 opp_pool=opp)
    print(f"[selfplay] done (+{n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
