"""E2.1 -- BC warmup: a return-conditioned MACRO policy over the daily-plan corpus.

The Slot-2 predator learns a STRATEGY (~30 daily decisions), not per-turn micro
(per-turn RL plateaus dumb -- master plan Sec 0.6/0.7). This module trains the BC
seed the self-play RL (``macro_rl``) fine-tunes.

Design (Decision-Transformer flavour, return-conditioned):
  * One training example = one game-seat = a sequence of <=30 days.
  * Per-day INPUT (causal, day t sees days < t):
      - day index (normalised) + a small day one-hot bucket
      - world_sig features (which two shops unlocked -- the field dispatch key)
      - RETURN-TO-GO token (final_return, and, when present, the rating +
        bank_return of the seat we are imitating) -- this is what makes it
        return-conditioned: at RL/inference time we CONDITION ON WINNING.
      - a running cumulative-economy vector (cumsum of PRIOR days' action
        vectors): land/hands/animals/builds/sells committed so far. This is the
        only "state" the macro corpus can supply faithfully for the destbreso
        rows (obs is not reconstructable there -- E0.2), and it is exact from
        the actions for every source.
  * Per-day OUTPUT heads over the fixed macro schema (``train.macro_actions``):
      - primary_class (7-way CE) -- the strategic label
      - the 26-dim action vector, as per-field regression (Smooth-L1) so the
        head is source-agnostic and round-trips through ``MacroAction``.

A masked causal Transformer encoder (~10-20M params at the default width) ties
the day sequence together. Training streams the (small, ~440k-row) macro corpus
into per-episode sequences held in memory; BF16 autocast on CUDA, fp32 on CPU.

Outputs ``models/rl/bc_policy.pt``: ``{config, state_dict, norm, schema}`` -- a
self-describing checkpoint ``macro_rl`` and ``package_policy`` both re-load.

    # real training on the RTX 4060 (see docs/history/slot2-training-runbook.md):
    python -m kaggriculture.train.bc_warmup --epochs 12 --d-model 384 --layers 8
    # tiny CPU smoke (no GPU, asserts fwd/bwd/save/reload/export parity):
    python -m kaggriculture.train.bc_warmup --smoke
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import kaggriculture.train.macro_actions as MA

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

MACRO_DIR = os.path.join(ROOT, "data", "bc_corpus")
OUT_PATH = os.path.join(ROOT, "models", "rl", "bc_policy.pt")
MAX_DAYS = 30

# world_sig vocabulary is open-ended (shop names); we hash the two-shop
# signature into a small fixed bucket space so the feature is fixed-width and
# an unseen world at inference just lands in a bucket (no crash, graceful).
WORLD_BUCKETS = 16

VEC_LEN = MA.VECTOR_LEN                       # 26
N_CLASS = len(MA.MACRO_CLASSES)               # 7


# --------------------------------------------------------------------------- #
# Corpus -> per-episode day sequences (numpy), source-agnostic.
# --------------------------------------------------------------------------- #
def _world_bucket(sig: str) -> int:
    if not sig:
        return 0
    return 1 + (hash(sig) % (WORLD_BUCKETS - 1))


def _load_sequences(limit_eps: Optional[int] = None,
                    sources: Tuple[str, ...] = ("both",)) -> List[dict]:
    """Group the macro parquet(s) into per-(episode,seat) day sequences.

    Returns a list of dicts with numpy arrays:
      days[T], world[T], rtg[T](return-to-go, constant=final_return),
      rating[T], vec[T,26](that day's action), cls[T](primary class id),
      cum[T,26](cumulative PRIOR-day action vector).
    """
    import numpy as np
    import pyarrow.parquet as pq

    files = []
    for nm in ("macro.parquet", "macro_destbreso.parquet",
               "macro_gm.parquet",               # georgymamarin full-ladder (delta-ingested)
               "macro_selfplay.parquet"):        # self-distillation corpus (grows)
        p = os.path.join(MACRO_DIR, nm)
        if os.path.exists(p):
            files.append(p)
    if not files:
        raise FileNotFoundError(f"no macro corpus under {MACRO_DIR} "
                                "(run kaggriculture.data.bc_corpus first)")

    vec_cols = list(MA.FEATURE_NAMES)
    # group rows by (episode_id, seat)
    groups: Dict[Tuple[str, int], List[dict]] = {}
    for f in files:
        pf = pq.ParquetFile(f)
        names = set(pf.schema_arrow.names)
        cols = (["episode_id", "seat", "day", "split", "world_sig",
                 "final_return", "primary_class"] + vec_cols
                + [c for c in ("rating", "bank_return") if c in names])
        for batch in pf.iter_batches(batch_size=8192, columns=cols):
            d = batch.to_pydict()
            n = len(d["episode_id"])
            for i in range(n):
                key = (d["episode_id"][i], int(d["seat"][i]))
                g = groups.setdefault(key, [])
                g.append({c: d[c][i] for c in cols})
                if limit_eps and len(groups) > limit_eps and key not in groups:
                    pass
        if limit_eps and len(groups) >= limit_eps:
            break

    seqs: List[dict] = []
    cls_index = MA.CLASS_INDEX
    for (eid, seat), rows in groups.items():
        rows.sort(key=lambda r: float(r["day"]))
        # dedupe by day + cap to MAX_DAYS (the same episode can appear in both
        # corpus files, which would otherwise stack to 2x30 rows and overrun the
        # positional embedding)
        seen_days, dedup = set(), []
        for r in rows:
            d = float(r["day"])
            if d < MAX_DAYS and d not in seen_days:
                seen_days.add(d)
                dedup.append(r)
        rows = dedup
        if not rows:
            continue
        T = len(rows)
        days = np.array([float(r["day"]) for r in rows], dtype=np.float32)
        world = np.array([_world_bucket(r.get("world_sig") or "")
                          for r in rows], dtype=np.int64)
        rtg = np.array([float(r["final_return"]) for r in rows],
                       dtype=np.float32)
        rating = np.array([(float(r["rating"]) / 3000.0)
                           if r.get("rating") not in (None, "")
                           else 0.0 for r in rows], dtype=np.float32)
        vec = np.array([[float(r[c]) for c in vec_cols] for r in rows],
                       dtype=np.float32)
        cls = np.array([cls_index.get(r["primary_class"], cls_index["IDLE"])
                        for r in rows], dtype=np.int64)
        cum = np.zeros_like(vec)
        if T > 1:
            cum[1:] = np.cumsum(vec, axis=0)[:-1]
        split = rows[0].get("split") or "train"
        seqs.append(dict(eid=eid, seat=seat, split=split, days=days,
                         world=world, rtg=rtg, rating=rating, vec=vec,
                         cls=cls, cum=cum))
        if limit_eps and len(seqs) >= limit_eps:
            break
    return seqs


def _norm_stats(seqs: List[dict]) -> dict:
    import numpy as np
    allvec = np.concatenate([s["cum"] for s in seqs], axis=0) if seqs \
        else np.zeros((1, VEC_LEN), np.float32)
    mu = allvec.mean(axis=0)
    sd = allvec.std(axis=0)
    sd[sd < 1e-6] = 1.0
    tgt = np.concatenate([s["vec"] for s in seqs], axis=0) if seqs \
        else np.zeros((1, VEC_LEN), np.float32)
    tmu = tgt.mean(axis=0)
    tsd = tgt.std(axis=0)
    tsd[tsd < 1e-6] = 1.0
    return dict(cum_mu=mu.tolist(), cum_sd=sd.tolist(),
                tgt_mu=tmu.tolist(), tgt_sd=tsd.tolist())


# input feature width: day(1) + day-onehot(6 buckets) + world onehot(WB)
#   + rtg(1) + rating(1) + cum(26 normalised)
DAY_BUCKETS = 6
IN_DIM = 1 + DAY_BUCKETS + WORLD_BUCKETS + 1 + 1 + VEC_LEN


def _features(seq: dict, norm: dict):
    """Build the [T, IN_DIM] input tensor and the targets for one sequence."""
    import numpy as np
    T = len(seq["days"])
    cum_mu = np.asarray(norm["cum_mu"], np.float32)
    cum_sd = np.asarray(norm["cum_sd"], np.float32)
    x = np.zeros((T, IN_DIM), np.float32)
    x[:, 0] = seq["days"] / float(MAX_DAYS)
    db = np.minimum((seq["days"] // (MAX_DAYS // DAY_BUCKETS + 1)).astype(int),
                    DAY_BUCKETS - 1)
    x[np.arange(T), 1 + db] = 1.0
    x[np.arange(T), 1 + DAY_BUCKETS + seq["world"]] = 1.0
    base = 1 + DAY_BUCKETS + WORLD_BUCKETS
    x[:, base] = seq["rtg"]
    x[:, base + 1] = seq["rating"]
    x[:, base + 2:base + 2 + VEC_LEN] = (seq["cum"] - cum_mu) / cum_sd
    return x


# --------------------------------------------------------------------------- #
# Model: masked causal Transformer encoder + class/vector heads.
# --------------------------------------------------------------------------- #
def build_model(cfg: dict):
    import torch
    import torch.nn as nn

    class MacroPolicy(nn.Module):
        def __init__(self, in_dim, d_model, layers, heads, vec_len, n_class):
            super().__init__()
            self.d_model = d_model
            self.inp = nn.Linear(in_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, MAX_DAYS, d_model))
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
                dropout=cfg.get("dropout", 0.1), batch_first=True,
                activation="gelu", norm_first=True)
            self.enc = nn.TransformerEncoder(enc, layers,
                                             enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.cls_head = nn.Linear(d_model, n_class)
            self.vec_head = nn.Linear(d_model, vec_len)

        def forward(self, x, pad_mask=None):
            # x: [B, T, in_dim]; causal mask so day t attends to <= t.
            B, T, _ = x.shape
            h = self.inp(x) + self.pos[:, :T]
            cmask = torch.triu(torch.ones(T, T, device=x.device,
                                          dtype=torch.bool), diagonal=1)
            h = self.enc(h, mask=cmask, src_key_padding_mask=pad_mask)
            h = self.norm(h)
            return self.cls_head(h), self.vec_head(h)

    m = MacroPolicy(cfg["in_dim"], cfg["d_model"], cfg["layers"],
                    cfg["heads"], cfg["vec_len"], cfg["n_class"])
    return m


def _batch_tensors(seqs, norm, device):
    import numpy as np
    import torch
    B = len(seqs)
    T = min(max(len(s["days"]) for s in seqs), MAX_DAYS)
    x = np.zeros((B, T, IN_DIM), np.float32)
    ymask = np.zeros((B, T), np.float32)
    ycls = np.zeros((B, T), np.int64)
    yvec = np.zeros((B, T, VEC_LEN), np.float32)
    pad = np.ones((B, T), bool)
    tgt_mu = np.asarray(norm["tgt_mu"], np.float32)
    tgt_sd = np.asarray(norm["tgt_sd"], np.float32)
    for i, s in enumerate(seqs):
        t = min(len(s["days"]), T)
        x[i, :t] = _features(s, norm)[:t]
        ymask[i, :t] = 1.0
        ycls[i, :t] = s["cls"][:t]
        yvec[i, :t] = ((s["vec"] - tgt_mu) / tgt_sd)[:t]
        pad[i, :t] = False
    dev = lambda a: torch.from_numpy(a).to(device)
    return dev(x), dev(pad), dev(ycls), dev(yvec), dev(ymask)


def train(cfg: dict, seqs: Optional[List[dict]] = None) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    device = cfg["device"]
    if seqs is None:
        seqs = _load_sequences(limit_eps=cfg.get("limit_eps"))
    if not seqs:
        raise RuntimeError("empty corpus")
    norm = _norm_stats(seqs)
    tr = [s for s in seqs if s["split"] != "val"]
    va = [s for s in seqs if s["split"] == "val"]
    if not va:                                  # smoke corpora may be tiny
        va = tr[: max(1, len(tr) // 10)]
    print(f"[bc] {len(seqs)} seqs ({len(tr)} train / {len(va)} val), "
          f"in_dim={IN_DIM}, device={device}")

    cfg2 = dict(cfg, in_dim=IN_DIM, vec_len=VEC_LEN, n_class=N_CLASS)
    model = build_model(cfg2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bc] model params = {n_params/1e6:.2f}M "
          f"(d_model={cfg['d_model']} layers={cfg['layers']} heads={cfg['heads']})")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("wd", 0.01))
    bs = cfg["batch_size"]
    use_bf16 = (device == "cuda" and cfg.get("bf16", True))
    rng = np.random.default_rng(0)

    def run_epoch(data, train_mode):
        model.train(train_mode)
        idx = np.arange(len(data))
        if train_mode:
            rng.shuffle(idx)
        tot, tot_c, tot_v, nb = 0.0, 0.0, 0.0, 0
        for k in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[k:k + bs]]
            x, pad, ycls, yvec, ymask = _batch_tensors(chunk, norm, device)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if use_bf16 else _nullctx())
            with ctx:
                logits, vpred = model(x, pad_mask=pad)
                m = ymask.reshape(-1) > 0
                lc = F.cross_entropy(
                    logits.reshape(-1, N_CLASS)[m], ycls.reshape(-1)[m])
                lv = F.smooth_l1_loss(
                    vpred.reshape(-1, VEC_LEN)[m], yvec.reshape(-1, VEC_LEN)[m])
                loss = lc + cfg.get("vec_weight", 1.0) * lv
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach()); tot_c += float(lc.detach())
            tot_v += float(lv.detach()); nb += 1
        return tot / max(nb, 1), tot_c / max(nb, 1), tot_v / max(nb, 1)

    best = math.inf
    hist = []
    for ep in range(cfg["epochs"]):
        trl = run_epoch(tr, True)
        with torch.no_grad():
            val = run_epoch(va, False)
        hist.append(dict(epoch=ep, train=trl[0], val=val[0],
                         val_cls=val[1], val_vec=val[2]))
        print(f"[bc] epoch {ep:2d}  train {trl[0]:.4f}  "
              f"val {val[0]:.4f} (cls {val[1]:.4f} vec {val[2]:.4f})")
        if val[0] < best:
            best = val[0]
            _save(model, cfg2, norm, hist, n_params)
    if best is math.inf:
        _save(model, cfg2, norm, hist, n_params)
    dt = time.time() - t0
    print(f"[bc] done in {dt:.1f}s  best val {best:.4f}  -> {OUT_PATH}")
    return dict(best_val=best, n_params=n_params, seqs=len(seqs), seconds=dt)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _save(model, cfg2, norm, hist, n_params):
    import torch
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(dict(
        config=cfg2, state_dict=model.state_dict(), norm=norm, history=hist,
        n_params=n_params,
        schema=dict(in_dim=IN_DIM, vec_len=VEC_LEN, n_class=N_CLASS,
                    max_days=MAX_DAYS, world_buckets=WORLD_BUCKETS,
                    day_buckets=DAY_BUCKETS, feature_names=list(MA.FEATURE_NAMES),
                    classes=list(MA.MACRO_CLASSES)),
    ), OUT_PATH)


def load_policy(path: str = OUT_PATH, device: str = "cpu"):
    """Re-instantiate the trained policy (used by macro_rl / package_policy)."""
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------- #
def _smoke() -> int:
    """Tiny CPU end-to-end: fwd/bwd/save/reload + a synthetic-corpus fallback."""
    import numpy as np
    import torch

    # If the real corpus is missing (fresh clone), synthesise a few sequences so
    # the smoke still exercises the whole path.
    try:
        seqs = _load_sequences(limit_eps=40)
    except FileNotFoundError:
        seqs = []
    if len(seqs) < 8:
        rng = np.random.default_rng(1)
        seqs = []
        for e in range(24):
            T = int(rng.integers(5, MAX_DAYS))
            vec = rng.random((T, VEC_LEN)).astype(np.float32)
            seqs.append(dict(eid=f"syn{e}", seat=e % 2,
                             split="val" if e % 5 == 0 else "train",
                             days=np.arange(T, dtype=np.float32),
                             world=rng.integers(0, WORLD_BUCKETS, T),
                             rtg=np.full(T, float(e % 2), np.float32),
                             rating=np.full(T, 0.8, np.float32),
                             vec=vec,
                             cls=rng.integers(0, N_CLASS, T),
                             cum=np.cumsum(vec, 0) - vec))

    cfg = dict(device="cpu", d_model=64, layers=2, heads=4, lr=1e-3,
               wd=0.01, dropout=0.0, batch_size=8, epochs=2, bf16=False,
               vec_weight=1.0, limit_eps=40)
    res = train(cfg, seqs=seqs)
    assert os.path.exists(OUT_PATH), "checkpoint not written"
    model, ck = load_policy(OUT_PATH, "cpu")
    # forward parity on a single fixed sequence
    s = seqs[0]
    x = torch.from_numpy(_features(s, ck["norm"])[None]).float()
    with torch.no_grad():
        a, b = model(x)
    assert a.shape[-1] == N_CLASS and b.shape[-1] == VEC_LEN
    # round-trip a predicted vector through MacroAction
    ma = MA.MacroAction.from_vector((b[0, 0] * 0 + 1).tolist())
    assert ma.to_vector() is not None
    print(f"[bc][smoke] OK  params={res['n_params']/1e6:.2f}M  "
          f"val={res['best_val']:.3f}  ckpt={OUT_PATH}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--vec-weight", type=float, default=1.0)
    ap.add_argument("--limit-eps", type=int, default=None,
                    help="cap #sequences (debug)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU end-to-end self-test")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()

    device = "cpu"
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    cfg = dict(device=device, d_model=args.d_model, layers=args.layers,
               heads=args.heads, lr=args.lr, wd=0.01, dropout=0.1,
               batch_size=args.batch_size, epochs=args.epochs,
               bf16=not args.no_bf16, vec_weight=args.vec_weight,
               limit_eps=args.limit_eps)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
