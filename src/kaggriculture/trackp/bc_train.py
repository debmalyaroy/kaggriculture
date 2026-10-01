"""Trackp BC trainer -- token-transformer behavior-cloning warm-start.

Version of record (tracked). Runs as a PRIVATE Kaggle kernel (GPU) over the
uploaded corpus dataset, or locally for a smoke (--smoke). Reads the packed
int32 token corpus (shards + vocab.json + norm.json, TOKEN_LAYOUT_VERSION=1),
trains a small order-invariant set-transformer with factorized action heads,
return-conditioned on rtg, split BY EPISODE (the split column). Saves
policy_bc.pt as the kernel/output artifact.

The autopilot (pipeline.trackp_autopilot, stage 7) stages a COPY of this file
plus configs/trackp_bc_kernel-metadata.json into a scratch dir and runs
`kaggle kernels push`. This file is standalone on Kaggle (no package import),
so keep it dependency-light (torch + numpy + pyarrow, all in the Kaggle image).

    python -m kaggriculture.trackp.bc_train --smoke        # local sanity
    (on Kaggle the script runs with defaults: 2 epochs, full corpus)

Row layout (see docs/history/trackp-token-spec.md):
  tokens : int32 [n_tokens, 14]  (type,x,y,ca,cb + 9 value fields)
  action : int32 [56] = [f_verb,f_arg] + 12*[verb,arg] + 10*[verb,arg,qty]
"""
import argparse, glob, json, os, time
import numpy as np
import torch                                    # trainer -> torch always required
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

TOK_W = 14
MAX_TOKENS = 128           # real max obs tokens ~115; model is order-invariant (no pos
                           # emb) so masked padding is identical -> 160->128 is free speed
MAX_HANDS = 12
MAX_MARKET = 10
ACT_W = 2 + MAX_HANDS * 2 + MAX_MARKET * 3     # 56
N_MOVER = 18
N_MARKET = 7
N_ARG = 10                     # 0 + crop/animal/product (verb-aware, max 9)
N_FARMER_ARG = N_ARG           # layout v2: farmer can PICKUP -> product id up to 9
QTY_CAP = 31


def _find_corpus():
    """Locate a dir with shard_*.parquet, independent of where this file lives:
    Kaggle mount, cwd/data, or any ancestor of this file / cwd."""
    bases = ["/kaggle/input", os.path.join(os.getcwd(), "data"), os.getcwd()]
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(1, 6):
        bases.append(os.path.join(here, *([".."] * up), "data"))
    seen = set()
    for base in bases:
        base = os.path.abspath(base)
        if base in seen or not os.path.isdir(base):
            continue
        seen.add(base)
        for root, _dirs, files in os.walk(base):
            if any(f.startswith("shard_") and f.endswith(".parquet") for f in files):
                return root
    raise SystemExit("no corpus dir with shard_*.parquet found")


def shard_list(corpus, limit_shards=0):
    shards = sorted(glob.glob(os.path.join(corpus, "shard_*.parquet")))
    return shards[:limit_shards] if limit_shards else shards


def iter_rows(corpus, split_keep, shuffle_buf=8192, seed=0, limit_shards=0,
              start_shard=0, worker_id=0, num_workers=1):
    """Yield (tokens, action, rtg, shard_idx). Resumable: ``start_shard`` skips
    whole shards (cheap -- no decode) so a resumed epoch continues near where it
    stopped. shard order is deterministic (sorted), so resume is reproducible.
    With num_workers>1, worker ``worker_id`` handles shards where
    ``shard_idx % num_workers == worker_id`` (disjoint parallel decode)."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed * 131 + worker_id)
    shards = shard_list(corpus, limit_shards)
    buf = []
    for si, sp in enumerate(shards):
        if si < start_shard or (si % num_workers) != worker_id:
            continue
        try:
            t = pq.read_table(sp, columns=["split", "rtg", "n_tokens", "tokens", "action"])
        except Exception:
            continue
        d = t.to_pydict()
        for i in range(t.num_rows):
            if int(d["split"][i]) != split_keep:
                continue
            nt = int(d["n_tokens"][i])
            tok = np.frombuffer(d["tokens"][i], np.int32).reshape(nt, TOK_W)
            act = np.frombuffer(d["action"][i], np.int32)
            buf.append((tok, act, float(d["rtg"][i]), si))
            if len(buf) >= shuffle_buf:
                j = rng.integers(len(buf))
                buf[j], buf[-1] = buf[-1], buf[j]
                yield buf.pop()
    rng.shuffle(buf)
    for r in buf:
        yield r


def collate(rows, norm_mean, norm_std):
    import torch
    b = len(rows)
    toks = np.zeros((b, MAX_TOKENS, TOK_W), np.float32)
    mask = np.zeros((b, MAX_TOKENS), np.bool_)
    acts = np.zeros((b, ACT_W), np.int64)
    rtgs = np.zeros((b, 1), np.float32)
    for k, r in enumerate(rows):
        tok, act, rtg = r[0], r[1], r[2]
        n = min(tok.shape[0], MAX_TOKENS)
        toks[k, :n] = (tok[:n].astype(np.float32) - norm_mean) / norm_std
        mask[k, :n] = True
        acts[k] = act
        rtgs[k, 0] = rtg
    return (torch.from_numpy(toks), torch.from_numpy(mask),
            torch.from_numpy(acts), torch.from_numpy(rtgs))


def _collate_with_si(rows, norm_mean, norm_std):
    return collate(rows, norm_mean, norm_std), max(r[3] for r in rows)


class _ShardDS(IterableDataset):
    """Module-level (picklable) dataset: each worker decodes a disjoint shard
    subset (shard_idx % num_workers == worker_id)."""
    def __init__(self, corpus, seed, limit_shards, start_shard):
        self.corpus, self.seed = corpus, seed
        self.limit_shards, self.start_shard = limit_shards, start_shard

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        return iter_rows(self.corpus, 0, seed=self.seed, limit_shards=self.limit_shards,
                         start_shard=self.start_shard, worker_id=wid, num_workers=nw)


def dl_batches(corpus, bs, norm_mean, norm_std, seed, limit_shards, start_shard, workers):
    """Parallel loader: `workers` processes decode disjoint shard subsets and
    prefetch, so the GPU isn't data-starved (the BC bottleneck). Yields the same
    ((toks,mask,acts,rtg), shard_idx) shape as `batches`."""
    from functools import partial
    dl = DataLoader(
        _ShardDS(corpus, seed, limit_shards, start_shard), batch_size=bs,
        num_workers=workers,
        collate_fn=partial(_collate_with_si, norm_mean=norm_mean, norm_std=norm_std),
        prefetch_factor=4, persistent_workers=False)
    yield from dl


def build_model(d_model=512, layers=6, heads=8, ff=1024):    # ~12.6M (bigger default)
    import torch
    import torch.nn as nn

    class BCPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.type_emb = nn.Embedding(16, d_model)
            self.field = nn.Linear(TOK_W, d_model)
            self.rtg = nn.Linear(1, d_model)
            enc = nn.TransformerEncoderLayer(d_model, heads, ff, batch_first=True,
                                             dropout=0.1)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.h_fverb = nn.Linear(d_model, N_MOVER)
            self.h_farg = nn.Linear(d_model, N_FARMER_ARG)
            self.h_hverb = nn.Linear(d_model, MAX_HANDS * N_MOVER)
            self.h_harg = nn.Linear(d_model, MAX_HANDS * N_ARG)
            self.h_mverb = nn.Linear(d_model, MAX_MARKET * N_MARKET)
            self.h_marg = nn.Linear(d_model, MAX_MARKET * N_ARG)
            self.h_mqty = nn.Linear(d_model, MAX_MARKET * (QTY_CAP + 1))

        def forward(self, toks, mask, rtg):
            b = toks.shape[0]
            typ = toks[:, :, 0].long().clamp(0, 15)
            x = self.type_emb(typ) + self.field(toks)
            cls = self.cls.expand(b, 1, -1) + self.rtg(rtg).unsqueeze(1)
            x = torch.cat([cls, x], dim=1)
            m = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device),
                           mask], dim=1)
            h = self.enc(x, src_key_padding_mask=~m)[:, 0]
            return dict(h=h,                                # CLS summary (RL value head)
                        fverb=self.h_fverb(h), farg=self.h_farg(h),
                        hverb=self.h_hverb(h).view(b, MAX_HANDS, N_MOVER),
                        harg=self.h_harg(h).view(b, MAX_HANDS, N_ARG),
                        mverb=self.h_mverb(h).view(b, MAX_MARKET, N_MARKET),
                        marg=self.h_marg(h).view(b, MAX_MARKET, N_ARG),
                        mqty=self.h_mqty(h).view(b, MAX_MARKET, QTY_CAP + 1))

    return BCPolicy()


def loss_fn(out, acts):
    import torch.nn.functional as F
    ce = F.cross_entropy
    fv, fa = acts[:, 0], acts[:, 1].clamp(0, N_FARMER_ARG - 1)
    L = ce(out["fverb"], fv) + ce(out["farg"], fa)
    o = 2
    hv = acts[:, o:o + MAX_HANDS * 2:2]
    ha = acts[:, o + 1:o + MAX_HANDS * 2:2].clamp(0, N_ARG - 1)
    L = L + ce(out["hverb"].reshape(-1, N_MOVER), hv.reshape(-1))
    L = L + ce(out["harg"].reshape(-1, N_ARG), ha.reshape(-1))
    o = 2 + MAX_HANDS * 2
    mv = acts[:, o:o + MAX_MARKET * 3:3]
    ma = acts[:, o + 1:o + MAX_MARKET * 3:3].clamp(0, N_ARG - 1)
    mq = acts[:, o + 2:o + MAX_MARKET * 3:3].clamp(0, QTY_CAP)
    L = L + ce(out["mverb"].reshape(-1, N_MARKET), mv.reshape(-1))
    L = L + ce(out["marg"].reshape(-1, N_ARG), ma.reshape(-1))
    L = L + ce(out["mqty"].reshape(-1, QTY_CAP + 1), mq.reshape(-1))
    return L


def batches(gen, bs, norm_mean, norm_std):
    """Yield (collated_batch, cur_shard_idx). cur_shard = max shard seen in the
    batch, used for resumable checkpointing."""
    rows = []
    for r in gen:
        rows.append(r)
        if len(rows) >= bs:
            si = max(x[3] for x in rows)
            yield collate(rows, norm_mean, norm_std), si
            rows = []
    if rows:
        si = max(x[3] for x in rows)
        yield collate(rows, norm_mean, norm_std), si


def _find_resume(out, explicit=None):
    """A checkpoint to resume from: explicit path, then any prior kernel's
    checkpoint mounted under /kaggle/input, then the working-dir out."""
    cands = []
    if explicit:
        cands.append(explicit)
    cands += sorted(glob.glob("/kaggle/input/**/policy_bc.pt", recursive=True))
    cands.append(out)
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def save_ckpt(path, model, opt, scaler, epoch, shard, gstep, done, norm_mean, norm_std,
              cfg):
    import torch
    tmp = path + ".tmp"
    torch.save({"state": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": int(epoch), "shard": int(shard), "global_step": int(gstep),
                "done": bool(done), "layout_version": 2,
                "norm_mean": norm_mean.tolist(), "norm_std": norm_std.tolist(),
                "config": cfg}, tmp)
    os.replace(tmp, path)                                  # atomic -> never a half file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="policy_bc.pt")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--ckpt-every", type=int, default=1000, help="steps between checkpoints")
    ap.add_argument("--max-hours", type=float, default=10.5,
                    help="self-stop + commit before Kaggle's 12h kill (proven guard)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel data-loader workers (0 = single-thread)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--corpus", default=None,
                    help="explicit corpus dir (else auto-find; also TRACKP_CORPUS env)")
    ap.add_argument("--d-model", type=int, default=512)      # ~12.6M model (bigger)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    a = ap.parse_args()
    import torch
    corpus = a.corpus or os.environ.get("TRACKP_CORPUS") or _find_corpus()
    print(f"[bc] corpus={corpus}", flush=True)
    try:
        nj = json.load(open(os.path.join(corpus, "norm.json")))
        norm_mean = np.array(nj["mean"], np.float32)
        norm_std = np.maximum(np.array(nj["std"], np.float32), 1e-3)
    except Exception:
        norm_mean = np.zeros(TOK_W, np.float32); norm_std = np.ones(TOK_W, np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (dev == "cuda")
    epochs = 1 if a.smoke else a.epochs
    lim = 2 if a.smoke else a.limit_shards
    # --- resolve model dims: a resume checkpoint's dims WIN (weights must match) ---
    rp = None if a.smoke else _find_resume(a.out, a.resume)
    dims = dict(d_model=a.d_model, layers=a.layers, heads=a.heads, ff=a.ff)
    resume_ck = None
    if rp and os.path.exists(rp):
        resume_ck = torch.load(rp, map_location=dev, weights_only=False)
        for k in ("d_model", "layers", "heads", "ff"):
            if k in (resume_ck.get("config") or {}):
                dims[k] = resume_ck["config"][k]
    model = build_model(**dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None      # fp16 AMP (T4 speedup)
    cfg = {"epochs": epochs, "bs": a.bs, "lr": a.lr, **dims}
    print(f"[bc] device={dev} amp={amp} dims={dims} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    train_model = model                          # raw model is saved; compiled one trains
    if not a.smoke and not a.no_compile:
        try:
            train_model = torch.compile(model)
            print("[bc] torch.compile ON", flush=True)
        except Exception as e:
            print(f"[bc] torch.compile unavailable ({e}) -- eager", flush=True)

    # --- resume (weights_only=False: opt/scaler payload needs full unpickle;
    #     torch>=2.6 defaults weights_only=True and would fail -- per the RSNA ckpt) ---
    start_epoch, start_shard, gstep = 0, 0, 0
    if resume_ck is not None:
        ck = resume_ck
        model.load_state_dict(ck["state"])
        for key, obj in (("opt", opt), ("scaler", scaler)):
            if ck.get(key) is not None and obj is not None:
                try:
                    obj.load_state_dict(ck[key])
                except Exception:
                    pass
        if ck.get("done"):
            print(f"[bc] resume {rp}: already DONE -- nothing to do", flush=True); return
        start_epoch = int(ck.get("epoch", 0)); start_shard = int(ck.get("shard", 0))
        gstep = int(ck.get("global_step", 0))
        print(f"[bc] RESUME from {rp}: epoch {start_epoch} shard {start_shard} "
              f"gstep {gstep}", flush=True)

    n_shards = max(1, len(shard_list(corpus, lim)))
    quart = {int(n_shards * q): int(q * 100) for q in (0.25, 0.5, 0.75)}
    t_start = time.time()

    for ep in range(start_epoch, epochs):
        model.train(); t0 = time.time(); seen = 0; run = 0.0
        sstart = start_shard if ep == start_epoch else 0
        done_q = {p for th, p in quart.items() if th <= sstart}
        if a.workers > 0 and not a.smoke:
            bgen = dl_batches(corpus, a.bs, norm_mean, norm_std, ep, lim, sstart, a.workers)
        else:
            bgen = batches(iter_rows(corpus, 0, seed=ep, limit_shards=lim,
                                     start_shard=sstart), a.bs, norm_mean, norm_std)
        for step, ((toks, mask, acts, rtg), si) in enumerate(bgen):
            toks, mask, acts, rtg = (toks.to(dev), mask.to(dev), acts.to(dev), rtg.to(dev))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(train_model(toks, mask, rtg), acts)
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            seen += toks.shape[0]; run += float(loss.detach()); gstep += 1
            if step % 50 == 0:
                print(f"[bc] ep{ep} step{step} shard{si}/{n_shards} "
                      f"loss={run/max(step+1,1):.3f} seen={seen:,} "
                      f"({seen/max(time.time()-t0,1):.0f}/s)", flush=True)
            if gstep % a.ckpt_every == 0:                  # frequent latest checkpoint
                save_ckpt(a.out, model, opt, scaler, ep, si, gstep, False,
                          norm_mean, norm_std, cfg)
                from kaggriculture.trackp import trainmetrics as _TM   # learning trace
                _TM.append(a.out, "bc", gstep=gstep, epoch=ep, shard=int(si),
                           loss=round(run / max(step + 1, 1), 4),
                           sps=int(seen / max(time.time() - t0, 1)))
            for th, pct in quart.items():                  # 25/50/75% milestone copies
                if si >= th and pct not in done_q:
                    done_q.add(pct)
                    save_ckpt(a.out.replace(".pt", f"_ep{ep}_q{pct}.pt"), model, opt,
                              scaler, ep, si, gstep, False, norm_mean, norm_std, cfg)
                    print(f"[bc] milestone ep{ep} {pct}% saved", flush=True)
            if not a.smoke and (time.time() - t_start) / 3600.0 >= a.max_hours:
                save_ckpt(a.out, model, opt, scaler, ep, si, gstep, False,
                          norm_mean, norm_std, cfg)
                print(f"[bc] TIME BUDGET {a.max_hours}h hit -- checkpoint saved, "
                      f"exiting to commit. Re-run to resume.", flush=True)
                return
            if a.smoke and step >= 5:
                break
        last = (ep == epochs - 1)
        # epoch complete -> checkpoint at the NEXT epoch boundary (shard 0)
        save_ckpt(a.out, model, opt, scaler, ep + 1, 0, gstep, last, norm_mean, norm_std, cfg)
        save_ckpt(a.out.replace(".pt", f"_ep{ep}_q100.pt"), model, opt, scaler, ep + 1, 0,
                  gstep, last, norm_mean, norm_std, cfg)
        print(f"[bc] epoch {ep} done in {time.time()-t0:.0f}s -> {a.out}", flush=True)
    print("[bc] DONE (all epochs)", flush=True)


if __name__ == "__main__":
    main()
