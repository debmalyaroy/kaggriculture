"""STREAMING trainer for the bandit heads (out-of-core). Loads ONE on-disk chunk
at a time, mini-batch SGD on GPU, discards it, over epochs -- the full dataset
never lives in memory. Feature standardisation is computed in one streaming pass.
Reads .local/nn/chunks/{sale_X,sale_Y,opp_X,opp_y}_NNN.npy from extract.py.

Run: python -m kaggriculture.bandit.nn.train_stream --kind sale
     python -m kaggriculture.bandit.nn.train_stream --kind opp
Writes .local/nn/{sale,opp}_nn.json (inline weights for config).
"""
import argparse, glob, json, os
import numpy as np
from kaggriculture.paths import ROOT
OUT = os.path.join(ROOT, ".local", "nn")
CH = os.path.join(OUT, "chunks")

def stream_stats(files):
    """Streaming mean/std over all X chunks (bounded memory: 32 accumulators)."""
    n = 0; s = None; ss = None
    for f in files:
        X = np.load(f)
        s = X.sum(0) if s is None else s + X.sum(0)
        ss = (X.astype(np.float64) ** 2).sum(0) if ss is None else ss + (X.astype(np.float64) ** 2).sum(0)
        n += len(X)
    mu = (s / n); var = ss / n - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-12)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sale", "opp", "clone", "joint"], required=True)
    ap.add_argument("--chunks-dir", default=CH, help="dir of {kind}_X/_y chunks (sp_chunks for clone)")
    ap.add_argument("--hidden", type=int, default=0)  # 0 -> 32 (sale) / 16 (opp/clone)
    ap.add_argument("--layers", type=str, default="",
                    help="comma hidden sizes, e.g. '64,64,32' -> deep net (overrides --hidden)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout on hidden layers (regularise deeper nets)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--mix", type=int, default=3, help="chunks concatenated+shuffled per group (cross-chunk mixing)")
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--shed-mask", action="store_true",
                    help="sale only: condition loss+AUC per product on rows where shed>0 "
                         "(the rail only consults the head for a product it is holding)")
    ap.add_argument("--tag", type=str, default="",
                    help="output suffix: writes {kind}{tag}_nn.json (keep '' to overwrite the live head)")
    ap.add_argument("--opp-weight", type=float, default=5.0,
                    help="joint only: loss weight on the opponent column so MTL does not sacrifice it")
    a = ap.parse_args()
    import copy
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cdir = a.chunks_dir
    # "joint" = the merged multiclass head: sale(9) + opponent(1) = 10 outputs,
    # sharing one trunk. Its features are sale_X (identical to opp_X); its labels
    # are the stacked joint_Y built by build_joint_chunks.py.
    xpat, ypat = {"sale": ("sale_X", "sale_Y"), "opp": ("opp_X", "opp_y"),
                  "clone": ("clone_X", "clone_y"), "joint": ("sale_X", "joint_Y")}[a.kind]
    xs = sorted(glob.glob(os.path.join(cdir, f"{xpat}_*.npy")))
    ys = sorted(glob.glob(os.path.join(cdir, f"{ypat}_*.npy")))
    assert xs and len(xs) == len(ys), f"no chunks in {cdir}"
    # VALIDATION = a FULL middle chunk (the last chunk is the tiny partial tail,
    # ~29k rows from one shard -> a noisy, unrepresentative val that makes
    # best-epoch selection meaningless). The middle full chunk is representative.
    vi = len(xs) // 2 if len(xs) > 2 else len(xs) - 1
    va_x, va_y = xs[vi], ys[vi]
    tr_x = [x for i, x in enumerate(xs) if i != vi] or xs
    tr_y = [y for i, y in enumerate(ys) if i != vi] or ys
    mu, sd, ntr = stream_stats(tr_x)
    F = np.load(tr_x[0]).shape[1]
    if a.kind in ("sale", "joint"):
        y0 = np.load(tr_y[0]); O = y0.reshape(len(y0), -1).shape[1]   # 9 (sale) / 10 (joint)
    else:
        O = 1
    # hidden topology: --layers "64,64,32" (deep) else a single --hidden layer
    if a.layers.strip():
        hidden = [int(h) for h in a.layers.split(",") if h.strip()]
    else:
        hidden = [a.hidden or (32 if a.kind in ("sale", "joint") else 16)]
    out_dim = O if a.kind in ("sale", "joint") else 1
    # shed-conditioning (Tier A): a product's "sell now?" label is only meaningful
    # on rows where the player HOLDS it (shed>0) -- which is exactly when the rail
    # consults the head. Masking removes the flood of cross-world trivial negatives
    # (e.g. CARROT rows in a world where CARROT never unlocked) that make rare
    # products look unlearnable. shed feature for product j is at index j*3+2.
    shed_mask = a.shed_mask and a.kind in ("sale", "joint")
    N_SALE = O if a.kind == "sale" else (9 if a.kind == "joint" else 0)  # joint: last col = opp
    SHED_IDX = [j * 3 + 2 for j in range(N_SALE)]

    def col_mask(Xraw):
        """Per-output held mask (n, out_dim). For joint, the opponent column (last)
        is always active -- the dump label applies on every row, not just held ones."""
        m = (Xraw[:, SHED_IDX] > 0)
        if a.kind == "joint":
            m = np.concatenate([m, np.ones((len(Xraw), 1), bool)], 1)
        return m
    # BOTH heads are now CLASSIFICATION (sale = 9 independent binary sell-now
    # labels, opp = 1 dump label) -> train logits with BCEWithLogitsLoss, report
    # AUC. The exported net carries sigmoid=true so Rust applies it at inference.
    seq = []
    prev = F
    for h in hidden:
        seq += [nn.Linear(prev, h), nn.ReLU()]
        if a.dropout > 0:
            seq += [nn.Dropout(a.dropout)]
        prev = h
    seq += [nn.Linear(prev, out_dim)]
    net = nn.Sequential(*seq).to(dev)
    # per-class imbalance weighting (streamed): pos_weight = neg/pos per output.
    # Under shed-mask the denominator is rows-where-held (conditional base rate),
    # which is the honest imbalance and lifts rare-but-available products.
    if shed_mask:
        pos = np.zeros(O); cnt = np.zeros(O)
        for fx, fy in zip(tr_x, tr_y):
            Xc = np.load(fx); yc = np.load(fy).reshape(len(Xc), -1).astype(np.float64)
            m = col_mask(Xc).astype(np.float64)
            pos += (yc * m).sum(0); cnt += m.sum(0)
        pos = np.maximum(pos, 1.0); cnt = np.maximum(cnt, 1.0)
        pw = np.clip((cnt - pos) / pos, 1.0, 50.0).astype(np.float32)
        print(f"{a.kind} SHED-MASK cond_pos_rate={np.round(pos/cnt,4).tolist()} "
              f"pos_weight={np.round(pw,2).tolist()}", flush=True)
    else:
        pos = None; tot = 0.0
        for f in tr_y:
            y = np.load(f).reshape(len(np.load(f)), -1).astype(np.float64)
            pos = y.sum(0) if pos is None else pos + y.sum(0)
            tot += len(y)
        pos = np.maximum(pos, 1.0)
        pw = np.clip((tot - pos) / pos, 1.0, 50.0).astype(np.float32)
        print(f"{a.kind} pos_rate={np.round(pos/max(tot,1),4).tolist()} pos_weight={np.round(pw,2).tolist()}", flush=True)
    # joint MTL: 9 sale columns vs 1 opp column -> the mean loss under-weights the
    # opponent task and the shared trunk sacrifices it (opp AUC collapses). Fix with
    # a per-column loss weight (up-weight the opp column) + per-element reduction.
    need_none = shed_mask or a.kind == "joint"
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev),
                                 reduction="none" if need_none else "mean")
    cw = None
    if a.kind == "joint":
        cwv = np.ones(O, np.float32); cwv[9] = a.opp_weight   # opp column = index 9
        cw = torch.tensor(cwv, device=dev)
        print(f"joint opp_weight={a.opp_weight} (col-9 loss weight)", flush=True)
    H = hidden  # for the log line below
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    muT = torch.tensor(mu, device=dev); sdT = torch.tensor(sd, device=dev)
    print(f"{a.kind} STREAM rows={ntr} F={F} O={O} H={H} chunks={len(tr_x)} mix={a.mix} "
          f"lr={a.lr} dev={dev} val_chunk={os.path.basename(va_x)}", flush=True)
    # pre-load the representative val chunk once (constant across epochs)
    Yv = np.load(va_y).reshape(-1, O if a.kind in ("sale", "joint") else 1)
    Xv_raw = np.load(va_x)
    Xvt = (torch.tensor(Xv_raw, device=dev) - muT) / sdT
    Yvt = torch.tensor(Yv.astype(np.float32), device=dev)
    Mvt = torch.tensor(col_mask(Xv_raw), device=dev) if shed_mask else None
    # BEST-EPOCH checkpoint: keep the weights at the best validation metric, not
    # the LAST epoch's (the old bug -- last epoch was often worse than mid-run).
    best_state = copy.deepcopy(net.state_dict()); best_metric = None; best_ep = -1
    for ep in range(a.epochs):
        # CROSS-CHUNK MIXING: each epoch, shuffle chunk order and process in
        # groups of `mix`; load+concat+shuffle each group so a minibatch draws
        # from several shards' distributions at once. Without this, each epoch is
        # a sequence of single-shard blocks -> unstable updates (bouncing AUC,
        # dead-sigmoid collapse). Memory bounded to `mix` chunks.
        order = np.random.permutation(len(tr_x))
        net.train()
        for g in range(0, len(order), a.mix):
            grp = order[g:g + a.mix]
            Xs = np.concatenate([np.load(tr_x[ci]) for ci in grp], 0)
            yl = [np.load(tr_y[ci]) for ci in grp]
            Ys = np.concatenate([y.reshape(len(y), -1) for y in yl], 0)
            Xt = (torch.tensor(Xs, device=dev) - muT) / sdT
            Yt = torch.tensor(Ys.astype(np.float32), device=dev)
            Mt = torch.tensor(col_mask(Xs).astype(np.float32), device=dev) if shed_mask else None
            p = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), a.batch):
                b = p[i:i + a.batch]; opt.zero_grad()
                if need_none:
                    per = lossf(net(Xt[b]), Yt[b])             # (batch, out_dim)
                    if shed_mask:
                        per = per * Mt[b]                      # zero out not-held products
                    if cw is not None:
                        per = per * cw                         # up-weight opp column (joint)
                    if shed_mask:
                        loss = per.sum() / Mt[b].sum().clamp_min(1.0)
                    else:
                        loss = per.mean()
                else:
                    loss = lossf(net(Xt[b]), Yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            del Xt, Yt, Xs, Ys
        net.eval()
        with torch.no_grad():
            # mean per-column AUC (both heads are classification now). Rank-based
            # AUC via cumulative-positive over the sorted scores, per output.
            logits = net(Xvt)
            pv = torch.sigmoid(logits)
            col_auc = [float("nan")] * pv.shape[1]   # column-aligned (keep identity)
            for j in range(pv.shape[1]):
                yv = Yvt[:, j]; sc = pv[:, j]
                if shed_mask:                       # AUC over held-rows only (conditional)
                    msk = Mvt[:, j]; yv = yv[msk]; sc = sc[msk]
                ordr = torch.argsort(sc)
                yvs = yv[ordr]; pos = yvs.sum().item(); neg = len(yvs) - pos
                if pos and neg:
                    col_auc[j] = (torch.cumsum(1 - yvs, 0) * yvs).sum().item() / (pos * neg)
            aucs = [x for x in col_auc if x == x]
            if a.kind == "joint":                   # task-balanced: opp counts as much as
                sale_a = [x for x in col_auc[:9] if x == x]   # the whole 9-product sale block
                sm = float(np.mean(sale_a)) if sale_a else 0.5
                op = col_auc[9] if col_auc[9] == col_auc[9] else 0.5
                metric = 0.5 * sm + 0.5 * op
                extra = f" sale={sm:.3f} opp={op:.3f}"
            else:
                metric = float(np.mean(aucs)) if aucs else 0.5
                extra = f" min={min(aucs):.3f}" if len(aucs) > 1 else ""
            improved = best_metric is None or metric > best_metric
            print(f"  ep{ep:2d} mean_auc={metric:.3f}{extra}{'  *' if improved else ''}", flush=True)
        if improved:
            best_metric = metric; best_ep = ep; best_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)  # ship the BEST epoch, not the last
    print(f"  >> best epoch {best_ep} metric={best_metric:.5f}", flush=True)
    del Xvt, Yvt
    # export EVERY Linear layer in order -> the general "layers" format the Rust
    # Mlp reads (arbitrary depth). Also emit legacy w1/b1/w2/b2 when the net is
    # exactly 2-deep, so older consumers still parse it.
    lins = [m for m in net if isinstance(m, nn.Linear)]
    layers = [{"w": m.weight.detach().cpu().numpy().tolist(),
               "b": m.bias.detach().cpu().numpy().tolist()} for m in lins]
    params = int(sum(m.weight.numel() + m.bias.numel() for m in lins))
    out = {"kind": a.kind, "hidden": H, "in": F, "out": O,
           "sigmoid": True,  # both heads are classification; Rust applies sigmoid at inference
           "mu": mu.tolist(), "sd": sd.tolist(), "layers": layers}
    if len(lins) == 2:  # legacy-compatible mirror
        out.update({"w1": layers[0]["w"], "b1": layers[0]["b"],
                    "w2": layers[1]["w"], "b2": layers[1]["b"]})
    out["shed_mask"] = bool(shed_mask)  # inference must apply the same held-only gating
    fname = f"{a.kind}{a.tag}_nn.json"
    json.dump(out, open(os.path.join(OUT, fname), "w"))
    print(f"wrote {fname} ({len(lins)} layers, {params} params, hidden={H}, shed_mask={shed_mask})", flush=True)

if __name__ == "__main__":
    main()
