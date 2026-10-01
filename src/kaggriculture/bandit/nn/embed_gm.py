"""Unsupervised opponent-STYLE embedding + clustering on the GM dataset.

Trains an AUTOENCODER (streaming/out-of-core) on the 107-feature GM rows:

    107 -> 64 -> 32 -> [16 bottleneck] -> 32 -> 64 -> 107   (MSE reconstruction)

The 16-d bottleneck is a learned trajectory-style embedding. We then run k-means
over the encoded GM rows to get K opponent-style CLUSTERS. Exports:

  * embed_encoder.json  -- the ENCODER as a Rust-loadable Mlp ("layers"+mu/sd),
                           so both Rust inference and numpy can produce embeddings.
  * embed_kmeans.json   -- K centroids (in embedding space) + per-cluster size.

This is the GM-based substrate the CLONE head uses (see clone_cluster.py): clone
detection = symmetry distance between MY embedding and the OPPONENT's embedding,
which needs no clone labels to TRAIN (only to validate). The cluster id is also a
reusable opponent-style feature for the router.

Run: python -m kaggriculture.bandit.nn.embed_gm --bottleneck 16 --clusters 8
"""
import argparse, glob, json, os
import numpy as np
from kaggriculture.paths import ROOT

OUT = os.path.join(ROOT, ".local", "nn")
CH = os.path.join(OUT, "chunks")


def stream_stats(files):
    n = 0; s = None; ss = None
    for f in files:
        X = np.load(f).astype(np.float64)
        s = X.sum(0) if s is None else s + X.sum(0)
        ss = (X ** 2).sum(0) if ss is None else ss + (X ** 2).sum(0)
        n += len(X)
    mu = s / n; var = ss / n - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-12)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32), n


def kmeans(Z, K, iters=25, seed=0):
    """Plain Lloyd k-means (numpy) so we take no sklearn dependency."""
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), K, replace=False)].copy()
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # n x K
        a = d.argmin(1)
        newC = np.array([Z[a == k].mean(0) if (a == k).any() else C[k] for k in range(K)])
        if np.allclose(newC, C):
            C = newC; break
        C = newC
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    a = d.argmin(1)
    inertia = float(d[np.arange(len(Z)), a].mean())
    return C, a, inertia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default=CH)
    ap.add_argument("--bottleneck", type=int, default=16)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mix", type=int, default=3)
    ap.add_argument("--kmeans-sample", type=int, default=200_000)
    a = ap.parse_args()
    import copy, torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xs = sorted(glob.glob(os.path.join(a.chunks_dir, "sale_X_*.npy")))  # any *_X: same feats
    assert xs, f"no feature chunks in {a.chunks_dir}"
    vi = len(xs) // 2 if len(xs) > 2 else len(xs) - 1
    tr = [x for i, x in enumerate(xs) if i != vi] or xs
    mu, sd, ntr = stream_stats(tr)
    F = np.load(tr[0]).shape[1]
    muT = torch.tensor(mu, device=dev); sdT = torch.tensor(sd, device=dev)
    enc_dims = [F, 64, 32, a.bottleneck]
    dec_dims = [a.bottleneck, 32, 64, F]

    def block(dims, last_relu):
        seq = []
        for i in range(len(dims) - 1):
            seq.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_relu:
                seq.append(nn.ReLU())
        return seq
    enc = nn.Sequential(*block(enc_dims, last_relu=False)).to(dev)
    dec = nn.Sequential(*block(dec_dims, last_relu=False)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    lossf = nn.MSELoss()
    print(f"AE rows={ntr} F={F} bottleneck={a.bottleneck} K={a.clusters} dev={dev} "
          f"val={os.path.basename(xs[vi])}", flush=True)
    Xv = (torch.tensor(np.load(xs[vi]), device=dev) - muT) / sdT
    best = None; best_state = None
    for ep in range(a.epochs):
        order = np.random.permutation(len(tr))
        enc.train(); dec.train()
        for g in range(0, len(order), a.mix):
            grp = order[g:g + a.mix]
            Xs = np.concatenate([np.load(tr[ci]) for ci in grp], 0)
            Xt = (torch.tensor(Xs, device=dev) - muT) / sdT
            p = torch.randperm(len(Xt), device=dev)
            for i in range(0, len(Xt), a.batch):
                b = p[i:i + a.batch]; opt.zero_grad()
                z = enc(Xt[b]); r = dec(z)
                loss = lossf(r, Xt[b]); loss.backward(); opt.step()
            del Xt
        enc.eval(); dec.eval()
        with torch.no_grad():
            vr = lossf(dec(enc(Xv)), Xv).item()
        improved = best is None or vr < best
        print(f"  ep{ep:2d} val_recon_mse={vr:.5f}{'  *' if improved else ''}", flush=True)
        if improved:
            best = vr; best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(dec.state_dict()))
    enc.load_state_dict(best_state[0])
    print(f"  >> best recon_mse={best:.5f}", flush=True)
    # export encoder as a Rust-loadable Mlp
    lins = [m for m in enc if isinstance(m, nn.Linear)]
    layers = [{"w": m.weight.detach().cpu().numpy().tolist(),
               "b": m.bias.detach().cpu().numpy().tolist()} for m in lins]
    enc_json = {"kind": "embed", "in": F, "out": a.bottleneck, "sigmoid": False,
                "mu": mu.tolist(), "sd": sd.tolist(), "layers": layers}
    json.dump(enc_json, open(os.path.join(OUT, "embed_encoder.json"), "w"))
    # encode a sample and cluster
    enc.eval()
    with torch.no_grad():
        Zv = enc(Xv).cpu().numpy()
    Zs = Zv[:a.kmeans_sample]
    C, assign, inertia = kmeans(Zs.astype(np.float64), a.clusters)
    sizes = np.bincount(assign, minlength=a.clusters).tolist()
    json.dump({"k": a.clusters, "centroids": C.tolist(), "sizes": sizes,
               "inertia": inertia, "bottleneck": a.bottleneck},
              open(os.path.join(OUT, "embed_kmeans.json"), "w"))
    print(f"wrote embed_encoder.json + embed_kmeans.json  "
          f"K={a.clusters} sizes={sizes} inertia={inertia:.4f}", flush=True)


if __name__ == "__main__":
    main()
