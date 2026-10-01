"""R2: native (dependency-free) inference of the macro policy — pure numpy.

The trackp seat runs the policy LIVE each macro decision. Macro inference is only
~30 calls/game (not per-turn), so a heavy Rust ONNX runtime is unnecessary: a
pure-numpy forward of the (small) transformer is exact and well under the 1 s/turn
budget, and it ships with NO extra dependency. This mirrors the torch model in
``bc_warmup`` exactly (norm_first encoder, GELU-erf, causal mask, LayerNorm eps
1e-5) and is validated bit-close to torch before use.

    python -m kaggriculture.train.native_infer --validate   # numpy == torch
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import math
import os
import sys

import numpy as np

import kaggriculture.train.bc_warmup as BC

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def export_weights(ckpt_path: str = BC.OUT_PATH) -> dict:
    """Pull the torch state_dict into a plain numpy weight bundle."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    cfg = ck["config"]
    w = {k: v.numpy().astype(np.float32) for k, v in sd.items()}
    return {"w": w, "cfg": cfg, "norm": ck["norm"], "schema": ck["schema"]}


def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    from math import sqrt
    # torch default GELU (exact, erf-based)
    from scipy.special import erf  # noqa
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def _gelu_noscipy(x):
    # erf via a numerically-good approximation (Abramowitz-Stegun 7.1.26)
    s = np.sign(x)
    a = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return 0.5 * x * (1.0 + s * y)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _mha(x, w, pfx, heads):
    T, d = x.shape
    dk = d // heads
    ipw = w[pfx + "in_proj_weight"]       # [3d, d]
    ipb = w[pfx + "in_proj_bias"]         # [3d]
    qkv = x @ ipw.T + ipb                 # [T, 3d]
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    q = q.reshape(T, heads, dk).transpose(1, 0, 2)   # [h, T, dk]
    k = k.reshape(T, heads, dk).transpose(1, 0, 2)
    v = v.reshape(T, heads, dk).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / math.sqrt(dk)  # [h, T, T]
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    scores = scores + mask
    attn = _softmax(scores)
    out = attn @ v                        # [h, T, dk]
    out = out.transpose(1, 0, 2).reshape(T, d)
    return out @ w[pfx + "out_proj.weight"].T + w[pfx + "out_proj.bias"]


def numpy_forward(bundle: dict, x: np.ndarray):
    """x: [T, in_dim] -> (cls_logits [T, n_class], vec [T, vec_len])."""
    w, cfg = bundle["w"], bundle["cfg"]
    heads = cfg["heads"]
    layers = cfg["layers"]
    T = x.shape[0]
    h = x @ w["inp.weight"].T + w["inp.bias"]
    h = h + w["pos"][0, :T]
    for i in range(layers):
        p = f"enc.layers.{i}."
        a = _mha(_ln(h, w[p + "norm1.weight"], w[p + "norm1.bias"]),
                 w, p + "self_attn.", heads)
        h = h + a
        f = _ln(h, w[p + "norm2.weight"], w[p + "norm2.bias"])
        f = _gelu_noscipy(f @ w[p + "linear1.weight"].T + w[p + "linear1.bias"])
        f = f @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
        h = h + f
    h = _ln(h, w["norm.weight"], w["norm.bias"])
    cls = h @ w["cls_head.weight"].T + w["cls_head.bias"]
    vec = h @ w["vec_head.weight"].T + w["vec_head.bias"]
    return cls, vec


def save_bundle(ckpt_path: str = BC.OUT_PATH, out: str = None) -> str:
    """R3: pack the policy WEIGHTS + norm + schema into one npz the trackp
    submission notebook embeds (so the agent is self-contained, no torch)."""
    b = export_weights(ckpt_path)
    out = out or os.path.join(ROOT, "models", "rl", "trackp_weights.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    flat = {f"w::{k}": v for k, v in b["w"].items()}
    flat["__cfg__"] = np.frombuffer(
        __import__("json").dumps(b["cfg"]).encode(), np.uint8)
    flat["__norm__"] = np.frombuffer(
        __import__("json").dumps(b["norm"]).encode(), np.uint8)
    flat["__schema__"] = np.frombuffer(
        __import__("json").dumps(b["schema"]).encode(), np.uint8)
    np.savez_compressed(out, **flat)
    print(f"[native] weight bundle -> {os.path.relpath(out, ROOT)} "
          f"({os.path.getsize(out)/1e6:.1f} MB, {len(b['w'])} tensors)")
    return out


def load_bundle(path: str) -> dict:
    """Reconstruct the weight bundle from the npz (what the submission does)."""
    import json
    z = np.load(path, allow_pickle=False)
    w = {k[3:]: z[k] for k in z.files if k.startswith("w::")}
    cfg = json.loads(bytes(z["__cfg__"]).decode())
    norm = json.loads(bytes(z["__norm__"]).decode())
    return {"w": w, "cfg": cfg, "norm": norm}


def validate(ckpt_path: str = BC.OUT_PATH, tol: float = 2e-3) -> dict:
    import torch
    bundle = export_weights(ckpt_path)
    model, ck = BC.load_policy(ckpt_path, "cpu")
    model.eval()
    max_c, max_v = 0.0, 0.0
    for T in (1, 5, BC.MAX_DAYS):
        x = np.random.randn(T, BC.IN_DIM).astype(np.float32)
        with torch.no_grad():
            tc, tv = model(torch.from_numpy(x[None]))
        nc, nv = numpy_forward(bundle, x)
        max_c = max(max_c, float(np.abs(tc.numpy()[0] - nc).max()))
        max_v = max(max_v, float(np.abs(tv.numpy()[0] - nv).max()))
    ok = max_c < tol and max_v < tol
    print(f"[native] numpy vs torch: cls<={max_c:.2e} vec<={max_v:.2e} "
          f"-> {'PASS' if ok else 'FAIL'} (tol {tol})")
    return dict(max_cls=max_c, max_vec=max_v, pass_=ok)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--ckpt", default=BC.OUT_PATH)
    args = ap.parse_args()
    r = validate(args.ckpt)
    return 0 if r["pass_"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
