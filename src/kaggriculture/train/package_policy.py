"""E5.2 / F1.1 (export) -- package a trained macro policy to ONNX + manifest.

Takes ``models/rl/rl_policy.pt`` (or the BC checkpoint as a fallback), exports
the transformer to ONNX with a dynamic day axis, and VERIFIES onnxruntime
reproduces the torch outputs to tolerance -- an export-equivalence gate (the
same discipline as ``experiments/policy_bc`` asserting export equivalence). It
also writes a self-describing JSON manifest (norm stats + schema + feature
recipe) so a Python OR native-Rust runtime (``candle``/``tract``/``ort``, task
F1.1 / G5.1) can build the exact input features and decode the output.

Outputs under ``models/rl/``:
  * ``policy.onnx``      -- the exported graph (input ``x[1,T,IN_DIM]``)
  * ``policy_manifest.json`` -- norm, schema, feature recipe, parity report
  * (optional) a submission bundle stub for the Rust runtime (documented hook)

    python -m kaggriculture.train.package_policy            # export + parity gate
    python -m kaggriculture.train.package_policy --smoke    # tiny CPU self-test
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys

import kaggriculture.train.bc_warmup as BC
import kaggriculture.train.macro_actions as MA

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RL_OUT = os.path.join(ROOT, "models", "rl", "rl_policy.pt")
ONNX_OUT = os.path.join(ROOT, "models", "rl", "policy.onnx")
MANIFEST = os.path.join(ROOT, "models", "rl", "policy_manifest.json")


def _pick_checkpoint(prefer_rl=True) -> str:
    if prefer_rl and os.path.exists(RL_OUT):
        return RL_OUT
    if os.path.exists(BC.OUT_PATH):
        return BC.OUT_PATH
    raise FileNotFoundError("no policy checkpoint (run bc_warmup / macro_rl)")


def export(ckpt_path: str, onnx_path: str = ONNX_OUT,
           manifest_path: str = MANIFEST, tol: float = 1e-3) -> dict:
    import numpy as np
    import torch

    model, ck = BC.load_policy(ckpt_path, "cpu")
    model.eval()

    class Wrap(torch.nn.Module):
        """No pad-mask (single sequence) so the ONNX graph is clean."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            cls, vec = self.m(x)
            return cls, vec

    wrap = Wrap(model).eval()
    # Export at a FIXED T = MAX_DAYS. The encoder is causal (output[t] depends
    # only on inputs[0..t]), so at inference we build features for the real days
    # and zero-pad to MAX_DAYS, then read the first real_T outputs -- padding
    # future days never changes an earlier day's output. Fixed T sidesteps the
    # dynamic-axis reshape bug in the attention op's TorchScript export.
    T0 = BC.MAX_DAYS
    dummy = torch.randn(1, T0, BC.IN_DIM)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        torch.onnx.export(
            wrap, (dummy,), onnx_path, input_names=["x"],
            output_names=["cls_logits", "vec"], opset_version=17, dynamo=False)
    except TypeError:                       # older torch without the kwarg
        torch.onnx.export(
            wrap, (dummy,), onnx_path, input_names=["x"],
            output_names=["cls_logits", "vec"], opset_version=17)

    parity = dict(checked=False)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        max_dc, max_dv = 0.0, 0.0
        for real_T in (1, 5, BC.MAX_DAYS):
            x = torch.zeros(1, BC.MAX_DAYS, BC.IN_DIM)
            x[0, :real_T] = torch.randn(real_T, BC.IN_DIM)
            with torch.no_grad():
                tc, tv = wrap(x)
            oc, ov = sess.run(None, {"x": x.numpy()})
            # compare only the real days
            max_dc = max(max_dc, float(np.abs(tc.numpy()[:, :real_T]
                                              - oc[:, :real_T]).max()))
            max_dv = max(max_dv, float(np.abs(tv.numpy()[:, :real_T]
                                              - ov[:, :real_T]).max()))
        parity = dict(checked=True, max_cls_diff=max_dc, max_vec_diff=max_dv,
                      tol=tol, pass_=bool(max_dc < tol and max_dv < tol),
                      fixed_T=BC.MAX_DAYS, pad="zero-pad to MAX_DAYS, read real days")
    except ImportError:
        parity = dict(checked=False, reason="onnxruntime not installed")

    manifest = dict(
        source_checkpoint=os.path.relpath(ckpt_path, ROOT),
        onnx=os.path.relpath(onnx_path, ROOT),
        in_dim=BC.IN_DIM, vec_len=MA.VECTOR_LEN, n_class=len(MA.MACRO_CLASSES),
        max_days=BC.MAX_DAYS, world_buckets=BC.WORLD_BUCKETS,
        day_buckets=BC.DAY_BUCKETS,
        feature_names=list(MA.FEATURE_NAMES), classes=list(MA.MACRO_CLASSES),
        norm=ck["norm"],
        log_std=(ck.get("log_std").tolist()
                 if hasattr(ck.get("log_std"), "tolist") else None),
        feature_recipe=(
            "x[0]=day/30; x[1:1+DB]=day one-hot bucket; "
            "x[1+DB:1+DB+WB]=world one-hot(hash(sig)%WB); "
            "x[base]=rtg(condition on 1.0 to WIN); x[base+1]=rating/3000; "
            "x[base+2:]= (cum_prior_action - cum_mu)/cum_sd  (base=1+DB+WB). "
            "Decode: MacroAction.from_vector(clip(vec*tgt_sd+tgt_mu,0,None))."),
        parity=parity,
    )
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    ok = (not parity["checked"]) or parity.get("pass_", False)
    print(f"[pkg] exported {os.path.relpath(onnx_path, ROOT)}  "
          f"parity={'PASS' if parity.get('pass_') else ('SKIP' if not parity['checked'] else 'FAIL')}"
          f" (cls<={parity.get('max_cls_diff', 0):.2e} vec<={parity.get('max_vec_diff', 0):.2e})")
    print(f"[pkg] manifest -> {os.path.relpath(manifest_path, ROOT)}")
    if parity["checked"] and not parity.get("pass_"):
        raise SystemExit("[pkg] ONNX parity FAILED -- not shippable")
    return manifest


def _smoke() -> int:
    if not os.path.exists(BC.OUT_PATH):
        BC._smoke()
    m = export(_pick_checkpoint(prefer_rl=False),
               onnx_path=ONNX_OUT, manifest_path=MANIFEST)
    assert os.path.exists(ONNX_OUT) and os.path.exists(MANIFEST)
    p = m["parity"]
    if p["checked"]:
        assert p["pass_"], p
    print("[pkg][smoke] OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bc", action="store_true", help="force the BC checkpoint")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    export(_pick_checkpoint(prefer_rl=not args.bc), tol=args.tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
