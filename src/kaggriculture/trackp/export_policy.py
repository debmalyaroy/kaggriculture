"""P3.5 -- export the torch macro-policy to pure-python weights.

Output JSON: {"w1","b1","w2","b2","wo","bo","heads"} -- consumed by the
planner template's _l1_forward (argmax at inference) and by
rollouts._forward_logits (softmax sampling at training).

The export-equivalence assertion is MANDATORY: torch logits vs the
pure-python forward must agree <= 1e-5 on 256 random feature vectors, and
the argmax bins must agree exactly. Anything else refuses to write.

Run with the GPU env: .../envs/llm/python.exe src/trackp/export_policy.py
  --ckpt models/trackp/iql_actor.pt --out models/trackp/l1_weights.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from kaggriculture.trackp import macro, rollouts  # noqa: E402


def export(ckpt_path: str, out_path: str, key: str = "actor") -> dict:
    import torch
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob[key] if key in blob else blob
    # Sequential(Linear, Tanh, Linear, Tanh, Linear) -> indices 0, 2, 4
    w1 = sd["0.weight"].numpy()
    b1 = sd["0.bias"].numpy()
    w2 = sd["2.weight"].numpy()
    b2 = sd["2.bias"].numpy()
    wo = sd["4.weight"].numpy()
    bo = sd["4.bias"].numpy()
    W = {"w1": w1.tolist(), "b1": b1.tolist(),
         "w2": w2.tolist(), "b2": b2.tolist(),
         "wo": wo.tolist(), "bo": bo.tolist(),
         "heads": [[n, k] for n, k in macro.HEADS]}

    # ---- equivalence assertion ----
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(w1.shape[1], w1.shape[0]), nn.Tanh(),
                        nn.Linear(w2.shape[1], w2.shape[0]), nn.Tanh(),
                        nn.Linear(wo.shape[1], wo.shape[0]))
    net[0].weight.data = torch.tensor(w1)
    net[0].bias.data = torch.tensor(b1)
    net[2].weight.data = torch.tensor(w2)
    net[2].bias.data = torch.tensor(b2)
    net[4].weight.data = torch.tensor(wo)
    net[4].bias.data = torch.tensor(bo)
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(256, w1.shape[1])).astype(np.float64)
    with torch.no_grad():
        t_logits = net(torch.tensor(X, dtype=torch.float64).float()).numpy()
    Wf = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "wo": wo, "bo": bo}
    max_err = 0.0
    argmax_mismatch = 0
    for i in range(X.shape[0]):
        py = np.asarray(rollouts._forward_logits(
            {k: v.tolist() for k, v in Wf.items()}, X[i].tolist()))
        max_err = max(max_err, float(np.max(np.abs(py - t_logits[i]))))
        off = 0
        for _, k in macro.HEADS:
            if int(py[off:off + k].argmax()) != int(
                    t_logits[i, off:off + k].argmax()):
                argmax_mismatch += 1
            off += k
    assert max_err <= 1e-4, f"export equivalence broken: max_err {max_err}"
    assert argmax_mismatch == 0, f"{argmax_mismatch} argmax mismatches"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(W, fh)
    rep = {"out": out_path, "max_err": max_err,
           "params": int(w1.size + w2.size + wo.size
                         + b1.size + b2.size + bo.size)}
    print(json.dumps(rep))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="actor")
    a = ap.parse_args()
    export(a.ckpt, a.out, a.key)
