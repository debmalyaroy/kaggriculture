"""Gate (queue Q10c): the Rust forward pass (crates/policy) == torch MacroNet.

    python python/learn/export_check.py [--weights DIR]

Uses weights/bc/LATEST when it exists, else a freshly initialized net (checks the format and the
math before any training). Inputs: 1,000 real sequences from the training caches when present,
else random. Builds policy-check, runs it, exits with its status.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MacroNet, NF, profile_names  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(RL, "data", "ops", "export_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--k", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    n_act = len(profile_names())
    wdir = a.weights
    if wdir is None:
        try:
            wdir = os.path.join(RL, "weights", "bc", open(os.path.join(RL, "weights", "bc", "LATEST")).read().strip())
        except OSError:
            wdir = None
    rng = np.random.default_rng(1)
    if wdir:
        net = MacroNet(n_act)
        net.load_state_dict(torch.load(os.path.join(wdir, "ckpt.pt"), map_location="cpu", weights_only=False)["model"])
    else:
        torch.manual_seed(0)
        net = MacroNet(n_act, mu=rng.normal(size=NF).astype(np.float32), sd=(0.5 + rng.random(NF)).astype(np.float32))
        wdir = os.path.join(WORK, "init")
        net.export(wdir, {"note": "random init for the export check"})
    if not os.path.exists(os.path.join(wdir, "weights.bin")):
        net.export(wdir, {"note": "exported by export_check"})
    x = None
    for f in ("league_obs.npy", "corpus_obs.npy"):
        p = os.path.join(RL, "data", "train", f)
        if os.path.exists(p):
            src = np.load(p, mmap_mode="r")
            idx = np.sort(rng.choice(len(src), size=min(a.k, len(src)), replace=False))
            x = np.asarray(src[idx], dtype=np.float32)
            break
    if x is None:
        x = rng.normal(size=(a.k, 30, NF)).astype(np.float32)
    net.eval()
    with torch.no_grad():
        o = net(torch.from_numpy(x))
    y = torch.cat([o["pi"], o["v"][..., None]], -1).numpy().astype("<f4")
    x.astype("<f4").tofile(os.path.join(WORK, "states.f32"))
    y.tofile(os.path.join(WORK, "expected.f32"))
    env = dict(os.environ, CARGO_TARGET_DIR=os.environ.get("CARGO_TARGET_DIR") or os.path.join(RL, "target-dev"))
    b = subprocess.run(["cargo", "build", "-p", "policy", "--release", "--bin", "policy-check"], cwd=RL, env=env,
                       capture_output=True, text=True)
    if b.returncode != 0:
        print(b.stderr[-2000:])
        sys.exit(1)
    r = subprocess.run([os.path.join(os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release"), "policy-check" + (".exe" if os.name == "nt" else "")), os.path.join(wdir, "weights.bin"),
                        os.path.join(WORK, "states.f32"), os.path.join(WORK, "expected.f32"), str(len(x))],
                       capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    print(f"export_check: weights {wdir}; inputs {'cache' if x is not None else 'random'}; rc {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
