"""TPP train/serve equivalence: PyTorch forward vs the Rust forward (tppcheck) on the same bcdump records.

    python python/tpp/check.py --net-dir weights/tpp/bc1 --records data/tpp/d1/part_00.bin [--n 256]
Fails (exit 1) unless every logit agrees within 1e-3 and every argmax matches.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--net-dir", required=True)
ap.add_argument("--records", required=True)
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bin", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "target-tpp", "release", "tppcheck.exe"))
a = ap.parse_args()
pt = sorted(f for f in os.listdir(a.net_dir) if f.endswith(".pt"))[-1]
net = T.Net()
net.load_state_dict(torch.load(os.path.join(a.net_dir, pt), map_location="cpu"))
net.eval()
out = os.path.join(a.net_dir, "check_logits.bin")
subprocess.run([a.bin, "--net", os.path.join(a.net_dir, "net.json"), "--records", a.records, "--n", str(a.n), "--out", out], check=True)
r = np.asarray(np.memmap(a.records, dtype=D.REC, mode="r")[:a.n])
with torch.no_grad():
    board = torch.from_numpy(r["board"].astype(np.float32) / 255.0)
    uc, uq, mi, mq, mh = net(board, torch.from_numpy(r["glob"].astype(np.float32)), torch.from_numpy(r["units"].astype(np.float32) / 255.0))
per = D.MAXU * (D.NUNIT + D.NUQ) + D.NMKT + D.NMKT * D.NMQ + 11
rs = np.fromfile(out, dtype="<f4").reshape(len(r), per)
worst, argbad, n = 0.0, 0, 0
for i in range(len(r)):
    nu = int(r["nu"][i])
    ru = rs[i, :D.MAXU * (D.NUNIT + D.NUQ)].reshape(D.MAXU, D.NUNIT + D.NUQ)[:nu]
    pu = torch.cat([uc[i, :nu], uq[i, :nu]], 1).numpy()
    rm = rs[i, D.MAXU * (D.NUNIT + D.NUQ):]
    pm = torch.cat([mi[i], mq[i].reshape(-1), mh[i]]).numpy()
    worst = max(worst, float(np.abs(ru - pu).max(initial=0)), float(np.abs(rm - pm).max()))
    argbad += int((ru[:, :D.NUNIT].argmax(1) != pu[:, :D.NUNIT].argmax(1)).sum())
    n += nu
print(f"[check] {len(r)} records, {n} units: max |rust - torch| = {worst:.2e}, unit argmax mismatches {argbad}")
sys.exit(0 if worst < 1e-3 and argbad == 0 else 1)
