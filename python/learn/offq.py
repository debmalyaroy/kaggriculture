"""Offline RL on the branch oracle's data: fitted Q(state, profile) -> a greedy policy (queue Q25).

    python python/learn/offq.py [--run weights/ppo/<run>] [--files 200] [--lams 2,5,10,20,50] [--threads 16]

Why this form: the inverse model (idm.py, Q23) showed a real player's day cannot be labelled with our
profiles (held-out top-1 0.043 vs chance 0.028), so offline RL on ladder games has no action labels.
The oracle's data has them exactly: for each material decision it holds the final margin of every
branched candidate from the same state. That is a counterfactual Q table, no bootstrapping needed.

1. Base = the PPO run's best.bin (trunk frozen). h(s) = the GRU state after day d's observation.
2. Q head: Linear(64, 36) fit by MSE on the centred Phi(margin / sigma) of each branched, allowed
   candidate (the state value cancels; only differences between profiles matter for the choice).
   Held out: the newest 10% of oracle files.
3. Policy for lambda: logits = pi_base(h) + lambda * q(h). Both heads are linear on the same h, so the
   export is an ordinary policy file (pi.weight += lambda * q.weight). lambda 0 = the base.
4. Held-out regret per lambda (Phi points x100 vs the best branched candidate), then for the best two
   lambdas the PPO validation composite vs v63 on the same seeds (head-to-head + panel + hard tapes,
   reference cached in the run as val_ref_v3.json), then the band gate on real players for the best.
Writes weights/offq/<ID>/{lam*.bin, report.json}.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo  # noqa: E402
from model import EXPORT_ORDER, MacroNet  # noqa: E402

RL = ppo.RL
N_ACT = 36


def load_bin(path, n_act):
    net = MacroNet(n_act)
    sd = net.state_dict()
    raw = np.fromfile(path, dtype="<f4")
    i = 0
    for k in EXPORT_ORDER:
        n = sd[k].numel()
        sd[k] = torch.tensor(raw[i:i + n]).reshape(sd[k].shape)
        i += n
    assert i == raw.size, (path, i, raw.size)
    net.load_state_dict(sd)
    return net


def oracle_rows(files, sigma):
    """Every oracle decision where the margin differed across candidates: obs [30, NF], day, Phi per
    candidate (NaN = not branched or not allowed), the allowed mask."""
    obs, day, phs, allow = [], [], [], []
    for f in files:
        pre = f[:-7]
        try:
            tsv = [ln.split("\t") for ln in open(pre + ".tsv", encoding="utf-8").read().splitlines()]
            traj = np.fromfile(pre + ".traj", dtype="<f4").reshape(len(tsv), 30, ppo.W)
        except (OSError, ValueError):
            continue
        at = {int(r[0]): i for i, r in enumerate(tsv)}
        for ln in open(f, encoding="utf-8").read().splitlines():
            seed, d, cells = ln.split("\t")
            c = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in cells.split(",")]
            if len({m for _, m in c}) < 2 or int(seed) not in at:
                continue
            g, d = at[int(seed)], int(d)
            bits = int(ppo.mask_bits(traj[g, d]).reshape(-1)[0])
            ph = np.full(N_ACT, np.nan, np.float32)
            for k, m in c:
                if (bits >> k) & 1:
                    ph[k] = 0.5 * (1 + math.erf(m / (sigma * 2 ** 0.5)))
            if np.isfinite(ph).sum() < 2:
                continue
            obs.append(traj[g, :, :ppo.NF])
            day.append(d)
            phs.append(ph)
            allow.append([(bits >> k) & 1 == 1 for k in range(N_ACT)])
    return np.array(obs, np.float32), np.array(day, np.int64), np.array(phs, np.float32), np.array(allow, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="PPO run dir (default: the newest with best.bin)")
    ap.add_argument("--files", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=5000.0)
    ap.add_argument("--lams", default="1,2,5,10,20,50")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-gate", action="store_true", help="stop after the held-out regret (no validation / band gate)")
    a = ap.parse_args()
    torch.set_num_threads(min(a.threads, 16))
    torch.manual_seed(0)
    run = a.run or sorted((d for d in glob.glob(os.path.join(RL, "weights", "ppo", "ppo-*")) if os.path.exists(os.path.join(d, "best.bin"))),
                          key=os.path.getmtime)[-1]
    oid = "offq-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = os.path.join(RL, "weights", "offq", oid)
    os.makedirs(out, exist_ok=True)
    base = load_bin(os.path.join(run, "best.bin"), N_ACT).eval()

    files = sorted(glob.glob(os.path.join(ppo.ORACLE, "o_*.oracle")))[-a.files:]
    n_hold = max(1, len(files) // 10)
    tr = oracle_rows(files[:-n_hold], a.sigma)
    te = oracle_rows(files[-n_hold:], a.sigma)
    print(f"[offq] base {os.path.basename(run)}/best.bin; oracle files {len(files)} ({n_hold} held out); rows train {len(tr[0])}, held-out {len(te[0])}", flush=True)

    def feats(rows):
        """The frozen trunk's state after day d and the base policy logits there."""
        obs, day = rows[0], rows[1]
        hs, lg = [], []
        with torch.no_grad():
            for i in range(0, len(obs), 2048):
                x = torch.tensor(obs[i:i + 2048])
                z = torch.relu(base.inp((x - base.mu) / base.sd))
                h = z.new_zeros(len(z), 64)
                hh = []
                for t in range(30):
                    h = base.gru(z[:, t], h)
                    hh.append(h)
                hd = torch.stack(hh, 1)[torch.arange(len(z)), torch.tensor(day[i:i + 2048])]
                hs.append(hd)
                lg.append(base.pi(hd))
        return torch.cat(hs), torch.cat(lg)

    h_tr, pi_tr = feats(tr)
    h_te, pi_te = feats(te)

    def target(ph):
        m = np.isfinite(ph)
        c = np.where(m, ph, 0.0)
        mean = c.sum(1, keepdims=True) / m.sum(1, keepdims=True)
        return torch.tensor(np.where(m, (ph - mean) * 10.0, 0.0), dtype=torch.float32), torch.tensor(m)

    y_tr, m_tr = target(tr[2])
    y_te, m_te = target(te[2])
    q = torch.nn.Linear(64, N_ACT)
    torch.nn.init.zeros_(q.weight)
    torch.nn.init.zeros_(q.bias)  # lambda * q starts as the base policy exactly
    opt = torch.optim.Adam(q.parameters(), lr=3e-3, weight_decay=1e-4)
    for ep in range(a.epochs):
        p = q(h_tr)
        p = p - (p * m_tr).sum(1, keepdim=True) / m_tr.sum(1, keepdim=True)  # centred over the branched set
        loss = (((p - y_tr) ** 2) * m_tr).sum() / m_tr.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                pt = q(h_te)
                pt = pt - (pt * m_te).sum(1, keepdim=True) / m_te.sum(1, keepdim=True)
                lt = (((pt - y_te) ** 2) * m_te).sum() / m_te.sum()
                l0 = ((y_te ** 2) * m_te).sum() / m_te.sum()
            print(f"[offq] epoch {ep}: train mse {loss.item():.4f}, held-out mse {lt.item():.4f} (predict-zero {l0.item():.4f})", flush=True)

    def regret(lam, h, pi, rows):
        """Phi points x100 lost vs the best branched candidate, over rows whose chosen profile was branched."""
        ph, allow = rows[2], rows[3]
        with torch.no_grad():
            s = (pi + lam * q(h)).numpy()
        s = np.where(allow, s, -1e9)
        k = s.argmax(1)
        got = ph[np.arange(len(k)), k]
        ok = np.isfinite(got)
        best = np.nanmax(ph, 1)
        return float(np.mean(best[ok] - got[ok]) * 100), float(ok.mean())

    ph35 = te[2][:, 35]
    ok35 = np.isfinite(ph35)
    rep = {"id": oid, "run": os.path.basename(run), "rows_train": int(len(tr[0])), "rows_heldout": int(len(te[0])),
           "heldout_regret_v63": round(float(np.mean(np.nanmax(te[2], 1)[ok35] - ph35[ok35]) * 100), 3), "lams": {}}
    for lam in [0.0] + [float(x) for x in a.lams.split(",")]:
        r, cov = regret(lam, h_te, pi_te, te)
        rep["lams"][str(lam)] = {"heldout_regret": round(r, 3), "coverage": round(cov, 3)}
        print(f"[offq] lambda {lam:g}: held-out regret {r:.3f} (coverage {cov:.2f}); always-v63 {rep['heldout_regret_v63']}", flush=True)
        if lam > 0:
            net = load_bin(os.path.join(run, "best.bin"), N_ACT)
            with torch.no_grad():
                net.pi.weight += lam * q.weight
                net.pi.bias += lam * q.bias
            ppo.export_bin(net, os.path.join(out, f"lam{lam:g}.bin"))
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    if a.no_gate:
        return

    # the PPO validation composite, same seeds and cached v63 reference as the run's own validation
    refp = os.path.join(run, f"val_ref_v{ppo.VAL_VERSION}.json")
    ref = json.load(open(refp))

    def combo(bin_):
        p = os.path.join(out, "val")
        h2h = ppo.vplay(["--weights", bin_], p, ppo.VAL_N, ppo.VAL_SEED0, a.threads, opp="fixed:35")
        sc = list(h2h.values())

        def delta(c, r):
            ks = [k for k in c if str(k) in r]
            return float(np.mean([c[k] - r[str(k)] for k in ks])) if ks else 0.0, sum(c[k] > r[str(k)] for k in ks), sum(c[k] < r[str(k)] for k in ks)
        pan = [delta(ppo.vplay(["--weights", bin_], p + "_p", ppo.VAL_PANEL_N, ppo.VAL_PANEL_SEED0, a.threads, opp=o), ref["panel"][o]) for o in ppo.VAL_PANEL]
        hard = delta(ppo.vplay(["--weights", bin_], p + "_h", ppo.VAL_HARD_N, ppo.VAL_HARD_SEED0, a.threads, tapes=ppo.HARD_TAPES), ref["hard"])
        c = (float(np.mean(sc)) - 0.5) + float(np.mean([x[0] for x in pan])) + hard[0]
        return {"combo": round(c, 4), "h2h_wdl": [sum(x == 1 for x in sc), sum(x == 0.5 for x in sc), sum(x == 0 for x in sc)],
                "panel_bw": [sum(x[1] for x in pan), sum(x[2] for x in pan)], "hard_bw": [hard[1], hard[2]]}

    rep["val"] = {"base": combo(os.path.join(run, "best.bin"))}
    print(f"[offq] validation base (best.bin): {rep['val']['base']}", flush=True)
    ranked = sorted((v["heldout_regret"], k) for k, v in rep["lams"].items() if float(k) > 0)
    for _, k in ranked[:2]:
        rep["val"][k] = combo(os.path.join(out, f"lam{float(k):g}.bin"))
        print(f"[offq] validation lambda {k}: {rep['val'][k]}", flush=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    best_k = max((k for k in rep["val"] if k != "base"), key=lambda k: rep["val"][k]["combo"])
    if rep["val"][best_k]["combo"] > rep["val"]["base"]["combo"]:
        r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand", os.path.join(out, f"lam{float(best_k):g}.bin"),
                            "--name", f"{oid}-lam{best_k}", "--ref-profile", "35", "--threads", str(a.threads)], cwd=RL, capture_output=True, text=True)
        rep["band_gate"] = (r.stdout.strip().splitlines() or [""])[-1]
        print(r.stdout[-1500:], flush=True)
    else:
        rep["band_gate"] = "skipped: no lambda beats the base on the validation composite"
        print(f"[offq] {rep['band_gate']}", flush=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    print(f"[offq] done -> {out}/report.json")


if __name__ == "__main__":
    main()
